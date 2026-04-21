import os
import math
from itertools import combinations
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("✓ GPU (CuPy) detected")
    _ = cp.zeros(1)
except ImportError:
    GPU_AVAILABLE = False
    cp = np
    print("⚠️ GPU not available (using NumPy)")

xp = cp if GPU_AVAILABLE else np

class SindyArgs:
    def __init__(self, win_len=5120, stride=None, overlap=0.5, d_max=2, simpl_rho=1.0, scale=1.0,
                 admm_rho=3.0, admm_overrelax=1.6, max_iters=100, fs=256.0,
                 win_sg=29, order=3, r_target_pc=0.95, k_min=600, k_max=1000000,
                 row_norm_nnz_thr=1e-6, param_abs_nnz_thr=1e-8,
                 tau2_q=0.75, tau3_q=0.75):
        self.win_len = win_len
        self.stride = stride
        self.overlap = overlap
        self.d_max = d_max
        self.simpl_rho = simpl_rho
        self.scale = scale
        self.admm_rho = admm_rho
        self.admm_overrelax = admm_overrelax
        self.max_iters = max_iters
        self.fs = fs
        self.win_sg = win_sg
        self.order = order
        self.r_target_pc = r_target_pc
        self.k_min = k_min
        self.k_max = k_max
        self.row_norm_nnz_thr = row_norm_nnz_thr
        self.param_abs_nnz_thr = param_abs_nnz_thr
        self.tau2_q = tau2_q
        self.tau3_q = tau3_q

# ─────────────── DIAGNOSTIC & SUPPORT HELPERS ────────────────
def precompute_index_patterns(n, dmax, use_xp=True):
    out = {}
    if dmax >= 2 and n >= 2:
        iu0, iu1 = np.triu_indices(n, k=1)
        pairs = np.stack([iu0.astype(np.int32), iu1.astype(np.int32)], axis=1)
        out['pairs'] = cp.asarray(pairs) if (use_xp and GPU_AVAILABLE) else pairs
    if dmax >= 3 and n >= 3:
        triples = np.array(list(combinations(range(n), 3)), dtype=np.int32)
        out['triples'] = cp.asarray(triples) if (use_xp and GPU_AVAILABLE) else triples
    if dmax >= 4 and n >= 4:
        quads = np.array(list(combinations(range(n), 4)), dtype=np.int32)
        out['quads'] = cp.asarray(quads) if (use_xp and GPU_AVAILABLE) else quads
    return out

def build_maps_from_patterns(n, dmax, idx):
    maps = {1:{}, 2:{}, 3:{}, 4:{}}
    row = 0
    if dmax >= 1:
        for i in range(n):
            row += 1
            maps[1][(i,)] = row
    if dmax >= 2 and 'pairs' in idx:
        for a, b in (idx['pairs'].get() if GPU_AVAILABLE else idx['pairs']):
            row += 1
            maps[2][(int(a), int(b))] = row
    if dmax >= 3 and 'triples' in idx:
        for i, j, k in (idx['triples'].get() if GPU_AVAILABLE else idx['triples']):
            row += 1
            maps[3][(int(i), int(j), int(k))] = row
    if dmax >= 4 and 'quads' in idx:
        for i, j, k, l in (idx['quads'].get() if GPU_AVAILABLE else idx['quads']):
            row += 1
            maps[4][(int(i), int(j), int(k), int(l))] = row
    g_total = row + 1
    return maps, g_total

def build_soc_edges_from_maps(dmax, maps):
    edges = []
    if dmax >= 2:
        for (i,j), child in maps[2].items():
            edges += [(child, maps[1][(i,)]), (child, maps[1][(j,)])]
    if dmax >= 3:
        for (i,j,k), child in maps[3].items():
            for face in combinations((i,j,k), 2):
                edges.append((child, maps[2][tuple(sorted(face))]))
    if dmax >= 4:
        for (i,j,k,l), child in maps[4].items():
            for face in combinations((i,j,k,l), 3):
                edges.append((child, maps[3][tuple(sorted(face))]))
    return edges

def build_library_fast(XT, n, dmax, idx):
    T = XT.shape[0]
    parts = [xp.ones((T, 1), dtype=XT.dtype)]
    if dmax >= 1:
        parts.append(XT)
    if dmax >= 2 and 'pairs' in idx:
        p = idx['pairs']
        parts.append(XT[:, p[:, 0]] * XT[:, p[:, 1]])
    if dmax >= 3 and 'triples' in idx:
        tr = idx['triples']
        parts.append(XT[:, tr[:,0]] * XT[:, tr[:,1]] * XT[:, tr[:,2]])
    if dmax >= 4 and 'quads' in idx:
        qd = idx['quads']
        parts.append(XT[:, qd[:,0]] * XT[:, qd[:,1]] * XT[:, qd[:,2]] * XT[:, qd[:,3]])
    return xp.concatenate(parts, axis=1)

def precompute_gram(Theta_std, Y_scaled):
    T = max(1, Theta_std.shape[0])
    G = (Theta_std.T @ Theta_std) / T
    eps_ridge = 5e-3
    G = G + eps_ridge * xp.eye(G.shape[0], dtype=G.dtype)
    B = (Theta_std.T @ Y_scaled)  / T
    return G, B

def project_rows_dykstra(Z, edges, alpha, iters=25, tol=1e-7):
    Zc = Z.copy()
    if not edges:
        return Zc
    ci = xp.asarray([e[0] for e in edges], dtype=xp.int32)
    pi = xp.asarray([e[1] for e in edges], dtype=xp.int32)
    E, ncols = ci.shape[0], Zc.shape[1]
    inc_c = xp.zeros((E, ncols), dtype=Zc.dtype)
    inc_p = xp.zeros((E, ncols), dtype=Zc.dtype)
    for _ in range(iters):
        Zold = Zc.copy()
        c_tmp = Zc[ci] - inc_c
        p_tmp = Zc[pi] - inc_p
        C = xp.linalg.norm(c_tmp, axis=1)
        P = xp.linalg.norm(p_tmp, axis=1)
        feas = (C <= alpha * P + 1e-12)
        if (~feas).any():
            idx = xp.where(~feas)[0]
            Ct, Pt = C[idx], P[idx]
            cti, pti = c_tmp[idx], p_tmp[idx]
            denom = 1.0 + alpha * alpha
            t = (alpha * Ct + Pt) / denom
            sc = xp.where(Ct > 0, (alpha * t) / Ct, 0.0)
            sp = xp.where(Pt > 0, t / Pt, 0.0)
            c_tmp[idx] = cti * sc[:, None]
            p_tmp[idx] = pti * sp[:, None]
        inc_c = c_tmp - (Zc[ci] - inc_c)
        inc_p = p_tmp - (Zc[pi] - inc_p)
        Zc[ci] = c_tmp
        Zc[pi] = p_tmp
        if float(xp.linalg.norm(Zc - Zold)) < tol:
            break
    return Zc

def degree_weights(maps, g, pair_lambda_mult=1.5):
    w = xp.ones(g, dtype=xp.float32)
    w[0] = 0.0
    for (_,), r in maps.get(1, {}).items():
        w[r] *= xp.array(1.0, dtype=xp.float32)
    for (_,_), r in maps.get(2, {}).items():
        w[r] *= xp.array(pair_lambda_mult, dtype=xp.float32)
    return w

def _window_stride_from_args(args):
    stride = args.stride
    if stride is None:
        stride = int(round(args.win_len * (1.0 - float(getattr(args, "overlap", 0.0)))))
    return max(1, min(int(stride), int(args.win_len)))

def build_edge_triangle_incidence(n_nodes, edges, triangles):
    edge_list = [tuple(sorted(edge)) for edge in edges]
    triangle_list = [tuple(sorted(tri)) for tri in triangles]

    edge_index = {edge: idx for idx, edge in enumerate(edge_list)}

    edge_incidence = np.zeros((n_nodes, len(edge_list)), dtype=np.int8)
    for e_idx, (i, j) in enumerate(edge_list):
        edge_incidence[i, e_idx] = 1
        edge_incidence[j, e_idx] = 1

    triangle_node_incidence = np.zeros((n_nodes, len(triangle_list)), dtype=np.int8)
    triangle_edge_incidence = np.zeros((len(edge_list), len(triangle_list)), dtype=np.int8)
    for t_idx, tri in enumerate(triangle_list):
        for node in tri:
            triangle_node_incidence[node, t_idx] = 1
        for edge in combinations(tri, 2):
            edge_key = tuple(sorted(edge))
            e_idx = edge_index.get(edge_key)
            if e_idx is not None:
                triangle_edge_incidence[e_idx, t_idx] = 1

    return {
        "edge_list": edge_list,
        "triangle_list": triangle_list,
        "edge_incidence": edge_incidence,
        "triangle_node_incidence": triangle_node_incidence,
        "triangle_edge_incidence": triangle_edge_incidence,
    }

def _window_result_payload(scores, res, n_nodes, w_start, w_end, window_index):
    node_scores = np.asarray(scores.get("S1", np.zeros(n_nodes)), dtype=float)
    edge_adjacency = np.asarray(scores.get("S2", np.zeros((n_nodes, n_nodes))), dtype=float)
    triangle_scores = scores.get("S3_max", {})

    topology = build_edge_triangle_incidence(n_nodes, res["edges"], res["triangles"])

    edge_feature_values = [float(edge_adjacency[i, j]) for i, j in topology["edge_list"]]
    triangle_feature_values = [float(triangle_scores.get(frozenset(tri), 0.0)) for tri in topology["triangle_list"]]

    return {
        "window_index": int(window_index),
        "window_start": int(w_start),
        "window_end": int(w_end),
        "window_length": int(w_end - w_start),
        "nodes": list(range(n_nodes)),
        "node_features": {
            "vector": node_scores,
            "adjacency": np.diag(node_scores),
        },
        "edges": topology["edge_list"],
        "edge_features": {
            "vector": edge_feature_values,
            "adjacency": edge_adjacency,
            "incidence": topology["edge_incidence"],
        },
        "triangles": topology["triangle_list"],
        "triangle_features": {
            "vector": triangle_feature_values,
            "incidence_nodes": topology["triangle_node_incidence"],
            "incidence_edges": topology["triangle_edge_incidence"],
        },
        "closed_triangles": [tuple(sorted(tri)) for tri in res["closed_triangles"]],
        "violating_triangles": [tuple(sorted(tri)) for tri in res["violating_triangles"]],
        "tau2": float(res["tau2"]),
        "tau3": float(res["tau3"]),
        "n_edges": int(res["n_edges"]),
        "n_triangles": int(res["n_triangles"]),
        "n_closed_triangles": int(res["n_closed_triangles"]),
        "n_violating_triangles": int(res["n_violating_triangles"]),
    }

def kkt_grad_row_norms(G, B, Xi):
    R = G @ Xi - B
    return (cp.asnumpy(cp.linalg.norm(R, axis=1)) if GPU_AVAILABLE else np.linalg.norm(R, axis=1))

def solve_admm_gl_soc(
    G, B, edges, lamb, rho_hier, w=None,
    max_iters=300, rho_admm=3.0, Xi0=None,
    log_every=25, adaptive_rho=True, overrelax=1.6,
    trace_every=1, return_trace=True, freeze_cholesky=False,
    early_proj_every=1, late_proj_every=1,
    early_proj_iters=15, late_proj_iters=50,
    early_proj_tol=1e-7,  late_proj_tol=1e-8,
):
    g, k = G.shape[0], B.shape[1]
    if w is None:
        w = xp.ones(g, dtype=G.dtype)

    Gs, Bs = G, B
    Is = xp.eye(g, dtype=G.dtype)

    Xi = xp.zeros((g, k), dtype=Gs.dtype) if Xi0 is None else Xi0.astype(Gs.dtype, copy=True)
    Z  = Xi.copy()
    C  = Xi.copy()
    Uz = xp.zeros_like(Xi)
    Uc = xp.zeros_like(Xi)

    A = Gs + (2.0 * rho_admm) * Is
    try:
        Lc = xp.linalg.cholesky(A)
        chol = True
    except Exception:
        chol = False

    lam = lamb

    for it in tqdm(range(max_iters)):
        Z_prev = Z.copy()
        C_prev = C.copy()

        RHS = Bs + rho_admm * (Z - Uz + C - Uc)
        if chol and not freeze_cholesky:
            Y = xp.linalg.solve(Lc, RHS)
            Xi = xp.linalg.solve(Lc.T, Y)
        else:
            Xi = xp.linalg.solve(A, RHS)

        Xi_hat = overrelax * Xi + (1.0 - overrelax) * Z_prev

        Wz = Xi_hat + Uz
        norms = xp.sqrt((Wz * Wz).sum(axis=1) + 1e-16)
        shrink = xp.maximum(0.0, 1.0 - (lam * w) / (rho_admm * norms))
        Z = (Wz.T * shrink).T

        two_thirds = (2 * max_iters) // 3
        if it < two_thirds:
            proj_every = early_proj_every; proj_iters = early_proj_iters; proj_tol = early_proj_tol
        else:
            proj_every = late_proj_every;  proj_iters = late_proj_iters;  proj_tol = late_proj_tol

        if proj_every and (it % proj_every) == 0:
            C = project_rows_dykstra(Xi_hat + Uc, edges, rho_hier, iters=proj_iters, tol=proj_tol)
        else:
            C = Xi_hat + Uc

        Uz += (Xi_hat - Z)
        Uc += (Xi_hat - C)

        r_pri = float(xp.linalg.norm(Xi - Z) + xp.linalg.norm(Xi - C))
        r_dual = float(rho_admm * xp.linalg.norm((Z - Z_prev) + (C - C_prev)))

        if adaptive_rho and it % 5 == 0 and it > 20:
            ratio = r_pri / max(r_dual, 1e-12)
            if ratio > 1.8:
                rho_admm *= 1.25; Uz /= 1.25; Uc /= 1.25
            elif ratio < 0.55:
                rho_admm /= 1.25; Uz *= 1.25; Uc *= 1.25
            rho_admm = float(min(max(rho_admm, 2.0), 10.0))
            A = Gs + (2.0 * rho_admm) * Is
            if not freeze_cholesky:
                try: Lc = xp.linalg.cholesky(A); chol = True
                except: chol = False

        if r_pri < 1e-3 and r_dual < 1e-3:
            break

    if it == (max_iters - 1):
        print(f"⚠️ ADMM did not converge after {max_iters} iterations (r_pri={r_pri:.2e}, r_dual={r_dual:.2e})")

    if return_trace: return Xi, C, {}
    else: return Xi, C

def _to_numpy_array(A_matrix, gpu_available=False, cp_module=None):
    if gpu_available and cp_module is not None:
        return cp_module.asnumpy(A_matrix).copy()
    return np.array(A_matrix, copy=True)

def _get_triple_structure(A, maps0, i, j, k):
    r_i  = maps0[1].get((i,))
    r_j  = maps0[1].get((j,))
    r_k  = maps0[1].get((k,))
    r_ij = maps0[2].get((i, j))
    r_ik = maps0[2].get((i, k))
    r_jk = maps0[2].get((j, k))

    if None in (r_i, r_j, r_k, r_ij, r_ik, r_jk): return None

    e_ij = max(abs(A[r_j, i]), abs(A[r_i, j]))
    e_ik = max(abs(A[r_k, i]), abs(A[r_i, k]))
    e_jk = max(abs(A[r_k, j]), abs(A[r_j, k]))
    t_i = abs(A[r_jk, i])
    t_j = abs(A[r_ik, j])
    t_k = abs(A[r_ij, k])
    m_ijk = min(e_ij, e_ik, e_jk)

    return {"rows": {"r_i": r_i, "r_j": r_j, "r_k": r_k, "r_ij": r_ij, "r_ik": r_ik, "r_jk": r_jk},
            "edges": {"e_ij": e_ij, "e_ik": e_ik, "e_jk": e_jk},
            "triangle_drivers": {"t_i": t_i, "t_j": t_j, "t_k": t_k},
            "bound": m_ijk}

def enforce_simplicial_hierarchy_clean(A_matrix, maps0, gpu_available=False, cp_module=None):
    if 1 not in maps0 or 2 not in maps0: return A_matrix.copy()

    A = _to_numpy_array(A_matrix, gpu_available=gpu_available, cp_module=cp_module)
    n = A.shape[1]

    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                info = _get_triple_structure(A, maps0, i, j, k)
                if info is None: continue
                m_ijk = info["bound"]
                rows = info["rows"]
                A[rows["r_jk"], i] = np.sign(A[rows["r_jk"], i]) * min(abs(A[rows["r_jk"], i]), m_ijk)
                A[rows["r_ik"], j] = np.sign(A[rows["r_ik"], j]) * min(abs(A[rows["r_ik"], j]), m_ijk)
                A[rows["r_ij"], k] = np.sign(A[rows["r_ij"], k]) * min(abs(A[rows["r_ij"], k]), m_ijk)

    if gpu_available and cp_module is not None:
        return cp_module.asarray(A)
    return A

def readout_scores_multi_mode(Xi, n, dmax, maps):
    A = cp.asnumpy(Xi) if GPU_AVAILABLE else Xi
    absA = np.abs(A)
    out = {}

    if dmax >= 1 and maps[1]:
        S1 = np.zeros(n)
        S2_dir = np.zeros((n, n))
        for j in range(n):
            row = maps[1][(j,)]
            S1[j] = float(np.linalg.norm(absA[row]))
            for i in range(n):
                if i == j: continue
                S2_dir[i, j] = absA[row, i]
        out['S1'] = S1
        out['S2'] = np.maximum(S2_dir, S2_dir.T)

    if dmax >= 2 and maps[2]:
        S3_max = {}
        S3_mean = {}
        S3_geom = {}
        pair_row = maps[2]
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    tri = frozenset((i, j, k))
                    r_jk = pair_row.get(tuple(sorted((j, k))))
                    r_ik = pair_row.get(tuple(sorted((i, k))))
                    r_ij = pair_row.get(tuple(sorted((i, j))))
                    if (r_jk is None) or (r_ik is None) or (r_ij is None): continue
                    
                    w_i, w_j, w_k = absA[r_jk, i], absA[r_ik, j], absA[r_ij, k]
                    weights = [w_i, w_j, w_k]
                    
                    s_max = max(weights)
                    if s_max > 1e-9: S3_max[tri] = s_max
                    s_mean = sum(weights) / 3.0
                    if s_mean > 1e-9: S3_mean[tri] = s_mean
                    s_geom = (w_i * w_j * w_k) ** (1.0/3.0)
                    if s_geom > 1e-9: S3_geom[tri] = s_geom

        out['S3_max'] = S3_max; out['S3_mean'] = S3_mean; out['S3_geom'] = S3_geom
    return out

def extract_simplicial_complex_edge_first(scores, n_nodes, tau2, tau3, S3_key="S3_max"):
    S2 = scores.get("S2", np.zeros((n_nodes, n_nodes)))
    S3 = scores.get(S3_key, {})

    edges = {
        frozenset((i, j))
        for i in range(n_nodes) for j in range(i + 1, n_nodes)
        if S2[i, j] >= tau2
    }

    candidate_triangles = set()
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            for k in range(j + 1, n_nodes):
                tri = frozenset((i, j, k))
                boundary = {frozenset((i, j)), frozenset((i, k)), frozenset((j, k))}
                if boundary.issubset(edges):
                    candidate_triangles.add(tri)

    triangles = {tri for tri in candidate_triangles if S3.get(tri, 0.0) >= tau3}
    return edges, triangles, candidate_triangles

def check_extracted_complex_closure(scores, n_nodes, tau2_q=0.75, tau3_q=0.75, S3_key="S3_max"):
    S2 = scores.get("S2", np.zeros((n_nodes, n_nodes)))
    S3_scores = scores.get(S3_key, {})

    edge_vals = [S2[i, j] for i in range(n_nodes) for j in range(i + 1, n_nodes) if S2[i, j] > 0]
    if edge_vals:
        tau2 = float(np.quantile(edge_vals, tau2_q))
    else: tau2 = np.inf

    tri_vals = list(S3_scores.values())
    if tri_vals: 
        tau3 = float(np.quantile(tri_vals, tau3_q))
    else: tau3 = np.inf

    edges, triangles, candidate_triangles = extract_simplicial_complex_edge_first(
        scores, n_nodes, tau2=tau2, tau3=tau3, S3_key=S3_key)

    closed_triangles = set()
    violating_triangles = set()
    for tri in triangles:
        i, j, k = sorted(tri)
        req_edges = {frozenset((i, j)), frozenset((i, k)), frozenset((j, k))}
        if req_edges.issubset(edges): closed_triangles.add(tri)
        else: violating_triangles.add(tri)

    return {
        "tau2": tau2, "tau3": tau3, "edges": edges, "triangles": triangles,
        "closed_triangles": closed_triangles, "violating_triangles": violating_triangles,
        "n_edges": len(edges), "n_triangles": len(triangles),
        "n_closed_triangles": len(closed_triangles), "n_violating_triangles": len(violating_triangles),
    }

# ─────────────── MAIN PROCESSING FUNCTION ────────────────
def process_data_in_windows(data, args):
    """
    Process dataset over overlapping/non-overlapping windows.
    Data format: `[nodes, channels]` i.e., shape (n, T).
    """
    n, T_raw = data.shape
    dt = 1.0 / args.fs

    half = (args.win_sg - 1) // 2
    # Savitzky-Golay for smoothing and derivative (Avoid boundary artifacts using whole array first or do we? I'll use whole dataset processing if applicable or ignore boundary)
    X_smooth = savgol_filter(data, args.win_sg, args.order, axis=1, mode='interp')[:, half:-half]
    dXdt_raw = savgol_filter(data, args.win_sg, args.order, deriv=1, delta=dt, axis=1, mode='interp')[:, half:-half]
    
    # Scale variables
    mu = X_smooth.mean(axis=1, keepdims=True)
    sig = X_smooth.std(axis=1, keepdims=True) + 1e-8
    X_full = (X_smooth - mu) / sig
    Y_full = dXdt_raw / sig

    idx_patterns = precompute_index_patterns(n, args.d_max, use_xp=True)
    maps0, g_total = build_maps_from_patterns(n, args.d_max, idx_patterns)
    edges0 = build_soc_edges_from_maps(args.d_max, maps0)
    w_degree_base = degree_weights(maps0, g_total)

    T_len = X_full.shape[1]
    stride = _window_stride_from_args(args)

    if T_len < args.win_len:
        raise ValueError(
            f"Input length {T_len} is shorter than the fixed window length {args.win_len}."
        )
    
    results = []

    # Iterate continuously over sliding windows.
    # Use notebook-aware tqdm and keep nested bars visible in notebook UIs.
    window_starts = list(range(0, T_len - args.win_len + 1, stride))
    show_window_progress = bool(getattr(args, "show_window_progress", True))
    disable_window_tqdm = bool(getattr(args, "disable_window_tqdm", False)) or (not show_window_progress)
    use_notebook_tqdm = bool(getattr(args, "use_notebook_tqdm", True))
    tqdm_cls = tqdm
    if use_notebook_tqdm:
        try:
            from tqdm.notebook import tqdm as notebook_tqdm
            tqdm_cls = notebook_tqdm
        except Exception:
            tqdm_cls = tqdm
    window_iter = enumerate(
        tqdm_cls(
            window_starts,
            total=len(window_starts),
            desc="sindy windows",
            leave=True,
            position=1,
            dynamic_ncols=True,
            mininterval=0.2,
            disable=disable_window_tqdm,
        )
    )
    for window_index, w_start in window_iter:
        w_end = w_start + args.win_len
        
        # 1. WINDOW DEPENDENT SLICING
        Xw = X_full[:, w_start:w_end]
        Yw = Y_full[:, w_start:w_end]

        x0 = np.median(Xw, axis=1, keepdims=True)
        Xc = Xw - x0
        sigma = np.maximum(
            1e-8,
            1.4826 * np.median(np.abs(Xc - np.median(Xc, axis=1, keepdims=True)), axis=1, keepdims=True)
        )
        d_pc = np.sqrt(((Xc / sigma) ** 2).sum(axis=0)) / np.sqrt(Xc.shape[0])
        keep = np.where(d_pc <= args.r_target_pc)[0]
        if keep.size < args.k_min:
            keep = np.argsort(d_pc)[:args.k_min]
        elif keep.size > args.k_max:
            keep = np.argsort(d_pc)[:args.k_max]
            
        keep = np.sort(keep)
        x0_res = x0.flatten()
        
        if GPU_AVAILABLE:
            Xw_gpu = cp.asarray(Xw)
            Yw_gpu = cp.asarray(Yw)
            x0_gpu = cp.asarray(x0_res.reshape(-1, 1))
            Xw_c = Xw_gpu - x0_gpu
            Xw_use = Xw_c[:, keep]
            Yw_use = Yw_gpu[:, keep]
        else:
            x0_cpu = x0_res.reshape(-1, 1)
            Xw_c = Xw - x0_cpu
            Xw_use = Xw_c[:, keep]
            Yw_use = Yw[:, keep]

        MAX_ROWS = 6000
        if Xw_use.shape[1] > MAX_ROWS:
            if GPU_AVAILABLE:
                idx = cp.linspace(0, Xw_use.shape[1] - 1, MAX_ROWS, dtype=cp.int32)
            else:
                idx = np.linspace(0, Xw_use.shape[1] - 1, MAX_ROWS, dtype=int)
            Xw_use = Xw_use[:, idx]
            Yw_use = Yw_use[:, idx]
            
        XT = Xw_use.T
        YT = Yw_use.T

        # 2. WINDOW DEPENDENT MATRICES
        Theta = build_library_fast(XT, n, args.d_max, idx_patterns)

        Theta_mean = xp.mean(Theta, axis=0, keepdims=True)
        Theta_stdv = xp.std(Theta, axis=0, keepdims=True) + 1e-8
        Thetaz = (Theta - Theta_mean) / Theta_stdv

        Y_mean = xp.mean(YT, axis=0, keepdims=True)
        Y_stdv = xp.std(YT, axis=0, keepdims=True) + 1e-8
        Y_scaled = (YT - Y_mean) / Y_stdv

        Thetaz   = Thetaz.astype(xp.float32, copy=False)
        Y_scaled = Y_scaled.astype(xp.float32, copy=False)
        
        # Window-specific Gram and Cross-correlation matrices
        G, B = precompute_gram(Thetaz, Y_scaled)
        
        T_eff = Thetaz.shape[0]
        G_eff = Thetaz.shape[1]

        if GPU_AVAILABLE:
            med_y = cp.median(Y_scaled, axis=0, keepdims=True)
            mad_y = cp.median(cp.abs(Y_scaled - med_y), axis=0, keepdims=True)
            res_std = float(1.4826 * cp.mean(mad_y))
        else:
            med_y = np.median(Y_scaled, axis=0, keepdims=True)
            mad_y = np.median(np.abs(Y_scaled - med_y), axis=0, keepdims=True)
            res_std = 1.4826 * float(np.mean(mad_y))

        col_std = xp.std(Thetaz, axis=0).astype(xp.float32)
        w_rows = xp.ones(G.shape[0], dtype=xp.float32)
        bad = (col_std < 1e-6)
        if (cp.any(bad).item() if GPU_AVAILABLE else bool(np.any(bad))):
            w_rows = w_rows.copy()
            w_rows[bad] = xp.array(3.0, dtype=w_rows.dtype)
            
        # 3. COMPUTE RUN_FOR_SCALE WITH WINDOW SPECIFIC MATRICES
        lam = args.scale * (res_std / max(1, math.sqrt(T_eff))) * math.sqrt(2.0 * math.log(max(2, G_eff)))
        w = (w_degree_base.astype(xp.float32) * w_rows).astype(xp.float32)
        
        rho_val = args.simpl_rho
        
        Xi_gpu, C_gpu, _ = solve_admm_gl_soc(
            G, B, edges0, lamb=lam, rho_hier=rho_val,
            w=w, max_iters=args.max_iters, rho_admm=args.admm_rho,
            Xi0=None, log_every=0, adaptive_rho=True, overrelax=args.admm_overrelax,
            trace_every=0, return_trace=True
        )

        Xi_gpu = project_rows_dykstra(C_gpu, edges0, rho_val, iters=200, tol=1e-9)

        # 4. FLOORING
        A_main = cp.asnumpy(Xi_gpu) if GPU_AVAILABLE else Xi_gpu
        row_norms = np.linalg.norm(A_main, axis=1)
        w_host = cp.asnumpy(w) if GPU_AVAILABLE else w
        tau_vec = (float(lam) * w_host) / float(args.admm_rho)

        has_active_child = np.zeros(len(row_norms), dtype=bool)
        if 3 in maps0 and maps0[3]:
            for (i, j, k), r_tri in maps0[3].items():
                if row_norms[r_tri] > args.row_norm_nnz_thr:
                    for (a, b) in [(i,j), (i,k), (j,k)]:
                        edge_key = tuple(sorted((a, b)))
                        r_edge = maps0[2].get(edge_key)
                        if r_edge is not None:
                            has_active_child[r_edge] = True
                            
        if 2 in maps0 and maps0[2]:
            for (i, j), r_edge in maps0[2].items():
                if row_norms[r_edge] > args.row_norm_nnz_thr:
                    has_active_child[maps0[1][(i,)]] = True
                    has_active_child[maps0[1][(j,)]] = True

        row_is_lin = np.zeros_like(row_norms, dtype=bool)
        for r in maps0.get(1, {}).values():
            row_is_lin[r] = True
            
        row_is_pair = np.zeros_like(row_norms, dtype=bool)
        for r in maps0.get(2, {}).values():
            row_is_pair[r] = True

        grad = kkt_grad_row_norms(G, B, Xi_gpu)

        cand = np.zeros_like(row_norms, dtype=bool)
        cand[row_is_lin]  |= (row_norms[row_is_lin]  <= 0.8 * tau_vec[row_is_lin])
        cand[row_is_pair] |= (row_norms[row_is_pair] <= 0.7 * tau_vec[row_is_pair])

        pair_parent_active = np.zeros_like(row_norms, dtype=bool)

        safe = grad <= (float(lam) * w_host + 1e-9)
        mask = cand & safe & (~has_active_child) & (~pair_parent_active)

        if np.any(mask):
            if GPU_AVAILABLE:
                Xi_gpu[cp.asarray(mask), :] = 0.0
            else:
                Xi_gpu[mask, :] = 0.0

        # 5. DEBIASING
        A_check = cp.asnumpy(Xi_gpu) if GPU_AVAILABLE else Xi_gpu
        active_mask = (np.linalg.norm(A_check, axis=1) > args.row_norm_nnz_thr)

        if np.any(active_mask):
            Xi_deb = Xi_gpu.copy()

            if GPU_AVAILABLE:
                active_idx = cp.where(cp.asarray(active_mask))[0]
                G_sub = G[cp.ix_(active_idx, active_idx)]
                L = float(cp.linalg.eigvalsh(G_sub)[-1])
            else:
                active_idx = np.where(active_mask)[0]
                G_sub = G[np.ix_(active_idx, active_idx)]
                L = float(np.linalg.eigvalsh(G_sub)[-1])

            alpha = 1.0 / (L + 1e-5)

            for _ in range(300):
                Grad = G @ Xi_deb - B
                Xi_deb = Xi_deb - alpha * Grad

                if GPU_AVAILABLE:
                    Xi_deb[cp.asarray(~active_mask), :] = 0.0
                else:
                    Xi_deb[~active_mask, :] = 0.0

                enforce_hierarchy = bool(getattr(args, "enforce_hierarchy", True))
                if enforce_hierarchy:
                    Xi_deb = enforce_simplicial_hierarchy_clean(
                        Xi_deb,
                        maps0,
                        gpu_available=GPU_AVAILABLE,
                        cp_module=cp if GPU_AVAILABLE else None,
                    )

            Xi_gpu = Xi_deb

        Xi_final = Xi_gpu

        # 6. EXTRACT EDGES AND TRIANGLES
        scores = readout_scores_multi_mode(Xi_final, n, args.d_max, maps0)
        
        # Use our defined tau2_q and tau3_q 
        res = check_extracted_complex_closure(
            scores=scores, 
            n_nodes=n, 
            tau2_q=args.tau2_q, 
            tau3_q=args.tau3_q, 
            S3_key="S3_max"
        )
        
        window_result = _window_result_payload(
            scores=scores,
            res=res,
            n_nodes=n,
            w_start=w_start,
            w_end=w_end,
            window_index=window_index,
        )
        results.append(window_result)
        
    return results
