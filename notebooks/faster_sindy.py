#!/usr/bin/env python3
"""
Modular graph-only (dmax=1) SINDy pipeline for sliding windows.

This module is designed for inputs of shape (N, T):
- N: number of nodes/channels
- T: number of timestamps

For each window, it:
1. builds a linear SINDy library [1, x_0, ..., x_{N-1}],
2. solves with STLSQ,
3. reads undirected edge scores from linear terms,
4. thresholds edges by quantile,
5. adds triangles by clique closure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import math

import numpy as np
from scipy.signal import savgol_filter

try:
    import cupy as cp  # type: ignore

    GPU_AVAILABLE = True
    # Probe a tiny kernel execution so driver/runtime incompatibility is caught early.
    _probe = cp.ones((4,), dtype=cp.float32)
    _probe_sum = float(cp.sum(_probe).item())
    if not np.isfinite(_probe_sum):
        raise RuntimeError("CuPy probe produced a non-finite value")
    print("CuPy runtime probe passed. GPU acceleration enabled.")
except Exception:
    cp = np  # fallback alias so type checks pass in function bodies
    GPU_AVAILABLE = False
    print("CuPy is unavailable or unusable. Falling back to CPU.")

# Force CPU mode override (set by notebook if FORCE_CPU_MODE=True)
FORCE_CPU_OVERRIDE = False

# Final decision: use GPU only if available AND not forced to CPU
if FORCE_CPU_OVERRIDE or not GPU_AVAILABLE:
    xp = np
else:
    xp = cp


@dataclass
class PreprocessConfig:
    fs: float = 256.0
    win_sg: int = 29
    poly_order: int = 3


@dataclass
class SelectionConfig:
    r_target_pc: float = 0.95
    k_min: int = 600
    k_max: int = 1_000_000
    max_rows: int = 6000


@dataclass
class SolverConfig:
    ridge_lambda: float = 1e-3
    stlsq_iters: int = 10
    lambda_scale: float = 1.0
    refit_lambda: float = 1e-6
    auto_relax_if_empty: bool = True
    relax_scale_factors: Tuple[float, ...] = field(default_factory=lambda: (0.7, 0.5, 0.3, 0.2, 0.1))


@dataclass
class ThresholdConfig:
    edge_quantile: float = 0.90


@dataclass
class WindowResult:
    window_id: int
    start: int
    end: int
    n_samples_used: int
    lambda_scale_used: float
    thr_stlsq: float
    tau2: float
    n_edges: int
    n_triangles: int
    nnz_rows: int
    nnz_params: int
    edge_scores: Dict[frozenset, float]
    tri_scores: Dict[frozenset, float]
    pred_edges: set
    pred_tris: set
    Xi: np.ndarray
    S2: np.ndarray
    S2_dir: np.ndarray


def to_numpy(x):
    # Convert only when the active backend provides asnumpy (e.g., CuPy).
    if hasattr(xp, "asnumpy"):
        return xp.asnumpy(x)
    return x


def _validate_input_matrix(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"Expected input shape (N, T), got {X.shape}")
    if X.shape[0] < 2:
        raise ValueError("Input needs at least 2 nodes (N >= 2)")
    if X.shape[1] < 10:
        raise ValueError("Input needs enough timestamps (T >= 10)")
    return X


def preprocess_timeseries(
    X_raw: np.ndarray,
    cfg: PreprocessConfig,
    drop_degenerate: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert raw (N, T) data into normalized X and derivative target Y.

    Returns:
      X: normalized state matrix, shape (N_kept, T_eff)
      Y: normalized derivative matrix, shape (N_kept, T_eff)
    """
    X_raw = _validate_input_matrix(X_raw)

    if drop_degenerate:
        keep = np.ptp(X_raw, axis=1) > 1e-12
        X_raw = X_raw[keep]
        if X_raw.size == 0:
            raise ValueError("All channels are degenerate after filtering")

    if cfg.win_sg % 2 == 0:
        raise ValueError("win_sg must be odd")
    if cfg.win_sg <= cfg.poly_order:
        raise ValueError("win_sg must be larger than poly_order")
    if cfg.win_sg >= X_raw.shape[1]:
        raise ValueError("win_sg must be smaller than number of timestamps")

    dt = 1.0 / cfg.fs
    half = (cfg.win_sg - 1) // 2

    X_smooth = savgol_filter(X_raw, cfg.win_sg, cfg.poly_order, axis=1, mode="interp")[:, half:-half]
    dXdt_raw = savgol_filter(
        X_raw,
        cfg.win_sg,
        cfg.poly_order,
        deriv=1,
        delta=dt,
        axis=1,
        mode="interp",
    )[:, half:-half]

    mu = X_smooth.mean(axis=1, keepdims=True)
    sig = X_smooth.std(axis=1, keepdims=True) + 1e-8

    X = (X_smooth - mu) / sig
    Y = dXdt_raw / sig
    return X, Y


def make_windows(T: int, window_size: int, stride: int) -> List[Tuple[int, int]]:
    """Create [start, end) windows over T samples."""
    if window_size <= 0:
        raise ValueError("window_size must be > 0")
    if stride <= 0:
        raise ValueError("stride must be > 0")
    if T < window_size:
        raise ValueError(f"window_size={window_size} exceeds available samples T={T}")

    out = []
    s = 0
    while s + window_size <= T:
        out.append((s, s + window_size))
        s += stride

    if not out:
        raise ValueError("No valid windows were created")
    return out


def select_local(Xw: np.ndarray, cfg: SelectionConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Robust local sample selector used inside each window."""
    x0 = np.median(Xw, axis=1, keepdims=True)
    Xc = Xw - x0

    sigma = np.maximum(
        1e-8,
        1.4826
        * np.median(
            np.abs(Xc - np.median(Xc, axis=1, keepdims=True)),
            axis=1,
            keepdims=True,
        ),
    )

    d_pc = np.sqrt(((Xc / sigma) ** 2).sum(axis=0)) / np.sqrt(Xc.shape[0])
    keep = np.where(d_pc <= cfg.r_target_pc)[0]

    if keep.size < cfg.k_min:
        keep = np.argsort(d_pc)[: cfg.k_min]
    elif keep.size > cfg.k_max:
        keep = np.argsort(d_pc)[: cfg.k_max]

    return x0, np.sort(keep), d_pc


def build_library_linear(XT):
    """Linear SINDy library for dmax=1: [1, x_0, ..., x_{N-1}]"""
    T = XT.shape[0]
    ones = xp.ones((T, 1), dtype=XT.dtype)
    return xp.concatenate([ones, XT], axis=1)


def precompute_gram(Theta_std, Y_scaled):
    """Compute Gram blocks for normal equations."""
    T = max(1, Theta_std.shape[0])
    G = (Theta_std.T @ Theta_std) / T
    G = G + 5e-3 * xp.eye(G.shape[0], dtype=G.dtype)
    B = (Theta_std.T @ Y_scaled) / T
    return G, B


def solve_stlsq(G, B, lam_ridge=1e-3, thr=1e-2, n_iter=8, keep_const=True):
    """Sequential Thresholded Least Squares (STLSQ) on Gram system."""
    g, n_targets = B.shape
    I = xp.eye(g, dtype=G.dtype)

    Xi = xp.linalg.solve(G + lam_ridge * I, B)

    for _ in range(n_iter):
        Xi_old = Xi.copy()

        for col in range(n_targets):
            support = xp.abs(Xi[:, col]) >= thr
            if keep_const:
                support[0] = True

            idx = xp.where(support)[0]
            if idx.size == 0:
                continue

            Gs = G[xp.ix_(idx, idx)]
            Bs = B[idx, col]
            sol = xp.linalg.solve(
                Gs + lam_ridge * xp.eye(len(idx), dtype=G.dtype),
                Bs,
            )

            Xi[:, col] = 0
            Xi[idx, col] = sol

        diff = float(xp.linalg.norm(Xi - Xi_old))
        if diff < 1e-6:
            break

    return Xi


def refit_on_support(G, B, Xi, lam_ridge=1e-6, keep_const=True):
    """Debias coefficients by refitting on discovered support."""
    _, n_targets = B.shape
    Xi_refit = xp.zeros_like(Xi)

    for col in range(n_targets):
        support = xp.abs(Xi[:, col]) > 0
        if keep_const:
            support[0] = True

        idx = xp.where(support)[0]
        if idx.size == 0:
            continue

        Gs = G[xp.ix_(idx, idx)]
        Bs = B[idx, col]
        sol = xp.linalg.solve(
            Gs + lam_ridge * xp.eye(len(idx), dtype=G.dtype),
            Bs,
        )
        Xi_refit[idx, col] = sol

    return Xi_refit


def readout_graph_scores_avg(Xi, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    From Xi shape (1+N, N):
      directed score src->tgt = |Xi[1+src, tgt]|
      undirected edge score   = average of two directions.
    """
    A = to_numpy(Xi)
    absA = np.abs(A)

    S2_dir = np.zeros((n, n), dtype=np.float64)
    for src in range(n):
        row = 1 + src
        for tgt in range(n):
            if src == tgt:
                continue
            S2_dir[src, tgt] = absA[row, tgt]

    S2 = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            s = 0.5 * (S2_dir[i, j] + S2_dir[j, i])
            S2[i, j] = s
            S2[j, i] = s

    return S2, S2_dir


def choose_edge_threshold(S2: np.ndarray, edge_quantile: float) -> float:
    """Choose edge threshold from positive undirected edge scores."""
    if edge_quantile < 0.0 or edge_quantile > 1.0:
        raise ValueError("edge_quantile must be in [0, 1]")

    n = S2.shape[0]
    vals = [S2[i, j] for i in range(n) for j in range(i + 1, n) if S2[i, j] > 0]
    if not vals:
        return float("inf")
    return float(np.quantile(np.asarray(vals, dtype=np.float64), edge_quantile))


def build_clique_complex_from_graph(
    S2: np.ndarray,
    tau2: float,
) -> Tuple[set, Dict[frozenset, float], set, Dict[frozenset, float]]:
    """
    Keep edges with S2[i, j] >= tau2 and add 2-simplices by clique closure.

    Returns:
      pred_edges, edge_scores, pred_tris, tri_scores
    """
    n = S2.shape[0]

    pred_edges = set()
    edge_scores: Dict[frozenset, float] = {}

    for i in range(n):
        for j in range(i + 1, n):
            s = float(S2[i, j])
            if s >= tau2:
                e = frozenset((i, j))
                pred_edges.add(e)
                edge_scores[e] = s

    pred_tris = set()
    tri_scores: Dict[frozenset, float] = {}

    for i in range(n):
        for j in range(i + 1, n):
            e1 = frozenset((i, j))
            if e1 not in pred_edges:
                continue
            for k in range(j + 1, n):
                e2 = frozenset((i, k))
                e3 = frozenset((j, k))
                if e2 in pred_edges and e3 in pred_edges:
                    tri = frozenset((i, j, k))
                    pred_tris.add(tri)
                    tri_scores[tri] = (edge_scores[e1] + edge_scores[e2] + edge_scores[e3]) / 3.0

    return pred_edges, edge_scores, pred_tris, tri_scores


def nnz_summary(Xi, row_thr=1e-6, param_thr=1e-8) -> Tuple[int, int]:
    A = to_numpy(Xi)
    row_norms = np.linalg.norm(A, axis=1)
    nnz_rows = int(np.sum(row_norms > row_thr))
    nnz_params = int(np.sum(np.abs(A) > param_thr))
    return nnz_rows, nnz_params


def solve_window(
    X_proc: np.ndarray,
    Y_proc: np.ndarray,
    start: int,
    end: int,
    window_id: int,
    sel_cfg: SelectionConfig,
    solver_cfg: SolverConfig,
    thr_cfg: ThresholdConfig,
) -> WindowResult:
    """Run full graph-only dmax=1 SINDy for a single [start, end) window."""
    global GPU_AVAILABLE, xp

    n = X_proc.shape[0]

    Xw = X_proc[:, start:end]
    Yw = Y_proc[:, start:end]

    x0, keep, _ = select_local(Xw, sel_cfg)
    Xw_c = Xw - x0

    Xw_use = Xw_c[:, keep]
    Yw_use = Yw[:, keep]

    if Xw_use.shape[1] > sel_cfg.max_rows:
        idx = np.linspace(0, Xw_use.shape[1] - 1, sel_cfg.max_rows, dtype=int)
        Xw_use = Xw_use[:, idx]
        Yw_use = Yw_use[:, idx]

    if Xw_use.shape[1] < 5:
        raise ValueError(f"Window {window_id}: too few usable samples after selection")

    try:
        # Use xp (which may be numpy if FORCE_CPU_OVERRIDE is set)
        XT = xp.asarray(Xw_use.T)
        YT = xp.asarray(Yw_use.T)

        Theta = build_library_linear(XT)

        Theta_mean = xp.mean(Theta, axis=0, keepdims=True)
        Theta_stdv = xp.std(Theta, axis=0, keepdims=True) + 1e-8
        Thetaz = (Theta - Theta_mean) / Theta_stdv

        Y_mean = xp.mean(YT, axis=0, keepdims=True)
        Y_stdv = xp.std(YT, axis=0, keepdims=True) + 1e-8
        Y_scaled = (YT - Y_mean) / Y_stdv

        Thetaz = Thetaz.astype(xp.float32, copy=False)
        Y_scaled = Y_scaled.astype(xp.float32, copy=False)

        G, B = precompute_gram(Thetaz, Y_scaled)

        T_eff = Thetaz.shape[0]
        G_eff = Thetaz.shape[1]

        med = xp.median(Y_scaled, axis=0, keepdims=True)
        mad = xp.median(xp.abs(Y_scaled - med), axis=0, keepdims=True)
        res_std = float(1.4826 * xp.mean(mad))

        base_thr = (res_std / max(1, math.sqrt(T_eff))) * math.sqrt(2.0 * math.log(max(2, G_eff)))

        def fit_for_scale(scale: float):
            thr_loc = float(scale * base_thr)
            Xi_loc = solve_stlsq(
                G,
                B,
                lam_ridge=solver_cfg.ridge_lambda,
                thr=thr_loc,
                n_iter=solver_cfg.stlsq_iters,
                keep_const=True,
            )
            Xi_loc = refit_on_support(
                G,
                B,
                Xi_loc,
                lam_ridge=solver_cfg.refit_lambda,
                keep_const=True,
            )
            nnz_rows_loc, nnz_params_loc = nnz_summary(Xi_loc)
            return Xi_loc, thr_loc, nnz_rows_loc, nnz_params_loc

        scale_used = float(solver_cfg.lambda_scale)
        Xi, thr, nnz_rows, nnz_params = fit_for_scale(scale_used)

        if solver_cfg.auto_relax_if_empty and nnz_params == 0:
            for factor in solver_cfg.relax_scale_factors:
                scale_try = float(solver_cfg.lambda_scale * factor)
                Xi_try, thr_try, nnz_rows_try, nnz_params_try = fit_for_scale(scale_try)
                if nnz_params_try > 0:
                    Xi, thr, nnz_rows, nnz_params = Xi_try, thr_try, nnz_rows_try, nnz_params_try
                    scale_used = scale_try
                    break

        S2, S2_dir = readout_graph_scores_avg(Xi, n)
        tau2 = choose_edge_threshold(S2, thr_cfg.edge_quantile)
        pred_edges, edge_scores, pred_tris, tri_scores = build_clique_complex_from_graph(S2, tau2)

        return WindowResult(
            window_id=window_id,
            start=start,
            end=end,
            n_samples_used=int(Xw_use.shape[1]),
            lambda_scale_used=scale_used,
            thr_stlsq=float(thr),
            tau2=float(tau2),
            n_edges=int(len(pred_edges)),
            n_triangles=int(len(pred_tris)),
            nnz_rows=int(nnz_rows),
            nnz_params=int(nnz_params),
            edge_scores=edge_scores,
            tri_scores=tri_scores,
            pred_edges=pred_edges,
            pred_tris=pred_tris,
            Xi=to_numpy(Xi),
            S2=S2,
            S2_dir=S2_dir,
        )
    except Exception as e:
        if GPU_AVAILABLE:
            err_txt = str(e).upper()
            if "CUDA" in err_txt or "CUPY" in type(e).__module__.lower():
                print(
                    f"Window {window_id}: GPU runtime failure ({type(e).__name__}). "
                    "Switching to CPU for this and remaining windows."
                )
                GPU_AVAILABLE = False
                xp = np
                return solve_window(X_proc, Y_proc, start, end, window_id, sel_cfg, solver_cfg, thr_cfg)
        raise


def run_sindy_windows(
    X_raw: np.ndarray,
    window_size: int,
    stride: int,
    preprocess_cfg: Optional[PreprocessConfig] = None,
    selection_cfg: Optional[SelectionConfig] = None,
    solver_cfg: Optional[SolverConfig] = None,
    threshold_cfg: Optional[ThresholdConfig] = None,
) -> List[WindowResult]:
    """
    Main entrypoint: run graph-only dmax=1 SINDy over sliding windows.

    Args:
      X_raw: array of shape (N, T)
      window_size: window length in samples
      stride: stride in samples
      preprocess_cfg: smoothing + derivative config
      selection_cfg: robust local sample selection config
      solver_cfg: STLSQ config
      threshold_cfg: edge threshold config

    Returns:
      List of WindowResult, one per window.
    """
    preprocess_cfg = preprocess_cfg or PreprocessConfig()
    selection_cfg = selection_cfg or SelectionConfig()
    solver_cfg = solver_cfg or SolverConfig()
    threshold_cfg = threshold_cfg or ThresholdConfig()

    X_proc, Y_proc = preprocess_timeseries(X_raw, preprocess_cfg, drop_degenerate=True)
    _, T_eff = X_proc.shape
    windows = make_windows(T_eff, window_size, stride)

    results: List[WindowResult] = []
    for i, (start, end) in enumerate(windows):
        res = solve_window(
            X_proc,
            Y_proc,
            start,
            end,
            window_id=i,
            sel_cfg=selection_cfg,
            solver_cfg=solver_cfg,
            thr_cfg=threshold_cfg,
        )
        results.append(res)

    return results


def results_to_records(results: Sequence[WindowResult]) -> List[dict]:
    """Flatten window results into simple dict records for DataFrame/CSV export."""
    rows = []
    for r in results:
        rows.append(
            {
                "window_id": r.window_id,
                "start": r.start,
                "end": r.end,
                "n_samples_used": r.n_samples_used,
                "lambda_scale_used": r.lambda_scale_used,
                "thr_stlsq": r.thr_stlsq,
                "tau2": r.tau2,
                "n_edges": r.n_edges,
                "n_triangles": r.n_triangles,
                "nnz_rows": r.nnz_rows,
                "nnz_params": r.nnz_params,
            }
        )
    return rows


def save_window_graphs(results: Sequence[WindowResult], outdir: str) -> None:
    """Save Xi, edge lists, and triangle lists for each window."""
    import os

    os.makedirs(outdir, exist_ok=True)

    for r in results:
        prefix = f"window_{r.window_id:04d}"

        np.save(os.path.join(outdir, f"{prefix}_Xi.npy"), r.Xi)
        np.save(os.path.join(outdir, f"{prefix}_S2.npy"), r.S2)
        np.save(os.path.join(outdir, f"{prefix}_S2_dir.npy"), r.S2_dir)

        with open(os.path.join(outdir, f"{prefix}_edges.txt"), "w", encoding="utf-8") as f:
            f.write("Predicted edges with scores\n")
            f.write("=" * 40 + "\n")
            for e, s in sorted(r.edge_scores.items(), key=lambda z: (-z[1], tuple(sorted(z[0])))):
                f.write(f"{tuple(sorted(e))}: {s:.8f}\n")

        with open(os.path.join(outdir, f"{prefix}_triangles.txt"), "w", encoding="utf-8") as f:
            f.write("Predicted clique triangles with average edge scores\n")
            f.write("=" * 60 + "\n")
            for tri, s in sorted(r.tri_scores.items(), key=lambda z: (-z[1], tuple(sorted(z[0])))):
                f.write(f"{tuple(sorted(tri))}: {s:.8f}\n")


if __name__ == "__main__":
    # Minimal usage example with random data (replace with your real matrix of shape N x T).
    rng = np.random.default_rng(0)
    N, T = 10, 20_000
    X_demo = rng.normal(size=(N, T))

    results = run_sindy_windows(
        X_demo,
        window_size=5120,
        stride=2560,
        preprocess_cfg=PreprocessConfig(fs=256.0, win_sg=29, poly_order=3),
        selection_cfg=SelectionConfig(r_target_pc=0.95, k_min=600, k_max=1_000_000, max_rows=6000),
        solver_cfg=SolverConfig(ridge_lambda=1e-3, stlsq_iters=10, lambda_scale=1.0, refit_lambda=1e-6),
        threshold_cfg=ThresholdConfig(edge_quantile=0.90),
    )

    print(f"Backend: {'GPU' if GPU_AVAILABLE else 'CPU'}")
    print(f"Windows solved: {len(results)}")
    if results:
        print(
            {
                "window_id": results[0].window_id,
                "start": results[0].start,
                "end": results[0].end,
                "n_edges": results[0].n_edges,
                "n_triangles": results[0].n_triangles,
            }
        )
