from __future__ import annotations

import argparse
import os
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm
import numpy as np
import torch
from scipy.io import loadmat
from scipy.signal import welch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from Network import create_model

try:
    from torcheeg.datasets.constants import SEED_ADJACENCY_MATRIX
except Exception:
    SEED_ADJACENCY_MATRIX = None


SEED_VII_ROOT = "/hdfs1/Data/Shubhajit/Project/New_TETON/New_code/SEED-VII"
EEG_PREPROCESSED_DIR = Path(SEED_VII_ROOT) / "EEG_preprocessed"
EMOTION_LABEL_FILE = Path(SEED_VII_ROOT) / "emotion_label_and_stimuli_order.xlsx"
CACHE_DIR = Path(__file__).resolve().parent / ".cache"

FS = 200
BANDS: Tuple[Tuple[int, int], ...] = ((1, 4), (4, 8), (8, 14), (14, 31), (31, 50))


class SimplicialComplex:
    def __init__(self, windows, label):
        self.windows = windows
        self.label = label

def single_subject_collate_fn(batch):
    """Return one subject sample as a dict for the simplicial models.

    The simplicial networks in Network.py expect a subject-level mapping with a
    "windows" key containing the per-window simplicial snapshots.
    """
    if len(batch) != 1:
        raise ValueError("single_subject_collate_fn expects batch_size=1")
    sample = batch[0]
    return {
        "windows": sample.windows,
        "label": torch.tensor(int(sample.label), dtype=torch.long),
        "subject_id": getattr(sample, "subject_id", "unknown"),
        "n_windows": len(sample.windows),
    }

def _xlsx_shared_strings(z: zipfile.ZipFile) -> List[str]:
    path = "xl/sharedStrings.xml"
    if path not in z.namelist():
        return []
    root = ET.fromstring(z.read(path))
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    out: List[str] = []
    for si in root.iter(f"{ns}si"):
        texts = [t.text or "" for t in si.iter(f"{ns}t")]
        out.append("".join(texts))
    return out


def _xlsx_sheet_cells(z: zipfile.ZipFile, sheet_rel_path: str) -> Dict[str, str]:
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    shared = _xlsx_shared_strings(z)
    root = ET.fromstring(z.read(sheet_rel_path))
    cells: Dict[str, str] = {}
    for c in root.iter(f"{ns}c"):
        ref = c.attrib.get("r", "")
        t = c.attrib.get("t", "")
        v = c.find(f"{ns}v")
        is_node = c.find(f"{ns}is")
        if t == "s" and v is not None and v.text is not None:
            idx = int(v.text)
            cells[ref] = shared[idx] if 0 <= idx < len(shared) else ""
        elif t == "inlineStr" and is_node is not None:
            t_node = is_node.find(f"{ns}t")
            cells[ref] = (t_node.text or "") if t_node is not None else ""
        elif v is not None and v.text is not None:
            cells[ref] = v.text
    return cells


def load_seed_vii_emotion_labels(label_file: Path) -> Tuple[List[int], Dict[str, int]]:
    with zipfile.ZipFile(label_file) as z:
        cells = _xlsx_sheet_cells(z, "xl/worksheets/sheet1.xml")

    session_rows = []
    for ref, value in cells.items():
        m = re.fullmatch(r"A(\d+)", ref)
        if m and value.strip().lower().startswith("session"):
            session_rows.append(int(m.group(1)))
    session_rows = sorted(session_rows)
    if len(session_rows) != 4:
        raise ValueError(f"Expected 4 session rows, found {len(session_rows)} in {label_file}")

    labels_text: List[str] = []
    for row in session_rows:
        for col in "BCDEFGHIJKLMNOPQRSTU":
            val = cells.get(f"{col}{row}", "").strip()
            if val:
                labels_text.append(val)
    if len(labels_text) != 80:
        raise ValueError(f"Expected 80 trial labels, found {len(labels_text)}")

    emo2id: Dict[str, int] = {}
    for e in labels_text:
        if e not in emo2id:
            emo2id[e] = len(emo2id)
    labels = [emo2id[e] for e in labels_text]
    return labels, emo2id


def fallback_ring_adjacency(n_nodes: int = 62) -> np.ndarray:
    a = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for i in range(n_nodes):
        j = (i + 1) % n_nodes
        a[i, j] = 1
        a[j, i] = 1
    return a


def build_static_edge_index() -> torch.Tensor:
    if SEED_ADJACENCY_MATRIX is not None:
        adj = np.asarray(SEED_ADJACENCY_MATRIX, dtype=np.float32)
    else:
        adj = fallback_ring_adjacency(62)
    rows, cols = np.nonzero(adj)
    return torch.tensor(np.vstack([rows, cols]), dtype=torch.long)


def build_knn_edge_index(eeg_window: np.ndarray, k: int) -> torch.Tensor:
    # eeg_window: [62, T]
    corr = np.corrcoef(eeg_window)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 0.0)
    score = np.abs(corr)
    n = score.shape[0]
    edges = set()
    for i in range(n):
        nn = np.argpartition(-score[i], kth=min(k, n - 1) - 1)[:k]
        for j in nn:
            if i == j:
                continue
            a, b = (i, int(j)) if i < j else (int(j), i)
            edges.add((a, b))
    rows, cols = [], []
    for a, b in edges:
        rows.extend([a, b])
        cols.extend([b, a])
    return torch.tensor(np.vstack([rows, cols]), dtype=torch.long)


def window_to_features(eeg_window: np.ndarray) -> torch.Tensor:
    # eeg_window: [62, W] -> x: [62,5]
    freqs, psd = welch(eeg_window, fs=FS, nperseg=min(FS, eeg_window.shape[1]), axis=1)
    feats = []
    for low, high in BANDS:
        mask = (freqs >= low) & (freqs < high)
        power = np.trapezoid(psd[:, mask], freqs[mask], axis=1)
        feats.append(np.log(power + 1e-8))
    return torch.tensor(np.stack(feats, axis=1), dtype=torch.float32)

def construct_topological_snapshot(
    node_features: np.ndarray,
    edge_list: List[Any],
    triangle_list: List[Any],
    edge_score: Optional[Any] = None,
    triangle_score: Optional[Any] = None,
    agg_func: str = "mean",
) -> Tuple[Dict[int, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], List[Tuple[int, int]], List[Tuple[int, int, int]]]:
    """Construct lifted features, incidence matrices, and adjacency matrices for one window."""
    x0 = np.asarray(node_features, dtype=np.float32)
    if x0.ndim == 1:
        x0 = x0[:, None]
    if x0.ndim != 2:
        raise ValueError(f"node_features must be 1D or 2D, got shape {x0.shape}")

    n_nodes, feat_dim = x0.shape

    def _aggregate(values: np.ndarray) -> np.ndarray:
        if agg_func == "mean":
            return values.mean(axis=0)
        if agg_func == "sum":
            return values.sum(axis=0)
        if agg_func == "max":
            return values.max(axis=0)
        raise ValueError("agg_func must be 'mean', 'sum', or 'max'")

    def _parse_edge_item(item: Any) -> Tuple[int, int]:
        # Supported forms for backward compatibility:
        # - (u, v)
        # - (u, v, score)
        # - ((u, v), score)
        if not isinstance(item, (tuple, list)):
            raise ValueError(f"Invalid edge specification: {item}")
        if len(item) == 2 and isinstance(item[0], (tuple, list)) and len(item[0]) == 2:
            u, v = int(item[0][0]), int(item[0][1])
            return tuple(sorted((u, v)))
        if len(item) >= 2:
            u, v = int(item[0]), int(item[1])
            return tuple(sorted((u, v)))
        raise ValueError(f"Invalid edge specification: {item}")

    def _parse_triangle_item(item: Any) -> Tuple[int, int, int]:
        # Supported forms for backward compatibility:
        # - (u, v, w)
        # - (u, v, w, score)
        # - ((u, v, w), score)
        if not isinstance(item, (tuple, list)):
            raise ValueError(f"Invalid triangle specification: {item}")
        if len(item) == 2 and isinstance(item[0], (tuple, list)) and len(item[0]) == 3:
            u, v, w = int(item[0][0]), int(item[0][1]), int(item[0][2])
            return tuple(sorted((u, v, w)))
        if len(item) >= 3:
            u, v, w = int(item[0]), int(item[1]), int(item[2])
            return tuple(sorted((u, v, w)))
        raise ValueError(f"Invalid triangle specification: {item}")

    edges = sorted({_parse_edge_item(edge_item) for edge_item in edge_list})
    triangles_all = sorted({_parse_triangle_item(tri_item) for tri_item in triangle_list})

    def _normalize_simplex_score_input(
        simplices: List[Any],
        scores: Any,
        simplex_name: str,
    ) -> Dict[Any, float]:
        if scores is None:
            return {}

        if isinstance(scores, dict):
            out: Dict[Any, float] = {}
            for key, value in scores.items():
                if simplex_name == "edge":
                    if not isinstance(key, (tuple, list)) or len(key) != 2:
                        raise ValueError(f"{simplex_name}_score dict keys must be 2-tuples")
                    parsed_key = tuple(sorted((int(key[0]), int(key[1]))))
                else:
                    if not isinstance(key, (tuple, list)) or len(key) != 3:
                        raise ValueError(f"{simplex_name}_score dict keys must be 3-tuples")
                    parsed_key = tuple(sorted((int(key[0]), int(key[1]), int(key[2]))))
                out[parsed_key] = float(value)
            return out

        if isinstance(scores, (list, tuple, np.ndarray)):
            if len(scores) != len(simplices):
                raise ValueError(
                    f"{simplex_name}_score length ({len(scores)}) must match "
                    f"{simplex_name}_list length ({len(simplices)})"
                )
            out = {}
            for item, value in zip(simplices, scores):
                key = _parse_edge_item(item) if simplex_name == "edge" else _parse_triangle_item(item)
                out[key] = max(out.get(key, float("-inf")), float(value))
            return out

        raise ValueError(
            f"{simplex_name}_score must be None, a dict, or a sequence aligned with {simplex_name}_list"
        )

    edge_score_map: Dict[Tuple[int, int], float] = {edge: 1.0 for edge in edges}
    provided_edge_scores = _normalize_simplex_score_input(edge_list, edge_score, "edge")
    if provided_edge_scores:
        for e in edges:
            if e in provided_edge_scores:
                edge_score_map[e] = provided_edge_scores[e]

    tri_score_map: Dict[Tuple[int, int, int], float] = {tri: 1.0 for tri in triangles_all}
    provided_triangle_scores = _normalize_simplex_score_input(triangle_list, triangle_score, "triangle")

    if provided_triangle_scores:
        for tri in triangles_all:
            if tri in provided_triangle_scores:
                tri_score_map[tri] = provided_triangle_scores[tri]
    elif provided_edge_scores:
        for tri in triangles_all:
            u, v, w = tri
            e1 = tuple(sorted((u, v)))
            e2 = tuple(sorted((v, w)))
            e3 = tuple(sorted((u, w)))
            edge_vals = np.asarray(
                [edge_score_map.get(e1, 1.0), edge_score_map.get(e2, 1.0), edge_score_map.get(e3, 1.0)],
                dtype=np.float32,
            )
            tri_score_map[tri] = float(_aggregate(edge_vals))

    edge_to_idx = {edge: idx for idx, edge in enumerate(edges)}

    if edges:
        x1 = np.stack(
            [edge_score_map[(u, v)] * _aggregate(x0[[u, v]]) for (u, v) in edges],
            axis=0,
        ).astype(np.float32)
    else:
        x1 = np.zeros((0, feat_dim), dtype=np.float32)

    # B1: nodes -> edges
    n_edges = len(edges)
    B1 = np.zeros((n_nodes, n_edges), dtype=np.float32)
    for e_idx, (u, v) in enumerate(edges):
        B1[u, e_idx] = 1.0
        B1[v, e_idx] = 1.0

    # Keep only triangles whose edges exist in edge_list.
    valid_triangles: List[Tuple[int, int, int]] = []
    for tri in tri_score_map.keys():
        u, v, w = tri
        e1 = tuple(sorted((u, v)))
        e2 = tuple(sorted((v, w)))
        e3 = tuple(sorted((u, w)))
        if e1 in edge_to_idx and e2 in edge_to_idx and e3 in edge_to_idx:
            valid_triangles.append((u, v, w))

    n_triangles = len(valid_triangles)
    B2 = np.zeros((n_edges, n_triangles), dtype=np.float32)
    for t_idx, (u, v, w) in enumerate(valid_triangles):
        B2[edge_to_idx[tuple(sorted((u, v)))], t_idx] = 1.0
        B2[edge_to_idx[tuple(sorted((v, w)))], t_idx] = 1.0
        B2[edge_to_idx[tuple(sorted((u, w)))], t_idx] = 1.0

    if valid_triangles:
        x2 = np.stack(
            [tri_score_map[(u, v, w)] * _aggregate(x0[[u, v, w]]) for (u, v, w) in valid_triangles],
            axis=0,
        ).astype(np.float32)
    else:
        x2 = np.zeros((0, feat_dim), dtype=np.float32)

    A0 = (B1 @ B1.T).astype(np.float32)
    A1 = (B1.T @ B1 + B2 @ B2.T).astype(np.float32)
    A2 = (B2.T @ B2).astype(np.float32)

    features = {0: x0, 1: x1, 2: x2}
    incidences = {
        "rank_1": B1,
        "rank_2": B2,
    }
    adjacencies = {
        "rank_0": A0,
        "rank_1": A1,
        "rank_2": A2,
    }
    return features, incidences, adjacencies, edges, valid_triangles



def build_simplicies(
    eeg_dir: Path,
    labels_80: List[int],
    graph_method: str,
    knn_k: int,
    window_size: int,
    window_stride: int,
    max_subjects: Optional[int],
    cache_tag: str,
    use_cache: bool,
    args: argparse.Namespace
) -> List[Any]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"seed_vii_{args.sindy_backend}_{cache_tag}.pt"
    if use_cache and cache_path.exists():
        try:
            bundle = torch.load(cache_path, map_location="cpu", weights_only=False)
            return bundle["simplicies"]
        except Exception as e:
            print(f"[WARN] Failed to read cache {cache_path}: {e}")
            print("[WARN] Cache appears corrupted. Deleting and rebuilding...")
            try:
                cache_path.unlink()
            except Exception as del_err:
                print(f"[WARN] Could not delete corrupted cache: {del_err}")

    static_edge = build_static_edge_index()
    simplices: List[Dict] = []

    files = sorted(eeg_dir.glob("*.mat"), key=lambda p: int(p.stem))
    if max_subjects is not None:
        files = files[: max(1, max_subjects)]
    if args.sindy_backend == "parallel_sindy":
        from parallel_sindy import process_data_in_windows
        
    else:
        from faster_sindy import run_sindy_windows, SelectionConfig, SolverConfig, ThresholdConfig
    from faster_sindy import preprocess_timeseries, PreprocessConfig, SelectionConfig, SolverConfig, ThresholdConfig
    preprocess_cfg = PreprocessConfig(fs=args.fs, win_sg=args.win_sg, poly_order=3)
    selection_cfg = SelectionConfig(k_min=args.k_min, r_target_pc=args.r_target_pc)
    solver_cfg = SolverConfig()
    threshold_cfg = ThresholdConfig(edge_quantile=0.9)
    for mat_path in tqdm(files):
        subject_id = int(mat_path.stem)
        mat = loadmat(mat_path)
        for trial_id in tqdm(range(1, 81)):
            key = str(trial_id)
            if key not in mat:
                continue
            eeg_trial = np.asarray(mat[key], dtype=np.float32)  # [62, T]
            X_proc, _ = preprocess_timeseries(eeg_trial, preprocess_cfg, drop_degenerate=True)
            T = eeg_trial.shape[1]
            if T < window_size:
                continue
            trial_windows = {}
            if args.sindy_backend in ("sindy", "parallel_sindy"):
                windows = process_data_in_windows(eeg_trial, args)
                for w_idx, results in enumerate(windows):
                    start = int(results["window_start"])
                    end = int(results["window_end"])

                    node_temporal = X_proc[:, start:end].astype(np.float32, copy=False)
                    node_temporal = window_to_features(node_temporal)
                    edges_in = [tuple(sorted(tuple(edge))) for edge in results["edges"]]
                    triangles_in = [tuple(sorted(tuple(tri))) for tri in results["triangles"]]

                    
                    triangle_score = results["triangle_features"]["vector"]
                    edge_score = results["edge_features"]["vector"]

                    features, incidences, adjacencies, edges, triangles = construct_topological_snapshot(
                        node_features=node_temporal,
                        edge_list=edges_in,
                        triangle_list=triangles_in,
                        edge_score=edge_score,
                        triangle_score=triangle_score,
                        agg_func="mean",
                    )

                    trial_windows[w_idx] = {"features": features, "incidences": incidences, "adjacencies": adjacencies, "edges": edges, "triangles": triangles}
                    print(f"Subject {subject_id} Trial {trial_id} Window {w_idx}: edges {len(edges)}, triangles {len(triangles)}")
            else:

                windows = run_sindy_windows(X_raw=eeg_trial, window_size=window_size, stride=window_stride, preprocess_cfg=preprocess_cfg, selection_cfg=selection_cfg, solver_cfg=solver_cfg, threshold_cfg=threshold_cfg)
                for w_idx, results in enumerate(windows):
                    start = int(results.start)
                    end = int(results.end)

                    node_temporal = X_proc[:, start:end].astype(np.float32, copy=False)
                    node_temporal = window_to_features(node_temporal)

                    edges_in = [tuple(sorted(tuple(edge))) for edge in results.pred_edges]
                    triangles_in = [tuple(sorted(tuple(tri))) for tri in results.pred_tris]

                    features, incidences, adjacencies, edges, triangles = construct_topological_snapshot(
                        node_features=node_temporal,
                        edge_list=edges_in,
                        triangle_list=triangles_in,
                        agg_func="mean",
                    )

                    trial_windows[w_idx] = {"features": features, "incidences": incidences, "adjacencies": adjacencies, "edges": edges, "triangles": triangles}
                    print(f"Subject {subject_id} Trial {trial_id} Window {w_idx}: edges {len(edges)}, triangles {len(triangles)}")
            simplices_data = SimplicialComplex(windows=trial_windows, label=labels_80[trial_id - 1])
            simplices.append(simplices_data)

    if not simplices:
        raise ValueError("No simplices were created. Try smaller --window_sec or lower --max_subjects debug settings.")

    if use_cache:
        try:
            torch.save({"simplicies": simplices}, cache_path)
            print(f"Cached simplicies to: {cache_path}")
        except Exception as e:
            print(f"[WARN] Cache write failed at {cache_path}: {e}")
            print("[WARN] Continuing without cache.")
    return simplices


def normalize_simplicies_features(train_ds: List[Any], val_ds: List[Any], test_ds: List[Any]) -> None:
    all_x0 = torch.cat([torch.tensor(d.windows[w]["features"][0]) for d in train_ds for w in d.windows], dim=0)
    all_x1 = torch.cat([torch.tensor(d.windows[w]["features"][1]) for d in train_ds for w in d.windows], dim=0)
    all_x2 = torch.cat([torch.tensor(d.windows[w]["features"][2]) for d in train_ds for w in d.windows], dim=0)
    mu0 = all_x0.mean(dim=0, keepdim=True)
    sigma0 = all_x0.std(dim=0, keepdim=True).clamp_min(1e-6)
    mu1 = all_x1.mean(dim=0, keepdim=True)
    sigma1 = all_x1.std(dim=0, keepdim=True).clamp_min(1e-6)
    mu2 = all_x2.mean(dim=0, keepdim=True)
    sigma2 = all_x2.std(dim=0, keepdim=True).clamp_min(1e-6)
    for ds in (train_ds, val_ds, test_ds):
        for d in ds:
            for w in d.windows:
                d.windows[w]["features"][0] = (torch.tensor(d.windows[w]["features"][0]) - mu0) / sigma0
                d.windows[w]["features"][1] = (torch.tensor(d.windows[w]["features"][1]) - mu1) / sigma1
                d.windows[w]["features"][2] = (torch.tensor(d.windows[w]["features"][2]) - mu2) / sigma2


def make_model(
    model_name: str,
    in_features: int,
    num_classes: int,
    hidden: int,
    dropout: float,
    num_layers: int
):
    return create_model(
        model_name=model_name,
        input_dim=in_features,
        sccn_hidden=hidden,
        num_sccn_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
    )


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, criterion: torch.nn.Module, device: torch.device):
    model.eval()
    use_amp = device.type == "cuda"
    total_loss, total_correct, total, n_batches = 0.0, 0, 0, 0
    pbar = tqdm(loader, desc="Evaluating")
    for sample in pbar:
        label = sample["label"].to(device).view(-1)
        with torch.amp.autocast(enabled=use_amp, device_type='cuda'):
            logits = model(sample["windows"])
            loss = criterion(logits, label)
        pred = logits.argmax(dim=1)
        total_correct += int((pred == label).sum().item())
        total += label.numel()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1), total_correct / max(total, 1)


def train_once(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    args: argparse.Namespace
):
    labels = [int(d.label) for d in train_loader.dataset]
    counts = np.bincount(labels)
    weights = 1.0 / np.clip(counts, 1, None)
    weights = weights * (len(counts) / weights.sum())
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=20)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_val, best_state, stale = -1.0, None, 0
    for epoch in range(epochs + 1):
        model.train()
        total_loss, total_correct, total, n_batches = 0.0, 0, 0, 0
        pending_logits = []
        pending_labels = []
        all_preds = []
        all_labels = []
        pbar = tqdm(train_loader, desc="Training")
        for sample_idx, sample in enumerate(pbar):
            label = sample["label"].to(device).view(-1)
            optimizer.zero_grad()
            with torch.amp.autocast(enabled=use_amp, device_type='cuda'):
                logits = model(sample["windows"])
                pending_logits.append(logits)
                pending_labels.append(label)
            
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(label.cpu().numpy())

            if len(pending_logits) >= args.batch_size or sample_idx == len(train_loader) - 1:
                batch_logits = torch.cat(pending_logits, dim=0)
                batch_labels = torch.cat(pending_labels, dim=0)
                pending_logits.clear()
                pending_labels.clear()

                loss = criterion(batch_logits.float(), batch_labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
                total_correct += int((batch_logits.argmax(dim=1) == batch_labels).sum().item())
                total += batch_labels.numel()
            
                n_batches += 1

        tr_loss = total_loss / max(n_batches, 1)
        tr_acc = total_correct / max(total, 1)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_acc)

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1

        if epoch % 25 == 0:
            print(
                f"Epoch {epoch:>4} | train_loss {tr_loss:.4f} | train_acc {tr_acc*100:5.2f}% "
                f"| val_loss {val_loss:.4f} | val_acc {val_acc*100:5.2f}%"
            )
        if stale >= patience:
            print(f"Early stop at epoch {epoch}, no val improvement for {patience} epochs.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    return best_val, test_loss, test_acc


def parse_args():
    p = argparse.ArgumentParser(description="SEED-VII GCN/GIN with KNN graph + Optuna tuning")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=32, help="Gradient accumulation steps for one-subject batches.")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.35)
    p.add_argument(
        "--model",
        type=str,
        choices=["SCCN_LSTM", "SCCN_Pool", "SCCN_Attention", "SCCN_Transformer"],
        default="SCCN_Pool",
    )
    p.add_argument("--graph_method", type=str, choices=["seed_static", "knn_corr"], default="seed_static")
    p.add_argument("--knn_k", type=int, default=8)
    p.add_argument("--window_sec", type=float, default=16.0)
    p.add_argument("--stride_sec", type=float, default=8.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_subjects", type=int, default=None)
    p.add_argument("--patience", type=int, default=120)
    p.add_argument("--use_optuna", action="store_true")
    p.add_argument("--optuna_trials", type=int, default=30)
    p.add_argument("--optuna_epochs", type=int, default=120)
    p.add_argument("--no_cache", action="store_true", help="Disable graph cache read/write.")
    p.add_argument("--sindy_backend", type=str, choices=["sindy", "parallel_sindy", "faster_sindy"], default="parallel_sindy")
    p.add_argument(
        "--projection_layer",
        action="store_true",
        help="Enable the learned projection path in Network.py; otherwise use raw aligned features.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    labels_80, emo_map = load_seed_vii_emotion_labels(EMOTION_LABEL_FILE)
    print(f"Loaded labels for 80 trials. Emotion map: {emo_map}")

    args.fs = FS
    args.win_sg = 29
    args.order = 3
    args.d_max = 2
    args.stride = int(FS * args.stride_sec)
    args.win_len = int(FS * args.window_sec)
    args.r_target_pc = 0.99
    args.k_min = 700
    args.k_max = 1000000
    args.row_norm_nnz_thr = 1e-6
    args.param_abs_nnz_thr = 1e-8
    args.tau2_q = 0.75
    args.tau3_q = 0.75
    args.scale = 1.0
    args.simpl_rho = 1.0
    args.admm_rho = 3.0
    args.admm_overrelax = 1.6
    args.max_iters = 250

    win = int(FS * args.window_sec)
    stride = int(FS * args.stride_sec)
    cache_tag = f"raw_{args.graph_method}_k{args.knn_k}_w{win}_s{stride}_sub{args.max_subjects or 'all'}"
    dataset = build_simplicies(
        EEG_PREPROCESSED_DIR,
        labels_80,
        args.graph_method,
        args.knn_k,
        win,
        stride,
        args.max_subjects,
        cache_tag,
        use_cache=not args.no_cache,
        args=args
    )
    print(f"Simplicies: {len(dataset)}")

    y_all = [int(d.label) for d in dataset]
    idx = np.arange(len(dataset))
    train_idx, test_idx = train_test_split(idx, test_size=0.1, random_state=args.seed, stratify=y_all)
    y_train_tmp = [y_all[i] for i in train_idx]
    train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=args.seed, stratify=y_train_tmp)
    train_ds = [dataset[i] for i in train_idx]
    val_ds = [dataset[i] for i in val_idx]
    test_ds = [dataset[i] for i in test_idx]
    normalize_simplicies_features(train_ds, val_ds, test_ds)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU unavailable; using CPU.")

    num_workers = min(8, max(1, (os.cpu_count() or 2) // 2))
    loader_kwargs = {"num_workers": num_workers, "pin_memory": use_cuda, "persistent_workers": num_workers > 0}
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 4

    in_features = train_ds[0].windows[0]["features"][0].shape[1]
    n_classes = len(emo_map)
    projection_type = "ffnn" if args.projection_layer else "none"

    if args.use_optuna:
        try:
            import importlib

            optuna = importlib.import_module("optuna")
        except Exception as e:
            raise RuntimeError(
                "Optuna is not installed. Install it with: pip install optuna"
            ) from e

        def objective(trial):
            model_name = trial.suggest_categorical("model", ["gcn", "gin"])
            hidden = trial.suggest_int("hidden", 64, 256, step=32)
            num_layers = trial.suggest_int("num_layers", 2, 6)
            dropout = trial.suggest_float("dropout", 0.1, 0.6)
            lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
            wd = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
            bs = trial.suggest_categorical("batch_size", [64, 128, 256])

            trial_args = argparse.Namespace(**vars(args))
            trial_args.batch_size = bs

            tr_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=single_subject_collate_fn, **loader_kwargs)
            va_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=single_subject_collate_fn, **loader_kwargs)
            te_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=single_subject_collate_fn, **loader_kwargs)

            model = make_model(model_name, in_features, n_classes, hidden, dropout, num_layers).to(device)
            val_acc, _, _ = train_once(model, tr_loader, va_loader, te_loader, device, args.optuna_epochs, lr, wd, 40, trial_args)
            return val_acc

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=args.optuna_trials)
        bp = study.best_trial.params
        print(f"Optuna best params: {bp}, best val={study.best_value*100:.2f}%")
        args.model = bp["model"]
        args.hidden = int(bp["hidden"])
        args.num_layers = int(bp["num_layers"])
        args.dropout = float(bp["dropout"])
        args.lr = float(bp["lr"])
        args.weight_decay = float(bp["weight_decay"])
        args.batch_size = int(bp["batch_size"])

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=single_subject_collate_fn, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=single_subject_collate_fn, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=single_subject_collate_fn, **loader_kwargs)

    model = make_model(args.model, in_features, n_classes, args.hidden, args.dropout, args.num_layers).to(device)
    print(
        f"Final training: model={args.model}, graph={args.graph_method}, k={args.knn_k}, "
        f"projection={projection_type}, epochs={args.epochs}, accum_steps={args.batch_size}, hidden={args.hidden}, "
        f"layers={args.num_layers}, lr={args.lr:.2e}, wd={args.weight_decay:.2e}"
    )
    best_val, test_loss, test_acc = train_once(
        model, train_loader, val_loader, test_loader, device, args.epochs, args.lr, args.weight_decay, args.patience, args
    )
    print(f"Best Val Acc: {best_val*100:.2f}% | Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}%")


if __name__ == "__main__":
    main()
