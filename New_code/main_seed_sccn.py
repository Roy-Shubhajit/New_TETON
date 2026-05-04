from __future__ import annotations

import argparse
import os
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat
from scipy.signal import welch
from sklearn.model_selection import train_test_split
from torch.nn import BatchNorm1d, Linear, ReLU, Sequential
from torch_geometric.data import Data
from torch.utils.data import DataLoader
from torch_geometric.nn import GCNConv, GINConv, global_add_pool, global_mean_pool, global_max_pool
from topomodelx.nn.simplicial.sccn_layer import SCCNLayer
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


#define a collater function which just returns the batch as is as batch size is 1
def collate_fn(batch):
    return batch[0]


class complexes:
    def __init__(self, feature, adjacency, incidence, y, trial):
        self.feature = feature
        self.adjacency = adjacency
        self.incidence = incidence
        self.y = y
        self.trial = trial

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
        if isinstance(item, (set, frozenset)):
            item = tuple(sorted(int(v) for v in item))
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
        if isinstance(item, (set, frozenset)):
            item = tuple(sorted(int(v) for v in item))
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
                if isinstance(key, (set, frozenset)):
                    key = tuple(sorted(int(v) for v in key))
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

    features = {"rank_0": x0, "rank_1": x1, "rank_2": x2}
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
) -> List[Data]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"seed_vii_{args.sindy_backend}_SCCN_{cache_tag}.pt"
    if use_cache and cache_path.exists():
        try:
            bundle = torch.load(cache_path, map_location="cpu", weights_only=False)
            return bundle["simplices"]
        except Exception as e:
            print(f"[WARN] Failed to read cache {cache_path}: {e}")
            print("[WARN] Cache appears corrupted. Deleting and rebuilding...")
            try:
                cache_path.unlink()
            except Exception as del_err:
                print(f"[WARN] Could not delete corrupted cache: {del_err}")

    static_edge = build_static_edge_index()
    simplices: List[complexes] = []

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
            T = eeg_trial.shape[1]
            if T < window_size:
                continue

            for start in range(0, T - window_size + 1, window_stride):
                w = eeg_trial[:, start : start + window_size]
                x = window_to_features(w)
                if args.sindy_backend == "faster_sindy":
                    results = run_sindy_windows(
                        X_raw = w,
                        window_size = len(w),
                        stride = len(w),
                        preprocess_cfg = preprocess_cfg,
                        selection_cfg = selection_cfg,
                        solver_cfg = solver_cfg,
                        threshold_cfg = threshold_cfg,
                    )
                    pred_edges = results[0].pred_edges
                    pred_tris = results[0].pred_tris

                y = labels_80[trial_id - 1]
                features, incidences, adjacencies, _, _ = construct_topological_snapshot(
                    node_features=x,
                    edge_list=pred_edges,
                    triangle_list=pred_tris,
                    agg_func="mean",
                )

                simplices.append(
                    complexes(
                        feature=features,
                        adjacency=adjacencies,
                        incidence=incidences,
                        y=torch.tensor([y], dtype=torch.long),
                        trial=torch.tensor([trial_id], dtype=torch.long)
                    )
                )

    if not simplices:
        raise ValueError("No simplices were created. Try smaller --window_sec or lower --  debug settings.")

    if use_cache:
        try:
            torch.save({"simplices": simplices}, cache_path)
            print(f"Cached simplices to: {cache_path}")
        except Exception as e:
            # Do not fail training if cache write fails (disk quota/space/permissions).
            print(f"[WARN] Cache write failed at {cache_path}: {e}")
            print("[WARN] Continuing without cache.")
    return simplices


def normalize_graph_features(train_ds: List[Any], val_ds: List[Any], test_ds: List[Any]) -> None:
    train_x = torch.cat([d.x for d in train_ds], dim=0)
    mu = train_x.mean(dim=0, keepdim=True) if train_x.size(0) > 0 else torch.zeros(1)
    sigma = train_x.std(dim=0, keepdim=True).nan_to_num(1.0).clamp_min(1e-6) if train_x.size(0) > 1 else torch.ones(1)
    for ds in (train_ds, val_ds, test_ds):
        for d in ds:
            d.x = (d.x - mu) / sigma

def normalize_simplicial_features(train_ds: List[Any], val_ds: List[Any], test_ds: List[Any]) -> None:
    for ds in [train_ds, val_ds, test_ds]:
        for d in ds:
            for key in d.feature.keys():
                x = torch.tensor(d.feature[key], dtype=torch.float32)
                mu = x.mean(dim=0, keepdim=True) if x.size(0) > 0 else torch.zeros(1)
                sigma = x.std(dim=0, keepdim=True).nan_to_num(1.0).clamp_min(1e-6) if x.size(0) > 1 else torch.ones(1)
                d.feature[key] = (x - mu) / sigma


class GCN(torch.nn.Module):
    def __init__(self, in_features: int, num_classes: int, hidden: int, dropout: float, num_layers: int):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.convs = torch.nn.ModuleList([GCNConv(in_features, hidden)])
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        self.lin = Linear(hidden, num_classes)
        self.dropout = dropout

    def forward(self, data: Data):
        h = data.x
        for conv in self.convs:
            h = conv(h, data.edge_index).relu()
        h = global_mean_pool(h, data.batch)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.lin(h)


class GIN(torch.nn.Module):
    def __init__(self, in_features: int, num_classes: int, hidden: int, dropout: float, num_layers: int):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GINConv(Sequential(Linear(in_features, hidden), BatchNorm1d(hidden), ReLU(), Linear(hidden, hidden), ReLU()))
        )
        for _ in range(num_layers - 1):
            self.convs.append(
                GINConv(Sequential(Linear(hidden, hidden), BatchNorm1d(hidden), ReLU(), Linear(hidden, hidden), ReLU()))
            )
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, num_classes)
        self.dropout = dropout

    def forward(self, data: Data):
        h = data.x
        for conv in self.convs:
            h = conv(h, data.edge_index)
        h = global_add_pool(h, data.batch)
        h = self.lin1(h).relu()
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.lin2(h)
    
class SCCN(torch.nn.Module):
    def __init__(self, in_features: int, num_classes: int, hidden: int, dropout: float, num_layers: int):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.proj0 = Linear(in_features, hidden)
        self.proj1 = Linear(in_features, hidden)
        self.proj2 = Linear(in_features, hidden)
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(SCCNLayer(channels = hidden, max_rank=2, aggr_func="mean", update_func="relu"))
        self.lin = Linear(hidden * 3, num_classes)
        self.dropout = dropout

    def forward(self, data: complexes):
    
        #make sure all matrices are in gpu
        device = next(self.parameters()).device
        
        # Convert adjacency matrices to tensors on device
        adjacency_tensors = {}
        for key in data.adjacency.keys():
            adjacency_tensors[key] = torch.tensor(data.adjacency[key], dtype=torch.float32, device=device)
        
        # Convert incidence matrices to tensors on device
        incidence_tensors = {}
        for key in data.incidence.keys():
            incidence_tensors[key] = torch.tensor(data.incidence[key], dtype=torch.float32, device=device)
        
        # Convert features to tensors on device
        feature_tensors = {}
        for key in data.feature.keys():
            feature_tensors[key] = torch.tensor(data.feature[key], dtype=torch.float32, device=device)

        h0 = feature_tensors["rank_0"]
        h1 = feature_tensors["rank_1"]
        h2 = feature_tensors["rank_2"]

        h0 = self.proj0(h0)
        h1 = self.proj1(h1)
        h2 = self.proj2(h2)

        features = {"rank_0": h0, "rank_1": h1, "rank_2": h2}

        # Normalize types to torch.Tensor on device and avoid empty-tensor mean producing NaNs.
        for k in list(adjacency_tensors.keys()):
            val = adjacency_tensors[k]
            if not isinstance(val, torch.Tensor):
                adjacency_tensors[k] = torch.tensor(val, dtype=torch.float32, device=device)
            else:
                adjacency_tensors[k] = val.to(device)

        for k in list(incidence_tensors.keys()):
            val = incidence_tensors[k]
            if not isinstance(val, torch.Tensor):
                incidence_tensors[k] = torch.tensor(val, dtype=torch.float32, device=device)
            else:
                incidence_tensors[k] = val.to(device)

        for k in list(features.keys()):
            val = features[k]
            if not isinstance(val, torch.Tensor):
                features[k] = torch.tensor(val, dtype=torch.float32, device=device)
            else:
                features[k] = val.to(device)
            if features[k].dim() == 1:
                features[k] = features[k].unsqueeze(-1)

        def _safe_mean(tensor: torch.Tensor) -> torch.Tensor:
            if tensor.numel() == 0:
                feat_dim = tensor.size(1) if tensor.dim() >= 2 else self.proj0.out_features
                return torch.zeros((1, feat_dim), dtype=torch.float32, device=device)
            return torch.mean(tensor, dim=0, keepdim=True)

        for conv in self.convs:
            features = conv(features=features, incidences=incidence_tensors, adjacencies=adjacency_tensors)

        global_feature = torch.cat([
            _safe_mean(features["rank_0"]),
            _safe_mean(features["rank_1"]),
            _safe_mean(features["rank_2"]),
        ], dim=1)

        global_feature = F.dropout(global_feature, p=self.dropout, training=self.training)
        return self.lin(global_feature)


def make_model(model_name: str, in_features: int, num_classes: int, hidden: int, dropout: float, num_layers: int):
    if model_name.lower() == "gcn":
        return GCN(in_features, num_classes, hidden, dropout, num_layers)
    elif model_name.lower() == "gin":
        return GIN(in_features, num_classes, hidden, dropout, num_layers)
    elif model_name.lower() == "sccn":
        return SCCN(in_features, num_classes, hidden, dropout, num_layers)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, criterion: torch.nn.Module, device: torch.device):
    model.eval()
    total_loss, total_correct, total, n_batches = 0.0, 0, 0, 0
    pbar = tqdm(loader, desc="Evaluating")
    for sample in loader:
        y = sample.y.to(device).view(-1).long()
        logits = model(sample)
        loss = criterion(logits, y)
        pred = logits.argmax(dim=1)
        total_correct += int((pred == y).sum().item())
        total += y.numel()
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
    labels = [int(d.y.item()) for d in train_loader.dataset]
    counts = np.bincount(labels)
    weights = 1.0 / np.clip(counts, 1, None)
    weights = weights * (len(counts) / weights.sum())
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=20)

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
            y = sample.y.to(device).view(-1).long()
            optimizer.zero_grad()
            logits = model(sample)
            pending_logits.append(logits)
            pending_labels.append(y)

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.cpu().numpy())

            if len(pending_logits) >= args.batch_size or sample_idx == len(train_loader) - 1:
                logits_batch = torch.cat(pending_logits, dim=0)
                labels_batch = torch.cat(pending_labels, dim=0)
                loss = criterion(logits_batch, labels_batch)
                pending_logits.clear()
                pending_labels.clear()

                loss = criterion(logits_batch, labels_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

                print("Batch loss: {:.4f}".format(loss.item()))

                total_loss += loss.item()
                total_correct += int((torch.argmax(logits_batch, dim=1) == labels_batch).sum().item())
                total += labels_batch.numel()
                n_batches += 1
                pbar.set_postfix({"batch_loss": loss.item(), "acc": total_correct / max(total, 1)})

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
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.35)
    p.add_argument("--model", type=str, choices=["gcn", "gin", "sccn"], default="sccn")
    p.add_argument("--graph_method", type=str, choices=["seed_static", "knn_corr"], default="seed_static")
    p.add_argument("--knn_k", type=int, default=8)
    p.add_argument("--window_sec", type=float, default=8.0)
    p.add_argument("--stride_sec", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_subjects", type=int, default=None)
    p.add_argument("--patience", type=int, default=120)
    p.add_argument("--use_optuna", action="store_true")
    p.add_argument("--optuna_trials", type=int, default=30)
    p.add_argument("--optuna_epochs", type=int, default=120)
    p.add_argument("--no_cache", action="store_true", help="Disable graph cache read/write.")
    p.add_argument("--sindy_backend", type=str, choices=["sindy", "parallel_sindy", "faster_sindy"], default="parallel_sindy")
    return p.parse_args()


def main():
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')
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

    y_all = [int(d.y.item()) for d in dataset]
    idx = np.arange(len(dataset))
    train_idx, test_idx = train_test_split(idx, test_size=0.1, random_state=args.seed, stratify=y_all)
    y_train_tmp = [y_all[i] for i in train_idx]
    train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=args.seed, stratify=y_train_tmp)
    train_ds = [dataset[i] for i in train_idx]
    val_ds = [dataset[i] for i in val_idx]
    test_ds = [dataset[i] for i in test_idx]
    #normalize_simplicial_features(train_ds, val_ds, test_ds)

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

    in_features = train_ds[0].feature["rank_0"].shape[1]
    n_classes = len(emo_map)

    if args.use_optuna:
        try:
            import optuna
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

            tr_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, **loader_kwargs)
            va_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, **loader_kwargs)
            te_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, **loader_kwargs)

            model = make_model(model_name, in_features, n_classes, hidden, dropout, num_layers).to(device)
            val_acc, _, _ = train_once(model, tr_loader, va_loader, te_loader, device, args.optuna_epochs, lr, wd, 40, args=args)
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

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_fn, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_fn, **loader_kwargs)

    model = make_model(args.model, in_features, n_classes, args.hidden, args.dropout, args.num_layers).to(device)
    print(
        f"Final training: model={args.model}, graph={args.graph_method}, k={args.knn_k}, "
        f"epochs={args.epochs}, batch={args.batch_size}, hidden={args.hidden}, layers={args.num_layers}, "
        f"lr={args.lr:.2e}, wd={args.weight_decay:.2e}"
    )
    best_val, test_loss, test_acc = train_once(
        model, train_loader, val_loader, test_loader, device, args.epochs, args.lr, args.weight_decay, args.patience, args
    )
    print(f"Best Val Acc: {best_val*100:.2f}% | Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}%")


if __name__ == "__main__":
    main()
