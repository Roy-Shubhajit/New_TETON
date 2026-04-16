#!/usr/bin/env python3
"""
Data processing pipeline for ABIDE SINDy ML.

Handles:
1. Loading ABIDE dataset
2. Processing windows with faster_sindy.py / sindy.py
3. Creating simplicial complex representations
4. Creating PyTorch dataloaders with custom collator
"""

import json
import pickle
from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from tqdm import tqdm
import faster_sindy as faster_sindy_backend

from faster_sindy import (
    PreprocessConfig, SelectionConfig, SolverConfig, ThresholdConfig,
    preprocess_timeseries,
    run_sindy_windows
)
from Helper import (
    SimplicialComplex, cubic_spline_upsample_timeseries
)


FAST_CACHE_FILENAME = "processed_cache.pkl"


def _normalized_backend_name(sindy_backend: str) -> str:
    """Normalize the backend name used to separate exported datasets."""
    backend = str(sindy_backend).strip().lower()
    if backend not in {"faster_sindy", "sindy"}:
        raise ValueError("sindy_backend must be either 'faster_sindy' or 'sindy'")
    return backend


def _processed_backend_dir(output_dir: str, sindy_backend: str) -> Path:
    """Return the backend-specific export directory under the dataset root."""
    return Path(output_dir) / _normalized_backend_name(sindy_backend)


@dataclass
class _WindowBackendResult:
    """Backend-agnostic minimal window payload used by process_subject."""

    start: int
    end: int
    pred_edges: set
    pred_tris: set


def _sort_edges(edge_list: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Sort edge endpoints and return unique sorted edges."""
    return sorted({tuple(sorted((int(u), int(v)))) for u, v in edge_list})


def construct_topological_snapshot(
    node_features: np.ndarray,
    edge_list: List[Tuple[int, int]],
    triangle_list: List[Tuple[int, int, int]],
    agg_func: str = "mean",
) -> Tuple[Dict[int, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], List[Tuple[int, int]], List[Tuple[int, int, int]]]:
    """Construct lifted features, incidence matrices, and adjacency matrices for one window."""
    x0 = np.asarray(node_features, dtype=np.float32)
    if x0.ndim == 1:
        x0 = x0[:, None]
    if x0.ndim != 2:
        raise ValueError(f"node_features must be 1D or 2D, got shape {x0.shape}")

    n_nodes, feat_dim = x0.shape
    edges = _sort_edges(edge_list)
    edge_to_idx = {edge: idx for idx, edge in enumerate(edges)}

    def _aggregate(values: np.ndarray) -> np.ndarray:
        if agg_func == "mean":
            return values.mean(axis=0)
        if agg_func == "sum":
            return values.sum(axis=0)
        if agg_func == "max":
            return values.max(axis=0)
        raise ValueError("agg_func must be 'mean', 'sum', or 'max'")

    if edges:
        x1 = np.stack([_aggregate(x0[[u, v]]) for (u, v) in edges], axis=0).astype(np.float32)
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
    for tri in triangle_list:
        u, v, w = sorted((int(tri[0]), int(tri[1]), int(tri[2])))
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
        x2 = np.stack([_aggregate(x0[[u, v, w]]) for (u, v, w) in valid_triangles], axis=0).astype(np.float32)
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


class _NumpyCompatUnpickler(pickle.Unpickler):
    """Unpickler that remaps NumPy internal module paths across versions."""

    _MODULE_REMAP = {
        "numpy._core": "numpy.core",
        "numpy._core.numeric": "numpy.core.numeric",
        "numpy._core.multiarray": "numpy.core.multiarray",
        "numpy._core.umath": "numpy.core.umath",
    }

    def find_class(self, module: str, name: str):
        remapped_module = self._MODULE_REMAP.get(module, module)
        return super().find_class(remapped_module, name)


def _load_pickle_with_numpy_compat(cache_path: Path) -> Any:
    """Load pickle cache with NumPy module-path compatibility fallbacks."""
    with cache_path.open("rb") as fp:
        try:
            return pickle.load(fp)
        except ModuleNotFoundError:
            fp.seek(0)
            return _NumpyCompatUnpickler(fp).load()


def _force_sindy_processing_cpu() -> None:
    """Force SINDy preprocessing to run on CPU for subject processing."""
    faster_sindy_backend.FORCE_CPU_OVERRIDE = True
    faster_sindy_backend.xp = np


def _safe_subject_filename(subject_id: str) -> str:
    cleaned = [ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(subject_id)]
    return "".join(cleaned).strip("._") or "subject"


def _window_to_exportable(window: SimplicialComplex) -> Dict[str, Any]:
    features_dict = getattr(window, "features", {}) if hasattr(window, "features") else {}
    rank0_features = np.asarray(features_dict.get("rank_0", window.node_features), dtype=np.float32)
    rank1_features = np.asarray(features_dict.get("rank_1", window.edge_features), dtype=np.float32)
    rank2_features = np.asarray(features_dict.get("rank_2", window.triangle_features), dtype=np.float32)

    incidence_rank_1 = np.asarray(window.incidence.get("rank_1", window.incidence["B1"]), dtype=np.int8)
    incidence_rank_2 = np.asarray(window.incidence.get("rank_2", window.incidence["B2"]), dtype=np.int8)

    adjacency_rank_0 = np.asarray(window.adjacency.get("rank_0", window.adjacency["H0_up"]), dtype=np.float32)
    adjacency_rank_1 = np.asarray(window.adjacency.get("rank_1", window.adjacency["H1_down"]), dtype=np.float32)
    adjacency_rank_2 = np.asarray(window.adjacency.get("rank_2", window.adjacency["H2_down"]), dtype=np.float32)

    return {
        "node_features": rank0_features,
        "edge_features": rank1_features,
        "triangle_features": rank2_features,
        "rank_0_features": rank0_features,
        "rank_1_features": rank1_features,
        "rank_2_features": rank2_features,
        "edges": _edge_array(window.incidence["edges"]),
        "triangles": _triangle_array(window.incidence["triangles"]),
        "rank_1": incidence_rank_1,
        "rank_2": incidence_rank_2,
        "B1": incidence_rank_1,
        "B2": incidence_rank_2,
        "A0": adjacency_rank_0,
        "A1": adjacency_rank_1,
        "A2": adjacency_rank_2,
        "rank_0": adjacency_rank_0,
        "rank_1_adj": adjacency_rank_1,
        "rank_2_adj": adjacency_rank_2,
        "H0_up": adjacency_rank_0,
        "H1_down": adjacency_rank_1,
        "H1_up": np.asarray(window.adjacency.get("H1_up", np.zeros_like(adjacency_rank_1)), dtype=np.float32),
        "H2_down": adjacency_rank_2,
    }


def save_processed_dataset(
    samples: List[Dict[str, Any]],
    labels: List[int],
    split_indices: Dict[str, List[int]],
    metadata: Dict[str, Any],
    output_dir: str,
    sindy_backend: str,
) -> Dict[str, Any]:
    """Save processed simplicial dataset so others can reuse it without recomputation."""
    backend_name = _normalized_backend_name(sindy_backend)
    out_dir = _processed_backend_dir(output_dir, backend_name)
    subjects_dir = out_dir / "subjects"
    fast_cache_path = out_dir / FAST_CACHE_FILENAME
    out_dir.mkdir(parents=True, exist_ok=True)
    subjects_dir.mkdir(parents=True, exist_ok=True)

    subject_records: List[Dict[str, Any]] = []

    for i, sample in enumerate(samples):
        subject_id = str(sample.get("subject_id", f"subject_{i:04d}"))
        filename = f"{i:04d}_{_safe_subject_filename(subject_id)}.npz"
        file_path = subjects_dir / filename

        windows = sample.get("windows", [])
        window_payloads = [_window_to_exportable(window) for window in windows]

        # Use uncompressed npz for faster readback than savez_compressed.
        np.savez(
            file_path,
            label=np.int64(labels[i]),
            subject_id=np.array(subject_id),
            n_windows=np.int32(len(window_payloads)),
            windows=np.array(window_payloads, dtype=object),
        )

        subject_records.append(
            {
                "sample_index": i,
                "subject_id": subject_id,
                "label": int(labels[i]),
                "n_windows": int(len(window_payloads)),
                "file": str(Path("subjects") / filename),
            }
        )

    export_manifest = {
        "format": "abide_sindy_simplicial_v1",
        "sindy_backend": backend_name,
        "num_samples": len(samples),
        "splits": split_indices,
        "metadata": metadata,
        "subjects": subject_records,
    }

    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fp:
        json.dump(export_manifest, fp, indent=2)

    split_path = out_dir / "splits.json"
    with split_path.open("w", encoding="utf-8") as fp:
        json.dump(split_indices, fp, indent=2)

    # Fast cache for quick full-dataset reload.
    fast_payload = {
        "format": "abide_sindy_simplicial_fast_v1",
        "sindy_backend": backend_name,
        "samples": samples,
        "labels": labels,
        "splits": split_indices,
        "metadata": metadata,
    }
    with fast_cache_path.open("wb") as fp:
        pickle.dump(fast_payload, fp, protocol=pickle.HIGHEST_PROTOCOL)

    export_summary = {
        "output_dir": str(out_dir),
        "sindy_backend": backend_name,
        "manifest": str(manifest_path),
        "splits": str(split_path),
        "num_subject_files": len(subject_records),
        "fast_cache": str(fast_cache_path),
    }
    return export_summary


def _window_from_exportable(payload: Dict[str, Any]) -> SimplicialComplex:
    node_features = np.asarray(payload.get("rank_0_features", payload.get("node_features", [])), dtype=np.float32)

    edges_arr = np.asarray(payload.get("edges", []), dtype=np.int32)
    triangles_arr = np.asarray(payload.get("triangles", []), dtype=np.int32)

    edges = [tuple(int(v) for v in edge) for edge in edges_arr.tolist()] if edges_arr.size > 0 else []
    triangles = [tuple(int(v) for v in tri) for tri in triangles_arr.tolist()] if triangles_arr.size > 0 else []

    edge_features = np.asarray(payload.get("rank_1_features", payload.get("edge_features", [])), dtype=np.float32)
    triangle_features = np.asarray(payload.get("rank_2_features", payload.get("triangle_features", [])), dtype=np.float32)

    simp = SimplicialComplex(
        node_features=node_features,
        edges=edges,
        triangles=triangles,
        edge_features=edge_features,
        triangle_features=triangle_features,
    )

    # Preserve exact exported incidence/adjacency tensors when available.
    rank_1_inc = np.asarray(payload.get("rank_1", payload.get("B1", simp.incidence.get("B1"))), dtype=np.int8)
    rank_2_inc = np.asarray(payload.get("rank_2", payload.get("B2", simp.incidence.get("B2"))), dtype=np.int8)
    simp.incidence["rank_1"] = rank_1_inc
    simp.incidence["rank_2"] = rank_2_inc
    simp.incidence["B1"] = rank_1_inc
    simp.incidence["B2"] = rank_2_inc

    rank_0_adj = np.asarray(payload.get("rank_0", payload.get("A0", payload.get("H0_up", simp.adjacency.get("H0_up")))), dtype=np.float32)
    rank_1_adj = np.asarray(payload.get("rank_1_adj", payload.get("A1", payload.get("H1_down", simp.adjacency.get("H1_down")))), dtype=np.float32)
    rank_2_adj = np.asarray(payload.get("rank_2_adj", payload.get("A2", payload.get("H2_down", simp.adjacency.get("H2_down")))), dtype=np.float32)
    simp.adjacency["rank_0"] = rank_0_adj
    simp.adjacency["rank_1"] = rank_1_adj
    simp.adjacency["rank_2"] = rank_2_adj
    simp.adjacency["H0_up"] = rank_0_adj
    simp.adjacency["H1_down"] = rank_1_adj
    simp.adjacency["H2_down"] = rank_2_adj
    if "H1_up" in payload:
        simp.adjacency["H1_up"] = np.asarray(payload["H1_up"], dtype=np.float32)
    else:
        simp.adjacency["H1_up"] = np.zeros_like(rank_1_adj, dtype=np.float32)

    simp.features = {
        "rank_0": np.asarray(node_features, dtype=np.float32),
        "rank_1": np.asarray(edge_features, dtype=np.float32),
        "rank_2": np.asarray(triangle_features, dtype=np.float32),
    }
    simp.incidences = {
        "rank_1": rank_1_inc,
        "rank_2": rank_2_inc,
    }
    simp.adjacencies = {
        "rank_0": rank_0_adj,
        "rank_1": rank_1_adj,
        "rank_2": rank_2_adj,
    }

    return simp


def _ensure_temporal_matrix(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"Expected array with shape (n_items,) or (n_items, feature_dim), got {arr.shape}")
    return arr


def _edge_array(edges: List[Tuple[int, int]]) -> np.ndarray:
    arr = np.asarray(edges, dtype=np.int32)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=np.int32)
    return arr.reshape(-1, 2)


def _triangle_array(triangles: List[Tuple[int, int, int]]) -> np.ndarray:
    arr = np.asarray(triangles, dtype=np.int32)
    if arr.size == 0:
        return np.zeros((0, 3), dtype=np.int32)
    return arr.reshape(-1, 3)


def load_processed_dataset(
    output_dir: str,
    sindy_backend: str,
    allow_npz_fallback: bool = False,
) -> Tuple[List[Dict[str, Any]], List[int], Dict[str, List[int]], Dict[str, Any]]:
    """Load previously processed simplicial dataset export from disk.

    By default this loads only from the fast pickle cache. Legacy manifest+npz
    loading is available only when allow_npz_fallback=True.
    """
    backend_name = _normalized_backend_name(sindy_backend)
    out_dir = _processed_backend_dir(output_dir, backend_name)
    fast_cache_path = out_dir / FAST_CACHE_FILENAME
    manifest_path = out_dir / "manifest.json"

    if not fast_cache_path.exists():
        legacy_cache_path = Path(output_dir) / FAST_CACHE_FILENAME
        if legacy_cache_path.exists():
            fast_cache_path = legacy_cache_path
            out_dir = Path(output_dir)
        else:
            raise FileNotFoundError(f"Processed pickle cache not found at {fast_cache_path}")

    try:
        payload = _load_pickle_with_numpy_compat(fast_cache_path)

        if not isinstance(payload, dict):
            raise ValueError("Invalid fast cache payload type")

        samples = payload.get("samples", [])
        labels = payload.get("labels", [])
        split_indices = payload.get("splits", {})
        metadata = payload.get("metadata", {})
        if isinstance(metadata, dict):
            metadata = dict(metadata)
            metadata["_processed_cache_source"] = "fast_pickle"
            metadata["sindy_backend"] = payload.get("sindy_backend", backend_name)
        return samples, labels, split_indices, metadata
    except Exception as exc:
        if not allow_npz_fallback:
            raise RuntimeError(f"Failed to load pickle cache: {exc}") from exc

    if not manifest_path.exists():
        raise FileNotFoundError(f"Processed manifest not found at {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as fp:
        manifest = json.load(fp)

    subject_records = manifest.get("subjects", [])
    samples: List[Dict[str, Any]] = []
    labels: List[int] = []

    for record in subject_records:
        rel_file = record.get("file")
        if not rel_file:
            raise ValueError("Invalid manifest: subject record missing 'file'")

        npz_path = out_dir / rel_file
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing subject file: {npz_path}")

        with np.load(npz_path, allow_pickle=True) as data:
            label = int(data["label"])
            subject_raw = data["subject_id"]
            subject_id = str(subject_raw.item() if np.ndim(subject_raw) == 0 else subject_raw)

            windows_payload = data["windows"]
            windows = [_window_from_exportable(payload) for payload in windows_payload.tolist()]

        samples.append(
            {
                "windows": windows,
                "subject_id": subject_id,
                "n_windows": len(windows),
            }
        )
        labels.append(label)

    split_indices = manifest.get("splits", {})
    metadata = manifest.get("metadata", {})
    if isinstance(metadata, dict):
        metadata = dict(metadata)
        metadata["_processed_cache_source"] = "legacy_npz"
        metadata["sindy_backend"] = manifest.get("sindy_backend", backend_name)

    # Upgrade legacy cache to fast single-file cache for subsequent runs.
    if not fast_cache_path.exists():
        try:
            fast_payload = {
                "format": "abide_sindy_simplicial_fast_v1",
                "sindy_backend": backend_name,
                "samples": samples,
                "labels": labels,
                "splits": split_indices,
                "metadata": metadata,
            }
            with fast_cache_path.open("wb") as fp:
                pickle.dump(fast_payload, fp, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass

    return samples, labels, split_indices, metadata


class SimplicialDataset(Dataset):
    """
    PyTorch Dataset for sequences of simplicial complexes.
    
    Each sample is a sequence of windows from a single subject.
    """
    
    def __init__(
        self,
        samples: List[Dict[str, Any]],
        labels: List[int],
    ):
        """
        Initialize dataset.
        
        Args:
            samples: List of dicts with 'windows' (list of SimplicialComplex) and metadata
            labels: List of class labels (one per sample)
        """
        self.samples = samples
        self.labels = labels
        
        if len(samples) != len(labels):
            raise ValueError("Number of samples and labels must match")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample.
        
        Returns:
            Dict with 'windows' (list of SimplicialComplex), 'label', 'subject_id'
        """
        sample = self.samples[idx]
        label = self.labels[idx]
        
        return {
            "windows": sample.get("windows", []),
            "label": torch.tensor(label, dtype=torch.long),
            "subject_id": sample.get("subject_id", "unknown"),
            "n_windows": len(sample.get("windows", [])),
        }


def simplicial_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for simplicial complex dataset.
    
    Handles variable sequence lengths and creates padded tensors.
    
    Args:
        batch: List of samples from SimplicialDataset
    
    Returns:
        Dictionary with:
        - node_features: (batch_size, max_seq_len, max_nodes, node_feature_dim)
        - edge_features: (batch_size, max_seq_len, max_edges, edge_feature_dim)
        - triangle_features: (batch_size, max_seq_len, max_triangles, triangle_feature_dim)
        - labels: (batch_size,)
        - lengths: (batch_size,) - actual sequence lengths
        - subject_ids: List of subject IDs
    """
    
    # Extract components from batch
    batch_windows = [sample["windows"] for sample in batch]
    batch_labels = torch.stack([sample["label"] for sample in batch])
    batch_lengths = torch.tensor([len(windows) for windows in batch_windows], dtype=torch.long)
    batch_subject_ids = [sample["subject_id"] for sample in batch]
    
    # Find max dimensions
    max_seq_len = max(len(windows) for windows in batch_windows)
    
    batch_size = len(batch)
    max_nodes = 0
    max_edges = 0
    max_triangles = 0
    node_feat_dim = 1
    edge_feat_dim = 1
    triangle_feat_dim = 1
    
    for windows in batch_windows:
        for window in windows:
            max_nodes = max(max_nodes, window.n_nodes)
            max_edges = max(max_edges, window.incidence["n_edges"])
            max_triangles = max(max_triangles, window.incidence["n_triangles"])
            node_feat_dim = max(node_feat_dim, _ensure_temporal_matrix(window.node_features).shape[1])
            edge_arr = np.asarray(window.edge_features, dtype=np.float32)
            tri_arr = np.asarray(window.triangle_features, dtype=np.float32)
            if edge_arr.ndim == 1:
                edge_arr = edge_arr[:, None]
            if tri_arr.ndim == 1:
                tri_arr = tri_arr[:, None]
            if edge_arr.size > 0:
                edge_feat_dim = max(edge_feat_dim, edge_arr.shape[1])
            if tri_arr.size > 0:
                triangle_feat_dim = max(triangle_feat_dim, tri_arr.shape[1])
    
    # Initialize padded tensors
    node_features = torch.zeros(
        (batch_size, max_seq_len, max_nodes, node_feat_dim),
        dtype=torch.float32
    )
    edge_features = torch.zeros(
        (batch_size, max_seq_len, max_edges, edge_feat_dim),
        dtype=torch.float32
    )
    triangle_features = torch.zeros(
        (batch_size, max_seq_len, max_triangles, triangle_feat_dim),
        dtype=torch.float32
    )
    
    # Fill tensors
    for b_idx, windows in enumerate(batch_windows):
        for w_idx, window in enumerate(windows):
            # Node features
            n_nodes = window.n_nodes
            node_arr = _ensure_temporal_matrix(window.node_features)
            node_features[b_idx, w_idx, :n_nodes, :node_arr.shape[1]] = torch.from_numpy(node_arr)
            
            # Edge features
            edge_arr = np.asarray(window.edge_features, dtype=np.float32)
            if edge_arr.ndim == 1:
                edge_arr = edge_arr[:, None]
            n_edges = edge_arr.shape[0]
            if n_edges > 0:
                edge_features[b_idx, w_idx, :n_edges, :edge_arr.shape[1]] = torch.from_numpy(edge_arr)
            
            # Triangle features
            tri_arr = np.asarray(window.triangle_features, dtype=np.float32)
            if tri_arr.ndim == 1:
                tri_arr = tri_arr[:, None]
            n_triangles = tri_arr.shape[0]
            if n_triangles > 0:
                triangle_features[b_idx, w_idx, :n_triangles, :tri_arr.shape[1]] = torch.from_numpy(tri_arr)
    
    return {
        "node_features": node_features,
        "edge_features": edge_features,
        "triangle_features": triangle_features,
        "labels": batch_labels,
        "lengths": batch_lengths,
        "subject_ids": batch_subject_ids,
    }


def single_subject_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a single subject sample unchanged.

    The training loop processes one subject at a time, so the dataloader
    should yield raw subject dictionaries instead of stacking tensors.
    """
    if len(batch) != 1:
        raise ValueError("single_subject_collate_fn expects batch_size=1")
    return batch[0]


class ABIDESimplicialProcessor:
    """
    Main processor for converting ABIDE to simplicial complexes.
    """
    
    def __init__(
        self,
        data_dir: str = "./",
        save_intermediates: bool = False,
        upsample_factor: int = 100,
        sindy_backend: str = "faster_sindy",
    ):
        """
        Initialize processor.
        
        Args:
            data_dir: Directory containing ABIDE dataset
            save_intermediates: Whether to save intermediate results
            upsample_factor: Factor to upsample timeseries (e.g., 100 means 100x interpolation)
        """
        self.data_dir = Path(data_dir)
        self.save_intermediates = save_intermediates
        self.upsample_factor = upsample_factor
        self.sindy_backend = str(sindy_backend).strip().lower()
        if self.sindy_backend not in {"faster_sindy", "sindy"}:
            raise ValueError("sindy_backend must be either 'faster_sindy' or 'sindy'")
        
        # Create config objects for processing
        self.preprocess_cfg = PreprocessConfig(
            fs=256.0,
            win_sg=29,
            poly_order=3,
        )
        
        self.selection_cfg = SelectionConfig(
            r_target_pc=0.95,
            k_min=600,
            k_max=1_000_000,
            max_rows=6000,
        )
        
        self.solver_cfg = SolverConfig(
            ridge_lambda=1e-3,
            stlsq_iters=10,
            lambda_scale=1.0,
            refit_lambda=1e-6,
            auto_relax_if_empty=True,
        )
        
        self.threshold_cfg = ThresholdConfig(
            edge_quantile=0.90,
        )

        # Keep faster_sindy preprocessing CPU-bound; ML training uses GPU later.
        if self.sindy_backend == "faster_sindy":
            _force_sindy_processing_cpu()

    def _run_sindy_backend_windows(
        self,
        X_raw: np.ndarray,
        window_len: int,
        stride: int,
    ) -> List[_WindowBackendResult]:
        """Run selected backend and normalize window outputs."""
        if self.sindy_backend == "faster_sindy":
            backend_results = run_sindy_windows(
                X_raw,
                window_size=window_len,
                stride=stride,
                preprocess_cfg=self.preprocess_cfg,
                selection_cfg=self.selection_cfg,
                solver_cfg=self.solver_cfg,
                threshold_cfg=self.threshold_cfg,
            )
            return [
                _WindowBackendResult(
                    start=int(r.start),
                    end=int(r.end),
                    pred_edges={frozenset(tuple(sorted(tuple(e)))) for e in r.pred_edges},
                    pred_tris={frozenset(tuple(sorted(tuple(t)))) for t in r.pred_tris},
                )
                for r in backend_results
            ]

        # Legacy backend: adapt sindy.py payload to match faster_sindy window schema.
        import sindy as legacy_sindy

        args = legacy_sindy.SindyArgs(
            win_len=int(window_len),
            stride=int(stride),
            overlap=float(1.0 - (float(stride) / float(max(1, window_len)))),
            fs=float(self.preprocess_cfg.fs),
            win_sg=int(self.preprocess_cfg.win_sg),
            order=int(self.preprocess_cfg.poly_order),
            r_target_pc=float(self.selection_cfg.r_target_pc),
            k_min=int(self.selection_cfg.k_min),
            k_max=int(self.selection_cfg.k_max),
            scale=float(self.solver_cfg.lambda_scale),
            tau2_q=float(self.threshold_cfg.edge_quantile),
            tau3_q=float(self.threshold_cfg.edge_quantile),
        )

        backend_results = legacy_sindy.process_data_in_windows(X_raw, args)
        out: List[_WindowBackendResult] = []
        for r in backend_results:
            raw_edges = r.get("edges", [])
            raw_tris = r.get("triangles", [])
            out.append(
                _WindowBackendResult(
                    start=int(r.get("window_start", 0)),
                    end=int(r.get("window_end", 0)),
                    pred_edges={frozenset(tuple(sorted(tuple(e)))) for e in raw_edges},
                    pred_tris={frozenset(tuple(sorted(tuple(t)))) for t in raw_tris},
                )
            )
        return out
    
    def process_subject(
        self,
        subject_ts: np.ndarray,
        window_len: int,
        window_overlap: float,
        subject_id: str = "unknown",
    ) -> Dict[str, List[SimplicialComplex]]:
        """
        Process a single subject timeseries into windows of simplicial complexes.
        
        Args:
            subject_ts: Time series of shape (timepoints, rois)
            window_len: Length of each window
            window_overlap: Overlap between windows
            subject_id: Subject identifier
        
        Returns:
            Dict with 'windows' list of SimplicialComplex and metadata
        """
        
        windows = []
        
        try:
            # faster_sindy expects shape (N, T), while ABIDE comes as (T, N).
            X_raw = np.asarray(subject_ts, dtype=np.float64).T

            # Apply upsampling interpolation if factor > 1
            if self.upsample_factor > 1:
                X_raw = cubic_spline_upsample_timeseries(X_raw.T, factor=self.upsample_factor).T

            # Keep the exported features aligned with the exact preprocessing used by SINDy.
            X_proc, _ = preprocess_timeseries(X_raw, self.preprocess_cfg, drop_degenerate=True)
            n_nodes = X_proc.shape[0]

            stride = max(1, int(round(window_len * (1.0 - window_overlap))))

            # Run selected SINDy backend and normalize to a common window schema.
            sindy_windows = self._run_sindy_backend_windows(
                X_raw=X_raw,
                window_len=window_len,
                stride=stride,
            )

            if not sindy_windows:
                print(f"  [skip] Subject {subject_id}: no valid windows")
                return {"windows": [], "subject_id": subject_id, "error": "no_windows"}

            for w_idx, result in enumerate(sindy_windows):
                try:
                    start = int(result.start)
                    end = int(result.end)

                    # Temporal node values for this processed window.
                    node_temporal = X_proc[:, start:end].astype(np.float32, copy=False)

                    edges_in = [tuple(sorted(tuple(edge))) for edge in result.pred_edges]
                    triangles_in = [tuple(sorted(tuple(tri))) for tri in result.pred_tris]

                    features, incidences, adjacencies, edges, triangles = construct_topological_snapshot(
                        node_features=node_temporal,
                        edge_list=edges_in,
                        triangle_list=triangles_in,
                        agg_func="mean",
                    )

                    simp = SimplicialComplex(
                        node_features=features[0],
                        edges=edges,
                        triangles=triangles,
                        edge_features=features[1],
                        triangle_features=features[2],
                    )

                    # Use explicit matrices from snapshot builder.
                    rank_1_inc = incidences["rank_1"].astype(np.int8)
                    rank_2_inc = incidences["rank_2"].astype(np.int8)
                    simp.incidence["rank_1"] = rank_1_inc
                    simp.incidence["rank_2"] = rank_2_inc
                    simp.incidence["B1"] = rank_1_inc
                    simp.incidence["B2"] = rank_2_inc
                    simp.incidence["edges"] = edges
                    simp.incidence["triangles"] = triangles
                    simp.incidence["n_edges"] = len(edges)
                    simp.incidence["n_triangles"] = len(triangles)

                    rank_0_adj = adjacencies["rank_0"].astype(np.float32)
                    rank_1_adj = adjacencies["rank_1"].astype(np.float32)
                    rank_2_adj = adjacencies["rank_2"].astype(np.float32)
                    simp.adjacency["rank_0"] = rank_0_adj
                    simp.adjacency["rank_1"] = rank_1_adj
                    simp.adjacency["rank_2"] = rank_2_adj
                    simp.adjacency["H0_up"] = rank_0_adj
                    simp.adjacency["H1_down"] = rank_1_adj
                    simp.adjacency["H1_up"] = np.zeros_like(rank_1_adj, dtype=np.float32)
                    simp.adjacency["H2_down"] = rank_2_adj

                    simp.features = {
                        "rank_0": np.asarray(features[0], dtype=np.float32),
                        "rank_1": np.asarray(features[1], dtype=np.float32),
                        "rank_2": np.asarray(features[2], dtype=np.float32),
                    }
                    simp.incidences = {
                        "rank_1": rank_1_inc,
                        "rank_2": rank_2_inc,
                    }
                    simp.adjacencies = {
                        "rank_0": rank_0_adj,
                        "rank_1": rank_1_adj,
                        "rank_2": rank_2_adj,
                    }

                    windows.append(simp)

                except Exception as e:
                    print(f"  [warn] Subject {subject_id}, window {w_idx}: {e}")
                    continue
            
            if not windows:
                print(f"  [skip] Subject {subject_id}: no valid windows created")
                return {"windows": [], "subject_id": subject_id, "error": "no_valid_windows"}
            
            #print(f"  [ok] Subject {subject_id}: {len(windows)} windows created")
            return {
                "windows": windows,
                "subject_id": subject_id,
                "n_windows": len(windows),
                "n_nodes": n_nodes,
            }
        
        except Exception as e:
            print(f"  [error] Subject {subject_id}: {e}")
            return {"windows": [], "subject_id": subject_id, "error": str(e)}
    
    def process_dataset(
        self,
        abide_data,
        window_len: int,
        window_overlap: float,
    ) -> Tuple[List[Dict], List[int], List[str]]:
        """
        Process entire ABIDE dataset.
        
        Args:
            abide_data: Nilearn ABIDE dataset object
            window_len: Window length
            window_overlap: Window overlap ratio
        
        Returns:
            Tuple of (samples, labels, subject_ids)
        """
        
        samples = []
        labels = []
        subject_ids_list = []
        
        # Get all timeseries
        all_timeseries = abide_data.rois_cc200
        phenotypic_data = abide_data.phenotypic
        
        print(f"Processing {len(all_timeseries)} subjects...")
        
        for subject_idx in tqdm(range(len(all_timeseries)), desc="Subjects"):
            subject_idx = int(subject_idx)  # Ensure native Python int for pandas indexing
            
            # Get subject timeseries
            subject_ts = all_timeseries[subject_idx]  # Shape: (timepoints, rois)
            
            # Get subject ID
            subject_id = self._get_subject_id(abide_data, subject_idx)
            
            # Get label (DX_GROUP: 1=Autism, 2=Control)
            try:
                label_value = int(phenotypic_data.iloc[subject_idx]["DX_GROUP"])
                if label_value == 2:
                    label = 0  # Control
                elif label_value == 1:
                    label = 1  # Autism
                else:
                    print(f"  [skip] Subject {subject_idx}: unknown DX_GROUP={label_value}")
                    continue
            except Exception as e:
                print(f"  [skip] Subject {subject_idx}: no label found - {e}")
                continue
            
            # Process subject
            result = self.process_subject(
                subject_ts,
                window_len=window_len,
                window_overlap=window_overlap,
                subject_id=subject_id,
            )
            
            # Add to dataset if successful
            if result.get("windows"):
                samples.append(result)
                labels.append(label)
                subject_ids_list.append(subject_id)
        
        print(f"Processed {len(samples)} subjects successfully")
        return samples, labels, subject_ids_list
    
    @staticmethod
    def _get_subject_id(abide_data, index: int) -> str:
        """Extract subject ID from ABIDE dataset."""
        for attr in ("subject_id", "subject_ids", "subjects"):
            try:
                values = getattr(abide_data, attr, None)
                if values is not None:
                    value = values[index]
                    if isinstance(value, bytes):
                        return value.decode("utf-8", errors="ignore")
                    return str(value)
            except Exception:
                continue
        return f"subject_{index:04d}"


def create_dataloaders(
    data_dir: str = "./",
    n_subjects: Optional[int] = None,
    window_length: int = 1024,  # scaled by upsample factor
    window_overlap: float = 0.5,
    batch_size: int = 32,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    random_seed: int = 42,
    num_workers: int = 0,
    verbose: bool = True,
    save_processed: bool = True,
    processed_output_dir: str = "./abide_simplicial_dataset",
    load_processed_if_available: bool = True,
    upsample_factor: int = 100,
    sindy_backend: str = "faster_sindy",
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """
    Create train, validation, and test dataloaders for ABIDE data.
    
    Args:
        data_dir: Path to ABIDE dataset
        n_subjects: Number of subjects to process (None for all)
        window_length: Length of each processing window
        window_overlap: Overlap between windows
        batch_size: Kept for API compatibility; loaders are always subject-wise.
        train_split: Fraction for training (0-1)
        val_split: Fraction for validation (0-1)
        test_split: Fraction for testing (0-1)
        random_seed: Random seed for reproducibility
        num_workers: Number of workers for data loading
        verbose: Whether to print progress
        save_processed: Whether to export processed dataset cache to disk
        processed_output_dir: Directory where processed dataset export is written
        load_processed_if_available: If True, reuse existing processed export when present
        upsample_factor: Interpolation factor for upsampling timeseries
        sindy_backend: Backend solver to use, either 'faster_sindy' or 'sindy'
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader, metadata_dict)
    """
    
    # Validate splits
    total = train_split + val_split + test_split
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Splits must sum to 1.0, got {total}")
    
    # Set random seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # Normalize n_subjects: convert string "all" to None
    if isinstance(n_subjects, str) and n_subjects.lower() == "all":
        n_subjects = None
    elif isinstance(n_subjects, str):
        try:
            n_subjects = int(n_subjects)
        except ValueError:
            n_subjects = None

    processed_loaded = False
    split_indices: Dict[str, List[int]] = {}
    loaded_metadata: Dict[str, Any] = {}
    processed_backend_dir = _processed_backend_dir(processed_output_dir, sindy_backend)

    if load_processed_if_available:
        try:
            samples, labels, split_indices, loaded_metadata = load_processed_dataset(
                processed_output_dir,
                sindy_backend=sindy_backend,
            )
            subject_ids = [str(sample.get("subject_id", "unknown")) for sample in samples]
            loaded_backend = (
                loaded_metadata.get("sindy_backend", "faster_sindy")
                if isinstance(loaded_metadata, dict)
                else "faster_sindy"
            )
            if str(loaded_backend).lower() == str(sindy_backend).lower():
                processed_loaded = True
            else:
                if verbose:
                    print(
                        f"Processed cache backend mismatch (cache={loaded_backend}, requested={sindy_backend}). "
                        "Reprocessing..."
                    )
            if verbose:
                src = loaded_metadata.get("_processed_cache_source", "unknown") if isinstance(loaded_metadata, dict) else "unknown"
                print(f"Loaded processed dataset from {processed_backend_dir} (source={src})")
        except FileNotFoundError:
            if verbose:
                print(f"No processed dataset found at {processed_backend_dir}. Running processing...")
        except Exception as e:
            if verbose:
                print(f"Could not load processed dataset ({e}). Running processing...")

    if not processed_loaded:
        # Load ABIDE dataset
        if verbose:
            print(f"Loading ABIDE dataset from {data_dir}...")

        try:
            from nilearn import datasets
            abide_data = datasets.fetch_abide_pcp(
                data_dir=data_dir,
                derivatives="rois_cc200",
                n_subjects=n_subjects,
                verbose=1 if verbose else 0,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load ABIDE dataset: {e}")

        # Process dataset
        if verbose:
            print("Processing dataset into simplicial complexes (CPU preprocessing)...")

        processor = ABIDESimplicialProcessor(
            data_dir=data_dir,
            save_intermediates=False,
            upsample_factor=upsample_factor,
            sindy_backend=sindy_backend,
        )
        samples, labels, subject_ids = processor.process_dataset(
            abide_data,
            window_len=window_length,
            window_overlap=window_overlap,
        )
    
    if not samples:
        raise RuntimeError("No samples were successfully processed")
    
    if verbose:
        print(f"Total samples: {len(samples)}")
        print(f"Label distribution: {np.bincount(labels)}")
    
    # Create dataset
    dataset = SimplicialDataset(samples, labels)
    
    # Split dataset
    n_samples = len(dataset)
    n_train = int(n_samples * train_split)
    n_val = int(n_samples * val_split)
    n_test = n_samples - n_train - n_val
    
    has_saved_splits = (
        isinstance(split_indices, dict)
        and all(k in split_indices for k in ("train", "val", "test"))
    )

    if has_saved_splits:
        train_idx = [int(i) for i in split_indices["train"]]
        val_idx = [int(i) for i in split_indices["val"]]
        test_idx = [int(i) for i in split_indices["test"]]
        n_dataset = len(dataset)

        valid = all(0 <= i < n_dataset for i in train_idx + val_idx + test_idx)
        if not valid:
            raise ValueError("Loaded split indices contain out-of-range sample indices")

        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)
        test_dataset = Subset(dataset, test_idx)

        split_indices = {
            "train": train_idx,
            "val": val_idx,
            "test": test_idx,
        }
        n_train, n_val, n_test = len(train_idx), len(val_idx), len(test_idx)
    else:
        train_dataset, val_dataset, test_dataset = random_split(
            dataset,
            [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(random_seed),
        )

        split_indices = {
            "train": [int(i) for i in train_dataset.indices],
            "val": [int(i) for i in val_dataset.indices],
            "test": [int(i) for i in test_dataset.indices],
        }
    
    if verbose:
        print(f"Train: {n_train}, Validation: {n_val}, Test: {n_test}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=single_subject_collate_fn,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=single_subject_collate_fn,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=single_subject_collate_fn,
    )
    
    # Prepare metadata
    metadata = {
        "n_samples": len(dataset),
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "n_subjects": len(set(subject_ids)),
        "labels": labels,
        "subject_ids": subject_ids,
        "label_distribution": np.bincount(labels).tolist(),
        "loaded_from_processed": processed_loaded,
        "sindy_backend": str(sindy_backend).lower(),
    }

    if isinstance(loaded_metadata, dict):
        metadata.update({k: v for k, v in loaded_metadata.items() if k not in metadata})

    if save_processed and not processed_loaded:
        export_info = save_processed_dataset(
            samples=samples,
            labels=labels,
            split_indices=split_indices,
            metadata=metadata,
            output_dir=processed_output_dir,
            sindy_backend=sindy_backend,
        )
        metadata["processed_export"] = export_info
        if verbose:
            print(f"Saved processed dataset to: {export_info['output_dir']}")
            print(f"Fast cache: {export_info['fast_cache']}")
    elif processed_loaded:
        metadata["processed_export"] = {
            "output_dir": str(processed_backend_dir),
            "manifest": str(processed_backend_dir / "manifest.json"),
            "splits": str(processed_backend_dir / "splits.json"),
            "fast_cache": str(processed_backend_dir / FAST_CACHE_FILENAME),
            "num_subject_files": len(samples),
            "sindy_backend": _normalized_backend_name(sindy_backend),
        }
    

    return train_loader, val_loader, test_loader, metadata


if __name__ == "__main__":
    # Example usage
    train_loader, val_loader, test_loader, metadata = create_dataloaders(
        data_dir="./ABIDE_pcp",
        n_subjects="all",
        batch_size=1,
        verbose=True,
        save_processed=True,
        load_processed_if_available=True,
    )
    
    print(f"\nMetadata: {metadata}")
    
    # Test loading a batch
    for batch in train_loader:
        print(f"\nBatch keys: {batch.keys()}")
        print(f"Node features shape: {batch['node_features'].shape}")
        print(f"Edge features shape: {batch['edge_features'].shape}")
        print(f"Triangle features shape: {batch['triangle_features'].shape}")
        print(f"Labels: {batch['labels']}")
        print(f"Lengths: {batch['lengths']}")
        break
