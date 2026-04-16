#!/usr/bin/env python3
"""
Helper functions for ABIDE SINDy ML Pipeline.

Contains utilities for:
- Loading configuration files
- Creating incidence and adjacency matrices
- Processing simplicial complex data
- Creating feature vectors from topology
"""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from scipy import sparse
from itertools import combinations
from scipy.interpolate import CubicSpline


class Config:
    """Configuration management for the pipeline."""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize with config dictionary or defaults."""
        self.config = config_dict or self._default_config()
    
    @staticmethod
    def _default_config() -> Dict[str, Any]:
        """Return default configuration."""
        return {
            # Dataset settings
            "dataset": {
                "n_subjects": None,  # None fetches all subjects
                "derivatives": "rois_cc200",
                "data_dir": "./ABIDE_pcp",
            },
            # Processing settings
            "processing": {
                "upsample_factor": 100,
                "window_length_base": 10,
                "window_overlap": 0.25,
                "d_max": 2,
                "tau2_q": 0.8,
                "tau3_q": 0.5,
            },
            # Model settings
            "model": {
                "name": "TemporalSCCN_v3",
                "sccn_layers": 2,
                "lstm_hidden": 512,
                "lstm_layers": 2,
                "dropout": 0.3,
                "output_classes": 2,  # DX_GROUP (ASD vs Control)
            },
            # Training settings
            "training": {
                "batch_size": 32,  # Number of subjects per optimizer step
                "num_epochs": 50,
                "learning_rate": 0.001,
                "weight_decay": 1e-5,
                "train_split": 0.7,
                "val_split": 0.15,
                "test_split": 0.15,
                "random_seed": 42,
            },
            # Logging settings
            "logging": {
                "log_dir": "./logs",
                "checkpoint_dir": "./checkpoints",
                "verbose": True,
            }
        }
    
    @classmethod
    def from_file(cls, config_path: str) -> "Config":
        """Load configuration from YAML or JSON file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.suffix == '.json':
                config_dict = json.load(f)
            elif config_path.suffix in {'.yaml', '.yml'}:
                config_dict = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")
        
        return cls(config_dict)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot notation (e.g., 'model.name')."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value if value is not None else default
    
    def __getitem__(self, key: str) -> Any:
        """Allow dict-style access."""
        return self.get(key)
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self.config.copy()
    
    def save(self, config_path: str) -> None:
        """Save configuration to YAML file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

def cubic_spline_upsample_timeseries(subject_ts: np.ndarray, factor: int) -> np.ndarray:
    """Upsample one subject time series shaped [timepoints, rois]."""
    if factor < 1:
        raise ValueError("factor must be >= 1")

    num_timepoints, num_rois = subject_ts.shape
    if factor == 1:
        return subject_ts.copy()

    x_old = np.arange(num_timepoints, dtype=float)
    num_timepoints_new = num_timepoints * factor
    x_new = np.linspace(0.0, float(num_timepoints - 1), num_timepoints_new)

    upsampled = np.empty((num_timepoints_new, num_rois), dtype=float)
    for roi_idx in range(num_rois):
        spline = CubicSpline(x_old, subject_ts[:, roi_idx], bc_type="natural")
        upsampled[:, roi_idx] = spline(x_new)
    return upsampled

def build_simplicial_incidence_matrices(
    n_nodes: int,
    edges: List[Tuple[int, int]],
    triangles: List[Tuple[int, int, int]],
) -> Dict[str, np.ndarray]:
    """
    Build incidence matrices for simplicial complex.
    
    Incidence matrices B_r map r-cells to (r-1)-cells:
    - B_1: edges to nodes (n_nodes x n_edges)
    - B_2: triangles to edges (n_edges x n_triangles)
    
    Args:
        n_nodes: Number of nodes
        edges: List of (i, j) edges
        triangles: List of (i, j, k) triangles
    
    Returns:
        Dictionary with incidence matrices
    """
    edges = [tuple(sorted(e)) for e in edges]
    triangles = [tuple(sorted(t)) for t in triangles]
    
    # Remove duplicates and sort
    edges = sorted(set(edges))
    triangles = sorted(set(triangles))
    
    n_edges = len(edges)
    n_triangles = len(triangles)
    
    # B_1: nodes to edges incidence matrix (n_nodes x n_edges)
    B1 = np.zeros((n_nodes, n_edges), dtype=np.int8)
    for e_idx, (i, j) in enumerate(edges):
        B1[i, e_idx] = 1
        B1[j, e_idx] = 1
    
    # B_2: edges to triangles incidence matrix (n_edges x n_triangles)
    B2 = np.zeros((n_edges, n_triangles), dtype=np.int8)
    edge_to_idx = {edge: idx for idx, edge in enumerate(edges)}
    
    for t_idx, (i, j, k) in enumerate(triangles):
        # Each triangle has 3 edges
        tri_edges = [
            tuple(sorted([i, j])),
            tuple(sorted([j, k])),
            tuple(sorted([i, k])),
        ]
        for edge in tri_edges:
            if edge in edge_to_idx:
                B2[edge_to_idx[edge], t_idx] = 1
    
    return {
        "B1": B1,  # nodes to edges
        "B2": B2,  # edges to triangles
        "edges": np.array(edges, dtype=np.int32),
        "triangles": np.array(triangles, dtype=np.int32),
        "n_edges": n_edges,
        "n_triangles": n_triangles,
    }


def build_simplicial_adjacency_matrices(
    n_nodes: int,
    edges: List[Tuple[int, int]],
    triangles: List[Tuple[int, int, int]],
    edge_weights: Optional[np.ndarray] = None,
    triangle_weights: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Build adjacency matrices for simplicial complex.
    
    Adjacency matrices H_r map cells to cells via lower and upper cells:
    - H_0_up: node adjacency via edge (n_nodes x n_nodes)
    - H_1_down: edge adjacency via nodes (n_edges x n_edges)
    - H_1_up: edge adjacency via triangles (n_edges x n_edges)
    - H_2_down: triangle adjacency via edges (n_triangles x n_triangles)
    
    Args:
        n_nodes: Number of nodes
        edges: List of (i, j) edges
        triangles: List of (i, j, k) triangles
        edge_weights: Optional weights for edges
        triangle_weights: Optional weights for triangles
    
    Returns:
        Dictionary with adjacency matrices
    """
    edges = [tuple(sorted(e)) for e in edges]
    edges = sorted(set(edges))
    n_edges = len(edges)
    
    # H_0_up: node adjacency (via edges)
    H0_up = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for e_idx, (i, j) in enumerate(edges):
        weight = edge_weights[e_idx] if edge_weights is not None else 1.0
        H0_up[i, j] = weight
        H0_up[j, i] = weight
    
    # H_1_down: edge adjacency via shared nodes
    H1_down = np.zeros((n_edges, n_edges), dtype=np.float32)
    for e1_idx, (i1, j1) in enumerate(edges):
        for e2_idx, (i2, j2) in enumerate(edges):
            if e1_idx != e2_idx:
                # Edges are adjacent if they share a node
                if len({i1, j1} & {i2, j2}) > 0:
                    H1_down[e1_idx, e2_idx] = 1.0
    
    # H_1_up: edge adjacency via triangles
    H1_up = np.zeros((n_edges, n_edges), dtype=np.float32)
    triangles = [tuple(sorted(t)) for t in triangles]
    triangles = sorted(set(triangles))
    edge_to_idx = {edge: idx for idx, edge in enumerate(edges)}
    
    for tri in triangles:
        i, j, k = tri
        tri_edges = [
            tuple(sorted([i, j])),
            tuple(sorted([j, k])),
            tuple(sorted([i, k])),
        ]
        tri_edge_indices = [edge_to_idx[e] for e in tri_edges if e in edge_to_idx]
        # Connect all edges in triangle
        for idx1 in tri_edge_indices:
            for idx2 in tri_edge_indices:
                if idx1 != idx2:
                    H1_up[idx1, idx2] = 1.0
    
    # H_2_down: triangle adjacency via shared edges
    n_triangles = len(triangles)
    H2_down = np.zeros((n_triangles, n_triangles), dtype=np.float32)
    
    for t1_idx, tri1 in enumerate(triangles):
        for t2_idx, tri2 in enumerate(triangles):
            if t1_idx != t2_idx:
                # Triangles are adjacent if they share an edge
                tri1_edges = {tuple(sorted([tri1[i], tri1[j]])) 
                             for i, j in combinations(range(3), 2)}
                tri2_edges = {tuple(sorted([tri2[i], tri2[j]])) 
                             for i, j in combinations(range(3), 2)}
                if len(tri1_edges & tri2_edges) > 0:
                    H2_down[t1_idx, t2_idx] = 1.0
    
    return {
        "H0_up": H0_up,      # node adjacency via edges
        "H1_down": H1_down,  # edge adjacency via nodes
        "H1_up": H1_up,      # edge adjacency via triangles
        "H2_down": H2_down,  # triangle adjacency via edges
    }


def _ensure_temporal_matrix(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"Expected node features with shape (n_nodes,) or (n_nodes, window_length), got {arr.shape}")
    return arr


def create_node_edge_features(
    node_temporal_values: np.ndarray,
    edges: List[Tuple[int, int]],
) -> Dict[str, np.ndarray]:
    """
    Create edge features from node temporal values.
    
    Args:
        node_temporal_values: Shape (n_nodes,)
        edges: List of (i, j) edges
    
    Returns:
        Dictionary with edge feature statistics
    """
    x0 = _ensure_temporal_matrix(node_temporal_values)
    edges = [tuple(sorted(e)) for e in edges]
    n_edges = len(edges)
    feat_dim = x0.shape[1]
    
    # Edge features preserve the temporal axis and combine the corresponding nodes pointwise.
    edge_features = np.zeros((n_edges, feat_dim), dtype=np.float32)
    
    for e_idx, (i, j) in enumerate(edges):
        node_vals = x0[[i, j]]
        edge_features[e_idx] = node_vals.mean(axis=0)
    
    return {
        "edge_features": edge_features,
        "n_edge_features": feat_dim,
    }


def create_node_triangle_features(
    node_temporal_values: np.ndarray,
    triangles: List[Tuple[int, int, int]],
) -> Dict[str, np.ndarray]:
    """
    Create triangle features from node temporal values.
    
    Args:
        node_temporal_values: Shape (n_nodes,)
        triangles: List of (i, j, k) triangles
    
    Returns:
        Dictionary with triangle feature statistics
    """
    x0 = _ensure_temporal_matrix(node_temporal_values)
    triangles = [tuple(sorted(t)) for t in triangles]
    n_triangles = len(triangles)
    feat_dim = x0.shape[1]
    
    # Triangle features preserve the temporal axis and combine the corresponding nodes pointwise.
    triangle_features = np.zeros((n_triangles, feat_dim), dtype=np.float32)
    
    for t_idx, (i, j, k) in enumerate(triangles):
        node_vals = x0[[i, j, k]]
        triangle_features[t_idx] = node_vals.mean(axis=0)
    
    return {
        "triangle_features": triangle_features,
        "n_triangle_features": feat_dim,
    }


class SimplicialComplex:
    """
    Represents a simplicial complex for a single window.
    
    Contains node, edge, and triangle features along with incidence/adjacency matrices.
    """
    
    def __init__(
        self,
        node_features: np.ndarray,
        edges: List[Tuple[int, int]],
        triangles: List[Tuple[int, int, int]],
        edge_features: Optional[np.ndarray] = None,
        triangle_features: Optional[np.ndarray] = None,
        edge_weights: Optional[np.ndarray] = None,
        triangle_weights: Optional[np.ndarray] = None,
    ):
        """
        Initialize simplicial complex.
        
        Args:
            node_features: Shape (n_nodes,) or (n_nodes, window_length)
            edges: List of (i, j) edges
            triangles: List of (i, j, k) triangles
            edge_features: Optional shape (n_edges, feature_dim)
            triangle_features: Optional shape (n_triangles, feature_dim)
            edge_weights: Optional shape (n_edges,)
            triangle_weights: Optional shape (n_triangles,)
        """
        self.node_features = np.asarray(node_features, dtype=np.float32)
        self.n_nodes = len(self.node_features)
        self.edges = edges
        self.triangles = triangles
        
        # Build incidence matrices
        self.incidence = build_simplicial_incidence_matrices(
            self.n_nodes, edges, triangles
        )
        
        # Build adjacency matrices
        self.adjacency = build_simplicial_adjacency_matrices(
            self.n_nodes, edges, triangles, edge_weights, triangle_weights
        )
        
        # Set or compute edge features
        if edge_features is None:
            edge_feat_dict = create_node_edge_features(node_features, edges)
            self.edge_features = edge_feat_dict["edge_features"]
        else:
            self.edge_features = edge_features
        
        # Set or compute triangle features
        if triangle_features is None:
            tri_feat_dict = create_node_triangle_features(node_features, triangles)
            self.triangle_features = tri_feat_dict["triangle_features"]
        else:
            self.triangle_features = triangle_features
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "n_nodes": self.n_nodes,
            "node_features": self.node_features,
            "edges": self.edges,
            "triangles": self.triangles,
            "edge_features": self.edge_features,
            "triangle_features": self.triangle_features,
            "incidence": self.incidence,
            "adjacency": self.adjacency,
        }
    
    def get_sparse_incidence(self) -> Dict[str, sparse.csr_matrix]:
        """Get sparse versions of incidence matrices."""
        return {
            "B1": sparse.csr_matrix(self.incidence["B1"]),
            "B2": sparse.csr_matrix(self.incidence["B2"]),
        }
    
    def get_sparse_adjacency(self) -> Dict[str, sparse.csr_matrix]:
        """Get sparse versions of adjacency matrices."""
        return {
            "H0_up": sparse.csr_matrix(self.adjacency["H0_up"]),
            "H1_down": sparse.csr_matrix(self.adjacency["H1_down"]),
            "H1_up": sparse.csr_matrix(self.adjacency["H1_up"]),
            "H2_down": sparse.csr_matrix(self.adjacency["H2_down"]),
        }


def pad_sequence_windows(
    windows: List[SimplicialComplex],
    pad_value: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pad sequences of simplicial complexes to the same length.
    
    Args:
        windows: List of SimplicialComplex objects
        pad_value: Value to use for padding
    
    Returns:
        Tuple of (padded_node_features, padded_edge_features, padded_triangle_features)
        with shape (n_windows, max_dim, feature_dim)
    """
    if not windows:
        raise ValueError("Empty windows list")
    
    max_nodes = max(w.n_nodes for w in windows)
    max_edges = max(w.incidence["n_edges"] for w in windows)
    max_triangles = max(w.incidence["n_triangles"] for w in windows)
    
    n_windows = len(windows)
    node_feat_dim = windows[0].node_features.shape[1] if np.asarray(windows[0].node_features).ndim > 1 else 1
    n_edge_features = windows[0].edge_features.shape[1] if windows[0].edge_features.size > 0 else node_feat_dim
    n_tri_features = windows[0].triangle_features.shape[1] if windows[0].triangle_features.size > 0 else node_feat_dim
    
    node_features_pad = np.full((n_windows, max_nodes, node_feat_dim), pad_value, dtype=np.float32)
    edge_features_pad = np.full((n_windows, max_edges, n_edge_features), pad_value, dtype=np.float32)
    tri_features_pad = np.full((n_windows, max_triangles, n_tri_features), pad_value, dtype=np.float32)
    
    for w_idx, window in enumerate(windows):
        # Pad node features
        n = window.n_nodes
        node_arr = np.asarray(window.node_features, dtype=np.float32)
        if node_arr.ndim == 1:
            node_arr = node_arr[:, None]
        node_features_pad[w_idx, :n, :node_arr.shape[1]] = node_arr
        
        # Pad edge features
        n_e = window.edge_features.shape[0]
        if n_e > 0:
            edge_features_pad[w_idx, :n_e] = window.edge_features
        
        # Pad triangle features
        n_t = window.triangle_features.shape[0]
        if n_t > 0:
            tri_features_pad[w_idx, :n_t] = window.triangle_features
    
    return node_features_pad, edge_features_pad, tri_features_pad


def get_label_mapping(phenotype_data: np.ndarray) -> Dict[int, str]:
    """
    Get label mapping for classification task.
    
    Args:
        phenotype_data: Class labels (0, 1, etc.)
    
    Returns:
        Dictionary mapping class indices to names
    """
    mapping = {}
    for idx in np.unique(phenotype_data):
        if int(idx) == 0:
            mapping[int(idx)] = "Control"
        elif int(idx) == 1:
            mapping[int(idx)] = "ASD"
        else:
            mapping[int(idx)] = f"Class_{int(idx)}"
    return mapping


def print_config(config: Config) -> None:
    """Pretty print configuration."""
    import json
    print("Configuration:")
    print(json.dumps(config.to_dict(), indent=2))
