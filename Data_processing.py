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
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from tqdm import tqdm

from faster_sindy import (
    PreprocessConfig, SelectionConfig, SolverConfig, ThresholdConfig,
    run_sindy_windows
)
from Helper import (
    SimplicialComplex, cubic_spline_upsample_timeseries
)


def _safe_subject_filename(subject_id: str) -> str:
    cleaned = [ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(subject_id)]
    return "".join(cleaned).strip("._") or "subject"


def _window_to_exportable(window: SimplicialComplex) -> Dict[str, Any]:
    return {
        "node_features": np.asarray(window.node_features, dtype=np.float32),
        "edge_features": np.asarray(window.edge_features, dtype=np.float32),
        "triangle_features": np.asarray(window.triangle_features, dtype=np.float32),
        "edges": np.asarray(window.incidence["edges"], dtype=np.int32),
        "triangles": np.asarray(window.incidence["triangles"], dtype=np.int32),
        "B1": np.asarray(window.incidence["B1"], dtype=np.int8),
        "B2": np.asarray(window.incidence["B2"], dtype=np.int8),
        "H0_up": np.asarray(window.adjacency["H0_up"], dtype=np.float32),
        "H1_down": np.asarray(window.adjacency["H1_down"], dtype=np.float32),
        "H1_up": np.asarray(window.adjacency["H1_up"], dtype=np.float32),
        "H2_down": np.asarray(window.adjacency["H2_down"], dtype=np.float32),
    }


def save_processed_dataset(
    samples: List[Dict[str, Any]],
    labels: List[int],
    split_indices: Dict[str, List[int]],
    metadata: Dict[str, Any],
    output_dir: str,
) -> Dict[str, Any]:
    """Save processed simplicial dataset so others can reuse it without recomputation."""
    out_dir = Path(output_dir)
    subjects_dir = out_dir / "subjects"
    out_dir.mkdir(parents=True, exist_ok=True)
    subjects_dir.mkdir(parents=True, exist_ok=True)

    subject_records: List[Dict[str, Any]] = []

    for i, sample in enumerate(samples):
        subject_id = str(sample.get("subject_id", f"subject_{i:04d}"))
        filename = f"{i:04d}_{_safe_subject_filename(subject_id)}.npz"
        file_path = subjects_dir / filename

        windows = sample.get("windows", [])
        window_payloads = [_window_to_exportable(window) for window in windows]

        np.savez_compressed(
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

    export_summary = {
        "output_dir": str(out_dir),
        "manifest": str(manifest_path),
        "splits": str(split_path),
        "num_subject_files": len(subject_records),
    }
    return export_summary


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
        - node_features: (batch_size, max_seq_len, max_nodes)
        - edge_features: (batch_size, max_seq_len, max_edges, 4)
        - triangle_features: (batch_size, max_seq_len, max_triangles, 4)
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
    
    for windows in batch_windows:
        for window in windows:
            max_nodes = max(max_nodes, window.n_nodes)
            max_edges = max(max_edges, window.incidence["n_edges"])
            max_triangles = max(max_triangles, window.incidence["n_triangles"])
    
    # Initialize padded tensors
    node_features = torch.zeros(
        (batch_size, max_seq_len, max_nodes),
        dtype=torch.float32
    )
    edge_features = torch.zeros(
        (batch_size, max_seq_len, max_edges, 4),
        dtype=torch.float32
    )
    triangle_features = torch.zeros(
        (batch_size, max_seq_len, max_triangles, 4),
        dtype=torch.float32
    )
    
    # Fill tensors
    for b_idx, windows in enumerate(batch_windows):
        for w_idx, window in enumerate(windows):
            # Node features
            n_nodes = window.n_nodes
            node_features[b_idx, w_idx, :n_nodes] = torch.from_numpy(window.node_features)
            
            # Edge features
            n_edges = window.edge_features.shape[0]
            if n_edges > 0:
                edge_features[b_idx, w_idx, :n_edges] = torch.from_numpy(window.edge_features)
            
            # Triangle features
            n_triangles = window.triangle_features.shape[0]
            if n_triangles > 0:
                triangle_features[b_idx, w_idx, :n_triangles] = torch.from_numpy(
                    window.triangle_features
                )
    
    return {
        "node_features": node_features,
        "edge_features": edge_features,
        "triangle_features": triangle_features,
        "labels": batch_labels,
        "lengths": batch_lengths,
        "subject_ids": batch_subject_ids,
    }


class ABIDESimplicialProcessor:
    """
    Main processor for converting ABIDE to simplicial complexes.
    """
    
    def __init__(
        self,
        data_dir: str = "./",
        save_intermediates: bool = False,
        upsample_factor: int = 100,
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

            n_nodes, n_timepoints = X_raw.shape

            # Apply upsampling interpolation if factor > 1
            if self.upsample_factor > 1:
                X_raw = cubic_spline_upsample_timeseries(X_raw.T, factor=self.upsample_factor).T

            stride = max(1, int(round(window_len * (1.0 - window_overlap))))

            # Run SINDy window solver and use its predicted edges/triangles directly.
            sindy_windows = run_sindy_windows(
                X_raw,
                window_size=window_len,
                stride=stride,
                preprocess_cfg=self.preprocess_cfg,
                selection_cfg=self.selection_cfg,
                solver_cfg=self.solver_cfg,
                threshold_cfg=self.threshold_cfg,
            )

            if not sindy_windows:
                print(f"  [skip] Subject {subject_id}: no valid windows")
                return {"windows": [], "subject_id": subject_id, "error": "no_windows"}

            for w_idx, result in enumerate(sindy_windows):
                try:
                    start = int(result.start)
                    end = int(result.end)

                    # Temporal node values for this processed window.
                    node_temporal = X_raw[:, start:end].mean(axis=1).astype(np.float32, copy=False)

                    edges = [tuple(sorted(tuple(edge))) for edge in result.pred_edges]
                    triangles = [tuple(sorted(tuple(tri))) for tri in result.pred_tris]

                    simp = SimplicialComplex(
                        node_features=node_temporal,
                        edges=edges,
                        triangles=triangles,
                    )

                    windows.append(simp)

                except Exception as e:
                    print(f"  [warn] Subject {subject_id}, window {w_idx}: {e}")
                    continue
            
            if not windows:
                print(f"  [skip] Subject {subject_id}: no valid windows created")
                return {"windows": [], "subject_id": subject_id, "error": "no_valid_windows"}
            
            print(f"  [ok] Subject {subject_id}: {len(windows)} windows created")
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
    upsample_factor: int = 100,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """
    Create train, validation, and test dataloaders for ABIDE data.
    
    Args:
        data_dir: Path to ABIDE dataset
        n_subjects: Number of subjects to process (None for all)
        window_length: Length of each processing window
        window_overlap: Overlap between windows
        batch_size: Batch size for dataloaders
        train_split: Fraction for training (0-1)
        val_split: Fraction for validation (0-1)
        test_split: Fraction for testing (0-1)
        random_seed: Random seed for reproducibility
        num_workers: Number of workers for data loading
        verbose: Whether to print progress
        save_processed: Whether to export processed dataset to disk
        processed_output_dir: Directory where processed dataset export is written
        upsample_factor: Interpolation factor for upsampling timeseries
    
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
        print("Processing dataset into simplicial complexes...")
    
    processor = ABIDESimplicialProcessor(
        data_dir=data_dir,
        save_intermediates=False,
        upsample_factor=upsample_factor,
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
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
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
    }

    if save_processed:
        export_info = save_processed_dataset(
            samples=samples,
            labels=labels,
            split_indices=split_indices,
            metadata=metadata,
            output_dir=processed_output_dir,
        )
        metadata["processed_export"] = export_info
        if verbose:
            print(f"Saved processed dataset to: {export_info['output_dir']}")
            print(f"Manifest: {export_info['manifest']}")
    
    return train_loader, val_loader, test_loader, metadata


if __name__ == "__main__":
    # Example usage
    train_loader, val_loader, test_loader, metadata = create_dataloaders(
        data_dir="./ABIDE_pcp",
        n_subjects="all",
        batch_size=16,
        verbose=True,
        save_processed=True,
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
