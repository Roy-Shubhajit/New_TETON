#!/usr/bin/env python3
"""
Neural network architectures for simplicial complex classification.

These models process one subject at a time. Each subject is represented as a
sequence of simplicial-complex windows, and the temporal model is applied over
that single subject sequence before the training loop aggregates losses across
multiple subjects.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from topomodelx.nn.simplicial.sccn_layer import SCCNLayer


def _as_tensor(value: Any, device: torch.device, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Convert a numpy array or tensor to a tensor on the target device."""
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype) if dtype is not None else value.to(device=device)
    return torch.as_tensor(value, device=device, dtype=dtype)


def _rank_features(value: Any, device: torch.device) -> torch.Tensor:
    """Project raw rank features to a 2D tensor suitable for the linear encoders."""
    tensor = _as_tensor(value, device=device, dtype=torch.float32)
    if tensor.ndim == 0:
        tensor = tensor.view(1, 1)
    elif tensor.ndim == 1:
        tensor = tensor.unsqueeze(-1)
    elif tensor.ndim > 2:
        tensor = tensor.reshape(-1, tensor.shape[-1])
    return tensor


def _align_feature_dim(tensor: torch.Tensor, target_dim: int) -> torch.Tensor:
    """Align feature width to target_dim so projections are shape-safe."""
    if tensor.shape[-1] == target_dim:
        return tensor

    if target_dim == 1:
        return tensor.mean(dim=-1, keepdim=True)

    if tensor.shape[-1] > target_dim:
        return tensor[..., :target_dim]

    pad_width = target_dim - tensor.shape[-1]
    padding = torch.zeros(*tensor.shape[:-1], pad_width, device=tensor.device, dtype=tensor.dtype)
    return torch.cat([tensor, padding], dim=-1)


def _move_mapping(mapping: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Move numeric incidence or adjacency payloads to the target device."""
    moved: Dict[str, Any] = {}
    for key, value in mapping.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device=device, dtype=torch.float32)
        elif isinstance(value, np.ndarray):
            moved[key] = torch.as_tensor(value, device=device, dtype=torch.float32)
        else:
            moved[key] = value
    return moved


def _rectify_rank_mapping(mapping: Dict[Any, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    """Convert keys to 'rank_X' format and values to float32 tensors on the target device."""
    converted = {}
    for k, v in mapping.items():
        new_key = f"rank_{k}" if isinstance(k, int) or (isinstance(k, str) and k.isdigit()) else k
        if isinstance(v, torch.Tensor):
            converted[new_key] = v.to(device=device, dtype=torch.float32)
        else:
            converted[new_key] = torch.as_tensor(v, device=device, dtype=torch.float32)
    return converted


def _extract_windows(subject_data: Any) -> Sequence[Any]:
    """Return the list of simplicial windows for a subject sample."""
    if isinstance(subject_data, dict):
        if "windows" in subject_data:
            return subject_data["windows"]
        if {"features", "incidences", "adjacencies"}.issubset(subject_data):
            return [subject_data]
        if {"node_features", "edge_features", "triangle_features"}.issubset(subject_data):
            return [subject_data]
    if isinstance(subject_data, (list, tuple)):
        return subject_data
    raise TypeError("Subject data must be a dict with 'windows' or a sequence of windows")


def _get_rank_mapping(window: Any, attr_name: str, legacy_attr: str) -> Dict[str, Any]:
    """Return a rank-keyed mapping from a window, with legacy fallback support."""
    if isinstance(window, dict):
        mapping = window.get(attr_name)
    else:
        mapping = getattr(window, attr_name, None)
    if isinstance(mapping, dict):
        return mapping

    if not legacy_attr:
        return {}

    if isinstance(window, dict):
        legacy_mapping = window.get(legacy_attr)
    else:
        legacy_mapping = getattr(window, legacy_attr, None)
    if isinstance(legacy_mapping, dict):
        return legacy_mapping

    return {}


def _get_window_rank_features(window: Any) -> Dict[str, Any]:
    """Extract rank_0/rank_1/rank_2 feature tensors from a window."""
    features = _get_rank_mapping(window, "features", "")
    if features:
        return {
            "rank_0": features.get("rank_0", features.get(0, [])),
            "rank_1": features.get("rank_1", features.get(1, [])),
            "rank_2": features.get("rank_2", features.get(2, [])),
        }

    if isinstance(window, dict):
        return {
            "rank_0": window.get("node_features", []),
            "rank_1": window.get("edge_features", []),
            "rank_2": window.get("triangle_features", []),
        }

    return {
        "rank_0": getattr(window, "node_features", []),
        "rank_1": getattr(window, "edge_features", []),
        "rank_2": getattr(window, "triangle_features", []),
    }


def _get_window_rank_incidences(window: Any) -> Dict[str, Any]:
    """Extract rank_1/rank_2 incidence tensors from a window."""
    incidences = _get_rank_mapping(window, "incidences", "incidence")
    if incidences:
        return incidences

    if isinstance(window, dict):
        legacy = window.get("incidence", {})
    else:
        legacy = getattr(window, "incidence", {}) if hasattr(window, "incidence") else {}
    return {
        "rank_1": legacy.get("B1", []),
        "rank_2": legacy.get("B2", []),
    }


def _get_window_rank_adjacencies(window: Any) -> Dict[str, Any]:
    """Extract rank_0/rank_1/rank_2 adjacency tensors from a window."""
    adjacencies = _get_rank_mapping(window, "adjacencies", "adjacency")
    if adjacencies:
        return adjacencies

    if isinstance(window, dict):
        legacy = window.get("adjacency", {})
    else:
        legacy = getattr(window, "adjacency", {}) if hasattr(window, "adjacency") else {}
    return {
        "rank_0": legacy.get("H0_up", []),
        "rank_1": legacy.get("H1_down", []),
        "rank_2": legacy.get("H2_down", []),
    }


class _TemporalSCCNBase(nn.Module):
    """Shared helpers for the temporal SCCN variants."""

    def __init__(
        self,
        in_channel: int,
        hidden_channels: int,
        n_layers: int,
        dropout: float,
        update_func: str,
    ):
        super().__init__()
        self.in_channel = in_channel
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers

        # The initial projection maps input dimension straight to hidden_channels
        self.proj_0 = nn.Linear(in_channel, hidden_channels)
        self.proj_1 = nn.Linear(in_channel, hidden_channels)
        self.proj_2 = nn.Linear(in_channel, hidden_channels)

        self.conv_layers = nn.ModuleList(
            [
                SCCNLayer(
                    channels=hidden_channels,
                    max_rank=2,
                    update_func=update_func,
                )
                for _ in range(n_layers)
            ]
        )

    def _apply_sccn_stack(
        self,
        features: Dict[str, torch.Tensor],
        incidences: Dict[str, Any],
        adjacencies: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:

        rank_0 = features.get("rank_0", torch.empty((0, self.in_channel), device=next(self.parameters()).device))
        rank_1 = features.get("rank_1", torch.empty((0, self.in_channel), device=next(self.parameters()).device))
        rank_2 = features.get("rank_2", torch.empty((0, self.in_channel), device=next(self.parameters()).device))

        x_dict = {
            "rank_0": self.proj_0(rank_0) if rank_0.numel() > 0 else torch.empty((0, self.hidden_channels), device=rank_0.device),
            "rank_1": self.proj_1(rank_1) if rank_1.numel() > 0 else torch.empty((0, self.hidden_channels), device=rank_1.device),
            "rank_2": self.proj_2(rank_2) if rank_2.numel() > 0 else torch.empty((0, self.hidden_channels), device=rank_2.device),
        }

        if x_dict["rank_0"].numel() == 0:
            raise ValueError("Subject window has no node features")

        if x_dict["rank_1"].shape[0] == 0 or x_dict["rank_2"].shape[0] == 0:
            return x_dict

        for layer in self.conv_layers:
            x_dict = layer(x_dict, incidences, adjacencies)
        return x_dict


class TemporalSCCN_approach1(_TemporalSCCNBase):
    """SCCN over each window followed by a subject-level LSTM."""

    def __init__(
        self,
        in_channel,
        hidden_channels,
        out_channels,
        n_layers=2,
        dropout=0.3,
    ):
        super().__init__(
            in_channel,
            hidden_channels,
            n_layers,
            dropout,
            update_func="relu",
        )
        self.temporal_lstm = nn.LSTM(hidden_channels, hidden_channels, batch_first=True)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, windows):
        device = next(self.parameters()).device

        temporal_states: List[torch.Tensor] = []
        for window in windows:
            # We assume window is a dict with features, incidences, adjacencies
            if not isinstance(window, dict):
                # Fallback extraction if it's the raw object
                x_dict = _get_window_rank_features(window)
                inc_dict = _get_window_rank_incidences(window)
                adj_dict = _get_window_rank_adjacencies(window)
                x_dict = _rectify_rank_mapping(x_dict, device)
                incidences = _rectify_rank_mapping(inc_dict, device)
                adjacencies = _rectify_rank_mapping(adj_dict, device)
            else:
                x_dict = _rectify_rank_mapping(window.get("features", _get_window_rank_features(window)), device)
                incidences = _rectify_rank_mapping(window.get("incidences", _get_window_rank_incidences(window)), device)
                adjacencies = _rectify_rank_mapping(window.get("adjacencies", _get_window_rank_adjacencies(window)), device)

            x_dict = self._apply_sccn_stack(x_dict, incidences, adjacencies)
            temporal_states.append(x_dict["rank_0"].mean(dim=0, keepdim=True))

        if not temporal_states:
            raise ValueError("No valid temporal snapshots processed")

        seq = torch.stack(temporal_states, dim=1)
        lstm_out, _ = self.temporal_lstm(seq)
        subject_repr = lstm_out[:, -1, :]
        return self.classifier(subject_repr)


class TemporalSCCN_approach2(_TemporalSCCNBase):
    """Interleaved SCCN and LSTM updates across layers."""

    def __init__(
        self,
        in_channel,
        hidden_channels,
        out_channels,
        n_layers=2,
        dropout=0.3,
    ):
        super().__init__(
            in_channel,
            hidden_channels,
            n_layers,
            dropout,
            update_func="relu",
        )
        self.temp_layers = nn.ModuleList(
            [nn.LSTM(hidden_channels, hidden_channels, batch_first=True) for _ in range(n_layers)]
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * n_layers, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, windows):
        device = next(self.parameters()).device

        hidden_states: List[Optional[torch.Tensor]] = [None] * self.n_layers
        cell_states: List[Optional[torch.Tensor]] = [None] * self.n_layers
        #print(len(windows))
        for window_idx, window in windows.items():
            #print(window.keys())

            #check if .numel() is 0
            for key in ["features", "incidences", "adjacencies"]:
                if key not in window:
                    raise ValueError(f"Window {window_idx} missing '{key}' key")
                if not isinstance(window[key], dict):
                    raise ValueError(f"Window {window_idx} '{key}' is not a dict")
                for subkey, value in window[key].items():
                    if isinstance(value, torch.Tensor) and value.numel() == 0:
                        raise ValueError(f"Window {window_idx} '{key}' subkey '{subkey}' has empty tensor")
                    if isinstance(value, np.ndarray) and value.size == 0:
                        raise ValueError(f"Window {window_idx} '{key}' subkey '{subkey}' has empty array") 
            x_dict = _rectify_rank_mapping(window["features"], device)
            incidences = _rectify_rank_mapping(window["incidences"], device)
            adjacencies = _rectify_rank_mapping(window["adjacencies"], device)
            #print(f"X_dict keys: {list(x_dict.keys())}")
            #for key in x_dict.keys():
                #print(f"  {key} shape: {x_dict[key].shape}, dtype: {x_dict[key].dtype}")
            #print(f"Incidences keys: {list(incidences.keys())}")
            #for key in incidences.keys():
               # print(f"  {key} shape: {incidences[key].shape}, dtype: {incidences[key].dtype}")
            #print(f"Adjacencies keys: {list(adjacencies.keys())}")
            #for key in adjacencies.keys():
               # print(f"  {key} shape: {adjacencies[key].shape}, dtype: {adjacencies[key].dtype}")

            #check if all matrices are not nan
            

            # Apply input projections manually for approach2 since it interleaves layers differently
            #rank_0 = x_dict.get("rank_0", torch.empty((0, self.in_channel), device=device))
            #rank_1 = x_dict.get("rank_1", torch.empty((0, self.in_channel), device=device))
            #rank_2 = x_dict.get("rank_2", torch.empty((0, self.in_channel), device=device))
            
            #x_dict = {
             #   "rank_0": self.proj_0(rank_0) if rank_0.numel() > 0 else torch.empty((0, self.hidden_channels), device=device),
              #  "rank_1": self.proj_1(rank_1) if rank_1.numel() > 0 else torch.empty((0, self.hidden_channels), device=device),
               # "rank_2": self.proj_2(rank_2) if rank_2.numel() > 0 else torch.empty((0, self.hidden_channels), device=device),
            #}
            x_dict = {
                "rank_0": self.proj_0(x_dict["rank_0"]),
                "rank_1": self.proj_1(x_dict["rank_1"]),
                "rank_2": self.proj_2(x_dict["rank_2"]),
            }

            for key, mat in x_dict.items():
                if not torch.isfinite(mat).all():
                    raise ValueError(f"Non-finite values found in features for {key} at window {window_idx}")
            for key, mat in incidences.items():
                if not torch.isfinite(mat).all():
                    raise ValueError(f"Non-finite values found in incidences for {key} at window {window_idx}")
            for key, mat in adjacencies.items():
                if not torch.isfinite(mat).all():
                    raise ValueError(f"Non-finite values found in adjacencies for {key} at window {window_idx}")

            for key, mat in x_dict.items():
                if mat.numel() == 0:
                    raise ValueError(f"Empty tensor found in features for {key} at window {window_idx}")
            for key, mat in incidences.items():
                if mat.numel() == 0:
                    raise ValueError(f"Empty tensor found in incidences for {key} at window {window_idx}")
            for key, mat in adjacencies.items():
                if mat.numel() == 0:
                    raise ValueError(f"Empty tensor found in adjacencies for {key} at window {window_idx}")
                
            #print(f"X_dict keys: {list(x_dict.keys())}")
            #for key in x_dict.keys():
            #    print(f"  {key} shape: {x_dict[key].shape}, dtype: {x_dict[key].dtype}")
            #print(f"Incidences keys: {list(incidences.keys())}")
            #for key in incidences.keys():
            #    print(f"  {key} shape: {incidences[key].shape}, dtype: {incidences[key].dtype}")
            #print(f"Adjacencies keys: {list(adjacencies.keys())}")
            #for key in adjacencies.keys():
            #    print(f"  {key} shape: {adjacencies[key].shape}, dtype: {adjacencies[key].dtype}")
                

            for layer_idx in range(self.n_layers):
                x_dict = self.conv_layers[layer_idx](x_dict, incidences, adjacencies)
                for key, mat in x_dict.items():
                    if not torch.isfinite(mat).all():
                        raise ValueError(
                            f"Non-finite values introduced by SCCN layer {layer_idx} for {key} at window {window_idx}"
                        )
                if x_dict["rank_0"].shape[0] > 0:
                    z_0 = x_dict["rank_0"].mean(dim=0, keepdim=True)
                else:
                    z_0 = torch.zeros((1, self.hidden_channels), device=device)

                if hidden_states[layer_idx] is None:
                    hidden_states[layer_idx] = torch.zeros_like(z_0)
                    cell_states[layer_idx] = torch.zeros_like(z_0)

                _, (h_new, c_new) = self.temp_layers[layer_idx](
                    z_0.unsqueeze(1),
                    (hidden_states[layer_idx].unsqueeze(0), cell_states[layer_idx].unsqueeze(0)),
                )
                hidden_states[layer_idx] = h_new.squeeze(0)
                cell_states[layer_idx] = c_new.squeeze(0)

        if any(state is None for state in hidden_states):
            raise ValueError("No valid temporal snapshots processed")

        subject_repr = torch.cat([state for state in hidden_states if state is not None], dim=1)
        return self.classifier(subject_repr)


class TemporalSCCN_v3(_TemporalSCCNBase):
    """Subject-level temporal SCCN with per-rank LSTMs and attention."""

    def __init__(
        self,
        in_channel,
        hidden_channels,
        out_channels,
        n_layers=2,
        dropout=0.3,
    ):
        super().__init__(
            in_channel,
            hidden_channels,
            n_layers,
            dropout,
            update_func="relu",
        )
        self.lstm_0 = nn.LSTM(hidden_channels, hidden_channels, batch_first=True)
        self.lstm_1 = nn.LSTM(hidden_channels, hidden_channels, batch_first=True)
        self.lstm_2 = nn.LSTM(hidden_channels, hidden_channels, batch_first=True)
        attention_hidden = max(1, hidden_channels // 4)
        self.temporal_attention = nn.Sequential(
            nn.Linear(hidden_channels, attention_hidden),
            nn.Tanh(),
            nn.Linear(attention_hidden, 1),
        )
        self.fusion = nn.Sequential(
            nn.Linear(3 * hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels),
        )

    def forward(self, windows):
        device = next(self.parameters()).device

        temporal_z0: List[torch.Tensor] = []
        temporal_z1: List[torch.Tensor] = []
        temporal_z2: List[torch.Tensor] = []

        for window in windows:
            if not isinstance(window, dict):
                x_dict = _get_window_rank_features(window)
                inc_dict = _get_window_rank_incidences(window)
                adj_dict = _get_window_rank_adjacencies(window)
                x_dict = _rectify_rank_mapping(x_dict, device)
                incidences = _rectify_rank_mapping(inc_dict, device)
                adjacencies = _rectify_rank_mapping(adj_dict, device)
            else:
                x_dict = _rectify_rank_mapping(window.get("features", _get_window_rank_features(window)), device)
                incidences = _rectify_rank_mapping(window.get("incidences", _get_window_rank_incidences(window)), device)
                adjacencies = _rectify_rank_mapping(window.get("adjacencies", _get_window_rank_adjacencies(window)), device)
            
            x_dict = self._apply_sccn_stack(x_dict, incidences, adjacencies)

            if x_dict["rank_0"].shape[0] > 0:
                temporal_z0.append(x_dict["rank_0"].mean(dim=0, keepdim=True))
            else:
                temporal_z0.append(torch.zeros((1, self.hidden_channels), device=device))

            if x_dict["rank_1"].shape[0] > 0:
                temporal_z1.append(x_dict["rank_1"].mean(dim=0, keepdim=True))
            else:
                temporal_z1.append(torch.zeros((1, self.hidden_channels), device=device))

            if x_dict["rank_2"].shape[0] > 0:
                temporal_z2.append(x_dict["rank_2"].mean(dim=0, keepdim=True))
            else:
                temporal_z2.append(torch.zeros((1, self.hidden_channels), device=device))

        if not temporal_z0:
            raise ValueError("No valid temporal snapshots processed")

        seq_0 = torch.stack(temporal_z0, dim=1)
        seq_1 = torch.stack(temporal_z1, dim=1)
        seq_2 = torch.stack(temporal_z2, dim=1)

        out_0, _ = self.lstm_0(seq_0)
        out_1, _ = self.lstm_1(seq_1)
        out_2, _ = self.lstm_2(seq_2)

        attn_scores = self.temporal_attention(out_0.squeeze(0))
        attn_weights = torch.softmax(attn_scores, dim=0)

        agg_0 = (out_0.squeeze(0) * attn_weights).sum(dim=0, keepdim=True)
        agg_1 = (out_1.squeeze(0) * attn_weights).sum(dim=0, keepdim=True)
        agg_2 = (out_2.squeeze(0) * attn_weights).sum(dim=0, keepdim=True)

        fused = torch.cat([agg_0, agg_1, agg_2], dim=1)
        fused = self.fusion(fused)
        return self.classifier(fused)


class TemporalSCCN_Transformer(_TemporalSCCNBase):
    """Temporal SCCN using a Transformer encoder over subject windows."""

    def __init__(
        self,
        in_channel,
        hidden_channels,
        out_channels,
        n_layers=2,
        n_heads=4,
        dropout=0.3,
    ):
        super().__init__(
            in_channel,
            hidden_channels,
            n_layers,
            dropout,
            update_func="relu",
        )
        self.sequence_dim = hidden_channels * 3
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.sequence_dim,
            nhead=n_heads,
            dim_feedforward=self.sequence_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Properly initialize CLS token and positional encoding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.sequence_dim))
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        
        self.pos_encoder = nn.Parameter(torch.zeros(1, 512, self.sequence_dim))
        nn.init.normal_(self.pos_encoder, mean=0.0, std=0.02)
        
        # Layer norm before transformer for stable training
        self.embed_ln = nn.LayerNorm(self.sequence_dim)
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.sequence_dim),
            nn.Linear(self.sequence_dim, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, windows):
        device = next(self.parameters()).device

        temporal_features: List[torch.Tensor] = []
        for window in windows:
            if not isinstance(window, dict):
                x_dict = _get_window_rank_features(window)
                inc_dict = _get_window_rank_incidences(window)
                adj_dict = _get_window_rank_adjacencies(window)
                x_dict = _rectify_rank_mapping(x_dict, device)
                incidences = _rectify_rank_mapping(inc_dict, device)
                adjacencies = _rectify_rank_mapping(adj_dict, device)
            else:
                x_dict = _rectify_rank_mapping(window.get("features", _get_window_rank_features(window)), device)
                incidences = _rectify_rank_mapping(window.get("incidences", _get_window_rank_incidences(window)), device)
                adjacencies = _rectify_rank_mapping(window.get("adjacencies", _get_window_rank_adjacencies(window)), device)
            
            x_dict = self._apply_sccn_stack(x_dict, incidences, adjacencies)
            
            # Safely handle potentially empty rank features
            rank_features = []
            
            # rank_0 is always present
            if x_dict["rank_0"].shape[0] > 0:
                rank_features.append(x_dict["rank_0"].mean(dim=0))
            else:
                rank_features.append(torch.zeros(self.hidden_channels, device=device))
            
            # rank_1 may be empty
            if x_dict["rank_1"].shape[0] > 0:
                rank_features.append(x_dict["rank_1"].mean(dim=0))
            else:
                rank_features.append(torch.zeros(self.hidden_channels, device=device))
            
            # rank_2 may be empty
            if x_dict["rank_2"].shape[0] > 0:
                rank_features.append(x_dict["rank_2"].mean(dim=0))
            else:
                rank_features.append(torch.zeros(self.hidden_channels, device=device))
            
            temporal_features.append(torch.cat(rank_features, dim=0))

        if not temporal_features:
            raise ValueError("No valid temporal snapshots processed")

        seq = torch.stack(temporal_features, dim=0).unsqueeze(0)
        seq_len = seq.size(1) + 1
        if seq_len > self.pos_encoder.size(1):
            raise ValueError(
                f"Sequence length {seq_len} exceeds the maximum positional encoding length {self.pos_encoder.size(1)}"
            )

        cls_tokens = self.cls_token.expand(1, -1, -1)
        seq = torch.cat([cls_tokens, seq], dim=1)
        
        # Add positional encoding and apply layer norm for stability
        seq = seq + self.pos_encoder[:, :seq_len, :]
        seq = self.embed_ln(seq)

        out = self.transformer(seq)
        cls_output = out[:, 0, :]
        return self.classifier(cls_output)


def create_model(
    model_name: str,
    input_dim: int = 1,
    sccn_hidden: int = 64,
    num_sccn_layers: int = 2,
    lstm_hidden: int = 128,
    lstm_layers: int = 1,
    num_classes: int = 2,
    dropout: float = 0.0,
) -> nn.Module:
    """Factory function to create the requested temporal SCCN model."""
    aliases = {
        "SCCN_LSTM": "TemporalSCCN_approach1",
        "SCCN_Pool": "TemporalSCCN_approach2",
        "SCCN_Attention": "TemporalSCCN_v3",
        "SCCN_Transformer": "TemporalSCCN_Transformer",
    }
    resolved_name = aliases.get(model_name, model_name)

    if resolved_name == "TemporalSCCN_approach1":
        return TemporalSCCN_approach1(
            in_channel=input_dim,
            hidden_channels=sccn_hidden,
            out_channels=num_classes,
            n_layers=num_sccn_layers,
            dropout=dropout,
        )
    if resolved_name == "TemporalSCCN_approach2":
        return TemporalSCCN_approach2(
            in_channel=input_dim,
            hidden_channels=sccn_hidden,
            out_channels=num_classes,
            n_layers=num_sccn_layers,
            dropout=dropout,
        )
    if resolved_name == "TemporalSCCN_v3":
        return TemporalSCCN_v3(
            in_channel=input_dim,
            hidden_channels=sccn_hidden,
            out_channels=num_classes,
            n_layers=num_sccn_layers,
            dropout=dropout,
        )
    if resolved_name == "TemporalSCCN_Transformer":
        return TemporalSCCN_Transformer(
            in_channel=input_dim,
            hidden_channels=sccn_hidden,
            out_channels=num_classes,
            n_layers=num_sccn_layers,
            dropout=dropout,
        )

    raise ValueError(f"Unknown model: {model_name}")


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
