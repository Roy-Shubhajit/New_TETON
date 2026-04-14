#!/usr/bin/env python3
"""
Neural network architectures for simplicial complex classification.

Models combine SCCN layers (from TopoModelX) with LSTM layers to handle
temporal sequences of simplicial complexes.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
import numpy as np


class SCCNLayer(nn.Module):
    """
    Simplicial Complex Convolutional Network (SCCN) Layer.
    
    Wrapper around TopoModelX SCCN for ease of use. This simplified version
    implements message passing on simplicial complexes.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        max_rank: int = 2,
        dropout: float = 0.0,
    ):
        """
        Initialize SCCN layer.
        
        Args:
            in_channels: Input feature dimension
            out_channels: Output feature dimension
            max_rank: Maximum rank (0=nodes, 1=edges, 2=triangles)
            dropout: Dropout probability
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_rank = max_rank
        self.dropout = nn.Dropout(dropout)
        
        # Learnable transformations for each rank
        self.node_lin = nn.Linear(in_channels, out_channels)
        self.edge_lin = nn.Linear(in_channels, out_channels) if max_rank >= 1 else None
        self.tri_lin = nn.Linear(in_channels, out_channels) if max_rank >= 2 else None
        
        # Message aggregation weights
        self.node_agg_weight = nn.Parameter(torch.ones(1))
        self.edge_agg_weight = nn.Parameter(torch.ones(1)) if max_rank >= 1 else None
        self.tri_agg_weight = nn.Parameter(torch.ones(1)) if max_rank >= 2 else None
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None,
        triangle_features: Optional[torch.Tensor] = None,
        adjacency_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass through SCCN layer.
        
        Args:
            node_features: Shape (batch_size, n_nodes, in_channels)
            edge_features: Shape (batch_size, n_edges, in_channels)
            triangle_features: Shape (batch_size, n_triangles, in_channels)
            adjacency_dict: Dictionary with adjacency matrices
        
        Returns:
            Tuple of (node_out, edge_out, triangle_out)
        """
        # Transform node features
        node_out = self.node_lin(node_features)
        node_out = self.dropout(node_out)
        
        # Transform edge features (if present)
        edge_out = None
        if self.edge_lin is not None and edge_features is not None:
            edge_out = self.edge_lin(edge_features)
            edge_out = self.dropout(edge_out)
        
        # Transform triangle features (if present)
        tri_out = None
        if self.tri_lin is not None and triangle_features is not None:
            tri_out = self.tri_lin(triangle_features)
            tri_out = self.dropout(tri_out)
        
        return node_out, edge_out, tri_out


class TemporalSimplicialEncoder(nn.Module):
    """
    Temporal encoder that applies SCCN layers followed by aggregation.
    
    Processes a sequence of simplicial complexes (windows) and produces
    a fixed-size encoding.
    """
    
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        num_sccn_layers: int = 2,
        dropout: float = 0.3,
        pooling: str = "mean",
    ):
        """
        Initialize temporal encoder.
        
        Args:
            input_dim: Input feature dimension (usually 1 for temporal values)
            hidden_dim: Hidden feature dimension
            num_sccn_layers: Number of SCCN layers
            dropout: Dropout probability
            pooling: Pooling strategy ('mean', 'max', 'sum')
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.pooling = pooling
        
        # Initial embedding
        self.node_embed = nn.Linear(input_dim, hidden_dim)
        self.edge_embed = nn.Linear(4, hidden_dim)  # 4 edge features (sum, mean, max, min)
        self.tri_embed = nn.Linear(4, hidden_dim)  # 4 triangle features
        
        # SCCN layers
        self.sccn_layers = nn.ModuleList([
            SCCNLayer(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                max_rank=2,
                dropout=dropout,
            )
            for _ in range(num_sccn_layers)
        ])
        
        # Activation and normalization
        self.activation = nn.ReLU()
        self.layer_norm_node = nn.LayerNorm(hidden_dim)
        self.layer_norm_edge = nn.LayerNorm(hidden_dim)
        self.layer_norm_tri = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        triangle_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through temporal encoder.
        
        Args:
            node_features: Shape (batch_size, seq_len, max_nodes)
            edge_features: Shape (batch_size, seq_len, max_edges, 4)
            triangle_features: Shape (batch_size, seq_len, max_triangles, 4)
        
        Returns:
            Encoded features: Shape (batch_size, hidden_dim)
        """
        batch_size, seq_len, max_nodes = node_features.shape
        
        # Embed features
        node_feat = self.node_embed(node_features.unsqueeze(-1))  # (batch, seq, nodes, hidden)
        edge_feat = self.edge_embed(edge_features)  # (batch, seq, edges, hidden)
        tri_feat = self.tri_embed(triangle_features)  # (batch, seq, triangles, hidden)
        
        # Process through SCCN layers
        for sccn_layer in self.sccn_layers:
            # Apply SCCN to each window in sequence
            node_out_list = []
            edge_out_list = []
            tri_out_list = []
            
            for t in range(seq_len):
                n_out, e_out, t_out = sccn_layer(
                    node_feat[:, t],
                    edge_feat[:, t] if edge_feat.shape[1] > 0 else None,
                    tri_feat[:, t] if tri_feat.shape[1] > 0 else None,
                )
                node_out_list.append(n_out)
                if e_out is not None:
                    edge_out_list.append(e_out)
                if t_out is not None:
                    tri_out_list.append(t_out)
            
            node_feat = torch.stack(node_out_list, dim=1)  # (batch, seq, nodes, hidden)
            
            if edge_out_list:
                edge_feat = torch.stack(edge_out_list, dim=1)
            if tri_out_list:
                tri_feat = torch.stack(tri_out_list, dim=1)
            
            # Normalize
            node_feat = self.layer_norm_node(node_feat)
            if edge_feat.shape[1] > 0:
                edge_feat = self.layer_norm_edge(edge_feat)
            if tri_feat.shape[1] > 0:
                tri_feat = self.layer_norm_tri(tri_feat)
            
            # Activation
            node_feat = self.activation(node_feat)
            edge_feat = self.activation(edge_feat)
            tri_feat = self.activation(tri_feat)
        
        # Temporal aggregation: pool over time and space
        # node_feat shape: (batch, seq, nodes, hidden)
        if self.pooling == "mean":
            # Average over time and nodes
            encoded = node_feat.mean(dim=[1, 2])  # (batch, hidden)
        elif self.pooling == "max":
            encoded = node_feat.max(dim=1)[0].max(dim=1)[0]  # (batch, hidden)
        elif self.pooling == "sum":
            encoded = node_feat.sum(dim=[1, 2])  # (batch, hidden)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        
        return encoded


class SimplicialLSTMClassifier(nn.Module):
    """
    Combine SCCN encoder with LSTM and classification head.
    
    Architecture:
    1. Embed window-level simplicial structures with SCCN
    2. Process temporal sequence with LSTM
    3. Classify with MLP head
    """
    
    def __init__(
        self,
        input_dim: int = 1,
        sccn_hidden: int = 64,
        num_sccn_layers: int = 2,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        """
        Initialize classifier.
        
        Args:
            input_dim: Input feature dimension
            sccn_hidden: SCCN hidden dimension
            num_sccn_layers: Number of SCCN layers
            lstm_hidden: LSTM hidden dimension
            lstm_layers: Number of LSTM layers
            num_classes: Number of output classes
            dropout: Dropout probability
        """
        super().__init__()
        
        # Temporal simplicial encoder with SCCN
        self.encoder = TemporalSimplicialEncoder(
            input_dim=input_dim,
            hidden_dim=sccn_hidden,
            num_sccn_layers=num_sccn_layers,
            dropout=dropout,
            pooling="mean",
        )
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=sccn_hidden,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden // 2, num_classes),
        )
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        triangle_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through classifier.
        
        Args:
            node_features: Shape (batch_size, seq_len, max_nodes)
            edge_features: Shape (batch_size, seq_len, max_edges, 4)
            triangle_features: Shape (batch_size, seq_len, max_triangles, 4)
        
        Returns:
            Class logits: Shape (batch_size, num_classes)
        """
        batch_size, seq_len, _ = node_features.shape
        
        # Encode each window with SCCN
        encoded_windows = []
        for t in range(seq_len):
            enc = self.encoder(
                node_features[:, t:t+1],
                edge_features[:, t:t+1],
                triangle_features[:, t:t+1],
            )
            encoded_windows.append(enc)
        
        # Stack into sequence
        encoded_seq = torch.stack(encoded_windows, dim=1)  # (batch, seq_len, sccn_hidden)
        
        # Process through LSTM
        lstm_out, (h_n, c_n) = self.lstm(encoded_seq)  # h_n: (lstm_layers, batch, lstm_hidden)
        
        # Take the last hidden state
        last_hidden = h_n[-1]  # (batch, lstm_hidden)
        
        # Classify
        logits = self.classifier(last_hidden)  # (batch, num_classes)
        
        return logits


class SimplePooingClassifier(nn.Module):
    """
    Simpler classifier without LSTM, just pooling and MLP.
    
    For comparison - treats each window independently and averages.
    """
    
    def __init__(
        self,
        input_dim: int = 1,
        sccn_hidden: int = 64,
        num_sccn_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        """Initialize simple classifier."""
        super().__init__()
        
        self.encoder = TemporalSimplicialEncoder(
            input_dim=input_dim,
            hidden_dim=sccn_hidden,
            num_sccn_layers=num_sccn_layers,
            dropout=dropout,
            pooling="mean",
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(sccn_hidden, sccn_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(sccn_hidden // 2, num_classes),
        )
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        triangle_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            node_features: Shape (batch_size, seq_len, max_nodes)
            edge_features: Shape (batch_size, seq_len, max_edges, 4)
            triangle_features: Shape (batch_size, seq_len, max_triangles, 4)
        
        Returns:
            Class logits: Shape (batch_size, num_classes)
        """
        # Encode entire sequence
        encoded = self.encoder(node_features, edge_features, triangle_features)
        
        # Classify
        logits = self.classifier(encoded)
        
        return logits


class AttentionPoolingClassifier(nn.Module):
    """
    Classifier with attention-based pooling over temporal windows.
    
    Uses attention to weight windows and SCCN for encoding.
    """
    
    def __init__(
        self,
        input_dim: int = 1,
        sccn_hidden: int = 64,
        num_sccn_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        """Initialize attention-based classifier."""
        super().__init__()
        
        self.encoder = TemporalSimplicialEncoder(
            input_dim=input_dim,
            hidden_dim=sccn_hidden,
            num_sccn_layers=num_sccn_layers,
            dropout=dropout,
            pooling="mean",
        )
        
        # Attention weights for temporal windows
        self.attention = nn.Sequential(
            nn.Linear(sccn_hidden, sccn_hidden // 2),
            nn.ReLU(),
            nn.Linear(sccn_hidden // 2, 1),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(sccn_hidden, sccn_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(sccn_hidden // 2, num_classes),
        )
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        triangle_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with attention pooling.
        
        Args:
            node_features: Shape (batch_size, seq_len, max_nodes)
            edge_features: Shape (batch_size, seq_len, max_edges, 4)
            triangle_features: Shape (batch_size, seq_len, max_triangles, 4)
        
        Returns:
            Class logits: Shape (batch_size, num_classes)
        """
        batch_size, seq_len, _ = node_features.shape
        
        # Encode each window
        encoded_windows = []
        for t in range(seq_len):
            enc = self.encoder(
                node_features[:, t:t+1],
                edge_features[:, t:t+1],
                triangle_features[:, t:t+1],
            )
            encoded_windows.append(enc)
        
        # Stack windows
        encoded_seq = torch.stack(encoded_windows, dim=1)  # (batch, seq_len, sccn_hidden)
        
        # Compute attention weights
        attn_scores = self.attention(encoded_seq)  # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (batch, seq_len, 1)
        
        # Apply attention and pool
        weighted_sum = (encoded_seq * attn_weights).sum(dim=1)  # (batch, sccn_hidden)
        
        # Classify
        logits = self.classifier(weighted_sum)
        
        return logits


def create_model(
    model_name: str,
    input_dim: int = 1,
    sccn_hidden: int = 64,
    num_sccn_layers: int = 2,
    lstm_hidden: int = 128,
    lstm_layers: int = 1,
    num_classes: int = 2,
    dropout: float = 0.3,
) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        model_name: One of 'SCCN_LSTM', 'SCCN_Pool', 'SCCN_Attention'
        Other args as model-specific
    
    Returns:
        PyTorch model
    """
    if model_name == "SCCN_LSTM":
        return SimplicialLSTMClassifier(
            input_dim=input_dim,
            sccn_hidden=sccn_hidden,
            num_sccn_layers=num_sccn_layers,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            num_classes=num_classes,
            dropout=dropout,
        )
    elif model_name == "SCCN_Pool":
        return SimplePooingClassifier(
            input_dim=input_dim,
            sccn_hidden=sccn_hidden,
            num_sccn_layers=num_sccn_layers,
            num_classes=num_classes,
            dropout=dropout,
        )
    elif model_name == "SCCN_Attention":
        return AttentionPoolingClassifier(
            input_dim=input_dim,
            sccn_hidden=sccn_hidden,
            num_sccn_layers=num_sccn_layers,
            num_classes=num_classes,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
