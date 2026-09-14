"""Compact model definitions used by every revised experiment."""

from __future__ import annotations

import torch
from torch import nn


class AttentionPool(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(size, size // 2), nn.Tanh(), nn.Linear(size // 2, 1))

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(sequence), dim=1)
        return torch.sum(weights * sequence, dim=1)


class TemporalEncoder(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, depth: int, dropout: float):
        super().__init__()
        self.output_size = hidden_size * 2
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=depth,
            batch_first=True,
            dropout=dropout if depth > 1 else 0.0,
            bidirectional=True,
        )
        self.attention = AttentionPool(self.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.lstm(x)
        return self.attention(sequence)


class TimSFSARiskModel(nn.Module):
    """LSTM-attention Cox model with prediction-origin raw-feature fusion."""

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 256,
        depth: int = 2,
        dropout: float = 0.3,
        connector_size: int = 512,
        connector_dropout: float = 0.4,
        use_lstm: bool = True,
        use_connector: bool = True,
        linear_head: bool = False,
    ):
        super().__init__()
        self.n_features = n_features
        self.use_lstm = use_lstm
        self.use_connector = use_connector and use_lstm
        self.linear_head = linear_head
        if use_lstm:
            self.encoder = TemporalEncoder(n_features, hidden_size, depth, dropout)
            context_size = self.encoder.output_size
            if self.use_connector:
                self.connector = nn.Sequential(
                    nn.Linear(context_size, connector_size),
                    nn.LayerNorm(connector_size),
                    nn.GELU(),
                    nn.Dropout(connector_dropout),
                )
                context_size = connector_size
            fused_size = context_size + n_features
        else:
            self.encoder = None
            fused_size = n_features
        if linear_head:
            self.head = nn.Linear(fused_size, 1)
        else:
            hidden = max(16, min(128, fused_size // 2))
            self.head = nn.Sequential(
                nn.Linear(fused_size, hidden), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden, max(8, hidden // 4)), nn.GELU(),
                nn.Linear(max(8, hidden // 4), 1),
            )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        raw_origin = x[:, -1, :]
        if not self.use_lstm:
            return raw_origin
        context = self.encoder(x)
        if self.use_connector:
            context = self.connector(context)
        return torch.cat([context, raw_origin], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x)).squeeze(-1)


class MultiHorizonLSTMClassifier(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_horizons: int,
        hidden_size: int = 256,
        depth: int = 2,
        dropout: float = 0.3,
        connector_size: int = 512,
    ):
        super().__init__()
        self.encoder = TemporalEncoder(n_features, hidden_size, depth, dropout)
        self.connector = nn.Sequential(
            nn.Linear(self.encoder.output_size, connector_size),
            nn.LayerNorm(connector_size), nn.GELU(), nn.Dropout(dropout),
        )
        self.head = nn.Linear(connector_size + n_features, n_horizons)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context = self.connector(self.encoder(x))
        return self.head(torch.cat([context, x[:, -1, :]], dim=1))
