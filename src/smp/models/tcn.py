"""Temporal Convolutional Network (dilated causal 1D convolutions).

Causality is enforced by left-padding each conv and chopping the extra right-hand
outputs (``Chomp1d``), so position ``t`` never sees ``t+1``. Stacking blocks with
exponentially increasing dilation grows the receptive field geometrically.

The TCN uses only the past sequence (recent returns + past covariates); future
calendar covariates are ignored here (the TFT is what exploits those).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from smp.models.base import BaseForecaster, QuantileHead


class Chomp1d(nn.Module):
    """Remove the ``chomp`` right-most timesteps added by causal padding."""

    def __init__(self, chomp: int) -> None:
        super().__init__()
        self.chomp = chomp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.chomp].contiguous() if self.chomp > 0 else x


class TemporalBlock(nn.Module):
    """Two dilated causal convs with a residual connection."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            weight_norm(nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)),
            Chomp1d(pad),
            nn.ReLU(),
            nn.Dropout(dropout),
            weight_norm(nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)),
            Chomp1d(pad),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNForecaster(BaseForecaster):
    """Stack of temporal blocks + quantile head on the final timestep."""

    def __init__(
        self,
        n_past_features: int,
        n_future_features: int,
        horizon: int,
        quantiles: list[float],
        hidden_size: int = 64,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(n_past_features, n_future_features, horizon, quantiles, lr, weight_decay)
        layers = []
        in_ch = n_past_features
        for i in range(num_layers):
            layers.append(
                TemporalBlock(in_ch, hidden_size, kernel_size, dilation=2**i, dropout=dropout)
            )
            in_ch = hidden_size
        self.network = nn.Sequential(*layers)
        self.head = QuantileHead(hidden_size, horizon, self.n_quantiles)

    def forward(self, past: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        # past: (batch, lookback, features) -> conv wants (batch, channels, time)
        x = past.transpose(1, 2)
        y = self.network(x)
        last = y[:, :, -1]  # most recent timestep summarises the receptive field
        return self.head(last)
