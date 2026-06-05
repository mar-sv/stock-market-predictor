"""LSTM encoder forecaster.

Encodes the past sequence (recent returns + past covariates) with a stacked
LSTM and maps the final hidden state to the full multi-horizon, multi-quantile
forecast. Future calendar covariates are ignored here (the TFT exploits those).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from smp.models.base import BaseForecaster, QuantileHead


class LSTMForecaster(BaseForecaster):
    def __init__(
        self,
        n_past_features: int,
        n_future_features: int,
        horizon: int,
        quantiles: list[float],
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(n_past_features, n_future_features, horizon, quantiles, lr, weight_decay)
        self.lstm = nn.LSTM(
            input_size=n_past_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = QuantileHead(hidden_size, horizon, self.n_quantiles)

    def forward(self, past: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(past)          # (batch, lookback, hidden)
        last = self.dropout(out[:, -1])   # final timestep summary
        return self.head(last)
