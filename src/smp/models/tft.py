"""Temporal Fusion Transformer (from scratch, compact but faithful).

Implements the core TFT building blocks:

* **GLU / GateAddNorm** - gated skip connections used throughout.
* **GRN** (Gated Residual Network) - the TFT's universal nonlinear block.
* **VariableSelectionNetwork** - learns per-variable importance weights.
* **LSTM encoder/decoder** - local sequence processing of past & future.
* **InterpretableMultiHeadAttention** - shared-value heads for interpretability.
* **Quantile heads** - one set of quantiles per future step.

We have no static covariates (single series per model), so static enrichment is
applied without external context. Future-known inputs are the calendar
covariates; if those are disabled the decoder is fed a zero placeholder.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from smp.models.base import BaseForecaster


class GLU(nn.Module):
    """Gated Linear Unit: ``a * sigmoid(b)``."""

    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.fc = nn.Linear(input_size, output_size * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.fc(x).chunk(2, dim=-1)
        return a * torch.sigmoid(b)


class GateAddNorm(nn.Module):
    """Dropout -> GLU -> add residual -> LayerNorm."""

    def __init__(self, input_size: int, output_size: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.glu = GLU(input_size, output_size)
        self.norm = nn.LayerNorm(output_size)

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return self.norm(self.glu(self.dropout(x)) + residual)


class GRN(nn.Module):
    """Gated Residual Network (optionally conditioned on a context vector)."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float,
        context_size: int | None = None,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.context = (
            nn.Linear(context_size, hidden_size, bias=False) if context_size else None
        )
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.gate = GateAddNorm(hidden_size, output_size, dropout)
        self.skip = (
            nn.Linear(input_size, output_size) if input_size != output_size else None
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        residual = x if self.skip is None else self.skip(x)
        h = self.fc1(x)
        if self.context is not None and context is not None:
            h = h + self.context(context)
        h = self.fc2(self.elu(h))
        return self.gate(h, residual)


class VariableSelectionNetwork(nn.Module):
    """Per-variable GRN transforms combined by learned softmax weights."""

    def __init__(self, num_vars: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.num_vars = num_vars
        self.weight_grn = GRN(num_vars, hidden_size, num_vars, dropout)
        self.var_grns = nn.ModuleList(
            [GRN(1, hidden_size, hidden_size, dropout) for _ in range(num_vars)]
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, num_vars)
        weights = self.softmax(self.weight_grn(x)).unsqueeze(-1)  # (b,t,vars,1)
        transformed = torch.stack(
            [self.var_grns[i](x[..., i : i + 1]) for i in range(self.num_vars)],
            dim=-2,
        )  # (b, t, vars, hidden)
        return (weights * transformed).sum(dim=-2)  # (b, t, hidden)


class InterpretableMultiHeadAttention(nn.Module):
    """Multi-head attention with a value projection shared across heads."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_layers = nn.ModuleList(
            [nn.Linear(d_model, self.d_head) for _ in range(n_heads)]
        )
        self.k_layers = nn.ModuleList(
            [nn.Linear(d_model, self.d_head) for _ in range(n_heads)]
        )
        self.v_layer = nn.Linear(d_model, self.d_head)  # shared across heads
        self.out = nn.Linear(self.d_head, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        v = self.v_layer(x)  # (b, t, d_head)
        scale = self.d_head**0.5
        heads = []
        for i in range(self.n_heads):
            q = self.q_layers[i](x)
            k = self.k_layers[i](x)
            scores = q @ k.transpose(-2, -1) / scale  # (b, t, t)
            scores = scores.masked_fill(mask, float("-inf"))
            attn = self.dropout(torch.softmax(scores, dim=-1))
            heads.append(attn @ v)  # (b, t, d_head)
        head = torch.stack(heads, dim=0).mean(dim=0)  # average heads
        return self.out(head)


class TFTForecaster(BaseForecaster):
    def __init__(
        self,
        n_past_features: int,
        n_future_features: int,
        horizon: int,
        quantiles: list[float],
        hidden_size: int = 32,
        lstm_layers: int = 1,
        n_heads: int = 4,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(n_past_features, n_future_features, horizon, quantiles, lr, weight_decay)
        # Decoder needs at least one future variable; use a zero placeholder if
        # no calendar covariates are configured.
        self.future_vars = max(n_future_features, 1)
        self._has_future = n_future_features > 0
        if hidden_size % n_heads != 0:
            hidden_size = (hidden_size // n_heads) * n_heads or n_heads

        self.past_vsn = VariableSelectionNetwork(n_past_features, hidden_size, dropout)
        self.future_vsn = VariableSelectionNetwork(self.future_vars, hidden_size, dropout)

        self.encoder = nn.LSTM(hidden_size, hidden_size, lstm_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, lstm_layers, batch_first=True)
        self.lstm_gate = GateAddNorm(hidden_size, hidden_size, dropout)

        self.enrichment = GRN(hidden_size, hidden_size, hidden_size, dropout)
        self.attention = InterpretableMultiHeadAttention(hidden_size, n_heads, dropout)
        self.attn_gate = GateAddNorm(hidden_size, hidden_size, dropout)
        self.ff = GRN(hidden_size, hidden_size, hidden_size, dropout)
        self.ff_gate = GateAddNorm(hidden_size, hidden_size, dropout)

        self.output = nn.Linear(hidden_size, self.n_quantiles)

    def forward(self, past: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        b = past.shape[0]
        if not self._has_future:
            future = torch.zeros(b, self.horizon, 1, device=past.device, dtype=past.dtype)

        past_sel = self.past_vsn(past)        # (b, L, hidden)
        future_sel = self.future_vsn(future)  # (b, H, hidden)

        enc_out, (h, c) = self.encoder(past_sel)
        dec_out, _ = self.decoder(future_sel, (h, c))

        temporal = torch.cat([enc_out, dec_out], dim=1)      # (b, L+H, hidden)
        vsn_seq = torch.cat([past_sel, future_sel], dim=1)   # gated skip residual
        temporal = self.lstm_gate(temporal, vsn_seq)

        enriched = self.enrichment(temporal)

        # Causal mask: position i may attend only to j <= i.
        s = enriched.shape[1]
        mask = torch.triu(
            torch.ones(s, s, dtype=torch.bool, device=enriched.device), diagonal=1
        )
        attn = self.attention(enriched, mask)
        attn = self.attn_gate(attn, enriched)

        out = self.ff(attn)
        out = self.ff_gate(out, temporal)

        dec = out[:, -self.horizon :, :]      # decoder steps only
        return self.output(dec)               # (b, horizon, n_quantiles)
