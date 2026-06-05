"""Quantile (pinball) loss for multi-horizon, multi-quantile forecasts.

The pinball loss for a single quantile ``q`` and error ``e = y - y_hat`` is::

    max(q * e, (q - 1) * e)

which penalises under- and over-prediction asymmetrically so the model learns
the requested conditional quantile. We average equally over the batch, the
forecast horizons, and the quantiles.
"""

from __future__ import annotations

import torch


def quantile_loss(
    preds: torch.Tensor,
    target: torch.Tensor,
    quantiles: list[float],
) -> torch.Tensor:
    """Mean pinball loss.

    Args:
        preds: ``(batch, horizon, n_quantiles)`` predicted quantiles.
        target: ``(batch, horizon)`` actual values.
        quantiles: the quantile levels, aligned with the last dim of ``preds``.

    Returns:
        A scalar tensor (mean over batch, horizon, quantiles).
    """
    q = torch.tensor(quantiles, dtype=preds.dtype, device=preds.device)
    # error: (batch, horizon, n_quantiles)
    error = target.unsqueeze(-1) - preds
    losses = torch.maximum(q * error, (q - 1.0) * error)
    return losses.mean()
