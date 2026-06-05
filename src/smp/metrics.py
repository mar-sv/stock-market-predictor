"""Selection / evaluation metrics.

These are computed on **raw log-return space** (predictions inverse-transformed
back from the standardized target). The point forecast is the median quantile
(q50); direction is its sign.

Optuna selection uses ``val_quantile_loss``; the rest are reported for insight.
"""

from __future__ import annotations

import numpy as np

from smp.losses import quantile_loss
import torch


def _q50_index(quantiles: list[float]) -> int:
    """Index of the median quantile (closest to 0.5)."""
    return int(np.argmin([abs(q - 0.5) for q in quantiles]))


def compute_metrics(
    preds: np.ndarray,
    target: np.ndarray,
    quantiles: list[float],
) -> dict[str, float]:
    """Compute point + directional + probabilistic metrics.

    Args:
        preds: ``(n, horizon, n_quantiles)`` predictions in raw log-return space.
        target: ``(n, horizon)`` actuals in raw log-return space.
        quantiles: quantile levels aligned with ``preds`` last dim.

    Returns:
        Dict with overall metrics plus per-horizon directional accuracy.
    """
    qi = _q50_index(quantiles)
    point = preds[..., qi]  # (n, horizon)

    err = point - target
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))

    # Directional accuracy: did we get the sign of the move right?
    correct = np.sign(point) == np.sign(target)
    dir_acc = float(np.mean(correct))

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "directional_accuracy": dir_acc,
        "quantile_loss": float(
            quantile_loss(
                torch.tensor(preds), torch.tensor(target), quantiles
            ).item()
        ),
    }

    # Per-horizon directional accuracy (h=1..H) for insight into lead-time decay.
    for h in range(target.shape[1]):
        metrics[f"dir_acc_h{h + 1}"] = float(np.mean(correct[:, h]))

    return metrics
