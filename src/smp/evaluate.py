"""Shared evaluation helper used by both tuning and backtesting.

Runs a model over a dataloader, inverse-transforms predictions and targets back
to raw log-return space, and computes the selection/eval metrics.
"""

from __future__ import annotations

import numpy as np

from smp.metrics import compute_metrics


def evaluate_model(model, dataloader, dm, quantiles: list[float]):
    """Return ``(metrics, preds_raw, targets_raw)`` in raw log-return space."""
    preds_std, targets_std = model.predict_dataset(dataloader)
    preds = np.asarray(dm.inverse_target(preds_std))
    targets = np.asarray(dm.inverse_target(targets_std))
    if preds.shape[0] == 0:
        return {"quantile_loss": float("inf")}, preds, targets
    metrics = compute_metrics(preds, targets, quantiles)
    return metrics, preds, targets
