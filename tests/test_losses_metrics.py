"""Tests for the quantile loss and selection metrics."""

from __future__ import annotations

import numpy as np
import torch

from smp.losses import quantile_loss
from smp.metrics import compute_metrics


def test_median_quantile_loss_is_half_mae():
    # For q=0.5 the pinball loss equals 0.5 * |error|.
    preds = torch.zeros(4, 5, 1)
    target = torch.full((4, 5), 2.0)
    loss = quantile_loss(preds, target, [0.5])
    assert torch.isclose(loss, torch.tensor(1.0))  # 0.5 * |2 - 0|


def test_quantile_loss_is_asymmetric():
    # A high quantile (0.9) should penalise under-prediction more than over.
    target = torch.zeros(1, 1)
    under = quantile_loss(torch.tensor([[[-1.0]]]), target, [0.9])  # pred below target
    over = quantile_loss(torch.tensor([[[1.0]]]), target, [0.9])    # pred above target
    assert under > over


def test_directional_accuracy_perfect_and_zero():
    quantiles = [0.1, 0.5, 0.9]
    target = np.array([[0.01, -0.02], [-0.03, 0.04]])
    # q50 (index 1) shares the sign of target everywhere -> perfect.
    preds = np.zeros((2, 2, 3))
    preds[..., 1] = target
    m = compute_metrics(preds, target, quantiles)
    assert m["directional_accuracy"] == 1.0
    # Flip the sign -> zero directional accuracy.
    preds[..., 1] = -target
    m2 = compute_metrics(preds, target, quantiles)
    assert m2["directional_accuracy"] == 0.0
