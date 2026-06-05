"""Forward-pass shape tests for all three architectures."""

from __future__ import annotations

import pytest
import torch

from smp.models.registry import build_model

ARCHS = ["tcn", "lstm", "tft"]


@pytest.mark.parametrize("arch", ARCHS)
def test_forward_output_shape(arch):
    batch, lookback, horizon = 8, 20, 5
    n_past, n_future = 11, 4
    quantiles = [0.1, 0.5, 0.9]

    hparams = {"hidden_size": 16}
    if arch == "tft":
        hparams["n_heads"] = 2
    model = build_model(arch, n_past, n_future, horizon, quantiles, hparams)

    past = torch.randn(batch, lookback, n_past)
    future = torch.randn(batch, horizon, n_future)
    out = model(past, future)
    assert out.shape == (batch, horizon, len(quantiles))


def test_tft_handles_no_future_features():
    # Decoder must still work when calendar covariates are disabled.
    model = build_model("tft", 11, 0, 5, [0.1, 0.5, 0.9], {"hidden_size": 16, "n_heads": 2})
    past = torch.randn(4, 20, 11)
    future = torch.randn(4, 5, 0)
    out = model(past, future)
    assert out.shape == (4, 5, 3)
