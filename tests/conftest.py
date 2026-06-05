"""Shared test fixtures: a deterministic synthetic OHLCV series and a config."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    """A deterministic random-walk OHLCV frame on business days."""
    rng = np.random.default_rng(0)
    n = 600
    dates = pd.bdate_range("2015-01-01", periods=n)
    steps = rng.normal(0, 0.01, size=n)
    close = 100.0 * np.exp(np.cumsum(steps))
    high = close * (1 + rng.uniform(0, 0.01, n))
    low = close * (1 - rng.uniform(0, 0.01, n))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=pd.DatetimeIndex(dates, name="date"),
    )


@pytest.fixture
def cfg() -> dict:
    """A small config suitable for fast tests."""
    return {
        "data": {"tickers": ["TEST"], "start": "2015-01-01", "end": None, "interval": "1d"},
        "forecast": {"lookback": 20, "horizon": 5, "quantiles": [0.1, 0.5, 0.9]},
        "features": {
            "use_rsi": True,
            "use_macd": True,
            "use_volatility": True,
            "use_ma_ratios": True,
            "volatility_window": 20,
            "rsi_window": 14,
            "use_calendar": True,
        },
        "split": {"train": 0.7, "val": 0.15, "test": 0.15},
        "train": {"max_epochs": 1, "early_stopping_patience": 2, "accelerator": "cpu",
                  "num_workers": 0, "seed": 42},
        "optuna": {"n_trials": 1, "timeout": None, "architectures": ["tcn", "lstm", "tft"],
                   "direction": "minimize"},
        "mlflow": {"tracking_uri": "./mlruns", "experiment_prefix": "smp"},
    }
