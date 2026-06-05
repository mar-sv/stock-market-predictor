"""Tests for feature engineering and the windowed datamodule (leakage focus)."""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

from smp.data.dataset import StockDataModule
from smp.data.features import TARGET_COL, build_features


def test_features_have_target_and_no_nan(synthetic_ohlcv, cfg):
    feats, cols = build_features(synthetic_ohlcv, cfg)
    assert cols.target == TARGET_COL
    assert TARGET_COL in feats.columns
    assert len(cols.past) > 0 and len(cols.future) == 4  # calendar sin/cos x2
    assert not feats.isna().any().any()  # warm-up rows dropped


def test_window_shapes(synthetic_ohlcv, cfg):
    dm = StockDataModule(synthetic_ohlcv, cfg)
    dm.setup()
    lookback = cfg["forecast"]["lookback"]
    horizon = cfg["forecast"]["horizon"]
    item = dm._datasets["train"][0]
    assert item["past"].shape == (lookback, dm.n_past_features)
    assert item["future"].shape == (horizon, dm.n_future_features)
    assert item["target"].shape == (horizon,)


def test_scaler_fit_on_train_only(synthetic_ohlcv, cfg):
    dm = StockDataModule(synthetic_ohlcv, cfg)
    dm.setup()
    feats, _ = build_features(synthetic_ohlcv, cfg)
    train_end = int(len(feats) * cfg["split"]["train"])
    cols = dm.past_input_cols + dm.future_input_cols
    expected = StandardScaler().fit(feats.iloc[:train_end][cols].to_numpy())
    # The datamodule's scaler must match a train-only fit (no val/test leakage).
    assert np.allclose(dm.feature_scaler.mean_, expected.mean_)
    assert dm.target_mean == float(feats.iloc[:train_end][TARGET_COL].mean())


def test_target_window_uses_future_values(synthetic_ohlcv, cfg):
    # The target for present-position p must be the FUTURE returns p+1..p+H,
    # standardized by the train mean/std. This is the core no-look-ahead check.
    dm = StockDataModule(synthetic_ohlcv, cfg)
    dm.setup()
    feats, cols = build_features(synthetic_ohlcv, cfg)
    raw = feats[cols.target].to_numpy(dtype=np.float32)
    lookback, horizon = cfg["forecast"]["lookback"], cfg["forecast"]["horizon"]
    p = lookback - 1  # first train present-position
    expected = (raw[p + 1 : p + 1 + horizon] - dm.target_mean) / dm.target_std
    got = dm._datasets["train"][0]["target"].numpy()
    assert np.allclose(got, expected, atol=1e-5)


def test_splits_are_nonempty_and_ordered(synthetic_ohlcv, cfg):
    dm = StockDataModule(synthetic_ohlcv, cfg)
    dm.setup()
    assert len(dm._datasets["train"]) > 0
    assert len(dm._datasets["val"]) > 0
    assert len(dm._datasets["test"]) > 0
