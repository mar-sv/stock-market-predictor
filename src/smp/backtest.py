"""Out-of-sample backtest of the selected model on the test split.

We refit the chosen architecture (best hyperparameters) on the training data
(with early stopping on validation), then produce rolling forecasts over the
held-out test period and report:

* per-horizon point/probabilistic metrics (RMSE, MAE, directional accuracy),
* a simple **sign-based PnL** on the 1-day-ahead signal as an intuitive sanity
  check (position = sign(predicted next-day return); not a trading system).

A forecast-vs-actual plot is saved and returned for logging as an MLflow artifact.

Note: this is a fixed-origin test-period backtest with rolling windows. True
walk-forward retraining is a documented future extension.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from smp.data.dataset import StockDataModule
from smp.evaluate import evaluate_model
from smp.metrics import _q50_index
from smp.models.registry import build_model, params_to_kwargs
from smp.tuning.optuna_search import make_trainer


def fit_final_model(arch: str, params: dict, df: pd.DataFrame, cfg: dict):
    """Refit ``arch`` with ``params`` on train (early-stopped on val)."""
    quantiles = cfg["forecast"]["quantiles"]
    horizon = cfg["forecast"]["horizon"]

    dm = StockDataModule(df, cfg)
    dm.setup()
    model_kwargs, batch_size = params_to_kwargs(arch, params)
    dm.batch_size = batch_size

    model = build_model(
        arch,
        n_past_features=dm.n_past_features,
        n_future_features=dm.n_future_features,
        horizon=horizon,
        quantiles=quantiles,
        hparams=model_kwargs,
    )
    make_trainer(cfg).fit(model, datamodule=dm)
    return model, dm


def sign_based_pnl(preds: np.ndarray, targets: np.ndarray, quantiles: list[float]) -> dict:
    """Naive long/short PnL from the 1-day-ahead directional signal."""
    qi = _q50_index(quantiles)
    signal = np.sign(preds[:, 0, qi])      # position from next-day forecast
    realized = targets[:, 0]               # next-day log return
    strat = signal * realized
    if strat.size == 0 or strat.std() == 0:
        sharpe = 0.0
    else:
        sharpe = float(strat.mean() / strat.std() * np.sqrt(252))
    return {
        "pnl_total_logret": float(strat.sum()),
        "pnl_sharpe_annualized": sharpe,
        "pnl_hit_rate": float(np.mean(signal == np.sign(realized))),
    }


def plot_forecast(
    preds: np.ndarray, targets: np.ndarray, quantiles: list[float], out_path: Path, title: str
) -> Path:
    """Plot 1-day-ahead q50 forecast vs actual with the quantile band."""
    qi = _q50_index(quantiles)
    lo, hi = 0, len(quantiles) - 1
    x = np.arange(targets.shape[0])
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(x, targets[:, 0], label="actual", color="black", lw=1)
    ax.plot(x, preds[:, 0, qi], label="forecast q50", color="tab:blue", lw=1)
    if len(quantiles) >= 2:
        ax.fill_between(
            x, preds[:, 0, lo], preds[:, 0, hi], color="tab:blue", alpha=0.2,
            label=f"q{int(quantiles[lo]*100)}-q{int(quantiles[hi]*100)}",
        )
    ax.set_title(title)
    ax.set_xlabel("test-period step")
    ax.set_ylabel("1-day log return")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def backtest(arch: str, params: dict, df: pd.DataFrame, cfg: dict, plot_dir: Path):
    """Refit + evaluate on the test split; return ``(metrics, plot_path)``."""
    quantiles = cfg["forecast"]["quantiles"]
    model, dm = fit_final_model(arch, params, df, cfg)
    metrics, preds, targets = evaluate_model(model, dm.test_dataloader(), dm, quantiles)
    metrics.update(sign_based_pnl(preds, targets, quantiles))
    plot_path = plot_forecast(
        preds, targets, quantiles, plot_dir / f"{arch}_forecast.png",
        title=f"{arch.upper()} test-period 1-day forecast",
    )
    return metrics, plot_path
