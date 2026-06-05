# stock-market-predictor

Exploring three deep-learning forecasting architectures — **Temporal Fusion
Transformer (TFT)**, **LSTM**, and **Temporal CNN (TCN)** — built from scratch in
PyTorch + Lightning, and comparing how well they predict daily stock returns.

For **each ticker**, an [Optuna](https://optuna.org/) study per architecture tunes
hyperparameters; the best architecture is selected by a held-out, time-aware
validation metric. Every trial is tracked in [MLflow](https://mlflow.org/).

## What it predicts

- **Target:** per-day **log returns** (stationary; avoids the raw-price trap).
- **Horizon:** multi-horizon — the next **5 trading days** in one shot.
- **Loss:** **quantile (pinball)** loss at quantiles `[0.1, 0.5, 0.9]`, identical
  across all three models for a fair comparison.
- **Point forecast** = the median (q50); **direction** = `sign(q50)`.

> The network *trains* on the quantile loss; model *selection* ranks Optuna trials
> by the **validation quantile loss in raw return space** (with directional
> accuracy and RMSE reported alongside).

## Setup

This repo uses [`uv`](https://docs.astral.sh/uv/). No system Python required.

```powershell
uv venv --python 3.12
uv pip install -e ".[dev]"
```

## Run

```powershell
# Full run for all tickers in config/config.yaml (^GSPC, AAPL, MSFT)
uv run smp-train

# A single ticker
uv run smp-train --ticker AAPL

# A fast end-to-end smoke test (tiny search, 2 epochs)
uv run smp-train --config config/smoke.yaml

# Inspect experiments
uv run mlflow ui        # then open the served URL; reads ./mlruns

# Tests
uv run pytest
```

All knobs (tickers, date range, lookback, horizon, quantiles, Optuna trials,
splits) live in `config/config.yaml`.

## Layout

```
config/            config.yaml (main) + smoke.yaml (fast test)
src/smp/
  data/            loader (yfinance + cache), features, windowed datamodule
  models/          base + tcn + lstm + tft + registry (factory & search spaces)
  losses.py        quantile/pinball loss
  metrics.py       RMSE, MAE, directional accuracy
  evaluate.py      run model -> raw-space metrics
  tuning/          Optuna studies + selection
  tracking/        MLflow helpers (tracking only)
  backtest.py      out-of-sample test-period backtest + sign-based PnL + plot
  train.py         CLI entrypoint
tests/             loss/metrics, data (leakage), model forward-shape tests
```

## Leakage controls

Chronological splits, a purge gap of `horizon` between splits, and scalers /
target statistics fit on the **training rows only**.

## Possible extensions

Model Registry, a multi-task direction head, true walk-forward retraining,
intraday data, and a richer feature set.
