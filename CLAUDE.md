# CLAUDE.md

Guidance for working in this repo. Keep it current when commands or architecture change.

## What this is

A research project comparing three from-scratch deep-learning forecasters —
**TFT**, **LSTM**, **TCN** — on daily stock returns. Per ticker, an Optuna study
per architecture tunes hyperparameters; the best architecture is selected by a
time-aware validation metric; everything is tracked in MLflow.

## Environment (important)

- **There is no system Python on this machine** (`python`/`py` are the Microsoft
  Store stub and fail). Use [`uv`](https://docs.astral.sh/uv/) for everything —
  it's at `C:\Users\masv\.local\bin\uv.exe`.
- A uv-managed CPython 3.12 venv lives at `.venv`.

## Commands

```powershell
uv venv --python 3.12              # one-time: create the env
uv pip install -e ".[dev]"         # install project + dev deps

uv run pytest                      # run all tests
uv run pytest tests/test_data.py   # run one test file

uv run smp-train                           # full pipeline, all tickers in config
uv run smp-train --ticker AAPL             # one ticker
uv run smp-train --config config/smoke.yaml  # fast end-to-end smoke test
uv run smp-train --refresh-data            # ignore the parquet cache, re-download

uv run mlflow ui                   # browse experiments (reads ./mlruns)
```

## Core design decisions (don't silently change these)

- **Built from scratch in PyTorch + Lightning** — the goal is to learn the
  architectures. Do **not** replace them with a library (Darts, pytorch-forecasting).
- **Target = per-day log returns** (stationary). Direction is derived as
  `sign(q50)`, not a separate model.
- **Multi-horizon = next 5 trading days**, emitted in one forward pass.
- **Quantile (pinball) loss** at `[0.1, 0.5, 0.9]`, identical across all three
  models so the comparison is fair. Point forecast = q50.
- **Train vs. select:** networks train on quantile loss (`val_loss`); Optuna
  *selection* ranks by the validation quantile loss in **raw return space**
  (`smp.evaluate.evaluate_model`), reporting directional accuracy + RMSE too.
- **MLflow: tracking only** (local `./mlruns`). Registry deferred.

## Architecture / data flow

```
loader.load_ticker            yfinance OHLCV -> parquet cache (data/raw)
  -> features.build_features  log-return target + past covariates + calendar (future-known)
  -> dataset.StockDataModule  sliding windows; emits {past, future, target}
  -> models/*                 all subclass BaseForecaster, output (batch, horizon, n_quantiles)
  -> tuning.optuna_search     study per (ticker, arch); select_best_architecture picks winner
  -> backtest                 refit winner, rolling forecasts on test split, metrics + PnL + plot
  -> train.run_ticker         orchestrates the above; logs winner to MLflow
```

Key contracts:
- Every model implements `forward(past, future) -> (batch, horizon, n_quantiles)`
  and reuses `BaseForecaster` for train/val/optim. Add new models in
  `models/`, register them in `models/registry.py` (`MODEL_CLASSES`,
  `ARCH_MODEL_KEYS`, and `suggest_hparams`).
- TCN and LSTM use only `past`; the **TFT** also consumes `future` (calendar
  covariates). The TFT handles the no-calendar case with a zero placeholder.
- Predictions/targets come back in **standardized** space — always
  `dm.inverse_target(...)` before computing metrics or directions.

## Leakage controls (do not weaken)

In `data/dataset.py`: chronological splits, a **purge gap of `horizon`** between
splits, and the feature scaler + target mean/std fit on **training rows only**.
`tests/test_data.py` guards these — keep them green.

## Conventions

- Config is the source of truth (`config/config.yaml`); read it via
  `smp.config.load_config`. `config/smoke.yaml` is the fast variant.
- Repo-relative paths resolve through `smp.config.resolve_path`.
- Generated dirs are gitignored: `data/`, `mlruns/`, `mlartifacts/`, `artifacts/`,
  `.venv/`, `lightning_logs/`.

## Gotchas

- Newer MLflow flags the file store as deprecated; we opt in via
  `MLFLOW_ALLOW_FILE_STORE` (set in `tracking/mlflow_utils.setup_experiment`). To
  move to a DB backend, set `mlflow.tracking_uri: sqlite:///mlflow.db` in config.
- `num_workers: 0` in config — safest on Windows (avoids dataloader spawn issues).
- Full runs are CPU-bound and slow (9 studies for the default 3 tickers); use
  `config/smoke.yaml` to validate changes quickly.
```
