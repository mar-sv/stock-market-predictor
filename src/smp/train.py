"""CLI entrypoint: run the full pipeline for one ticker or all configured tickers.

Pipeline per ticker:
    load data -> features/datamodule -> Optuna study per architecture
    -> select best architecture -> refit + backtest on test split
    -> log the winner (params, test metrics, forecast plot) to MLflow.

Usage:
    smp-train                      # all tickers in config
    smp-train --ticker AAPL        # one ticker
    smp-train --config path.yaml --refresh-data
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow

from smp.backtest import backtest
from smp.config import load_config, resolve_path
from smp.data.loader import load_ticker
from smp.tracking import mlflow_utils
from smp.tuning.optuna_search import select_best_architecture


def run_ticker(ticker: str, cfg: dict, refresh: bool) -> dict:
    """Run selection + backtest for one ticker; return a small summary dict."""
    d = cfg["data"]
    df = load_ticker(
        ticker,
        start=d["start"],
        end=d.get("end"),
        interval=d.get("interval", "1d"),
        cache_dir=d.get("cache_dir", "data/raw"),
        refresh=refresh,
    )

    print(f"\n=== {ticker}: {len(df)} rows ({df.index.min().date()} .. {df.index.max().date()}) ===")
    winner, results = select_best_architecture(ticker, df, cfg)

    print(f"  validation quantile loss by architecture:")
    for r in sorted(results, key=lambda r: r.best_value):
        flag = "  <-- best" if r.arch == winner.arch else ""
        print(f"    {r.arch:5s}: {r.best_value:.6f}{flag}")

    # Refit the winner and backtest on the held-out test split.
    plot_dir = resolve_path("artifacts") / ticker
    test_metrics, plot_path = backtest(winner.arch, winner.best_params, df, cfg, plot_dir)

    # Log the winner as a dedicated MLflow run.
    mlflow_utils.setup_experiment(cfg, ticker)
    with mlflow_utils.start_run(
        run_name=f"BEST-{winner.arch}", tags={"arch": winner.arch, "ticker": ticker, "stage": "best"}
    ):
        mlflow_utils.log_params_flat({"arch": winner.arch, "ticker": ticker, **winner.best_params})
        mlflow_utils.log_metrics({f"val_{k}": v for k, v in winner.best_metrics.items()})
        mlflow_utils.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.log_artifact(str(plot_path))

    print(
        f"  WINNER {winner.arch} | test RMSE {test_metrics.get('rmse'):.5f} "
        f"| test dir-acc {test_metrics.get('directional_accuracy'):.3f} "
        f"| PnL Sharpe {test_metrics.get('pnl_sharpe_annualized'):.2f}"
    )
    return {"ticker": ticker, "arch": winner.arch, **test_metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="Stock market predictor pipeline")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--ticker", default=None, help="Single ticker (default: all in config)")
    parser.add_argument("--refresh-data", action="store_true", help="Re-download cached data")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tickers = [args.ticker] if args.ticker else cfg["data"]["tickers"]

    summaries = [run_ticker(t, cfg, args.refresh_data) for t in tickers]

    print("\n=== Summary ===")
    for s in summaries:
        print(
            f"  {s['ticker']:8s} -> {s['arch']:5s} | RMSE {s.get('rmse'):.5f} | "
            f"dir-acc {s.get('directional_accuracy'):.3f}"
        )


if __name__ == "__main__":
    main()
