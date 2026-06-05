"""Per-ticker, per-architecture Optuna search with MLflow tracking.

For each architecture we run an Optuna study that minimises the **validation
quantile loss** (computed in raw log-return space). Every trial is logged as an
MLflow run. The best architecture for a ticker is the one whose best trial has
the lowest validation quantile loss.

Note the deliberate split of concerns:

* the network *trains* on the quantile loss (``val_loss`` logged by Lightning),
* model *selection* ranks trials by the raw-space validation quantile loss
  returned by :func:`smp.evaluate.evaluate_model` (with directional accuracy and
  RMSE logged alongside for insight).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import optuna
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from smp.data.dataset import StockDataModule
from smp.evaluate import evaluate_model
from smp.models.registry import build_model, suggest_hparams
from smp.tracking import mlflow_utils


def make_trainer(cfg: dict) -> pl.Trainer:
    """Build a quiet Lightning trainer with early stopping on ``val_loss``."""
    tcfg = cfg["train"]
    return pl.Trainer(
        max_epochs=tcfg["max_epochs"],
        accelerator=tcfg.get("accelerator", "auto"),
        devices=1,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=tcfg["early_stopping_patience"])
        ],
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )


@dataclass
class StudyResult:
    arch: str
    best_value: float
    best_params: dict[str, Any]
    best_metrics: dict[str, float]


def run_study(arch: str, ticker: str, dm: StockDataModule, cfg: dict) -> StudyResult:
    """Run one Optuna study for ``arch`` on a prepared datamodule."""
    quantiles = cfg["forecast"]["quantiles"]
    horizon = cfg["forecast"]["horizon"]
    ocfg = cfg["optuna"]

    def objective(trial: optuna.Trial) -> float:
        pl.seed_everything(cfg["train"].get("seed", 42), workers=True)
        hp = suggest_hparams(arch, trial)
        dm.batch_size = hp["batch_size"]

        model = build_model(
            arch,
            n_past_features=dm.n_past_features,
            n_future_features=dm.n_future_features,
            horizon=horizon,
            quantiles=quantiles,
            hparams=hp["model"],
        )

        run_name = f"{arch}-trial{trial.number}"
        with mlflow_utils.start_run(run_name=run_name, tags={"arch": arch, "ticker": ticker}):
            mlflow_utils.log_params_flat({"arch": arch, "ticker": ticker, **hp})
            trainer = make_trainer(cfg)
            trainer.fit(model, datamodule=dm)
            metrics, _, _ = evaluate_model(model, dm.val_dataloader(), dm, quantiles)
            mlflow_utils.log_metrics({f"val_{k}": v for k, v in metrics.items()})

        trial.set_user_attr("metrics", metrics)
        return metrics["quantile_loss"]

    study = optuna.create_study(
        direction=ocfg.get("direction", "minimize"),
        study_name=f"{ticker}-{arch}",
    )
    study.optimize(objective, n_trials=ocfg["n_trials"], timeout=ocfg.get("timeout"))

    best = study.best_trial
    return StudyResult(
        arch=arch,
        best_value=study.best_value,
        best_params=best.params,
        best_metrics=best.user_attrs.get("metrics", {}),
    )


def select_best_architecture(
    ticker: str, df: pd.DataFrame, cfg: dict
) -> tuple[StudyResult, list[StudyResult]]:
    """Run a study per architecture for one ticker; return (winner, all results)."""
    mlflow_utils.setup_experiment(cfg, ticker)

    # Build the datamodule once and reuse across all studies/trials for the ticker.
    dm = StockDataModule(df, cfg)
    dm.setup()

    results = [run_study(arch, ticker, dm, cfg) for arch in cfg["optuna"]["architectures"]]
    winner = min(results, key=lambda r: r.best_value)
    return winner, results
