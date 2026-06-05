"""Model factory + per-architecture Optuna search spaces.

Keeps the mapping from architecture name -> class and the hyperparameter
suggestions in one place so the tuning and CLI code stay model-agnostic.
"""

from __future__ import annotations

from typing import Any

from smp.models.base import BaseForecaster
from smp.models.lstm import LSTMForecaster
from smp.models.tcn import TCNForecaster
from smp.models.tft import TFTForecaster

MODEL_CLASSES: dict[str, type[BaseForecaster]] = {
    "tcn": TCNForecaster,
    "lstm": LSTMForecaster,
    "tft": TFTForecaster,
}

# Constructor keys per architecture (excluding the shared dimension args).
ARCH_MODEL_KEYS: dict[str, list[str]] = {
    "tcn": ["hidden_size", "num_layers", "kernel_size"],
    "lstm": ["hidden_size", "num_layers"],
    "tft": ["hidden_size", "lstm_layers", "n_heads"],
}
COMMON_KEYS = ["lr", "dropout", "weight_decay"]


def params_to_kwargs(arch: str, params: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """Split a flat Optuna params dict into ``(model_kwargs, batch_size)``."""
    keys = ARCH_MODEL_KEYS[arch] + COMMON_KEYS
    model_kwargs = {k: params[k] for k in keys if k in params}
    return model_kwargs, int(params.get("batch_size", 64))


def build_model(
    arch: str,
    n_past_features: int,
    n_future_features: int,
    horizon: int,
    quantiles: list[float],
    hparams: dict[str, Any],
) -> BaseForecaster:
    """Instantiate a model by architecture name with the given hyperparameters."""
    cls = MODEL_CLASSES[arch]
    return cls(
        n_past_features=n_past_features,
        n_future_features=n_future_features,
        horizon=horizon,
        quantiles=quantiles,
        **hparams,
    )


def suggest_hparams(arch: str, trial) -> dict[str, Any]:
    """Suggest a hyperparameter set for ``arch`` using an Optuna trial.

    Common knobs (lr, dropout, weight_decay) are shared; the rest are
    architecture-specific. ``batch_size`` is returned separately under the key
    ``batch_size`` and consumed by the datamodule, not the model constructor.
    """
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.4)
    weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    if arch == "tcn":
        model = {
            "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128]),
            "num_layers": trial.suggest_int("num_layers", 2, 6),
            "kernel_size": trial.suggest_categorical("kernel_size", [2, 3, 5]),
        }
    elif arch == "lstm":
        model = {
            "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128]),
            "num_layers": trial.suggest_int("num_layers", 1, 3),
        }
    elif arch == "tft":
        model = {
            "hidden_size": trial.suggest_categorical("hidden_size", [16, 32, 64]),
            "lstm_layers": trial.suggest_int("lstm_layers", 1, 2),
            "n_heads": trial.suggest_categorical("n_heads", [1, 2, 4]),
        }
    else:
        raise ValueError(f"Unknown architecture: {arch!r}")

    model.update(lr=lr, dropout=dropout, weight_decay=weight_decay)
    return {"model": model, "batch_size": batch_size}
