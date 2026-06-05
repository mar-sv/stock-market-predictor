"""Thin MLflow helpers (tracking only — no model registry yet).

One experiment per ticker (``<prefix>/<ticker>``); each Optuna trial is a run.
We log flattened params, scalar metrics, and artifact files. The layout is
deliberately registry-friendly so a Model Registry step can be added later
without reshaping anything.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from numbers import Number
from typing import Any

import mlflow

from smp.config import resolve_path


def setup_experiment(cfg: dict, ticker: str) -> str:
    """Point MLflow at the configured tracking URI and select the ticker's experiment.

    Returns the experiment name.
    """
    # The local file store is in "maintenance mode" in newer MLflow; opt in
    # explicitly since tracking-only experiment logging is all we need.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    uri = cfg["mlflow"]["tracking_uri"]
    # Local relative URIs are resolved against the repo root for stability.
    if "://" not in uri:
        uri = resolve_path(uri).as_uri()
    mlflow.set_tracking_uri(uri)
    name = f"{cfg['mlflow']['experiment_prefix']}/{ticker}"
    mlflow.set_experiment(name)
    return name


def _flatten(prefix: str, obj: Any, out: dict[str, Any]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(f"{prefix}.{k}" if prefix else str(k), v, out)
    else:
        out[prefix] = obj


def log_params_flat(params: dict[str, Any]) -> None:
    """Log a (possibly nested) param dict as flat ``a.b.c`` keys."""
    flat: dict[str, Any] = {}
    _flatten("", params, flat)
    mlflow.log_params(flat)


def log_metrics(metrics: dict[str, Any]) -> None:
    """Log only the numeric entries of a metrics dict."""
    numeric = {k: float(v) for k, v in metrics.items() if isinstance(v, Number)}
    mlflow.log_metrics(numeric)


@contextmanager
def start_run(run_name: str, tags: dict[str, str] | None = None):
    """Context manager wrapping ``mlflow.start_run`` with a name and tags."""
    with mlflow.start_run(run_name=run_name, tags=tags) as run:
        yield run
