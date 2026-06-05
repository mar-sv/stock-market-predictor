"""Configuration loading and a typed view over ``config/config.yaml``.

We keep this intentionally light: a single ``load_config`` that returns a plain
dict (so YAML stays the source of truth), plus a couple of helpers for the paths
and values used all over the codebase.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Repo root = three parents up from this file (src/smp/config.py -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML config into a dict, resolving ``null`` end-date to today.

    Args:
        path: Optional path to a config file. Defaults to ``config/config.yaml``.
    """
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh)
    return cfg


def resolve_path(relative: str | Path) -> Path:
    """Resolve a config-relative path against the repo root."""
    p = Path(relative)
    return p if p.is_absolute() else (REPO_ROOT / p)
