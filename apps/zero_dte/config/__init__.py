"""Configuration helpers for Zero-DTE application."""
from __future__ import annotations

import yaml
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping


_CFG_PATH = Path(__file__).with_suffix("").with_name("day_type.yaml")


@lru_cache(maxsize=1)
def load_daytype_config(path: str | Path | None = None) -> Mapping[str, Any]:
    """Load YAML config for day-type classifier.

    If *path* is *None* the default *day_type.yaml* in the same package is
    used. The result is cached so subsequent calls are cheap.
    """
    p = Path(path) if path else _CFG_PATH
    try:
        with p.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
            return data
    except FileNotFoundError:
        # Absent config is not fatal – caller can apply defaults.
        return {}
