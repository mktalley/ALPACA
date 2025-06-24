"""Ensure required market data (minute bars + Greeks snapshots) are cached locally.

If any requested *symbol, date* pair is missing from the on-disk cache, invoke
fetch_0dte_day.main() to download and store the files.
"""

from __future__ import annotations

import importlib
from datetime import date
from pathlib import Path
from typing import Iterable

from . import data as _data

_CACHE = _data._CACHE_ROOT


def _have_greeks(ul: str, exp: date) -> bool:
    return (_CACHE / "greeks" / ul.upper() / f"{exp:%Y%m%d}.parquet").exists()


def _have_bars(ul: str, exp: date) -> bool:
    return (_CACHE / "option_bars" / ul.upper() / f"{exp:%Y%m%d}.parquet").exists()


def ensure_data(ul: str, days: Iterable[date]):
    """Download missing data for *ul* over *days* using fetch_0dte_day."""
    need: list[date] = []
    for d in days:
        if not (_have_greeks(ul, d) and _have_bars(ul, d)):
            need.append(d)
    if not need:
        return
    fetch_mod = importlib.import_module("apps.zero_dte.fetch_0dte_day")
    for d in need:
        # mimic CLI args programmatically
        out_p = _CACHE / "option_bars" / ul.upper() / f"{d:%Y%m%d}.parquet"
        out_p.parent.mkdir(parents=True, exist_ok=True)
        print("Downloading missing data for", d)
        fetch_mod.main.__wrapped__(  # type: ignore[attr-defined]
            [
                "--underlying",
                ul,
                "--date",
                str(d),
                "--out",
                str(out_p),
            ]
        )
