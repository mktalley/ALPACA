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
    """True if minute option parquet for *ul, exp* already cached."""
    return (_CACHE / "option" / ul.upper() / f"{exp:%Y%m%d}.parquet").exists()

import os


def ensure_data(ul: str, days: Iterable[date]):
    """Ensure local cache has option & stock data for *ul* over *days*.

    If ALPACA credentials are missing **or** environment variable ``ZERO_DTE_OFFLINE=1``
    is set, this function becomes a no-op so that back-tests can still run using the
    synthetic pricing fall-backs in apps.zero_dte.data.
    """
    if os.getenv("ZERO_DTE_OFFLINE") == "1" or not os.getenv("ALPACA_API_KEY"):
        # Pure-offline: rely on synthetic data generators (deterministic)
        return

    need: list[date] = []
    for d in days:
        if not (_have_greeks(ul, d) and _have_bars(ul, d)):
            need.append(d)
    if not need:
        return

    fetch_mod = importlib.import_module("apps.zero_dte.fetch_0dte_day")
    for d in need:
        out_p = _CACHE / "option_bars" / ul.upper() / f"{d:%Y%m%d}.parquet"
        out_p.parent.mkdir(parents=True, exist_ok=True)
        print("Downloading missing data for", d)
        import sys
        _orig_argv = sys.argv[:]
        sys.argv = [
            "fetch_0dte_day",
            "--underlying",
            ul,
            "--date",
            str(d),
            "--out",
            str(out_p),
        ]
        try:
            fetch_mod.main()
        finally:
            sys.argv = _orig_argv
