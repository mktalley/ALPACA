"""Lightweight Alpaca wrappers with on-disk parquet caching for back-tests."""

from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path
from typing import Union

import pandas as pd

try:
    from alpaca.data.historical.option import OptionHistoricalDataClient
    from alpaca.data.historical.stock import StockHistoricalDataClient
    from alpaca.data.requests import OptionBarsRequest, StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    _HAS_ALPACA = True
except Exception:  # pragma: no cover
    # SDK not available – we will fall back to synthetic bars via .pricing
    _HAS_ALPACA = False
    OptionHistoricalDataClient = StockHistoricalDataClient = None  # type: ignore
    OptionBarsRequest = StockBarsRequest = TimeFrame = None  # type: ignore

CACHE = Path(".cache_bt")
CACHE.mkdir(exist_ok=True)

_STOCK = StockHistoricalDataClient() if _HAS_ALPACA else None
_OPT = OptionHistoricalDataClient() if _HAS_ALPACA else None


def _fp(tag: str, symbol: str, start: date, end: date) -> Path:
    return CACHE / f"{tag}_{symbol}_{start}_{end}.parq"


def _load_or(tag: str, path: Path, fetch_func):
    if path.exists():
        return pd.read_parquet(path)
    df = fetch_func()
    df.to_parquet(path)
    return df


def get_stock_bars(symbol: str, start: date, end: date):  # noqa: D401
    """Return 1-minute underlying bars, *synthetic* if Alpaca unavailable."""

    if not _HAS_ALPACA:
        from .pricing import get_stock_bars as _fake  # local fallback

        return _fake(symbol, start, end)

    p = _fp("stk1m", symbol, start, end)
    return _load_or(
        "stk1m",
        p,
        lambda: _STOCK.get_stock_bars(
            StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Minute,
                start=start,
                end=end,
            )
        ).df,
    )


def get_option_bars(symbol: str, start: date, end: date) -> pd.DataFrame:
    """1-min bars for one option contract – cached."""
    p = _fp("opt1m", symbol, start, end)
    return _load_or(
        "opt1m",
        p,
        lambda: _OPT.get_option_bars(
            OptionBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Minute,
                start=start,
                end=end,
            )
        ).df,
    )
