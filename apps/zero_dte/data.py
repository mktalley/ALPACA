"""Data helpers for 0-DTE back-tests.

We keep the implementation intentionally simple:
    • Thin wrappers around Alpaca HistoricalDataClient endpoints.
    • Local parquet cache (one file per (symbol, date)).
    • Utilities return pandas.DataFrame objects with a DatetimeIndex (UTC).

Environment variables (same as live trading code):
    ALPACA_API_KEY, ALPACA_SECRET_KEY – forwarded to the SDK automatically.

This module does **not** call Settings so that it can be used in tests without
needing a full .env file.
"""
from __future__ import annotations

import os
import pathlib
from datetime import datetime, date, time as dt_time, timedelta
from typing import Literal

import random

import pandas as pd
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import OptionBarsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame

__all__ = [
    "get_option_bars",
    "get_stock_bars",
]

# ---------------------------------------------------------------------------
# Caching helpers
# ---------------------------------------------------------------------------

_CACHE_ROOT = pathlib.Path(os.environ.get("ALPACA_CACHE_DIR", "~/.alpaca_cache")).expanduser()


def _cache_path(symbol: str, bar_date: date, kind: Literal["option", "stock"]) -> pathlib.Path:
    """Return the parquet cache path for a single-day bar file."""
    yyyy_mm_dd = bar_date.strftime("%Y%m%d")
    return _CACHE_ROOT / kind / symbol / f"{yyyy_mm_dd}.parquet"


# ---------------------------------------------------------------------------
# Client singletons – one per process
# ---------------------------------------------------------------------------

_stock_client: StockHistoricalDataClient | None = None
_option_client: OptionHistoricalDataClient | None = None


def _ensure_clients() -> tuple[StockHistoricalDataClient, OptionHistoricalDataClient]:
    global _stock_client, _option_client
    if _stock_client is None:
        _stock_client = StockHistoricalDataClient(
            api_key=os.getenv("ALPACA_API_KEY"),
            secret_key=os.getenv("ALPACA_API_SECRET"),
        )
    if _option_client is None:
        _option_client = OptionHistoricalDataClient(
            api_key=os.getenv("ALPACA_API_KEY"),
            secret_key=os.getenv("ALPACA_API_SECRET"),
        )
    return _stock_client, _option_client


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


def get_stock_bars(
    symbol: str,
    bar_date: date,
    timeframe: TimeFrame = TimeFrame.Minute,
) -> pd.DataFrame:
    """Return a DataFrame with minute bars for *symbol* on *bar_date*.

    The result is cached to
    ~/.alpaca_cache/stock/{symbol}/{yyyymmdd}.parquet so repeated calls are fast
    and do not hit the API.
    """
    path = _cache_path(symbol, bar_date, "stock")
    if path.exists():
        cached = _read_parquet_safe(path)
        if not cached.empty:
            return cached

    # ------------------------------------------------------------------
    # 1. Try Alpaca API
    # ------------------------------------------------------------------
    try:
        s_client, _ = _ensure_clients()
        start = datetime.combine(bar_date, dt_time(0, 0))
        end = start + timedelta(days=1)
        req = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=timeframe,
            start=start,
            end=end,
        )
        barset = s_client.get_stock_bars(req)
        df_all = barset.df
    except Exception:
        df_all = pd.DataFrame()

    if not df_all.empty:
        if isinstance(df_all.index, pd.MultiIndex):
            df = df_all.xs(symbol, level="symbol").copy().reset_index()
        else:
            df = df_all.reset_index()
            if "symbol" in df.columns:
                df = df[df["symbol"] == symbol]
        # Normalise column names
        if "timestamp" not in df.columns and "index" in df.columns:
            df.rename(columns={"index": "timestamp"}, inplace=True)
        df = df.set_index("timestamp").sort_index()
    else:
        # ------------------------------------------------------------------
        # 2. Fallback – generate synthetic bars
        # ------------------------------------------------------------------
        df = _synthetic_stock_bars(symbol, bar_date)

    _write_parquet_safe(df, path)
    return df

# ---------------------------------------------------------------------------
# Safe parquet helpers (tests may run without pyarrow)
# ---------------------------------------------------------------------------

from pathlib import Path


def _read_parquet_safe(path: Path) -> pd.DataFrame:
    """Return DataFrame or empty df if parquet engine unavailable."""
    try:
        return pd.read_parquet(path)
    except (ImportError, ValueError, OSError):
        # Either pyarrow/fastparquet missing or file corrupted – ignore.
        return pd.DataFrame()


def _write_parquet_safe(df: pd.DataFrame, path: Path) -> None:
    """Silently skip write if parquet engine unavailable."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
    except (ImportError, ValueError, OSError):
        # Cannot write parquet – tolerate in test environment
        pass


def get_option_bars(
    option_symbol: str,
    bar_date: date,
    timeframe: TimeFrame = TimeFrame.Minute,
) -> pd.DataFrame:
    """Return minute bars for *option_symbol* on *bar_date*.

    If Alpaca data are unavailable the function synthesises prices via
    Black-Scholes fed with the synthetic (or cached) equity bars.
    The returned DataFrame always has a UTC DatetimeIndex.
    """
    path = _cache_path(option_symbol, bar_date, "option")
    if path.exists():
        cached = _read_parquet_safe(path)
        if not cached.empty:
            return cached
        # Cached file was empty → rebuild

    # ------------------------------------------------------------------
    # 1. Try Alpaca
    # ------------------------------------------------------------------
    try:
        _, o_client = _ensure_clients()
        start = datetime.combine(bar_date, dt_time(0, 0))
        end = start + timedelta(days=1)
        req = OptionBarsRequest(symbol_or_symbols=[option_symbol], timeframe=timeframe, start=start, end=end)
        barset = o_client.get_option_bars(req)
        df_all = barset.df
    except Exception:
        df_all = pd.DataFrame()

    if not df_all.empty:
        # Flatten MultiIndex or filter single-index frame
        if isinstance(df_all.index, pd.MultiIndex):
            df = df_all.xs(option_symbol, level="symbol").copy().reset_index()
        else:
            df = df_all.reset_index()
            if "symbol" in df.columns:
                df = df[df["symbol"] == option_symbol]
        # Normalise column names → timestamp
        if "timestamp" not in df.columns and "index" in df.columns:
            df.rename(columns={"index": "timestamp"}, inplace=True)
        df = df.set_index("timestamp").sort_index()
    else:
        # ------------------------------------------------------------------
        # 2. Fallback – build synthetic option bars
        # ------------------------------------------------------------------
        df = _synthetic_option_bars(option_symbol, bar_date)
        # ensure proper index type
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.set_index(df.index)

    # Cache & return
    _write_parquet_safe(df, path)
    return df


def _synthetic_stock_bars(symbol: str, bar_date: date, start_price: float = 100.0) -> pd.DataFrame:
    """Generate synthetic minute bars (random walk) for offline back-tests."""
    minutes = 390  # 6.5h trading session
    prices = [start_price]
    for _ in range(minutes - 1):
        prices.append(prices[-1] * (1 + random.gauss(0, 0.0005)))
    rng = pd.date_range(
        datetime.combine(bar_date, dt_time(9, 30), tzinfo=ZoneInfo("America/New_York")).astimezone(ZoneInfo("UTC")),
        periods=minutes,
        freq="min",
    )
    df = pd.DataFrame({
        "open": prices,
        "high": [p * 1.001 for p in prices],
        "low":  [p * 0.999 for p in prices],
        "close": prices,
    }, index=rng)
    return df



# ---------------------------------------------------------------------------
# Synthetic option pricing fallback (Black-Scholes on equity bars)
# ---------------------------------------------------------------------------
from math import log, sqrt, erf
from zoneinfo import ZoneInfo
Eastern = ZoneInfo("America/New_York")

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def _bs_price(S: float, K: float, T: float, sigma: float, opt_type: str) -> float:
    if T <= 0:
        return max(0.0, S - K) if opt_type.upper() == 'C' else max(0.0, K - S)
    d1 = (log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    if opt_type.upper() == 'C':
        return S * _norm_cdf(d1) - K * _norm_cdf(d2)
    else:
        return K * _norm_cdf(-d2) - S * _norm_cdf(-d1)

def _parse_occ_symbol(sym: str):
    i = 0
    while i < len(sym) and not sym[i].isdigit():
        i += 1
    underlying = sym[:i]
    expiry = datetime.strptime(sym[i:i+6], "%y%m%d").date()
    opt_type = sym[i+6]
    strike = int(sym[-8:]) / 100.0
    return underlying, expiry, opt_type, strike

def _synthetic_option_bars(option_symbol: str, bar_date: date, sigma: float = 0.2) -> pd.DataFrame:
    underlying, expiry, opt_type, strike = _parse_occ_symbol(option_symbol)
    stock_bars = get_stock_bars(underlying, bar_date)
    from datetime import time as _time
    expiry_dt = datetime.combine(expiry, _time(16,0), tzinfo=Eastern)
    prices = []
    for ts, row in stock_bars.iterrows():
        S = row["close"]
        T = max((expiry_dt - ts).total_seconds(), 0) / (365 * 24 * 3600)
        p = _bs_price(S, strike, T, sigma, opt_type)
        prices.append(p)
    df = pd.DataFrame({"open": prices, "close": prices}, index=stock_bars.index)
    return df

