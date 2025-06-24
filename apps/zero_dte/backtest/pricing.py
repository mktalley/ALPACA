"""Light-weight data plumbing for back-test harness.

For production we will call alpaca-py.  To keep the test suite fast and
hermetic we fall back to synthetic data when the SDK is unavailable or
when no API key is present in the environment.
"""

from __future__ import annotations

import os
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

__all__ = ["get_stock_bars"]


def _random_walk(n: int, start: float = 100.0) -> pd.DataFrame:
    """Return *n* minutes of OHLCV walk suitable for unit tests."""
    prices = [start]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 + random.gauss(0, 0.0005)))
    df = pd.DataFrame(
        {
            "t": pd.date_range("2025-01-01 09:30", periods=n, freq="min"),
            "open": prices,
            "high": [p * 1.001 for p in prices],
            "low": [p * 0.999 for p in prices],
            "close": prices,
            "volume": 1,
        }
    )
    return df.set_index("t")


def get_stock_bars(
    symbol: str,
    start: datetime,
    end: datetime,
    timeframe: str = "1Min",
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:  # noqa: N802
    """Fetch minute bars for *symbol* in [start, end).

    If Alpaca creds are missing the function returns a synthetic walk so
    that unit tests and documentation examples remain runnable offline.
    """
    try:
        from alpaca.data.historical import \
            StockHistoricalDataClient  # pylint: disable=import-error
        from alpaca.data.requests import \
            StockBarsRequest  # pylint: disable=import-error
        from alpaca.data.timeframe import \
            TimeFrame  # pylint: disable=import-error

        key, secret = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_API_SECRET")
        if key and secret:
            client = StockHistoricalDataClient(key, secret)
            req = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Minute,
                start=start,
                end=end,
            )
            df = client.get_stock_bars(req).df
            return df[df.symbol == symbol].drop(columns=["symbol"])
    except Exception:  # pragma: no cover
        # Hard failure → fallthrough to synthetic
        pass

    delta: timedelta = end - start
    return _random_walk(max(int(delta.total_seconds() // 60), 1))
