"""Smoke test: offline back-test uses synthetic bars and returns one TradeResult."""

import os
from datetime import date, time as dt_time

# Force offline mode (skip Alpaca API calls & downloads)
os.environ["ZERO_DTE_OFFLINE"] = "1"

from apps.zero_dte.grids import parse_grid
from apps.zero_dte.simulator import run_backtest


def test_offline_backtest_runs():
    params = parse_grid("target_pct=[0.3]; stop_pct=[0.15]")
    res = run_backtest(
        "SPY", date(2025, 5, 1), date(2025, 5, 1), params, dt_time(9, 35), dt_time(15, 45)
    )
    # Synthetic data generator always returns exactly one TradeResult per param set
    assert len(res) == 1
