"""Very small offline smoke-test: replay 1 synthetic day to make sure the
zero_dte simulator never crashes due to refactors.

We run with ZERO_DTE_OFFLINE=1 so that no network calls occur; the data module
will generate deterministic synthetic price bars.
"""
from datetime import date, time
import os

import pandas as pd

from apps.zero_dte.simulator import run_backtest
from apps.zero_dte.defaults import default_grid

os.environ.setdefault("ZERO_DTE_OFFLINE", "1")


def test_one_day_smoke():
    results = run_backtest(
        symbol="SPY",
        start=date(2025, 6, 2),
        end=date(2025, 6, 2),
        grid_params=default_grid(),
        entry_time=time(9, 45),
        exit_cutoff=time(15, 30),
        strategy="strangle",
    )
    # Should return list[TradeResult] with deterministic synthetic PnL
    assert results, "No trades returned"
    pnl_series = pd.Series([r.pnl for r in results])
    # Synthetic generator always returns between -200 and +200 per trade
    assert pnl_series.between(-300, 300).all()
