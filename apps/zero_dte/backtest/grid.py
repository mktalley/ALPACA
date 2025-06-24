"""Exhaustive grid search over parameter dict on historical date list."""

from __future__ import annotations

import itertools
from datetime import datetime
from typing import Any, Dict, Iterable

import pandas as pd

from .pricing import get_stock_bars
from .simulator import StrangleSimulator, TradeResult


def _daterange(start: datetime, end: datetime) -> Iterable[datetime]:
    curr = start
    while curr < end:
        yield curr
        curr += pd.Timedelta(days=1)


def run_grid(
    symbol: str,
    start: datetime,
    end: datetime,
    params: Dict[str, list[Any]],
) -> pd.DataFrame:
    keys = list(params)
    rows = []
    for values in itertools.product(*params.values()):
        combo = dict(zip(keys, values, strict=True))
        pnl_sum = 0.0
        n_trades = 0
        for day in _daterange(start, end):
            bars = get_stock_bars(symbol, day, day + pd.Timedelta(days=1))
            # naive 10-AM entry example
            entry_time = bars.index[30]  # 10:00 if bars start 9:30
            sim = StrangleSimulator(
                bars,
                entry_time=entry_time,
                entry_credit=2.0,  # placeholder credit
                qty=1,
                stop_loss_pct=combo.get("STOP_LOSS_PCT", 0.2),
                profit_target_pct=combo.get("PROFIT_TARGET_PCT", 0.65),
            )
            res: TradeResult = sim.run()
            pnl_sum += res.pnl
            n_trades += 1
        rows.append({**combo, "pnl": pnl_sum, "trades": n_trades})
    return pd.DataFrame(rows)
