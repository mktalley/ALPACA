"""Exhaustive grid search over parameter dict on historical date list."""

from __future__ import annotations

import itertools
from datetime import datetime
from typing import Any, Dict, Iterable, List

import pandas as pd

from apps.zero_dte.data import AlpacaBarLoader
from .simulator import StrangleSimulator


def _daterange(start: datetime, end: datetime) -> Iterable[datetime]:
    curr = start
    while curr < end:
        yield curr
        curr += pd.Timedelta(days=1)


def _score_combo(args):
    """Helper function for ProcessPoolExecutor.

    Parameters
    ----------
    args : tuple(symbol, datetime start, datetime end, combo_dict)
    """
    symbol, start_d, end_d, combo = args
    pnl_sum = 0.0
    n_trades = 0
    loader = AlpacaBarLoader()
    for day in _daterange(start_d, end_d):
        bars = loader.get(symbol, day, day + pd.Timedelta(days=1))
        entry_time = bars.index[30]  # 10:00 assuming 9:30 open
        sim = StrangleSimulator(
            bars,
            entry_time=entry_time,
            entry_credit=2.0,
            qty=1,
            stop_loss_pct=combo.get("STOP_LOSS_PCT", 0.2),
            profit_target_pct=combo.get("PROFIT_TARGET_PCT", 0.65),
            slippage=combo.get("SLIPPAGE", 0.0),
            commission=combo.get("COMMISSION", 0.0),
        )
        pnl_sum += sim.run().pnl
        n_trades += 1
    return {**combo, "pnl": pnl_sum, "trades": n_trades}


def run_grid(
    symbol: str,
    start: datetime,
    end: datetime,
    params: Dict[str, list[Any]],
    *,
    n_jobs: int = 1,
) -> pd.DataFrame:
    keys = list(params)
    combos: List[Dict[str, Any]] = [
        dict(zip(keys, vs, strict=True)) for vs in itertools.product(*params.values())
    ]

    if n_jobs > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            tasks = [
                (symbol, start, end, c) for c in combos
            ]
            rows = list(ex.map(_score_combo, tasks))
    else:
        rows = [_score_combo((symbol, start, end, c)) for c in combos]

    return pd.DataFrame(rows)
