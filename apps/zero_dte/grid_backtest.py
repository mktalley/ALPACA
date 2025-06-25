"""Simple parameter-grid runner for 0-DTE back-tests.

This module relies on :pyfunc:`apps.zero_dte.simulator.run_backtest` to perform
per-day simulations.  It merely expands a *grid* (dict of param→list) into the
Cartesian product, launches simulations and persists the results to a CSV so
users can quickly pivot-table the PnL distribution.

Example
-------
>>> from datetime import date, time
>>> from apps.zero_dte import grid_backtest as gb
>>> grid = {
...     "target_pct": [0.25, 0.35],
...     "stop_pct": [1.5, 2.0],
... }
>>> gb.run_grid(
...     symbol="SPY",
...     start=date(2025, 5, 1),
...     end=date(2025, 5, 31),
...     param_grid=grid,
...     entry_time=time(9, 45),
...     exit_cutoff=time(15, 30),
...     outfile="results.csv",
... )
"""
from __future__ import annotations

import csv
import itertools
import logging
import pathlib
from datetime import date, time as dt_time
from typing import Any, Dict, Iterable, Iterator, List

from .simulator import TradeResult, run_backtest
from .zero_dte_app import Settings

log = logging.getLogger(__name__)

__all__ = [
    "expand_grid",
    "run_grid",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def expand_grid(param_grid: Dict[str, Iterable[Any]]) -> Iterator[Dict[str, Any]]:
    """Yield dicts for every combination of *param_grid* values."""

    if not param_grid:
        yield {}
        return

    keys = list(param_grid)
    value_lists = [list(param_grid[k]) for k in keys]
    for combo in itertools.product(*value_lists):
        yield {k: v for k, v in zip(keys, combo, strict=True)}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_grid(
    *,
    symbol: str,
    start: date,
    end: date,
    param_grid: Dict[str, Iterable[Any]],
    entry_time: dt_time,
    exit_cutoff: dt_time,
    strategy: str = "strangle",
    outfile: str | pathlib.Path | None = None,
) -> List[TradeResult]:
    """Run back-test for **every** param combination and return list of trades.

    If *outfile* is provided the trades are appended to (or create) a CSV with
    columns::
        grid_id, date, entry_dt, exit_dt, pnl, param1, param2, ...
    """

    # Expand grid into raw param dicts – simulator will instantiate Settings
    grid_params: List[Dict[str, Any]] = list(expand_grid(param_grid))

    log.info("Running grid with %s combinations over %s→%s", len(grid_params), start, end)

    # -------------------- simulate --------------------
    trades = run_backtest(
        symbol=symbol,
        start=start,
        end=end,
        grid_params=grid_params,
        entry_time=entry_time,
        exit_cutoff=exit_cutoff,
        strategy=strategy,
    )

    log.info("Completed simulation → %s trades", len(trades))

    # -------------------- persist CSV --------------------
    if outfile:
        outfile = pathlib.Path(outfile)
        first_write = not outfile.exists()
        with outfile.open("a", newline="") as fp:
            w = csv.writer(fp)
            if first_write:
                header = [
                    "grid_id",
                    "date",
                    "entry_dt",
                    "exit_dt",
                    "pnl",
                ] + list(param_grid)
                w.writerow(header)
            for tr in trades:
                row = [
                    tr.grid_id,
                    tr.entry_dt.date(),
                    tr.entry_dt.isoformat(timespec="seconds"),
                    tr.exit_dt.isoformat(timespec="seconds"),
                    round(tr.pnl, 2),
                ] + [tr.params.get(k) for k in param_grid]
                w.writerow(row)
        log.info("Results appended to %s", outfile)

    return trades
