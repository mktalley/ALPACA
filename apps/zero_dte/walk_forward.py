#!/usr/bin/env python3
"""Walk-forward wrapper around simulator.run_backtest.

The walk-forward procedure repeatedly splits the overall period into contiguous
*train* and *test* windows.  For each split we grid-search the supplied param
combinations on the *train* period, pick the set with the best P&L, and then
simulate the chosen params on the following *test* period.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import date
from datetime import time as dt_time
from datetime import timedelta
from pathlib import Path
from typing import List

import pandas as pd

from .defaults import default_grid
from .grids import parse_grid
from .simulator import TradeResult, run_backtest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _next_weekday(d: date, n: int) -> date:
    """Return date *n* weekdays (Mon–Fri) after *d* (inclusive count)."""
    day = d
    step = 1 if n >= 0 else -1
    remaining = abs(n)
    while remaining:
        day += timedelta(days=step)
        if day.weekday() < 5:
            remaining -= 1
    return day


def _walk_splits(start: date, end: date, train_days: int, test_days: int):
    """Yield successive (train_start, train_end, test_start, test_end) tuples."""
    window_start = start
    # Ensure we always have enough room for a full train+test window
    while True:
        train_start = window_start
        train_end = _next_weekday(train_start, train_days - 1)
        test_start = _next_weekday(train_end, 1)
        test_end = _next_weekday(test_start, test_days - 1)
        if test_end > end:
            break
        yield (train_start, train_end, test_start, test_end)
        window_start = _next_weekday(test_start, test_days)  # slide by test window


# ---------------------------------------------------------------------------
# Core walk-forward routine
# ---------------------------------------------------------------------------


def run_walk_forward(
    *,
    symbol: str,
    start: date,
    end: date,
    grid_params: List[dict],
    train_days: int = 5,
    test_days: int = 1,
    entry_time: dt_time = dt_time(9, 35),
    exit_cutoff: dt_time = dt_time(15, 45),
    strategy: str = "strangle",
):
    # Ensure required market data cached for whole period
    from datetime import timedelta

    from .dl_missing import ensure_data

    days = []
    d = start
    while d <= end:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    ensure_data(symbol, days)

    """Perform walk-forward optimisation returning list[TradeResult]."""
    all_trades: List[TradeResult] = []
    splits = list(_walk_splits(start, end, train_days, test_days))
    if not splits:
        raise ValueError("Date range too short for requested windows")

    for idx, (tr_start, tr_end, te_start, te_end) in enumerate(splits, 1):
        # 1. Grid search on training window
        best_params = None
        best_pnl = float("-inf")
        for params in grid_params:
            train_res = run_backtest(
                symbol=symbol,
                start=tr_start,
                end=tr_end,
                grid_params=[params],
                entry_time=entry_time,
                exit_cutoff=exit_cutoff,
                strategy=strategy,
            )
            pnl = sum(r.pnl for r in train_res)
            if pnl > best_pnl:
                best_pnl, best_params = pnl, params
        if best_params is None:
            # Could not find any trades; skip this split
            continue
        # 2. Evaluate on test window with chosen params
        test_res = run_backtest(
            symbol=symbol,
            start=te_start,
            end=te_end,
            grid_params=[best_params],
            entry_time=entry_time,
            exit_cutoff=exit_cutoff,
            strategy=strategy,
        )
        # Flag trades with window index for traceability
        for tr in test_res:
            tr.params = best_params
            tr.grid_id = idx  # reuse grid_id to mark split
        all_trades.extend(test_res)
    return all_trades


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def _d(s: str) -> date:
    return date.fromisoformat(s)


def _dt(s: str) -> dt_time:
    hh, mm = map(int, s.split(":"))
    return dt_time(hh, mm)


def main():
    ap = argparse.ArgumentParser("0-DTE walk-forward back-test CLI")
    ap.add_argument("--underlying", default="SPY")
    ap.add_argument("--start", type=_d, required=True)
    ap.add_argument("--end", type=_d, required=True)
    ap.add_argument(
        "--train", type=int, default=5, help="Training window length (weekdays)"
    )
    ap.add_argument("--test", type=int, default=1, help="Test window length (weekdays)")
    ap.add_argument(
        "--grid", default="", help="Grid string; default built-ins if empty"
    )
    ap.add_argument("--entry", type=_dt, default=dt_time(9, 35))
    ap.add_argument("--cutoff", type=_dt, default=dt_time(15, 45))
    ap.add_argument("--strategy", choices=["strangle", "condor"], default="strangle")
    ap.add_argument("--out")
    args = ap.parse_args()

    grid_params = parse_grid(args.grid) if args.grid else default_grid()
    trades = run_walk_forward(
        symbol=args.underlying,
        start=args.start,
        end=args.end,
        grid_params=grid_params,
        train_days=args.train,
        test_days=args.test,
        entry_time=args.entry,
        exit_cutoff=args.cutoff,
        strategy=args.strategy,
    )
    df = pd.DataFrame([t.__dict__ for t in trades])
    if args.out:
        out_path = Path(args.out)
        df.to_parquet(out_path)
        df.to_csv(out_path.with_suffix(".csv"), index=False)
        print("Wrote", out_path)
    else:
        print(df.head())


if __name__ == "__main__":
    main()
