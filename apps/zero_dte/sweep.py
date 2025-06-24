"""Parallel grid-sweep driver around `apps.zero_dte.backtest.run_backtest`.

Example
-------
python -m apps.zero_dte.sweep \
    --underlying SPY --start 2025-06-01 --end 2025-06-30 \
    --entry 09:35 --cutoff 15:45 \
    --grid "STOP_LOSS_PCT=[0.2,0.25]; PROFIT_TARGET_PCT=[0.4,0.5]" \
    --workers 4 --out sweep.parquet
"""
from __future__ import annotations

import argparse
import os
import math
from datetime import date, time as dt_time
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List

import pandas as pd

from .grids import parse_grid
from .defaults import default_grid
from .simulator import run_backtest, TradeResult

# ---------------------------------------------------------------------------
# Helper utils
# ---------------------------------------------------------------------------

def _dt(s: str) -> dt_time:
    hh, mm = map(int, s.split(":"))
    return dt_time(hh, mm)


def _d(s: str) -> date:
    return date.fromisoformat(s)


def _chunk(lst: List[dict], n: int):
    """Yield *n* roughly equal chunks from *lst*."""
    k = math.ceil(len(lst) / n)
    for i in range(0, len(lst), k):
        yield lst[i : i + k]


# Worker --------------------------------------------------------------------

def _worker(args):
    (
        symbol,
        start,
        end,
        entry_time,
        cutoff,
        strategy,
        grid_subset,
    ) = args
    res = run_backtest(
        symbol=symbol,
        start=start,
        end=end,
        grid_params=grid_subset,
        entry_time=entry_time,
        exit_cutoff=cutoff,
        strategy=strategy,
    )
    # Convert to serialisable list[dict]
    return [r.__dict__ for r in res]


# CLI -----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser("0-DTE grid-sweep multiprocess driver")
    ap.add_argument("--underlying", default="SPY")
    ap.add_argument("--start", type=_d, required=True)
    ap.add_argument("--end", type=_d, required=True)
    ap.add_argument("--entry", type=_dt, default=dt_time(9, 35))
    ap.add_argument("--cutoff", type=_dt, default=dt_time(15, 45))
    ap.add_argument("--grid", default="", help="Grid string; default built-ins if omitted")
    ap.add_argument("--strategy", choices=["strangle", "condor"], default="strangle")
    ap.add_argument("--workers", type=int, default=min(4, cpu_count()))
    ap.add_argument("--out", required=False, help="Output parquet path")
    args = ap.parse_args()

    grid = parse_grid(args.grid) if args.grid else default_grid()
    if not grid:
        print("Grid is empty – nothing to run.")
        return

    # Partition work
    chunks = list(_chunk(grid, args.workers))
    worker_args = [
        (
            args.underlying,
            args.start,
            args.end,
            args.entry,
            args.cutoff,
            args.strategy,
            subset,
        )
        for subset in chunks
    ]

    print(f"Launching {len(chunks)} worker processes × {sum(len(c) for c in chunks)} combos…")
    with Pool(processes=len(chunks)) as pool:
        results_lists = pool.map(_worker, worker_args)

    flat: List[dict] = [item for sub in results_lists for item in sub]
    if not flat:
        print("No trades produced.")
        return

    df = pd.DataFrame(flat)
    if args.out:
        out_path = Path(args.out)
        df.to_parquet(out_path)
        print("Saved", out_path)
    else:
        print(df.head())


if __name__ == "__main__":
    # Ensure env creds load once in parent (multiprocessing forks env)
    if not os.getenv("ALPACA_API_KEY") and Path(".env").exists():
        from dotenv import load_dotenv

        load_dotenv()
    main()
