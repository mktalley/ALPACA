#!/usr/bin/env python3
"""CLI wrapper around simulator.run_backtest.

Usage example::
    python -m apps.zero_dte.backtest \
       --underlying SPX --start 2025-05-01 --end 2025-05-31 \
       --entry 09:35 --cutoff 15:45 \
       --grid "STOP_LOSS_PCT=[0.3,0.4]; PROFIT_TARGET_PCT=[0.15,0.2]" \
       --out may25.parquet
"""
from __future__ import annotations

import argparse
import os
from datetime import date
from datetime import time as dt_time
from pathlib import Path

import pandas as pd

from .defaults import default_grid
from .grids import parse_grid
from .simulator import run_backtest


def _load_dotenv():
    """Lightweight .env loader (only KEY=VALUE lines)."""
    here = Path(__file__).resolve()
    root = here.parent.parent.parent  # go to repo root
    env_path = root / ".env"
    if not env_path.exists():
        return
    with env_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            os.environ.setdefault(k, v)


# Load env before anything else (so SDK picks credentials)
_load_dotenv()


def _dt(s: str) -> dt_time:
    hh, mm = map(int, s.split(":"))
    return dt_time(hh, mm)


def _d(s: str) -> date:
    return date.fromisoformat(s)


def main():
    p = argparse.ArgumentParser("0-DTE back-test CLI")
    p.add_argument("--underlying", default="SPX")
    p.add_argument("--start", type=_d, required=True)
    p.add_argument("--end", type=_d, required=True)
    p.add_argument("--entry", type=_dt, default=dt_time(9, 35))
    p.add_argument("--cutoff", type=_dt, default=dt_time(15, 45))
    p.add_argument(
        "--grid",
        required=False,
        default="",
        help="Grid string. If empty use built-in defaults.",
    )
    p.add_argument("--strategy", choices=["strangle", "condor"], default="strangle")
    p.add_argument("--out", required=False)
    args = p.parse_args()
    # Ensure local option bars/greeks cache is available before simulation
    from datetime import timedelta

    from .dl_missing import ensure_data

    day = args.start
    needed = []
    while day <= args.end:
        if day.weekday() < 5:  # Mon-Fri only
            needed.append(day)
        day += timedelta(days=1)
    ensure_data(args.underlying, needed)

    params = parse_grid(args.grid) if args.grid else default_grid()
    results = run_backtest(
        symbol=args.underlying,
        start=args.start,
        end=args.end,
        grid_params=params,
        entry_time=args.entry,
        exit_cutoff=args.cutoff,
        strategy=args.strategy,
    )

    # dump to df
    df = pd.DataFrame([r.__dict__ for r in results])
    if args.out:
        out_path = Path(args.out)
        # Parquet for lossless persistence
        df.to_parquet(out_path)
        # CSV companion for quick eyeballing in spreadsheets
        csv_path = out_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        print("Wrote", out_path, "and", csv_path)
    else:
        print(df.head())


if __name__ == "__main__":
    main()
