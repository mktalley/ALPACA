"""Bulk downloader for 0-DTE option minute bars + Greeks.

Example
-------
python -m apps.zero_dte.fetch_bulk \
    --symbols SPY,QQQ \
    --start 2025-06-01 --end 2025-06-30

The script walks every trading day in the inclusive [start, end] range and
invokes `apps.zero_dte.dl_missing.ensure_data()` which pulls data only when the
local parquet cache is missing.  This guarantees the command is idempotent and
safe to re-run.

Environment
~~~~~~~~~~~
Same as the rest of the zero_dte tools:
    ALPACA_API_KEY / ALPACA_API_SECRET – required for live API access.  If they
    are missing **or** the environment variable ``ZERO_DTE_OFFLINE=1`` is set,
    the function becomes a no-op (simulator will later fall back to synthetic
    prices).
"""
from __future__ import annotations

import argparse
import os
from datetime import date, timedelta
from pathlib import Path
from typing import List

from .dl_missing import ensure_data

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _d(s: str) -> date:  # YYYY-MM-DD → date
    return date.fromisoformat(s)


def _trading_days(start: date, end: date) -> List[date]:
    days: List[date] = []
    d = start
    while d <= end:
        if d.weekday() < 5:  # Monday-Friday only (no holiday calendar yet)
            days.append(d)
        d += timedelta(days=1)
    return days


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():  # noqa: D401 – CLI entry-point
    ap = argparse.ArgumentParser("Bulk 0-DTE downloader")
    ap.add_argument(
        "--symbols",
        required=True,
        help="Comma-separated list of underlyings (e.g. SPY,QQQ)",
    )
    ap.add_argument("--start", type=_d, required=True)
    ap.add_argument("--end", type=_d, required=True)
    args = ap.parse_args()

    syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if not syms:
        ap.error("--symbols must contain at least one symbol")

    days = _trading_days(args.start, args.end)
    if not days:
        ap.error("Date range contains no trading days")

    print(
        f"Ensuring option bars + Greeks cache for {len(days)} days and {syms} …",
        flush=True,
    )

    for sym in syms:
        ensure_data(sym, days)
        print(f"  ✓ {sym} done", flush=True)

    print("All done.  Local cache directory:", os.environ.get("ALPACA_CACHE_DIR", "~/.alpaca_cache"))


if __name__ == "__main__":
    main()
