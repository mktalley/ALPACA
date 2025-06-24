"""Fetch minute bars for an equity on a single day and save to parquet + csv.

Usage
-----
python -m apps.zero_dte.fetch_equity_day \
    --symbol SPY \
    --date 2024-06-21 \
    --out data/spy_equity_20240621.parquet

Requires ALPACA_API_KEY / ALPACA_API_SECRET in env or .env.
"""

from __future__ import annotations

import argparse
import os
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


def main():
    ap = argparse.ArgumentParser("Fetch equity minute bars for a day")
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--date", required=True, help="YYYY-MM-DD trading date")
    ap.add_argument("--out", required=True, help="Output parquet file path")
    args = ap.parse_args()

    trade_date = date.fromisoformat(args.date)
    out_path = Path(args.out).expanduser()

    if not os.getenv("ALPACA_API_KEY") and Path(".env").exists():
        from dotenv import load_dotenv

        load_dotenv()

    client = StockHistoricalDataClient(
        api_key=os.getenv("ALPACA_API_KEY"),
        secret_key=os.getenv("ALPACA_API_SECRET"),
    )

    start_dt = datetime.combine(trade_date, datetime.min.time())
    end_dt = start_dt + timedelta(days=1)
    req = StockBarsRequest(
        symbol_or_symbols=[args.symbol],
        timeframe=TimeFrame.Minute,
        start=start_dt,
        end=end_dt,
        limit=10000,
    )
    barset = client.get_stock_bars(req)
    df = barset.df.reset_index()
    if df.empty:
        print("No bars fetched.")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    csv_path = out_path.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df):,} rows to {out_path} and {csv_path}")


if __name__ == "__main__":
    main()
