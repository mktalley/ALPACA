"""Fetch all minute bars for 0-DTE SPY options for a single day.

Usage
-----
python -m apps.zero_dte.fetch_0dte_day \
    --underlying SPY \
    --date 2025-06-20 \
    --out data/spy_0dte_20250620.parquet

Outputs both parquet and CSV side-by-side.

Notes
-----
* Relies on ALPACA_API_KEY / ALPACA_API_SECRET in environment or .env
* Uses OptionHistoricalDataClient v2 endpoints via SDK.
* Snapshot endpoint returns *all* contracts; we filter with OCC YYMMDD segment.
"""

from __future__ import annotations

import argparse
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List

import pandas as pd

from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import OptionBarsRequest, OptionChainRequest
from alpaca.data.timeframe import TimeFrame

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _contracts_for_day(
    client: OptionHistoricalDataClient, underlying: str, exp: date
) -> List[str]:
    """Return list of OCC option-contract symbols for *underlying* that expire on *exp*."""
    # Use Trading API contracts endpoint because it includes inactive/expired contracts.
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import AssetStatus
    from alpaca.trading.requests import GetOptionContractsRequest

    trading = TradingClient(
        os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_API_SECRET"), paper=True
    )
    req = GetOptionContractsRequest(
        underlying_symbols=[underlying],
        expiration_date=exp,
        status=AssetStatus.INACTIVE,  # include expired contracts
        limit=10_000,
    )
    resp = trading.get_option_contracts(req)

    # The SDK returns either a typed response object (attr ``option_contracts``)
    # or raw dict when `use_raw_data=True`. Handle both.
    contracts = (
        resp.option_contracts
        if hasattr(resp, "option_contracts")
        else resp.get("option_contracts", [])
    )

    # Filter to valid OCC symbols (e.g. SPY240621C00410000)
    import re

    base = underlying.upper()
    pat = re.compile(rf"^{base}(\d{{6}}|\d{{7}})[CP]\d{{8}}$")
    return [
        c.symbol if hasattr(c, "symbol") else c["symbol"]
        for c in contracts
        if pat.match(c.symbol if hasattr(c, "symbol") else c["symbol"])
    ]


def _fetch_bars_for_symbols(
    client: OptionHistoricalDataClient,
    symbols: List[str],
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """Download minute bars for *symbols* between start/end (UTC)."""
    dfs: List[pd.DataFrame] = []
    CHUNK = 800  # stay below 1000 symbol hard-limit
    for i in range(0, len(symbols), CHUNK):
        chunk = symbols[i : i + CHUNK]
        req = OptionBarsRequest(
            symbol_or_symbols=chunk,
            timeframe=TimeFrame.Minute,
            start=start,
            end=end,
            limit=10_000,
            feed="opra",  # OPRA feed
        )
        barset = client.get_option_bars(req)
        if barset.df.empty:
            continue
        dfs.append(barset.df.reset_index())
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Fetch 0-DTE option minute bars for one day"
    )
    ap.add_argument("--underlying", default="SPY")
    ap.add_argument(
        "--date",
        required=True,
        help="YYYY-MM-DD expiration / trade date (must be a Friday)",
    )
    ap.add_argument("--out", required=True, help="Output parquet file path")
    args = ap.parse_args()

    exp = date.fromisoformat(args.date)
    out_path = Path(args.out).expanduser()

    # Ensure API creds
    if not os.getenv("ALPACA_API_KEY") and Path(".env").exists():
        from dotenv import load_dotenv

        load_dotenv()

    client = OptionHistoricalDataClient(
        api_key=os.getenv("ALPACA_API_KEY"),
        secret_key=os.getenv("ALPACA_API_SECRET"),
    )

    print("Enumerating contracts …", end="", flush=True)
    symbols = _contracts_for_day(client, args.underlying, exp)
    print(f" {len(symbols)} found")
    if not symbols:
        print("No contracts returned; check expiration date or entitlements.")
        return

    # Market hours 09:30-16:00 US/Eastern; Alpaca stores in UTC.
    # Use 00:00-23:59 UTC to catch any bars.
    start_dt = datetime.combine(exp, datetime.min.time())
    end_dt = start_dt + timedelta(days=1)

    print("Downloading minute bars …", flush=True)
    df = _fetch_bars_for_symbols(client, symbols, start_dt, end_dt)
    if df.empty:
        print("No bars fetched. Exiting.")
        return

    # ------------------------------------------------------------------
    # 2. Fetch option snapshots to capture Greeks (delta) once during day
    # ------------------------------------------------------------------
    from alpaca.data.requests import OptionSnapshotRequest

    SNAP_CHUNK = 100  # Alpaca snapshot API hard limit
    snaps = []
    for i in range(0, len(symbols), SNAP_CHUNK):
        chunk = symbols[i : i + SNAP_CHUNK]
        req = OptionSnapshotRequest(symbol_or_symbols=chunk, feed="opra")
        resp = client.get_option_snapshot(req)
        # resp is Mapping[str, Snapshot]; transform to rows
        for sym in chunk:
            snap = resp.get(sym)
            if not snap:
                continue
            greeks = getattr(snap, "greeks", None)
            if not greeks:
                continue
            snaps.append(
                {
                    "symbol": sym,
                    "delta": greeks.delta,
                    "theta": greeks.theta,
                    "gamma": greeks.gamma,
                    "vega": greeks.vega,
                }
            )
    df_g = pd.DataFrame(snaps)

    # ------------------------------------------------------------------
    # 3. Save raw parquet/csv outputs
    # ------------------------------------------------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    csv_path = out_path.with_suffix(".csv")
    df.to_csv(csv_path, index=False)

    if not df_g.empty:
        # Save alongside bar file for archival
        g_path = out_path.with_name(out_path.stem + "_greeks.parquet")
        df_g.to_parquet(g_path)
        print(f"Greeks saved to {g_path}")
        # Also drop a copy into cache for simulator fast access
        from apps.zero_dte import data as _data

        g_cache = (
            _data._CACHE_ROOT
            / "greeks"
            / args.underlying.upper()
            / f"{exp:%Y%m%d}.parquet"
        )
        g_cache.parent.mkdir(parents=True, exist_ok=True)
        df_g.to_parquet(g_cache)

    print(f"Saved {len(df):,} bar rows to {out_path} and {csv_path}")

    # ------------------------------------------------------------------
    # 3. Populate 0-DTE cache used by apps.zero_dte.data
    # ------------------------------------------------------------------
    from apps.zero_dte import data as _data

    # Equity bars first (this will internally cache)
    _data.get_stock_bars(args.underlying, exp)

    rows_written = 0
    for sym, g in df.groupby("symbol"):
        g2 = g.drop(columns=["symbol"]).set_index("timestamp").sort_index()
        p = _data._cache_path(sym, exp, "option")
        _data._write_parquet_safe(g2, p)
        rows_written += len(g2)
    print(f"Cached {rows_written:,} option-bar rows into {_data._CACHE_ROOT}")


if __name__ == "__main__":
    main()
