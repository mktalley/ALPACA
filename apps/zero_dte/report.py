"""Command-line report generator for 0-DTE back-test parquet files."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .analytics import metrics_from_trades, pivot_heatmap, equity_curve


def flatten_params(df: pd.DataFrame) -> pd.DataFrame:
    """Expand the *params* dict column into individual columns."""
    if "params" not in df.columns:
        return df
    params_expanded = pd.json_normalize(df["params"])
    params_expanded.index = df.index
    return pd.concat([df.drop(columns=["params"]), params_expanded], axis=1)


def main():
    ap = argparse.ArgumentParser("0-DTE back-test reporter")
    ap.add_argument("file", help="Parquet file output by apps.zero_dte.backtest")
    ap.add_argument("--heat", nargs=2, metavar=("X", "Y"), help="Pivot two columns into a heat-map of mean PnL")
    ap.add_argument("--equity", action="store_true", help="Plot equity curve")
    args = ap.parse_args()

    path = Path(args.file)
    df = pd.read_parquet(path)
    df = flatten_params(df)

    m = metrics_from_trades(df)
    for k, v in m.items():
        print(f"{k:15s}: {v:,.2f}")

    # Plotting
    if args.heat:
        x, y = args.heat
        pivot_heatmap(df, x, y, outfile=path.with_suffix(f"_{x}_{y}_heat.png"))
        print("Saved heat-map png")
    if args.equity:
        equity_curve(df, outfile=path.with_suffix("_equity.png"))
        print("Saved equity curve png")


if __name__ == "__main__":
    main()
