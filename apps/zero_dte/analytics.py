"""Analytics helpers for 0-DTE back-tests.

Most functions accept a *pandas.DataFrame* that comes from::

    df = pandas.DataFrame([r.__dict__ for r in run_backtest(...)])

Columns expected:
* pnl – per-trade profit/loss
* entry_dt / exit_dt – timezone-aware datetimes
* params – dict of the parameters used (flattened by caller if needed)
"""
from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

import math
import pandas as pd

try:
    import seaborn as sns  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore
except ModuleNotFoundError:  # pragma: no cover – plots optional in CI
    sns = None  # type: ignore
    plt = None  # type: ignore

__all__ = [
    "metrics_from_trades",
    "pivot_heatmap",
    "equity_curve",
]


def metrics_from_trades(df: pd.DataFrame) -> Mapping[str, float]:
    """Return basic summary metrics for a back-test results DataFrame."""
    if df.empty:
        return {}

    total = df["pnl"].sum()
    win_rate = (df["pnl"] > 0).mean()
    avg = df["pnl"].mean()
    std = df["pnl"].std(ddof=0)
    sharpe = (avg / std * math.sqrt(252)) if std else float("nan")

    # equity curve based on trade close chronological order
    curve = df.sort_values("exit_dt")["pnl"].cumsum()
    peak = curve.expanding().max()
    drawdown = curve - peak
    max_dd = drawdown.min()

    return {
        "total_pnl": float(total),
        "trade_count": int(len(df)),
        "win_rate": float(win_rate),
        "avg_pnl": float(avg),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
    }


def pivot_heatmap(
    df: pd.DataFrame,
    x: str,
    y: str,
    value: str = "pnl",
    outfile: str | None = None,
    agg: str = "mean",
):
    """Create a heat-map (Seaborn) of *value* aggregated over X/Y parameters.

    *x*/*y* must be columns in *df* (so caller should flatten params beforehand).
    If *outfile* is given, the PNG is saved there; otherwise plt.show().
    """
    if sns is None or plt is None:
        raise RuntimeError("seaborn / matplotlib not available – install to use plotting helpers")

    # Pivot into matrix
    table = df.pivot_table(index=y, columns=x, values=value, aggfunc=agg)
    fig, ax = plt.subplots(figsize=(1 + 0.6 * table.shape[1], 1 + 0.6 * table.shape[0]))
    sns.heatmap(table, annot=True, fmt=".1f", cmap="RdYlGn", center=0, ax=ax)
    ax.set_title(f"{value} ({agg})")
    plt.tight_layout()
    if outfile:
        fig.savefig(outfile, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def equity_curve(df: pd.DataFrame, outfile: str | None = None):
    """Plot cumulative PnL over time (based on exit_dt chronological order)."""
    if sns is None or plt is None:
        raise RuntimeError("seaborn / matplotlib not available – install to use plotting helpers")

    track = df.sort_values("exit_dt")["pnl"].cumsum()
    fig, ax = plt.subplots(figsize=(8, 4))
    track.plot(ax=ax)
    ax.set_ylabel("Cumulative PnL")
    ax.set_xlabel("Trade # (chronological)")
    ax.set_title("Equity Curve")
    ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    if outfile:
        fig.savefig(outfile, dpi=150)
        plt.close(fig)
    else:
        plt.show()
