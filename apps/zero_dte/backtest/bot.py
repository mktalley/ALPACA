"""Orchestration helper for 0-DTE back-tests.

The *bot* delegates the heavy-lifting to existing helpers:

* :pyfunc:`apps.zero_dte.backtest.grid.run_grid` for exhaustive parameter
  search.
* :pyclass:`apps.zero_dte.backtest.StrangleSimulator` for the path-wise PNL
  of a single strangle.

It adds a paper-thin CLI so researchers can launch experiments straight from
 the command line or Jupyter while keeping the public API ridiculously small.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd

from .grid import run_grid as _run_grid

__all__ = [
    "BacktestBot",
]


class BacktestBot:
    """High-level convenience wrapper around :pyfunc:`grid.run_grid`."""

    def __init__(self, symbol: str) -> None:  # noqa: D401
        self.symbol = symbol

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run_single(
        self,
        start: datetime,
        end: datetime,
        **params: Any,
    ) -> pd.DataFrame:
        """Run exactly one parameter combination and return a 1-row frame."""
        param_dict: Dict[str, List[Any]] = {k: [v] for k, v in params.items()}
        return _run_grid(self.symbol, start, end, param_dict)

    def run_grid(
        self,
        start: datetime,
        end: datetime,
        grid: Dict[str, List[Any]],
        *,
        n_jobs: int = 1,
    ) -> pd.DataFrame:
        """Exhaustive grid search. Delegates to :pyfunc:`grid.run_grid`."""
        return _run_grid(self.symbol, start, end, grid, n_jobs=n_jobs)

    # ------------------------------------------------------------------
    # CLI helper – intentionally tiny to avoid external deps
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_grid(specs: list[str]) -> Dict[str, List[Any]]:
        grid: Dict[str, List[Any]] = {}
        for raw in specs:
            key, _, val_str = raw.partition("=")
            if not _:
                raise SystemExit(f"Invalid --grid entry '{raw}'. Use NAME=v1,v2 …")
            vals: list[str] = val_str.split(",")
            # Heuristic: try int → float → str
            parsed: list[Any] = []
            for v in vals:
                try:
                    parsed.append(int(v))
                    continue
                except ValueError:
                    pass
                try:
                    parsed.append(float(v))
                except ValueError:
                    parsed.append(v)
            grid[key] = parsed
        return grid


def _main(argv: list[str] | None = None) -> None:  # pragma: no cover
    p = argparse.ArgumentParser(description="0-DTE back-test grid runner")
    p.add_argument("--symbol", required=True)
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--slippage", type=float, default=0.0, help="$ per contract side")
    p.add_argument("--commission", type=float, default=0.0, help="$ per contract side")
    p.add_argument("--jobs", type=int, default=1, help="parallel workers")
    p.add_argument(
        "--grid",
        action="append",
        metavar="KEY=V1,V2,…",
        required=True,
        help="repeatable – parameter search space",
    )
    ns = p.parse_args(argv)

    start_ts = pd.to_datetime(ns.start)
    end_ts = pd.to_datetime(ns.end)
    grid = BacktestBot._parse_grid(ns.grid)
    grid.setdefault("SLIPPAGE", [ns.slippage])
    grid.setdefault("COMMISSION", [ns.commission])

    bot = BacktestBot(ns.symbol)
    df = bot.run_grid(start_ts, end_ts, grid, n_jobs=ns.jobs)
    # Pretty-print top 10 rows by pnl desc
    pd.options.display.width = 0  # noqa: RUF100
    print(df.sort_values("pnl", ascending=False).head(10))


if __name__ == "__main__":  # pragma: no cover
    _main()
