"""Hyper-parameter optimisation for 0-DTE strategies using Optuna.

Usage
-----
python -m apps.zero_dte.optimise \
    --underlying SPY \
    --start 2024-05-01 --end 2024-06-21 \
    --strategy strangle \
    --n-trials 50 \
    --out config/zero_dte.yaml

The script loads historical bars via simulator.run_backtest and searches for the
parameter set that maximises total PnL over the period.  Drawdown and negative
PnL trials are penalised, so the optimiser converges to robust combos.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Ensure .env variables are present for historical-data downloads
# ---------------------------------------------------------------------------
import os
import re
from pathlib import Path

_env_path = Path(__file__).resolve().parent.parent.parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        k, v = map(str.strip, _line.split("=", 1))
        os.environ.setdefault(k, v)


import argparse
import importlib
from datetime import date
from datetime import time as dt_time
from pathlib import Path

import optuna
import yaml

from .simulator import run_backtest

# Bounds for every numeric field in Settings (floats are inclusive ranges, ints are inclusive [low, high])
_DEFAULT_PARAM_SPACE = {
    # Risk / exits
    "STOP_LOSS_PCT": (0.05, 0.50),
    "PROFIT_TARGET_PCT": (0.10, 0.90),  # strangle only
    "CONDOR_TARGET_PCT": (0.10, 0.60),  # condor only
    "CONDOR_WING_SPREAD": (1, 4),  # int – narrower so strikes exist
    # Sizing / risk
    "RISK_PCT_PER_TRADE": (0.002, 0.01),
    "MAX_LOSS_PER_TRADE": (200, 1500),
    # Liquidity filters
    "MAX_SPREAD_PCT": (0.005, 0.05),
    "MIN_IV": (0.05, 0.60),
    # Event / scaling
    "EVENT_MOVE_PCT": (0.0005, 0.005),
    "QTY": (1, 20),  # int
}


def _d(s: str) -> date:
    return date.fromisoformat(s)


def _t(s: str) -> dt_time:
    hh, mm = map(int, s.split(":"))
    return dt_time(hh, mm)


def main():
    p = argparse.ArgumentParser("0-DTE hyper-parameter optimiser")
    p.add_argument("--underlying", default="SPY")
    p.add_argument("--start", type=_d, required=True)
    p.add_argument("--end", type=_d, required=True)
    p.add_argument("--strategy", choices=["strangle", "condor"], default="strangle")
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--entry", type=_t, default=dt_time(9, 35))
    p.add_argument("--cutoff", type=_t, default=dt_time(15, 45))
    p.add_argument(
        "--out", default="config/zero_dte.yaml", help="Path to write best-param YAML"
    )
    args = p.parse_args()

    strategy = args.strategy

    # Build search space dict tailored to strategy
    search_space = {}
    for k, v in _DEFAULT_PARAM_SPACE.items():
        if strategy == "strangle" and k.startswith("CONDOR"):
            continue
        if strategy == "condor" and k == "PROFIT_TARGET_PCT":
            continue
        search_space[k] = v

    # For anchor times we search minutes offset from market open
    MARKET_OPEN = 9 * 60 + 30  # 9:30 ET
    MARKET_CLOSE = 16 * 60  # 16:00 ET

    def _minutes_to_time(minutes: int) -> dt_time:
        h, m = divmod(minutes, 60)
        return dt_time(h, m)

    def objective(trial: optuna.Trial):
        params = {}
        for name, rng in search_space.items():
            if isinstance(rng[0], int):
                params[name] = trial.suggest_int(name, rng[0], rng[1])
            else:
                params[name] = trial.suggest_float(name, rng[0], rng[1])
        results = run_backtest(
            symbol=args.underlying,
            start=args.start,
            end=args.end,
            grid_params=[params],  # run_backtest expects list
            entry_time=args.entry,
            exit_cutoff=args.cutoff,
            strategy=strategy,
        )
        if not results:
            # No trades: heavy penalty but still return a score so trial finishes
            return -1_000.0

        # --- Risk-adjusted scoring -----------------------------------------
        from collections import defaultdict

        import numpy as np

        daily_pnl: dict[str, float] = defaultdict(float)
        for r in results:
            day = r.exit_dt.date().isoformat()
            daily_pnl[day] += r.pnl
        pnl_series = np.array(list(daily_pnl.values()), dtype=float)
        total_pnl = pnl_series.sum()
        sharpe = 0.0
        if len(pnl_series) > 1 and pnl_series.std() > 1e-6:
            sharpe = pnl_series.mean() / pnl_series.std()

        # equity curve for drawdown
        cum = np.cumsum(pnl_series)
        peak = np.maximum.accumulate(cum)
        drawdowns = peak - cum
        max_dd = drawdowns.max(initial=0.0)

        # Objective: maximise pnl with Sharpe bonus and drawdown penalty
        score = total_pnl + sharpe * 500 - max_dd * 0.5
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.n_trials)

    best = study.best_params
    print("Best params:", best)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        yaml.safe_dump(best, fh)
    print("Wrote best params to", out_path)


if __name__ == "__main__":
    main()
