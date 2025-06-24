#!/usr/bin/env python3
"""
Smoke test for equity-based drawdown/profit triggers in zero_dte_app.

Usage:
  python tools/scripts/smoke_equity_triggers.py \
    --current <current_equity> [--drawdown <pct>] [--profit <pct>] [--start <start_equity>]

Example:
  # Simulate a 12% drawdown
  python tools/scripts/smoke_equity_triggers.py --current 88 --drawdown 0.10

"""
import argparse
from threading import Event

from apps.zero_dte.zero_dte_app import Settings


def main():
    parser = argparse.ArgumentParser(
        description="Smoke test for equity-based drawdown and profit triggers"
    )
    parser.add_argument(
        "--drawdown",
        type=float,
        help="Override DAILY_DRAWDOWN_PCT (e.g. 0.1 for 10%)",
    )
    parser.add_argument(
        "--profit",
        type=float,
        help="Override DAILY_PROFIT_PCT (e.g. 0.2 for 20%)",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=100.0,
        help="Simulated starting equity (default 100.0)",
    )
    parser.add_argument(
        "--current",
        type=float,
        required=True,
        help="Simulated current equity",
    )
    args = parser.parse_args()

    # Load settings with overrides
    env_overrides = {}
    if args.drawdown is not None:
        env_overrides["DAILY_DRAWDOWN_PCT"] = args.drawdown
    if args.profit is not None:
        env_overrides["DAILY_PROFIT_PCT"] = args.profit

    cfg = Settings(**env_overrides)
    shutdown_event = Event()

    start_equity = args.start
    current_equity = args.current

    drawdown_pct = (
        (start_equity - current_equity) / start_equity if start_equity > 0 else 0
    )
    profit_pct = (
        (current_equity - start_equity) / start_equity if start_equity > 0 else 0
    )

    print(f"Start equity: {start_equity:.2f}")
    print(f"Current equity: {current_equity:.2f}")
    print(
        f"Drawdown: {drawdown_pct*100:.2f}% (threshold {cfg.DAILY_DRAWDOWN_PCT*100:.2f}%)"
    )
    print(f"Profit: {profit_pct*100:.2f}% (threshold {cfg.DAILY_PROFIT_PCT*100:.2f}%)")

    if cfg.DAILY_DRAWDOWN_PCT > 0 and drawdown_pct >= cfg.DAILY_DRAWDOWN_PCT:
        print("=> Drawdown trigger WOULD fire (liquidate & shutdown).")
    else:
        print("=> Drawdown trigger would NOT fire.")

    if cfg.DAILY_PROFIT_PCT > 0 and profit_pct >= cfg.DAILY_PROFIT_PCT:
        print("=> Profit trigger WOULD fire (liquidate & shutdown).")
    else:
        print("=> Profit trigger would NOT fire.")


if __name__ == "__main__":
    main()
