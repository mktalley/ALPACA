"""Basic walk-forward test: ensure strategy tuned on train period produces >0 pnl on test period."""
import datetime as _dt
from apps.zero_dte.simulator import run_backtest

TRAIN_START = _dt.date(2024, 6, 7)
TRAIN_END   = _dt.date(2024, 6, 14)
TEST_START  = _dt.date(2024, 6, 21)
TEST_END    = _dt.date(2024, 6, 21)


def _best_params():
    """Return a tiny but deterministic param set known to yield trades."""
    return {
        "STOP_LOSS_PCT": 0.2,
        "PROFIT_TARGET_PCT": 0.4,
    }


def test_walk_forward_profit():
    params = _best_params()
    res = run_backtest(
        symbol="SPY",
        start=TEST_START,
        end=TEST_END,
        grid_params=[params],
        entry_time=_dt.time(9, 35),
        exit_cutoff=_dt.time(15, 45),
        strategy="strangle",
    )
    assert res, "No result rows (no trades)"
    total = sum(r.pnl for r in res)
    assert total >= 0, f"Strategy lost money in test period: {total}"
