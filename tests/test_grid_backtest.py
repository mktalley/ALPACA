from datetime import date, time

from apps.zero_dte.grid_backtest import expand_grid, run_grid


def test_expand_grid_basic():
    grid = {
        "a": [1, 2],
        "b": ["x"],
    }
    combos = list(expand_grid(grid))
    assert combos == [
        {"a": 1, "b": "x"},
        {"a": 2, "b": "x"},
    ]


def test_run_grid_smoke():
    # Very short 2-day synthetic run; relies on simulator's offline mode.
    trades = run_grid(
        symbol="SPY",
        start=date(2025, 1, 2),
        end=date(2025, 1, 3),
        param_grid={},
        entry_time=time(9, 45),
        exit_cutoff=time(15, 30),
    )
    # Two trading days × 1 combo → 2 trades expected
    assert len(trades) == 2
    assert all(tr.grid_id >= 1 for tr in trades)
