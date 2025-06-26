import pandas as pd
import pytest

from apps.zero_dte.backtest.bot import BacktestBot


@pytest.mark.parametrize(
    "specs, expected",
    [
        (["A=1,2,3"], {"A": [1, 2, 3]}),
        (["X=foo,bar"], {"X": ["foo", "bar"]}),
        (["P=1.5,2"], {"P": [1.5, 2.0]}),
        (["A=1", "B=x,y"], {"A": [1], "B": ["x", "y"]}),
    ],
)
def test_parse_grid(specs, expected):
    from apps.zero_dte.backtest.bot import BacktestBot as _B

    assert _B._parse_grid(specs) == expected


def test_run_single_synthetic():
    bot = BacktestBot("XYZ")
    start = pd.Timestamp("2025-01-03")
    end = start + pd.Timedelta(days=1)
    df = bot.run_single(start, end, STOP_LOSS_PCT=0.2, PROFIT_TARGET_PCT=0.5)

    assert len(df) == 1
    assert "pnl" in df.columns
    assert df.iloc[0].trades == 1
