import pandas as pd
from datetime import date, timedelta

from apps.zero_dte.market_structure import classify_day, DayType


def _make_bars(prices, day):
    idx = pd.date_range(start=f"{day} 09:30", periods=len(prices), freq="T", tz="US/Eastern")
    return pd.DataFrame(
        {
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "volume": [100] * len(prices),
        },
        index=idx,
    )


def _patch(monkeypatch, today_df, yday_df):
    def fake_get_stock_bars(symbol, bar_date, timeframe=None):
        if bar_date == today_df.index[0].date():
            return today_df
        return yday_df

    monkeypatch.setattr(
        "apps.zero_dte.market_structure.get_stock_bars", fake_get_stock_bars, raising=True
    )


def test_gap_go(monkeypatch):
    today = date(2025, 6, 23)
    yday = today - timedelta(days=1)
    bars_today = _make_bars([110] * 30, today)
    bars_yday = _make_bars([100] * 30, yday)

    _patch(monkeypatch, bars_today, bars_yday)

    dt = classify_day("TEST", stock_client=None, cfg={"today": today})
    assert dt == DayType.GAP_GO


def test_gap_trap(monkeypatch):
    today = date(2025, 6, 24)
    yday = today - timedelta(days=1)
    # price touches 100 (fills gap)
    bars_today = _make_bars([110, 109, 100, 102, 104] + [105] * 25, today)
    bars_yday = _make_bars([100] * 30, yday)

    _patch(monkeypatch, bars_today, bars_yday)

    dt = classify_day("TEST", stock_client=None, cfg={"today": today})
    assert dt == DayType.GAP_TRAP


def test_trend_up(monkeypatch):
    today = date(2025, 6, 25)
    yday = today - timedelta(days=1)
    # slight up-gap under threshold
    bars_today = _make_bars(list(range(100, 131)), today)  # +30 bucks over 30 min
    bars_yday = _make_bars([99] * 30, yday)

    _patch(monkeypatch, bars_today, bars_yday)

    dt = classify_day("TEST", stock_client=None, cfg={"gap_pct_threshold": 2.0, "today": today})
    assert dt == DayType.TREND_UP


def test_inside(monkeypatch):
    today = date(2025, 6, 26)
    yday = today - timedelta(days=1)
    bars_today = _make_bars([100] * 30, today)  # flat inside
    bars_yday = _make_bars([110] * 10 + [90] * 20, yday)

    _patch(monkeypatch, bars_today, bars_yday)

    dt = classify_day("TEST", stock_client=None, cfg={"today": today})
    assert dt == DayType.INSIDE
