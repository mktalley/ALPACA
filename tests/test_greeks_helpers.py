"""Unit tests for _load_greeks_df and _pick_strikes_by_delta."""

from datetime import date

import pandas as pd

from apps.zero_dte import simulator as sim


def test_load_greeks_df_returns_df(tmp_path, monkeypatch):
    ul = "SPY"
    exp = date(2025, 6, 20)
    root = tmp_path / "cache" / "greeks" / ul
    root.mkdir(parents=True)
    df = pd.DataFrame({"symbol": ["SPY250620C00450000"], "delta": [0.25]})
    path = root / f"{exp:%Y%m%d}.parquet"
    # Without pyarrow dependency: store via pickle and monkeypatch pandas.read_parquet
    df.to_pickle(path)
    monkeypatch.setattr(sim.pd, "read_parquet", lambda p: pd.read_pickle(p))

    monkeypatch.setattr(sim._data, "_CACHE_ROOT", tmp_path / "cache")

    out = sim._load_greeks_df(ul, exp)
    assert not out.empty
    assert list(out.columns) == ["symbol", "delta"]


def test_pick_strikes_fallback(tmp_path, monkeypatch):
    """If no greeks file exists return strangle strikes via spot."""
    ul = "SPY"
    exp = date(2025, 6, 20)
    spot_price = 500.5

    # stub get_stock_bars to return dummy df with open price
    def _dummy_stock_bars(sym, d):
        return pd.DataFrame(
            {"open": [spot_price]}, index=[pd.Timestamp("2025-06-20T14:30Z")]
        )

    monkeypatch.setattr(sim._data, "_CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(sim._data, "get_stock_bars", _dummy_stock_bars)

    call_sym, put_sym = sim._pick_strikes_by_delta(ul, exp, 0.1)
    assert call_sym.startswith(ul)
    assert put_sym.startswith(ul)
    import math

    expected_call = "C" + str((math.ceil(spot_price) + 1) * 100).zfill(8)
    expected_put = "P" + str((math.floor(spot_price) - 1) * 100).zfill(8)
    assert call_sym.endswith(expected_call)
    assert put_sym.endswith(expected_put)
