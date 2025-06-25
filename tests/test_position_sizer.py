import pytest

from apps.zero_dte.position_sizer import PositionSizer, StrategyTag


def test_qty_condor_basic():
    sizer = PositionSizer(equity=50_000, risk_pct=0.02)
    # iron condor width 5, credit 1 -> risk per condor = 4
    qty = sizer.qty_for_condor(credit=1.0, width=5.0)
    # risk-based qty = 250, premium cap (100/1) = 100 ⇒ choose lower
    assert qty == 100


def test_qty_condor_min_cap():
    sizer = PositionSizer(equity=10_000, risk_pct=0.02)
    # credit nearly width, risk =1; but max premium cap 100 => qty <=10
    qty = sizer.qty_for_condor(credit=10.0, width=11.0)
    assert qty == 10


def test_kill_switch():
    sizer = PositionSizer(equity=20_000, daily_realised_pl=-2200)  # 11% loss
    assert sizer.kill_switch() is True
    qty = sizer.qty_for_condor(credit=1.0, width=5.0)
    assert qty == 0


def test_vix_scaling():
    sizer_high = PositionSizer(equity=30_000, vix=35.0, risk_pct=0.02)
    qty_high = sizer_high.qty_for_condor(credit=1.0, width=5.0)

    sizer_low = PositionSizer(equity=30_000, vix=15.0, risk_pct=0.02)
    qty_low = sizer_low.qty_for_condor(credit=1.0, width=5.0)

    assert qty_high < qty_low
