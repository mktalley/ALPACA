from datetime import datetime
import pytest

import pandas as pd
from apps.zero_dte.backtest.simulator import StrangleSimulator


def test_fee_math_flat_day():
    # synthetic bars: price flat so mid stays constant
    idx = pd.date_range("2024-01-02 09:30", periods=60, freq="min", tz="US/Eastern")
    bars = pd.DataFrame({"close": 480.0}, index=idx)

    sim = StrangleSimulator(
        bars,
        entry_time=idx[0],
        entry_credit=2.0,  # $2.00 credit
        qty=1,
        stop_loss_pct=0.5,  # unreachable
        profit_target_pct=0.5,  # unreachable
        slippage=0.05,
        commission=0.65,
        legs=2,
    )

    res = sim.run()
    # Exit at cutoff, mid unchanged => exit_credit=2.0
    # open_credit net = 2 - 0.05*2 - 0.65*2 = 0.6
    # close_cost  net = 2 + 0.05*2 + 0.65*2 = 4.0
    # pnl = (0.6 - 3.4)*100 = -280
    assert res.pnl == pytest.approx(-280.0)
