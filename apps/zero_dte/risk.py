"""Risk & position-sizing utilities for 0-DTE bot.

Phase-2 implements a minimal helper ``calc_position_size`` used by builders
so we can unit-test sizing without pulling live account snapshots.
"""

from __future__ import annotations

import math

__all__ = ["calc_position_size"]


def calc_position_size(
    account_equity: float,
    max_risk_per_trade: float,
    *,
    credit: float,
    width: float,
) -> int:
    """Return contract quantity constrained by *max_risk_per_trade*.

    Iron-condor risk is ``width − credit`` per contract.  We round *down* to the
    nearest integer >=1.  A safeguard ensures we always return at least 1 so
    back-tests don’t accidentally skip trades due to float precision.
    """

    risk_per_contract = max(width - credit, 0.01)
    max_contracts = math.floor(max_risk_per_trade / risk_per_contract)
    return max(max_contracts, 1)