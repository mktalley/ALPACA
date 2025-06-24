"""Position sizing & risk management helpers for 0-DTE bot.

The class encapsulates the *hybrid* sizing model agreed with the user:

• Primary rule – risk **risk_pct** (default 2 %) of account *equity* per trade.
• Daily kill-switch once *realised P/L* crosses –daily_cap_pct (default –5 %).
• VIX sensitivity – if VIX > vix_high (**25**) size is scaled down *vix_scale_high*
  (default 0.5 → 50 %).  If VIX < vix_low (**14**) no boost is applied – we keep
  it simple.
• Strategy-specific fixed credit targets (``FIXED_PREMIUM_TARGET``).

Public API
==========
``PositionSizer`` is initialised once per session and reused by builders.

    sizer = PositionSizer(equity=50_000, daily_pl=-300, vix=19.2)
    qty = sizer.qty_for_condor(credit=1.1, width=5.0)

A thin ``qty_for_spread`` convenience covers verticals/strangles that have
fixed risk *width* on one side only.
"""

from __future__ import annotations

import math
import os
from enum import StrEnum
from typing import Final

__all__ = [
    "PositionSizer",
]


class StrategyTag(StrEnum):
    IRON_CONDOR = "IRON_CONDOR"
    STRADDLE = "STRADDLE"
    STRANGLE = "STRANGLE"
    CREDIT_VERTICAL = "CREDIT_VERTICAL"
    DEBIT_VERTICAL = "DEBIT_VERTICAL"


FIXED_PREMIUM_TARGET: Final[dict[StrategyTag, float]] = {
    StrategyTag.IRON_CONDOR: 100.0,
    StrategyTag.STRADDLE: 100.0,
}


class PositionSizer:
    """Compute contract quantity based on risk-budget parameters."""

    def __init__(
        self,
        *,
        equity: float,
        daily_realised_pl: float = 0.0,
        vix: float | None = None,
        risk_pct: float | None = None,
        daily_cap_pct: float | None = None,
        vix_high: float = 25.0,
        vix_low: float = 14.0,
        vix_scale_high: float = 0.5,
    ) -> None:
        env = os.getenv
        self.equity = equity
        self.daily_realised_pl = daily_realised_pl
        self.vix = vix
        self.risk_pct = risk_pct or float(env("ZDTE_RISK_PCT", 0.02))
        self.daily_cap_pct = daily_cap_pct or float(env("ZDTE_DAILY_CAP_PCT", 0.05))
        self.vix_high, self.vix_low, self.vix_scale_high = vix_high, vix_low, vix_scale_high

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def kill_switch(self) -> bool:
        """Return *True* if daily loss cap breached."""
        return self.daily_realised_pl <= -self.equity * self.daily_cap_pct

    # Condor / strangle risk is width − credit (defined-risk)
    def qty_for_condor(
        self,
        *,
        credit: float,
        width: float,
        strategy: StrategyTag = StrategyTag.IRON_CONDOR,
    ) -> int:
        if self.kill_switch():
            return 0
        risk_per_contract = max(width - credit, 0.01)
        return self._qty_by_risk(risk_per_contract, strategy, credit)

    # Single-sided credit vertical
    def qty_for_spread(
        self,
        *,
        credit: float,
        width: float,
        strategy: StrategyTag = StrategyTag.CREDIT_VERTICAL,
    ) -> int:
        return self.qty_for_condor(credit=credit, width=width, strategy=strategy)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _qty_by_risk(self, risk_per_contract: float, strategy: StrategyTag, credit: float) -> int:
        # 1) base risk budget
        available_risk = self.equity * self.risk_pct

        # 2) VIX scaling
        if self.vix is not None and self.vix > self.vix_high:
            available_risk *= self.vix_scale_high  # cut size by 50 %

        # 3) fixed-premium override (acts as *upper* bound)
        premium_target = FIXED_PREMIUM_TARGET.get(strategy)
        if premium_target is not None and credit > 0:
            contracts_premium_cap = math.floor(premium_target / credit)
        else:
            contracts_premium_cap = math.inf  # type: ignore[assignment]

        # 4) risk-based quantity
        contracts_risk = math.floor(available_risk / risk_per_contract)

        qty = min(contracts_risk, contracts_premium_cap)
        return max(qty, 1)
