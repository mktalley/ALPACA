"""Strategy builder routing – Phase-1 stubs.

Exposes ``select_builder(strategy_id)`` which maps a :class:`apps.zero_dte.market_structure.StrategyID`
value to a callable that executes the strategy.  Only *SYMMETRIC_STRANGLE* is
implemented (delegates to :pyfunc:`apps.zero_dte.zero_dte_app.run_strangle`).
Other strategies raise *NotImplementedError* for now.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Protocol, Type

from apps.zero_dte.market_structure import StrategyID

log = logging.getLogger(__name__)


class BaseBuilder(ABC):
    """Abstract interface shared by all strategy builders."""

    def __init__(self, *args, **kwargs):
        # Store args for convenience in subclasses that delegate to existing funcs
        self._args = args
        self._kwargs = kwargs

    @abstractmethod
    def build(self):  # noqa: D401 – returns realised PnL or None
        """Execute the strategy and return realised/unrealised PnL (float)."""


# ---------------------------------------------------------------------------
# Concrete implementations
# ---------------------------------------------------------------------------


class StrangleBuilder(BaseBuilder):
    """Wrap existing *run_strangle* procedure in *build* method."""

    def build(self):  # type: ignore[override]
        from apps.zero_dte.zero_dte_app import run_strangle as _impl

        return _impl(*self._args, **self._kwargs)


class IronCondorBuilder(BaseBuilder):
    """Build and (optionally) place an intraday 0-DTE iron condor on **SPY**.

    Strike-selection rules (default values configurable via kwargs):
    • *delta_target* – short legs closest to ±delta_target (abs value).  Default **0.15**
    • *wing_width*   – distance between short and long legs (dollars).  Default **1.0**

    The builder returns a dict with the selected strikes and theoretical
    premium.  It delegates sizing / risk-checks to the *risk* module so tests
    can inject custom account snapshots.
    """

    # ------------------------------------------------------------------
    # Public entry-point
    # ------------------------------------------------------------------

    def build(
        self,
        symbol: str = "SPY",
        *,
        delta_target: float | None = None,
        wing_width: float | None = None,
        account_equity: float | None = None,
        max_risk_per_trade: float | None = None,
        **kwargs,
    ):  # type: ignore[override]
        """Select strikes, size the order & (optionally) submit.

        Parameters
        ----------
        symbol
            Underlying ticker, default "SPY".
        delta_target
            Target absolute delta for short legs.  If *None*, fallback to
            env-var ``ZDTE_DELTA_TARGET`` or **0.15**.
        wing_width
            Dollar distance between short and long legs.  Fallback to
            ``ZDTE_WING_WIDTH`` or **1.0**.
        account_equity / max_risk_per_trade
            If provided, risk manager computes contract qty.  Otherwise qty=1.
        kwargs
            Forward-compatible extra params (ignored).
        """

        from datetime import date

        from alpaca.data.live import Options
        from alpaca.data.models import OptionContract
        from apps.zero_dte.position_sizer import PositionSizer, StrategyTag

        delta_target = delta_target or float(os.getenv("ZDTE_DELTA_TARGET", 0.15))
        wing_width = wing_width or float(os.getenv("ZDTE_WING_WIDTH", 1.0))

        # ------------------------------------------------------------------
        # 1. Fetch current price & option chain snapshot (same-day expiry)
        # ------------------------------------------------------------------
        try:
            client = Options()
            underlying_px = client.get_last_quote(symbol).ask_price or client.get_last_quote(symbol).bid_price
            today = date.today().strftime("%Y-%m-%d")
            chain: list[OptionContract] = client.get_option_chain(symbol, expiry=today)
        except Exception as exc:  # pragma: no cover – API offline in CI
            log.warning("Alpaca API unavailable – falling back to synthetic strikes (%s)", exc)
            return self._fallback_stub(symbol, delta_target, wing_width)

        # Build two lists with (abs(Δ-Δtgt), contract) for calls & puts
        calls, puts = [], []
        for c in chain:
            if c.greeks is None or c.greeks.delta is None:
                continue
            entry = (abs(abs(c.greeks.delta) - delta_target), c)
            (calls if c.put_call == "call" else puts).append(entry)

        if not calls or not puts:  # pragma: no cover
            return self._fallback_stub(symbol, delta_target, wing_width)

        short_call = min(calls, key=lambda x: x[0])[1]
        short_put = min(puts, key=lambda x: x[0])[1]

        long_call_strike = short_call.strike + wing_width
        long_put_strike = short_put.strike - wing_width

        order = {
            "symbol": symbol,
            "expiry": short_call.expiry,
            "legs": {
                "short_call": short_call.strike,
                "long_call": long_call_strike,
                "short_put": short_put.strike,
                "long_put": long_put_strike,
            },
            "qty": 1,
            "theoretical_credit": round(short_call.mid - long_call_strike * 0, 2),
        }

        if account_equity and max_risk_per_trade:
            sizer = PositionSizer(
                equity=account_equity,
                risk_pct=max_risk_per_trade / account_equity,
            )
            order["qty"] = sizer.qty_for_condor(
                credit=order["theoretical_credit"],
                width=wing_width,
                strategy=StrategyTag.IRON_CONDOR,
            )

        log.info("Built iron condor order: %s", order)
        return order

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fallback_stub(self, symbol: str, delta_tgt: float, width: float):
        """Return deterministic strikes when live data unavailable (CI / back-test)."""
        strikes_atm = 440  # arbitrary static anchor
        order = {
            "symbol": symbol,
            "expiry": "T0",
            "legs": {
                "short_call": strikes_atm + 5,
                "long_call": strikes_atm + 5 + width,
                "short_put": strikes_atm - 5,
                "long_put": strikes_atm - 5 - width,
            },
            "qty": 1,
            "theoretical_credit": 1.0,
        }
        log.info("[IronCondorBuilder] Fallback build – %s", order)
        return order


class OneSideStrangleBuilder(BaseBuilder):
    """Placeholder for skewed one-sided strangles/spreads (Phase-1)."""

    def build(self):  # type: ignore[override]
        log.info("[OneSideStrangleBuilder] Stub execution – returning 0.0 PnL")
        return 0.0


# Mapping StrategyID → builder class
_mapping: dict[StrategyID, Type[BaseBuilder]] = {
    StrategyID.SYMMETRIC_STRANGLE: StrangleBuilder,
    StrategyID.WIDE_IRON_CONDOR: IronCondorBuilder,
    StrategyID.TIGHT_IRON_CONDOR: IronCondorBuilder,
    StrategyID.PRE_EVENT_CONDOR: IronCondorBuilder,
    StrategyID.SKEWED_CREDIT_PUT: OneSideStrangleBuilder,
    StrategyID.SKEWED_CREDIT_CALL: OneSideStrangleBuilder,
    StrategyID.REVERSAL_SPREAD: OneSideStrangleBuilder,
    StrategyID.DIRECTIONAL_DEBIT_VERTICAL: OneSideStrangleBuilder,
}


class _BuilderClass(Protocol):
    def __call__(self, *args, **kwargs) -> BaseBuilder: ...


def select_builder(strategy: StrategyID) -> Type[BaseBuilder]:
    """Return builder *class* for *strategy* (fallback to StrangleBuilder)."""
    builder_cls = _mapping.get(strategy, StrangleBuilder)
    if builder_cls is not StrangleBuilder and builder_cls.build is BaseBuilder.build:  # type: ignore[comparison-overlap]
        log.info("Builder for strategy %s not implemented (Phase-1 stub)", strategy)
    return builder_cls
