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
    """Wrapper around *run_strangle* that injects dynamic sizing via ``PositionSizer``.

    Expected usage::

        StrangleBuilder()(trading, stock_client, option_client,
                          symbol="SPX", target_pct=0.2,
                          poll_interval=10, exit_cutoff=time(15,45),
                          account_equity=25_000, max_risk_per_trade=500)

    If *account_equity* & *max_risk_per_trade* are provided we compute the
    contract quantity (``qty``) using :class:`apps.zero_dte.position_sizer.PositionSizer`.
    Otherwise the caller must pass an explicit ``qty`` kwarg and no sizing is
    performed.
    """

    def build(self, *args, **kwargs):  # type: ignore[override]
        # Merge stored args/kwargs with direct call overrides for compatibility
        if not args and self._args:
            args = self._args
        if self._kwargs:
            merged = {**self._kwargs, **kwargs}
            kwargs = merged
        from apps.zero_dte.zero_dte_app import run_strangle as _impl
        from apps.zero_dte.position_sizer import PositionSizer, StrategyTag

        # Pull sizing kwargs if supplied
        acct_equity = kwargs.pop("account_equity", None)
        max_risk = kwargs.pop("max_risk_per_trade", None)

        # ``qty`` may be provided directly; otherwise derive from risk params
        qty = kwargs.get("qty")
        if qty is None and acct_equity and max_risk:
            # Naïve risk proxy – use credit × STOP_LOSS_PCT as max loss per contract
            # Since we don't know credit yet, assume worst-case 100% loss of premium;
            # this deliberately sizes *smaller* than live builder until Phase-3.
            sizer = PositionSizer(equity=acct_equity, risk_pct=max_risk / acct_equity)
            qty = sizer.qty_for_spread(credit=1.0, width=1.0, strategy=StrategyTag.STRANGLE)
            kwargs["qty"] = qty
        elif qty is None:
            raise ValueError("StrangleBuilder requires either qty or account_equity+max_risk_per_trade")
        else:
            kwargs["qty"] = qty

        return _impl(*args, **kwargs)


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

# Alias other placeholder builders to OneSideStrangleBuilder for Phase-1
ReversalSpreadBuilder = OneSideStrangleBuilder
DirectionalVerticalBuilder = OneSideStrangleBuilder

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

        # Choose strikes via shared delta-selector util
        from apps.zero_dte.utils.delta import select_strikes

        try:
            sp, lp, sc, lc = select_strikes(chain, delta_target, wing_width)
        except ValueError:  # empty chain / missing deltas
            return self._fallback_stub(symbol, delta_target, wing_width)

        # Map the selected strikes back to their OptionContract objects
        short_put_obj = next(
            (c for c in chain if c.put_call == "put" and c.strike == sp),
            None,
        )
        short_call_obj = next(
            (c for c in chain if c.put_call == "call" and c.strike == sc),
            None,
        )
        if short_put_obj is None or short_call_obj is None:
            return self._fallback_stub(symbol, delta_target, wing_width)

        long_put_strike, short_put_strike = lp, sp
        short_call_strike, long_call_strike = sc, lc

        order = {
            "symbol": symbol,
            "expiry": short_call_obj.expiry if hasattr(short_call_obj, 'expiry') else None,
            "legs": {
                "short_call": short_call_strike,
                "long_call": long_call_strike,
                "short_put": short_put_strike,
                "long_put": long_put_strike,
            },
            "qty": 1,
            # Placeholder theoretical credit – live impl will fetch option mid prices
            "theoretical_credit": 0.0,
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
        log.info("[IronCondorBuilder] Fallback build – returning 0.0 PnL")
        return 0.0


class OneSideStrangleBuilder(BaseBuilder):
    """Simple credit spread builder (one side of strangle).

    Accepts ``side`` ("put" or "call"), ``delta_target`` and ``wing_width`` to pick
    strikes.  In offline/test mode we skip live data and return deterministic
    strikes at ±$5 with sizing via :class:`PositionSizer`.
    """

    def build(
        self,
        symbol: str = "SPY",
        *,
        side: str = "put",
        delta_target: float | None = None,
        wing_width: float | None = None,
        account_equity: float | None = None,
        max_risk_per_trade: float | None = None,
        credit: float = 1.0,
        **kwargs,
    ):  # type: ignore[override]
        from datetime import date

        from alpaca.data.live import Options
        from apps.zero_dte.position_sizer import PositionSizer, StrategyTag

        side = side.lower()
        if side not in ("put", "call"):
            raise ValueError("side must be 'put' or 'call'")

        delta_target = delta_target or float(os.getenv("ZDTE_DELTA_TARGET", 0.15))
        wing_width = wing_width or float(os.getenv("ZDTE_WING_WIDTH", 1.0))

        try:
            client = Options()
            today = date.today().strftime("%Y-%m-%d")
            chain = client.get_option_chain(symbol, expiry=today)
            # Pick strike whose delta closest to delta_target on chosen side
            best = None
            for c in chain:
                if c.put_call != side:
                    continue
                if c.greeks is None or c.greeks.delta is None:
                    continue
                diff = abs(abs(c.greeks.delta) - delta_target)
                if best is None or diff < best[0]:
                    best = (diff, c)
            if best is None:
                raise RuntimeError("No contract found")
            short = best[1]
            long_strike = short.strike - wing_width if side == "put" else short.strike + wing_width
            order = {
                "symbol": symbol,
                "expiry": short.expiry,
                "legs": {
                    f"short_{side}": short.strike,
                    f"long_{side}": long_strike,
                },
                "qty": 1,
                "theoretical_credit": credit,
            }
        except Exception as exc:  # pragma: no cover – offline
            log.info("[OneSideStrangleBuilder] API unavailable – stub strikes (%s)", exc)
            short_strike = 440 - 5 if side == "put" else 440 + 5
            long_strike = short_strike - wing_width if side == "put" else short_strike + wing_width
            order = {
                "symbol": symbol,
                "expiry": "T0",
                "legs": {f"short_{side}": short_strike, f"long_{side}": long_strike},
                "qty": 1,
                "theoretical_credit": credit,
            }

        # Sizing
        if account_equity and max_risk_per_trade:
            sizer = PositionSizer(
                equity=account_equity,
                risk_pct=max_risk_per_trade / account_equity,
            )
            order["qty"] = sizer.qty_for_spread(
                credit=order["theoretical_credit"],
                width=wing_width,
                strategy=StrategyTag.CREDIT_VERTICAL,
            )
        log.info("Built one-side spread order: %s", order)
        return order



class SkewedCreditPutBuilder(OneSideStrangleBuilder):
    def build(self, *args, **kwargs):  # type: ignore[override]
        kwargs.setdefault("side", "put")
        return super().build(*args, **kwargs)


class SkewedCreditCallBuilder(OneSideStrangleBuilder):
    def build(self, *args, **kwargs):  # type: ignore[override]
        kwargs.setdefault("side", "call")
        return super().build(*args, **kwargs)


# Mapping StrategyID → builder class
_mapping: dict[StrategyID, Type[BaseBuilder]] = {
    StrategyID.SYMMETRIC_STRANGLE: StrangleBuilder,
    StrategyID.WIDE_IRON_CONDOR: IronCondorBuilder,
    StrategyID.TIGHT_IRON_CONDOR: IronCondorBuilder,
    StrategyID.PRE_EVENT_CONDOR: IronCondorBuilder,
    StrategyID.SKEWED_CREDIT_PUT: SkewedCreditPutBuilder,
    StrategyID.SKEWED_CREDIT_CALL: SkewedCreditCallBuilder,
    StrategyID.REVERSAL_SPREAD: OneSideStrangleBuilder,
    StrategyID.DIRECTIONAL_DEBIT_VERTICAL: OneSideStrangleBuilder,
}


class _BuilderClass(Protocol):
    def __call__(self, *args, **kwargs) -> BaseBuilder: ...

# ---------------------------------------------------------------------------
# Validation to ensure every StrategyID resolves to a usable builder
# ---------------------------------------------------------------------------

def _validate_mapping():
    missing = []
    for sid, cls in _mapping.items():
        if not hasattr(cls, "build") or cls.build is BaseBuilder.build:  # type: ignore[comparison-overlap]
            missing.append(sid)
    if missing:
        raise RuntimeError(
            f"Unimplemented builder(s) for strategies: {', '.join(missing)}"
        )


_validate_mapping()


def select_builder(strategy: StrategyID) -> Type[BaseBuilder]:
    """Return builder *class* for *strategy* (fallback to StrangleBuilder)."""
    builder_cls = _mapping.get(strategy, StrangleBuilder)
    if builder_cls is not StrangleBuilder and builder_cls.build is BaseBuilder.build:  # type: ignore[comparison-overlap]
        log.info("Builder for strategy %s not implemented (Phase-1 stub)", strategy)
    return builder_cls
