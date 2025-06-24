"""Market-structure / day-type classification & strategy routing (Phase-1).

This module is intentionally lightweight: it introduces only *types* and a
stub `classify_day()` implementation so that other parts of the bot can start
wiring against the public interface while the statistical logic is filled in
later iterations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from datetime import time as dt_time
from datetime import timedelta
from enum import StrEnum
from typing import Mapping

import pandas as pd

from alpaca.data.historical import OptionHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.timeframe import TimeFrame
from apps.zero_dte.config import load_daytype_config

from .data import get_stock_bars

log = logging.getLogger(__name__)

import os

# ---------------------------------------------------------------------------
# Public enums (importable by the rest of the bot)
# ---------------------------------------------------------------------------


class DayType(StrEnum):
    """High-level intraday market regimes recognised by the bot."""

    GAP_GO = "GAP_GO"  # gaps and continues in same direction
    GAP_TRAP = "GAP_TRAP"  # gaps then fully retraces / reverses
    TREND_UP = "TREND_UP"  # steady intraday up-trend
    TREND_DOWN = "TREND_DOWN"  # steady intraday down-trend
    INSIDE = "INSIDE"  # session stays within yesterday's range
    RANGE = "RANGE"  # whipsaw / low-trend day inside a wide range
    EVENT_RISK_HIGH_IV_CRUSH = "EVENT_RISK_HIGH_IV_CRUSH"  # macro event then vol crush
    UNCLASSIFIED = "UNCLASSIFIED"  # fallback / unknown


class StrategyID(StrEnum):
    """Abstract identifiers understood by trade-builder layer."""

    SKEWED_CREDIT_PUT = "SKEWED_CREDIT_PUT"
    SKEWED_CREDIT_CALL = "SKEWED_CREDIT_CALL"
    REVERSAL_SPREAD = "REVERSAL_SPREAD"
    DIRECTIONAL_DEBIT_VERTICAL = "DIRECTIONAL_DEBIT_VERTICAL"
    TIGHT_IRON_CONDOR = "TIGHT_IRON_CONDOR"
    WIDE_IRON_CONDOR = "WIDE_IRON_CONDOR"
    PRE_EVENT_CONDOR = "PRE_EVENT_CONDOR"
    SYMMETRIC_STRANGLE = "SYMMETRIC_STRANGLE"


# ---------------------------------------------------------------------------
# Strategy selector – configurable mapping DayType → StrategyID
# ---------------------------------------------------------------------------


_DEFAULT_MAP: Mapping[DayType, StrategyID] = {
    DayType.GAP_GO: StrategyID.SKEWED_CREDIT_PUT,  # or CALL if bearish – builder decides
    DayType.GAP_TRAP: StrategyID.REVERSAL_SPREAD,
    DayType.TREND_UP: StrategyID.DIRECTIONAL_DEBIT_VERTICAL,
    DayType.TREND_DOWN: StrategyID.DIRECTIONAL_DEBIT_VERTICAL,
    DayType.INSIDE: StrategyID.TIGHT_IRON_CONDOR,
    DayType.RANGE: StrategyID.WIDE_IRON_CONDOR,
    DayType.EVENT_RISK_HIGH_IV_CRUSH: StrategyID.PRE_EVENT_CONDOR,
    DayType.UNCLASSIFIED: StrategyID.SYMMETRIC_STRANGLE,
}


@dataclass(slots=True)
class StrategySelector:
    """Return the *StrategyID* for a given *DayType*.

    The mapping can be overridden via *mapping* (e.g. parsed from YAML).
    """

    @classmethod
    def from_config(cls, cfg: Mapping[str, object] | None = None) -> "StrategySelector":
        mapping = _DEFAULT_MAP.copy()
        if cfg and "strategy_map" in cfg:
            for k, v in cfg["strategy_map"].items():
                try:
                    dt = DayType(k)
                    sid = StrategyID(v)
                    mapping[dt] = sid
                except ValueError:
                    log.warning("Invalid strategy_map entry %s -> %s", k, v)
        return cls(mapping=mapping)

    mapping: Mapping[DayType, StrategyID] = field(
        default_factory=lambda: _DEFAULT_MAP.copy()
    )

    def strategy_for_day(self, day_type: DayType) -> StrategyID:
        return self.mapping.get(day_type, StrategyID.SYMMETRIC_STRANGLE)


# ---------------------------------------------------------------------------
# Day-type classifier
# ---------------------------------------------------------------------------


def classify_day(
    symbol: str,
    stock_client: StockHistoricalDataClient,
    option_client: OptionHistoricalDataClient | None = None,
    cfg: dict | None = None,
) -> DayType:
    """Heuristic classification of the current session type.

    Parameters
    ----------
    symbol
        Underlying equity ticker (e.g. "SPY").
    stock_client / option_client
        Authenticated Alpaca data clients, reused to avoid extra handshakes.
    cfg
        Optional dict of thresholds (gap %, ATR lookback, IV rank bounds, etc.).

    Notes
    -----
    For Phase-1 this implementation is *minimal* – the function demonstrates
    the expected signature and produces a deterministic yet naive result to
    keep the bot operational while we iterate on richer metrics.
    """

    # Merge runtime overrides with YAML defaults (runtime wins)
    yaml_cfg = load_daytype_config()
    cfg = {**yaml_cfg, **(cfg or {})}

    today = cfg.get("today", date.today())
    log.debug("Classifying day-type for %s on %s", symbol, today)

    # Helper thresholds
    gap_thr = cfg.get("gap_pct_threshold", 0.5)
    retrace_thr = cfg.get("gap_retrace_threshold", 40.0)
    or_window = int(cfg.get("or_window_minutes", 30))
    atr_lookback = int(cfg.get("atr_lookback_days", 14))
    range_mult = float(cfg.get("range_wide_multiplier", 1.8))
    high_iv_rank = float(cfg.get("high_iv_rank", 80))
    vix_change_thr = float(cfg.get("vix_change_pct", 8.0))

    # ------------------------------------------------------------------
    # 0. Economic calendar check via FMP (fallback to YAML)
    # ------------------------------------------------------------------
    try:
        import zoneinfo

        import pytz
        import requests  # runtime deps only when used

        api_key = os.getenv("FMP_API_KEY", "nAlmF6ETNXUum8UzPpAodz5V4cFcsPkM")
        url = f"https://financialmodelingprep.com/api/v3/economic_calendar?from={today}&to={today}&apikey={api_key}"
        resp = requests.get(url, timeout=5)
        if resp.ok:
            events = resp.json()
        else:
            events = []
    except Exception:
        events = []

    if not events:
        # fallback to static list
        events_default = cfg.get("econ_events_default", [])
        import yaml

        cal_path = cfg.get(
            "econ_calendar_path", "apps/zero_dte/data/econ_calendar.yaml"
        )
        try:
            with open(cal_path) as fh:
                _CAL = yaml.safe_load(fh) or {}
            if today.isoformat() in _CAL:
                events = _CAL[today.isoformat()]
        except FileNotFoundError:
            pass

    # ------------------------------------------------------------------
    # 1. Retrieve bars & historical data
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # 1. Retrieve yesterday + today minute bars
    # ------------------------------------------------------------------
    try:
        bars_today = get_stock_bars(symbol, today, timeframe=TimeFrame.Minute)
        bars_yday = get_stock_bars(
            symbol, today - timedelta(days=1), timeframe=TimeFrame.Minute
        )
    except Exception as exc:  # pragma: no cover – defensive
        log.warning("Cannot download bars – marking UNCLASSIFIED: %s", exc)
        return DayType.UNCLASSIFIED

    if bars_today.empty or bars_yday.empty:
        return DayType.UNCLASSIFIED

    # Gap size (%)
    open_today = bars_today.iloc[0].open
    close_yday = bars_yday.iloc[-1].close
    gap_pct = (open_today - close_yday) / close_yday * 100
    prev_high = bars_yday.high.max()
    prev_low = bars_yday.low.min()

    gap_outside = open_today > prev_high or open_today < prev_low

    # Simple thresholds (configurable)
    gap_thr = cfg.get("gap_pct_threshold", 0.4)
    retrace_thr = cfg.get("gap_retrace_threshold", 50.0)  # % of gap

    # Track running high/low to ~10:00 for retrace check
    cutoff_naive = datetime.combine(today, dt_time(hour=10, minute=0))
    cutoff = bars_today.index.tz.localize(cutoff_naive)
    sub = bars_today[bars_today.index <= cutoff]

    gap_filled = (
        (sub.low.min() <= close_yday <= sub.high.max())  # touched prev close
        if gap_pct > 0  # up-gap
        else (sub.high.max() >= close_yday >= sub.low.min())
    )

    # ------------------------------------------------------------------
    # Decision tree (very rough – will be improved later)
    # ------------------------------------------------------------------
    if gap_outside and abs(gap_pct) > gap_thr:
        if gap_filled:
            return DayType.GAP_TRAP
        return DayType.GAP_GO

    # Trend determination via simple slope of VWAP 09:30-12:00
    noon_naive = datetime.combine(today, dt_time(hour=12))
    noon_cut = bars_today.index.tz.localize(noon_naive)
    vwap = (bars_today.close * bars_today.volume).cumsum() / bars_today.volume.cumsum()
    early = vwap[bars_today.index <= noon_cut]
    if len(early) < 2:
        return DayType.INSIDE  # not enough data – default calm
    slope = early.iloc[-1] - early.iloc[0]

    if slope > 0.2:  # ~0.2$ early slope ⇒ uptrend
        return DayType.TREND_UP
    if slope < -0.2:
        return DayType.TREND_DOWN

    # Inside day if today range within y-day range
    if (bars_today.high.max() <= bars_yday.high.max()) and (
        bars_today.low.min() >= bars_yday.low.min()
    ):
        return DayType.INSIDE

    return DayType.UNCLASSIFIED
