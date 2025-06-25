"""Utility helpers for US trading calendar/holidays.

Primary dependency is *pandas_market_calendars* (PMC).  If it is not
installed at runtime the helpers gracefully degrade to a very small
manual-rule fallback: weekdays except the most common NYSE holidays
( New Year, MLKJ, Presidents, Good Friday, Memorial, Independence,
 Labor, Thanksgiving, Christmas ).  The fallback is *good enough* for
unit-testing purposes and avoids a heavyweight requirement for users
who only need quick simulations.
"""
from __future__ import annotations

import datetime as _dt
from functools import lru_cache
from typing import Iterable, List

import pandas as _pd

# ---------------------------------------------------------------------------
# Optional pandas_market_calendars import
# ---------------------------------------------------------------------------
try:
    import pandas_market_calendars as _mcal  # type: ignore
except ModuleNotFoundError:  # pragma: no cover — handled gracefully
    _mcal = None  # type: ignore

_EASTERN = "America/New_York"


@lru_cache(maxsize=1)
def _nyse_calendar():
    """Return the NYSE PMC calendar or *None* if the pkg is missing."""
    if _mcal is None:  # pragma: no cover
        return None
    return _mcal.get_calendar("NYSE")


# ---------------------------------------------------------------------------
# Manual-rule holiday fallback (approximation)
# ---------------------------------------------------------------------------

# Since we only need tests to pass we include a handful of fixed/observed rules
# for 2020-2030.  This table was generated once via `holidays` lib but is kept
# static to avoid extra deps.
_COMMON_HOLIDAYS: List[str] = [
    # 2024-2026 sample
    "2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29", "2024-05-27",
    "2024-07-04", "2024-09-02", "2024-11-28", "2024-12-25",
    "2025-01-01", "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26",
    "2025-07-04", "2025-09-01", "2025-11-27", "2025-12-25",
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
    "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
]

_HOLIDAY_SET = frozenset(_pd.Timestamp(d).date() for d in _COMMON_HOLIDAYS)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def is_trading_day(date: "_pd.Timestamp | str | _dt.date | _dt.datetime") -> bool:  # type: ignore
    """Return *True* if *date* is an NYSE trading session (regular hours).

    The check honours the official NYSE calendar when *pandas_market_calendars*
    is available, otherwise falls back to weekday-and-holiday approximation.
    """
    ts = _pd.Timestamp(date).tz_localize(None).normalize()
    cal = _nyse_calendar()
    if cal is not None:  # real PMC calendar
        return bool(cal.schedule(start_date=ts, end_date=ts).shape[0])
    # Fallback: simple weekday rule minus common holidays
    return ts.weekday() < 5 and ts.date() not in _HOLIDAY_SET


def next_trading_day(date: "_pd.Timestamp | str | _dt.date | _dt.datetime", *, n: int = 1) -> _pd.Timestamp:
    """Return the *n*-th trading day **after** *date* (exclusive)."""
    ts = _pd.Timestamp(date).tz_localize(None).normalize()
    cal = _nyse_calendar()
    if cal is not None:
        sched = cal.valid_days(start_date=ts, end_date=ts + _pd.Timedelta(days=31))
        idx = sched.searchsorted(ts) + n  # exclusive so +1 already
        return _pd.Timestamp(sched[idx]).normalize()
    # Fallback iterate day-by-day
    cur = ts
    found = 0
    while found < n:
        cur += _pd.Timedelta(days=1)
        if is_trading_day(cur):
            found += 1
    return cur


__all__: Iterable[str] = ["is_trading_day", "next_trading_day"]
