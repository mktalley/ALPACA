"""Strategy abstractions for 0-DTE back-tests.

Only *very* light logic is implemented here so that users can subclass
:class:`Strategy` and override a handful of methods to encode their own
entry/exit logic.  The engine is deliberately primitive – one trade per
session – because most production 0-DTE workflows (iron-condors, strangles,
credit spreads etc.) keep at most a single position open per side.

The module does **not** depend on the Alpaca SDK directly which keeps the
unit-test surface small.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, time, timedelta
from types import MappingProxyType
from typing import Final, Mapping

import pandas as pd

__all__: list[str] = [
    "Strategy",
    "SimplePutSell",
]


class Strategy(ABC):
    """Base-class for stateless parametric entry/exit logic.

    A concrete subclass should be *pure* – i.e. its methods must not mutate
    global state so that the back-test can run safely in parallel workers.
    Each instance receives a *parameters* mapping at construction time and the
    back-test engine exposes it verbatim on the resulting ``TradeLog`` so that
    grid-search bookkeeping is trivial.
    """

    #: Friendly name used by CLI – defaults to class name.
    name: Final[str] = "Strategy"

    def __init__(self, **params):
        self._params: Mapping[str, object] = MappingProxyType(dict(params))

    # ---------------------------------------------------------------------
    # Required API – must be implemented by concrete strategy
    # ---------------------------------------------------------------------

    @abstractmethod
    def entry_time(self) -> time:  # noqa: D401 – simple time accessor
        """Return *Eastern* time at which the position should be opened."""

    @abstractmethod
    def exit_time(self) -> time:  # noqa: D401
        """Return *Eastern* time at which the position should be closed."""

    @abstractmethod
    def choose_contract(self, chain: pd.DataFrame) -> str:
        """Select option symbol from OCC *chain* DataFrame.

        ``chain`` columns are assumed to include at least ``symbol`` &
        ``delta``.  Strategies are free to ignore columns they don't need.
        The function must *return* the selected OCC code so that the engine
        can later fetch minute bars only for the symbol actually traded.
        """

    # ------------------------------------------------------------------
    # Helpers available to subclasses (do **not** override in user code)
    # ------------------------------------------------------------------

    @property
    def params(self) -> Mapping[str, object]:  # noqa: D401
        return self._params

    # Convenience so ``print(strategy)`` dumps the param-dict.
    def __repr__(self):  # noqa: D401
        param_str = ", ".join(f"{k}={v}" for k, v in self._params.items())
        return f"<{self.__class__.__name__} {param_str}>"


# ---------------------------------------------------------------------------
# Example concrete strategy – always sells ATM put at 10:00 ET, closes 15:55
# ---------------------------------------------------------------------------


class SimplePutSell(Strategy):
    """Sell at-the-money put once per day, exit before close.

    Parameters
    ----------
    entry_time : str, default "10:00"
        HH:MM Eastern – when to open.
    exit_time : str, default "15:55"
        HH:MM Eastern – when to exit.
    """

    name = "put-sell"

    def __init__(self, entry_time: str = "10:00", exit_time: str = "15:55"):
        super().__init__(entry_time=entry_time, exit_time=exit_time)
        self._entry_time = _parse_hhmm(entry_time)
        self._exit_time = _parse_hhmm(exit_time)

    # Required overrides -------------------------------------------------

    def entry_time(self) -> time:  # noqa: D401 – typed alias to property
        return self._entry_time

    def exit_time(self) -> time:  # noqa: D401
        return self._exit_time

    def choose_contract(self, chain: pd.DataFrame) -> str:  # noqa: D401
        # ATM ≈ |delta| ~ 0.50 – pick the first matching option.
        chain_sorted = chain.loc[(chain["delta"].abs() >= 0.45) & (chain["delta"].abs() <= 0.55)]
        if chain_sorted.empty:
            # Fallback: choose *closest* delta
            idx = (chain["delta"].abs() - 0.50).abs().idxmin()
            return chain.loc[idx, "symbol"]
        return chain_sorted.iloc[0]["symbol"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_hhmm(value: str) -> time:
    if value.count(":") != 1:
        raise ValueError(f"Invalid HH:MM time literal: {value!r}")
    h, m = map(int, value.split(":"))
    return time(hour=h, minute=m)
