"""Strategy builder routing – Phase-1 stubs.

Exposes ``select_builder(strategy_id)`` which maps a :class:`apps.zero_dte.market_structure.StrategyID`
value to a callable that executes the strategy.  Only *SYMMETRIC_STRANGLE* is
implemented (delegates to :pyfunc:`apps.zero_dte.zero_dte_app.run_strangle`).
Other strategies raise *NotImplementedError* for now.
"""

from __future__ import annotations

import logging
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
    """Minimal wrapper for iron-condor entry helpers.

    For Phase-1 we avoid live API calls; the builder merely records that the
    strategy *would* run and returns a placeholder PnL of ``0.0``.  Concrete
    logic will be plugged in Phase-2.
    """

    def build(self):  # type: ignore[override]
        log.info("[IronCondorBuilder] Stub execution – returning 0.0 PnL")
        return 0.0


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
