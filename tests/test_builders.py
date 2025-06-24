import pytest

from apps.zero_dte.builders import (IronCondorBuilder, StrangleBuilder,
                                    select_builder)
from apps.zero_dte.market_structure import StrategyID


def test_select_builder_defaults():
    assert select_builder(StrategyID.SYMMETRIC_STRANGLE) is StrangleBuilder


def test_select_builder_not_implemented():
    builder_cls = select_builder(StrategyID.WIDE_IRON_CONDOR)
    assert builder_cls is IronCondorBuilder
    assert builder_cls().build() == 0.0
