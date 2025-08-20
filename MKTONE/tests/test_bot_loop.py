"""Tests for the bot loop respecting market hours and positions."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

# Ensure the src package is on the path
ROOT = Path(__file__).resolve().parents[1]
sys.path.extend([str(ROOT), str(ROOT.parent)])

import src.bot as bot  # noqa: E402  # pylint: disable=C0413


def test_loop_honors_market_hours(monkeypatch):
    """Strategy should run only while the market is open."""
    trading = Mock()
    trading.get_clock = Mock(
        side_effect=[SimpleNamespace(is_open=True), SimpleNamespace(is_open=False)]
    )
    trading.get_all_positions = Mock(return_value=[])
    strategy = Mock()

    monkeypatch.setattr(bot, "get_trading_client", lambda: trading)
    monkeypatch.setattr(bot, "get_stock_client", Mock())
    monkeypatch.setattr(
        bot, "ZeroDTECreditSpread", lambda trading_client, stock_client: strategy
    )
    monkeypatch.setattr(bot.time, "sleep", lambda x: None)

    bot.main(poll_interval=0)

    assert strategy.run.call_count == 1
    assert trading.get_clock.call_count == 2


def test_loop_no_duplicate_positions(monkeypatch):
    """Strategy should not run again when positions remain open."""
    trading = Mock()
    trading.get_clock = Mock(
        side_effect=[
            SimpleNamespace(is_open=True),
            SimpleNamespace(is_open=True),
            SimpleNamespace(is_open=False),
        ]
    )
    trading.get_all_positions = Mock(side_effect=[[], ["existing"]])
    strategy = Mock()

    monkeypatch.setattr(bot, "get_trading_client", lambda: trading)
    monkeypatch.setattr(bot, "get_stock_client", Mock())
    monkeypatch.setattr(
        bot, "ZeroDTECreditSpread", lambda trading_client, stock_client: strategy
    )
    monkeypatch.setattr(bot.time, "sleep", lambda x: None)

    bot.main(poll_interval=0)

    assert strategy.run.call_count == 1
    assert trading.get_all_positions.call_count == 2
