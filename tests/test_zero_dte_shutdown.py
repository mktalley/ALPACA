import pytest

import apps.zero_dte.zero_dte_app as z
from apps.zero_dte.zero_dte_app import _execute_trade_and_update, Settings


class DummyClient:
    def __init__(self):
        self.closed = False

    def close_all_positions(self):
        self.closed = True


def setup_function():
    # Reset global state before each test
    z.daily_pnl = 0.0
    z.daily_start_equity = 100.0
    z.type_shutdown.clear()


def test_drawdown_triggers_liquidation_and_shutdown():
    client = DummyClient()

    # Define a dummy trade that returns -10 PnL (10% loss > default 5%)
    def dummy_trade(trading_client):
        return -10.0

    # Use a small drawdown threshold
    cfg = Settings(DAILY_DRAWDOWN_PCT=0.05)

    _execute_trade_and_update(dummy_trade, (client,), cfg)

    # Liquidation should have occurred and shutdown flag set
    assert client.closed, "Expected close_all_positions() to be called"
    assert z.type_shutdown.is_set(), "Expected shutdown event to be set"


def test_no_liquidation_if_within_drawdown():
    client = DummyClient()

    # Define a dummy trade that returns -4 PnL (4% loss < 5%)
    def dummy_trade(trading_client):
        return -4.0

    cfg = Settings(DAILY_DRAWDOWN_PCT=0.05)

    _execute_trade_and_update(dummy_trade, (client,), cfg)

    # No liquidation and no shutdown for small drawdown
    assert not client.closed, "Expected no liquidation for drawdown below threshold"
    assert not z.type_shutdown.is_set(), "Expected no shutdown event for drawdown below threshold"
