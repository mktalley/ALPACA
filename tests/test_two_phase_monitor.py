import pytest
from datetime import time
import logging

import apps.zero_dte.two_phase as tp_module
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.trading.client import TradingClient


class DummyTrade:
    def __init__(self, price):
        self.price = price


def make_fake_trades(symbols, prices):
    return {s: DummyTrade(p) for s, p in zip(symbols, prices)}


@ pytest.mark.parametrize(
    "prices, entry_credit, credit_target_pct, expected_pnl",
    [
        ([3, 2, 1, 1], 2.0, 0.5, 1.0),   # profit scenario
        ([1, 1, 5, 5], 3.0, 0.5, -11.0),  # small loss scenario
    ]
)

def test_monitor_and_exit_condor_basic(prices, entry_credit, credit_target_pct, expected_pnl, monkeypatch, caplog):
    symbols = ['A', 'B', 'C', 'D']
    monkeypatch.setattr(
        OptionHistoricalDataClient,
        'get_option_latest_trade',
        lambda client, req: make_fake_trades(symbols, prices)
    )
    monkeypatch.setattr(
        TradingClient, 'submit_order',
        lambda trading, order: type('R', (), {'id': 'x', 'status': 'filled'})
    )

    caplog.set_level(logging.INFO)
    result = tp_module.monitor_and_exit_condor(
        trading=None,
        option_client=None,
        symbols=symbols,
        entry_credit=entry_credit,
        qty=1,
        credit_target_pct=credit_target_pct,
        poll_interval=0,
        exit_cutoff=time(23, 59)
    )
    assert result == pytest.approx(expected_pnl)
    # verify log contains correct PnL formatting
    assert f"Exit PnL={expected_pnl:+0.2f}" in caplog.text


def test_monitor_and_exit_condor_cap(monkeypatch, caplog):
    symbols = ['W', 'X', 'Y', 'Z']
    prices = [0, 0, 2000, 2000]  # sold=0, bought=4000
    entry_credit = 3.0
    monkeypatch.setattr(
        OptionHistoricalDataClient,
        'get_option_latest_trade',
        lambda client, req: make_fake_trades(symbols, prices)
    )
    monkeypatch.setattr(
        TradingClient, 'submit_order',
        lambda trading, order: type('R', (), {'id': 'x', 'status': 'filled'})
    )

    cfg = tp_module.Settings()
    caplog.set_level(logging.INFO)
    result = tp_module.monitor_and_exit_condor(
        trading=None,
        option_client=None,
        symbols=symbols,
        entry_credit=entry_credit,
        qty=1,
        credit_target_pct=0.5,
        poll_interval=0,
        exit_cutoff=time(23, 59)
    )
    assert result == pytest.approx(-cfg.MAX_LOSS_PER_TRADE)
    # verify capped
    assert f"Exit PnL={-cfg.MAX_LOSS_PER_TRADE:+0.2f}" in caplog.text
