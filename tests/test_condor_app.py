import pytest
from datetime import time
from requests.exceptions import RequestException

from apps.zero_dte.two_phase import monitor_and_exit_condor
from alpaca.trading.enums import OrderClass, OrderSide
from alpaca.trading.requests import MarketOrderRequest

class DummyTrade:
    def __init__(self, price):
        self.price = price


def test_monitor_and_exit_condor_retry(monkeypatch):
    # simulate transient error once, then prices that hit target
    seq = [True, False]
    symbols = ['SC', 'LC', 'SP', 'LP']
    prices_seq = [[1.0, 0.5, 1.0, 0.5]]

    def fake_trade(self, req):
        if seq.pop(0):
            raise RequestException('transient error')
        pr = prices_seq.pop(0)
        return {sym: DummyTrade(pr[i]) for i, sym in enumerate(symbols)}

    monkeypatch.setattr(
        'apps.zero_dte.two_phase.OptionHistoricalDataClient.get_option_latest_trade',
        fake_trade,
    )
    monkeypatch.setattr('apps.zero_dte.two_phase.time.sleep', lambda _: None)

    calls = []
    def fake_submit(self, order_data):
        calls.append(order_data)
        class R: pass
        r = R(); r.id = 'CID'; r.status = 'filled'; return r

    monkeypatch.setattr(
        'apps.zero_dte.two_phase.TradingClient.submit_order', fake_submit
    )

    result = monitor_and_exit_condor(
        trading=None,
        option_client=None,
        symbols=symbols,
        entry_credit=2.0,
        qty=1,
        credit_target_pct=0.5,
        poll_interval=0.01,
        exit_cutoff=time(23, 59),
    )
    assert seq == []  # consumed the transient flag
    assert len(calls) == 1  # one exit order
    assert result is True


def test_monitor_and_exit_condor_stop_loss(monkeypatch):
    # simulate prices breaching stop-loss first
    symbols = ['SC', 'LC', 'SP', 'LP']
    def fake_trade(self, req):
        # prices [2.0, 0.0, 2.0, 0.0] -> debit = 4.0
        return {sym: DummyTrade(2.0 if i % 2 == 0 else 0.0) for i, sym in enumerate(symbols)}

    monkeypatch.setattr(
        'apps.zero_dte.two_phase.OptionHistoricalDataClient.get_option_latest_trade', fake_trade
    )
    monkeypatch.setattr('apps.zero_dte.two_phase.time.sleep', lambda _: None)

    calls = []
    def fake_submit(self, order_data):
        calls.append(order_data)
        class R: pass
        r = R(); r.id = 'SLID'; r.status = 'filled'; return r

    monkeypatch.setattr(
        'apps.zero_dte.two_phase.TradingClient.submit_order', fake_submit
    )

    result = monitor_and_exit_condor(
        trading=None,
        option_client=None,
        symbols=symbols,
        entry_credit=2.0,
        qty=1,
        credit_target_pct=0.5,
        poll_interval=0.01,
        exit_cutoff=time(23, 59),
    )
    assert len(calls) == 1  # exit on stop-loss
    assert result is False
