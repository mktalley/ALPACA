import pytest
from alpaca.trading.client import TradingClient
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.historical.option import OptionHistoricalDataClient
from datetime import time

from apps.zero_dte.zero_dte_app import (
    choose_otm_strangle_contracts,
    submit_strangle,
    monitor_and_exit_strangle,
)
from alpaca.trading.enums import OrderClass, OrderSide
from alpaca.trading.requests import MarketOrderRequest


class DummyOpt:
    def __init__(self, symbol, strike, contract_type):
        self.symbol = symbol
        self.strike = strike
        self.contract_type = contract_type


class DummyTrade:
    def __init__(self, price):
        self.price = price


@pytest.fixture
def fake_chain():
    # underlying spot at 100 → OTM call >100 and put <100
    return [
        DummyOpt("SYM_C_101", 101, "call"),
        DummyOpt("SYM_C_105", 105, "call"),
        DummyOpt("SYM_P_99", 99, "put"),
        DummyOpt("SYM_P_95", 95, "put"),
    ]


def test_choose_otm_strangle(monkeypatch, fake_chain):
    # stub option chain and underlying price
    monkeypatch.setattr(
        "apps.zero_dte.zero_dte_app.TradingClient.get_option_contracts",
        lambda self, req: fake_chain,
    )
    monkeypatch.setattr(
        "apps.zero_dte.zero_dte_app.get_underlying_price",
        lambda stock, s: 100.0,
    )

    call_sym, put_sym = choose_otm_strangle_contracts(
        trading=None, stock_client=None, symbol="SYM"
    )
    assert call_sym == "SYM_C_101"
    assert put_sym == "SYM_P_99"


def test_submit_strangle(monkeypatch):
    sold = {}

    def fake_submit(self, order_data):
        sold['order'] = order_data
        class R:
            pass

        r = R()
        r.id = "ABC"
        r.status = "new"
        r.legs = order_data.legs
        return r

    monkeypatch.setattr(
        "apps.zero_dte.zero_dte_app.TradingClient.submit_order", fake_submit
    )

    resp = submit_strangle(trading=None, call_symbol="C1", put_symbol="P1", qty=2)

    o: MarketOrderRequest = sold['order']
    assert o.order_class == OrderClass.MLEG
    assert len(o.legs) == 2
    assert o.legs[0].symbol == "C1"
    assert o.legs[1].symbol == "P1"
    assert resp.id == "ABC"


def test_monitor_and_exit_strangle(monkeypatch):
    # simulate prices that meet target immediately
    prices = [2.0, 1.0]



def test_run_strangle_sizing_capped(monkeypatch):
    """
    Test that run_strangle caps the quantity based on RISK_PCT_PER_TRADE and STOP_LOSS_PCT.
    Equity=1000, prices sum=20, STOP_LOSS_PCT=0.20 → risk_per_contract=4.0, allowed_risk=1000*0.01=10 → max_contracts=2
    Original qty=5 → capped to 2
    """
    from datetime import time as dt_time
    # Mock choose_otm_strangle_contracts
    monkeypatch.setattr(
        "apps.zero_dte.zero_dte_app.choose_otm_strangle_contracts",
        lambda trading, stock_client, symbol: ("C1", "P1"),
    )
    # Mock account equity
    class FakeAccount:
        equity = "1000"
    monkeypatch.setattr(
        "apps.zero_dte.zero_dte_app.TradingClient.get_account",
        lambda self: FakeAccount(),
    )
    # Mock option latest trade prices
    class FakeTrade:
        def __init__(self, price):
            self.price = price
    monkeypatch.setattr(
        "apps.zero_dte.zero_dte_app.OptionHistoricalDataClient.get_option_latest_trade",
        lambda self, req: {"C1": FakeTrade(10.0), "P1": FakeTrade(10.0)},
    )
    # Capture submitted qty
    submitted = {}
    def fake_submit(trading, call_sym, put_sym, qty):
        submitted['qty'] = qty
        class DummyResp:
            legs = []
        return DummyResp()
    monkeypatch.setattr(
        "apps.zero_dte.zero_dte_app.submit_strangle",
        fake_submit,
    )
    # Skip monitoring and exit
    monkeypatch.setattr(
        "apps.zero_dte.zero_dte_app.monitor_and_exit_strangle",
        lambda *args, **kwargs: 0.0,
    )
    # Run with original qty greater than max_contracts
    from apps.zero_dte.zero_dte_app import run_strangle
    result = run_strangle(
        trading=None,
        stock_client=None,
        option_client=None,
        symbol="SYM",
        qty=5,
        target_pct=0.5,
        poll_interval=0.01,
        exit_cutoff=dt_time(23, 59),
    )
    assert submitted.get('qty') == 2, f"Expected qty capped to 2, got {submitted.get('qty')}"


def test_run_strangle_sizing_no_cap(monkeypatch):
    """
    Test that run_strangle does not reduce qty when max_contracts exceeds original qty.
    Equity=1000, prices sum=2, STOP_LOSS_PCT=0.20 → risk_per_contract=0.4, allowed_risk=1000*0.01=10 → max_contracts=25
    Original qty=5 → remains 5
    """
    from datetime import time as dt_time
    monkeypatch.setattr(
        "apps.zero_dte.zero_dte_app.choose_otm_strangle_contracts",
        lambda trading, stock_client, symbol: ("C1", "P1"),
    )
    class FakeAccount:
        equity = "1000"
    monkeypatch.setattr(
        "apps.zero_dte.zero_dte_app.TradingClient.get_account",
        lambda self: FakeAccount(),
    )
    class FakeTrade:
        def __init__(self, price):
            self.price = price
    monkeypatch.setattr(
        "apps.zero_dte.zero_dte_app.OptionHistoricalDataClient.get_option_latest_trade",
        lambda self, req: {"C1": FakeTrade(1.0), "P1": FakeTrade(1.0)},
    )
    submitted = {}
    def fake_submit(trading, call_sym, put_sym, qty):
        submitted['qty'] = qty
        class DummyResp:
            legs = []
        return DummyResp()
    monkeypatch.setattr(
        "apps.zero_dte.zero_dte_app.submit_strangle",
        fake_submit,
    )
    monkeypatch.setattr(
        "apps.zero_dte.zero_dte_app.monitor_and_exit_strangle",
        lambda *args, **kwargs: 0.0,
    )
    from apps.zero_dte.zero_dte_app import run_strangle
    result = run_strangle(
        trading=None,
        stock_client=None,
        option_client=None,
        symbol="SYM",
        qty=5,
        target_pct=0.5,
        poll_interval=0.01,
        exit_cutoff=dt_time(23, 59),
    )
    assert submitted.get('qty') == 5, f"Expected qty unchanged at 5, got {submitted.get('qty')}"


def test_run_strangle_sizing_skip(monkeypatch):
    """
    Test that run_strangle skips the trade when risk budget yields qty < 1.
    Equity=100, prices sum=200, STOP_LOSS_PCT=0.20 → risk_per_contract=40, allowed_risk=100*0.01=1 → max_contracts=0 -> skip
    """
    from datetime import time as dt_time
    monkeypatch.setattr(
        "apps.zero_dte.zero_dte_app.choose_otm_strangle_contracts",
        lambda trading, stock_client, symbol: ("C1", "P1"),
    )
    class FakeAccount:
        equity = "100"
    monkeypatch.setattr(
        "apps.zero_dte.zero_dte_app.TradingClient.get_account",
        lambda self: FakeAccount(),
    )
    class FakeTrade:
        def __init__(self, price):
            self.price = price
    monkeypatch.setattr(
        "apps.zero_dte.zero_dte_app.OptionHistoricalDataClient.get_option_latest_trade",
        lambda self, req: {"C1": FakeTrade(100.0), "P1": FakeTrade(100.0)},
    )
    # Capture if submit_strangle is called
    called = {'invoked': False}
    def fake_submit(trading, call_sym, put_sym, qty):
        called['invoked'] = True
        class DummyResp:
            legs = []
        return DummyResp()
    monkeypatch.setattr(
        "apps.zero_dte.zero_dte_app.submit_strangle",
        fake_submit,
    )
    from apps.zero_dte.zero_dte_app import run_strangle
    result = run_strangle(
        trading=None,
        stock_client=None,
        option_client=None,
        symbol="SYM",
        qty=5,
        target_pct=0.5,
        poll_interval=0.01,
        exit_cutoff=dt_time(23, 59),
    )
    assert not called['invoked'], "Expected skip of submit_strangle when qty < 1"
    assert result is None, "Expected run_strangle to return None when skipping trade"

    def fake_trade(self, req):
        p = prices.pop(0)
        return {"C1": DummyTrade(p), "P1": DummyTrade(p)}

    monkeypatch.setattr(
        "apps.zero_dte.zero_dte_app.OptionHistoricalDataClient.get_option_latest_trade",
        fake_trade,
    )
    calls = []

    def fake_submit(self, order_data):
        calls.append(order_data)
        class R:
            pass

        r = R()
        r.id = "X"
        r.status = "filled"
        return r

    monkeypatch.setattr(
        "apps.zero_dte.zero_dte_app.TradingClient.submit_order", fake_submit
    )

    # entry_price_sum=2.0, target_pct=0.5 → target_sum=3.0
    monitor_and_exit_strangle(
        trading=None,
        option_client=None,
        symbols=["C1", "P1"],
        entry_price_sum=2.0,
        qty=1,
        target_pct=0.5,
        poll_interval=0.01,
        exit_cutoff=time(23, 59),
    )

    # should trigger exactly one exit order
    assert len(calls) == 1
    exit_order = calls[0]
    assert exit_order.order_class == OrderClass.MLEG
    assert exit_order.legs[0].side == OrderSide.SELL


def test_monitor_and_exit_strangle_retry(monkeypatch):
    from requests.exceptions import RequestException
    # Setup fake trade to raise once then succeed
    call_log = []
    prices = [2.0, 2.0]
    def fake_trade(self, req):
        if not call_log:
            call_log.append('err')
            raise RequestException('transient error')
        return {
            'C1': DummyTrade(prices.pop(0)),
            'P1': DummyTrade(prices.pop(0)),
        }
    monkeypatch.setattr(
        'apps.zero_dte.zero_dte_app.OptionHistoricalDataClient.get_option_latest_trade',
        fake_trade,
    )
    # avoid delays
    monkeypatch.setattr('apps.zero_dte.zero_dte_app.time.sleep', lambda _: None)
    calls = []
    def fake_submit(self, order_data):
        calls.append(order_data)
        class R: pass
        r = R()
        r.id = 'R'
        r.status = 'filled'
        return r
    monkeypatch.setattr(
        'apps.zero_dte.zero_dte_app.TradingClient.submit_order',
        fake_submit,
    )
    # entry_price_sum=2.0, target_pct=0.5 → target_sum=3.0
    monitor_and_exit_strangle(
        trading=None,
        option_client=None,
        symbols=['C1', 'P1'],
        entry_price_sum=2.0,
        qty=1,
        target_pct=0.5,
        poll_interval=0.01,
        exit_cutoff=time(23, 59),
    )
    assert call_log, "Expected at least one retry on transient error"
    assert len(calls) == 1


def test_monitor_and_exit_strangle_stop_loss(monkeypatch):
    # Setup fake trade to always return low prices breaching stop-loss
    def fake_trade(self, req):
        return {
            'C1': DummyTrade(0.5),
            'P1': DummyTrade(0.5),
        }
    monkeypatch.setattr(
        'apps.zero_dte.zero_dte_app.OptionHistoricalDataClient.get_option_latest_trade',
        fake_trade,
    )
    monkeypatch.setattr('apps.zero_dte.zero_dte_app.time.sleep', lambda _: None)
    calls = []
    def fake_submit(self, order_data):
        calls.append(order_data)
        class R: pass
        r = R()
        r.id = 'S'
        r.status = 'filled'
        return r
    monkeypatch.setattr(
        'apps.zero_dte.zero_dte_app.TradingClient.submit_order',
        fake_submit,
    )
    # entry_price_sum=2.0, stop_loss_pct=0.3 → stop_loss_sum=1.4, current_sum=1.0 breaches
    monitor_and_exit_strangle(
        trading=None,
        option_client=None,
        symbols=['C1', 'P1'],
        entry_price_sum=2.0,
        qty=1,
        target_pct=0.5,
        poll_interval=0.01,
        exit_cutoff=time(23, 59),
    )
    assert len(calls) == 1

