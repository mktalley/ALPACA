from datetime import time as dt_time

import pytest

from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.trading.client import \
    TradingClient  # This import is incorrect; should be alpaca.trading.client
# We'll import run_two_phase properly
from apps.zero_dte.two_phase import run_two_phase


def test_run_two_phase_sizing_capped(monkeypatch):
    # Equity=1000, entry_credit_est = pr_sc+pr_sp-pr_lc-pr_lp = 20 (10+10-0-0)
    # STOP_LOSS_PCT=0.20, risk_per_contract=4, allowed_risk=10 (1000*0.01), max_contracts=2
    # original qty=5, capped to 2

    # Mock choose_iron_condor_contracts to return 4 leg symbols
    monkeypatch.setattr(
        "apps.zero_dte.two_phase.choose_iron_condor_contracts",
        lambda trading, stock_client, symbol, wing: ("SC", "SP", "LC", "LP"),
    )

    # Mock account equity
    class FakeAccount:
        equity = "1000"

    monkeypatch.setattr(
        "apps.zero_dte.two_phase.TradingClient.get_account",
        lambda self: FakeAccount(),
    )

    # Mock option trades prices
    class FakeTrade:
        def __init__(self, price):
            self.price = price

    def fake_latest_trade(self, req):
        return {
            "SC": FakeTrade(10.0),
            "SP": FakeTrade(10.0),
            "LC": FakeTrade(0.0),
            "LP": FakeTrade(0.0),
        }

    monkeypatch.setattr(
        "apps.zero_dte.two_phase.OptionHistoricalDataClient.get_option_latest_trade",
        fake_latest_trade,
    )
    # Capture submit_iron_condor calls
    called = {}

    def fake_submit(trading, sc, sp, lc, lp, qty):
        called["qty"] = qty

        class DummyResp:
            legs = []

        return DummyResp()

    monkeypatch.setattr(
        "apps.zero_dte.two_phase.submit_iron_condor",
        fake_submit,
    )
    # Stub monitor_and_exit_condor to skip
    monkeypatch.setattr(
        "apps.zero_dte.two_phase.monitor_and_exit_condor",
        lambda *args, **kwargs: True,
    )
    result = run_two_phase(
        trading=None,
        stock_client=None,
        option_client=None,
        symbol="SYM",
        qty=5,
        profit_target_pct=0.5,
        poll_interval=0.01,
        exit_cutoff=dt_time(23, 59),
        condor_target_pct=0.25,
    )
    assert called.get("qty") == 2, f"Expected qty capped to 2, got {called.get('qty')}"
    assert result is True


def test_run_two_phase_sizing_no_cap(monkeypatch):
    # Equity=1000, entry_credit_est=2 (1+1-0-0), risk_per_contract=0.4, allowed_risk=10, max_contracts=25
    # original qty=5 -> remains 5
    monkeypatch.setattr(
        "apps.zero_dte.two_phase.choose_iron_condor_contracts",
        lambda trading, stock_client, symbol, wing: ("SC", "SP", "LC", "LP"),
    )

    class FakeAccount:
        equity = "1000"

    monkeypatch.setattr(
        "apps.zero_dte.two_phase.TradingClient.get_account",
        lambda self: FakeAccount(),
    )

    class FakeTrade:
        def __init__(self, price):
            self.price = price

    def fake_latest_trade(self, req):
        return {
            "SC": FakeTrade(1.0),
            "SP": FakeTrade(1.0),
            "LC": FakeTrade(0.0),
            "LP": FakeTrade(0.0),
        }

    monkeypatch.setattr(
        "apps.zero_dte.two_phase.OptionHistoricalDataClient.get_option_latest_trade",
        fake_latest_trade,
    )
    called = {}

    def fake_submit(trading, sc, sp, lc, lp, qty):
        called["qty"] = qty

        class DummyResp:
            legs = []

        return DummyResp()

    monkeypatch.setattr(
        "apps.zero_dte.two_phase.submit_iron_condor",
        fake_submit,
    )
    monkeypatch.setattr(
        "apps.zero_dte.two_phase.monitor_and_exit_condor",
        lambda *args, **kwargs: False,
    )
    result = run_two_phase(
        trading=None,
        stock_client=None,
        option_client=None,
        symbol="SYM",
        qty=5,
        profit_target_pct=0.5,
        poll_interval=0.01,
        exit_cutoff=dt_time(23, 59),
        condor_target_pct=0.25,
    )
    assert (
        called.get("qty") == 5
    ), f"Expected qty unchanged at 5, got {called.get('qty')}"
    assert result is False


def test_run_two_phase_sizing_skip(monkeypatch):
    # Equity=100, entry_credit_est=200 (100+100-0-0), risk_per_contract=40, allowed_risk=1, max_contracts=0 -> skip
    monkeypatch.setattr(
        "apps.zero_dte.two_phase.choose_iron_condor_contracts",
        lambda trading, stock_client, symbol, wing: ("SC", "SP", "LC", "LP"),
    )

    class FakeAccount:
        equity = "100"

    monkeypatch.setattr(
        "apps.zero_dte.two_phase.TradingClient.get_account",
        lambda self: FakeAccount(),
    )

    class FakeTrade:
        def __init__(self, price):
            self.price = price

    def fake_latest_trade(self, req):
        return {
            "SC": FakeTrade(100.0),
            "SP": FakeTrade(100.0),
            "LC": FakeTrade(0.0),
            "LP": FakeTrade(0.0),
        }

    monkeypatch.setattr(
        "apps.zero_dte.two_phase.OptionHistoricalDataClient.get_option_latest_trade",
        fake_latest_trade,
    )
    invoked = False

    def fake_submit(trading, sc, sp, lc, lp, qty):
        nonlocal invoked
        invoked = True

        class DummyResp:
            legs = []

        return DummyResp()

    monkeypatch.setattr(
        "apps.zero_dte.two_phase.submit_iron_condor",
        fake_submit,
    )
    result = run_two_phase(
        trading=None,
        stock_client=None,
        option_client=None,
        symbol="SYM",
        qty=5,
        profit_target_pct=0.5,
        poll_interval=0.01,
        exit_cutoff=dt_time(23, 59),
        condor_target_pct=0.25,
    )
    assert not invoked, "Expected submit_iron_condor not called when qty < 1"
    assert result is False
