import sys
from pathlib import Path
from types import SimpleNamespace
import datetime as dt

import pytest

# add path to strategy module
sys.path.append(str(Path(__file__).resolve().parents[1] / "MKTONE" / "src"))

from strategies.zero_dte_credit_spread import ZeroDTECreditSpread, dt as zdt  # noqa: E402


class DummyTradingClient:
    def __init__(self, credits, market_close):
        self.credits = iter(credits)
        self.market_close = market_close
        self.closed = False

    def submit_order(self, order):
        return SimpleNamespace(id="1", filled_avg_price=-1.0)

    def get_open_position(self, order_id):
        if self.closed:
            return None
        try:
            credit = next(self.credits)
        except StopIteration:
            return None
        return SimpleNamespace(current_credit=credit)

    def close_position(self, order_id):
        self.closed = True

    def get_clock(self):
        return SimpleNamespace(market_close=self.market_close)


class FakeDateTime(dt.datetime):
    """Class to control datetime.now() in tests."""

    @classmethod
    def now(cls, tz=None):
        return cls.fixed_now


@pytest.fixture(autouse=True)
def patch_create_order(monkeypatch):
    order = SimpleNamespace(limit_price=-1.0)
    monkeypatch.setattr(ZeroDTECreditSpread, "_create_order", lambda self: order)


def run_strategy(credits, now, market_close):
    client = DummyTradingClient(credits, market_close)
    FakeDateTime.fixed_now = now
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(zdt, "datetime", FakeDateTime)
    strat = ZeroDTECreditSpread(trading_client=client, stock_client=None)
    strat.run()
    monkeypatch.undo()
    return client


def test_take_profit_closes_position():
    now = dt.datetime(2024, 1, 1, 12, tzinfo=dt.timezone.utc)
    mc = now + dt.timedelta(hours=1)
    client = run_strategy([0.7], now, mc)
    assert client.closed


def test_stop_loss_closes_position():
    now = dt.datetime(2024, 1, 1, 12, tzinfo=dt.timezone.utc)
    mc = now + dt.timedelta(hours=1)
    client = run_strategy([1.6], now, mc)
    assert client.closed


def test_end_of_day_closes_position():
    now = dt.datetime(2024, 1, 1, 12, tzinfo=dt.timezone.utc)
    mc = now + dt.timedelta(minutes=4)
    client = run_strategy([1.0], now, mc)
    assert client.closed
