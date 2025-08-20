codex/update-order-creation-to-use-market-data
import datetime as dt
from types import SimpleNamespace

from MKTONE.src.strategies.zero_dte_credit_spread import (
    ZeroDTECreditSpread,
    build_option_symbol,
)


class DummyQuoteClient:
    def __init__(self, data):
        self.data = data

    def get_option_chain(self, req):  # noqa: D401 - simple stub
        return self.data


def _make_chain(symbol: str, exp: dt.date, short: float, long: float):
    short_symbol = build_option_symbol(symbol, exp, short, "P")
    long_symbol = build_option_symbol(symbol, exp, long, "P")
    chain = {
        short_symbol: SimpleNamespace(
            latest_quote=SimpleNamespace(bid_price=1.0, ask_price=1.2)
        ),
        long_symbol: SimpleNamespace(
            latest_quote=SimpleNamespace(bid_price=0.5, ask_price=0.7)
        ),
    }
    return chain, short_symbol, long_symbol


def test_limit_price_matches_mid_quotes():
    exp = dt.date.today()
    chain, _, _ = _make_chain("SPY", exp, 100, 99)
    option_client = DummyQuoteClient(chain)

    strat = ZeroDTECreditSpread(
        trading_client=None,
        stock_client=None,
        option_client=option_client,
        strike_width=1.0,
    )

    strat._latest_price = lambda: 100  # price rounded to 100
    strat._determine_direction = lambda: "bull"

    order = strat._create_order()

    expected_credit = (1.1 - 0.6)  # mid(short) - mid(long)
    assert order.limit_price == -expected_credit
    assert strat.profit_target_amt == expected_credit * strat.profit_target
    assert strat.stop_loss_amt == expected_credit * strat.stop_loss
=======
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
main
