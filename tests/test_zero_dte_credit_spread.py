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
