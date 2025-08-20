"""Zero DTE credit spread strategy implementation."""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from typing import Any


def build_option_symbol(underlying: str, exp: dt.date, strike: float, opt_type: str) -> str:
    """Return OCC option symbol for the given params."""
    return f"{underlying}{exp:%y%m%d}{opt_type}{int(strike * 1000):08d}"


@dataclass
class ZeroDTECreditSpread:
    """Simple momentum based credit spread."""

    trading_client: Any
    stock_client: Any
    symbol: str = "SPY"
    qty: int = 1
    ma_window: int = 5
    strike_width: float = 1.0
    profit_target: float = 0.2
    stop_loss: float = 0.5

    def _latest_price(self) -> float:
        from alpaca.data.requests import StockLatestTradeRequest

        req = StockLatestTradeRequest(symbol_or_symbols=self.symbol)
        trade = self.stock_client.get_stock_latest_trade(req)
        return float(trade.price)

    def _moving_average(self) -> float:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        end = dt.datetime.now(dt.timezone.utc)
        start = end - dt.timedelta(minutes=self.ma_window * 2)
        req = StockBarsRequest(
            symbol_or_symbols=self.symbol,
            timeframe=TimeFrame.Minute,
            start=start,
            end=end,
            limit=self.ma_window,
        )
        bars = self.stock_client.get_stock_bars(req).df
        return float(bars["close"].mean())

    def _determine_direction(self) -> str:
        price = self._latest_price()
        ma = self._moving_average()
        logging.info("Price %.2f vs MA %.2f", price, ma)
        return "bull" if price > ma else "bear"

    def _create_order(self):
        from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
        from alpaca.trading.requests import LimitOrderRequest, OptionLegRequest

        direction = self._determine_direction()
        exp = dt.date.today()
        price = self._latest_price()
        atm = round(price)
        if direction == "bull":
            short_strike = atm
            long_strike = atm - self.strike_width
            opt_type = "P"
            limit_price = -0.5  # receive credit
        else:
            short_strike = atm
            long_strike = atm + self.strike_width
            opt_type = "C"
            limit_price = -0.5
        short_symbol = build_option_symbol(self.symbol, exp, short_strike, opt_type)
        long_symbol = build_option_symbol(self.symbol, exp, long_strike, opt_type)
        legs = [
            OptionLegRequest(symbol=short_symbol, ratio_qty=1, side=OrderSide.SELL),
            OptionLegRequest(symbol=long_symbol, ratio_qty=1, side=OrderSide.BUY),
        ]
        order = LimitOrderRequest(
            order_class=OrderClass.MLEG,
            legs=legs,
            qty=self.qty,
            time_in_force=TimeInForce.DAY,
            limit_price=limit_price,
        )
        return order

    def run(self) -> None:
        """Execute one cycle of the strategy."""
        order = self._create_order()
        logging.info("Submitting order: %s", order)
        try:
            self.trading_client.submit_order(order)
            logging.info("Order submitted successfully.")
        except Exception as exc:  # pragma: no cover - network
            logging.exception("Order submission failed: %s", exc)

