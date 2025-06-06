#!/usr/bin/env python3
"""
0DTE OTM Strangle (Call + Put) & Target-Exit Example
Designed for institutional clients:
  • Configurable via env/.env
  • Structured with pydantic BaseSettings
  • Logging, retries, time checks
  • Easy to extend to multi-leg (MLEG) or other underlyings
"""

import time
import logging
from datetime import date, datetime, time as dt_time
from typing import Dict

try:
    from pydantic import BaseSettings
except Exception:
    class BaseSettings:
        """Dummy BaseSettings for environments without pydantic v1 BaseSettings"""
        pass


from pydantic import Field, validator

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
from alpaca.trading.requests import (
    GetOptionContractsRequest,
    MarketOrderRequest,
    OptionLegRequest,
)
from alpaca.data.requests import (
    StockLatestTradeRequest,
    OptionLatestTradeRequest,
)
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.historical.option import OptionHistoricalDataClient


class Settings(BaseSettings):
    ALPACA_API_KEY: str
    ALPACA_API_SECRET: str
    PAPER: bool = True

    UNDERLYING: str = Field("SPY", description="Underlying symbol to trade options on")
    QTY: int = Field(1, description="Number of contracts per leg")
    PROFIT_TARGET_PCT: float = Field(
        0.50, description="Profit target as a percent of entry price (e.g. 0.5 = 50%)"
    )
    POLL_INTERVAL: float = Field(120.0, description="Seconds between price checks")
    EXIT_CUTOFF: dt_time = Field(
        dt_time(15, 45), description="Hard exit time (wall clock) if target not hit"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @validator("EXIT_CUTOFF", pre=True)
    def parse_exit_time(cls, v):
        if isinstance(v, str):
            return dt_time.fromisoformat(v)
        return v



def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_underlying_price(stock_client: StockHistoricalDataClient, symbol: str) -> float:
    resp = stock_client.get_stock_latest_trade(
        StockLatestTradeRequest(symbol_or_symbols=[symbol])
    )
    trade = resp[symbol]
    return float(trade.price)


def choose_atm_contract(
    trading: TradingClient,
    stock_client: StockHistoricalDataClient,
    symbol: str,
) -> str:
    today = date.today().isoformat()
    chain = TradingClient.get_option_contracts(trading,
        GetOptionContractsRequest(
            underlying_symbols=[symbol],
            expiration_date=today,
            limit=500,
        )
    )
    spot = get_underlying_price(stock_client, symbol)
    calls = [c for c in chain if c.contract_type == "call"]
    atm = min(calls, key=lambda c: abs(c.strike - spot))
    logging.info(
        "Underlying %s spot=%.2f → ATM call %s @ strike=%.2f",
        symbol,
        spot,
        atm.symbol,
        atm.strike,
    )
    return atm.symbol


def submit_buy(
    trading: TradingClient, option_symbol: str, qty: int
) -> Dict:
    leg = OptionLegRequest(
        symbol=option_symbol,
        qty=qty,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
    )
    order = MarketOrderRequest(
        symbol=option_symbol,
        qty=qty,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
        legs=[leg],
    )
    resp = TradingClient.submit_order(trading, order)
    logging.info("Submitted BUY order %s; status=%s", resp.id, resp.status)
    return resp




def choose_otm_strangle_contracts(
    trading: TradingClient,
    stock_client: StockHistoricalDataClient,
    symbol: str,
) -> tuple[str, str]:
    """
    Fetches today's option chain and returns the nearest OTM call and put symbols.
    """
    today = date.today().isoformat()
    chain = TradingClient.get_option_contracts(trading,
        GetOptionContractsRequest(
            underlying_symbols=[symbol],
            expiration_date=today,
            limit=500,
        )
    )
    spot = get_underlying_price(stock_client, symbol)
    calls = [c for c in chain if c.contract_type == "call" and c.strike > spot]
    puts = [c for c in chain if c.contract_type == "put" and c.strike < spot]
    if not calls or not puts:
        raise RuntimeError(f"Could not find OTM strikes for {symbol} on {today}")
    # nearest OTM
    call = min(calls, key=lambda c: c.strike - spot)
    put = min(puts, key=lambda p: spot - p.strike)
    logging.info(
        "OTM Strangle for %s: call %s (@%.2f) & put %s (@%.2f)",
        symbol, call.symbol, call.strike, put.symbol, put.strike
    )
    return call.symbol, put.symbol


def submit_strangle(
    trading: TradingClient, call_symbol: str, put_symbol: str, qty: int
):
    """
    Submits a multi-leg market BUY strangle (call + put).
    """
    call_leg = OptionLegRequest(
        symbol=call_symbol,
        ratio_qty=qty,
        side=OrderSide.BUY,
    )
    put_leg = OptionLegRequest(
        symbol=put_symbol,
        ratio_qty=qty,
        side=OrderSide.BUY,
    )
    order = MarketOrderRequest(
        qty=qty,
        time_in_force=TimeInForce.DAY,
        order_class=OrderClass.MLEG,
        legs=[call_leg, put_leg],
    )
    resp = TradingClient.submit_order(trading, order)
    logging.info("Submitted STRANGLE BUY order %s; status=%s", resp.id, resp.status)
    return resp


def monitor_and_exit_strangle(
    trading: TradingClient,
    option_client: OptionHistoricalDataClient,
    symbols: list[str],
    entry_price_sum: float,
    qty: int,
    target_pct: float,
    poll_interval: float,
    exit_cutoff: dt_time,
):
    """
    Polls the legs' last trade prices, exits both when combined profit or at cutoff.
    """
    target_sum = entry_price_sum * (1 + target_pct)
    logging.info(
        "Monitoring strangle %s: entry_sum=%.4f, target_sum=%.4f, cutoff=%s",
        symbols, entry_price_sum, target_sum, exit_cutoff.isoformat(),
    )

    while True:
        now = datetime.now()
        if now.time() >= exit_cutoff:
            logging.warning("Cutoff time reached (%s). Exiting...", exit_cutoff)
            break
        prices = []
        for sym in symbols:
            trade = OptionHistoricalDataClient.get_option_latest_trade(option_client, 
                OptionLatestTradeRequest(symbol_or_symbols=[sym])
            )[sym]
            prices.append(float(trade.price))
        current_sum = sum(prices)
        logging.debug("Current prices %s → %.4f (sum)", symbols, current_sum)
        if current_sum >= target_sum:
            logging.info("Target hit: %.4f >= %.4f", current_sum, target_sum)
            break
        time.sleep(poll_interval)

    # exit legs
    legs = [
        OptionLegRequest(symbol=sym, ratio_qty=qty, side=OrderSide.SELL)
        for sym in symbols
    ]
    exit_order = MarketOrderRequest(
        qty=qty,
        time_in_force=TimeInForce.DAY,
        order_class=OrderClass.MLEG,
        legs=legs,
    )
    resp = TradingClient.submit_order(trading, exit_order)
    logging.info("Submitted STRANGLE SELL exit order %s; status=%s", resp.id, resp.status)
    return resp

def monitor_and_exit(
    trading: TradingClient,
    option_client: OptionHistoricalDataClient,
    entry_symbol: str,
    entry_price: float,
    qty: int,
    target_pct: float,
    poll_interval: float,
    exit_cutoff: dt_time,
):
    target_price = entry_price * (1 + target_pct)
    logging.info(
        "Monitoring %s: entry=%.4f, target=%.4f, cutoff=%s",
        entry_symbol,
        entry_price,
        target_price,
        exit_cutoff.isoformat(),
    )

    while True:
        now = datetime.now()
        if now.time() >= exit_cutoff:
            logging.warning("Cutoff time reached (%s). Exiting...", exit_cutoff)
            break

        trade = OptionHistoricalDataClient.get_option_latest_trade(option_client, 
            OptionLatestTradeRequest(symbol_or_symbols=[entry_symbol])
        )[entry_symbol]
        current = float(trade.price)
        logging.debug("%s @ %.4f", entry_symbol, current)
        if current >= target_price:
            logging.info("Target hit: %.4f >= %.4f", current, target_price)
            break

        time.sleep(poll_interval)

    leg = OptionLegRequest(
        symbol=entry_symbol,
        qty=qty,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.DAY,
    )
    exit_order = MarketOrderRequest(
        symbol=entry_symbol,
        qty=qty,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.DAY,
        legs=[leg],
    )
    resp = TradingClient.submit_order(trading, exit_order)
    logging.info("Submitted SELL exit order %s; status=%s", resp.id, resp.status)
    return resp


def main():
    setup_logging()
    cfg = Settings()
    logging.info(
        "Starting 0DTE strategy on %s (paper=%s)",
        cfg.UNDERLYING,
        cfg.PAPER,
    )

    trading = TradingClient(
        cfg.ALPACA_API_KEY, cfg.ALPACA_API_SECRET, paper=cfg.PAPER
    )
    stock_hist = StockHistoricalDataClient(
        api_key=cfg.ALPACA_API_KEY,
        secret_key=cfg.ALPACA_API_SECRET,
        sandbox=cfg.PAPER,
    )
    option_hist = OptionHistoricalDataClient(
        api_key=cfg.ALPACA_API_KEY,
        secret_key=cfg.ALPACA_API_SECRET,
        sandbox=cfg.PAPER,
    )

    # Choose OTM strikes for strangle
    call_sym, put_sym = choose_otm_strangle_contracts(trading, stock_hist, cfg.UNDERLYING)
    # Submit OTM strangle
    strangle_resp = submit_strangle(trading, call_sym, put_sym, cfg.QTY)
    # Sum entry prices from both legs
    try:
        entry_price_sum = sum(float(leg.filled_avg_price or 0.0) for leg in strangle_resp.legs)
    except Exception:
        logging.error("Failed to compute entry price sum from response legs. Exiting.")
        return
    if entry_price_sum <= 0:
        logging.error("Entry price sum is zero or invalid. Exiting.")
        return

    # Monitor & exit strangle
    monitor_and_exit_strangle(
        trading,
        option_hist,
        symbols=[call_sym, put_sym],
        entry_price_sum=entry_price_sum,
        qty=cfg.QTY,
        target_pct=cfg.PROFIT_TARGET_PCT,
        poll_interval=cfg.POLL_INTERVAL,
        exit_cutoff=cfg.EXIT_CUTOFF,
    )
    logging.info("Strategy run complete.")


if __name__ == "__main__":
    main()
