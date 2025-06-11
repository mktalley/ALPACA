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
from datetime import date, datetime, time as dt_time, timedelta
from zoneinfo import ZoneInfo
from typing import Dict
import os
import threading
# Configure timezone for Eastern Time logging
os.environ['TZ'] = 'America/New_York'
time.tzset()
import pydantic
from pydantic import SecretStr
# Workaround: alias Secret to SecretStr so pydantic-settings can import Secret
pydantic.Secret = SecretStr





import pydantic
from pydantic import SecretStr
# Workaround: alias Secret to SecretStr so pydantic-settings can import Secret
setattr(pydantic, 'Secret', SecretStr)

# Prefer pydantic BaseSettings; fall back to pydantic_settings only if available without errors
try:
    # pydantic_settings may not be compatible with the installed pydantic version
    from pydantic_settings import BaseSettings
except Exception:
    from pydantic import BaseSettings



from pydantic import Field, field_validator

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

import math, requests, statistics
from alpaca.data.requests import StockBarsRequest, OptionSnapshotRequest
from alpaca.data.timeframe import TimeFrame

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
        dt_time(22, 45), description="Hard exit time (wall clock) if target not hit"
    )
    MAX_TRADES: int = Field(6, description="Maximum number of strangle trades per day")
    TRADE_START: dt_time = Field(
        dt_time(9, 45), description="Earliest time to enter trades (ET)"
    )
    TRADE_END: dt_time = Field(
        dt_time(12, 30), description="Latest time to enter trades (ET)"
    )

    class Config:
        env_file = "apps/.env"
        env_file_encoding = "utf-8"

    @field_validator("EXIT_CUTOFF", mode="before")
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
    option_resp = TradingClient.get_option_contracts(trading,
        GetOptionContractsRequest(
            underlying_symbols=[symbol],
            expiration_date=today,
            limit=500,
        )
    )
    if isinstance(option_resp, list):
        chain = option_resp
    else:
        chain = option_resp.option_contracts or []
    spot = get_underlying_price(stock_client, symbol)

    # normalize contract type and strike attributes
    def ct(c):
        if hasattr(c, 'contract_type'):
            return c.contract_type
        t = getattr(c, 'type', None)
        return t.value if hasattr(t, 'value') else t

    def st(c):
        if hasattr(c, 'strike'):
            return c.strike
        return getattr(c, 'strike_price', 0)

    # filter OTM calls and puts
    calls = [c for c in chain if ct(c) == 'call' and st(c) > spot]
    puts = [c for c in chain if ct(c) == 'put' and st(c) < spot]
    if not calls or not puts:
        raise RuntimeError(f"Could not find OTM strikes for {symbol} on {today}")

    # nearest OTM
    call = min(calls, key=lambda c: st(c) - spot)
    put = min(puts, key=lambda p: spot - st(p))
    logging.info(
        "OTM Strangle for %s: call %s (@%.2f) & put %s (@%.2f)",
        symbol, call.symbol, st(call), put.symbol, st(put)
    )
    return call.symbol, put.symbol
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



def run_strangle(
    trading: TradingClient,
    stock_client: StockHistoricalDataClient,
    option_client: OptionHistoricalDataClient,
    symbol: str,
    qty: int,
    target_pct: float,
    poll_interval: float,
    exit_cutoff: dt_time,
):
    """
    Helper to place and monitor a single OTM strangle.
    """
    try:
        call_sym, put_sym = choose_otm_strangle_contracts(trading, stock_client, symbol)
        strangle_resp = submit_strangle(trading, call_sym, put_sym, qty)
        start_time = time.time()
        fill_timeout = 30  # seconds
        filled_prices = []
        for leg in strangle_resp.legs:
            leg_id = leg.id
            logging.info("Waiting for fill on leg %s", leg_id)
            while True:
                current_leg = trading.get_order_by_id(leg_id)
                avg_price = current_leg.filled_avg_price
                if avg_price is not None and float(avg_price) > 0:
                    filled_prices.append(float(avg_price))
                    logging.info("Leg %s filled at %.4f", leg_id, float(avg_price))
                    break
                if time.time() - start_time > fill_timeout:
                    logging.error("Timeout waiting for fill on leg %s. Aborting this strangle.", leg_id)
                    return
                time.sleep(1)
        entry_price_sum = sum(filled_prices)
        logging.info("Entry prices %s → entry_price_sum=%.4f", filled_prices, entry_price_sum)
        monitor_and_exit_strangle(
            trading,
            option_client,
            symbols=[call_sym, put_sym],
            entry_price_sum=entry_price_sum,
            qty=qty,
            target_pct=target_pct,
            poll_interval=poll_interval,
            exit_cutoff=exit_cutoff,
        )
    except Exception as e:
        logging.error("Error in run_strangle: %s", e, exc_info=True)


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
        sandbox=False,  # always use live market data endpoint
    )
    option_hist = OptionHistoricalDataClient(
        api_key=cfg.ALPACA_API_KEY,
        secret_key=cfg.ALPACA_API_SECRET,
        sandbox=False  # use live data endpoint,
    )
    # Wait until market is open before submitting orders
    clock = trading.get_clock()
    while not clock.is_open:
        logging.warning(
            "Market is closed (next open at %s). Sleeping 60 seconds...",
            clock.next_open.isoformat(),
        )
        time.sleep(60)
        clock = trading.get_clock()



    # Hybrid scheduling: fixed anchor trades + event-triggered trades
    today = date.today()
    trade_start_dt = datetime.combine(today, cfg.TRADE_START)
    trade_end_dt = datetime.combine(today, cfg.TRADE_END)
    mid_dt = trade_start_dt + (trade_end_dt - trade_start_dt) / 2

    EVENT_MOVE_PCT = 0.005

    threads: list[threading.Thread] = []
    trades_done = 0
    # initial spot price
    last_spot = get_underlying_price(stock_hist, cfg.UNDERLYING)

    anchors = [trade_start_dt, mid_dt, trade_end_dt]
    for i, anchor in enumerate(anchors):
        if trades_done >= cfg.MAX_TRADES:
            break
        now = datetime.now()
        if now < anchor:
            secs = (anchor - now).total_seconds()
            logging.info("Sleeping until anchor trade #%d at %s", trades_done + 1, anchor.time())
            time.sleep(secs)
        logging.info("Anchor trade #%d at %s", trades_done + 1, anchor.time())
        t = threading.Thread(
            target=run_strangle,
            args=(trading, stock_hist, option_hist, cfg.UNDERLYING, cfg.QTY, cfg.PROFIT_TARGET_PCT, cfg.POLL_INTERVAL, cfg.EXIT_CUTOFF),
            daemon=True,
        )
        t.start()
        threads.append(t)
        trades_done += 1
        last_spot = get_underlying_price(stock_hist, cfg.UNDERLYING)

        # Event-triggered trades until next anchor
        if i < len(anchors) - 1:
            next_anchor = anchors[i + 1]
            while trades_done < cfg.MAX_TRADES and datetime.now() < next_anchor:
                time.sleep(cfg.POLL_INTERVAL)
                curr_spot = get_underlying_price(stock_hist, cfg.UNDERLYING)
                pct_move = abs(curr_spot - last_spot) / last_spot
                if pct_move >= EVENT_MOVE_PCT:
                    logging.info("Price moved %.2f%%, triggering event trade #%d", pct_move * 100, trades_done + 1)
                    t = threading.Thread(
                        target=run_strangle,
                        args=(trading, stock_hist, option_hist, cfg.UNDERLYING, cfg.QTY, cfg.PROFIT_TARGET_PCT, cfg.POLL_INTERVAL, cfg.EXIT_CUTOFF),
                        daemon=True,
                    )
                    t.start()
                    threads.append(t)
                    trades_done += 1
                    last_spot = curr_spot

    # Wait for all strangle threads to complete
    for t in threads:
        t.join()
    logging.info("All %d trades completed", trades_done)


if __name__ == "__main__":
    main()
