#!/usr/bin/env python3
from __future__ import annotations
import sys
import os
# Remove potentially invalid UNDERLYING env var to prevent parsing errors
os.environ.pop("UNDERLYING", None)

# Ensure project root is on PYTHONPATH so the 'alpaca' package is importable
file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(file_dir, os.pardir, os.pardir))
from pathlib import Path
import yaml

# Load YAML overrides (config/zero_dte.yaml) into environment so pydantic Settings can see them.
def _apply_yaml_overrides():
    cfg_path = Path("config/zero_dte.yaml")
    if not cfg_path.exists():
        return
    try:
        data = yaml.safe_load(cfg_path.read_text()) or {}
    except Exception as e:
        print("Warning: could not parse YAML config", cfg_path, e)
        return
    for key, val in data.items():
        # Lists → comma-separated; bool/float/int → str
        if isinstance(val, list):
            os.environ.setdefault(key, ",".join(map(str, val)))
        else:
            os.environ.setdefault(key, str(val))

_apply_yaml_overrides()

if project_root not in sys.path:
    sys.path.insert(0, project_root)


"""
0DTE OTM Strangle (Call + Put) & Target-Exit Example
Designed for institutional clients:
  • Configurable via apps/zero_dte/.env (copy apps/zero_dte/.env.example)
  • Structured with pydantic BaseSettings
  • Logging, retries, time checks
  • Uses STOP_LOSS_PCT and RISK_PCT_PER_TRADE for dynamic sizing
  • Easy to extend to multi-leg (MLEG) or other underlyings
"""

import time
import logging
from logging.handlers import TimedRotatingFileHandler
from datetime import date, datetime, time as dt_time, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, List
import os
import threading
import signal  # for graceful shutdown
import argparse
import re  # regex for log analysis
# Configure timezone for Eastern Time logging
import random  # for jittering anchor times
os.environ['TZ'] = 'America/New_York'
time.tzset()
daily_pnl: float = 0.0  # cumulative P&L for the trading day
daily_start_equity: float = 0.0  # starting equity for the trading day

import pydantic
from pydantic import SecretStr
# Workaround: alias Secret to SecretStr so pydantic-settings can import Secret
pydantic.Secret = SecretStr


import pandas as pd




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



from pydantic import Field, field_validator, model_validator

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
    OptionLatestQuoteRequest,
    OptionSnapshotRequest,
)

# counters for monitoring filtered trades
_spread_skipped_count: int = 0
_iv_skipped_count: int = 0


def submit_complex_order(
    trading: TradingClient,
    legs: list[OptionLegRequest],
    qty: int,
) -> object:
    """
    Submit a multi-leg market order (MLEG) with the given legs and quantity.
    """
    order = MarketOrderRequest(
        qty=qty,
        time_in_force=TimeInForce.DAY,
        order_class=OrderClass.MLEG,
        legs=legs,
    )
    resp = TradingClient.submit_order(trading, order)
    logging.info(
        "Submitted COMPLEX order %s; status=%s",
        getattr(resp, 'id', None),
        getattr(resp, 'status', None),
    )
    return resp


from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.historical.option import OptionHistoricalDataClient

import math, requests, statistics
from alpaca.data.requests import StockBarsRequest, OptionSnapshotRequest
from alpaca.data.timeframe import TimeFrame

# Placeholder for two-phase strategy; will import dynamically in main or be stubbed in tests
run_two_phase = None

# Simple retry helper for external API calls with circuit breaker
_failure_count = 0
_circuit_open_until = 0.0

def safe_api_call(func, *args, retries: int = 3, backoff: float = 1.0, **kwargs):
    """Call func(*args, **kwargs) with retries, exponential backoff on exception, and circuit breaker on repeated failures."""
    global _failure_count, _circuit_open_until
    now = time.time()
    # Circuit open: reject calls until cooldown expires
    if now < _circuit_open_until:
        raise RuntimeError(
            f"Circuit is open until {datetime.fromtimestamp(_circuit_open_until)}; skipping API call {func.__name__}"
        )
    attempt = 0
    while True:
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            attempt += 1
            if attempt > retries:
                # after retry exhaustion, count a failure
                _failure_count += 1
                cfg = Settings()
                threshold = getattr(cfg, 'FAILURE_THRESHOLD', 5)
                cooldown = getattr(cfg, 'COOLDOWN_SEC', 300.0)
                if _failure_count >= threshold:
                    _circuit_open_until = now + cooldown
                    logging.critical(
                        "Circuit breaker tripped for %s: pausing API calls for %.1fs", func.__name__, cooldown
                    )
                logging.error("API call %s failed after %d attempts: %s", func.__name__, retries, e)
                raise
            delay = backoff * (2 ** (attempt - 1))
            logging.warning(
                "API call %s failed on attempt %d/%d: %s – retrying in %.1fs",
                func.__name__, attempt, retries, e, delay,
            )
            time.sleep(delay)
            continue
        else:
            # reset failure count on success
            _failure_count = 0
            return result


# Global shutdown event
type_shutdown = threading.Event()

# Helper to update and check daily P&L thresholds
def _update_daily_pnl(pnl: float, cfg) -> None:
    global daily_pnl
    daily_pnl += pnl
    logging.info("Updated DAILY P&L: %.4f", daily_pnl)
    # Check thresholds
    if cfg.DAILY_PROFIT_TARGET > 0 and daily_pnl >= cfg.DAILY_PROFIT_TARGET:
        logging.info("Daily profit target reached (%.4f >= %.4f); shutting down further trades.", daily_pnl, cfg.DAILY_PROFIT_TARGET)
        type_shutdown.set()
    if cfg.DAILY_LOSS_LIMIT > 0 and daily_pnl <= -cfg.DAILY_LOSS_LIMIT:
        logging.info("Daily loss limit reached (%.4f <= -%.4f); shutting down further trades.", daily_pnl, cfg.DAILY_LOSS_LIMIT)
        type_shutdown.set()

# Graceful shutdown handler

# Wrapper to execute trade function and update daily P&L
def _execute_trade_and_update(target, args, cfg) -> None:
    logging.info("Executing trade %s with args %s", target.__name__, args)
    try:
        result = target(*args)
        if result is None:
            logging.warning("Trade %s returned None; cannot compute P&L", target.__name__)
            return
        try:
            pnl_val = float(result)
        except (TypeError, ValueError):
            logging.warning("Trade %s returned non-numeric result: %s", target.__name__, result)
            return
        _update_daily_pnl(pnl_val, cfg)

        # Check percent drawdown circuit breaker
        if daily_start_equity > 0:
            drawdown_pct = -daily_pnl / daily_start_equity
            if drawdown_pct >= cfg.DAILY_DRAWDOWN_PCT:
                logging.critical(
                    "Daily drawdown exceeded %.2f%% (%.2f / %.2f); initiating shutdown and liquidation.",
                    cfg.DAILY_DRAWDOWN_PCT * 100,
                    -daily_pnl,
                    daily_start_equity,
                )
                # Liquidate all open positions immediately
                try:
                    trading_client = args[0]
                    trading_client.close_all_positions()
                    logging.info("All open positions closed due to drawdown limit.")
                except Exception as e:
                    logging.error("Error closing positions on drawdown: %s", e, exc_info=True)
                type_shutdown.set()
    except Exception as e:
        logging.error("Error executing trade %s: %s", target.__name__, e, exc_info=True)
        return

def handle_signal(signum, frame):
    logging.info("Received signal %s; shutting down gracefully...", signum)
    type_shutdown.set()



class Settings(BaseSettings):
    
    @model_validator(mode='after')
    def validate_thresholds(cls, values):
        """
        Ensure STOP_LOSS_PCT is less than PROFIT_TARGET_PCT and RISK_PCT_PER_TRADE is between 0 and 1.
        """
        if values.STOP_LOSS_PCT >= values.PROFIT_TARGET_PCT:
            raise ValueError(f"STOP_LOSS_PCT ({values.STOP_LOSS_PCT}) must be less than PROFIT_TARGET_PCT ({values.PROFIT_TARGET_PCT})")
        if not (0 < values.RISK_PCT_PER_TRADE < 1):
            raise ValueError(f"RISK_PCT_PER_TRADE ({values.RISK_PCT_PER_TRADE}) must be between 0 and 1")
        return values

class Settings(BaseSettings):
    DAILY_PROFIT_TARGET: float = Field(0.0, description="Stop trading once we’ve made $X today (0 to disable)")
    DAILY_LOSS_LIMIT:    float = Field(0.0, description="Stop trading once we’ve lost $X today (0 to disable)")
    DAILY_DRAWDOWN_PCT: float = Field(0.03, description="Stop trading if cumulative drawdown >= X% of starting equity (default 3%)")
    DAILY_PROFIT_PCT: float = Field(0.0, description="Stop trading and liquidate when account equity increases by X% intraday (0 to disable)")

    ALPACA_API_KEY: str = ""
    ALPACA_API_SECRET: str = ""
    PAPER: bool = True

    UNDERLYING: List[str] = Field(["SPY"], description="Comma-separated list of underlying symbols to trade options on")
    @field_validator("UNDERLYING", mode="before")
    def split_underlying(cls, v):
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        elif isinstance(v, list):
            return v
        else:
            raise ValueError("UNDERLYING must be a string or list of strings")
    QTY: int = Field(10, description="Number of contracts per leg")
    PROFIT_TARGET_PCT: float = Field(
        0.65, description="Profit target as a percent of entry price (e.g. 0.65 = 65%)"
    )
    POLL_INTERVAL: float = Field(120.0, description="Seconds between price checks")
    EXIT_CUTOFF: dt_time = Field(
        dt_time(22, 45), description="Hard exit time (wall clock) if target not hit"
    )
    STOP_LOSS_PCT: float = Field(
        0.25, description="Max allowable loss as a percent of entry (e.g. 0.25 = 25%)"
    )
    RISK_PCT_PER_TRADE: float = Field(
        0.0025, description="Percent of equity to risk per trade (e.g. 0.0025 = 0.25%)"
    )
    MAX_LOSS_PER_TRADE: float = Field(
        1000.0, description="Maximum loss per trade (dollars)"
    )
    MAX_SPREAD_PCT: float = Field(
        0.02, description="Max bid-ask spread as pct of midpoint (e.g. 0.02 = 2%)"
    )
    MIN_IV: float = Field(
        0.10, description="Minimum implied volatility required (e.g. 0.10 = 10%)"
    )


    CONDOR_TARGET_PCT: float = Field(
        0.25, description="Profit target as a percent for the condor phase in two-phase strategy"
    )
    CONDOR_WING_SPREAD: int = Field(
        2, description="Wing width in strikes for the condor phase"
    )

    MAX_TRADES: int = Field(15, description="Maximum number of strangle trades per day")
    EVENT_MOVE_PCT: float = Field(
        0.0015, description="Price move percent to trigger event trades (e.g. 0.0015 = 0.15%)"
    )
    ANCHOR_MOVE_PCT_MAX: float = Field(
        0.006, description="Max 09:30-10:00 price range to allow anchor entries (e.g. 0.006 = 0.6%)"
    )
    IV_RANK_THRESHOLD: float = Field(
        0.5, description="Maximum allowed IV rank (0-1) for anchor and event trades"
    )

    TRADE_START: dt_time = Field(
        dt_time(10, 30), description="Earliest time to enter trades (ET)"
    )
    TRADE_END: dt_time = Field(
        dt_time(12, 30), description="Latest time to enter trades (ET)"
    )

    class Config:
        env_file = "apps/zero_dte/.env"
        env_file_encoding = "utf-8"

    @field_validator("EXIT_CUTOFF", mode="before")
    VIX_HIGH: float = Field(18.0, description="VIX threshold to switch to deep OTM strikes")

    def parse_exit_time(cls, v):
        if isinstance(v, str):
            return dt_time.fromisoformat(v)
        return v



def setup_logging():
    """
    Configure logging for 0DTE bot:
      - StreamHandler at DEBUG level to console.
      - TimedRotatingFileHandler at INFO level to apps/logs/zero_dte.log.
    """
    # Prepare log directory
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logs_dir = os.path.join(root, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, 'zero_dte.log')

    # Get root logger and clear existing handlers
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    # Console handler: DEBUG and above
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-8s [%(threadName)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S %Z%z"
    ))
    logger.addHandler(stream_handler)

    # File handler: INFO and above, rotate daily, keep 7 days
    file_handler = TimedRotatingFileHandler(
        filename=log_path, when='midnight', interval=1,
        backupCount=7, encoding='utf-8'
    )
    file_handler.suffix = "%Y-%m-%d"
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-8s [%(threadName)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S %Z%z"
    ))
    logger.addHandler(file_handler)




def get_underlying_price(stock_client: StockHistoricalDataClient, symbol: str) -> float:
    """Fetch latest spot price with retries."""
    resp = safe_api_call(
        stock_client.get_stock_latest_trade,
        StockLatestTradeRequest(symbol_or_symbols=[symbol])
    )
    trade = resp[symbol]
    return float(trade.price)

# ---------------------------------------------------------------------------
# Volatility helpers
# ---------------------------------------------------------------------------

def _morning_move_pct(stock_client: StockHistoricalDataClient, symbol: str, ref_date: date | None = None) -> float:
    """Return 09:30–10:00 ET high-low range as percent of 09:30 open."""
    if ref_date is None:
        ref_date = date.today()
    tz = ZoneInfo("America/New_York")
    start_et = datetime.combine(ref_date, dt_time(9, 30), tzinfo=tz)
    end_et   = datetime.combine(ref_date, dt_time(10, 0), tzinfo=tz)
    start_utc= start_et.astimezone(ZoneInfo("UTC"))
    end_utc  = end_et.astimezone(ZoneInfo("UTC"))
    bars = safe_api_call(
        stock_client.get_stock_bars,
        StockBarsRequest(symbol_or_symbols=[symbol], timeframe=TimeFrame.Minute, start=start_utc, end=end_utc)
    ).df
    if bars.empty:
        return 0.0


    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(symbol, level="symbol")
    open_price = bars.iloc[0]["open"]
    hi = bars["high"].max()
    lo = bars["low"].min()
    return (hi - lo) / open_price
def _iv_rank(stock_client: StockHistoricalDataClient, symbol: str, lookback_days: int = 30) -> float:
    """Return a placeholder IV rank between 0 and 1.
    TODO: Implement proper IV rank using option IV history.
    """
    logging.debug("IV rank placeholder for %s", symbol)
    return 0.3


def _should_enter_trade(stock_client: StockHistoricalDataClient,
                        option_hist: OptionHistoricalDataClient,
                        symbol: str,
                        cfg: Settings,
                        is_anchor: bool) -> bool:
    """Return True if context filters allow a new entry."""
    try:
        iv_rank = _iv_rank(stock_client, symbol)
        if iv_rank > cfg.IV_RANK_THRESHOLD:
            logging.info("%s IV rank %.2f > threshold %.2f; skipping trade", symbol, iv_rank, cfg.IV_RANK_THRESHOLD)
            return False
    except Exception as e:
        logging.warning("IV rank check failed for %s: %s", symbol, e)

    if is_anchor:
        try:
            move_pct = _morning_move_pct(stock_client, symbol)
            if move_pct > cfg.ANCHOR_MOVE_PCT_MAX:
                logging.info("%s morning move %.3f > %.3f; skipping anchor trade", symbol, move_pct, cfg.ANCHOR_MOVE_PCT_MAX)
                return False
        except Exception as e:
            logging.warning("Morning move check failed for %s: %s", symbol, e)
    return True

    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(symbol, level="symbol")
    open_price = bars.iloc[0]["open"]
    hi = bars["high"].max()
    lo = bars["low"].min()
    return (hi - lo) / open_price

    """Fetch spot price with retries."""
    resp = safe_api_call(
        stock_client.get_stock_latest_trade,
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
    return call.symbol, put.symbol


def choose_iron_condor_contracts(
    trading: TradingClient,
    stock_client: StockHistoricalDataClient,
    symbol: str,
    wing_spread: int,
) -> tuple[str, str, str, str]:
    """
    Fetch today’s option chain and build an iron condor with defined risk.
    Returns symbols for short call, short put, long call, long put.
    """
    today = date.today().isoformat()
    opt_resp = TradingClient.get_option_contracts(
        trading,
        GetOptionContractsRequest(
            underlying_symbols=[symbol],
            expiration_date=today,
            limit=500,
        ),
    )
    if isinstance(opt_resp, list):
        chain = opt_resp
    else:
        chain = opt_resp.option_contracts or []
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

    # separate OTM calls and puts
    calls = sorted([c for c in chain if ct(c) == 'call' and st(c) > spot], key=lambda c: st(c))
    puts = sorted([p for p in chain if ct(p) == 'put' and st(p) < spot], key=lambda p: st(p), reverse=True)
    # ensure enough strikes for wings
    if len(calls) <= wing_spread or len(puts) <= wing_spread:
        raise RuntimeError(f"Insufficient strikes to build iron condor for {symbol}")

    short_call = calls[0]
    long_call = calls[wing_spread]
    short_put = puts[0]
    long_put = puts[wing_spread]
    logging.info(
        "Iron condor for %s: short_call=%s, short_put=%s, long_call=%s, long_put=%s",
        symbol,
        short_call.symbol,
        short_put.symbol,
        long_call.symbol,
        long_put.symbol,
    )
    return short_call.symbol, short_put.symbol, long_call.symbol, long_put.symbol


def submit_iron_condor(
    trading: TradingClient,
    short_call: str,
    short_put: str,
    long_call: str,
    long_put: str,
    qty: int,
) -> object:
    """
    Submit a defined-risk iron condor (sell short call/put, buy wings).
    """
    legs = [
        OptionLegRequest(symbol=short_call, ratio_qty=1, side=OrderSide.SELL),
        OptionLegRequest(symbol=short_put, ratio_qty=1, side=OrderSide.SELL),
        OptionLegRequest(symbol=long_call, ratio_qty=1, side=OrderSide.BUY),
        OptionLegRequest(symbol=long_put, ratio_qty=1, side=OrderSide.BUY),
    ]
    order = MarketOrderRequest(
        qty=qty,
        time_in_force=TimeInForce.DAY,
        order_class=OrderClass.MLEG,
        legs=legs,
    )
    resp = TradingClient.submit_order(trading, order)
    logging.info(
        "Submitted IRON CONDOR entry order %s; status=%s",
        getattr(resp, 'id', None),
        getattr(resp, 'status', None),
    )
    return resp

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
        ratio_qty=1,
        side=OrderSide.BUY,
    )
    put_leg = OptionLegRequest(
        symbol=put_symbol,
        ratio_qty=1,
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



def monitor_and_exit_strangle(
    trading: TradingClient,
    option_client: OptionHistoricalDataClient,
    symbols: list[str],
    entry_price_sum: float,
    qty: int,
    target_pct: float,
    poll_interval: float,
    exit_cutoff: dt_time,
) -> float:
    """
    Monitor a multi-leg strangle and exit on profit target or stop-loss using a single MLEG order.
    """
    try:
        cfg = Settings()
        cfg = Settings()
    except Exception as e:
        logging.warning("Failed to load Settings in monitor_and_exit_strangle; using default STOP_LOSS_PCT=0.0 (%s)", e)
        class _CfgDefault:
            STOP_LOSS_PCT = 0.0
        cfg = _CfgDefault()
    target_sum = entry_price_sum * (1 + target_pct)
    stop_loss_sum = entry_price_sum * (1 - cfg.STOP_LOSS_PCT)
    global strangle_pnl_history
    strangle_pnl_history = []
    logging.info(
        "Monitoring strangle %s: entry_sum=%.4f, target_sum=%.4f, stop_loss_sum=%.4f, cutoff=%s",
        symbols, entry_price_sum, target_sum, stop_loss_sum, exit_cutoff.isoformat(),
    )

    while True:
        now = datetime.now()
        if now.time() >= exit_cutoff:
            logging.warning("Strangle cutoff time reached (%s). Exiting monitor.", exit_cutoff)
            break
        try:
            trades = OptionHistoricalDataClient.get_option_latest_trade(
                option_client,
                OptionLatestTradeRequest(symbol_or_symbols=symbols)
            )
            prices = [float(trades[sym].price) for sym in symbols]
        except requests.exceptions.RequestException as err:
            logging.warning(
                "Error fetching strangle prices %s: %s – retrying in %.1fs",
                symbols, err, poll_interval,
            )
            time.sleep(poll_interval)
            continue
                # Tighter logging: record equity & PnL snapshot
        ts = datetime.now()
        try:
            account = TradingClient.get_account(trading)
            equity = float(account.equity)
        except Exception as e:
            logging.warning(
                "Unable to fetch equity during strangle monitor: %s", e
            )
            equity = None
        current_sum = sum(prices)
        trade_pnl = (current_sum - entry_price_sum) * qty
        if equity is not None:
            logging.debug(
                "Current equity=%.2f, trade PnL=%.2f", equity, trade_pnl
            )
        else:
            logging.debug(
                "Trade PnL=%.2f", trade_pnl
            )
        strangle_pnl_history.append((ts, trade_pnl))
        try:
            account = TradingClient.get_account(trading)
            equity = float(account.equity)
            pnl = equity - daily_start_equity
            logging.debug(
                "Current equity=%.2f, PnL=%.2f", equity, pnl
            )
        except Exception as e:
            logging.warning(
                "Unable to fetch equity during strangle monitor: %s", e
            )
        current_sum = sum(prices)
        logging.debug(
            "Current strangle sum %s → %.4f", symbols, current_sum
        )
        if current_sum >= target_sum:
            logging.info(
                "Strangle target hit: %.4f >= %.4f", current_sum, target_sum
            )
            break
        if current_sum <= stop_loss_sum:
            logging.info(
                "Strangle stop-loss hit: %.4f <= %.4f", current_sum, stop_loss_sum
            )
            break
        time.sleep(poll_interval)

    # Build exit legs and submit complex order
    legs = [
        OptionLegRequest(symbol=sym, ratio_qty=1, side=OrderSide.SELL)
        for sym in symbols
    ]
    resp = submit_complex_order(trading, legs, qty)
    
    # Compute and return trade PnL
    try:
        # Use last observed option prices to compute exit sum
                # Fetch latest exit trade prices for each leg
        exit_trades = OptionHistoricalDataClient.get_option_latest_trade(
            option_client,
            OptionLatestTradeRequest(symbol_or_symbols=symbols)
        )
        exit_prices = [float(exit_trades[s].price) for s in symbols]
        exit_price_sum = sum(exit_prices)
        # PnL per contract per share: exit - entry; convert to USD: *qty contracts *100 shares
        # PnL per contract per share: exit - entry; convert to USD: *qty contracts *100 shares
        pnl = (exit_price_sum - entry_price_sum) * qty * 100  # profit from net credit (sell - buy)
        logging.info("Exit PnL=%.2f", pnl)
    except Exception as e:
        logging.warning(
            "Error computing exit-based PnL; falling back to equity difference (%s)", e
        )
        try:
            account_now = TradingClient.get_account(trading)
            equity_now = float(account_now.equity)
            pnl = equity_now - daily_start_equity
            logging.info("Fallback PnL from equity change=%.2f", pnl)
        except Exception as e2:
            pnl = 0.0
            logging.warning(
                "Unable to compute fallback equity-based PnL; returning 0.0 (%s)", e2
            )
    return pnl



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
        try:
            cfg = Settings()  # load settings for pre-trade filters
        except Exception as e:
            logging.warning("Failed to load Settings in run_strangle; using fallback defaults: %s", e)
            class _CfgDefault:
                MAX_SPREAD_PCT = 0.02
                MIN_IV = 0.10
                STOP_LOSS_PCT = 0.20
                RISK_PCT_PER_TRADE = 0.01
            cfg = _CfgDefault()

        # Dynamic position sizing based on risk per trade
        # Pre-trade bid-ask spread filter
        try:
            latest_quotes = OptionHistoricalDataClient.get_option_latest_quote(
                option_client,
                OptionLatestQuoteRequest(symbol_or_symbols=[call_sym, put_sym])
            )
            for s, q in latest_quotes.items():
                spread = q.ask_price - q.bid_price
                midpoint = (q.ask_price + q.bid_price) / 2
                spread_pct = spread / midpoint if midpoint > 0 else float('inf')
                if spread_pct > cfg.MAX_SPREAD_PCT:
                    global _spread_skipped_count
                    _spread_skipped_count += 1
                    logging.warning(
                        "Bid-ask spread %.2f%% for %s > MAX_SPREAD_PCT %.2f%%; skipping strangle.",
                        spread_pct * 100,
                        s,
                        cfg.MAX_SPREAD_PCT * 100,
                    )
                    logging.info(
                        "Total spread-skipped count: %d", _spread_skipped_count
                    )
                    return
        except Exception as e:
            logging.warning(
                "Error fetching quotes for spread filter: %s; proceeding anyway.",
                e,
            )

        # Pre-trade implied volatility filter
        try:
            snapshots = OptionHistoricalDataClient.get_option_snapshot(
                option_client,
                OptionSnapshotRequest(symbol_or_symbols=[call_sym, put_sym])
            )
            for s, snap in snapshots.items():
                iv = snap.implied_volatility
                if iv is None or iv < cfg.MIN_IV:
                    global _iv_skipped_count
                    _iv_skipped_count += 1
                    logging.warning(
                        "IV %.2f%% for %s < MIN_IV %.2f%%; skipping strangle.",
                        (iv or 0.0) * 100,
                        s,
                        cfg.MIN_IV * 100,
                    )
                    logging.info(
                        "Total IV-skipped count: %d", _iv_skipped_count
                    )
                    return
        except Exception as e:
            logging.warning(
                "Error fetching snapshots for IV filter: %s; proceeding anyway.", e
            )



        try:
            # fetch current equity
            account = TradingClient.get_account(trading)
            equity = float(account.equity)
            # estimate option credit for sizing (credit from selling both legs)
            trades = OptionHistoricalDataClient.get_option_latest_trade(
                option_client,
                OptionLatestTradeRequest(symbol_or_symbols=[call_sym, put_sym])
            )
            pr_call = float(trades[call_sym].price)
            pr_put = float(trades[put_sym].price)
            entry_credit_est = pr_call + pr_put
                
            risk_per_contract = entry_credit_est * cfg.STOP_LOSS_PCT
            allowed_risk = equity * cfg.RISK_PCT_PER_TRADE
            max_contracts = int(allowed_risk / risk_per_contract) if risk_per_contract > 0 else 0
            original_qty = qty
            qty = min(original_qty, max_contracts) if max_contracts > 0 else 0
            if qty < 1:
                logging.warning(
                    "Risk per trade too low for strangle (allowed %.2f, per contract %.2f), skipping strangle.",
                    allowed_risk,
                    risk_per_contract,
                )
                return
            logging.info(
                "Strangle sizing: entry_credit_est=%.4f, risk_per_contract=%.4f, allowed_risk=%.4f -> qty=%d (orig %d)",
                entry_credit_est,
                risk_per_contract,
                allowed_risk,
                qty,
                original_qty,
            )
            # Pre-check options buying power against estimated margin requirement for naked strangle
            try:
                account_bp = TradingClient.get_account(trading)
                options_bp = float(account_bp.options_buying_power)
                # margin requirement = premium credit * qty * 100
                required_margin = entry_credit_est * qty * 100
                if options_bp < required_margin:
                    logging.warning(
                        "Insufficient options buying power for naked strangle: required ~$%.2f, available $%.2f; skipping strangle.",
                        required_margin,
                        options_bp,
                    )
                    return
                logging.info(
                    "Proceeding with naked strangle; estimated margin requirement $%.2f <= available $%.2f",
                    required_margin,
                    options_bp,
                )
            except Exception as e:
                logging.warning("Error checking options buying power: %s", e)

        except Exception as e:
            logging.warning(
                "Error computing dynamic position size in strangle: %s; using default qty=%d",
                e,
                qty,
            )
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
        pnl = monitor_and_exit_strangle(
            trading,
            option_client,
            symbols=[call_sym, put_sym],
            entry_price_sum=entry_price_sum,
            qty=qty,
            target_pct=target_pct,
            poll_interval=poll_interval,
            exit_cutoff=exit_cutoff,
        )
        logging.info("Strangle complete for %s; PnL=%.4f", symbol, pnl)
        return pnl
    except Exception as e:
        logging.error("Error in run_strangle: %s", e, exc_info=True)



def schedule_for_symbol(
    symbol: str,
    cfg: Settings,
    trading: TradingClient,
    stock_hist: StockHistoricalDataClient,
    option_hist: OptionHistoricalDataClient,
    strategy: str,
):
    """Schedule and execute trades for a single symbol."""
    logging.info("Scheduling trades for symbol %s", symbol)
    # Prepare anchor times for today
    today = date.today()
    trade_start_dt = datetime.combine(today, cfg.TRADE_START)
    # Use the earlier of TRADE_END or EXIT_CUTOFF for final anchor to avoid near-expiration entries
    anchor_end_time = min(cfg.TRADE_END, cfg.EXIT_CUTOFF)
    trade_end_dt = datetime.combine(today, anchor_end_time)
    duration = trade_end_dt - trade_start_dt
    # Generate anchor times with ±5 min jitter (clamped to trading window)
    raw_anchors = [
        trade_start_dt + duration * frac
        for frac in (0, 0.25, 0.5, 0.75, 1)
    ]
    anchors = []
    for base in raw_anchors:
        jitter_offset = timedelta(seconds=random.uniform(-300, 300))
        jittered = base + jitter_offset
        if jittered < trade_start_dt:
            jittered = trade_start_dt
        elif jittered > trade_end_dt:
            jittered = trade_end_dt
        anchors.append(jittered)
    threads: list[threading.Thread] = []
    trades_done = 0
    # initial spot price
    last_spot = get_underlying_price(stock_hist, symbol)

    def _build_args(qty: int) -> tuple:
        base = (
            trading,
            stock_hist,
            option_hist,
            symbol,
            qty,
            cfg.PROFIT_TARGET_PCT,
            cfg.POLL_INTERVAL,
            cfg.EXIT_CUTOFF,
        )
        if strategy == "two_phase":
            return base + (cfg.CONDOR_TARGET_PCT,)
        return base

    # Determine thread target
    if strategy == "two_phase":
        if run_two_phase is None:
            from apps.zero_dte.two_phase import run_two_phase as _rp
            globals()["run_two_phase"] = _rp
        thread_target = run_two_phase
    else:
        thread_target = run_strangle

    anchor_args = _build_args(cfg.QTY)
    event_args  = _build_args(max(1, cfg.QTY // 2))

    event_trade_since_anchor = False    # Continuous polling for anchors and event-triggered trades
    next_anchor_idx = next((idx for idx, t in enumerate(anchors) if t >= datetime.now()), len(anchors) - 1)
    while not type_shutdown.is_set() and trades_done < cfg.MAX_TRADES:
        now = datetime.now()
        # Poll live equity to enforce drawdown/profit limits
        try:
            account = trading.get_account()
            current_equity = float(account.equity)
            # Calculate drawdown and profit percentages
            drawdown_pct = (daily_start_equity - current_equity) / daily_start_equity if daily_start_equity > 0 else 0
            profit_pct = (current_equity - daily_start_equity) / daily_start_equity if daily_start_equity > 0 else 0
            logging.info(
                "Equity: %.2f, drawdown %.2f%%, profit %.2f%%",
                current_equity,
                drawdown_pct * 100,
                profit_pct * 100,
            )
            # Check equity-based drawdown limit
            if cfg.DAILY_DRAWDOWN_PCT > 0 and drawdown_pct >= cfg.DAILY_DRAWDOWN_PCT:
                logging.critical(
                    "Equity drawdown %.2f%% >= limit %.2f%%; liquidating and shutting down",
                    drawdown_pct * 100,
                    cfg.DAILY_DRAWDOWN_PCT * 100,
                )
                trading.close_all_positions()
                type_shutdown.set()
                break
            # Check equity-based profit target
            if cfg.DAILY_PROFIT_PCT > 0 and profit_pct >= cfg.DAILY_PROFIT_PCT:
                logging.info(
                    "Equity profit %.2f%% >= limit %.2f%%; liquidating and shutting down",
                    profit_pct * 100,
                    cfg.DAILY_PROFIT_PCT * 100,
                )
                trading.close_all_positions()
                type_shutdown.set()
                break
        except Exception as e:
            logging.warning("Equity check failed: %s", e)

        # Anchor scheduling
        if next_anchor_idx < len(anchors) and now >= anchors[next_anchor_idx]:
            anchor = anchors[next_anchor_idx]
            logging.info(
                "Symbol %s anchor trade #%d at %s",
                symbol,
                trades_done + 1,
                anchor.time(),
            )
            if _should_enter_trade(stock_hist, option_hist, symbol, cfg, is_anchor=True):
                t = threading.Thread(
                    target=_execute_trade_and_update,
                    args=(thread_target, anchor_args, cfg),
                    daemon=True,
                )
                t.start()
                threads.append(t)
                trades_done += 1
                last_spot = get_underlying_price(stock_hist, symbol)
                event_trade_since_anchor = False
            next_anchor_idx += 1
            # If final anchor fired, exit scheduling loop
            if next_anchor_idx >= len(anchors):
                break
            continue
        # Event-triggered trades between anchors
        if next_anchor_idx < len(anchors) and not event_trade_since_anchor:
            curr_spot = get_underlying_price(stock_hist, symbol)
            if abs(curr_spot - last_spot) / last_spot >= cfg.EVENT_MOVE_PCT:
                logging.info(
                    "Symbol %s price move triggered event trade #%d",
                    symbol,
                    trades_done + 1,
                )
                t = threading.Thread(
                    target=_execute_trade_and_update,
                    args=(thread_target, event_args, cfg),
                    daemon=True,
                )
                t.start()
                threads.append(t)
                trades_done += 1
                event_trade_since_anchor = True
                last_spot = curr_spot
        time.sleep(cfg.POLL_INTERVAL)


    # Wait for all threads to complete
    for t in threads:
        t.join()
    logging.info(
        "Symbol %s: all %d trades completed",
        symbol,
        trades_done,
    )



def main(strategy: str = "two_phase", auto_reenter: bool = False):
    setup_logging()
    logging.info(
        "Starting 0DTE strategy on %s (paper=%s)",
        cfg.UNDERLYING,
        cfg.PAPER,
    )

    trading = TradingClient(
        cfg.ALPACA_API_KEY, cfg.ALPACA_API_SECRET, paper=cfg.PAPER
    )
    # Initialize daily P&L and starting equity (percent drawdown circuit breaker)
    global daily_pnl, daily_start_equity
    daily_pnl = 0.0
    try:
        account = trading.get_account()
        daily_start_equity = float(account.equity)
        logging.info("Starting equity for today: %.2f", daily_start_equity)
    except Exception as e:
        logging.warning("Unable to fetch starting equity: %s", e)
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
            "Market is closed (next open at %s). Sleeping 10 minutes...",
            clock.next_open.isoformat(),
        )
        time.sleep(600)
        clock = trading.get_clock()
    # Schedule trading tasks for each symbol
    symbol_threads: list[threading.Thread] = []
    for symbol in cfg.UNDERLYING:
        if type_shutdown.is_set():
            break
        t = threading.Thread(
            target=schedule_for_symbol,
            args=(symbol, cfg, trading, stock_hist, option_hist, strategy),
            daemon=True,
        )
        t.start()
        symbol_threads.append(t)
    # Wait for all per-symbol schedules to complete
    for t in symbol_threads:
        t.join()
    logging.info("All symbols processed: %s", cfg.UNDERLYING)

def analyze_logs_for_date(trading_date: date):
    """
    Read the log file(s) for the given trading date, summarize key metrics, and log recommendations.
    Attempts to read both the date-based log (YYYY-MM-DD.log) and the strangle-specific log (zero_dte_strangle.log).
    """
    # Locate logs directory in project root
    file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(file_dir, os.pardir, os.pardir))
    log_dir = os.path.join(project_root, 'logs')
    lines: list[str] = []
    # Read primary date-based log
    primary_log = os.path.join(log_dir, f"{trading_date.isoformat()}.log")
    try:
        lines.extend(open(primary_log).read().splitlines())
    except FileNotFoundError:
        logging.warning("Date-based log file not found for analysis: %s", primary_log)
    # Read strangle-specific log and include only entries for this date
    strangle_log = os.path.join(log_dir, 'zero_dte_strangle.log')
    try:
        slines = open(strangle_log).read().splitlines()
        date_prefix = trading_date.isoformat()
        lines.extend([ln for ln in slines if ln.startswith(date_prefix)])
    except FileNotFoundError:
        # No strangle-specific log; skip silently
        pass
    # Proceed with analysis
    errors = [ln for ln in lines if "ERROR" in ln]
    pnls: list[float] = []
    for ln in lines:
        # Only capture final strangle-complete PnL entries
        m = re.search(r"Strangle complete for .+; PnL=([-+]?\d+\.\d+)", ln)
        if m:
            pnls.append(float(m.group(1)))
    total_trades = len(pnls)
    total_pnl = sum(pnls)
    avg_pnl = total_pnl / total_trades if total_trades else 0.0
    logging.info(
        "End-of-day analysis for %s: trades=%d, total_PnL=%.2f, avg_PnL=%.2f",
        trading_date.isoformat(), total_trades, total_pnl, avg_pnl
    )
    if errors:
        logging.warning("Errors encountered during trading day (%d):", len(errors))
        for e in errors:
            logging.warning("%s", e)
        logging.info("Recommendation: Investigate errors and add resilience (retries, exception handling).")
    if avg_pnl < 0:
        logging.info(
            "Recommendation: Strategy resulted in negative average PnL (%.2f). Consider adjusting targets or filters.",
            avg_pnl
        )
    else:
        logging.info("Recommendation: Strategy yielded positive average PnL. Continue monitoring and fine-tuning thresholds.")







if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["strangle", "two_phase"], default="two_phase",
                        help="Trading strategy to run: single-phase strangle or two-phase flip")
    parser.add_argument("--auto-reenter", action="store_true",
                        help="Enable automatic re-entry of a new trading cycle on drawdown shutdown.")
    args = parser.parse_args()
    strategy = args.strategy
    auto_reenter = args.auto_reenter

    # Register signal handlers in main thread
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Outer loop: run trading cycles for each day
    while True:
        # Immediate re-entry loop: repeat main() if shutdown triggered by drawdown
        while True:
            # Reset log handlers so setup_logging can reconfigure
            for h in logging.root.handlers[:]:
                logging.root.removeHandler(h)
            main(strategy)
            if not auto_reenter or not type_shutdown.is_set():
                break
            logging.info("Auto-reenter enabled: restarting trading cycle with fresh PnL and equity.")
            type_shutdown.clear()
        # End-of-day analysis
        analyze_logs_for_date(date.today())
        # Calculate sleep until next trading day
        tomorrow = date.today() + timedelta(days=1)
        next_start = datetime.combine(tomorrow, cfg.TRADE_START)
        secs = (next_start - datetime.now()).total_seconds()
        logging.info("Sleeping until next trading day start at %s (%.0f seconds)", next_start, secs)
        time.sleep(secs)

