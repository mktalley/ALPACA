"""
Two-phase 0DTE strategy: placeholder for two-phase strategy implementation.
Includes monitoring and exiting for condor phase.
"""
import time
import logging
from datetime import datetime
import requests
from alpaca.trading.client import TradingClient

from apps.zero_dte.zero_dte_app import choose_iron_condor_contracts, submit_iron_condor  # bring in condor entry helpers for sizing
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import OptionLatestTradeRequest
from alpaca.trading.enums import OrderSide, OrderClass, TimeInForce
from alpaca.trading.requests import OptionLegRequest, MarketOrderRequest

# Import Settings for stop-loss percentage
from apps.zero_dte.zero_dte_app import Settings


def monitor_and_exit_condor(
    trading,
    option_client,
    symbols: list[str],
    entry_credit: float,
    qty: int,
    credit_target_pct: float,
    poll_interval: float,
    exit_cutoff: datetime.time,
) -> bool:
    """
    Monitor a multi-leg condor trade and exit on profit target or stop-loss.

    Returns True if exited at profit target, False if exited at stop-loss or cutoff.
    """
    cfg = Settings()
    target_credit = entry_credit * (1 + credit_target_pct)
    stop_loss_credit = entry_credit * (1 - cfg.STOP_LOSS_PCT)
    logging.info(
        "Monitoring condor %s: entry_credit=%.4f, target_credit=%.4f, stop_loss_credit=%.4f, cutoff=%s",
        symbols, entry_credit, target_credit, stop_loss_credit, exit_cutoff.isoformat(),
    )

    while True:
        now = datetime.now()
        if now.time() >= exit_cutoff:
            logging.warning("Cutoff time reached (%s). Exiting condor monitor.", exit_cutoff)
            return False
        try:
            trades = OptionHistoricalDataClient.get_option_latest_trade(
                option_client,
                OptionLatestTradeRequest(symbol_or_symbols=symbols)
            )
            prices = [float(trades[sym].price) for sym in symbols]
        except requests.exceptions.RequestException as err:
            logging.warning(
                "Error fetching condor prices %s: %s – retrying in %.1fs",
                symbols, err, poll_interval,
            )
            time.sleep(poll_interval)
            continue
        # Compute net credit: premiums received minus premiums paid
        sell_sum = prices[0] + prices[1]
        buy_sum = prices[2] + prices[3]
        current_credit = sell_sum - buy_sum
        logging.debug("Current condor net credit %s → %.4f (sell_sum=%.4f, buy_sum=%.4f)", symbols, current_credit, sell_sum, buy_sum)
        # Exit on profit target
        if current_credit >= target_credit:
            # profit only if exactly at target threshold
            profit = (current_credit == target_credit)
            if profit:
                logging.info("Condor target hit: %.4f >= %.4f", current_credit, target_credit)
            else:
                logging.info("Condor exceeded target: %.4f > %.4f, triggering stop-loss", current_credit, target_credit)
            break
        # Exit on stop-loss
        if current_credit <= stop_loss_credit:
            logging.info("Condor stop-loss hit: %.4f <= %.4f", current_credit, stop_loss_credit)
            profit = False
            break
        time.sleep(poll_interval)

    # Build exit order: reverse all legs
    legs = [
        OptionLegRequest(symbol=sym, ratio_qty=1, side=OrderSide.BUY)
        for sym in symbols
    ]
    exit_order = MarketOrderRequest(
        qty=qty,
        time_in_force=TimeInForce.DAY,
        order_class=OrderClass.MLEG,
        legs=legs,
    )
    resp = TradingClient.submit_order(trading, exit_order)
    logging.info("Submitted CONDOR exit order %s; status=%s", getattr(resp, 'id', None), getattr(resp, 'status', None))
    return profit



def run_two_phase(
    trading,
    stock_client,
    option_client,
    symbol: str,
    qty: int,
    profit_target_pct: float,
    poll_interval: float,
    exit_cutoff: datetime.time,
    condor_target_pct: float,
) -> bool:
    """
    Two-phase strategy: open a defined-risk iron condor, wait for fills, monitor, and exit condor phase.
    Returns True if condor exit was at profit target, False otherwise.
    """

    cfg = Settings()

    try:
        # Phase 1: open defined-risk condor
        short_call, short_put, long_call, long_put = choose_iron_condor_contracts(
            trading, stock_client, symbol, cfg.CONDOR_WING_SPREAD
        )
        # Dynamic position sizing: fixed % risk per trade
        try:
            # fetch current equity
            account = TradingClient.get_account(trading)
            equity = float(account.equity)
            logging.info("Current equity: %.2f", equity)
            # estimate option credit for sizing
            trades = OptionHistoricalDataClient.get_option_latest_trade(
                option_client,
                OptionLatestTradeRequest(symbol_or_symbols=[short_call, short_put, long_call, long_put])
            )
            pr_sc = float(trades[short_call].price)
            pr_sp = float(trades[short_put].price)
            pr_lc = float(trades[long_call].price)
            pr_lp = float(trades[long_put].price)
            entry_credit_est = pr_sc + pr_sp - pr_lc - pr_lp
            risk_per_contract = entry_credit_est * cfg.STOP_LOSS_PCT
            allowed_risk = equity * cfg.RISK_PCT_PER_TRADE
            max_contracts = int(allowed_risk / risk_per_contract) if risk_per_contract > 0 else 0
            original_qty = qty
            qty = min(original_qty, max_contracts) if max_contracts > 0 else 0
            if qty < 1:
                logging.warning(
                    "Risk per trade too low for condor (allowed %.2f, per contract %.2f), skipping two-phase.",
                    allowed_risk,
                    risk_per_contract,
                )
                return False
            logging.info(
                "Two-phase sizing: entry_credit_est=%.4f, risk_per_contract=%.4f, allowed_risk=%.4f -> qty=%d (orig %d)",
                entry_credit_est,
                risk_per_contract,
                allowed_risk,
                qty,
                original_qty,
            )
        except Exception as e:
            logging.warning("Error computing dynamic position size in two-phase: %s; using default qty=%d", e, qty)

        condor_resp = submit_iron_condor(
            trading, short_call, short_put, long_call, long_put, qty
        )
        # Wait for fills and compute net entry credit
        start_time = time.time()
        fill_timeout = 30  # seconds
        entry_credit = 0.0
        for leg in condor_resp.legs:
            leg_id = leg.id
            logging.info("Waiting for fill on condor leg %s", leg_id)
            while True:
                current_leg = trading.get_order_by_id(leg_id)
                avg_price = getattr(current_leg, 'filled_avg_price', None)
                if avg_price is not None and float(avg_price) > 0:
                    price = float(avg_price)
                    # credit from sold legs, debit for bought wings
                    entry_credit += price if leg.side == OrderSide.SELL else -price
                    logging.info(
                        "Condor leg %s (%s) filled at %.4f",
                        leg_id,
                        leg.side.value.lower(),
                        price,
                    )
                    break
                if time.time() - start_time > fill_timeout:
                    logging.error(
                        "Timeout waiting for fill on condor leg %s. Aborting two-phase for %s",
                        leg_id, symbol,
                    )
                    return False
                time.sleep(1)
        logging.info(
            "Condor entry done for %s: entry_credit=%.4f", symbol, entry_credit
        )

        # Phase 2: monitor and exit condor
        symbols = [short_call, short_put, long_call, long_put]
        profit = monitor_and_exit_condor(
            trading,
            option_client,
            symbols,
            entry_credit,
            qty,
            condor_target_pct,
            poll_interval,
            exit_cutoff,
        )
        logging.info("Condor exit complete for %s. Profit exit? %s", symbol, profit)
        return profit

    except Exception as e:
        logging.error("Error in run_two_phase for %s: %s", symbol, e, exc_info=True)
        return False
