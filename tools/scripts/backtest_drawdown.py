#!/usr/bin/env python3
"""
Quick backtest for daily drawdowns on SPY over the past year using Alpaca Market Data API.
Identifies days where the 1-day return exceeded the negative drawdown threshold (e.g., -5%).
"""
import os
import statistics
import sys
from datetime import datetime, timedelta

from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest

# Configuration
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_API_SECRET")
SYMBOL = "SPY"
TIMEFRAME = "1Day"
DRAWDOWN_PCT = 0.05  # 5% threshold


def main():
    if not API_KEY or not API_SECRET:
        print("Please set ALPACA_API_KEY and ALPACA_API_SECRET in your environment.")
        sys.exit(1)

    client = StockHistoricalDataClient(
        api_key=API_KEY,
        secret_key=API_SECRET,
        sandbox=False,
    )

    end = datetime.utcnow()
    start = end - timedelta(days=365)

    request_params = StockBarsRequest(
        symbol_or_symbols=[SYMBOL],
        start=start.isoformat() + "Z",
        end=end.isoformat() + "Z",
        timeframe=TIMEFRAME,
        limit=1000,
    )
    barset = client.get_stock_bars(request_params)
    bars = barset[SYMBOL]
    if len(bars) < 2:
        print("Not enough data to compute drawdowns.")
        sys.exit(1)

    # Compute daily returns
    returns = []  # list of (date, return)
    for i in range(1, len(bars)):
        prev = bars[i - 1]
        curr = bars[i]
        ret = (curr.close - prev.close) / prev.close
        returns.append((curr.timestamp.date(), ret))

    # Identify drawdown days
    ddays = [(d, r) for d, r in returns if r <= -DRAWDOWN_PCT]
    max_dd = min(returns, key=lambda x: x[1])  # most negative return
    avg = statistics.mean(r for _, r in returns)
    sd = statistics.stdev(r for _, r in returns)

    print(f"Analyzed {len(returns)} trading days for {SYMBOL}.")
    print(f"Average daily return: {avg:.2%} (stdev {sd:.2%})")
    print(f"Max single-day drawdown: {max_dd[1]:.2%} on {max_dd[0]}")
    print(f"Days with drawdown >= {DRAWDOWN_PCT*100:.0f}%: {len(ddays)}")
    for d, r in ddays:
        print(f"  {d}: {r:.2%}")


if __name__ == "__main__":
    main()
