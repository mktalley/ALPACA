# Zero-DTE Strangle & Two-Phase App

This application executes 0DTE (zero days to expiration) OTM strangle and two-phase strategies on options for a given underlying symbol using Alpaca.

## Overview

- Single-phase **strangle** strategy: sell OTM call + put, monitor PnL, exit on profit target or stop loss.
- Two-phase strategy: after initial strangle leg, flip into an iron condor on target, then exit on condor profit target.
- Built with Pydantic `Settings` for configuration via environment variables.
- Structured for easy extension and reliable order handling with retries and sizing safeguards.

## Prerequisites

- Python 3.8+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- Alpaca API credentials (paper or live)

## Setup

1. Copy the example environment file:
   ```bash
   cp apps/zero_dte/.env.example apps/zero_dte/.env
   ```
2. Edit `apps/zero_dte/.env` and fill in:
   ```dotenv
   ALPACA_API_KEY=your_api_key
   ALPACA_API_SECRET=your_api_secret
   PAPER=true               # use paper account
   UNDERLYING=SPY           # comma-separated symbols
   QTY=10                   # default contracts per leg
   PROFIT_TARGET_PCT=0.75   # profit target %
   STOP_LOSS_PCT=0.20       # stop loss %
   RISK_PCT_PER_TRADE=0.01  # max equity % to risk per trade
   ...                     # see .env.example for full list
   ```

## Configuration Settings

All settings are defined in `Settings` (Pydantic `BaseSettings`) in `zero_dte_app.py`. Key fields:

| Variable                | Default   | Description                                                                      |
|-------------------------|-----------|----------------------------------------------------------------------------------|
| `UNDERLYING`            | `["SPY"]`| Comma-separated list of symbol(s) to trade.                                      |
| `QTY`                   | `10`      | Number of contracts per leg.                                                     |
| `PROFIT_TARGET_PCT`     | `0.75`    | Profit target as percent of entry price.                                         |
| `STOP_LOSS_PCT`         | `0.20`    | Max loss percent per contract.                                                   |
| `RISK_PCT_PER_TRADE`    | `0.01`    | Percent of account equity to risk per trade (dynamic sizing).                   |
| `CONDOR_TARGET_PCT`     | `0.25`    | Profit target percent for condor phase (two-phase strategy).                     |
| `CONDOR_WING_SPREAD`    | `2`       | Strike width for condor wings.                                                   |
| `MAX_TRADES`            | `20`      | Maximum number of trades per day.                                                |
| `POLL_INTERVAL`         | `120.0`   | Seconds between PnL and price checks.                                            |
| `EXIT_CUTOFF`           | `22:45:00`| Wall-clock time to force exit if target not hit.                                  |
| `TRADE_START`/`TRADE_END` | `09:45:00`/`12:30:00` | Earliest and latest times for initial entries.                         |

## Running the App

- **Strangle only**:
  ```bash
  python apps/zero_dte/zero_dte_app.py --strategy strangle
  ```

- **Two-phase (strangle → condor)**:
  ```bash
  python apps/zero_dte/zero_dte_app.py --strategy two_phase
  ```

The app will:
- Wait for market open
- Execute scheduled strangle/condor trades
- Monitor fills and PnL, perform exits on target or stop
- Log all activity to console and `logs/YYYY-MM-DD.log`
- Perform end-of-day analysis and sleep until next trading day

## Logs & Analysis

Logs are written to `logs/YYYY-MM-DD.log`. At EOD the app summarizes:
- Total trades, PnL stats
- Any errors or warnings
- Recommendations for thresholds or resilience improvements

---

For questions or contributions, see the main repository README.