# MKTONE 0DTE Trading Bot

MKTONE is an automated options trading bot focused on **0DTE (zero days to expiration)** strategies using the [Alpaca](https://alpaca.markets) API. It demonstrates a simple credit spread approach that sells premium in the direction of the current intraday trend and manages risk with profit and loss targets.

## Strategy Overview

1. **Momentum filter** – calculate a short moving average on the underlying symbol (default `SPY`).
2. **Direction** –
   - If price is above the moving average, sell a put credit spread.
   - If price is below the moving average, sell a call credit spread.
3. **Risk management** – close the spread when either:
   - 20% of the received credit is captured, or
   - the spread loses 50% of the credit (stop loss), or
   - the market closes.

The code serves as a starting point and should be back‑tested and adapted before live use.

## Project Layout

```
MKTONE/
├── README.md
├── requirements.txt
├── .gitignore
├── logs/
├── src/
│   ├── bot.py
│   ├── config.py
│   └── strategies/
│       └── zero_dte_credit_spread.py
└── tests/
    └── test_imports.py
```

## Getting Started

1. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Provide your Alpaca API credentials using environment variables or an `.env` file:

```
APCA_API_KEY_ID="your-key"
APCA_API_SECRET_KEY="your-secret"
APCA_API_BASE_URL="https://paper-api.alpaca.markets"
```

3. Run the bot:

```bash
python -m src.bot
```

Logs will be written to `logs/bot.log`.
