"""Configuration helpers for the MKTONE bot."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load variables from a .env file if present
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    load_dotenv()

API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
BASE_URL = os.getenv("APCA_API_BASE_URL")


def get_trading_client():  # type: ignore[override]
    """Return an authenticated ``TradingClient`` instance."""
    from alpaca.trading.client import TradingClient

    return TradingClient(API_KEY, API_SECRET, paper=True, url_override=BASE_URL)


def get_stock_client():
    """Return a client for fetching stock market data."""
    from alpaca.data.historical import StockHistoricalDataClient

    return StockHistoricalDataClient(API_KEY, API_SECRET)


def get_option_client():
    """Return a client for fetching options data."""
    from alpaca.data.historical import OptionHistoricalDataClient

    return OptionHistoricalDataClient(API_KEY, API_SECRET)

