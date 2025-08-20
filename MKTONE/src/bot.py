"""Entry point for running the MKTONE bot."""

import logging
from pathlib import Path

from .config import get_trading_client, get_stock_client
from .strategies.zero_dte_credit_spread import ZeroDTECreditSpread

LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "bot.log"
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def main() -> None:
    trading = get_trading_client()
    stock = get_stock_client()
    strategy = ZeroDTECreditSpread(trading_client=trading, stock_client=stock)
    strategy.run()


if __name__ == "__main__":  # pragma: no cover
    main()

