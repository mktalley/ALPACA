"""Entry point for running the MKTONE bot."""

import logging
import time
from pathlib import Path

from .config import get_option_client, get_stock_client, get_trading_client
from .strategies.zero_dte_credit_spread import ZeroDTECreditSpread

LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "bot.log"
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def main(poll_interval: float = 60) -> None:
    trading = get_trading_client()
    stock = get_stock_client()
codex/update-order-creation-to-use-market-data
    option = get_option_client()
    strategy = ZeroDTECreditSpread(
        trading_client=trading, stock_client=stock, option_client=option
    )

    strategy = ZeroDTECreditSpread(trading_client=trading, stock_client=stock)
main

    while True:
        clock = trading.get_clock()
        if not clock.is_open:
            break
        if not trading.get_all_positions():
            strategy.run()
        time.sleep(poll_interval)


if __name__ == "__main__":  # pragma: no cover
    main()

