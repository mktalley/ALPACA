"""Vectorised minute-bar simulator for a single 0-DTE strangle."""

from __future__ import annotations

from datetime import datetime
from datetime import time as dt_time
from typing import Iterable, NamedTuple

import numpy as np
import pandas as pd


class TradeResult(NamedTuple):
    qty: int
    entry_time: datetime
    exit_time: datetime
    entry_credit: float
    exit_credit: float
    pnl: float


class StrangleSimulator:
    """PNL path for one strangle given underlying minute bars."""

    def __init__(
        self,
        bars: pd.DataFrame,
        entry_time: datetime,
        entry_credit: float,
        qty: int,
        stop_loss_pct: float = 0.20,
        profit_target_pct: float = 0.65,
        *,
        slippage: float = 0.0,  # absolute $ per contract, per side
        commission: float = 0.0,  # absolute $ per contract, per side
        legs: int = 2,
        exit_cutoff: dt_time | None = None,
    ) -> None:
        self.bars = bars
        self.entry_time = entry_time
        self.entry_credit = entry_credit
        self.qty = qty
        self.stop_loss = entry_credit * (1 + stop_loss_pct)
        self.profit_target = entry_credit * (1 - profit_target_pct)
        self.exit_cutoff = exit_cutoff or dt_time(22, 45)
        self.slippage = slippage
        self.commission = commission
        self.legs = legs

    def run(self) -> TradeResult:
        idx = self.bars.index
        after_entry = idx >= self.entry_time

        prices = self.bars.loc[after_entry, "close"]
        rel_move = prices.pct_change().fillna(0).cumsum()
        mid = self.entry_credit * (1 + 0.2 * rel_move)

        hit_target = mid <= self.profit_target
        hit_stop = mid >= self.stop_loss
        cutoff_mask = mid.index.time >= self.exit_cutoff

        exit_idx = np.where(hit_target | hit_stop | cutoff_mask)[0]
        exit_pos = int(exit_idx[0]) if len(exit_idx) else -1

        exit_credit = float(mid.iloc[exit_pos])
        exit_time = mid.index[exit_pos]

        # cost = buy-back mid + slippage per leg + commission per leg
        cost_to_close = (
            exit_credit
            + self.slippage * self.legs
            + self.commission * self.legs
        )
        open_credit = (
            self.entry_credit
            - self.slippage * self.legs
            - self.commission * self.legs
        )
        pnl = (open_credit - cost_to_close) * 100 * self.qty
        return TradeResult(
            self.qty, self.entry_time, exit_time, open_credit, cost_to_close, pnl
        )
