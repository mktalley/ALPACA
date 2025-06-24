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
        exit_cutoff: dt_time | None = None,
    ) -> None:
        self.bars = bars
        self.entry_time = entry_time
        self.entry_credit = entry_credit
        self.qty = qty
        self.stop_loss = entry_credit * (1 + stop_loss_pct)
        self.profit_target = entry_credit * (1 - profit_target_pct)
        self.exit_cutoff = exit_cutoff or dt_time(22, 45)

    def run(self) -> TradeResult:
        idx = self.bars.index
        after_entry = idx >= self.entry_time
        # synthetic mid-credit path = entry ± 0.2% of underlying move
        rel_move = self.bars.loc[after_entry, "close"].pct_change().fillna(0).cumsum()
        mid = self.entry_credit * (1 + 0.2 * rel_move)

        # exit conditions
        hit_target = mid <= self.profit_target
        hit_stop = mid >= self.stop_loss
        cutoff_mask = self.bars.index.time >= self.exit_cutoff

        exit_idx = np.where(hit_target | hit_stop | cutoff_mask)[0]
        exit_pos = exit_idx[0] if len(exit_idx) else -1

        exit_credit = float(mid.iloc[exit_pos])
        exit_time = idx[after_entry][exit_pos]
        pnl = (
            (self.entry_credit - exit_credit) * 100 * self.qty
        )  # credit received minus cost to buy back
        return TradeResult(
            self.qty, self.entry_time, exit_time, self.entry_credit, exit_credit, pnl
        )
