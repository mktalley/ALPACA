"""Default parameter grid and helpers for 0-DTE back-tests."""
from __future__ import annotations

import itertools
from typing import Dict, List, Any

# You can tune these lists – they are the first levers most traders adjust.
DEFAULT_PARAM_GRID: Dict[str, List[Any]] = {
    # risk management
    "STOP_LOSS_PCT": [0.15, 0.20, 0.25],  # 15–25 % debit loss
    "PROFIT_TARGET_PCT": [0.40, 0.50, 0.60],  # 40–60 % debit profit
    # sizing – keep constant to isolate risk/target impact first
    "QTY": [10],
    # condor wing distance (ignored by strangle simulator but harmless)
    "CONDOR_WING_SPREAD": [2],
}


def default_grid() -> List[Dict[str, Any]]:
    """Return a list of parameter dictionaries representing the Cartesian product
    of *DEFAULT_PARAM_GRID*.
    """
    keys = list(DEFAULT_PARAM_GRID)
    combos = itertools.product(*[DEFAULT_PARAM_GRID[k] for k in keys])
    return [dict(zip(keys, combo)) for combo in combos]
