from __future__ import annotations

"""Simple utility to build parameter dictionaries from a concise grid string.

Example::
    parse_grid("STOP=[0.1,0.2]; TARGET=[0.3]")
    # -> [{'STOP': 0.1, 'TARGET': 0.3}, {'STOP': 0.2, 'TARGET': 0.3}]
"""

import ast
import itertools
from typing import Any, Dict, List


def _parse_list(text: str):
    """Parse a text representing a python literal list/tuple"""
    try:
        val = ast.literal_eval(text)
    except Exception as e:
        raise ValueError(f"Could not parse list literal: {text}") from e
    if not isinstance(val, (list, tuple)):
        raise ValueError(f"Expected list literal but got: {val}")
    return list(val)


def parse_grid(grid_str: str) -> List[Dict[str, Any]]:
    """Convert a semi-colon separated *KEY=[values]* string into a list of param dicts."""
    if not grid_str:
        return [{}]
    pieces = [p.strip() for p in grid_str.split(";") if p.strip()]
    param_lists = {}
    for piece in pieces:
        if "=" not in piece:
            raise ValueError(f"Grid token missing '=': {piece}")
        key, val = piece.split("=", 1)
        key = key.strip()
        val = val.strip()
        if not (val.startswith("[") and val.endswith("]")):
            raise ValueError(f"Grid value must be a list literal: {val}")
        param_lists[key] = _parse_list(val)
    # Build cartesian product
    keys = list(param_lists)
    combos = list(itertools.product(*[param_lists[k] for k in keys]))
    return [dict(zip(keys, combo)) for combo in combos]
