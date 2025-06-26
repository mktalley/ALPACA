"""Utility helpers for delta-based strike selection.

The helpers are **data-frame agnostic**: they operate on any pandas DataFrame
or sequence that exposes at least the following columns/attributes:
    • ``strike`` (float/int)
    • ``delta``  (float, signed ‑ puts usually negative)
    • ``put_call`` or ``side`` designating option type ("put" / "call")

They are therefore usable with Alpaca-PY *OptionContract* objects **or** the
synthetic tables used in the unit-test suite.
"""
from __future__ import annotations

from typing import Literal, Sequence, Tuple

import pandas as pd

Side = Literal["put", "call"]


# ---------------------------------------------------------------------------
# Core selection logic
# ---------------------------------------------------------------------------

def _normalise_frame(chain: "pd.DataFrame | Sequence") -> pd.DataFrame:  # type: ignore[type-arg]
    """Return *chain* as a DataFrame with strike, delta, side columns."""
    if isinstance(chain, pd.DataFrame):
        df = chain.copy()
    else:  # convert from sequence of objects having attributes
        df = pd.DataFrame(
            {
                "strike": [c.strike for c in chain],
                "delta": [getattr(getattr(c, "greeks", c), "delta", None) for c in chain],
                "side": [getattr(c, "put_call", getattr(c, "side", None)) for c in chain],
            }
        )
    # Normalise column names
    if "put_call" in df.columns and "side" not in df.columns:
        df.rename(columns={"put_call": "side"}, inplace=True)
    if "side" not in df.columns:
        raise ValueError("Option chain must contain 'side' or 'put_call' column")
    return df


def select_short_leg(
    chain: "pd.DataFrame | Sequence", target_delta: float, side: Side
) -> pd.Series:
    """Return the contract row whose *abs(delta)* is closest to *target_delta*.

    Parameters
    ----------
    chain
        Option chain snapshot (DataFrame or sequence of objects).
    target_delta
        Absolute target delta (e.g., ``0.15`` picks ~±15Δ).
    side
        "put" or "call".
    """
    df = _normalise_frame(chain)
    if side not in ("put", "call"):
        raise ValueError("side must be 'put' or 'call'")

    # Filter by side and drop rows with NaN deltas
    mask_side = df["side"].str.lower() == side
    df = df.loc[mask_side & df["delta"].notna()].copy()
    if df.empty:
        raise ValueError("No contracts matching side and delta present in chain")

    df["abs_delta_diff"] = (df["delta"].abs() - target_delta).abs()
    return df.sort_values("abs_delta_diff").iloc[0]


def select_strikes(
    chain: "pd.DataFrame | Sequence",
    target_delta: float,
    wing_width: float,
) -> Tuple[float, float, float, float]:
    """Return tuple ``(short_put, long_put, short_call, long_call)``.

    The short legs are chosen at ±target_delta; wing width (dollars) is added
    symmetrically to build an iron-condor style structure.
    """
    short_put = select_short_leg(chain, target_delta, "put")
    short_call = select_short_leg(chain, target_delta, "call")

    long_put_strike = float(short_put.strike) - wing_width
    long_call_strike = float(short_call.strike) + wing_width

    return (
        float(short_put.strike),
        long_put_strike,
        float(short_call.strike),
        long_call_strike,
    )


__all__ = [
    "select_short_leg",
    "select_strikes",
]
