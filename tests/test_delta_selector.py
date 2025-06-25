import pandas as pd

from apps.zero_dte.utils.delta import select_short_leg, select_strikes


def _mock_chain():
    rng = []
    # Build synthetic chain ±20 strikes with linear delta mapping
    for strike in range(430, 451):
        delta = -(451 - strike) / 100  # put deltas from ~-0.21 to 0
        rng.append({"strike": strike, "delta": delta, "side": "put"})
    for strike in range(451, 472):
        delta = (strike - 451) / 100  # call deltas 0→0.21
        rng.append({"strike": strike, "delta": delta, "side": "call"})
    return pd.DataFrame(rng)


def test_select_short_leg():
    chain = _mock_chain()
    short_put = select_short_leg(chain, 0.15, "put")
    short_call = select_short_leg(chain, 0.15, "call")
    assert short_put["strike"] == 436  # 0.15 delta approx
    assert short_call["strike"] == 466  # 0.15 delta approx


def test_select_strikes():
    chain = _mock_chain()
    sp, lp, sc, lc = select_strikes(chain, 0.10, 5.0)
    assert sp == 441  # delta_diff minimal for 0.10 (~441 put)
    assert lp == 436.0
    assert sc == 461  # ~0.10 delta call
    assert lc == 466.0
