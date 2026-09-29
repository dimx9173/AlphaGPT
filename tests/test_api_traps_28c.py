"""Two API traps that fail silently, now made to fail loudly.

Both were found while building the cost attribution and the random-formula
probe, and both share a shape: the pipeline runs to completion, every
validation passes, and the number that comes out is simply the wrong number.

1. GRAMMAR was a module global set only inside main(). Any other caller --
   a probe, a test, a notebook -- inherited the import-time default 'full' and
   sent reduced formulas to the full evaluator. Token indices are
   grammar-specific (reduced token 20 is DELAY1, full token 20 is JUMP), so the
   reduced formula was computed as a DIFFERENT FUNCTION and reported the
   result honestly. Nothing crashed.

2. load_data returns the 8-hour settlement MASK in its fourth slot, where
   callers expect per-coin rate arrays. Indexing the mask by coin name raises
   IndexError, but passing None instead is worse: it silently reverts to the
   flat +0.0005 rate that overstates funding by ~16x and inverts the sign on
   about 30% of events, which this project already fixed once.

Both guards are asserted here against the real modules, not against a mock.
"""
from __future__ import annotations

import numpy as np
import pytest

import research.ga_28c_30m_3y as GA
from research.grammar_reduced_28c import validate_reduced


# --- trap 1: grammar routing -------------------------------------------------

def test_resolve_grammar_refuses_a_formula_from_the_wrong_grammar():
    """A reduced formula must not be silently evaluated as a full one."""
    bad = [20] * 12
    assert not validate_reduced(bad)
    with pytest.raises(ValueError, match="not valid in the reduced grammar"):
        GA.resolve_grammar(bad, "reduced")


def test_resolve_grammar_accepts_a_valid_reduced_formula():
    good = [11, 1, 7, 2, 14, 7, 15, 6, 15, 18, 4, 22]
    assert validate_reduced(good)
    assert GA.resolve_grammar(good, "reduced") == "reduced"


def test_the_full_path_is_not_gated_by_the_reduced_check():
    """The guard is one-directional on purpose.

    Tokens 0-11 are features in both grammars, so a full formula is not
    required to satisfy reduced RPN validity. Gating it would reject valid
    full-grammar programs.
    """
    assert GA.resolve_grammar([20] * 12, "full") == "full"


def test_grammar_travels_with_the_call_not_a_global():
    """The point of the fix: no global needs mutating to be correct.

    An importing caller must not have to reach into the module and set
    GA.GRAMMAR, because that is exactly what made the bug invisible.
    """
    original = GA.GRAMMAR
    try:
        GA.GRAMMAR = "full"          # what an importer used to inherit
        good = [11, 1, 7, 2, 14, 7, 15, 6, 15, 18, 4, 22]
        # The explicit parameter wins over the stale global.
        assert GA.resolve_grammar(good, "reduced") == "reduced"
    finally:
        GA.GRAMMAR = original


def test_formula_signal_rejects_the_wrong_grammar_before_evaluating():
    maps = {"BTC": np.zeros((12, 100))}
    with pytest.raises(ValueError):
        GA.formula_signal([20] * 12, maps, "BTC", grammar="reduced")


# --- trap 2: funding mask vs per-coin rates ---------------------------------

def test_evaluate_refuses_a_mask_where_rates_are_expected():
    """An ndarray in the funding_by_coin slot is the documented mistake."""
    from research.splits_28c import search_splits, split_indices
    common, maps, returns, mask = GA.load_data(null="none")
    n = len(common)
    lockbox = split_indices(n, common[0], common[-1])["lockbox"]
    scale_end = search_splits(n, common[0], common[-1])["train"][1]
    good = [11, 1, 7, 2, 14, 7, 15, 6, 15, 18, 4, 22]
    with pytest.raises(TypeError, match="dict keyed by coin"):
        GA.evaluate(good, maps, returns, lockbox[0], lockbox[1], mask,
                    scale_end, common, funding_by_coin=mask, grammar="reduced")


def test_evaluate_refuses_a_dict_missing_the_coin():
    from research.splits_28c import search_splits, split_indices
    common, maps, returns, mask = GA.load_data(null="none")
    n = len(common)
    lockbox = split_indices(n, common[0], common[-1])["lockbox"]
    scale_end = search_splits(n, common[0], common[-1])["train"][1]
    good = [11, 1, 7, 2, 14, 7, 15, 6, 15, 18, 4, 22]
    with pytest.raises(TypeError, match="dict keyed by coin"):
        GA.evaluate(good, maps, returns, lockbox[0], lockbox[1], mask,
                    scale_end, common, funding_by_coin={}, grammar="reduced")


def test_the_live_constants_are_not_the_ones_assumed_by_default():
    """FEE and LEV are load-bearing and have been misremembered before.

    An earlier cost script assumed fee=0.0005 and lev=1.0; the real values are
    0.0004 and 2.0. The assumption did not crash, it just produced a 'net'
    column that disagreed with the recorded artifacts.
    """
    assert GA.FEE == 0.0004
    assert GA.LEV == 2.0
