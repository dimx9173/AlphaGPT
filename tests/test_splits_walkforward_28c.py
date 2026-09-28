"""The walk-forward protocol must not leak, or it is decoration.

A test that only checks the fold boundaries exist is worthless here. The
failure mode being guarded against is specific: a fold whose training window
shares information with its validation window through a rolling factor, or a
holdout that a later selection pass can still see. Both are silent -- the code
runs, the numbers look plausible, and the conclusion is wrong.
"""
from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from research.splits_walkforward_28c import (  # noqa: E402
    walk_forward_folds, fold_summaries, LONGEST_FACTOR_LOOKBACK)
from research.data_contract_28c import COMMON_28_BARS, EMBARGO_BARS  # noqa: E402

N = COMMON_28_BARS


def test_folds_are_chronological_and_expanding():
    """Training windows grow and never move backwards in time."""
    folds = walk_forward_folds(N, n_folds=4)
    for f in folds:
        assert f["train"][0] == 0, "every fold trains from the start"
    ends = [f["train"][1] for f in folds]
    assert ends == sorted(ends), "training window must expand, not shrink"
    assert ends[0] < ends[-1], "the last fold should see the most history"


def test_validation_never_overlaps_its_own_training():
    """The direct leak: a bar scored in validation also trained the model."""
    for f in walk_forward_folds(N, n_folds=4):
        train_start, train_end = f["train"]
        val_start, val_end = f["validation"]
        assert train_end <= val_start, (
            f"fold {f['index']} training overlaps validation: "
            f"train ends {train_end}, validation starts {val_start}")


def test_fold_boundary_is_purged_beyond_any_factor_lookback():
    """A rolling factor must not carry state across the fold boundary.

    The last factor that can still see the final training bar is the one whose
    window reaches into the validation segment. If that bar is within
    LONGEST_FACTOR_LOOKBACK of the first validation bar, the fold leaks.
    """
    for f in walk_forward_folds(N, n_folds=4):
        val_start = f["validation"][0]
        last_train_bar = f["train"][1] - 1
        gap = val_start - last_train_bar
        assert gap > LONGEST_FACTOR_LOOKBACK, (
            f"fold {f['index']} gap is {gap} bars, but a factor can look back "
            f"{LONGEST_FACTOR_LOOKBACK} bars; state crosses the boundary")


def test_holdout_is_the_final_fold_and_is_unique():
    """There must be exactly one holdout, and it must be last."""
    folds = walk_forward_folds(N, n_folds=4)
    holdouts = [f for f in folds if f["is_holdout"]]
    assert len(holdouts) == 1, "exactly one fold may be the holdout"
    assert holdouts[0]["index"] == len(folds) - 1, "holdout must be the last fold"
    assert holdouts[0]["validation"][1] == N, "holdout must run to the end of the sample"


def test_holdout_is_never_a_training_window():
    """No fold may train on any bar the holdout scores."""
    holdout_start = walk_forward_folds(N, n_folds=4)[-1]["validation"][0]
    for f in walk_forward_folds(N, n_folds=4):
        assert f["train"][1] < holdout_start, (
            f"fold {f['index']} trains up to {f['train'][1]}, "
            f"which reaches into the holdout starting {holdout_start}")


def test_folds_do_not_overlap_each_other():
    """Validation segments are disjoint, so no bar is scored twice."""
    folds = walk_forward_folds(N, n_folds=4)
    for a, b in zip(folds, folds[1:]):
        assert a["validation"][1] <= b["validation"][0], (
            f"fold {a['index']} validation overlaps fold {b['index']}")


def test_rejects_too_few_folds():
    with pytest.raises(ValueError):
        walk_forward_folds(N, n_folds=1)


def test_rejects_fold_count_that_collapses_the_sample():
    """Asking for 40 folds on a 2-year sample must fail loudly, not silently
    produce folds of a handful of bars."""
    with pytest.raises(ValueError):
        walk_forward_folds(N, n_folds=40)


def test_summaries_report_holdout_and_days():
    """The artifact must be able to state which fold was the holdout."""
    rows = fold_summaries(walk_forward_folds(N, n_folds=4), N)
    assert len(rows) == 4
    assert sum(r["is_holdout"] for r in rows) == 1
    for r in rows:
        assert r["validation_bars"] > 0
        assert r["validation_days"] > 0
