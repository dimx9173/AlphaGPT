"""Expanding-window walk-forward folds for the 28c contract.

The existing contract uses one train window and one lockbox. That is enough
to measure a candidate, but not enough to measure a search: a genetic search
over 100 generations is allowed to see the same validation window tens of
thousands of times, and none of that is penalised. A lockbox inspected ten
times is no longer a lockbox, which is exactly what the null control exposed
-- ten real searches and ten null searches produced indistinguishable
distributions, so the honest question is no longer "does this formula work"
but "does a search that claims to work survive a protocol that gives it no
reusable validation window".

This module builds expanding-window folds. Fold i trains on everything
strictly before its validation segment, and fold i's validation segment is
never part of any later fold's training data in a way that lets the search
"learn from it twice" -- the folds are chronological, and the final fold is
reserved as a one-shot holdout that is scored only after selection is
complete.

Purging: a validation segment is bracketed by an embargo on both sides equal
to the longest lookback any factor can use. Without that, a training bar
immediately before the validation window can share information with the first
validation bar through a rolling factor, and the fold boundary leaks.
"""
from __future__ import annotations

import numpy as np

from .data_contract_28c import EMBARGO_BARS

# The longest trailing window any factor in research/causal_12f.py uses.
# NORM_WINDOW is 200; CORR and DECAY also look back. Taking 200 bars on each
# side means a fold boundary cannot be crossed by any factor state.
LONGEST_FACTOR_LOOKBACK = 200


def walk_forward_folds(n: int, n_folds: int = 4,
                       embargo_bars: int = EMBARGO_BARS,
                       lookback: int = LONGEST_FACTOR_LOOKBACK) -> list[dict]:
    """Expanding-window folds, each with a train window and a validation window.

    Fold 0 trains on the first ``base`` bars. Each later fold extends the
    training window to the start of the new validation segment, so information
    only ever flows forward in time.

    Returns a list of dicts, each with:
      index        fold number, 0-based
      train        (start, end) half-open bar range
      validation   (start, end) half-open bar range
      is_holdout   True only for the final fold

    The final fold is the holdout. It must be scored once, after selection,
    and its result is not allowed to influence anything.
    """
    if n_folds < 2:
        raise ValueError("walk-forward needs at least 2 folds")
    if embargo_bars < 0:
        raise ValueError("embargo must be non-negative")
    if lookback < 0:
        raise ValueError("lookback must be non-negative")

    # Each fold needs a validation window that survives purging on both sides
    # AND is long enough to measure anything. A fold with a few hundred bars
    # produces a Sharpe with a confidence interval wider than the number
    # itself, which is how a search ends up "validated" on noise. The floor is
    # the larger of the purge cost and a 30-day window at 30m bars.
    min_validation = n // (n_folds + 1)
    purge_cost = 2 * embargo_bars + lookback
    minimum_useful = max(purge_cost, 30 * 48)   # 30 days
    if min_validation <= minimum_useful:
        raise ValueError(
            f"cannot build {n_folds} folds: each validation window would be "
            f"{min_validation} bars, below the {minimum_useful}-bar floor "
            f"(purge costs {purge_cost}, 30-day window is {30 * 48})"
        )

    folds: list[dict] = []
    validation_len = n // (n_folds + 1)
    for i in range(n_folds):
        val_start = n - (n_folds - i) * validation_len
        val_end = n - (n_folds - i - 1) * validation_len
        if i == n_folds - 1:
            val_end = n
        # Purge on both sides: the last `embargo_bars` of training are dropped
        # and the first `embargo_bars` of validation are dropped.
        train_end = val_start - embargo_bars
        val_start_purged = val_start + embargo_bars
        if train_end <= lookback or val_start_purged >= val_end:
            raise ValueError(f"fold {i} collapsed under embargo/lookback")
        folds.append({
            "index": i,
            "train": (0, train_end),
            "validation": (val_start_purged, val_end),
            "is_holdout": i == n_folds - 1,
        })
    return folds


def fold_summaries(folds: list[dict], n: int, bar_minutes: int = 30) -> list[dict]:
    """Human-readable fold table, for logging and artifact provenance."""
    out = []
    for f in folds:
        out.append({
            "fold": f["index"],
            "train_start": f["train"][0],
            "train_end": f["train"][1],
            "train_bars": f["train"][1] - f["train"][0],
            "validation_start": f["validation"][0],
            "validation_end": f["validation"][1],
            "validation_bars": f["validation"][1] - f["validation"][0],
            "is_holdout": f["is_holdout"],
            "validation_days": (f["validation"][1] - f["validation"][0]) * bar_minutes / 1440.0,
        })
    return out
