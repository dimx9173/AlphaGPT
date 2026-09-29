"""The iid null is a frozen control and must stay frozen.

Every "the search harvests noise, not edge" artifact in this project was
measured against one specific iid null, and its reported Sharpe is quoted in
the write-ups. Refactoring the null construction -- even to fix a real
numerical bug -- changes which experiment those artifacts describe, while
leaving the name, the CLI flag and the artifact schema untouched.

That is the same failure as moving a benchmark after seeing the result. It
happened here: routing the iid null through the numerically safe
_rebuild_bars reconstruction shifted the price path and moved the re-scored
lockbox Sharpe by up to 0.36, which is enough to move a median and to change
the sign of a real-vs-null comparison.

The diagnostic nulls (iid1/iidw/iidg) are deliberately NOT frozen, because
they were introduced together with the new reconstruction and have no
historical artifact depending on them.

The recorded iid batch is the reference. If this test fails, either the
control moved or the artifacts are stale; both need investigating, and neither
may be resolved by re-running the batch.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import research.ga_28c_30m_3y as GA
from research.accounting_28c import load_real_funding
from research.splits_28c import search_splits, split_indices

ROOT = Path(__file__).resolve().parents[1]
IID_BATCH = ROOT / "results" / "gram_reduced_runs"

pytestmark = pytest.mark.skipif(
    not IID_BATCH.exists(), reason="iid batch not present")


def _rescore(prefix: str, null: str):
    common, maps, returns, mask = GA.load_data(null=null)
    funding = {c: load_real_funding(c, np.asarray(common, dtype=np.int64))
               for c in GA.COINS_28C}
    n = len(common)
    lockbox = split_indices(n, common[0], common[-1])["lockbox"]
    scale_end = search_splits(n, common[0], common[-1])["train"][1]
    out = []
    for f in sorted(IID_BATCH.glob(f"{prefix}*.json")):
        a = json.loads(f.read_text())
        r = GA.evaluate(a["formula"], maps, returns, lockbox[0], lockbox[1],
                        mask, scale_end, common, funding_by_coin=funding,
                        grammar="reduced")
        out.append((a["oos"]["portfolio_sharpe"], r["portfolio_sharpe"]))
    return out


def test_iid_null_reproduces_its_recorded_batch():
    """Every iid artifact must re-score to its recorded Sharpe.

    1e-5 rather than exact: re-running the accumulation in a different order
    changes the last few ulps, and on a Sharpe near 1.9 that shows in the 5th
    decimal. 0.36 is what moving the control looks like, and this tolerance is
    four orders of magnitude below it.
    """
    pairs = _rescore("null_", "iid")
    assert len(pairs) == 10
    for recorded, rescored in pairs:
        assert abs(recorded - rescored) <= 1e-5, (
            f"iid null moved: recorded {recorded:+.6f}, rescored "
            f"{rescored:+.6f}. Either the control construction changed or "
            "these artifacts are stale. Re-running the batch is not a fix: "
            "it replaces the control rather than restoring it.")


def test_the_real_batch_reproduces_its_recorded_batch():
    pairs = _rescore("real_", "none")
    assert len(pairs) == 10
    for recorded, rescored in pairs:
        assert abs(recorded - rescored) <= 1e-5


def test_the_iid_median_is_the_published_one():
    """Pin the headline number so a silent shift cannot pass review.

    +0.822 is quoted in docs/reduced_grammar_null_calibration_28c.md as the
    null median that beats the real data. If the control moves, every
    comparison built on that sentence has to be re-derived.
    """
    vals = [b for _, b in _rescore("null_", "iid")]
    assert abs(float(np.median(vals)) - 0.822) < 5e-3
