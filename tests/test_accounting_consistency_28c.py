"""Guards for two defects that were silent because they never crashed.

1. `threshold_fit_28c_3y.py` passed a zero funding array into the shared
   accounting, modelling a funding-free perpetual and overstating every long
   threshold. It looked correct because `accounting_bar_returns` accepts a
   zero array without complaint.
2. `train_12f_30m.py` emitted bare `acceptance_passed: False` /
   `live_adopted: False` literals, which read as "a gate ran and did not pass".
   Its reward is not equity-compound-v2, so the honest answer is an explicit
   fail-closed record saying no gate was evaluated.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from research.acceptance_schema_28c import FAIL  # noqa: E402
from research.accounting_28c import (  # noqa: E402
    FUND_RATE, accounting_bar_returns, scheduled_funding_rates,
)

BARS_PER_DAY = 48


# a known 00:00 UTC bar, so index 0 is a funding event
_MIDNIGHT_MS = 1725667200000


def _ts(n):
    return _MIDNIGHT_MS + np.arange(n) * 1_800_000


# ------------------------------------------------------ funding is not optional

def test_threshold_fit_does_not_pass_a_zero_funding_array():
    src = open(os.path.join(ROOT, "research", "threshold_fit_28c_3y.py")).read()
    # the call site must derive funding from timestamps, not a zero array
    assert "np.zeros_like(pos), LEV" not in src, \
        "threshold_fit still passes a zero funding array to accounting_bar_returns"
    assert "scheduled_funding_rates(timestamps[start:end]" in src


def test_scheduled_funding_is_non_zero_and_sparse():
    ts = _ts(BARS_PER_DAY * 20)
    fnd = scheduled_funding_rates(ts, FUND_RATE)
    assert set(np.unique(fnd)) <= {0.0, FUND_RATE}
    assert fnd[0] == FUND_RATE                      # 00:00 UTC
    assert fnd[8 * BARS_PER_DAY] == FUND_RATE        # 08:00 UTC
    assert fnd[16 * BARS_PER_DAY] == FUND_RATE       # 16:00 UTC
    assert int((fnd != 0).sum()) == 60               # exactly 3 per day x 20


def test_funding_changes_the_result_it_is_applied_to():
    """A funding-free perpetual is not a small difference; it is a different
    strategy. This is the assertion that would have caught the defect."""
    n = BARS_PER_DAY * 30
    ts = _ts(n)
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.004, n); r[0] = 0.0
    pos = np.full(n, 0.25)
    with_f = accounting_bar_returns(pos, r, 0.0004, scheduled_funding_rates(ts, FUND_RATE), 2.0)
    without = accounting_bar_returns(pos, r, 0.0004, np.zeros(n), 2.0)
    assert not np.allclose(with_f, without)
    # a long pays funding, so the funded leg is strictly worse here
    assert with_f.sum() < without.sum()


# ------------------------------------------------------ honest fail-closed record

def test_train_12f_emits_no_bare_false_literals():
    src = open(os.path.join(ROOT, "research", "train_12f_30m.py")).read()
    assert '"live_adopted": False' not in src
    assert '"acceptance_passed": False' not in src
    assert "derive_live_adopted(acceptance)" in src


def test_train_12f_records_that_no_gate_was_run():
    src = open(os.path.join(ROOT, "research", "train_12f_30m.py")).read()
    assert '"acceptance_gate_run": False' in src
    assert "make_acceptance(" in src
    # the reason must name the real cause, not just "fail"
    assert "MemeBacktest" in src
    assert "equity-compound-v2" in src


def test_train_12f_record_is_fail_closed_and_explained():
    from research.acceptance_schema_28c import derive_live_adopted, make_acceptance
    acc = make_acceptance(
        FAIL, "no regime gate evaluated: legacy MemeBacktest reward path",
        min_positive_coins=24, oos_mdd=False, oos_solvent=False,
        oos_portfolio_sharpe=0.0, legacy_bar=1.0, criteria={})
    assert acc["verdict"] == FAIL
    assert acc["verdict_reason"]
    assert derive_live_adopted(acc) is False
