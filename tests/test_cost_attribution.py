"""PRP-4: cost attribution, and the structural reason it cannot affect selection.

The first implementation put gross/fee/funding inside `evaluate()`. It was
behaviourally neutral, and it was still the wrong shape: `evaluate()` is where
`reward` is built, so the fields sat one line from the only value selection
depends on, and proving they never touched it required a source scan. Six review
rounds found one bypass after another in that scan, which is what happens when a
check has to prove a negative about code paths.

The fix is structural. `evaluate()` is byte-identical to its pre-PRP-4 form and
mentions none of the attribution fields. The decomposition lives in
`research/cost_attribution_28c.py` and runs after the reward is fixed.

These tests therefore check three things:

  1. the accounting contract, which is the only shared surface
  2. that evaluate() is untouched, by comparing its source to a recorded copy
  3. that the attribution module reconciles its own drags

There is deliberately no scan of the selection path. There is nothing to scan:
the fields do not exist anywhere in the evaluator.
"""
import ast
import json
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from research import accounting_28c as A
from research import cost_attribution_28c as CA

GA_PATH = os.path.join(ROOT, "research", "ga_28c_30m_3y.py")

# The commit at which the attribution was added. evaluate() must be identical to
# its source there.
PRP4_START = "8110505"

ATTRIBUTION_KEYS = {"gross_sharpe", "fee_drag", "funding_drag"}


class TestAccountingContract:
    def test_out_parameter_yields_components_that_reconstruct_net(self):
        """gross - fee - funding must equal the returned net exactly. These come
        out of the same call, so this is exact arithmetic, not a reconstruction."""
        rng = np.random.default_rng(12345)
        n = 500
        pos = rng.uniform(-0.25, 0.25, n)
        ret = rng.normal(0.0, 0.01, n)
        fnd = np.zeros(n)
        fnd[::8] = 0.0005
        comp = {}
        net = A.accounting_bar_returns(pos, ret, 0.0004, fnd, 2.0, out=comp)
        assert {"gross", "fee", "funding_cost"} <= set(comp)
        assert np.array_equal(comp["gross"] - comp["fee"] - comp["funding_cost"], net)

    def test_return_is_unchanged_when_out_is_omitted(self):
        """The default path must be bit-identical, not merely close. Every other
        caller in the repo passes no `out`."""
        rng = np.random.default_rng(999)
        n = 400
        pos = rng.uniform(-0.25, 0.25, n)
        ret = rng.normal(0.0, 0.01, n)
        fnd = np.zeros(n)
        fnd[::8] = -0.0003
        plain = A.accounting_bar_returns(pos, ret, 0.0004, fnd, 2.0)
        comp = {}
        with_out = A.accounting_bar_returns(pos, ret, 0.0004, fnd, 2.0, out=comp)
        assert isinstance(plain, np.ndarray), "return shape must not become a tuple"
        assert np.array_equal(plain, with_out)

    def test_fee_charged_only_where_the_position_moves(self):
        """The first bar always carries turnover because the previous position is
        defined as zero, so a book opened from flat pays one bar and then nothing.
        Asserting zero fee everywhere would be wrong, not strict."""
        n = 50
        comp = {}
        A.accounting_bar_returns(np.full(n, 0.1), np.zeros(n), 0.0004, np.zeros(n),
                                 2.0, out=comp)
        assert comp["fee"][0] == pytest.approx(0.1 * 0.0004 * 2.0)
        assert np.allclose(comp["fee"][1:], 0.0)
        assert np.allclose(comp["gross"], 0.0)

    def test_funding_follows_position_sign(self):
        """A short receives when the rate is positive. This is the term whose sign
        the constant-rate assumption historically got wrong, so it is pinned."""
        n = 20
        comp = {}
        net = A.accounting_bar_returns(np.full(n, -0.2), np.zeros(n), 0.0004,
                                       np.full(n, 0.0005), 2.0, out=comp)
        assert np.all(comp["funding_cost"] < 0)
        assert np.all(net > 0)


class TestEvaluateIsUntouched:
    """The structural guarantee. If this passes, no scan is needed, because the
    attribution fields do not appear in the evaluator at all."""

    def test_evaluate_source_matches_the_pre_attribution_commit(self):
        prev = subprocess.run(["git", "show", f"{PRP4_START}:research/ga_28c_30m_3y.py"],
                               cwd=ROOT, capture_output=True, text=True)
        assert prev.returncode == 0, f"cannot read {PRP4_START}: git show failed"

        def extract(src):
            tree = ast.parse(src)
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name == "evaluate":
                    return ast.dump(node)
            raise AssertionError("evaluate not found")

        assert extract(prev.stdout) == extract(open(GA_PATH).read()), (
            "evaluate() has changed since the attribution was added. If that is "
            "intended, this PRP's neutrality claim no longer holds and the change "
            "needs its own review.")

    def test_attribution_fields_appear_nowhere_in_the_evaluator(self):
        src = open(GA_PATH).read()
        for key in ATTRIBUTION_KEYS:
            assert key not in src, (
                f"{key} appears in the GA module. The attribution belongs in "
                "research/cost_attribution_28c.py; putting it back in evaluate() "
                "reopens the leak this design removed.")

    def test_shape_reward_signature_unchanged(self):
        import inspect
        import research.ga_28c_30m_3y as ga
        assert not (set(inspect.signature(ga.shape_reward).parameters) & ATTRIBUTION_KEYS)

    def test_selection_score_is_exactly_the_train_reward(self):
        """Exact, not approx. An epsilon-weighted leak such as
        reward + 1e-9 * something would rank differently while staying inside a
        relative tolerance, and it is exercised with both a large and a tiny
        value so a small coefficient cannot hide either."""
        import research.ga_28c_30m_3y as ga
        for big, tiny in ((99.0, 1e-9), (-99.0, -1e-12)):
            a = {"reward": 2.0, "gross_sharpe": big, "fee_drag": tiny}
            b = {"reward": 1.0, "gross_sharpe": -big, "fee_drag": -tiny}
            assert ga.selection_score(a, b, mode="train") == 2.0
            assert ga.selection_score(b, a, mode="train") == 1.0


class TestAttributionReconciles:
    def test_chained_drag_sums_back_to_gross(self):
        rng = np.random.default_rng(2024)
        T = 3000
        pgross = rng.normal(0.0002, 0.01, T)
        pfee = np.abs(rng.normal(0.0, 0.0004, T))
        pfund = rng.normal(0.0, 0.0002, T)
        ts = np.arange(T) * 30 * 60 * 1000
        r = CA.attribute(pgross - pfee - pfund, pgross, pfee, pfund, ts)
        resid = r["gross_sharpe"] - r["fee_drag"] - r["funding_drag"] - r["net_sharpe"]
        assert resid == pytest.approx(0.0, abs=1e-9)

    def test_drag_order_is_fee_first(self):
        """The identity holds for ANY chain order, so it cannot detect a swap.
        fee_drag must be the loss from removing fees alone, which is different
        from the loss from removing funding first."""
        rng = np.random.default_rng(77)
        T = 2000
        pgross = rng.normal(0.0003, 0.008, T)
        pfee = np.abs(rng.normal(0.0, 0.0005, T))
        pfund = rng.normal(0.0001, 0.0002, T)
        ts = np.arange(T) * 30 * 60 * 1000
        r = CA.attribute(pgross - pfee - pfund, pgross, pfee, pfund, ts)

        fee_first = (A.metrics(pgross, ts)["sharpe"]
                     - A.metrics(pgross - pfee, ts)["sharpe"])
        funding_first = (A.metrics(pgross, ts)["sharpe"]
                         - A.metrics(pgross - pfund, ts)["sharpe"])
        assert r["fee_drag"] == pytest.approx(fee_first, abs=1e-12)
        assert abs(funding_first - fee_first) > 1e-6, (
            "the two orders coincide here, so this test cannot discriminate them")

    def test_attribution_matches_the_accounting_it_decomposes(self):
        """The module must not re-derive the net series. It takes the accounting's
        own output, so a change to the accounting cannot silently desynchronise
        the attribution."""
        rng = np.random.default_rng(5)
        n = 300
        pos = rng.uniform(-0.25, 0.25, n)
        ret = rng.normal(0.0, 0.01, n)
        fnd = np.zeros(n)
        fnd[::4] = 0.0004
        net, g, f, u = CA.decompose_leg(pos, ret, 0.0004, fnd, 2.0)
        plain = A.accounting_bar_returns(pos, ret, 0.0004, fnd, 2.0)
        assert np.array_equal(net, plain)
        assert np.array_equal(g - f - u, net)


class TestPilotFreeze:
    def test_frozen_constants(self):
        import research.ga_28c_30m_3y as ga
        assert ga.FORMULA_LEN == 12
        assert ga.FEE == 0.0004
        assert ga.FUND == 0.0005
        assert ga.LEV == 2.0
        assert ga.ACTIVITY_TARGET == 0.01
        assert ga.DEFAULT_SELECT_ON == "train"
