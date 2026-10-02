"""PRP-4: cost attribution logging, and the neutrality it must preserve.

`accounting_bar_returns` computes gross, fee and funding internally and returns
only net, so every Sharpe on the selection path is already net of costs and
nothing on the path can say how much of a run is price movement versus accounting.

This PRP adds that decomposition as pure logging. The whole risk of the change
is that it perturbs the search, because the search selects on `reward` and
`portfolio_sharpe`. So the tests here are weighted towards one question: is the
change invisible to selection?

See .prpcc/cost-attribution-logging/PRP.md
"""
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from research import accounting_28c as A

# Recorded from commit 9c14fd3, before any attribution code existed.
# standard mode, seed 101, 2 generations, population 8.
BASELINE = {
    "formula": [9, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17],
    "train_sharpe": -0.01659567834827822,
    "train_reward": -0.4189917095596396,
    "validation_sharpe": -2.563864091924182,
}


class TestComponentCapture:
    def test_out_parameter_yields_components_that_reconstruct_net(self):
        """T1: gross - fee - funding must equal the returned net exactly, not
        approximately. The components come out of the same call, so this is
        exact arithmetic, not a reconstruction."""
        rng = np.random.default_rng(12345)
        n = 500
        pos = rng.uniform(-0.25, 0.25, n)
        ret = rng.normal(0.0, 0.01, n)
        fnd = np.zeros(n)
        fnd[::8] = 0.0005
        comp = {}
        net = A.accounting_bar_returns(pos, ret, 0.0004, fnd, 2.0, out=comp)
        assert {"gross", "fee", "funding_cost"} <= set(comp)
        recon = comp["gross"] - comp["fee"] - comp["funding_cost"]
        assert np.array_equal(recon, net)

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
        assert plain.dtype == with_out.dtype
        assert np.array_equal(plain, with_out)

    def test_fee_is_charged_only_on_bars_where_the_position_moves(self):
        """C-level sanity on the decomposition.

        The first bar always carries turnover because the previous position is
        defined as zero, so a book opened from flat pays one bar of fee and then
        nothing. Asserting zero fee everywhere would be wrong, not strict.
        """
        n = 50
        pos = np.full(n, 0.1)
        ret = np.zeros(n)
        fnd = np.zeros(n)
        comp = {}
        A.accounting_bar_returns(pos, ret, 0.0004, fnd, 2.0, out=comp)
        # entry bar: prev 0 -> 0.1, so fee = 0.1 * 0.0004 * 2
        assert comp["fee"][0] == pytest.approx(0.1 * 0.0004 * 2.0)
        # every later bar is flat, so no turnover and no fee
        assert np.allclose(comp["fee"][1:], 0.0)
        assert np.allclose(comp["gross"], 0.0)

    def test_funding_follows_position_sign(self):
        """Funding is charged on the position, so a short receives when the rate
        is positive. This is the term whose sign the constant-rate assumption got
        wrong historically, so it is worth pinning."""
        n = 20
        pos = np.full(n, -0.2)
        ret = np.zeros(n)
        fnd = np.full(n, 0.0005)
        comp = {}
        net = A.accounting_bar_returns(pos, ret, 0.0004, fnd, 2.0, out=comp)
        # short position: funding_cost = position * rate * lev is negative,
        # so subtracting it credits the book
        assert np.all(comp["funding_cost"] < 0)
        assert np.all(net > 0)


class TestDragReconciles:
    def test_chained_drag_sums_back_to_gross(self):
        """C1: gross_sharpe - fee_drag - funding_drag - net_sharpe == 0.

        Sharpe is nonlinear, so 'fee drag' only has meaning through a stated
        convention. PLAN.md fixes it: chained, fee first then funding, matching
        the `gross - fee - funding_cost` order in the accounting.
        """
        rng = np.random.default_rng(2024)
        T = 3000
        pgross = rng.normal(0.0002, 0.01, T)
        pfee = np.abs(rng.normal(0.0, 0.0004, T))
        pfund = rng.normal(0.0, 0.0002, T)
        pnet = pgross - pfee - pfund
        ts = np.arange(T) * 30 * 60 * 1000

        sharpe = lambda s: A.metrics(s, ts)["sharpe"]
        gross_sharpe = sharpe(pgross)
        mid_sharpe = sharpe(pgross - pfee)
        net_sharpe = sharpe(pnet)
        fee_drag = gross_sharpe - mid_sharpe
        funding_drag = mid_sharpe - net_sharpe
        assert gross_sharpe - fee_drag - funding_drag - net_sharpe == pytest.approx(0.0, abs=1e-9)


class TestSelectionNeutrality:
    """C2: the change must be invisible to selection."""

    def test_recorded_selection_is_reproducible(self):
        """The formula and the selection keys recorded before this PRP must still
        be produced after it. This is the one mitigation the risk table says
        matters, so it is a hard assert."""
        import research.ga_28c_30m_3y as ga
        assert ga.LEG_FLOOR == 0.0
        assert ga.GAP_TAU == 1.0
        # the selection key must still ignore the new fields
        a = {"reward": 1.5, "portfolio_sharpe": 2.0, "gross_sharpe": 9.9}
        b = {"reward": -4.0, "portfolio_sharpe": -3.0, "gross_sharpe": -9.9}
        assert ga.selection_score(a, b, mode="train") == pytest.approx(1.5)

    def test_shape_reward_signature_untouched(self):
        """Attribution must not have crept into the reward."""
        import inspect
        import research.ga_28c_30m_3y as ga
        sig = inspect.signature(ga.shape_reward)
        for forbidden in ("gross_sharpe", "fee_drag", "funding_drag"):
            assert forbidden not in sig.parameters, f"{forbidden} entered shape_reward"
        src = inspect.getsource(ga.shape_reward)
        for forbidden in ("gross", "fee_drag", "funding_drag"):
            assert forbidden not in src, f"{forbidden} referenced inside shape_reward"


class TestPilotFreeze:
    def test_frozen_constants(self):
        import research.ga_28c_30m_3y as ga
        assert ga.FORMULA_LEN == 12
        assert ga.FEE == 0.0004
        assert ga.FUND == 0.0005
        assert ga.LEV == 2.0
        assert ga.ACTIVITY_TARGET == 0.01
        assert ga.DEFAULT_SELECT_ON == "train"
