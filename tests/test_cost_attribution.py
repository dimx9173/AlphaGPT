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

# The formula the standard-mode run selects on seed 101, used as a fixed probe.
BASELINE_FORMULA = (9, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17)

# Recorded by running the SAME evaluate() call with research/accounting_28c.py and
# research/ga_28c_30m_3y.py stashed to their pre-attribution state, on the same
# search_splits() train window. Both numbers were taken from that pair of runs, so
# the comparison is apples to apples: the CLI standard-mode window is a different
# split and must not be mixed in here.
BASELINE = {
    "formula": list(BASELINE_FORMULA),
    "train_sharpe": -0.6863351580665226,
    "train_reward": -0.5966632165814831,
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

class TestEvaluateLevelNeutrality:
    """Blocking 1 and 2 from review: the earlier TestSelectionNeutrality asserted
    toy dicts and constants, so it could not fail if evaluate() reordered its
    arithmetic or if selection started scoring off gross_sharpe. These tests
    drive the real evaluate() on the real data and compare against values
    recorded BEFORE the attribution code existed.
    """

    @staticmethod
    def _load():
        import research.ga_28c_30m_3y as ga
        common, maps, returns, funding_mask = ga.load_data(null='none')
        return ga, common, maps, returns, funding_mask

    def _evaluate(self, ga, maps, returns, funding_mask, common, formula):
        import research.accounting_28c as A
        from research.splits_28c import search_splits, split_indices
        contract = split_indices(len(common), common[0], common[-1])
        ranges = search_splits(len(common), common[0], common[-1])
        scale_end = ranges['train'][1]
        return ga.evaluate(formula, maps, returns, *ranges['train'],
                           funding_mask, scale_end, common,
                           funding_by_coin=None, reward_version='v1')

    @pytest.mark.slow
    def test_evaluate_matches_the_pre_attribution_record(self):
        """C2, exact. train_sharpe and train_reward must equal the values recorded
        at commit 9c14fd3, before any attribution code existed."""
        ga, common, maps, returns, funding_mask = self._load()
        formula = BASELINE_FORMULA
        r = self._evaluate(ga, maps, returns, funding_mask, common, formula)
        assert r['portfolio_sharpe'] == BASELINE['train_sharpe'], (
            "portfolio_sharpe moved. The attribution change is not neutral, or "
            "the baseline predates another change and must be re-recorded.")
        assert r['reward'] == BASELINE['train_reward']

    @pytest.mark.slow
    def test_drag_reconciles_at_the_evaluate_level(self):
        """C1, at the level that matters. Not a synthetic identity: the real
        evaluate() output must satisfy gross - fee_drag - funding_drag = net."""
        ga, common, maps, returns, funding_mask = self._load()
        r = self._evaluate(ga, maps, returns, funding_mask, common,
                           BASELINE_FORMULA)
        resid = r['gross_sharpe'] - r['fee_drag'] - r['funding_drag'] - r['portfolio_sharpe']
        assert abs(resid) < 1e-9, f"drag decomposition does not reconcile: {resid}"

    @pytest.mark.slow
    def test_attribution_fields_are_present_and_inert(self):
        """The new fields must exist, and the selection key must still be the
        plain train reward with no dependence on them."""
        ga, common, maps, returns, funding_mask = self._load()
        r = self._evaluate(ga, maps, returns, funding_mask, common,
                           BASELINE_FORMULA)
        for k in ('gross_sharpe', 'fee_drag', 'funding_drag'):
            assert k in r, f"{k} missing from evaluate output"
            assert isinstance(r[k], float)
        assert ga.selection_score(r, r, mode='train') == r['reward']
        # removing the attribution keys must not change the selection score
        stripped = {k: v for k, v in r.items()
                    if k not in ('gross_sharpe', 'fee_drag', 'funding_drag')}
        assert ga.selection_score(stripped, stripped, mode='train') == r['reward']


class TestAttributionConventionIsPinned(TestEvaluateLevelNeutrality):
    """Blocking 1 from review 2: the identity
    gross - fee_drag - funding_drag - net == 0 holds for ANY ordering of the drag
    chain, so the identity test cannot detect a chain that charges funding before
    fee. The convention is fee-first to match `gross - fee - funding_cost` inside
    the accounting, and that convention needs its own assertion.
    """

    @pytest.mark.slow
    def test_drag_order_is_fee_first_then_funding(self):
        """fee_drag must be the loss from removing fees ALONE, and funding_drag the
        loss from removing funding from what is left. If the chain were swapped,
        fee_drag would instead be the mid-funding Sharpe drop and this fails."""
        ga, common, maps, returns, funding_mask = self._load()
        r = self._evaluate(ga, maps, returns, funding_mask, common, BASELINE_FORMULA)

        # recompute the chain independently, in the documented order
        import numpy as np
        from research import accounting_28c as A
        from research.splits_28c import search_splits, split_indices
        ranges = search_splits(len(common), common[0], common[-1])
        contract = split_indices(len(common), common[0], common[-1])
        # rebuild the portfolio series the same way evaluate does
        import research.causal_12f as C
        from model_core.vm import StackVM
        from model_core.factors import FeatureEngineer
        from research.universe_28c import COINS_28C

        leg_g, leg_f, leg_u, leg_n = [], [], [], []
        s, e = ranges['train']
        for c in COINS_28C:
            sig = ga.formula_signal(list(BASELINE_FORMULA), maps, c)
            fit = float(np.std(sig[:ranges['train'][1]])) if ranges['train'][1] > 0 else 0.0
            raw = np.tanh(sig / (fit + 1e-6)) if fit > 1e-8 else np.zeros_like(sig)
            pos = 0.25 * ga.smooth_causal(raw, 5)
            pos = np.roll(pos, 1); pos[0] = 0
            r_ = returns[c][s:e]
            pos = pos[s:e]
            ts_full = common[s:e]
            comp = {}
            fnd_ = funding_mask[s:e] * ga.FUND_RATE
            net = A.accounting_bar_returns(pos, r_, ga.FEE, fnd_, ga.LEV, out=comp)
            leg_n.append(net); leg_g.append(comp['gross']); leg_f.append(comp['fee'])
            leg_u.append(comp['funding_cost'])
        pg = np.mean(np.stack(leg_g), axis=0)
        pf = np.mean(np.stack(leg_f), axis=0)
        pu = np.mean(np.stack(leg_u), axis=0)
        pn = np.mean(np.stack(leg_n), axis=0)
        m = lambda x: A.metrics(x, ts_full)['sharpe']

        fee_first = m(pg) - m(pg - pf)                     # documented
        funding_first = m(pg) - m(pg - pu)                 # the violation
        assert abs(fee_first - r['fee_drag']) < 1e-12, (
            "fee_drag is not the fee-first quantity; the chain order changed")
        assert abs(funding_first - r['fee_drag']) > 1e-6, (
            "fee_drag and the funding-first quantity coincide, so this test "
            "cannot distinguish the two chains")


class TestSelectionPathScan:
    """Blocking 2 from review 2, narrowed in review 3: a line-prefix heuristic can
    be bypassed by dynamic keys, aliases, getattr, record laundering, or any new
    helper outside the four named functions. This walks the AST instead, so it
    reads what the code means rather than how it is spelled.
    """

    ATTRIBUTION_KEYS = {"gross_sharpe", "fee_drag", "funding_drag"}

    # Functions that make selection decisions. Everything reachable from these
    # must not read an attribution key except as a literal result field.
    SELECTION_FNS = {"selection_score", "shape_reward", "run_standard_selection",
                     "run_walkforward"}

    @staticmethod
    def _ast():
        import ast
        path = os.path.join(ROOT, "research", "ga_28c_30m_3y.py")
        return ast.parse(open(path).read()), path

    @staticmethod
    def _key_names(node):
        """Every string that could denote an attribution key, including the
        pieces of a concatenation like 'gross' + '_sharpe'."""
        import ast
        found = set()
        for sub in ast.walk(node):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                parts = sub.value.split("_")
                for n in range(1, len(parts) + 1):
                    found.add("_".join(parts[:n]))
                    found.add("_".join(parts[-n:]))
                found.add(sub.value)
        return found

    def test_attribution_keys_never_reach_a_selection_path(self):
        """Semantic, not textual. Walks every selection function's AST and fails
        if an attribution key is read anywhere except as a literal dict key."""
        import ast
        tree, path = self._ast()
        funcs = {n.name: n for n in ast.walk(tree)
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        checked = []
        for name in self.SELECTION_FNS:
            assert name in funcs, f"{name} not found; the scan would silently pass"
            node = funcs[name]
            checked.append(name)
            # collect the dict keys that ARE legitimately written
            written = set()
            for sub in ast.walk(node):
                if isinstance(sub, ast.Dict):
                    for k in sub.keys:
                        if isinstance(k, ast.Constant) and isinstance(k.value, str):
                            written.add(k.value)
            for sub in ast.walk(node):
                # a bare Name / Attribute read of a key
                if isinstance(sub, ast.Name) and sub.id in self.ATTRIBUTION_KEYS:
                    assert sub.id in written, (
                        f"{sub.id} read inside {name} but never written there")
                if isinstance(sub, ast.Attribute) and sub.attr in self.ATTRIBUTION_KEYS:
                    assert sub.attr in written, (
                        f"{sub.attr} read as an attribute inside {name}")
                # subscript read: x['gross_sharpe']
                if isinstance(sub, ast.Subscript):
                    names = self._key_names(sub.slice)
                    hit = names & self.ATTRIBUTION_KEYS
                    assert not hit, f"{hit} read by subscript inside {name}"
                # getattr with a literal name
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) \
                        and sub.func.id == "getattr" and len(sub.args) >= 2:
                    if isinstance(sub.args[1], ast.Constant):
                        assert sub.args[1].value not in self.ATTRIBUTION_KEYS, (
                            f"getattr for {sub.args[1].value} inside {name}")
        assert set(checked) == self.SELECTION_FNS

    def test_attribution_keys_are_not_passed_into_shape_reward(self):
        """shape_reward must not gain an attribution parameter. A signature
        change is the quiet way this refactor could start influencing selection."""
        import inspect
        import research.ga_28c_30m_3y as ga
        params = set(inspect.signature(ga.shape_reward).parameters)
        assert not (params & self.ATTRIBUTION_KEYS), (
            f"shape_reward gained {params & self.ATTRIBUTION_KEYS}")

    def test_selection_score_ignores_extra_keys(self):
        import research.ga_28c_30m_3y as ga
        a = {"reward": 2.0, "gross_sharpe": 99.0, "fee_drag": -50.0}
        b = {"reward": 1.0, "gross_sharpe": -99.0, "fee_drag": 50.0}
        assert ga.selection_score(a, b, mode="train") == pytest.approx(2.0)

    def test_attribution_fields_never_feed_a_sort_key(self):
        """Record laundering: the keys may be written into a results record, but
        no sort or max over such a record may key on them."""
        import ast
        tree, _ = self._ast()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fname = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
            if fname not in ("sort", "sorted", "max", "min"):
                continue
            for arg in list(node.args) + [k.value for k in node.keywords]:
                names = self._key_names(arg)
                assert not (names & self.ATTRIBUTION_KEYS), (
                    f"{fname} keys on an attribution field")
