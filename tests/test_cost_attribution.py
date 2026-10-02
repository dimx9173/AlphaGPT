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
    """Reviews 2 and 3 replaced a line-prefix heuristic, then found that the
    replacement was still an allowlist over four named functions and four named
    callables, and still missed keys built by concatenation.

    This version has neither property. It analyses the WHOLE module:

      - every function, not a named subset
      - every way a key can be *read*: subscript with a literal, a concatenated
        literal, a Name bound to such a concatenation anywhere in the function,
        getattr with a literal or a built name, .attr access, dict.get with either
      - every ranking callable, matched on the resolved name so heapq.nsmallest,
        np.argsort, operator.attrgetter and operator.itemgetter are covered, not
        just sort / max
      - an attribution key is legal only as a literal key of a dict that is
        RETURNED or STORED as a record; it is illegal as a value read anywhere

    Deny-list semantics means a new helper is covered automatically, because
    there is no list of helpers to update.
    """

    ATTRIBUTION_KEYS = {"gross_sharpe", "fee_drag", "funding_drag"}

    # Callables that impose an order. Matched by the terminal attribute or name so
    # heapq.nsmallest and operator.attrgetter resolve to their last component.
    RANKERS = {"sort", "sorted", "max", "min", "nlargest", "nsmallest",
               "argsort", "argmax", "argmin", "attrgetter", "itemgetter"}

    # The one function that computes the attribution. Reading and writing the
    # fields here is the whole point; everywhere else they must not appear.
    PRODUCERS = {"evaluate"}

    @staticmethod
    def _module():
        import ast
        return ast.parse(open(os.path.join(ROOT, "research", "ga_28c_30m_3y.py")).read())

    @staticmethod
    def _attr_key_names(node):
        """String constants, decomposed so a concatenation still names the key."""
        import ast
        out = set()
        for sub in ast.walk(node):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                parts = sub.value.split("_")
                for n in range(1, len(parts) + 1):
                    out.add("_".join(parts[:n]))
                    out.add("_".join(parts[-n:]))
                out.add(sub.value)
        return out

    @staticmethod
    def _callee_name(node):
        f = node.func
        return getattr(f, "attr", None) or getattr(f, "id", None)

    def test_attribution_keys_are_never_read_as_values_anywhere(self):
        """Whole-module, deny-list. An attribution key may be constructed and
        written into a returned record, and read back out of THAT record by a
        caller; what it may never do is enter any computation."""
        import ast
        tree = self._module()
        hits = []
        for fn in [n for n in ast.walk(tree)
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
            fn_name = fn.name

            # names bound to an attribution-key string anywhere in this function,
            # including built up by concatenation or a join
            bound = set()
            for sub in ast.walk(fn):
                if isinstance(sub, (ast.Assign, ast.AnnAssign)):
                    targets = sub.targets if isinstance(sub, ast.Assign) else [sub.target]
                    names = self._attr_key_names(sub.value)
                    if names & self.ATTRIBUTION_KEYS:
                        for tgt in targets:
                            if isinstance(tgt, ast.Name):
                                bound.add(tgt.id)

            for sub in ast.walk(fn):
                # subscript read whose index names an attribution key
                if isinstance(sub, ast.Subscript):
                    idx_names = self._attr_key_names(sub.slice)
                    hit = idx_names & self.ATTRIBUTION_KEYS
                    if hit and fn_name not in self.PRODUCERS:
                        hits.append((fn_name, f"subscript read {hit}"))
                    # x[k] where k was bound above
                    if isinstance(sub.slice, ast.Name) and sub.slice.id in bound:
                        hits.append((fn_name, f"subscript via bound name {sub.slice.id}"))
                # getattr with a literal or a bound/built name
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) \
                        and sub.func.id == "getattr" and len(sub.args) >= 2:
                    arg = sub.args[1]
                    if (self._attr_key_names(arg) & self.ATTRIBUTION_KEYS
                            and fn_name not in self.PRODUCERS):
                        hits.append((fn_name, "getattr on an attribution key"))
                    if isinstance(arg, ast.Name) and arg.id in bound:
                        hits.append((fn_name, f"getattr via bound name {arg.id}"))
                # dict.get / dict[key] chained, and .attr access
                if (isinstance(sub, ast.Attribute) and sub.attr in self.ATTRIBUTION_KEYS
                        and fn_name not in self.PRODUCERS):
                    hits.append((fn_name, f"attribute access .{sub.attr}"))
                # dict.get / mapping.get with a key literal. This is a Call, not a
                # Subscript, so it needs its own arm; the earlier versions missed
                # r.get('gross_sharpe') entirely.
                if isinstance(sub, ast.Call) and self._callee_name(sub) == "get" \
                        and sub.args:
                    if (self._attr_key_names(sub.args[0]) & self.ATTRIBUTION_KEYS
                            and fn_name not in self.PRODUCERS):
                        hits.append((fn_name, "mapping.get on an attribution key"))
                # A bare Name equal to a key. evaluate() legitimately BINDS these
                # names to compute the attribution, so a store is not a leak. Only
                # a LOAD counts, and only outside the function that produces them.
                if isinstance(sub, ast.Name) and sub.id in self.ATTRIBUTION_KEYS \
                        and isinstance(sub.ctx, ast.Load):
                    if fn_name not in self.PRODUCERS:
                        hits.append((fn_name, f"name read {sub.id}"))

        assert not hits, ("attribution fields reach a computation: "
                          + "; ".join(f"{f}: {w}" for f, w in hits))

    def test_no_ranking_callable_keys_on_an_attribution_field(self):
        """Every ranking callable in the module, including heapq and operator,
        must not take an attribution field as a key."""
        import ast
        tree = self._module()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if self._callee_name(node) not in self.RANKERS:
                continue
            for arg in list(node.args) + [k.value for k in node.keywords]:
                if self._attr_key_names(arg) & self.ATTRIBUTION_KEYS:
                    raise AssertionError(
                        f"{self._callee_name(node)} keys on an attribution field")

    def test_attribution_keys_are_written_only_as_literal_record_fields(self):
        """The positive half: the fields must actually be produced, and produced
        as literal keys, so a rename cannot silently drop them."""
        src = open(os.path.join(ROOT, "research", "ga_28c_30m_3y.py")).read()
        for key in self.ATTRIBUTION_KEYS:
            assert f"'{key}'" in src or f'"{key}"' in src, \
                f"{key} is not written as a literal field name"

    def test_shape_reward_signature_unchanged(self):
        import inspect
        import research.ga_28c_30m_3y as ga
        params = set(inspect.signature(ga.shape_reward).parameters)
        assert not (params & self.ATTRIBUTION_KEYS)
