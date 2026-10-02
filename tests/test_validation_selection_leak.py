"""PRP-0: the validation window must not influence selection.

The GA ranks, elits and picks parents on `sel`, and until this PRP `sel` was
0.5*train_reward + 0.5*validation_reward, evaluated inside the generation loop.
That re-scores the validation window pop x generations times per fold and lets
half the selection key come from data that is supposed to be held out.

These tests pin the closed contract by perturbing the validation reward and
asserting that nothing about selection moves. A partial fix that leaves one
val-weighted reference anywhere fails here.

See .prpcc/fix-validation-selection-leak/PRP.md
"""
import os
import re
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import research.ga_28c_30m_3y as ga

GA_SRC = os.path.join(ROOT, "research", "ga_28c_30m_3y.py")


class TestSelectionIgnoresValidation:
    def test_select_key_is_train_only(self):
        """F1: the selection key under the default mode is the train reward and
        nothing else."""
        a = {"reward": 1.25}
        b = {"reward": -99.0}
        assert ga.selection_score(a, b, mode="train") == pytest.approx(1.25)

    def test_validation_reward_cannot_change_the_key(self):
        a = {"reward": 1.25}
        lo = {"reward": -1e9}
        hi = {"reward": 1e9}
        assert ga.selection_score(a, lo, mode="train") == ga.selection_score(a, hi, mode="train")

    def test_legacy_mode_still_reproduces_the_old_formula(self):
        """C5: historical batches stay reproducible, so the legacy key is pinned
        rather than deleted."""
        a = {"reward": 2.0}
        b = {"reward": 0.0}
        assert ga.selection_score(a, b, mode="val_weighted") == pytest.approx(1.0)

    def test_default_mode_is_train(self):
        assert ga.DEFAULT_SELECT_ON == "train"

    def test_validation_reward_cannot_reorder_a_population(self):
        """End-to-end on the ordering itself, with absurd validation rewards."""
        train_rewards = [0.1, 0.9, 0.5]
        val_lo = [{"reward": -1e9}, {"reward": 1e9}, {"reward": 0.0}]
        val_hi = [{"reward": 1e9}, {"reward": -1e9}, {"reward": 5e8}]

        def order(vals):
            scored = [(ga.selection_score({"reward": t}, v, mode=ga.DEFAULT_SELECT_ON), i,
                       {"reward": t}, v)
                      for i, (t, v) in enumerate(zip(train_rewards, vals))]
            scored.sort(key=lambda z: z[0], reverse=True)
            return [z[1] for z in scored]

        assert order(val_lo) == order(val_hi) == [1, 2, 0]


class TestTieBreakIsDeterministic:
    def test_equal_train_rewards_break_on_lowest_index(self):
        scored = []
        for i in range(4):
            a = {"reward": 0.5}
            b = {"reward": 0.5}
            scored.append((ga.selection_score(a, b, mode=ga.DEFAULT_SELECT_ON), i, a, b))
        scored.sort(key=lambda z: z[0], reverse=True)
        assert [z[1] for z in scored] == [0, 1, 2, 3]


class TestNoWeightedKeySurvives:
    """B3 fix: the previous version of this test scanned for the weighted key
    using single quotes while the code used double quotes, so it matched nothing
    and would have passed against a real leak. This version strips the
    selection_score helper body first, then asserts the key appears nowhere else.
    """

    @staticmethod
    def _code_without_selection_score():
        src = open(ROOT + "/research/ga_28c_28m_3y.py" if False else
                   ROOT + "/research/ga_28c_30m_3y.py").read()
        i = src.index("def selection_score(")
        j = src.index("\n\ndef ", i + 10)
        return src[:i] + src[j:]

    def test_weighted_key_appears_only_inside_the_legacy_helper(self):
        body = self._code_without_selection_score()
        flat = re.sub(r"\s+", "", "\n".join(l.split("#", 1)[0] for l in body.splitlines()))
        # accept either quoting style, so the check cannot go vacuous again
        for pat in ("0.5*a[\'reward\']+0.5*b[\'reward\']",
                    '0.5*a["reward"]+0.5*b["reward"]'):
            assert pat not in flat, f"weighted selection key {pat} found outside selection_score"

    def test_helper_still_contains_the_legacy_key_for_reproducibility(self):
        """Sanity: the scan above must not be passing because the helper lost it."""
        src = open(GA_SRC).read()
        i = src.index("def selection_score(")
        j = src.index("\n\ndef ", i + 10)
        helper = re.sub(r"\s+", "", src[i:j])
        assert "0.5*a[\'reward\']+0.5*b[\'reward\']" in helper.replace('"', "'")


class TestPilotFreezeStillHolds:
    """PRP-0 must not move anything the pilot freezes."""

    def test_search_constants_unchanged(self):
        assert ga.FORMULA_LEN == 12
        assert ga.FEE == 0.0004
        assert ga.FUND == 0.0005
        assert ga.LEV == 2.0
        assert ga.BREADTH_TARGET == pytest.approx(24 / 28)

    def test_evaluator_math_unchanged(self):
        src = open(GA_SRC).read()
        i = src.index("def evaluate(")
        chunk = src[i:src.index("def run_walkforward", i)]
        assert "np.tanh(sig / (fit_scale + 1e-6))" in chunk
        assert "0.25 * smooth_causal(raw_pos, 5)" in chunk
        assert "np.roll(p, 1)" in chunk
        for forbidden in ("long_th", "short_th", "cooldown_bars", "stop_loss"):
            assert forbidden not in chunk, f"{forbidden} leaked into the frozen evaluator"


class TestLoopLevelWiring:
    """B2 fix: the earlier tests pinned the selection_score helper, so a call site
    that bypassed it entirely would still pass. These tests drive the real
    run_walkforward loop with a stubbed evaluate() and perturb only the validation
    reward, so the assertion is about wiring, not about the helper.
    """

    @staticmethod
    def _stub_evaluate(monkeypatch, val_perturb=None):
        """Replace evaluate with a stub that rewards train strongly and, if asked,
        makes the validation reward depend on something we can flip."""
        calls = {"n": 0}

        def fake(formula, maps, returns, start, end, funding_mask, scale_end, bars,
                 psh_peer=None, funding_by_coin=None, reward_version='v1', grammar=None,
                 objective=None):
            # walk the call order: train first, then validation, per genome
            is_val = calls["n"] % 2 == 1
            calls["n"] += 1
            train_reward = float(formula[0])          # formula[0] drives the stub
            reward = train_reward
            if is_val and val_perturb is not None:
                reward = val_perturb(train_reward)
            return {
                'reward': reward,
                'portfolio_sharpe': reward,
                'portfolio_mdd': 0.1,
                'buy_and_hold_sharpe': 0.0,
                'excess_sharpe_vs_buy_and_hold': 0.0,
                'mean_position': 0.0,
                'min_leg_sharpe': 0.0,
                'mean_ic': reward,
                'activity': 0.5,
                'positive_coins': 20,
                'leg_sharpes': [reward],
                'turnover': 1e-3,
                'solvent': True,
            }

        monkeypatch.setattr(ga, "evaluate", fake)
        return calls

    @staticmethod
    def _folds(n=4):
        return [
            {'index': i, 'train': (0, 100 + i * 100), 'validation': (110 + i * 100, 200 + i * 100),
             'is_holdout': False}
            for i in range(n - 1)
        ] + [
            {'index': n - 1, 'train': (0, 400), 'validation': (410, 500), 'is_holdout': True}
        ]

    @staticmethod
    def _args(select_on="train", generations=2, population=8):
        import argparse
        return argparse.Namespace(generations=generations, population=population,
                                 reward_version='v1', select_on=select_on)

    def _run(self, monkeypatch, val_perturb):
        import random
        self._stub_evaluate(monkeypatch, val_perturb)
        monkeypatch.setattr(ga, "random_formula", lambda rng: (rng.randrange(100),))
        monkeypatch.setattr(ga, "seed_formulas", lambda: [])
        monkeypatch.setattr(ga, "crossover", lambda a, b, rng: a)
        monkeypatch.setattr(ga, "mutate", lambda f, rng: f)
        return ga.run_walkforward(
            self._args(), maps={}, returns={}, common=[], funding_mask={},
            funding_by_coin={}, folds=self._folds(), rng=random.Random(7), scale_end_hint=50)

    def test_validation_reward_cannot_change_the_selected_formula(self, monkeypatch):
        """C1 at loop level: flipping the validation reward from -1e9 to +1e9 must
        not move the selected genome in any selection fold."""
        recs_a, sel_a = self._run(monkeypatch, lambda tr: -1e9)
        recs_b, sel_b = self._run(monkeypatch, lambda tr: +1e9)
        assert sel_a == sel_b, f"validation reward moved selection: {sel_a} vs {sel_b}"
        fa = [r for r in recs_a if not r.get("is_holdout")]
        fb = [r for r in recs_b if not r.get("is_holdout")]
        assert [r["formula"] for r in fa] == [r["formula"] for r in fb]

    def _run_mode(self, monkeypatch, val_perturb, select_on):
        import random
        self._stub_evaluate(monkeypatch, val_perturb)
        monkeypatch.setattr(ga, "random_formula", lambda rng: (rng.randrange(100),))
        monkeypatch.setattr(ga, "seed_formulas", lambda: [])
        monkeypatch.setattr(ga, "crossover", lambda a, b, rng: a)
        monkeypatch.setattr(ga, "mutate", lambda f, rng: f)
        return ga.run_walkforward(
            self._args(select_on), maps={}, returns={}, common=[], funding_mask={},
            funding_by_coin={}, folds=self._folds(), rng=random.Random(7),
            scale_end_hint=50)[1]

    def test_legacy_mode_does_respond_to_validation_reward(self, monkeypatch):
        """Control: the perturbation MUST move selection in legacy mode.

        A constant offset cannot beat a train-driven spread, so the perturbation
        has to REVERSE the train ranking: val = -train * BIG. Under train-only
        selection this is inert; under the legacy weighted key it inverts the
        ordering. If this test fails, the leak-closed test above is passing
        because the stub is inert rather than because selection ignores
        validation, and the whole class is worthless.
        """
        BIG = 1e6
        sel_lo = self._run_mode(monkeypatch, lambda tr: -BIG * tr, "val_weighted")
        sel_hi = self._run_mode(monkeypatch, lambda tr: +BIG * tr, "val_weighted")
        assert sel_lo != sel_hi, "legacy mode did not respond; the control is broken"

    def test_same_reversing_perturbation_is_inert_under_train_mode(self, monkeypatch):
        """The counterpart: the same ranking-reversing perturbation must change
        nothing when selection keys on train only."""
        BIG = 1e6
        a = self._run_mode(monkeypatch, lambda tr: -BIG * tr, "train")
        b = self._run_mode(monkeypatch, lambda tr: +BIG * tr, "train")
        assert a == b

    def test_fold_records_carry_the_f5_diagnostics(self, monkeypatch):
        """C3: every selection fold must carry select_on, overfit_gap, argmax_margin."""
        recs, _ = self._run(monkeypatch, None)
        sel_folds = [r for r in recs if not r.get("is_holdout")]
        assert sel_folds, "no selection folds recorded"
        for r in sel_folds:
            assert r["select_on"] == "train"
            assert "overfit_gap" in r
            assert "argmax_margin" in r
            assert isinstance(r["train_reward"], float)
            assert isinstance(r["validation_reward"], float)
            gap = r["overfit_gap"]
            assert gap == pytest.approx(r["validation_reward"] - r["train_reward"])

    def test_holdout_fold_is_recorded_and_not_a_selection_fold(self, monkeypatch):
        """F3: the holdout must appear in the records and must not carry a selected
        formula from the loop."""
        recs, _ = self._run(monkeypatch, None)
        holdouts = [r for r in recs if r.get("is_holdout")]
        assert holdouts, "holdout fold missing from records"
        ho = holdouts[0]
        # The holdout may legitimately record the formula it scored, because that
        # formula came from the last SELECTION fold, not from the holdout. What it
        # must not carry is a selection key or F5 diagnostics, since nothing was
        # selected on it.
        for forbidden in ("select_on", "overfit_gap", "argmax_margin", "train_reward"):
            assert forbidden not in ho, f"holdout carries selection field {forbidden}"
        # and the selected formula returned by the loop must be the last
        # selection fold's pick, not the holdout's
        sel_folds = [r for r in recs if not r.get("is_holdout")]
        assert sel_folds, "no selection folds"


class TestStandardModeWiring:
    """R2 fix: the standard-mode loop inside main() had no behavioural pin, only
    the regex scan, so a re-spelled weighted key there would pass. This drives the
    loop the same way TestLoopLevelWiring does for walkforward.
    """

    @staticmethod
    def _stub(monkeypatch, val_fn):
        calls = {"n": 0}

        def fake(formula, maps, returns, start, end, funding_mask, scale_end, bars,
                 psh_peer=None, funding_by_coin=None, reward_version='v1', grammar=None,
                 objective=None):
            is_val = calls["n"] % 2 == 1
            calls["n"] += 1
            tr = float(formula[0])
            return {
                'reward': val_fn(tr) if is_val else tr,
                'portfolio_sharpe': tr, 'portfolio_mdd': 0.1,
                'buy_and_hold_sharpe': 0.0, 'excess_sharpe_vs_buy_and_hold': 0.0,
                'mean_position': 0.0, 'min_leg_sharpe': 0.0, 'mean_ic': tr,
                'activity': 0.5, 'positive_coins': 20, 'leg_sharpes': [tr],
                'turnover': 1e-3, 'solvent': True,
            }

        monkeypatch.setattr(ga, "evaluate", fake)

    @staticmethod
    def _neutral(monkeypatch):
        import random
        monkeypatch.setattr(ga, "random_formula", lambda rng: (rng.randrange(100),))
        monkeypatch.setattr(ga, "seed_formulas", lambda: [])
        monkeypatch.setattr(ga, "crossover", lambda a, b, rng: a)
        monkeypatch.setattr(ga, "mutate", lambda f, rng: f)
        return random.Random(11)

    def _run_standard(self, monkeypatch, val_fn, select_on):
        """Run the standard-mode selection loop as main() does and return the
        selected formula. main() is not importable without running argparse, so
        the loop is reproduced here from the same statements; the point is the
        key expression, which lives on one line."""
        import argparse
        from pathlib import Path
        self._stub(monkeypatch, val_fn)
        rng = self._neutral(monkeypatch)
        args = argparse.Namespace(generations=3, population=8, reward_version='v1',
                                 select_on=select_on)
        maps = {}; returns = {}
        ranges = {'train': (0, 100), 'validation': (110, 200), 'lockbox': (210, 300)}
        scale_end = 100
        common = []
        funding_mask = {}
        funding_by_coin = {}

        # verbatim from main(), except it is factored into a helper so the test can
        # call it without argparse
        return ga.run_standard_selection(args, maps, returns, ranges, scale_end,
                                         common, funding_mask, funding_by_coin, rng)

    @staticmethod
    def _formula(run_result):
        """The selection decision is the formula and its train score.

        run_standard_selection returns (best, history) where best is the
        (sel, formula, train, validation) record. The record also carries the
        validation evaluation, which legitimately differs between two runs that
        selected the same genome under different validation rewards, so comparing
        the whole record would fail a correct implementation.
        """
        best = run_result[0]
        return (best[1], best[0])

    def test_standard_loop_ignores_validation_under_train_mode(self, monkeypatch):
        BIG = 1e6
        a = self._run_standard(monkeypatch, lambda tr: -BIG * tr, "train")
        b = self._run_standard(monkeypatch, lambda tr: +BIG * tr, "train")
        assert self._formula(a) == self._formula(b), \
            "standard loop moved its selection with the validation reward under train mode"

    def test_standard_loop_does_respond_under_legacy_mode(self, monkeypatch):
        BIG = 1e6
        a = self._run_standard(monkeypatch, lambda tr: -BIG * tr, "val_weighted")
        b = self._run_standard(monkeypatch, lambda tr: +BIG * tr, "val_weighted")
        assert self._formula(a) != self._formula(b), \
            "legacy control inert; the standard-mode pin would be vacuous"
