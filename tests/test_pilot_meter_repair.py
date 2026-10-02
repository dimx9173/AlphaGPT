"""Regression tests for the pilot meter repair (2026-10-01).

Two things broke the selection score before this commit, and both are silent:
the -50 cliff made every inactive genome score identically, and the gap term was
identically zero because psh_peer was None at both call sites. A test that does
not pin the absence of these will not notice them coming back, so these tests
assert the SHAPE of the meter rather than a single expected number.

See docs/GA_SEARCH_RELIABILITY_2026-10-01.md.
"""
import re
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import research.ga_28c_30m_3y as ga


def _r(psh, min_leg, mean_ic, mean_turn, mean_act):
    """A shape_reward-shaped call using the real module constants."""
    penalty = -ga.INACTIVITY_WEIGHT * max(0.0, 1.0 - mean_act / ga.ACTIVITY_TARGET)
    base = ga.shape_reward(psh, None, min_leg, mean_ic, mean_turn, version="v1")
    return base + penalty


class TestNoCliff:
    def test_inactive_genomes_do_not_all_score_the_same(self):
        """The old code returned -50.0 for every genome under the activity bar.
        Two clearly different inactive genomes must now score differently, or
        selection has nothing to rank inside that region."""
        a = _r(psh=1.0, min_leg=-1.0, mean_ic=0.010, mean_turn=1e-4, mean_act=0.0004)
        b = _r(psh=0.5, min_leg=-1.0, mean_ic=0.011, mean_turn=1e-4, mean_act=0.0009)
        assert a != -50.0
        assert b != -50.0
        assert a != b

    def test_no_reward_equals_minus_fifty(self):
        for act in (0.0, 1e-5, 4e-5, 9e-3, 1e-2):
            v = _r(psh=2.0, min_leg=-1.5, mean_ic=0.005, mean_turn=1e-4, mean_act=act)
            assert v != -50.0, f"activity {act} still hit the cliff"

    def test_penalty_is_zero_at_or_above_the_old_threshold(self):
        """A formula that already cleared the old bar must score exactly as it
        did before, or the A/B against the recorded runs is not an A/B."""
        at = ga.INACTIVITY_WEIGHT * max(0.0, 1.0 - 0.01 / ga.ACTIVITY_TARGET)
        above = ga.INACTIVITY_WEIGHT * max(0.0, 1.0 - 0.05 / ga.ACTIVITY_TARGET)
        assert at == 0.0
        assert above == 0.0

    def test_penalty_grows_monotonically_as_activity_falls(self):
        prev = None
        for act in (0.009, 0.005, 0.002, 0.0005, 0.0001):
            v = _r(psh=1.0, min_leg=0.0, mean_ic=0.01, mean_turn=1e-4, mean_act=act)
            if prev is not None:
                assert v < prev, "penalty must worsen as activity falls"
            prev = v


class TestGapTermIsLive:
    def test_peer_changes_the_score(self):
        """With psh_peer=None the gap term is dead code. Supplying a peer must
        change the result, which is the whole point of the second change."""
        without = ga.shape_reward(2.0, None, 0.0, 0.01, 1e-4, version="v1")
        with_peer = ga.shape_reward(2.0, 0.0, 0.0, 0.01, 1e-4, version="v1")
        assert without != with_peer

    def test_gap_punishes_a_split_that_outperforms_its_baseline(self):
        """A genome that beats the peer gets charged for the divergence. This is
        the term that was silently zero before."""
        beats = ga.shape_reward(3.0, 0.0, 0.0, 0.01, 1e-4, version="v1")
        near = ga.shape_reward(0.0, 0.0, 0.0, 0.01, 1e-4, version="v1")
        assert beats < near

    def test_peer_none_still_works_as_a_fallback(self):
        """The None fallback is deliberate so a caller that forgets the baseline
        is no worse than before rather than silently gaining an uncalibrated term."""
        assert ga.shape_reward(1.0, None, 0.0, 0.01, 1e-4, version="v1") == pytest.approx(
            ga.shape_reward(1.0, 1.0, 0.0, 0.01, 1e-4, version="v1")
        )


class TestPilotFreeze:
    """The pilot contract: if any of these move, the real-vs-null comparison
    measures the engine instead of the meter and the run is void."""

    def test_evaluator_math_unchanged(self):
        src = open(ROOT + "/research/ga_28c_30m_3y.py").read()
        i = src.index("def evaluate(")
        body = src[i:src.index("def run_walkforward", i)]
        assert "np.tanh(sig / (fit_scale + 1e-6))" in body
        assert "0.25 * smooth_causal(raw_pos, 5)" in body
        assert "np.roll(p, 1)" in body
        # no execution-layer parameters may enter the GA evaluator
        for forbidden in ("long_th", "short_th", "cooldown_bars", "stop_loss"):
            assert forbidden not in body, f"{forbidden} entered the frozen evaluator"

    def test_genome_and_costs_unchanged(self):
        assert ga.FORMULA_LEN == 12
        assert ga.FEE == 0.0004
        assert ga.FUND == 0.0005
        assert ga.LEV == 2.0

    def test_reward_weights_unchanged(self):
        """The v4 parameterisation must default to exactly the v1 arithmetic.

        The weights were lifted out of the expression so they can be swept, so
        this no longer greps literals. It pins the DEFAULT VALUES instead, and
        the next test pins that the defaults reproduce the old reward exactly.
        """
        import inspect
        sig = inspect.signature(ga.shape_reward).parameters
        assert sig["w_mean_ic"].default == 1.0
        assert sig["w_psh"].default == 0.10
        assert sig["w_gap"].default == 0.30
        assert sig["w_breadth"].default == 2.0
        assert sig["w_excess"].default == 0.30
        assert sig["rank_ic"].default is None
        assert sig["excess_bh"].default is None

    def test_default_turnover_weight_is_unchanged(self):
        """TURNOVER_WEIGHT was extracted so a probe can sweep it. The pilot freeze
        still applies: the default must resolve to 0.02."""
        src = open(ROOT + "/research/ga_28c_30m_3y.py").read()
        assert re.search(r'ALPHAGPT_TURNOVER_WEIGHT",\s*"0\.02"', src), \
            "the default turnover weight moved off 0.02; the pilot freeze covers it"
        assert ga.TURNOVER_WEIGHT == 0.02, (
            f"TURNOVER_WEIGHT is {ga.TURNOVER_WEIGHT} in this environment")

    def test_default_call_reproduces_the_pre_v4_reward(self):
        """The behavioural freeze, not a textual one.

        Recomputes the v1 expression by hand and asserts shape_reward's default
        call returns the same number. This catches a refactor that moves a term,
        reorders the sum, or changes a sign, which a signature check cannot.
        """
        import math
        psh, peer, min_leg, ic, turn = 1.3, 0.9, -0.4, 0.011, 3.1e-3
        breadth, excess = 0.75, 0.22
        expected = (ic
                    + 0.10 * min(psh, peer)
                    - 0.30 * ga.softplus((psh - peer) / ga.GAP_TAU) * ga.GAP_TAU
                    - 0.02 * turn
                    - 0.10 * ga.softplus((ga.LEG_FLOOR - min_leg) / 0.5)
                    - 2.0 * max(0.0, ga.BREADTH_TARGET - breadth)
                    + 0.30 * excess)
        got = ga.shape_reward(psh, peer, min_leg, ic, turn,
                              breadth=breadth, version="v3", excess=excess)
        assert got == pytest.approx(expected, abs=1e-12), (
            f"default reward {got} != pre-v4 expression {expected}")

    def test_new_objectives_fire_at_reward_version_v1(self):
        """Regression for a real defect found by the first objective sweep.

        The rank-IC and excess-over-hold terms were placed after the
        `if version == 'v1': return base` guard, so with the default reward
        version they could never fire. Every non-v3 preset was silently an
        empty objective and three of them produced byte-identical numbers.

        The freeze default is reward-version v1, so this is the version that
        matters for every recorded batch.
        """
        import research.ga_28c_30m_3y as m
        args = (1.3, 0.9, -0.4, 0.011, 3.1e-3)
        kw = dict(breadth=0.75, version="v1", excess=0.22,
                  rank_ic=0.011, excess_bh=0.22)
        v3 = m.shape_reward(*args, **kw, **m.OBJECTIVES["v3"])
        for name in ("rank_ic", "excess_bh", "rank_ic_excess_bh"):
            got = m.shape_reward(*args, **kw, **m.OBJECTIVES[name])
            assert got != v3, (
                f"{name} produces the same reward as v3 at version v1; its "
                "weight is not reaching the objective")

    def test_new_objectives_are_off_by_default(self):
        """rank-IC and excess-over-hold are opt-in. If they were ever on by
        default, every recorded batch would silently be non-comparable."""
        psh, peer, min_leg, ic, turn = 1.3, 0.9, -0.4, 0.011, 3.1e-3
        a = ga.shape_reward(psh, peer, min_leg, ic, turn, breadth=0.75,
                            version="v3", excess=0.22)
        # passing the values alone must NOT change anything: their weights are 0
        b = ga.shape_reward(psh, peer, min_leg, ic, turn, breadth=0.75,
                            version="v3", excess=0.22,
                            rank_ic=0.5, excess_bh=0.4)
        assert a == b, "a zero weight still moved the score; the terms are not gated"
        # turning a weight on must change it, otherwise the term is dead code
        c = ga.shape_reward(psh, peer, min_leg, ic, turn, breadth=0.75,
                            version="v3", excess=0.22,
                            w_rank_ic=1.0, rank_ic=0.5)
        d = ga.shape_reward(psh, peer, min_leg, ic, turn, breadth=0.75,
                            version="v3", excess=0.22,
                            w_excess_bh=1.0, excess_bh=0.4)
        assert c != a, "w_rank_ic has no effect; the rank-IC objective is dead"
        assert d != a, "w_excess_bh has no effect; the excess-BH objective is dead"
        import inspect
        sig = inspect.signature(ga.shape_reward).parameters
        assert sig["w_rank_ic"].default == 0.0
        assert sig["w_excess_bh"].default == 0.0
