"""Tests for the reduced grammar.

The reduced grammar exists to make a null result mean something: under the full
29-token grammar a block-bootstrap null search reaches a *higher* median lockbox
Sharpe than the real data, so a positive result from that grammar is not
evidence. These tests pin the three properties the reduction depends on:

  1. the reduced grammar is a strict subset of the full one, with no new
     reachability, so it can only remove hypotheses;
  2. sampling, mutation and crossover never emit an invalid program under
     either grammar;
  3. reduced token indices are NOT the same function as full token indices, so
     a reduced formula must be evaluated by the reduced evaluator. Silently
     reusing the full evaluator is the failure mode this file exists to catch.
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import research.ga_28c_30m_3y as G
from research.formula_grammar import is_valid, ALL_TOKENS, OP_ARITY as FULL_ARITY
from research.grammar_reduced_28c import (
    REDUCED_ARITY, REDUCED_OPS, REDUCED_TOKENS, REDUCED_OP_COUNT,
    decode, evaluate_formula_reduced, validate_reduced,
)


@pytest.fixture(autouse=True)
def _restore_grammar():
    """Every test may switch the module-level grammar; put it back."""
    saved = G.GRAMMAR
    yield
    G.GRAMMAR = saved


# ---------------------------------------------------------------------------
# Shape of the reduced grammar
# ---------------------------------------------------------------------------

def test_reduced_operators_are_a_subset_of_the_full_operators():
    """No operator family is invented; the reduction only removes.

    The subset is over operator *names*. Token indices are re-packed, so index
    18 is SIGN (arity 1) in the full grammar and GATE (arity 3) in the reduced
    one; comparing indices would compare unrelated functions.
    """
    from model_core.ops import OPS_CONFIG
    full_by_name = {cfg[0]: int(cfg[2]) for cfg in OPS_CONFIG}
    for token, (name, arity) in REDUCED_OPS.items():
        assert name in full_by_name, f"reduced operator {name} is not in the full set"
        assert full_by_name[name] == arity, (
            f"{name} arity {arity} does not match the full grammar's "
            f"{full_by_name[name]}; the two evaluators would disagree")
        assert 0 <= token < REDUCED_TOKENS


def test_reduced_token_count_is_consistent():
    # Tokens are 0..FEATURE_COUNT-1 for factors, then one index per operator.
    # If the upper bound were computed as a count and used as an inclusive max,
    # the last operator (CORR, the highest token) would fall outside the grammar
    # and be silently unreachable. Assert the boundary directly.
    assert REDUCED_TOKENS == 12 + len(REDUCED_OPS)
    assert max(REDUCED_OPS) == REDUCED_TOKENS - 1
    assert REDUCED_TOKENS < ALL_TOKENS
    assert REDUCED_OP_COUNT == len(REDUCED_OPS) == 11


def test_reduced_operator_names_are_the_kept_families():
    names = {name for name, _ in REDUCED_OPS.values()}
    assert names == {"ADD", "SUB", "MUL", "DIV", "NEG", "SIGN", "GATE",
                     "DECAY", "DELAY1", "DELTA", "CORR"}
    dropped = {"ABS", "JUMP", "MAX3", "ZSCORE", "RANK", "TS_RANK"}
    assert not (names & dropped), "a dropped operator came back"


def test_reduced_arities_match_the_declared_ops():
    assert REDUCED_ARITY == {k: a for k, (_, a) in REDUCED_OPS.items()}


# ---------------------------------------------------------------------------
# Sampling operators
# ---------------------------------------------------------------------------

def test_random_formula_full_grammar_is_valid():
    import random
    G.GRAMMAR = "full"
    rng = random.Random(11)
    for _ in range(400):
        assert is_valid(G.random_formula(rng))


def test_random_formula_reduced_is_valid_and_in_grammar():
    import random
    G.GRAMMAR = "reduced"
    rng = random.Random(11)
    for _ in range(400):
        f = G.random_formula(rng)
        assert validate_reduced(f)
        assert _is_valid_r(f)
        for t in f:
            assert 0 <= t < REDUCED_TOKENS


def test_mutation_and_crossover_stay_valid_under_both_grammars():
    import random
    for grammar in ("full", "reduced"):
        G.GRAMMAR = grammar
        rng = random.Random(5)
        check = is_valid if grammar == "full" else validate_reduced
        for _ in range(200):
            a = G.random_formula(rng)
            b = G.random_formula(rng)
            assert check(G.mutate(a, rng))
            assert check(G.crossover(a, b, rng))


def test_seed_formulas_are_valid_under_the_active_grammar():
    G.GRAMMAR = "full"
    for f in G.seed_formulas():
        assert is_valid(f)
    G.GRAMMAR = "reduced"
    for f in G.seed_formulas():
        assert validate_reduced(f)


def _is_valid_r(formula):
    return G._is_valid_r(formula)


# ---------------------------------------------------------------------------
# The evaluator must follow the grammar
# ---------------------------------------------------------------------------

def test_reduced_tokens_are_not_interchangeable_with_full_tokens():
    """The whole point of a separate evaluator.

    A reduced token index means a different operator than the full index of
    the same number, so evaluating a reduced program with the full evaluator
    returns a different series rather than an error. If this test ever passes
    trivially the two grammars have converged and the duplicate evaluator can
    be deleted.
    """
    from research.causal_12f import evaluate_formula
    assert set(REDUCED_ARITY) != set(FULL_ARITY)
    # Build the full names for the shared indices.
    from model_core.ops import OPS_CONFIG
    full_names = {12 + i: cfg[0] for i, cfg in enumerate(OPS_CONFIG)}
    differing = [t for t in set(REDUCED_ARITY) & set(FULL_ARITY)
                 if REDUCED_OPS[t][0] != full_names[t]]
    assert differing, "no reduced/full index collision, so the test is vacuous"

    # A concrete program that is legal in both grammars and uses a
    # colliding index, evaluated both ways, must disagree.
    t = differing[0]
    feats = np.random.RandomState(3).randn(12, 128)
    prog = [5] + [t] * 11
    a = evaluate_formula_reduced(prog, feats)
    b = evaluate_formula(prog, feats)
    assert a is not None and b is not None
    assert not np.allclose(a, b), (
        f"token {t} evaluated identically under both grammars; "
        "the reduced evaluator is not being exercised")


def test_reduced_evaluator_rejects_tokens_outside_its_grammar():
    feats = np.random.RandomState(1).randn(12, 64)
    # 23..28 exist only in the full grammar (ZSCORE..CORR are dropped).
    for bad_token in (24, 25, 26):
        prog = [5] + [bad_token] * 11
        assert evaluate_formula_reduced(prog, feats) is None


def test_reduced_evaluator_rejects_illegal_stack():
    feats = np.random.RandomState(1).randn(12, 64)
    # Starts with a 2-arity operator on an empty stack.
    assert evaluate_formula_reduced([12, 5, 5, 16, 16, 16, 16, 16, 16, 16, 16, 16], feats) is None


def test_reduced_evaluator_agrees_with_full_on_the_shared_operators():
    """Identical operator names must compute identical numbers."""
    from research.causal_12f import evaluate_formula
    import research.grammar_reduced_28c as R
    from model_core.ops import OPS_CONFIG
    full_index = {cfg[0]: 12 + i for i, cfg in enumerate(OPS_CONFIG)}
    feats = np.random.RandomState(7).randn(12, 200)
    PAD_R = 17  # reduced token for SIGN
    PAD_F = full_index["SIGN"]
    for reduced_tok, (name, _arity) in REDUCED_OPS.items():
        full_tok = full_index[name]
        a_arity = REDUCED_ARITY[reduced_tok]
        # Pad with unary operators so the program is exactly 12 tokens and ends
        # at stack depth 1; an evaluator that returns None is a bug in the
        # program, not a disagreement, so assert legality first.
        if a_arity == 3:
            prog = [1, 5, 0, reduced_tok]
            pad = 8
        elif a_arity == 2:
            prog = [1, 5, reduced_tok]
            pad = 9
        else:
            prog = [5, reduced_tok]
            pad = 10
        prog = prog + [PAD_R] * pad
        assert len(prog) == 12
        assert validate_reduced(prog), f"test program for {name} is not legal"
        a = evaluate_formula_reduced(prog, feats)
        assert a is not None, f"{name} evaluated to None under the reduced grammar"
        # Map every reduced operator through its NAME, not its index: reduced 17
        # is SIGN while full 17 is ABS, and mixing the two silently changes the
        # function being compared.
        red_to_full = {rt: full_index[nm] for rt, (nm, _a) in REDUCED_OPS.items()}
        mapped = [red_to_full[t] if t >= 12 else t for t in prog]
        b = evaluate_formula(mapped, feats)
        assert b is not None, f"{name} evaluated to None under the full grammar"
        assert np.allclose(a, b), (
            f"{name}: reduced and full evaluators disagree "
            f"(max diff {float(np.max(np.abs(a - b)))})")


def test_formula_signal_follows_the_active_grammar():
    G.GRAMMAR = "full"
    assert G._spec()["max_token"] == ALL_TOKENS
    G.GRAMMAR = "reduced"
    assert G._spec()["max_token"] == REDUCED_TOKENS


# ---------------------------------------------------------------------------
# The reduction is real
# ---------------------------------------------------------------------------

def test_reduced_grammar_is_strictly_smaller():
    """Fewer hypotheses, measured not asserted."""
    @lru_cache(maxsize=None)
    def count(depth, slots):
        if slots == 0:
            return 1 if depth == 1 else 0
        if depth > 12 or depth < 1 or depth > slots + 1:
            return 0
        total = 0
        for t in range(12):
            if G._can_finish_r(depth + 1, slots - 1, 12, REDUCED_ARITY):
                total += count(depth + 1, slots - 1)
        for t, a in REDUCED_ARITY.items():
            if depth >= a and G._can_finish_r(depth - a + 1, slots - 1, 12, REDUCED_ARITY):
                total += count(depth - a + 1, slots - 1)
        return total

    # The first token of an RPN program is always a factor, so the count starts
    # from depth 1. Seeding at depth 0 would hit the `depth < 1` guard, which
    # describes post-token states rather than the start of a program.
    reduced_total = count(1, 11)
    assert reduced_total > 0
    # Uniform-draw estimate for the full grammar, same estimator style.
    import random
    from research.formula_grammar import is_valid as full_valid
    rng = random.Random(0)
    trials = 20000
    ok = sum(1 for _ in range(trials)
             if full_valid(tuple(rng.randrange(ALL_TOKENS) for _ in range(12))))
    full_total = ALL_TOKENS ** 12 * ok / trials
    assert reduced_total < full_total / 100, (
        f"reduction only {full_total / reduced_total:.1f}x, expected >100x")


def test_can_finish_agrees_with_brute_force():
    """The reachability prune must not cut off a completable prefix."""

    def brute(depth, slots):
        if slots == 0:
            return depth == 1
        if depth < 1 or depth > 12 or depth > slots + 1:
            return False
        for t in range(REDUCED_TOKENS):
            if t < 12:
                if brute(depth + 1, slots - 1):
                    return True
            elif t in REDUCED_ARITY and depth >= REDUCED_ARITY[t]:
                if brute(depth - REDUCED_ARITY[t] + 1, slots - 1):
                    return True
        return False

    for depth in range(0, 8):
        for slots in range(0, 11):
            assert bool(G._can_finish_r(depth, slots, 12, REDUCED_ARITY)) == brute(depth, slots), (
                f"reachability disagrees at depth={depth} slots={slots}")


def test_decode_names_every_token():
    G.GRAMMAR = "reduced"
    import random
    rng = random.Random(2)
    f = G.random_formula(rng)
    text = decode(f)
    assert text and "UNK" not in text
    assert len(text.split()) == 12
