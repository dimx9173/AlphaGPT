"""Variable-length formulas: the search must be able to leave 12 tokens."""
import random

import pytest

import research.ga_28c_30m_3y as ga


@pytest.fixture(params=("full", "reduced"))
def grammar(request):
    old = ga.GRAMMAR
    ga.GRAMMAR = request.param
    yield request.param
    ga.GRAMMAR = old


def _valid(g):
    from research.formula_grammar import is_valid
    return is_valid if g == "full" else ga.validate_reduced


def test_length_bounds_cover_the_historical_length():
    assert ga.FORMULA_MIN_LEN <= 12 <= ga.FORMULA_MAX_LEN


def test_random_formula_has_variable_length(grammar):
    rng = random.Random(123)
    lengths = [len(ga.random_formula(rng)) for _ in range(200)]
    # Should span the full min-max range, not be stuck at 12
    assert min(lengths) <= ga.FORMULA_MIN_LEN + 1
    assert max(lengths) >= ga.FORMULA_MAX_LEN - 1
    # Not all 12
    assert any(l != 12 for l in lengths)


def test_random_formula_is_valid(grammar):
    rng = random.Random(456)
    check = _valid(grammar)
    for _ in range(200):
        f = ga.random_formula(rng)
        assert check(f), f"invalid: {f}"

def test_mutation_can_change_length(grammar):
    rng = random.Random(789)
    check = _valid(grammar)
    changes = 0
    for _ in range(200):
        parent = ga.random_formula(rng)
        child = ga.mutate(parent, rng)
        assert check(child)
        if len(child) != len(parent):
            changes += 1
    # With 20% length-change branch, expect ~40 length changes
    assert changes >= 20, f"only {changes} length changes out of 200"


def test_crossover_produces_valid_offspring(grammar):
    rng = random.Random(999)
    check = _valid(grammar)
    for _ in range(200):
        a = ga.random_formula(rng)
        b = ga.random_formula(rng)
        child = ga.crossover(a, b, rng)
        assert check(child)
        # Length should be reasonable (between min and max)
        assert ga.FORMULA_MIN_LEN <= len(child) <= ga.FORMULA_MAX_LEN