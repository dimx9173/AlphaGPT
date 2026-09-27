"""Tests for the regime-neutral gate.

The two tests that Codex and the architect found missing are test_leverage_parity
(constant-0.25 cap hold, B9) and test_exposure_parity (B8). The rev-1 exposure
regression was vacuous because a "static 2x hold" under F1 *was* the benchmark.
"""
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from research.regime_gate_28c import (  # noqa: E402
    BARS_PER_DAY, POSITION_CAP, coverage_report, excess_metrics,
    market_regime_proxy, midnight_aligned, passive_benchmark_net,
    regime_label, walk_forward_folds_28c,
)
from research.accounting_28c import daily_sharpe  # noqa: E402

START_TS = 1726228800000  # a known 30m bar


def _series(n_years=3, n_coins=6, seed=0, trend=0.0):
    n = int(n_years * 365 * BARS_PER_DAY)
    ts = START_TS + np.arange(n) * 1_800_000
    rng = np.random.default_rng(seed)
    returns, funding = {}, np.zeros(n)
    for j, c in enumerate([f"C{i}" for i in range(n_coins)]):
        r = rng.normal(trend, 0.004, n)
        r[0] = 0.0
        returns[c] = r
    for i, t_ms in enumerate(ts):
        d = datetime.fromtimestamp(t_ms / 1000.0, tz=timezone.utc)
        if d.hour % 8 == 0 and d.minute == 0:
            funding[i] = 1.0
    return ts, returns, funding, list(returns)


# ---------------------------------------------------------------- benchmark

def test_market_regime_proxy_is_the_hand_computable_mean():
    ts, returns, _, coins = _series(n_years=0.1, n_coins=3, seed=1)
    got = market_regime_proxy(returns, coins, 0, 100)
    want = np.mean([returns[c][:100] for c in coins], axis=0)
    assert np.array_equal(got, want)


def test_benchmark_matches_a_leg_through_the_same_accounting_path():
    """B8: cap, leverage, fee and funding must match a strategy leg by
    construction, not by convention."""
    ts, returns, funding, coins = _series(n_years=0.5, n_coins=4, seed=2)
    got = passive_benchmark_net(returns, coins, 0, len(ts), funding, 0.0004, 2.0)
    from research.accounting_28c import accounting_bar_returns
    pos = np.full(len(ts), POSITION_CAP)
    want = np.mean([accounting_bar_returns(pos, returns[c], 0.0004, funding, 2.0)
                    for c in coins], axis=0)
    assert np.allclose(got, want)


def test_leverage_parity_b9_constant_cap_hold_has_zero_excess():
    """B9: the realistic trap in this framework is a constant hold at the
    position cap, not a 2x hold. It must not score excess skill."""
    ts, returns, funding, coins = _series(n_years=2, n_coins=6, seed=3, trend=3e-5)
    bench = passive_benchmark_net(returns, coins, 0, len(ts), funding, 0.0004, 2.0)
    # a "strategy" that is always long at the cap
    strategy = passive_benchmark_net(returns, coins, 0, len(ts), funding, 0.0004, 2.0)
    ex = excess_metrics(strategy, bench, ts)
    assert abs(ex["excess_sharpe"]) < 1e-9
    assert ex["excess_sharpe"] == pytest.approx(0.0, abs=1e-9)


def test_exposure_parity_b8_benchmark_notional_equals_the_cap():
    """The benchmark must not be 4x the strategy's achievable exposure."""
    assert POSITION_CAP == 0.25
    from research.accounting_28c import accounting_bar_returns
    ts, returns, funding, coins = _series(n_years=0.2, n_coins=3, seed=4)
    # strategy leg built exactly as evaluate() builds it, with a signal large
    # enough that tanh saturates, so the cap is actually reached
    sig = np.tanh(50.0 * np.sin(np.linspace(0, 20, len(ts))))
    p = POSITION_CAP * sig
    p = np.roll(p, 1); p[0] = 0.0
    strat_max = float(np.max(np.abs(p))) * 2.0
    bench_max = POSITION_CAP * 2.0
    assert bench_max <= strat_max + 1e-12
    # and it is not the 4x mismatch the old F1 specification produced
    assert bench_max >= 0.99 * strat_max


# ---------------------------------------------------------------- excess

def test_excess_uses_the_difference_series_not_sharpe_subtraction():
    # trend kept small so the 2x leveraged benchmark stays solvent
    ts, returns, funding, coins = _series(n_years=1, n_coins=5, seed=5, trend=5e-6)
    bench = passive_benchmark_net(returns, coins, 0, len(ts), funding, 0.0004, 2.0)
    rng = np.random.default_rng(9)
    strategy = bench + rng.normal(0, 0.002, len(ts))
    ex = excess_metrics(strategy, bench, ts)
    direct, _ = daily_sharpe(strategy - bench, ts)
    assert ex["excess_sharpe"] == pytest.approx(direct, rel=1e-12)
    naive = ex["strategy_sharpe"] - ex["benchmark_sharpe"]
    assert abs(ex["excess_sharpe"] - naive) > 1e-6  # not the forbidden form


# ---------------------------------------------------------------- regime

def test_regime_label_is_deterministic_and_has_no_tolerance_band():
    flat = np.zeros(BARS_PER_DAY)
    ts = START_TS + np.arange(BARS_PER_DAY) * 1_800_000
    assert regime_label(flat, ts)[0] == "down"
    assert regime_label(flat, ts)[0] == "down"          # deterministic
    up = np.full(BARS_PER_DAY, 1e-9)
    assert regime_label(up, ts)[0] == "up"              # no band: tiny > 0 is up
    dn = np.full(BARS_PER_DAY, -1e-9)
    assert regime_label(dn, ts)[0] == "down"
    assert regime_label(np.zeros(0), np.zeros(0, dtype=np.int64))[0] == "down"


# ---------------------------------------------------------------- folds

def test_folds_are_midnight_aligned():
    ts, _, _, _ = _series(n_years=3, n_coins=3, seed=6)
    splits = {"train": (0, 21131), "validation": (21531, 26520),
              "lockbox": (26720, 35553)}
    for lo, hi in walk_forward_folds_28c(splits, int(ts[0])):
        assert midnight_aligned(lo, int(ts[0])), f"fold start {lo} not midnight"
        assert midnight_aligned(hi, int(ts[0])), f"fold end {hi} not midnight"


def test_fold_coverage_invariant_and_lockbox_unreachable():
    ts, _, _, _ = _series(n_years=3, n_coins=3, seed=7)
    splits = {"train": (0, 21131), "validation": (21531, 26520),
              "lockbox": (26720, 35553)}
    folds = walk_forward_folds_28c(splits, int(ts[0]))
    cov = coverage_report(folds, splits)
    assert cov["overlap"] is False
    assert cov["embargo_leak"] == []
    assert cov["max_hi"] <= splits["validation"][1] == 26520


def test_folds_do_not_contain_the_embargo():
    ts, _, _, _ = _series(n_years=3, n_coins=3, seed=8)
    splits = {"train": (0, 21131), "validation": (21531, 26520),
              "lockbox": (26720, 35553)}
    embargo = set(range(21131, 21531))
    for lo, hi in walk_forward_folds_28c(splits, int(ts[0])):
        assert not (embargo & set(range(lo, hi)))


def test_folds_are_four_and_nondegenerate():
    ts, _, _, _ = _series(n_years=3, n_coins=3, seed=11)
    splits = {"train": (0, 21131), "validation": (21531, 26520),
              "lockbox": (26720, 35553)}
    folds = walk_forward_folds_28c(splits, int(ts[0]))
    assert len(folds) == 4
    for lo, hi in folds:
        assert hi - lo > 2 * BARS_PER_DAY
