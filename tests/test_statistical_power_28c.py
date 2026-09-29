"""Tests for statistical_power and for artifact provenance.

Two separate things are pinned here.

1. `statistical_power` must build its interval from the Sharpe it is given.
   It used to hardcode a per-period Sharpe of 0.5, which in Lo's formula is an
   annualised Sharpe of about 9.6. The error was small in absolute terms
   (~12% on the half-width) and never flipped a verdict, which is exactly why
   it survived: nothing was obviously wrong downstream.

2. The v3c artifact is not reproducible from the current tree. That is a
   property of the repository's history, not a passing observation, and the
   only way to keep it true is to keep saying it out loud. If someone restores
   the old LIQ_SCORE definition these tests fail, which is the correct signal
   that every number measured on the old factor set has to be re-derived.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from research.regime_gate_28c import statistical_power

BARS_PER_DAY = 48  # 30-minute bars


# ---------------------------------------------------------------------------
# The interval depends on the Sharpe it is given
# ---------------------------------------------------------------------------

def test_half_width_matches_the_lo_formula_exactly():
    """half_width = 1.96 * sqrt((1 + 0.5 * S^2) / years), S per-period."""
    days = 184
    sr_annual = 0.479
    p = statistical_power(days * BARS_PER_DAY, observed_sharpe=sr_annual)
    years = days / 365.25
    per_period = sr_annual / math.sqrt(365.25)
    expected = 1.96 * math.sqrt((1.0 + 0.5 * per_period ** 2) / years)
    assert p["sharpe_ci95_half_width"] == pytest.approx(expected, abs=1e-4)
    assert p["observed_per_period_sharpe"] == pytest.approx(per_period, abs=1e-6)


def test_hardcoded_zero_point_five_is_not_used():
    """The old formula substituted S = 0.5, i.e. annualised Sharpe ~9.6.

    At any realistic Sharpe the two differ, and a caller must be able to see
    which one produced a number.
    """
    days = 184
    p = statistical_power(days * BARS_PER_DAY, observed_sharpe=0.479)
    years = days / 365.25
    old = 1.96 * math.sqrt((1.0 + 0.5 ** 2) / years)
    assert p["sharpe_ci95_half_width"] < old
    assert not math.isclose(p["sharpe_ci95_half_width"], old, rel_tol=1e-6)


def test_half_width_is_monotone_in_observed_sharpe():
    """A larger estimate implies a fatter-tailed return, hence a wider interval.

    This is the term the old code was trying to express and could not, because
    it was not reading the Sharpe at all.
    """
    days = 184
    widths = [statistical_power(days * BARS_PER_DAY, observed_sharpe=s)["sharpe_ci95_half_width"]
              for s in (0.1, 0.5, 1.0, 2.0, 5.0)]
    assert widths == sorted(widths), widths
    assert widths[0] < widths[-1]


def test_missing_sharpe_is_reported_as_unidentified_not_defaulted():
    p = statistical_power(300 * BARS_PER_DAY)
    assert p["observed_annualised_sharpe"] is None
    assert p["observed_per_period_sharpe"] is None
    assert p["serial_correction_unidentified"] is True
    assert "lower bound" in p["note"]
    # 1.0 is the tightest defensible value of (1 + S^2/2) with S unknown.
    years = 300 / 365.25
    assert p["sharpe_ci95_half_width"] == pytest.approx(1.96 * math.sqrt(1.0 / years), abs=1e-4)


def test_achieved_t_is_annualised_sharpe_times_sqrt_years():
    days = 184
    sr = 0.479
    p = statistical_power(days * BARS_PER_DAY, observed_sharpe=sr)
    assert p["achieved_t_statistic"] == pytest.approx(sr * math.sqrt(days / 365.25), abs=1e-4)
    assert p["achieved_t_statistic"] < p["t_required_for_95pct"]


def test_shorter_window_is_less_resolvable():
    short = statistical_power(184 * BARS_PER_DAY, observed_sharpe=0.479)
    long_ = statistical_power(3000 * BARS_PER_DAY, observed_sharpe=0.479)
    assert short["sharpe_ci95_half_width"] > long_["sharpe_ci95_half_width"]
    assert not short["can_resolve_sharpe_1_0"]
    assert long_["can_resolve_sharpe_1_0"]


def test_can_resolve_1_0_needs_about_3_84_years_at_realistic_sharpe():
    """The number quoted in docs/statistical_power_ceiling_28c.md."""
    sr = 0.479
    per_period = sr / math.sqrt(365.25)
    years = (1.96 ** 2) * (1 + 0.5 * per_period ** 2) / 1.0 ** 2
    assert years == pytest.approx(3.84, abs=0.01)


# ---------------------------------------------------------------------------
# The v3c artifact does not belong to the current tree
# ---------------------------------------------------------------------------

V3C = ROOT / "results" / "ga_28c_100gen_v3c_seed42.json"


@pytest.mark.skipif(not V3C.exists(), reason="v3c artifact is gitignored and absent")
def test_v3c_formula_depends_on_the_factor_that_de2e353_changed():
    """v3c's formula contains LIQ_SCORE, so a factor change moves its number.

    If this test starts failing because the formula no longer contains factor 1,
    the artifact has been superseded and the provenance warning in the
    statistical-power document must be revisited rather than deleted.
    """
    art = json.loads(V3C.read_text())
    assert 1 in art["formula"], "v3c no longer uses LIQ_SCORE; re-check the doc"


@pytest.mark.skipif(not V3C.exists(), reason="v3c artifact is gitignored and absent")
def test_v3c_recorded_sharpe_is_not_reproducible_from_the_current_tree():
    """The recorded +0.478 cannot be regenerated, and that must stay visible.

    It would be easy to "fix" this by editing the recorded number, which would
    make the artifact consistent and its history false. The check is against
    the artifact's own value so a re-baseline has to be an explicit act.
    """
    import research.ga_28c_30m_3y as G
    from research.splits_28c import split_indices

    art = json.loads(V3C.read_text())
    common, maps, returns, funding_mask = G.load_data(null="none")
    sp = split_indices(len(common), common[0], common[-1])
    lo, hi = sp["lockbox"]
    res = G.evaluate(tuple(art["formula"]), maps, returns, lo, hi, funding_mask,
                     sp["train"][1], common, funding_by_coin=None,
                     reward_version="v1")
    recorded = art["oos"]["portfolio_sharpe"]
    assert abs(res["portfolio_sharpe"] - recorded) > 0.1, (
        f"v3c now reproduces {res['portfolio_sharpe']:.4f} against a recorded "
        f"{recorded:.4f}. Either the factor history was restored or the "
        "artifact was re-baselined; the statistical-power document must be "
        "updated either way.")


def test_current_liq_score_is_not_the_old_constant():
    """The reason the artifact above cannot be reproduced, pinned directly."""
    from research.causal_12f import causal_features, constant_features
    # A deterministic ramp would make PRESSURE, CLOSE_POS and MOM_REV constant
    # for a legitimate reason, so use a seeded random walk instead: every
    # factor is then non-degenerate and a failure means a real regression.
    rs = np.random.RandomState(0)
    steps = rs.normal(0.0, 0.01, 600)
    close = 100.0 * np.exp(np.cumsum(steps))
    open_ = close * (1.0 + rs.normal(0.0, 0.002, 600))
    spread = np.abs(rs.normal(0.0, 0.004, 600)) + 0.001
    high = np.maximum(open_, close) * (1.0 + spread)
    low = np.minimum(open_, close) * (1.0 - spread)
    volume = 1000.0 * np.exp(rs.normal(0.0, 0.3, 600))
    data = {"close": close, "open": open_, "high": high, "low": low, "volume": volume}
    f = causal_features(data)
    liq = f[1]
    assert np.std(liq) > 1e-6, "LIQ_SCORE has gone back to being a constant"
    # The placeholder was exactly 0.4 on every bar.
    assert not np.allclose(liq, 0.4)
    assert constant_features(f) == []
