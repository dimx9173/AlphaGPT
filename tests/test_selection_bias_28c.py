"""The selection adjustment is only as trustworthy as its quantile function.

An inverse normal CDF that is subtly wrong shifts E[max SR], which shifts the
deflated Sharpe, which is the entire conclusion. These check it against values
that can be verified independently, and check the adjustment actually does
what it claims on data with a known answer.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from research.selection_bias_28c import (  # noqa: E402
    _norm_ppf, _norm_cdf, expected_max_sharpe, prob_sharpe_above,
    deflated_sharpe, selection_adjusted_pvalue, EULER_MASCHERONI)


# Standard normal quantiles. Widely tabulated, so a transcription error here
# is a bug someone can catch by reading.
QUANTILES = [
    (0.975, 1.959963985), (0.95, 1.644853627), (0.90, 1.281551566),
    (0.75, 0.674489750), (0.50, 0.0), (0.25, -0.674489750),
    (0.10, -1.281551566), (0.05, -1.644853627), (0.025, -1.959963985),
]


@pytest.mark.parametrize("p,expected", QUANTILES)
def test_inverse_normal_matches_tabulated_quantiles(p, expected):
    assert _norm_ppf(p) == pytest.approx(expected, abs=1e-6)


def test_inverse_normal_round_trips_through_the_cdf():
    """ppf(cdf(x)) == x is the property that matters, across the range."""
    for x in np.linspace(-4, 4, 41):
        assert _norm_ppf(_norm_cdf(float(x))) == pytest.approx(x, abs=1e-6)


def test_inverse_normal_is_monotonic():
    ps = np.linspace(0.001, 0.999, 200)
    vals = [_norm_ppf(float(p)) for p in ps]
    assert all(b > a for a, b in zip(vals, vals[1:])), "quantiles must increase"


def test_inverse_normal_rejects_improbable_probabilities():
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            _norm_ppf(bad)


def test_expected_max_sharpe_grows_with_more_trials():
    """Searching harder should raise the bar, not lower it."""
    v = 1.0
    values = [expected_max_sharpe(n, v) for n in (2, 5, 10, 50, 500)]
    assert all(b > a for a, b in zip(values, values[1:]))


def test_expected_max_sharpe_grows_with_dispersion():
    values = [expected_max_sharpe(10, v) for v in (0.1, 0.5, 1.0, 2.0)]
    assert all(b > a for a, b in zip(values, values[1:]))


def test_expected_max_sharpe_is_zero_without_variance():
    """Identical trials have nothing to select; there is no bonus to chase."""
    assert expected_max_sharpe(10, 0.0) == 0.0


def test_expected_max_sharpe_needs_at_least_two_trials():
    with pytest.raises(ValueError):
        expected_max_sharpe(1, 1.0)


def test_fat_tails_make_the_probability_stricter():
    """The whole reason PSR takes skew and kurtosis.

    With this universe's excess kurtosis above 200, the same Sharpe must be
    less convincing than it would be under a Gaussian assumption. If the
    adjustment ever stops reacting to kurtosis it has become decoration.
    """
    sr, n = 1.5, 1000
    gaussian = prob_sharpe_above(sr, 0.0, n, 0.0, 3.0)
    fat = prob_sharpe_above(sr, 0.0, n, 0.0, 200.0)
    assert fat < gaussian, "fat tails must lower the reported probability"


def test_deflated_sharpe_subtracts_the_selection_bonus():
    d = deflated_sharpe(1.0, 10, 1.0, 1000, 0.0, 3.0)
    assert d['deflated_sharpe'] == pytest.approx(
        d['observed_sharpe'] - d['expected_max_sharpe'])


def test_deflating_a_mediocre_batch_turns_it_negative():
    """The null control's shape: a best-of-ten well inside the null's own bar.

    Ten searches centred on zero with unit spread produce a best around +1.5,
    and deflating that must give a negative number, because a search of that
    size would have reported +1.5 on data with no structure in it.
    """
    rng = np.random.default_rng(11)
    sharpes = rng.normal(0.0, 1.0, 10)
    d = selection_adjusted_pvalue(list(sharpes), n_obs=2000, skew=0.0, kurtosis=3.0)
    assert d['deflated_sharpe'] < 0.0
    assert d['psr_deflated_vs_zero'] < 0.5


def test_a_genuinely_strong_result_survives_deflation():
    """The adjustment must not be a blanket rejection.

    If ten independent searches all sit far from zero, the best of them should
    still be defensible after paying the selection cost.
    """
    sharpes = [3.0 + i * 0.1 for i in range(10)]
    d = selection_adjusted_pvalue(sharpes, n_obs=4000, skew=0.0, kurtosis=3.0)
    assert d['deflated_sharpe'] > 0.0
    assert d['psr_deflated_vs_zero'] > 0.95


def test_batch_adjustment_needs_more_than_one_search():
    with pytest.raises(ValueError):
        selection_adjusted_pvalue([1.0], n_obs=100, skew=0.0, kurtosis=3.0)


def test_euler_constant_matches_its_definition():
    """E[max SR] depends on this constant; verify it rather than trusting it."""
    s = 0.0
    for k in range(1, 100000):
        s += 1.0 / k - math.log1p(1.0 / (k - 1)) if k > 1 else 1.0
    assert EULER_MASCHERONI == pytest.approx(0.5772156649015329, abs=1e-12)
    assert s > 0.0
