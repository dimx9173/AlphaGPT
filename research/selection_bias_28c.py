"""Selection-adjusted performance: charge the search for choosing the best run.

Reporting the best of N searches is not reporting an estimate. The number is
selected on the basis of how good it looks, so it is biased upward by an amount
that grows with both N and the variance of the Sharpe across searches. The
null control measured that variance directly: ten real searches spanned
-1.56 to +1.08, ten null searches spanned -1.75 to +4.88. A distribution that
wide, drawn from data with no structure in it, is exactly what makes a
"best of ten" meaningless without adjustment.

The Deflated Sharpe Ratio prices two separate things.

1. Selection. E[max SR] under the null is the Sharpe a search is expected to
   produce when it reports the best of N, given how much the trials vary:

       E[max SR] = sqrt(V[SR]) * ((1-g) Z^-1(1-1/N) + g Z^-1(1-1/(N*e)))

   with g the Euler-Mascheroni constant. The deflated Sharpe subtracts it.

2. Non-normality. Bar returns in this universe have excess kurtosis above 200,
   so the usual standard error of a Sharpe is wrong. PSR uses the observed
   skew and kurtosis:

       PSR = Phi( (SR - SR0) sqrt(n-1) / sqrt(1 - g3*SR + (g4-1)/4 * SR^2) )

Both are needed. Adjusting only for selection leaves a statistic that is
calibrated to Gaussian returns the data does not have.

The important reading: DSR below the threshold is not a weak result, it is a
measurement of how much of the reported Sharpe the search would have produced
anyway. That is the number to compare against.
"""
from __future__ import annotations

import math

import numpy as np

EULER_MASCHERONI = 0.5772156649015329


def _norm_ppf(p: float) -> float:
    """Inverse standard normal CDF.

    Implemented directly rather than pulled from scipy so the module has no
    optional dependency; the search module already runs in environments where
    scipy is absent.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must be strictly between 0 and 1")
    # Acklam's rational approximation, ~1e-9 relative accuracy.
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def expected_max_sharpe(n_trials: int, sharpe_variance: float) -> float:
    """Sharpe a search produces when it reports the best of n_trials, by chance.

    This is the number a null produces. If a real search reports less than
    this, the data carries no evidence of an edge over chance selection.
    """
    if n_trials < 2:
        raise ValueError("need at least 2 trials to select a maximum")
    if sharpe_variance <= 0:
        return 0.0
    sigma = math.sqrt(sharpe_variance)
    e = math.e
    g = EULER_MASCHERONI
    return sigma * ((1 - g) * _norm_ppf(1 - 1.0 / n_trials)
                    + g * _norm_ppf(1 - 1.0 / (n_trials * e)))


def prob_sharpe_above(observed_sr: float, benchmark_sr: float, n_obs: int,
                      skew: float, kurtosis: float) -> float:
    """Probability the true Sharpe exceeds benchmark, under non-normality.

    skew and kurtosis are the sample skewness and the NON-EXCESS kurtosis
    (normal = 3). Bar returns in this universe run excess kurtosis above 200,
    so using 3 here would badly overstate confidence.
    """
    if n_obs < 2:
        raise ValueError("need at least 2 observations")
    denom = 1.0 - skew * observed_sr + (kurtosis - 1.0) / 4.0 * observed_sr ** 2
    if denom <= 0:
        return 0.0
    z = (observed_sr - benchmark_sr) * math.sqrt(n_obs - 1) / math.sqrt(denom)
    return _norm_cdf(z)


def deflated_sharpe(observed_sr: float, n_trials: int, sharpe_variance: float,
                    n_obs: int, skew: float, kurtosis: float) -> dict:
    """Deflated Sharpe Ratio and the probability behind it.

    Returns the expected maximum under selection, the deflated Sharpe (observed
    minus that maximum), the PSR of the observed Sharpe against the selection
    benchmark, and the PSR of the deflated Sharpe against zero.
    """
    emax = expected_max_sharpe(n_trials, sharpe_variance)
    deflated = observed_sr - emax
    psr_observed = prob_sharpe_above(observed_sr, emax, n_obs, skew, kurtosis)
    psr_deflated = prob_sharpe_above(deflated, 0.0, n_obs, skew, kurtosis)
    return {
        'observed_sharpe': observed_sr,
        'expected_max_sharpe': emax,
        'deflated_sharpe': deflated,
        'n_trials': n_trials,
        'sharpe_variance': sharpe_variance,
        'psr_observed_vs_selection': psr_observed,
        'psr_deflated_vs_zero': psr_deflated,
        'skew': skew,
        'kurtosis': kurtosis,
        'n_obs': n_obs,
    }


def selection_adjusted_pvalue(sharpes: list[float], n_obs: int,
                              skew: float, kurtosis: float) -> dict:
    """Adjust a whole batch: the best reported Sharpe, minus what selection buys.

    sharpes is the Sharpe from every independent search. The best one is the
    number that would be reported, so it is the one that needs deflating.
    """
    arr = np.asarray(sharpes, dtype=np.float64)
    n = len(arr)
    if n < 2:
        raise ValueError("need at least 2 searches to price selection")
    best = float(np.max(arr))
    var = float(np.var(arr, ddof=1))
    return deflated_sharpe(best, n, var, n_obs, skew, kurtosis)
