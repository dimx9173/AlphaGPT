#!/usr/bin/env python3
"""Null data for the 28c contract: same statistics, no real relationship to predict.

The point is to answer one question about the search itself, not about the
market. Ten real searches produced a median lockbox Sharpe of +0.78 and a
train-to-lockbox correlation of -0.82. Before tuning a reward function against
that, it is worth knowing whether the search reports an edge when none exists.
If it does, no reward change can be evaluated until the process is fixed,
because the measurement itself would be untrustworthy.

A null must be fair. If the synthetic series were obviously unrealistic the
search would trivially fail and the test would prove nothing. So each null
preserves what the strategy actually consumes and destroys only the
relationship that could be traded:

  iid    block-bootstrapped log returns and volume, drawn independently per
         coin. Preserves the marginal return distribution, its fat tails,
         volatility clustering, and the return/volume relationship that the
         FOMO and LOG_VOL factors read. Destroys cross-coin correlation and
         any real predictability.
  xsec   the real series for one coin paired with another coin's bars.
         Preserves each coin's own time series intact but breaks the
         cross-sectional relationship a multi-coin portfolio depends on.

  none   the real contract, for baseline comparison.

Everything downstream is untouched: same features, same grammar, same splits,
same accounting. Only the data changes, so a difference in results is
attributable to the data and not to a different code path.
"""
from __future__ import annotations

import numpy as np


def _block_starts(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Ordered block boundaries covering [0, n)."""
    if block <= 1:
        return np.arange(n)
    return np.arange(0, n, block)


def block_bootstrap_series(log_ret: np.ndarray, volume: np.ndarray,
                           block: int, rng: np.random.Generator):
    """Resample (return, volume) pairs in contiguous blocks.

    Returns and volumes are resampled together and contiguously, because the
    volume factors read the volume *at* a return, not a marginal one. Drawing
    them separately would break a relationship the search can see, and the
    result would flatter the real data by comparison.
    """
    n = len(log_ret)
    starts = _block_starts(n, block, rng)
    need = int(np.ceil(n / block))
    picked = starts[rng.integers(0, len(starts), need)]
    idx = np.concatenate([np.arange(s, min(s + block, n)) for s in picked])[:n]
    if len(idx) < n:                      # tail safety if the last block is short
        idx = np.concatenate([idx, np.arange(n - (n - len(idx)), n)])
    return log_ret[idx], volume[idx]


def make_null(kind: str, data: dict, seed: int, block: int = 48):
    """Rebuild one coin's bars from a null resample of its own series.

    ``data`` carries aligned open/high/low/close/volume arrays.
    Returns the same keys so callers cannot tell the difference.
    """
    rng = np.random.default_rng(seed)
    close = np.asarray(data['close'], dtype=np.float64)
    volume = np.asarray(data['volume'], dtype=np.float64)
    n = len(close)
    log_ret = np.diff(np.log(np.maximum(close, 1e-12)), prepend=np.log(close[0]))
    r, v = block_bootstrap_series(log_ret, volume, block, rng)
    # The iid null is the frozen control: every "the search harvests noise"
    # artifact was measured against it, and its reported Sharpe is quoted in
    # the write-ups. It is rebuilt here EXACTLY as it was before _rebuild_bars
    # existed, so those artifacts still reproduce.
    #
    # Routing iid through the shifted reconstruction instead changed the price
    # path and moved the re-scored lockbox Sharpe by up to 0.36, which is large
    # enough to change a median. That is the same failure as moving a
    # benchmark: the control stayed the same NAME while becoming a different
    # experiment. The new reconstruction is only safe for the diagnostic nulls,
    # which were introduced alongside it.
    new_close = np.maximum(close[0] * np.exp(np.cumsum(r)), 1e-12)
    prev = np.concatenate(([new_close[0]], new_close[:-1]))
    open_ = np.where(rng.random(n) < 0.5, prev, new_close)
    hi = np.maximum(open_, new_close) * (1.0 + rng.random(n) * 0.004)
    lo = np.minimum(open_, new_close) * (1.0 - rng.random(n) * 0.004)
    return {'open': open_, 'high': hi, 'low': lo, 'close': new_close,
            'volume': np.maximum(v, 0.0), 'timestamp': list(data['timestamp'])}


def permute_cross_sectional(data_by_coin: dict, coins: list, seed: int) -> dict:
    """Keep every coin's own time series, rotate which coin it is paired with.

    Each coin's bars move to a different coin's slot. A single-coin factor
    still behaves normally; the cross-coin structure that a portfolio of 28
    positions depends on is gone.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(coins))
    out = {}
    for i, c in enumerate(coins):
        out[c] = data_by_coin[coins[order[i]]]
    return out


# ---------------------------------------------------------------------------
# Diagnostic nulls: which preserved property is the search reading?
# ---------------------------------------------------------------------------
#
# The iid null keeps fat tails (kurtosis 236 -> 173), keeps volatility
# clustering (AC(1) -0.043 -> -0.036), and keeps the per-coin marginal return
# distribution (p99/median 8.5 -> 8.5). It removes only cross-coin
# co-movement and any genuine predictability. Yet a search over the iid null
# reports a median lockbox Sharpe of +0.82, higher than the +0.78 the same
# search reports on the real data.
#
# Those three preserved properties are the candidates. Each mode below breaks
# exactly one of them, so the search result says which one mattered:
#
#   iid1   resample at the bar level instead of in blocks. Breaks volatility
#          clustering; keeps the marginal distribution and the fat tails.
#   iidw   winsorise the resampled returns at the 1%/99% tails. Breaks fat
#          tails; keeps the marginal centre and, with iid1, the clustering.
#   iidg   rank-transform the resampled returns onto a Gaussian marginal.
#          Breaks the tail shape completely and keeps a normal marginal.
#
# If the null's Sharpe collapses under iid1, the search was reading
# autocorrelation, and the honest conclusion is that the factor set and the
# position construction are not measuring a causal relationship. If it
# survives iid1 and collapses under iidw, it was reading tail asymmetry. If it
# survives all three, the position construction itself generates signal from
# scale, which is a bug to find rather than a property to exploit.
#
# None of these are "easier" nulls in the sense of being unrealistic. Each is
# strictly less realistic than iid, and each is used to attribute an effect,
# not to pass or fail a candidate.

def _winsorise(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    a, b = np.quantile(x, lo), np.quantile(x, hi)
    return np.clip(x, a, b)


def _rank_to_gaussian(x: np.ndarray) -> np.ndarray:
    """Monotone map of x onto a standard normal marginal, order preserved.

    Uses Acklam's rational approximation to the inverse normal CDF, refined
    by one Halley step against erfc. The accuracy is ~1e-15 relative, which
    matters here because the output is fed straight back into a search whose
    whole failure mode is picking up small numerical artifacts.
    """
    from math import sqrt
    n = len(x)
    order = np.argsort(np.argsort(x, kind="stable"), kind="stable")
    u = (order + 0.5) / n

    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    plow, phigh = 0.02425, 1 - 0.02425

    x_ = np.empty_like(u)
    lo_m = u < plow
    hi_m = u > phigh
    mid_m = ~(lo_m | hi_m)

    q = np.sqrt(-2.0 * np.log(np.maximum(u[lo_m], 1e-300)))
    x_[lo_m] = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = np.sqrt(-2.0 * np.log(1.0 - u[hi_m]))
    x_[hi_m] = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /                 ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = u[mid_m] - 0.5
    r = q * q
    x_[mid_m] = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
                (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)

    # One Halley refinement step, skipped in the tails.
    #
    # The correction divides by a pdf, which underflows to zero in the far
    # tail and turns the step into a division by zero. The tails are where
    # Acklam is already accurate to full double precision, so there is nothing
    # to gain there and a diverging step to lose. The guard is on |x|.
    safe = np.abs(x_) < 30.0
    if np.any(safe):
        xs = x_[safe]
        e = _phi(xs) - u[safe]
        pdf = np.exp(-0.5 * xs * xs) / sqrt(2.0 * np.pi)
        u_e = e / pdf
        x_[safe] = xs - u_e / (1.0 + 0.5 * xs * u_e)
    return x_


def _erf_vec(x: np.ndarray) -> np.ndarray:
    """Abramowitz-Stegun 7.1.26 erf, |error| < 1.5e-7.

    Note this computes erf, not erfc, despite the approximation having been
    copied around under both names. The distinction is not cosmetic:
    Phi(x) = 0.5 * (1 + erf(x / sqrt(2))), so returning the complement here
    inverts the normal CDF about 1/2. That inversion is symmetric and does not
    fail loudly, it just pushes the inverse-CDF refinement the wrong way and
    drives the output to roughly 1e4 where it should be about 4.
    """
    sign = np.sign(x)
    ax = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * ax)
    y = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
          - 0.284496736) * t + 0.254829592) * t
    return sign * (1.0 - y * np.exp(-ax * ax))


def _phi(x: np.ndarray) -> np.ndarray:
    """Standard normal CDF, vectorised."""
    from math import sqrt
    return 0.5 * (1.0 + _erf_vec(x / sqrt(2.0)))



def _rebuild_bars(close0: float, r: np.ndarray, volume: np.ndarray,
                  rng: np.random.Generator) -> dict:
    """Rebuild OHLCV around a return path, in a numerically safe range.

    The obvious form, close0 * exp(cumsum(r)), overflows to inf as soon as the
    cumulative sum is large, and the subsequent clamp to a floor turns the
    whole tail of the series into a constant. That produced a null whose
    prices all read 1e-12, which is not a distribution at all. Shifting the
    cumulative path by its maximum keeps the exponent non-positive, so the
    price series can never overflow and the shape of the return process is
    preserved exactly.
    """
    cum = np.cumsum(r)
    path = np.exp(cum - cum.max())
    new_close = np.maximum(float(close0), 1e-12) * path
    new_close = np.maximum(new_close, 1e-12)
    n = len(new_close)
    prev = np.concatenate(([new_close[0]], new_close[:-1]))
    open_ = np.where(rng.random(n) < 0.5, prev, new_close)
    hi = np.maximum(open_, new_close) * (1.0 + rng.random(n) * 0.004)
    lo = np.minimum(open_, new_close) * (1.0 - rng.random(n) * 0.004)
    return {'open': open_, 'high': hi, 'low': lo, 'close': new_close,
            'volume': np.maximum(volume, 0.0)}


def make_diagnostic_null(kind: str, data: dict, seed: int, block: int = 48):
    """Rebuild one coin's bars from a null that breaks one preserved property."""
    rng = np.random.default_rng(seed)
    close = np.asarray(data['close'], dtype=np.float64)
    volume = np.asarray(data['volume'], dtype=np.float64)
    n = len(close)
    log_ret = np.diff(np.log(np.maximum(close, 1e-12)), prepend=np.log(close[0]))

    b = 1 if kind in ("iid1", "iidw", "iidg") else block
    r, v = block_bootstrap_series(log_ret, volume, b, rng)

    if kind == "iidw":
        # Break fat tails, keep clustering (block resample unchanged).
        r = _winsorise(r, 0.01, 0.99)
    elif kind == "iidg":
        # Break the tail shape entirely, keep a normal marginal.
        r = _rank_to_gaussian(r)
        r = r * float(np.std(log_ret)) if np.std(log_ret) > 0 else r

    out = _rebuild_bars(close[0], r, v, rng)
    out['timestamp'] = list(data['timestamp'])
    return out


def null_mode_kinds():
    return ("iid", "xsec", "iid1", "iidw", "iidg", "none")
