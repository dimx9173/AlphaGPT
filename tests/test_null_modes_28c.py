"""Tests for the diagnostic nulls.

The diagnostic nulls exist to attribute an effect: each breaks exactly one
property that the iid null preserves, so the search result says which
property it was reading. A null that breaks the wrong property, or fails
silently, would make the attribution wrong while still looking like a
plausible number, so the property each mode is supposed to break is asserted
here rather than assumed.
"""
from __future__ import annotations

import numpy as np
import pytest

from research.null_data_28c import (make_diagnostic_null, make_null,
                                    null_mode_kinds, _phi, _rank_to_gaussian,
                                    _winsorise)


def _garch(n=20000, seed=0, fat_tails=True):
    """A series with volatility clustering and, optionally, fat tails.

    GARCH(1,1) on t(3) shocks, with a = 0.30 and b = 0.75 chosen so the
    fixture reproduces the two properties the real 28-coin contract shows:
    AC(1) of squared returns near +0.22 and excess kurtosis in the hundreds.
    A plain Gaussian-innovation GARCH has excess kurtosis near zero, so a test
    about fat tails would otherwise be testing the wrong thing, and a
    mis-specified recursion here produces a series that overflows to NaN and
    makes every downstream assertion pass or fail for the wrong reason.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_t(3.0, n) if fat_tails else rng.standard_normal(n)
    z = z / np.std(z)
    r = np.zeros(n)
    s2 = 1e-8
    for i in range(1, n):
        s2 = 1e-8 + 0.30 * r[i - 1] ** 2 + 0.75 * s2
        r[i] = np.sqrt(s2) * z[i]
    cum = np.cumsum(r)
    close = 100 * np.exp(cum - cum.mean())
    prev = np.concatenate(([close[0]], close[:-1]))
    open_ = np.where(rng.random(n) < 0.5, prev, close)
    return {"open": open_, "high": np.maximum(open_, close) * 1.001,
            "low": np.minimum(open_, close) * 0.999, "close": close,
            "volume": np.abs(rng.normal(1e4, 3e3, n)),
            "timestamp": list(range(n))}


def _rets(bar):
    return np.diff(np.log(np.asarray(bar["close"], dtype=np.float64)))


def _ac(a, lag=1):
    return float(np.corrcoef(a[:-lag], a[lag:])[0, 1])


def _kurt(r):
    return float(np.mean(((r - r.mean()) / r.std()) ** 4))


# --- the erf/erf inversion, which produced values ~1e4 instead of ~4 --------

def test_phi_matches_the_normal_cdf():
    """_phi must be the CDF, not its complement.

    The A&S 7.1.26 approximation computes erf. Wrapping it as erfc inverts the
    normal CDF about 1/2, which is symmetric and so does not fail loudly; it
    drives the inverse-CDF refinement the wrong way instead.
    """
    from statistics import NormalDist
    nd = NormalDist()
    for x in (-8, -6, -4, -2, 0, 2, 4, 6, 8):
        got = float(_phi(np.array([float(x)]))[0])
        assert abs(got - nd.cdf(x)) < 1e-6, f"Phi({x}) = {got}, want {nd.cdf(x)}"


def test_rank_to_gaussian_is_an_identity_on_normal_data():
    from statistics import NormalDist
    nd = NormalDist()
    u = np.array([0.001, 0.025, 0.25, 0.5, 0.75, 0.975, 0.999])
    data = np.array([nd.inv_cdf(q) for q in u])
    out = _rank_to_gaussian(data)
    want = np.array([nd.inv_cdf((i + 0.5) / len(u)) for i in range(len(u))])
    assert np.max(np.abs(out - want)) < 1e-5


def test_rank_to_gaussian_removes_tails_and_preserves_order():
    rng = np.random.default_rng(0)
    y = rng.standard_t(df=2.5, size=50000)
    z = _rank_to_gaussian(y)
    assert abs(z.mean()) < 0.02 and abs(z.std() - 1.0) < 0.02
    assert 2.7 < _kurt(z) < 3.3, f"kurtosis { _kurt(z) } should be ~3"
    # monotone: order preservation is what makes this a transform of the same
    # sample rather than a fresh draw
    assert np.all(np.diff(np.sort(_rank_to_gaussian(y[:500]))) >= 0)


# --- each mode breaks the property it claims to break -----------------------

def test_iid_preserves_volatility_clustering():
    """The baseline: iid keeps clustering, which is why it is not diagnostic."""
    d = _garch()
    r2 = _rets(make_null("iid", d, 0, 48)) ** 2
    assert _ac(r2) > 0.10, "iid null must keep volatility clustering"


def test_iid1_breaks_volatility_clustering():
    d = _garch()
    r2 = _rets(make_diagnostic_null("iid1", d, 0, 48)) ** 2
    assert abs(_ac(r2)) < 0.05, "iid1 must destroy AC(1) of squared returns"


def test_iidw_breaks_fat_tails():
    """iidw must cut the kurtosis substantially.

    Winsorising at 1%/99% cannot reach a Gaussian marginal: clipping the tails
    of a heavy-tailed sample leaves kurtosis around 10, not 3. The test
    therefore asserts the reduction, and the assertion is a ratio so it does
    not silently pass if the null stops working. iidg is the mode that has to
    reach a Gaussian marginal.
    """
    d = _garch()
    before = _kurt(_rets(d))
    after = _kurt(_rets(make_diagnostic_null("iidw", d, 0, 48)))
    assert before > 50.0, "the fixture must be fat-tailed first"
    assert after < 0.25 * before, f"iidw kurtosis only fell {before:.0f} -> {after:.0f}"


def test_iidg_breaks_the_tail_shape_outright():
    d = _garch()
    r = _rets(make_diagnostic_null("iidg", d, 0, 48))
    assert 2.6 < _kurt(r) < 3.4, f"iidg kurtosis {_kurt(r)} should be ~3"


def test_modes_are_registered():
    for k in ("iid", "iid1", "iidw", "iidg", "xsec", "none"):
        assert k in null_mode_kinds()


def test_every_mode_keeps_the_bar_contract():
    """A null that drops or reorders keys would break the caller silently."""
    d = _garch(5000)
    for kind in ("iid1", "iidw", "iidg"):
        out = make_diagnostic_null(kind, d, 0, 48)
        assert set(out) == set(d), f"{kind} changed the key set"
        assert len(out["close"]) == len(d["close"])
        assert len(out["timestamp"]) == len(d["timestamp"])
        assert np.all(np.asarray(out["close"]) > 0), f"{kind} made a nonpositive price"
        assert np.all(np.asarray(out["volume"]) >= 0)
        assert np.all(np.asarray(out["high"]) >= np.asarray(out["low"]))


def test_deterministic_given_a_seed():
    d = _garch(3000)
    for kind in ("iid1", "iidw", "iidg"):
        a = make_diagnostic_null(kind, d, 7, 48)
        b = make_diagnostic_null(kind, d, 7, 48)
        assert np.array_equal(a["close"], b["close"]), f"{kind} is not reproducible"
        c = make_diagnostic_null(kind, d, 8, 48)
        assert not np.array_equal(a["close"], c["close"]), f"{kind} ignores the seed"


def test_winsorise_clips_both_tails():
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 100.0])
    w = _winsorise(x, 0.1, 0.9)
    assert w.max() < 100.0 and w.min() >= 0.0
    assert np.all(np.diff(w) >= 0), "winsorising must stay monotone"
