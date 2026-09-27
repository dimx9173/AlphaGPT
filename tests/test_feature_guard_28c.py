#!/usr/bin/env python3
"""A factor with no variance carries no information, and nothing else catches it.

Both defects this guards are the same defect: a missing input became a constant,
and a constant is a legal number, so nothing objected.

  funding    a flat +0.0005 at every settlement  -> credited a net-short book
             9.6%/yr the venue never paid, and flipped the v3c lockbox Sharpe
             from -0.624 to +0.479
  LIQ_SCORE  data.get('liquidity', 1e7)          -> a constant 0.4 on every bar
             of every coin, still occupying a slot in the formula grammar

Neither showed up in a Sharpe check, a drawdown check, or a formula validation.
They are only visible from the input side, which is what these tests look at.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.causal_12f import (  # noqa: E402
    FEATURE_NAMES,
    assert_features_vary,
    causal_features,
    constant_features,
)


def _ohlcv(n=900, seed=0):
    """Synthetic bars with genuine variation in every price relationship."""
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    # High and low bracket close asymmetrically. A symmetric envelope makes
    # CLOSE_POS identically 0.5, which is a property of the generator, not of
    # the factors, and it would make this file test the wrong thing.
    up = rng.uniform(0.001, 0.02, n)
    down = rng.uniform(0.001, 0.02, n)
    high = close * (1.0 + up)
    low = close * (1.0 - down)
    body = (rng.uniform(-0.8, 0.8, n) * np.minimum(up, down))
    return {
        "open": close + body,
        "high": high,
        "low": low,
        "close": close,
        "volume": rng.uniform(1e5, 1e6, n),
    }


def test_ohlcv_only_data_yields_no_constant_factor():
    """Every factor must be computable from the columns the contract actually has."""
    F = causal_features(_ohlcv())
    assert constant_features(F) == [], f"constant factors: {constant_features(F)}"


def test_liq_score_is_not_a_placeholder():
    """LIQ_SCORE must carry variance. It used to be exactly 0.4 everywhere."""
    F = causal_features(_ohlcv())
    liq = F[FEATURE_NAMES.index("LIQ_SCORE")]
    assert float(np.ptp(liq)) > 1e-3, "LIQ_SCORE is still effectively constant"
    assert not np.allclose(liq, 0.4), "LIQ_SCORE is still the old 0.4 placeholder"


def test_liq_score_uses_the_log_scale():
    """Without the log, Amihud's heavy tail collapses the factor to ~1e-5 variation.

    That passes a naive non-constant check while being useless, so the bound
    here is deliberately generous but far above the collapsed value.
    """
    F = causal_features(_ohlcv())
    liq = F[FEATURE_NAMES.index("LIQ_SCORE")]
    assert float(np.std(liq)) > 0.1, "LIQ_SCORE needs the log scale to be usable"


def test_liq_score_is_causal():
    """Rewriting the future must not change the past.

    Liquidity scored off anything non-causal would leak, and a liquidity factor
    is exactly the kind of input where a peeking implementation is easy to
    write by accident.
    """
    data = _ohlcv()
    a = causal_features(data)
    perturbed = {k: np.array(v, copy=True) for k, v in data.items()}
    cut = len(perturbed["close"]) // 2
    for k in perturbed:
        perturbed[k][cut:] *= 3.0
    b = causal_features(perturbed)
    idx = FEATURE_NAMES.index("LIQ_SCORE")
    assert np.allclose(a[idx, :cut - 210], b[idx, :cut - 210]), "LIQ_SCORE leaks the future"


def test_guard_rejects_the_old_placeholder():
    """The exact historical defect must now be a hard error."""
    F = causal_features(_ohlcv())
    F[FEATURE_NAMES.index("LIQ_SCORE")] = 0.4
    assert "LIQ_SCORE" in constant_features(F)
    with pytest.raises(ValueError, match="LIQ_SCORE"):
        assert_features_vary(F, context="BTC on the 28c contract")


def test_guard_message_names_the_missing_input():
    """An error that does not say what to fix is an error nobody fixes."""
    F = causal_features(_ohlcv())
    F[1] = 0.4
    with pytest.raises(ValueError) as exc:
        assert_features_vary(F)
    assert "needs" in str(exc.value)


def test_guard_passes_on_healthy_features():
    assert_features_vary(causal_features(_ohlcv()), context="synthetic")


def test_guard_rejects_a_wrongly_shaped_array():
    with pytest.raises(ValueError, match="n_factors"):
        assert_features_vary(np.zeros((12, 100, 2)))


def test_feature_cache_is_versioned():
    """A stale cache must not shadow current factor code.

    This is not hypothetical: the old cache files on disk carried the constant
    0.4, and an unversioned cache would have quietly reinstated it after the
    factor was fixed.
    """
    src = (ROOT / "research" / "ga_28c_30m_3y.py").read_text()
    assert "FEATURE_CACHE_VERSION" in src
    assert "-{FEATURE_CACHE_VERSION}.npy" in src


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
