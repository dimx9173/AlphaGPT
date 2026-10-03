"""The 36-dimension expansion: 12 existing + 24 measured-low-correlation additions."""
import numpy as np
import pytest

import research.causal_36f as M
import research.causal_12f as B


def test_ships_exactly_24():
    assert len(M.SHIPPED_24) == 24
    assert len(set(M.SHIPPED_24)) == 24


def test_no_shipped_factor_duplicates_an_existing_name():
    assert not set(M.SHIPPED_24) & set(B.FEATURE_NAMES)


def test_total_dimension_is_36():
    assert len(B.FEATURE_NAMES) + len(M.SHIPPED_24) == 36


def _bars(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    return {
        "open": c * (1 + rng.normal(0, 0.001, n)),
        "high": c * (1 + abs(rng.normal(0, 0.002, n))),
        "low": c * (1 - abs(rng.normal(0, 0.002, n))),
        "close": c,
        "volume": np.abs(rng.normal(1e6, 1e5, n)),
        "timestamp": (np.arange(n) * 1800 * 1000) + 1600000000000,
    }


def test_every_shipped_bars_factor_varies():
    """A factor that does not vary is a placeholder, and a search will use it."""
    bars = _bars()
    for nm in M.SHIPPED_24:
        if nm not in M.F:
            continue
        out = np.asarray(M.F[nm][0](bars), dtype=np.float64)
        assert out.shape == (len(bars["close"]),), nm
        assert np.isfinite(out).all(), nm
        assert float(np.ptp(out)) > 1e-9, nm
        assert float(np.std(out)) > 1e-9, nm


def test_indices_0_to_11_are_unchanged():
    """Every artifact recorded against the old 12 must stay decodable."""
    bars = _bars()
    old = B.causal_features(bars)
    new = M.shipped_bars_only(bars)
    assert new.shape[0] == 12 + sum(1 for n in M.SHIPPED_24 if n in M.F)
    assert np.array_equal(old, new[:12])


def test_split_between_bars_and_funding():
    nbars = sum(1 for n in M.SHIPPED_24 if n in M.F)
    nfund = sum(1 for n in M.SHIPPED_24 if n in M.FUNDING_FACTORS)
    assert nbars + nfund == 24
    assert nfund == 4, "the funding block is FUND_LAST, FUND_COUNTDOWN, FUND_X_MOM, FUND_FLIP_7"


@pytest.mark.parametrize("nm", [n for n in M.SHIPPED_24 if n in M.F])
def test_truncation_does_not_move_the_factor(nm):
    """Causal factors must not change when the future is removed."""
    from research.factor_corr_harness import causal_violation
    assert not causal_violation(M.F[nm][0], _bars(n=2500))
