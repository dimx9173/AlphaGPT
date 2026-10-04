"""The multi-timeframe block: 36 factors plus 5 measured coarse-grid survivors."""
import numpy as np
import pytest

import research.causal_36f as C
import research.causal_mtf_28c as M
import research.grammar_reduced_36c as G


def _bars(n=4000, seed=3):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    return {
        "open": c * (1 + rng.normal(0, 0.001, n)),
        "high": c * (1 + abs(rng.normal(0, 0.002, n))),
        "low": c * (1 - abs(rng.normal(0, 0.002, n))),
        "close": c,
        "volume": np.abs(rng.normal(1e6, 1e5, n)),
        "timestamp": (np.arange(n) * 3600 * 1000) + 1726228800000,
    }


def test_ships_exactly_five():
    assert len(M.MTF_FACTORS) == 5


def test_cut_factors_are_documented():
    """The six that were measured and rejected must stay named with their reason.

    A rejected factor that is simply deleted is a factor that will be
    re-proposed next time somebody reads the round-3 shortlist.
    """
    src = (M.__file__)
    text = open(src).read()
    for cut in ("4h_RANGEMEAN", "4h_VWAPDEV", "4h_DOLLARIMBAL",
                "4h_TREND_TSTAT", "4h_DONCH", "4h_DD"):
        assert cut not in M.MTF_FACTORS
        assert cut in text, cut


def test_coarse_volume_sums_not_samples():
    """A coarse volume that took the last fine bar would be a mislabelled 1h factor."""
    b = _bars(400)
    cb = M.coarse_bars(b, "4h")
    assert np.isclose(cb["volume"][:3].sum(), b["volume"][:12].sum())


def test_coarse_ohlc_is_ordered():
    b = _bars(2000)
    cb = M.coarse_bars(b, "4h")
    assert (cb["high"] >= cb["low"]).all()
    assert (cb["high"] >= cb["close"]).all()
    assert (cb["low"] <= cb["close"]).all()


def test_alignment_never_looks_forward():
    """A primary bar inside a coarse bar must see the PREVIOUS coarse value.

    This is the property that makes the whole design causal rather than
    approximately causal.
    """
    coarse_close = np.array([100.0, 200.0, 300.0])
    values = np.array([1.0, 2.0, 3.0])
    primary = np.array([99.0, 100.0, 150.0, 200.0, 250.0, 300.0])
    out = M.align_to_primary(values, coarse_close, primary)
    assert out[0] == 0.0            # before the first close: no information
    assert out[1] == 1.0            # at the first close
    assert out[2] == 1.0            # inside the second coarse bar: still the first
    assert out[3] == 2.0
    assert out[4] == 2.0
    assert out[5] == 3.0


def test_coarse_factors_are_causal_under_truncation():
    """Truncating the series must not move any coarse factor value."""
    b = _bars(4000)
    full = M.coarse_bars(b, "4h")
    part = M.coarse_bars({k: v[:3000] for k, v in b.items()}, "4h")
    shared = set(part["timestamp"]) & set(full["timestamp"])
    assert len(shared) > 100
    fi = {t: i for i, t in enumerate(full["timestamp"])}
    pi = {t: i for i, t in enumerate(part["timestamp"])}
    for name, factory in M.COARSE_FACTORS.items():
        a = factory(full)
        c = factory(part)
        for t in sorted(shared):
            assert np.isclose(a[fi[t]], c[pi[t]], rtol=1e-9, atol=1e-12), name


def test_grammar_binds_at_41():
    old = G.FEATURE_COUNT
    try:
        G.with_feature_count(41)
        assert G.FEATURE_COUNT == 41
        assert min(G.REDUCED_OPS) == 41
        assert G.REDUCED_TOKENS == 41 + len(G.REDUCED_OPS)
        # At 41 features token 41 is the first operator, ADD with arity 2, so a
        # valid program needs two factors under it. A bare ADD underflows and a
        # lone factor is not a program.
        assert G.validate_reduced((0, 1, 41))       # F0 F1 ADD
        assert not G.validate_reduced((0, 41))      # ADD with one arg underflows
        assert not G.validate_reduced((41,))        # a bare operator underflows
        assert G.validate_reduced((0,))             # a lone factor is a program
    finally:
        G.with_feature_count(old)
