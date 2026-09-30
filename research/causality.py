"""Causality assertion for signals, so a look-ahead fails loudly.

A cross-sectional momentum book in this project reported Sharpe +16.6 with a
95% CI of [14.8, 20.7] and P(Sharpe <= 0) = 0.0000. It was look-ahead: the
signal was a trailing window ending at bar t, and the position was applied to
r[t] without the one-bar lag this repo uses everywhere else, so the position
read its own outcome. Every statistical check passed, because the artifact was
present in every resampled block and so survived resampling.

The framework guards against missing data, wrong accounting and selection
bias. It did not guard against causality of the signal, because that was a
convention followed by hand in each file. This module makes it a test.

THE TEST
--------
For a split point k, replace every return at or after k with an independent
draw and rebuild the signal. If the signal is a causal function of bars up to
t, the rebuilt signal must equal the original at every bar BEFORE k, because
none of its inputs changed. A signal that peeks differs at some bar before k.

The first peeking bar is found by binary search over k, so the check costs
O(log T) rebuilds rather than O(T).

The replacement draw is random rather than zero, because zeroing the tail can
make a normalisation constant degenerate and produce a spurious failure.

SCOPE, STATED PLAINLY
---------------------
`assert_causal` tests whether a signal is a causal function of the array you
pass it. Two consequences follow, and both have bitten already:

  - It does NOT catch a POSITION built from a causal signal but applied
    without the one-bar lag, because such a position is still a causal
    function of the signal. Use `assert_position_lagged` for that.
  - It says nothing about a signal built from something OTHER than returns,
    such as a funding charge. A carry signal that reads the settlement it is
    scored against is depending on the future OF ITS OWN INPUT, and this
    function is the right test for it, but only if you pass the funding array
    as `returns`. The accounting alignment is a separate question answered by
    `assert_position_lagged`, not by this.

The terminal bar is excluded. It has no future, so it cannot be wrong in this
sense, and including it would reject every signal.
"""
from __future__ import annotations
import numpy as np

__all__ = ["assert_causal", "assert_position_lagged", "first_peek_bar"]


def _suffix_draw(r, k, rng):
    """A copy of r with bars >= k replaced by an independent draw."""
    alt = r.copy()
    tail = alt[..., k:]
    if tail.size:
        scale = float(np.std(r)) if np.std(r) > 0 else 1.0
        alt[..., k:] = rng.normal(0.0, scale, size=tail.shape)
    return alt


def first_peek_bar(signal, rebuild, returns, *, seed=0, warmup=0):
    """Smallest bar index whose signal value depends on a return at or after it.

    Returns None when the signal is causal. Binary search over the split point:
    the predicate "the signal before k is unchanged when the tail is replaced"
    is monotone in k, so the first violating k is found in O(log T) rebuilds.
    """
    s = np.asarray(signal, dtype=np.float64)
    r = np.asarray(returns, dtype=np.float64)
    T = s.shape[-1]
    if T < 4:
        return None
    rng = np.random.default_rng(seed)
    # The final bar has no future, so its value is unconstrained by
    # construction. Searching to T-1 would flag every signal, including a
    # strictly causal one, at the last index. The search therefore stops at
    # T-2: the last bar that could have been built from a bar that was
    # wrongly included.
    lo, hi = max(1, warmup + 1), T - 2
    if hi < lo:
        return None
    if _intact_at(s, rebuild, r, hi, rng):
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        if _intact_at(s, rebuild, r, mid, rng):
            lo = mid + 1
        else:
            hi = mid
    return lo


def _intact_at(s, rebuild, r, k, rng):
    """True when the signal up to AND INCLUDING bar k ignores the tail.

    The head must include index k. A trailing window that ends at bar t uses
    r[t]; if the head stopped at k-1 the one-bar overlap at the boundary would
    be invisible and the check would pass a signal that peeks by exactly one
    bar -- which is the bug it exists to catch.
    """
    s_alt = np.asarray(rebuild(_suffix_draw(r, k, rng)), dtype=np.float64)
    if s_alt.shape != s.shape:
        raise AssertionError(
            f"rebuild returned shape {s_alt.shape}, expected {s.shape}. The "
            f"rebuild must depend on nothing but the returns passed to it.")
    head_slice = slice(0, k + 1)
    head = np.abs(s[..., head_slice] - s_alt[..., head_slice])
    live = np.any(np.abs(s[..., head_slice]) > 0, axis=tuple(range(s.ndim - 1))) \
        if s.ndim > 1 else np.abs(s[..., head_slice]) > 0
    if not live.any():
        return True
    return not bool((head > 0).any())


def assert_causal(signal, rebuild, returns, *, name="signal", warmup=0, seed=0):
    """Assert `rebuild` is a causal function of `returns`.

    Raises AssertionError naming the first bar whose value moved. Returns a
    dict on success.
    """
    s = np.asarray(signal, dtype=np.float64)
    r = np.asarray(returns, dtype=np.float64)
    if s.shape[-1] != r.shape[-1]:
        raise ValueError(f"length mismatch: signal {s.shape[-1]} vs returns {r.shape[-1]}")
    if s.shape[-1] < 4:
        raise AssertionError(f"{name}: too few bars ({s.shape[-1]}) to test causality")
    k = first_peek_bar(s, rebuild, r, seed=seed, warmup=warmup)
    if k is not None:
        raise AssertionError(
            f"{name} is not causal. Its value at bar {k} depends on a return at "
            f"or after bar {k}. A rolling window ending at bar t peeks by "
            f"construction. Lag the POSITION by one bar before applying it to "
            f"returns -- that is the convention in research/ga_28c_30m_3y.py "
            f"and every other position path in this repo.")
    return {"causal": True, "bars": int(s.shape[-1])}


def assert_position_lagged(raw_position, applied_position, *, name="position",
                           atol=1e-12):
    """Assert the applied position equals the raw one lagged by exactly one bar.

    The repo builds `raw` from a signal, then applies `np.roll(raw, 1)` with
    `applied[0] = 0` before any return, fee or funding is charged. Applying the
    unlagged series lets the position read the bar it is scored on, which is the
    defect that produced a reported Sharpe of +16.6 in this project.
    """
    raw = np.asarray(raw_position, dtype=np.float64)
    app = np.asarray(applied_position, dtype=np.float64)
    if raw.shape != app.shape:
        raise ValueError(f"{name}: shape mismatch {raw.shape} vs {app.shape}")
    expect = np.roll(raw, 1, axis=-1)
    zero = [slice(None)] * expect.ndim
    zero[-1] = 0
    expect[tuple(zero)] = 0.0
    d = np.abs(expect - app)
    if float(d.max()) > atol:
        k = int(np.argmax(d.max(axis=tuple(range(app.ndim - 1))) if app.ndim > 1
                          else d.max(axis=0)))
        raise AssertionError(
            f"{name} is not lagged. The applied series differs from "
            f"np.roll(raw, 1) by up to {float(d.max()):.6g} (worst at bar {k}). "
            f"The framework lags every position by one bar before charging "
            f"returns, fees or funding; an unlagged position is scored on the "
            f"bar it already read.")
    return {"lagged": True, "max_abs_diff": float(d.max()), "bars": int(raw.shape[-1])}
