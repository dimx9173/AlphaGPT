"""The causality guard must catch the exact bug it was written for."""
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.causality import assert_causal, assert_position_lagged  # noqa: E402

N = 3000
RNG = np.random.default_rng(7)
RET = RNG.normal(0.0, 0.01, N)


def _rolling_win(win):
    """A trailing sum ending at bar t -- the shape that peeks."""
    def build(r):
        c = np.concatenate(([0.0], np.cumsum(r)))
        out = np.zeros(len(r))
        out[win - 1:] = c[win:] - c[:-win]
        return out
    return build


def test_causal_signal_passes():
    """A signal built from PAST bars only is unchanged when the future moves."""
    win = 288
    def build(r):
        # out[t] = sum(r[t-win : t]) -- the window ENDS at t-1, strictly past.
        c = np.concatenate(([0.0], np.cumsum(r)))
        n = len(r)
        out = np.zeros(n)
        out[win:] = c[win:n] - c[:n - win]
        return out
    out = assert_causal(build(RET), build, RET, name="past_only")
    assert out["causal"]


def test_trailing_window_ending_at_t_is_caught():
    """A rolling window ending at bar t peeks; the guard must reject it."""
    build = _rolling_win(288)
    with pytest.raises(AssertionError, match="not causal"):
        assert_causal(build(RET), build, RET, name="trailing_window")


def test_future_dependent_signal_is_caught():
    def build(r):
        return np.roll(r, -1)                 # explicitly the future bar
    with pytest.raises(AssertionError, match="not causal"):
        assert_causal(build(RET), build, RET, name="next_bar")


def test_position_lag_guard_catches_unlagged():
    raw = np.tanh(np.arange(N) % 7 - 3)
    applied = raw.copy()                      # NOT lagged
    with pytest.raises(AssertionError, match="not lagged"):
        assert_position_lagged(raw, applied)


def test_position_lag_guard_accepts_lagged():
    raw = np.tanh(np.arange(N) % 7 - 3)
    applied = np.roll(raw, 1)
    applied[0] = 0.0
    out = assert_position_lagged(raw, applied)
    assert out["lagged"]


def test_guard_names_the_offending_bar():
    """The failure message must point at a bar, not just say 'failed'."""
    build = _rolling_win(50)
    with pytest.raises(AssertionError) as ei:
        assert_causal(build(RET), build, RET, name="trailing_window")
    assert "bar" in str(ei.value)
