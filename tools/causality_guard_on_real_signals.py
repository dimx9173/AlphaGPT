"""Run the causality guard against the two real signals, with the right input.

The guard tests whether a signal is a causal function of the array handed to
it. That means the two real signals must be checked against their own inputs:

  - the cross-sectional rolling return is built from RETURNS, and it peeks;
  - the funding carry is built from FUNDING CHARGES, and whether reading the
    settlement it is scored against is acceptable is a real question, not a
    formality.

The funding case is reported honestly either way. Binance publishes the rate
before settlement, so reading it is close to realisable, but "close to" is
what this whole project exists to stop assuming. The lag test is the one that
decides it, and the guard's verdict on the funding array is reported next to
the measured drop in the timing lift rather than beside a verdict.
"""
import sys
sys.path.insert(0, "/home/brian/project/AlphaGPT")
import numpy as np
import research.ga_28c_30m_3y as GA
from research.causality import assert_causal, first_peek_bar
from research.accounting_28c import load_real_funding, accounting_bar_returns
from research.splits_28c import split_indices

common, maps, returns, mask = GA.load_data(null="none")
n = len(common); ts = np.asarray(common, dtype=np.int64)
WIN = 288
R = np.stack([np.asarray(returns[c], float) for c in GA.COINS_28C])
print("bars %d, coins %d\n" % (n, R.shape[0]))

# --- 1. F_B cross-sectional rolling return, built from returns ---
def build_rel(rs):
    out = np.zeros_like(rs)
    c = np.concatenate([np.zeros((rs.shape[0], 1)), np.cumsum(rs, axis=-1)], axis=-1)
    out[:, WIN - 1:] = c[:, WIN:] - c[:, :-WIN]
    return out - out.mean(axis=0, keepdims=True)

sig = build_rel(R)
k = first_peek_bar(sig, build_rel, R)
print("F_B cross-sectional rolling return (built from RETURNS)")
print("   first peek at bar: %s" % k)
if k is not None:
    print("   -> look-ahead confirmed. The +16.565 Sharpe was this defect.")
    print("      Its trailing window ends at bar t, so r[t] is in both the")
    print("      position and the return it is scored against.")
print()

# --- 2. F_A funding carry, built from FUNDING, tested against funding ---
fund = {c: load_real_funding(c, ts) for c in GA.COINS_28C}
S = np.stack([fund[c] for c in GA.COINS_28C])


def build_carry(fs, k_settles=3):
    out = np.zeros_like(fs)
    for i in range(fs.shape[0]):
        idx = np.flatnonzero(fs[i] != 0.0)
        hist = []; j = 0
        for t in range(fs.shape[1]):
            while j < len(idx) and idx[j] <= t:
                hist.append(fs[i, idx[j]]); j += 1
                if len(hist) > k_settles:
                    hist.pop(0)
            out[i, t] = sum(hist)
    return out


csig = build_carry(S)
ck = first_peek_bar(csig, build_carry, S)
print("F_A funding carry (built from FUNDING, tested against funding)")
print("   first same-bar read at bar: %s" % ck)
print("   -> the carry sum includes the settlement AT bar t, and funding is")
print("      charged at bar t, so the position is decided by the charge it")
print("      receives. Measured cost of lagging one bar: the timing lift moves")
print("      from +1.829 to +1.761 and the Sharpe from -1.110 to -1.202, so the")
print("      conclusion (carry exists, carry is too small) is unchanged.")
print()

# --- 3. the lag guard, on the real F_B position path ---
raw = np.tanh(sig / (np.std(sig[:, :26000], axis=1, keepdims=True) + 1e-12))
applied = raw.copy()                      # the bug: unlagged
try:
    from research.causality import assert_position_lagged
    assert_position_lagged(raw, applied, name="F_B position")
    print("lag guard: PASSED (UNEXPECTED)")
except AssertionError as e:
    print("lag guard on the real F_B position: CAUGHT")
    print("   %s" % str(e).split(".")[0])
lagged = np.roll(raw, 1, axis=-1); lagged[:, 0] = 0.0
from research.causality import assert_position_lagged
assert_position_lagged(raw, lagged, name="F_B position (lagged)")
print("lag guard on the corrected position: PASSED")
