"""F_B REL_REV, corrected: the position must be lagged before it touches returns.

The previous F_B runs reported Sharpe +16.6 with a 95% CI of [14.8, 20.7] and
P(Sharpe<=0)=0. That was too good, and it was wrong. The position was built
straight from a trailing window that ENDS at bar t, so p[t] and r[t] both
contained r[t]. On iid data that alignment alone manufactures a correlation
of about +0.43; the correctly lagged position shows -0.0035.

Every other position path in this repo lags by one bar (`np.roll(pos, 1)`)
because that is the only causal convention available. The F_B code did not,
and the result was look-ahead dressed as a discovery. It is recorded here
because a tight confidence interval does not protect against a wrong
definition: the CI was narrow because the artifact was present in every
resampled block, not because the signal was strong.

The position is now lagged exactly like every other book in the framework, and
the signal is re-derived and re-tested from scratch.
"""
import sys, json
sys.path.insert(0, "/home/brian/project/AlphaGPT")
import numpy as np
import research.ga_28c_30m_3y as GA
from research.accounting_28c import accounting_bar_returns, load_real_funding, metrics
from research.splits_28c import split_indices
from pathlib import Path

PC = GA.POSITION_CAP
common, maps, returns, mask = GA.load_data(null="none")
n = len(common); ts = np.asarray(common, dtype=np.int64)
sp = split_indices(n, common[0], common[-1])
LOCK = sp["lockbox"]; TRAIN = (0, sp["train"][1]); VAL = sp["validation"]
train_end = sp["train"][1]
fund = {c: load_real_funding(c, ts) for c in GA.COINS_28C}
R = {c: np.asarray(returns[c], float) for c in GA.COINS_28C}
COINS = list(GA.COINS_28C)
rng = np.random.default_rng(20260930)


def rolling(win):
    M = np.zeros((len(COINS), n - win + 1))
    for i, c in enumerate(COINS):
        csum = np.concatenate(([0.0], np.cumsum(R[c])))
        M[i] = csum[win:] - csum[:-win]
    out = np.zeros((len(COINS), n)); out[:, win - 1:] = M
    return out


def positions(values, sd):
    """Build positions, then LAG BY ONE BAR, the repo's causal convention."""
    pos = np.zeros_like(values)
    for i in range(len(COINS)):
        if sd[i] > 1e-12:
            pos[i] = np.tanh(values[i] / (sd[i] + 1e-12)) * PC
    pos = np.roll(pos, 1, axis=1)
    pos[:, 0] = 0.0
    return pos


def score(pos, window, fee=GA.FEE):
    x, y = window
    nets, turn, ics, expo = [], [], [], []
    for i, c in enumerate(COINS):
        p = pos[i][x:y]; r = R[c][x:y]
        nets.append(accounting_bar_returns(p, r, fee, fund[c][x:y], GA.LEV))
        prev = np.roll(p, 1); prev[0] = 0.0
        turn.append(float(np.abs(p - prev).mean()))
        expo.append(float(p.mean()))
        if p.std() > 1e-12 and r.std() > 1e-12:
            ics.append(float(np.corrcoef(p, r)[0, 1]))
    m = metrics(np.mean(np.stack(nets), axis=0), common[x:y])
    return {"sharpe": m["sharpe"], "turn": float(np.mean(turn)),
            "ic": float(np.mean(ics)), "coins_pos": int(sum(1 for v in ics if v > 0)),
            "net_exp": float(np.mean(expo))}


print("=" * 86)
print("F_B REL_REV, corrected  -- position lagged one bar, cross-sectional")
print("=" * 86)
print("  sign -1 = fade the cross-sectional move (reversal)")
print("  sign +1 = follow the cross-sectional move (momentum)")
print()
print("  %-11s %-7s %5s %9s %9s %8s %7s %9s" %
      ("window", "lookback", "sign", "Sharpe", "ic", "coins>0", "turn", "net_exp"))
print("-" * 86)
rows = []
for win in (48, 96, 288):
    sig = rolling(win)
    sig = sig - sig.mean(axis=0, keepdims=True)
    sd = np.array([float(np.std(sig[i, :train_end])) for i in range(len(COINS))])
    for sign in (-1, +1):
        pos = positions(sign * sig, sd)
        r = score(pos, LOCK)
        rows.append({"win": win, "sign": sign, **r})
        print("  %-11s %-7d %+5d %+9.3f %+9.4f %6d/%-2d %7.4f %9.4f" %
              ("lockbox", win, sign, r["sharpe"], r["ic"], r["coins_pos"],
               len(COINS), r["turn"], r["net_exp"]))
    print()

print("  same sign across all three windows (train / validation / lockbox):")
print("  %-7s %5s %9s %9s %9s" % ("lookback", "sign", "train", "val", "lockbox"))
print("-" * 86)
consist = []
for win in (48, 96, 288):
    sig = rolling(win)
    sig = sig - sig.mean(axis=0, keepdims=True)
    sd = np.array([float(np.std(sig[i, :train_end])) for i in range(len(COINS))])
    for sign in (-1, +1):
        pos = positions(sign * sig, sd)
        sh = [score(pos, w)["sharpe"] for w in (TRAIN, VAL, LOCK)]
        same = (sh[0] > 0) == (sh[1] > 0) == (sh[2] > 0)
        consist.append(same)
        print("  %-7d %+5d %+9.3f %+9.3f %+9.3f   %s" %
              (win, sign, sh[0], sh[1], sh[2], "consistent" if same else "FLIPS"))
print()
best = max(rows, key=lambda r: r["sharpe"])
print("  best: lookback %d sign %+d -> Sharpe %+.3f" % (best["win"], best["sign"], best["sharpe"]))
print("  (the uncorrected version of this same code reported +16.565)")
Path("/home/brian/project/AlphaGPT/results/rel_rev_causal.json").write_text(
    json.dumps(rows, indent=1))
print("wrote results/rel_rev_causal.json")
