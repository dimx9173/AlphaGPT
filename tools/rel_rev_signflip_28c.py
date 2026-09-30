"""F_B follow-up: separate signal from cost, and test both signs.

The first F_B pass returned Sharpe -21..-44 with a NEGATIVE IC of -0.17, which
means the book as built (fade the cross-sectional move) is anti-predictive:
if that holds, the cross-section is trending, not reverting, and the factor
should be sign-flipped. But a Sharpe of -44 is also the magnitude a large
turnover cost produces, so the two explanations cannot be separated from that
run alone.

This separates them three ways:
  1. fee = 0, so the signal stands alone with no cost to hide behind or to
     manufacture the effect;
  2. both signs, because a negative IC under one sign is evidence about the
     other sign;
  3. all three windows, because a real cross-sectional effect should not
     appear only in the lockbox.

The ranking-null is repeated for each, so "beats the null" is still judged
against a book with identical turnover and identical market exposure.
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


def demean(M):
    return M - M.mean(axis=0, keepdims=True)


def pos_from(sig, sign):
    pos = np.zeros_like(sig)
    for i in range(len(COINS)):
        sd = float(np.std(sig[i, :train_end]))
        if sd > 1e-12:
            pos[i] = np.tanh(sign * sig[i] / (sd + 1e-12)) * PC
    return pos


def null_pos(sig, sign):
    key = rng.random(sig.shape)
    s = np.take_along_axis(sig, np.argsort(key, axis=0), axis=0)
    pos = np.zeros_like(s)
    for i in range(len(COINS)):
        sd = float(np.std(sig[i, :train_end]))
        if sd > 1e-12:
            pos[i] = np.tanh(sign * s[i] / (sd + 1e-12)) * PC
    return pos


def score(pos, window, fee):
    x, y = window
    nets, ic, net_exp = [], [], []
    for i, c in enumerate(COINS):
        p = pos[i][x:y]; r = R[c][x:y]
        nets.append(accounting_bar_returns(p, r, fee, fund[c][x:y], GA.LEV))
        net_exp.append(float(p.mean()))
        if p.std() > 1e-12 and r.std() > 1e-12:
            ic.append(float(np.corrcoef(p, r)[0, 1]))
    m = metrics(np.mean(np.stack(nets), axis=0), common[x:y])
    return {"sharpe": m["sharpe"], "ic": float(np.mean(ic)),
            "net_exposure": float(np.mean(net_exp))}


WIN = 288   # best lookback from the first pass
sig = demean(rolling(WIN))
print("=" * 88)
print("F_B follow-up  lookback %d bars (1 day), fee 0 to isolate the signal" % WIN)
print("=" * 88)
print("  sign -1 = fade the move (reversal, as originally built)")
print("  sign +1 = follow the move (cross-sectional momentum)")
print()
print("  %-8s %-6s %-4s %9s %9s %9s %9s %10s" %
      ("window", "sign", "fee", "Sharpe", "ic", "null", "lift", "net_exp"))
print("-" * 88)
rows = []
for wname, w in (("train", TRAIN), ("validation", VAL), ("lockbox", LOCK)):
    for sign in (-1, +1):
        pos = pos_from(sig, sign)
        npos = null_pos(sig, sign)
        for fee in (0.0, GA.FEE):
            r = score(pos, w, fee)
            nl = score(npos, w, fee)
            lift = r["sharpe"] - nl["sharpe"]
            rows.append({"window": wname, "sign": sign, "fee": fee, **r,
                         "null_sharpe": nl["sharpe"], "lift": lift})
            print("  %-8s %+6d %-4.4f %+9.3f %+9.4f %+9.3f %+9.3f %10.4f" %
                  (wname, sign, fee, r["sharpe"], r["ic"], nl["sharpe"], lift,
                   r["net_exposure"]))
    print()

print("Reading")
print("-" * 88)
z = [r for r in rows if r["fee"] == 0.0 and r["sign"] == 1]
if z:
    print("  sign +1, zero fee, IC by window:")
    for r in z:
        print("    %-11s IC %+0.4f  Sharpe %+8.3f  lift over null %+8.3f"
              % (r["window"], r["ic"], r["sharpe"], r["lift"]))
    consistent = all(r["ic"] > 0 for r in z)
    print("  IC same sign in all three windows: %s" % consistent)
print()
print("  A real cross-sectional effect has the same IC sign in train, validation")
print("  and lockbox. The existing 12 factors flip 5/12 -> 8/12 -> 5/12 and are")
print("  rejected for exactly that reason. Consistency is the test, not magnitude.")
Path("/home/brian/project/AlphaGPT/results/rel_rev_signflip.json").write_text(
    json.dumps(rows, indent=1))
print("wrote results/rel_rev_signflip.json")
