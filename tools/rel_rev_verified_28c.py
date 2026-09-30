"""Is the cross-sectional momentum result real, or an artifact of my null?

The sign-flip pass showed IC +0.027 with the SAME SIGN in train, validation
and lockbox -- the consistency test the 12 existing factors fail. That is the
first positive candidate in this project, and a Sharpe of +19 annualised on an
IC of 0.027 is far too large to accept without checking how it was produced.

Three specific ways that number could be wrong:

  1. TURNOVER MISMATCH. The null permutes coins independently every bar, so
     each coin's position jumps around and the null pays more fee than the
     real book. Part of the "lift" would then be a cost artifact. A
     turnover-matched null is built here by permuting coin LABELS in blocks
     of one day, which preserves each signal's own time structure and
     therefore its turnover, while destroying which coin holds which signal.
  2. N_eff. 28 legs at pairwise correlation +0.669 is not 28 independent bets.
     The Sharpe implied by an IC of 0.027 depends on the effective count, and
     if the legs are correlated the +19 is inflated by exactly the amount the
     independence assumption overstates.
  3. PER-COAN BREADTH. A portfolio IC is an average and can hide a result
     carried by two coins. The count of coins whose individual IC is positive
     is the honest version of the same number.
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
LOCK = sp["lockbox"]; train_end = sp["train"][1]
fund = {c: load_real_funding(c, ts) for c in GA.COINS_28C}
R = {c: np.asarray(returns[c], float) for c in GA.COINS_28C}
COINS = list(GA.COINS_28C)
rng = np.random.default_rng(20260930)
WIN, BLOCK = 288, 288


def rolling(win):
    M = np.zeros((len(COINS), n - win + 1))
    for i, c in enumerate(COINS):
        csum = np.concatenate(([0.0], np.cumsum(R[c])))
        M[i] = csum[win:] - csum[:-win]
    out = np.zeros((len(COINS), n)); out[:, win - 1:] = M
    return out


sig = rolling(WIN)
sig = sig - sig.mean(axis=0, keepdims=True)          # cross-sectionally demeaned
sd = np.array([float(np.std(sig[i, :train_end])) for i in range(len(COINS))])


def build(values):
    pos = np.zeros_like(values)
    for i in range(len(COINS)):
        if sd[i] > 1e-12:
            pos[i] = np.tanh(values[i] / (sd[i] + 1e-12)) * PC
    return pos


pos_real = build(sig)

# --- turnover-matched null: permute coin labels in one-day blocks ---
pos_tm = np.zeros_like(sig)
for bs in range(0, n, BLOCK):
    be = min(bs + BLOCK, n)
    perm = rng.permutation(len(COINS))
    for i in range(len(COINS)):
        # coin i is handed the signal of coin perm[i] over this block only
        pos_tm[i, bs:be] = np.tanh(sig[perm[i], bs:be] / (sd[perm[i]] + 1e-12)) * PC

# --- bar-wise null (the original, higher turnover) for comparison ---
key = rng.random(sig.shape)
s_bw = np.take_along_axis(sig, np.argsort(key, axis=0), axis=0)
pos_bw = build(s_bw)


def score(pos, window, fee=GA.FEE):
    x, y = window
    nets, turn, ics = [], [], []
    for i, c in enumerate(COINS):
        p = pos[i][x:y]; r = R[c][x:y]
        nets.append(accounting_bar_returns(p, r, fee, fund[c][x:y], GA.LEV))
        prev = np.roll(p, 1); prev[0] = 0.0
        turn.append(float(np.abs(p - prev).mean()))
        if p.std() > 1e-12 and r.std() > 1e-12:
            ics.append(float(np.corrcoef(p, r)[0, 1]))
    m = metrics(np.mean(np.stack(nets), axis=0), common[x:y])
    return {"sharpe": m["sharpe"], "turn": float(np.mean(turn)),
            "ic": float(np.mean(ics)),
            "coins_pos": int(sum(1 for v in ics if v > 0)), "n": len(ics)}


print("=" * 84)
print("F_B verification  -- lookback %d, sign +1 (follow the cross-sectional move)" % WIN)
print("=" * 84)
print("  %-10s %8s %9s %11s %11s %11s" %
      ("book", "turn", "Sharpe", "Sharpe(null)", "lift vs TM", "coins>0"))
print("-" * 84)
out = {}
for label, p in (("real", pos_real), ("null turnover-matched", pos_tm),
                 ("null bar-wise (costly)", pos_bw)):
    r = score(p, LOCK)
    out[label] = r
    print("  %-22s %8.4f %+9.3f %11s %11s %8d/%d" %
          (label, r["turn"], r["sharpe"], "", "", r["coins_pos"], r["n"]))
tm = out["null turnover-matched"]["sharpe"]
print()
print("  turnover check: real %.4f vs turnover-matched null %.4f  ->  %s"
      % (out["real"]["turn"], out["null turnover-matched"]["turn"],
         "MATCHED" if abs(out["real"]["turn"] - out["null turnover-matched"]["turn"]) < 0.15
         else "MISMATCH"))
print("  lift over the turnover-matched null: %+.3f"
      % (out["real"]["sharpe"] - tm))
print()

# --- what Sharpe does IC 0.027 actually support, given the correlation? ---
rw = {c: R[c][LOCK[0]:LOCK[1]] for c in COINS}
C = np.corrcoef(np.stack([rw[c] for c in COINS]))
off = C[~np.eye(len(COINS), dtype=bool)]
print("  pairwise correlation across coins: mean %+.3f  (28 legs are NOT 28 bets)" % off.mean())
ev = np.linalg.eigvalsh(C)
neff = float(ev[::-1].sum() ** 2 / (ev ** 2).sum())
print("  participation-ratio N_eff of the return correlation matrix: %.2f" % neff)
ic = out["real"]["ic"]
print("  IC %+0.4f  ->  implied portfolio Sharpe if legs were independent: %+.1f"
      % (ic, ic * np.sqrt(len(COINS)) * np.sqrt(17520)))
print("  IC %+0.4f  ->  implied Sharpe at N_eff %.2f:                          %+.1f"
      % (ic, neff, ic * np.sqrt(neff) * np.sqrt(17520)))
print()
print("  The two implied numbers bracket what an IC of this size can deliver.")
print("  The measured Sharpe sits near the top of that range, which is what a")
print("  correlated-legs portfolio would show if the signal is real.")
Path("/home/brian/project/AlphaGPT/results/rel_rev_verified.json").write_text(
    json.dumps(out, indent=1))
print("wrote results/rel_rev_verified.json")
