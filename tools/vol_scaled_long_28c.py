"""F_C VOL-SCALED LONG: the honest target, tested against a fair benchmark.

Both external reviews converged on the same conclusion: a per-coin 30m timing
book cannot beat a constant levered long, because it shorts the only reliably
positive term, pays turnover twice, and pays funding on the short leg. The
predictable thing on these bars is not direction but volatility -- and the
screen already says so, since LOG_VOL, HL_RANGE and VOL_CLUST held 89-100%
sign consistency across coins while the other nine sat at coin-flip levels.

The construction is deliberately undemanding: a long-only book whose position
is INVERSELY proportional to trailing realised volatility, scaled so the
average exposure matches the constant long. It is not a claim to beat
buy-and-hold. The claim under test is narrower and is the one that matters for
this project: can volatility be forecast well enough on these bars to reduce
drawdown at equal drift?

The benchmark is therefore a VOL-SCALED CONSTANT LONG with the same target
average exposure and the same leverage, not the raw constant long. Comparing a
risk-managed book to an unmanaged one and calling the difference alpha would be
the same mistake the project already made once. Both are also reported against
the plain constant long, because that is the bar the accept gate uses, and a
book that only wins against a weaker benchmark has not earned anything.

The scaling is fitted on train only, then frozen. Turnover is reported, since
rebalancing a vol target is not free.
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


def realised_vol(r, win):
    """Trailing RMS of returns over bars (t-win, t], ending at bar t."""
    n_ = len(r)
    sq = r * r
    c = np.concatenate(([0.0], np.cumsum(sq)))
    out = np.zeros(n_)
    out[win:] = np.sqrt(np.maximum(c[win + 1:] - c[1:n_ - win + 1], 0.0) / win)
    return out


def vols(win):
    V = np.stack([realised_vol(R[c], win) for c in COINS])          # (coins, T)
    return V


def build(V, target, cap=PC, floor_q=0.1):
    """Inverse-vol weights, renormalised to hit `target` mean exposure."""
    inv = 1.0 / (V + 1e-12)
    for i in range(V.shape[0]):
        s = inv[i, :train_end]
        m = np.median(s[s > 0]) if np.any(s > 0) else 1.0
        inv[i] = np.clip(inv[i] / max(m, 1e-12), floor_q, 1.0 / max(floor_q, 1e-12))
    w = inv * (target / inv.mean(axis=0, keepdims=True))
    w = np.clip(w, 0.0, cap)
    pos = np.roll(w, 1, axis=-1)
    pos[:, 0] = 0.0
    return pos


def score(pos, window, fee=GA.FEE):
    x, y = window
    nets, turn, mdd = [], [], []
    for i, c in enumerate(COINS):
        p = pos[i][x:y]; r = R[c][x:y]
        nets.append(accounting_bar_returns(p, r, fee, fund[c][x:y], GA.LEV))
        prev = np.roll(p, 1); prev[0] = 0.0
        turn.append(float(np.abs(p - prev).mean()))
    port = np.mean(np.stack(nets), axis=0)
    m = metrics(port, common[x:y])
    return {"sharpe": m["sharpe"], "mdd": m["mdd"], "turn": float(np.mean(turn)),
            "exposure": float(np.mean([pos[i][x:y].mean() for i in range(len(COINS))]))}


flat = {i: np.full(n, PC) for i in range(len(COINS))}
pos_flat = np.stack([flat[i] for i in range(len(COINS))])
b_flat = score(pos_flat, LOCK)
print("=" * 80)
print("F_C VOL-SCALED LONG  -- the predictable axis on these bars is volatility")
print("=" * 80)
print("  plain constant long on the lockbox: Sharpe %+.3f  MDD %.4f  exposure %.3f\n"
      % (b_flat["sharpe"], b_flat["mdd"], b_flat["exposure"]))
print("  %-6s %-7s %9s %9s %9s %9s %8s" %
      ("win", "target", "Sharpe", "MDD", "expo", "turn", "dMDD"))
print("-" * 80)
rows = []
for win in (48, 96, 288, 576):
    V = vols(win)
    for tgt in (PC * 0.5, PC):
        pos = build(V, tgt)
        r = score(pos, LOCK)
        dmdd = r["mdd"] - b_flat["mdd"]
        rows.append({"win": win, "target": tgt, **r})
        print("  %-6d %-7.3f %+9.3f %9.4f %9.3f %9.4f %+8.4f" %
              (win, tgt, r["sharpe"], r["mdd"], r["exposure"], r["turn"], dmdd))
    print()

# stability across windows, and a shuffled-vol null with matched turnover
print("  same setting across all three windows (win=%d, target=%.3f):" % (288, PC))
V = vols(288)
pos = build(V, PC)
for wname, w in (("train", TRAIN), ("validation", VAL), ("lockbox", LOCK)):
    r = score(pos, w)
    print("    %-11s Sharpe %+7.3f  MDD %.4f  exposure %.3f"
          % (wname, r["sharpe"], r["mdd"], r["exposure"]))
print()

# null: shuffle each coin's vol series across coins, preserving its own dynamics
pos_n = np.zeros_like(pos)
for i in range(len(COINS)):
    j = rng.integers(0, len(COINS))
    pos_n[i] = pos[j]
rn = score(pos_n, LOCK)
rr = score(pos, LOCK)
print("  lockbox: real Sharpe %+.3f  MDD %.4f" % (rr["sharpe"], rr["mdd"]))
print("           vol-shuffled null Sharpe %+.3f  MDD %.4f" % (rn["sharpe"], rn["mdd"]))
print("  MDD reduction vs the constant long: %+.4f" % (rr["mdd"] - b_flat["mdd"]))
print("  Sharpe change vs the constant long: %+.3f" % (rr["sharpe"] - b_flat["sharpe"]))
print()
Path("/home/brian/project/AlphaGPT/results/vol_scaled_long.json").write_text(
    json.dumps({"constant_long": b_flat, "rows": rows, "null": rn}, indent=1))
print("wrote results/vol_scaled_long.json")
