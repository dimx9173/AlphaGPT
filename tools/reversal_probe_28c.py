"""Is the strong mean reversion the VR test found exploitable after fees?

The variance-ratio test rejected iid on signed 30m returns by a wide margin
(VR(4)=0.24, VR(16)=0.06). Both reviewers argued the returns are directionless.
Those two claims cannot both be right. A VR far from 1 means real dependence.

The dependence could still be unexploitable, so this measures it directly:
  1. the raw autocorrelation function against its own iid null band, to see
     whether the reversion is bar-level structure or sub-bar microstructure
     bounce that averages out by the close;
  2. the net Sharpe of a plain contrarian book (fade the previous bar) after
     the same fee, funding and leverage the framework uses, against the same
     constant long.
"""
import sys, math
sys.path.insert(0, "/home/brian/project/AlphaGPT")
import numpy as np
import research.ga_28c_30m_3y as GA
from research.accounting_28c import accounting_bar_returns, load_real_funding, metrics
from research.splits_28c import split_indices

common, maps, returns, mask = GA.load_data(null="none")
n = len(common)
lb = split_indices(n, common[0], common[-1])["lockbox"]
a, b = lb
funding = {c: load_real_funding(c, np.asarray(common, dtype=np.int64)) for c in GA.COINS_28C}
N_EFF = 1.7

print("=" * 74)
print("1. Autocorrelation of signed 30m returns, lockbox, vs the iid null")
print("=" * 74)
print("%4s %10s %12s %12s %8s" % ("lag", "acf_raw", "acf_null", "t_eff", "verdict"))
rng = np.random.default_rng(20260930)
for lag in (1, 2, 3, 6, 12, 24, 48, 96, 288):
    raw, null = [], []
    for c in GA.COINS_28C:
        r = np.asarray(returns[c], float)[a:b]
        x = r - r.mean()
        raw.append(float(np.sum(x[:-lag] * x[lag:]) / np.sum(x * x)))
        # same marginal, shuffled within coin -> destroys dependence, keeps distribution
        null.append(float(np.corrcoef(r[:-lag], rng.permutation(r[lag:]))[0, 1]))
    raw = np.array(raw); null = np.array(null)
    t_raw = raw.mean() / (raw.std(ddof=1) / np.sqrt(len(raw)))
    t_eff = t_raw / np.sqrt(len(raw) / N_EFF)
    v = "REAL" if abs(t_eff) > 1.96 else "noise"
    print("%4d %+10.4f %+12.4f %+12.2f %8s" % (lag, raw.mean(), null.mean(), t_eff, v))

print()
print("=" * 74)
print("2. Is it exploitable? plain contrarian book vs constant long")
print("=" * 74)
print("   same fee=%.4f leverage=%.1f, real Binance funding, position cap %.2f"
      % (GA.FEE, GA.LEV, GA.POSITION_CAP))
print()
for lag in (1, 2, 3, 6, 12, 24):
    row = []
    for wname, (x, y) in (("train", (0, a)), ("lockbox", (a, b))):
        nets, bhs = [], []
        for c in GA.COINS_28C:
            r = np.asarray(returns[c], float)[x:y]
            sig = np.zeros(len(r))
            sig[lag:] = -np.asarray(returns[c], float)[x:y][:-lag]   # fade the lag-bar return
            sd = float(np.std(sig[: max(1, (y - x) // 2)]))
            pos = np.tanh(sig / (sd + 1e-9)) * GA.POSITION_CAP if sd > 1e-9 else sig
            pos = np.roll(pos, 1); pos[0] = 0.0
            f = funding[c][x:y]
            nets.append(accounting_bar_returns(pos, r, GA.FEE, f, GA.LEV))
            bhs.append(accounting_bar_returns(np.full(len(pos), GA.POSITION_CAP), r, GA.FEE, f, GA.LEV))
        sh = metrics(np.mean(np.stack(nets), axis=0), common[x:y])["sharpe"]
        bsh = metrics(np.mean(np.stack(bhs), axis=0), common[x:y])["sharpe"]
        row.append((sh, bsh, sh - bsh))
    print("  lag %2d:  train Sharpe %+.3f (B&H %+.3f, excess %+.3f)   |   "
          "lockbox Sharpe %+.3f (B&H %+.3f, excess %+.3f)"
          % (lag, row[0][0], row[0][1], row[0][2], row[1][0], row[1][1], row[1][2]))
print()
print("   excess > 0 on BOTH windows is the bar. Anything else is drift noise.")
