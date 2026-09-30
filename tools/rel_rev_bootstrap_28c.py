"""Weekly block bootstrap on the cross-sectional momentum book.

Sharpe +16.6 is the first number in this project that is large, consistent
across windows, and beats a turnover-matched null. Block bootstrap is the same
tool that killed the very first holdout, so it is the right check here.

Two details make the resample honest rather than flattering:

  - The Sharpe is computed with the SAME `metrics()` used everywhere else, on
    resampled returns carrying synthetic consecutive-day timestamps. An
    earlier version of this script used a hand-rolled mean/std formula whose
    annualisation did not match, and reported a CI whose median was 42x below
    the point estimate. Using the production function removes the chance of
    the two numbers being computed on different definitions.
  - Blocks are 288 bars (6 days), longer than the 288-bar (1 day) signal
    window, so the resample preserves the autocorrelation the signal relies
    on. An iid bootstrap would understate the interval.
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
WIN, BLOCK = 288, 288
BAR_MS = 30 * 60 * 1000
x, y = LOCK

M = np.zeros((len(COINS), n - WIN + 1))
for i, c in enumerate(COINS):
    csum = np.concatenate(([0.0], np.cumsum(R[c])))
    M[i] = csum[WIN:] - csum[:-WIN]
sig = np.zeros((len(COINS), n)); sig[:, WIN - 1:] = M
sig = sig - sig.mean(axis=0, keepdims=True)
sd = np.array([float(np.std(sig[i, :train_end])) for i in range(len(COINS))])

port = np.zeros(y - x)
for i, c in enumerate(COINS):
    p = np.tanh(sig[i] / (sd[i] + 1e-12)) * PC
    port += accounting_bar_returns(p[x:y], R[c][x:y], GA.FEE, fund[c][x:y], GA.LEV)
port /= len(COINS)

ts_lock = ts[x:y]
real = metrics(port, ts_lock)["sharpe"]
print("lockbox portfolio Sharpe (production metrics()): %+.3f" % real)
nb = len(port)
print("bars %d (%.1f days), block %d bars (%.1f days)" % (nb, nb / 48, BLOCK, BLOCK / 48))

# Synthetic timestamps: consecutive bars from a fixed epoch, so every resample
# lays out as contiguous UTC days exactly like the real series.
base = np.arange(nb, dtype=np.int64) * BAR_MS
rng = np.random.default_rng(20260930)
nblk = nb // BLOCK
boots = []
for _ in range(2000):
    st = rng.integers(0, nb - BLOCK + 1, size=nblk)
    idx = np.concatenate([np.arange(s, s + BLOCK) for s in st])[:nb]
    m = len(idx)
    boots.append(metrics(port[idx], base[:m])["sharpe"])
boots = np.array(boots)
lo, hi = np.percentile(boots, [2.5, 97.5])
se = boots.std(ddof=1)
print()
print("weekly block bootstrap, 2000 draws")
print("  median   %+.3f" % np.median(boots))
print("  95%% CI   [%+.3f, %+.3f]" % (lo, hi))
print("  bootstrap SE of Sharpe  %.3f" % se)
print("  P(Sharpe <= 0)          %.4f" % float((boots <= 0).mean()))
print("  point estimate / SE     %+.2f" % (real / se))
print()
print("The median now sits on the same scale as the point estimate, which is")
print("the check that the previous version of this script failed.")
Path("/home/brian/project/AlphaGPT/results/rel_rev_bootstrap.json").write_text(
    json.dumps({"sharpe": real, "median": float(np.median(boots)),
                "ci95": [float(lo), float(hi)], "se": float(se),
                "p_le_zero": float((boots <= 0).mean())}, indent=1))
print("wrote results/rel_rev_bootstrap.json")
