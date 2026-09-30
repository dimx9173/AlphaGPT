"""Variance-ratio test: is 30m return predictable at all?

The cheapest decisive experiment, proposed in the factor review. It costs a
few CPU-minutes, runs no search, and has no selection lift to explain away.

Lo-MacKinlay: for returns r_t, the variance ratio

    VR(q) = Var(sum of q consecutive returns) / (q * Var(one return))

is 1 under iid. VR > 1 means the series trends, VR < 1 means it mean-reverts.
The homoskedastic test statistic is

    z(q) = sqrt(n(q-1)) * (VR(q) - 1) / sqrt(2q - 5 + 2.5q^2/q)

Two halves, and the split is the whole point:

  - signed returns:  if VR is indistinguishable from 1, no linear or
    grammar recombination of 30m returns can predict direction, and further
    directional search is best-of-N mining. This is the kill shot.
  - absolute returns: if |r| is NOT white, volatility is predictable on the
    same bars, which is the one thing the screen says still has signal
    (LOG_VOL, HL_RANGE, VOL_CLUST held 89-100% sign consistency).

Pooling is done across coins, but the t-statistic is divided by the effective
number of independent bets. 28 coins at N_eff 1.7 is not 28 observations, and
treating it as such is how a null gets promoted to a finding.
"""
import sys, json, math
sys.path.insert(0, "/home/brian/project/AlphaGPT")
import numpy as np
import research.ga_28c_30m_3y as GA
from research.splits_28c import split_indices
from pathlib import Path

def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


common, maps, returns, mask = GA.load_data(null="none")
n = len(common)
lockbox = split_indices(n, common[0], common[-1])["lockbox"]
a, b = lockbox
print("lockbox bars %d..%d  (%d bars, %.1f days)" % (a, b, b - a, (b - a) / 48))
print()


def vr(r, q):
    r = np.asarray(r, float)
    n_ = len(r)
    if n_ <= q + 10:
        return None, None
    x = r - r.mean()
    v1 = float(np.sum(x * x) / (n_ - 1))
    if v1 <= 0:
        return None, None
    s_q = r[q:][: (len(r) - q) // q * q].reshape(-1, q).sum(axis=1)
    vq = float(np.sum(s_q ** 2) / (n_ - q))
    vr_ = vq / (q * v1)
    var = 2 * q - 5 + 2.5 * q ** 2 / q
    z = np.sqrt(n_ * (q - 1)) * (vr_ - 1) / np.sqrt(var)
    return vr_, z


N_EFF = 1.7
results = {}
for label, key, transform in (("signed returns (direction)", "signed", lambda r: r),
                              ("absolute returns |r| (vol)", "abs", lambda r: np.abs(r))):
    print("--- %s ---" % label)
    zs = {q: [] for q in (4, 8, 16, 32)}
    vrs = {q: [] for q in (4, 8, 16, 32)}
    for c in GA.COINS_28C:
        r = transform(np.asarray(returns[c], float))[a:b]
        for q in zs:
            v, z = vr(r, q)
            if z is not None:
                zs[q].append(z)
                vrs[q].append(v)
    row = {}
    for q in (4, 8, 16, 32):
        z = np.array(zs[q])
        if len(z) < 3:
            continue
        vr_mean = float(np.mean(vrs[q]))
        pooled = z.mean() / (z.std(ddof=1) / np.sqrt(len(z)))
        adj = pooled / np.sqrt(len(z) / N_EFF)
        p = 2 * (1 - norm_cdf(abs(adj)))
        verdict = "REJECT iid" if p < 0.05 else "white"
        row[str(q)] = {"VR": round(vr_mean, 4), "t_raw": round(pooled, 2),
                       "t_eff": round(adj, 2), "p": round(p, 4), "verdict": verdict}
        print("  q=%2d  VR=%.3f  t_raw=%+6.2f  t_eff(N_eff=1.7)=%+6.2f  p=%.4f  %s"
              % (q, vr_mean, pooled, adj, p, verdict))
    results[key] = row
    print()

print("Interpretation")
print("-" * 72)
print("  signed white      -> no direction predictable; stop directional search.")
print("  |r| NOT white     -> volatility predictable; pivot to vol/risk timing.")
print("  both white        -> nothing on these bars is predictable; freeze the")
print("                      factor set as a negative control and stop.")
Path("/home/brian/project/AlphaGPT/results/variance_ratio_lockbox.json").write_text(
    json.dumps(results, indent=1))
print("\nwrote results/variance_ratio_lockbox.json")
