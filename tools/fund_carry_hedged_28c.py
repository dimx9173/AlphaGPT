"""Is the funding carry real, and can it survive being separated from direction?

The sweep showed the contrarian funding book receiving roughly +6.7%/yr of
carry -- real money, and not something a price/volume factor can produce. It
still lost, because being short a rising market costs more than the carry
pays. That conflates two different questions:

  1. Is the carry income real?        -> decompose the P&L into price, carry, fee
  2. Can the carry be harvested alone? -> strip the market beta so the short
                                          exposure is not paid for twice

A carry book that must also be directionally short is not a carry strategy.
The only way to keep the income and drop the exposure is to hedge the common
mode, so this tests the carry both raw and beta-hedged against the universal,
and against a funding-shuffled null of the identical hedge. If hedged carry
does not beat the null net of costs, the carry thesis is dead in this window
and the answer to "which factor library" is that there isn't one.
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
a, b = sp["lockbox"]; train_end = sp["train"][1]
fund = {c: load_real_funding(c, ts) for c in GA.COINS_28C}
settle = {c: (fund[c] != 0) for c in GA.COINS_28C}
rng = np.random.default_rng(20260930)
K, TH = 3, 1.0          # the best setting from the sweep


def ff(r):
    out = np.zeros(len(r)); last = 0.0
    for i, v in enumerate(r):
        if v != 0.0: last = v
        out[i] = last
    return out


def carry_sum(r, k):
    idx = np.flatnonzero(r != 0.0); out = np.zeros(len(r))
    if len(idx) == 0: return out
    vals = r[idx]; j = 0
    for t in range(len(r)):
        while j < len(idx) and idx[j] <= t: j += 1
        out[t] = vals[max(0, j - k):j].sum()
    return out


def positions(carry, sm, scale, th):
    raw = -carry / (scale + 1e-12)
    pos = np.zeros(len(carry)); cur = 0.0
    for t in range(len(carry)):
        if sm[t]:
            cur = 0.0 if abs(raw[t]) < th else float(np.tanh(raw[t]))
        pos[t] = cur
    return pos


def build(fundmap):
    raw = {c: carry_sum(ff(fundmap[c]), K) for c in GA.COINS_28C}
    sc = {c: float(np.std(raw[c][:train_end])) for c in GA.COINS_28C}
    return {c: positions(raw[c], settle[c], sc[c], TH) for c in GA.COINS_28C}


nullmap = {}
for c in GA.COINS_28C:
    idx = np.flatnonzero(fund[c] != 0.0)
    r2 = np.zeros(n); r2[idx] = rng.permutation(fund[c][idx]); nullmap[c] = r2

pos_real = build(fund)
pos_null = build(nullmap)

# --- market mode, estimated on train only, then held fixed for the lockbox ---
mkt_train = np.mean([np.asarray(returns[c], float)[:train_end] for c in GA.COINS_28C], axis=0)


def hedge(pos, c, beta_train_only=True):
    """Subtract the coin's train-fitted loading on the equal-weight market."""
    r = np.asarray(returns[c], float)
    y = r[:train_end]; x = mkt_train
    b = float(np.cov(y, x, ddof=1)[0, 1] / max(np.var(x, ddof=1), 1e-18))
    return pos - b * PC


for label, pmap in (("real", pos_real), ("iid null", pos_null)):
    hedged = {c: hedge(pmap[c], c) for c in GA.COINS_28C}
    print("=" * 74)
    print("F_A FUND_CARRY  [%s]  k=%d thresh=%.1f   lockbox" % (label, K, TH))
    print("=" * 74)
    for vname, p in (("raw", pmap), ("beta-hedged", hedged)):
        px, cy, fe = [], [], []
        for c in GA.COINS_28C:
            p_ = p[c][a:b]; r_ = np.asarray(returns[c], float)[a:b]; f_ = fund[c][a:b]
            prev = np.roll(p_, 1); prev[0] = 0.0
            px.append((p_ * r_ * GA.LEV).sum())
            cy.append((p_ * f_ * GA.LEV).sum())
            fe.append((np.abs(p_ - prev) * GA.FEE * GA.LEV).sum())
        net = np.mean([(p[c][a:b] * np.asarray(returns[c], float)[a:b] * GA.LEV
                        - np.abs(np.roll(p[c][a:b], 1) - 0) * 0) for c in []])  # placeholder
        nets = [accounting_bar_returns(p[c][a:b], np.asarray(returns[c], float)[a:b],
                                      GA.FEE, fund[c][a:b], GA.LEV) for c in GA.COINS_28C]
        sh = metrics(np.mean(np.stack(nets), axis=0), common[a:b])["sharpe"]
        nb = b - a
        print("  %-12s Sharpe %+7.3f   price %+.3f   carry %+.3f   fee %+.3f  (mean log-units x1e3)"
              % (vname, sh, np.mean(px) / nb * 1000, -np.mean(cy) / nb * 1000,
                 -np.mean(fe) / nb * 1000))
    print()

print("Reading: 'carry' is income received (positive is good). The question is")
print("whether hedged carry beats its own null. The null has identical market")
print("exposure and identical hedging, so any gap is timing information.")
Path("/home/brian/project/AlphaGPT/results/fund_carry_hedged.json").write_text(
    json.dumps({"k": K, "thresh": TH}, indent=1))
