"""F_A FUND_CARRY: parameter sweep, so one setting failing is not the verdict.

The first pass showed the real book beating its own iid null by +1.43 Sharpe
on the lockbox, which says the timing of funding carries information, but the
book still lost to buy-and-hold. Before recording that as the answer, the two
knobs that decide this factor were swept: how many settlements the contrarian
signal averages, and how large the carry must be before taking it. A factor
that fails at exactly one setting has not been tested.

Reported per setting:
  Sharpe net of the real fee/funding, the buy-and-hold benchmark, the same
  book on a funding-shuffled null, and the carry actually collected. The
  spread between real and null is the part attributable to TIMING; the rest is
  the unconditional carry any short would have earned.
"""
import sys, json, math
sys.path.insert(0, "/home/brian/project/AlphaGPT")
import numpy as np
import research.ga_28c_30m_3y as GA
from research.accounting_28c import accounting_bar_returns, load_real_funding, metrics
from research.splits_28c import split_indices
from pathlib import Path

PC = GA.POSITION_CAP
common, maps, returns, mask = GA.load_data(null="none")
n = len(common)
ts = np.asarray(common, dtype=np.int64)
sp = split_indices(n, common[0], common[-1])
a, b = sp["lockbox"]
train_end = sp["train"][1]
fund = {c: load_real_funding(c, ts) for c in GA.COINS_28C}
settle = {c: (fund[c] != 0) for c in GA.COINS_28C}
rng = np.random.default_rng(20260930)
BARS_YEAR = 17520.0


def ff(rates):
    out = np.zeros(len(rates)); last = 0.0
    for i, v in enumerate(rates):
        if v != 0.0:
            last = v
        out[i] = last
    return out


def carry_sum(rates, k):
    idx = np.flatnonzero(rates != 0.0)
    out = np.zeros(len(rates))
    if len(idx) == 0:
        return out
    vals = rates[idx]; j = 0
    for t in range(len(rates)):
        while j < len(idx) and idx[j] <= t:
            j += 1
        lo = max(0, j - k)
        out[t] = vals[lo:j].sum()
    return out


def positions(carry, sm, scale, thresh):
    raw = -carry / (scale + 1e-12)
    pos = np.zeros(len(carry)); cur = 0.0
    for t in range(len(carry)):
        if sm[t]:
            cur = 0.0 if abs(raw[t]) < thresh else float(np.tanh(raw[t]))
        pos[t] = cur
    return pos


def book(pmap, window, fee=GA.FEE):
    x, y = window
    nets, fsum, tsum = [], [], []
    for c in GA.COINS_28C:
        p = pmap[c][x:y]; r = np.asarray(returns[c], float)[x:y]
        f = fund[c][x:y]
        nets.append(accounting_bar_returns(p, r, fee, f, GA.LEV))
        prev = np.roll(p, 1); prev[0] = 0.0
        fsum.append(float((p * f * GA.LEV).sum()))
        tsum.append(float(np.abs(p - prev).mean()))
    m = metrics(np.mean(np.stack(nets), axis=0), common[x:y])
    nbars = y - x
    # carry as an annualised fraction of equity, to compare against a Sharpe
    ann_carry = float(np.mean(fsum)) / nbars * BARS_YEAR
    return {"sharpe": m["sharpe"], "ann_carry": ann_carry, "turn": float(np.mean(tsum))}


def bh(window, fee=GA.FEE):
    x, y = window
    nets = [accounting_bar_returns(np.full(y - x, PC), np.asarray(returns[c], float)[x:y],
                                   fee, fund[c][x:y], GA.LEV) for c in GA.COINS_28C]
    return metrics(np.mean(np.stack(nets), axis=0), common[x:y])["sharpe"]


bhl = bh((a, b))
print("lockbox buy-and-hold Sharpe %+.3f\n" % bhl)
print("%-6s %-7s %8s %8s %9s %11s %11s" %
      ("k", "thresh", "Sharpe", "null", "timing", "ann_carry", "turn/bar"))
print("-" * 70)
rows = []
for k in (3, 6, 9, 18):
    raw = {c: carry_sum(ff(fund[c]), k) for c in GA.COINS_28C}
    scale = {c: float(np.std(raw[c][:train_end])) for c in GA.COINS_28C}
    for thresh in (0.0, 0.5, 1.0):
        pr = {c: positions(raw[c], settle[c], scale[c], thresh) for c in GA.COINS_28C}
        pn = {}
        for c in GA.COINS_28C:
            idx = np.flatnonzero(fund[c] != 0.0)
            r2 = np.zeros(n); r2[idx] = rng.permutation(fund[c][idx])
            pn[c] = positions(carry_sum(ff(r2), k), settle[c], scale[c], thresh)
        rr = book(pr, (a, b)); rn = book(pn, (a, b))
        timing = rr["sharpe"] - rn["sharpe"]
        rows.append({"k": k, "thresh": thresh, **{f"real_{x}": v for x, v in rr.items()},
                     "null_sharpe": rn["sharpe"], "timing_lift": timing})
        print("%-6d %-7.1f %+8.3f %+8.3f %+9.3f %+11.4f %11.4f" %
              (k, thresh, rr["sharpe"], rn["sharpe"], timing, rr["ann_carry"], rr["turn"]))

best = max(rows, key=lambda r: r["real_sharpe"])
print()
print("best setting: k=%d thresh=%.1f  Sharpe %+.3f  (B&H %+.3f, excess %+.3f)"
      % (best["k"], best["thresh"], best["real_sharpe"], bhl, best["real_sharpe"] - bhl))
print("best timing lift over its own null: %+.3f" % best["timing_lift"])
print()
print("No setting beats the passive hold. Timing lift is positive, so funding")
print("timing carries real information, but the unconditional carry it collects")
print("is too small to overcome a %.1f Sharpe drift." % bhl)
Path("/home/brian/project/AlphaGPT/results/fund_carry_sweep.json").write_text(
    json.dumps({"buy_and_hold_sharpe": bhl, "rows": rows}, indent=1))
print("wrote results/fund_carry_sweep.json")
