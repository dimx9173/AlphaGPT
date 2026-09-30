"""Is the funding-carry result contaminated by same-bar settlement alignment?

F_A reported a positive timing lift over a funding-shuffled null and ~+6.7%/yr
of carry. Before recording that as a real effect, one detail needs checking.

Funding is charged on the settlement bar itself: `funding_cost = position *
funding_rate * leverage` uses the rate at bar t against the position at bar t.
The F_A position was rebuilt AT settlement bars, using a carry sum that
included the rate at bar t. So the book was short (or long) into a payment
whose size and sign it had already read on that same bar.

On Binance the rate is published shortly before settlement, so this is close to
realisable -- but "close to" is exactly the kind of assumption that turns a
null into a finding, and the framework's own convention is to lag the position
one bar before it touches any return or charge. This re-runs F_A both ways on
identical data, with the scale and thresholds held fixed, so the size of the
effect is measured rather than argued.
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
settle = {c: (fund[c] != 0) for c in GA.COINS_28C}
COINS = list(GA.COINS_28C)
rng = np.random.default_rng(20260930)
K, TH = 3, 1.0


def ff(r):
    out = np.zeros(len(r)); last = 0.0
    for i, v in enumerate(r):
        if v != 0.0: last = v
        out[i] = last
    return out


def carry_sum(r, k):
    idx = np.flatnonzero(r != 0.0); out = np.zeros(len(r))
    vals = r[idx]; j = 0
    for t in range(len(r)):
        while j < len(idx) and idx[j] <= t: j += 1
        out[t] = vals[max(0, j - k):j].sum()
    return out


def build(rates_map, lag):
    """lag=True rebuilds the position one bar after the settlement it read."""
    pos = {}
    for c in COINS:
        raw = carry_sum(ff(rates_map[c]), K)
        sd = float(np.std(raw[:train_end]))
        step = np.zeros(n); cur = 0.0
        for t in range(n):
            if settle[c][t]:
                v = -raw[t] / (sd + 1e-12)
                cur = 0.0 if abs(v) < TH else float(np.tanh(v))
            step[t] = cur
        pos[c] = np.roll(step, 1) if lag else step
        if lag:
            pos[c][0] = 0.0
    return pos


nullmap = {}
for c in COINS:
    idx = np.flatnonzero(fund[c] != 0.0)
    r2 = np.zeros(n); r2[idx] = rng.permutation(fund[c][idx]); nullmap[c] = r2


def score(pmap, window, fee=GA.FEE):
    x, y = window
    nets, carry, turn = [], [], []
    for c in COINS:
        p = pmap[c][x:y]; r = np.asarray(GA.returns_load if False else returns[c], float)[x:y]
        f = fund[c][x:y]
        nets.append(accounting_bar_returns(p, r, fee, f, GA.LEV))
        carry.append(float((p * f * GA.LEV).sum()))
        prev = np.roll(p, 1); prev[0] = 0.0
        turn.append(float(np.abs(p - prev).mean()))
    m = metrics(np.mean(np.stack(nets), axis=0), common[x:y])
    return {"sharpe": m["sharpe"], "carry": float(np.mean(carry)) / (y - x) * 17520,
            "turn": float(np.mean(turn))}


def bh(window, fee=GA.FEE):
    x, y = window
    nets = [accounting_bar_returns(np.full(y - x, PC), np.asarray(returns[c], float)[x:y],
                                   fee, fund[c][x:y], GA.LEV) for c in COINS]
    return metrics(np.mean(np.stack(nets), axis=0), common[x:y])["sharpe"]


bhl = bh(LOCK)
print("=" * 82)
print("F_A funding carry: same-bar settlement alignment vs a one-bar lag")
print("=" * 82)
print("  buy-and-hold Sharpe on the lockbox: %+.3f\n" % bhl)
print("  %-26s %9s %9s %11s %10s" % ("variant", "Sharpe", "null", "timing", "ann_carry"))
print("-" * 82)
out = {}
for lag in (False, True):
    pr = build(fund, lag); pn = build(nullmap, lag)
    r = score(pr, LOCK); nl = score(pn, LOCK)
    name = "lagged 1 bar (causal)" if lag else "same-bar (look-ahead)"
    out[name] = {"real": r, "null": nl, "excess": r["sharpe"] - bhl,
                 "timing_lift": r["sharpe"] - nl["sharpe"]}
    print("  %-26s %+9.3f %+9.3f %+11.3f %10.4f" %
          (name, r["sharpe"], nl["sharpe"], r["sharpe"] - nl["sharpe"], r["carry"]))
print()
a = out["same-bar (look-ahead)"]; b = out["lagged 1 bar (causal)"]
print("  timing lift  same-bar %+.3f  ->  lagged %+.3f   (drop %+.3f)"
      % (a["timing_lift"], b["timing_lift"], a["timing_lift"] - b["timing_lift"]))
print("  carry        same-bar %+.4f  ->  lagged %+.4f   (drop %+.4f)"
      % (a["real"]["carry"], b["real"]["carry"], a["real"]["carry"] - b["real"]["carry"]))
print()
print("  Both variants remain far below the buy-and-hold of %+.3f." % bhl)
Path("/home/brian/project/AlphaGPT/results/fund_carry_settlement_alignment.json").write_text(
    json.dumps(out, indent=1))
print("wrote results/fund_carry_settlement_alignment.json")
