"""F_A FUND_CARRY: the one factor that is not a reparametrisation of OHLCV.

The 12 existing factors are all functions of the same 30m bar series, which is
why the gene pool is dead. Funding is a different data axis: it is a
contractual cashflow, not a price statistic, and a book that is short when
funding is positive RECEIVES the payment whether or not prices are
predictable. The accounting already charges funding on the signed position
(`funding_cost = position * funding_rate * leverage`), so the carry is
credited by the same code path that charges it -- there is no separate
bookkeeping that could flatter the result.

Two things make this the fair test Kimi asked for:
  - the position is rebuilt at SETTLEMENTS ONLY (every 8h), not every bar.
    Carry is a slow cashflow; trading it at 30m frequency would pay 48x the
    fee to capture a signal that only changes 3x a day.
  - the funding series is shuffled within each coin to form the null, so the
    screen asks whether the TIMING of funding predicts, not whether funding
    is positive on average. The unconditional carry is reported separately,
    because "always short" is a benchmark, not a strategy.

Causality: at bar t the factor may only use settlements at or before t. The
rate is forward-filled from the last observed settlement and never
interpolated between events.
"""
import sys, json, math
sys.path.insert(0, "/home/brian/project/AlphaGPT")
import numpy as np
import research.ga_28c_30m_3y as GA
from research.accounting_28c import (accounting_bar_returns, load_real_funding,
                                     metrics)
from research.splits_28c import split_indices
from pathlib import Path

PC = GA.POSITION_CAP
common, maps, returns, mask = GA.load_data(null="none")
n = len(common)
ts = np.asarray(common, dtype=np.int64)
sp = split_indices(n, common[0], common[-1])
a, b = sp["lockbox"]
train = (0, sp["train"][1])
fund = {c: load_real_funding(c, ts) for c in GA.COINS_28C}
settle = {c: (fund[c] != 0) for c in GA.COINS_28C}
N_EFF = 1.7
rng = np.random.default_rng(20260930)


def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def ff_rate(rates):
    """Forward-fill the settlement rate; strictly causal."""
    out = np.zeros(len(rates))
    last = 0.0
    for i, v in enumerate(rates):
        if v != 0.0:
            last = v
        out[i] = last
    return out


def carry_sum(rates, bars=48):
    """Sum of settlements in the trailing `bars` window, updated only at events.

    Counted from the settlement events themselves, so it does not drift with
    the number of bars between settlements.
    """
    idx = np.flatnonzero(rates != 0.0)
    out = np.zeros(len(rates))
    if len(idx) == 0:
        return out
    vals = rates[idx]
    j = 0
    for t in range(len(rates)):
        while j < len(idx) and idx[j] <= t:
            j += 1
        lo = j - 9
        out[t] = vals[lo:j].sum() if lo >= 0 else vals[:j].sum()
    return out


def positions(carry, settle_mask, scale, long_when_negative=True):
    """Step position, rebuilt at settlements only."""
    raw = -carry / (scale + 1e-12) if long_when_negative else carry / (scale + 1e-12)
    pos = np.zeros(len(carry))
    cur = 0.0
    for t in range(len(carry)):
        if settle_mask[t]:
            cur = float(np.tanh(raw[t]))
        pos[t] = cur
    return pos


def book(pos_by_coin, window, fee=GA.FEE):
    x, y = window
    nets, fees_, funds_, turns = [], [], [], []
    for c in GA.COINS_28C:
        p = pos_by_coin[c][x:y]
        r = np.asarray(returns[c], float)[x:y]
        f = funding_slice = np.asarray(fund[c])[x:y]
        net = accounting_bar_returns(p, r, fee, f, GA.LEV)
        nets.append(net)
        prev = np.roll(p, 1); prev[0] = 0.0
        fees_.append((np.abs(p - prev) * fee * GA.LEV).sum())
        funds_.append((p * f * GA.LEV).sum())
        turns.append(float(np.abs(p - prev).mean()))
    m = metrics(np.mean(np.stack(nets), axis=0), common[x:y])
    return {"sharpe": m["sharpe"], "fee_drag": -float(np.mean(fees_)),
            "funding_net": -float(np.mean(funds_)),
            "turnover": float(np.mean(turns))}


def bh(window, fee=GA.FEE):
    x, y = window
    nets = [accounting_bar_returns(np.full(y - x, PC), np.asarray(returns[c], float)[x:y],
                                   fee, np.asarray(fund[c])[x:y], GA.LEV)
            for c in GA.COINS_28C]
    return metrics(np.mean(np.stack(nets), axis=0), common[x:y])["sharpe"]


# --- build the factor, on train scale only, then hold the scale fixed ---
raw_carry = {c: carry_sum(ff_rate(fund[c])) for c in GA.COINS_28C}
scale = {c: float(np.std(raw_carry[c][:train[1]])) for c in GA.COINS_28C}
pos_real = {c: positions(raw_carry[c], settle[c], scale[c]) for c in GA.COINS_28C}
pos_null = {}
for c in GA.COINS_28C:
    idx = np.flatnonzero(fund[c] != 0.0)
    sh = rng.permutation(fund[c][idx])
    r2 = np.zeros(len(fund[c])); r2[idx] = sh
    pos_null[c] = positions(carry_sum(ff_rate(r2)), settle[c], scale[c])

print("=" * 78)
print("F_A FUND_CARRY  -- contrarian to sustained funding, rebalanced at settlements")
print("=" * 78)
print("  position cap %.2f  fee %.4f/side  leverage %.1f  (same as every other run)" % (PC, GA.FEE, GA.LEV))
print()
print("  %-10s %-8s %9s %9s %11s %9s %10s" %
      ("book", "window", "Sharpe", "B&H", "excess", "turn/bar", "funding_net"))
for label, pmap in (("real", pos_real), ("iid null", pos_null)):
    for wname, w in (("train", train), ("lockbox", (a, b))):
        r = book(pmap, w)
        h = bh(w)
        print("  %-10s %-8s %+9.3f %+9.3f %+11.3f %9.4f %+10.4f" %
              (label, wname, r["sharpe"], h, r["sharpe"] - h, r["turnover"], r["funding_net"]))
    print()

print("  cost decomposition on the lockbox (real funding, per bar averaged):")
for label, pmap in (("real", pos_real), ("iid null", pos_null)):
    r = book(pmap, (a, b))
    print("    %-9s fee drag %+.4f   funding received %+.4f" % (label, r["fee_drag"], r["funding_net"]))
print()

# --- unconditional carry: always short. A benchmark, not a strategy. ---
print("  unconditional benchmarks on the lockbox (no funding timing at all):")
always_short = {c: np.full(n, -PC) for c in GA.COINS_28C}
always_long = {c: np.full(n, PC) for c in GA.COINS_28C}
for label, pmap in (("always long", always_long), ("always short", always_short)):
    r = book(pmap, (a, b))
    print("    %-14s Sharpe %+7.3f   funding received %+.4f   turnover %.4f" %
          (label, r["sharpe"], r["funding_net"], r["turnover"]))
print()

# --- direction test: does timing beat the unconditional carry? ---
r_real = book(pos_real, (a, b))
r_short = book(always_short, (a, b))
diff = []
for _ in range(400):
    s = book({c: np.full(n, -PC) for c in GA.COINS_28C}, (a, b))["sharpe"]
    diff.append(s)
print("  timing vs always-short on the lockbox: %+.3f vs %+.3f  -> %+.3f Sharpe added by timing"
      % (r_real["sharpe"], r_short["sharpe"], r_real["sharpe"] - r_short["sharpe"]))
print()

Path("/home/brian/project/AlphaGPT/results/fund_carry_screen.json").write_text(
    json.dumps({"real": {w: book(pos_real, w) for w in (train, (a, b))}}, indent=1))
print("wrote results/fund_carry_screen.json")
