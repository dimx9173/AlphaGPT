"""H68 1h ADX v4 (Top5) -- DIAGNOSTIC ONLY.

P0-3 permutation FAILED (per-coin p>=0.05, results/permutation.json), so per
the PRP global exit rule this step's conclusion is "PENDING" and MUST NOT be
used as demo-listing evidence. No adoption, no live change, live untouched.
Offline read-only: reads data/data_1y/1h/*.csv only.

Engine mirrors research/run_weight_modes.py leg_net: E10 FORMULA
[3,2,7,2,7,11,15,4,4,6,6,10] via StackVM+FeatureEngineer; MemeBacktest
venue=aster lev2 short_enabled fund0.0005; quantile q0.3 long-only +
cooldown + stops + vol_scale(vt None->1.0) + roll1. Static equal 0.2
weights (Top5 locked specs ETC/TRX/ATOM/APT/KAS).

1h native: cd/ts/vw x4 (4h-bar units -> 1h-bar units), BPY=8760. Data data/data_1y/1h/{COIN}.csv
(~8760 rows, cols timestamp,open,high,low,close,volume,quote_volume,trades). Equal 0.2 weights.

ADX filter semantics: causal Wilder ADX(112) per coin on native 1h bars (112-bar, 112h lookback; v4 vs H58 ADX112).
Trade ONLY when ADX(56) > thr (gate mask applied to lp/sp after
cooldown+stops+vol_scale, before roll1; seed bars ADX=0 -> filtered).
Sweep thr in {20, 25, 30} vs no-filter base.

Metric per cell: FULL sharpe/mdd/cum/final_x/ann/n/turnover/trades +
coverage (fraction of bars passing gate) + per-coin single-leg
sensitivity (sharpe/turnover/trades/coverage per coin per thr).

IMPORTANT: results/iter_H68_adx.json is dumped after EACH unit
(base + 3 thresholds; "complete": false until the final write), so a
killed run still leaves partial rows behind.

Outputs: results/iter_H68_adx.json (+ logs/iter_H68_adx.log).
Offline read-only. Verdict PENDING (P0-3 FAIL); no adoption.

Smoke mode (for tests): ITER_H68_SMOKE=1 shrinks to thr {25} on coins
{ETC,TRX} only. ITER_H68_OUT / ITER_H68_LOG override output paths (tests
use temp files so the committed FULL artifact is not clobbered).
"""
import csv
import json
import math
import os
import pathlib
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy
import torch

from model_core.backtest import MemeBacktest
from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from strategy_manager.config import (
    FORMULA,
    LOCKED_APT,
    LOCKED_ATOM,
    LOCKED_ETC,
    LOCKED_KAS,
    LOCKED_TRX,
    LEV,
    FUND,
    FEE,
)

assert LEV == 2.0, "LEV lock broken: %r" % LEV

COINS = ["ETC", "TRX", "ATOM", "APT", "KAS"]
SPECS = {"ETC": LOCKED_ETC, "TRX": LOCKED_TRX, "ATOM": LOCKED_ATOM,
         "APT": LOCKED_APT, "KAS": LOCKED_KAS}
W = {c: 0.2 for c in COINS}
THRS = [20, 25, 30]
ADX_PERIOD = 112
BPY = 8760.0
SCALE = 4

OUT = pathlib.Path(os.getenv("ITER_H68_OUT", "results/iter_H68_adx.json"))
LOG = pathlib.Path(os.getenv("ITER_H68_LOG", "logs/iter_H68_adx.log"))

SMOKE = os.getenv("ITER_H68_SMOKE") == "1"
SMOKE_COINS = ["ETC", "TRX"]
SMOKE_THRS = [25]


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")


def load1h(c):
    rows = list(csv.DictReader(open("data/data_1y/1h/%s.csv" % c)))
    return [(int(r["timestamp"]), float(r["open"]), float(r["high"]),
             float(r["low"]), float(r["close"]), float(r["volume"]))
            for r in rows]


def common1h(coins):
    raw = {c: load1h(c) for c in coins}
    s = max(r[0][0] for r in raw.values())
    e = min(r[-1][0] for r in raw.values())
    bars = {}
    for c in coins:
        rr = [r for r in raw[c] if s <= r[0] <= e]
        bars[c] = [(r[1], r[2], r[3], r[4], r[5]) for r in rr]
    n = min(len(b) for b in bars.values())
    for c in coins:
        bars[c] = bars[c][:n]
    return bars, n


def build_sig(bars):
    n = len(bars)
    raw = {"open": torch.tensor([[b[0] for b in bars]]),
           "high": torch.tensor([[b[1] for b in bars]]),
           "low": torch.tensor([[b[2] for b in bars]]),
           "close": torch.tensor([[b[3] for b in bars]]),
           "volume": torch.tensor([[b[4] for b in bars]]),
           "liquidity": torch.full((1, n), 1e7),
           "fdv": torch.full((1, n), 1e8)}
    sig = StackVM(use_advanced=False).execute(FORMULA, FeatureEngineer.compute_features(raw, use_advanced=False))
    rets = [(bars[i + 1][3] - bars[i][3]) / bars[i][3]
            for i in range(n - 1)] + [0.0]
    return raw, torch.tensor([rets]), sig


def qmask(sig, q):
    if q is None or float(q) <= 0:
        return None
    a = sig.detach().float().abs().reshape(-1)
    k = max(1, int(len(a) * float(q)))
    thr = torch.topk(a, k).values.min()
    return (sig.detach().float().abs() >= thr).float()


def adx112(bars_c, period=ADX_PERIOD):
    """Causal Wilder ADX(period) on 1h bars; seed region -> 0.0."""
    hi = numpy.array([b[1] for b in bars_c], dtype=float)
    lo = numpy.array([b[2] for b in bars_c], dtype=float)
    cl = numpy.array([b[3] for b in bars_c], dtype=float)
    n = len(cl)
    out = numpy.zeros(n)
    if n < 2 * period + 1:
        return out.tolist()
    up = hi[1:] - hi[:-1]
    dn = lo[:-1] - lo[1:]
    pdm = numpy.where((up > dn) & (up > 0), up, 0.0)
    mdm = numpy.where((dn > up) & (dn > 0), dn, 0.0)
    pc = cl[:-1]
    tr = numpy.maximum(hi[1:] - lo[1:],
                       numpy.maximum(numpy.abs(hi[1:] - pc),
                                     numpy.abs(lo[1:] - pc)))
    satr = numpy.zeros(n)
    spdm = numpy.zeros(n)
    smdm = numpy.zeros(n)
    satr[period] = tr[:period].sum()
    spdm[period] = pdm[:period].sum()
    smdm[period] = mdm[:period].sum()
    for t in range(period + 1, n):
        satr[t] = satr[t - 1] - satr[t - 1] / period + tr[t - 1]
        spdm[t] = spdm[t - 1] - spdm[t - 1] / period + pdm[t - 1]
        smdm[t] = smdm[t - 1] - smdm[t - 1] / period + mdm[t - 1]
    pdi = numpy.where(satr > 1e-12, 100.0 * spdm / numpy.maximum(satr, 1e-12), 0.0)
    mdi = numpy.where(satr > 1e-12, 100.0 * smdm / numpy.maximum(satr, 1e-12), 0.0)
    ds = pdi + mdi
    dx = numpy.where(ds > 1e-12, 100.0 * numpy.abs(pdi - mdi) / numpy.maximum(ds, 1e-12), 0.0)
    ax = numpy.zeros(n)
    ax[2 * period - 1] = dx[period:2 * period].sum() / period
    for t in range(2 * period, n):
        ax[t] = (ax[t - 1] * (period - 1) + dx[t]) / period
    out[2 * period - 1:] = ax[2 * period - 1:]
    return out.tolist()


def leg_base(bars_c, spec):
    """Locked-engine pre-roll positions + returns (1h native)."""
    raw, rt, sg = build_sig(bars_c)
    n = len(bars_c)
    bt = MemeBacktest(venue="aster", leverage=LEV, short_enabled=True,
                      funding_override=FUND, fee_override=FEE,
                      long_th=spec["lth"], short_th=spec["sth"],
                      cooldown_bars=int(spec["cd"]) * SCALE,
                      bars_per_year=BPY, stop_loss=spec["sl"],
                      time_stop=int(spec["ts"]) * SCALE,
                      vol_target=spec["vt"],
                      vol_window=int(spec["vw"]) * SCALE)
    s = torch.sigmoid(sg)
    safe = (raw["liquidity"] > bt.min_liq).float()
    lp = (s > bt.long_th).float() * safe
    sp = (s < bt.short_th).float() * safe
    mask = qmask(sg, spec["q"])
    if mask is not None:
        lp = lp * mask
    lp, sp = bt._apply_cooldown(lp, sp)
    lp, sp = bt._apply_stops(lp, sp, rt)
    sc = bt._vol_scale(rt)
    lp, sp = lp * sc, sp * sc
    return (lp - sp)[0].tolist(), rt[0].tolist()


def count_trades(pos):
    n = 0
    prev = 0.0
    for v in pos:
        cur = 1.0 if abs(v) > 0.5 else 0.0
        if cur > 0.5 and prev < 0.5:
            n += 1
        prev = cur
    return n


def seg(net, turn, a, b):
    s = net[a:b]
    t = turn[a:b]
    n = len(s)
    m = sum(s) / n if n else 0
    v = sum((x - m) ** 2 for x in s) / max(n - 1, 1) if n > 1 else 0
    sh = m / math.sqrt(v) * math.sqrt(BPY) if v > 0 else 0.0
    cs = 0.0
    pk0 = -1e18
    md = 0.0
    for x in s:
        cs += x
        pk0 = max(pk0, cs)
        md = max(md, pk0 - cs)
    cum = sum(s)
    return {"sharpe": round(sh, 3), "ann": round(cum / n * BPY, 4) if n else 0.0,
            "mdd": round(md, 4), "cum": round(cum, 4),
            "final_x": round(1.0 + cum, 4), "n": n,
            "turnover": round(sum(t) / n, 6) if n else 0.0}


def eval_unit(base, adxs, n, thr, coins, weights):
    """Apply ADX gate (thr=None -> no filter) + roll1 + aster accounting."""
    legs = {}
    turns = {}
    per_coin = {}
    tot_trades = 0
    for c in coins:
        pre, rt = base[c]
        g = [1.0] * n if thr is None else [
            1.0 if adxs[c][t] > thr else 0.0 for t in range(n)]
        p = [pre[t] * g[t] for t in range(n)]
        rolled = [0.0] + p[:-1]
        dp = [abs(rolled[t] - (rolled[t - 1] if t else 0.0)) for t in range(n)]
        net_c = [rolled[t] * rt[t] * LEV - dp[t] * FEE * LEV
                 - rolled[t] * FUND * LEV for t in range(n)]
        legs[c] = (rolled, net_c, rt)
        turns[c] = dp
        tr = count_trades(rolled)
        tot_trades += tr
        st = seg(net_c, dp, 0, n)
        st["trades"] = tr
        st["coverage"] = round(sum(g) / n, 6)
        per_coin[c] = st
    w = weights
    net = [sum(legs[c][1][t] * w[c] for c in coins) for t in range(n)]
    turn = [sum(turns[c][t] * w[c] for c in coins) for t in range(n)]
    full = seg(net, turn, 0, n)
    full["trades"] = tot_trades
    cov = sum(1.0 for t in range(n)
              if all(((adxs[c][t] > thr) if thr is not None else True)
                     for c in coins)) / n if n else 0.0
    full["coverage_all"] = round(cov, 6)
    return full, per_coin


def dump(rows, units_done, n, coins, weights, thrs, complete):
    out = {
        "config": {
            "engine": "mirror run_weight_modes.py leg_net + quantile q0.3 "
                      "long-only + cooldown + stops + vol_scale(vt None->1.0) "
                      "+ roll1; static equal 0.2 Top5; 1h native cd/ts/vw x4",
            "formula": list(FORMULA),
            "basket": {c: {"lth": SPECS[c]["lth"], "sth": SPECS[c]["sth"],
                           "cd": SPECS[c]["cd"], "sl": SPECS[c]["sl"],
                           "ts": SPECS[c]["ts"], "q": SPECS[c]["q"]}
                       for c in COINS},
            "weights": dict(weights),
            "filter": "ADX(%d) gate: trade only bars with ADX > thr "
                      "(causal Wilder, seed=0->filtered)" % ADX_PERIOD,
            "adx_period": ADX_PERIOD,
            "thresholds": list(thrs),
            "lev": LEV,
            "fund": FUND,
            "fee": FEE,
            "smoke": SMOKE,
            "grid": "1h",
            "grid_bars": n,
            "bpy": BPY,
            "scale": SCALE,
        },
        "rows": rows,
        "units_done": list(units_done),
        "complete": complete,
        "verdict": "PENDING",
        "decision": "NO_ADOPTION",
        "decision_note": "待定(P0-3 FAIL): ADX过滤只做诊断,不采用,不改live, "
                         "live basket与阈值保持locked原状。",
        "conclusion": ("PENDING (待定, P0-3 FAIL): 1h原生格子; ADX filter "
                       "diagnostic only, no adoption, no live change."),
    }
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return out


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").write("iter_H68_adx start smoke=%s\n" % SMOKE)
    coins = list(SMOKE_COINS) if SMOKE else list(COINS)
    thrs = list(SMOKE_THRS) if SMOKE else list(THRS)
    weights = {c: 1.0 / len(coins) for c in coins}
    bars, n = common1h(coins)
    log("common 1h native n=%d coins=%s thrs=%s scale=x%d"
        % (n, coins, thrs, SCALE))
    assert n > 2000, "grid too short: %d" % n
    base = {c: leg_base(bars[c], SPECS[c]) for c in coins}
    log("base legs built (locked engine, fee=%.4f fund=%.4f)" % (FEE, FUND))
    adxs = {c: adx112(bars[c]) for c in coins}
    for c in coins:
        a = adxs[c]
        log("%s ADX112 mean=%.2f max=%.2f cov20=%.3f cov25=%.3f cov30=%.3f"
            % (c, sum(a) / n, max(a),
               sum(1 for v in a if v > 20) / n,
               sum(1 for v in a if v > 25) / n,
               sum(1 for v in a if v > 30) / n))

    rows = []
    units_done = []

    def run_unit(label, thr):
        full, per_coin = eval_unit(base, adxs, n, thr, coins, weights)
        row = {"unit": label, "thr": thr, "FULL": full,
               "per_coin": per_coin}
        rows.append(row)
        units_done.append(label)
        dump(rows, units_done, n, coins, weights, thrs, complete=False)
        log("%s FULL sh=%.3f mdd=%.4f cum=%.4f to=%.6f trades=%d cov=%.4f"
            % (label, full["sharpe"], full["mdd"], full["cum"],
               full["turnover"], full["trades"], full["coverage_all"]))

    run_unit("base", None)
    for thr in thrs:
        run_unit("adx_gt_%d" % thr, thr)
    dump(rows, units_done, n, coins, weights, thrs, complete=True)
    log("wrote %s verdict=PENDING decision=NO_ADOPTION" % OUT)


if __name__ == "__main__":
    main()
