"""H108 1h Chaikin money-flow oscillator gate (Top5) -- DIAGNOSTIC ONLY.

P0-3 permutation FAILED (per-coin p>=0.05, results/permutation.json), so per
the PRP global exit rule this step's conclusion is "PENDING" and MUST NOT be
used as demo-listing evidence. No adoption, no live change, live untouched.
Offline read-only: reads data/data_1y/1h/*.csv only.

Engine mirrors research/run_weight_modes.py leg_net: E10 FORMULA
[3,2,7,2,7,11,15,4,4,6,6,10] via StackVM+FeatureEngineer; MemeBacktest
venue=aster lev2 short_enabled fund0.0005; quantile q0.3 long-only +
cooldown + stops + vol_scale(vt None->1.0) + roll1. Static equal 0.2
weights (Top5 locked specs ETC/TRX/ATOM/APT/KAS).

1h native: cd/ts/vw x4 (4h-bar units -> 1h-bar units), BPY=8760.
Data data/data_1y/1h/{COIN}.csv (~8760 rows,
cols timestamp,open,high,low,close,volume,quote_volume,trades).
Equal 0.2 weights.

Chaikin entry-gate semantics: per coin on native 1h bars, compute
Money Flow Multiplier MFM = ((close-low)-(high-close))/(high-low)
(0 when high==low), Money Flow Volume MFV = MFM * volume, ADL = cumsum(MFV),
Chaikin osc = EMA_fast(ADL) - EMA_slow(ADL). ENTRY ONLY directional sign gate:
a new long entry (flat->long) or a short->long flip is allowed only when
osc[t] > 0; a new short entry (flat->short) or a long->short flip only when
osc[t] < 0; continuations (same-sign holds) and exits (pre==0 -> flat) always
pass; osc==0 blocks new entries/flips. A blocked flip holds the previous side
(no forced exit). Gate is applied to pre-roll positions (after
cooldown+stops+vol_scale), before roll1. Warmup (first max(slow,fast) bars):
osc=0.0 neutral (entries blocked). Requires volume+quote_volume columns
(quote_volume asserted present; MFV uses volume, standard Chaikin).
Sweep (slow,fast) in {(27,10), (48,24)} vs no-filter base.

Metric per cell: FULL sharpe/mdd/cum/final_x/ann/n/turnover/trades +
coverage (frac osc>0 / osc<0 / nonzero) + per-coin single-leg sensitivity
(sharpe/turnover/trades/coverage per coin per params).

IMPORTANT: results/iter_H108_chaikin.json is dumped after EACH unit
(base + 2 param cells; "complete": false until the final write), so a
killed run still leaves partial rows behind.

Outputs: results/iter_H108_chaikin.json (+ logs/iter_H108_chaikin.log).
Offline read-only. Verdict PENDING (P0-3 FAIL); no adoption.

Smoke mode (for tests): ITER_H108_SMOKE=1 shrinks to (27,10) only on coins
{ETC,TRX}. ITER_H108_OUT / ITER_H108_LOG override output paths (tests
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
PARAMS = [(27, 10), (48, 24)]
BPY = 8760.0
SCALE = 4

OUT = pathlib.Path(os.getenv("ITER_H108_OUT", "results/iter_H108_chaikin.json"))
LOG = pathlib.Path(os.getenv("ITER_H108_LOG", "logs/iter_H108_chaikin.log"))

SMOKE = os.getenv("ITER_H108_SMOKE") == "1"
SMOKE_COINS = ["ETC", "TRX"]
SMOKE_PARAMS = [(27, 10)]


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")


def load1h(c):
    rows = list(csv.DictReader(open("data/data_1y/1h/%s.csv" % c)))
    assert "volume" in rows[0] and "quote_volume" in rows[0], "volume+quote_volume required"
    return [(int(r["timestamp"]), float(r["open"]), float(r["high"]),
             float(r["low"]), float(r["close"]), float(r["volume"]),
             float(r["quote_volume"]))
            for r in rows]


def common1h(coins):
    raw = {c: load1h(c) for c in coins}
    s = max(r[0][0] for r in raw.values())
    e = min(r[-1][0] for r in raw.values())
    bars = {}
    for c in coins:
        rr = [r for r in raw[c] if s <= r[0] <= e]
        bars[c] = [(r[1], r[2], r[3], r[4], r[5], r[6]) for r in rr]
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
    sig = StackVM().execute(FORMULA, FeatureEngineer.compute_features(raw))
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


def ema_series(x, period):
    n = len(x)
    out = [0.0] * n
    if n == 0:
        return out
    k = 2.0 / (period + 1.0)
    seed_n = min(period, n)
    sma = sum(x[:seed_n]) / seed_n
    for t in range(n):
        if t < seed_n - 1:
            out[t] = sma
        elif t == seed_n - 1:
            # seed with SMA then apply one EMA step on last seed bar
            e = sma
            e = x[t] * k + e * (1.0 - k)
            out[t] = e
        else:
            out[t] = x[t] * k + out[t - 1] * (1.0 - k)
    return out


def chaikin_osc(bars_c, slow, fast):
    """Chaikin money-flow oscillator: ADL then EMA_fast - EMA_slow.

    Warmup: first max(slow,fast) bars -> 0.0 neutral (blocks entries).
    """
    n = len(bars_c)
    mfv = []
    for b in bars_c:
        hi, lo, cl, vo = b[1], b[2], b[3], b[4]
        rng = hi - lo
        if rng == 0.0:
            mfm = 0.0
        else:
            mfm = ((cl - lo) - (hi - cl)) / rng
        mfv.append(mfm * vo)
    adl = []
    cs = 0.0
    for v in mfv:
        cs += v
        adl.append(cs)
    ef = ema_series(adl, fast)
    es = ema_series(adl, slow)
    w = max(slow, fast)
    return [(ef[t] - es[t]) if t >= w else 0.0 for t in range(n)]


def leg_base(bars_c, spec):
    """Locked-engine pre-roll positions + returns (1h native)."""
    raw, rt, sg = build_sig(bars_c)
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


def apply_chaikin_gate(pre, osc):
    """Entry-only directional Chaikin sign gate on pre-roll positions.

    pre==0 -> flat always passes (exits preserved). Flat->long entry or
    short->long flip requires osc[t] > 0; flat->short or long->short flip
    requires osc[t] < 0; same-sign continuation passes; osc==0 blocks new
    entries/flips. A blocked flip holds the previous side (no forced exit).
    """
    n = len(pre)
    g = [0.0] * n
    prev = 0.0
    for t in range(n):
        cur = pre[t]
        o = osc[t]
        if cur == 0.0:
            g[t] = 0.0
        elif prev == 0.0:
            if cur > 0:
                g[t] = cur if o > 0 else 0.0
            else:
                g[t] = cur if o < 0 else 0.0
        elif (cur > 0) == (prev > 0):
            g[t] = cur
        else:
            if cur > 0:
                g[t] = cur if o > 0 else prev
            else:
                g[t] = cur if o < 0 else prev
        prev = g[t]
    return g


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


def cov_stats(osc):
    n = len(osc)
    if not n:
        return {"long": 0.0, "short": 0.0, "nonzero": 0.0}
    nl = sum(1 for v in osc if v > 0)
    ns = sum(1 for v in osc if v < 0)
    return {"long": round(nl / n, 6), "short": round(ns / n, 6),
            "nonzero": round((nl + ns) / n, 6)}


def eval_unit(base, oscs, n, params, coins, weights):
    """Apply Chaikin directional entry gate (params=None -> no filter) + roll1 + accounting."""
    legs = {}
    turns = {}
    per_coin = {}
    tot_trades = 0
    for c in coins:
        pre, rt = base[c]
        if params is None:
            p = list(pre)
            cv = {"long": 1.0, "short": 0.0, "nonzero": 1.0}
        else:
            p = apply_chaikin_gate(pre, oscs[c])
            cs = cov_stats(oscs[c])
            cv = cs
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
        st["coverage_long"] = cv["long"]
        st["coverage_short"] = cv["short"]
        st["coverage"] = cv["nonzero"]
        per_coin[c] = st
    w = weights
    net = [sum(legs[c][1][t] * w[c] for c in coins) for t in range(n)]
    turn = [sum(turns[c][t] * w[c] for c in coins) for t in range(n)]
    full = seg(net, turn, 0, n)
    full["trades"] = tot_trades
    if params is None:
        full["coverage_long"] = 1.0
        full["coverage_short"] = 0.0
        full["coverage"] = 1.0
        full["coverage_all"] = 1.0
    else:
        fl = sum(per_coin[c]["coverage_long"] for c in coins) / len(coins)
        fs = sum(per_coin[c]["coverage_short"] for c in coins) / len(coins)
        fn = sum(per_coin[c]["coverage"] for c in coins) / len(coins)
        full["coverage_long"] = round(fl, 6)
        full["coverage_short"] = round(fs, 6)
        full["coverage"] = round(fn, 6)
        full["coverage_all"] = round(fn, 6)
    return full, per_coin


def dump(rows, units_done, n, coins, weights, params, complete):
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
            "filter": "Chaikin(slow,fast) directional entry-only sign gate: "
                      "MFM=((close-low)-(high-close))/(high-low) (0 if high==low), "
                      "MFV=MFM*volume, ADL=cumsum(MFV), osc=EMA_fast(ADL)-EMA_slow(ADL); "
                      "flat->long / short->long flip only if osc>0, flat->short / "
                      "long->short flip only if osc<0, osc==0 blocks entries/flips; "
                      "holds/exits pass; warmup first max(slow,fast) bars osc=0",
            "chaikin_params": [list(p) for p in params],
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
        "decision_note": "待定(P0-3 FAIL): Chaikin osc sign-gate只做诊断,不采用,不改live, "
                         "live basket与阈值保持locked原状。",
        "conclusion": ("PENDING (待定, P0-3 FAIL): 1h原生格子; Chaikin money-flow "
                       "oscillator sign gate diagnostic only, no adoption, no live change."),
    }
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return out


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").write("iter_H108_chaikin start smoke=%s\n" % SMOKE)
    coins = list(SMOKE_COINS) if SMOKE else list(COINS)
    params = list(SMOKE_PARAMS) if SMOKE else list(PARAMS)
    weights = {c: 1.0 / len(coins) for c in coins}
    bars, n = common1h(coins)
    log("common 1h native n=%d coins=%s params=%s scale=x%d"
        % (n, coins, params, SCALE))
    assert n > 2000, "grid too short: %d" % n
    base = {c: leg_base(bars[c], SPECS[c]) for c in coins}
    log("base legs built (locked engine, fee=%.4f fund=%.4f)" % (FEE, FUND))
    osc_cache = {}
    for (slow, fast) in params:
        osc_cache[(slow, fast)] = {c: chaikin_osc(bars[c], slow, fast) for c in coins}
        for c in coins:
            cs = cov_stats(osc_cache[(slow, fast)][c])
            log("%s Chaikin(%d,%d) long=%.4f short=%.4f" % (c, slow, fast, cs["long"], cs["short"]))

    rows = []
    units_done = []

    def run_unit(label, pr):
        oscs = None if pr is None else osc_cache[pr]
        full, per_coin = eval_unit(base, oscs, n, pr, coins, weights)
        row = {"unit": label, "params": None if pr is None else list(pr),
               "slow": None if pr is None else pr[0],
               "fast": None if pr is None else pr[1],
               "FULL": full, "per_coin": per_coin}
        rows.append(row)
        units_done.append(label)
        dump(rows, units_done, n, coins, weights, params, complete=False)
        log("%s FULL sh=%.3f mdd=%.4f cum=%.4f final_x=%.4f ann=%.4f n=%d to=%.6f trades=%d cov=%.4f (L=%.4f/S=%.4f)"
            % (label, full["sharpe"], full["mdd"], full["cum"], full["final_x"],
               full["ann"], full["n"], full["turnover"], full["trades"],
               full["coverage"], full["coverage_long"], full["coverage_short"]))

    run_unit("base", None)
    for pr in params:
        run_unit("chaikin_%d_%d" % pr, pr)
    dump(rows, units_done, n, coins, weights, params, complete=True)
    log("wrote %s verdict=PENDING decision=NO_ADOPTION" % OUT)


if __name__ == "__main__":
    main()
