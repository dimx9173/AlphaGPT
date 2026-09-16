"""H78 1h Parabolic SAR filter (Top5) -- DIAGNOSTIC ONLY.

P0-3 permutation FAILED (per-coin p>=0.05, results/permutation.json), so per
the PRP global exit rule this step's conclusion is "PENDING" and MUST NOT be
used as demo-listing evidence. No adoption, no live change, live untouched.
Offline read-only: reads data/data_1y/1h/*.csv only.

Engine mirrors research/run_weight_modes.py leg_net: E10 FORMULA
[3,2,7,2,7,11,15,4,4,6,6,10] via StackVM+FeatureEngineer; MemeBacktest
venue=aster lev2 short_enabled fund0.0005; quantile q0.3 long-only +
cooldown + stops + vol_scale(vt None->1.0) + roll1. Static equal 0.2
weights (Top5 locked specs ETC/TRX/ATOM/APT/KAS).

1h native: cd/ts/vw x4 (4h-bar units -> 1h-bar units), BPY=8760. Data
data/data_1y/1h/{COIN}.csv (~8760 rows, cols
timestamp,open,high,low,close,volume,quote_volume,trades). Equal 0.2 weights.

SAR filter semantics: causal Parabolic SAR per coin on native 1h bars
(standard AF step/max, prior-bar clamp, reversal on current-bar high/low;
all inputs known at bar close, gate applied before roll1 so execution is
next bar). Directional gate: LONG kept only where SAR < close (SAR below
price = uptrend), SHORT kept only where SAR > close (SAR above price =
downtrend). Gate applied to lp/sp after cooldown+stops+vol_scale, before
roll1. Sweep (step,maxaf) in {(0.01,0.10),(0.02,0.20),(0.04,0.20)} vs
no-filter base.

Metric per unit: FULL sharpe/mdd/cum/final_x/ann/n/turnover/trades +
long_cov/short_cov (mean per-coin fraction of bars passing each side) +
per-coin single-leg sensitivity (sharpe/turnover/trades/long_cov/short_cov
per coin per param cell).

IMPORTANT: results/iter_H78_sar.json is dumped after EACH unit
(base + 3 param cells; "complete": false until the final write), so a
killed run still leaves partial rows behind.

Outputs: results/iter_H78_sar.json (+ logs/iter_h78_sar.log).
Offline read-only. Verdict PENDING (P0-3 FAIL); no adoption.

Smoke mode (for tests): ITER_H78_SMOKE=1 shrinks to params {(0.02,0.20)}
on coins {ETC,TRX} only. ITER_H78_OUT / ITER_H78_LOG override output paths
(tests use temp files so the committed FULL artifact is not clobbered).
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
SAR_PARAMS = [(0.01, 0.10), (0.02, 0.20), (0.04, 0.20)]
BPY = 8760.0
SCALE = 4

OUT = pathlib.Path(os.getenv("ITER_H78_OUT", "results/iter_H78_sar.json"))
LOG = pathlib.Path(os.getenv("ITER_H78_LOG", "logs/iter_h78_sar.log"))

SMOKE = os.getenv("ITER_H78_SMOKE") == "1"
SMOKE_COINS = ["ETC", "TRX"]
SMOKE_PARAMS = [(0.02, 0.20)]


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


def psar(bars_c, step=0.02, maxaf=0.20):
    """Causal Parabolic SAR on 1h bars; returns (sar list, close list).

    Standard AF scheme: af starts at step, += step on each new extreme,
    capped at maxaf. SAR[t] uses prior SAR/EP plus prior-bar clamp
    (min of prior lows in uptrend / max of prior highs in downtrend);
    reversal checked against the current bar high/low (known at bar close).
    """
    hi = [b[1] for b in bars_c]
    lo = [b[2] for b in bars_c]
    cl = [b[3] for b in bars_c]
    n = len(cl)
    sar = [0.0] * n
    if n == 0:
        return [], []
    if n == 1:
        return [lo[0]], list(cl)
    up = cl[1] >= cl[0]
    ep = hi[0] if up else lo[0]
    s = lo[0] if up else hi[0]
    af = float(step)
    sar[0] = s
    for t in range(1, n):
        s = s + af * (ep - s)
        if up:
            s = min(s, lo[t - 1], lo[t - 2] if t >= 2 else lo[t - 1])
            if lo[t] < s:
                up = False
                s = ep
                ep = lo[t]
                af = float(step)
            else:
                if hi[t] > ep:
                    ep = hi[t]
                    af = min(af + float(step), float(maxaf))
        else:
            s = max(s, hi[t - 1], hi[t - 2] if t >= 2 else hi[t - 1])
            if hi[t] > s:
                up = True
                s = ep
                ep = hi[t]
                af = float(step)
            else:
                if lo[t] < ep:
                    ep = lo[t]
                    af = min(af + float(step), float(maxaf))
        sar[t] = s
    return sar, list(cl)


def leg_base(bars_c, spec):
    """Locked-engine pre-roll lp/sp + returns (1h native)."""
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
    return lp[0].tolist(), sp[0].tolist(), rt[0].tolist()


def count_trades_dir(lp, sp):
    tl, ts, pl, ps = 0, 0, 0.0, 0.0
    for a, b in zip(lp, sp):
        cl = 1.0 if abs(a) > 0.5 else 0.0
        cs = 1.0 if abs(b) > 0.5 else 0.0
        if cl > 0.5 > pl:
            tl += 1
        if cs > 0.5 > ps:
            ts += 1
        pl, ps = cl, cs
    return tl + ts


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


def eval_unit(base, sars, closes, n, params, coins, weights):
    """Apply directional SAR gate (params=None -> no filter) + roll1 + aster."""
    legs = {}
    turns = {}
    per_coin = {}
    tot_trades = 0
    lcs, scs = [], []
    for c in coins:
        lp0, sp0, rt = base[c]
        if params is None:
            gl = [1.0] * n
            gs = [1.0] * n
        else:
            sar = sars[c]
            cl = closes[c]
            gl = [1.0 if sar[t] < cl[t] else 0.0 for t in range(n)]
            gs = [1.0 if sar[t] > cl[t] else 0.0 for t in range(n)]
        lp = [lp0[t] * gl[t] for t in range(n)]
        sp = [sp0[t] * gs[t] for t in range(n)]
        p = [lp[t] - sp[t] for t in range(n)]
        rolled = [0.0] + p[:-1]
        dp = [abs(rolled[t] - (rolled[t - 1] if t else 0.0)) for t in range(n)]
        net_c = [rolled[t] * rt[t] * LEV - dp[t] * FEE * LEV
                 - rolled[t] * FUND * LEV for t in range(n)]
        legs[c] = (rolled, net_c, rt)
        turns[c] = dp
        tr = count_trades_dir(lp, sp)
        tot_trades += tr
        lc = round(sum(gl) / n, 6)
        sc = round(sum(gs) / n, 6)
        lcs.append(lc)
        scs.append(sc)
        st = seg(net_c, dp, 0, n)
        st["trades"] = tr
        st["long_cov"] = lc
        st["short_cov"] = sc
        per_coin[c] = st
    w = weights
    net = [sum(legs[c][1][t] * w[c] for c in coins) for t in range(n)]
    turn = [sum(turns[c][t] * w[c] for c in coins) for t in range(n)]
    full = seg(net, turn, 0, n)
    full["trades"] = tot_trades
    full["long_cov"] = round(sum(lcs) / len(lcs), 6) if lcs else 0.0
    full["short_cov"] = round(sum(scs) / len(scs), 6) if scs else 0.0
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
            "filter": "Parabolic SAR directional gate: LONG only where SAR < "
                      "close, SHORT only where SAR > close (causal, prior-bar "
                      "clamp, reversal on bar high/low)",
            "sar_params": [[a, b] for a, b in params],
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
        "decision_note": "\u5f85\u5b9a(P0-3 FAIL): SAR\u8fc7\u6ee4\u53ea\u505a\u8bca\u65ad,\u4e0d\u91c7\u7528,\u4e0d\u6539live, "
                         "live basket\u4e0e\u9608\u503c\u4fdd\u6301locked\u539f\u72b6\u3002",
        "conclusion": ("PENDING (\u5f85\u5b9a, P0-3 FAIL): 1h\u539f\u751f\u683c\u5b50; SAR filter "
                       "diagnostic only, no adoption, no live change."),
    }
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return out


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").write("iter_H78_sar start smoke=%s\n" % SMOKE)
    coins = list(SMOKE_COINS) if SMOKE else list(COINS)
    params = list(SMOKE_PARAMS) if SMOKE else list(SAR_PARAMS)
    weights = {c: 1.0 / len(coins) for c in coins}
    bars, n = common1h(coins)
    log("common 1h native n=%d coins=%s params=%s scale=x%d"
        % (n, coins, params, SCALE))
    assert n > 2000, "grid too short: %d" % n
    base = {c: leg_base(bars[c], SPECS[c]) for c in coins}
    log("base legs built (locked engine, fee=%.4f fund=%.4f)" % (FEE, FUND))
    sars, closes = {}, {}
    for c in coins:
        cov = {}
        for (a, b) in params:
            s, cl = psar(bars[c], a, b)
            cov[(a, b)] = (sum(1 for t in range(n) if s[t] < cl[t]) / n,
                           sum(1 for t in range(n) if s[t] > cl[t]) / n)
        sars[c] = {p: psar(bars[c], *p)[0] for p in params}
        closes[c] = [b[3] for b in bars[c]]
        for (a, b) in params:
            lc, sc = cov[(a, b)]
            log("%s SAR(%.2f,%.2f) long_cov=%.3f short_cov=%.3f"
                % (c, a, b, lc, sc))

    rows = []
    units_done = []

    def run_unit2(label, p):
        if p is None:
            sar_map = {c: None for c in coins}
        else:
            sar_map = {c: sars[c][p] for c in coins}
        full, per_coin = eval_unit(base, sar_map, closes, n, p, coins, weights)
        row = {"unit": label,
               "sar": None if p is None else [p[0], p[1]],
               "FULL": full, "per_coin": per_coin}
        rows.append(row)
        units_done.append(label)
        dump(rows, units_done, n, coins, weights, params, complete=False)
        log("%s FULL sh=%.3f mdd=%.4f cum=%.4f to=%.6f trades=%d lcov=%.4f scov=%.4f"
            % (label, full["sharpe"], full["mdd"], full["cum"],
               full["turnover"], full["trades"], full["long_cov"],
               full["short_cov"]))

    rows.clear()
    units_done.clear()
    run_unit2("base", None)
    for p in params:
        run_unit2("sar_af%.2f_max%.2f" % p, p)
    dump(rows, units_done, n, coins, weights, params, complete=True)
    log("wrote %s verdict=PENDING decision=NO_ADOPTION" % OUT)


if __name__ == "__main__":
    main()
