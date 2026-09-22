"""H113 1h BB Bollinger Bands %B entry gate (Top5) -- DIAGNOSTIC ONLY.

P0-3 permutation FAILED (per-coin p>=0.05, results/permutation.json), so per
the PRP global exit rule this step's conclusion is "PENDING (待定)" and MUST NOT be
used as demo-listing evidence. No adoption, no live change, live chain untouched.
Offline read-only: reads data/data_1y/1h/*.csv only.

Engine mirrors research/run_weight_modes.py leg_net: E10 FORMULA
[3,2,7,2,7,11,15,4,4,6,6,10] via StackVM+FeatureEngineer; MemeBacktest
venue=aster lev2 short_enabled fund0.0005 fee0.0004; quantile q0.3
long-only mask + cooldown + stops + vol_scale(vt None->1.0) + roll1. Static equal 0.2
weights (Top5 locked specs ETC/TRX/ATOM/APT/KAS).

1h native: cd/ts/vw x4 (4h-bar units -> 1h-bar units), BPY=8760.
Data data/data_1y/1h/{COIN}.csv (~8760 rows,
cols timestamp,open,high,low,close,volume,quote_volume,trades).
Equal 0.2 weights.

Bollinger %B entry-only: long entries/flips allowed only when %B<0.2 (near lower band), short entries/flips only when %B>0.8 (near upper band); %B=(close-SMA_lower)/(SMA_upper-SMA_lower); holds/exits pass
Sweep params in [(20, 2.0), (50, 2.0)] vs no-filter base.

Metric per cell: FULL sharpe/mdd/cum/final_x/ann/n/turnover/trades +
coverage (fraction of long-open bars + short-open bars) + per-coin
single-leg sensitivity (sharpe/turnover/trades/coverage per coin per params).

IMPORTANT: results/iter_H113_bb.json is dumped after EACH unit
(base + 2 param cells; "complete": false until the final write), so a
killed run still leaves partial rows behind.

Outputs: results/iter_H113_bb.json (+ logs/iter_H113_bb.log).
Offline read-only. Verdict PENDING (P0-3 FAIL); no adoption.

Smoke mode (for tests): ITER_H113_SMOKE=1 shrinks to [(20, 2.0)] on coins
{ETC,TRX} only. ITER_H113_OUT / ITER_H113_LOG override output paths (tests
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

from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest
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
PARAMS = [(20, 2.0), (50, 2.0)]
BPY = 8760.0
SCALE = 4

OUT = pathlib.Path(os.getenv("ITER_H113_OUT", "results/iter_H113_bb.json"))
LOG = pathlib.Path(os.getenv("ITER_H113_LOG", "logs/iter_H113_bb.log"))

SMOKE = os.getenv("ITER_H113_SMOKE") == "1"
SMOKE_COINS = ["ETC", "TRX"]
SMOKE_PARAMS = [(20, 2.0)]


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
    bars = {}
    n = None
    for c in coins:
        bars[c] = load1h(c)
        if n is None:
            n = len(bars[c])
        else:
            assert len(bars[c]) == n, "length mismatch %s vs %s" % (c, n)
    return bars, n


def qmask(sig, q):
    if q is None or float(q) <= 0:
        return None
    a = sig.detach().float().abs().reshape(-1)
    k = max(1, int(len(a) * float(q)))
    thr = torch.topk(a, k).values.min()
    return (sig.detach().float().abs() >= thr).float()


# Filter-specific gate functions
def bb_gates(bars_c, period, std_mult):
    """Causal Bollinger Bands %B entry gates. Returns (long_ok, short_ok) bool lists."""
    cl = [b[3] for b in bars_c]
    n = len(cl)
    sma = [0.0] * n
    std = [0.0] * n
    for t in range(n):
        if t < period - 1:
            sma[t] = cl[t]
            std[t] = 0.0
        else:
            window = cl[t - period + 1:t + 1]
            sma[t] = sum(window) / period
            var = sum((x - sma[t]) ** 2 for x in window) / period
            std[t] = math.sqrt(var)
    upper = [sma[t] + std_mult * std[t] for t in range(n)]
    lower = [sma[t] - std_mult * std[t] for t in range(n)]
    pctb = [0.5] * n
    for t in range(n):
        if upper[t] != lower[t]:
            pctb[t] = (cl[t] - lower[t]) / (upper[t] - lower[t])
    long_ok = [pctb[t] < 0.2 for t in range(n)]
    short_ok = [pctb[t] > 0.8 for t in range(n)]
    return long_ok, short_ok


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


def apply_entry_gate(pre, long_ok, short_ok):
    """Entry-only directional gate on pre-roll positions.

    pre==0 -> flat always passes (exits preserved). Flat->long entry needs
    long_ok[t]; flat->short entry needs short_ok[t]. Long<->short flips need
    the target side's gate; a blocked flip holds the previous side (no
    forced exit). Same-sign continuations always pass.
    """
    n = len(pre)
    g = [0.0] * n
    prev = 0.0
    for t in range(n):
        cur = pre[t]
        if cur == 0.0:
            g[t] = 0.0
        elif cur > 0:
            if prev > 0:
                g[t] = cur
            else:
                g[t] = cur if long_ok[t] else (prev if prev < 0 else 0.0)
        else:
            if prev < 0:
                g[t] = cur
            else:
                g[t] = cur if short_ok[t] else (prev if prev > 0 else 0.0)
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


def eval_unit(base, gates, n, param, coins, weights):
    """Apply directional entry gate (param=None -> no filter) + roll1 + accounting."""
    legs = {}
    turns = {}
    per_coin = {}
    tot_trades = 0
    for c in coins:
        pre, rt = base[c]
        if param is None:
            p = list(pre)
            cov_l = 1.0
            cov_s = 1.0
            cov = 1.0
        else:
            long_ok, short_ok = gates[c]
            p = apply_entry_gate(pre, long_ok, short_ok)
            cov_l = sum(1 for v in long_ok if v) / n if n else 0.0
            cov_s = sum(1 for v in short_ok if v) / n if n else 0.0
            cov = (cov_l + cov_s) / 2.0
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
        st["coverage"] = round(cov, 6)
        st["coverage_long"] = round(cov_l, 6)
        st["coverage_short"] = round(cov_s, 6)
        per_coin[c] = st
    w = weights
    net = [sum(legs[c][1][t] * w[c] for c in coins) for t in range(n)]
    turn = [sum(turns[c][t] * w[c] for c in coins) for t in range(n)]
    full = seg(net, turn, 0, n)
    full["trades"] = tot_trades
    if param is None:
        cov_all = 1.0
    else:
        cov_all = sum(per_coin[c]["coverage"] for c in coins) / len(coins) if coins else 0.0
    full["coverage_all"] = round(cov_all, 6)
    return full, per_coin


def build_sig(bars):
    """Build signal via E10 engine (StackVM + FeatureEngineer)."""
    n = len(bars)
    raw = {"open": torch.tensor([[b[1] for b in bars]]),
           "high": torch.tensor([[b[2] for b in bars]]),
           "low": torch.tensor([[b[3] for b in bars]]),
           "close": torch.tensor([[b[4] for b in bars]]),
           "volume": torch.tensor([[b[5] for b in bars]]),
           "liquidity": torch.full((1, n), 1e7), "fdv": torch.full((1, n), 1e8)}
    fe = FeatureEngineer.compute_features(raw, use_advanced=False)
    sig = StackVM(use_advanced=False).execute(FORMULA, fe)
    rets = [(bars[i + 1][4] - bars[i][4]) / bars[i][4] for i in range(n - 1)] + [0.0]
    return raw, torch.tensor([rets]), sig

def dump(rows, units_done, n, coins, weights, params_list, complete):
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
            "filter": "Bollinger %B entry-only: long entries/flips allowed only when %B<0.2 (near lower band), short entries/flips only when %B>0.8 (near upper band); %B=(close-SMA_lower)/(SMA_upper-SMA_lower); holds/exits pass",
            "params": [[20, 2.0], [50, 2.0]],
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
        "decision_note": "待定(P0-3 FAIL): BB entry過濾只做診斷,不採用,不改live, live basket與閾值保持locked原狀。",
        "conclusion": ("PENDING (待定, P0-3 FAIL): 1h原生格子; BB entry "
                       "filter diagnostic only, no adoption, no live change."),
    }
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return out


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").write("iter_H113_bb start smoke=%s\n" % SMOKE)
    coins = list(SMOKE_COINS) if SMOKE else list(COINS)
    params_list = list(SMOKE_PARAMS) if SMOKE else list(PARAMS)
    weights = {c: 1.0 / len(coins) for c in coins}
    bars, n = common1h(coins)
    log("common 1h native n=%d coins=%s params=%s scale=x%d"
        % (n, coins, params_list, SCALE))
    assert n > 2000, "grid too short: %d" % n
    base = {c: leg_base(bars[c], SPECS[c]) for c in coins}
    log("base legs built (locked engine, fee=%.4f fund=%.4f)" % (FEE, FUND))
    gates_cache = {}
    for pr in params_list:
        gates_cache[pr] = {c: bb_gates(bars[c], *pr) for c in coins}
        for c in coins:
            lo, so = gates_cache[pr][c]
            cl = sum(1 for v in lo if v) / n
            cs = sum(1 for v in so if v) / n
            log("%s bb_gates(%s) long_cov=%.4f short_cov=%.4f" % (c, pr, cl, cs))

    rows = []
    units_done = []

    def run_unit(label, param):
        gates = None if param is None else gates_cache[param]
        full, per_coin = eval_unit(base, gates, n, param, coins, weights)
        row = {"unit": label, "param": list(param) if param else None, "FULL": full,
               "per_coin": per_coin}
        rows.append(row)
        units_done.append(label)
        dump(rows, units_done, n, coins, weights, params_list, complete=False)
        log("%s FULL sh=%.3f mdd=%.4f cum=%.4f to=%.6f trades=%d cov=%.4f"
            % (label, full["sharpe"], full["mdd"], full["cum"],
               full["turnover"], full["trades"], full["coverage_all"]))

    run_unit("base", None)
    for pr in params_list:
        run_unit("bb_%s" % str(pr).replace("(", "").replace(")", "").replace(", ", "_"), pr)
    dump(rows, units_done, n, coins, weights, params_list, complete=True)
    log("wrote %s verdict=PENDING decision=NO_ADOPTION" % OUT)


if __name__ == "__main__":
    main()
