"""H97 1h Ichimoku v2 cloud filter (Top5) -- DIAGNOSTIC ONLY.

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

Ichimoku filter semantics: causal Ichimoku(18,52,104, disp 52) v2 per coin on
native 1h bars. Cloud at bar t is built from data up to t-52
(spanA[t] = (tenkan[t-52]+kijun[t-52])/2, spanB[t] = snb104[t-52]);
first valid bar is t=155 (0-indexed), seed bars -> filtered (no trade).
Directional gate applied to lp/sp AFTER cooldown+stops+vol_scale, BEFORE
roll1: long leg requires close above cloud, short leg requires close below
cloud, inside-cloud bars are flat for the gated side.

Sweep modes (ablation): base (no filter) | ichi_long (long gated, short
unchanged) | ichi_short (short gated, long unchanged) | ichi_both
(long above cloud AND short below cloud).

Metric per cell: FULL sharpe/mdd/cum/final_x/ann/n/turnover/trades +
coverage (fraction of bars passing gate) + per-coin single-leg
sensitivity (sharpe/turnover/trades/coverage per coin per mode).

IMPORTANT: results/iter_H97_ichi.json is dumped after EACH unit
("complete": false until the final write), so a killed run still leaves
partial rows behind.

Outputs: results/iter_H97_ichi.json (+ logs/iter_h97_ichi.log).
Offline read-only. Verdict PENDING (P0-3 FAIL); no adoption.

Smoke mode (for tests): ITER_H97_SMOKE=1 shrinks to modes
{base, ichi_both} on coins {ETC,TRX} only. ITER_H97_OUT / ITER_H97_LOG
override output paths (tests use temp files so the committed FULL
artifact is not clobbered).
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
assert list(FORMULA) == [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10], "E10 FORMULA lock broken"

COINS = ["ETC", "TRX", "ATOM", "APT", "KAS"]
SPECS = {"ETC": LOCKED_ETC, "TRX": LOCKED_TRX, "ATOM": LOCKED_ATOM,
         "APT": LOCKED_APT, "KAS": LOCKED_KAS}
MODES = ["base", "ichi_long", "ichi_short", "ichi_both"]
TENKAN_P, KIJUN_P, SENKOU_P, DISP = 18, 52, 104, 52
FIRST_VALID = SENKOU_P + DISP - 1  # 155 (0-indexed)
BPY = 8760.0
SCALE = 4

OUT = pathlib.Path(os.getenv("ITER_H97_OUT", "results/iter_H97_ichi.json"))
LOG = pathlib.Path(os.getenv("ITER_H97_LOG", "logs/iter_h97_ichi.log"))

SMOKE = os.getenv("ITER_H97_SMOKE") == "1"
SMOKE_COINS = ["ETC", "TRX"]
SMOKE_MODES = ["base", "ichi_both"]


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


def ichi_gates(bars_c, ten=TENKAN_P, kij=KIJUN_P, snb=SENKOU_P, disp=DISP):
    """Causal Ichimoku gates on 1h bars.

    Returns (above, below): above[t]=1 iff close[t] above the causal
    cloud, below[t]=1 iff close[t] below it. Bars before FIRST_VALID
    (t < snb+disp-1) are seed -> (0, 0).
    """
    hi = numpy.array([b[1] for b in bars_c], dtype=float)
    lo = numpy.array([b[2] for b in bars_c], dtype=float)
    cl = numpy.array([b[3] for b in bars_c], dtype=float)
    n = len(cl)
    above = numpy.zeros(n)
    below = numpy.zeros(n)
    if n < snb + disp:
        return above.tolist(), below.tolist()
    tenkan = numpy.full(n, numpy.nan)
    kijun = numpy.full(n, numpy.nan)
    snb_raw = numpy.full(n, numpy.nan)
    for t in range(ten - 1, n):
        tenkan[t] = (hi[t - ten + 1:t + 1].max() + lo[t - ten + 1:t + 1].min()) / 2.0
    for t in range(kij - 1, n):
        kijun[t] = (hi[t - kij + 1:t + 1].max() + lo[t - kij + 1:t + 1].min()) / 2.0
    for t in range(snb - 1, n):
        snb_raw[t] = (hi[t - snb + 1:t + 1].max() + lo[t - snb + 1:t + 1].min()) / 2.0
    for t in range(snb + disp - 1, n):
        sa = (tenkan[t - disp] + kijun[t - disp]) / 2.0
        sb = snb_raw[t - disp]
        top = sa if sa >= sb else sb
        bot = sb if sa >= sb else sa
        if cl[t] > top:
            above[t] = 1.0
        elif cl[t] < bot:
            below[t] = 1.0
    return above.tolist(), below.tolist()


def leg_base(bars_c, spec):
    """Locked-engine pre-roll long/short legs + returns (1h native)."""
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


def side_gates(mode, above, below, n):
    """(long_gate, short_gate, pass_mask) for a sweep mode."""
    one = [1.0] * n
    if mode == "base":
        return one, one, one
    if mode == "ichi_long":
        return above, one, above
    if mode == "ichi_short":
        return one, below, below
    if mode == "ichi_both":
        both = [1.0 if (above[t] > 0.5 or below[t] > 0.5) else 0.0
                for t in range(n)]
        return above, below, both
    raise ValueError("unknown mode %r" % mode)


def eval_unit(base, clouds, n, mode, coins, weights):
    """Apply Ichimoku directional gates + roll1 + aster accounting."""
    legs = {}
    turns = {}
    per_coin = {}
    tot_trades = 0
    for c in coins:
        lp0, sp0, rt = base[c]
        gl, gs, pm = side_gates(mode, clouds[c][0], clouds[c][1], n)
        lp = [lp0[t] * gl[t] for t in range(n)]
        sp = [sp0[t] * gs[t] for t in range(n)]
        p = [lp[t] - sp[t] for t in range(n)]
        rolled = [0.0] + p[:-1]
        dp = [abs(rolled[t] - (rolled[t - 1] if t else 0.0)) for t in range(n)]
        net_c = [rolled[t] * rt[t] * LEV - dp[t] * FEE * LEV
                 - rolled[t] * FUND * LEV for t in range(n)]
        legs[c] = (rolled, net_c, rt, pm)
        turns[c] = dp
        tr = count_trades(rolled)
        tot_trades += tr
        st = seg(net_c, dp, 0, n)
        st["trades"] = tr
        st["coverage"] = round(sum(pm) / n, 6)
        st["coverage_long"] = round(sum(gl) / n, 6)
        st["coverage_short"] = round(sum(gs) / n, 6)
        per_coin[c] = st
    w = weights
    net = [sum(legs[c][1][t] * w[c] for c in coins) for t in range(n)]
    turn = [sum(turns[c][t] * w[c] for c in coins) for t in range(n)]
    full = seg(net, turn, 0, n)
    full["trades"] = tot_trades
    cov = sum(1.0 for t in range(n)
              if all(legs[c][3][t] > 0.5 for c in coins)) / n if n else 0.0
    full["coverage_all"] = round(cov, 6)
    full["coverage_long"] = round(
        sum(per_coin[c]["coverage_long"] * w[c] for c in coins), 6)
    full["coverage_short"] = round(
        sum(per_coin[c]["coverage_short"] * w[c] for c in coins), 6)
    return full, per_coin


def dump(rows, units_done, n, coins, weights, modes, complete):
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
            "filter": "Ichimoku(18,52,104,disp52) v2 directional gate: long only "
                      "above cloud, short only below cloud (causal; "
                      "seed=filtered)",
            "ichimoku": {"tenkan": TENKAN_P, "kijun": KIJUN_P,
                         "senkou": SENKOU_P, "displacement": DISP,
                         "first_valid_bar": FIRST_VALID},
            "modes": list(modes),
            "trades_def": "opened entries flat->nonzero (H58 sibling-count)",
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
        "decision_note": "待定(P0-3 FAIL): Ichimoku云过滤只做诊断,不采用,不改live, "
                         "live basket与阈值保持locked原状。",
        "conclusion": ("PENDING (待定, P0-3 FAIL): 1h原生格子; Ichimoku filter "
                       "diagnostic only, no adoption, no live change."),
    }
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return out


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").write("iter_H97_ichi start smoke=%s\n" % SMOKE)
    coins = list(SMOKE_COINS) if SMOKE else list(COINS)
    modes = list(SMOKE_MODES) if SMOKE else list(MODES)
    weights = {c: 1.0 / len(coins) for c in coins}
    bars, n = common1h(coins)
    log("common 1h native n=%d coins=%s modes=%s scale=x%d"
        % (n, coins, modes, SCALE))
    assert n > 2000, "grid too short: %d" % n
    base = {c: leg_base(bars[c], SPECS[c]) for c in coins}
    log("base legs built (locked engine, fee=%.4f fund=%.4f)" % (FEE, FUND))
    clouds = {c: ichi_gates(bars[c]) for c in coins}
    for c in coins:
        ab, be = clouds[c]
        log("%s above=%.4f below=%.4f inside=%.4f" %
            (c, sum(ab) / n, sum(be) / n,
             1.0 - (sum(ab) + sum(be)) / n))

    rows = []
    units_done = []

    def run_unit(mode):
        full, per_coin = eval_unit(base, clouds, n, mode, coins, weights)
        row = {"unit": mode, "mode": mode, "FULL": full,
               "per_coin": per_coin}
        rows.append(row)
        units_done.append(mode)
        dump(rows, units_done, n, coins, weights, modes, complete=False)
        log("%s FULL sh=%.3f mdd=%.4f cum=%.4f to=%.6f trades=%d cov=%.4f"
            % (mode, full["sharpe"], full["mdd"], full["cum"],
               full["turnover"], full["trades"], full["coverage_all"]))

    for mode in modes:
        run_unit(mode)
    dump(rows, units_done, n, coins, weights, modes, complete=True)
    log("wrote %s verdict=PENDING decision=NO_ADOPTION" % OUT)


if __name__ == "__main__":
    main()
