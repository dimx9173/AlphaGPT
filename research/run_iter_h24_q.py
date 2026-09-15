"""H24 1h quantile sweep (Top5) -- DIAGNOSTIC ONLY.

P0-3 permutation FAILED (per-coin p>=0.05), so per the PRP global exit
rule this step's conclusion is "PENDING" and MUST NOT be used as
demo-listing evidence. No adoption, no live change, live untouched.
Offline read-only: reads data/data_1y/1h/*.csv only.

Engine mirrors research/run_weight_modes.py leg_net: E10 FORMULA
[3,2,7,2,7,11,15,4,4,6,6,10] via StackVM+FeatureEngineer; MemeBacktest
venue=aster lev2 short_enabled fund0.0005; quantile q long-only mask +
cooldown + stops + vol_scale(vt None->1.0) + roll1. The sweep overrides
the locked q0.3 per row; everything else stays locked.

Top5 locked specs (4h units): ETC(0.88/0.12/cd18/None/ts24)
TRX(0.85/0.12/cd6/0.05/ts24) ATOM(0.85/0.15/cd6/0.05/ts24)
APT(0.88/0.12/cd18/None/ts24) KAS(0.88/0.12/cd6/None/ts24).
Equal 0.2 weights.

1h native: data/data_1y/1h/{COIN}.csv direct (~8760 rows:
timestamp,open,high,low,close,volume,quote_volume,trades);
cd/ts/vw x4 (4h-bar units -> 1h-bar units); BPY=8760.

Sweep: q in {0.1,0.2,0.3,0.4,0.5} -> per-q FULL sharpe/turnover (+ann/
mdd/cum/final_x/n) + long/short attribution (entries + net-PnL share)
+ FULL_fee2x sharpe. fee2x knee = max perpendicular distance from the
chord on (q, FULL fee2x sharpe). Decision forced KEEP_q0.3 (no adoption).

Output: results/iter_H24_q.json (+ logs/iter_H24_q.log).
Incremental dump: results JSON rewritten after each unit (partial survives).

Smoke mode (for tests): ITER_H24_SMOKE=1 shrinks to coins {ETC,TRX},
first 3000 bars, q {0.1,0.3,0.5}. ITER_H24_OUT / ITER_H24_LOG override
output paths (tests use a temp file so the committed FULL artifact is
not clobbered).
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
    LOCKED_ATOM,
    LOCKED_APT,
    LOCKED_ETC,
    LOCKED_KAS,
    LOCKED_TRX,
    LEV,
    FUND,
    FEE,
    FEE2X,
)

assert LEV == 2.0, "LEV lock broken: %r" % LEV

COINS = ["ETC", "TRX", "ATOM", "APT", "KAS"]
BASE_SPECS = {
    "ETC": dict(LOCKED_ETC),
    "TRX": dict(LOCKED_TRX),
    "ATOM": dict(LOCKED_ATOM),
    "APT": dict(LOCKED_APT),
    "KAS": dict(LOCKED_KAS),
}
W = {c: 0.2 for c in COINS}
QS = [0.1, 0.2, 0.3, 0.4, 0.5]
SMOKE_QS = [0.1, 0.3, 0.5]
SMOKE_COINS = ["ETC", "TRX"]
SMOKE_BARS = 3000
BPY = 8760.0
SCALE = 4  # 4h-bar units -> 1h-bar units

OUT = pathlib.Path(os.getenv("ITER_H24_OUT", "results/iter_H24_q.json"))
LOG = pathlib.Path(os.getenv("ITER_H24_LOG", "logs/iter_H24_q.log"))
SMOKE = os.getenv("ITER_H24_SMOKE") == "1"

_partial = {}


def log(m):
    print(m, flush=True)
    open(LOG, "a").write(m + "\n")


def dump():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(_partial, indent=1, ensure_ascii=False))


def load1h(c):
    rows = list(csv.DictReader(open("data/data_1y/1h/%s.csv" % c)))
    return [(int(r["timestamp"]), float(r["open"]), float(r["high"]),
             float(r["low"]), float(r["close"]), float(r["volume"]))
            for r in rows]


def common1h(coins):
    """Timestamp-intersected native 1h bars (no aggregation)."""
    raw = {c: load1h(c) for c in coins}
    ts = set(r[0] for r in raw[coins[0]])
    for c in coins[1:]:
        ts &= set(r[0] for r in raw[c])
    ts_sorted = sorted(ts)
    bars = {}
    for c in coins:
        m = {r[0]: r for r in raw[c]}
        bars[c] = [(m[t][1], m[t][2], m[t][3], m[t][4], m[t][5])
                   for t in ts_sorted]
    return ts_sorted, bars


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


def leg_pnl(raw, rt, sig, spec, fee, fund, q):
    """Mirror research/run_weight_modes.py leg_net, 1h-native params."""
    bt = MemeBacktest(venue="aster", leverage=LEV, short_enabled=True,
                      funding_override=fund, fee_override=fee,
                      long_th=spec["lth"], short_th=spec["sth"],
                      cooldown_bars=int(spec["cd"]) * SCALE,
                      bars_per_year=BPY, stop_loss=spec["sl"],
                      time_stop=int(spec["ts"]) * SCALE,
                      vol_target=spec["vt"],
                      vol_window=int(spec["vw"]) * SCALE)
    sg = torch.sigmoid(sig)
    safe = (raw["liquidity"] > bt.min_liq).float()
    lp = (sg > bt.long_th).float() * safe
    sp = (sg < bt.short_th).float() * safe
    mk = qmask(sig, q)
    if mk is not None:
        lp = lp * mk
    lp, sp = bt._apply_cooldown(lp, sp)
    lp, sp = bt._apply_stops(lp, sp, rt)
    sc = bt._vol_scale(rt)
    lp, sp = lp * sc, sp * sc
    lp = lp.roll(1, dims=1)
    lp[:, 0] = 0
    sp = sp.roll(1, dims=1)
    sp[:, 0] = 0
    turn = (lp - lp.roll(1, dims=1)).abs() + (sp - sp.roll(1, dims=1)).abs()
    lturn = (lp - lp.roll(1, dims=1)).abs()
    sturn = (sp - sp.roll(1, dims=1)).abs()
    rate = bt.base_fee
    gross = (lp - sp) * rt * bt.leverage
    tx = turn * rate
    fnd = (lp - sp) * bt.default_funding_rate * bt.leverage
    net = (gross - tx * bt.leverage - fnd)[0].tolist()
    lp_l = lp[0].tolist()
    sp_l = sp[0].tolist()
    lt_l = lturn[0].tolist()
    st_l = sturn[0].tolist()
    rt_l = rt[0].tolist()
    long_net, short_net = [], []
    for t in range(len(net)):
        ln = lp_l[t] * rt_l[t] * LEV - lt_l[t] * rate * LEV - lp_l[t] * fund * LEV
        sn = -sp_l[t] * rt_l[t] * LEV - st_l[t] * rate * LEV + sp_l[t] * fund * LEV
        long_net.append(ln)
        short_net.append(sn)
    pos = (lp - sp)[0].tolist()
    entries_l = entries_s = 0
    prev = 0.0
    for v in pos:
        cur = 1.0 if v > 0.5 else (-1.0 if v < -0.5 else 0.0)
        if cur != 0.0 and prev == 0.0:
            if cur > 0:
                entries_l += 1
            else:
                entries_s += 1
        prev = cur
    return {"net": net, "turn": turn[0].tolist(),
            "long_net": long_net, "short_net": short_net,
            "entries_l": entries_l, "entries_s": entries_s}


def seg(net, turn, a, b):
    s = net[a:b]
    t = turn[a:b]
    n = len(s)
    m = sum(s) / n if n else 0.0
    v = sum((x - m) ** 2 for x in s) / max(n - 1, 1) if n > 1 else 0.0
    sh = m / math.sqrt(v) * math.sqrt(BPY) if v > 0 else 0.0
    cs, pk, md = 0.0, -1e18, 0.0
    for x in s:
        cs += x
        pk = max(pk, cs)
        md = max(md, pk - cs)
    cum = sum(s)
    return {"sharpe": round(sh, 3), "ann": round(cum / n * BPY, 4) if n else 0.0,
            "mdd": round(md, 4), "cum": round(cum, 4),
            "final_x": round(1.0 + cum, 4), "n": n,
            "turnover": round(sum(t) / n, 6) if n else 0.0}


def side_split(long_net, short_net, entries_l, entries_s, a, b):
    ln = sum(long_net[a:b])
    sn = sum(short_net[a:b])
    tot = ln + sn
    if abs(tot) < 1e-12:
        shr_l, shr_s = 0.0, 0.0
    else:
        shr_l = round(ln / abs(tot) if tot >= 0 else -ln / abs(tot), 4)
        shr_s = round(sn / abs(tot) if tot >= 0 else -sn / abs(tot), 4)
    return {"long_entries": entries_l, "short_entries": entries_s,
            "long_pnl": round(ln, 4), "short_pnl": round(sn, 4),
            "long_share": shr_l, "short_share": shr_s}


def find_knee(xs, ys):
    n = len(xs)
    if n < 3:
        return xs[0], 0
    x0, x1 = xs[0], xs[-1]
    y0, y1 = ys[0], ys[-1]
    dx, dy = (x1 - x0), (y1 - y0)
    norm = math.sqrt(dx * dx + dy * dy)
    if norm < 1e-12:
        return xs[0], 0
    best_i, best_d = 0, -1e18
    for i in range(n):
        d = abs(dy * xs[i] - dx * ys[i] + x1 * y0 - y1 * x0) / norm
        if d > best_d:
            best_d, best_i = d, i
    return xs[best_i], best_i


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").write("iter_H24_q start smoke=%s\n" % SMOKE)
    coins = list(SMOKE_COINS) if SMOKE else list(COINS)
    qs = list(SMOKE_QS) if SMOKE else list(QS)
    w = 1.0 / len(coins)
    _partial["config"] = {
        "engine": "mirror run_weight_modes leg_net + quantile q long-only mask + cooldown + stops + vol_scale(vt None->1.0) + roll1; static equal Top5; 1h native cd/ts/vw x4",
        "formula": list(FORMULA),
        "basket": {c: dict(BASE_SPECS[c]) for c in coins},
        "weights": {c: w for c in coins},
        "venue": "aster",
        "lev": LEV,
        "fund": FUND,
        "fee": FEE,
        "fee2x": FEE2X,
        "qs": list(qs),
        "grid": "1h",
        "bpy": BPY,
        "scale": SCALE,
        "coins": list(coins),
        "smoke": SMOKE,
        "knee": "max perpendicular distance from chord on (q, FULL fee2x sharpe)",
        "attribution": "per-leg long_net/short_net split (fee+funding allocated by side turnover/position); entries count 0->side flips",
        "note": "E10 FORMULA untouched; q sweep overrides locked q0.3 per row only; offline read-only; PENDING, no adoption, live untouched",
    }
    _partial["rows"] = []
    _partial["units_done"] = []
    _partial["verdict"] = "PENDING"
    _partial["decision"] = "KEEP_q0.3"
    dump()
    ts, bars = common1h(coins)
    n = len(ts)
    if SMOKE:
        n = min(n, SMOKE_BARS)
        bars = {c: bars[c][:n] for c in coins}
    log("common 1h n=%d coins=%s qs=%s smoke=%s scale=x%d" % (n, coins, qs, SMOKE, SCALE))
    assert n > 2000, "grid too short: %d" % n
    _partial["config"]["grid_bars"] = n
    dump()
    mats = {c: build_sig(bars[c]) for c in coins}
    log("signals built (E10 FORMULA, locked)")
    _partial["units_done"].append("signals")
    dump()
    for q in qs:
        legs, legs2 = {}, {}
        for c in coins:
            raw, rt, sg = mats[c]
            legs[c] = leg_pnl(raw, rt, sg, BASE_SPECS[c], FEE, FUND, q)
            legs2[c] = leg_pnl(raw, rt, sg, BASE_SPECS[c], FEE2X, FUND, q)
        net = [sum(legs[c]["net"][t] * w for c in coins) for t in range(n)]
        turn = [sum(legs[c]["turn"][t] * w for c in coins) for t in range(n)]
        lnet = [sum(legs[c]["long_net"][t] * w for c in coins) for t in range(n)]
        snet = [sum(legs[c]["short_net"][t] * w for c in coins) for t in range(n)]
        el = sum(legs[c]["entries_l"] for c in coins)
        es = sum(legs[c]["entries_s"] for c in coins)
        full = seg(net, turn, 0, n)
        net2 = [sum(legs2[c]["net"][t] * w for c in coins) for t in range(n)]
        turn2 = [sum(legs2[c]["turn"][t] * w for c in coins) for t in range(n)]
        full2 = seg(net2, turn2, 0, n)
        row = {"q": q, "FULL": full, "FULL_fee2x": full2,
               "gap_fee2x_sharpe": round(full2["sharpe"] - full["sharpe"], 3),
               "attribution_full": side_split(lnet, snet, el, es, 0, n)}
        _partial["rows"].append(row)
        _partial["units_done"].append("q_%.1f" % q)
        dump()  # INCREMENTAL DUMP after each unit
        a = row["attribution_full"]
        log("q=%.1f FULL sh=%.3f mdd=%.4f final_x=%.4f to=%.6f | fee2x sh=%.3f gap=%+.3f | L/S=%d/%d pnl=%.4f/%.4f" % (
            q, full["sharpe"], full["mdd"], full["final_x"], full["turnover"],
            full2["sharpe"], row["gap_fee2x_sharpe"], a["long_entries"],
            a["short_entries"], a["long_pnl"], a["short_pnl"]))
    rows = _partial["rows"]
    curve = [r["FULL_fee2x"]["sharpe"] for r in rows]
    base_curve = [r["FULL"]["sharpe"] for r in rows]
    knee_q, knee_i = find_knee(qs, curve)
    r03 = next((r for r in rows if abs(r["q"] - 0.3) < 1e-9), None)
    if r03 is not None:
        others = [r["FULL_fee2x"]["sharpe"] for r in rows if abs(r["q"] - 0.3) > 1e-9]
        mean_others = sum(others) / len(others) if others else 0.0
        gap = (r03["FULL_fee2x"]["sharpe"] - mean_others) / max(abs(mean_others), 1e-9)
        isolated = bool(gap > 0.15 and all(
            r03["FULL_fee2x"]["sharpe"] > r["FULL_fee2x"]["sharpe"] for r in rows
            if abs(r["q"] - 0.3) > 1e-9))
        q03_check = {"FULL_fee2x": r03["FULL_fee2x"]["sharpe"],
                     "FULL": r03["FULL"]["sharpe"],
                     "mean_others_fee2x": round(mean_others, 3),
                     "gap_pct": round(gap * 100, 1),
                     "isolated_peak": isolated}
    else:
        q03_check = {"FULL_fee2x": None, "FULL": None,
                     "mean_others_fee2x": None, "gap_pct": None,
                     "isolated_peak": False,
                     "note": "smoke grid has no q0.3 row" if SMOKE else "no q0.3 row"}
    _partial["fee2x_curve"] = [{"q": q, "FULL_fee2x_sharpe": v} for q, v in zip(qs, curve)]
    _partial["base_curve"] = [{"q": q, "FULL_sharpe": v} for q, v in zip(qs, base_curve)]
    _partial["knee"] = {"q": knee_q, "idx": knee_i, "curve": curve}
    _partial["q03_check"] = q03_check
    _partial["verdict"] = "PENDING"
    _partial["decision"] = "KEEP_q0.3"
    _partial["decision_note"] = ("P0-3 permutation FAIL => H24 verdict PENDING; "
                                 "fee2x knee q=%.1f reported only, no adoption, "
                                 "live q0.3 untouched." % knee_q)
    _partial["conclusion"] = ("PENDING: H24 1h Top5 q-sweep knee q=%.1f "
                              "(fee2x curve %s); KEEP_q0.3, no live change." % (knee_q, curve))
    dump()
    log("wrote %s verdict=PENDING decision=KEEP_q0.3 knee=%.1f" % (OUT, knee_q))


if __name__ == "__main__":
    main()
