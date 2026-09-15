"""H11 per-coin sl x ts x cd-small joint grid (1h native, Top5) -- DIAGNOSTIC ONLY.

P0-3 permutation FAILED (per-coin p>=0.05, results/permutation.json), so per
the PRP global exit rule this step's conclusion is "PENDING (\u5f85\u5b9a)" and MUST
NOT be used as demo-listing evidence. No adoption, no live change, live
chain untouched. Offline read-only: reads data/data_1y/1h/*.csv only.

Engine mirrors research/run_weight_modes.py leg_net: E10 FORMULA
[3,2,7,2,7,11,15,4,4,6,6,10] via StackVM+FeatureEngineer; MemeBacktest
venue=aster lev2 short_enabled fund0.0005 fee0.0004; quantile q0.3
long-only mask + cooldown + stops + vol_scale(vt None->1.0) + roll1.

Top5 locked specs: ETC(0.88/0.12/cd18/None/ts24) TRX(0.85/0.12/cd6/0.05/ts24)
ATOM(0.85/0.15/cd6/0.05/ts24) APT(0.88/0.12/cd18/None/ts24)
KAS(0.88/0.12/cd6/None/ts24), q0.3. Equal 0.2 weights unless round
varies them (this round varies sl/ts/cd only).

1h native: data/data_1y/1h/{COIN}.csv direct (~8760 rows, cols
timestamp,open,high,low,close,volume,quote_volume,trades). 4h-bar params
scale x4 inside the engine: cooldown_bars=cd*4, time_stop=ts*4,
vol_window=vw*4 (stop-loss is a return fraction, unscaled). BPY=8760.

Grid (per-coin joint): sl {None,0.03,0.05} x ts {12,24} x cd {3,6} =
18 combos per coin. Each run varies ONE coin's (sl,ts,cd) while the
other four stay locked; the basket (equal 0.2) FULL sharpe/mdd is
reported. 5 coins x 18 = 90 per-coin rows.
Uniform compromise: same 18 combos applied to ALL five coins at once
(18 uniform rows, basket FULL). Selection = max FULL sharpe
(tie-break: min FULL mdd). Report only, no adoption.

Outputs: results/iter_H11_combo.json (+ logs/iter_H11_combo.log).
Incremental: results JSON dumped after EACH unit (partial survives).

Smoke mode (for tests): ITER_H11_SMOKE=1 shrinks to coins {ETC,TRX},
sl {None,0.05} x ts {24} x cd {6} (2 runs/coin + 2 uniform).
ITER_H11_OUT / ITER_H11_LOG override output paths (tests use a temp
file so the committed FULL artifact is not clobbered).
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
SL_GRID = [None, 0.03, 0.05]
TS_GRID = [12, 24]
CD_GRID = [3, 6]
BPY = 8760.0
SCALE = 4  # 4h-bar units -> 1h-bar units

OUT = pathlib.Path(os.getenv("ITER_H11_OUT", "results/iter_H11_combo.json"))
LOG = pathlib.Path(os.getenv("ITER_H11_LOG", "logs/iter_H11_combo.log"))

SMOKE = os.getenv("ITER_H11_SMOKE") == "1"
SMOKE_COINS = ["ETC", "TRX"]
SMOKE_SL = [None, 0.05]
SMOKE_TS = [24]
SMOKE_CD = [6]

_partial = {}


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")


def dump():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(_partial, indent=1, ensure_ascii=False))


def load1h(coin):
    rows = list(csv.DictReader(open("data/data_1y/1h/%s.csv" % coin)))
    return [(int(r["timestamp"]), float(r["open"]), float(r["high"]),
             float(r["low"]), float(r["close"]), float(r["volume"]))
            for r in rows]


def common1h(coins):
    """Timestamp-intersected native 1h bars (no aggregation)."""
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
    return bars


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
    # quantile q0.3 long-only mask
    if q is None or float(q) <= 0:
        return None
    a = sig.detach().float().abs().reshape(-1)
    k = max(1, int(len(a) * float(q)))
    thr = torch.topk(a, k).values.min()
    return (sig.detach().float().abs() >= thr).float()


def leg_net(raw, rt, sig, spec, fee, fund):
    """Mirror research/run_weight_modes.py leg_net, 1h-native params."""
    bt = MemeBacktest(venue="aster", leverage=LEV, short_enabled=True,
                      funding_override=fund, fee_override=fee,
                      long_th=spec["lth"], short_th=spec["sth"],
                      cooldown_bars=int(spec["cd"]) * SCALE,
                      bars_per_year=BPY,
                      stop_loss=spec["sl"],
                      time_stop=int(spec["ts"]) * SCALE,
                      vol_target=spec["vt"],
                      vol_window=int(spec["vw"]) * SCALE)
    sg = torch.sigmoid(sig)
    safe = (raw["liquidity"] > bt.min_liq).float()
    lp = (sg > bt.long_th).float() * safe
    sp = (sg < bt.short_th).float() * safe
    mk = qmask(sig, spec["q"])
    if mk is not None:
        lp = lp * mk
    # cooldown + stops + vol_scale(vt None->1.0) + roll1
    lp, sp = bt._apply_cooldown(lp, sp)
    lp, sp = bt._apply_stops(lp, sp, rt)
    sc = bt._vol_scale(rt)
    lp, sp = lp * sc, sp * sc
    lp = lp.roll(1, dims=1)
    lp[:, 0] = 0
    sp = sp.roll(1, dims=1)
    sp[:, 0] = 0
    turn = (lp - lp.roll(1, dims=1)).abs() + (sp - sp.roll(1, dims=1)).abs()
    gross = (lp - sp) * rt * bt.leverage
    tx = turn * bt.base_fee * bt.leverage
    fnd = (lp - sp) * bt.default_funding_rate * bt.leverage
    return (gross - tx - fnd)[0].tolist(), turn[0].tolist()


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


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").write("iter_H11 start smoke=%s\n" % SMOKE)
    coins = list(SMOKE_COINS) if SMOKE else list(COINS)
    sl_grid = list(SMOKE_SL) if SMOKE else list(SL_GRID)
    ts_grid = list(SMOKE_TS) if SMOKE else list(TS_GRID)
    cd_grid = list(SMOKE_CD) if SMOKE else list(CD_GRID)
    _partial["config"] = {
        "engine": "mirror run_weight_modes leg_net + quantile q0.3 long-mask + cooldown + stops + vol_scale(vt None->1.0) + roll1; 1h native, cd/ts/vw x4",
        "formula": list(FORMULA),
        "basket": {c: {"lth": BASE_SPECS[c]["lth"], "sth": BASE_SPECS[c]["sth"],
                       "cd": BASE_SPECS[c]["cd"], "sl": BASE_SPECS[c]["sl"],
                       "ts": BASE_SPECS[c]["ts"], "q": BASE_SPECS[c]["q"]}
                  for c in COINS},
        "weights": dict(W),
        "sl_grid": list(sl_grid),
        "ts_grid": list(ts_grid),
        "cd_grid": list(cd_grid),
        "cd_unit": "4h-bar, engine x4 internally",
        "ts_unit": "4h-bar, engine x4 internally",
        "per_coin_cells": len(sl_grid) * len(ts_grid) * len(cd_grid),
        "uniform_cells": len(sl_grid) * len(ts_grid) * len(cd_grid),
        "venue": "aster",
        "lev": LEV,
        "fund": FUND,
        "fee": FEE,
        "bpy": BPY,
        "scale": SCALE,
        "grid": "1h",
        "smoke": SMOKE,
        "partial": True,
        "note": "per-coin joint sl x ts x cd-small grid, one coin varied per run, others locked; uniform compromise applies one combo to all coins. E10 FORMULA untouched; live chain untouched; offline read-only.",
    }
    _partial["per_coin_rows"] = []
    _partial["uniform_rows"] = []
    _partial["verdict"] = "PENDING"
    dump()

    bars = common1h(COINS)
    n = len(bars[COINS[0]])
    log("common 1h n=%d coins=%s sl=%s ts=%s cd=%s smoke=%s" % (
        n, coins, sl_grid, ts_grid, cd_grid, SMOKE))
    assert n > 8000, "1h grid too short: %d" % n
    _partial["config"]["grid_bars"] = n
    dump()
    mats = {c: build_sig(bars[c]) for c in COINS}
    log("signals built (E10 FORMULA, locked)")
    _partial.setdefault("units_done", []).append("signals")
    dump()

    leg_cache = {}

    def get_leg(c, sl, ts, cd):
        key = (c, sl, ts, cd)
        if key not in leg_cache:
            spec = dict(BASE_SPECS[c])
            spec["sl"], spec["ts"], spec["cd"] = sl, ts, cd
            raw, rt, sg = mats[c]
            leg_cache[key] = leg_net(raw, rt, sg, spec, FEE, FUND)
        return leg_cache[key]

    def basket_full(sl_map, ts_map, cd_map):
        legs, turns = {}, {}
        for c in COINS:
            legs[c], turns[c] = get_leg(c, sl_map[c], ts_map[c], cd_map[c])
        net = [sum(legs[c][t] * W[c] for c in COINS) for t in range(n)]
        turn = [sum(turns[c][t] * W[c] for c in COINS) for t in range(n)]
        return seg(net, turn, 0, n)

    base_sl = {c: BASE_SPECS[c]["sl"] for c in COINS}
    base_ts = {c: BASE_SPECS[c]["ts"] for c in COINS}
    base_cd = {c: BASE_SPECS[c]["cd"] for c in COINS}
    base_full = basket_full(base_sl, base_ts, base_cd)
    _partial["base_FULL"] = base_full
    _partial["units_done"].append("base")
    dump()
    log("BASE FULL sh=%.3f mdd=%.4f final_x=%.4f to=%.6f" % (
        base_full["sharpe"], base_full["mdd"],
        base_full["final_x"], base_full["turnover"]))

    def row_for(sl, ts, cd, full):
        return {"sl": sl, "ts": ts, "cd": cd, "FULL": full,
                "d_sharpe_vs_base": round(full["sharpe"] - base_full["sharpe"], 3),
                "d_mdd_vs_base": round(full["mdd"] - base_full["mdd"], 4)}

    def pick_best(rows):
        if not rows:
            return None
        top = max(r["FULL"]["sharpe"] for r in rows)
        tied = [r for r in rows if r["FULL"]["sharpe"] == top]
        return min(tied, key=lambda r: r["FULL"]["mdd"])

    # per-coin joint grid: vary one coin, others locked
    for c in coins:
        for sl in sl_grid:
            for ts in ts_grid:
                for cd in cd_grid:
                    sl_map = dict(base_sl)
                    ts_map = dict(base_ts)
                    cd_map = dict(base_cd)
                    sl_map[c], ts_map[c], cd_map[c] = sl, ts, cd
                    full = basket_full(sl_map, ts_map, cd_map)
                    r = {"coin": c}
                    r.update(row_for(sl, ts, cd, full))
                    _partial["per_coin_rows"].append(r)
                    _partial["units_done"].append("%s_sl%s_ts%d_cd%d" % (c, sl, ts, cd))
                    log("%s sl=%s ts=%d cd=%d sh=%.3f mdd=%.4f final_x=%.4f dsh=%+.3f dmdd=%+.4f" % (
                        c, sl, ts, cd, full["sharpe"], full["mdd"],
                        full["final_x"], r["d_sharpe_vs_base"], r["d_mdd_vs_base"]))
                    dump()

    per_best = {}
    for c in coins:
        rows_c = [r for r in _partial["per_coin_rows"] if r["coin"] == c]
        per_best[c] = pick_best(rows_c)
        b = per_best[c]
        log("best[%s]: sl=%s ts=%d cd=%d sh=%.3f mdd=%.4f" % (
            c, b["sl"], b["ts"], b["cd"],
            b["FULL"]["sharpe"], b["FULL"]["mdd"]))
    _partial["per_coin_best"] = per_best
    dump()

    # uniform compromise: one combo applied to ALL coins
    for sl in sl_grid:
        for ts in ts_grid:
            for cd in cd_grid:
                sl_map = {c: sl for c in COINS}
                ts_map = {c: ts for c in COINS}
                cd_map = {c: cd for c in COINS}
                full = basket_full(sl_map, ts_map, cd_map)
                r = row_for(sl, ts, cd, full)
                _partial["uniform_rows"].append(r)
                _partial["units_done"].append("U_sl%s_ts%d_cd%d" % (sl, ts, cd))
                log("U sl=%s ts=%d cd=%d sh=%.3f mdd=%.4f final_x=%.4f dsh=%+.3f dmdd=%+.4f" % (
                    sl, ts, cd, full["sharpe"], full["mdd"],
                    full["final_x"], r["d_sharpe_vs_base"], r["d_mdd_vs_base"]))
                dump()

    uniform_best = pick_best(_partial["uniform_rows"])
    _partial["uniform_best"] = uniform_best
    if uniform_best is None:
        best_note = "no uniform cell; no selection"
    else:
        best_note = ("max FULL sharpe (tie-break min mdd): sl=%s ts=%d cd=%d" % (
            uniform_best["sl"], uniform_best["ts"], uniform_best["cd"]))
    log("uniform_best: %s (%s)" % (
        ({k: uniform_best[k] for k in ("sl", "ts", "cd")} if uniform_best else None),
        best_note))

    _partial["config"]["partial"] = False
    _partial["best_note"] = best_note
    _partial["verdict"] = "PENDING"
    _partial["conclusion"] = ("\u5f85\u5b9a (P0-3 FAIL): H11 1h per-coin sl x ts x cd-small "
                              "joint grid diagnostic only; per-coin best + uniform "
                              "compromise are shelf values, no adoption, no live "
                              "change, live untouched.")
    dump()
    log("wrote %s verdict=PENDING" % OUT)
    print(json.dumps({"base_FULL": base_full, "per_coin_best": per_best,
                      "uniform_best": uniform_best,
                      "verdict": "PENDING"}, indent=1))


if __name__ == "__main__":
    main()
