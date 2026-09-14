"""Z3 15m time-stop sweep (Top5) -- DIAGNOSTIC ONLY.

P0-3 permutation FAILED (per-coin p>=0.05, results/permutation.json), so per
the PRP global exit rule this step's conclusion is "PENDING" and MUST NOT
be used as demo-listing evidence. No adoption, no live change, live chain
untouched. Offline read-only: reads data/data_1y/15m/*.csv only.

Premise: Top5 locked specs (E10 FORMULA [3,2,7,2,7,11,15,4,4,6,6,10]):
  ETC(0.88/0.12/cd18/None/ts24) TRX(0.85/0.12/cd6/0.05/ts24)
  ATOM(0.85/0.15/cd6/0.05/ts24) APT(0.88/0.12/cd18/None/ts24)
  KAS(0.88/0.12/cd6/None/ts24), q0.3 long-mask, equal 0.2 weights,
  aster perp 2x fund0.0005 fee0.0004.

15m NATIVE grid (no 4h aggregation): bars are raw 15m rows (~35040, 1y).
4h-bar params scale x16 inside the engine: cooldown_bars=cd*16,
time_stop=ts*16, vol_window=vw*16. BPY=35040. Stop-loss is a return
fraction, unscaled (sl stays per-coin locked in this sweep).

Sweep A (uniform): ts {12,24,36,48} applied to ALL 5 coins
(cd/sl stays per-coin locked). 4 cells.
Sweep B (per-coin sensitivity): each coin's ts over {12,24,36,48} with
the other four at locked base (ts 24). 5x4 = 20 runs.

Metric per cell/run: FULL sharpe/mdd/turnover (+ann/cum/n).
Selection: best no-drop dd-min cell = min FULL mdd among uniform cells
with FULL sharpe >= base FULL sharpe. Shelf value only, no adoption.

Outputs: results/iter_Z3_ts.json (+ logs/iter_z3_ts.log).

Smoke mode (for tests): ITER_Z3_SMOKE=1 shrinks to ts {24,36}
(2 uniform cells) + per-coin ts {24,36} (2 coins x 2 = 4 runs),
coins {ETC,TRX}, first 3000 bars. OUT/LOG overridable via
ITER_Z3_OUT / ITER_Z3_LOG.
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
TS_GRID = [12, 24, 36, 48]
BPY = 35040.0
SCALE = 16  # 4h-bar params -> 15m bars

OUT = pathlib.Path(os.getenv("ITER_Z3_OUT", "results/iter_Z3_ts.json"))
LOG = pathlib.Path(os.getenv("ITER_Z3_LOG", "logs/iter_z3_ts.log"))

SMOKE = os.getenv("ITER_Z3_SMOKE") == "1"
SMOKE_TS = [24, 36]
SMOKE_COINS = ["ETC", "TRX"]


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")


def load15m(coin):
    rows = list(csv.DictReader(open("data/data_1y/15m/%s.csv" % coin)))
    return [(int(r["timestamp"]), float(r["open"]), float(r["high"]),
             float(r["low"]), float(r["close"]), float(r["volume"]))
            for r in rows]


def common15m(coins):
    """Timestamp-intersected native 15m bars (no aggregation)."""
    raw = {c: load15m(c) for c in coins}
    s = max(r[0][0] for r in raw.values())
    e = min(r[-1][0] for r in raw.values())
    bars, closes = {}, {}
    for c in coins:
        rr = [r for r in raw[c] if s <= r[0] <= e]
        bars[c] = [(r[1], r[2], r[3], r[4], r[5]) for r in rr]
        closes[c] = [b[3] for b in bars[c]]
    n = min(len(b) for b in bars.values())
    for c in coins:
        bars[c] = bars[c][:n]
        closes[c] = closes[c][:n]
    return bars, closes


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


def leg_net(raw, rt, sig, lth, sth, cd, sl, ts, vt, vw, q, fee, fund):
    """Mirror research/run_weight_modes.py leg_net, 15m-native params."""
    bt = MemeBacktest(venue="aster", leverage=LEV, short_enabled=True,
                      funding_override=fund, fee_override=fee,
                      long_th=lth, short_th=sth,
                      cooldown_bars=cd * SCALE, bars_per_year=BPY,
                      stop_loss=sl, time_stop=ts * SCALE,
                      vol_target=vt, vol_window=vw * SCALE)
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


def portfolio(legs, turns, coins, n):
    w = 1.0 / len(coins)
    net = [sum(legs[c][t] * w for c in coins) for t in range(n)]
    turn = [sum(turns[c][t] * w for c in coins) for t in range(n)]
    return net, turn


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").write("iter_z3_ts start\n")
    coins = list(SMOKE_COINS) if SMOKE else list(COINS)
    ts_grid = list(SMOKE_TS) if SMOKE else list(TS_GRID)
    w = {c: 1.0 / len(coins) for c in coins}
    bars, _closes = common15m(coins)
    n = len(bars[coins[0]])
    if SMOKE:
        n = min(n, 3000)
        bars = {c: bars[c][:n] for c in coins}
    h2a = n // 2
    log("common 15m n=%d h2a=%d coins=%s ts=%s smoke=%s" % (n, h2a, coins, ts_grid, SMOKE))
    assert n > 2000, "15m grid too short: %d" % n
    if not SMOKE:
        assert n > 30000, "15m grid too short: %d" % n
    mats = {c: build_sig(bars[c]) for c in coins}
    log("signals built (E10 FORMULA, locked)")

    def run_leg(c, ts):
        spec = BASE_SPECS[c]
        raw, rt, sg = mats[c]
        return leg_net(raw, rt, sg, spec["lth"], spec["sth"], spec["cd"],
                       spec["sl"], ts, spec["vt"], spec["vw"], spec["q"], FEE, FUND)

    def eval_cell(ts_map):
        legs, turns = {}, {}
        for c in coins:
            legs[c], turns[c] = run_leg(c, ts_map[c])
        net, turn = portfolio(legs, turns, coins, n)
        return net, turn, seg(net, turn, 0, n), seg(net, turn, h2a, n)["sharpe"]

    base_ts = {c: BASE_SPECS[c]["ts"] for c in coins}
    _, _, base_full, base_h2 = eval_cell(base_ts)
    log("BASE FULL sh=%.3f mdd=%.4f final_x=%.4f to=%.6f H2=%.3f" % (
        base_full["sharpe"], base_full["mdd"], base_full["final_x"],
        base_full["turnover"], base_h2))

    uniform_rows = []
    for ts in ts_grid:
        ts_map = {c: ts for c in coins}
        _net, _turn, full, h2 = eval_cell(ts_map)
        row = {"ts": ts, "FULL": full, "H2_sharpe": h2,
               "d_sharpe_vs_base": round(full["sharpe"] - base_full["sharpe"], 3),
               "d_mdd_vs_base": round(full["mdd"] - base_full["mdd"], 4),
               "d_turnover_vs_base": round(full["turnover"] - base_full["turnover"], 6)}
        uniform_rows.append(row)
        log("U ts=%d sh=%.3f mdd=%.4f final_x=%.4f H2=%.3f to=%.6f dsh=%+.3f dmdd=%+.4f dto=%+.6f" % (
            ts, full["sharpe"], full["mdd"], full["final_x"], h2,
            full["turnover"], row["d_sharpe_vs_base"], row["d_mdd_vs_base"],
            row["d_turnover_vs_base"]))

    percoin_rows = []
    for c in coins:
        for ts in ts_grid:
            ts_map = dict(base_ts)
            ts_map[c] = ts
            _net, _turn, full, h2 = eval_cell(ts_map, )
            row = {"coin": c, "ts": ts, "FULL": full, "H2_sharpe": h2,
                   "d_sharpe_vs_base": round(full["sharpe"] - base_full["sharpe"], 3),
                   "d_mdd_vs_base": round(full["mdd"] - base_full["mdd"], 4),
                   "d_turnover_vs_base": round(full["turnover"] - base_full["turnover"], 6)}
            percoin_rows.append(row)
            log("P %s ts=%d sh=%.3f mdd=%.4f final_x=%.4f H2=%.3f to=%.6f dsh=%+.3f" % (
                c, ts, full["sharpe"], full["mdd"], full["final_x"], h2,
                full["turnover"], row["d_sharpe_vs_base"]))

    percoin_best = {}
    for c in coins:
        crs = [r for r in percoin_rows if r["coin"] == c]
        best = max(crs, key=lambda r: (r["FULL"]["sharpe"], -r["FULL"]["mdd"]))
        percoin_best[c] = {"ts": best["ts"], "FULL": best["FULL"],
                           "H2_sharpe": best["H2_sharpe"]}

    nodrop = [r for r in uniform_rows if r["FULL"]["sharpe"] >= base_full["sharpe"]]
    best_cell = min(nodrop, key=lambda r: r["FULL"]["mdd"]) if nodrop else None
    if best_cell is None:
        best_note = "no uniform cell with sharpe>=base; no selection"
    else:
        best_note = ("min FULL mdd among uniform cells with sharpe>=base: "
                     "ts=%d" % (best_cell["ts"]))
    log("best_no_drop_dd_min: %s (%s)" % (
        ({k: best_cell[k] for k in ("ts",)} if best_cell else None), best_note))

    out = {
        "config": {
            "engine": "mirror run_weight_modes leg_net + quantile q0.3 long-mask + cooldown + stops + vol_scale(vt None->1.0) + roll1; 15m native, cd/ts/vw x16",
            "formula": list(FORMULA),
            "basket": {c: dict(BASE_SPECS[c]) for c in coins},
            "weights": dict(w),
            "coins": list(coins),
            "ts_grid": list(ts_grid),
            "ts_unit": "4h-bar, engine x16 internally",
            "uniform_cells": len(uniform_rows),
            "percoin_runs": len(percoin_rows),
            "venue": "aster",
            "lev": LEV,
            "fund": FUND,
            "fee": FEE,
            "bpy": BPY,
            "scale": SCALE,
            "grid": "15m",
            "grid_bars": n,
            "h2_start": h2a,
            "smoke": SMOKE,
            "note": "uniform ts {12,24,36,48} + per-coin ts sensitivity; sl/cd locked. E10 FORMULA untouched; live chain untouched; offline read-only.",
        },
        "base_FULL": base_full,
        "base_H2_sharpe": base_h2,
        "uniform_rows": uniform_rows,
        "percoin_rows": percoin_rows,
        "percoin_best": percoin_best,
        "best_no_drop_dd_min": best_cell,
        "best_note": best_note,
        "verdict": "PENDING",
        "decision": "PENDING",
        "conclusion": ("PENDING (P0-3 FAIL): Z3 15m time-stop sweep diagnostic only; "
                       "no-drop dd-min selection is shelf value, no adoption, "
                       "no live change, live untouched."),
    }
    OUT.write_text(json.dumps(out, indent=1))
    log("wrote %s verdict=PENDING" % OUT)
    print(json.dumps({"base_FULL": base_full, "base_H2": base_h2,
                      "best_no_drop_dd_min": best_cell,
                      "verdict": "PENDING"}, indent=1))


if __name__ == "__main__":
    main()
