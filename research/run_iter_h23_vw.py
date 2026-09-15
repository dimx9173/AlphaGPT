"""H23 1h vol-window sweep (Top5) -- DIAGNOSTIC ONLY.

P0-3 permutation FAILED (per-coin p>=0.05, results/permutation.json), so per
the PRP global exit rule this step's conclusion is "PENDING" and MUST
NOT be used as demo-listing evidence. No adoption, no live change, live
chain untouched. Offline read-only: reads data/data_1y/1h/*.csv only.

Premise: Top5 locked specs (E10 FORMULA [3,2,7,2,7,11,15,4,4,6,6,10]):
  ETC(0.88/0.12/cd18/None/ts24) TRX(0.85/0.12/cd6/0.05/ts24)
  ATOM(0.85/0.15/cd6/0.05/ts24) APT(0.88/0.12/cd18/None/ts24)
  KAS(0.88/0.12/cd6/None/ts24), q0.3 long-mask, equal 0.2 weights,
  aster perp 2x fund0.0005 fee0.0004.

1h NATIVE grid (no 4h aggregation): data/data_1y/1h/{COIN}.csv direct
(~8760 rows). 4h-bar params scale x4 inside the engine:
cooldown_bars=cd*4, time_stop=ts*4, vol_window=vw*4. BPY=8760.
Stop-loss is a return fraction, unscaled.

Sweep (uniform): vw {6,12,24} x vt {None,0.006,0.012} applied to ALL
5 coins (sl/ts/cd stay per-coin locked). 3x3 = 9 cells. vt None means
vol_scale 1.0 (no targeting); vt set means MemeBacktest inverse-vol
scaling clamp 0.2-2.0 on trailing price-vol.

Metric per cell: FULL sharpe/mdd (+cum/final_x/ann/n/turnover, H2 sharpe).
Selection: best sharpe = max FULL sharpe (tie-break: min FULL mdd).
Report only, no adoption.

Outputs: results/iter_H23_vw.json (+ logs/iter_h23_vw.log).
Incremental: results JSON dumped after EACH cell (partial survives).

Smoke mode (for tests): ITER_H23_SMOKE=1 shrinks to vw {12} x
vt {None,0.012} (2 uniform cells).
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
VW_GRID = [6, 12, 24]
VT_GRID = [None, 0.006, 0.012]
BPY = 8760.0
SCALE = 4  # 4h-bar units -> 1h-bar units

OUT = pathlib.Path(os.getenv("ITER_H23_OUT", "results/iter_H23_vw.json"))
LOG = pathlib.Path(os.getenv("ITER_H23_LOG", "logs/iter_h23_vw.log"))

SMOKE = os.getenv("ITER_H23_SMOKE") == "1"
SMOKE_VW = [12]
SMOKE_VT = [None, 0.012]


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")


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
    """Mirror research/run_weight_modes.py leg_net, 1h-native params."""
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
    # quantile q0.3 long-only mask
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


def portfolio(legs, turns, n):
    net = [sum(legs[c][t] * W[c] for c in COINS) for t in range(n)]
    turn = [sum(turns[c][t] * W[c] for c in COINS) for t in range(n)]
    return net, turn


def dump_partial(payload):
    OUT.write_text(json.dumps(payload, indent=1))


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").write("iter_h23 start\n")
    vw_grid = SMOKE_VW if SMOKE else VW_GRID
    vt_grid = SMOKE_VT if SMOKE else VT_GRID
    bars, _closes = common1h(COINS)
    n = len(bars[COINS[0]])
    h2a = n // 2
    log("common 1h n=%d h2a=%d vw=%s vt=%s smoke=%s" % (n, h2a, vw_grid, vt_grid, SMOKE))
    assert n > 8000, "1h grid too short: %d" % n
    mats = {c: build_sig(bars[c]) for c in COINS}
    log("signals built (E10 FORMULA, locked)")

    def run_leg(c, vw, vt):
        spec = BASE_SPECS[c]
        raw, rt, sg = mats[c]
        return leg_net(raw, rt, sg, spec["lth"], spec["sth"], spec["cd"],
                       spec["sl"], spec["ts"], vt, vw, spec["q"], FEE, FUND)

    def eval_cell(vw, vt):
        legs, turns = {}, {}
        for c in COINS:
            legs[c], turns[c] = run_leg(c, vw, vt)
        net, turn = portfolio(legs, turns, n)
        return net, turn, seg(net, turn, 0, n), seg(net, turn, h2a, n)["sharpe"]

    _, _, base_full, base_h2 = eval_cell(12, None)
    log("BASE vw=12 vt=None FULL sh=%.3f mdd=%.4f final_x=%.4f to=%.6f H2=%.3f" % (
        base_full["sharpe"], base_full["mdd"], base_full["final_x"],
        base_full["turnover"], base_h2))

    def base_payload(rows):
        best = None
        if rows:
            best = max(rows, key=lambda r: (r["FULL"]["sharpe"], -r["FULL"]["mdd"]))
        return {
            "config": {
                "engine": "mirror run_weight_modes leg_net + quantile q0.3 long-mask + cooldown + stops + vol_scale(vt None->1.0) + roll1; 1h native, cd/ts/vw x4",
                "formula": list(FORMULA),
                "basket": {c: dict(BASE_SPECS[c]) for c in COINS},
                "weights": dict(W),
                "vw_grid": list(vw_grid),
                "vt_grid": list(vt_grid),
                "vw_unit": "4h-bar, engine x4 internally",
                "vt_unit": "per-bar vol target fraction (None = no targeting, scale 1.0)",
                "uniform_cells": len(rows),
                "venue": "aster",
                "lev": LEV,
                "fund": FUND,
                "fee": FEE,
                "bpy": BPY,
                "scale": SCALE,
                "grid_bars": n,
                "h2_start": h2a,
                "smoke": SMOKE,
                "partial": True,
                "note": "vw 3 x vt 3 = 9 uniform cells. E10 FORMULA untouched; live chain untouched; offline read-only.",
            },
            "base_FULL": base_full,
            "base_H2_sharpe": base_h2,
            "uniform_rows": rows,
            "best_sharpe": best,
            "verdict": "PENDING_P03_FAIL",
            "decision": "NO_ADOPTION_KEEP_EQUAL",
        }

    uniform_rows = []
    for vw in vw_grid:
        for vt in vt_grid:
            _net, _turn, full, h2 = eval_cell(vw, vt)
            row = {"vw": vw, "vt": vt, "FULL": full, "H2_sharpe": h2,
                   "d_sharpe_vs_base": round(full["sharpe"] - base_full["sharpe"], 3),
                   "d_mdd_vs_base": round(full["mdd"] - base_full["mdd"], 4)}
            uniform_rows.append(row)
            log("U vw=%d vt=%s sh=%.3f mdd=%.4f final_x=%.4f H2=%.3f to=%.6f dsh=%+.3f dmdd=%+.4f" % (
                vw, vt, full["sharpe"], full["mdd"], full["final_x"], h2,
                full["turnover"], row["d_sharpe_vs_base"], row["d_mdd_vs_base"]))
            dump_partial(base_payload(list(uniform_rows)))

    best_cell = max(uniform_rows, key=lambda r: (r["FULL"]["sharpe"], -r["FULL"]["mdd"]))
    best_note = ("max FULL sharpe among uniform cells (tie-break min mdd): "
                 "vw=%d vt=%s" % (best_cell["vw"], best_cell["vt"]))
    log("best_sharpe: %s (%s)" % (
        ({k: best_cell[k] for k in ("vw", "vt")}), best_note))

    out = base_payload(uniform_rows)
    out["config"]["partial"] = False
    out["best_note"] = best_note
    out["verdict"] = "PENDING_P03_FAIL"
    out["decision"] = "NO_ADOPTION_KEEP_EQUAL"
    out["decision_note"] = ("P0-3 FAIL, no adoption: vol-window contrast is shelf value only; "
                            "equal 0.2 weights and locked specs stay, live untouched.")
    out["conclusion"] = ("PENDING (P0-3 FAIL): H23 1h vol-window sweep diagnostic only; "
                         "vw x vt uniform contrast is shelf value, no adoption, "
                         "no live change, live untouched.")
    OUT.write_text(json.dumps(out, indent=1))
    log("wrote %s verdict=PENDING_P03_FAIL" % OUT)
    print(json.dumps({"base_FULL": base_full, "base_H2": base_h2,
                      "best_sharpe": best_cell,
                      "verdict": "PENDING_P03_FAIL"}, indent=1))


if __name__ == "__main__":
    main()
