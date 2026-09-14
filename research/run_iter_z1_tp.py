"""Z1 15m take-profit sweep (Top5) -- DIAGNOSTIC ONLY.

P0-3 permutation FAILED (per-coin p>=0.05, results/permutation.json), so per
the PRP global exit rule this step's conclusion is "PENDING" and MUST NOT be
used as demo-listing evidence. No adoption, no live change, live untouched.
Offline read-only: reads data/data_1y/15m/*.csv only.

Premise: Top5 locked specs (E10 FORMULA [3,2,7,2,7,11,15,4,4,6,6,10]):
  ETC(0.88/0.12/cd18/None/ts24) TRX(0.85/0.12/cd6/0.05/ts24)
  ATOM(0.85/0.15/cd6/0.05/ts24) APT(0.88/0.12/cd18/None/ts24)
  KAS(0.88/0.12/cd6/None/ts24), q0.3 long-mask, equal 0.2 weights,
  aster perp 2x fund0.0005 fee0.0004.

15m NATIVE grid (no 4h aggregation): bars are raw 15m rows (~35040, 1y).
4h-bar params scale x16 inside the engine: cooldown_bars=cd*16,
time_stop=ts*16, vol_window=vw*16. BPY=35040. TP/trailing are return
fractions, unscaled.

Sweep (uniform, all 5 coins): take_profit {None,0.05,0.10,0.20} x
trailing-drop {None,0.03,0.05} = 4x3 = 12 cells. Fixed TP rides inside
MemeBacktest._apply_stops (exc >= tp kills the position). Trailing is a
post-stop give-back exit applied on binary positions before vol_scale:
track per-position excursion since entry and its peak; force flat when
armed (peak > 0) and peak - excursion >= drop. No separate activation
level (activation = any profit). TP=None + trail=None reproduces the
locked config; TP set + trail set applies both (engine TP kill first,
then trailing trims survivors).

Metric per cell: FULL sharpe/mdd/final_x + H2 sharpe + turnover
(H2 = second half of the 15m grid). Selection: best no-drop dd-min cell
= min FULL mdd among uniform cells with FULL sharpe >= base FULL sharpe.

Outputs: results/iter_Z1_tp.json (+ logs/iter_z1_tp.log).

Smoke mode (for tests): ITER_Z1_SMOKE=1 shrinks to tp {None,0.10} x
trail {None,0.03} (4 cells).
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

COINS = ["ETC", "TRX", "ATOM", "APT", "KAS"]
BASE_SPECS = {
    "ETC": dict(LOCKED_ETC),
    "TRX": dict(LOCKED_TRX),
    "ATOM": dict(LOCKED_ATOM),
    "APT": dict(LOCKED_APT),
    "KAS": dict(LOCKED_KAS),
}
W = {c: 0.2 for c in COINS}
TP_GRID = [None, 0.05, 0.10, 0.20]
TRAIL_GRID = [None, 0.03, 0.05]
BPY = 35040.0
SCALE = 16  # 4h-bar params -> 15m bars

OUT = pathlib.Path("results/iter_Z1_tp.json")
LOG = pathlib.Path("logs/iter_z1_tp.log")

SMOKE = os.getenv("ITER_Z1_SMOKE") == "1"
SMOKE_TP = [None, 0.10]
SMOKE_TRAIL = [None, 0.03]


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


def apply_trailing(long_pos, short_pos, target_ret, drop):
    """Give-back exit on binary positions (pre-vol_scale, post-stops).

    Per row: track entry excursion exc (cum cur*R since entry) and its
    peak. A fresh entry arms once peak > 0 (any profit); force flat when
    peak - exc >= drop. Natural flats/resets pass through. drop=None/0
    is a no-op. Returns tensors on the input device.
    """
    if drop is None or float(drop) <= 0:
        return long_pos, short_pos
    d = float(drop)
    L = long_pos.detach().float().cpu().tolist()
    S = short_pos.detach().float().cpu().tolist()
    R = target_ret.detach().float().cpu().tolist()
    if not isinstance(R[0], list):
        R = [R]
        L = [L] if not isinstance(L[0], list) else L
        S = [S] if not isinstance(S[0], list) else S
    out_l, out_s = [], []
    for i in range(len(L)):
        rl, rs = [], []
        cur, exc, peak = 0.0, 0.0, 0.0
        for t in range(len(L[i])):
            want = 1.0 if L[i][t] > 0.5 else (-1.0 if S[i][t] > 0.5 else 0.0)
            if want != cur:
                cur, exc, peak = want, 0.0, 0.0
            if cur != 0.0:
                exc += cur * R[i][t] if t < len(R[i]) else 0.0
                peak = max(peak, exc)
                if peak > 0 and peak - exc >= d:
                    cur, exc, peak = 0.0, 0.0, 0.0
            rl.append(1.0 if cur > 0 else 0.0)
            rs.append(1.0 if cur < 0 else 0.0)
        out_l.append(rl)
        out_s.append(rs)
    dev = long_pos.device
    return torch.tensor(out_l, device=dev), torch.tensor(out_s, device=dev)


def leg_net(raw, rt, sig, lth, sth, cd, sl, ts, vt, vw, q, tp, trail, fee, fund):
    """Mirror research/run_weight_modes.py leg_net, 15m-native + TP/trail."""
    bt = MemeBacktest(venue="aster", leverage=LEV, short_enabled=True,
                      funding_override=fund, fee_override=fee,
                      long_th=lth, short_th=sth,
                      cooldown_bars=cd * SCALE, bars_per_year=BPY,
                      stop_loss=sl, time_stop=ts * SCALE,
                      take_profit=tp,
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
    lp, sp = apply_trailing(lp, sp, rt, trail)
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


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").write("iter_z1 start\n")
    tp_grid = SMOKE_TP if SMOKE else TP_GRID
    trail_grid = SMOKE_TRAIL if SMOKE else TRAIL_GRID
    bars, _closes = common15m(COINS)
    n = len(bars[COINS[0]])
    h2a = n // 2
    log("common 15m n=%d h2a=%d tp=%s trail=%s smoke=%s" % (n, h2a, tp_grid, trail_grid, SMOKE))
    assert n > 30000, "15m grid too short: %d" % n
    mats = {c: build_sig(bars[c]) for c in COINS}
    log("signals built (E10 FORMULA, locked)")

    def eval_cell(tp, trail):
        legs, turns = {}, {}
        for c in COINS:
            spec = BASE_SPECS[c]
            raw, rt, sg = mats[c]
            legs[c], turns[c] = leg_net(raw, rt, sg, spec["lth"], spec["sth"],
                                        spec["cd"], spec["sl"], spec["ts"],
                                        spec["vt"], spec["vw"], spec["q"],
                                        tp, trail, FEE, FUND)
        net, turn = portfolio(legs, turns, n)
        return net, turn, seg(net, turn, 0, n), seg(net, turn, h2a, n)["sharpe"]

    _, _, base_full, base_h2 = eval_cell(None, None)
    log("BASE FULL sh=%.3f mdd=%.4f final_x=%.4f to=%.6f H2=%.3f" % (
        base_full["sharpe"], base_full["mdd"], base_full["final_x"],
        base_full["turnover"], base_h2))

    uniform_rows = []
    for tp in tp_grid:
        for trail in trail_grid:
            _net, _turn, full, h2 = eval_cell(tp, trail)
            row = {"tp": tp, "trail": trail, "FULL": full, "H2_sharpe": h2,
                   "d_sharpe_vs_base": round(full["sharpe"] - base_full["sharpe"], 3),
                   "d_mdd_vs_base": round(full["mdd"] - base_full["mdd"], 4)}
            uniform_rows.append(row)
            log("U tp=%s trail=%s sh=%.3f mdd=%.4f final_x=%.4f H2=%.3f to=%.6f dsh=%+.3f dmdd=%+.4f" % (
                tp, trail, full["sharpe"], full["mdd"], full["final_x"], h2,
                full["turnover"], row["d_sharpe_vs_base"], row["d_mdd_vs_base"]))

    nodrop = [r for r in uniform_rows if r["FULL"]["sharpe"] >= base_full["sharpe"]]
    best_cell = min(nodrop, key=lambda r: r["FULL"]["mdd"]) if nodrop else None
    if best_cell is None:
        best_note = "no uniform cell with sharpe>=base; no selection"
    else:
        best_note = ("min FULL mdd among uniform cells with sharpe>=base: "
                     "tp=%s trail=%s" % (best_cell["tp"], best_cell["trail"]))
    log("best_no_drop_dd_min: %s (%s)" % (
        ({k: best_cell[k] for k in ("tp", "trail")} if best_cell else None), best_note))

    out = {
        "config": {
            "engine": "mirror run_weight_modes leg_net + quantile q0.3 long-mask + cooldown + stops + vol_scale(vt None->1.0) + roll1; 15m native, cd/ts/vw x16",
            "formula": list(FORMULA),
            "basket": {c: dict(BASE_SPECS[c]) for c in COINS},
            "weights": dict(W),
            "tp_grid": list(tp_grid),
            "trail_grid": list(trail_grid),
            "uniform_cells": len(uniform_rows),
            "venue": "aster",
            "lev": LEV,
            "fund": FUND,
            "fee": FEE,
            "bpy": BPY,
            "scale": SCALE,
            "grid_bars": n,
            "h2_start": h2a,
            "smoke": SMOKE,
            "trail_spec": "post-stop give-back exit on binary positions pre-vol_scale: armed once peak excursion > 0, flat when peak - excursion >= drop; no separate activation",
            "note": "tp 4 x trail 3 = 12 uniform cells. E10 FORMULA untouched; live chain untouched; offline read-only.",
        },
        "base_FULL": base_full,
        "base_H2_sharpe": base_h2,
        "uniform_rows": uniform_rows,
        "best_no_drop_dd_min": best_cell,
        "best_note": best_note,
        "verdict": "PENDING",
        "conclusion": ("PENDING (P0-3 FAIL): Z1 15m take-profit sweep diagnostic only; "
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
