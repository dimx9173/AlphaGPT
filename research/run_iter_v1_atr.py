"""V1 15m ATR-stop sweep (Top5) -- DIAGNOSTIC ONLY.

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
time_stop=ts*16, vol_window=vw*16. BPY=35040. ATR window is 96x15m bars
(=24h), unscaled; sl/ts/cd stay per-coin locked in this sweep.

Sweep A (uniform): atr_mult {None,1.5,2.0,3.0} applied to ALL 5 coins
(cd/sl/ts stay per-coin locked). 4 cells.
Sweep B (per-coin sensitivity): each coin's atr_mult over
{None,1.5,2.0,3.0} with the other four at locked base (atr None).
5x4 = 20 runs.

ATR-stop spec: Wilder ATR(96) on 15m high/low/close, close-normalized
atr_norm(t) = ATR(t)/close(t). Post-stop exit on binary positions before
vol_scale: track per-position excursion since entry (cum cur*R); force
flat when excursion <= -mult*atr_norm(t) (current-bar ATR, adverse move
beyond mult ATRs). atr_mult=None reproduces the locked config.

Metric per cell/run: FULL sharpe/mdd/final_x/turnover (+ann/cum/n)
+ H2 sharpe. Selection: best no-drop dd-min cell = min FULL mdd among
uniform cells with FULL sharpe >= base FULL sharpe. Shelf value only,
no adoption.

Outputs: results/iter_V1_atr.json (+ logs/iter_v1_atr.log).
Incremental dump after each cell: partial JSON survives interrupts.

Smoke mode (for tests): ITER_V1_SMOKE=1 shrinks to atr_mult {None,2.0}
(2 uniform cells) + per-coin atr_mult {None,2.0} (2 coins x 2 = 4 runs),
coins {ETC,TRX}, first 3000 bars. OUT/LOG overridable via
ITER_V1_OUT / ITER_V1_LOG.
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
ATR_GRID = [None, 1.5, 2.0, 3.0]
ATR_WINDOW = 96  # 96x15m = 24h
BPY = 35040.0
SCALE = 16  # 4h-bar params -> 15m bars

OUT = pathlib.Path(os.getenv("ITER_V1_OUT", "results/iter_V1_atr.json"))
LOG = pathlib.Path(os.getenv("ITER_V1_LOG", "logs/iter_v1_atr.log"))

SMOKE = os.getenv("ITER_V1_SMOKE") == "1"
SMOKE_ATR = [None, 2.0]
SMOKE_COINS = ["ETC", "TRX"]


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")


def dump(res):
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(res, indent=1))
    os.replace(tmp, OUT)


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


def atr_norm_96(bars, window=96):
    """Wilder ATR(window) on 15m high/low/close, close-normalized list."""
    n = len(bars)
    hs = [b[1] for b in bars]
    ls = [b[2] for b in bars]
    cs = [b[3] for b in bars]
    tr = [0.0] * n
    for t in range(n):
        if t == 0:
            tr[t] = hs[t] - ls[t]
        else:
            tr[t] = max(hs[t] - ls[t], abs(hs[t] - cs[t - 1]),
                        abs(ls[t] - cs[t - 1]))
    atr = [0.0] * n
    w = max(int(window), 1)
    if n == 0:
        return []
    seed = sum(tr[:min(w, n)]) / min(w, n)
    for t in range(n):
        if t < w:
            atr[t] = sum(tr[:t + 1]) / (t + 1)
        elif t == w:
            atr[t] = seed
        else:
            atr[t] = (atr[t - 1] * (w - 1) + tr[t]) / w
    out = []
    for t in range(n):
        c = cs[t]
        out.append(float(atr[t] / c) if c > 0 else 0.0)
    return out


def apply_atr_stop(long_pos, short_pos, target_ret, atrn, mult):
    """ATR adverse-excursion exit on binary positions (pre-vol_scale, post-stops).

    Per row: track entry excursion exc (cum cur*R since entry). Force flat
    when exc <= -mult*atrn[t] (current-bar normalized ATR). Natural
    flats/resets pass through. mult=None/0 is a no-op. Returns tensors
    on the input device.
    """
    if mult is None or float(mult) <= 0:
        return long_pos, short_pos
    m = float(mult)
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
        cur, exc = 0.0, 0.0
        for t in range(len(L[i])):
            want = 1.0 if L[i][t] > 0.5 else (-1.0 if S[i][t] > 0.5 else 0.0)
            if want != cur:
                cur, exc = want, 0.0
            if cur != 0.0:
                exc += cur * R[i][t] if t < len(R[i]) else 0.0
                thr = m * (atrn[t] if t < len(atrn) else 0.0)
                if thr > 0 and exc <= -thr:
                    cur, exc = 0.0, 0.0
            rl.append(1.0 if cur > 0 else 0.0)
            rs.append(1.0 if cur < 0 else 0.0)
        out_l.append(rl)
        out_s.append(rs)
    dev = long_pos.device
    return torch.tensor(out_l, device=dev), torch.tensor(out_s, device=dev)


def leg_net(raw, rt, sig, atrn, lth, sth, cd, sl, ts, vt, vw, q, atr_mult,
            fee, fund):
    """Mirror research/run_weight_modes.py leg_net, 15m-native + ATR stop."""
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
    lp, sp = apply_atr_stop(lp, sp, rt, atrn, atr_mult)
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
    open(LOG, "w").write("iter_v1_atr start\n")
    coins = list(SMOKE_COINS) if SMOKE else list(COINS)
    atr_grid = list(SMOKE_ATR) if SMOKE else list(ATR_GRID)
    w = {c: 1.0 / len(coins) for c in coins}
    bars, _closes = common15m(coins)
    n = len(bars[coins[0]])
    if SMOKE:
        n = min(n, 3000)
        bars = {c: bars[c][:n] for c in coins}
    h2a = n // 2
    log("common 15m n=%d h2a=%d coins=%s atr=%s smoke=%s" % (n, h2a, coins, atr_grid, SMOKE))
    assert n > 2000, "15m grid too short: %d" % n
    if not SMOKE:
        assert n > 30000, "15m grid too short: %d" % n
    mats = {c: build_sig(bars[c]) for c in coins}
    atrs = {c: atr_norm_96(bars[c], ATR_WINDOW) for c in coins}
    log("signals built (E10 FORMULA, locked) + ATR%d" % ATR_WINDOW)

    res = {
        "config": {
            "engine": "mirror run_weight_modes leg_net + quantile q0.3 long-mask + cooldown + stops + vol_scale(vt None->1.0) + roll1; 15m native, cd/ts/vw x16",
            "formula": list(FORMULA),
            "basket": {c: dict(BASE_SPECS[c]) for c in coins},
            "weights": dict(w),
            "coins": list(coins),
            "atr_grid": list(atr_grid),
            "atr_window": ATR_WINDOW,
            "atr_unit": "mult of close-normalized Wilder ATR(96x15m); unscaled",
            "uniform_cells": len(atr_grid),
            "percoin_runs": len(coins) * len(atr_grid),
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
            "atr_spec": "post-stop ATR adverse-excursion exit on binary positions pre-vol_scale: flat when excursion <= -mult*atr_norm(t); None reproduces locked config",
            "note": "uniform atr_mult {None,1.5,2.0,3.0} + per-coin atr sensitivity; sl/cd/ts locked. E10 FORMULA untouched; live chain untouched; offline read-only.",
        },
        "status": "partial",
    }
    dump(res)

    def run_leg(c, atr_mult):
        spec = BASE_SPECS[c]
        raw, rt, sg = mats[c]
        return leg_net(raw, rt, sg, atrs[c], spec["lth"], spec["sth"],
                       spec["cd"], spec["sl"], spec["ts"], spec["vt"],
                       spec["vw"], spec["q"], atr_mult, FEE, FUND)

    def eval_cell(atr_map):
        legs, turns = {}, {}
        for c in coins:
            legs[c], turns[c] = run_leg(c, atr_map[c])
        net, turn = portfolio(legs, turns, coins, n)
        return net, turn, seg(net, turn, 0, n), seg(net, turn, h2a, n)["sharpe"]

    base_atr = {c: None for c in coins}
    _, _, base_full, base_h2 = eval_cell(base_atr)
    log("BASE FULL sh=%.3f mdd=%.4f final_x=%.4f to=%.6f H2=%.3f" % (
        base_full["sharpe"], base_full["mdd"], base_full["final_x"],
        base_full["turnover"], base_h2))
    res["base_FULL"] = base_full
    res["base_H2_sharpe"] = base_h2
    res["uniform_rows"] = []
    res["percoin_rows"] = []
    dump(res)

    for am in atr_grid:
        atr_map = {c: am for c in coins}
        _net, _turn, full, h2 = eval_cell(atr_map)
        row = {"atr_mult": am, "FULL": full, "H2_sharpe": h2,
               "d_sharpe_vs_base": round(full["sharpe"] - base_full["sharpe"], 3),
               "d_mdd_vs_base": round(full["mdd"] - base_full["mdd"], 4),
               "d_turnover_vs_base": round(full["turnover"] - base_full["turnover"], 6)}
        res["uniform_rows"].append(row)
        dump(res)
        log("U atr=%s sh=%.3f mdd=%.4f final_x=%.4f H2=%.3f to=%.6f dsh=%+.3f dmdd=%+.4f dto=%+.6f" % (
            am, full["sharpe"], full["mdd"], full["final_x"], h2,
            full["turnover"], row["d_sharpe_vs_base"], row["d_mdd_vs_base"],
            row["d_turnover_vs_base"]))

    for c in coins:
        for am in atr_grid:
            atr_map = dict(base_atr)
            atr_map[c] = am
            _net, _turn, full, h2 = eval_cell(atr_map)
            row = {"coin": c, "atr_mult": am, "FULL": full, "H2_sharpe": h2,
                   "d_sharpe_vs_base": round(full["sharpe"] - base_full["sharpe"], 3),
                   "d_mdd_vs_base": round(full["mdd"] - base_full["mdd"], 4),
                   "d_turnover_vs_base": round(full["turnover"] - base_full["turnover"], 6)}
            res["percoin_rows"].append(row)
            dump(res)
            log("P %s atr=%s sh=%.3f mdd=%.4f final_x=%.4f H2=%.3f to=%.6f dsh=%+.3f" % (
                c, am, full["sharpe"], full["mdd"], full["final_x"], h2,
                full["turnover"], row["d_sharpe_vs_base"]))

    percoin_best = {}
    for c in coins:
        crs = [r for r in res["percoin_rows"] if r["coin"] == c]
        best = max(crs, key=lambda r: (r["FULL"]["sharpe"], -r["FULL"]["mdd"]))
        percoin_best[c] = {"atr_mult": best["atr_mult"], "FULL": best["FULL"],
                           "H2_sharpe": best["H2_sharpe"]}
    res["percoin_best"] = percoin_best

    nodrop = [r for r in res["uniform_rows"] if r["FULL"]["sharpe"] >= base_full["sharpe"]]
    best_cell = min(nodrop, key=lambda r: r["FULL"]["mdd"]) if nodrop else None
    if best_cell is None:
        best_note = "no uniform cell with sharpe>=base; no selection"
    else:
        best_note = ("min FULL mdd among uniform cells with sharpe>=base: "
                     "atr_mult=%s" % (best_cell["atr_mult"],))
    res["best_no_drop_dd_min"] = best_cell
    res["best_note"] = best_note
    log("best_no_drop_dd_min: %s (%s)" % (
        ({k: best_cell[k] for k in ("atr_mult",)} if best_cell else None), best_note))

    res["config"]["uniform_cells"] = len(res["uniform_rows"])
    res["config"]["percoin_runs"] = len(res["percoin_rows"])
    res["verdict"] = "PENDING"
    res["decision"] = "PENDING"
    res["conclusion"] = ("PENDING (P0-3 FAIL): V1 15m ATR-stop sweep "
                         "diagnostic only; no-drop dd-min selection is shelf value, "
                         "no adoption, no live change, live untouched.")
    res["status"] = "done"
    dump(res)
    log("wrote %s verdict=PENDING" % OUT)
    print(json.dumps({"base_FULL": base_full, "base_H2": base_h2,
                      "best_no_drop_dd_min": best_cell,
                      "verdict": "PENDING"}, indent=1))


if __name__ == "__main__":
    main()
