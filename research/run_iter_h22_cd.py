"""H22 1h cooldown sweep (Top5) -- DIAGNOSTIC ONLY.

P0-3 permutation FAILED (per-coin p>=0.05, results/permutation.json), so per
the PRP global exit rule this step's conclusion is "PENDING" and MUST NOT
be used as demo-listing evidence. No adoption, no live change, live chain
untouched. Offline read-only: reads data/data_1y/1h/*.csv only.

Premise: Top5 locked specs (E10 FORMULA [3,2,7,2,7,11,15,4,4,6,6,10]):
  ETC(0.88/0.12/cd18/None/ts24) TRX(0.85/0.12/cd6/0.05/ts24)
  ATOM(0.85/0.15/cd6/0.05/ts24) APT(0.88/0.12/cd18/None/ts24)
  KAS(0.88/0.12/cd6/None/ts24), q0.3 long-only, equal 0.2 weights,
  aster perp 2x fund0.0005 fee0.0004.

Engine mirrors research/run_weight_modes.py leg_net: quantile q0.3
long-only + cooldown + stops + vol_scale(vt None->1.0) + roll1.
1h NATIVE grid (no aggregation): data/data_1y/1h/{COIN}.csv direct
(~8760 rows, cols timestamp,open,high,low,close,volume,quote_volume,
trades). 4h-bar params scale x4 inside the engine:
cooldown_bars=cd*4, time_stop=ts*4, vol_window=vw*4. BPY=8760.
Stop-loss is a return fraction, unscaled (sl/ts stay per-coin locked).

Sweep A (uniform): cd {3,6,12,24} applied to ALL 5 coins
(sl/ts stay per-coin locked). 4 cells.
Sweep B (per-coin sensitivity): each coin's cd over {3,6,12,24} with
the other four at locked base. 5x4 = 20 runs.

Metric per cell/run: FULL sharpe/turnover/trades (+ann/cum/final_x/mdd/
entries/flips/exits/n) + H2 sharpe/trades. trades = entries
(flat->non-zero) + flips (long<->short direct switch) summed over the
5 executed legs. Selection: best no-drop dd-min = min FULL mdd among
uniform cells with FULL sharpe >= base FULL sharpe. Knee (P0-2 bar):
first turnover cut >=20% vs base with sharpe no-drop, min turnover
wins. Shelf value only, no adoption.

Outputs: results/iter_H22_cd.json (+ logs/iter_h22_cd.log).
Incremental: results JSON dumped after EACH unit (partial survives).

Smoke mode (for tests): ITER_H22_SMOKE=1 shrinks to cd {6,12}
(2 uniform cells) + per-coin cd {6,12} on coins {ETC,TRX}
(2 coins x 2 = 4 runs), first 3000 bars. OUT/LOG overridable via
ITER_H22_OUT / ITER_H22_LOG.
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
CD_GRID = [3, 6, 12, 24]
KNEE_CUT = 0.20
BPY = 8760.0
SCALE = 4  # 4h-bar units -> 1h-bar units

OUT = pathlib.Path(os.getenv("ITER_H22_OUT", "results/iter_H22_cd.json"))
LOG = pathlib.Path(os.getenv("ITER_H22_LOG", "logs/iter_h22_cd.log"))

SMOKE = os.getenv("ITER_H22_SMOKE") == "1"
SMOKE_CD = [6, 12]
SMOKE_COINS = ["ETC", "TRX"]


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
    pos = lp - sp
    gross = pos * rt * bt.leverage
    tx = turn * bt.base_fee * bt.leverage
    fnd = pos * bt.default_funding_rate * bt.leverage
    return (gross - tx - fnd)[0].tolist(), turn[0].tolist(), pos[0].tolist()


def count_trades(pos, a, b):
    """Opened-directional-trade counts over pos[a:b.

    entries: flat->non-zero; flips: long<->short direct switch;
    exits: non-zero->flat; trades = entries + flips.
    Slice-aware: previous position is pos[a-1] (or 0.0 at grid start).
    """
    entries = flips = exits = 0
    prev = pos[a - 1] if a > 0 else 0.0
    for t in range(a, b):
        cur = pos[t]
        if cur != 0.0 and prev == 0.0:
            entries += 1
        elif cur == 0.0 and prev != 0.0:
            exits += 1
        elif cur != 0.0 and prev != 0.0 and (cur > 0) != (prev > 0):
            flips += 1
        prev = cur
    return {"trades": entries + flips, "entries": entries,
            "flips": flips, "exits": exits}


def seg_basket(net, turn, poss, coins, a, b):
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
    out = {"sharpe": round(sh, 3), "ann": round(cum / n * BPY, 4) if n else 0.0,
           "mdd": round(md, 4), "cum": round(cum, 4),
           "final_x": round(1.0 + cum, 4), "n": n,
           "turnover": round(sum(t) / n, 6) if n else 0.0}
    tot = {"trades": 0, "entries": 0, "flips": 0, "exits": 0}
    for c in coins:
        ct = count_trades(poss[c], a, b)
        for k in tot:
            tot[k] += ct[k]
    out.update(tot)
    return out


def portfolio(legs, turns, coins, n):
    w = 1.0 / len(coins)
    net = [sum(legs[c][t] * w for c in coins) for t in range(n)]
    turn = [sum(turns[c][t] * w for c in coins) for t in range(n)]
    return net, turn


def dump_partial(payload):
    OUT.write_text(json.dumps(payload, indent=1))


def base_payload(coins, cd_grid, n, h2a, rows_u, rows_p):
    return {
        "config": {
            "engine": "mirror run_weight_modes leg_net + quantile q0.3 long-only + cooldown + stops + vol_scale(vt None->1.0) + roll1; 1h native, cd/ts/vw x4",
            "formula": list(FORMULA),
            "basket": {c: dict(BASE_SPECS[c]) for c in coins},
            "weights": {c: round(1.0 / len(coins), 4) for c in coins},
            "coins": list(coins),
            "cd_grid": list(cd_grid),
            "cd_unit": "4h-bar, engine x4 internally",
            "knee_cut": KNEE_CUT,
            "knee_rule": "turnover cut>=%.0f%% vs base + FULL sharpe>=base; min turnover wins" % (KNEE_CUT * 100),
            "uniform_cells": len(rows_u),
            "percoin_runs": len(rows_p),
            "venue": "aster",
            "lev": LEV,
            "fund": FUND,
            "fee": FEE,
            "bpy": BPY,
            "scale": SCALE,
            "grid": "1h",
            "grid_bars": n,
            "h2_start": h2a,
            "smoke": SMOKE,
            "partial": True,
            "trades_def": "trades = entries (flat->non-zero) + flips (long<->short); summed over legs; exits reported separately",
            "note": "uniform cd + per-coin cd sensitivity; sl/ts locked. E10 FORMULA untouched; live chain untouched; offline read-only.",
        },
        "uniform_rows": list(rows_u),
        "percoin_rows": list(rows_p),
        "verdict": "PENDING",
    }


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").write("iter_h22_cd start\n")
    coins = list(SMOKE_COINS) if SMOKE else list(COINS)
    cd_grid = list(SMOKE_CD) if SMOKE else list(CD_GRID)
    bars, _closes = common1h(coins)
    n = len(bars[coins[0]])
    if SMOKE:
        n = min(n, 3000)
        bars = {c: bars[c][:n] for c in coins}
    h2a = n // 2
    log("common 1h n=%d h2a=%d coins=%s cd=%s smoke=%s" % (n, h2a, coins, cd_grid, SMOKE))
    assert n > 2000, "1h grid too short: %d" % n
    if not SMOKE:
        assert n > 8000, "1h grid too short: %d" % n
    mats = {c: build_sig(bars[c]) for c in coins}
    log("signals built (E10 FORMULA, locked)")

    def run_leg(c, cd):
        spec = BASE_SPECS[c]
        raw, rt, sg = mats[c]
        return leg_net(raw, rt, sg, spec["lth"], spec["sth"], cd,
                       spec["sl"], spec["ts"], spec["vt"], spec["vw"], spec["q"], FEE, FUND)

    def eval_cell(cd_map):
        legs, turns, poss = {}, {}, {}
        for c in coins:
            legs[c], turns[c], poss[c] = run_leg(c, cd_map[c])
        net, turn = portfolio(legs, turns, coins, n)
        full = seg_basket(net, turn, poss, coins, 0, n)
        h2 = seg_basket(net, turn, poss, coins, h2a, n)
        return net, turn, full, h2["sharpe"], h2["trades"]

    base_cd = {c: BASE_SPECS[c]["cd"] for c in coins}
    _, _, base_full, base_h2, base_h2_tr = eval_cell(base_cd)
    log("BASE FULL sh=%.3f mdd=%.4f final_x=%.4f to=%.6f tr=%d H2=%.3f H2tr=%d" % (
        base_full["sharpe"], base_full["mdd"], base_full["final_x"],
        base_full["turnover"], base_full["trades"], base_h2, base_h2_tr))
    dump_partial({**base_payload(coins, cd_grid, n, h2a, [], []),
                  "base_FULL": base_full, "base_H2_sharpe": base_h2,
                  "base_H2_trades": base_h2_tr})

    uniform_rows = []
    for cd in cd_grid:
        cd_map = {c: cd for c in coins}
        _net, _turn, full, h2, h2tr = eval_cell(cd_map)
        row = {"cd": cd, "FULL": full, "H2_sharpe": h2, "H2_trades": h2tr,
               "d_sharpe_vs_base": round(full["sharpe"] - base_full["sharpe"], 3),
               "d_mdd_vs_base": round(full["mdd"] - base_full["mdd"], 4),
               "d_turnover_vs_base": round(full["turnover"] - base_full["turnover"], 6),
               "turnover_cut_vs_base": round(
                   (base_full["turnover"] - full["turnover"]) / base_full["turnover"], 4)
               if base_full["turnover"] > 0 else 0.0}
        uniform_rows.append(row)
        log("U cd=%d sh=%.3f mdd=%.4f final_x=%.4f H2=%.3f to=%.6f tr=%d dsh=%+.3f dmdd=%+.4f cut=%.1f%%" % (
            cd, full["sharpe"], full["mdd"], full["final_x"], h2,
            full["turnover"], full["trades"],
            row["d_sharpe_vs_base"], row["d_mdd_vs_base"],
            row["turnover_cut_vs_base"] * 100))
        dump_partial({**base_payload(coins, cd_grid, n, h2a, list(uniform_rows), []),
                      "base_FULL": base_full, "base_H2_sharpe": base_h2,
                      "base_H2_trades": base_h2_tr})

    percoin_rows = []
    for c in coins:
        for cd in cd_grid:
            cd_map = dict(base_cd)
            cd_map[c] = cd
            _net, _turn, full, h2, h2tr = eval_cell(cd_map)
            row = {"coin": c, "cd": cd, "FULL": full, "H2_sharpe": h2, "H2_trades": h2tr,
                   "d_sharpe_vs_base": round(full["sharpe"] - base_full["sharpe"], 3),
                   "d_mdd_vs_base": round(full["mdd"] - base_full["mdd"], 4),
                   "d_turnover_vs_base": round(full["turnover"] - base_full["turnover"], 6)}
            percoin_rows.append(row)
            log("P %s cd=%d sh=%.3f mdd=%.4f final_x=%.4f H2=%.3f to=%.6f tr=%d dsh=%+.3f" % (
                c, cd, full["sharpe"], full["mdd"], full["final_x"], h2,
                full["turnover"], full["trades"], row["d_sharpe_vs_base"]))
            dump_partial({**base_payload(coins, cd_grid, n, h2a, list(uniform_rows), list(percoin_rows)),
                          "base_FULL": base_full, "base_H2_sharpe": base_h2,
                          "base_H2_trades": base_h2_tr})

    percoin_best = {}
    for c in coins:
        crs = [r for r in percoin_rows if r["coin"] == c]
        best = max(crs, key=lambda r: (r["FULL"]["sharpe"], -r["FULL"]["mdd"]))
        percoin_best[c] = {"cd": best["cd"], "FULL": best["FULL"],
                           "H2_sharpe": best["H2_sharpe"],
                           "H2_trades": best["H2_trades"]}

    nodrop = [r for r in uniform_rows if r["FULL"]["sharpe"] >= base_full["sharpe"]]
    best_cell = min(nodrop, key=lambda r: r["FULL"]["mdd"]) if nodrop else None
    if best_cell is None:
        best_note = "no uniform cell with sharpe>=base; no selection"
    else:
        best_note = ("min FULL mdd among uniform cells with sharpe>=base: "
                     "cd=%d" % (best_cell["cd"]))
    log("best_no_drop_dd_min: %s (%s)" % (
        ({k: best_cell[k] for k in ("cd",)} if best_cell else None), best_note))

    cand = [r for r in uniform_rows
            if r["turnover_cut_vs_base"] >= KNEE_CUT
            and r["FULL"]["sharpe"] >= base_full["sharpe"]]
    cand.sort(key=lambda r: (r["FULL"]["turnover"], -r["FULL"]["sharpe"]))
    knee = cand[0] if cand else None
    if knee is None:
        knee_note = "no uniform cell with turnover cut>=%.0f%% + sharpe no-drop; knee=null" % (KNEE_CUT * 100)
    else:
        knee_note = ("knee cd=%d: turnover cut=%.1f%% (>=%.0f%%) sharpe %+.3f vs base (hold)" % (
            knee["cd"], knee["turnover_cut_vs_base"] * 100, KNEE_CUT * 100,
            knee["d_sharpe_vs_base"]))
    log("knee: %s (%s)" % (
        ({k: knee[k] for k in ("cd",)} if knee else None), knee_note))

    out = base_payload(coins, cd_grid, n, h2a, uniform_rows, percoin_rows)
    out["config"]["partial"] = False
    out.update({
        "base_FULL": base_full,
        "base_H2_sharpe": base_h2,
        "base_H2_trades": base_h2_tr,
        "percoin_best": percoin_best,
        "best_no_drop_dd_min": best_cell,
        "best_note": best_note,
        "knee": knee,
        "knee_note": knee_note,
        "verdict": "PENDING",
        "decision": "PENDING",
        "conclusion": ("PENDING (P0-3 FAIL): H22 1h cooldown sweep diagnostic only; "
                       "knee + no-drop dd-min + per-coin best are shelf value, no adoption, "
                       "no live change, live untouched."),
    })
    OUT.write_text(json.dumps(out, indent=1))
    log("wrote %s verdict=PENDING" % OUT)
    print(json.dumps({"base_FULL": base_full, "base_H2": base_h2,
                      "best_no_drop_dd_min": best_cell,
                      "knee": knee,
                      "verdict": "PENDING"}, indent=1))


if __name__ == "__main__":
    main()
