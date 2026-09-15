"""H5 cost-gate stress (1h native, Top5) -- DIAGNOSTIC ONLY.

P0-3 permutation FAILED (results/permutation.json), so per the PRP global
exit rule this step's conclusion is "PENDING" and MUST NOT be used as
demo-listing evidence. Y1B_COST_K stays default OFF; this step only
recommends a value for future enablement. No live behavior change.

Engine mirrors research/run_weight_modes.py leg_net: E10 FORMULA
[3,2,7,2,7,11,15,4,4,6,6,10] via StackVM+FeatureEngineer; MemeBacktest
venue=aster lev2 short_enabled fund0.0005; quantile q0.3 long-only +
cooldown + stops + vol_scale(vt None->1.0) + roll1. Static equal 0.2
weights (Top5 locked specs ETC/TRX/ATOM/APT/KAS).

1h native: cd/ts/vw x4, BPY=8760. Data data/data_1y/1h/{COIN}.csv
(~8760 rows: timestamp,open,high,low,close,volume,quote_volume,trades).

Cost gate semantics (mirrors strategy_manager/y1b_executor.apply_swap_gates
cost branch, funding as |funding| cost):
  edge_per_unit(t) = LEV * |sg(t)-0.5| * trail_vol_60(t)
  block a fresh entry (0 -> position) iff edge < K*(fee+slip+|fund|).
Once entered, the position rides until its natural flat (entry-only gate:
no mid-trade veto churn). Live wiring of edge_per_unit is NOT done here
(executor still passes edge=None); the formula above is the spec for it.

Accounting: stress view uses SYMMETRIC costs --
  tx  = |dpos| * (fee+slip) * LEV, fnd = |pos| * fund * LEV.
Signed-funding rebates (net-short earns funding) would mask cost
sensitivity entirely, defeating a cost-stress diagnostic; documented here.

Sweep: 5 cost cells (fee, slip_bp, fund) x K {0,1,2,3,5} = 25 runs.
  base   (0.0004, 0, 0.0005)  unit=0.0009
  fee2x  (0.0008, 0, 0.0005)  unit=0.0013
  slip10 (0.0004,10, 0.0005)  unit=0.0019
  slip20 (0.0004,20, 0.0005)  unit=0.0029
  fundhi (0.0008, 0, 0.002)   unit=0.0028
slip = slip_bp * 1e-4. unit = fee + slip + fund.

Bite rule (per cost cell, vs same-cell K=0): turnover cut >= 20%.
Recommendation rule: smallest K>0 that bites the top cell (fundhi)
while staying near-no-op (blocked < 5%) at the base cell -- a tail
guard silent in normal regimes.

IMPORTANT: results/iter_H5_cost.json is dumped after EACH unit run
(incremental; "complete": false until the final write), so a killed run
still leaves partial rows behind.

Outputs: results/iter_H5_cost.json (+ logs/iter_H5_cost.log).
Offline read-only: reads data/data_1y/1h/*.csv only. No orders.

Smoke mode (for tests): ITER_H5_SMOKE=1 shrinks to the top cell
x K {0,2} only (2 runs). ITER_H5_OUT / ITER_H5_LOG override output
paths (tests use a temp file so the committed artifact is not clobbered).
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
)

assert LEV == 2.0, "LEV lock broken: %r" % LEV

COINS = ["ETC", "TRX", "ATOM", "APT", "KAS"]
SPECS = {"ETC": LOCKED_ETC, "TRX": LOCKED_TRX, "ATOM": LOCKED_ATOM,
         "APT": LOCKED_APT, "KAS": LOCKED_KAS}
W = {c: 0.2 for c in COINS}
KS = [0, 1, 2, 3, 5]
COSTS = [
    {"label": "base", "fee": 0.0004, "slip_bp": 0, "fund": 0.0005},
    {"label": "fee2x", "fee": 0.0008, "slip_bp": 0, "fund": 0.0005},
    {"label": "slip10", "fee": 0.0004, "slip_bp": 10, "fund": 0.0005},
    {"label": "slip20", "fee": 0.0004, "slip_bp": 20, "fund": 0.0005},
    {"label": "fundhi", "fee": 0.0008, "slip_bp": 0, "fund": 0.002},
]
TOP_CELL = "fundhi"
BASE_CELL = "base"
BITE_CUT = 0.20
NOOP_BLOCKED = 0.05
BPY = 8760.0
SCALE = 4
VOL_WIN = 60

OUT = pathlib.Path(os.getenv("ITER_H5_OUT", "results/iter_H5_cost.json"))
LOG = pathlib.Path(os.getenv("ITER_H5_LOG", "logs/iter_H5_cost.log"))

SMOKE = os.getenv("ITER_H5_SMOKE") == "1"
SMOKE_COSTS = [c for c in COSTS if c["label"] == TOP_CELL]
SMOKE_KS = [0, 2]


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


def leg_base(bars_c, spec):
    """Locked-engine base position/edge cache for one coin (1h native)."""
    raw, rt, sg = build_sig(bars_c)
    n = len(bars_c)
    bt = MemeBacktest(venue="aster", leverage=LEV, short_enabled=True,
                      funding_override=0.0005, fee_override=0.0004,
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
    lp = lp.roll(1, dims=1)
    lp[:, 0] = 0
    sp = sp.roll(1, dims=1)
    sp[:, 0] = 0
    pos = (lp - sp)[0].tolist()
    sg_l = s[0].tolist()
    rt_l = rt[0].tolist()
    r = numpy.array(rt_l)
    vol = [float(numpy.std(r[max(0, t - VOL_WIN):t])) if t > 5 else 1e-9
           for t in range(n)]
    edge = [LEV * abs(sg_l[t] - 0.5) * (vol[t] or 1e-9) for t in range(n)]
    return pos, rt_l, edge


def apply_entry_gate(pos, edge, thr):
    """Block fresh 0->position entries with edge < thr; ride once entered."""
    out = list(pos)
    held = False
    for t in range(len(pos)):
        if not held and abs(pos[t]) > 0.5:
            if edge[t] < thr:
                out[t] = 0.0
            else:
                held = True
        elif held and abs(pos[t]) < 0.5:
            held = False
    return out


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


def eval_cell(base, n, fee, slip, fund, k):
    unit = fee + slip + fund
    thr = k * unit
    legs = {}
    for c in COINS:
        pos, _rt, edge = base[c]
        legs[c] = apply_entry_gate(pos, edge, thr) if k > 0 else list(pos)
    tx_rate = fee + slip
    Ww = 1.0 / len(COINS)
    arr = {c: numpy.array(legs[c]) for c in COINS}
    rts = {c: numpy.array(base[c][1]) for c in COINS}
    port = numpy.zeros(n)
    turn = numpy.zeros(n)
    for c in COINS:
        p = arr[c]
        dp = numpy.abs(p - numpy.concatenate([[0.0], p[:-1]]))
        port += Ww * (p * rts[c] * LEV - dp * tx_rate * LEV
                      - numpy.abs(p) * fund * LEV)
        turn += Ww * dp
    st = seg(port.tolist(), turn.tolist(), 0, n)
    tot = sum(sum(1 for t in range(n) if abs(base[c][0][t]) > 0.5)
              for c in COINS)
    blocked = sum(sum(1 for t in range(n)
                      if abs(base[c][0][t]) > 0.5 and abs(legs[c][t]) < 0.5)
                  for c in COINS)
    return st, (blocked / tot if tot else 0.0)


def dump(rows, by_cell, n, costs, ks, complete, first_bite=None, rec=None):
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
            "weights": dict(W),
            "costs": costs,
            "Ks": ks,
            "top_cell": TOP_CELL,
            "base_cell": BASE_CELL,
            "lev": LEV,
            "edge_def": "LEV*|sg-0.5|*trail_vol_60, entry-only gate, "
                        "abs-cost accounting",
            "bite_cut": BITE_CUT,
            "noop_blocked": NOOP_BLOCKED,
            "smoke": SMOKE,
            "grid": "1h",
            "grid_bars": n,
            "bpy": BPY,
            "scale": SCALE,
        },
        "rows": rows,
        "cells_done": sorted(by_cell.keys()),
        "complete": complete,
        "first_bite": first_bite or {},
        "recommendation": rec or {"Y1B_COST_K": None,
                                  "reason": "incomplete run"},
        "verdict": "PENDING",
        "decision": "GATE_OFF",
        "decision_note": "待定,不动YZ1B_COST_K(保持默认OFF)。"
                        "本步只推荐备选K值,不启用,不改现货行为。",
        "conclusion": ("PENDING (待定): 1h原生格子; Y1B_COST_K stays "
                       "default OFF, no adoption, no live change; "
                       "recommended shelf value K=%s for future enablement "
                       "only." % ((rec or {}).get("Y1B_COST_K"))),
    }
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return out


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").write("iter_H5_cost start smoke=%s\n" % SMOKE)
    costs = SMOKE_COSTS if SMOKE else COSTS
    ks = SMOKE_KS if SMOKE else KS
    bars, n = common1h(COINS)
    log("common 1h native n=%d smoke=%s scale=x%d" % (n, SMOKE, SCALE))
    assert n > 5000, "grid too short: %d" % n
    base = {c: leg_base(bars[c], SPECS[c]) for c in COINS}
    log("base legs built (locked engine, fee=0.0004 fund=0.0005)")

    rows = []
    by_cell = {}
    for cell in costs:
        slip = cell["slip_bp"] * 1e-4
        unit = cell["fee"] + slip + cell["fund"]
        for k in ks:
            st, blocked = eval_cell(base, n, cell["fee"], slip,
                                    cell["fund"], k)
            row = {"cost": cell["label"], "fee": cell["fee"],
                   "slip_bp": cell["slip_bp"], "slip": round(slip, 5),
                   "fund": cell["fund"], "unit": round(unit, 5), "K": k,
                   "thr": round(k * unit, 5), "FULL": st,
                   "blocked_frac": round(blocked, 4)}
            rows.append(row)
            by_cell.setdefault(cell["label"], {})[k] = row
            log("%s K=%d thr=%.5f sh=%.3f cum=%.3f to=%.5f blocked=%.4f"
                % (cell["label"], k, k * unit, st["sharpe"], st["cum"],
                   st["turnover"], blocked))
            k0 = by_cell[cell["label"]][ks[0]]
            t0 = k0["FULL"]["turnover"]
            for kk, r in by_cell[cell["label"]].items():
                r["turnover_cut_vs_k0"] = round(
                    (t0 - r["FULL"]["turnover"]) / t0, 4) if t0 else 0.0
                r["d_sharpe_vs_k0"] = round(
                    r["FULL"]["sharpe"] - k0["FULL"]["sharpe"], 3)
            # INCREMENTAL DUMP after each unit run
            dump(rows, by_cell, n, costs, ks, False)
            log("incremental dump: unit %s K=%d done (%d rows) -> %s"
                % (cell["label"], k, len(rows), OUT))

    first_bite = {}
    for cell in costs:
        bite = None
        for k in ks:
            if k == ks[0]:
                continue
            if by_cell[cell["label"]][k]["turnover_cut_vs_k0"] >= BITE_CUT:
                bite = k
                break
        first_bite[cell["label"]] = bite
    log("first_bite (>=%.0f%% turnover cut): %s" % (BITE_CUT * 100, first_bite))

    rec_k, rec_reason = None, ""
    if not SMOKE:
        top = by_cell[TOP_CELL]
        base_rows = by_cell[BASE_CELL]
        for k in KS:
            if k == 0:
                continue
            bites_top = top[k]["turnover_cut_vs_k0"] >= BITE_CUT
            noop_base = base_rows[k]["blocked_frac"] < NOOP_BLOCKED
            if bites_top and noop_base:
                rec_k = k
                rec_reason = (
                    "smallest K with turnover_cut>=%.0f%% at %s (cut=%.1f%%) "
                    "while base-cell blocked=%.2f%% (<%.0f%%): tail guard "
                    "silent in normal regimes, binds under stress" % (
                        BITE_CUT * 100, TOP_CELL,
                        top[k]["turnover_cut_vs_k0"] * 100,
                        base_rows[k]["blocked_frac"] * 100,
                        NOOP_BLOCKED * 100))
                break
        if rec_k is None:
            rec_reason = "no K satisfies bite-top + noop-base jointly"
    else:
        rec_reason = "smoke mode: recommendation deferred to full run"
    rec = {"Y1B_COST_K": rec_k, "reason": rec_reason}
    log("recommendation Y1B_COST_K=%s (%s)" % (rec_k, rec_reason))

    dump(rows, by_cell, n, costs, ks, True, first_bite, rec)
    log("wrote %s complete=true rec_K=%s" % (OUT, rec_k))


if __name__ == "__main__":
    main()
