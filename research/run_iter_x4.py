"""X4 cost-gate stress (Top5) -- DIAGNOSTIC ONLY.

P0-3 permutation FAILED (per-coin p=0.060/0.159 >= 0.05,
results/permutation.json), so per the PRP global exit rule this step's
conclusion is "PENDING (待定)" and MUST NOT be used as demo-listing evidence.
Y1B_COST_K stays default OFF; this step only recommends a value for future
enablement. No live behavior change.

Premise: results/swap_frontier.json has knee=null -- eps {0..0.2} x
min-hold {0,1,2} moves turnover only 0.0895 -> 0.0889 (-0.7%) because the
E10 signal is sparse (positions flip rarely; |sg-0.5| p50 ~0.49, so the
hyst band almost never binds). X4 asks the follow-up: under HIGH显式成本
(high explicit cost: slip 10/20bp x funding 0.001/0.002, 4 cells + base
reference), when does the Y1B_COST_K gate bite, and which K is worth
keeping on the shelf?

Engine mirrors research/run_qsweep.py exactly: E10 FORMULA, Top5 basket
(ETC/TRX/ATOM/APT/KAS, equal 20%), aster perp 2x, q0.3 long-mask +
cooldown + stops + vol_scale (locked vt None -> 1.0) + roll1. Base
positions/returns are built ONCE with the locked (FEE/FUND) engine; the
K sweep only re-prices costs and applies the entry gate, so K=0 at the
base cell reproduces the locked config up to accounting choice.

Cost gate semantics (mirrors strategy_manager/y1b_executor.apply_swap_gates
cost branch, funding as |funding| cost):
  edge_per_unit(t) = LEV * |sg(t)-0.5| * trail_vol_60(t)
  block a fresh entry (0 -> position) iff edge < K*(FEE2X+slip+|fund|).
Once entered, the position rides until its natural flat (entry-only gate:
no mid-trade veto churn). Live wiring of edge_per_unit is NOT done here
(executor still passes edge=None); the formula above is the spec for it.

Accounting: stress view uses SYMMETRIC costs --
  tx  = |dpos| * (FEE2X+slip) * LEV, fnd = |pos| * fund * LEV.
Signed-funding rebates (net-short earns funding) would mask cost
sensitivity entirely, defeating a cost-stress diagnostic; documented here.

Bite rule (per cost cell, vs same-cell K=0): turnover cut >= 20%
(same bar as the P0-2 knee rule). Recommendation rule (coded below):
smallest K>0 that bites the top stress cell (20bp/0.002) while staying
near-no-op (blocked < 5%) at the base cell -- a tail guard silent in
normal regimes.

Outputs: results/iter_X4_cost.json (+ logs/iter_x4.log).
Offline read-only: reads data/data_15m_3y/*.csv only. No orders.

Smoke mode (for tests): ITER_X4_SMOKE=1 shrinks to the top stress cell
x K {0,2} only (2 runs).
"""
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
from research.run_qsweep import (
    BASKET_SPECS,
    COINS,
    FEE,
    FUND,
    build_sig,
    common_4h,
    quantile_mask_long,
    seg_stats,
)
from strategy_manager.config import FEE2X, LEV

KS = [0, 1, 2, 3, 5]
COSTS = [
    {"label": "base-5bp-0.0005", "slip": 0.0005, "fund": 0.0005},
    {"label": "10bp-0.001", "slip": 0.001, "fund": 0.001},
    {"label": "10bp-0.002", "slip": 0.001, "fund": 0.002},
    {"label": "20bp-0.001", "slip": 0.002, "fund": 0.001},
    {"label": "20bp-0.002", "slip": 0.002, "fund": 0.002},
]
TOP_CELL = "20bp-0.002"
BASE_CELL = "base-5bp-0.0005"
BITE_CUT = 0.20
NOOP_BLOCKED = 0.05
BARS_PER_YEAR = 2190.0
VOL_WIN = 60

OUT = pathlib.Path(os.getenv("ITER_X4_OUT", "results/iter_X4_cost.json"))
LOG = pathlib.Path(os.getenv("ITER_X4_LOG", "logs/iter_x4.log"))

SMOKE = os.getenv("ITER_X4_SMOKE") == "1"
SMOKE_COSTS = [c for c in COSTS if c["label"] == TOP_CELL]
SMOKE_KS = [0, 2]


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")


def leg_base(bars_c, spec):
    """Locked-engine base position/edge cache for one coin."""
    raw, rt, sg = build_sig(bars_c)
    n = len(bars_c)
    bt = MemeBacktest(venue="aster", leverage=LEV, short_enabled=True,
                      funding_override=FUND, fee_override=FEE,
                      long_th=spec["lth"], short_th=spec["sth"],
                      cooldown_bars=spec["cd"], bars_per_year=BARS_PER_YEAR,
                      stop_loss=spec["sl"], time_stop=spec["ts"],
                      vol_target=spec["vt"], vol_window=spec["vw"])
    s = torch.sigmoid(sg)
    safe = (raw["liquidity"] > bt.min_liq).float()
    lp = (s > bt.long_th).float() * safe
    sp = (s < bt.short_th).float() * safe
    mask = quantile_mask_long(sg, 0.3)
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


def eval_cell(base, n, slip, fund, k):
    unit = FEE2X + slip + fund
    thr = k * unit
    legs = {}
    for c in COINS:
        pos, _rt, edge = base[c]
        legs[c] = apply_entry_gate(pos, edge, thr) if k > 0 else list(pos)
    tx_rate = FEE2X + slip
    W = 1.0 / len(COINS)
    port = [sum(W * legs[c][t] * base[c][1][t] * LEV
                - W * abs(legs[c][t] - (legs[c][t - 1] if t else 0.0)) * tx_rate * LEV
                - W * abs(legs[c][t]) * fund * LEV for c in COINS)
            for t in range(n)]
    turn = [sum(W * abs(legs[c][t] - (legs[c][t - 1] if t else 0.0))
                for c in COINS) for t in range(n)]
    st = seg_stats(port, turn, 0, n)
    tot = sum(sum(1 for t in range(n) if abs(base[c][0][t]) > 0.5) for c in COINS)
    blocked = sum(sum(1 for t in range(n)
                      if abs(base[c][0][t]) > 0.5 and abs(legs[c][t]) < 0.5)
                  for c in COINS)
    return st, (blocked / tot if tot else 0.0)


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").write("iter_x4 start\n")
    costs = SMOKE_COSTS if SMOKE else COSTS
    ks = SMOKE_KS if SMOKE else KS
    bars, _closes = common_4h(COINS)
    n = len(bars[COINS[0]])
    log("common 4h bars n=%d" % n)
    assert n > 2000, "grid too short: %d" % n
    base = {c: leg_base(bars[c], BASKET_SPECS[c]) for c in COINS}
    log("base legs built (locked engine, FEE=%.4f FUND=%.4f)" % (FEE, FUND))

    rows = []
    by_cell = {}
    for cell in costs:
        for k in ks:
            st, blocked = eval_cell(base, n, cell["slip"], cell["fund"], k)
            unit = FEE2X + cell["slip"] + cell["fund"]
            row = {"cost": cell["label"], "slip": cell["slip"],
                   "fund": cell["fund"], "unit": round(unit, 5), "K": k,
                   "thr": round(k * unit, 5), "FULL": st,
                   "blocked_frac": round(blocked, 4)}
            rows.append(row)
            by_cell.setdefault(cell["label"], {})[k] = row
            log("%s K=%d thr=%.5f sh=%.3f cum=%.3f to=%.5f blocked=%.4f"
                % (cell["label"], k, k * unit, st["sharpe"], st["cum"],
                   st["turnover"], blocked))

    for row in rows:
        k0 = by_cell[row["cost"]][ks[0]]
        t0 = k0["FULL"]["turnover"]
        row["turnover_cut_vs_k0"] = round((t0 - row["FULL"]["turnover"]) / t0, 4) if t0 else 0.0
        row["d_sharpe_vs_k0"] = round(row["FULL"]["sharpe"] - k0["FULL"]["sharpe"], 3)

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
                        base_rows[k]["blocked_frac"] * 100, NOOP_BLOCKED * 100))
                break
        if rec_k is None:
            rec_reason = "no K satisfies bite-top + noop-base jointly"
    else:
        rec_reason = "smoke mode: recommendation deferred to full run"
    log("recommendation Y1B_COST_K=%s (%s)" % (rec_k, rec_reason))

    out = {
        "config": {
            "weights": {c: 0.2 for c in COINS},
            "costs": costs,
            "Ks": ks,
            "fee2x": FEE2X,
            "lev": LEV,
            "edge_def": "LEV*|sg-0.5|*trail_vol_60, entry-only gate, abs-cost accounting",
            "bite_cut": BITE_CUT,
            "noop_blocked": NOOP_BLOCKED,
            "smoke": SMOKE,
            "grid_bars": n,
        },
        "rows": rows,
        "first_bite": first_bite,
        "recommendation": {"Y1B_COST_K": rec_k, "reason": rec_reason},
        "conclusion": ("PENDING (待定): P0-3 permutation FAILED, Y1B_COST_K stays "
                       "default OFF, no adoption, no live change; recommended shelf "
                       "value K=%s for future enablement only." % rec_k),
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps({"first_bite": first_bite,
                      "recommendation": out["recommendation"],
                      "conclusion": out["conclusion"]}, indent=1))


if __name__ == "__main__":
    main()
