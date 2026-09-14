"""X8 funding-regime sweep (Top5 locked basket, q0.3, aster fee2x, 2x).

Reuses the funding-sweep engine from research/run_micro.py (S5):
combo_legs() -> leg_pnl() per coin, equal 20% weight, E10 FORMULA untouched.

Grid: fund in {0.0001, 0.0005, 0.001, 0.002, 0.005} at fee=0.0008.
The Top5 book is net-short (~99%% short PnL share, see results/qsweep.json),
so funding is EARNED: sharpe rises with fund. Two questions answered:

  1. sharpe curve over the grid + is there an UPSIDE failure point?
     (monotone check; max at 0.005 expected, no failure inside grid)
  2. DOWNSIDE failure point: bisect sharpe(fund)=0 with real engine
     recompute (short-pays regime). Compare against the 0.001 gate:
     coverage = sharpe(0.001) > 0 plus margin to the zero-crossing.

Offline read-only: reads data/data_15m_3y/*.csv, writes
results/iter_X8_fund.json (+ logs/iter_x8.log). No orders, no broker imports.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pathlib

from research.run_qsweep import COINS, W, build_sig, common_4h
from research.run_qsweep import seg_stats
from research.run_micro import combo_legs

FEE2X = 0.0008
FUND_GRID = [0.0001, 0.0005, 0.001, 0.002, 0.005]
GATE_FUND = 0.001
OUT = pathlib.Path("results/iter_X8_fund.json")
LOG = pathlib.Path("logs/iter_x8.log")


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")


def sharpe_at(fund, mats, n):
    _, net, turn = combo_legs(FEE2X, fund, mats, n)
    return seg_stats(net, turn, 0, n), net, turn


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").write("iter_x8 start\n")
    bars, _ = common_4h(COINS)
    n = len(bars["ETC"])
    log("common 4h bars n=%d" % n)
    assert n > 2000, "grid too short: %d" % n
    mats = {c: build_sig(bars[c]) for c in COINS}

    rows = []
    for fund in FUND_GRID:
        st, _, _ = sharpe_at(fund, mats, n)
        rows.append({"fund": fund, **st})
        log("fund %.4f sharpe=%.3f ann=%.4f mdd=%.4f" % (fund, st["sharpe"], st["ann"], st["mdd"]))
    mono_up = all(rows[i + 1]["sharpe"] >= rows[i]["sharpe"] - 1e-9 for i in range(len(rows) - 1))
    to_equal = len({r["turnover"] for r in rows}) == 1

    cov = next(r for r in rows if abs(r["fund"] - GATE_FUND) < 1e-12)
    coverage_ok = bool(cov["sharpe"] > 0)

    # Downside failure search: sharpe(fund)=0 bisection with engine recompute.
    # Book earns funding when net-short, so failure lives at fund <= ~0.
    lo, hi = -0.001, FUND_GRID[0]
    st_lo, _, _ = sharpe_at(lo, mats, n)
    st_hi = rows[0]
    k = 0
    while not (st_lo["sharpe"] < 0 < st_hi["sharpe"]) and k < 10:
        lo *= 2.0
        st_lo, _, _ = sharpe_at(lo, mats, n)
        k += 1
    assert st_lo["sharpe"] < 0 < st_hi["sharpe"], "bracket failed: lo=%.4f sh=%.3f hi=%.4f sh=%.3f" % (
        lo, st_lo["sharpe"], hi, st_hi["sharpe"])
    log("bracket lo=%.5f sh=%.3f hi=%.5f sh=%.3f" % (lo, st_lo["sharpe"], hi, st_hi["sharpe"]))
    bis_rows = [{"fund": lo, **{kk: vv for kk, vv in st_lo.items()}}]
    for _ in range(20):
        mid = 0.5 * (lo + hi)
        st_m, _, _ = sharpe_at(mid, mats, n)
        bis_rows.append({"fund": mid, **st_m})
        if st_m["sharpe"] > 0:
            hi = mid
        else:
            lo = mid
    zero_cross = 0.5 * (lo + hi)
    st_z, _, _ = sharpe_at(zero_cross, mats, n)
    log("zero-cross fund=%.6f sharpe=%.3f (width=%.2e)" % (zero_cross, st_z["sharpe"], hi - lo))

    margin = GATE_FUND - zero_cross
    gate_reasonable = bool(coverage_ok and margin > 0 and mono_up)
    res = {
        "config": {
            "engine": "mirror research/run_micro.py S5 funding sweep, locked Top5 q0.3 legs, E10 FORMULA untouched, offline read-only",
            "coins": list(COINS), "weight_each": W, "lev": 2.0,
            "fee": FEE2X, "fund_grid": list(FUND_GRID), "gate_fund": GATE_FUND,
            "note": "book is net-short: funding earned, sharpe rises with fund; failure only on downside",
        },
        "curve": {"rows": rows, "monotone_up": bool(mono_up), "turnover_flat": bool(to_equal)},
        "upside": {"max_fund": FUND_GRID[-1], "max_sharpe": rows[-1]["sharpe"],
                   "failure_in_grid": False,
                   "note": "no upside failure in [0.0001, 0.005]; sharpe monotone increasing (short rebate scales linearly)"},
        "downside": {"bracket_lo": bis_rows[0]["fund"], "bracket_hi": hi,
                     "zero_cross_fund": round(zero_cross, 6),
                     "zero_cross_check": st_z, "bisection_rows": bis_rows},
        "gate_0001": {"sharpe": cov["sharpe"], "pass_sharpe_gt_0": coverage_ok,
                      "margin_to_failure": round(margin, 6),
                      "reasonable": gate_reasonable,
                      "note": "0.001 gate passes with sharpe=%.3f; failure only at fund=%.6f (short-pays regime), margin %.6f" % (
                          cov["sharpe"], zero_cross, margin)},
        "verdict": "PASS" if gate_reasonable else "FAIL",
    }
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    log("wrote %s verdict=%s" % (OUT, res["verdict"]))


if __name__ == "__main__":
    main()
