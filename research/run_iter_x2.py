"""X2 stop-loss / time-stop grid (Top5) -- DIAGNOSTIC ONLY.

P0-3 permutation FAILED (per-coin p=0.060/0.159 >= 0.05,
results/permutation.json), so per the PRP global exit rule this step's
conclusion is "PENDING (待定)" and MUST NOT be used as demo-listing
evidence. 待定, 不動 live.

Grid: per coin sl in {None, 0.03, 0.05, 0.08} x ts in {12, 24, 36}
  => 5 coins x 4 x 3 = 60 per-coin cells (one coin perturbed, rest
  locked) + 12 uniform cells (all coins same sl/ts) + base = 73 evals.
Per cell: FULL + H2 maxDD / sharpe / final(cum) + turnover.
Selection: dd-minimum cell subject to FULL sharpe >= base (sharpe 不跌).
No adoption: best cell is reported, not written to any config.

Engine mirrors research/run_qsweep.py exactly: E10 FORMULA, Top5
basket (ETC/TRX/ATOM/APT/KAS, equal 20%), aster perp 2x, fund 0.0005,
base fee 0.0004, quantile_mask_long(q=0.3) on the long leg only +
cooldown + stops + vol_scale (locked vt None -> 1.0) + roll1.

Outputs: results/iter_X2_stops.json (+ logs/iter_x2.log).
Offline read-only: reads data/data_15m_3y/*.csv only. No orders.

Smoke mode (for tests): ITER_X2_SMOKE=1 shrinks to 2 cells
(per-coin ETC sl=0.05/ts=12 + uniform sl=0.05/ts=24) + base.
"""
import json
import os
import pathlib
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.run_qsweep import (
    BASKET_SPECS,
    COINS,
    FEE,
    FUND,
    build_sig,
    common_4h,
    leg_pnl,
    seg_stats,
)

SL_GRID = [None, 0.03, 0.05, 0.08]
TS_GRID = [12, 24, 36]
H2_LEN = 493

OUT = pathlib.Path(os.getenv("ITER_X2_OUT", "results/iter_X2_stops.json"))
LOG = pathlib.Path(os.getenv("ITER_X2_LOG", "logs/iter_x2.log"))

SMOKE = os.getenv("ITER_X2_SMOKE") == "1"


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")


def eval_basket(mats, n, specs):
    legs_net, legs_turn = [], []
    for c in COINS:
        raw, rt, sg = mats[c]
        r = leg_pnl(raw, rt, sg, specs[c], FEE, FUND, specs[c].get("q", 0.3))
        legs_net.append(r["net"])
        legs_turn.append(r["turn"])
    w = 1.0 / len(COINS)
    net = [sum(legs_net[i][t] * w for i in range(len(COINS))) for t in range(n)]
    turn = [sum(legs_turn[i][t] * w for i in range(len(COINS))) for t in range(n)]
    h2a = n - H2_LEN
    return (seg_stats(net, turn, 0, n), seg_stats(net, turn, h2a, n))


def cell_specs(base, scope, coin, sl, ts):
    specs = {c: dict(base[c]) for c in COINS}
    targets = list(COINS) if scope == "uniform" else [coin]
    for c in targets:
        specs[c]["sl"] = sl
        specs[c]["ts"] = ts
    return specs


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").write("iter_x2 start smoke=%s\n" % SMOKE)
    bars, _ = common_4h(COINS)
    n = len(bars["ETC"])
    log("common 4h bars n=%d smoke=%s" % (n, SMOKE))
    assert n > 2000, "grid too short: %d" % n
    mats = {c: build_sig(bars[c]) for c in COINS}
    log("signals built")

    base = {c: dict(BASKET_SPECS[c]) for c in COINS}
    base_full, base_h2 = eval_basket(mats, n, base)
    log("base FULL sharpe=%.3f mdd=%.4f final=%.4f to=%.5f | H2 sharpe=%.3f mdd=%.4f"
        % (base_full["sharpe"], base_full["mdd"], base_full["cum"],
           base_full["turnover"], base_h2["sharpe"], base_h2["mdd"]))

    if SMOKE:
        plan = [("per_coin", "ETC", 0.05, 12),
                ("uniform", None, 0.05, 24)]
    else:
        plan = ([("per_coin", c, sl, ts) for c in COINS
                 for sl in SL_GRID for ts in TS_GRID]
                + [("uniform", None, sl, ts)
                   for sl in SL_GRID for ts in TS_GRID])
    rows = []
    for scope, coin, sl, ts in plan:
        specs = cell_specs(base, scope, coin, sl, ts)
        full, h2 = eval_basket(mats, n, specs)
        rows.append({"scope": scope, "coin": coin, "sl": sl, "ts": ts,
                     "FULL": full, "H2": h2,
                     "d_sharpe": round(full["sharpe"] - base_full["sharpe"], 4),
                     "d_mdd": round(full["mdd"] - base_full["mdd"], 4),
                     "d_final": round(full["cum"] - base_full["cum"], 4)})
        log("%s %s sl=%s ts=%d FULL sh=%.3f(d%+.3f) dd=%.4f(d%+.4f) "
            "final=%.4f(d%+.4f) to=%.5f | H2 sh=%.3f dd=%.4f"
            % (scope, coin, sl, ts, full["sharpe"],
               rows[-1]["d_sharpe"], full["mdd"], rows[-1]["d_mdd"],
               full["cum"], rows[-1]["d_final"], full["turnover"],
               h2["sharpe"], h2["mdd"]))

    # selection: dd minimum subject to FULL sharpe >= base (sharpe 不跌)
    eligible = [r for r in rows if r["FULL"]["sharpe"] >= base_full["sharpe"]]
    order = sorted(range(len(eligible)),
                   key=lambda i: (eligible[i]["FULL"]["mdd"],
                                  -eligible[i]["FULL"]["sharpe"]))
    best = dict(eligible[order[0]]) if order else None
    per_coin_best = {}
    for c in COINS:
        cand = [r for r in rows
                if r["scope"] == "per_coin" and r["coin"] == c
                and r["FULL"]["sharpe"] >= base_full["sharpe"]]
        if cand:
            b = min(cand, key=lambda r: (r["FULL"]["mdd"],
                                         -r["FULL"]["sharpe"]))
            per_coin_best[c] = {"sl": b["sl"], "ts": b["ts"],
                                "FULL": b["FULL"],
                                "d_mdd": b["d_mdd"],
                                "d_sharpe": b["d_sharpe"]}
        else:
            per_coin_best[c] = None
    if best:
        cut = ((base_full["mdd"] - best["FULL"]["mdd"]) / base_full["mdd"]
               if base_full["mdd"] > 1e-12 else 0.0)
        log("best %s %s sl=%s ts=%d dd=%.4f (cut %+.1f%%) sh=%.3f(d%+.3f) "
            "final=%.4f(d%+.4f) | eligible %d/%d"
            % (best["scope"], best["coin"], best["sl"], best["ts"],
               best["FULL"]["mdd"], cut * 100, best["FULL"]["sharpe"],
               best["d_sharpe"], best["FULL"]["cum"], best["d_final"],
               len(eligible), len(rows)))
    else:
        cut = 0.0
        log("no no-drop cell: all %d cells drop sharpe vs base %.3f"
            % (len(rows), base_full["sharpe"]))

    res = {
        "config": {
            "engine": "mirror run_qsweep.py leg_pnl + quantile_mask_long(q=0.3, long-only) + cooldown + stops + vol_scale(vt None -> 1.0) + roll1",
            "formula": "E10 LOCKED (see strategy_manager.config.FORMULA)",
            "basket": {c: {kk: base[c][kk]
                            for kk in ("lth", "sth", "cd", "sl", "ts", "q")}
                       for c in COINS},
            "weights": {c: 1.0 / len(COINS) for c in COINS},
            "venue": "aster", "lev": 2.0, "fund": FUND, "fee": FEE,
            "sl_grid": list(SL_GRID), "ts_grid": list(TS_GRID),
            "select_rule": "dd minimum subject to FULL sharpe >= base "
                           "(dd 最小且 sharpe 不跌); report only, no adoption",
            "smoke": SMOKE,
            "grid_bars": n, "h2_len": H2_LEN,
            "note": "E10 FORMULA untouched; live chain untouched; offline read-only; "
                    "P0-3 FAIL => conclusion PENDING (待定), not demo evidence",
        },
        "base_FULL": base_full,
        "base_H2": base_h2,
        "rows": rows,
        "best_overall": best,
        "per_coin_best": per_coin_best,
        "selection": {"n_cells": len(rows), "n_eligible": len(eligible),
                      "mdd_cut_pct": round(cut * 100, 1) if best else 0.0,
                      "found_no_drop_dd_min": best is not None},
        "decision": "PENDING (待定)",
        "conclusion": ("PENDING (待定): P0-3 permutation FAIL => diagnostic only; "
                       "no adoption; live chain untouched; "
                       + ("best no-drop dd-min cell reported for reference."
                          if best else "no sharpe-no-drop cell found.")),
    }
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    log("wrote %s decision=PENDING (待定)" % OUT)


if __name__ == "__main__":
    main()
