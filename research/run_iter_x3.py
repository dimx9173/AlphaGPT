"""X3 threshold fine-tune (Top5) -- ROBUSTNESS ONLY, DIAGNOSTIC.

P0-3 permutation FAILED (per-coin p=0.060/0.159 >= 0.05,
results/permutation.json), so per the PRP global exit rule this step's
conclusion is "PENDING (待定)" and MUST NOT be used as demo-listing evidence.

Premise: coarse grid concluded island(孤島) for uniform sth/cd/ts/q moves.
X3 asks the follow-up: per-coin lth/sth +/-0.02 fine-tune -- does FULL
sharpe stay flat (plateau/robust, 高原驗證) or jump (sensitive)?
One coin perturbed at a time, all others locked. No optimum pursuit:
the verdict is flat-vs-sensitive only; no parameter is adopted.

Engine mirrors research/run_grid_coarse.py exactly: E10 FORMULA, Top5
basket (ETC/TRX/ATOM/APT/KAS, equal 20%), aster perp 2x, fund 0.0005,
base fee 0.0004, quantile_mask_long(q) on the long leg only + cooldown +
stops + vol_scale (locked vt None -> 1.0) + roll1.

Base = locked hand config (BASKET_SPECS). Perturbations: per coin
lth +/-0.02, sth +/-0.02 => 5*4 = 20 runs + base = 21 evals.

Flat rule: max |d FULL sharpe| <= FLAT_TOL (0.15) => flat(穩健),
else sensitive(敏感). 0.15 keeps it a robustness check, not an
optimum hunt: observed coarse-grid cell gaps are >> 0.15.

Outputs: results/iter_X3_thresh.json (+ logs/iter_x3.log).
Offline read-only: reads data/data_15m_3y/*.csv only. No orders.

Smoke mode (for tests): ITER_X3_SMOKE=1 shrinks to ETC sth +/-0.02
only (base + 2 runs).
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

DELTA = 0.02
FLAT_TOL = 0.15
PARAMS = ("lth", "sth")
SGNS = (+0.02, -0.02)

OUT = pathlib.Path("results/iter_X3_thresh.json")
LOG = pathlib.Path("logs/iter_x3.log")

SMOKE = os.getenv("ITER_X3_SMOKE") == "1"
SMOKE_COINS = ["ETC"]
SMOKE_PARAMS = ("sth",)


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
    return net, seg_stats(net, turn, 0, n)


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").write("iter_x3 start smoke=%s\n" % SMOKE)
    bars, _ = common_4h(COINS + ["BTC"])
    n = len(bars["ETC"])
    log("common 4h bars n=%d smoke=%s" % (n, SMOKE))
    assert n > 2000, "grid too short: %d" % n
    mats = {c: build_sig(bars[c]) for c in COINS}
    log("signals built")

    base_specs = {c: dict(BASKET_SPECS[c]) for c in COINS}
    _, base_full = eval_basket(mats, n, base_specs)
    base_sh = base_full["sharpe"]
    log("base FULL sharpe=%.3f turnover=%.4f mdd=%.4f"
        % (base_sh, base_full["turnover"], base_full["mdd"]))

    coins = SMOKE_COINS if SMOKE else list(COINS)
    params = SMOKE_PARAMS if SMOKE else PARAMS
    rows = []
    for coin in coins:
        for param in params:
            for sgn in SGNS:
                specs = {c: dict(base_specs[c]) for c in COINS}
                new_v = round(specs[coin][param] + sgn, 4)
                specs[coin][param] = new_v
                _, full = eval_basket(mats, n, specs)
                d = round(full["sharpe"] - base_sh, 4)
                rows.append({"coin": coin, "param": param,
                             "delta": sgn, "new_value": new_v,
                             "FULL": full, "d_sharpe": d})
                log("%s %s%+.2f -> %.2f FULL sharpe=%.3f d=%+.4f to=%.4f mdd=%.4f"
                    % (coin, param, sgn, new_v, full["sharpe"], d,
                       full["turnover"], full["mdd"]))

    order = sorted(range(len(rows)), key=lambda i: abs(rows[i]["d_sharpe"]), reverse=True)
    rows = [rows[i] for i in order]
    max_abs = abs(rows[0]["d_sharpe"]) if rows else 0.0
    lth_max = max([abs(r["d_sharpe"]) for r in rows if r["param"] == "lth"] or [0.0])
    sth_max = max([abs(r["d_sharpe"]) for r in rows if r["param"] == "sth"] or [0.0])
    flat = bool(max_abs <= FLAT_TOL)
    verdict = "flat(穩健)" if flat else "sensitive(敏感)"
    log("max|d|=%.4f (lth %.4f / sth %.4f) tol=%.2f => %s"
        % (max_abs, lth_max, sth_max, FLAT_TOL, verdict))

    res = {
        "config": {
            "engine": "mirror run_grid_coarse.py leg_pnl + quantile_mask_long(q, long-only) + cooldown + stops + vol_scale(vt None -> 1.0) + roll1",
            "formula": "E10 LOCKED (see strategy_manager.config.FORMULA)",
            "basket": {c: {kk: base_specs[c][kk]
                            for kk in ("lth", "sth", "cd", "sl", "ts", "q")}
                       for c in COINS},
            "weights": {c: 1.0 / len(COINS) for c in COINS},
            "venue": "aster", "lev": 2.0, "fund": FUND, "fee": FEE,
            "delta": DELTA, "params": list(params),
            "flat_tol": FLAT_TOL,
            "flat_rule": "max|d FULL sharpe| <= %.2f => flat(穩健), else sensitive(敏感); robustness only, no adoption" % FLAT_TOL,
            "smoke": SMOKE,
            "grid_bars": n,
            "note": "E10 FORMULA untouched; live chain untouched; offline read-only; "
                    "P0-3 FAIL => conclusion PENDING (待定), not demo evidence",
        },
        "base_FULL": base_full,
        "rows": rows,
        "verdict": {"verdict": verdict, "flat": flat,
                    "max_abs_d_sharpe": round(max_abs, 4),
                    "lth_max_abs_d": round(lth_max, 4),
                    "sth_max_abs_d": round(sth_max, 4),
                    "tol": FLAT_TOL, "n_runs": len(rows)},
        "conclusion": ("待定 (P0-3 FAIL): X3 per-coin lth/sth +/-0.02 %s, "
                       "max|d sharpe|=%.4f (tol %.2f); no optimum pursuit, no adoption"
                       % (verdict, max_abs, FLAT_TOL)),
    }
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    log("wrote %s conclusion=%s" % (OUT, res["conclusion"]))


if __name__ == "__main__":
    main()
