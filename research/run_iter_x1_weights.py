"""X1 weight optimization (Top5) -- DIAGNOSTIC ONLY.

P0-3 permutation FAILED (per-coin p>=0.05, results/permutation.json), so per
the PRP global exit rule this step's conclusion is "PENDING (待定, P0-3 FAIL)"
and MUST NOT change the live default (equal 20%) or serve as demo evidence.

Beyond P1-1's equal/invvol/invvol_cap contrast (results/weight_modes.json),
this step tries, on the locked Top5 basket (ETC/TRX/ATOM/APT/KAS, E10 FORMULA,
aster perp 2x, fund 0.0005, base fee 0.0004):

  - meanvar : max-Sharpe (long-only, sum=1, box 0.05-0.40) on in-sample leg
              net returns (fee-included), SLSQP.
  - riskpar : equal-risk-contribution on leg-net covariance, same box.
  - kas10   : fixed candidate, KAS down-weighted to 0.10 (rest 0.225).

Reference rows equal/invvol/invvol_cap are recomputed here as STATIC weights
(invvol from full-sample realized vol via y1b_basket.invvol_weights clip
0.10-0.35; invvol_cap = same weights x static lev scale to vol_target 0.35)
so every candidate shares one engine. P1-1 dynamic numbers are quoted for
traceability only.

Engine mirrors research/run_qsweep.py exactly: leg_pnl + quantile q0.3
long-only + cooldown + stops + vol_scale (locked vt None -> 1.0) + roll1.

Objective: max FULL sharpe s.t. turnover < 0.16 AND 12fold median >= 1.5.
Verdict is always PENDING_P03_FAIL; decision always KEEP_equal.

Outputs: results/iter_X1_weights.json (+ logs/iter_x1_weights.log).
Offline read-only: reads data/data_15m_3y/*.csv only. No orders.

Smoke mode (for tests): ITER_X1_SMOKE=1 restricts to equal + kas10.
"""
import json
import math
import os
import pathlib
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.run_qsweep import (
    BASKET_SPECS,
    BARS_PER_YEAR,
    COINS,
    FEE,
    FEE2X,
    FUND,
    H2_LEN,
    build_sig,
    common_4h,
    leg_pnl,
    seg_stats,
)
from strategy_manager.y1b_basket import invvol_weights

OUT = pathlib.Path("results/iter_X1_weights.json")
LOG = pathlib.Path("logs/iter_x1_weights.log")

SMOKE = os.getenv("ITER_X1_SMOKE") == "1"
W_LO, W_HI = 0.05, 0.40
TO_MAX = 0.16
MED_MIN = 1.5
VOL_TARGET = 0.35
LEV_MIN, LEV_MAX = 0.25, 2.0


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")


def fold12(net, turn):
    n = len(net)
    fn = n // 12
    ss = []
    for i in range(12):
        a = i * fn
        b = a + fn if i < 11 else n
        ss.append(seg_stats(net, turn, a, b)["sharpe"])
    srt = sorted(ss)
    return {"sharpes": [round(x, 3) for x in ss],
            "mean": round(sum(ss) / len(ss), 3),
            "median": round(srt[len(srt) // 2], 3),
            "n_pos": sum(1 for x in ss if x > 0)}


def combine(legs_net, legs_turn, w):
    n = len(legs_net[COINS[0]])
    net = [sum(legs_net[c][t] * w[c] for c in COINS) for t in range(n)]
    turn = [sum(legs_turn[c][t] * w[c] for c in COINS) for t in range(n)]
    return net, turn


def eval_candidate(legs_net, legs_turn, legs2_net, w, n, h2a):
    net, turn = combine(legs_net, legs_turn, w)
    net2, turn2 = combine(legs2_net, legs_turn, w)
    full = seg_stats(net, turn, 0, n)
    h2 = seg_stats(net, turn, h2a, n)
    full2x = seg_stats(net2, turn2, 0, n)
    f12 = fold12(net, turn)
    elig = bool(full["turnover"] < TO_MAX and f12["median"] >= MED_MIN)
    return {"weights": {c: round(w[c], 4) for c in COINS},
            "FULL": full, "H2": h2, "FULL_fee2x": full2x, "fold12": f12,
            "eligible": elig}


def meanvar_weights(legs_net):
    import numpy as np
    try:
        from scipy.optimize import minimize
    except Exception:
        return {c: 0.2 for c in COINS}, "scipy-missing-fallback-equal"
    M = np.array([legs_net[c] for c in COINS], dtype=float)  # 5 x n
    mu = M.mean(axis=1)
    cov = np.cov(M)
    cov = cov + np.eye(len(COINS)) * 1e-12

    def neg_sharpe(w):
        v = float(w @ cov @ w)
        if v <= 0:
            return 0.0
        return -float(w @ mu) / math.sqrt(v) * math.sqrt(BARS_PER_YEAR)

    x0 = np.ones(len(COINS)) / len(COINS)
    r = minimize(neg_sharpe, x0, method="SLSQP",
                 bounds=[(W_LO, W_HI)] * len(COINS),
                 constraints={"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},
                 options={"maxiter": 1000, "ftol": 1e-12})
    w = np.clip(r.x, W_LO, W_HI)
    w = w / w.sum()
    return {c: float(w[i]) for i, c in enumerate(COINS)}, "SLSQP-maxSharpe"


def riskparity_weights(legs_net, iters=2000):
    import numpy as np
    M = np.array([legs_net[c] for c in COINS], dtype=float)
    cov = np.cov(M) + np.eye(len(COINS)) * 1e-12
    n = len(COINS)
    w = np.ones(n) / n
    for _ in range(iters):
        sw = cov @ w
        v = float(w @ sw)
        if v <= 0:
            break
        rc = w * sw / v
        w = w * np.sqrt((1.0 / n) / np.maximum(rc, 1e-12))
        w = np.clip(w, W_LO, W_HI)
        s = w.sum()
        if s <= 0:
            w = np.ones(n) / n
            break
        w = w / s
    return {c: float(w[i]) for i, c in enumerate(COINS)}, "cyclical-sqrt"


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").write("iter_x1_weights start smoke=%s\n" % SMOKE)
    bars, closes = common_4h(COINS)
    n = len(bars["ETC"])
    log("common 4h bars n=%d" % n)
    assert n > 2000, "grid too short: %d" % n
    mats = {c: build_sig(bars[c]) for c in COINS}
    log("signals built")

    legs_net, legs_turn, legs2_net = {}, {}, {}
    for c in COINS:
        raw, rt, sg = mats[c]
        r = leg_pnl(raw, rt, sg, BASKET_SPECS[c], FEE, FUND, 0.3)
        legs_net[c], legs_turn[c] = r["net"], r["turn"]
        r2 = leg_pnl(raw, rt, sg, BASKET_SPECS[c], FEE2X, FUND, 0.3)
        legs2_net[c] = r2["net"]
    log("legs built (fee + fee2x)")

    vols = {}
    for c in COINS:
        cc = closes[c]
        rets = [(cc[i + 1] - cc[i]) / cc[i] for i in range(len(cc) - 1) if cc[i]]
        m = sum(rets) / len(rets)
        var = sum((x - m) ** 2 for x in rets) / max(len(rets) - 1, 1)
        vols[c] = math.sqrt(max(var, 0.0)) * math.sqrt(BARS_PER_YEAR)
    w_inv = invvol_weights(vols, 0.10, 0.35)
    log("static invvol vols=%s w=%s" % (
        {c: round(vols[c], 4) for c in COINS},
        {c: round(w_inv[c], 4) for c in COINS}))

    import numpy as np
    M = np.array([legs_net[c] for c in COINS], dtype=float)
    mu = M.mean(axis=1)
    cov = np.cov(M)
    wv = np.array([w_inv[c] for c in COINS])
    pv = float(wv @ cov @ wv)
    port_vol = math.sqrt(max(pv, 0.0)) * math.sqrt(BARS_PER_YEAR)
    ls = min(max(VOL_TARGET / port_vol if port_vol > 1e-9 else 1.0, LEV_MIN), LEV_MAX)
    log("invvol_cap static port_vol=%.4f lev_scale=%.4f" % (port_vol, ls))

    cands = {}
    cands["equal"] = ({"ETC": 0.2, "TRX": 0.2, "ATOM": 0.2, "APT": 0.2, "KAS": 0.2},
                      "locked-default", 1.0)
    cands["kas10"] = ({"ETC": 0.225, "TRX": 0.225, "ATOM": 0.225, "APT": 0.225, "KAS": 0.10},
                      "fixed-KAS-0.10", 1.0)
    if not SMOKE:
        cands["invvol"] = (dict(w_inv), "static-fullsample-realized-vol-clip0.10-0.35", 1.0)
        cands["invvol_cap"] = (dict(w_inv), "static-invvol-x-lev-scale", round(ls, 4))
        w_mv, mv_how = meanvar_weights(legs_net)
        cands["meanvar"] = (w_mv, mv_how, 1.0)
        w_rp, rp_how = riskparity_weights(legs_net)
        cands["riskpar"] = (w_rp, rp_how, 1.0)

    h2a = n - H2_LEN
    rows = {}
    for name, (w, how, lev) in cands.items():
        ln = {c: [x * lev for x in legs_net[c]] for c in COINS}
        lt = {c: [x * lev for x in legs_turn[c]] for c in COINS}
        l2 = {c: [x * lev for x in legs2_net[c]] for c in COINS}
        ev = eval_candidate(ln, lt, l2, w, n, h2a)
        ev["method"] = how
        ev["lev_scale"] = lev
        rows[name] = ev
        log("%-10s FULL sh=%.3f to=%.5f med12=%.3f | H2 sh=%.3f | fee2x=%.3f | elig=%s | w=%s" % (
            name, ev["FULL"]["sharpe"], ev["FULL"]["turnover"], ev["fold12"]["median"],
            ev["H2"]["sharpe"], ev["FULL_fee2x"]["sharpe"], ev["eligible"], ev["weights"]))

    elig = {k: v for k, v in rows.items() if v["eligible"]}
    best = max(elig, key=lambda k: rows[k]["FULL"]["sharpe"]) if elig else None
    res = {
        "config": {
            "engine": "mirror run_qsweep leg_pnl + quantile q0.3 long-only + cooldown + stops + roll1; static weights",
            "formula": "E10 LOCKED (see strategy_manager.config.FORMULA)",
            "basket": {c: dict(BASKET_SPECS[c]) for c in COINS},
            "coins": list(COINS),
            "venue": "aster", "lev": 2.0, "fund": FUND, "fee": FEE, "fee2x": FEE2X,
            "box": [W_LO, W_HI],
            "objective": "max FULL sharpe s.t. turnover<%.2f AND fold12 median>=%.1f" % (TO_MAX, MED_MIN),
            "smoke": SMOKE, "grid_bars": n, "h2_len": H2_LEN,
            "vols_fullsample_ann": {c: round(vols[c], 4) for c in COINS},
            "invvol_cap_static": {"port_vol": round(port_vol, 4), "vol_target": VOL_TARGET,
                                  "lev_scale": round(ls, 4)},
            "note": "E10 FORMULA untouched; live chain untouched; offline read-only; P0-3 FAIL => conclusion PENDING (待定), not demo evidence",
        },
        "rows": rows,
        "selection": {"eligible": sorted(elig),
                      "best": best,
                      "best_FULL_sharpe": rows[best]["FULL"]["sharpe"] if best else None},
        "verdict": "PENDING_P03_FAIL",
        "decision": "KEEP_equal",
        "decision_note": "待定(P0-3 FAIL),不切預設。權重對照僅供研究, live 維持 equal 20%。",
    }
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    log("wrote %s verdict=%s decision=%s best=%s" % (OUT, res["verdict"], res["decision"], best))


if __name__ == "__main__":
    main()
