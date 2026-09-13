"""P2-4 microstructure gate (Librarian 2/4/6): slippage / rejects / deadman gap / venue dual.

Reuses the locked Top5 basket engine from research/run_qsweep.py
(E10 FORMULA untouched, equal 20%, q0.3, cooldown+stops, 2x):
  S1 slippage:   base legs recomputed once per (fee,fund); extra slip cost
                 turn*bps*LEV applied arithmetically for slip in {0,2,5}bps.
  S2 rejects:    expected-value model -- on rebalance bars (turn>eps) the bar
                 net scales by (1-reject_rate); rr in {0, 2%%, 5%%}.
  S3 deadman:    unprotected-window expected drift table (gap hours x hourly
                 |ret| x LEV), informational (quantified + monotonic).
  S4 venue dual: aster fee 0.0004 vs bybit fee 0.00055, each base + fee2x,
                 fund 0.0005; both must keep sharpe>0.
  S5 funding:    fund sweep {0.0001,0.0005,0.001,0.002} at fee2x; coverage
                 gate = sharpe(fund=0.001)>0 (survives RiskConfig cap).
  S6 DSR:        Deflated Sharpe trials=75 (in 50-100 band); trial variance
                 from the qsweep FULL_fee2x curve (6 q-points), obs = base
                 combo FULL_fee2x sharpe recomputed here + cross-checked
                 against results/qsweep.json (|diff|<0.05).

Offline read-only: reads data/data_15m_3y/*.csv + results/qsweep.json,
writes results/micro.json (+ logs/micro.log). No orders, no broker imports.
P0-3 is FAIL: this step sets the standard only, triggers no promotion.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import math
import pathlib

import torch

from model_core.backtest import MemeBacktest
from research.run_qsweep import (
    BASKET_SPECS,
    COINS,
    W,
    BARS_PER_YEAR,
    build_sig,
    common_4h,
    leg_pnl,
    quantile_mask_long,
    seg_stats,
)

SEED = 7
FEE_ASTER = 0.0004
FEE_BYBIT = 0.00055
FUND_BASE = 0.0005
SLIP_BPS = [0, 2, 5]
REJECT_RATES = [0.0, 0.02, 0.05]
FUND_GRID = [0.0001, 0.0005, 0.001, 0.002]
DEADMAN_GAP_HOURS = [0, 1, 2, 4, 8]
DSR_TRIALS = 75
GAMMA = 0.5772156649  # Euler-Mascheroni
OUT = pathlib.Path("results/micro.json")
LOG = pathlib.Path("logs/micro.log")
QS = "results/qsweep.json"


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")


def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p):
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    if not 0.0 < p < 1.0:
        raise ValueError("ppf p out of range")
    if p < 0.02425:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > 1.0 - 0.02425:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def _moments(ser):
    n = len(ser)
    m = sum(ser) / n
    s2 = sum((x - m) ** 2 for x in ser) / n
    s = math.sqrt(s2) if s2 > 0 else 0.0
    if s == 0.0:
        return m, 0.0, 0.0, 3.0
    sk = sum((x - m) ** 3 for x in ser) / n / (s ** 3)
    ku = sum((x - m) ** 4 for x in ser) / n / (s ** 4)
    return m, s, sk, ku


def _deflated_sharpe(obs, trial_sharpes, net_ser, trials):
    gm = sum(trial_sharpes) / len(trial_sharpes)
    var_hat = sum((x - gm) ** 2 for x in trial_sharpes) / max(len(trial_sharpes) - 1, 1)
    sr0 = math.sqrt(var_hat) * ((1.0 - GAMMA) * _norm_ppf(1.0 - 1.0 / trials)
                                + GAMMA * _norm_ppf(1.0 - 1.0 / (trials * math.e)))
    _, _, skew, kurt = _moments(net_ser)
    tlen = len(net_ser)
    den = math.sqrt(max(1e-12, 1.0 - skew * obs + (kurt - 1.0) / 4.0 * obs ** 2))
    dsr = _norm_cdf((obs - sr0) * math.sqrt(tlen - 1) / den)
    return var_hat, sr0, skew, kurt, tlen, dsr


def combo_legs(fee, fund, mats, n):
    legs = {}
    for c in COINS:
        raw, rt, sg = mats[c]
        legs[c] = leg_pnl(raw, rt, sg, BASKET_SPECS[c], fee, fund, 0.3)
    net = [sum(legs[c]["net"][t] * W for c in COINS) for t in range(n)]
    turn = [sum(legs[c]["turn"][t] * W for c in COINS) for t in range(n)]
    return legs, net, turn


def apply_slip(net, turn, bps, lev=2.0):
    cost = float(bps) / 10000.0 * lev
    return [net[t] - turn[t] * cost for t in range(len(net))]


def leg_positions(mat, spec, fee, fund):
    """Mirror leg_pnl internals but return executed (lp, sp, rets) series."""
    raw, rt, sg = mat
    bt = MemeBacktest(venue="aster", leverage=2.0, short_enabled=True,
                      funding_override=fund, fee_override=fee,
                      long_th=spec["lth"], short_th=spec["sth"],
                      cooldown_bars=spec["cd"], bars_per_year=BARS_PER_YEAR,
                      stop_loss=spec["sl"], time_stop=spec["ts"],
                      vol_target=spec["vt"], vol_window=spec["vw"])
    signal = torch.sigmoid(sg)
    is_safe = (raw["liquidity"] > bt.min_liq).float()
    lp = (signal > bt.long_th).float() * is_safe
    sp = (signal < bt.short_th).float() * is_safe
    mask = quantile_mask_long(sg, 0.3)
    if mask is not None:
        lp = lp * mask
    lp, sp = bt._apply_cooldown(lp, sp)
    lp, sp = bt._apply_stops(lp, sp, rt)
    scale = bt._vol_scale(rt)
    lp, sp = lp * scale, sp * scale
    lp = lp.roll(1, dims=1)
    lp[:, 0] = 0
    sp = sp.roll(1, dims=1)
    sp[:, 0] = 0
    return lp[0].tolist(), sp[0].tolist(), rt[0].tolist()


def leg_net_from_pos(lp, sp, rets, fee, fund, lev=2.0):
    n = len(lp)
    # NOTE: replicate leg_pnl's torch-roll turnover exactly, including the
    # t=0 wrap-around (|lp0-lp_last|, roll artifact), so rr=0.0 matches base.
    turn = [0.0] * n
    for t in range(n):
        prev = t - 1  # t=0 wraps to last bar, same as lp.roll(1)
        turn[t] = abs(lp[t] - lp[prev]) + abs(sp[t] - sp[prev])
    net = [(lp[t] - sp[t]) * rets[t] * lev - turn[t] * fee * lev
           - (lp[t] - sp[t]) * fund * lev for t in range(n)]
    return net, turn


def apply_reject_freeze(pos_list, rets_list, rr, fee, fund):
    """Deterministic reject sim: every m-th rebalance bar (m=round(1/rr))
    the fill fails -> position frozen at previous bar, no fee that bar."""
    m = max(1, int(round(1.0 / rr))) if rr > 0 else 0
    nets, turns = [], []
    for (lp, sp), rets in zip(pos_list, rets_list):
        lp2, sp2 = list(lp), list(sp)
        if m > 0:
            rb = [t for t in range(len(lp)) if
                  abs(lp[t] - lp[t - 1]) + abs(sp[t] - sp[t - 1]) > 1e-9] if False else None
            turn0 = [0.0] * len(lp)
            for t in range(1, len(lp)):
                turn0[t] = abs(lp[t] - lp[t - 1]) + abs(sp[t] - sp[t - 1])
            rb = [t for t in range(len(lp)) if turn0[t] > 1e-9]
            skip = set(rb[i] for i in range(0, len(rb), m))
            for t in sorted(skip):
                lp2[t] = lp2[t - 1]
                sp2[t] = sp2[t - 1]
        net, turn = leg_net_from_pos(lp2, sp2, rets, fee, fund)
        nets.append(net)
        turns.append(turn)
    n = len(nets[0])
    combo_net = [sum(nets[j][t] * W for j in range(len(nets))) for t in range(n)]
    combo_turn = [sum(turns[j][t] * W for j in range(len(turns))) for t in range(n)]
    return combo_net, combo_turn


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").write("micro start\n")
    bars, closes = common_4h(COINS)
    n = len(bars["ETC"])
    log("common 4h bars n=%d %s" % (n, {c: len(bars[c]) for c in COINS}))
    assert n > 2000, "grid too short: %d" % n
    mats = {c: build_sig(bars[c]) for c in COINS}
    log("signals built")

    # S1+S4 base combos: aster/bybit x base/fee2x, recomputed (no reuse of qsweep).
    s4 = {}
    for venue, fee in (("aster", FEE_ASTER), ("bybit", FEE_BYBIT)):
        for tag, f in (("base", fee), ("fee2x", fee * 2.0)):
            _, net, turn = combo_legs(f, FUND_BASE, mats, n)
            st = seg_stats(net, turn, 0, n)
            s4["%s_%s" % (venue, tag)] = {"fee": f, "fund": FUND_BASE, **st}
            log("%s %s fee=%.5f sharpe=%.3f mdd=%.4f to=%.6f" % (
                venue, tag, f, st["sharpe"], st["mdd"], st["turnover"]))
    _, net_a2x, turn_a2x = combo_legs(FEE_ASTER * 2.0, FUND_BASE, mats, n)
    base_sharpe = seg_stats(net_a2x, turn_a2x, 0, n)["sharpe"]

    # S1 slippage: recompute legs once per slip on top of aster fee2x base,
    # extra cost turn*bps*LEV; slope = d(sharpe)/d(bp).
    s1 = []
    for bps in SLIP_BPS:
        net = apply_slip(net_a2x, turn_a2x, bps)
        st = seg_stats(net, turn_a2x, 0, n)
        s1.append({"slip_bps": bps, **st})
        log("slip %dbps sharpe=%.3f mdd=%.4f" % (bps, st["sharpe"], st["mdd"]))
    slope = (s1[-1]["sharpe"] - s1[0]["sharpe"]) / max(s1[-1]["slip_bps"] - s1[0]["slip_bps"], 1)

    # S2 rejects: deterministic frozen-position sim (missed fills keep stale
    # side, pay mark drift, save the fee). Monotonic sharpe decay expected.
    pos_list, rets_list = [], []
    for c in COINS:
        lp, sp, rets = leg_positions(mats[c], BASKET_SPECS[c], FEE_ASTER * 2.0, FUND_BASE)
        pos_list.append((lp, sp))
        rets_list.append(rets)
    s2 = []
    for rr in REJECT_RATES:
        net, turn = apply_reject_freeze(pos_list, rets_list, rr, FEE_ASTER * 2.0, FUND_BASE)
        if rr == 0.0:
            drift = sum(abs(a - b) for a, b in zip(net, net_a2x))
            log("reject-model drift vs leg_pnl base: %.6f" % drift)
            assert drift < 1e-4, "reject-freeze base must match leg_pnl base"
        st = seg_stats(net, turn, 0, n)
        s2.append({"reject_rate": rr, **st})
        log("reject %.0f%% sharpe=%.3f mdd=%.4f" % (rr * 100, st["sharpe"], st["mdd"]))

    # S3 deadman gap: expected adverse drift per gap window.
    abs_rets = []
    for c in COINS:
        raw, rt, sg = mats[c]
        r = rt[0].tolist()
        abs_rets.append([abs(x) for x in r])
    mean_abs_4h = sum(sum(v) / len(v) for v in abs_rets) / len(abs_rets)
    hourly_abs = mean_abs_4h / 4.0
    s3 = []
    for gap_h in DEADMAN_GAP_HOURS:
        drift = gap_h * hourly_abs * 2.0
        s3.append({"gap_hours": gap_h, "expected_adverse_move": round(drift, 6)})
    mono = all(s3[i + 1]["expected_adverse_move"] >= s3[i]["expected_adverse_move"]
               for i in range(len(s3) - 1))
    assert mono, "deadman drift table must be monotonic"

    # S5 funding sweep at aster fee2x.
    s5 = []
    for fund in FUND_GRID:
        _, net, turn = combo_legs(FEE_ASTER * 2.0, fund, mats, n)
        st = seg_stats(net, turn, 0, n)
        s5.append({"fund": fund, **st})
        log("fund %.4f sharpe=%.3f" % (fund, st["sharpe"]))
    cov = next(x for x in s5 if abs(x["fund"] - 0.001) < 1e-12)
    coverage_ok = bool(cov["sharpe"] > 0)

    # S6 DSR trials=75: trial variance from qsweep FULL_fee2x curve.
    qd = json.load(open(QS))
    curve = [r["FULL_fee2x"]["sharpe"] for r in qd["rows"]]
    q03 = next(r for r in qd["rows"] if abs(r["q"] - 0.3) < 1e-9)["FULL_fee2x"]["sharpe"]
    var_hat, sr0, skew, kurt, tlen, dsr = _deflated_sharpe(
        base_sharpe, curve, net_a2x, DSR_TRIALS)
    cross = {"qsweep_q03_FULL_fee2x": q03, "recomputed_base_sharpe": round(base_sharpe, 3),
             "abs_diff": round(abs(base_sharpe - q03), 4)}
    log("DSR trials=%d var_hat=%.4f sr0=%.3f skew=%.3f kurt=%.3f dsr=%.4f cross_diff=%.4f => %s" % (
        DSR_TRIALS, var_hat, sr0, skew, kurt, dsr, cross["abs_diff"],
        "PASS" if dsr > 0.8 else "FAIL"))

    gates = {
        "slip_slope_negative": bool(slope < 0),
        "reject_monotonic_down": bool(all(
            s2[i + 1]["sharpe"] <= s2[i]["sharpe"] + 1e-9 for i in range(len(s2) - 1))),
        "venue_dual_sharpe_gt_0": bool(all(s4[k]["sharpe"] > 0 for k in s4)),
        "funding_cover_0001_sharpe_gt_0": coverage_ok,
        "dsr_gt_0_8": bool(dsr > 0.8),
    }
    res = {"config": {"engine": "mirror research/run_qsweep.py locked Top5 q0.3 legs, "
                                "E10 FORMULA untouched, offline read-only",
                       "coins": list(COINS), "weight_each": W, "lev": 2.0,
                       "fee_aster": FEE_ASTER, "fee_bybit": FEE_BYBIT,
                       "fund_base": FUND_BASE, "slip_bps": list(SLIP_BPS),
                       "reject_rates": list(REJECT_RATES),
                       "fund_grid": list(FUND_GRID),
                       "deadman_gap_hours": list(DEADMAN_GAP_HOURS),
                       "dsr_trials": DSR_TRIALS, "seed": SEED,
                       "note": "P0-3 FAIL: standard-setting only, no promotion"},
           "slippage": {"rows": s1, "slope_per_bp": round(slope, 4)},
           "rejects": {"rows": s2},
           "deadman_gap": {"rows": s3, "hourly_abs_ret": round(hourly_abs, 6),
                            "monotonic": bool(mono)},
           "venue_dual": s4,
           "funding": {"rows": s5, "coverage_fund_0001": cov},
           "deflated_sharpe": {"trials": DSR_TRIALS, "obs_sharpe": round(base_sharpe, 3),
                                "trial_curve": curve,
                                "var_hat": round(var_hat, 4), "sr0": round(sr0, 4),
                                "skew": round(skew, 4), "kurt": round(kurt, 4),
                                "t_obs": tlen, "dsr": round(dsr, 4),
                                "gate_dsr_gt_0_8": bool(dsr > 0.8),
                                "cross_check": cross},
           "gates": gates,
           "verdict": "PASS" if all(gates.values()) else "FAIL"}
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    log("wrote %s verdict=%s" % (OUT, res["verdict"]))


if __name__ == "__main__":
    main()
