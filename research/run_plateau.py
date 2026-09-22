"""P0-3: frozen-OOS plateau scan + permutation test (read-only, no orders).

Reads research/oos_freeze.json for the cut index. All selection scans use
in-sample bars [0:cut) only. OOS [cut:N) appears solely as a read-only
locked-point diagnostic, never as selection input.
No broker imports, no orders.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import math
import pathlib

import torch

from research.run_aa import build_mats, combo, leg_series, load_bars, stats

SEED = 7
COINS = ["ETC", "TRX"]
FEE = 0.0004
FUND = 0.0005
# Locked Y1b legs: (lth, sth, cd, sl, ts)
LOCKED = {"ETC": (0.88, 0.12, 18, None, 24), "TRX": (0.85, 0.12, 6, 0.05, 24)}
BASE_STH = 0.12
STH_GRID = [round(BASE_STH * (1.0 + d), 3) for d in (-0.2, -0.1, 0.0, 0.1, 0.2)]
N_PERM = 5000
GAMMA = 0.5772156649  # Euler-Mascheroni


def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p):
    # Acklam rational approximation, double precision.
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


def _eval_combo(mats, sth_etc, sth_trx):
    legs = []
    trs = []
    tos = []
    for coin, sth in (("ETC", sth_etc), ("TRX", sth_trx)):
        lth, _, cd, sl, ts = LOCKED[coin]
        raw, rt, sg = mats[coin]
        net, t, to = leg_series(raw, rt, sg, lth, sth, cd, sl, fee=FEE, fund=FUND)
        legs.append(net)
        trs.append(t)
        tos.append(to)
    cb = combo(legs, [0.5, 0.5])
    s = stats(cb, trades=sum(trs), turnover=sum(tos) / len(tos))
    return s, cb, legs


def _plateau_verdict(cells, sth_grid, band=0.3, min_size=6):
    smax = max(c["sharpe"] for c in cells)
    high = {(c["i"], c["j"]) for c in cells if c["sharpe"] >= smax - band}
    amax = next(c for c in cells if c["sharpe"] == smax)
    seed = (amax["i"], amax["j"])
    seen = {seed}
    stack = [seed]
    while stack:
        i, j = stack.pop()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            q = (i + di, j + dj)
            if q in high and q not in seen:
                seen.add(q)
                stack.append(q)
    locked = next(c for c in cells if c["sth_etc"] == BASE_STH and c["sth_trx"] == BASE_STH)
    locked_high = locked["sharpe"] >= smax - band
    size = len(seen)
    ok = bool(size >= min_size and locked_high)
    return ok, {"smax": smax, "band": band, "min_size": min_size,
                "plateau_size": size, "locked_sharpe": locked["sharpe"],
                "locked_high": bool(locked_high),
                "amax": {"sth_etc": amax["sth_etc"], "sth_trx": amax["sth_trx"],
                         "sharpe": amax["sharpe"]}}


def main():
    freeze = json.load(open("research/oos_freeze.json"))
    cut = int(freeze["cut"]["cut_index"])
    logp = pathlib.Path("logs/plateau.log")
    logp.parent.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(msg, flush=True)
        with open(logp, "a") as f:
            f.write(msg + "\n")

    open(logp, "w").write("plateau start\n")
    full = {c: load_bars(c) for c in COINS}
    n = min(len(b) for b in full.values())
    log("4h bars n=%d cut=%d oos_bars=%d" % (n, cut, n - cut))
    assert cut == 6580, "freeze cut moved: %s" % cut
    assert n > cut, "no OOS bars available"

    mats_is = {c: build_mats(full[c][0:cut]) for c in COINS}
    mats_oos = {c: build_mats(full[c][cut:n]) for c in COINS}

    # 1. plateau grid: sth +/-20%%, 5pts each coin => 25 cells, in-sample only.
    cells = []
    for i, se in enumerate(STH_GRID):
        for j, st in enumerate(STH_GRID):
            s, _, _ = _eval_combo(mats_is, se, st)
            cells.append({"i": i, "j": j, "sth_etc": se, "sth_trx": st,
                          "sharpe": s["sharpe"], "ann": s["ann"], "mdd": s["mdd"],
                          "cum": s["cum"], "trades": s["trades"],
                          "turnover": s["turnover"]})
            log("grid etc=%.3f trx=%.3f sharpe=%.3f ann=%.3f mdd=%.4f to=%.4f" % (
                se, st, s["sharpe"], s["ann"], s["mdd"], s["turnover"]))
    ok, info = _plateau_verdict(cells, STH_GRID)
    verdict = "PASS" if ok else "FAIL"
    log("plateau smax=%.3f size=%d locked=%.3f high=%s => %s" % (
        info["smax"], info["plateau_size"], info["locked_sharpe"],
        info["locked_high"], verdict))

    # 2. locked-point OOS diagnostic (read-only, never selection input).
    s_oos, _, _ = _eval_combo(mats_oos, BASE_STH, BASE_STH)
    s_is, cb_is, legs_is = _eval_combo(mats_is, BASE_STH, BASE_STH)
    log("locked IS sharpe=%.3f OOS(diagnostic) sharpe=%.3f n_oos=%d" % (
        s_is["sharpe"], s_oos["sharpe"], n - cut))

    plateau = {
        "config": {"engine": "mirror research/run_aa.py leg_series + quantile q0.3 long-only",
                   "in_sample": [0, cut], "oos": [cut, n],
                   "oos_role": "diagnostic-only, no selection on OOS",
                   "sth_grid_etc": STH_GRID, "sth_grid_trx": STH_GRID,
                   "locked": {"sth": BASE_STH, "etc_cd": 18, "trx_cd": 6},
                   "fee": FEE, "fund": FUND, "seed": SEED},
        "heatmap": cells,
        "plateau": {"verdict": verdict, **info},
        "locked_insample": s_is,
        "locked_oos_diagnostic": {**s_oos, "diagnostic_only": True},
    }
    pathlib.Path("results").mkdir(parents=True, exist_ok=True)
    json.dump(plateau, open("results/plateau.json", "w"), indent=1)
    log("saved results/plateau.json verdict=%s" % verdict)

    # 3. permutation: per-coin signal shuffle x200 + both-shuffled x200.
    obs = s_is["sharpe"]
    g = torch.Generator().manual_seed(SEED)
    per_coin = {}
    for coin in COINS:
        raw, rt, sg = mats_is[coin]
        nn = sg.shape[1]
        other = 1 - COINS.index(coin)
        fixed = legs_is[other]
        lth, _, cd, sl, ts = LOCKED[coin]
        nulls = []
        for _ in range(N_PERM):
            perm = torch.randperm(nn, generator=g)
            psig = sg[:, perm]
            net, t, to = leg_series(raw, rt, psig, lth, BASE_STH, cd, sl,
                                    fee=FEE, fund=FUND)
            pair = [net, fixed] if COINS.index(coin) == 0 else [fixed, net]
            cb = combo(pair, [0.5, 0.5])
            nulls.append(stats(cb)["sharpe"])
        nulls_sorted = sorted(nulls)
        p = (1.0 + sum(1 for x in nulls if x >= obs)) / (1.0 + N_PERM)
        per_coin[coin] = {"n_perm": N_PERM, "observed": obs,
                          "null_mean": round(sum(nulls) / len(nulls), 4),
                          "null_std": round((sum((x - sum(nulls) / len(nulls)) ** 2
                                                 for x in nulls) / max(len(nulls) - 1, 1)) ** 0.5, 4),
                          "null_median": nulls_sorted[len(nulls_sorted) // 2],
                          "null_max": max(nulls), "p_value": round(p, 4),
                          "gate_p_lt_0_05": bool(p < 0.05)}
        log("%s perm5000 null_mean=%.3f std=%.3f max=%.3f obs=%.3f p=%.4f => %s" % (
            coin, per_coin[coin]["null_mean"], per_coin[coin]["null_std"],
            per_coin[coin]["null_max"], obs, p,
            "PASS" if p < 0.05 else "FAIL"))

    both_nulls = []
    for _ in range(N_PERM):
        pair = []
        for coin in COINS:
            raw, rt, sg = mats_is[coin]
            perm = torch.randperm(sg.shape[1], generator=g)
            lth, _, cd, sl, ts = LOCKED[coin]
            net, t, to = leg_series(raw, rt, sg[:, perm], lth, BASE_STH, cd, sl,
                                    fee=FEE, fund=FUND)
            pair.append(net)
        both_nulls.append(stats(combo(pair, [0.5, 0.5]))["sharpe"])
    p_both = (1.0 + sum(1 for x in both_nulls if x >= obs)) / (1.0 + N_PERM)

    # 4. Deflated Sharpe (Bailey & Lopez de Prado), trials = 5 x n_coins.
    trials = 5 * len(COINS)
    grid_sharpes = [c["sharpe"] for c in cells]
    gm = sum(grid_sharpes) / len(grid_sharpes)
    var_hat = sum((x - gm) ** 2 for x in grid_sharpes) / max(len(grid_sharpes) - 1, 1)
    sr0 = math.sqrt(var_hat) * ((1.0 - GAMMA) * _norm_ppf(1.0 - 1.0 / trials)
                                + GAMMA * _norm_ppf(1.0 - 1.0 / (trials * math.e)))
    _, _, skew, kurt = _moments(cb_is)
    tlen = len(cb_is)
    den = math.sqrt(max(1e-12, 1.0 - skew * obs + (kurt - 1.0) / 4.0 * obs ** 2))
    dsr = _norm_cdf((obs - sr0) * math.sqrt(tlen - 1) / den)
    log("DSR trials=%d var_hat=%.4f sr0=%.3f skew=%.3f kurt=%.3f dsr=%.4f => %s" % (
        trials, var_hat, sr0, skew, kurt, dsr, "PASS" if dsr > 0.8 else "FAIL"))

    perm = {
        "config": {"in_sample": [0, cut], "n_perm": N_PERM, "seed": SEED,
                   "trials": trials, "engine": "signal shuffle per coin, locked params",
                   "oos_role": "not used; in-sample only"},
        "observed_combo_sharpe": obs,
        "per_coin": per_coin,
        "both_shuffled": {"n_perm": N_PERM, "p_value": round(p_both, 4),
                          "null_mean": round(sum(both_nulls) / len(both_nulls), 4),
                          "null_max": round(max(both_nulls), 4),
                          "gate_p_lt_0_05": bool(p_both < 0.05)},
        "deflated_sharpe": {"trials": trials, "var_hat": round(var_hat, 4),
                            "sr0": round(sr0, 4), "skew": round(skew, 4),
                            "kurt": round(kurt, 4), "t_obs": tlen,
                            "dsr": round(dsr, 4),
                            "gate_dsr_gt_0_8": bool(dsr > 0.8)},
        "gates": {"perm_p_lt_0_05": bool(all(v["gate_p_lt_0_05"] for v in per_coin.values())
                                         and p_both < 0.05),
                  "dsr_gt_0_8": bool(dsr > 0.8)},
    }
    json.dump(perm, open("results/permutation.json", "w"), indent=1)
    log("saved results/permutation.json gates=%s" % json.dumps(perm["gates"]))


if __name__ == "__main__":
    main()
