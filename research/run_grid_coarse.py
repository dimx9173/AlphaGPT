"""P2-2 coarse full-grid sweep (Librarian 3a) -- DIAGNOSTIC ONLY.

P0-3 permutation FAILED (per-coin p=0.060/0.159 >= 0.05,
results/permutation.json), so per the PRP global exit rule this step's
conclusion is "PENDING (待定)" and MUST NOT be used as demo-listing evidence.

Grid: sth 0.08-0.15 step 0.01 (8) x cd {6,12,18,24} (4) x ts {12,24,36} (3)
  x q {0.25,0.3,0.35} (3) = 288 combos (within the PRP 200-500 range).

Engine mirrors research/run_qsweep.py exactly: E10 FORMULA, Top5 basket
(ETC/TRX/ATOM/APT/KAS, equal 20%), aster perp 2x, fund 0.0005, base fee
0.0004, quantile_mask_long(q) on the long leg only + cooldown + stops +
vol_scale (locked vt None -> 1.0) + roll1. Per-coin lth/sl stay at their
locked values; the grid overrides sth/cd/ts/q uniformly.

Objective (per combo, base fee):
  obj = fold12_median_sharpe(FULL net) - LAM*turnover_FULL - MU*max_dd_FULL
  LAM=5.0, MU=1.0 (turnover~0.09 -> ~0.44 penalty; mdd~0.8 -> ~0.8 penalty;
  same order as median sharpe ~2, so no term dominates blindly).

Outputs: Top5, sth-cd heatmap (max obj over ts,q), plateau verdict
(top10% connectivity), White Reality Check (block bootstrap vs the
hand-config benchmark, p<0.05 gate), hand-config-in-plateau diagnostic.

Writes results/grid_coarse.json (+ logs/grid_coarse.log).
Offline read-only: reads data/data_15m_3y/*.csv only. No orders.

Smoke mode (for tests): GRID_COARSE_SMOKE=1 shrinks to
  sth {0.11,0.12} x cd {6,12} x ts {24} x q {0.3} = 4 combos, WRC B=20.
"""
import json
import math
import os
import pathlib
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from research.run_qsweep import (
    BASKET_SPECS,
    COINS,
    FEE,
    FUND,
    H2_LEN,
    build_sig,
    common_4h,
    leg_pnl,
    seg_stats,
)

STH = [round(0.08 + 0.01 * i, 2) for i in range(8)]
CD = [6, 12, 18, 24]
TS = [12, 24, 36]
Q = [0.25, 0.3, 0.35]
LAM = 5.0
MU = 1.0
WRC_B = 1000
WRC_BLOCK = 24
WRC_SEED = 7
N_FOLD = 12
TOP_FRAC = 0.10

OUT = pathlib.Path("results/grid_coarse.json")
LOG = pathlib.Path("logs/grid_coarse.log")

SMOKE = os.getenv("GRID_COARSE_SMOKE") == "1"
if SMOKE:
    STH = [0.11, 0.12]
    CD = [6, 12]
    TS = [24]
    Q = [0.3]
    WRC_B = 20


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")


def fold_median(net):
    n = len(net)
    fn = n // N_FOLD
    ss = []
    for fi in range(N_FOLD):
        a = fi * fn
        b = a + fn if fi < N_FOLD - 1 else n
        s = net[a:b]
        m = sum(s) / len(s)
        var = sum((x - m) ** 2 for x in s) / max(len(s) - 1, 1)
        ss.append(m / math.sqrt(var) * math.sqrt(2190.0) if var > 0 else 0.0)
    ordered = sorted(ss)
    med = (ordered[N_FOLD // 2 - 1] + ordered[N_FOLD // 2]) / 2.0
    return round(med, 3), [round(x, 3) for x in ss]


def eval_combo(mats, n, sth, cd, ts, q):
    legs_net, legs_turn = [], []
    for c in COINS:
        spec = dict(BASKET_SPECS[c])
        spec["sth"] = sth
        spec["cd"] = cd
        spec["ts"] = ts
        raw, rt, sg = mats[c]
        r = leg_pnl(raw, rt, sg, spec, FEE, FUND, q)
        legs_net.append(r["net"])
        legs_turn.append(r["turn"])
    w = 1.0 / len(COINS)
    net = [sum(legs_net[i][t] * w for i in range(len(COINS))) for t in range(n)]
    turn = [sum(legs_turn[i][t] * w for i in range(len(COINS))) for t in range(n)]
    full = seg_stats(net, turn, 0, n)
    h2 = seg_stats(net, turn, n - H2_LEN, n)
    med, folds = fold_median(net)
    obj = med - LAM * full["turnover"] - MU * full["mdd"]
    return {"net": net, "turn": turn, "FULL": full, "H2": h2,
            "fold_median": med, "folds": folds, "objective": round(obj, 4)}


def plateau_verdict(rows):
    k = max(1, int(len(rows) * TOP_FRAC))
    top = rows[:k]
    key = {(r["sth"], r["cd"], r["ts"], r["q"]) for r in top}
    si = {v: i for i, v in enumerate(STH)}
    ci = {v: i for i, v in enumerate(CD)}
    ti = {v: i for i, v in enumerate(TS)}
    qi = {v: i for i, v in enumerate(Q)}
    idx = {(si[r["sth"]], ci[r["cd"]], ti[r["ts"]], qi[r["q"]]) for r in top}

    def neighbours(p):
        a, b, c, d = p
        for q in ((a + 1, b, c, d), (a - 1, b, c, d), (a, b + 1, c, d),
                  (a, b - 1, c, d), (a, b, c + 1, d), (a, b, c - 1, d),
                  (a, b, c, d + 1), (a, b, c, d - 1)):
            if q in idx:
                yield q

    seen = set()
    best = 0
    for p in idx:
        if p in seen:
            continue
        stack, comp = [p], 0
        seen.add(p)
        while stack:
            u = stack.pop()
            comp += 1
            for v in neighbours(u):
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        best = max(best, comp)
    frac = best / len(idx)
    connected = bool(frac >= 0.5)
    return connected, {"top_n": k, "largest_cc": best, "cc_frac": round(frac, 3),
                       "rule": "largest-4nbr-CC >= 50% of top10% => plateau(\u9023\u7247)"}


def white_reality_check(nets, bench, b, block, seed):
    """White (2000)-style RC: best grid mean vs hand-config benchmark.

    Joint circular block bootstrap over centered excess series preserves
    cross-model correlation (data-snooping adjusted). Returns (p, best_obs).
    """
    g = torch.Generator().manual_seed(seed)
    D = torch.tensor(nets, dtype=torch.float64) - torch.tensor(bench, dtype=torch.float64)
    obs = D.mean(dim=1)
    best_obs = float(obs.max())
    Dc = D - obs.unsqueeze(1)
    G, N = Dc.shape
    nblk = max(1, N // block)
    exceed = 0
    for _ in range(b):
        starts = torch.randint(0, N, (nblk,), generator=g).tolist()
        idx = [(s + o) % N for s in starts for o in range(block)][:N]
        stat = float(Dc[:, idx].mean(dim=1).max())
        if stat >= best_obs:
            exceed += 1
    p = (1.0 + exceed) / (1.0 + b)
    return round(p, 4), round(best_obs, 5), exceed


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").write("grid_coarse start smoke=%s\n" % SMOKE)
    bars, _ = common_4h(COINS + ["BTC"])
    n = len(bars["ETC"])
    log("common 4h bars n=%d smoke=%s" % (n, SMOKE))
    assert n > 2000, "grid too short: %d" % n
    mats = {c: build_sig(bars[c]) for c in COINS}
    log("signals built")

    rows = []
    nets = []
    for sth in STH:
        for cd in CD:
            for ts in TS:
                for q in Q:
                    r = eval_combo(mats, n, sth, cd, ts, q)
                    rows.append({"sth": sth, "cd": cd, "ts": ts, "q": q,
                                 "FULL": r["FULL"], "H2": r["H2"],
                                 "fold_median": r["fold_median"],
                                 "folds": r["folds"],
                                 "objective": r["objective"]})
                    nets.append(r["net"])
                    log("sth=%.2f cd=%d ts=%d q=%.2f obj=%.4f med=%.3f "
                        "FULL sh=%.3f to=%.4f mdd=%.4f H2 sh=%.3f"
                        % (sth, cd, ts, q, r["objective"], r["fold_median"],
                           r["FULL"]["sharpe"], r["FULL"]["turnover"],
                           r["FULL"]["mdd"], r["H2"]["sharpe"]))
    order = sorted(range(len(rows)), key=lambda i: rows[i]["objective"], reverse=True)
    rows = [rows[i] for i in order]
    nets_sorted = [nets[i] for i in order]
    _best = {k: rows[0][k] for k in ("sth", "cd", "ts", "q", "fold_median", "objective")}
    log("grid done combos=%d best=%s" % (len(rows), json.dumps(_best)))

    top5 = [{k: r[k] for k in ("sth", "cd", "ts", "q", "FULL", "H2",
                               "fold_median", "objective")} for r in rows[:5]]

    heat = {}
    for r in rows:
        key = (r["sth"], r["cd"])
        if key not in heat or r["objective"] > heat[key]["objective"]:
            heat[key] = {"sth": r["sth"], "cd": r["cd"],
                         "objective": r["objective"],
                         "arg_ts": r["ts"], "arg_q": r["q"],
                         "fold_median": r["fold_median"],
                         "FULL_sharpe": r["FULL"]["sharpe"]}
    heatmap = [heat[(s, c)] for s in STH for c in CD]
    smax = max(c["objective"] for c in heatmap)
    hot = [c for c in heatmap if c["objective"] >= smax - 0.3]
    log("heatmap sth x cd cells=%d smax=%.4f hot(>=smax-0.3)=%d"
        % (len(heatmap), smax, len(hot)))

    connected, plat = plateau_verdict(rows)
    verdict = "plateau(\u9023\u7247)" if connected else "island(\u5b64\u5cf6)"
    log("plateau top10%% n=%d largest_cc=%d frac=%.3f => %s"
        % (plat["top_n"], plat["largest_cc"], plat["cc_frac"], verdict))

    hand_specs = {c: dict(BASKET_SPECS[c]) for c in COINS}
    legs_net, legs_turn = [], []
    for c in COINS:
        raw, rt, sg = mats[c]
        r = leg_pnl(raw, rt, sg, hand_specs[c], FEE, FUND,
                    hand_specs[c].get("q", 0.3))
        legs_net.append(r["net"])
        legs_turn.append(r["turn"])
    w = 1.0 / len(COINS)
    hand_net = [sum(legs_net[i][t] * w for i in range(len(COINS))) for t in range(n)]
    hand_turn = [sum(legs_turn[i][t] * w for i in range(len(COINS))) for t in range(n)]
    hand_full = seg_stats(hand_net, hand_turn, 0, n)
    hand_med, hand_folds = fold_median(hand_net)
    hand_obj = hand_med - LAM * hand_full["turnover"] - MU * hand_full["mdd"]
    rank = 1 + sum(1 for r in rows if r["objective"] > hand_obj)
    k = max(1, int(len(rows) * TOP_FRAC))
    sth_vals = sorted({r["sth"] for r in rows[:k]})
    cd_vals = sorted({r["cd"] for r in rows[:k]})
    hand_sth_set = sorted({hand_specs[c]["sth"] for c in COINS})
    hand_in_box = bool(any(s in sth_vals for s in hand_sth_set)
                       and any(hand_specs[c]["cd"] in cd_vals for c in COINS))
    hand = {"params": {c: {kk: hand_specs[c][kk]
                           for kk in ("lth", "sth", "cd", "sl", "ts", "q")}
                       for c in COINS},
            "FULL": hand_full, "fold_median": hand_med,
            "folds": hand_folds, "objective": round(hand_obj, 4),
            "rank_vs_grid": rank, "grid_n": len(rows),
            "top10_sth": sth_vals, "top10_cd": cd_vals,
            "in_plateau_box": hand_in_box}
    log("hand obj=%.4f med=%.3f rank=%d/%d in_box=%s"
        % (hand_obj, hand_med, rank, len(rows), hand_in_box))

    p_val, best_obs, exceed = white_reality_check(nets_sorted, hand_net, WRC_B, WRC_BLOCK, WRC_SEED)
    wrc_pass = bool(p_val < 0.05)
    log("WRC B=%d block=%d best_excess=%.5f exceed=%d p=%.4f => %s"
        % (WRC_B, WRC_BLOCK, best_obs, exceed, p_val, "PASS" if wrc_pass else "FAIL"))

    res = {
        "config": {
            "engine": "mirror run_qsweep.py leg_pnl + quantile_mask_long(q, long-only) + cooldown + stops + vol_scale(vt None -> 1.0) + roll1",
            "formula": "E10 LOCKED (see strategy_manager.config.FORMULA)",
            "basket": {c: {kk: BASKET_SPECS[c][kk]
                            for kk in ("lth", "sth", "cd", "sl", "ts", "q")}
                       for c in COINS},
            "weights": {c: 1.0 / len(COINS) for c in COINS},
            "venue": "aster", "lev": 2.0, "fund": FUND, "fee": FEE,
            "grid": {"sth": list(STH), "cd": list(CD), "ts": list(TS), "q": list(Q)},
            "combos": len(rows), "smoke": SMOKE,
            "objective": "fold12_median_sharpe - %s*turnover_FULL - %s*max_dd_FULL (base fee)" % (LAM, MU),
            "lam": LAM, "mu": MU, "n_fold": N_FOLD, "h2_len": H2_LEN,
            "grid_bars": n,
            "wrc": {"B": WRC_B, "block": WRC_BLOCK, "seed": WRC_SEED,
                    "benchmark": "hand-config Top5 net series (mean excess)",
                    "method": "joint circular block bootstrap on centered excess (White-2000 style)"},
            "note": "E10 FORMULA untouched; live chain untouched; offline read-only; "
                    "P0-3 FAIL => conclusion PENDING (\u5f85\u5b9a), not demo evidence",
        },
        "rows": [{k: r[k] for k in ("sth", "cd", "ts", "q", "FULL", "H2",
                                    "fold_median", "folds", "objective")} for r in rows],
        "top5": top5,
        "heatmap_sth_cd": heatmap,
        "plateau": {"verdict": verdict, "connected": connected, **plat},
        "wrc": {"p_value": p_val, "best_excess_mean": best_obs,
                "exceed": exceed, "gate_p_lt_0_05": wrc_pass},
        "hand_config": hand,
        "conclusion": ("\u5f85\u5b9a (P0-3 FAIL): grid %s, WRC %s, hand rank %d/%d %s"
                       % (verdict, "p=%.4f PASS" % p_val if wrc_pass else "p=%.4f FAIL" % p_val,
                          rank, len(rows),
                          "in-plateau-box" if hand_in_box else "off-plateau-box")),
    }
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    log("wrote %s conclusion=%s" % (OUT, res["conclusion"]))



if __name__ == "__main__":
    main()
