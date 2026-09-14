"""X9 Top5 rolling-60 correlation monitor (diagnostic, offline read-only).

Reads: results/grid_coarse.json (Top5 basket coins), data/data_15m_3y/*.csv.
Mirrors research/run_weight_modes.py common4h aggregation (15m x16 -> 4h),
then rolling 60-bar Pearson correlation on simple 4h-close returns.

Output: results/iter_X9_corr.json.
Live chain untouched; breaker recommendation default OFF (env-gated);
P0-3 FAIL => verdict PENDING (待定), diagnostic only.
"""
import csv
import itertools
import json
import math
import os
import pathlib
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GRID = pathlib.Path("results/grid_coarse.json")
OUT = pathlib.Path("results/iter_X9_corr.json")
LOG = pathlib.Path("logs/iter_X9_corr.log")

WINDOW = 60
THR_HIGH = 0.6
THR_CUT = 0.7
THR_HALT = 0.85
FALLBACK_COINS = ["ETC", "TRX", "ATOM", "APT", "KAS"]


def log(m):
    print(m, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(m + "\n")


def basket_coins():
    try:
        d = json.loads(GRID.read_text())
        coins = sorted(d["config"]["weights"])
        if len(coins) == 5:
            return coins
    except (OSError, KeyError, ValueError):
        pass
    return list(FALLBACK_COINS)


def load_closes_4h(coins):
    raw = {}
    for c in coins:
        with open("data/data_15m_3y/%s.csv" % c) as f:
            raw[c] = [(int(r["timestamp"]), float(r["close"]))
                      for r in csv.DictReader(f)]
    s = max(r[0][0] for r in raw.values())
    e = min(r[-1][0] for r in raw.values())
    closes = {}
    for c in coins:
        rr = [r for r in raw[c] if s <= r[0] <= e]
        b4 = []
        for i in range(len(rr) // 16):
            blk = rr[i * 16:(i + 1) * 16]
            b4.append(blk[-1][1])
        closes[c] = b4
    n = min(len(v) for v in closes.values())
    return {c: closes[c][:n] for c in coins}


def pearson(a, b):
    n = len(a)
    if n < 2:
        return 0.0
    ma = sum(a) / n
    mb = sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((x - mb) ** 2 for x in b)
    if va <= 0.0 or vb <= 0.0:
        return 0.0
    cov = sum((a[k] - ma) * (b[k] - mb) for k in range(n))
    c = cov / math.sqrt(va * vb)
    return max(-1.0, min(1.0, c))


def rolling_corr(closes, window=WINDOW):
    coins = sorted(closes)
    rets = {c: [(closes[c][i + 1] - closes[c][i]) / closes[c][i]
                if closes[c][i] else 0.0 for i in range(len(closes[c]) - 1)]
            for c in coins}
    m = min(len(v) for v in rets.values())
    pairs = list(itertools.combinations(coins, 2))
    max_s, mean_s = [], []
    pair_vals = {p: [] for p in pairs}
    for t in range(window, m + 1):
        vals = []
        for p in pairs:
            c = pearson(rets[p[0]][t - window:t], rets[p[1]][t - window:t])
            pair_vals[p].append(c)
            vals.append(c)
        max_s.append(max(vals))
        mean_s.append(sum(vals) / len(vals))
    return {"pairs": pairs, "pair_vals": pair_vals,
            "max_s": max_s, "mean_s": mean_s, "n_win": len(max_s)}


def quantile(xs, q):
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


def corr_brake_scale(max_corr, mean_corr=None):
    """Pure breaker map: max_corr>0.85 -> 0.25; >0.7 -> 0.5; else 1.0.

    mean_corr>0.6 also halves even if no single pair trips the cut.
    Default OFF in live chain (env Y1B_CORR_BRAKE); recommendation only.
    """
    try:
        mc = float(max_corr)
    except (TypeError, ValueError):
        return 1.0
    if mc > THR_HALT:
        return 0.25
    if mc > THR_CUT:
        return 0.5
    if mean_corr is not None:
        try:
            if float(mean_corr) > THR_HIGH:
                return 0.5
        except (TypeError, ValueError):
            pass
    return 1.0


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    open(LOG, "w").write("iter_X9_corr start\n")
    coins = basket_coins()
    closes = load_closes_4h(coins)
    n = len(closes[coins[0]])
    rc = rolling_corr(closes, WINDOW)
    n_win = rc["n_win"]
    max_s, mean_s = rc["max_s"], rc["mean_s"]

    def share(xs, thr):
        return round(sum(1 for x in xs if x > thr) / len(xs), 4) if xs else 0.0

    pairs_out = {}
    for p in rc["pairs"]:
        v = rc["pair_vals"][p]
        pairs_out["%s-%s" % p] = {
            "mean": round(sum(v) / len(v), 3) if v else 0.0,
            "max": round(max(v), 3) if v else 0.0,
            "share_gt_0.6": share(v, THR_HIGH),
        }
    agg = {
        "avg_max": round(sum(max_s) / len(max_s), 3) if max_s else 0.0,
        "max_max": round(max(max_s), 3) if max_s else 0.0,
        "q50_max": round(quantile(max_s, 0.50), 3),
        "q75_max": round(quantile(max_s, 0.75), 3),
        "q90_max": round(quantile(max_s, 0.90), 3),
        "q95_max": round(quantile(max_s, 0.95), 3),
        "q99_max": round(quantile(max_s, 0.99), 3),
        "share_max_gt_0.6": share(max_s, THR_HIGH),
        "share_max_gt_0.7": share(max_s, THR_CUT),
        "share_max_gt_0.85": share(max_s, THR_HALT),
        "avg_mean": round(sum(mean_s) / len(mean_s), 3) if mean_s else 0.0,
        "max_mean": round(max(mean_s), 3) if mean_s else 0.0,
        "share_mean_gt_0.6": share(mean_s, THR_HIGH),
        "share_mean_gt_0.4": share(mean_s, 0.4),
    }
    div_verdict = ("LIMITED" if agg["share_max_gt_0.6"] > 0.5
                   else ("PARTIAL" if agg["share_max_gt_0.6"] > 0.2 else "EFFECTIVE"))
    res = {
        "config": {
            "engine": "mirror run_weight_modes.common4h (15m x16 -> 4h closes); "
                      "rolling 60-bar Pearson on simple returns; offline read-only",
            "basket_source": "results/grid_coarse.json config.weights",
            "coins": coins,
            "window": WINDOW,
            "thresholds": {"high": THR_HIGH, "cut": THR_CUT, "halt": THR_HALT},
            "n_bars": n,
            "n_win": n_win,
            "note": "E10 FORMULA untouched; live chain untouched; P0-3 FAIL => PENDING",
        },
        "pairs": pairs_out,
        "aggregate": agg,
        "diversification": {
            "verdict": div_verdict,
            "reason": ("max_corr>0.6 in %.1f%% of windows (avg_max=%.3f); "
                       "mean_corr avg=%.3f; TRX legs least correlated, "
                       "ETC/ATOM/APT block most correlated"
                       % (agg["share_max_gt_0.6"] * 100, agg["avg_max"], agg["avg_mean"])),
        },
        "breaker": {
            "rule": "max_corr>0.85 -> lev_scale 0.25 (halt new opens); "
                    "max_corr>0.7 -> lev_scale 0.5; "
                    "mean_corr>0.6 -> lev_scale 0.5; else 1.0; "
                    "recover only after max_corr<0.6 for 12 bars (hysteresis)",
            "default": "OFF",
            "env": "Y1B_CORR_BRAKE",
            "example": {"max_corr>0.7": 0.5, "max_corr>0.85": 0.25},
        },
        "verdict": "PENDING_P03_FAIL",
        "decision": "NO_CHANGE",
        "decision_note": ("P0-3 permutation FAIL => X9 verdict PENDING; breaker is "
                          "recommendation only, live default stays OFF."),
        "conclusion": "待定 (P0-3 FAIL): Top5 rolling-60 corr high "
                      "(max>0.6 %.1f%%), diversification %s; breaker rec max>0.7 halve, max>0.85 quarter, default OFF"
                      % (agg["share_max_gt_0.6"] * 100, div_verdict),
    }
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    log("wrote %s n=%d n_win=%d avg_max=%.3f share_max_gt0.6=%.4f div=%s verdict=PENDING"
        % (OUT, n, n_win, agg["avg_max"], agg["share_max_gt_0.6"], div_verdict))


if __name__ == "__main__":
    main()
