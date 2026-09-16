"""validate_12f_floor_12fold.py -- floor formula 12fold robustness on 30m Top5 (offline).

Loads results/train_12f_30m_floor_best.json formula (12f vocab), runs per-fold
MemeBacktest sharpe on 30m 1y (17520 bars -> 12 folds x 1460 bars) for Top5
with E10 per-coin thresholds. Gates (E10 AB A1): mean>0, median>1.5, n_pos>=8.
Verdict PENDING always (diagnostic); no adoption, live untouched.
Output: results/validate_12f_floor_12fold.json
"""
import csv
import json
import os
import pathlib
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["USE_ADVANCED"] = "1"

import torch

from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest

COINS = ["ETC", "TRX", "ATOM", "APT", "KAS"]
TH = {"ETC": (0.88, 0.12, 18, None), "TRX": (0.85, 0.12, 6, 0.05),
      "ATOM": (0.85, 0.15, 6, 0.05), "APT": (0.88, 0.12, 18, None),
      "KAS": (0.88, 0.12, 6, None)}
BPY = 17520.0
NFOLD = 12


def load_coin(c):
    with open(f"data/data_1y/30m/{c}.csv") as f:
        rows = list(csv.DictReader(f))
    return rows


def fold_metrics(formula, rows, spec, a, b):
    seg = rows[a:b]
    n = len(seg)
    raw = {"open": torch.tensor([[float(r["open"]) for r in seg]]),
           "high": torch.tensor([[float(r["high"]) for r in seg]]),
           "low": torch.tensor([[float(r["low"]) for r in seg]]),
           "close": torch.tensor([[float(r["close"]) for r in seg]]),
           "volume": torch.tensor([[float(r["volume"]) for r in seg]]),
           "liquidity": torch.full((1, n), 1e7),
           "fdv": torch.full((1, n), 1e8)}
    feats = FeatureEngineer.compute_features(raw, use_advanced=True)
    res = StackVM(use_advanced=True).execute(formula, feats)
    if res is None:
        return {"sharpe": -10.0, "mdd": 1.0, "turnover": 0.0, "err": "vm_none"}
    tgt = torch.tensor([[(float(seg[i + 1]["close"]) - float(seg[i]["close"])) / float(seg[i]["close"]) if i < n - 1 else 0.0 for i in range(n)]])
    lth, sth, cd, sl = spec
    bt = MemeBacktest(venue="aster", leverage=2.0, short_enabled=True,
                      funding_override=0.0005, long_th=lth, short_th=sth,
                      cooldown_bars=cd, bars_per_year=BPY, stop_loss=sl)
    sc, _ = bt.evaluate(res, raw, tgt)
    m = bt.last_metrics
    return {"sharpe": round(float(m["sharpe"]), 3), "mdd": round(float(m["max_dd"]), 4),
            "turnover": round(float(m["turnover"]), 5)}


def main():
    best = json.loads(pathlib.Path("results/train_12f_30m_floor_best.json").read_text())
    formula = best["formula"]
    data = {c: load_coin(c) for c in COINS}
    n = min(len(v) for v in data.values())
    per = n // NFOLD
    folds = []
    for k in range(NFOLD):
        a, b = k * per, (k + 1) * per if k < NFOLD - 1 else n
        per_coin = {c: fold_metrics(formula, data[c], TH[c], a, b) for c in COINS}
        vals = [v["sharpe"] for v in per_coin.values()]
        folds.append({"fold": k, "bars": [a, b], "per_coin": per_coin,
                      "basket_mean": round(sum(vals) / len(vals), 3)})
    means = [f["basket_mean"] for f in folds]
    med = statistics.median(means)
    avg = sum(means) / len(means)
    npos = sum(1 for v in means if v > 0)
    gates = {"mean_gt_0": avg > 0, "median_ge_1_5": med >= 1.5, "n_pos_ge_8": npos >= 8}
    out = {"formula": formula, "decode": best.get("decode"), "train_score": best.get("score"),
           "folds": folds, "summary": {"mean": round(avg, 3), "median": round(med, 3),
           "n_pos": npos, "gates": gates,
           "pass_all": all(gates.values())},
           "verdict": "PENDING", "decision": "NO_ADOPTION",
           "note": "diagnostic 12fold on train window (in-sample); fresh-OOS permutation still required; live untouched"}
    pathlib.Path("results/validate_12f_floor_12fold.json").write_text(json.dumps(out, indent=1))
    print(f"DONE mean={avg:.3f} median={med:.3f} n_pos={npos} gates={gates}")


if __name__ == "__main__":
    main()
