"""Q2 OOS verify (standalone): top3 from train_q5_best.json on all 5 coins
LAST-15% (FULL/H1/H2) with default AND Q1 params + H2 side split, vs baseline."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run_q2 as Q
import json
from model_core.config import ModelConfig
from model_core.vm import StackVM
import torch

device = ModelConfig.DEVICE
vm = StackVM()
tr = json.load(open("results/train_q5_best.json"))
top3 = [(t["train_score"], t["formula"]) for t in tr["top3"]]
all_bars = Q.load_4h(Q.TRAIN_COINS)
report = {"baseline_formula": Q.BASELINE_FORMULA,
          "baseline_decoded": Q.decode(Q.BASELINE_FORMULA),
          "config": {"long_th": Q.LONG_TH, "short_th": Q.SHORT_TH, "cooldown": Q.COOLDOWN,
                     "short_enabled": True, "venue": "aster", "leverage": 2.0,
                     "funding_override": 0.0005},
          "q1_params": Q.Q1, "per_coin": {}}
for coin in Q.TRAIN_COINS:
    bars = all_bars[coin]
    nn = len(bars)
    no = nn - int(nn * Q.TRAIN_FRAC)
    f, t, r = Q.build_single(bars[nn - no:])
    T = f.shape[2] if f.dim() == 3 else f.shape[1]
    q = Q.Q1.get(coin)
    print(f"OOS {coin}: bars={nn} oos={T} q1={q}", flush=True)
    centry = {"oos_bars": T, "q1_params": q, "results": [], "baseline": None}
    for sc, fl in top3:
        e = {"formula": fl, "decoded": Q.decode(fl), "train_score": sc}
        e["default"] = Q.eval_segs(vm, fl, f, t, r, device, Q.LONG_TH, Q.SHORT_TH, Q.COOLDOWN, None)
        e["q1"] = (Q.eval_segs(vm, fl, f, t, r, device, q["lth"], q["sth"], q["cd"], q["sl"])
                   if q is not None else e["default"])
        e["side_default_H2"] = {s: (Q.eval_segs(vm, fl, f, t, r, device, Q.LONG_TH, Q.SHORT_TH,
                                                Q.COOLDOWN, None, side=s, segs=("H2",)) or {}).get("H2")
                                for s in ("both", "long", "short")}
        e["side_q1_H2"] = ({s: (Q.eval_segs(vm, fl, f, t, r, device, q["lth"], q["sth"], q["cd"],
                                            q["sl"], side=s, segs=("H2",)) or {}).get("H2")
                            for s in ("both", "long", "short")} if q is not None else e["side_default_H2"])
        centry["results"].append(e)
        print("OOS " + coin + " " + json.dumps(e), flush=True)
    b = {"formula": Q.BASELINE_FORMULA, "decoded": Q.decode(Q.BASELINE_FORMULA)}
    b["default"] = Q.eval_segs(vm, Q.BASELINE_FORMULA, f, t, r, device, Q.LONG_TH, Q.SHORT_TH, Q.COOLDOWN, None)
    b["q1"] = (Q.eval_segs(vm, Q.BASELINE_FORMULA, f, t, r, device, q["lth"], q["sth"], q["cd"], q["sl"])
               if q is not None else b["default"])
    b["side_default_H2"] = {s: (Q.eval_segs(vm, Q.BASELINE_FORMULA, f, t, r, device, Q.LONG_TH, Q.SHORT_TH,
                                            Q.COOLDOWN, None, side=s, segs=("H2",)) or {}).get("H2")
                            for s in ("both", "long", "short")}
    b["side_q1_H2"] = ({s: (Q.eval_segs(vm, Q.BASELINE_FORMULA, f, t, r, device, q["lth"], q["sth"], q["cd"],
                                        q["sl"], side=s, segs=("H2",)) or {}).get("H2")
                        for s in ("both", "long", "short")} if q is not None else b["side_default_H2"])
    centry["baseline"] = b
    print("BASE " + coin + " " + json.dumps(b), flush=True)
    report["per_coin"][coin] = centry
json.dump(report, open("results/backtest_Q2_train.json", "w"), indent=1)
print("OOS DONE -> backtest_Q2_train.json", flush=True)
