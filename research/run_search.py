"""Iter15: random formula search directly optimizing Sharpe (BTC 4h, unbiased)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import csv, json, random
import torch
from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest
from model_core.ops import OPS_CONFIG

FEATS = 6
ARITY = [c[2] for c in OPS_CONFIG]
OFF = 6
random.seed(123)

def rand_formula(length=12):
    toks, depth = [], 0
    for i in range(length):
        remaining = length - i
        # must keep depth feasible: need (depth-1) more ops最小?
        if depth < 2:
            t = random.randrange(FEATS); toks.append(t); depth += 1
        else:
            # bias to feats to keep valid
            if random.random() < 0.45:
                # op
                oi = random.randrange(len(OPS_CONFIG))
                ar = ARITY[oi]
                if ar <= depth and (depth - ar + 1) + (remaining - 1) >= 1:
                    toks.append(OFF + oi); depth = depth - ar + 1
                else:
                    t = random.randrange(FEATS); toks.append(t); depth += 1
            else:
                t = random.randrange(FEATS); toks.append(t); depth += 1
    # fix ending: reduce to depth 1 with binary ops
    while depth > 1:
        # ADD two top
        toks.append(OFF + 0); depth -= 1
    return toks

with open("data/data_15m_3y/BTC.csv") as f:
    rows = list(csv.DictReader(f))
bars = []
for i in range(0, len(rows), 16):
    blk = rows[i:i+16]
    if len(blk) < 16: break
    bars.append({"close": float(blk[-1]["close"]), "open": float(blk[0]["open"]),
                 "high": max(float(x["high"]) for x in blk),
                 "low": min(float(x["low"]) for x in blk),
                 "volume": sum(float(x["volume"]) for x in blk)})
n = len(bars)
raw = {"open": torch.tensor([[b["open"] for b in bars]]),
       "high": torch.tensor([[b["high"] for b in bars]]),
       "low": torch.tensor([[b["low"] for b in bars]]),
       "close": torch.tensor([[b["close"] for b in bars]]),
       "volume": torch.tensor([[b["volume"] for b in bars]]),
       "liquidity": torch.full((1, n), 1e7), "fdv": torch.full((1, n), 1e8)}
feats = FeatureEngineer.compute_features(raw)
rets = [(bars[i+1]["close"]-bars[i]["close"])/bars[i]["close"] for i in range(n-1)] + [0.0]
target = torch.tensor([rets])
vm = StackVM()
results = []
N = 400
for k in range(N):
    ft = rand_formula(12)
    sig = vm.execute(ft, feats)
    if sig is None: continue
    bt = MemeBacktest(venue="aster", leverage=2.0, short_enabled=True,
                      funding_override=0.0005, long_th=0.88, short_th=0.12,
                      cooldown_bars=6, bars_per_year=2190.0)
    fit, cum = bt.evaluate(sig, raw, target)
    m = bt.last_metrics
    results.append({"formula": ft, "sharpe": round(m["sharpe"],3),
                    "ann": round(float(cum)/n*2190.0,3), "dd": round(m["max_dd"],3),
                    "turn": round(m["turnover"],4), "fit": round(float(fit),2)})
    if len(results) % 50 == 0:
        print(f"done {len(results)} valid, best sharpe={max(r['sharpe'] for r in results):.3f}", flush=True)
results.sort(key=lambda r: r["sharpe"], reverse=True)
with open("results/backtest_search_BTC.jsonl", "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")
print(f"valid={len(results)}/{N}")
for r in results[:10]:
    print(r)
