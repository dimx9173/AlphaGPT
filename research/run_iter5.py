"""Iter5: quantile filter sweep on TRX F_MOM (best base: 0.88/0.12/cd3)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import csv, json
import torch
from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest

def load_4h(coin):
    import csv as _c
    with open(f"data/data_15m_3y/{coin}.csv") as f:
        rows = list(_c.DictReader(f))
    bars = []
    for i in range(0, len(rows), 16):
        blk = rows[i:i+16]
        if len(blk) < 16: break
        bars.append({"close": float(blk[-1]["close"]), "open": float(blk[0]["open"]),
                     "high": max(float(x["high"]) for x in blk),
                     "low": min(float(x["low"]) for x in blk),
                     "volume": sum(float(x["volume"]) for x in blk)})
    return bars

bars = load_4h("TRX")
n = len(bars)
raw = {"open": torch.tensor([[b["open"] for b in bars]]),
       "high": torch.tensor([[b["high"] for b in bars]]),
       "low": torch.tensor([[b["low"] for b in bars]]),
       "close": torch.tensor([[b["close"] for b in bars]]),
       "volume": torch.tensor([[b["volume"] for b in bars]]),
       "liquidity": torch.full((1, n), 1e7), "fdv": torch.full((1, n), 1e8)}
feats = FeatureEngineer.compute_features(raw)
sig = StackVM().execute([0, 2, 6, 15], feats)
rets = [(bars[i+1]["close"]-bars[i]["close"])/bars[i]["close"] for i in range(n-1)] + [0.0]
target = torch.tensor([rets])
out = []
for q in [None, 0.30, 0.20, 0.10, 0.05]:
    for cd in [3, 6]:
        bt = MemeBacktest(venue="aster", leverage=2.0, short_enabled=True,
                          funding_override=0.0005, long_th=0.88, short_th=0.12,
                          cooldown_bars=cd, bars_per_year=2190.0)
        bt.set_quantile_filter(q)
        fit, cum = bt.evaluate(sig, raw, target)
        m = bt.last_metrics
        ann = float(cum)/n*2190.0
        out.append({"q": q, "cd": cd, "fit": round(float(fit),2), "cum": round(float(cum),3),
                    "ann": round(ann,3), "sharpe": round(m["sharpe"],3),
                    "dd": round(m["max_dd"],3), "turn": round(m["turnover"],4)})
        print(out[-1])
with open("results/backtest_iter5_TRX.jsonl", "w") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")
