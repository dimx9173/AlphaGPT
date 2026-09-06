"""Iter7: take-profit sweep on TRX F_MOM 0.88/0.12/cd3/sl3%."""
import csv, json
import torch
from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest

with open("data_15m_3y/TRX.csv") as f:
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
sig = StackVM().execute([0, 2, 6, 15], FeatureEngineer.compute_features(raw))
rets = [(bars[i+1]["close"]-bars[i]["close"])/bars[i]["close"] for i in range(n-1)] + [0.0]
target = torch.tensor([rets])
out = []
for tp in [None, 0.05, 0.08, 0.12, 0.20]:
    for lev in [1.0, 2.0]:
        bt = MemeBacktest(venue="aster", leverage=lev, short_enabled=True,
                          funding_override=0.0005, long_th=0.88, short_th=0.12,
                          cooldown_bars=3, bars_per_year=2190.0,
                          stop_loss=0.03, take_profit=tp)
        fit, cum = bt.evaluate(sig, raw, target)
        m = bt.last_metrics
        ann = float(cum)/n*2190.0
        out.append({"tp": tp, "lev": lev, "fit": round(float(fit),2),
                    "cum": round(float(cum),3), "ann": round(ann,3),
                    "sharpe": round(m["sharpe"],3), "dd": round(m["max_dd"],3),
                    "turn": round(m["turnover"],4)})
        print(out[-1])
with open("backtest_iter7_TRX.jsonl", "w") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")
best = max(out, key=lambda r: r["sharpe"])
print("BEST:", best)
