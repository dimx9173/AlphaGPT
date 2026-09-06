"""Long-only quantile filter on BTC (short leg untouched)."""
import csv
import torch
from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest

with open("data_15m_3y/BTC.csv") as f:
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
sig = StackVM().execute([3,2,7,2,7,11,15,4,4,6,6,10], feats)
rets = [(bars[i+1]["close"]-bars[i]["close"])/bars[i]["close"] for i in range(n-1)] + [0.0]
target = torch.tensor([rets])
for q in [None, 0.5, 0.3, 0.2, 0.1]:
    bt = MemeBacktest(venue="aster", leverage=2.0, short_enabled=True,
                      funding_override=0.0005, long_th=0.88, short_th=0.12,
                      cooldown_bars=6, bars_per_year=2190.0)
    bt.set_quantile_filter(q, side="long")
    fit, cum = bt.evaluate(sig, raw, target)
    m = bt.last_metrics
    print({"q": q, "fit": round(float(fit),2), "cum": round(float(cum),3),
           "ann": round(float(cum)/n*2190.0,3), "sharpe": round(m["sharpe"],3),
           "dd": round(m["max_dd"],3), "lt": round(m["long_turnover"],4),
           "st": round(m["short_turnover"],4)}, flush=True)
