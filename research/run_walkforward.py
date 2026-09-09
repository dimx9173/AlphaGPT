"""Walk-forward: baseline formula evaluated per half-year block (BTC 4h, unbiased)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import csv
import torch
from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest

FORMULA = [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10]
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
NB = 6
bsz = n // NB
print(f"total bars={n}, blocks={NB}, per-block={bsz}")
for b in range(NB):
    sl = slice(b*bsz, (b+1)*bsz if b < NB-1 else n)
    bb = bars[sl]
    m = len(bb)
    raw = {"open": torch.tensor([[x["open"] for x in bb]]),
           "high": torch.tensor([[x["high"] for x in bb]]),
           "low": torch.tensor([[x["low"] for x in bb]]),
           "close": torch.tensor([[x["close"] for x in bb]]),
           "volume": torch.tensor([[x["volume"] for x in bb]]),
           "liquidity": torch.full((1, m), 1e7), "fdv": torch.full((1, m), 1e8)}
    sig = StackVM().execute(FORMULA, FeatureEngineer.compute_features(raw))
    rets = [(bb[i+1]["close"]-bb[i]["close"])/bb[i]["close"] for i in range(m-1)] + [0.0]
    bt = MemeBacktest(venue="aster", leverage=2.0, short_enabled=True,
                      funding_override=0.0005, long_th=0.88, short_th=0.12,
                      cooldown_bars=6, bars_per_year=2190.0)
    fit, cum = bt.evaluate(sig, raw, torch.tensor([rets]))
    mm = bt.last_metrics
    print({"block": b, "bars": m, "fit": round(float(fit),2),
           "cum": round(float(cum),3), "ann": round(float(cum)/m*2190.0,3),
           "sharpe": round(mm["sharpe"],3), "dd": round(mm["max_dd"],3)}, flush=True)
