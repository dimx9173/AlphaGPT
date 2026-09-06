"""Regime filter: flat when trailing volatility > quantile gate (BTC trained formula)."""
import csv, json
import torch
from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest
import math

FORMULA = [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10]
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
sig = StackVM().execute(FORMULA, FeatureEngineer.compute_features(raw))
rets = [(bars[i+1]["close"]-bars[i]["close"])/bars[i]["close"] for i in range(n-1)] + [0.0]
target = torch.tensor([rets])
# trailing 24-bar stdev of 4h returns as vol gauge
import statistics
vol = []
for t in range(n):
    w = rets[max(0, t-24):t+1]
    vol.append(statistics.pstdev(w) if len(w) > 1 else 0.0)
out = []
for q in [None, 0.90, 0.80, 0.70]:
    thr = sorted(vol)[int(n * q)] if q else None
    mask = torch.tensor([[1.0 if (thr is None or v <= thr) else 0.0 for v in vol]])
    masked_sig = sig * mask
    bt = MemeBacktest(venue="aster", leverage=2.0, short_enabled=True,
                      funding_override=0.0005, long_th=0.88, short_th=0.12,
                      cooldown_bars=6, bars_per_year=2190.0)
    fit, cum = bt.evaluate(masked_sig, raw, target)
    m = bt.last_metrics
    cov = sum(1 for v in vol if thr is not None and v > thr) / n if q else 0.0
    out.append({"q": q, "cover": round(1 - cov, 3), "fit": round(float(fit), 2),
                "cum": round(float(cum), 3), "ann": round(float(cum)/n*2190.0, 3),
                "sharpe": round(m["sharpe"], 3), "dd": round(m["max_dd"], 3),
                "turn": round(m["turnover"], 4)})
    print(out[-1])
