"""Asymmetry + cross-coin check (unbiased, trained formula, 0.88/0.12/cd6)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import csv
import torch
from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest

FORMULA = [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10]

def load(coin):
    with open(f"data/data_15m_3y/{coin}.csv") as f:
        rows = list(csv.DictReader(f))
    bars = []
    for i in range(0, len(rows), 16):
        blk = rows[i:i+16]
        if len(blk) < 16: break
        bars.append({"close": float(blk[-1]["close"]), "open": float(blk[0]["open"]),
                     "high": max(float(x["high"]) for x in blk),
                     "low": min(float(x["low"]) for x in blk),
                     "volume": sum(float(x["volume"]) for x in blk)})
    return bars

def run(coin, long_th, short_th, short_on):
    bars = load(coin)
    n = len(bars)
    raw = {"open": torch.tensor([[b["open"] for b in bars]]),
           "high": torch.tensor([[b["high"] for b in bars]]),
           "low": torch.tensor([[b["low"] for b in bars]]),
           "close": torch.tensor([[b["close"] for b in bars]]),
           "volume": torch.tensor([[b["volume"] for b in bars]]),
           "liquidity": torch.full((1, n), 1e7), "fdv": torch.full((1, n), 1e8)}
    sig = StackVM(use_advanced=False).execute(FORMULA, FeatureEngineer.compute_features(raw, use_advanced=False))
    rets = [(bars[i+1]["close"]-bars[i]["close"])/bars[i]["close"] for i in range(n-1)] + [0.0]
    bt = MemeBacktest(venue="aster", leverage=2.0, short_enabled=short_on,
                      funding_override=0.0005, long_th=long_th, short_th=short_th,
                      cooldown_bars=6, bars_per_year=2190.0)
    fit, cum = bt.evaluate(sig, raw, torch.tensor([rets]))
    m = bt.last_metrics
    r = {"coin": coin, "lth": long_th, "sth": short_th, "short_on": short_on,
         "fit": round(float(fit),2), "cum": round(float(cum),3),
         "ann": round(float(cum)/n*2190.0,3), "sharpe": round(m["sharpe"],3),
         "dd": round(m["max_dd"],3), "turn": round(m["turnover"],4),
         "lt": round(m["long_turnover"],4), "st": round(m["short_turnover"],4)}
    print(r, flush=True)
    return r

# BTC asymmetry: both / long-only (short_th=-1 never triggers... use short_on=False) / short-only (lth=2.0 never)
run("BTC", 0.88, 0.12, True)
run("BTC", 0.88, 0.12, False)
run("BTC", 2.00, 0.12, True)
# cross-coin with best known params
for c in ["TRX", "BNB", "ETH"]:
    run(c, 0.88, 0.12, True)
