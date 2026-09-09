"""Stress: baseline x fee multipliers + cross-coin (unbiased)."""
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

def run(coin, fee_mult, label):
    bars = load(coin)
    n = len(bars)
    raw = {"open": torch.tensor([[b["open"] for b in bars]]),
           "high": torch.tensor([[b["high"] for b in bars]]),
           "low": torch.tensor([[b["low"] for b in bars]]),
           "close": torch.tensor([[b["close"] for b in bars]]),
           "volume": torch.tensor([[b["volume"] for b in bars]]),
           "liquidity": torch.full((1, n), 1e7), "fdv": torch.full((1, n), 1e8)}
    sig = StackVM().execute(FORMULA, FeatureEngineer.compute_features(raw))
    rets = [(bars[i+1]["close"]-bars[i]["close"])/bars[i]["close"] for i in range(n-1)] + [0.0]
    bt = MemeBacktest(venue="aster", leverage=2.0, short_enabled=True,
                      funding_override=0.0005, long_th=0.88, short_th=0.12,
                      cooldown_bars=6, bars_per_year=2190.0,
                      fee_override=0.0004 * fee_mult)
    fit, cum = bt.evaluate(sig, raw, torch.tensor([rets]))
    m = bt.last_metrics
    print({"coin": coin, "test": label, "fee_mult": fee_mult,
           "ann": round(float(cum)/n*2190.0,3), "sharpe": round(m["sharpe"],3),
           "dd": round(m["max_dd"],3), "turn": round(m["turnover"],4)}, flush=True)

run("BTC", 1.0, "base")
run("BTC", 2.0, "costx2")
run("BTC", 3.0, "costx3")
for c in ["ETH", "TRX", "BNB", "SOL", "ATOM", "DOT"]:
    run(c, 1.0, "cross")
