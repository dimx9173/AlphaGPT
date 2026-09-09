"""OOS: trained formula [3,2,7,2,7,11,15,4,4,6,6,10] on remaining 24 coins."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import csv, json, sys
import torch
from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest

FORMULA = [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10]
KW = dict(venue="aster", leverage=2.0, short_enabled=True, funding_override=0.0005,
          long_th=0.88, short_th=0.12, cooldown_bars=3, bars_per_year=2190.0, stop_loss=0.03)

def run_coin(coin):
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
    n = len(bars)
    raw = {"open": torch.tensor([[b["open"] for b in bars]]),
           "high": torch.tensor([[b["high"] for b in bars]]),
           "low": torch.tensor([[b["low"] for b in bars]]),
           "close": torch.tensor([[b["close"] for b in bars]]),
           "volume": torch.tensor([[b["volume"] for b in bars]]),
           "liquidity": torch.full((1, n), 1e7), "fdv": torch.full((1, n), 1e8)}
    sig = StackVM().execute(FORMULA, FeatureEngineer.compute_features(raw))
    rets = [(bars[i+1]["close"]-bars[i]["close"])/bars[i]["close"] for i in range(n-1)] + [0.0]
    bt = MemeBacktest(**KW)
    fit, cum = bt.evaluate(sig, raw, torch.tensor([rets]))
    m = bt.last_metrics
    return {"coin": coin, "fit": round(float(fit),2), "cum": round(float(cum),3),
            "ann": round(float(cum)/n*2190.0,3), "sharpe": round(m["sharpe"],3),
            "dd": round(m["max_dd"],3), "turn": round(m["turnover"],4)}

if __name__ == "__main__":
    import os
    coins = sys.argv[1:] or [f[:-4] for f in sorted(os.listdir("data/data_15m_3y")) if f.endswith(".csv")]
    out = [run_coin(c) for c in coins]
    tag = "all" if len(coins) > 1 else coins[0]
    with open(f"results/backtest_oos_{tag}.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    for r in sorted(out, key=lambda r: r["sharpe"], reverse=True):
        print(f"{r['coin']:6s} ann={r['ann']:7.3f} sharpe={r['sharpe']:6.3f} dd={r['dd']:.3f} turn={r['turn']:.4f}")
