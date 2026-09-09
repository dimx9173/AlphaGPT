"""Iter4: TRX fine sweep (threshold x cooldown x leverage x formula)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import csv, json
import torch
from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest

FORMULAS = {"F_MOM": [0, 2, 6, 15], "F_REV": [4, 3, 7],
            "F_REV_D": [4, 3, 7, 15], "F_PX": [2, 4, 7]}
GRID = []
for lth, sth in [(0.88, 0.12), (0.90, 0.10), (0.92, 0.08), (0.94, 0.06)]:
    for cd in [3, 6, 12]:
        for lev in [1.0, 2.0, 3.0]:
            GRID.append((lth, sth, cd, lev))

def load_4h(coin):
    import csv as _c
    with open(f"data/data_15m_3y/{coin}.csv") as f:
        rows = list(_c.DictReader(f))
    bars = []
    for i in range(0, len(rows), 16):
        blk = rows[i:i+16]
        if len(blk) < 16:
            break
        bars.append({"open": float(blk[0]["open"]),
                     "high": max(float(x["high"]) for x in blk),
                     "low": min(float(x["low"]) for x in blk),
                     "close": float(blk[-1]["close"]),
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
vm = StackVM()
rets = [(bars[i+1]["close"] - bars[i]["close"]) / bars[i]["close"] for i in range(n - 1)] + [0.0]
target = torch.tensor([rets])
out = []
for fname, ftoks in FORMULAS.items():
    sig = vm.execute(ftoks, feats)
    if sig is None:
        print(f"TRX {fname}: VM invalid"); continue
    for (lth, sth, cd, lev) in GRID:
        bt = MemeBacktest(venue="aster", leverage=lev, short_enabled=True,
                          funding_override=0.0005, long_th=lth, short_th=sth,
                          cooldown_bars=cd, bars_per_year=2190.0)
        fit, cum = bt.evaluate(sig, raw, target)
        m = bt.last_metrics
        # annualized return: cum is sum of per-bar pnl on 1x notional-ish; annualize
        ann = float(cum) / n * 2190.0
        out.append({"formula": fname, "lth": lth, "sth": sth, "cd": cd, "lev": lev,
                    "fit": round(float(fit), 2), "cum": round(float(cum), 3),
                    "ann": round(ann, 3), "sharpe": round(m["sharpe"], 3),
                    "dd": round(m["max_dd"], 3), "turn": round(m["turnover"], 4)})
with open("results/backtest_iter4_TRX.jsonl", "w") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")
top = sorted(out, key=lambda r: r["sharpe"], reverse=True)[:10]
print("TOP10 by sharpe:")
for r in top:
    print(r)
pos = [r for r in out if r["sharpe"] >= 2.0]
print(f"\nsharpe>=2.0 count: {len(pos)}")
ann10 = [r for r in out if r["ann"] >= 0.10]
print(f"ann>=10% count: {len(ann10)}")
