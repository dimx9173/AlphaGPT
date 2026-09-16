"""Combo: equal-weight portfolio of per-coin best-config net-pnl series (unbiased)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import csv, json
import torch
from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest
import math

FORMULA = [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10]
COINS = ["BTC", "TRX", "BNB", "ETH"]
KW = dict(venue="aster", leverage=2.0, short_enabled=True, funding_override=0.0005,
          long_th=0.88, short_th=0.12, cooldown_bars=6, bars_per_year=2190.0)

def coin_pnl(coin):
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
    sig = StackVM(use_advanced=False).execute(FORMULA, FeatureEngineer.compute_features(raw, use_advanced=False))
    rets = [(bars[i+1]["close"]-bars[i]["close"])/bars[i]["close"] for i in range(n-1)] + [0.0]
    # replicate evaluate internals to capture per-bar net pnl series
    bt = MemeBacktest(**KW)
    signal = torch.sigmoid(sig)
    is_safe = (raw["liquidity"] > bt.min_liq).float()
    lp = (signal > bt.long_th).float() * is_safe
    sp = ((signal < bt.short_th).float() * is_safe)
    lp, sp = bt._apply_cooldown(lp, sp)
    lp = lp.roll(1, dims=1); lp[:, 0] = 0
    sp = sp.roll(1, dims=1); sp[:, 0] = 0
    tgt = torch.tensor([rets])
    turn = (lp - lp.roll(1, dims=1)).abs() + (sp - sp.roll(1, dims=1)).abs()
    tx = turn * (bt.base_fee + torch.clamp(bt.trade_size / (raw["liquidity"] + 1e-9), 0.0, 0.05))
    gross = (lp - sp) * tgt * bt.leverage
    fund = (lp - sp) * bt.default_funding_rate * bt.leverage
    net = (gross - tx * bt.leverage - fund)[0].tolist()
    return net

series = {c: coin_pnl(c) for c in COINS}
n = min(len(v) for v in series.values())
combo = [sum(series[c][t] for c in COINS) / len(COINS) for t in range(n)]
mean = sum(combo) / n
var = sum((x - mean) ** 2 for x in combo) / max(n - 1, 1)
sharpe = mean / math.sqrt(var) * math.sqrt(2190.0) if var > 0 else 0.0
cum = sum(combo)
ann = cum / n * 2190.0
# max dd on cumsum
cs, peak, mdd = 0.0, -1e18, 0.0
for x in combo:
    cs += x
    peak = max(peak, cs)
    mdd = max(mdd, peak - cs)
print(f"combo {COINS}: cum={cum:.3f} ann={ann:.3f} sharpe={sharpe:.3f} mdd={mdd:.3f} n={n}")
# correlation matrix
import statistics
for i, a in enumerate(COINS):
    row = []
    for b in COINS:
        xa, xb = series[a][:n], series[b][:n]
        ma, mb = statistics.mean(xa), statistics.mean(xb)
        cov = sum((x-ma)*(y-mb) for x, y in zip(xa, xb)) / n
        sa = statistics.pstdev(xa); sb = statistics.pstdev(xb)
        row.append(round(cov/(sa*sb) if sa and sb else 0, 2))
    print(a, row)
with open("results/backtest_combo.json", "w") as f:
    json.dump({"coins": COINS, "cum": cum, "ann": ann, "sharpe": sharpe, "mdd": mdd, "n": n}, f)
