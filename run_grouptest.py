"""Combo: equal-weight portfolio of per-coin best-config net-pnl series (unbiased)."""
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
    with open(f"data_15m_3y/{coin}.csv") as f:
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

import statistics
SETS = {
    'single_BTC': ['BTC'],
    'single_ETC': ['ETC'],
    'single_TRX': ['TRX'],
    'prev_naive_BTC_TRX_BNB_ETH': ['BTC','TRX','BNB','ETH'],
    'within_main_BTC_ETH': ['BTC','ETH'],
    'within_meme_DOGE_SHIB': ['DOGE','SHIB'],
    'within_pay_XRP_XLM': ['XRP','XLM'],
    'within_new_SOL_AVAX': ['SOL','AVAX'],
    'within_main_BTC_ETC_LINK': ['BTC','ETC','LINK'],
    'cross_BTC_TRX_XRP_SOL': ['BTC','TRX','XRP','SOL'],
    'cross_ETC_TRX_XRP_SOL': ['ETC','TRX','XRP','SOL'],
    'cross_BTC_TRX_DOGE_XRP': ['BTC','TRX','DOGE','XRP'],
    'cross_ETC_TRX_DOGE_XRP': ['ETC','TRX','DOGE','XRP'],
    'cross_BTC_TRX_BCH': ['BTC','TRX','BCH'],
    'cross_ETC_TRX_BCH_SOL': ['ETC','TRX','BCH','SOL'],
}
allcoins = sorted({c for s in SETS.values() for c in s})
print('computing coins:', allcoins, flush=True)
series = {c: coin_pnl(c) for c in allcoins}
rows = []
for name, COINS in SETS.items():
    n = min(len(series[c]) for c in COINS)
    combo = [sum(series[c][t] for c in COINS)/len(COINS) for t in range(n)]
    mean = sum(combo)/n
    var = sum((x-mean)**2 for x in combo)/max(n-1,1)
    sharpe = mean/math.sqrt(var)*math.sqrt(2190.0) if var>0 else 0.0
    cum = sum(combo); ann = cum/n*2190.0
    cs, peak, mdd = 0.0, -1e18, 0.0
    for x in combo:
        cs += x; peak = max(peak, cs); mdd = max(mdd, peak-cs)
    corrs = []
    for i in range(len(COINS)):
        for j in range(i+1, len(COINS)):
            xa, xb = series[COINS[i]][:n], series[COINS[j]][:n]
            ma, mb = statistics.mean(xa), statistics.mean(xb)
            cov = sum((x-ma)*(y-mb) for x,y in zip(xa,xb))/n
            sa, sb = statistics.pstdev(xa), statistics.pstdev(xb)
            corrs.append(cov/(sa*sb) if sa and sb else 0)
    mcorr = sum(corrs)/len(corrs) if corrs else 1.0
    print(name, COINS, round(sharpe,3), round(ann,3), round(mdd,3), round(mcorr,2), flush=True)
    rows.append({'name':name,'coins':COINS,'sharpe':sharpe,'ann':ann,'mdd':mdd,'pnlcorr':mcorr,'n':n})
import json as _j
open('backtest_grouptest.json','w').write(_j.dumps(rows, indent=1))
print('saved backtest_grouptest.json', flush=True)
