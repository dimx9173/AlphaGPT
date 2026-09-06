"""ATOM sweep (unbiased, trained formula)."""
import csv
import torch
from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest
FORMULA=[3,2,7,2,7,11,15,4,4,6,6,10]
with open("data_15m_3y/ATOM.csv") as f:
    rows=list(csv.DictReader(f))
bars=[]
for i in range(0,len(rows),16):
    blk=rows[i:i+16]
    if len(blk)<16: break
    bars.append({"close":float(blk[-1]["close"]),"open":float(blk[0]["open"]),"high":max(float(x["high"]) for x in blk),"low":min(float(x["low"]) for x in blk),"volume":sum(float(x["volume"]) for x in blk)})
n=len(bars)
raw={"open":torch.tensor([[b["open"] for b in bars]]),"high":torch.tensor([[b["high"] for b in bars]]),"low":torch.tensor([[b["low"] for b in bars]]),"close":torch.tensor([[b["close"] for b in bars]]),"volume":torch.tensor([[b["volume"] for b in bars]]),"liquidity":torch.full((1,n),1e7),"fdv":torch.full((1,n),1e8)}
sig=StackVM().execute(FORMULA,FeatureEngineer.compute_features(raw))
rets=[(bars[i+1]["close"]-bars[i]["close"])/bars[i]["close"] for i in range(n-1)]+[0.0]
target=torch.tensor([rets])
for lth,sth in [(0.85,0.15),(0.88,0.12),(0.90,0.10),(0.92,0.08)]:
    for cd in [0,3,6,12]:
        for sl in [None,0.03,0.05]:
            bt=MemeBacktest(venue="aster",leverage=2.0,short_enabled=True,funding_override=0.0005,long_th=lth,short_th=sth,cooldown_bars=cd,bars_per_year=2190.0,stop_loss=sl)
            fit,cum=bt.evaluate(sig,raw,target)
            m=bt.last_metrics
            print({"lth":lth,"sth":sth,"cd":cd,"sl":sl,"sharpe":round(m["sharpe"],3),"ann":round(float(cum)/n*2190.0,3),"dd":round(m["max_dd"],3),"turn":round(m["turnover"],4)},flush=True)
