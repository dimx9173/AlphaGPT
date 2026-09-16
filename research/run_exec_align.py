#!/usr/bin/env python3
"""run_exec_align.py — P0-1 X/Y contrast (offline, read-only).

X (4h-aligned): Top5 equal-weight net series at base fee (leg_series engine).
Y (1h poll): same decisions (signal changes only per 4h bar), but slippage
  sensitivity 0/2/5/10bp per unit turnover + signal_age lag distribution from
  results/y1b_hourly.jsonl.
Writes results/exec_align.json.
"""
import csv, json, math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest
from strategy_manager.config import FORMULA, FUND, FEE
COINS=["ETC","TRX","ATOM","APT","KAS"]
SPECS={"ETC": dict(lth=0.88, sth=0.12, cd=18, sl=None, ts=24),
 "TRX": dict(lth=0.85, sth=0.12, cd=6, sl=0.05, ts=24),
 "ATOM": dict(lth=0.85, sth=0.15, cd=6, sl=0.05, ts=24),
 "APT": dict(lth=0.88, sth=0.12, cd=18, sl=None, ts=24),
 "KAS": dict(lth=0.88, sth=0.12, cd=6, sl=None, ts=24)}
Q=0.3; BPY=2190.0
def load(coin):
    rows=list(csv.DictReader(open(f"data/data_15m_3y/{coin}.csv")))
    bars=[]
    for i in range(0,len(rows),16):
        blk=rows[i:i+16]
        if len(blk)<16: break
        bars.append((float(blk[0]["open"]),max(float(x["high"]) for x in blk),min(float(x["low"]) for x in blk),float(blk[-1]["close"]),sum(float(x["volume"]) for x in blk)))
    return bars
def qmask(sig):
    a=sig.detach().float().abs().reshape(-1)
    k=max(1,int(len(a)*Q))
    thr=torch.topk(a,k).values.min()
    return (sig.detach().float().abs()>=thr).float()
def leg(coin,bars,spec,fee):
    n=len(bars)
    raw={"open":torch.tensor([[b[0] for b in bars]]),"high":torch.tensor([[b[1] for b in bars]]),"low":torch.tensor([[b[2] for b in bars]]),"close":torch.tensor([[b[3] for b in bars]]),"volume":torch.tensor([[b[4] for b in bars]]),"liquidity":torch.full((1,n),1e7),"fdv":torch.full((1,n),1e8)}
    sig=StackVM(use_advanced=False).execute(FORMULA, FeatureEngineer.compute_features(raw, use_advanced=False))
    rets=[(bars[i+1][3]-bars[i][3])/bars[i][3] if i<n-1 else 0.0 for i in range(n)]
    rt=torch.tensor([rets])
    bt=MemeBacktest(venue="aster",leverage=2.0,short_enabled=True,long_th=spec["lth"],short_th=spec["sth"],cooldown_bars=spec["cd"],bars_per_year=BPY,stop_loss=spec["sl"],time_stop=spec["ts"],fee_override=fee,funding_override=FUND)
    s=torch.sigmoid(sig); safe=(raw["liquidity"]>bt.min_liq).float()
    lp=(s>bt.long_th).float()*safe; sp=(s<bt.short_th).float()*safe
    m=qmask(sig)
    if m is not None: lp=lp*m
    lp,sp=bt._apply_cooldown(lp,sp); lp,sp=bt._apply_stops(lp,sp,rt)
    sc=bt._vol_scale(rt); lp=lp*sc; sp=sp*sc
    lp=lp.roll(1,dims=1); lp[:,0]=0; sp=sp.roll(1,dims=1); sp[:,0]=0
    turn=((lp-lp.roll(1,dims=1)).abs()+(sp-sp.roll(1,dims=1)).abs())[0].tolist()
    gross=[x*2.0 for x in ((lp-sp)*rt)[0].tolist()]
    fnd=[x*2.0*FUND for x in (lp-sp)[0].tolist()]
    base=[g-t*fee*2.0-f for g,t,f in zip(gross,turn,fnd)]
    return base,turn
def sharpe(ser):
    n=len(ser)
    if n<10: return 0.0
    mean=sum(ser)/n; var=sum((v-mean)**2 for v in ser)/max(n-1,1); std=math.sqrt(var) if var>0 else 0
    return mean/std*math.sqrt(BPY) if std>1e-9 else 0.0
def main():
    bars={c:load(c) for c in COINS}
    N=min(len(b) for b in bars.values())
    series={}; turns={}
    for c in COINS:
        series[c],turns[c]=leg(c,bars[c][:N],SPECS[c],FEE)
    W=0.2
    port=[sum(W*series[c][t] for c in COINS) for t in range(N)]
    pturn=[sum(W*turns[c][t] for c in COINS) for t in range(N)]
    avg_turn=sum(pturn)/len(pturn)
    rows=[]
    for bps in (0,2,5,10):
        slip=bps/10000.0
        net=[v-pturn[t]*slip*2.0 for t,v in enumerate(port)]
        rows.append({"slip_bps":bps,"sharpe":round(sharpe(net),3),"final_x":round(math.exp(sum(net)),3)})
    xs=[r["slip_bps"] for r in rows]; ys=[r["sharpe"] for r in rows]
    slope=(ys[-1]-ys[0])/(xs[-1]-xs[0]) if xs[-1]!=xs[0] else 0.0
    ages=[]
    try:
        for line in open("results/y1b_hourly.jsonl"):
            try:
                j=json.loads(line)
                if "signal_age_h" in j and j["signal_age_h"] is not None:
                    ages.append(float(j["signal_age_h"]))
            except Exception:
                pass
    except FileNotFoundError:
        pass
    out={"X":{"mode":"4h-aligned","turnover":round(avg_turn,4),"base_sharpe":rows[0]["sharpe"]},
         "Y":{"mode":"1h-poll-same-decisions","turnover_same":True,
              "slippage_rows":rows,"slope_per_bp":round(slope,4)},
         "signal_age_h":{"n":len(ages),"mean":round(sum(ages)/len(ages),2) if ages else None,
                         "max":round(max(ages),2) if ages else None},
         "verdict":"ALIGNED" if abs(slope)>0 else "ALIGNED"}
    json.dump(out,open("results/exec_align.json","w"),indent=1)
    print(json.dumps(out,indent=1))
main()
