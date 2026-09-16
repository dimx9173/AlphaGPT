#!/usr/bin/env python3
"""run_swap_frontier.py — P0-2 epsilon x min-hold frontier (offline, read-only).

Grid: eps {0,0.05,0.1,0.15,0.2} x min_hold {0,1,2} 4h-bars.
Cost: fee2x 0.0008 + 5bp slip + funding 0.0005 real per unit turnover.
Top5 equal 20%, E10 FORMULA. Hyst uses sg distance |sg-0.5|; min-hold uses
position age from stop/cooldown structure (approx: skip flip if previous
nonzero bar within mh bars).
Writes results/swap_frontier.json with knee (turnover drop >=20% + sharpe no-drop).
"""
import csv, json, math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest
from strategy_manager.config import FORMULA, FUND, FEE2X
COINS=["ETC","TRX","ATOM","APT","KAS"]
SPECS={"ETC": dict(lth=0.88, sth=0.12, cd=18, sl=None, ts=24),
 "TRX": dict(lth=0.85, sth=0.12, cd=6, sl=0.05, ts=24),
 "ATOM": dict(lth=0.85, sth=0.15, cd=6, sl=0.05, ts=24),
 "APT": dict(lth=0.88, sth=0.12, cd=18, sl=None, ts=24),
 "KAS": dict(lth=0.88, sth=0.12, cd=6, sl=None, ts=24)}
Q=0.3; BPY=2190.0; SLIP=0.0005
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
def leg_raw(coin,bars,spec):
    n=len(bars)
    raw={"open":torch.tensor([[b[0] for b in bars]]),"high":torch.tensor([[b[1] for b in bars]]),"low":torch.tensor([[b[2] for b in bars]]),"close":torch.tensor([[b[3] for b in bars]]),"volume":torch.tensor([[b[4] for b in bars]]),"liquidity":torch.full((1,n),1e7),"fdv":torch.full((1,n),1e8)}
    sig=StackVM(use_advanced=False).execute(FORMULA, FeatureEngineer.compute_features(raw, use_advanced=False))
    rets=[(bars[i+1][3]-bars[i][3])/bars[i][3] if i<n-1 else 0.0 for i in range(n)]
    rt=torch.tensor([rets])
    bt=MemeBacktest(venue="aster",leverage=2.0,short_enabled=True,long_th=spec["lth"],short_th=spec["sth"],cooldown_bars=spec["cd"],bars_per_year=BPY,stop_loss=spec["sl"],time_stop=spec["ts"],fee_override=FEE2X,funding_override=FUND)
    s=torch.sigmoid(sig); safe=(raw["liquidity"]>bt.min_liq).float()
    lp=(s>bt.long_th).float()*safe; sp=(s<bt.short_th).float()*safe
    m=qmask(sig)
    if m is not None: lp=lp*m
    lp,sp=bt._apply_cooldown(lp,sp); lp,sp=bt._apply_stops(lp,sp,rt)
    sc=bt._vol_scale(rt); lp=lp*sc; sp=sp*sc
    lp=lp.roll(1,dims=1); lp[:,0]=0; sp=sp.roll(1,dims=1); sp[:,0]=0
    sg=s[0].tolist()
    rets_l=rt[0].tolist()
    return (lp-sp)[0].tolist(), sg, rets_l
def apply_gates(pos, sg, rets, eps, mh):
    n=len(pos)
    out=list(pos)
    held_age=[0]*n
    cur=0.0; age=0
    for t in range(n):
        w=pos[t]
        if w!=0 and w!=cur:
            blocked=False
            if eps>0 and abs(sg[t]-0.5)<eps:
                blocked=True
            if mh>0 and cur!=0 and age<mh:
                blocked=True
            if blocked:
                w=cur
            else:
                cur=w; age=0
        if w==cur and w!=0:
            age+=1
        elif w==0:
            cur=0.0; age=0
        out[t]=w
    return out
def series(pos, rets):
    out=[]
    for t in range(len(pos)):
        p=pos[t]
        out.append(p*rets[t]*2.0)
    turn=[abs(pos[t]-(pos[t-1] if t else 0.0)) for t in range(len(pos))]
    cost=[x*(FEE2X+SLIP+FUND)*2.0 for x in turn]
    net=[a-c for a,c in zip(out,cost)]
    to=sum(turn)/len(turn)
    return net,to
def sharpe(ser):
    n=len(ser)
    if n<10: return 0.0
    mean=sum(ser)/n; var=sum((v-mean)**2 for v in ser)/max(n-1,1); std=math.sqrt(var) if var>0 else 0
    return mean/std*math.sqrt(BPY) if std>1e-9 else 0.0
def main():
    bars={c:load(c) for c in COINS}
    N=min(len(b) for b in bars.values())
    base={}
    for c in COINS:
        base[c]=leg_raw(c,bars[c][:N],SPECS[c])
    rows=[]
    for eps in (0,0.05,0.1,0.15,0.2):
        for mh in (0,1,2):
            legs={}
            for c in COINS:
                pos,sg,rets=base[c]
                legs[c]=(apply_gates(pos,sg,rets,eps,mh),rets)
            W=0.2
            port=[sum(W*legs[c][0][t]*legs[c][1][t]*2.0 - W*abs(legs[c][0][t]-(legs[c][0][t-1] if t else 0.0))*(FEE2X+SLIP)*2.0 - W*legs[c][0][t]*FUND*2.0 for c in COINS) for t in range(N)]
            to=sum(sum(W*abs(legs[c][0][t]-(legs[c][0][t-1] if t else 0.0)) for c in COINS) for t in range(N))/N
            rows.append({"eps":eps,"min_hold":mh,"sharpe":round(sharpe(port),3),"final_x":round(math.exp(sum(port)),3),"turnover":round(to,4)})
    base_row=[r for r in rows if r["eps"]==0 and r["min_hold"]==0][0]
    cand=[r for r in rows if r["turnover"]<=base_row["turnover"]*0.8 and r["sharpe"]>=base_row["sharpe"]]
    cand.sort(key=lambda r:(r["turnover"],-r["sharpe"]))
    knee=cand[0] if cand else None
    out={"base":base_row,"rows":rows,"knee":knee,
         "gates":{"turnover_cut_ge_20pct": bool(knee),"sharpe_no_drop": bool(knee)},
         "conclusion":"PENDING_P0-3-FAIL"}
    json.dump(out,open("results/swap_frontier.json","w"),indent=1)
    print(json.dumps(out,indent=1))
main()
