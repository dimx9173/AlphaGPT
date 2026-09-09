"""Z2 gate x vol: trailing-200 y2_vol sharpe>1 gate vs plain and y2_vol ledgers."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if os.path.dirname(os.path.abspath(__file__)) else "."))
import pathlib as _pl
_root = _pl.Path(__file__).resolve().parent.parent
os.chdir(str(_root))
import json, math, csv, torch
from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest

FORMULA = [3,2,7,2,7,11,15,4,4,6,6,10]
COINS = ['ETC','TRX']
Q=0.3
FEE=0.0004; FUND=0.0005; LEV=2.0
# plain = Y1b: X5+S2+ts24 no vol
PLAIN={'ETC':(0.88,0.12,18,None,24,'both',None,24),'TRX':(0.85,0.12,6,0.05,24,'both',None,24)}
# y2_vol = Y2 optimal: +vt0.01 vw12
Y2={'ETC':(0.88,0.12,18,None,24,'both',0.01,12),'TRX':(0.85,0.12,6,0.05,24,'both',0.01,12)}
PORT={'ETC':0.5,'TRX':0.5}
SEGS={'FULL':(0,6580),'H2':(6077,6570),'B':(6380,6580),'C':(6080,6580)}
GATE_WIN=200; GATE_THRESH=1.0

def load_bars(coin):
    rows=list(csv.DictReader(open(f'data/data_15m_3y/{coin}.csv')))
    bars=[]
    for i in range(0,len(rows),16):
        blk=rows[i:i+16]
        if len(blk)<16: break
        bars.append((float(blk[0]['open']), max(float(x['high']) for x in blk), min(float(x['low']) for x in blk), float(blk[-1]['close']), sum(float(x['volume']) for x in blk)))
    return bars

def leg_net_pos(bars,lth,sth,cd,sl,ts,side,q,vt,vw):
    n=len(bars)
    raw={'open':torch.tensor([[b[0] for b in bars]]),'high':torch.tensor([[b[1] for b in bars]]),'low':torch.tensor([[b[2] for b in bars]]),'close':torch.tensor([[b[3] for b in bars]]),'volume':torch.tensor([[b[4] for b in bars]]),'liquidity':torch.full((1,n),1e7),'fdv':torch.full((1,n),1e8)}
    sig=StackVM().execute(FORMULA, FeatureEngineer.compute_features(raw))
    rets=[(bars[i+1][3]-bars[i][3])/bars[i][3] if i<n-1 else 0.0 for i in range(n)]
    bt=MemeBacktest(venue='aster',leverage=LEV,short_enabled=True,long_th=lth,short_th=sth,cooldown_bars=cd,bars_per_year=2190.0,stop_loss=sl,time_stop=ts,vol_target=vt,vol_window=vw,funding_override=FUND)
    sg=torch.sigmoid(sig)
    safe=(raw['liquidity']>bt.min_liq).float()
    lp=(sg>bt.long_th).float()*safe; sp=(sg<bt.short_th).float()*safe
    if q is not None:
        a=sig.detach().float().abs().reshape(-1); k=max(1,int(len(a)*float(q))); thr=torch.topk(a,k).values.min()
        mask=(sig.detach().float().abs()>=thr).float()
        lp=lp*mask
    lp,sp=bt._apply_cooldown(lp,sp)
    lp,sp=bt._apply_stops(lp,sp,torch.tensor([rets]))
    sc=bt._vol_scale(torch.tensor([rets]))
    # _vol_scale returns tensor or 1.0
    if not isinstance(sc,float):
        lp=lp*sc; sp=sp*sc
    if side=='long': sp=sp*0.0
    if side=='short': lp=lp*0.0
    lp=lp.roll(1,dims=1); lp[:,0]=0; sp=sp.roll(1,dims=1); sp[:,0]=0
    tgt=torch.tensor([rets])
    turn=(lp-lp.roll(1,dims=1)).abs()+(sp-sp.roll(1,dims=1)).abs()
    tx=turn*(bt.base_fee+torch.clamp(bt.trade_size/(raw['liquidity']+1e-9),0.0,0.05))
    gross=(lp-sp)*tgt*bt.leverage; fnd=(lp-sp)*bt.default_funding_rate*bt.leverage
    net=(gross-tx*bt.leverage-fnd)[0].tolist()
    pos=(lp-sp)[0].tolist()
    return net,pos

def run_ledger(poss,px,n):
    coins=sorted(PORT); eq=[1.0]; ledger=[]
    cur={c:0.0 for c in coins}; ent={c:0.0 for c in coins}
    for t in range(n):
        for c in coins:
            want=poss[c][t]
            want=1.0 if want>0.5 else (-1.0 if want<-0.5 else 0.0)
            if t>0 and want!=cur[c]:
                fill=px[c][t-1]
                if cur[c]!=0.0:
                    move=(fill-ent[c])/ent[c]*(1.0 if cur[c]>0 else -1.0)
                    netp=move*LEV - FEE*LEV*2.0 - FUND*LEV
                    eq.append(eq[-1]+eq[-1]*PORT[c]*netp)
                    ledger.append({'t':t,'coin':c,'side':int(cur[c]),'fill':fill,'move':round(move,6),'eq':round(eq[-1],4)})
                if want!=0.0: ent[c]=fill
                cur[c]=want
        if len(eq)<t+2: eq.append(eq[-1])
    return eq[:n], ledger

def eq_stats(eq,ledger):
    rets=[(eq[i+1]-eq[i])/eq[i] if eq[i] else 0.0 for i in range(len(eq)-1)]
    mean=sum(rets)/max(len(rets),1)
    var=sum((x-mean)**2 for x in rets)/max(len(rets)-1,1)
    sharpe=mean/math.sqrt(var)*math.sqrt(2190.0) if var>0 else 0.0
    peak=eq[0]; mdd=0.0
    for v in eq: peak=max(peak,v); mdd=max(mdd,(peak-v)/peak if peak else 0.0)
    by={}
    for e in ledger: by[e['coin']]=by.get(e['coin'],0)+1
    longs=sum(1 for e in ledger if e['side']==1)
    return {'final_x':round(eq[-1],4),'sharpe':round(sharpe,3),'mdd':round(mdd,4),'trades':len(ledger),'by':by,'longs':longs}

def seg_stats(eq,ledger,a,b):
    sub=eq[a:b] if b<=len(eq) else eq[a:]
    if not sub: sub=[1.0]
    base=sub[0]
    norm=[x/base for x in sub]
    rets=[(norm[i+1]-norm[i])/norm[i] if norm[i] else 0.0 for i in range(len(norm)-1)]
    mean=sum(rets)/max(len(rets),1)
    var=sum((x-mean)**2 for x in rets)/max(len(rets)-1,1)
    sharpe=mean/math.sqrt(var)*math.sqrt(2190.0) if var>0 else 0.0
    peak=norm[0]; mdd=0.0
    for v in norm: peak=max(peak,v); mdd=max(mdd,(peak-v)/peak if peak else 0.0)
    seg_led=[e for e in ledger if a<=e['t']<b]
    longs=sum(1 for e in seg_led if e['side']==1)
    return {'x':round(sub[-1]/sub[0],4) if sub[0] else 1.0,'sharpe':round(sharpe,3),'mdd':round(mdd,4),'trades':len(seg_led),'longs':longs}

def main():
    full={c:load_bars(c) for c in COINS}
    n=min(len(b) for b in full.values())
    print(f'full 4h bars n={n}',flush=True)
    px={c:[b[3] for b in full[c][:n]] for c in COINS}
    # compute per-coin net/pos for plain and y2
    plain_net={}; plain_pos={}; y2_net={}; y2_pos={}
    for c in COINS:
        lth,sth,cd,sl,ts,side,vt,vw = PLAIN[c]
        net,pos=leg_net_pos(full[c][:n],lth,sth,cd,sl,ts,side,Q,vt,vw)
        plain_net[c]=net; plain_pos[c]=pos
        lth2,sth2,cd2,sl2,ts2,side2,vt2,vw2 = Y2[c]
        net2,pos2=leg_net_pos(full[c][:n],lth2,sth2,cd2,sl2,ts2,side2,Q,vt2,vw2)
        y2_net[c]=net2; y2_pos[c]=pos2
    # combo per-bar net for gate calculation (0.5* etc +0.5*trx)
    y2_combo=[0.5*y2_net['ETC'][t]+0.5*y2_net['TRX'][t] for t in range(n)]
    plain_combo=[0.5*plain_net['ETC'][t]+0.5*plain_net['TRX'][t] for t in range(n)]
    eq_plain, led_plain = run_ledger(plain_pos, px, n)
    eq_y2, led_y2 = run_ledger(y2_pos, px, n)
    s_plain=eq_stats(eq_plain,led_plain); s_y2=eq_stats(eq_y2,led_y2)
    for k,(a,b) in SEGS.items():
        s_plain['seg_'+k]=seg_stats(eq_plain,led_plain,a,b)
        s_y2['seg_'+k]=seg_stats(eq_y2,led_y2,a,b)
    print('plain FULL',s_plain,flush=True)
    print('y2 FULL',s_y2,flush=True)
    drift={'d_trades':s_y2['trades']-s_plain['trades'],'d_final_x':round(s_y2['final_x']-s_plain['final_x'],4),'d_sharpe':round(s_y2['sharpe']-s_plain['sharpe'],3),'longs_filtered':s_plain['longs']-s_y2['longs']}
    print('drift y2-plain',drift,flush=True)
    # gate: trailing-200 y2 sharpe>1
    import math as _m
    def trailing_sharpe(arr,win):
        out=[0.0]*len(arr)
        for t in range(win,len(arr)):
            seg=arr[t-win:t]
            mu=sum(seg)/win
            var=sum((x-mu)**2 for x in seg)/max(win-1,1)
            sh=mu/_m.sqrt(var)*_m.sqrt(2190.0) if var>0 else 0.0
            out[t]=sh
        return out
    ts_y2=trailing_sharpe(y2_combo,GATE_WIN)
    ts_plain=trailing_sharpe(plain_combo,GATE_WIN)
    # shadows: LED-switch per-bar position source
    # shadow main: y2 active when ts_y2>1
    def build_shadow(active_fn):
        spos={c:[0.0]*n for c in COINS}
        for t in range(n):
            use_y2=active_fn(t)
            for c in COINS:
                spos[c][t]=y2_pos[c][t] if use_y2 else plain_pos[c][t]
        eq,led=run_ledger(spos,px,n)
        s=eq_stats(eq,led)
        for k,(a,b) in SEGS.items():
            s['seg_'+k]=seg_stats(eq,led,a,b)
        # coverage
        cov=sum(1 for t in range(n) if active_fn(t))/n
        return eq,led,s,cov
    eq_main,led_main,s_main,cov_main=build_shadow(lambda t: ts_y2[t]>GATE_THRESH if t>=GATE_WIN else False)
    eq_alt,led_alt,s_alt,cov_alt=build_shadow(lambda t: ts_plain[t]<GATE_THRESH if t>=GATE_WIN else False)
    print('shadow main FULL',s_main,'cov',cov_main,flush=True)
    print('shadow alt FULL',s_alt,'cov',cov_alt,flush=True)
    verdict={
        'main_beats_plain_B': s_main['seg_B']['sharpe']>s_plain['seg_B']['sharpe'],
        'main_beats_plain_C': s_main['seg_C']['sharpe']>s_plain['seg_C']['sharpe'],
        'main_full_no_collapse': s_main['sharpe']>2.0,
        'alt_beats_plain_B': s_alt['seg_B']['sharpe']>s_plain['seg_B']['sharpe'],
        'alt_beats_plain_C': s_alt['seg_C']['sharpe']>s_plain['seg_C']['sharpe'],
        'pass': bool(s_main['seg_B']['sharpe']>s_plain['seg_B']['sharpe'] and s_main['seg_C']['sharpe']>s_plain['seg_C']['sharpe'] and s_main['sharpe']>2.0),
        'decision': 'ADOPT shadow main_y2_gt1' if (s_main['seg_B']['sharpe']>s_plain['seg_B']['sharpe'] and s_main['seg_C']['sharpe']>s_plain['seg_C']['sharpe']) else 'REJECT both'
    }
    res={
        'config':{'formula':FORMULA,'fee':FEE,'fund':FUND,'lev':LEV,'q':Q,'plain':{k:[v[0],v[1],v[2],v[3]] for k,v in PLAIN.items()},'y2':{k:[v[0],v[1],v[2],v[3],v[6],v[7]] for k,v in Y2.items()},'weights':PORT,'full_n':n,'segments':SEGS,'gate':'trailing-200 y2 sharpe>1 main, trailing-200 plain<1 alt'},
        'ledgers':{'plain':s_plain,'y2_vol':s_y2},
        'drift_y2_minus_plain':drift,
        'gate':{'rule':'y2 active iff trailing-200 y2 sharpe>1 (causal window t-200:t)','window':GATE_WIN,'thresh':GATE_THRESH,'alt_rule':'plain trailing-200 sharpe<1','shadows':{'main_y2_gt1':{**s_main,'coverage':round(cov_main,4)},'alt_plain_lt1':{**s_alt,'coverage':round(cov_alt,4)}}},
        'verdict':verdict
    }
    open('results/backtest_Z2.json','w').write(json.dumps(res,indent=1))
    print('saved results/backtest_Z2.json verdict',verdict,flush=True)

if __name__=='__main__':
    main()
