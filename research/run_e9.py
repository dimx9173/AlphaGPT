"""E9 Shadow Gate & Ledger 实盘模拟 — 对照 W3 的 ledger 法在 AA 分支上。

Plain:  Y1b vtNone  = ETC(0.88,0.12,cd18,None,ts24,both)+TRX(0.85,0.12,cd6,0.05,ts24,both), q0.3, vtNone
Z1:     same legs + vt0.012 vw12 (cd18/6 sth0.12)
AA-H1:  AA H1-best idx01 sth0.10/etc15/trx9 + vt0.012 vw12
AA-H2:  AA H2-best idx13 sth0.12/etc15/trx9 + vt0.012 vw12
Weights 50/50, FORMULA [3,2,7,2,7,11,15,4,4,6,6,10], lev2 fund0.0005 fee0.0004
Ledger: run_ledger per-trade netp = move*LEV - FEE*LEV*2 - FUND*LEV 50/50 split via leg_net_pos
Gate: causal trailing-200 per-bar-net sharpe
  main= overlay-active iff trailing-200 overlay sharpe > thresh
  alt = plain-active iff trailing-200 plain sharpe < thresh  (overlay when plain < thresh)
  thresh in [0.8,1.0,1.5] x win200 x 3 overlays = 18
Segments: FULL[0:6580]/H2[6077:6570]/B[6380:6580]/C[6080:6580]
Decision: PASS iff B>plain_B AND C>plain_C AND FULL no collapse (sharpe>0.5 & mdd<1.0)
Mirror research/run_w3.py + run_z2.py engines.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if os.path.dirname(os.path.abspath(__file__)) else ".")
import pathlib as _pl
_root = _pl.Path(__file__).resolve().parent.parent if "research" in str(_pl.Path(__file__).resolve()) else _pl.Path(".")
os.chdir(str(_root))
import json, math, csv
import torch
from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest

FORMULA = [3,2,7,2,7,11,15,4,4,6,6,10]
COINS = ['ETC','TRX']
PORT = {'ETC':0.5,'TRX':0.5}
FEE = 0.0004
FUND = 0.0005
LEV = 2.0
Q = 0.3
SEGS = {'FULL':(0,6580),'H2':(6077,6570),'B':(6380,6580),'C':(6080,6580)}
GATE_WIN = 200
THRESHES = [0.8,1.0,1.5]

# (lth, sth, cd, sl, ts, side, vt, vw)  8-tuple per leg
PLAIN_SPEC = {
    'ETC': (0.88,0.12,18,None,24,'both',None,12),
    'TRX': (0.85,0.12,6,0.05,24,'both',None,12),
}
Z1_SPEC = {
    'ETC': (0.88,0.12,18,None,24,'both',0.012,12),
    'TRX': (0.85,0.12,6,0.05,24,'both',0.012,12),
}
AA_H1_SPEC = {
    'ETC': (0.88,0.10,15,None,24,'both',0.012,12),
    'TRX': (0.85,0.10,9,0.05,24,'both',0.012,12),
}
AA_H2_SPEC = {
    'ETC': (0.88,0.12,15,None,24,'both',0.012,12),
    'TRX': (0.85,0.12,9,0.05,24,'both',0.012,12),
}

OVERLAYS = {'Z1': Z1_SPEC, 'AA-H1': AA_H1_SPEC, 'AA-H2': AA_H2_SPEC}

def log(msg):
    print(msg, flush=True)
    try:
        with open("logs/e9.log","a") as f:
            f.write(msg+"\n")
    except:
        pass

def load_bars(coin):
    rows = list(csv.DictReader(open(f'data/data_15m_3y/{coin}.csv')))
    bars=[]
    for i in range(0,len(rows),16):
        blk=rows[i:i+16]
        if len(blk)<16: break
        bars.append((float(blk[0]['open']), max(float(x['high']) for x in blk), min(float(x['low']) for x in blk), float(blk[-1]['close']), sum(float(x['volume']) for x in blk)))
    return bars

def leg_net_pos(bars, lth, sth, cd, sl, ts, side, q, vt, vw):
    n=len(bars)
    raw={'open':torch.tensor([[b[0] for b in bars]]),'high':torch.tensor([[b[1] for b in bars]]),'low':torch.tensor([[b[2] for b in bars]]),'close':torch.tensor([[b[3] for b in bars]]),'volume':torch.tensor([[b[4] for b in bars]]),'liquidity':torch.full((1,n),1e7),'fdv':torch.full((1,n),1e8)}
    sig=StackVM(use_advanced=False).execute(FORMULA, FeatureEngineer.compute_features(raw, use_advanced=False))
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

def run_ledger(poss, px, n):
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

def eq_stats(eq, ledger):
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

def seg_stats(eq, ledger, a, b):
    n=len(eq)
    a=max(0,min(a,n)); b=max(a+1,min(b,n))
    sub=eq[a:b]
    base=sub[0] if sub else 1.0
    norm=[x/base for x in sub] if base else [1.0]*len(sub)
    rets=[(norm[i+1]-norm[i])/norm[i] if norm[i] else 0.0 for i in range(len(norm)-1)]
    mean=sum(rets)/max(len(rets),1)
    var=sum((x-mean)**2 for x in rets)/max(len(rets)-1,1)
    sharpe=mean/math.sqrt(var)*math.sqrt(2190.0) if var>0 else 0.0
    peak=norm[0] if norm else 1.0; mdd=0.0
    for v in norm: peak=max(peak,v); mdd=max(mdd,(peak-v)/peak if peak else 0.0)
    seg_led=[e for e in ledger if a<=e['t']<b]
    longs=sum(1 for e in seg_led if e['side']==1)
    by={}
    for e in seg_led: by[e['coin']]=by.get(e['coin'],0)+1
    return {'x':round(sub[-1]/sub[0],4) if sub and sub[0] else 1.0,'sharpe':round(sharpe,3),'mdd':round(mdd,4),'trades':len(seg_led),'longs':longs,'by':by}

def trailing_sharpe(ser, t, win):
    w=ser[t-win:t]
    mean=sum(w)/win
    var=sum((x-mean)**2 for x in w)/(win-1)
    return mean/math.sqrt(var)*math.sqrt(2190.0) if var>0 else 0.0

def main():
    open("logs/e9.log","w").write("")
    log("E9 start: AA ledger + Z1/AA-H1/AA-H2 + 18 gates (mirror W3)")
    full={c:load_bars(c) for c in COINS}
    n=min(len(b) for b in full.values())
    log(f"full 4h bars n={n}")
    assert n==6580, f"expected 6580 got {n}"
    px={c:[b[3] for b in full[c][:n]] for c in COINS}

    specs={'plain': PLAIN_SPEC, **OVERLAYS}
    nets={}; poss={}
    for name,spec in specs.items():
        nets[name]={}; poss[name]={}
        for c in COINS:
            lth,sth,cd,sl,ts,side,vt,vw = spec[c]
            net,pos=leg_net_pos(full[c][:n], lth, sth, cd, sl, ts, side, Q, vt, vw)
            nets[name][c]=net; poss[name][c]=pos
        log(f"leg done {name} q={Q} spec={spec}")

    ledgers={}
    for name in specs:
        eq,led=run_ledger(poss[name], px, n)
        st=eq_stats(eq,led)
        st['segments']={k: seg_stats(eq,led,a,b) for k,(a,b) in SEGS.items()}
        st['eq']=eq
        st['ledger']=led
        ledgers[name]=st
        log(f"{name} FULL final_x={st['final_x']} sharpe={st['sharpe']} mdd={st['mdd']} trades={st['trades']} by={st['by']} longs={st['longs']}")
        for k in ['FULL','H2','B','C']:
            s=st['segments'][k]
            log(f"  {k} x={s['x']} sh={s['sharpe']} mdd={s['mdd']} tr={s['trades']}")

    plain_st=ledgers['plain']
    for ov in OVERLAYS:
        d={'d_trades': ledgers[ov]['trades']-plain_st['trades'], 'd_final_x': round(ledgers[ov]['final_x']-plain_st['final_x'],4), 'd_sharpe': round(ledgers[ov]['sharpe']-plain_st['sharpe'],3), 'longs_filtered': plain_st['longs']-ledgers[ov]['longs']}
        log(f"drift {ov}-plain {d}")

    combo_plain=[0.5*nets['plain']['ETC'][t]+0.5*nets['plain']['TRX'][t] for t in range(n)]
    combo_over={ov:[0.5*nets[ov]['ETC'][t]+0.5*nets[ov]['TRX'][t] for t in range(n)] for ov in OVERLAYS}

    plain_B = plain_st['segments']['B']
    plain_C = plain_st['segments']['C']

    shadows=[]
    for ov in ['Z1','AA-H1','AA-H2']:
        for thresh in THRESHES:
            for gtype in ['main','alt']:
                gate=[False]*n
                for t in range(GATE_WIN, n):
                    if gtype=='main':
                        gate[t]= trailing_sharpe(combo_over[ov], t, GATE_WIN) > thresh
                    else:
                        gate[t]= trailing_sharpe(combo_plain, t, GATE_WIN) < thresh
                spos={c:[poss[ov][c][t] if gate[t] else poss['plain'][c][t] for t in range(n)] for c in COINS}
                eq,led=run_ledger(spos, px, n)
                st=eq_stats(eq,led)
                st['segments']={k: seg_stats(eq,led,a,b) for k,(a,b) in SEGS.items()}
                st['coverage']=round(sum(gate)/n,4)
                st['overlay']=ov; st['thresh']=thresh; st['gate']=gtype; st['window']=GATE_WIN
                st['id']=f"{ov}_{gtype}_thr{thresh}_w{GATE_WIN}"
                b_sh = st['segments']['B']['sharpe']; c_sh = st['segments']['C']['sharpe']
                full_ok = st['sharpe']>0.5 and st['mdd']<1.0
                pass_ = (b_sh > plain_B['sharpe']) and (c_sh > plain_C['sharpe']) and full_ok
                st['pass']=bool(pass_)
                st['pass_x']= bool(st['segments']['B']['x']>plain_B['x'] and st['segments']['C']['x']>plain_C['x'] and full_ok)
                shadows.append(st)
                log(f"shadow {st['id']} cov={st['coverage']} FULL x={st['final_x']} sh={st['sharpe']} mdd={st['mdd']} tr={st['trades']} | B sh={b_sh} x={st['segments']['B']['x']} vs plain {plain_B['sharpe']}/{plain_B['x']} | C sh={c_sh} x={st['segments']['C']['x']} vs {plain_C['sharpe']}/{plain_C['x']} | pass={pass_}")

    pass_sh = [s for s in shadows if s['pass']]
    pass_x = [s for s in shadows if s['pass_x']]
    log(f"SUMMARY pass_sharpe={len(pass_sh)}/18 pass_x={len(pass_x)}/18")
    for s in shadows:
        log(f"  {s['id']} cov={s['coverage']} FULL sh={s['sharpe']} mdd={s['mdd']} B_sh={s['segments']['B']['sharpe']} vs plain {plain_B['sharpe']} C_sh={s['segments']['C']['sharpe']} vs {plain_C['sharpe']} pass={s['pass']}")

    best=None
    if pass_sh:
        pass_sh_sorted=sorted(pass_sh, key=lambda s: (s['mdd'], -s['coverage']))
        best=pass_sh_sorted[0]
        log(f"BEST pass_sh {best['id']} mdd={best['mdd']} cov={best['coverage']} B_sh={best['segments']['B']['sharpe']} C_sh={best['segments']['C']['sharpe']}")
    elif pass_x:
        pass_x_sorted=sorted(pass_x, key=lambda s: (s['mdd'], -s['coverage']))
        best=pass_x_sorted[0]
        log(f"BEST pass_x (fallback) {best['id']}")

    clean_ledgers={}
    for name in specs:
        st=ledgers[name]
        clean_ledgers[name]={'final_x':st['final_x'],'sharpe':st['sharpe'],'mdd':st['mdd'],'trades':st['trades'],'by':st['by'],'longs':st['longs'],'segments':{k: {kk: st['segments'][k][kk] for kk in ['x','sharpe','mdd','trades','by','longs']} for k in SEGS}}
        clean_ledgers[name]['segments']['FULL']['final_x']=clean_ledgers[name]['final_x']

    clean_shadows=[]
    for s in shadows:
        clean_shadows.append({'id':s['id'],'overlay':s['overlay'],'gate':s['gate'],'window':s['window'],'thresh':s['thresh'],'coverage':s['coverage'],'final_x':s['final_x'],'sharpe':s['sharpe'],'mdd':s['mdd'],'trades':s['trades'],'by':s['by'],'longs':s['longs'],'segments':{k:{kk:s['segments'][k][kk] for kk in ['x','sharpe','mdd','trades','by','longs']} for k in SEGS},'pass':s['pass'],'pass_x':s['pass_x'],'vs_plain':{'B_plain_sh':plain_B['sharpe'],'B_sh':s['segments']['B']['sharpe'],'B_plain_x':plain_B['x'],'B_x':s['segments']['B']['x'],'C_plain_sh':plain_C['sharpe'],'C_sh':s['segments']['C']['sharpe'],'C_plain_x':plain_C['x'],'C_x':s['segments']['C']['x']}})

    # heat table for docs
    heat={}
    for ov in OVERLAYS:
        heat[ov]={}
        for gtype in ['main','alt']:
            heat[ov][gtype]={str(th): next(s for s in clean_shadows if s['overlay']==ov and s['gate']==gtype and s['thresh']==th) for th in THRESHES}

    res={
        'config':{'formula': FORMULA,'fee':FEE,'fund':FUND,'lev':LEV,'q':Q,'weights':PORT,'full_n':n,'segments':{k:list(v) for k,v in SEGS.items()},'plain_spec':{c:list(PLAIN_SPEC[c]) for c in COINS},'overlays':{k:{c:list(v[c]) for c in COINS} for k,v in OVERLAYS.items()},'gate_window':GATE_WIN,'gate_threshes':THRESHES,'gate_rules':{'main':'overlay active iff trailing-200 overlay per-bar-net sharpe > thresh','alt':'plain active iff trailing-200 plain sharpe < thresh (overlay when plain sharpe < thresh)'},'note':'E9 ledger: per-trade netp=move*LEV-FEE*LEV*2-FUND*LEV 50/50 split, pos via leg_net_pos with quantile0.3+cooldown+stops+vol_scale+roll1. Overlays Z1/AA-H1/AA-H2 vs plain Y1b vtNone.'},
        'ledgers': clean_ledgers,
        'shadows': clean_shadows,
        'heat': {ov:{gtype:{str(th): {'coverage': heat[ov][gtype][str(th)]['coverage'],'sharpe': heat[ov][gtype][str(th)]['sharpe'],'mdd': heat[ov][gtype][str(th)]['mdd'],'final_x': heat[ov][gtype][str(th)]['final_x'],'B_sh': heat[ov][gtype][str(th)]['segments']['B']['sharpe'],'C_sh': heat[ov][gtype][str(th)]['segments']['C']['sharpe'],'pass': heat[ov][gtype][str(th)]['pass']} for th in THRESHES} for gtype in ['main','alt']} for ov in OVERLAYS},
        'summary':{'total':18,'pass_sharpe':len(pass_sh),'pass_x':len(pass_x),'pass_ids_sharpe':[s['id'] for s in pass_sh],'pass_ids_x':[s['id'] for s in pass_x],'plain_B':plain_B,'plain_C':plain_C,'plain_FULL':{'final_x':plain_st['final_x'],'sharpe':plain_st['sharpe'],'mdd':plain_st['mdd']}},
        'best': ({'id':best['id'],'overlay':best['overlay'],'gate':best['gate'],'thresh':best['thresh'],'window':best['window'],'coverage':best['coverage'],'sharpe':best['sharpe'],'mdd':best['mdd'],'final_x':best['final_x'],'B_sh':best['segments']['B']['sharpe'],'C_sh':best['segments']['C']['sharpe'],'B_x':best['segments']['B']['x'],'C_x':best['segments']['C']['x']} if best else None),
        'decision':('PASS adopt shadow '+best['id'] if best and best['pass'] else ('REJECT no shadow passes B>C and full no collapse (keep plain)' if not pass_sh else 'REJECT')),
    }
    res['verdict']=res['decision']
    open('results/backtest_E9.json','w').write(json.dumps(res, indent=2, ensure_ascii=False))
    log(f"saved results/backtest_E9.json best={best['id'] if best else None} decision={res['decision']}")
    log(f"FULL ledgers: plain={clean_ledgers['plain']} Z1={clean_ledgers['Z1']} AA-H1={clean_ledgers['AA-H1']} AA-H2={clean_ledgers['AA-H2']}")

if __name__=='__main__':
    main()
