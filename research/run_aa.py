import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json, math, csv, itertools, pathlib
import torch
from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest

FORMULA = [3,2,7,2,7,11,15,4,4,6,6,10]
COINS = ['ETC','TRX']
BASE_FEE = 0.0004
FEE2X = 0.0008
FUND = 0.0005
Q = 0.3
VT = 0.012
VW = 12
TS = 24

SEGS = {'H1': (5584,6077), 'H2': (6077,6570), 'B': (6380,6580), 'C': (6080,6580), 'FULL': (0,6580)}
SEGS_LIST = ['H1','H2','B','C','FULL']

STH_GRID = [0.10, 0.11, 0.12]
ETC_CD_GRID = [15,18,21]
TRX_CD_GRID = [6,9]

def load_bars(coin):
    rows = list(csv.DictReader(open(f'data/data_15m_3y/{coin}.csv')))
    bars=[]
    for i in range(0,len(rows),16):
        blk=rows[i:i+16]
        if len(blk)<16:
            break
        bars.append((float(blk[0]['open']), max(float(x['high']) for x in blk), min(float(x['low']) for x in blk), float(blk[-1]['close']), sum(float(x['volume']) for x in blk)))
    return bars

def build_mats(bars):
    n=len(bars)
    raw={'open':torch.tensor([[b[0] for b in bars]]),
         'high':torch.tensor([[b[1] for b in bars]]),
         'low':torch.tensor([[b[2] for b in bars]]),
         'close':torch.tensor([[b[3] for b in bars]]),
         'volume':torch.tensor([[b[4] for b in bars]]),
         'liquidity':torch.full((1,n),1e7),
         'fdv':torch.full((1,n),1e8)}
    sig=StackVM().execute(FORMULA, FeatureEngineer.compute_features(raw))
    rets=[(bars[i+1][3]-bars[i][3])/bars[i][3] for i in range(n-1)] + [0.0]
    return raw, torch.tensor([rets]), sig

def quantile_mask_long(sig,q):
    if q is None:
        return None
    a=sig.detach().float().abs().reshape(-1)
    k=max(1,int(len(a)*float(q)))
    thr=torch.topk(a,k).values.min()
    return (sig.detach().float().abs() >= thr).float()

def leg_series(raw, rets_t, sig, lth, sth, cd, sl, fee=None, fund=None, side='both', q=Q, vt=VT, vw=VW, ts=TS):
    kw=dict(venue='aster', leverage=2.0, short_enabled=True, long_th=lth, short_th=sth, cooldown_bars=cd, bars_per_year=2190.0, stop_loss=sl, time_stop=ts, vol_target=vt, vol_window=vw)
    kw['funding_override']=FUND if fund is None else fund
    if fee is not None:
        kw['fee_override']=fee
    bt=MemeBacktest(**kw)
    signal=torch.sigmoid(sig)
    is_safe=(raw['liquidity']>bt.min_liq).float()
    lp=(signal>bt.long_th).float()*is_safe
    sp=(signal<bt.short_th).float()*is_safe
    mask=quantile_mask_long(sig,q)
    if mask is not None:
        lp=lp*mask
    lp,sp=bt._apply_cooldown(lp,sp)
    lp,sp=bt._apply_stops(lp,sp, rets_t)
    scale=bt._vol_scale(rets_t)
    lp,sp=lp*scale, sp*scale
    if side=='long':
        sp=sp*0.0
    if side=='short':
        lp=lp*0.0
    lp=lp.roll(1,dims=1); lp[:,0]=0
    sp=sp.roll(1,dims=1); sp[:,0]=0
    turn=(lp - lp.roll(1,dims=1)).abs() + (sp - sp.roll(1,dims=1)).abs()
    tx=turn*(bt.base_fee+torch.clamp(bt.trade_size/(raw['liquidity']+1e-9),0.0,0.05))
    gross=(lp-sp)*rets_t*bt.leverage
    fnd=(lp-sp)*bt.default_funding_rate*bt.leverage
    net=(gross - tx*bt.leverage - fnd)[0].tolist()
    turnl=turn[0].tolist()
    pos=(lp-sp)[0].tolist()
    trades=sum(1 for t in range(len(pos)) if pos[t]!=0.0 and (t==0 or pos[t-1]==0.0))
    turnover=sum(turnl)/len(turnl) if turnl else 0.0
    return net, trades, turnover

def stats(ser, trades=0, turnover=0.0):
    n=len(ser)
    mean=sum(ser)/n if n else 0
    var=sum((x-mean)**2 for x in ser)/max(n-1,1) if n>1 else 0
    sharpe=mean/math.sqrt(var)*math.sqrt(2190.0) if var>0 else 0.0
    cum=sum(ser)
    ann=cum/n*2190.0 if n else 0
    cs,peak,mdd=0.0,-1e18,0.0
    for x in ser:
        cs+=x
        peak=max(peak,cs)
        mdd=max(mdd,peak-cs)
    return {'sharpe':round(sharpe,3),'ann':round(ann,4),'mdd':round(mdd,4),'cum':round(cum,4),'n':n,'trades':trades,'turnover':round(turnover,6)}

def combo(ser_list, weights=None):
    m=min(len(s) for s in ser_list)
    k=len(ser_list)
    w=weights or [1.0/k]*k
    sw=sum(w)
    return [sum(ser_list[i][t]*w[i]/sw for i in range(k)) for t in range(m)]

def pearson(a,b):
    n=len(a)
    if n==0 or len(b)==0:
        return 0.0
    ma=sum(a)/n
    mb=sum(b)/len(b)
    num=sum((a[i]-ma)*(b[i]-mb) for i in range(min(n,len(b))))
    va=sum((x-ma)**2 for x in a)
    vb=sum((x-mb)**2 for x in b)
    den=math.sqrt(va*vb) if va>0 and vb>0 else 0
    return round(num/den,4) if den>0 else 0.0

def eval_all(mats_seg, fee, fund, spec_pairs):
    legs,tr,tos=[],[],[]
    for coin,(lth,sth,cd,sl,ts) in spec_pairs:
        raw,rt,sg=mats_seg[coin]
        net,t,to=leg_series(raw,rt,sg,lth,sth,cd,sl,fee=fee,fund=fund)
        legs.append(net); tr.append(t); tos.append(to)
    cb=combo(legs,[0.5,0.5])
    s=stats(cb,trades=sum(tr),turnover=sum(tos)/len(tos) if tos else 0)
    s['trades_by']={spec_pairs[i][0]:tr[i] for i in range(len(spec_pairs))}
    return s, cb, legs

def main():
    logp=pathlib.Path('logs/aa.log')
    logp.parent.mkdir(parents=True, exist_ok=True)
    open(logp,'w').write('AA start\n')
    def log(msg):
        print(msg, flush=True)
        with open(logp,'a') as f: f.write(msg+'\n')

    full={c:load_bars(c) for c in COINS}
    n=min(len(b) for b in full.values())
    log(f'full 4h bars n={n} ETC={len(full["ETC"])} TRX={len(full["TRX"])}')
    assert n==6580, f'expected 6580 got {n}'
    # build mats per segment
    mats={}
    for seg,(a,b) in SEGS.items():
        mats[seg]={c: build_mats(full[c][a:b]) for c in COINS}
        log(f'{seg} [{a}:{b}] n={b-a} built')

    grid=[]
    for sth in STH_GRID:
        for etc_cd in ETC_CD_GRID:
            for trx_cd in TRX_CD_GRID:
                grid.append((sth, etc_cd, trx_cd, False))
    # 2 control sth0.09
    grid.append((0.09, 18, 6, True))
    grid.append((0.09, 18, 9, True))
    assert len(grid)==20, f'grid {len(grid)} !=20'

    rows=[]
    for idx,(sth, etc_cd, trx_cd, is_ctrl) in enumerate(grid):
        spec_pairs=[('ETC',(0.88, sth, etc_cd, None, TS)), ('TRX',(0.85, sth, trx_cd, 0.05, TS))]
        segs={}
        feat={}
        for seg in SEGS_LIST:
            s,cb,legs = eval_all(mats[seg], fee=BASE_FEE, fund=FUND, spec_pairs=spec_pairs)
            segs[seg]=s
            feat[seg]=(cb,legs)
        # fee2x for H2/B/C
        for seg in ['H2','B','C']:
            s2,_,_ = eval_all(mats[seg], fee=FEE2X, fund=FUND, spec_pairs=spec_pairs)
            segs[seg+'_fee2x']=s2
        # corr on H2
        _, legs_h2 = feat['H2']
        corr = pearson(legs_h2[0], legs_h2[1]) if len(legs_h2)==2 else 0.0
        row={
            'idx': idx,
            'sth': sth, 'etc_cd': etc_cd, 'trx_cd': trx_cd, 'control': bool(is_ctrl),
            'lth_etc':0.88,'lth_trx':0.85,'sl_etc':None,'sl_trx':0.05,'ts':TS,'vt':VT,'vw':VW,'q':Q,'weights':[0.5,0.5],
            'H1': segs['H1'], 'H2': segs['H2'], 'B': segs['B'], 'C': segs['C'], 'FULL': segs['FULL'],
            'H2_fee2x': segs['H2_fee2x'], 'B_fee2x': segs['B_fee2x'], 'C_fee2x': segs['C_fee2x'],
            'corr_ETC_TRX_H2': corr,
            'fee_decay_H2': round(segs['H2']['sharpe']-segs['H2_fee2x']['sharpe'],3),
            'fee_decay_B': round(segs['B']['sharpe']-segs['B_fee2x']['sharpe'],3),
            'fee_decay_C': round(segs['C']['sharpe']-segs['C_fee2x']['sharpe'],3),
        }
        rows.append(row)
        tag='CTRL' if is_ctrl else 'GRID'
        log(f'{tag} {idx:02d} sth={sth:.2f} etc_cd={etc_cd} trx_cd={trx_cd} | H1 sh={segs["H1"]["sharpe"]:.3f} ann={segs["H1"]["ann"]:.3f} dd={segs["H1"]["mdd"]:.4f} tr={segs["H1"]["trades"]} to={segs["H1"]["turnover"]:.4f} || H2 sh={segs["H2"]["sharpe"]:.3f} ann={segs["H2"]["ann"]:.4f} dd={segs["H2"]["mdd"]:.4f} tr={segs["H2"]["trades"]} to={segs["H2"]["turnover"]:.4f} H2_2x={segs["H2_fee2x"]["sharpe"]:.3f} corr={corr:.3f} || B sh={segs["B"]["sharpe"]:.3f} B2x={segs["B_fee2x"]["sharpe"]:.3f} || C sh={segs["C"]["sharpe"]:.3f} C2x={segs["C_fee2x"]["sharpe"]:.3f} || FULL sh={segs["FULL"]["sharpe"]:.3f} ann={segs["FULL"]["ann"]:.4f} dd={segs["FULL"]["mdd"]:.4f} cum={segs["FULL"]["cum"]:.4f}')

    # unbiased selection: H1-best
    h1_best = max(rows, key=lambda r: r['H1']['sharpe'])
    h2_best = max(rows, key=lambda r: r['H2']['sharpe'])
    log(f'H1-best idx={h1_best["idx"]} sth={h1_best["sth"]} etc={h1_best["etc_cd"]} trx={h1_best["trx_cd"]} H1={h1_best["H1"]["sharpe"]:.3f} -> H2={h1_best["H2"]["sharpe"]:.3f} B={h1_best["B"]["sharpe"]:.3f} C={h1_best["C"]["sharpe"]:.3f} FULL={h1_best["FULL"]["sharpe"]:.3f}')
    log(f'H2-best idx={h2_best["idx"]} sth={h2_best["sth"]} etc={h2_best["etc_cd"]} trx={h2_best["trx_cd"]} H2={h2_best["H2"]["sharpe"]:.3f} B={h2_best["B"]["sharpe"]:.3f} C={h2_best["C"]["sharpe"]:.3f} FULL={h2_best["FULL"]["sharpe"]:.3f} H1={h2_best["H1"]["sharpe"]:.3f}')

    unbiased_pass = bool(h1_best['H2']['sharpe'] > 3.0)
    decision = 'ADOPT_H1_BEST' if unbiased_pass else 'REJECT_KEEP_Z1'
    log(f'Unbiased gate H2>3.0: H1-best H2={h1_best["H2"]["sharpe"]:.3f} => {"PASS" if unbiased_pass else "FAIL"} decision={decision} (Z1 baseline sth0.12/etc18/trx6 H2 5.01 Z/W4)')

    # stress: fund x fee on B/C for both candidates
    funds=[0.0003,0.0005]
    fees=[0.0004,0.0008]
    stress={}
    for label, cand in [('H1_best',h1_best),('H2_best',h2_best)]:
        spec_pairs=[('ETC',(0.88, cand['sth'], cand['etc_cd'], None, TS)), ('TRX',(0.85, cand['sth'], cand['trx_cd'], 0.05, TS))]
        grid_bc=[]
        worst_B=None
        worst_C=None
        for fund in funds:
            for fee in fees:
                b,_ ,_ = eval_all(mats['B'], fee=fee, fund=fund, spec_pairs=spec_pairs)
                c,_ ,_ = eval_all(mats['C'], fee=fee, fund=fund, spec_pairs=spec_pairs)
                grid_bc.append({'fund':fund,'fee':fee,'B_sharpe':b['sharpe'],'B_ann':b['ann'],'C_sharpe':c['sharpe'],'C_ann':c['ann'],'B_mdd':b['mdd'],'C_mdd':c['mdd']})
                if worst_B is None or b['sharpe']<worst_B:
                    worst_B=b['sharpe']
                if worst_C is None or c['sharpe']<worst_C:
                    worst_C=c['sharpe']
                log(f'  stress {label} fund={fund:.4f} fee={fee:.4f} B={b["sharpe"]:.3f} C={c["sharpe"]:.3f}')
        pass_cond = bool(worst_B>0 and worst_C>1)
        stress[label]={'grid':grid_bc,'worst_B':round(worst_B,3),'worst_C':round(worst_C,3),'PASS':pass_cond}
        log(f'  stress {label} worst B={worst_B:.3f} worst C={worst_C:.3f} => {"PASS" if pass_cond else "FAIL"} (req B>0 C>1)')

    # compare W4/Z1 deltas: fetch Z1 best from existing if available else note
    import pathlib as _pl
    z1_ref=None
    try:
        z1_ref=json.load(open('results/backtest_Z1.json'))
    except: pass
    w4_ref=None
    try:
        w4_ref=json.load(open('results/backtest_W4_next.json'))
    except: pass

    # also compute Z1 literal sth0.12/etc18/trx6 for delta; find in rows
    z1_row = next((r for r in rows if r['sth']==0.12 and r['etc_cd']==18 and r['trx_cd']==6), None)

    res={
        'config':{
            'formula':FORMULA,
            'engine':'mirror run_z1.py/run_w4.py leg_series + quantile_mask_long(q0.3 long-only) + _vol_scale post-stops pre-roll + roll1',
            'fixed':{'lth_etc':0.88,'lth_trx':0.85,'sl_etc':None,'sl_trx':0.05,'ts':TS,'vt':VT,'vw':VW,'q':Q,'fund':FUND,'fee':BASE_FEE,'weights':[0.5,0.5],'venue':'aster','lev':2.0},
            'segments':{'H1':[5584,6077],'H2':[6077,6570],'B':[6380,6580],'C':[6080,6580],'FULL':[0,6580]},
            'grid':'sth[0.10,0.11,0.12] x etc_cd[15,18,21] x trx_cd[6,9]=18 + sth0.09 x etc18/trx6,9 ctrl=2 =>20',
            'bars':'15m x16 ->4h n=6580',
        },
        'rows':rows,
        'h1_best':{'idx':h1_best['idx'],'sth':h1_best['sth'],'etc_cd':h1_best['etc_cd'],'trx_cd':h1_best['trx_cd'],'control':h1_best['control'],'H1':h1_best['H1'],'H2':h1_best['H2'],'H2_fee2x':h1_best['H2_fee2x'],'B':h1_best['B'],'B_fee2x':h1_best['B_fee2x'],'C':h1_best['C'],'C_fee2x':h1_best['C_fee2x'],'FULL':h1_best['FULL'],'corr':h1_best['corr_ETC_TRX_H2']},
        'h2_best':{'idx':h2_best['idx'],'sth':h2_best['sth'],'etc_cd':h2_best['etc_cd'],'trx_cd':h2_best['trx_cd'],'control':h2_best['control'],'H1':h2_best['H1'],'H2':h2_best['H2'],'H2_fee2x':h2_best['H2_fee2x'],'B':h2_best['B'],'B_fee2x':h2_best['B_fee2x'],'C':h2_best['C'],'C_fee2x':h2_best['C_fee2x'],'FULL':h2_best['FULL'],'corr':h2_best['corr_ETC_TRX_H2']},
        'unbiased':{'H1_best_H2':h1_best['H2']['sharpe'],'threshold':3.0,'PASS':unbiased_pass,'decision':decision},
        'stress':stress,
        'z1_row': ({'sth':z1_row['sth'],'etc_cd':z1_row['etc_cd'],'trx_cd':z1_row['trx_cd'],'H1':z1_row['H1'],'H2':z1_row['H2'],'B':z1_row['B'],'C':z1_row['C'],'FULL':z1_row['FULL'],'B_fee2x':z1_row['B_fee2x'],'C_fee2x':z1_row['C_fee2x'],'H2_fee2x':z1_row['H2_fee2x']} if z1_row else None),
        'refs':{'z1_available': bool(z1_ref is not None),'w4_available': bool(w4_ref is not None)},
    }
    open('results/backtest_AA.json','w').write(json.dumps(res,indent=1,ensure_ascii=False))
    log(f'saved results/backtest_AA.json unbiased={decision} H1-best sth={h1_best["sth"]}/etc{h1_best["etc_cd"]}/trx{h1_best["trx_cd"]} H2={h1_best["H2"]["sharpe"]:.3f} H2-best sth={h2_best["sth"]}/etc{h2_best["etc_cd"]}/trx{h2_best["trx_cd"]} H2={h2_best["H2"]["sharpe"]:.3f}')

if __name__=='__main__':
    main()
