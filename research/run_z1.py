import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json, math, csv
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
BASE_SPEC = {
    'ETC': (0.88, 0.12, 18, None, 24, 'both'),
    'TRX': (0.85, 0.12, 6, 0.05, 24, 'both'),
}
VAR2_SPEC = {
    'ETC': (0.88, 0.12, 18, 0.02, 24, 'both'),
    'TRX': (0.85, 0.12, 6, 0.02, 24, 'short'),
}
VT_GRID = [0.008, 0.01, 0.012, 0.015]
VW_GRID = [8,12,24]
SL_VARIANTS = [
    ('base', BASE_SPEC),
    ('sl0.02_TRXoff', VAR2_SPEC),
]
SEGS = {'H2': (6077,6570), 'B': (6380,6580), 'C': (6080,6580)}

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
    sig=StackVM(use_advanced=False).execute(FORMULA, FeatureEngineer.compute_features(raw, use_advanced=False))
    rets=[(bars[i+1][3]-bars[i][3])/bars[i][3] for i in range(n-1)] + [0.0]
    return raw, torch.tensor([rets]), sig

def quantile_mask_long(sig,q):
    if q is None:
        return None
    a=sig.detach().float().abs().reshape(-1)
    k=max(1,int(len(a)*float(q)))
    thr=torch.topk(a,k).values.min()
    return (sig.detach().float().abs() >= thr).float()

def leg_series(raw, rets_t, sig, lth, sth, cd, sl, fee=None, fund=None, side='both', q=Q, vt=None, vw=24, ts=0):
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

def eval_seg(mats_seg, fee, spec, vt, vw):
    legs,tr,tos=[],[],[]
    for c in COINS:
        raw,rt,sg=mats_seg[c]
        lth,sth,cd,sl,ts,side=spec[c]
        net,t,to=leg_series(raw,rt,sg,lth,sth,cd,sl,fee=fee,side=side,q=Q,vt=vt,vw=vw,ts=ts)
        legs.append(net)
        tr.append(t)
        tos.append(to)
    cb=combo(legs,[0.5,0.5])
    s=stats(cb,trades=sum(tr),turnover=sum(tos)/len(tos) if tos else 0)
    s['trades_by']={c:tr[i] for i,c in enumerate(COINS)}
    return s

def vol_contrast_example(mats_seg, spec, vt, vw):
    c='ETC'
    raw, rets_t, sig=mats_seg[c]
    lth,sth,cd,sl,ts,side=spec[c]
    kw=dict(venue='aster', leverage=2.0, short_enabled=True, long_th=lth, short_th=sth, cooldown_bars=cd, bars_per_year=2190.0, stop_loss=sl, time_stop=ts, vol_target=vt, vol_window=vw, funding_override=FUND, fee_override=BASE_FEE)
    bt=MemeBacktest(**kw)
    signal=torch.sigmoid(sig)
    is_safe=(raw['liquidity']>bt.min_liq).float()
    lp=(signal>bt.long_th).float()*is_safe
    sp=(signal<bt.short_th).float()*is_safe
    mask=quantile_mask_long(sig,Q)
    if mask is not None:
        lp=lp*mask
    lp_c,sp_c=bt._apply_cooldown(lp,sp)
    lp_s,sp_s=bt._apply_stops(lp_c,sp_c,rets_t)
    scale=bt._vol_scale(rets_t)
    lp_pre=lp_s.clone()
    sp_pre=sp_s.clone()
    lp_post=lp_pre*scale
    sp_post=sp_pre*scale
    n=lp_pre.shape[1]
    if isinstance(scale,float):
        scale_list=[scale]*n
    else:
        scale_list=scale[0].tolist() if hasattr(scale,'tolist') else [float(scale)]*n
    lp_pre_l=lp_pre[0].tolist()
    sp_pre_l=sp_pre[0].tolist()
    lp_post_l=lp_post[0].tolist()
    sp_post_l=sp_post[0].tolist()
    candidates=[]
    for t in range(n):
        if lp_pre_l[t]>0 or sp_pre_l[t]>0:
            candidates.append(t)
            if len(candidates)>=20:
                break
    picks=[]
    if candidates:
        picks.append(candidates[0])
        if len(candidates)>1:
            mid=candidates[len(candidates)//2]
            picks.append(mid)
        if len(candidates)>2:
            picks.append(candidates[-1])
    else:
        picks=[0, n//2, n-1]
    while len(picks)<3:
        picks.append(max(0,n-1-len(picks)))
    picks=picks[:3]
    out=[]
    for t in picks:
        out.append({'bar':int(t), 'lp_pre':round(float(lp_pre_l[t]),4), 'sp_pre':round(float(sp_pre_l[t]),4), 'scale':round(float(scale_list[t]),4), 'lp_post':round(float(lp_post_l[t]),4), 'sp_post':round(float(sp_post_l[t]),4)})
    return out

def main():
    full={c:load_bars(c) for c in COINS}
    n=min(len(b) for b in full.values())
    print(f'full 4h bars n={n}',flush=True)
    assert n==6580, f'expected 6580 got {n}'
    mats={}
    for name,(a,b) in SEGS.items():
        mats[name]={c: build_mats(full[c][a:b]) for c in COINS}
        print(f'{name} [{a}:{b}] n={b-a} built',flush=True)
    rows=[]
    for vt in VT_GRID:
        for vw in VW_GRID:
            for var_name, spec in SL_VARIANTS:
                h2=eval_seg(mats['H2'], fee=BASE_FEE, spec=spec, vt=vt, vw=vw)
                b_seg=eval_seg(mats['B'], fee=BASE_FEE, spec=spec, vt=vt, vw=vw)
                c_seg=eval_seg(mats['C'], fee=BASE_FEE, spec=spec, vt=vt, vw=vw)
                b2=eval_seg(mats['B'], fee=FEE2X, spec=spec, vt=vt, vw=vw)
                c2=eval_seg(mats['C'], fee=FEE2X, spec=spec, vt=vt, vw=vw)
                row={'vol_target':vt,'vol_window':vw,'sl_variant':var_name,
                     'H2':h2,'B':b_seg,'C':c_seg,'B_fee2x':b2,'C_fee2x':c2}
                row['fee_decay_B']=round(b_seg['sharpe']-b2['sharpe'],3)
                row['fee_decay_C']=round(c_seg['sharpe']-c2['sharpe'],3)
                rows.append(row)
                print(f'Z1 vt={vt} vw={vw} var={var_name} H2(sh={h2["sharpe"]:.3f} ann={h2["ann"]:.4f} dd={h2["mdd"]:.4f} tr={h2["trades"]} to={h2["turnover"]:.4f}) B(sh={b_seg["sharpe"]:.3f} ann={b_seg["ann"]:.4f} dd={b_seg["mdd"]:.4f} tr={b_seg["trades"]} to={b_seg["turnover"]:.4f} B2x={b2["sharpe"]:.3f}) C(sh={c_seg["sharpe"]:.3f} ann={c_seg["ann"]:.4f} dd={c_seg["mdd"]:.4f} tr={c_seg["trades"]} to={c_seg["turnover"]:.4f} C2x={c2["sharpe"]:.3f})',flush=True)
    filtered=[r for r in rows if max(r['B']['turnover'], r['C']['turnover']) < 0.15]
    pool=filtered if filtered else rows
    best=max(pool, key=lambda r: r['B']['sharpe']+r['C']['sharpe'])
    best_to=max(best['B']['turnover'], best['C']['turnover'])
    global_best=max(rows, key=lambda r: r['B']['sharpe']+r['C']['sharpe'])
    print(f'FILTERED pool {len(filtered)}/24, BEST vt={best["vol_target"]} vw={best["vol_window"]} var={best["sl_variant"]} B={best["B"]["sharpe"]:.3f} C={best["C"]["sharpe"]:.3f} to={best_to:.4f}',flush=True)
    print(f'GLOBAL best vt={global_best["vol_target"]} vw={global_best["vol_window"]} var={global_best["sl_variant"]} B={global_best["B"]["sharpe"]:.3f} C={global_best["C"]["sharpe"]:.3f}',flush=True)
    pass_obj={
        'best_vt':best['vol_target'],
        'best_vw':best['vol_window'],
        'best_sl_variant':best['sl_variant'],
        'B_gt_2.5': bool(best['B']['sharpe']>2.5),
        'C_gt_3.0': bool(best['C']['sharpe']>3.0),
        'turnover_lt_0.15': bool(best_to<0.15),
        'fee2x_B_gt_1.0': bool(best['B_fee2x']['sharpe']>1.0),
    }
    pass_obj['overall']=bool(pass_obj['B_gt_2.5'] and pass_obj['C_gt_3.0'] and pass_obj['fee2x_B_gt_1.0'] and pass_obj['turnover_lt_0.15'])
    spec_map={n:s for n,s in SL_VARIANTS}
    best_spec=spec_map[best['sl_variant']]
    contrast_B=vol_contrast_example(mats['B'], best_spec, best['vol_target'], best['vol_window'])
    contrast_C=vol_contrast_example(mats['C'], best_spec, best['vol_target'], best['vol_window'])
    print('VOL contrast B segment (ETC) 3 bars:',contrast_B,flush=True)
    print('VOL contrast C segment (ETC) 3 bars:',contrast_C,flush=True)
    res={
        'config':{
            'base':'Y2 optimal: FORMULA [3,2,7,2,7,11,15,4,4,6,6,10], ETC(0.88/0.12/cd18/None/ts24)+TRX(0.85/0.12/cd6/0.05/ts24), q=0.3 long-gate, 50/50, aster 2x fund0.0005 fee0.0004',
            'formula':FORMULA,
            'base_spec':{k:list(v[:4])+[v[4],v[5]] for k,v in BASE_SPEC.items()},
            'variant_spec':{k:list(v[:4])+[v[4],v[5]] for k,v in VAR2_SPEC.items()},
            'sl_variants':['base: ETC None/TRX 0.05 both ts24','sl0.02_TRXoff: ETC 0.02 both / TRX 0.02 short-only ts24'],
            'weights':{'ETC':0.5,'TRX':0.5},'lev':2.0,'q':Q,'q_side':'long','side':'both','fund':FUND,'fee':BASE_FEE,'fee2x':FEE2X,
            'full_n':n,
            'segments':{'H2':[6077,6570],'B':[6380,6580],'C':[6080,6580]},
            'grid':'vt[0.008,0.01,0.012,0.015] x vw[8,12,24] x sl_variant[base, sl0.02_TRXoff] = 24 rows, per-leg _vol_scale post-stops pre-roll',
            'note':'replicates research/run_y2.py leg_series + quantile_mask_long + MemeBacktest._vol_scale; vol scaling after _apply_stops before roll'
        },
        'rows':rows,
        'best':{'vol_target':best['vol_target'],'vol_window':best['vol_window'],'sl_variant':best['sl_variant'],'H2':best['H2'],'B':best['B'],'C':best['C'],'B_fee2x':best['B_fee2x'],'C_fee2x':best['C_fee2x'],'max_turnover':best_to,'fee_decay_B':best['fee_decay_B'],'fee_decay_C':best['fee_decay_C']},
        'global_best':{'vol_target':global_best['vol_target'],'vol_window':global_best['vol_window'],'sl_variant':global_best['sl_variant'],'B_sharpe':global_best['B']['sharpe'],'C_sharpe':global_best['C']['sharpe'],'H2_sharpe':global_best['H2']['sharpe']},
        'PASS':pass_obj,
        'vol_contrast':{'note':'best scaling before/after on ETC, scale = clamp(vt/trailing_vol,0.2,2.0) with roll1, applied post-stops pre-roll','B_ETC_3bars':contrast_B,'C_ETC_3bars':contrast_C},
        'filtered_count':len(filtered)
    }
    open('results/backtest_Z1.json','w').write(json.dumps(res,indent=1))
    print(f'saved results/backtest_Z1.json BEST vt={best["vol_target"]} vw={best["vol_window"]} var={best["sl_variant"]} B={best["B"]["sharpe"]:.3f} C={best["C"]["sharpe"]:.3f} to={best_to:.4f} B2x={best["B_fee2x"]["sharpe"]:.3f} PASS={pass_obj["overall"]}',flush=True)

if __name__=='__main__':
    main()
