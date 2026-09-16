"""Z3 weight & third-leg expansion on Y2 optimal base.
Base: ETC(0.88/0.12/cd18/ts24) + TRX(0.85/0.12/cd6/sl0.05/ts24), vt0.01 vw12, q0.3 50/50.
Grid A weight: wETC [0.3,0.4,0.5,0.6,0.7] 5 rows, corr ETC_TRX.
Grid B third leg at w 0.5/0.3/0.7 three anchors, third coin DOGE/BTC/SOL (same as TRX 0.85/0.12/cd6/sl0.05/ts24/vt0.01) eq 33/33/33 vs 40/40/20 => 3*3*2=18 rows.
Slices fixed: H2[6077:6570] B[6380:6580] C[6080:6580], each H2/B/C sharpe/ann/mdd/trades + fee2x B/C.
Mirrors run_y2 leg_series with time_stop=24 added per Z3 spec.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json, math, csv
import torch
from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest

FORMULA = [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10]
BASE_FEE = 0.0004
FEE2X = 0.0008
FUND = 0.0005
Q = 0.3
VT = 0.01
VW = 12
TS = 24
# per-coin params: (lth, sth, cd, sl, ts)
P_ETC  = (0.88, 0.12, 18, None, 24)
P_TRX  = (0.85, 0.12, 6, 0.05, 24)
P_THIRD = (0.85, 0.12, 6, 0.05, 24)  # DOGE/BTC/SOL reuse TRX spec
COINS_BASE = ['ETC','TRX']
THIRD_COINS = ['DOGE','BTC','SOL']
WEIGHT_GRID = [0.3,0.4,0.5,0.6,0.7]
THIRD_WEIGHT_ANCHORS = [0.5,0.3,0.7]  # wETC value anchor; thirdleg explore at these w
THIRD_MIXES = ['eq333','40_40_20']  # eq 33/33/33 vs 40/40/20 (ETC/TRX/third)
# slices fixed per task
SLICE = {'H2': (6077,6570), 'B': (6380,6580), 'C': (6080,6580)}

def load_bars(coin):
    rows = list(csv.DictReader(open('data/data_15m_3y/' + coin + '.csv')))
    bars=[]
    for i in range(0,len(rows),16):
        blk=rows[i:i+16]
        if len(blk)<16: break
        bars.append((float(blk[0]['open']), max(float(x['high']) for x in blk), min(float(x['low']) for x in blk), float(blk[-1]['close']), sum(float(x['volume']) for x in blk)))
    return bars

def build_mats(bars):
    n=len(bars)
    raw={'open': torch.tensor([[b[0] for b in bars]]),
         'high': torch.tensor([[b[1] for b in bars]]),
         'low': torch.tensor([[b[2] for b in bars]]),
         'close': torch.tensor([[b[3] for b in bars]]),
         'volume': torch.tensor([[b[4] for b in bars]]),
         'liquidity': torch.full((1,n),1e7),
         'fdv': torch.full((1,n),1e8)}
    sig=StackVM(use_advanced=False).execute(FORMULA, FeatureEngineer.compute_features(raw, use_advanced=False))
    rets=[(bars[i+1][3]-bars[i][3])/bars[i][3] for i in range(n-1)]+[0.0]
    return raw, torch.tensor([rets]), sig

def quantile_mask_long(sig,q):
    if q is None: return None
    a=sig.detach().float().abs().reshape(-1)
    k=max(1,int(len(a)*float(q)))
    thr=torch.topk(a,k).values.min()
    return (sig.detach().float().abs()>=thr).float()

def leg_series(raw, rets_t, sig, lth, sth, cd, sl, ts, fee=None, fund=None, side='both', q=Q, vt=VT, vw=VW):
    kw=dict(venue='aster', leverage=2.0, short_enabled=True, long_th=lth, short_th=sth, cooldown_bars=cd, bars_per_year=2190.0, stop_loss=sl, time_stop=ts, vol_target=vt, vol_window=vw)
    kw['funding_override']=FUND if fund is None else fund
    if fee is not None: kw['fee_override']=fee
    bt=MemeBacktest(**kw)
    signal=torch.sigmoid(sig)
    is_safe=(raw['liquidity']>bt.min_liq).float()
    lp=(signal>bt.long_th).float()*is_safe
    sp=(signal<bt.short_th).float()*is_safe
    mask=quantile_mask_long(sig,q)
    if mask is not None: lp=lp*mask
    lp,sp=bt._apply_cooldown(lp,sp)
    lp,sp=bt._apply_stops(lp,sp,rets_t)
    scale=bt._vol_scale(rets_t)
    lp,sp=lp*scale, sp*scale
    if side=='long': sp=sp*0.0
    if side=='short': lp=lp*0.0
    lp=lp.roll(1,dims=1); lp[:,0]=0
    sp=sp.roll(1,dims=1); sp[:,0]=0
    turn=(lp-lp.roll(1,dims=1)).abs()+(sp-sp.roll(1,dims=1)).abs()
    tx=turn*(bt.base_fee + torch.clamp(bt.trade_size/(raw['liquidity']+1e-9),0.0,0.05))
    gross=(lp-sp)*rets_t*bt.leverage
    fnd=(lp-sp)*bt.default_funding_rate*bt.leverage
    net=(gross - tx*bt.leverage - fnd)[0].tolist()
    turnl=turn[0].tolist()
    pos=(lp-sp)[0].tolist()
    trades=sum(1 for t in range(len(pos)) if pos[t]!=0.0 and (t==0 or pos[t-1]==0.0))
    turnover=sum(turnl)/len(turnl)
    return net,trades,turnover

def stats(ser,trades=0,turnover=0.0):
    n=len(ser)
    mean=sum(ser)/n
    var=sum((x-mean)**2 for x in ser)/max(n-1,1)
    sharpe=mean/math.sqrt(var)*math.sqrt(2190.0) if var>0 else 0.0
    cum=sum(ser)
    ann=cum/n*2190.0
    cs,peak,mdd=0.0,-1e18,0.0
    for x in ser:
        cs+=x; peak=max(peak,cs); mdd=max(mdd,peak-cs)
    return {'sharpe':round(sharpe,3),'ann':round(ann,4),'mdd':round(mdd,4),'cum':round(cum,4),'n':n,'trades':trades,'turnover':round(turnover,6)}

def combo(ser_list, weights=None):
    m=min(len(s) for s in ser_list)
    k=len(ser_list)
    w=weights or [1.0/k]*k
    sw=sum(w)
    return [sum(ser_list[i][t]*w[i]/sw for i in range(k)) for t in range(m)]

def corr(a,b):
    n=min(len(a),len(b))
    if n<2: return 0.0
    ma=sum(a[:n])/n; mb=sum(b[:n])/n
    num=sum((a[i]-ma)*(b[i]-mb) for i in range(n))
    da=sum((a[i]-ma)**2 for i in range(n)); db=sum((b[i]-mb)**2 for i in range(n))
    den=math.sqrt(da*db)
    return round(num/den,3) if den>1e-12 else 0.0

def eval_combo(mats_seg, fee, weights, coins):
    # coins order matches weights
    legs,tr,tos=[],[],[]
    # map coin->params
    param_map={'ETC':P_ETC,'TRX':P_TRX,'DOGE':P_THIRD,'BTC':P_THIRD,'SOL':P_THIRD}
    for c in coins:
        raw,rt,sg=mats_seg[c]
        lth,sth,cd,sl,ts=param_map[c]
        net,t,to=leg_series(raw,rt,sg,lth,sth,cd,sl,ts,fee=fee,side='both',q=Q,vt=VT,vw=VW)
        legs.append(net); tr.append(t); tos.append(to)
    cb=combo(legs, weights)
    # corr: for 2 legs = ETC_TRX, for 3 legs = pairwise avg? include ETC_TRX as primary
    if len(coins)==2:
        cval=corr(legs[0],legs[1])
    else:
        # record ETC_TRX corr and also avg pairwise
        c_etc_trx=corr(legs[0],legs[1])
        cval=c_etc_trx
    s=stats(cb,trades=sum(tr),turnover=sum(tos)/len(tos))
    s['trades_by']={c:tr[i] for i,c in enumerate(coins)}
    return s, legs

def main():
    all_coins=list(set(COINS_BASE+THIRD_COINS))
    full={c:load_bars(c) for c in all_coins}
    n=min(len(b) for b in full.values())
    print('full 4h bars per coin n=%d (min across %s)'%(n, all_coins),flush=True)
    # build seg mats per spec slices (absolute indices)
    seg={}
    for seg_name,(a,b) in SLICE.items():
        seg[seg_name]={c: build_mats(full[c][a:b]) for c in all_coins}
        print('seg %s [%d:%d] n=%d'%(seg_name,a,b,b-a),flush=True)
    print('Z3 base ETC%s TRX%s vt=%.2f vw=%d q=%.1f ts=%d fee=%.4f fund=%.4f'%(str(P_ETC),str(P_TRX),VT,VW,Q,TS,BASE_FEE,FUND),flush=True)
    # Grid A weight sweep
    weight_rows=[]
    for w in WEIGHT_GRID:
        w_etc=w; w_trx=1-w
        weights=[w_etc,w_trx]
        row={'wETC':w_etc,'wTRX':w_trx,'weights':weights,'coins':['ETC','TRX']}
        for seg_name in ['H2','B','C']:
            s,_=eval_combo(seg[seg_name],BASE_FEE,weights,['ETC','TRX'])
            row[seg_name]=s
        for seg_name in ['B','C']:
            s,_=eval_combo(seg[seg_name],FEE2X,weights,['ETC','TRX'])
            row[seg_name+'_fee2x']=s
        # correlations per segment
        for seg_name in ['H2','B','C']:
            _,legs=eval_combo(seg[seg_name],BASE_FEE,weights,['ETC','TRX'])
            row['corr_'+seg_name]=corr(legs[0],legs[1])
        # also fees decay
        row['fee_decay_B']=round(row['B']['sharpe']-row['B_fee2x']['sharpe'],3)
        row['fee_decay_C']=round(row['C']['sharpe']-row['C_fee2x']['sharpe'],3)
        weight_rows.append(row)
        print('W wETC=%.1f H2(sh=%.3f ann=%.4f dd=%.4f tr=%d to=%.4f) B(sh=%.3f ann=%.4f dd=%.4f tr=%d to=%.4f B2x=%.3f) C(sh=%.3f C2x=%.3f) corr(H2/B/C)=%.3f/%.3f/%.3f' %(
            w_etc,row['H2']['sharpe'],row['H2']['ann'],row['H2']['mdd'],row['H2']['trades'],row['H2']['turnover'],
            row['B']['sharpe'],row['B']['ann'],row['B']['mdd'],row['B']['trades'],row['B']['turnover'],row['B_fee2x']['sharpe'],
            row['C']['sharpe'],row['C_fee2x']['sharpe'],
            row['corr_H2'],row['corr_B'],row['corr_C']),flush=True)
    best_weight=max(weight_rows, key=lambda r: r['B']['sharpe']+r['C']['sharpe'])
    # Grid B third leg
    thirdleg_rows=[]
    for w_anchor in THIRD_WEIGHT_ANCHORS:
        for third in THIRD_COINS:
            for mix in THIRD_MIXES:
                if mix=='eq333':
                    weights=[1/3,1/3,1/3]
                    w_desc='33/33/33'
                else:
                    # 40/40/20 => ETC 0.4 TRX 0.4 third 0.2 regardless of anchor? Or anchored mix 40/40/20 scaled?
                    # Use 0.4/0.4/0.2 per spec example
                    weights=[0.4,0.4,0.2]
                    w_desc='40/40/20'
                coins=['ETC','TRX',third]
                row={'w_anchor':w_anchor,'third':third,'mix':mix,'mix_desc':w_desc,'weights':[round(x,4) for x in weights],'coins':coins}
                for seg_name in ['H2','B','C']:
                    s,_=eval_combo(seg[seg_name],BASE_FEE,weights,coins)
                    row[seg_name]=s
                for seg_name in ['B','C']:
                    s,_=eval_combo(seg[seg_name],FEE2X,weights,coins)
                    row[seg_name+'_fee2x']=s
                row['fee_decay_B']=round(row['B']['sharpe']-row['B_fee2x']['sharpe'],3)
                row['fee_decay_C']=round(row['C']['sharpe']-row['C_fee2x']['sharpe'],3)
                # corr ETC_TRX in this 3-leg combo context
                for seg_name in ['H2','B','C']:
                    _,legs=eval_combo(seg[seg_name],BASE_FEE,weights,coins)
                    row['corr_ETC_TRX_'+seg_name]=corr(legs[0],legs[1])
                thirdleg_rows.append(row)
                print('THIRD anchor=%.1f %s %s H2=%.3f B=%.3f (2x=%.3f) C=%.3f (2x=%.3f) etc_trx_corr_H2=%.3f'%(
                    w_anchor,third,mix,row['H2']['sharpe'],row['B']['sharpe'],row['B_fee2x']['sharpe'],row['C']['sharpe'],row['C_fee2x']['sharpe'],row['corr_ETC_TRX_H2']),flush=True)
    best_third=max(thirdleg_rows, key=lambda r: r['B']['sharpe']+r['C']['sharpe']) if thirdleg_rows else None
    # PASS per task: best_weight B>3 & fee2xB>2
    pass_flag=bool(best_weight['B']['sharpe']>3 and best_weight['B_fee2x']['sharpe']>2)
    res={
        'config':{
            'base':'Y2 optimal ETC(0.88/0.12/cd18/ts24)+TRX(0.85/0.12/cd6/sl0.05/ts24), vt0.01 vw12, q0.3 50/50',
            'formula':FORMULA,
            'params':{'ETC':list(P_ETC),'TRX':list(P_TRX),'THIRD_same_as_TRX':list(P_THIRD)},
            'vol':{'vt':VT,'vw':VW},'q':Q,'ts':TS,'fee':BASE_FEE,'fee2x':FEE2X,'fund':FUND,
            'slices':{k:list(v) for k,v in SLICE.items()},
            'full_n':n,
            'weight_grid':WEIGHT_GRID,
            'third_grid':{'anchors':THIRD_WEIGHT_ANCHORS,'coins':THIRD_COINS,'mixes':THIRD_MIXES},
            'note':'mirrors run_y2 leg_series + _vol_scale post-stops; slices fixed H2[6077:6570] B[6380:6580] C[6080:6580] as Z3 spec; third leg params reuse TRX (0.85/0.12/cd6/sl0.05/ts24/vt0.01)'
        },
        'weight_rows':weight_rows,
        'thirdleg_rows':thirdleg_rows,
        'best_weight':{'wETC':best_weight['wETC'],'wTRX':best_weight['wTRX'],'weights':best_weight['weights'],'H2':best_weight['H2'],'B':best_weight['B'],'C':best_weight['C'],'B_fee2x':best_weight['B_fee2x'],'C_fee2x':best_weight['C_fee2x'],'corr_H2':best_weight['corr_H2'],'corr_B':best_weight['corr_B'],'corr_C':best_weight['corr_C'],'fee_decay_B':best_weight['fee_decay_B'],'fee_decay_C':best_weight['fee_decay_C']},
        'best_thirdleg':{'w_anchor':best_third['w_anchor'],'third':best_third['third'],'mix':best_third['mix'],'weights':best_third['weights'],'coins':best_third['coins'],'H2':best_third['H2'],'B':best_third['B'],'C':best_third['C'],'B_fee2x':best_third['B_fee2x'],'C_fee2x':best_third['C_fee2x'],'corr_ETC_TRX_H2':best_third['corr_ETC_TRX_H2'],'corr_ETC_TRX_B':best_third['corr_ETC_TRX_B'],'corr_ETC_TRX_C':best_third['corr_ETC_TRX_C'],'fee_decay_B':best_third['fee_decay_B'],'fee_decay_C':best_third['fee_decay_C']} if best_third else None,
        'PASS':{'best_weight_B_gt_3':bool(best_weight['B']['sharpe']>3),'best_weight_fee2xB_gt_2':bool(best_weight['B_fee2x']['sharpe']>2),'overall':pass_flag}
    }
    open('results/backtest_Z3.json','w').write(json.dumps(res,indent=1))
    print('BEST_WEIGHT wETC=%.2f B=%.3f C=%.3f B2x=%.3f corr_B=%.3f'%(best_weight['wETC'],best_weight['B']['sharpe'],best_weight['C']['sharpe'],best_weight['B_fee2x']['sharpe'],best_weight['corr_B']),flush=True)
    if best_third:
        print('BEST_THIRD anchor=%.1f %s %s B=%.3f C=%.3f B2x=%.3f'%(best_third['w_anchor'],best_third['third'],best_third['mix'],best_third['B']['sharpe'],best_third['C']['sharpe'],best_third['B_fee2x']['sharpe']),flush=True)
    print('PASS overall=%s (B>3 %s fee2xB>2 %s)'%(pass_flag, best_weight['B']['sharpe']>3, best_weight['B_fee2x']['sharpe']>2),flush=True)
    print('saved results/backtest_Z3.json',flush=True)

if __name__=='__main__':
    main()
