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


SEGS = {'H1': (5584,6077), 'H2': (6077,6570), 'B': (6380,6580), 'C': (6080,6580), 'FULL': (0,6580)}
SEGS_LIST = ['H1','H2','B','C','FULL']
FEE_SCAN = [0.0002, 0.0004, 0.0008, 0.0012]
FUNDS = [0.0003, 0.0005]
FEES = [0.0004, 0.0008]

BASE = dict(sth=0.10, etc_cd=15, trx_cd=9, vt=0.012, vw=12, ts=24, q=0.3)
VARIANTS = [
  ("base",   dict(BASE)),
  ("cd+2",   dict(BASE, etc_cd=17, trx_cd=11)),
  ("vt0.010",dict(BASE, vt=0.010)),
  ("vt0.015",dict(BASE, vt=0.015)),
  ("ts12",   dict(BASE, ts=12)),
  ("ts36",   dict(BASE, ts=36)),
  ("q0.25",  dict(BASE, q=0.25)),
  ("q0.35",  dict(BASE, q=0.35)),
]

def leg_series_v(raw, rets_t, sig, lth, sth, cd, sl, fee, fund, q, vt, vw, ts):
    return leg_series(raw, rets_t, sig, lth, sth, cd, sl, fee=fee, fund=fund, side='both', q=q, vt=vt, vw=vw, ts=ts)

def eval_all_v(mats_seg, fee, fund, sth, etc_cd, trx_cd, q, vt, vw, ts):
    spec = [('ETC',(0.88, sth, etc_cd, None, ts)), ('TRX',(0.85, sth, trx_cd, 0.05, ts))]
    legs, tr, tos = [], [], []
    for coin,(lth,s2,cd,sl,ts2) in spec:
        raw,rt,sg = mats_seg[coin]
        net,t,to = leg_series_v(raw,rt,sg,lth,s2,cd,sl,fee,fund,q,vt,vw,ts2)
        legs.append(net); tr.append(t); tos.append(to)
    cb = combo(legs,[0.5,0.5])
    s = stats(cb, trades=sum(tr), turnover=sum(tos)/len(tos) if tos else 0)
    s['trades_by'] = {spec[i][0]: tr[i] for i in range(len(spec))}
    return s, cb, legs

def fold_median(cb, n_fold=12):
    import statistics
    n = len(cb); fn = n // n_fold
    ss = []
    for fi in range(n_fold):
        a = fi*fn; b = a+fn if fi < n_fold-1 else n
        ss.append(stats(cb[a:b])['sharpe'])
    ss_sorted = sorted(ss)
    med = (ss_sorted[5]+ss_sorted[6])/2 if len(ss_sorted)==12 else statistics.median(ss_sorted)
    return round(med,3), [round(x,3) for x in ss]

def main():
    logp = pathlib.Path('logs/e13.log')
    logp.parent.mkdir(parents=True, exist_ok=True)
    open(logp,'w').write('E13 start\n')
    def log(msg):
        print(msg, flush=True)
        with open(logp,'a') as f: f.write(msg+'\n')
    full = {c: load_bars(c) for c in COINS}
    n = min(len(b) for b in full.values())
    log(f'full 4h bars n={n}')
    assert n == 6580, f'expected 6580 got {n}'
    mats = {}
    for seg,(a,b) in SEGS.items():
        mats[seg] = {c: build_mats(full[c][a:b]) for c in COINS}
        log(f'{seg} [{a}:{b}] built')
    out_vars = {}
    for vname, v in VARIANTS:
        log(f'=== variant {vname} {v} ===')
        segs = {}
        for seg in SEGS_LIST:
            s,cb,legs = eval_all_v(mats[seg], BASE_FEE, FUND, v['sth'], v['etc_cd'], v['trx_cd'], v['q'], v['vt'], v['vw'], v['ts'])
            segs[seg] = s
            log(f'  {seg} sh={s["sharpe"]:.3f} ann={s["ann"]:.4f} mdd={s["mdd"]:.4f} to={s["turnover"]:.6f} tr={s["trades"]}')
        # fee curve on FULL (mirror E3)
        fee_curve = []
        for fee in FEE_SCAN:
            s,cb,_ = eval_all_v(mats['FULL'], fee, FUND, v['sth'], v['etc_cd'], v['trx_cd'], v['q'], v['vt'], v['vw'], v['ts'])
            med, folds = fold_median(cb)
            fee_curve.append({'fee': fee, 'sharpe': s['sharpe'], 'ann': s['ann'], 'mdd': s['mdd'], 'cum': s['cum'], 'turnover': s['turnover'], 'fold_median_sharpe': med, 'folds_sharpe': folds})
            log(f'  fee {fee:.4f} FULL sh={s["sharpe"]:.3f} med12={med:.3f} to={s["turnover"]:.6f}')
        # stress grid fund x fee on B/C (mirror AA)
        grid = []
        worstB = None; worstC = None
        for fund in FUNDS:
            for fee in FEES:
                b,_,_ = eval_all_v(mats['B'], fee, fund, v['sth'], v['etc_cd'], v['trx_cd'], v['q'], v['vt'], v['vw'], v['ts'])
                c,_,_ = eval_all_v(mats['C'], fee, fund, v['sth'], v['etc_cd'], v['trx_cd'], v['q'], v['vt'], v['vw'], v['ts'])
                grid.append({'fund': fund, 'fee': fee, 'B_sharpe': b['sharpe'], 'B_ann': b['ann'], 'B_mdd': b['mdd'], 'C_sharpe': c['sharpe'], 'C_ann': c['ann'], 'C_mdd': c['mdd']})
                worstB = b['sharpe'] if worstB is None or b['sharpe'] < worstB else worstB
                worstC = c['sharpe'] if worstC is None or c['sharpe'] < worstC else worstC
                log(f'  stress fund={fund:.4f} fee={fee:.4f} B={b["sharpe"]:.3f} C={c["sharpe"]:.3f}')
        worstB = round(worstB,3); worstC = round(worstC,3)
        to = segs['FULL']['turnover']
        fsh = segs['FULL']['sharpe']
        h2 = segs['H2']['sharpe']
        base_med = next(x['fold_median_sharpe'] for x in fee_curve if x['fee']==BASE_FEE)
        gate = bool(to < 0.16 and fsh > 2.0 and h2 > 3.5)
        out_vars[vname] = {'params': v, 'segments': segs, 'fee_curve': fee_curve, 'stress_grid': grid, 'worst_B': worstB, 'worst_C': worstC, 'base_fold_median': base_med, 'gate': {'turnover_lt_016': bool(to<0.16), 'full_gt_20': bool(fsh>2.0), 'h2_gt_35': bool(h2>3.5), 'pass': gate}}
        log(f'  => worstB={worstB:.3f} worstC={worstC:.3f} to={to:.6f} FULL={fsh:.3f} H2={h2:.3f} med12={base_med:.3f} gate={"PASS" if gate else "FAIL"}')
    # rank: worstB desc among gate pass, else worstB desc
    ranked = sorted(out_vars.items(), key=lambda kv: (kv[1]['gate']['pass'], kv[1]['worst_B']), reverse=True)
    best = ranked[0][0]
    log(f'BEST {best} ranked={[k for k,_ in ranked]}')
    res = {
      'meta': {'formula': FORMULA, 'coins': COINS, 'base': 'AA-H1 sth0.10/etc15/trx9 vt0.012 ts24 q0.3', 'fund': FUND, 'fee_base': BASE_FEE, 'fee_scan': FEE_SCAN, 'funds': FUNDS, 'fees': FEES, 'segments': {k:list(v) for k,v in SEGS.items()}, 'note': 'E13 mirror research/run_e3.py fee_curve + research/run_aa.py stress grid (fund x fee 4-cell, B/C worst). FORMULA locked.'},
      'variants': out_vars,
      'rank': [k for k,_ in ranked],
      'best': best,
    }
    open('results/results_E13_fee.json','w').write(json.dumps(res, indent=1, ensure_ascii=False))
    log(f'saved results/results_E13_fee.json best={best}')
    log('E13 done')

if __name__ == '__main__':
    main()
