
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
VW = 12
TS = 24
# default vt for AA compare
VT_DEFAULT = 0.012

SEGS = {'H1': (5584,6077), 'H2': (6077,6570), 'B': (6380,6580), 'C': (6080,6580), 'FULL': (0,6580)}
SEGS_LIST = ['H1','H2','B','C','FULL']

# AA grid for recalc
STH_GRID_AA = [0.10, 0.11, 0.12]
ETC_CD_GRID_AA = [15,18,21]
TRX_CD_GRID_AA = [6,9]

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

def leg_series(raw, rets_t, sig, lth, sth, cd, sl, fee=None, fund=None, side='both', q=Q, vt=VT_DEFAULT, vw=VW, ts=TS):
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

def spearman(x,y):
    # rank correlation
    n=len(x)
    if n!=len(y) or n<2:
        return 0.0
    # rank: 1 highest
    def rankdata(v):
        # average rank for ties not needed much, use simple argsort
        idx=sorted(range(len(v)), key=lambda i: v[i], reverse=True)
        r=[0]*len(v)
        for rank, i in enumerate(idx):
            r[i]=rank+1
        # handle ties: average
        # detect ties
        vals={}
        for i,val in enumerate(v):
            vals.setdefault(val, []).append(i)
        for val, lst in vals.items():
            if len(lst)>1:
                avg=sum(r[i] for i in lst)/len(lst)
                for i in lst:
                    r[i]=avg
        return r
    rx=rankdata(x)
    ry=rankdata(y)
    # pearson on ranks
    mx=sum(rx)/n
    my=sum(ry)/n
    num=sum((rx[i]-mx)*(ry[i]-my) for i in range(n))
    va=sum((rx[i]-mx)**2 for i in range(n))
    vb=sum((ry[i]-my)**2 for i in range(n))
    den=math.sqrt(va*vb) if va>0 and vb>0 else 0
    if den==0:
        return 0.0
    return round(num/den,4)

def eval_all(mats_seg, fee, fund, spec_pairs, vt_map=None):
    # vt_map: dict coin->vt else use spec_pairs vt? spec_pairs now includes vt via closure
    # spec_pairs: list of (coin,(lth,sth,cd,sl,ts,vt))
    legs,tr,tos=[],[],[]
    for coin, spec in spec_pairs:
        if len(spec)==6:
            lth,sth,cd,sl,ts,vt = spec
        else:
            lth,sth,cd,sl,ts = spec
            vt = VT_DEFAULT
        raw,rt,sg=mats_seg[coin]
        net,t,to=leg_series(raw,rt,sg,lth,sth,cd,sl,fee=fee,fund=fund,vt=vt)
        legs.append(net); tr.append(t); tos.append(to)
    cb=combo(legs,[0.5,0.5])
    s=stats(cb,trades=sum(tr),turnover=sum(tos)/len(tos) if tos else 0)
    s['trades_by']={spec_pairs[i][0]:tr[i] for i in range(len(spec_pairs))}
    return s, cb, legs

def main():
    logp=pathlib.Path('logs/e4.log')
    logp.parent.mkdir(parents=True, exist_ok=True)
    open(logp,'w').write('E4 start\n')
    def log(msg):
        print(msg, flush=True)
        with open(logp,'a') as f: f.write(msg+'\n')

    full={c:load_bars(c) for c in COINS}
    n=min(len(b) for b in full.values())
    log(f'full 4h bars n={n} ETC={len(full["ETC"])} TRX={len(full["TRX"])}')
    assert n==6580, f'expected 6580 got {n}'
    mats={}
    for seg,(a,b) in SEGS.items():
        mats[seg]={c: build_mats(full[c][a:b]) for c in COINS}
        log(f'{seg} [{a}:{b}] n={b-a} built')
    # also FULL mats for 12fold split
    full_mats = {c: build_mats(full[c]) for c in COINS}

    # Build E4 new grid 20 rows per spec
    # sth 0.09/0.10 x etc12/15/18 x vt0.008/0.012 =>12
    # + sth0.11/0.12 x etc15/18 x vt None/0.015 =>8  =>20
    e4_new=[]
    for sth in [0.09,0.10]:
        for etc_cd in [12,15,18]:
            for vt in [0.008,0.012]:
                e4_new.append((sth, etc_cd, 9, vt))
    for sth in [0.11,0.12]:
        for etc_cd in [15,18]:
            for vt in [None,0.015]:
                e4_new.append((sth, etc_cd, 9, vt))
    assert len(e4_new)==20, f'e4_new {len(e4_new)}'
    # AA recalc grid 20
    aa_grid=[]
    for sth in STH_GRID_AA:
        for etc_cd in ETC_CD_GRID_AA:
            for trx_cd in TRX_CD_GRID_AA:
                aa_grid.append((sth, etc_cd, trx_cd, VT_DEFAULT))
    aa_grid.append((0.09, 18, 6, VT_DEFAULT))
    aa_grid.append((0.09, 18, 9, VT_DEFAULT))
    assert len(aa_grid)==20

    FOLD_N = 6580//12  # 548
    fold_bounds=[(i*FOLD_N, (i+1)*FOLD_N if i<11 else 6580) for i in range(12)]

    def run_grid(grid, label):
        rows=[]
        for idx,(sth, etc_cd, trx_cd, vt) in enumerate(grid):
            spec_pairs=[('ETC',(0.88, sth, etc_cd, None, TS, vt)), ('TRX',(0.85, sth, trx_cd, 0.05, TS, vt))]
            segs={}
            feat={}
            for seg in SEGS_LIST:
                s,cb,legs = eval_all(mats[seg], fee=BASE_FEE, fund=FUND, spec_pairs=spec_pairs)
                segs[seg]=s
                feat[seg]=(cb,legs)
            for seg in ['H2','B','C']:
                s2,_,_ = eval_all(mats[seg], fee=FEE2X, fund=FUND, spec_pairs=spec_pairs)
                segs[seg+'_fee2x']=s2
            _, legs_h2 = feat['H2']
            corr = pearson(legs_h2[0], legs_h2[1]) if len(legs_h2)==2 else 0.0
            # 12fold on FULL
            fold_sharpes=[]
            folds=[]
            for fi,(fa,fb) in enumerate(fold_bounds):
                # build mats for fold slice from full_mats raw? easier: rebuild from full bars slice
                # use full_mats slicing via eval_all on mats constructed from full[c][fa:fb]
                fold_mats={c: build_mats(full[c][fa:fb]) for c in COINS}
                s_fold,_,_ = eval_all(fold_mats, fee=BASE_FEE, fund=FUND, spec_pairs=spec_pairs)
                fold_sharpes.append(s_fold['sharpe'])
                folds.append({'fold':fi,'range':[fa,fb],'sharpe':s_fold['sharpe'],'ann':s_fold['ann'],'mdd':s_fold['mdd'],'trades':s_fold['trades'],'turnover':s_fold['turnover']})
            fold_mean=round(sum(fold_sharpes)/len(fold_sharpes),3) if fold_sharpes else 0.0
            fold_median=round(sorted(fold_sharpes)[len(fold_sharpes)//2],3) if fold_sharpes else 0.0
            fold_min=round(min(fold_sharpes),3) if fold_sharpes else 0.0
            fold_max=round(max(fold_sharpes),3) if fold_sharpes else 0.0
            n_pos=sum(1 for x in fold_sharpes if x>0)
            row={
                'idx': idx,
                'sth': sth, 'etc_cd': etc_cd, 'trx_cd': trx_cd, 'vt': vt, 'vw': VW, 'q': Q,
                'lth_etc':0.88,'lth_trx':0.85,'sl_etc':None,'sl_trx':0.05,'ts':TS,'weights':[0.5,0.5],
                'H1': segs['H1'], 'H2': segs['H2'], 'B': segs['B'], 'C': segs['C'], 'FULL': segs['FULL'],
                'H2_fee2x': segs['H2_fee2x'], 'B_fee2x': segs['B_fee2x'], 'C_fee2x': segs['C_fee2x'],
                'corr_ETC_TRX_H2': corr,
                'fee_decay_H2': round(segs['H2']['sharpe']-segs['H2_fee2x']['sharpe'],3),
                'fee_decay_B': round(segs['B']['sharpe']-segs['B_fee2x']['sharpe'],3),
                'fee_decay_C': round(segs['C']['sharpe']-segs['C_fee2x']['sharpe'],3),
                'folds': folds,
                'fold_sharpes': [round(x,3) for x in fold_sharpes],
                'fold_mean': fold_mean,
                'fold_median': fold_median,
                'fold_min': fold_min,
                'fold_max': fold_max,
                'fold_n_pos': n_pos,
            }
            rows.append(row)
            vt_str='None' if vt is None else f'{vt:.3f}'
            log(f'{label} {idx:02d} sth={sth:.2f} etc_cd={etc_cd} trx_cd={trx_cd} vt={vt_str} | H1 sh={segs["H1"]["sharpe"]:.3f} H2 sh={segs["H2"]["sharpe"]:.3f} H2_2x={segs["H2_fee2x"]["sharpe"]:.3f} FULL sh={segs["FULL"]["sharpe"]:.3f} fold_mean={fold_mean:.3f} med={fold_median:.3f} min={fold_min:.3f} npos={n_pos}/12 corr={corr:.3f} to_H1={segs["H1"]["turnover"]:.4f} to_H2={segs["H2"]["turnover"]:.4f}')
        return rows

    log('=== E4 NEW 20 ===')
    rows_new = run_grid(e4_new, 'NEW')
    log('=== AA RECALC 20 ===')
    rows_aa = run_grid(aa_grid, 'AA')

    # Unbiased selection on NEW only + combined
    # top3 by H1 for new
    new_sorted = sorted(rows_new, key=lambda r: r['H1']['sharpe'], reverse=True)
    new_top3 = new_sorted[:3]
    aa_sorted = sorted(rows_aa, key=lambda r: r['H1']['sharpe'], reverse=True)
    aa_top = aa_sorted[0]

    # combined for spearman
    combined = rows_new + rows_aa
    h1_vals = [r['H1']['sharpe'] for r in combined]
    h2_vals = [r['H2']['sharpe'] for r in combined]
    spear_combined = spearman(h1_vals, h2_vals)
    spear_new = spearman([r['H1']['sharpe'] for r in rows_new], [r['H2']['sharpe'] for r in rows_new])
    spear_aa = spearman([r['H1']['sharpe'] for r in rows_aa], [r['H2']['sharpe'] for r in rows_aa])

    log(f'NEW H1-top3:')
    for r in new_top3:
        vt_str='None' if r['vt'] is None else f"{r['vt']:.3f}"
        log(f"  sth={r['sth']:.2f} etc={r['etc_cd']} trx={r['trx_cd']} vt={vt_str} H1={r['H1']['sharpe']:.3f} -> H2={r['H2']['sharpe']:.3f} FULL={r['FULL']['sharpe']:.3f} fold_mean={r['fold_mean']:.3f} B={r['B']['sharpe']:.3f} C={r['C']['sharpe']:.3f}")
    log(f'AA H1-best idx={aa_top["idx"]} sth={aa_top["sth"]} etc={aa_top["etc_cd"]} trx={aa_top["trx_cd"]} H1={aa_top["H1"]["sharpe"]:.3f} -> H2={aa_top["H2"]["sharpe"]:.3f} FULL={aa_top["FULL"]["sharpe"]:.3f} fold_mean={aa_top["fold_mean"]:.3f}')
    log(f'Spearman H1<->H2 NEW={spear_new:.4f} AA={spear_aa:.4f} COMBINED40={spear_combined:.4f} (W4 negative warning if <0)')

    # Promotion gate: NEW H1-best
    new_best = new_top3[0]
    gate_h2 = new_best['H2']['sharpe'] > 3.5
    gate_full = new_best['FULL']['sharpe'] > 2.0
    gate_fold = new_best['fold_mean'] > 1.7
    gate_pass = bool(gate_h2 and gate_full and gate_fold)
    decision = 'PROMOTE_E4' if gate_pass else 'KEEP_AA'
    # also keep AA winner reference: AA h1best is sth0.10/etc15/trx9 vt0.012? verify matches AA file
    # load AA json to compare
    try:
        aa_json=json.load(open('results/backtest_AA.json'))
        aa_h1best_ref=aa_json.get('h1_best',{})
    except:
        aa_h1best_ref={}

    challenger=None
    if gate_pass:
        challenger={'sth':new_best['sth'],'etc_cd':new_best['etc_cd'],'trx_cd':new_best['trx_cd'],'vt':new_best['vt'],'H1':new_best['H1'],'H2':new_best['H2'],'FULL':new_best['FULL'],'fold_mean':new_best['fold_mean'],'fold_median':new_best['fold_median'],'fold_min':new_best['fold_min']}
    log(f'Gate NEW H1-best H2>3.5? {new_best["H2"]["sharpe"]:.3f} => {gate_h2} | FULL>2.0? {new_best["FULL"]["sharpe"]:.3f} => {gate_full} | fold_mean>1.7? {new_best["fold_mean"]:.3f} => {gate_fold} => {"PASS" if gate_pass else "FAIL"} decision={decision}')

    # 5 recommendations: new top3 + aa top + best FULL among new? or best H2?
    new_best_full = max(rows_new, key=lambda r: r['FULL']['sharpe'])
    new_best_h2 = max(rows_new, key=lambda r: r['H2']['sharpe'])
    recs=[]
    for r in new_top3:
        recs.append({'rank':'H1_top'+str(new_top3.index(r)+1),'sth':r['sth'],'etc_cd':r['etc_cd'],'trx_cd':r['trx_cd'],'vt':r['vt'],'H1':r['H1'],'H2':r['H2'],'FULL':r['FULL'],'fold_mean':r['fold_mean'],'B':r['B'],'C':r['C']})
    recs.append({'rank':'FULL_best_NEW','sth':new_best_full['sth'],'etc_cd':new_best_full['etc_cd'],'trx_cd':new_best_full['trx_cd'],'vt':new_best_full['vt'],'H1':new_best_full['H1'],'H2':new_best_full['H2'],'FULL':new_best_full['FULL'],'fold_mean':new_best_full['fold_mean']})
    recs.append({'rank':'H2_best_NEW','sth':new_best_h2['sth'],'etc_cd':new_best_h2['etc_cd'],'trx_cd':new_best_h2['trx_cd'],'vt':new_best_h2['vt'],'H1':new_best_h2['H1'],'H2':new_best_h2['H2'],'FULL':new_best_h2['FULL'],'fold_mean':new_best_h2['fold_mean']})

    res={
        'config':{
            'formula':FORMULA,
            'engine':'mirror run_aa.py leg_series + quantile_mask_long(q0.3 long-only) + _vol_scale post-stops pre-roll + roll1',
            'fixed':{'lth_etc':0.88,'lth_trx':0.85,'sl_etc':None,'sl_trx':0.05,'ts':TS,'vw':VW,'q':Q,'fund':FUND,'fee':BASE_FEE,'weights':[0.5,0.5],'venue':'aster','lev':2.0},
            'segments':{'H1':[5584,6077],'H2':[6077,6570],'B':[6380,6580],'C':[6080,6580],'FULL':[0,6580]},
            'fold_bounds': fold_bounds,
            'grid_new':'sth[0.09,0.10]x etc[12,15,18]x vt[0.008,0.012]=12 + sth[0.11,0.12]x etc[15,18]x vt[None,0.015]=8 =>20 new trx_cd=9 fixed',
            'grid_aa':'sth[0.10,0.11,0.12]x etc[15,18,21]x trx[6,9]=18 + sth0.09x2 ctrl=2 =>20 recalc vt0.012',
            'bars':'15m x16 ->4h n=6580',
        },
        'rows_new': rows_new,
        'rows_aa_recalc': rows_aa,
        'new_top3': [{'idx':r['idx'],'sth':r['sth'],'etc_cd':r['etc_cd'],'trx_cd':r['trx_cd'],'vt':r['vt'],'H1':r['H1'],'H2':r['H2'],'H2_fee2x':r['H2_fee2x'],'B':r['B'],'B_fee2x':r['B_fee2x'],'C':r['C'],'C_fee2x':r['C_fee2x'],'FULL':r['FULL'],'fold_mean':r['fold_mean'],'fold_median':r['fold_median'],'fold_min':r['fold_min'],'fold_max':r['fold_max'],'fold_n_pos':r['fold_n_pos'],'corr':r['corr_ETC_TRX_H2'],'turnover_H1':r['H1']['turnover'],'turnover_H2':r['H2']['turnover']} for r in new_top3],
        'aa_h1_best': {'idx':aa_top['idx'],'sth':aa_top['sth'],'etc_cd':aa_top['etc_cd'],'trx_cd':aa_top['trx_cd'],'vt':aa_top['vt'],'H1':aa_top['H1'],'H2':aa_top['H2'],'FULL':aa_top['FULL'],'fold_mean':aa_top['fold_mean']},
        'spearman': {'new_20': spear_new, 'aa_20': spear_aa, 'combined_40': spear_combined},
        'promotion_gate': {'thresholds': 'H2>3.5 and FULL>2.0 and fold_mean>1.7','new_h1_best': {'sth':new_best['sth'],'etc_cd':new_best['etc_cd'],'trx_cd':new_best['trx_cd'],'vt':new_best['vt'],'H1_sharpe':new_best['H1']['sharpe'],'H2_sharpe':new_best['H2']['sharpe'],'FULL_sharpe':new_best['FULL']['sharpe'],'fold_mean':new_best['fold_mean'],'fold_median':new_best['fold_median']}, 'checks': {'H2_gt_3.5': gate_h2,'FULL_gt_2.0': gate_full,'fold_gt_1.7': gate_fold}, 'PASS': gate_pass, 'decision': decision, 'challenger': challenger, 'keep_AA_ref': aa_h1best_ref},
        'recommendations_5': recs,
        'w4_warning': 'H1->H2 negative correlation repeats W4 if spearman <0',
    }
    open('results/backtest_E4.json','w').write(json.dumps(res,indent=1,ensure_ascii=False))
    log(f'saved results/backtest_E4.json decision={decision} new_best vt={new_best["vt"]} spear_new={spear_new:.3f} spear_combined={spear_combined:.3f}')

if __name__=='__main__':
    main()
