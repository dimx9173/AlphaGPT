import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json, math, csv, random
import torch
from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest
from model_core.vocab import FORMULA_VOCAB

BASE_FORMULA = [3,2,7,2,7,11,15,4,4,6,6,10]
COINS_BASE = ['ETC','TRX']
COINS_X = ['AVAX','SHIB']
BASE_FEE = 0.0004
FEE2X = 0.0008
FUND = 0.0005
Q = 0.3
# Y1b = ETC(0.88/0.12/cd18/None/ts24)+TRX(0.85/0.12/cd6/0.05/ts24), q0.3, 50/50, aster 2x fund0.0005 fee0.0004
BASE_SPEC = {
    'ETC': (0.88, 0.12, 18, None, 24, 'both'),
    'TRX': (0.85, 0.12, 6, 0.05, 24, 'both'),
}
SEGS = {'H1': (5584,6077), 'H2': (6077,6570), 'B': (6380,6580), 'C': (6080,6580), 'FULL': (0,6580)}
VOCAB_SIZE = FORMULA_VOCAB.size  # 23

def load_bars(coin):
    rows = list(csv.DictReader(open(f'data/data_15m_3y/{coin}.csv')))
    bars=[]
    for i in range(0,len(rows),16):
        blk=rows[i:i+16]
        if len(blk)<16:
            break
        bars.append((float(blk[0]['open']), max(float(x['high']) for x in blk), min(float(x['low']) for x in blk), float(blk[-1]['close']), sum(float(x['volume']) for x in blk)))
    return bars

def build_mats(bars, formula):
    n=len(bars)
    raw={'open':torch.tensor([[b[0] for b in bars]]),
         'high':torch.tensor([[b[1] for b in bars]]),
         'low':torch.tensor([[b[2] for b in bars]]),
         'close':torch.tensor([[b[3] for b in bars]]),
         'volume':torch.tensor([[b[4] for b in bars]]),
         'liquidity':torch.full((1,n),1e7),
         'fdv':torch.full((1,n),1e8)}
    sig=StackVM().execute(formula, FeatureEngineer.compute_features(raw))
    rets=[(bars[i+1][3]-bars[i][3])/bars[i][3] for i in range(n-1)] + [0.0]
    return raw, torch.tensor([rets]), sig

def quantile_mask_long(sig,q):
    if q is None:
        return None
    a=sig.detach().float().abs().reshape(-1)
    k=max(1,int(len(a)*float(q)))
    thr=torch.topk(a,k).values.min()
    return (sig.detach().float().abs() >= thr).float()

def leg_series(raw, rets_t, sig, lth, sth, cd, sl, fee=None, fund=None, side='both', q=Q, ts=24, tp=None):
    kw=dict(venue='aster', leverage=2.0, short_enabled=True, long_th=lth, short_th=sth, cooldown_bars=cd, bars_per_year=2190.0, stop_loss=sl, time_stop=ts, take_profit=tp)
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
    if n==0:
        return {'sharpe':0.0,'ann':0.0,'mdd':0.0,'cum':0.0,'n':0,'trades':trades,'turnover':round(turnover,6)}
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

def decode_formula(formula):
    names = FORMULA_VOCAB.token_names
    return [names[i] if 0 <= i < len(names) else f"UNK{i}" for i in formula]

def eval_combo(mats_seg, coins, spec_map, fee, tp=None):
    legs,tr,tos=[],[],[]
    by_coin={}
    for c in coins:
        raw,rt,sg=mats_seg[c]
        lth,sth,cd,sl,ts,side=spec_map[c]
        net,t,to=leg_series(raw,rt,sg,lth,sth,cd,sl,fee=fee,side=side,q=Q,ts=ts,tp=tp)
        legs.append(net)
        tr.append(t)
        tos.append(to)
        by_coin[c]={'net':net,'trades':t,'turnover':to}
    cb=combo(legs,[1.0/len(legs)]*len(legs))
    s=stats(cb,trades=sum(tr),turnover=sum(tos)/len(tos) if tos else 0)
    s['trades_by']={c:tr[i] for i,c in enumerate(coins)}
    s['turnover_by']={c:round(tos[i],6) for i,c in enumerate(coins)}
    return s, by_coin

def per_coin_side_split(mats_seg, coin, spec_entry, fee=BASE_FEE, tp=None):
    raw,rt,sg=mats_seg[coin]
    lth,sth,cd,sl,ts,side=spec_entry
    out={}
    for sd in ['both','long','short']:
        net,t,to=leg_series(raw,rt,sg,lth,sth,cd,sl,fee=fee,side=sd,q=Q,ts=ts,tp=tp)
        out[sd]=stats(net,trades=t,turnover=to)
    return out

# --- AD1 candidate generation ---
def _is_valid_formula(formula):
    # quick VM check on dummy features
    try:
        dummy=torch.randn(1, FORMULA_VOCAB.feature_count, 100)
        sig=StackVM().execute(formula, dummy)
        return sig is not None
    except Exception:
        return False

def gen_ad1_candidates(seed=42):
    random.seed(seed)
    base=BASE_FORMULA[:]
    cands=[]
    # 5 single-point valid mutations (loop until valid)
    attempts=0
    while len([c for c in cands if c['kind']=='single'])<5 and attempts<500:
        attempts+=1
        f=base[:]
        pos=random.randrange(len(f))
        old=f[pos]
        new=random.randrange(VOCAB_SIZE)
        if new==old:
            continue
        f[pos]=new
        if not _is_valid_formula(f):
            continue
        cands.append({'formula':f[:], 'kind':'single','pos':pos,'old':old,'new':new,'decode':decode_formula(f)})
    # 5 fully random valid
    attempts=0
    while len([c for c in cands if c['kind']=='random'])<5 and attempts<2000:
        attempts+=1
        f=[random.randrange(VOCAB_SIZE) for _ in range(len(base))]
        if not _is_valid_formula(f):
            continue
        cands.append({'formula':f[:], 'kind':'random','decode':decode_formula(f)})
    # If still not enough valid, fill with remaining invalid but flagged
    while len([c for c in cands if c['kind']=='single'])<5:
        f=base[:]
        pos=random.randrange(len(f))
        old=f[pos]
        new=random.randrange(VOCAB_SIZE)
        if new==old:
            new=(new+1)%VOCAB_SIZE
        f[pos]=new
        cands.append({'formula':f[:], 'kind':'single','pos':pos,'old':old,'new':new,'decode':decode_formula(f),'valid':_is_valid_formula(f)})
    while len([c for c in cands if c['kind']=='random'])<5:
        f=[random.randrange(VOCAB_SIZE) for _ in range(len(base))]
        cands.append({'formula':f[:], 'kind':'random','decode':decode_formula(f),'valid':_is_valid_formula(f)})
    return cands[:10]

def folds_12(mats_full, coins, spec_map, fee=BASE_FEE, tp=None):
    n=mats_full[coins[0]][0]['open'].shape[1]
    fold_n=n//12
    out=[]
    for i in range(12):
        a=i*fold_n
        b=(i+1)*fold_n if i<11 else n
        # slice per coin
        sliced={}
        for c in coins:
            raw,rt,sg=mats_full[c]
            slc=lambda t: t[:,a:b] if hasattr(t,'shape') and len(t.shape)==2 and t.shape[1]==n else t
            # need to re-slice mats correctly: raw dict values, rets, sig
            raw_s={k: (v[:,a:b] if hasattr(v,'shape') and len(v.shape)==2 and v.shape[1]==n else v) for k,v in raw.items()}
            rt_s=rt[:,a:b]
            sg_s=sg[:,a:b]
            sliced[c]=(raw_s,rt_s,sg_s)
        s,_=eval_combo(sliced, coins, spec_map, fee=fee, tp=tp)
        s['fold']=i
        s['bars']=[a,b]
        out.append(s)
    return out

def main():
    log_lines=[]
    def log(msg):
        print(msg,flush=True)
        log_lines.append(msg)
    full_bars={c:load_bars(c) for c in COINS_BASE + COINS_X}
    n_by={c:len(full_bars[c]) for c in full_bars}
    n=min(n_by.values())
    log(f'bars n per coin: {n_by} min={n}')
    # Allow 6570 for alt coins, but ETC/TRX must be 6580
    assert full_bars['ETC'].__len__()==6580 and full_bars['TRX'].__len__()==6580, f'ETC/TRX must be 6580 got ETC={len(full_bars["ETC"])} TRX={len(full_bars["TRX"])}'
    # also need per-coin mats per formula candidate - built inside loop
    # For baseline Y1b mats with BASE_FORMULA
    base_mats_full={c: build_mats(full_bars[c], BASE_FORMULA) for c in COINS_BASE + COINS_X}
    mats_by_seg_base={}
    for seg,(a,b) in SEGS.items():
        mats_by_seg_base[seg]={c: ( {k:(v[:,a:b] if hasattr(v,'shape') and len(v.shape)==2 and v.shape[1]==full_bars[c].__len__() or True else v) for k,v in base_mats_full[c][0].items()}, base_mats_full[c][1][:,a:b], base_mats_full[c][2][:,a:b]) for c in COINS_BASE + COINS_X}
        # above slicing uses mats_full n, simpler: rebuild from sliced bars
        # redo with bar slices for correctness
    # Rebuild seg mats from bar slices to avoid edge slice issues
    seg_mats_base={}
    for seg,(a,b) in SEGS.items():
        seg_mats_base[seg]={}
        for c in COINS_BASE + COINS_X:
            sl=full_bars[c][a:b]
            seg_mats_base[seg][c]=build_mats(sl, BASE_FORMULA)

    # ============ AD1 ============
    cands=gen_ad1_candidates(seed=42)
    log(f'AD1 candidates {len(cands)} (5 single +5 random) seed42')
    for i,cd in enumerate(cands):
        log(f' AD1 #{i} [{cd["kind"]}] {cd["formula"]} decode={cd["decode"]}')

    ad1_rows=[]
    for idx,cd in enumerate(cands):
        formula=cd['formula']
        # build per-seg mats for this formula (only need ETC+TRX)
        seg_mats={}
        for seg,(a,b) in SEGS.items():
            seg_mats[seg]={}
            for c in COINS_BASE:
                sl=full_bars[c][a:b]
                seg_mats[seg][c]=build_mats(sl, formula)
        # H1 ranking (unbiased)
        h1,_=eval_combo(seg_mats['H1'], COINS_BASE, BASE_SPEC, fee=BASE_FEE)
        h2,_=eval_combo(seg_mats['H2'], COINS_BASE, BASE_SPEC, fee=BASE_FEE)
        bseg,_=eval_combo(seg_mats['B'], COINS_BASE, BASE_SPEC, fee=BASE_FEE)
        cseg,_=eval_combo(seg_mats['C'], COINS_BASE, BASE_SPEC, fee=BASE_FEE)
        full,_=eval_combo(seg_mats['FULL'], COINS_BASE, BASE_SPEC, fee=BASE_FEE)
        # fee2x and extra
        h2_fee,_=eval_combo(seg_mats['H2'], COINS_BASE, BASE_SPEC, fee=FEE2X)
        b_fee,_=eval_combo(seg_mats['B'], COINS_BASE, BASE_SPEC, fee=FEE2X)
        c_fee,_=eval_combo(seg_mats['C'], COINS_BASE, BASE_SPEC, fee=FEE2X)
        full_fee,_=eval_combo(seg_mats['FULL'], COINS_BASE, BASE_SPEC, fee=FEE2X)
        h1_sh_side={}
        h2_sh_side={}
        # per-coin side split for H2 and FULL (for report)
        full_side={}
        h2_side={}
        for c in COINS_BASE:
            full_side[c]=per_coin_side_split(seg_mats['FULL'], c, BASE_SPEC[c], fee=BASE_FEE)
            h2_side[c]=per_coin_side_split(seg_mats['H2'], c, BASE_SPEC[c], fee=BASE_FEE)
        # 12-fold on FULL for reference
        # build full mats for folds
        full_mats={c: build_mats(full_bars[c], formula) for c in COINS_BASE}
        folds=folds_12(full_mats, COINS_BASE, BASE_SPEC, fee=BASE_FEE)

        row={'idx':idx,'kind':cd['kind'],'formula':formula,'decode':cd['decode'],
             'H1':h1,'H2':h2,'B':bseg,'C':cseg,'FULL':full,
             'H2_fee2x':h2_fee,'B_fee2x':b_fee,'C_fee2x':c_fee,'FULL_fee2x':full_fee,
             'per_coin_FULL_side':full_side,'per_coin_H2_side':h2_side,
             'folds':folds}
        if cd['kind']=='single':
            row['mut']={'pos':cd['pos'],'old':cd['old'],'new':cd['new']}
        ad1_rows.append(row)
        log(f'AD1 #{idx} {cd["kind"]} H1 sh={h1["sharpe"]:.3f} ann={h1["ann"]:.4f} mdd={h1["mdd"]:.4f} tr={h1["trades"]} to={h1["turnover"]:.4f} | H2 sh={h2["sharpe"]:.3f} tr={h2["trades"]} | B sh={bseg["sharpe"]:.3f} | C sh={cseg["sharpe"]:.3f} | FULL sh={full["sharpe"]:.3f} mdd={full["mdd"]:.4f} tr={full["trades"]}')

    # rank by H1 sharpe for selection
    ad1_ranked=sorted(ad1_rows, key=lambda r: r['H1']['sharpe'], reverse=True)
    best_h1=ad1_ranked[0]
    log(f'AD1 H1-best #{best_h1["idx"]} kind={best_h1["kind"]} formula={best_h1["formula"]} H1={best_h1["H1"]["sharpe"]:.3f} -> H2={best_h1["H2"]["sharpe"]:.3f} FULL={best_h1["FULL"]["sharpe"]:.3f}')

    # baseline row for comparison (BASE_FORMULA on same segs)
    baseline_full_mats={c: build_mats(full_bars[c], BASE_FORMULA) for c in COINS_BASE}
    baseline_folds=folds_12(baseline_full_mats, COINS_BASE, BASE_SPEC, fee=BASE_FEE)
    baseline_row={}
    for seg in ['H1','H2','B','C','FULL']:
        s,_=eval_combo(seg_mats_base[seg], COINS_BASE, BASE_SPEC, fee=BASE_FEE)
        baseline_row[seg]=s
    for seg in ['H2','B','C','FULL']:
        s,_=eval_combo(seg_mats_base[seg], COINS_BASE, BASE_SPEC, fee=FEE2X)
        baseline_row[seg+'_fee2x']=s
    baseline_sides={}
    for c in COINS_BASE:
        baseline_sides[c+'_FULL']=per_coin_side_split(seg_mats_base['FULL'], c, BASE_SPEC[c])
        baseline_sides[c+'_H2']=per_coin_side_split(seg_mats_base['H2'], c, BASE_SPEC[c])
    baseline_row['folds']=baseline_folds
    baseline_row['formula']=BASE_FORMULA
    baseline_row['decode']=decode_formula(BASE_FORMULA)
    log(f'BASELINE H1={baseline_row["H1"]["sharpe"]:.3f} H2={baseline_row["H2"]["sharpe"]:.3f} B={baseline_row["B"]["sharpe"]:.3f} C={baseline_row["C"]["sharpe"]:.3f} FULL={baseline_row["FULL"]["sharpe"]:.3f} H2_fee2x={baseline_row["H2_fee2x"]["sharpe"]:.3f}')

    # ============ AD2 take-profit on Y1b baseline ============
    TP_GRID=[None, 0.06, 0.10, 0.15]
    ad2_rows=[]
    for tp in TP_GRID:
        h2,_=eval_combo(seg_mats_base['H2'], COINS_BASE, BASE_SPEC, fee=BASE_FEE, tp=tp)
        bseg,_=eval_combo(seg_mats_base['B'], COINS_BASE, BASE_SPEC, fee=BASE_FEE, tp=tp)
        cseg,_=eval_combo(seg_mats_base['C'], COINS_BASE, BASE_SPEC, fee=BASE_FEE, tp=tp)
        full,_=eval_combo(seg_mats_base['FULL'], COINS_BASE, BASE_SPEC, fee=BASE_FEE, tp=tp)
        h2_fee,_=eval_combo(seg_mats_base['H2'], COINS_BASE, BASE_SPEC, fee=FEE2X, tp=tp)
        b_fee,_=eval_combo(seg_mats_base['B'], COINS_BASE, BASE_SPEC, fee=FEE2X, tp=tp)
        c_fee,_=eval_combo(seg_mats_base['C'], COINS_BASE, BASE_SPEC, fee=FEE2X, tp=tp)
        full_fee,_=eval_combo(seg_mats_base['FULL'], COINS_BASE, BASE_SPEC, fee=FEE2X, tp=tp)
        h1,_=eval_combo(seg_mats_base['H1'], COINS_BASE, BASE_SPEC, fee=BASE_FEE, tp=tp)
        folds=folds_12(baseline_full_mats, COINS_BASE, BASE_SPEC, fee=BASE_FEE, tp=tp)
        tp_label='None' if tp is None else str(tp)
        # per-coin side for FULL/H2
        per_coin={}
        for c in COINS_BASE:
            per_coin[c+'_FULL']=per_coin_side_split(seg_mats_base['FULL'], c, BASE_SPEC[c], tp=tp)
            per_coin[c+'_H2']=per_coin_side_split(seg_mats_base['H2'], c, BASE_SPEC[c], tp=tp)
        row={'take_profit':tp_label,'take_profit_raw':tp,'H1':h1,'H2':h2,'B':bseg,'C':cseg,'FULL':full,'H2_fee2x':h2_fee,'B_fee2x':b_fee,'C_fee2x':c_fee,'FULL_fee2x':full_fee,'folds':folds,'per_coin':per_coin}
        ad2_rows.append(row)
        log(f'AD2 TP={tp_label} H1={h1["sharpe"]:.3f} H2={h2["sharpe"]:.3f} FULL={full["sharpe"]:.3f} B={bseg["sharpe"]:.3f} C={cseg["sharpe"]:.3f} H2_fee2x={h2_fee["sharpe"]:.3f} to H2={h2["turnover"]:.4f} FULL to={full["turnover"]:.4f} folds mean={sum(f["sharpe"] for f in folds)/len(folds):.3f}')

    # AD2 gain check: TP must improve both H2 and FULL sharpe vs None and turnover not +>0.02
    base_tp_row=next(r for r in ad2_rows if r['take_profit']=='None')
    ad2_gain=[]
    for r in ad2_rows:
        if r['take_profit']=='None':
            continue
        tp=r['take_profit']
        d_h2=r['H2']['sharpe']-base_tp_row['H2']['sharpe']
        d_full=r['FULL']['sharpe']-base_tp_row['FULL']['sharpe']
        d_to_h2=r['H2']['turnover']-base_tp_row['H2']['turnover']
        d_to_full=r['FULL']['turnover']-base_tp_row['FULL']['turnover']
        d_to=max(d_to_h2,d_to_full)
        both_up=bool(d_h2>0 and d_full>0)
        to_ok=bool(d_to <= 0.02)
        gain=bool(both_up and to_ok)
        ad2_gain.append({'take_profit':tp,'d_H2':round(d_h2,3),'d_FULL':round(d_full,3),'d_turnover_max':round(d_to,4),'both_up':both_up,'turnover_ok':to_ok,'gain':gain})
        log(f'AD2 gain TP={tp} dH2={d_h2:.3f} dFULL={d_full:.3f} dto={d_to:.4f} both_up={both_up} to_ok={to_ok} GAIN={gain}')

    # ============ AD3 cross-coin single leg ============
    ad3_rows=[]
    for coin in COINS_X:
        # reuse Y1b same params but single coin leg: need spec for that coin -> mirror TRX? Task says Y1b 同參 + q0.3 in 雙腿單幣上跑 H2/B/C
        #Interpretation: run single-coin backtest with Y1b params - for AVAX/SHIB use TRX-like? We'll map: use TRX spec (0.85/0.12/cd6/0.05/ts24) as template, but task says "Y1b 同參" -> apply Y1b per-leg thresholds? For single coin, we test both ETC-spec and TRX-spec? Spec says "各用 Y1b 同參 + q0.3 在雙腿單幣上跑 H2/B/C (2行)" - ambiguous. Choose: use baseline single-leg with coin's own spec = TRX spec for altcoins (more generic). Report both interpretations via TRX template.
        # We'll also try ETC template for contrast but primary is TRX template.
        # For this task, use TRX spec for both AVAX/SHIB: (0.85,0.12,6,0.05,24,'both')
        spec_trx_for_coin={'AVAX':(0.85,0.12,6,0.05,24,'both'), 'SHIB':(0.85,0.12,6,0.05,24,'both')}[coin]
        spec_map={coin: spec_trx_for_coin}
        # need single coin combo (1 leg)
        for seg in ['H2','B','C']:
            pass
        # build seg mats for this coin with BASE_FORMULA
        coin_full_mats=build_mats(full_bars[coin], BASE_FORMULA)
        # use same leg_series but single coin
        segs_to_eval=['H2','B','C']
        res_seg={}
        for seg in segs_to_eval:
            a,b=SEGS[seg]
            raw,rt,sg=build_mats(full_bars[coin][a:b], BASE_FORMULA)
            mats_single={coin:(raw,rt,sg)}
            s,_=eval_combo(mats_single, [coin], spec_map, fee=BASE_FEE)
            s_fee,_=eval_combo(mats_single, [coin], spec_map, fee=FEE2X)
            side=per_coin_side_split(mats_single, coin, spec_map[coin])
            res_seg[seg]=s
            res_seg[seg+'_fee2x']=s_fee
            res_seg[seg+'_side']=side
        # 12-fold on full for this coin
        folds_single=folds_12({coin: coin_full_mats}, [coin], spec_map, fee=BASE_FEE)
        row={'coin':coin,'spec':list(spec_map[coin]),'q':Q,'H2':res_seg['H2'],'H2_fee2x':res_seg['H2_fee2x'],'H2_side':res_seg['H2_side'],'B':res_seg['B'],'B_fee2x':res_seg['B_fee2x'],'B_side':res_seg['B_side'],'C':res_seg['C'],'C_fee2x':res_seg['C_fee2x'],'C_side':res_seg['C_side'],'folds':folds_single,'folds_summary':{'mean':round(sum(f['sharpe'] for f in folds_single)/len(folds_single),3),'min':round(min(f['sharpe'] for f in folds_single),3),'n_pos':sum(1 for f in folds_single if f['sharpe']>0)}}
        ad3_rows.append(row)
        log(f'AD3 {coin} H2={row["H2"]["sharpe"]:.3f} fee2x={row["H2_fee2x"]["sharpe"]:.3f} B={row["B"]["sharpe"]:.3f} C={row["C"]["sharpe"]:.3f} folds mean={row["folds_summary"]["mean"]} min={row["folds_summary"]["min"]} npos={row["folds_summary"]["n_pos"]} trH2={row["H2"]["trades"]}')

    # Total rows 10+4+2=16 check
    total_rows=len(ad1_rows)+len(ad2_rows)+len(ad3_rows)
    assert total_rows==16, total_rows

    # PASS logic for AD
    # AD1: did H1-best beat baseline on H2 and FULL?
    ad1_beats_h2 = best_h1['H2']['sharpe'] > baseline_row['H2']['sharpe']
    ad1_beats_full = best_h1['FULL']['sharpe'] > baseline_row['FULL']['sharpe']
    ad1_gain = bool(ad1_beats_h2 and ad1_beats_full)
    # AD2: any TP gain?
    ad2_any_gain = any(g['gain'] for g in ad2_gain)
    # AD3: any coin with H2>1.5 and C>1.0 as candidate third leg?
    ad3_candidates=[r for r in ad3_rows if r['H2']['sharpe']>1.5 and r['C']['sharpe']>1.0]

    PASS={'AD1_H1_best_beats_baseline_H2': bool(ad1_beats_h2),'AD1_H1_best_beats_baseline_FULL': bool(ad1_beats_full),'AD1_gain': bool(ad1_gain),'AD2_any_TP_gain': bool(ad2_any_gain),'AD3_candidates': [r['coin'] for r in ad3_candidates],'overall_AD1_found_better': bool(ad1_gain),'overall_TP_effective': bool(ad2_any_gain),'overall_third_leg': bool(len(ad3_candidates)>0)}

    res={
        'config':{
            'base_formula': BASE_FORMULA,
            'base_decode': decode_formula(BASE_FORMULA),
            'base_spec': {k:list(v) for k,v in BASE_SPEC.items()},
            'coins_base': COINS_BASE,
            'coins_x': COINS_X,
            'q': Q,
            'venue':'aster','lev':2.0,'fund':FUND,'fee':BASE_FEE,'fee2x':FEE2X,
            'segments':{k:list(v) for k,v in SEGS.items()},
            'AD1':'5 single-point +5 random seed42, H1[5584:6077] rank, verify H2/B/C/FULL unbiased',
            'AD2':'take_profit [None,0.06,0.10,0.15] on Y1b baseline, H2/B/C/FULL+fee2x+turnover+12fold; gain requires H2&FULL sharpe up and dto<=0.02',
            'AD3':'AVAX/SHIB single leg Y1b TRX-spec (0.85/0.12/cd6/0.05/ts24) q0.3 H2/B/C',
            'note':'engine mirrors run_z1 leg_series+quantile+_vol_scale+stops; 16 rows total 10+4+2'
        },
        'baseline': baseline_row,
        'AD1':{'candidates_meta':[{'idx':c['idx'],'kind':c['kind'],'formula':c['formula'],'decode':c['decode'], 'mut':c.get('mut')} for c in ad1_rows],'rows':ad1_rows,'ranked_idx':[r['idx'] for r in ad1_ranked],'H1_best':{'idx':best_h1['idx'],'kind':best_h1['kind'],'formula':best_h1['formula'],'decode':best_h1['decode'],'H1':best_h1['H1'],'H2':best_h1['H2'],'B':best_h1['B'],'C':best_h1['C'],'FULL':best_h1['FULL'],'H2_fee2x':best_h1['H2_fee2x'],'FULL_fee2x':best_h1['FULL_fee2x']}},
        'AD2':{'rows':ad2_rows,'gain_check':ad2_gain},
        'AD3':{'rows':ad3_rows},
        'PASS':PASS,
        'total_rows':total_rows
    }
    open('results/backtest_AD.json','w').write(json.dumps(res,indent=1))
    open('logs/ad.log','w').write("\n".join(log_lines))
    print(f'saved results/backtest_AD.json total_rows={total_rows} AD1 H1-best #{best_h1["idx"]} H1={best_h1["H1"]["sharpe"]:.3f} H2={best_h1["H2"]["sharpe"]:.3f} FULL={best_h1["FULL"]["sharpe"]:.3f} baseline H2={baseline_row["H2"]["sharpe"]:.3f} FULL={baseline_row["FULL"]["sharpe"]:.3f} AD1_gain={ad1_gain} AD2_any_gain={ad2_any_gain} AD3_cands={[r["coin"] for r in ad3_candidates]}',flush=True)

if __name__=='__main__':
    main()
