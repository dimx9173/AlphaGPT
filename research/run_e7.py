
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json, math, csv, itertools
import torch
from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest

FORMULA = [3,2,7,2,7,11,15,4,4,6,6,10]
COINS = ['ETC','TRX']
BASE_FEE = 0.0004
FEE2X = 0.0008
FUND = 0.0005
# Base spec: Y1b = ETC(0.88/0.12/cd18/None/ts24)+TRX(0.85/0.12/cd6/0.05/ts24), both
BASE = {
    'ETC': dict(lth=0.88, sth=0.12, cd=18, sl=None, ts=24),
    'TRX': dict(lth=0.85, sth=0.12, cd=6, sl=0.05, ts=24),
}
# segment definitions 4h bars (6580 total from 15m x16)
SEGS = {'H2': (6077,6570), 'B': (6380,6580), 'C': (6080,6580), 'FULL': (0,6580)}

# 5 switches: q(quantile on long), sl(TRX sl only as designed), ts, vt, tp
# For ablation we toggle each atomically. For L9 we treat 4 dims (q, sl_TRX, ts, vt); tp included as 5th in all-open vs all-closed but orthogonal subset is 4 dims.
# Task spec: q[None,0.3 long], sl_TRX[None,0.05], ts[12,24], vt[None,0.012] -> L9 orthogonal subset 9 rows + all-open + all-closed =11
# tp [None,0.06] as 5th switch: included in full-open (0.06) vs full-closed (None); L9 rows use tp=None to isolate 4-factor orthogonality. tp marginal computed separately via single-factor delta.

# Orthogonal L9 subset: use standard L9 3^4 would be 9; but task says 3^2 approx with 4 dims 2 levels -> we need 9 diverse binary combos + 2 extremes.
# Design: enumerate all 16 combos then pick 9 max-min Hamming distance + add extremes.
# Simpler: hand-pick 9 that are orthogonal-ish covering each level balance and pair balance.

L9_HAND = [
    # q, sl_TRX, ts, vt, tp, label
    (0.3, 0.05, 24, 0.012, None, "L9-01"),  # will be all-open minus tp
    (0.3, 0.05, 12, None, None, "L9-02"),
    (0.3, None, 24, None, None, "L9-03"),
    (0.3, None, 12, 0.012, None, "L9-04"),
    (None, 0.05, 24, None, None, "L9-05"),
    (None, 0.05, 12, 0.012, None, "L9-06"),
    (None, None, 24, 0.012, None, "L9-07"),
    (None, None, 12, None, None, "L9-08"),
    (0.3, 0.05, 24, 0.012, 0.06, "L9-09_mix_tp"),  # tp on mid-point
]

ALL_OPEN = (0.3, 0.05, 24, 0.012, 0.06)
ALL_CLOSED = (None, None, 12, None, None)  # ts12 as closed (minimal hold), q/sl/vt/tp off; alternative ts24 off? Use 12 as minimal.

# For marginal analysis we need single-factor toggling vs ALL_CLOSED baseline (vtNone, qNone, slNone, tpNone, ts12)
# Factor order: q, sl, ts(24 vs 12), vt, tp
FACTOR_NAMES = ['q0.3_long','sl_TRX0.05','ts24','vt0.012','tp0.06']

def load_bars(coin):
    rows = list(csv.DictReader(open(f'data/data_15m_3y/{coin}.csv')))
    bars=[]
    for i in range(0,len(rows),16):
        blk=rows[i:i+16]
        if len(blk)<16: break
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

def quantile_mask_long(sig, q):
    if q is None: return None
    a=sig.detach().float().abs().reshape(-1)
    k=max(1,int(len(a)*float(q)))
    thr=torch.topk(a,k).values.min()
    return (sig.detach().float().abs() >= thr).float()

def leg_series(raw, rets_t, sig, lth, sth, cd, sl, fee, side='both', q=None, ts=0, vt=None, vw=24, tp=None, clamp=(0.2,2.0)):
    kw=dict(venue='aster', leverage=2.0, short_enabled=True, long_th=lth, short_th=sth, cooldown_bars=cd, bars_per_year=2190.0, stop_loss=sl, time_stop=ts, take_profit=tp, vol_target=vt, vol_window=vw)
    kw['funding_override']=FUND
    if fee is not None: kw['fee_override']=fee
    bt=MemeBacktest(**kw)
    # monkey-patch clamp if needed
    if clamp != (0.2,2.0) and vt is not None:
        orig_vol_scale = bt._vol_scale
        def patched_vol_scale(target_ret):
            if not bt.vol_target: return 1.0
            r = target_ret.detach().float()
            if r.dim()==1: r=r.unsqueeze(0)
            w=max(bt.vol_window,2)
            pad=r[:,:1].repeat(1,w-1)
            rp=torch.cat([pad,r],dim=1)
            win=rp.unfold(1,w,1)
            vol=win.std(dim=-1)+1e-9
            scale=(bt.vol_target/vol).clamp(clamp[0],clamp[1])
            scale=scale.roll(1,dims=1); scale[:,0]=1.0
            return scale
        bt._vol_scale = patched_vol_scale
    signal=torch.sigmoid(sig)
    is_safe=(raw['liquidity']>bt.min_liq).float()
    lp=(signal>bt.long_th).float()*is_safe
    sp=(signal<bt.short_th).float()*is_safe
    mask=quantile_mask_long(sig,q)
    if mask is not None: lp=lp*mask
    lp,sp=bt._apply_cooldown(lp,sp)
    lp,sp=bt._apply_stops(lp,sp, rets_t)
    scale=bt._vol_scale(rets_t)
    lp,sp=lp*scale, sp*scale
    if side=='long': sp=sp*0.0
    if side=='short': lp=lp*0.0
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
    if n==0: return {'sharpe':0.0,'ann':0.0,'mdd':0.0,'cum':0.0,'n':0,'trades':trades,'turnover':round(turnover,6)}
    mean=sum(ser)/n
    var=sum((x-mean)**2 for x in ser)/max(n-1,1) if n>1 else 0
    sharpe=mean/math.sqrt(var)*math.sqrt(2190.0) if var>0 else 0.0
    cum=sum(ser); ann=cum/n*2190.0 if n else 0
    cs,peak,mdd=0.0,-1e18,0.0
    for x in ser: cs+=x; peak=max(peak,cs); mdd=max(mdd,peak-cs)
    return {'sharpe':round(sharpe,3),'ann':round(ann,4),'mdd':round(mdd,4),'cum':round(cum,4),'n':n,'trades':trades,'turnover':round(turnover,6)}

def combo_series(legs):
    m=min(len(s) for s in legs); k=len(legs)
    return [sum(legs[i][t] for i in range(k))/k for t in range(m)]

def spec_for_config(q, sl_trx, ts, vt, tp):
    # ETC sl stays None always (baseline), TRX sl = sl_trx
    return {
        'ETC': dict(lth=BASE['ETC']['lth'], sth=BASE['ETC']['sth'], cd=BASE['ETC']['cd'], sl=None, ts=ts),
        'TRX': dict(lth=BASE['TRX']['lth'], sth=BASE['TRX']['sth'], cd=BASE['TRX']['cd'], sl=sl_trx, ts=ts),
    }, vt, tp, q

def eval_all_segments(mats_seg_dict, fee, q, sl_trx, ts, vt, tp, clamp=(0.2,2.0), vw=24):
    spec, vt_, tp_, q_ = spec_for_config(q, sl_trx, ts, vt, tp)
    out={}
    for seg in ['H2','B','C','FULL']:
        legs=[]; tr=[]; tos=[]
        for c in COINS:
            raw,rt,sg=mats_seg_dict[seg][c]
            cf=spec[c]
            net,t,to=leg_series(raw,rt,sg,cf['lth'],cf['sth'],cf['cd'],cf['sl'],fee=fee,side='both',q=q_,ts=cf['ts'],vt=vt_,vw=vw,tp=tp_,clamp=clamp)
            legs.append(net); tr.append(t); tos.append(to)
        cb=combo_series(legs)
        s=stats(cb,trades=sum(tr),turnover=sum(tos)/len(tos))
        s['trades_by']={c:tr[i] for i,c in enumerate(COINS)}
        s['turnover_by']={c:round(tos[i],6) for i,c in enumerate(COINS)}
        # fee2x counterpart for H2/B/C/FULL? compute on demand
        out[seg]=s
    # also compute fee2x for each seg (except FULL we compute anyway)
    fee2x_out={}
    for seg in ['H2','B','C','FULL']:
        legs=[]; tr=[]; tos=[]
        for c in COINS:
            raw,rt,sg=mats_seg_dict[seg][c]
            cf=spec[c]
            net,t,to=leg_series(raw,rt,sg,cf['lth'],cf['sth'],cf['cd'],cf['sl'],fee=FEE2X,side='both',q=q_,ts=cf['ts'],vt=vt_,vw=vw,tp=tp_,clamp=clamp)
            legs.append(net); tr.append(t); tos.append(to)
        cb=combo_series(legs)
        s=stats(cb,trades=sum(tr),turnover=sum(tos)/len(tos))
        fee2x_out[seg]=s
    # 12-fold mean on FULL fee=BASE_FEE
    # build full mats for folding
    return out, fee2x_out

def folds_12mean(full_bars, fee, q, sl_trx, ts, vt, tp, clamp=(0.2,2.0), vw=24):
    n = len(full_bars[COINS[0]])
    fold_n = n//12
    spec, vt_, tp_, q_ = spec_for_config(q, sl_trx, ts, vt, tp)
    sharpes=[]
    for i in range(12):
        a=i*fold_n; b=(i+1)*fold_n if i<11 else n
        legs=[]
        for c in COINS:
            sl = full_bars[c][a:b]
            raw,rt,sg = build_mats(sl)
            cf=spec[c]
            net,_,_=leg_series(raw,rt,sg,cf['lth'],cf['sth'],cf['cd'],cf['sl'],fee=fee,side='both',q=q_,ts=cf['ts'],vt=vt_,vw=vw,tp=tp_,clamp=clamp)
            legs.append(net)
        cb=combo_series(legs)
        s=stats(cb)
        sharpes.append(s['sharpe'])
    return round(sum(sharpes)/len(sharpes),3), sharpes

def main():
    import pathlib
    log_lines=[]
    def log(msg):
        print(msg,flush=True)
        log_lines.append(msg)
    full_bars={c: load_bars(c) for c in COINS}
    n=min(len(v) for v in full_bars.values())
    log(f"bars 4h n={n} ETC={len(full_bars['ETC'])} TRX={len(full_bars['TRX'])}")
    assert len(full_bars['ETC'])==6580 and len(full_bars['TRX'])==6580
    mats_seg={}
    for seg,(a,b) in SEGS.items():
        mats_seg[seg]={c: build_mats(full_bars[c][a:b]) for c in COINS}
        log(f"{seg} [{a}:{b}] built")
    # Build orthogonal rows
    ortho_configs=[]
    for tup in L9_HAND:
        q,sl,ts,vt,tp,label=tup
        ortho_configs.append((q,sl,ts,vt,tp,label))
    # add all-open and all-closed as separate rows
    ortho_configs.append((ALL_OPEN[0],ALL_OPEN[1],ALL_OPEN[2],ALL_OPEN[3],ALL_OPEN[4],"ALL_OPEN"))
    ortho_configs.append((ALL_CLOSED[0],ALL_CLOSED[1],ALL_CLOSED[2],ALL_CLOSED[3],ALL_CLOSED[4],"ALL_CLOSED"))
    log(f"orthogonal grid: {len(ortho_configs)} rows (9 L9 +2 extremes)")
    rows=[]
    for q,sl,ts,vt,tp,label in ortho_configs:
        segs, segs2x = eval_all_segments(mats_seg, BASE_FEE, q, sl, ts, vt, tp)
        m12, _ = folds_12mean(full_bars, BASE_FEE, q, sl, ts, vt, tp)
        row={'label':label,'config':{'q':q,'sl_TRX':sl,'ts':ts,'vt':vt,'tp':tp,
               'ETC':{'lth':0.88,'sth':0.12,'cd':18,'sl':None,'ts':ts},
               'TRX':{'lth':0.85,'sth':0.12,'cd':6,'sl':sl,'ts':ts}},
             'H2':segs['H2'],'B':segs['B'],'C':segs['C'],'FULL':segs['FULL'],
             'H2_fee2x':segs2x['H2'],'B_fee2x':segs2x['B'],'C_fee2x':segs2x['C'],'FULL_fee2x':segs2x['FULL'],
             'mean12':m12}
        # also fees decay
        row['fee_decay_H2']=round(segs['H2']['sharpe']-segs2x['H2']['sharpe'],3)
        row['fee_decay_B']=round(segs['B']['sharpe']-segs2x['B']['sharpe'],3)
        rows.append(row)
        log(f"{label}: q={q} sl={sl} ts={ts} vt={vt} tp={tp} | H2 sh={segs['H2']['sharpe']:.3f} to={segs['H2']['turnover']:.4f} B sh={segs['B']['sharpe']:.3f} C sh={segs['C']['sharpe']:.3f} FULL sh={segs['FULL']['sharpe']:.3f} fee2x_H2={segs2x['H2']['sharpe']:.3f} mean12={m12:.3f}")
    # === Marginal gains vs ALL_CLOSED on H2 sharpe ===
    closed = [r for r in rows if r['label']=='ALL_CLOSED'][0]
    base_sh = closed['H2']['sharpe']
    base_to = closed['H2']['turnover']
    # single-factor rows: toggle one factor on from closed
    singles = {
        'q0.3_long': (0.3, None, 12, None, None),
        'sl_TRX0.05': (None, 0.05, 12, None, None),
        'ts24': (None, None, 24, None, None),
        'vt0.012': (None, None, 12, 0.012, None),
        'tp0.06': (None, None, 12, None, 0.06),
    }
    marginals=[]
    for fname, cfg in singles.items():
        q,sl,ts,vt,tp=cfg
        segs, segs2x = eval_all_segments(mats_seg, BASE_FEE, q, sl, ts, vt, tp)
        m12,_=folds_12mean(full_bars, BASE_FEE, q, sl, ts, vt, tp)
        d_sh = round(segs['H2']['sharpe']-base_sh,3)
        d_to = round(segs['H2']['turnover']-base_to,3)
        gain = segs['H2']['sharpe']
        log(f"  marginal {fname}: H2 sh={gain:.3f} Δ={d_sh:+.3f} to={segs['H2']['turnover']:.4f} d_to={d_to:+.4f} mean12={m12:.3f}")
        marginals.append({'factor':fname,'cfg':{'q':q,'sl_TRX':sl,'ts':ts,'vt':vt,'tp':tp},
                          'H2_sharpe':segs['H2']['sharpe'],'H2_turnover':segs['H2']['turnover'],
                          'delta_sharpe_vs_closed':d_sh,'delta_turnover':d_to,'mean12':m12,
                          'H2':segs['H2'],'B':segs['B'],'C':segs['C'],'FULL':segs['FULL']})
    # also compute ALL_OPEN delta
    open_row=[r for r in rows if r['label']=='ALL_OPEN'][0]
    open_delta = round(open_row['H2']['sharpe']-base_sh,3)
    log(f"ALL_OPEN H2 Δ vs CLOSED = {open_delta:+.3f} (open {open_row['H2']['sharpe']:.3f} vs closed {base_sh:.3f})")
    # variance decomposition: portion of total variance explained by each factor (approx via single-factor deltas)
    total_open_gain = open_delta if open_delta!=0 else 1.0
    for m in marginals:
        m['variance_share'] = round(m['delta_sharpe_vs_closed']/total_open_gain,3) if total_open_gain!=0 else 0.0
    # Redundancy: delta <0.1 and delta_turnover >0.03
    redundant=[]
    for m in marginals:
        is_red = (m['delta_sharpe_vs_closed'] < 0.1) and (m['delta_turnover'] > 0.03)
        m['redundant'] = bool(is_red)
        if is_red: redundant.append(m['factor'])
        log(f"    -> {m['factor']}: redundant={is_red} (Δ={m['delta_sharpe_vs_closed']:.3f}, d_to={m['delta_turnover']:.3f})")
    # Pruned stack: keep non-redundant with positive gain, plus always keep at least best factor
    # Sort by delta desc, keep those not redundant and delta>0.05; if all redundant keep top 2
    kept = [m['factor'] for m in sorted(marginals, key=lambda x: x['delta_sharpe_vs_closed'], reverse=True) if not m['redundant'] and m['delta_sharpe_vs_closed']>0.05]
    if not kept:
        kept = [sorted(marginals, key=lambda x: x['delta_sharpe_vs_closed'], reverse=True)[0]['factor']]
    log(f"PRUNED stack kept: {kept}  redundant: {redundant}")
    # Build pruned config: apply kept factors on top of closed
    pruned_cfg = {'q':None,'sl':None,'ts':12,'vt':None,'tp':None}
    for f in kept:
        if f=='q0.3_long': pruned_cfg['q']=0.3
        elif f=='sl_TRX0.05': pruned_cfg['sl']=0.05
        elif f=='ts24': pruned_cfg['ts']=24
        elif f=='vt0.012': pruned_cfg['vt']=0.012
        elif f=='tp0.06': pruned_cfg['tp']=0.06
    pruned_segs, pruned_2x = eval_all_segments(mats_seg, BASE_FEE, pruned_cfg['q'], pruned_cfg['sl'], pruned_cfg['ts'], pruned_cfg['vt'], pruned_cfg['tp'])
    pruned_m12,_=folds_12mean(full_bars, BASE_FEE, pruned_cfg['q'], pruned_cfg['sl'], pruned_cfg['ts'], pruned_cfg['vt'], pruned_cfg['tp'])
    log(f"PRUNED eval: q={pruned_cfg['q']} sl={pruned_cfg['sl']} ts={pruned_cfg['ts']} vt={pruned_cfg['vt']} tp={pruned_cfg['tp']} | H2 {pruned_segs['H2']['sharpe']:.3f} B {pruned_segs['B']['sharpe']:.3f} C {pruned_segs['C']['sharpe']:.3f} FULL {pruned_segs['FULL']['sharpe']:.3f} fee2x_H2 {pruned_2x['H2']['sharpe']:.3f} mean12 {pruned_m12:.3f} to {pruned_segs['H2']['turnover']:.4f}")
    # === vol cautious variant: clamp 0.5-1.5 vs 0.2-2.0, vw 8/24 at vt0.012, keep other at ALL_OPEN-minus-one? Use open config with vt only to isolate vt effect, and also open full
    vt_variants=[]
    for clamp, vw, label in [((0.5,1.5),24,"vt0.012_clamp0.5-1.5_vw24"), ((0.2,2.0),8,"vt0.012_clamp0.2-2.0_vw8"), ((0.2,2.0),24,"vt0.012_clamp0.2-2.0_vw24_baseline")]:
        segs, segs2x = eval_all_segments(mats_seg, BASE_FEE, 0.3, 0.05, 24, 0.012, None, clamp=clamp, vw=vw)
        m12,_=folds_12mean(full_bars, BASE_FEE, 0.3, 0.05, 24, 0.012, None, clamp=clamp, vw=vw)
        vt_variants.append({'label':label,'clamp':list(clamp),'vw':vw,'H2':segs['H2'],'B':segs['B'],'C':segs['C'],'FULL':segs['FULL'],'H2_fee2x':segs2x['H2'],'mean12':m12})
        log(f"VT var {label}: clamp={clamp} vw={vw} H2 sh={segs['H2']['sharpe']:.3f} to={segs['H2']['turnover']:.4f} B={segs['B']['sharpe']:.3f} C={segs['C']['sharpe']:.3f} FULL={segs['FULL']['sharpe']:.3f} mean12={m12:.3f}")
    # Save JSON
    res={
        'config':{
            'formula':FORMULA,'formula_decode':['FOMO','PRESSURE','SUB','PRESSURE','SUB','ABS','DECAY','DEV','DEV','ADD','ADD','NEG'],
            'coins':COINS,'weights':'50/50','venue':'aster','lev':2.0,'fund':FUND,'fee':BASE_FEE,'fee2x':FEE2X,
            'base_spec':{'ETC':[0.88,0.12,18,None,24,'both'],'TRX':[0.85,0.12,6,0.05,24,'both']},
            'segments':SEGS,'engine':'leg_series + quantile_mask_long + _vol_scale post-stops pre-roll (mirrors run_ad/run_z1)',
            'grid':'L9( q[None,0.3] x sl_TRX[None,0.05] x ts[12,24] x vt[None,0.012]) hand-picked 9 orthogonal + ALL_OPEN(q0.3/sl0.05/ts24/vt0.012/tp0.06) + ALL_CLOSED(None/None/12/None/None)=11 rows; 50/50 combo; fee2x on all segments',
            'marginals_note':'Δ vs ALL_CLOSED on H2 sharpe; redundant if Δ<0.1 and Δturnover>0.03',
            'vt_variants_note':'vt0.012 caution clamp 0.5-1.5 vw24 vs vw8/24 baseline, other switches at Y1b(Z1) open-minus-tp to isolate vt',
        },
        'orthogonal_rows':rows,
        'marginals':marginals,
        'closed_H2_sharpe':base_sh,'closed_H2_turnover':base_to,
        'open_H2_sharpe':open_row['H2']['sharpe'],'open_delta':open_delta,
        'redundant':redundant,
        'pruned':{'kept_factors':kept,'cfg':pruned_cfg,'H2':pruned_segs['H2'],'B':pruned_segs['B'],'C':pruned_segs['C'],'FULL':pruned_segs['FULL'],'H2_fee2x':pruned_2x['H2'],'B_fee2x':pruned_2x['B'],'C_fee2x':pruned_2x['C'],'FULL_fee2x':pruned_2x['FULL'],'mean12':pruned_m12},
        'vt_variants':vt_variants,
        'PASS':{
            'orthogonal_rows':len(rows)==11,
            'marginals_computed':len(marginals)==5,
            'vt_variants':len(vt_variants)==3,
        }
    }
    open('results/backtest_E7.json','w').write(json.dumps(res,indent=2,ensure_ascii=False))
    open('logs/e7.log','w').write('\n'.join(log_lines))
    log(f"saved results/backtest_E7.json ({len(rows)} rows, {len(marginals)} marginals, {len(vt_variants)} vt variants)")
    log(f"PASS {json.dumps(res['PASS'])}")
    # also print summary table for doc
    log("MARGIN TABLE (H2 sharpe Δ vs ALL_CLOSED):")
    for m in sorted(marginals, key=lambda x: x['delta_sharpe_vs_closed'], reverse=True):
        log(f"  {m['factor']:12s}  Δ={m['delta_sharpe_vs_closed']:+6.3f}  d_to={m['delta_turnover']:+6.3f}  share={m['variance_share']*100:5.1f}%  redundant={m['redundant']}")

if __name__=='__main__':
    main()
