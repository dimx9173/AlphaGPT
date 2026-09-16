
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if os.path.dirname(os.path.abspath(__file__)) else ".")
import pathlib as _pl
root = _pl.Path(__file__).parent.parent if _pl.Path(__file__).parent.name=="research" else _pl.Path(".")
os.chdir(str(root))
import csv, math, json, itertools, pathlib
import torch
from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest

FORMULA = [3,2,7,2,7,11,15,4,4,6,6,10]
COINS = ['ETC','TRX']
FUND_BASE = 0.0005
FEE_BASE = 0.0004
FEE2X = 0.0008
FULL_N = 6580
FOLD_N = FULL_N // 12
Q = 0.3

# baseline specs for Y1b/Z1/AA-H1
BASELINES = {
    "Y1b": {"label":"Y1b_both_q03_vtNone", "vt":None, "vw":12, "sth":0.12, "etc_cd":18, "trx_cd":6, "ts":24, "side":"both"},
    "Z1":  {"label":"Z1_both_q03_vt012", "vt":0.012, "vw":12, "sth":0.12, "etc_cd":18, "trx_cd":6, "ts":24, "side":"both"},
    "AA-H1": {"label":"AA-H1_sth0.10_15_9_vt012", "vt":0.012, "vw":12, "sth":0.10, "etc_cd":15, "trx_cd":9, "ts":24, "side":"both"},
}
# also we need FULL baseline both reference for comparison with short specialist at H2

SEGS = {'H1':(5584,6077), 'H2':(6077,6570), 'B':(6380,6580), 'C':(6080,6580), 'FULL':(0,6580)}

def load_bars(coin):
    rows=list(csv.DictReader(open(f'data/data_15m_3y/{coin}.csv')))
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
         'liquidity':torch.full((1,n), 1e7),
         'fdv':torch.full((1,n), 1e8)}
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

def leg_series_with_pos(raw, rets_t, sig, lth, sth, cd, sl, ts, side, vt, vw, fee, fund, q):
    kw=dict(venue='aster', leverage=2.0, short_enabled=True, long_th=lth, short_th=sth, cooldown_bars=cd, bars_per_year=2190.0, stop_loss=sl, time_stop=ts, vol_target=vt, vol_window=vw)
    kw['funding_override']=FUND_BASE if fund is None else fund
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
    lp,sp=bt._apply_stops(lp,sp,rets_t)
    scale=bt._vol_scale(rets_t)
    lp = lp * scale
    sp = sp * scale
    if side=='long':
        sp=sp*0.0
    if side=='short':
        lp=lp*0.0
    lp=lp.roll(1,dims=1); lp[:,0]=0
    sp=sp.roll(1,dims=1); sp[:,0]=0
    turn=(lp - lp.roll(1,dims=1)).abs() + (sp - sp.roll(1,dims=1)).abs()
    turn_l=(lp - lp.roll(1,dims=1)).abs()
    turn_s=(sp - sp.roll(1,dims=1)).abs()
    tx=turn*(bt.base_fee+torch.clamp(bt.trade_size/(raw['liquidity']+1e-9),0.0,0.05))
    gross=(lp-sp)*rets_t*bt.leverage
    fnd=(lp-sp)*bt.default_funding_rate*bt.leverage
    net=(gross - tx*bt.leverage - fnd)[0].tolist()
    pos=(lp-sp)[0].tolist()
    lp_list=lp[0].tolist()
    sp_list=sp[0].tolist()
    turn_list=turn[0].tolist()
    turn_l_list=turn_l[0].tolist()
    turn_s_list=turn_s[0].tolist()
    trades=sum(1 for t in range(len(pos)) if pos[t]!=0.0 and (t==0 or pos[t-1]==0.0))
    turnover=sum(turn_list)/len(turn_list) if turn_list else 0.0
    long_turnover=sum(turn_l_list)/len(turn_l_list) if turn_l_list else 0.0
    short_turnover=sum(turn_s_list)/len(turn_s_list) if turn_s_list else 0.0
    pos_rate=sum(1 for v in pos if v!=0)/len(pos) if pos else 0
    long_pos_rate=sum(1 for v in lp_list if v!=0)/len(lp_list) if lp_list else 0
    short_pos_rate=sum(1 for v in sp_list if v!=0)/len(sp_list) if sp_list else 0
    return {'net':net,'pos':pos,'lp':lp_list,'sp':sp_list,'trades':trades,'turnover':turnover,'long_turnover':long_turnover,'short_turnover':short_turnover,'pos_rate':pos_rate,'long_pos_rate':long_pos_rate,'short_pos_rate':short_pos_rate,'turn':turn_list}

def stats(ser, trades=0, turnover=0.0, pos_rate=0.0, extra=None):
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
    out={'sharpe':round(sharpe,3),'ann':round(ann,4),'mdd':round(mdd,4),'cum':round(cum,4),'n':n,'trades':trades,'turnover':round(turnover,6),'pos_rate':round(pos_rate,4)}
    if extra:
        out.update(extra)
    return out

def combo(nets, weights=None):
    m=min(len(s) for s in nets)
    k=len(nets)
    w=weights or [1.0/k]*k
    sw=sum(w)
    return [sum(nets[i][t]*w[i]/sw for i in range(k)) for t in range(m)]

def eval_combo(bars_dict, spec_dict, fee, fund, q):
    legs=[]; metas=[]
    for c in COINS:
        lth, sth, cd, sl, ts, side, vt, vw = spec_dict[c]
        raw, rt, sg = build_mats(bars_dict[c])
        res = leg_series_with_pos(raw, rt, sg, lth, sth, cd, sl, ts, side, vt, vw, fee, fund, q)
        legs.append(res['net'])
        metas.append(res)
    cb = combo(legs, [0.5,0.5])
    avg_to = sum(m['turnover'] for m in metas)/len(metas)
    avg_pr = sum(m['pos_rate'] for m in metas)/len(metas)
    avg_lpr = sum(m['long_pos_rate'] for m in metas)/len(metas)
    avg_spr = sum(m['short_pos_rate'] for m in metas)/len(metas)
    avg_lto = sum(m['long_turnover'] for m in metas)/len(metas)
    avg_sto = sum(m['short_turnover'] for m in metas)/len(metas)
    s = stats(cb, trades=sum(m['trades'] for m in metas), turnover=avg_to, pos_rate=avg_pr, extra={'long_pos_rate':round(avg_lpr,4),'short_pos_rate':round(avg_spr,4),'long_turnover':round(avg_lto,6),'short_turnover':round(avg_sto,6),'trades_by':{COINS[i]:metas[i]['trades'] for i in range(len(COINS))}})
    return s, cb, metas

# Holding bucket decomposition on FULL combo net with per-trade attribution
def duration_bucket(d):
    if 1 <= d <= 6:
        return '1-6'
    elif 7 <= d <= 24:
        return '7-24'
    elif d >= 25:
        return '25+'
    else:
        return 'flat'

def decompose_buckets(metas, combo_net):
    # metas: per-leg dict with pos, net etc
    # We decompose at leg level then aggregate? For contribution we need per-bar net attribution weighted.
    # Simplify: use combo_net and combo position proxy for bucketing, then cum per bucket.
    # Combo position = weighted avg of pos lists (0.5 each)
    n = len(combo_net)
    pos0 = metas[0]['pos']
    pos1 = metas[1]['pos']
    cpos = [(pos0[t]*0.5 + pos1[t]*0.5) for t in range(n)]
    # Build trade episodes on cpos with sign-aware split
    # Identify episodes: contiguous non-zero cpos with same sign
    episodes = [] # list of (start, end, duration)
    i=0
    while i<n:
        if cpos[i]!=0:
            sign = 1 if cpos[i]>0 else -1
            j=i+1
            while j<n and cpos[j]!=0 and ((cpos[j]>0)==(sign>0)):
                j+=1
            # episode i..j-1
            episodes.append((i, j-1, j-i))
            i=j
        else:
            i+=1
    # Map each bar index to its episode bucket
    bar_to_bucket = {}
    bucket_trades = {'1-6':0,'7-24':0,'25+':0}
    for (s,e,d) in episodes:
        b = duration_bucket(d)
        bucket_trades[b]+=1
        for t in range(s,e+1):
            bar_to_bucket[t]=b
    # Aggregate cum per bucket from combo_net
    bucket_cum = {'1-6':0.0,'7-24':0.0,'25+':0.0,'flat':0.0}
    for t, v in enumerate(combo_net):
        b = bar_to_bucket.get(t, 'flat')
        bucket_cum[b]+=float(v)
    total_cum = sum(combo_net)
    # contribution ratios (guard zero)
    bucket_pct = {}
    for k in ['1-6','7-24','25+']:
        bucket_pct[k]= round(bucket_cum[k]/total_cum,4) if abs(total_cum)>1e-12 else 0.0
    # avg bars held per bucket
    # need avg duration per bucket
    bucket_avg_bars = {}
    for k in ['1-6','7-24','25+']:
        durs = [d for (s,e,d) in episodes if duration_bucket(d)==k]
        bucket_avg_bars[k]= round(sum(durs)/len(durs),2) if durs else 0.0
    # also per-leg bucket decomposition for reference (short vs long legs)
    per_leg_buckets={}
    for idx,c in enumerate(COINS):
        pos = metas[idx]['pos']
        net = metas[idx]['net']
        eps=[]
        i=0
        while i<len(pos):
            if pos[i]!=0:
                sign=1 if pos[i]>0 else -1
                j=i+1
                while j<len(pos) and pos[j]!=0 and ((pos[j]>0)==(sign>0)):
                    j+=1
                eps.append((i,j-1,j-i))
                i=j
            else:
                i+=1
        bar_bucket={}
        trades_leg={'1-6':0,'7-24':0,'25+':0}
        for (s,e,d) in eps:
            b=duration_bucket(d)
            trades_leg[b]+=1
            for t in range(s,e+1):
                bar_bucket[t]=b
        cum_leg={'1-6':0.0,'7-24':0.0,'25+':0.0,'flat':0.0}
        for t,v in enumerate(net):
            b=bar_bucket.get(t,'flat')
            cum_leg[b]+=float(v)
        per_leg_buckets[c]={'trades':trades_leg,'cum':{k:round(v,4) for k,v in cum_leg.items() if k in ['1-6','7-24','25+']}, 'avg_bars':{k: round(sum([d for (_,_,d) in eps if duration_bucket(d)==k])/max(1, sum(1 for (_,_,d) in eps if duration_bucket(d)==k)),2) if trades_leg[k]>0 else 0.0 for k in ['1-6','7-24','25+']}, 'spans':len(eps)}
    return {'combo_episodes':len(episodes),'bucket_trades':bucket_trades,'bucket_cum':{k:round(v,4) for k,v in bucket_cum.items() if k in ['1-6','7-24','25+']}, 'bucket_pct':bucket_pct, 'bucket_avg_bars':bucket_avg_bars, 'episodes':episodes[:10], 'per_leg':per_leg_buckets}

def main():
    log_lines=[]
    def log(m):
        print(m, flush=True)
        log_lines.append(m)
    full={c: load_bars(c) for c in COINS}
    n=min(len(v) for v in full.values())
    log(f"E6 full 4h bars n={n} FORMULA {FORMULA} COINS {COINS} fund {FUND_BASE} fee {FEE_BASE}/{FEE2X} Q {Q}")
    assert n==FULL_N, f"expected {FULL_N} got {n}"

    # === Part 2: duration bucket decomposition on FULL for 3 baselines ===
    baseline_buckets={}
    baseline_segments={}  # store FULL segment stats for reference
    # Build specs for 3 baselines
    def make_baseline_spec(name):
        cfg = BASELINES[name]
        spec={}
        # lth etc 0.88 sl None, trx 0.85 sl 0.05, side as cfg side, vt/vw, ts
        spec['ETC']=(0.88, cfg['sth'], cfg['etc_cd'], None, cfg['ts'], cfg['side'], cfg['vt'], cfg['vw'])
        spec['TRX']=(0.85, cfg['sth'], cfg['trx_cd'], 0.05, cfg['ts'], cfg['side'], cfg['vt'], cfg['vw'])
        return spec, cfg
    for name in ['Y1b','Z1','AA-H1']:
        spec,cfg = make_baseline_spec(name)
        bars_full={c: full[c] for c in COINS}
        s, cb, metas = eval_combo(bars_full, spec, fee=FEE_BASE, fund=FUND_BASE, q=Q)
        decomp = decompose_buckets(metas, cb)
        baseline_buckets[name]=decomp
        baseline_segments[name]=s
        log(f"\n=== BUCKET {name} {cfg['label']} vt={cfg['vt']} sth={cfg['sth']} etc_cd={cfg['etc_cd']} trx_cd={cfg['trx_cd']} ts={cfg['ts']} side={cfg['side']} ===")
        log(f"  FULL sh={s['sharpe']} ann={s['ann']} mdd={s['mdd']} cum={s['cum']} trades={s['trades']} turnover={s['turnover']} pos={s['pos_rate']} lpr={s['long_pos_rate']} spr={s['short_pos_rate']}")
        log(f"  BUCKET combo cum {decomp['bucket_cum']} pct {decomp['bucket_pct']} trades {decomp['bucket_trades']} avg_bars {decomp['bucket_avg_bars']} episodes {decomp['combo_episodes']}")
        for c in COINS:
            pl=decomp['per_leg'][c]
            log(f"    leg {c} trades {pl['trades']} cum {pl['cum']} avg_bars {pl['avg_bars']}")

    # === Part 3: short-only specialist grid 27 rows ===
    STH_GRID=[0.08,0.10,0.12]
    TRX_CD_GRID=[6,9,12]
    TS_GRID=[12,24,36]
    VT=0.012; VW=12
    LTH_ETC=0.88; LTH_TRX=0.85
    ETC_CD_FIXED=18
    grid=[]
    for sth in STH_GRID:
        for trx_cd in TRX_CD_GRID:
            for ts in TS_GRID:
                grid.append((sth, trx_cd, ts))
    assert len(grid)==27
    rows=[]
    for idx,(sth,trx_cd,ts) in enumerate(grid):
        spec={
            'ETC':(LTH_ETC, sth, ETC_CD_FIXED, None, ts, 'short', VT, VW),
            'TRX':(LTH_TRX, sth, trx_cd, 0.05, ts, 'short', VT, VW),
        }
        seg_results={}
        for seg_name,(a,b) in SEGS.items():
            bars={c: full[c][a:b] for c in COINS}
            s,cb,metas = eval_combo(bars, spec, fee=FEE_BASE, fund=FUND_BASE, q=Q)
            entry=dict(s)
            entry_cb = cb  # keep for turnover per seg already in s
            # fee2x for H2/B/C/FULL
            if seg_name in ('H2','B','C','FULL'):
                s2,_,_ = eval_combo(bars, spec, fee=FEE2X, fund=FUND_BASE, q=Q)
                entry['fee2x']=dict(s2)
            seg_results[seg_name]=entry
        # also record turnover explicitly (already in sharpe dict)
        row={'idx':idx,'sth':sth,'etc_cd':ETC_CD_FIXED,'trx_cd':trx_cd,'ts':ts,'vt':VT,'vw':VW,'lth_etc':LTH_ETC,'lth_trx':LTH_TRX,'sl_etc':None,'sl_trx':0.05,'side':'short','q':Q, 'segments':seg_results}
        rows.append(row)
        log(f"GRID {idx:02d} sth={sth:.2f} trx_cd={trx_cd} ts={ts} | H1 sh={seg_results['H1']['sharpe']:.3f} ann={seg_results['H1']['ann']:.3f} tr={seg_results['H1']['trades']} to={seg_results['H1']['turnover']:.5f} | H2 sh={seg_results['H2']['sharpe']:.3f} ann={seg_results['H2']['ann']:.4f} tr={seg_results['H2']['trades']} to={seg_results['H2']['turnover']:.5f} (2x{seg_results['H2']['fee2x']['sharpe']:.3f}) | B sh={seg_results['B']['sharpe']:.3f}(2x{seg_results['B']['fee2x']['sharpe']:.3f}) C sh={seg_results['C']['sharpe']:.3f}(2x{seg_results['C']['fee2x']['sharpe']:.3f}) FULL sh={seg_results['FULL']['sharpe']:.3f}(2x{seg_results['FULL']['fee2x']['sharpe']:.3f}) trFULL={seg_results['FULL']['trades']} toFULL={seg_results['FULL']['turnover']:.5f}")

    # Rank by H1 (unbiased) top3
    rows_sorted_H1 = sorted(rows, key=lambda r: r['segments']['H1']['sharpe'], reverse=True)
    top3_H1 = rows_sorted_H1[:3]
    rows_sorted_H2 = sorted(rows, key=lambda r: r['segments']['H2']['sharpe'], reverse=True)
    top3_H2 = rows_sorted_H2[:3]

    # short specialist verdict: H1-top1 H2>3.5 wins vs both ?
    # both baseline for comparison: need Y1b and Z1 both H2 sharpe
    # Y1b H2 ~4.091, Z1 H2 ~5.013 (from AB A1/A2). Also get them from baseline_segments vs grid.
    # Compute both_H2 for reference via evaluating both side at Z1(Y1b?) Actually both on same Z1 vt0.012 spec but side both
    # For short specialist we compare its H2 against both-mode H2 (Z1 both). Already have baseline Z1 H2.
    # Re-evaluate both baseline H2 quickly for exact same segments
    both_Z1_spec={'ETC':(0.88,0.12,18,None,24,'both',0.012,12),'TRX':(0.85,0.12,6,0.05,24,'both',0.012,12)}
    bars_H2={c: full[c][6077:6570] for c in COINS}
    both_H2_stats,_,_=eval_combo(bars_H2, both_Z1_spec, fee=FEE_BASE, fund=FUND_BASE, q=Q)
    both_H2_sharpe=both_H2_stats['sharpe']
    # Also Y1b both without vol
    both_Y1b_spec={'ETC':(0.88,0.12,18,None,24,'both',None,12),'TRX':(0.85,0.12,6,0.05,24,'both',None,12)}
    both_Y1b_H2_stats,_,_=eval_combo(bars_H2, both_Y1b_spec, fee=FEE_BASE, fund=FUND_BASE, q=Q)
    both_Y1b_H2_sharpe=both_Y1b_H2_stats['sharpe']
    log(f"\n=== SHORT SPECIALIST VERDICT ===")
    log(f"Both Z1 (vt0.012) H2 sharpe={both_H2_sharpe:.3f} | Both Y1b H2={both_Y1b_H2_sharpe:.3f}")
    for rank, r in enumerate(top3_H1,1):
        h1=r['segments']['H1']['sharpe']; h2=r['segments']['H2']['sharpe']; b=r['segments']['B']['sharpe']; c=r['segments']['C']['sharpe']; f=r['segments']['FULL']['sharpe']
        win_vs_both = (h2 > both_H2_sharpe)
        win_gate = (h2 > 3.5)
        # short special wins both only if h2>3.5 and h2>both
        # spec says: require H2>3.5 to count as win vs both
        overall_pass = bool(win_gate and win_vs_both)
        log(f"  H1-rank{rank} idx={r['idx']:02d} sth={r['sth']:.2f} trx_cd={r['trx_cd']} ts={r['ts']} | H1 {h1:.3f} -> H2 {h2:.3f}(2x{r['segments']['H2']['fee2x']['sharpe']:.3f}) B {b:.3f} C {c:.3f} FULL {f:.3f} | gate>3.5={win_gate} vs_both({both_H2_sharpe:.3f})={win_vs_both} => PASS={overall_pass}")
    # Also log H2-rank top3 for curiosity
    for rank, r in enumerate(top3_H2,1):
        log(f"  H2-rank{rank} idx={r['idx']:02d} sth={r['sth']:.2f} trx_cd={r['trx_cd']} ts={r['ts']} H2 {r['segments']['H2']['sharpe']:.3f} H1 {r['segments']['H1']['sharpe']:.3f}")

    h1_top1 = top3_H1[0]
    h1_top1_H2 = h1_top1['segments']['H2']['sharpe']
    specialist_pass = bool(h1_top1_H2 > 3.5 and h1_top1_H2 > both_H2_sharpe)
    specialist_label = "SHORT_SPECIALIST_WIN_BOTH" if specialist_pass else "SHORT_SPECIALIST_FAIL_VS_BOTH"
    log(f"Specialist top1 H1->H2 {h1_top1_H2:.3f} vs both_Z1 {both_H2_sharpe:.3f} vs gate 3.5 => {specialist_label}")

    # Build bucket decomposition table for docs (markdown friendly structure)
    # Also compute short vs long contribution ratio: buckets that are long vs short already both aggregated but we can compute total short cum via extra?
    # For now short vs long ratio was not bucket but we can compute overall short/long separation.
    # Provide readable bucket table including per baseline

    out={
        'config':{
            'formula':FORMULA,'coins':COINS,'full_n':FULL_N,'fold_n':FOLD_N,'fund_base':FUND_BASE,'fee_base':FEE_BASE,'fee2x':FEE2X,
            'segments':SEGS,'q':Q,'note':'E6 mirror run_ab.py: leg_series + quantile_mask_long abs top-q long-only + _apply_cooldown + _apply_stops(time_stop varied) + _vol_scale clamp0.2-2 roll1 + roll1, combo 50/50, lev2. Buckets [1-6,7-24,25+] on FULL via combo position episodes (sign-aware). Short grid 3*3*3=27 on Z1 vt0.012 short-only (ETC cd18 fixed, TRX cd varied).',
            'baselines':BASELINES,
            'grid_spec':'sth[0.08,0.10,0.12] x trx_cd[6,9,12] x ts[12,24,36] short-only Z1 vt0.012',
            'buckets':['1-6','7-24','25+'],
            'both_H2_ref':{'Z1_vt012':both_H2_sharpe,'Y1b_vtNone':both_Y1b_H2_sharpe}
        },
        'bucket_decomposition':{
            k:{'baseline_cfg':BASELINES[k],'FULL_stats':baseline_segments[k],'decomp':baseline_buckets[k]} for k in ['Y1b','Z1','AA-H1']
        },
        'short_grid':rows,
        'short_top3_H1':[ {'idx':r['idx'],'sth':r['sth'],'etc_cd':r['etc_cd'],'trx_cd':r['trx_cd'],'ts':r['ts'],'H1':r['segments']['H1'],'H2':r['segments']['H2'],'H2_fee2x':r['segments']['H2']['fee2x'],'B':r['segments']['B'],'B_fee2x':r['segments']['B']['fee2x'],'C':r['segments']['C'],'C_fee2x':r['segments']['C']['fee2x'],'FULL':r['segments']['FULL'],'FULL_fee2x':r['segments']['FULL']['fee2x']} for r in top3_H1],
        'short_top3_H2':[ {'idx':r['idx'],'sth':r['sth'],'trx_cd':r['trx_cd'],'ts':r['ts'],'H2':r['segments']['H2'],'H1':r['segments']['H1']} for r in top3_H2],
        'unbiased_verdict':{'H1_top1_H2':h1_top1_H2,'gate':3.5,'both_Z1_H2':both_H2_sharpe,'both_Y1b_H2':both_Y1b_H2_sharpe,'pass_vs_both':specialist_pass,'label':specialist_label, 'H1_top1_spec':{'sth':h1_top1['sth'],'trx_cd':h1_top1['trx_cd'],'ts':h1_top1['ts']}}
    }
    pathlib.Path("results/backtest_E6.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    pathlib.Path("logs/e6.log").write_text("\n".join(log_lines), encoding="utf-8")
    print("Wrote results/backtest_E6.json and logs/e6.log")
    # console summary
    print("\nBUCKET SUMMARY FULL:")
    for name in ['Y1b','Z1','AA-H1']:
        d=baseline_buckets[name]
        print(f"{name:6s} cum {d['bucket_cum']} pct {d['bucket_pct']} trades {d['bucket_trades']} avg_bars {d['bucket_avg_bars']}")
    print(f"\nSHORT GRID H1 TOP3:")
    for r in top3_H1:
        print(f" idx={r['idx']:02d} sth={r['sth']:.2f} cd={r['trx_cd']} ts={r['ts']} H1 {r['segments']['H1']['sharpe']:.3f} -> H2 {r['segments']['H2']['sharpe']:.3f} B {r['segments']['B']['sharpe']:.3f} FULL {r['segments']['FULL']['sharpe']:.3f} toFULL {r['segments']['FULL']['turnover']:.5f}")
    print(f"Specialist verdict: {specialist_label} H1-top1 H2 {h1_top1_H2:.3f} vs both_Z1 {both_H2_sharpe:.3f} gate 3.5")

if __name__=="__main__":
    main()
