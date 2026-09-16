
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if os.path.dirname(os.path.abspath(__file__)) else ".")
# ensure project root
import pathlib as _pl
root = _pl.Path(__file__).parent.parent if _pl.Path(__file__).parent.name=="research" else _pl.Path(".")
os.chdir(str(root))
import csv, math, json
import torch
from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest

FORMULA = [3,2,7,2,7,11,15,4,4,6,6,10]
COINS = ['ETC','TRX']
FUND_BASE = 0.0005
FEE_BASE = 0.0004
FEE2X = 0.0008
Q = 0.3
FULL_N = 6580
FOLD_N = FULL_N // 12  # 548

# Specs
Y1B_SPEC = {
    'ETC': (0.88, 0.12, 18, None, 24, 'both'),
    'TRX': (0.85, 0.12, 6, 0.05, 24, 'both'),
}
Z1_SPEC = {
    'ETC': (0.88, 0.12, 18, None, 24, 'both', 0.012, 12),
    'TRX': (0.85, 0.12, 6, 0.05, 24, 'both', 0.012, 12),
}
# For Z1/Y1b spec unpack helpers
def unpack_spec(spec, coin):
    vals = spec[coin]
    if len(vals)==6:
        lth,sth,cd,sl,ts,side = vals
        vt,vw = None,12
    elif len(vals)==8:
        lth,sth,cd,sl,ts,side,vt,vw = vals
    else:
        raise ValueError(vals)
    return lth,sth,cd,sl,ts,side,vt,vw

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

def leg_series(raw, rets_t, sig, lth, sth, cd, sl, ts, side, vt, vw, fee, fund, q=Q):
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
    # scale may be scalar float 1.0 or tensor
    lp = lp * scale
    sp = sp * scale
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
    # also compute long/short split for turnover decomposition if needed
    turn_l=(lp - lp.roll(1,dims=1)).abs()
    turn_s=(sp - sp.roll(1,dims=1)).abs()
    # stats needed
    turn_list=turn[0].tolist()
    pos=(lp-sp)[0].tolist()
    lp_list=lp[0].tolist()
    sp_list=sp[0].tolist()
    trades=sum(1 for t in range(len(pos)) if pos[t]!=0.0 and (t==0 or pos[t-1]==0.0))
    turnover=sum(turn_list)/len(turn_list) if turn_list else 0.0
    long_turnover=sum(turn_l[0].tolist())/len(turn_l[0].tolist()) if turn_l.numel()>0 else 0
    short_turnover=sum(turn_s[0].tolist())/len(turn_s[0].tolist()) if turn_s.numel()>0 else 0
    # pos_rate
    pos_rate=sum(1 for v in pos if v!=0)/len(pos) if pos else 0
    long_pos_rate=sum(1 for v in lp_list if v!=0)/len(lp_list) if lp_list else 0
    short_pos_rate=sum(1 for v in sp_list if v!=0)/len(sp_list) if sp_list else 0
    return {
        'net':net, 'trades':trades, 'turnover':turnover,
        'long_turnover':long_turnover, 'short_turnover':short_turnover,
        'pos_rate':pos_rate, 'long_pos_rate':long_pos_rate, 'short_pos_rate':short_pos_rate,
        'pos':pos, 'lp':lp_list, 'sp':sp_list,
        'turn':turn_list
    }

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

def eval_combo_for_bars(bars_dict, spec, fee, fund, side_override=None):
    # bars_dict: {coin: bars_list slice}
    # spec: dict coin -> tuple
    legs=[]
    meta_trades=[]
    meta_turnovers=[]
    meta_pos_rates=[]
    meta_lpos=[]
    meta_spos=[]
    meta_lturn=[]
    meta_sturn=[]
    for c in COINS:
        raw,rt,sg=build_mats(bars_dict[c])
        lth,sth,cd,sl,ts,side,vt,vw=unpack_spec(spec,c)
        if side_override is not None:
            side=side_override
        res=leg_series(raw,rt,sg,lth,sth,cd,sl,ts,side,vt,vw,fee,fund)
        legs.append(res['net'])
        meta_trades.append(res['trades'])
        meta_turnovers.append(res['turnover'])
        meta_pos_rates.append(res['pos_rate'])
        meta_lpos.append(res['long_pos_rate'])
        meta_spos.append(res['short_pos_rate'])
        meta_lturn.append(res['long_turnover'])
        meta_sturn.append(res['short_turnover'])
    cb=combo(legs,[0.5,0.5])
    avg_to=sum(meta_turnovers)/len(meta_turnovers) if meta_turnovers else 0
    avg_pr=sum(meta_pos_rates)/len(meta_pos_rates) if meta_pos_rates else 0
    avg_lpr=sum(meta_lpos)/len(meta_lpos) if meta_lpos else 0
    avg_spr=sum(meta_spos)/len(meta_spos) if meta_spos else 0
    s=stats(cb, trades=sum(meta_trades), turnover=avg_to, pos_rate=avg_pr, extra={'long_pos_rate':round(avg_lpr,4),'short_pos_rate':round(avg_spr,4),'long_turnover':round(sum(meta_lturn)/len(meta_lturn),6) if meta_lturn else 0,'short_turnover':round(sum(meta_sturn)/len(meta_sturn),6) if meta_sturn else 0,'trades_by':{COINS[i]:meta_trades[i] for i in range(len(COINS))}})
    return s, cb

def eval_side_split(bars_dict, spec, fee, fund):
    out={}
    for side in ['both','long','short']:
        s,_=eval_combo_for_bars(bars_dict,spec,fee,fund,side_override=side)
        out[side]=s
    return out

def fold_stats(folds):
    # folds: list of stat dicts
    sharpes=[f['sharpe'] for f in folds]
    anns=[f['ann'] for f in folds]
    mdds=[f['mdd'] for f in folds]
    cums=[f['cum'] for f in folds]
    trades=[f['trades'] for f in folds]
    n=len(sharpes)
    def mean(x): return sum(x)/len(x) if x else 0
    def median(x):
        s=sorted(x)
        mid=n//2
        return s[mid] if n%2==1 else (s[mid-1]+s[mid])/2
    return {
        'mean_sharpe':round(mean(sharpes),3),
        'median_sharpe':round(median(sharpes),3),
        'min_sharpe':round(min(sharpes),3) if sharpes else 0,
        'max_sharpe':round(max(sharpes),3) if sharpes else 0,
        'mean_ann':round(mean(anns),4),
        'median_ann':round(median(anns),4),
        'min_ann':round(min(anns),4) if anns else 0,
        'max_ann':round(max(anns),4) if anns else 0,
        'mean_mdd':round(mean(mdds),4),
        'max_mdd':round(max(mdds),4),
        'mean_cum':round(mean(cums),4),
        'sum_cum':round(sum(cums),4),
        'n_pos':sum(1 for x in sharpes if x>0),
        'n_neg':sum(1 for x in sharpes if x<0),
        'sharpes':[round(x,3) for x in sharpes],
        'anns':[round(x,4) for x in anns],
    }

import pathlib

def main():
    log_lines=[]
    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    full={c: load_bars(c) for c in COINS}
    n=min(len(v) for v in full.values())
    log(f"full 4h bars n={n}")
    assert n==FULL_N, f"expected {FULL_N} got {n}"

    # segments
    segs={
        'H2': (6077,6570),
        'B': (6380,6580),
        'C': (6080,6580),
        'C1': (6080,6330),
        'C2': (6330,6580),
    }

    # 1) H2/B/C/C1/C2 contrast Y1b vs Z1-best (fee base + fee2x for B/C)
    seg_results={}
    for name,(a,b) in segs.items():
        bars={c: full[c][a:b] for c in COINS}
        y1b,_=eval_combo_for_bars(bars, Y1B_SPEC, fee=FEE_BASE, fund=FUND_BASE)
        z1,_=eval_combo_for_bars(bars, Z1_SPEC, fee=FEE_BASE, fund=FUND_BASE)
        entry={'Y1b':y1b,'Z1':z1,'delta_sharpe':round(z1['sharpe']-y1b['sharpe'],3),'delta_ann':round(z1['ann']-y1b['ann'],4)}
        # fee2x for B/C only
        if name in ('B','C'):
            y1b_f2,_=eval_combo_for_bars(bars, Y1B_SPEC, fee=FEE2X, fund=FUND_BASE)
            z1_f2,_=eval_combo_for_bars(bars, Z1_SPEC, fee=FEE2X, fund=FUND_BASE)
            entry['Y1b_fee2x']=y1b_f2
            entry['Z1_fee2x']=z1_f2
            entry['delta_fee2x_sharpe']=round(z1_f2['sharpe']-y1b_f2['sharpe'],3)
        seg_results[name]=entry
        log(f"SEG {name} [{a}:{b}] Y1b sh={y1b['sharpe']} ann={y1b['ann']} Z1 sh={z1['sharpe']} ann={z1['ann']} delta={entry['delta_sharpe']}")
        if name in ('B','C'):
            log(f"  fee2x: Y1b sh={y1b_f2['sharpe']} Z1 sh={z1_f2['sharpe']} delta={entry['delta_fee2x_sharpe']}")

    # 2) 12-fold
    folds_Y1b=[]
    folds_Z1=[]
    for i in range(12):
        start=i*FOLD_N
        end=(i+1)*FOLD_N if i<11 else n
        bars={c: full[c][start:end] for c in COINS}
        s1,_=eval_combo_for_bars(bars, Y1B_SPEC, fee=FEE_BASE, fund=FUND_BASE)
        s2,_=eval_combo_for_bars(bars, Z1_SPEC, fee=FEE_BASE, fund=FUND_BASE)
        s1['fold']=i; s1['range']=[start,end]
        s2['fold']=i; s2['range']=[start,end]
        folds_Y1b.append(s1)
        folds_Z1.append(s2)
        log(f"fold {i:2d} [{start:4d}:{end:4d}] Y1b sh={s1['sharpe']:6.3f} ann={s1['ann']:7.4f} mdd={s1['mdd']:6.4f} trades={s1['trades']} | Z1 sh={s2['sharpe']:6.3f} ann={s2['ann']:7.4f} mdd={s2['mdd']:6.4f} trades={s2['trades']} delta={round(s2['sharpe']-s1['sharpe'],3)}")

    summary_Y1b=fold_stats(folds_Y1b)
    summary_Z1=fold_stats(folds_Z1)
    # worst delta
    deltas=[folds_Z1[i]['sharpe']-folds_Y1b[i]['sharpe'] for i in range(12)]
    worst_idx=int(min(range(12), key=lambda i: deltas[i]))
    worst_delta=round(min(deltas),3)
    log(f"12-fold summary Y1b mean={summary_Y1b['mean_sharpe']} median={summary_Y1b['median_sharpe']} min={summary_Y1b['min_sharpe']} max={summary_Y1b['max_sharpe']} n_pos={summary_Y1b['n_pos']}")
    log(f"12-fold summary Z1  mean={summary_Z1['mean_sharpe']} median={summary_Z1['median_sharpe']} min={summary_Z1['min_sharpe']} max={summary_Z1['max_sharpe']} n_pos={summary_Z1['n_pos']}")
    log(f"delta mean {round(summary_Z1['mean_sharpe']-summary_Y1b['mean_sharpe'],3)} median {round(summary_Z1['median_sharpe']-summary_Y1b['median_sharpe'],3)} worst delta {worst_delta} at fold {worst_idx}")

    # 3) pressure grid fund[0.0003,0.0005,0.001] x fee[0.0004,0.0008] on B/C 6 grids each config
    FUNDS=[0.0003,0.0005,0.001]
    FEES=[0.0004,0.0008]
    pressure={}
    for seg_name in ('B','C'):
        a,b=segs[seg_name]
        bars={c: full[c][a:b] for c in COINS}
        grid=[]
        for fund in FUNDS:
            for fee in FEES:
                y,_=eval_combo_for_bars(bars, Y1B_SPEC, fee=fee, fund=fund)
                z,_=eval_combo_for_bars(bars, Z1_SPEC, fee=fee, fund=fund)
                grid.append({
                    'fund':fund,'fee':fee,
                    'Y1b_sharpe':y['sharpe'],'Y1b_ann':y['ann'],'Y1b_mdd':y['mdd'],'Y1b_trades':y['trades'],
                    'Z1_sharpe':z['sharpe'],'Z1_ann':z['ann'],'Z1_mdd':z['mdd'],'Z1_trades':z['trades'],
                    'delta_sharpe':round(z['sharpe']-y['sharpe'],3)
                })
        # worst B per config?
        worstB_Y1b=min(g['Y1b_sharpe'] for g in grid)
        worstB_Z1=min(g['Z1_sharpe'] for g in grid)
        pressure[seg_name]={'grid':grid,'worst_Y1b':round(worstB_Y1b,3),'worst_Z1':round(worstB_Z1,3)}
        log(f"pressure {seg_name}: worst Y1b={worstB_Y1b:.3f} worst Z1={worstB_Z1:.3f}")

    # 4) side split short-only vs both on H2/B/C Y1b vs Z1-best
    side_results={}
    for seg_name in ('H2','B','C'):
        a,b=segs[seg_name]
        bars={c: full[c][a:b] for c in COINS}
        res={}
        for cfg_name,spec in [('Y1b',Y1B_SPEC),('Z1',Z1_SPEC)]:
            split=eval_side_split(bars, spec, fee=FEE_BASE, fund=FUND_BASE)
            res[cfg_name]=split
            log(f"side {seg_name} {cfg_name}: both sh={split['both']['sharpe']} long sh={split['long']['sharpe']} short sh={split['short']['sharpe']} long_pos={split['both']['long_pos_rate']} short_pos={split['both']['short_pos_rate']} long_tr={split['long']['trades']} short_tr={split['short']['trades']}")
            # also fee2x side?
        side_results[seg_name]=res
        # note if long near zero
        for cfg_name in ('Y1b','Z1'):
            both=side_results[seg_name][cfg_name]['both']
            long_s=side_results[seg_name][cfg_name]['long']
            # check if long contribution near zero
            # we note if long_pos_rate <0.01 or long sharpe close to 0
            if both['long_pos_rate']<0.02:
                log(f"  NOTE {seg_name} {cfg_name} both long_pos_rate {both['long_pos_rate']} near zero despite side=both")

    # 5) decision
    # criteria: H2>Y1b AND B/C fee2x > threshold AND 12-fold not collapse AND pressure worstB>0
    # define thresholds: B_fee2x>1.0, C_fee2x>1.0 maybe stricter B>1.5 C>2.0 ?
    h2_ok = seg_results['H2']['Z1']['sharpe'] > seg_results['H2']['Y1b']['sharpe']
    b_fee2x = seg_results['B']['Z1_fee2x']['sharpe']
    c_fee2x = seg_results['C']['Z1_fee2x']['sharpe']
    # thresholds inspired by Y2/Z1: B_fee2x>1.0, C_fee2x>1.5 ?
    b_fee2x_ok = b_fee2x > 1.0
    c_fee2x_ok = c_fee2x > 1.0
    # 12-fold not collapse: mean not down >0.3, median >0, n_pos >=10, min>=-1.5, worst delta not < -1.0
    fold_mean_ok = summary_Z1['mean_sharpe'] > summary_Y1b['mean_sharpe'] - 0.3
    fold_median_ok = summary_Z1['median_sharpe'] >= 0.5  # arbitrary
    fold_npos_ok = summary_Z1['n_pos'] >= 9
    fold_min_ok = summary_Z1['min_sharpe'] >= -1.5
    worst_delta_ok = worst_delta >= -1.0
    fold_ok = fold_mean_ok and fold_npos_ok and fold_min_ok and worst_delta_ok
    # pressure worstB>0 for Z1
    pressure_ok = pressure['B']['worst_Z1'] > 0 and pressure['C']['worst_Z1'] > 0
    # Additional: C1/C2 both >0.5
    c1_ok = seg_results['C1']['Z1']['sharpe'] > 0.5
    c2_ok = seg_results['C2']['Z1']['sharpe'] > 0.5
    c_split_ok = c1_ok and c2_ok

    decision_ok = h2_ok and b_fee2x_ok and c_fee2x_ok and fold_ok and pressure_ok and c_split_ok
    decision = "ADOPT Z1-best" if decision_ok else "KEEP Y1b (reject Z1-best)"
    reason=[]
    if not h2_ok: reason.append(f"H2 not >Y1b ({seg_results['H2']['Z1']['sharpe']} vs {seg_results['H2']['Y1b']['sharpe']})")
    if not b_fee2x_ok: reason.append(f"B fee2x {b_fee2x} <=1.0")
    if not c_fee2x_ok: reason.append(f"C fee2x {c_fee2x} <=1.0")
    if not fold_ok: reason.append(f"fold collapse mean {summary_Z1['mean_sharpe']} vs Y1b {summary_Y1b['mean_sharpe']} n_pos {summary_Z1['n_pos']} min {summary_Z1['min_sharpe']} worst_delta {worst_delta}")
    if not c_split_ok: reason.append(f"C1/C2 {seg_results['C1']['Z1']['sharpe']}/{seg_results['C2']['Z1']['sharpe']} not both >0.5")
    if not pressure_ok: reason.append(f"pressure worstB Z1 B={pressure['B']['worst_Z1']} C={pressure['C']['worst_Z1']} not >0")
    log(f"DECISION: {decision}")
    if reason: log("reasons: " + "; ".join(reason))
    else: log("all gates passed")

    out={
        'config':{
            'formula': FORMULA,
            'coins': COINS,
            'full_n': FULL_N,
            'fold_size': FOLD_N,
            'fund_base': FUND_BASE,
            'fee_base': FEE_BASE,
            'fee2x': FEE2X,
            'q': Q,
            'segments': segs,
            'Y1b_spec': {k:list(v) for k,v in Y1B_SPEC.items()},
            'Z1_spec': {k:list(v) for k,v in Z1_SPEC.items()},
            'vt':0.012,'vw':12,'note':'Z1-best = Y1b + vt0.012 vw12 per-leg _vol_scale clamp0.2-2.0 roll1 post-stops pre-roll q0.3 long-only quantile 50/50 ETC0.88/0.12/cd18/slNone + TRX0.85/0.12/cd6/sl0.05 ts24 both aster lev2 fund0.0005 fee0.0004/0.0008'
        },
        'segments': seg_results,
        'folds': {'Y1b':folds_Y1b,'Z1':folds_Z1,'summary_Y1b':summary_Y1b,'summary_Z1':summary_Z1,'worst_delta':worst_delta,'worst_fold':worst_idx,'deltas':[round(x,3) for x in deltas]},
        'pressure': pressure,
        'side_split': side_results,
        'decision': {
            'adopt': bool(decision_ok),
            'verdict': decision,
            'gates': {
                'H2_gt_Y1b': bool(h2_ok),
                'B_fee2x_gt1': bool(b_fee2x_ok),
                'C_fee2x_gt1': bool(c_fee2x_ok),
                'fold_ok': bool(fold_ok),
                'fold_detail': {'mean_ok':bool(fold_mean_ok),'median_ok':bool(fold_median_ok),'npos_ok':bool(fold_npos_ok),'min_ok':bool(fold_min_ok),'worst_delta_ok':bool(worst_delta_ok)},
                'pressure_worstB_gt0': bool(pressure_ok),
                'C_split_gt0.5': bool(c_split_ok)
            },
            'reasons': reason,
            'thresholds': 'H2>Y1b and B/C fee2x>1.0 and 12fold mean not <-0.3 vs Y1b & n_pos>=9 & min>=-1.5 & worst_delta>=-1.0 and pressure worstB/C>0 and C1/C2>0.5'
        }
    }

    pathlib.Path("results/backtest_W1_z1verify.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    pathlib.Path("logs/w1.log").write_text("\n".join(log_lines))
    log("Wrote results/backtest_W1_z1verify.json and logs/w1.log")
    # also print key summary json
    print(json.dumps({
        'segments_delta':{k:{'Y1b_sh':v['Y1b']['sharpe'],'Z1_sh':v['Z1']['sharpe'],'delta':v['delta_sharpe'],'Y1b_fee2x':v.get('Y1b_fee2x',{}).get('sharpe'),'Z1_fee2x':v.get('Z1_fee2x',{}).get('sharpe')} for k,v in seg_results.items()},
        'folds_summary_Y1b':summary_Y1b,
        'folds_summary_Z1':summary_Z1,
        'worst_delta':worst_delta,
        'worst_fold':worst_idx,
        'pressure':{k:{'worst_Y1b':v['worst_Y1b'],'worst_Z1':v['worst_Z1']} for k,v in pressure.items()},
        'decision':decision,
        'gates':out['decision']['gates']
    }, indent=2, ensure_ascii=False))

if __name__=='__main__':
    main()
