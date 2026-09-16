
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
FOLD_N = FULL_N // 12  # 548
Q = 0.3

# baseline Y1b specs per coin
BASE_ETC = (0.88, 0.12, 18, None, 24)  # lth, sth, cd, sl, ts
BASE_TRX = (0.85, 0.12, 6, 0.05, 24)

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

def leg_series(raw, rets_t, sig, lth, sth, cd, sl, ts, side, vt, vw, fee, fund, q):
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
    tx=turn*(bt.base_fee+torch.clamp(bt.trade_size/(raw['liquidity']+1e-9),0.0,0.05))
    gross=(lp-sp)*rets_t*bt.leverage
    fnd=(lp-sp)*bt.default_funding_rate*bt.leverage
    net=(gross - tx*bt.leverage - fnd)[0].tolist()
    turn_l=(lp - lp.roll(1,dims=1)).abs()
    turn_s=(sp - sp.roll(1,dims=1)).abs()
    turn_list=turn[0].tolist()
    pos=(lp-sp)[0].tolist()
    lp_list=lp[0].tolist()
    sp_list=sp[0].tolist()
    trades=sum(1 for t in range(len(pos)) if pos[t]!=0.0 and (t==0 or pos[t-1]==0.0))
    turnover=sum(turn_list)/len(turn_list) if turn_list else 0.0
    long_turnover=sum(turn_l[0].tolist())/len(turn_l[0].tolist()) if turn_l.numel()>0 else 0
    short_turnover=sum(turn_s[0].tolist())/len(turn_s[0].tolist()) if turn_s.numel()>0 else 0
    pos_rate=sum(1 for v in pos if v!=0)/len(pos) if pos else 0
    long_pos_rate=sum(1 for v in lp_list if v!=0)/len(lp_list) if lp_list else 0
    short_pos_rate=sum(1 for v in sp_list if v!=0)/len(sp_list) if sp_list else 0
    return {'net':net,'trades':trades,'turnover':turnover,'long_turnover':long_turnover,'short_turnover':short_turnover,'pos_rate':pos_rate,'long_pos_rate':long_pos_rate,'short_pos_rate':short_pos_rate}

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
    legs=[]
    meta=[]
    for c in COINS:
        lth, sth, cd, sl, ts, side, vt, vw = spec_dict[c]
        raw, rt, sg = build_mats(bars_dict[c])
        res = leg_series(raw, rt, sg, lth, sth, cd, sl, ts, side, vt, vw, fee, fund, q)
        legs.append(res['net'])
        meta.append(res)
    cb = combo(legs, [0.5,0.5])
    avg_to = sum(m['turnover'] for m in meta)/len(meta)
    avg_pr = sum(m['pos_rate'] for m in meta)/len(meta)
    avg_lpr = sum(m['long_pos_rate'] for m in meta)/len(meta)
    avg_spr = sum(m['short_pos_rate'] for m in meta)/len(meta)
    avg_lto = sum(m['long_turnover'] for m in meta)/len(meta)
    avg_sto = sum(m['short_turnover'] for m in meta)/len(meta)
    s = stats(cb, trades=sum(m['trades'] for m in meta), turnover=avg_to, pos_rate=avg_pr, extra={'long_pos_rate':round(avg_lpr,4),'short_pos_rate':round(avg_spr,4),'long_turnover':round(avg_lto,6),'short_turnover':round(avg_sto,6),'trades_by':{COINS[i]:meta[i]['trades'] for i in range(len(COINS))}})
    return s, cb

def fold_stats_from_list(stats_list):
    sharpes=[x['sharpe'] for x in stats_list]
    anns=[x['ann'] for x in stats_list]
    mdds=[x['mdd'] for x in stats_list]
    cums=[x['cum'] for x in stats_list]
    n=len(sharpes)
    def mean(v): return sum(v)/len(v) if v else 0
    def median(v):
        s=sorted(v); mid=n//2
        return s[mid] if n%2==1 else (s[mid-1]+s[mid])/2 if n else 0
    return {'mean_sharpe':round(mean(sharpes),3),'median_sharpe':round(median(sharpes),3),'min_sharpe':round(min(sharpes),3) if sharpes else 0,'max_sharpe':round(max(sharpes),3) if sharpes else 0,'mean_ann':round(mean(anns),4),'median_ann':round(median(anns),4),'min_ann':round(min(anns),4) if anns else 0,'max_ann':round(max(anns),4) if anns else 0,'mean_mdd':round(mean(mdds),4),'max_mdd':round(max(mdds),4),'mean_cum':round(mean(cums),4),'sum_cum':round(sum(cums),4),'n_pos':sum(1 for x in sharpes if x>0),'n_neg':sum(1 for x in sharpes if x<0),'sharpes':[round(x,3) for x in sharpes],'anns':[round(x,4) for x in anns],'mdds':[round(x,4) for x in mdds],'cums':[round(x,4) for x in cums]}

def main():
    log_lines=[]
    def log(m):
        print(m, flush=True)
        log_lines.append(m)
    full={c: load_bars(c) for c in COINS}
    n=min(len(v) for v in full.values())
    log(f"full 4h bars n={n} FORMULA {FORMULA} COINS {COINS} fund {FUND_BASE} fee {FEE_BASE}/{FEE2X}")
    assert n==FULL_N, f"expected {FULL_N} got {n}"
    segs={'H2':(6077,6570),'B':(6380,6580),'C':(6080,6580),'C1':(6080,6330),'C2':(6330,6580),'FULL':(0,6580)}
    # define 11 configs
    # helper to create spec dict from params
    def make_spec(side_mode, vt, vw, lth_etc=None, lth_trx=None, q=Q):
        # side_mode: 'both' or 'short'
        spec={}
        etc_lth = BASE_ETC[0] if lth_etc is None else lth_etc
        trx_lth = BASE_TRX[0] if lth_trx is None else lth_trx
        spec['ETC']=(etc_lth, BASE_ETC[1], BASE_ETC[2], BASE_ETC[3], BASE_ETC[4], side_mode, vt, vw)
        spec['TRX']=(trx_lth, BASE_TRX[1], BASE_TRX[2], BASE_TRX[3], BASE_TRX[4], side_mode, vt, vw)
        return spec

    configs=[]
    # A both q0.3
    configs.append({'id':'A1','label':'A_both_q03_vtNone','side':'both','q':0.3,'vt':None,'vw':12,'spec':make_spec('both',None,12)})
    configs.append({'id':'A2','label':'A_both_q03_vt012','side':'both','q':0.3,'vt':0.012,'vw':12,'spec':make_spec('both',0.012,12)})
    # B short q0.3
    configs.append({'id':'B1','label':'B_short_q03_vtNone','side':'short','q':0.3,'vt':None,'vw':12,'spec':make_spec('short',None,12)})
    configs.append({'id':'B2','label':'B_short_q03_vt012','side':'short','q':0.3,'vt':0.012,'vw':12,'spec':make_spec('short',0.012,12)})
    # C short qNone
    configs.append({'id':'C1','label':'C_short_qNone_vtNone','side':'short','q':None,'vt':None,'vw':12,'spec':make_spec('short',None,12)})
    configs.append({'id':'C2','label':'C_short_qNone_vt012','side':'short','q':None,'vt':0.012,'vw':12,'spec':make_spec('short',0.012,12)})
    # D both qNone
    configs.append({'id':'D1','label':'D_both_qNone_vtNone','side':'both','q':None,'vt':None,'vw':12,'spec':make_spec('both',None,12)})
    configs.append({'id':'D2','label':'D_both_qNone_vt012','side':'both','q':None,'vt':0.012,'vw':12,'spec':make_spec('both',0.012,12)})
    # E lth experiments on both q0.3 vt0.012
    for lth in [0.88,0.92,0.98]:
        # uniform lth for both coins as experiment (so ETC and TRX both = lth)
        spec=make_spec('both',0.012,12, lth_etc=lth, lth_trx=lth)
        configs.append({'id':f'E_lth{lth:.2f}'.replace('.','p'),'label':f'E_both_q03_vt012_lth{lth:.2f}','side':'both','q':0.3,'vt':0.012,'vw':12,'lth':lth,'spec':spec})
    # There will be 11 configs (2+2+2+2+3=11)
    assert len(configs)==11, len(configs)

    rows=[]
    for cfg in configs:
        log(f"\n=== {cfg['id']} {cfg['label']} side={cfg['side']} q={cfg['q']} vt={cfg['vt']} lth={cfg.get('lth','base')} ===")
        spec = cfg['spec']
        q = cfg['q']
        seg_results={}
        for seg_name,(a,b) in segs.items():
            bars={c: full[c][a:b] for c in COINS}
            s,_=eval_combo(bars, spec, fee=FEE_BASE, fund=FUND_BASE, q=q)
            entry=dict(s)
            # fee2x for B/C/FULL maybe but spec requires fee2x B/C ; compute for B,C and FULL for completeness
            if seg_name in ('B','C','FULL'):
                s2,_=eval_combo(bars, spec, fee=FEE2X, fund=FUND_BASE, q=q)
                entry['fee2x']=dict(s2)
                log(f"  {seg_name:4s} base sh={s['sharpe']:6.3f} ann={s['ann']:7.4f} mdd={s['mdd']:.4f} cum={s['cum']:.4f} tr={s['trades']} to={s['turnover']:.5f} pos={s['pos_rate']:.4f} lpr={s['long_pos_rate']:.4f} spr={s['short_pos_rate']:.4f} | fee2x sh={s2['sharpe']:.3f} ann={s2['ann']:.4f}")
            else:
                log(f"  {seg_name:4s} base sh={s['sharpe']:6.3f} ann={s['ann']:7.4f} mdd={s['mdd']:.4f} cum={s['cum']:.4f} tr={s['trades']} to={s['turnover']:.5f} pos={s['pos_rate']:.4f} lpr={s['long_pos_rate']:.4f} spr={s['short_pos_rate']:.4f}")
            seg_results[seg_name]=entry
        # 12-fold
        folds=[]
        for i in range(12):
            start=i*FOLD_N
            end=(i+1)*FOLD_N if i<11 else n
            bars={c: full[c][start:end] for c in COINS}
            s,_=eval_combo(bars, spec, fee=FEE_BASE, fund=FUND_BASE, q=q)
            s['fold']=i
            s['range']=[start,end]
            folds.append(s)
        summary=fold_stats_from_list(folds)
        log(f"  12fold mean_sh={summary['mean_sharpe']:.3f} median={summary['median_sharpe']:.3f} min={summary['min_sharpe']:.3f} max={summary['max_sharpe']:.3f} n_pos={summary['n_pos']}/12")
        rows.append({'id':cfg['id'],'label':cfg['label'],'side':cfg['side'],'q':cfg['q'],'vt':cfg['vt'],'vw':cfg['vw'],'lth':cfg.get('lth',None),'spec':{k:list(v) for k,v in spec.items()},'segments':seg_results,'folds':folds,'fold_summary':summary})

    # compute deltas for quick log: both vs short
    out={'config':{'formula':FORMULA,'coins':COINS,'full_n':FULL_N,'fold_n':FOLD_N,'fund_base':FUND_BASE,'fee_base':FEE_BASE,'fee2x':FEE2X,'segments':segs,'q_base':Q,'note':'AB: A both q0.3 vs B short q0.3 vs C short qNone vs D both qNone each vtNone/vt012, E lth sweep 0.88/0.92/0.98 on both q0.3 vt012 unified lth for both coins (TRX also = lth for E). Engine mirror run_w1 leg_series+quantile+_vol_scale+stops+roll1. 12fold 548bars.'},'rows':rows}
    pathlib.Path("results/backtest_AB.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    pathlib.Path("logs/ab.log").write_text("\n".join(log_lines))
    print("Wrote results/backtest_AB.json and logs/ab.log")
    # summary table for console
    print("\nSUMMARY H2/B/C sharpe:")
    for r in rows:
        h2=r['segments']['H2']['sharpe']; b=r['segments']['B']['sharpe']; c=r['segments']['C']['sharpe']; b2=r['segments']['B']['fee2x']['sharpe']; c2=r['segments']['C']['fee2x']['sharpe']
        print(f"{r['id']:8s} {r['label']:24s} H2 {h2:5.2f} B {b:5.2f}(2x{b2:5.2f}) C {c:5.2f}(2x{c2:5.2f}) lpr_H2 {r['segments']['H2']['long_pos_rate']:.4f} spr_H2 {r['segments']['H2']['short_pos_rate']:.4f} Full_sh {r['segments']['FULL']['sharpe']:.3f} 12mMean {r['fold_summary']['mean_sharpe']:.3f} n_pos {r['fold_summary']['n_pos']}")

if __name__=="__main__":
    main()
