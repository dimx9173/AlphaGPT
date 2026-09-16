import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import csv, math, json, pathlib
import torch
from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest

FORMULA = [3,2,7,2,7,11,15,4,4,6,6,10]
COINS = ['ETC','TRX']
Q = 0.3
VT = 0.012
VW = 12
TS = 24
FUND_BASE = 0.0005
FEE_BASE = 0.0004
FEE2X = 0.0008
N = 6580
FOLD_N = 6580//12  # 548

# 4 configs for E1:
# Y1b(vtNone), Z1(0.12/18/6 vt0.012), AA-H1best(0.10/15/9 vt0.012), AA-H2best(0.12/15/9 vt0.012)
# All share ETC 0.88/TRX0.85, sl ETC None TRX 0.05, ts24, q0.3, 50/50, aster 2x
CONFIGS = {
    'Y1b': {'sth':0.12,'etc_cd':18,'trx_cd':6,'vt':None,'vw':12},
    'Z1':  {'sth':0.12,'etc_cd':18,'trx_cd':6,'vt':0.012,'vw':12},
    'AA_H1best': {'sth':0.10,'etc_cd':15,'trx_cd':9,'vt':0.012,'vw':12},
    'AA_H2best': {'sth':0.12,'etc_cd':15,'trx_cd':9,'vt':0.012,'vw':12},
}

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
         'liquidity':torch.full((1,n),1e7),
         'fdv':torch.full((1,n),1e8)}
    sig=StackVM(use_advanced=False).execute(FORMULA, FeatureEngineer.compute_features(raw, use_advanced=False))
    rets=[(bars[i+1][3]-bars[i][3])/bars[i][3] for i in range(n-1)] + [0.0]
    return raw, torch.tensor([rets]), sig

def quantile_mask_long(sig,q):
    if q is None: return None
    a=sig.detach().float().abs().reshape(-1)
    k=max(1,int(len(a)*float(q)))
    thr=torch.topk(a,k).values.min()
    return (sig.detach().float().abs() >= thr).float()

def leg_series(raw, rets_t, sig, lth, sth, cd, sl, fee=None, fund=None, side='both', q=Q, vt=VT, vw=VW, ts=TS):
    # mirror run_aa.py exactly: leg_series + quantile + _vol_scale + stops + roll1
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

def eval_combo_on_bars(mats_by_coin, cfg, fee, fund):
    legs=[]; trs=[]; tos=[]
    for coin, (lth,cd,sl) in [('ETC',(0.88,cfg['etc_cd'],None)), ('TRX',(0.85,cfg['trx_cd'],0.05))]:
        raw,rt,sg=mats_by_coin[coin]
        net,t,to=leg_series(raw,rt,sg,lth,cfg['sth'],cd,sl,fee=fee,fund=fund,vt=cfg['vt'],vw=cfg['vw'])
        legs.append(net); trs.append(t); tos.append(to)
    cb=combo(legs,[0.5,0.5])
    s=stats(cb,trades=sum(trs),turnover=sum(tos)/len(tos) if tos else 0)
    s['trades_by']={c:trs[i] for i,c in enumerate(COINS)}
    return s, cb

def walkforward_for_config(full, cfg, fee, fund, n_folds):
    fold_size = N // n_folds
    m = N
    # WF semantics: for each fold k, test = [k*fold_size : (k+1)*fold_size) (last fold to m)
    # Training would be prior, but evaluation is test-only per standard walk-forward report
    # We report test-fold sharpes sequence like run_w1 12fold
    folds=[]
    for i in range(n_folds):
        a=i*fold_size
        b=(i+1)*fold_size if i < n_folds-1 else m
        mats={c: build_mats(full[c][a:b]) for c in COINS}
        s,_=eval_combo_on_bars(mats, cfg, fee=fee, fund=fund)
        s['fold']=i; s['range']=[a,b]
        folds.append(s)
    # also sequential IC: report whole? we also want summary
    return folds

def fold_summary(folds):
    sharpes=[f['sharpe'] for f in folds]
    n=len(sharpes)
    mean=sum(sharpes)/n if n else 0
    srt=sorted(sharpes)
    med=srt[n//2] if n%2==1 else (srt[n//2-1]+srt[n//2])/2
    max_dd=max(f['mdd'] for f in folds) if folds else 0
    # compute median sharpe vs mean
    n_pos=sum(1 for x in sharpes if x>0)
    anns=[f['ann'] for f in folds]
    return {
        'sharpes':[round(x,3) for x in sharpes],
        'mean':round(mean,3),
        'median':round(med,3),
        'min':round(min(sharpes),3) if sharpes else 0,
        'max':round(max(sharpes),3) if sharpes else 0,
        'n_pos':int(n_pos),
        'n_neg':int(n-n_pos),
        'max_dd':round(max_dd,4),
        'mean_ann':round(sum(anns)/len(anns),4) if anns else 0,
        'median_ann':round(sorted(anns)[n//2] if n%2==1 else (sorted(anns)[n//2-1]+sorted(anns)[n//2])/2,4) if anns else 0,
    }

def main():
    logp=pathlib.Path('logs/e1.log')
    logp.parent.mkdir(parents=True, exist_ok=True)
    log_lines=[]
    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)
        with open(logp,'a') as f: f.write(msg+'\n')
    open(logp,'w').write('E1 start\n')
    full={c:load_bars(c) for c in COINS}
    n=min(len(v) for v in full.values())
    log(f'full 4h bars n={n} ETC={len(full["ETC"])} TRX={len(full["TRX"])}')
    assert n==N, f'expected {N} got {n}'

    # 12fold + 8/12/24 fold WF per config (8fold and 24fold + 12fold already)
    # The task: 12fold (548bar) + 8fold/24fold WF
    wf_specs=[8,12,24]
    results12={}
    resultsWF={}
    for cfg_name,cfg in CONFIGS.items():
        # 12fold detailed (548bar)
        folds12=walkforward_for_config(full, cfg, fee=FEE_BASE, fund=FUND_BASE, n_folds=12)
        summ12=fold_summary(folds12)
        results12[cfg_name]={'folds':folds12,'summary':summ12}
        log(f"12FOLD {cfg_name} vt={cfg['vt']} sth={cfg['sth']} etc{cfg['etc_cd']} trx{cfg['trx_cd']} mean={summ12['mean']} med={summ12['median']} min={summ12['min']} max={summ12['max']} n_pos={summ12['n_pos']}/{len(folds12)} max_dd={summ12['max_dd']} sharpes={summ12['sharpes']}")
        # WF 8/12/24
        wf_entry={}
        for nf in wf_specs:
            if nf==12:
                # reuse
                wf_entry[f'WF{nf}']={'folds':folds12,'summary':summ12}
            else:
                folds=walkforward_for_config(full, cfg, fee=FEE_BASE, fund=FUND_BASE, n_folds=nf)
                summ=fold_summary(folds)
                wf_entry[f'WF{nf}']={'folds':folds,'summary':summ}
                log(f"WF{nf:2d}  {cfg_name} mean={summ['mean']} med={summ['median']} min={summ['min']} max={summ['max']} n_pos={summ['n_pos']}/{nf} max_dd={summ['max_dd']} sharpes={summ['sharpes']}")
        resultsWF[cfg_name]=wf_entry

    # Stress: AA-H1best only, fund[0.0003,0.0005,0.001] x fee[0.0004,0.0008]=6 on B/C (same as W1)
    stress={}
    cfg_h1=CONFIGS['AA_H1best']
    segs={'B':(6380,6580),'C':(6080,6580)}
    funds=[0.0003,0.0005,0.001]
    fees=[0.0004,0.0008]
    for seg_name,(a,b) in segs.items():
        mats={c: build_mats(full[c][a:b]) for c in COINS}
        grid=[]
        for fund in funds:
            for fee in fees:
                s,_=eval_combo_on_bars(mats, cfg_h1, fee=fee, fund=fund)
                grid.append({'fund':fund,'fee':fee,'sharpe':s['sharpe'],'ann':s['ann'],'mdd':s['mdd'],'cum':s['cum'],'trades':s['trades']})
                log(f"STRESS AA-H1best {seg_name} fund={fund:.4f} fee={fee:.4f} sh={s['sharpe']:.3f} ann={s['ann']:.4f} mdd={s['mdd']:.4f} tr={s['trades']}")
        worst=min(g['sharpe'] for g in grid)
        worst_entry=min(grid, key=lambda g: g['sharpe'])
        stress[seg_name]={'grid':grid,'worst_sharpe':round(worst,3),'worst_entry':worst_entry}
        log(f"STRESS AA-H1best {seg_name} worst={worst:.3f} at {worst_entry}")

    # Decision: if any AA candidate 12fold median > Y1b and mean>1.7 and n_pos>=9 and WF8 n_pos>=6 -> challenger else keep Y1b
    y1b_summ = results12['Y1b']['summary']
    h1_summ = results12['AA_H1best']['summary']
    h2_summ = results12['AA_H2best']['summary']
    z1_summ = results12['Z1']['summary']
    wf8_h1 = resultsWF['AA_H1best']['WF8']['summary']
    wf8_h2 = resultsWF['AA_H2best']['WF8']['summary']
    def qualifies(summ, wf8_summ, label):
        cond_med = summ['median'] > y1b_summ['median']
        cond_mean = summ['mean'] > 1.7
        cond_npos = summ['n_pos'] >= 9
        cond_wf8 = wf8_summ['n_pos'] >= 6
        ok = bool(cond_med and cond_mean and cond_npos and cond_wf8)
        return {'label':label,'median':summ['median'],'mean':summ['mean'],'n_pos':summ['n_pos'],'wf8_n_pos':wf8_summ['n_pos'],'y1b_median':y1b_summ['median'],'y1b_mean':y1b_summ['mean'],'gates':{'median_gt_Y1b':bool(cond_med),'mean_gt1.7':bool(cond_mean),'npos_ge9':bool(cond_npos),'wf8_npos_ge6':bool(cond_wf8)},'PASS':ok}
    q_h1 = qualifies(h1_summ, wf8_h1, 'AA_H1best')
    q_h2 = qualifies(h2_summ, wf8_h2, 'AA_H2best')
    log(f"QUALIFY AA_H1best med {q_h1['median']} vs Y1b {q_h1['y1b_median']} mean {q_h1['mean']} n_pos {q_h1['n_pos']} wf8_npos {q_h1['wf8_n_pos']} gates={q_h1['gates']} => {'PASS' if q_h1['PASS'] else 'FAIL'}")
    log(f"QUALIFY AA_H2best med {q_h2['median']} vs Y1b {q_h2['y1b_median']} mean {q_h2['mean']} n_pos {q_h2['n_pos']} wf8_npos {q_h2['wf8_n_pos']} gates={q_h2['gates']} => {'PASS' if q_h2['PASS'] else 'FAIL'}")
    if q_h1['PASS'] or q_h2['PASS']:
        # prefer H1 if both pass (unbiased), else whoever passes
        if q_h1['PASS'] and q_h2['PASS']:
            challenger = 'AA_H1best' if h1_summ['median'] >= h2_summ['median'] else 'AA_H2best'
        elif q_h1['PASS']:
            challenger = 'AA_H1best'
        else:
            challenger = 'AA_H2best'
        verdict = f'UPGRADE challenger {challenger}'
        adopt = True
    else:
        verdict = 'KEEP Y1b'
        adopt = False
        challenger = None
    log(f"DECISION: {verdict}")

    # Build output
    out={
        'config':{
            'formula':FORMULA,
            'coins':COINS,
            'N':N,
            'fold_n':FOLD_N,
            'q':Q,'TS':TS,'fund_base':FUND_BASE,'fee_base':FEE_BASE,'fee2x':FEE2X,
            'configs':CONFIGS,
            'note':'mirror run_aa.py leg_series + quantile q0.3 long-only + _vol_scale clamp0.2-2.0 post-stops pre-roll + roll1, 50/50, aster lev2, lth ETC0.88 TRX0.85 sl ETC None TRX0.05 ts24'
        },
        '12fold':{k:{'summary':v['summary'],'folds':v['folds']} for k,v in results12.items()},
        'walkforward':{cfg:{k2:{'summary':v2['summary'],'folds':v2['folds']} for k2,v2 in wf.items()} for cfg,wf in resultsWF.items()},
        'stress_AA_H1best': stress,
        'decision':{'y1b_summary':y1b_summ,'qualify':{'AA_H1best':q_h1,'AA_H2best':q_h2},'verdict':verdict,'adopt':adopt,'challenger':challenger,'thresholds':'12fold median>Y1b AND mean>1.7 AND n_pos>=9 AND WF8 n_pos>=6 => challenger else keep Y1b'},
    }
    # prune folds for lighter display? keep full
    pathlib.Path('results/backtest_E1.json').write_text(json.dumps(out, indent=2, ensure_ascii=False))
    # also write concise summary to log
    log(f"Wrote results/backtest_E1.json decision={verdict}")
    # print for CLI
    print(json.dumps({
        '12fold_summary':{k:v['summary'] for k,v in results12.items()},
        'wf8_summary':{k:resultsWF[k]['WF8']['summary'] for k in CONFIGS},
        'wf24_summary':{k:resultsWF[k]['WF24']['summary'] for k in CONFIGS},
        'stress_worst':{k:{'worst':v['worst_sharpe'],'entry':v['worst_entry']} for k,v in stress.items()},
        'decision':verdict,
        'qualify':{'AA_H1best':q_h1,'AA_H2best':q_h2}
    }, indent=2, ensure_ascii=False))

if __name__=='__main__':
    main()
