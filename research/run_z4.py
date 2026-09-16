
"""Z4 12-fold + funding/fee pressure + time splits for Y2-optimal vs Y1b control."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if os.path.dirname(os.path.abspath(__file__)) else "."))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if os.path.dirname(os.path.abspath(__file__)) else ".")
# Actually ensure cwd is project root
import pathlib as _pl
root = _pl.Path(__file__).parent.parent if _pl.Path(__file__).parent.name=="research" else _pl.Path(".")
os.chdir(str(root))
import json, math, csv
import torch
from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest

FORMULA = [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10]
COINS = ['ETC','TRX']
X5 = {'ETC': (0.88, 0.12, 18, None), 'TRX': (0.85, 0.12, 6, 0.05)}
Q = 0.3
FUND_BASE = 0.0005
FEE_BASE = 0.0004
FEE2X = 0.0008
FULL_N = 6580
FOLD_N = FULL_N // 12  # 548
FUNDS = [0.0003, 0.0005, 0.001]
FEES = [0.0004, 0.0008]

# Y2 optimal: X5 + ts24 + vt0.01 vw12
Y2_SPEC = {
    'ETC': (0.88, 0.12, 18, None, 24, None, 'both', 0.01, 12),
    'TRX': (0.85, 0.12, 6, 0.05, 24, None, 'both', 0.01, 12),
}
# Y1b: X5 + ts24 no vol (b_X5_X4ts24)
Y1B_SPEC = {
    'ETC': (0.88, 0.12, 18, None, 24, None, 'both', None, 12),
    'TRX': (0.85, 0.12, 6, 0.05, 24, None, 'both', None, 12),
}

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
    raw={'open':torch.tensor([[b[0] for b in bars]]),'high':torch.tensor([[b[1] for b in bars]]),'low':torch.tensor([[b[2] for b in bars]]),'close':torch.tensor([[b[3] for b in bars]]),'volume':torch.tensor([[b[4] for b in bars]]),'liquidity':torch.full((1,n),1e7),'fdv':torch.full((1,n),1e8)}
    sig=StackVM(use_advanced=False).execute(FORMULA, FeatureEngineer.compute_features(raw, use_advanced=False))
    rets=[(bars[i+1][3]-bars[i][3])/bars[i][3] for i in range(n-1)]+[0.0]
    return raw, torch.tensor([rets]), sig

def quantile_mask_long(sig,q):
    if q is None: return None
    a=sig.detach().float().abs().reshape(-1)
    k=max(1,int(len(a)*float(q)))
    thr=torch.topk(a,k).values.min()
    return (sig.detach().float().abs()>=thr).float()

def leg_series(raw, rets_t, sig, lth, sth, cd, sl, time_stop=0, take_profit=None, side='both', q=Q, vt=None, vw=12, fee=None, fund=None):
    kw=dict(venue='aster', leverage=2.0, short_enabled=True, long_th=lth, short_th=sth, cooldown_bars=cd, bars_per_year=2190.0, stop_loss=sl, time_stop=time_stop, take_profit=take_profit, vol_target=vt, vol_window=vw)
    kw['funding_override']=FUND_BASE if fund is None else fund
    if fee is not None: kw['fee_override']=fee
    bt=MemeBacktest(**kw)
    signal=torch.sigmoid(sig)
    is_safe=(raw['liquidity']>bt.min_liq).float()
    lp=(signal>bt.long_th).float()*is_safe
    sp=(signal<bt.short_th).float()*is_safe
    mask=quantile_mask_long(sig,q)
    if mask is not None:
        lp=lp*mask
    # cooldown -> stops -> vol_scale matches Y2
    lp,sp=bt._apply_cooldown(lp,sp)
    lp,sp=bt._apply_stops(lp,sp,rets_t)
    scale=bt._vol_scale(rets_t)
    lp,sp=lp*scale, sp*scale
    if side=='long': sp=sp*0.0
    if side=='short': lp=lp*0.0
    lp=lp.roll(1,dims=1); lp[:,0]=0
    sp=sp.roll(1,dims=1); sp[:,0]=0
    turn=(lp - lp.roll(1,dims=1)).abs() + (sp - sp.roll(1,dims=1)).abs()
    tx=turn * (bt.base_fee + torch.clamp(bt.trade_size/(raw['liquidity']+1e-9),0.0,0.05))
    gross=(lp - sp)*rets_t*bt.leverage
    fnd=(lp - sp)*bt.default_funding_rate*bt.leverage
    net=(gross - tx*bt.leverage - fnd)[0].tolist()
    pos=(lp - sp)[0].tolist()
    trades=sum(1 for t in range(len(pos)) if pos[t]!=0.0 and (t==0 or pos[t-1]==0.0))
    pos_rate=sum(1 for v in pos if v!=0.0)/len(pos) if pos else 0
    turnover=sum(turn[0].tolist())/len(turn[0].tolist()) if turn.numel()>0 else 0
    return net, trades, pos_rate, turnover

def stats(ser, trades=0, pos_rate=0, turnover=0):
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
        mdd=max(mdd, peak - cs)
    return {'sharpe':round(sharpe,3),'ann':round(ann,4),'mdd':round(mdd,4),'cum':round(cum,4),'n':n,'trades':trades,'pos_rate':round(pos_rate,4),'turnover':round(turnover,6)}

def combo(ser_list, weights=None):
    m=min(len(s) for s in ser_list)
    k=len(ser_list)
    w=weights or [1.0/k]*k
    sw=sum(w)
    return [sum(ser_list[i][t]*w[i]/sw for i in range(k)) for t in range(m)]

def eval_combo_for_slice(bars_dict, spec, fee, fund):
    # bars_dict: {coin: bars_slice}
    legs=[]
    tr_list=[]; pr_list=[]; to_list=[]
    for c in COINS:
        raw,rt,sg=build_mats(bars_dict[c])
        lth,sth,cd,sl,ts,tp,side,vt,vw=spec[c]
        net,t,pr,to=leg_series(raw,rt,sg,lth,sth,cd,sl,time_stop=ts,take_profit=tp,side=side,q=Q,vt=vt,vw=vw,fee=fee,fund=fund)
        legs.append(net)
        tr_list.append(t); pr_list.append(pr); to_list.append(to)
    cb=combo(legs,[0.5,0.5])
    # cb is avg of legs weighted
    # need pos_rate for combo: approximate as mean of legs
    avg_pr=sum(pr_list)/len(pr_list) if pr_list else 0
    avg_to=sum(to_list)/len(to_list) if to_list else 0
    s=stats(cb, trades=sum(tr_list), pos_rate=avg_pr, turnover=avg_to)
    s['trades_by']={c:tr_list[i] for i,c in enumerate(COINS)}
    return s, cb

def run_for_spec(spec, full_bars):
    # A: 12-fold 548 each. Partition 0:6580 contiguously. Use floor 548; last fold extended to cover remainder
    n=FULL_N
    folds=[]
    # build folds using ceil distribution: fold boundaries = round robin
    # Use simple: fold i = [i*548, (i+1)*548) except last extends to n
    for i in range(12):
        start=i*FOLD_N
        end=(i+1)*FOLD_N if i<11 else n
        # ensure not exceed n
        if end>n: end=n
        # for i<11 end = (i+1)*548; for i=11 covers 6028:6580 = 552
        seg={c: full_bars[c][start:end] for c in COINS}
        s,_=eval_combo_for_slice(seg, spec, fee=FEE_BASE, fund=FUND_BASE)
        s['fold']=i
        s['range']=[start,end]
        folds.append(s)
    # pressure B/C 6 grids
    segB={c: full_bars[c][n-200:n] for c in COINS}
    segC={c: full_bars[c][n-500:n] for c in COINS}
    # also C1 C2 splits
    segC1={c: full_bars[c][6080:6330] for c in COINS}
    segC2={c: full_bars[c][6330:6580] for c in COINS}
    pressure=[]
    for fund in FUNDS:
        for fee in FEES:
            sb,_=eval_combo_for_slice(segB, spec, fee=fee, fund=fund)
            sc,_=eval_combo_for_slice(segC, spec, fee=fee, fund=fund)
            pressure.append({'fund':fund,'fee':fee,'B_sharpe':sb['sharpe'],'B_ann':sb['ann'],'B_mdd':sb['mdd'],'B_trades':sb['trades'],'C_sharpe':sc['sharpe'],'C_ann':sc['ann'],'C_mdd':sc['mdd'],'C_trades':sc['trades']})
    sC1,_=eval_combo_for_slice(segC1, spec, fee=FEE_BASE, fund=FUND_BASE)
    sC2,_=eval_combo_for_slice(segC2, spec, fee=FEE_BASE, fund=FUND_BASE)
    # summary
    sharpes=[f['sharpe'] for f in folds]
    n_pos=sum(1 for x in sharpes if x>0)
    min_sh=min(sharpes) if sharpes else 0
    mean_sh=sum(sharpes)/len(sharpes) if sharpes else 0
    worstB=min(p['B_sharpe'] for p in pressure)
    worstC=min(p['C_sharpe'] for p in pressure)
    # PASS per spec: n_pos>=10 and min>=-1.0 and worstB>0 worstC>1 and C1/C2>0.5
    c1_ok=sC1['sharpe']>0.5
    c2_ok=sC2['sharpe']>0.5
    ratio_ok = (min(sC1['sharpe'], sC2['sharpe'])/max(sC1['sharpe'], sC2['sharpe']) >0.5) if max(sC1['sharpe'], sC2['sharpe'])>0 else False
    # task says C1/C2>0.5 — interpret as both >0.5 and ratio>0.5
    pass_pressure = (worstB>0 and worstC>1)
    pass_folds = (n_pos>=10 and min_sh>=-1.0)
    pass_c = (c1_ok and c2_ok and ratio_ok) if (sC1['sharpe']>0 and sC2['sharpe']>0) else (c1_ok and c2_ok)
    # Actually task states PASS: y2 12-fold n_pos>=10 且 min>=-1.0 且 pressure worstB>0 worstC>1 且 C1/C2>0.5
    # We'll implement C1/C2>0.5 as both >0.5
    pass_c_simple = (sC1['sharpe']>0.5 and sC2['sharpe']>0.5)
    PASS = bool(pass_folds and pass_pressure and pass_c_simple)
    return {
        'folds': folds,
        'folds_summary': {'n_pos':n_pos,'min_sharpe':round(min_sh,3),'mean_sharpe':round(mean_sh,3),'sharpes':sharpes},
        'pressure': pressure,
        'pressure_summary': {'worstB':round(worstB,3),'worstC':round(worstC,3)},
        'C1': sC1,
        'C2': sC2,
        'C_summary': {'C1_sharpe':sC1['sharpe'],'C2_sharpe':sC2['sharpe'],'both_gt0.5':bool(c1_ok and c2_ok),'ratio':round(min(sC1['sharpe'],sC2['sharpe'])/max(sC1['sharpe'],sC2['sharpe']),3) if max(sC1['sharpe'],sC2['sharpe'])!=0 else 0},
        'PASS': PASS,
        'PASS_detail': {'folds_ok':bool(pass_folds),'pressure_ok':bool(pass_pressure),'C_ok':bool(pass_c_simple)}
    }

def main():
    full={c: load_bars(c) for c in COINS}
    n=min(len(v) for v in full.values())
    print(f"full 4h n={n}")
    assert n==FULL_N, f"expected {FULL_N} got {n}"
    y2=run_for_spec(Y2_SPEC, full)
    y1b=run_for_spec(Y1B_SPEC, full)
    comparison={
        'd_sharpe_folds_mean': round(y2['folds_summary']['mean_sharpe'] - y1b['folds_summary']['mean_sharpe'],3),
        'worst_fold_delta': round(y2['folds_summary']['min_sharpe'] - y1b['folds_summary']['min_sharpe'],3),
        'y2_mean': y2['folds_summary']['mean_sharpe'],
        'y1b_mean': y1b['folds_summary']['mean_sharpe'],
        'y2_min': y2['folds_summary']['min_sharpe'],
        'y1b_min': y1b['folds_summary']['min_sharpe']
    }
    config={
        'formula': FORMULA,
        'full_n': FULL_N,
        'fold_size': FOLD_N,
        'fee_base': FEE_BASE,
        'fund_base': FUND_BASE,
        'pressure_grid': {'fund':FUNDS,'fee':FEES},
        'y2_spec': {k: list(v) for k,v in Y2_SPEC.items()},
        'y1b_spec': {k: list(v) for k,v in Y1B_SPEC.items()},
        'q': Q,
        'segments': {'B':[6380,6580],'C':[6080,6580],'C1':[6080,6330],'C2':[6330,6580]},
        'PASS_criteria': 'y2 12-fold n_pos>=10 and min>=-1.0 and pressure worstB>0 worstC>1 and C1/C2>0.5',
        'note': 'Y2=X5+ts24+vt0.01 vw12 (ETC 0.88/0.12/cd18 + TRX 0.85/0.12/cd6/sl0.05 ts24 vol0.01/12) Y1b=X5+ts24 no vol'
    }
    out={'config':config,'y2':y2,'y1b':y1b,'comparison':comparison}
    import pathlib as _p
    _p.Path('results/backtest_Z4.json').write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps({'y2_PASS':y2['PASS'],'y1b_PASS':y1b['PASS'],'comparison':comparison}, indent=2))
    print("Wrote results/backtest_Z4.json")

if __name__=='__main__':
    main()
