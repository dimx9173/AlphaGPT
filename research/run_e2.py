import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if os.path.dirname(os.path.abspath(__file__)) else "."))
import pathlib as _pl
root = _pl.Path(__file__).parent.parent if _pl.Path(__file__).parent.name=="research" else _pl.Path(".")
os.chdir(str(root))
import csv, math, json, pathlib
import torch
from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest

FORMULA = [3,2,7,2,7,11,15,4,4,6,6,10]
COINS = ['ETC','TRX']
FUND = 0.0005
FEE_BASE = 0.0004
FEE2X = 0.0008
Q = 0.3
VT_NONE = None
VW = 12
TS = 24
BARS_PER_YEAR = 2190.0

# 8 slices: 6 tail buffet + H2 + FULL
SLICES = {
    'S100': (6480, 6580),   # last-100
    'B':    (6380, 6580),   # last-200
    'S300': (6280, 6580),   # last-300
    'C':    (6080, 6580),   # last-500
    'S800': (5780, 6580),   # last-800
    'S200_2': (6180, 6380), # 倒二 200
    'H2':   (6077, 6570),
    'FULL': (0, 6580),
}
TAIL_KEYS = ['S100','B','S300','C','S800','S200_2']

CONFIGS = [
    {
        'id': 'Y1b',
        'label': 'Y1b(vtNone)',
        'sth': 0.12, 'etc_cd': 18, 'trx_cd': 6,
        'vt': None, 'vw': 12, 'q': 0.3,
        'spec': {
            'ETC': (0.88, 0.12, 18, None, TS, 'both', None, 12),
            'TRX': (0.85, 0.12, 6, 0.05, TS, 'both', None, 12),
        }
    },
    {
        'id': 'Z1',
        'label': 'Z1(0.12/18/6 vt0.012)',
        'sth': 0.12, 'etc_cd': 18, 'trx_cd': 6,
        'vt': 0.012, 'vw': 12, 'q': 0.3,
        'spec': {
            'ETC': (0.88, 0.12, 18, None, TS, 'both', 0.012, 12),
            'TRX': (0.85, 0.12, 6, 0.05, TS, 'both', 0.012, 12),
        }
    },
    {
        'id': 'AA_H1',
        'label': 'AA-H1(0.10/15/9)',
        'sth': 0.10, 'etc_cd': 15, 'trx_cd': 9,
        'vt': 0.012, 'vw': 12, 'q': 0.3,
        'spec': {
            'ETC': (0.88, 0.10, 15, None, TS, 'both', 0.012, 12),
            'TRX': (0.85, 0.10, 9, 0.05, TS, 'both', 0.012, 12),
        }
    },
    {
        'id': 'AA_H2',
        'label': 'AA-H2(0.12/15/9)',
        'sth': 0.12, 'etc_cd': 15, 'trx_cd': 9,
        'vt': 0.012, 'vw': 12, 'q': 0.3,
        'spec': {
            'ETC': (0.88, 0.12, 15, None, TS, 'both', 0.012, 12),
            'TRX': (0.85, 0.12, 9, 0.05, TS, 'both', 0.012, 12),
        }
    },
    {
        'id': 'AB_A2',
        'label': 'AB-A2(Z1 both q0.3 vt0.012)',
        'sth': 0.12, 'etc_cd': 18, 'trx_cd': 6,
        'vt': 0.012, 'vw': 12, 'q': 0.3,
        'spec': {
            'ETC': (0.88, 0.12, 18, None, TS, 'both', 0.012, 12),
            'TRX': (0.85, 0.12, 6, 0.05, TS, 'both', 0.012, 12),
        }
    },
]

def load_bars(coin):
    rows=list(csv.DictReader(open(f'data/data_15m_3y/{coin}.csv')))
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

def leg_series(raw, rets_t, sig, lth, sth, cd, sl, ts, side, vt, vw, fee, fund, q):
    kw=dict(venue='aster', leverage=2.0, short_enabled=True, long_th=lth, short_th=sth, cooldown_bars=cd, bars_per_year=BARS_PER_YEAR, stop_loss=sl, time_stop=ts, vol_target=vt, vol_window=vw)
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
    # scale may be float 1.0 or tensor
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
    turn_list=turn[0].tolist()
    pos=(lp-sp)[0].tolist()
    trades=sum(1 for t in range(len(pos)) if pos[t]!=0.0 and (t==0 or pos[t-1]==0.0))
    turnover=sum(turn_list)/len(turn_list) if turn_list else 0.0
    pos_rate=sum(1 for v in pos if v!=0)/len(pos) if pos else 0
    return {'net':net,'trades':trades,'turnover':turnover,'pos_rate':pos_rate,'pos':pos}

def stats(ser, trades=0, turnover=0.0, pos_rate=0.0, extra=None):
    n=len(ser)
    mean=sum(ser)/n if n else 0
    var=sum((x-mean)**2 for x in ser)/max(n-1,1) if n>1 else 0
    sharpe=mean/math.sqrt(var)*math.sqrt(BARS_PER_YEAR) if var>0 else 0.0
    cum=sum(ser)
    ann=cum/n*BARS_PER_YEAR if n else 0
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

def eval_combo(bars_dict, spec, fee, fund, q):
    legs=[]
    metas=[]
    for c in COINS:
        lth, sth, cd, sl, ts, side, vt, vw = spec[c]
        raw, rt, sg = build_mats(bars_dict[c])
        res = leg_series(raw, rt, sg, lth, sth, cd, sl, ts, side, vt, vw, fee, fund, q)
        legs.append(res['net'])
        metas.append(res)
    cb = combo(legs, [0.5,0.5])
    avg_to = sum(m['turnover'] for m in metas)/len(metas) if metas else 0
    avg_pr = sum(m['pos_rate'] for m in metas)/len(metas) if metas else 0
    s = stats(cb, trades=sum(m['trades'] for m in metas), turnover=avg_to, pos_rate=avg_pr, extra={'trades_by':{COINS[i]:metas[i]['trades'] for i in range(len(COINS))}})
    return s, cb, legs

def main():
    logp=pathlib.Path('logs/e2.log')
    logp.parent.mkdir(parents=True, exist_ok=True)
    outp=pathlib.Path('results/backtest_E2.json')
    outp.parent.mkdir(parents=True, exist_ok=True)
    def log(msg):
        print(msg, flush=True)
        with open(logp,'a') as f:
            f.write(msg+'\n')
    # clear log
    open(logp,'w').write('')
    log('E2 start: 8 slices x 5 configs = 40 cells (each with fee2x)')
    full={c:load_bars(c) for c in COINS}
    n=min(len(v) for v in full.values())
    log(f'full 4h bars n={n} FORMULA {FORMULA} COINS {COINS} fund {FUND} fee {FEE_BASE}/{FEE2X} Q={Q}')
    assert n==6580, f'expected 6580 got {n}'
    for k,(a,b) in SLICES.items():
        log(f'slice {k:6s} [{a}:{b}] n={b-a}')
    # prebuild mats per slice per coin to avoid recompute build_mats many times? We will just slice bars and call eval_combo which builds mats.
    results={}
    heat_sharpe={}
    heat_ann={}
    heat_mdd={}
    heat_cum={}
    heat_trades={}
    heat_turnover={}
    heat_fee2x_sharpe={}
    # judge per config vs Z1 on 6 tails
    for cfg in CONFIGS:
        cid=cfg['id']
        log(f"\n=== CONFIG {cid} {cfg['label']} sth={cfg['sth']} etc_cd={cfg['etc_cd']} trx_cd={cfg['trx_cd']} vt={cfg['vt']} q={cfg['q']} ===")
        spec=cfg['spec']
        q=cfg['q']
        results[cid]={'label':cfg['label'],'sth':cfg['sth'],'etc_cd':cfg['etc_cd'],'trx_cd':cfg['trx_cd'],'vt':cfg['vt'],'vw':cfg['vw'],'q':q,'spec':{k:list(v) if v[3] is not None else [v[0],v[1],v[2],None,v[4],v[5],v[6],v[7]] for k,v in spec.items()},'slices':{}}
        for sname,(a,b) in SLICES.items():
            bars={c: full[c][a:b] for c in COINS}
            s,_cb,_legs = eval_combo(bars, spec, fee=FEE_BASE, fund=FUND, q=q)
            s2,_cb2,_ = eval_combo(bars, spec, fee=FEE2X, fund=FUND, q=q)
            s['fee2x'] = s2
            # extras: fee decay
            s['fee_decay_sharpe'] = round(s['sharpe']-s2['sharpe'],3)
            results[cid]['slices'][sname]=s
            log(f"  {sname:6s} base sh={s['sharpe']:6.3f} ann={s['ann']:7.4f} mdd={s['mdd']:.4f} cum={s['cum']:.4f} tr={s['trades']:3d} to={s['turnover']:.5f} pos={s['pos_rate']:.3f} | fee2x sh={s2['sharpe']:6.3f} ann={s2['ann']:.4f} decay={s['fee_decay_sharpe']:.3f}")
        # fill heat
        heat_sharpe[cid]={k: results[cid]['slices'][k]['sharpe'] for k in SLICES}
        heat_ann[cid]={k: results[cid]['slices'][k]['ann'] for k in SLICES}
        heat_mdd[cid]={k: results[cid]['slices'][k]['mdd'] for k in SLICES}
        heat_cum[cid]={k: results[cid]['slices'][k]['cum'] for k in SLICES}
        heat_trades[cid]={k: results[cid]['slices'][k]['trades'] for k in SLICES}
        heat_turnover[cid]={k: results[cid]['slices'][k]['turnover'] for k in SLICES}
        heat_fee2x_sharpe[cid]={k: results[cid]['slices'][k]['fee2x']['sharpe'] for k in SLICES}

    # heatmap table print
    log("\n=== HEATMAP sharpe (5 configs x 8 slices) ===")
    header = "config".ljust(16) + "".join(k.ljust(10) for k in SLICES.keys())
    log(header)
    log("-"*len(header))
    for cfg in CONFIGS:
        cid=cfg['id']
        row = cid.ljust(16) + "".join(f"{heat_sharpe[cid][k]:6.2f}".ljust(10) for k in SLICES)
        # mark beats Z1
        log(row)
    log("\n=== HEATMAP fee2x sharpe ===")
    header2 = "config".ljust(16) + "".join(k.ljust(10) for k in SLICES.keys())
    log(header2)
    log("-"*len(header2))
    for cfg in CONFIGS:
        cid=cfg['id']
        row = cid.ljust(16) + "".join(f"{heat_fee2x_sharpe[cid][k]:6.2f}".ljust(10) for k in SLICES)
        log(row)
    # judgment vs Z1 on 6 tails
    log("\n=== JUDGMENT vs Z1 on 6 tail slices (S100,B,S300,C,S800,S200_2) ===")
    z1_sh = heat_sharpe['Z1']
    for cfg in CONFIGS:
        if cfg['id']=='Z1':
            continue
        cid=cfg['id']
        beats=[k for k in TAIL_KEYS if heat_sharpe[cid][k] > z1_sh[k]]
        n_beats=len(beats)
        narrow_flag = False
        # regime narrow: only C/H2 win but B/100 collapse
        # check: if C beats but B and S100 lose
        c_beats = 'C' in beats
        b_beats = 'B' in beats
        s100_beats = 'S100' in beats
        if c_beats and not b_beats and not s100_beats:
            narrow_flag=True
        log(f"{cid:8s} beats Z1 on {n_beats}/6 tails: {beats} | broad={n_beats>=5} narrow={narrow_flag} | note={'REGIME廣譜' if n_beats>=5 else ('REGIME窄(僅C/H2勝B/100崩)' if narrow_flag else '無廣譜')}")
        # also fees
    # weakest slice per config
    log("\n=== WEAKEST SLICE per config (min sharpe among 6 tails) ===")
    for cfg in CONFIGS:
        cid=cfg['id']
        vals=[(k, heat_sharpe[cid][k]) for k in TAIL_KEYS]
        vals_sorted=sorted(vals, key=lambda x: x[1])
        weakest=vals_sorted[0]
        log(f"{cid:8s} weakest tail={weakest[0]} sh={weakest[1]:.3f} | ranking tail sorted {vals_sorted}")
    # also weakest fee2x
    log("\n=== WEAKEST FEE2X per config ===")
    for cfg in CONFIGS:
        cid=cfg['id']
        vals=[(k, heat_fee2x_sharpe[cid][k]) for k in TAIL_KEYS]
        weakest=min(vals, key=lambda x: x[1])
        log(f"{cid:8s} weakest fee2x tail={weakest[0]} sh={weakest[1]:.3f}")
    # summary
    log("\n=== BUFFET SUMMARY ===")
    # compute mean tail sharpe per config
    for cfg in CONFIGS:
        cid=cfg['id']
        tail_sh=[heat_sharpe[cid][k] for k in TAIL_KEYS]
        mean_tail=sum(tail_sh)/len(tail_sh)
        fee_tail=[heat_fee2x_sharpe[cid][k] for k in TAIL_KEYS]
        mean_fee=sum(fee_tail)/len(fee_tail)
        log(f"{cid:8s} mean tail sh={mean_tail:.3f} fee2x_mean={mean_fee:.3f} | H2={heat_sharpe[cid]['H2']:.3f} FULL={heat_sharpe[cid]['FULL']:.3f}")

    # build JSON
    verdict={}
    for cfg in CONFIGS:
        if cfg['id']=='Z1':
            continue
        cid=cfg['id']
        beats=[k for k in TAIL_KEYS if heat_sharpe[cid][k] > z1_sh[k]]
        n_beats=len(beats)
        c_beats='C' in beats
        b_beats='B' in beats
        s100_beats='S100' in beats
        narrow = c_beats and not b_beats and not s100_beats
        broad = n_beats>=5
        if broad:
            verdict[cid]='broad'
        elif narrow:
            verdict[cid]='narrow'
        else:
            verdict[cid]='neutral'
    weakest_map={}
    for cfg in CONFIGS:
        cid=cfg['id']
        vals=[(k, heat_sharpe[cid][k]) for k in TAIL_KEYS]
        weakest=min(vals, key=lambda x: x[1])
        vals2=[(k, heat_fee2x_sharpe[cid][k]) for k in TAIL_KEYS]
        weakest2=min(vals2, key=lambda x: x[1])
        weakest_map[cid]={'sharpe':{'slice':weakest[0],'value':weakest[1]},'fee2x':{'slice':weakest2[0],'value':weakest2[1]}}

    out={
        'meta':{'formula': FORMULA, 'coins': COINS, 'slices': SLICES, 'tail_keys': TAIL_KEYS, 'fee_base': FEE_BASE, 'fee2x': FEE2X, 'fund': FUND, 'q': Q, 'bars_per_year': BARS_PER_YEAR, 'n_full': n},
        'configs': results,
        'heatmap':{'sharpe': heat_sharpe, 'fee2x_sharpe': heat_fee2x_sharpe, 'ann': heat_ann, 'mdd': heat_mdd, 'cum': heat_cum, 'trades': heat_trades, 'turnover': heat_turnover},
        'judgment':{'beats_vs_Z1':{cfg['id']:[k for k in TAIL_KEYS if heat_sharpe[cfg['id']][k] > z1_sh[k]] for cfg in CONFIGS if cfg['id']!='Z1'},'beats_count':{cfg['id']:len([k for k in TAIL_KEYS if heat_sharpe[cfg['id']][k] > z1_sh[k]]) for cfg in CONFIGS if cfg['id']!='Z1'},'verdict': verdict,'weakest_slice': weakest_map},
        'slices_order': list(SLICES.keys()),
        'tail_keys': TAIL_KEYS,
    }
    with open(outp,'w') as f:
        json.dump(out,f,indent=2,ensure_ascii=False)
    log(f"\nWritten {outp} with {len(CONFIGS)} configs x {len(SLICES)} slices")

if __name__=='__main__':
    main()
