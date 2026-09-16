import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json, math
import torch
from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest
FORMULA = [3,2,7,2,7,11,15,4,4,6,6,10]
BASE = {'ETC':(0.88,0.12,12,None),'TRX':(0.85,0.15,6,0.05)}
def load_bars(coin):
    import csv
    rows = list(csv.DictReader(open('data/data_15m_3y/'+coin+'.csv')))
    bars = []
    for i in range(0, len(rows), 16):
        blk = rows[i:i+16]
        if len(blk) < 16: break
        bars.append((float(blk[0]['open']), max(float(x['high']) for x in blk), min(float(x['low']) for x in blk), float(blk[-1]['close']), sum(float(x['volume']) for x in blk)))
    return bars
def build_mats(bars):
    import torch
    n = len(bars)
    raw = {'open':torch.tensor([[b[0] for b in bars]]),'high':torch.tensor([[b[1] for b in bars]]),'low':torch.tensor([[b[2] for b in bars]]),'close':torch.tensor([[b[3] for b in bars]]),'volume':torch.tensor([[b[4] for b in bars]]),'liquidity':torch.full((1,n),1e7),'fdv':torch.full((1,n),1e8)}
    sig = StackVM(use_advanced=False).execute(FORMULA, FeatureEngineer.compute_features(raw, use_advanced=False))
    rets = [(bars[i+1][3]-bars[i][3])/bars[i][3] for i in range(n-1)] + [0.0]
    return raw, torch.tensor([rets]), sig
def net_series(raw, rets_t, sig, lth, sth, cd, sl, ts=0):
    import torch
    from model_core.backtest import MemeBacktest
    bt = MemeBacktest(venue='aster', leverage=2.0, short_enabled=True, funding_override=0.0005, long_th=lth, short_th=sth, cooldown_bars=cd, bars_per_year=2190.0, stop_loss=sl, time_stop=ts)
    signal = torch.sigmoid(sig)
    is_safe = (raw['liquidity'] > bt.min_liq).float()
    lp = (signal > bt.long_th).float() * is_safe
    sp = (signal < bt.short_th).float() * is_safe
    lp, sp = bt._apply_cooldown(lp, sp)
    lp, sp = bt._apply_stops(lp, sp, rets_t)
    lp = lp.roll(1, dims=1); lp[:, 0] = 0
    sp = sp.roll(1, dims=1); sp[:, 0] = 0
    turn = (lp - lp.roll(1, dims=1)).abs() + (sp - sp.roll(1, dims=1)).abs()
    tx = turn * (bt.base_fee + torch.clamp(bt.trade_size / (raw['liquidity'] + 1e-9), 0.0, 0.05))
    gross = (lp - sp) * rets_t * bt.leverage
    fnd = (lp - sp) * bt.default_funding_rate * bt.leverage
    net = (gross - tx * bt.leverage - fnd)[0].tolist()
    return net
def stats(ser):
    import math
    n = len(ser); mean = sum(ser)/n
    var = sum((x-mean)**2 for x in ser)/max(n-1,1)
    sharpe = mean/math.sqrt(var)*math.sqrt(2190.0) if var > 0 else 0.0
    cum = sum(ser); ann = cum/n*2190.0
    cs, peak, mdd = 0.0, -1e18, 0.0
    for x in ser:
        cs += x; peak = max(peak, cs); mdd = max(mdd, peak-cs)
    return sharpe, ann, mdd, cum, n
def combo_stats(ser_list, weights=None):
    m = min(len(s) for s in ser_list)
    k = len(ser_list)
    w = weights or [1.0/k]*k
    sw = sum(w)
    cb = [sum(ser_list[i][t]*w[i]/sw for i in range(k)) for t in range(m)]
    return stats(cb) + (m,)
mats = {}
seglen = {}
for coin in ['ETC','TRX']:
    bars = load_bars(coin)
    n = len(bars); cut = int(n*0.85)
    oos = bars[cut:]; h = len(oos)//2
    r1,t1,s1 = build_mats(oos[:h]); r2,t2,s2 = build_mats(oos[h:]); rf,tf,sf = build_mats(bars)
    mats[coin] = {'h1':(r1,t1,s1),'h2':(r2,t2,s2),'full':(rf,tf,sf)}
    seglen[coin] = {'h1':len(oos[:h]),'h2':len(oos[h:]),'full':n}
    print('mats '+coin+' h1='+str(len(oos[:h]))+' h2='+str(len(oos[h:]))+' full='+str(n), flush=True)
# N1 grid
etc_th=[(0.88,0.12),(0.85,0.15),(0.90,0.10)]
trx_th=[(0.85,0.15),(0.88,0.12),(0.80,0.20)]
etc_cd=[6,12,18]; trx_cd=[3,6,12]
n1=[]
for ecd in etc_cd:
    for tcd in trx_cd:
        for (el,es) in etc_th:
            for (tl,ts_) in trx_th:
                row={'etc':{'lth':el,'sth':es,'cd':ecd,'sl':None},'trx':{'lth':tl,'sth':ts_,'cd':tcd,'sl':0.05}}
                line='ETC(%s,%s,cd%d) TRX(%s,%s,cd%d)'%(el,es,ecd,tl,ts_,tcd)
                for seg in ['h1','h2','full']:
                    se=[net_series(*mats['ETC'][seg],el,es,ecd,None), net_series(*mats['TRX'][seg],tl,ts_,tcd,0.05)]
                    sh,an,md,cu,nn,m=combo_stats(se)
                    row[seg]={'sharpe':sh,'ann':an,'mdd':md,'cum':cu,'n':m}
                    line+=' | %s sh=%.3f'%(seg,sh)
                print(line, flush=True)
                n1.append(row)
by_h1=sorted(n1,key=lambda r:r['h1']['sharpe'],reverse=True)
print('--- N1 top3 by H1 ---', flush=True)
for r in by_h1[:3]:
    print('ETC(%s,%s,cd%d) TRX(%s,%s,cd%d) H1=%.3f H2=%.3f FULL=%.3f'%(r['etc']['lth'],r['etc']['sth'],r['etc']['cd'],r['trx']['lth'],r['trx']['sth'],r['trx']['cd'],r['h1']['sharpe'],r['h2']['sharpe'],r['full']['sharpe']), flush=True)
# N2 time-stop sweep on baseline
n2=[]
for ts in [0,6,12,24,48]:
    el,es,ecd,esl=0.88,0.12,12,None
    tl,ts_,tcd,tsl=0.85,0.15,6,0.05
    row={'time_stop':ts}
    line='ts=%d'%ts
    for seg in ['h1','h2','full']:
        se=[net_series(*mats['ETC'][seg],el,es,ecd,esl,ts), net_series(*mats['TRX'][seg],tl,ts_,tcd,tsl,ts)]
        sh,an,md,cu,nn,m=combo_stats(se)
        row[seg]={'sharpe':sh,'ann':an,'mdd':md,'cum':cu,'n':m}
        line+=' | %s sh=%.3f ann=%.3f dd=%.3f'%(seg,sh,an,md)
    print(line, flush=True)
    n2.append(row)
by_h1_n2=sorted(n2,key=lambda r:r['h1']['sharpe'],reverse=True)
print('N2 H1-best ts=%d H1=%.3f H2=%.3f'%(by_h1_n2[0]['time_stop'],by_h1_n2[0]['h1']['sharpe'],by_h1_n2[0]['h2']['sharpe']), flush=True)
out={'formula':FORMULA,'venue':'aster perp 2x fund 0.0005','segments':seglen,'n1':n1,'n1_top3_h1':[{'etc':r['etc'],'trx':r['trx'],'h1':r['h1'],'h2':r['h2'],'full':r['full']} for r in by_h1[:3]],'n1_h1best':{'etc':by_h1[0]['etc'],'trx':by_h1[0]['trx'],'h1':by_h1[0]['h1'],'h2':by_h1[0]['h2'],'full':by_h1[0]['full']},'n2':n2,'n2_h1best':{'time_stop':by_h1_n2[0]['time_stop'],'h1':by_h1_n2[0]['h1'],'h2':by_h1_n2[0]['h2'],'full':by_h1_n2[0]['full']}}
open('results/backtest_iterN1N2.json','w').write(json.dumps(out,indent=1))
print('saved backtest_iterN1N2.json n1=%d n2=%d'%(len(n1),len(n2)), flush=True)
