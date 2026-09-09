import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json, math
import torch
from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest
FORMULA = [3,2,7,2,7,11,15,4,4,6,6,10]
BEST = {'ETC':(0.88,0.12,12,None),'TRX':(0.85,0.15,6,0.05)}
BPY = 2190.0
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
    sig = StackVM().execute(FORMULA, FeatureEngineer.compute_features(raw))
    rets = [(bars[i+1][3]-bars[i][3])/bars[i][3] for i in range(n-1)] + [0.0]
    return raw, torch.tensor([rets]), sig
def net_series(raw, rets_t, sig, lth, sth, cd, sl, tp=None, lev=2.0):
    import torch
    from model_core.backtest import MemeBacktest
    bt = MemeBacktest(venue='aster', leverage=lev, short_enabled=True, funding_override=0.0005, long_th=lth, short_th=sth, cooldown_bars=cd, bars_per_year=BPY, stop_loss=sl, take_profit=tp)
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
    to = turn[0].tolist()
    return net, to
def stats(ser):
    import math
    n = len(ser); mean = sum(ser)/n
    var = sum((x-mean)**2 for x in ser)/max(n-1,1)
    sharpe = mean/math.sqrt(var)*math.sqrt(BPY) if var > 0 else 0.0
    cum = sum(ser); ann = cum/n*BPY
    cs, peak, mdd = 0.0, -1e18, 0.0
    for x in ser:
        cs += x; peak = max(peak, cs); mdd = max(mdd, peak-cs)
    return sharpe, ann, mdd, cum, n
def combo_stats(ser_list, to_list=None, weights=None):
    m = min(len(s) for s in ser_list)
    k = len(ser_list)
    w = weights or [1.0/k]*k
    sw = sum(w)
    cb = [sum(ser_list[i][t]*w[i]/sw for i in range(k)) for t in range(m)]
    sh, an, md, cu, nn = stats(cb)
    to = None
    if to_list is not None:
        mt = min(len(s) for s in to_list)
        tcb = [sum(to_list[i][t]*w[i]/sw for i in range(k)) for t in range(mt)]
        to = sum(tcb)/len(tcb)*BPY
    return sh, an, md, cu, nn, m, to
mats = {}
for coin in sorted(BEST):
    bars = load_bars(coin)
    n = len(bars); cut = int(n*0.85)
    oos = bars[cut:]; h = len(oos)//2
    mats[coin] = {'h1':build_mats(oos[:h]),'h2':build_mats(oos[h:]),'full':build_mats(bars),'params':BEST[coin]}
    print('mats '+coin+' h1=%d h2=%d full=%d' % (len(oos[:h]), len(oos[h:]), n), flush=True)
CLS = ['ETC','TRX']
# N3: SL x TP grid, lev 2.0
print('=== N3 SL/TP sweep ETC+TRX eq lev2 ===', flush=True)
n3 = []
for sl in [None,0.03,0.05,0.08]:
    for tp in [None,0.08,0.15]:
        outs = {}; line = 'SL=%s TP=%s' % (sl,tp)
        for seg in ['h1','h2','full']:
            ser, tos = [], []
            for c in CLS:
                lth,sth,cd,_ = mats[c]['params']
                s,t = net_series(*mats[c][seg], lth,sth,cd,sl,tp,2.0)
                ser.append(s); tos.append(t)
            sh,an,md,cu,nn,m,to = combo_stats(ser,tos)
            outs[seg] = {'sharpe':round(sh,4),'ann':round(an,4),'mdd':round(md,4),'cum':round(cu,4),'n':m,'turnover':round(to,1)}
            line += ' | %s sh=%.3f ann=%.3f dd=%.3f to=%.0f' % (seg,sh,an,md,to)
        print(line, flush=True)
        n3.append({'sl':sl,'tp':tp,**outs})
n3s = sorted(n3, key=lambda r: r['h1']['sharpe'], reverse=True)
print('--- N3 top3 by H1 sharpe ---', flush=True)
for r in n3s[:3]:
    print('SL=%s TP=%s H1sh=%.3f H2sh=%.3f FULLsh=%.3f H2ann=%.3f H2dd=%.3f' % (r['sl'],r['tp'],r['h1']['sharpe'],r['h2']['sharpe'],r['full']['sharpe'],r['h2']['ann'],r['h2']['mdd']), flush=True)
# N4: leverage sweep on H1-best N3 config
best = n3s[0]
print('=== N4 leverage sweep on SL=%s TP=%s ===' % (best['sl'],best['tp']), flush=True)
n4 = []
for lev in [1.0,1.5,2.0,2.5,3.0]:
    outs = {}; line = 'lev=%.1f' % lev
    for seg in ['h1','h2','full']:
        ser, tos = [], []
        for c in CLS:
            lth,sth,cd,_ = mats[c]['params']
            s,t = net_series(*mats[c][seg], lth,sth,cd,best['sl'],best['tp'],lev)
            ser.append(s); tos.append(t)
        sh,an,md,cu,nn,m,to = combo_stats(ser,tos)
        outs[seg] = {'sharpe':round(sh,4),'ann':round(an,4),'mdd':round(md,4),'cum':round(cu,4),'n':m,'turnover':round(to,1)}
        line += ' | %s sh=%.3f ann=%.3f dd=%.3f' % (seg,sh,an,md)
    print(line, flush=True)
    n4.append({'leverage':lev,**outs})
json.dump({'N3':n3,'N3_top3_H1':n3s[:3],'N3_best':{'sl':best['sl'],'tp':best['tp']},'N4':n4}, open('results/backtest_iterN3N4.json','w'), indent=1)
print('saved backtest_iterN3N4.json', flush=True)
