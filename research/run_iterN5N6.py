
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json, math
import torch
from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest
FORMULA = [3,2,7,2,7,11,15,4,4,6,6,10]
ETC_BASE = (0.88,0.12,12,None)
TRX_BASE = (0.85,0.15,6,0.05)
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
def net_series_side(raw, rets_t, sig, lth, sth, cd, sl, fee=None, fund=None, side='both'):
    import torch
    from model_core.backtest import MemeBacktest
    kw = dict(venue='aster', leverage=2.0, short_enabled=True, long_th=lth, short_th=sth, cooldown_bars=cd, bars_per_year=2190.0, stop_loss=sl)
    kw['funding_override'] = 0.0005 if fund is None else fund
    if fee is not None: kw['fee_override'] = fee
    bt = MemeBacktest(**kw)
    signal = torch.sigmoid(sig)
    is_safe = (raw['liquidity'] > bt.min_liq).float()
    lp = (signal > bt.long_th).float() * is_safe
    sp = (signal < bt.short_th).float() * is_safe
    lp, sp = bt._apply_cooldown(lp, sp)
    lp, sp = bt._apply_stops(lp, sp, rets_t)
    if side == 'long': sp = sp * 0.0
    if side == 'short': lp = lp * 0.0
    lp = lp.roll(1, dims=1); lp[:, 0] = 0
    sp = sp.roll(1, dims=1); sp[:, 0] = 0
    turn = (lp - lp.roll(1, dims=1)).abs() + (sp - sp.roll(1, dims=1)).abs()
    tx = turn * (bt.base_fee + torch.clamp(bt.trade_size / (raw['liquidity'] + 1e-9), 0.0, 0.05))
    gross = (lp - sp) * rets_t * bt.leverage
    fnd = (lp - sp) * bt.default_funding_rate * bt.leverage
    net = (gross - tx * bt.leverage - fnd)[0].tolist()
    return net
def pos_counts(raw, rets_t, sig, lth, sth, cd, sl, side='both'):
    import torch
    from model_core.backtest import MemeBacktest
    bt = MemeBacktest(venue='aster', leverage=2.0, short_enabled=True, funding_override=0.0005, long_th=lth, short_th=sth, cooldown_bars=cd, bars_per_year=2190.0, stop_loss=sl)
    signal = torch.sigmoid(sig)
    is_safe = (raw['liquidity'] > bt.min_liq).float()
    lp = (signal > bt.long_th).float() * is_safe
    sp = (signal < bt.short_th).float() * is_safe
    lp, sp = bt._apply_cooldown(lp, sp)
    lp, sp = bt._apply_stops(lp, sp, rets_t)
    if side == 'long': sp = sp * 0.0
    if side == 'short': lp = lp * 0.0
    lp = lp.roll(1, dims=1); lp[:, 0] = 0
    sp = sp.roll(1, dims=1); sp[:, 0] = 0
    lp0 = lp[0].tolist(); sp0 = sp[0].tolist()
    def entries(v):
        c=0
        prev=0
        for x in v:
            xi=int(round(x))
            if xi==1 and prev==0: c+=1
            prev=xi
        return c
    return {'long_bars':int(sum(round(x) for x in lp0)), 'short_bars':int(sum(round(x) for x in sp0)), 'long_entries':entries(lp0), 'short_entries':entries(sp0), 'n':len(lp0)}
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
for coin in ['ETC','TRX']:
    bars = load_bars(coin)
    n = len(bars); cut = int(n*0.85)
    oos = bars[cut:]; h = len(oos)//2
    r1, t1, s1 = build_mats(oos[:h])
    r2, t2, s2 = build_mats(oos[h:])
    rf, tf, sf = build_mats(bars)
    mats[coin] = {'h1':(r1,t1,s1),'h2':(r2,t2,s2),'full':(rf,tf,sf)}
    print('mats '+coin+' h1='+str(len(oos[:h]))+' h2='+str(len(oos[h:]))+' full='+str(n), flush=True)
out = {'meta':{'formula':FORMULA,'etc_base':list([0.88,0.12,12,None]),'trx_base':list([0.85,0.15,6,0.05]),'combo':'equal-weight ETC+TRX','protocol':'H1 select / H2 verify + FULL'}}
# N5a short_th sweep
print('=== N5a TRX short_th sweep (lth=0.85 fixed) ===', flush=True)
n5a=[]
for sth in [0.08,0.10,0.15,0.20,0.25]:
    tp=(0.85,sth,6,0.05)
    row={'trx':list([0.85,sth,6,0.05])}
    line='sth='+str(sth)
    for seg in ['h1','h2','full']:
        se=[net_series_side(*mats['ETC'][seg], *ETC_BASE), net_series_side(*mats['TRX'][seg], *tp)]
        sh,an,md,cu,nn,m=combo_stats(se)
        row[seg]={'sharpe':sh,'ann':an,'mdd':md,'cum':cu}
        line+=' | '+seg+' sh='+str(round(sh,3))+' ann='+str(round(an,3))+' dd='+str(round(md,3))
    print(line, flush=True)
    n5a.append(row)
out['N5a_short_sweep_lth085']=n5a
# H1-best by sharpe
h1best=max(n5a, key=lambda r: r['h1']['sharpe'])
best_sth=h1best['trx'][1]
print('H1-best short_th='+str(best_sth)+' sh='+str(round(h1best['h1']['sharpe'],3)), flush=True)
out['N5a_h1best_short']=best_sth
# N5b long_th sweep with short fixed at H1-best
print('=== N5b TRX long_th sweep (sth='+str(best_sth)+' fixed) ===', flush=True)
n5b=[]
for lth in [0.80,0.85,0.88,0.92]:
    tp=(lth,best_sth,6,0.05)
    row={'trx':list([lth,best_sth,6,0.05])}
    line='lth='+str(lth)
    for seg in ['h1','h2','full']:
        se=[net_series_side(*mats['ETC'][seg], *ETC_BASE), net_series_side(*mats['TRX'][seg], *tp)]
        sh,an,md,cu,nn,m=combo_stats(se)
        row[seg]={'sharpe':sh,'ann':an,'mdd':md,'cum':cu}
        line+=' | '+seg+' sh='+str(round(sh,3))+' ann='+str(round(an,3))+' dd='+str(round(md,3))
    print(line, flush=True)
    n5b.append(row)
out['N5b_long_sweep_sthfix']=n5b
out['N5b_sth_fixed']=best_sth
h1best2=max(n5b, key=lambda r: r['h1']['sharpe'])
print('H1-best long_th='+str(h1best2['trx'][0])+' sh='+str(round(h1best2['h1']['sharpe'],3)), flush=True)
out['N5b_h1best_long']=h1best2['trx'][0]
# N6 side comparison
print('=== N6 side: both vs short-only vs long-only ===', flush=True)
n6=[]
for item, coins, params in [('ETC',['ETC'],[ETC_BASE]),('TRX',['TRX'],[TRX_BASE]),('ETC_TRX',['ETC','TRX'],[ETC_BASE,TRX_BASE])]:
    row={'item':item}
    line=item
    for seg in ['h1','h2','full']:
        for side in ['both','short','long']:
            se=[net_series_side(*mats[c][seg], *p, side=side) for c,p in zip(coins,params)]
            sh,an,md,cu,nn,m=combo_stats(se)
            cnts=[pos_counts(*mats[c][seg], *p, side=side) for c,p in zip(coins,params)]
            tot={'long_bars':sum(c['long_bars'] for c in cnts),'short_bars':sum(c['short_bars'] for c in cnts),'long_entries':sum(c['long_entries'] for c in cnts),'short_entries':sum(c['short_entries'] for c in cnts),'n':cnts[0]['n']}
            row[seg+'_'+side]={'sharpe':sh,'ann':an,'mdd':md,'cum':cu,'counts':tot}
        line+=' | '+seg+' B='+str(round(row[seg+'_both']['sharpe'],3))+' S='+str(round(row[seg+'_short']['sharpe'],3))+' L='+str(round(row[seg+'_long']['sharpe'],3))
    print(line, flush=True)
    for seg in ['h1','h2','full']:
        for side in ['both','short','long']:
            c=row[seg+'_'+side]['counts']
            print('  '+seg+'_'+side+' ann='+str(round(row[seg+'_'+side]['ann'],3))+' dd='+str(round(row[seg+'_'+side]['mdd'],3))+' LB='+str(c['long_bars'])+' SB='+str(c['short_bars'])+' LE='+str(c['long_entries'])+' SE='+str(c['short_entries']), flush=True)
    n6.append(row)
out['N6_side']=n6
open('results/backtest_iterN5N6.json','w').write(json.dumps(out, indent=1))
print('saved backtest_iterN5N6.json', flush=True)
