import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import csv, json, math, statistics
import torch
from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest
FORMULA = [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10]
BEST = {'ETC':(0.88,0.12,12,None),'TRX':(0.85,0.15,6,0.05),'BCH':(0.85,0.15,6,0.05),'SOL':(0.9,0.1,3,0.03),'DOGE':(0.85,0.15,6,0.05),'XRP':(0.88,0.12,12,None),'BTC':(0.88,0.12,6,None),'SHIB':(0.85,0.15,6,None),'AVAX':(0.85,0.15,12,0.03)}
CANDS = {'A_ETC_TRX_BCH':['ETC','TRX','BCH'],'B_ETC_TRX_BCH_SOL':['ETC','TRX','BCH','SOL'],'C_ETC_TRX_SOL':['ETC','TRX','SOL'],'D_ETC_TRX_DOGE_XRP':['ETC','TRX','DOGE','XRP'],'E_DOGE_SHIB':['DOGE','SHIB'],'F_ETC':['ETC'],'G_TRX':['TRX']}
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
def net_series(raw, rets_t, sig, lth, sth, cd, sl):
    import torch
    bt = MemeBacktest(venue='aster', leverage=2.0, short_enabled=True, funding_override=0.0005, long_th=lth, short_th=sth, cooldown_bars=cd, bars_per_year=2190.0, stop_loss=sl)
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
series = {}
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
mats = {}
for coin in sorted(BEST):
    bars = load_bars(coin)
    n = len(bars); cut = int(n*0.85)
    oos = bars[cut:]; h = len(oos)//2
    r1, t1, s1 = build_mats(oos[:h])
    r2, t2, s2 = build_mats(oos[h:])
    rf, tf, sf = build_mats(bars)
    lth, sth, cd, sl = BEST[coin]
    mats[coin] = {'h1':(r1,t1,s1),'h2':(r2,t2,s2),'full':(rf,tf,sf),'params':(lth,sth,cd,sl)}
    print('mats '+coin+' h1='+str(len(oos[:h]))+' h2='+str(len(oos[h:]))+' full='+str(n), flush=True)
def combo_stats(ser_list, weights=None):
    m = min(len(s) for s in ser_list)
    k = len(ser_list)
    w = weights or [1.0/k]*k
    sw = sum(w)
    cb = [sum(ser_list[i][t]*w[i]/sw for i in range(k)) for t in range(m)]
    return stats(cb) + (m,)
import json as _j
print('=== R6 weight sweep ETC:TRX (H1/H2/FULL) ===', flush=True)
r6 = []
for wi in [0,10,20,30,40,50,60,70,80,90,100]:
    w = wi/100.0
    outs = {}
    line = 'wETC='+str(round(w,1))
    for seg in ['h1','h2','full']:
        se = [net_series_side(*mats['ETC'][seg], *mats['ETC']['params']), net_series_side(*mats['TRX'][seg], *mats['TRX']['params'])]
        sh, an, md, cu, nn, m = combo_stats(se, [w, 1-w])
        outs[seg] = {'sharpe':sh,'ann':an,'mdd':md}
        line += ' | '+seg+' sh='+str(round(sh,3))+' dd='+str(round(md,3))
    print(line, flush=True)
    r6.append({'wETC':w, **outs})
open('results/backtest_grouptest6_weight.json','w').write(_j.dumps(r6, indent=1))
print('=== R7 third leg addon to ETC+TRX (equal, H1/H2/FULL) ===', flush=True)
r7 = []
for X in ['SOL','SHIB','AVAX','DOGE','BCH','XRP','BTC']:
    CL = ['ETC','TRX',X]
    outs = {}
    line = 'ETC+TRX+'+X
    for seg in ['h1','h2','full']:
        se = [net_series_side(*mats[c][seg], *mats[c]['params']) for c in CL]
        sh, an, md, cu, nn, m = combo_stats(se)
        outs[seg] = {'sharpe':sh,'ann':an,'mdd':md}
        line += ' | '+seg+' sh='+str(round(sh,3))+' dd='+str(round(md,3))
    print(line, flush=True)
    r7.append({'leg':X, 'coins':CL, **outs})
open('results/backtest_grouptest7_thirdleg.json','w').write(_j.dumps(r7, indent=1))
print('=== R8 long/short decomposition (H2 + FULL) ===', flush=True)
r8 = []
for item, CL in [('ETC',['ETC']),('TRX',['TRX']),('ETC_TRX',['ETC','TRX'])]:
    line = item
    outs = {}
    for seg in ['h2','full']:
        for side in ['long','short','both']:
            se = [net_series_side(*mats[c][seg], *mats[c]['params'], side=side) for c in CL]
            sh, an, md, cu, nn, m = combo_stats(se)
            outs[seg+'_'+side] = {'sharpe':round(sh,3),'ann':round(an,3),'mdd':round(md,3)}
        line += ' | '+seg+' L='+str(outs[seg+'_long']['sharpe'])+' S='+str(outs[seg+'_short']['sharpe'])+' B='+str(outs[seg+'_both']['sharpe'])
    print(line, flush=True)
    r8.append({'item':item, **outs})
open('results/backtest_grouptest8_side.json','w').write(_j.dumps(r8, indent=1))
print('=== R9 fee sweep ETC+TRX eq (fund fixed, H2 + FULL) ===', flush=True)
r9 = []
for fee in [0.0002,0.0004,0.0008,0.0012]:
    line = 'fee='+str(fee)
    outs = {}
    for seg in ['h2','full']:
        se = [net_series_side(*mats[c][seg], *mats[c]['params'], fee=fee, fund=0.0005) for c in ['ETC','TRX']]
        sh, an, md, cu, nn, m = combo_stats(se)
        outs[seg] = {'sharpe':sh,'ann':an,'mdd':md}
        line += ' | '+seg+' sh='+str(round(sh,3))+' ann='+str(round(an,3))
    print(line, flush=True)
    r9.append({'fee':fee, **outs})
open('results/backtest_grouptest9_fee.json','w').write(_j.dumps(r9, indent=1))
print('=== R10 final lock: ETC+TRX eq, 6-fold WF full history ===', flush=True)
se_full = [net_series_side(*mats[c]['full'], *mats[c]['params']) for c in ['ETC','TRX']]
sh, an, md, cu, nn, m = combo_stats(se_full)
print('full sh='+str(round(sh,3))+' ann='+str(round(an,3))+' dd='+str(round(md,3))+' n='+str(m), flush=True)
folds = []
q = m//6
for k in range(6):
    seg = [s[k*q:(k+1)*q] if k < 5 else s[k*q:] for s in se_full]
    sh2, an2, md2, cu2, nn2, m2 = combo_stats(seg)
    folds.append({'fold':k,'sharpe':round(sh2,3),'ann':round(an2,3),'mdd':round(md2,3),'n':m2})
    print('fold'+str(k)+' sh='+str(round(sh2,3))+' ann='+str(round(an2,3))+' dd='+str(round(md2,3)), flush=True)
open('results/backtest_grouptest10_final.json','w').write(_j.dumps({'full':{'sharpe':sh,'ann':an,'mdd':md,'n':m},'folds':folds}, indent=1))
print('saved R6-R10 jsons', flush=True)
