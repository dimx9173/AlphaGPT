import csv, json, math, statistics
import torch
from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest
FORMULA = [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10]
BEST = {'AVAX':(0.85,0.15,12,0.03),'BCH':(0.85,0.15,6,0.05),'BTC':(0.88,0.12,6,None),'DOGE':(0.85,0.15,6,0.05),'ETC':(0.88,0.12,12,None),'SHIB':(0.85,0.15,6,None),'SOL':(0.9,0.1,3,0.03),'TRX':(0.85,0.15,6,0.05),'XRP':(0.88,0.12,12,None)}
def load_bars(coin):
    import csv
    rows = list(csv.DictReader(open('data_15m_3y/'+coin+'.csv')))
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
def net_series(raw, rets_t, sig, lth, sth, cd, sl, fee=None, fund=None):
    import torch
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
    lp = lp.roll(1, dims=1); lp[:, 0] = 0
    sp = sp.roll(1, dims=1); sp[:, 0] = 0
    turn = (lp - lp.roll(1, dims=1)).abs() + (sp - sp.roll(1, dims=1)).abs()
    tx = turn * (bt.base_fee + torch.clamp(bt.trade_size / (raw['liquidity'] + 1e-9), 0.0, 0.05))
    gross = (lp - sp) * rets_t * bt.leverage
    fnd = (lp - sp) * bt.default_funding_rate * bt.leverage
    net = (gross - tx * bt.leverage - fnd)[0].tolist()
    t = turn[0].tolist()
    return net, sum(t)/len(t)
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
COINS = sorted(BEST)
print('coins:', COINS, flush=True)
base = {}; turns = {}; cost2 = {}
for coin in COINS:
    bars = load_bars(coin)
    n = len(bars); cut = int(n*0.85)
    erw, eret, esig = build_mats(bars[cut:])
    lth, sth, cd, sl = BEST[coin]
    net, tn = net_series(erw, eret, esig, lth, sth, cd, sl)
    net2, tn2 = net_series(erw, eret, esig, lth, sth, cd, sl, fee=0.0008, fund=0.0005)
    base[coin] = net; cost2[coin] = net2; turns[coin] = tn
    sh, an, md, cu, nn = stats(net)
    sh2, an2, md2, cu2, nn2 = stats(net2)
    print(str(coin)+' turn='+str(round(tn,4))+' base_sh='+str(round(sh,3))+' base_ann='+str(round(an,3))+' cost2_sh='+str(round(sh2,3))+' cost2_ann='+str(round(an2,3)), flush=True)
SETS = {'single_ETC':['ETC'],'single_TRX':['TRX'],'single_BTC':['BTC'],'cross_ETC_TRX_BCH_SOL':['ETC','TRX','BCH','SOL'],'cross_ETC_TRX_XRP_SOL':['ETC','TRX','XRP','SOL'],'cross_ETC_TRX_DOGE_XRP':['ETC','TRX','DOGE','XRP'],'within_meme_DOGE_SHIB':['DOGE','SHIB'],'cross_BTC_TRX_BCH':['BTC','TRX','BCH'],'within_new_SOL_AVAX':['SOL','AVAX']}
print('=== TEST base vs costx2 ===', flush=True)
rows = []
for name, CL in SETS.items():
    m = min(len(base[c]) for c in CL)
    cb = [sum(base[c][t] for c in CL)/len(CL) for t in range(m)]
    c2 = [sum(cost2[c][t] for c in CL)/len(CL) for t in range(m)]
    sh, an, md, cu, nn = stats(cb)
    sh2, an2, md2, cu2, nn2 = stats(c2)
    print(str(name)+' '+str(CL)+' base_sh='+str(round(sh,3))+' base_ann='+str(round(an,3))+' base_dd='+str(round(md,3))+' cost2_sh='+str(round(sh2,3))+' cost2_ann='+str(round(an2,3))+' cost2_dd='+str(round(md2,3)), flush=True)
    rows.append({'name':name,'coins':CL,'base_sharpe':sh,'base_ann':an,'base_mdd':md,'cost2_sharpe':sh2,'cost2_ann':an2,'cost2_mdd':md2,'n':nn})
print('=== leave-one-out on ETC_TRX_BCH_SOL (base cost) ===', flush=True)
FULL = ['ETC','TRX','BCH','SOL']
for drop in FULL:
    CL = [c for c in FULL if c != drop]
    m = min(len(base[c]) for c in CL)
    cb = [sum(base[c][t] for c in CL)/len(CL) for t in range(m)]
    sh, an, md, cu, nn = stats(cb)
    print('drop_'+str(drop)+' '+str(CL)+' sh='+str(round(sh,3))+' ann='+str(round(an,3))+' dd='+str(round(md,3)), flush=True)
print('=== test halves stability (base cost) ===', flush=True)
for name, CL in SETS.items():
    m = min(len(base[c]) for c in CL)
    cb = [sum(base[c][t] for c in CL)/len(CL) for t in range(m)]
    h = m//2
    sh1, an1, md1, cu1, nn1 = stats(cb[:h])
    sh2, an2, md2, cu2, nn2 = stats(cb[h:])
    print(str(name)+' first_half_sh='+str(round(sh1,3))+' ann='+str(round(an1,3))+' | second_half_sh='+str(round(sh2,3))+' ann='+str(round(an2,3)), flush=True)
import json as _j
open('backtest_grouptest_robust_feeonly.json','w').write(_j.dumps(rows, indent=1))
print('saved backtest_grouptest_robust.json', flush=True)
