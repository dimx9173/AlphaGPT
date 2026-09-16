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
    sig = StackVM(use_advanced=False).execute(FORMULA, FeatureEngineer.compute_features(raw, use_advanced=False))
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
for coin in sorted(BEST):
    bars = load_bars(coin)
    n = len(bars); cut = int(n*0.85)
    oos = bars[cut:]
    h = len(oos)//2
    r1, t1, s1 = build_mats(oos[:h])
    r2, t2, s2 = build_mats(oos[h:])
    lth, sth, cd, sl = BEST[coin]
    series[coin] = (net_series(r1, t1, s1, lth, sth, cd, sl), net_series(r2, t2, s2, lth, sth, cd, sl))
    full_raw, full_tret, full_sig = build_mats(bars)
    series[coin + '_FULL'] = net_series(full_raw, full_tret, full_sig, lth, sth, cd, sl)
print('=== H1 select / H2 verify (equal + sharpe-weighted) ===', flush=True)
CANDS2 = {'Z_ETC_TRX':['ETC','TRX'],'C_ETC_TRX_SOL':['ETC','TRX','SOL'],'A_ETC_TRX_BCH':['ETC','TRX','BCH'],'B_ETC_TRX_BCH_SOL':['ETC','TRX','BCH','SOL'],'F_ETC':['ETC'],'G_TRX':['TRX']}
H1SH = {'ETC':4.3857813905187415,'TRX':2.58986933638759,'BCH':5.614,'SOL':1.18}
rows = []
for name, CL in CANDS2.items():
    m1 = min(len(series[c][0]) for c in CL)
    m2 = min(len(series[c][1]) for c in CL)
    cb1 = [sum(series[c][0][t] for c in CL)/len(CL) for t in range(m1)]
    cb2 = [sum(series[c][1][t] for c in CL)/len(CL) for t in range(m2)]
    sh1, an1, md1, cu1, nn1 = stats(cb1)
    sh2, an2, md2, cu2, nn2 = stats(cb2)
    w = [max(H1SH[c],0.01) for c in CL]
    sw = sum(w)
    wb1 = [sum(series[c][0][t]*w[i]/sw for i,c in enumerate(CL)) for t in range(m1)]
    wb2 = [sum(series[c][1][t]*w[i]/sw for i,c in enumerate(CL)) for t in range(m2)]
    wsh1, wan1, wmd1, wcu1, wnn1 = stats(wb1)
    wsh2, wan2, wmd2, wcu2, wnn2 = stats(wb2)
    print(str(name)+' '+str(CL)+' EQ:H1='+str(round(sh1,3))+' H2='+str(round(sh2,3))+' dd='+str(round(md2,3))+' | W:H1='+str(round(wsh1,3))+' H2='+str(round(wsh2,3))+' dd='+str(round(wmd2,3)), flush=True)
    rows.append({'name':name,'coins':CL,'eq_h1':sh1,'eq_h2':sh2,'eq_h2_mdd':md2,'w_h1':wsh1,'w_h2':wsh2,'w_h2_mdd':wmd2})
import json as _j
open('results/backtest_grouptest5_pw.json','w').write(_j.dumps(rows, indent=1))
print('saved backtest_grouptest5_pw.json', flush=True)
print('=== walk-forward 4 folds full history ===', flush=True)
for name, CL in CANDS2.items():
    f = min(len(series[c+'_FULL']) for c in CL)
    combo = [sum(series[c+'_FULL'][t] for c in CL)/len(CL) for t in range(f)]
    q = f//4
    outs = []
    for k in range(4):
        seg = combo[k*q:(k+1)*q] if k < 3 else combo[k*q:]
        sh, an, md, cu, nn = stats(seg)
        outs.append(str(round(sh,2)))
    sh, an, md, cu, nn = stats(combo)
    print(str(name)+' full_sh='+str(round(sh,3))+' ann='+str(round(an,3))+' dd='+str(round(md,3))+' folds='+str(outs), flush=True)
