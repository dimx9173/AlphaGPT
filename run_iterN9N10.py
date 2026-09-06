import json, math
import torch
from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest
FORMULA = [3,2,7,2,7,11,15,4,4,6,6,10]
BEST = {'ETC':(0.88,0.12,12,None),'TRX':(0.85,0.15,6,0.05),'AVAX':(0.85,0.15,12,0.03),'SHIB':(0.85,0.15,6,None),'DOGE':(0.85,0.15,6,0.05),'SOL':(0.9,0.1,3,0.03)}
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
def leg_out(raw, rets_t, sig, lth, sth, cd, sl, fee=None, fund=None):
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
for coin in sorted(BEST):
    bars = load_bars(coin)
    n = len(bars); cut = int(n*0.85)
    oos = bars[cut:]; h = len(oos)//2
    r1, t1, s1 = build_mats(oos[:h])
    r2, t2, s2 = build_mats(oos[h:])
    rf, tf, sf = build_mats(bars)
    mats[coin] = {'h1':(r1,t1,s1),'h2':(r2,t2,s2),'full':(rf,tf,sf),'params':BEST[coin],'n_full':n,'n_h1':len(oos[:h]),'n_h2':len(oos[h:])}
    print('mats '+coin+' h1='+str(len(oos[:h]))+' h2='+str(len(oos[h:]))+' full='+str(n), flush=True)
# ---------- N9 ----------
out = {'N9':{'singles':{},'pair_ETC_TRX':{}}, 'N10':{}}
for coin in ['AVAX','SHIB','DOGE','SOL','ETC','TRX']:
    d = {}
    for seg in ['h1','h2','full']:
        net, to = leg_out(*mats[coin][seg], *mats[coin]['params'])
        sh, an, md, cu, nn = stats(net)
        d[seg] = {'sharpe':round(sh,3),'ann':round(an,3),'mdd':round(md,3),'cum':round(cu,4),'n':nn}
    out['N9']['singles'][coin] = d
    print('N9 '+coin+' h1=%s h2=%s full=%s' % (d['h1']['sharpe'], d['h2']['sharpe'], d['full']['sharpe']), flush=True)
for seg in ['h1','h2','full']:
    se = [leg_out(*mats[c][seg], *mats[c]['params'])[0] for c in ['ETC','TRX']]
    sh, an, md, cu, nn, m = combo_stats(se)
    out['N9']['pair_ETC_TRX'][seg] = {'sharpe':round(sh,3),'ann':round(an,3),'mdd':round(md,3),'cum':round(cu,4),'n':m}
    print('N9 pair ETC+TRX '+seg+' sh='+str(round(sh,3)), flush=True)
h2pair = out['N9']['pair_ETC_TRX']['h2']['sharpe']
out['N9']['guard_pass'] = all(out['N9']['singles'][c]['h2']['sharpe'] < h2pair for c in ['AVAX','SHIB','DOGE','SOL'])
print('N9 guard_pass='+str(out['N9']['guard_pass']), flush=True)
# ---------- N10 ----------
se_full = [leg_out(*mats[c]['full'], *mats[c]['params'])[0] for c in ['ETC','TRX']]
sh, an, md, cu, nn, m = combo_stats(se_full)
out['N10']['full'] = {'sharpe':round(sh,3),'ann':round(an,3),'mdd':round(md,3),'cum':round(cu,4),'n':m}
print('N10 full sh=%s ann=%s dd=%s n=%s' % (round(sh,3),round(an,3),round(md,3),m), flush=True)
folds = []
q = m//6
for k in range(6):
    seg = [s[k*q:(k+1)*q] if k < 5 else s[k*q:] for s in se_full]
    sh2, an2, md2, cu2, nn2, m2 = combo_stats(seg)
    folds.append({'fold':k,'sharpe':round(sh2,3),'ann':round(an2,3),'mdd':round(md2,3),'cum':round(cu2,4),'n':m2})
    print('fold%d sh=%s ann=%s dd=%s n=%s' % (k,round(sh2,3),round(an2,3),round(md2,3),m2), flush=True)
out['N10']['folds'] = folds
fees = []
for fee in [0.0004, 0.0008]:
    row = {'fee':fee}
    for seg in ['h2','full']:
        se = [leg_out(*mats[c][seg], *mats[c]['params'], fee=fee, fund=0.0005)[0] for c in ['ETC','TRX']]
        sh3, an3, md3, cu3, nn3, m3 = combo_stats(se)
        row[seg] = {'sharpe':round(sh3,3),'ann':round(an3,3),'mdd':round(md3,3),'cum':round(cu3,4),'n':m3}
    print('fee %s h2_sh=%s full_sh=%s' % (fee, row['h2']['sharpe'], row['full']['sharpe']), flush=True)
    fees.append(row)
out['N10']['fee_sweep'] = fees
to = {}
for c in ['ETC','TRX']:
    net, tlist = leg_out(*mats[c]['full'], *mats[c]['params'])
    mu = sum(tlist)/len(tlist)
    to[c] = {'mean_per_bar':round(mu,5),'annualized':round(mu*2190.0,1),'n':len(tlist)}
to['pair_avg_annualized'] = round((to['ETC']['annualized']+to['TRX']['annualized'])/2,1)
out['N10']['turnover'] = to
print('turnover '+str(to), flush=True)
out['meta'] = {'formula':FORMULA,'venue':'aster','leverage':2.0,'funding':0.0005,'fee_baseline':'aster default 0.0004','split':'85% cut, OOS halves = H1/H2; FULL=6570 bars 4h','pair':'ETC+TRX 50/50 equal-weight, no tweaks'}
json.dump(out, open('backtest_iterN9N10.json','w'), indent=1)
print('saved backtest_iterN9N10.json', flush=True)
