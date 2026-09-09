"""R2 pair-construction: short-base (baseline ETC+TRX short-only) + long-satellite (Q2 cand BTC+ETC long-only)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json, math
import torch
from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest

BASELINE = [3,2,7,2,7,11,15,4,4,6,6,10]
Q2CAND   = [0,20,18,14,20,12,14,18,11,14,14,14]

SHORT_PARAMS = {'ETC':(0.88,0.12,12,None), 'TRX':(0.85,0.15,6,0.05)}
LONG_PARAMS  = {'BTC':(0.85,0.15,6,None), 'ETC':(0.85,0.15,6,None)}
FEE, FUND, LEV = 0.0004, 0.0005, 2.0

def load_bars(coin):
    import csv
    rows = list(csv.DictReader(open('data/data_15m_3y/'+coin+'.csv')))
    bars = []
    for i in range(0, len(rows), 16):
        blk = rows[i:i+16]
        if len(blk) < 16: break
        bars.append((float(blk[0]['open']), max(float(x['high']) for x in blk), min(float(x['low']) for x in blk), float(blk[-1]['close']), sum(float(x['volume']) for x in blk)))
    return bars

def build_mats(bars, formula):
    n = len(bars)
    raw = {'open':torch.tensor([[b[0] for b in bars]]),'high':torch.tensor([[b[1] for b in bars]]),'low':torch.tensor([[b[2] for b in bars]]),'close':torch.tensor([[b[3] for b in bars]]),'volume':torch.tensor([[b[4] for b in bars]]),'liquidity':torch.full((1,n),1e7),'fdv':torch.full((1,n),1e8)}
    sig = StackVM().execute(formula, FeatureEngineer.compute_features(raw))
    rets = [(bars[i+1][3]-bars[i][3])/bars[i][3] for i in range(n-1)] + [0.0]
    return raw, torch.tensor([rets]), sig

def net_series_side(raw, rets_t, sig, lth, sth, cd, sl, fee=None, fund=None, side='both'):
    kw = dict(venue='aster', leverage=LEV, short_enabled=True, long_th=lth, short_th=sth, cooldown_bars=cd, bars_per_year=2190.0, stop_loss=sl)
    kw['funding_override'] = FUND if fund is None else fund
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

def stats(ser):
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

def corr(a, b):
    m = min(len(a), len(b)); a=a[:m]; b=b[:m]
    ma=sum(a)/m; mb=sum(b)/m
    va=sum((x-ma)**2 for x in a); vb=sum((x-mb)**2 for x in b)
    if va==0 or vb==0: return 0.0
    return sum((a[i]-ma)*(b[i]-mb) for i in range(m))/math.sqrt(va*vb)

# ---- build mats ----
mats = {}  # mats[coin][formula_key][seg]
coins = ['ETC','TRX','BTC']
bars_all = {c: load_bars(c) for c in coins}
for coin in coins:
    bars = bars_all[coin]
    n = len(bars); cut = int(n*0.85)
    oos = bars[cut:]; h = len(oos)//2
    for fkey, formula in [('base',BASELINE),('q2',Q2CAND)]:
        r1,t1,s1 = build_mats(oos[:h], formula)
        r2,t2,s2 = build_mats(oos[h:], formula)
        rf,tf,sf = build_mats(bars, formula)
        mats.setdefault(coin,{})[fkey] = {'h1':(r1,t1,s1),'h2':(r2,t2,s2),'full':(rf,tf,sf)}
    print(f'mats {coin} n={n} oos={len(oos)} h={h}', flush=True)

def leg_series(seg, fee=None, fund=None):
    # short leg: baseline short-only
    s_etc = net_series_side(*mats['ETC']['base'][seg], *SHORT_PARAMS['ETC'], fee=fee, fund=fund, side='short')
    s_trx = net_series_side(*mats['TRX']['base'][seg], *SHORT_PARAMS['TRX'], fee=fee, fund=fund, side='short')
    # long leg: Q2 long-only
    l_btc = net_series_side(*mats['BTC']['q2'][seg], *LONG_PARAMS['BTC'], fee=fee, fund=fund, side='long')
    l_etc = net_series_side(*mats['ETC']['q2'][seg], *LONG_PARAMS['ETC'], fee=fee, fund=fund, side='long')
    return {'S_ETC':s_etc,'S_TRX':s_trx,'L_BTC':l_btc,'L_ETC':l_etc}

mixes = {
 'SHORT-only': [1,1,0,0],
 'LONG-only': [0,0,1,1],
 'LS70/30': [0.7/2,0.7/2,0.3/2,0.3/2],
 'LS50/50': [0.5/2,0.5/2,0.5/2,0.5/2],
 'LS30/70': [0.3/2,0.3/2,0.7/2,0.7/2],
}
order = ['S_ETC','S_TRX','L_BTC','L_ETC']
out = {'config':{'baseline':BASELINE,'q2cand':Q2CAND,'short_params':{k:list(v) for k,v in SHORT_PARAMS.items()},
 'long_params':{k:list(v) for k,v in LONG_PARAMS.items()},'fee':FEE,'fund':FUND,'lev':LEV,
 'mixes':{k:v for k,v in mixes.items()}}, 'segments':{}, 'h2_leg_corr':{}, 'fee2x':{}, 'pass':{}}
for seg in ['h1','h2','full']:
    legs = leg_series(seg, fee=FEE, fund=FUND)
    segs = {}
    for name, w in mixes.items():
        ser_list = [legs[k] for k in order]
        # zero-weight legs: combo_stats with all 4 weights handles it (weights sum renormalized, but zero entries contribute 0)
        # for SHORT-only/LONG-only renormalize over active legs only:
        if name=='SHORT-only': ser_list2=[legs['S_ETC'],legs['S_TRX']]; w2=None
        elif name=='LONG-only': ser_list2=[legs['L_BTC'],legs['L_ETC']]; w2=None
        else: ser_list2=ser_list; w2=w
        sh,an,md,cu,nn,m = combo_stats(ser_list2, w2)
        segs[name] = {'sharpe':round(sh,3),'ann':round(an,4),'mdd':round(md,4),'cum':round(cu,4),'n':m}
        print(f'{seg} {name}: sh={sh:.3f} ann={an:.4f} dd={md:.4f} cum={cu:.4f} n={m}', flush=True)
    out['segments'][seg] = segs
    # leg-level combos for corr
    s_leg = combo_stats([legs['S_ETC'],legs['S_TRX']])[0:1]  # placeholder
    # build avg series
    m = min(len(legs[k]) for k in order)
    s_avg = [ (legs['S_ETC'][t]+legs['S_TRX'][t])/2 for t in range(m)]
    l_avg = [ (legs['L_BTC'][t]+legs['L_ETC'][t])/2 for t in range(m)]
    if seg=='h2':
        out['h2_leg_corr'] = {'S_vs_L':round(corr(s_avg,l_avg),3),
          'S_ETC_vs_S_TRX':round(corr(legs['S_ETC'],legs['S_TRX']),3),
          'L_BTC_vs_L_ETC':round(corr(legs['L_BTC'],legs['L_ETC']),3),
          'S_ETC_vs_L_BTC':round(corr(legs['S_ETC'],legs['L_BTC']),3),
          'S_ETC_vs_L_ETC':round(corr(legs['S_ETC'],legs['L_ETC']),3),
          'S_TRX_vs_L_BTC':round(corr(legs['S_TRX'],legs['L_BTC']),3),
          'S_TRX_vs_L_ETC':round(corr(legs['S_TRX'],legs['L_ETC']),3)}
        print('H2 leg corr: '+json.dumps(out['h2_leg_corr']), flush=True)
        # per-leg stats detail
        for nm, ss in [('S_leg_avg',s_avg),('L_leg_avg',l_avg),('S_ETC',legs['S_ETC']),('S_TRX',legs['S_TRX']),('L_BTC',legs['L_BTC']),('L_ETC',legs['L_ETC'])]:
            sh,an,md,cu,nn = stats(ss)
            print(f'H2 {nm}: sh={sh:.3f} ann={an:.4f} dd={md:.4f}', flush=True)
            out.setdefault('h2_legs',{})[nm] = {'sharpe':round(sh,3),'ann':round(an,4),'mdd':round(md,4),'cum':round(cu,4)}

# best LS mix by H2 sharpe
cands = {k:v for k,v in out['segments']['h2'].items() if k.startswith('LS')}
best = max(cands, key=lambda k: cands[k]['sharpe'])
print('best LS mix (H2 sharpe): '+best, flush=True)
out['best_mix'] = best
# fee 2x stress on best mix
bw = mixes[best]
for seg in ['h2','full']:
    legs = leg_series(seg, fee=FEE*2, fund=FUND)
    ser_list=[legs[k] for k in order]
    sh,an,md,cu,nn,m = combo_stats(ser_list, bw)
    out['fee2x'][seg] = {'sharpe':round(sh,3),'ann':round(an,4),'mdd':round(md,4),'cum':round(cu,4),'n':m}
    print(f'fee2x {seg} {best}: sh={sh:.3f} ann={an:.4f} dd={md:.4f}', flush=True)

# PASS bar: LS H2 sharpe>2.0 AND H2 dd<0.25 AND FULL sharpe>2.0 (best LS mix)
h2 = out['segments']['h2'][best]; fu = out['segments']['full'][best]
passed = (h2['sharpe']>2.0) and (h2['mdd']<0.25) and (fu['sharpe']>2.0)
out['pass'] = {'best':best,'h2_sharpe':h2['sharpe'],'h2_mdd':h2['mdd'],'full_sharpe':fu['sharpe'],'PASS':bool(passed)}
print('PASS: '+json.dumps(out['pass']), flush=True)
open('results/backtest_R2_pair.json','w').write(json.dumps(out, indent=1))
print('saved backtest_R2_pair.json', flush=True)
