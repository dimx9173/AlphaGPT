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
def load_bars(coin):
    import csv
    rows = list(csv.DictReader(open('data/data_1y/30m/'+coin+'.csv')))
    bars = []
    for i in range(0, len(rows), 8):
        blk = rows[i:i+8]
        if len(blk) < 8: break
        bars.append((float(blk[0]['open']), max(float(x['high']) for x in blk), min(float(x['low']) for x in blk), float(blk[-1]['close']), sum(float(x['volume']) for x in blk)))
    return bars
def build_mats(bars):
    import torch
    n = len(bars)
    raw = {'open':torch.tensor([[b[0] for b in bars]]),'high':torch.tensor([[b[1] for b in bars]]),'low':torch.tensor([[b[2] for b in bars]]),'close':torch.tensor([[b[3] for b in bars]]),'volume':torch.tensor([[b[4] for b in bars]]),'liquidity':torch.full((1,n),1e7),'fdv':torch.full((1,n),1e8)}
    sig = StackVM(use_advanced=False).execute(FORMULA, FeatureEngineer.compute_features(raw, use_advanced=False))
    rets = [(bars[i+1][3]-bars[i][3])/bars[i][3] for i in range(n-1)] + [0.0]
    return raw, torch.tensor([rets]), sig
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
    turn_list = turn[0].tolist()
    return net, turn_list
# mats
mats = {}
for coin in ['ETC','TRX']:
    bars = load_bars(coin)
    n = len(bars); cut = int(n*0.85)
    oos = bars[cut:]; h = len(oos)//2
    r1, t1, s1 = build_mats(oos[:h])
    r2, t2, s2 = build_mats(oos[h:])
    rf, tf, sf = build_mats(bars)
    lth, sth, cd, sl = BEST[coin]
    mats[coin] = {'h1':(r1,t1,s1),'h2':(r2,t2,s2),'full':(rf,tf,sf),'params':(lth,sth,cd,sl),'n_full':n,'n_h1':len(oos[:h]),'n_h2':len(oos[h:])}
    print('mats '+coin+' h1='+str(len(oos[:h]))+' h2='+str(len(oos[h:]))+' full='+str(n), flush=True)
# ---- N7 rotation ----
def leg_pos(raw, rets_t, sig, lth, sth, cd, sl):
    from model_core.backtest import MemeBacktest
    bt = MemeBacktest(venue='aster', leverage=2.0, short_enabled=True, long_th=lth, short_th=sth, cooldown_bars=cd, bars_per_year=2190.0, stop_loss=sl, funding_override=0.0005)
    signal = torch.sigmoid(sig)
    strength = (signal - 0.5).abs()
    is_safe = (raw['liquidity'] > bt.min_liq).float()
    lp = (signal > bt.long_th).float() * is_safe
    sp = (signal < bt.short_th).float() * is_safe
    lp, sp = bt._apply_cooldown(lp, sp)
    lp, sp = bt._apply_stops(lp, sp, rets_t)
    lp = lp.roll(1, dims=1); lp[:,0]=0
    sp = sp.roll(1, dims=1); sp[:,0]=0
    return bt, signal, strength, lp, sp
def rotation_stats(seg):
    rE,tE,sE = mats['ETC'][seg]; rT,tT,sT = mats['TRX'][seg]
    pE = BEST['ETC']; pT = BEST['TRX']
    btE, sigE, strE, lpE, spE = leg_pos(rE,tE,sE,*pE)
    btT, sigT, strT, lpT, spT = leg_pos(rT,tT,sT,*pT)
    # winner raw by strength at signal bar, then rolled +1 for execution lag (matches lp/sp roll)
    winE_raw = (strE >= strT).float()
    winE = winE_raw.roll(1, dims=1); winE[:,0]=0.5  # bar0 unused (positions 0); use 0.5 to keep neutral
    winT = 1.0 - winE
    # at bar0 positions are 0 anyway
    lpR = winE*lpE + winT*lpT
    spR = winE*spE + winT*spT
    # fix bar0: force 0
    lpR[:,0]=0; spR[:,0]=0
    # rets per leg
    # gross uses each leg's own rets; funding same rate; tx from rotated turnover
    lev=2.0; fund=0.0005
    base_fee = btE.base_fee
    liq = rE['liquidity']
    gross = (winE*(lpE-spE)*tE + winT*(lpT-spT)*tT)*lev
    fnd = (winE*(lpE-spE) + winT*(lpT-spT))*fund*lev
    turn = (lpR - lpR.roll(1,dims=1)).abs() + (spR - spR.roll(1,dims=1)).abs()
    # turn at bar0: lpR-roll => lpR[0]-lpR[-1]; correct to position change from flat
    # replicate manual convention: turn uses roll which wraps; bar0 turn includes wrap artifact. Match net_series_side convention (same artifact) for comparability.
    tx = turn * (base_fee + torch.clamp(1000.0/(liq+1e-9),0.0,0.05))
    net = (gross - tx*lev - fnd)[0].tolist()
    turnl = turn[0].tolist()
    sh,an,md,cu,nn = stats(net)
    turnover = sum(turnl)/len(turnl)
    # diagnostics: share of bars won by ETC, overlap both-flat rate, both-active rate
    winE_list = winE[0].tolist()
    # active = lp+sp>0.5 per leg (rolled positions)
    actE = ((lpE+spE)>0.5).float()[0].tolist()
    actT = ((lpT+spT)>0.5).float()[0].tolist()
    shrE = sum(1 for v in winE_list if v>0.9)/len(winE_list)
    both_act = sum(1 for a,b in zip(actE,actT) if a>0.5 and b>0.5)/len(actE)
    both_flat = sum(1 for a,b in zip(actE,actT) if a<0.5 and b<0.5)/len(actE)
    return {'net':net,'turn':turnl,'sharpe':sh,'ann':an,'mdd':md,'cum':cu,'n':nn,'turnover':turnover,'share_ETC':shrE,'both_active':both_act,'both_flat':both_flat}
print('=== N7 rotation vs equal-weight ===', flush=True)
n7={}
for seg in ['h1','h2','full']:
    netE, turnE = net_series_side(*mats['ETC'][seg], *mats['ETC']['params'])
    netT, turnT = net_series_side(*mats['TRX'][seg], *mats['TRX']['params'])
    sh,an,md,cu,nn,m = combo_stats([netE,netT])
    to_eq = sum((a+b)/2 for a,b in zip(turnE,turnT))/len(turnE)
    rot = rotation_stats(seg)
    n7[seg]={'eq':{'sharpe':sh,'ann':an,'mdd':md,'cum':cu,'n':m,'turnover':to_eq},'rot':{k:rot[k] for k in ['sharpe','ann','mdd','cum','n','turnover','share_ETC','both_active','both_flat']}}
    print(f"{seg} EQ sh={sh:.3f} ann={an:.3f} dd={md:.3f} to={to_eq:.4f} | ROT sh={rot['sharpe']:.3f} ann={rot['ann']:.3f} dd={rot['mdd']:.3f} to={rot['turnover']:.4f} shareETC={rot['share_ETC']:.2f} bothact={rot['both_active']:.2f} bothflat={rot['both_flat']:.2f}", flush=True)
# ---- N8 vol-target sweep ----
def net_series_vol(raw, rets_t, sig, lth, sth, cd, sl, vt, vw=24):
    from model_core.backtest import MemeBacktest
    bt = MemeBacktest(venue='aster', leverage=2.0, short_enabled=True, long_th=lth, short_th=sth, cooldown_bars=cd, bars_per_year=2190.0, stop_loss=sl, funding_override=0.0005, vol_target=vt, vol_window=vw)
    signal = torch.sigmoid(sig)
    is_safe = (raw['liquidity'] > bt.min_liq).float()
    lp = (signal > bt.long_th).float() * is_safe
    sp = (signal < bt.short_th).float() * is_safe
    lp, sp = bt._apply_cooldown(lp, sp)
    lp, sp = bt._apply_stops(lp, sp, rets_t)
    scale = bt._vol_scale(rets_t)
    lp, sp = lp*scale, sp*scale
    lp = lp.roll(1,dims=1); lp[:,0]=0
    sp = sp.roll(1,dims=1); sp[:,0]=0
    turn = (lp - lp.roll(1,dims=1)).abs() + (sp - sp.roll(1,dims=1)).abs()
    tx = turn * (bt.base_fee + torch.clamp(bt.trade_size/(raw['liquidity']+1e-9),0.0,0.05))
    gross = (lp-sp)*rets_t*bt.leverage
    fnd = (lp-sp)*bt.default_funding_rate*bt.leverage
    net = (gross - tx*bt.leverage - fnd)[0].tolist()
    return net, turn[0].tolist()
print('=== N8 vol-target sweep (eq ETC+TRX) ===', flush=True)
n8=[]
for vt in [None,0.02,0.05,0.10]:
    row={'vol_target':vt,'vol_window':24}
    line=f"vt={vt}"
    for seg in ['h1','h2','full']:
        ne,te = net_series_vol(*mats['ETC'][seg], *mats['ETC']['params'], vt)
        nt,tt = net_series_vol(*mats['TRX'][seg], *mats['TRX']['params'], vt)
        sh,an,md,cu,nn,m = combo_stats([ne,nt])
        to = sum((a+b)/2 for a,b in zip(te,tt))/len(te)
        row[seg]={'sharpe':sh,'ann':an,'mdd':md,'cum':cu,'n':m,'turnover':to}
        line+=f" | {seg} sh={sh:.3f} ann={an:.3f} dd={md:.3f} to={to:.4f}"
    print(line, flush=True)
    n8.append(row)
out={'baseline':{'formula':FORMULA,'best':{k:{'lth':v[0],'sth':v[1],'cd':v[2],'sl':v[3]} for k,v in BEST.items()},'venue':'aster','leverage':2.0,'fund':0.0005,'split':'OOS last15pct H1/H2, FULL full-history','n':{c:{s:mats[c][s][1].shape[1] for s in ['h1','h2','full']} for c in ['ETC','TRX']}},'N7':n7,'N8':n8}
open('results/backtest_iterN7N8_30m.json','w').write(json.dumps(out,indent=1))
print('saved run_iterN5N6_30m_30m_30m.json', flush=True)
