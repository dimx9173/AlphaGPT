import csv, json, math, statistics
import torch
from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest
FORMULA = [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10]
GRID = [(0.85,0.15,6,None),(0.85,0.15,6,0.05),(0.88,0.12,6,None),(0.88,0.12,3,0.03),
        (0.90,0.10,6,None),(0.90,0.10,3,0.03),(0.92,0.08,6,None),(0.92,0.08,12,None),
        (0.85,0.15,12,0.03),(0.90,0.10,12,0.03),(0.88,0.12,12,None),(0.92,0.08,3,0.03)]
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
    o = torch.tensor([[b[0] for b in bars]])
    h = torch.tensor([[b[1] for b in bars]])
    l = torch.tensor([[b[2] for b in bars]])
    c = torch.tensor([[b[3] for b in bars]])
    v = torch.tensor([[b[4] for b in bars]])
    raw = {'open':o,'high':h,'low':l,'close':c,'volume':v,'liquidity':torch.full((1,n),1e7),'fdv':torch.full((1,n),1e8)}
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
    fund = (lp - sp) * bt.default_funding_rate * bt.leverage
    net = (gross - tx * bt.leverage - fund)[0].tolist()
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
SETS = {'single_BTC':['BTC'],'single_ETC':['ETC'],'single_TRX':['TRX'],'prev_naive_BTC_TRX_BNB_ETH':['BTC','TRX','BNB','ETH'],'within_main_BTC_ETH':['BTC','ETH'],'within_meme_DOGE_SHIB':['DOGE','SHIB'],'within_pay_XRP_XLM':['XRP','XLM'],'within_new_SOL_AVAX':['SOL','AVAX'],'within_main_BTC_ETC_LINK':['BTC','ETC','LINK'],'cross_BTC_TRX_XRP_SOL':['BTC','TRX','XRP','SOL'],'cross_ETC_TRX_XRP_SOL':['ETC','TRX','XRP','SOL'],'cross_BTC_TRX_DOGE_XRP':['BTC','TRX','DOGE','XRP'],'cross_ETC_TRX_DOGE_XRP':['ETC','TRX','DOGE','XRP'],'cross_BTC_TRX_BCH':['BTC','TRX','BCH'],'cross_ETC_TRX_BCH_SOL':['ETC','TRX','BCH','SOL']}
allcoins = sorted({c for s in SETS.values() for c in s})
print('coins:', allcoins, flush=True)
best = {}
test_series = {}
train_stats = {}
for coin in allcoins:
    bars = load_bars(coin)
    n = len(bars); cut = int(n*0.85)
    traw, tret, tsig = build_mats(bars[:cut])
    pick = None; pick_sh = -1e18
    for (lth, sth, cd, sl) in GRID:
        bt = MemeBacktest(venue='aster', leverage=2.0, short_enabled=True, funding_override=0.0005, long_th=lth, short_th=sth, cooldown_bars=cd, bars_per_year=2190.0, stop_loss=sl)
        bt.evaluate(tsig, traw, tret)
        sh = bt.last_metrics['sharpe']
        if sh > pick_sh:
            pick_sh = sh; pick = (lth, sth, cd, sl)
    best[coin] = pick
    traw2, tret2, tsig2 = traw, tret, tsig
    tr_net = net_series(traw2, tret2, tsig2, *pick)
    tsh, tann, tmdd, tcum, tn = stats(tr_net)
    train_stats[coin] = (round(tsh,3), round(tann,3), round(tmdd,3))
    erw, eret, esig = build_mats(bars[cut:])
    te_net = net_series(erw, eret, esig, *pick)
    test_series[coin] = te_net
    sh2, an2, md2, cu2, nn2 = stats(te_net)
    print(str(coin)+': best_train='+str(pick)+' train_sharpe='+str(round(pick_sh,3))+' test_sharpe='+str(round(sh2,3))+' test_ann='+str(round(an2,3))+' test_mdd='+str(round(md2,3))+' ntest='+str(nn2), flush=True)
print('=== TEST combos (equal-weight, per-coin train-best params) ===', flush=True)
import statistics
rows = []
for name, COINS in SETS.items():
    m = min(len(test_series[c]) for c in COINS)
    combo = [sum(test_series[c][t] for c in COINS)/len(COINS) for t in range(m)]
    sh, an, md, cu, nn = stats(combo)
    corrs = []
    for i in range(len(COINS)):
        xa = test_series[COINS[i]][:m]
        for j in range(i+1, len(COINS)):
            xb = test_series[COINS[j]][:m]
            ma = statistics.mean(xa); mb = statistics.mean(xb)
            cov = sum((x-ma)*(y-mb) for x, y in zip(xa, xb))/m
            sa = statistics.pstdev(xa); sb = statistics.pstdev(xb)
            corrs.append(cov/(sa*sb) if sa and sb else 0)
    mc = sum(corrs)/len(corrs) if corrs else 1.0
    print(str(name)+' '+str(COINS)+' sharpe='+str(round(sh,3))+' ann='+str(round(an,3))+' mdd='+str(round(md,3))+' pnlcorr='+str(round(mc,2)), flush=True)
    rows.append({'name':name,'coins':COINS,'params':{c:list(best[c]) for c in COINS},'sharpe':sh,'ann':an,'mdd':md,'pnlcorr':mc,'n':nn})
import json as _j
open('backtest_grouptest_oos.json','w').write(_j.dumps(rows, indent=1))
print('saved backtest_grouptest_oos.json', flush=True)
