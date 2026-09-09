"""Q1 baseline: 5-coin (BTC/SOL/ETC/TRX/DOGE) grid both-legs + side split.
Formula [3,2,7,2,7,11,15,4,4,6,6,10], aster perp 2x, fund 0.0005, fee 0.0004 (aster default).
Pattern reused from run_bsd1.py / run_grouptest678910.py.
Per coin: 6570 4h bars, 85% cut (5584 train), OOS 986 -> H1/H2 493/493.
Grid 2x3x3x3=54 rows, H1-best by H1 sharpe, verify H2 + FULL.
Side split on H1-best: both/long/short on H1/H2/FULL.
Combos (equal-weight, H1-best params): ALL5, MAJORS, EDGE, ALL5exETC.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import csv, json, math, statistics
import torch
from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest

FORMULA = [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10]
COINS = ['BTC', 'SOL', 'ETC', 'TRX', 'DOGE']
GRID_LTH = [0.85, 0.88]
GRID_STH = [0.10, 0.12, 0.15]
GRID_CD = [3, 6, 12]
GRID_SL = [None, 0.03, 0.05]
COMBOS = {'ALL5': ['BTC', 'SOL', 'ETC', 'TRX', 'DOGE'],
          'MAJORS': ['BTC', 'SOL', 'DOGE'],
          'EDGE': ['ETC', 'TRX'],
          'ALL5exETC': ['BTC', 'SOL', 'TRX', 'DOGE']}

def load_bars(coin):
    rows = list(csv.DictReader(open('data/data_15m_3y/' + coin + '.csv')))
    bars = []
    for i in range(0, len(rows), 16):
        blk = rows[i:i + 16]
        if len(blk) < 16:
            break
        bars.append((float(blk[0]['open']), max(float(x['high']) for x in blk),
                     min(float(x['low']) for x in blk), float(blk[-1]['close']),
                     sum(float(x['volume']) for x in blk)))
    return bars

def build_mats(bars):
    n = len(bars)
    raw = {'open': torch.tensor([[b[0] for b in bars]]),
           'high': torch.tensor([[b[1] for b in bars]]),
           'low': torch.tensor([[b[2] for b in bars]]),
           'close': torch.tensor([[b[3] for b in bars]]),
           'volume': torch.tensor([[b[4] for b in bars]]),
           'liquidity': torch.full((1, n), 1e7),
           'fdv': torch.full((1, n), 1e8)}
    sig = StackVM().execute(FORMULA, FeatureEngineer.compute_features(raw))
    rets = [(bars[i + 1][3] - bars[i][3]) / bars[i][3] for i in range(n - 1)] + [0.0]
    return raw, torch.tensor([rets]), sig

def net_series_side(raw, rets_t, sig, lth, sth, cd, sl, side='both'):
    bt = MemeBacktest(venue='aster', leverage=2.0, short_enabled=True,
                      funding_override=0.0005, long_th=lth, short_th=sth,
                      cooldown_bars=cd, bars_per_year=2190.0, stop_loss=sl)
    signal = torch.sigmoid(sig)
    is_safe = (raw['liquidity'] > bt.min_liq).float()
    lp = (signal > bt.long_th).float() * is_safe
    sp = (signal < bt.short_th).float() * is_safe
    lp, sp = bt._apply_cooldown(lp, sp)
    lp, sp = bt._apply_stops(lp, sp, rets_t)
    if side == 'long':
        sp = sp * 0.0
    if side == 'short':
        lp = lp * 0.0
    lp = lp.roll(1, dims=1); lp[:, 0] = 0
    sp = sp.roll(1, dims=1); sp[:, 0] = 0
    turn = (lp - lp.roll(1, dims=1)).abs() + (sp - sp.roll(1, dims=1)).abs()
    tx = turn * (bt.base_fee + torch.clamp(bt.trade_size / (raw['liquidity'] + 1e-9), 0.0, 0.05))
    gross = (lp - sp) * rets_t * bt.leverage
    fnd = (lp - sp) * bt.default_funding_rate * bt.leverage
    return (gross - tx * bt.leverage - fnd)[0].tolist()

def stats(ser):
    n = len(ser)
    mean = sum(ser) / n
    var = sum((x - mean) ** 2 for x in ser) / max(n - 1, 1)
    sharpe = mean / math.sqrt(var) * math.sqrt(2190.0) if var > 0 else 0.0
    cum = sum(ser)
    ann = cum / n * 2190.0
    cs, peak, mdd = 0.0, -1e18, 0.0
    for x in ser:
        cs += x
        peak = max(peak, cs)
        mdd = max(mdd, peak - cs)
    return {'sharpe': sharpe, 'ann': ann, 'mdd': mdd, 'cum': cum, 'n': n}

def combo_stats(ser_list):
    m = min(len(s) for s in ser_list)
    k = len(ser_list)
    cb = [sum(ser_list[i][t] for i in range(k)) / k for t in range(m)]
    return stats(cb), cb

def main():
    out = {'formula': FORMULA, 'venue': 'aster', 'leverage': 2.0,
           'funding': 0.0005, 'fee': 0.0004, 'fee_note': 'aster venue default',
           'grid': {'lth': GRID_LTH, 'sth': GRID_STH, 'cd': GRID_CD,
                    'sl': ['None', 0.03, 0.05]},
           'selection': 'per-coin H1-best by H1 sharpe', 'coins': {}}
    series = {}
    for coin in COINS:
        bars = load_bars(coin)
        n = len(bars)
        cut = int(n * 0.85)
        oos = bars[cut:]
        h = len(oos) // 2
        assert n == 6570 and cut == 5584 and len(oos) == 986 and h == 493, (n, cut, len(oos), h)
        segs = {'h1': oos[:h], 'h2': oos[h:], 'full': bars}
        mats = {k: build_mats(v) for k, v in segs.items()}
        rows = []
        for lth in GRID_LTH:
            for sth in GRID_STH:
                for cd in GRID_CD:
                    for sl in GRID_SL:
                        ser = net_series_side(*mats['h1'], lth, sth, cd, sl)
                        s = stats(ser)
                        rows.append({'lth': lth, 'sth': sth, 'cd': cd, 'sl': sl,
                                     'h1_sharpe': s['sharpe'], 'h1_ann': s['ann'],
                                     'h1_mdd': s['mdd']})
        assert len(rows) == 54, len(rows)
        rows.sort(key=lambda r: r['h1_sharpe'], reverse=True)
        best = rows[0]
        res = {'n_full': n, 'n_train': cut, 'n_h1': h, 'n_h2': len(oos) - h,
               'grid_rows': len(rows),
               'best': {k: best[k] for k in ['lth', 'sth', 'cd', 'sl']}}
        series[coin] = {}
        for seg in ['h1', 'h2', 'full']:
            series[coin][seg] = {}
            for side in ['both', 'long', 'short']:
                ser = net_series_side(*mats[seg], best['lth'], best['sth'],
                                      best['cd'], best['sl'], side=side)
                series[coin][seg][side] = ser
                res[seg + '_' + side] = stats(ser)
        res['grid_top5'] = rows[:5]
        out['coins'][coin] = res
        print(coin + ' best=' + str(res['best'])
              + ' | H2 both=' + str(round(res['h2_both']['sharpe'], 3))
              + ' long=' + str(round(res['h2_long']['sharpe'], 3))
              + ' short=' + str(round(res['h2_short']['sharpe'], 3))
              + ' | FULL both=' + str(round(res['full_both']['sharpe'], 3)),
              flush=True)
    out['combos'] = {}
    for name, cl in COMBOS.items():
        co = {'coins': cl, 'note': 'equal-weight mean of per-coin H1-best both-leg series'}
        print('--- combo ' + name + ' ' + str(cl) + ' ---', flush=True)
        for seg in ['h1', 'h2', 'full']:
            co2 = {}
            for side in ['both', 'long', 'short']:
                se = [series[c][seg][side] for c in cl]
                s, cb = combo_stats(se)
                co2[side] = s
                if seg == 'h2':
                    co2[side + '_n'] = s['n']
            co[seg] = co2
            print(name + ' ' + seg + ' both=' + str(round(co2['both']['sharpe'], 3))
                  + ' long=' + str(round(co2['long']['sharpe'], 3))
                  + ' short=' + str(round(co2['short']['sharpe'], 3)), flush=True)
        a = [series[c]['h2']['both'] for c in cl]
        corr = {}
        for i in range(len(cl)):
            for j in range(i + 1, len(cl)):
                try:
                    v = statistics.correlation(a[i], a[j])
                except Exception:
                    v = 0.0
                corr[cl[i] + '_' + cl[j]] = {'r': v, 'n_shared': min(len(a[i]), len(a[j]))}
        co['h2_pairwise_corr'] = corr
        print(name + ' H2 corr: ' + str({k: round(v['r'], 3) for k, v in corr.items()}), flush=True)
        out['combos'][name] = co
    open('results/backtest_Q1_5coin.json', 'w').write(json.dumps(out, indent=1))
    print('saved backtest_Q1_5coin.json', flush=True)

if __name__ == '__main__':
    main()
