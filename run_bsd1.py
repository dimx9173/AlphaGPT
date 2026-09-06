import csv, json, math, statistics
import torch
from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest

FORMULA = [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10]
COINS = ['BTC', 'SOL', 'DOGE', 'ASTER']
GRID_LTH = [0.85, 0.88]
GRID_STH = [0.10, 0.12, 0.15]
GRID_CD = [3, 6, 12]
GRID_SL = [None, 0.03, 0.05]

def load_bars(coin):
    rows = list(csv.DictReader(open('data_15m_3y/' + coin + '.csv')))
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

def net_series_side(raw, rets_t, sig, lth, sth, cd, sl):
    bt = MemeBacktest(venue='aster', leverage=2.0, short_enabled=True,
                      funding_override=0.0005, long_th=lth, short_th=sth,
                      cooldown_bars=cd, bars_per_year=2190.0, stop_loss=sl)
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
    return sharpe, ann, mdd, cum, n

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
        segs = {'h1': oos[:h], 'h2': oos[h:], 'full': bars}
        mats = {k: build_mats(v) for k, v in segs.items()}
        rows = []
        for lth in GRID_LTH:
            for sth in GRID_STH:
                for cd in GRID_CD:
                    for sl in GRID_SL:
                        ser = net_series_side(*mats['h1'], lth, sth, cd, sl)
                        sh, an, md, cu, nn = stats(ser)
                        rows.append({'lth': lth, 'sth': sth, 'cd': cd, 'sl': sl,
                                     'h1_sharpe': sh, 'h1_ann': an, 'h1_mdd': md})
        assert len(rows) <= 54
        rows.sort(key=lambda r: r['h1_sharpe'], reverse=True)
        best = rows[0]
        res = {'n_full': n, 'n_train': cut, 'n_h1': len(segs['h1']),
               'n_h2': len(segs['h2']), 'grid_rows': len(rows),
               'best': {k: best[k] for k in ['lth', 'sth', 'cd', 'sl']}}
        series[coin] = {}
        for seg in ['h1', 'h2', 'full']:
            ser = net_series_side(*mats[seg], best['lth'], best['sth'],
                                  best['cd'], best['sl'])
            series[coin][seg] = ser
            sh, an, md, cu, nn = stats(ser)
            res[seg] = {'sharpe': sh, 'ann': an, 'mdd': md, 'cum': cu, 'n': nn}
        res['grid_top5'] = rows[:5]
        out['coins'][coin] = res
        print(coin + ' n=' + str(n) + ' best=' + str(res['best'])
              + ' | H1 sh=' + str(round(res['h1']['sharpe'], 3))
              + ' H2 sh=' + str(round(res['h2']['sharpe'], 3))
              + ' FULL sh=' + str(round(res['full']['sharpe'], 3)), flush=True)
    # Combo: positional overlap alignment to common END (histories share end
    # time within ~1 day; 4h-bar phase differs by <4h so exact-ts join fails).
    # Overlap length L = ASTER full length (2107): tails of BTC/SOL/DOGE full
    # series + ASTER full. ASTER's own split (cut=1790, h=158) defines the
    # shared calendar windows: combo H1 = idx 1790:1948, H2 = idx 1948:.
    import datetime
    L = min(len(series[c]['full']) for c in COINS)
    assert L == len(series['ASTER']['full']), L
    tails = {c: series[c]['full'][-L:] for c in COINS}
    cb_full = [sum(tails[c][t] for c in COINS) / len(COINS) for t in range(L)]
    acut = out['coins']['ASTER']['n_train']
    ah = out['coins']['ASTER']['n_h1']
    windows = {'h1': (acut, acut + ah), 'h2': (acut + ah, L), 'full': (0, L)}
    combo = {'note': 'equal-weight mean of 4 coins at per-coin H1-best params, '
                     'positionally aligned to common end (overlap L=' + str(L) + '); '
                     'H1/H2 windows follow ASTER split (short-history caveat)',
             'overlap_len': L}
    for seg, (a, b) in windows.items():
        cb = cb_full[a:b]
        sh, an, md, cu, nn = stats(cb)
        combo[seg] = {'sharpe': sh, 'ann': an, 'mdd': md, 'cum': cu, 'n': nn,
                      'idx': [a, b]}
        print('combo ' + seg + ' sh=' + str(round(sh, 3))
              + ' ann=' + str(round(an, 3)) + ' dd=' + str(round(md, 3))
              + ' n=' + str(nn), flush=True)
    # pairwise corr on H2 window over the same aligned tails
    a, b = windows['h2']
    h2t = {c: tails[c][a:b] for c in COINS}
    corr = {}
    for i in range(len(COINS)):
        for j in range(i + 1, len(COINS)):
            x, y = COINS[i], COINS[j]
            try:
                v = statistics.correlation(h2t[x], h2t[y])
            except Exception:
                v = 0.0
            corr[x + '_' + y] = {'r': v, 'n_shared': b - a}
    combo['h2_pairwise_corr'] = corr
    print('H2 corr: ' + str({k: round(v['r'], 3) for k, v in corr.items()}), flush=True)
    out['combo'] = combo
    open('backtest_BSD1_baseline.json', 'w').write(json.dumps(out, indent=1))
    print('saved backtest_BSD1_baseline.json', flush=True)

if __name__ == '__main__':
    main()
