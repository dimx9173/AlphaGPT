"""S4 cooldown/threshold re-tune for tail regime (H1-anchored, no H2 cherry-pick).
Baseline formula [3,2,7,2,7,11,15,4,4,6,6,10] on ETC+TRX equal-weight.
Aster perp 2x, fund 0.0005, fee 0.0004 (Q1 baseline venue config).
Grid (joint combo rows, shared lth/sth/sl; cooldown low/med/high mapped
per-coin: ETC [6,12,18], TRX [3,6,12]): 2*3*3*2 = 36 combos (cap ~48).
SELECT top3 on frozen H1 (bars[5584:6077]) ONLY by H1 combo sharpe.
VERIFY top3 on frozen H2 (bars[6077:6570]) + fresh B(last 200) + C(last 500).
PASS bar: H1-best row has B > 1.0 AND C > 1.5 AND H2 > 1.5.
Method mirrors research/run_q1.py (net series) + research/run_p2.py (trades).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import csv, json, math
import torch
from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest

FORMULA = [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10]
COINS = ['ETC', 'TRX']
GRID_LTH = [0.85, 0.88]
GRID_STH = [0.10, 0.12, 0.15]
CD_MAP = {'low': (6, 3), 'med': (12, 6), 'high': (18, 12)}  # (ETC_cd, TRX_cd)
GRID_CD_LVL = ['low', 'med', 'high']
GRID_SL = [None, 0.05]
H1_LO, H1_HI = 5584, 6077   # frozen H1
H2_LO, H2_HI = 6077, 6570   # frozen H2

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
    sig = StackVM(use_advanced=False).execute(FORMULA, FeatureEngineer.compute_features(raw, use_advanced=False))
    rets = [(bars[i + 1][3] - bars[i][3]) / bars[i][3] for i in range(n - 1)] + [0.0]
    return raw, torch.tensor([rets]), sig

def net_series_trades(raw, rets_t, sig, lth, sth, cd, sl):
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
    pos = (lp - sp)[0].tolist()
    trades = sum(1 for t in range(len(pos)) if pos[t] != 0.0 and (t == 0 or pos[t - 1] == 0.0))
    return net, trades

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

def combo_eval(mats, lth, sth, cd_etc, cd_trx, sl):
    cds = {'ETC': cd_etc, 'TRX': cd_trx}
    legs, tr = [], []
    by = {}
    for c in COINS:
        raw, rt, sg = mats[c]
        net, t = net_series_trades(raw, rt, sg, lth, sth, cds[c], sl)
        legs.append(net)
        tr.append(t)
        s = stats(net)
        s['trades'] = t
        by[c] = s
    m = min(len(s) for s in legs)
    cb = [(legs[0][t] + legs[1][t]) / 2.0 for t in range(m)]
    s = stats(cb)
    s['trades'] = sum(tr)
    s['trades_by'] = {c: tr[i] for i, c in enumerate(COINS)}
    return s, by

def main():
    full = {c: load_bars(c) for c in COINS}
    n = min(len(b) for b in full.values())
    assert n >= 6570 and H2_HI <= n, (n, H2_HI)
    segs = {'H1': {c: full[c][H1_LO:H1_HI] for c in COINS},
            'H2': {c: full[c][H2_LO:H2_HI] for c in COINS},
            'B': {c: full[c][n - 200:n] for c in COINS},
            'C': {c: full[c][n - 500:n] for c in COINS}}
    assert all(len(v) == 493 for v in segs['H1'].values()), [len(v) for v in segs['H1'].values()]
    assert all(len(v) == 493 for v in segs['H2'].values()), [len(v) for v in segs['H2'].values()]
    mats = {k: {c: build_mats(v[c]) for c in COINS} for k, v in segs.items()}
    print('n_full=%d H1=%d H2=%d B=%d C=%d' % (n, len(segs['H1']['ETC']),
          len(segs['H2']['ETC']), len(segs['B']['ETC']), len(segs['C']['ETC'])), flush=True)
    rows = []
    for lth in GRID_LTH:
        for sth in GRID_STH:
            for lvl in GRID_CD_LVL:
                cd_etc, cd_trx = CD_MAP[lvl]
                for sl in GRID_SL:
                    s, _ = combo_eval(mats['H1'], lth, sth, cd_etc, cd_trx, sl)
                    rows.append({'lth': lth, 'sth': sth, 'cd_level': lvl,
                                 'cd_etc': cd_etc, 'cd_trx': cd_trx, 'sl': sl,
                                 'H1_sharpe': s['sharpe'], 'H1_ann': s['ann'],
                                 'H1_mdd': s['mdd'], 'H1_trades': s['trades'],
                                 'H1_trades_by': s['trades_by']})
    assert len(rows) == 36, len(rows)
    rows.sort(key=lambda r: r['H1_sharpe'], reverse=True)
    out = {'formula': FORMULA, 'venue': 'aster', 'leverage': 2.0, 'funding': 0.0005,
           'fee': 0.0004, 'weights': {'ETC': 0.5, 'TRX': 0.5},
           'grid': {'lth': GRID_LTH, 'sth': GRID_STH,
                    'cd_map_ETC_TRX': {k: list(v) for k, v in CD_MAP.items()},
                    'sl': ['None', 0.05], 'n_combos': len(rows)},
           'segments': {'H1': [H1_LO, H1_HI], 'H2': [H2_LO, H2_HI],
                        'B_last200': [n - 200, n], 'C_last500': [n - 500, n],
                        'full_n': n},
           'selection': 'H1-sharpe top3 on frozen H1 only', 'top3': []}
    for rank, row in enumerate(rows[:3]):
        entry = {'rank': rank, 'params': {k: row[k] for k in
                 ['lth', 'sth', 'cd_level', 'cd_etc', 'cd_trx', 'sl']}}
        for seg in ['H1', 'H2', 'B', 'C']:
            s, by = combo_eval(mats[seg], row['lth'], row['sth'], row['cd_etc'],
                               row['cd_trx'], row['sl'])
            entry[seg] = {'sharpe': s['sharpe'], 'ann': s['ann'], 'mdd': s['mdd'],
                          'cum': s['cum'], 'n': s['n'], 'trades': s['trades'],
                          'trades_by': s['trades_by']}
            entry[seg + '_bycoin'] = {c: {'sharpe': by[c]['sharpe'], 'ann': by[c]['ann'],
                                          'mdd': by[c]['mdd'], 'trades': by[c]['trades']}
                                      for c in COINS}
        out['top3'].append(entry)
        print('rank%d lth=%s sth=%s cd=%s(ETC%d/TRX%d) sl=%s | H1=%.3f t%d | H2=%.3f t%d | B=%.3f t%d | C=%.3f t%d' % (
            rank, row['lth'], row['sth'], row['cd_level'], row['cd_etc'], row['cd_trx'],
            row['sl'], entry['H1']['sharpe'], entry['H1']['trades'],
            entry['H2']['sharpe'], entry['H2']['trades'],
            entry['B']['sharpe'], entry['B']['trades'],
            entry['C']['sharpe'], entry['C']['trades']), flush=True)
    best = out['top3'][0]
    out['PASS'] = {'H2_gt_1.5': best['H2']['sharpe'] > 1.5,
                   'B_gt_1.0': best['B']['sharpe'] > 1.0,
                   'C_gt_1.5': best['C']['sharpe'] > 1.5}
    out['PASS']['overall'] = all(out['PASS'].values())
    print('PASS ' + json.dumps(out['PASS']), flush=True)
    # full grid H1 ranking for appendix
    out['grid_all_H1'] = [{'lth': r['lth'], 'sth': r['sth'], 'cd_level': r['cd_level'],
                           'sl': r['sl'], 'H1_sharpe': r['H1_sharpe'],
                           'H1_trades': r['H1_trades']} for r in rows]
    open('results/backtest_S4_retune.json', 'w').write(json.dumps(out, indent=1))
    print('saved backtest_S4_retune.json', flush=True)

if __name__ == '__main__':
    main()
