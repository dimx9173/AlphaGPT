"""S1 short-only gate retest on fresh tails (locked ETC+TRX baseline).
Locked: FORMULA [3,2,7,2,7,11,15,4,4,6,6,10], ETC (0.88/0.12/cd12/None)
+ TRX (0.85/0.15/cd6/sl0.05), 50/50 eq, aster perp 2x fund 0.0005 fee 0.0004 (fee2x 0.0008).
Segments (side=short ONLY): H2frozen = bars[6077:6570] of 6570-era grid (frozen indices);
B = fresh last-200 4h bars; C = last-500 4h bars (of current bar count).
Per-coin short + 50/50 combo short. PASS: B-short sharpe>1.0 AND B-short-fee2x>0.8 AND C-short>1.5.
Helpers imported from research/run_p2.py (load_bars/build_mats/leg_series/combo/stats).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import json
from run_p2 import load_bars, build_mats, leg_series, stats, combo, FORMULA, BEST, COINS, BASE_FEE, FUND

FEE2X = 0.0008
H2_LO, H2_HI = 6077, 6570  # frozen 6570-era grid indices

def percoin_short(mats_seg, fee):
    out = {}
    for c in COINS:
        raw, rt, sg = mats_seg[c]
        lth, sth, cd, sl = BEST[c]
        net, t = leg_series(raw, rt, sg, lth, sth, cd, sl, fee=fee, side='short')
        s = stats(net)
        s['trades'] = t
        out[c] = s
    return out

def combo_short(mats_seg, fee):
    legs, tr = [], {}
    for c in COINS:
        raw, rt, sg = mats_seg[c]
        lth, sth, cd, sl = BEST[c]
        net, t = leg_series(raw, rt, sg, lth, sth, cd, sl, fee=fee, side='short')
        legs.append(net)
        tr[c] = t
    cb = combo(legs, [0.5, 0.5])
    s = stats(cb)
    s['trades'] = sum(tr.values())
    s['trades_by'] = tr
    return s

def main():
    full = {c: load_bars(c) for c in COINS}
    n = min(len(b) for b in full.values())
    print('full 4h bars n=' + str(n), flush=True)
    assert n >= H2_HI, 'need >=6570 bars for frozen window, got %d' % n
    segH = {c: build_mats(full[c][H2_LO:H2_HI]) for c in COINS}
    segB = {c: build_mats(full[c][n - 200:n]) for c in COINS}
    segC = {c: build_mats(full[c][n - 500:n]) for c in COINS}
    print('H2frozen n=%d | B tail200 | C last500' % (H2_HI - H2_LO), flush=True)

    res = {}
    res['H2frozen_short'] = {'percoin': percoin_short(segH, BASE_FEE), 'combo': combo_short(segH, BASE_FEE)}
    res['H2frozen_short_fee2x'] = {'combo': combo_short(segH, FEE2X)}
    res['B_tail200_short'] = {'percoin': percoin_short(segB, BASE_FEE), 'combo': combo_short(segB, BASE_FEE)}
    res['B_tail200_short_fee2x'] = {'percoin': percoin_short(segB, FEE2X), 'combo': combo_short(segB, FEE2X)}
    res['C_last500_short'] = {'percoin': percoin_short(segC, BASE_FEE), 'combo': combo_short(segC, BASE_FEE)}
    res['C_last500_short_fee2x'] = {'percoin': percoin_short(segC, FEE2X), 'combo': combo_short(segC, FEE2X)}

    b = res['B_tail200_short']['combo']
    bf = res['B_tail200_short_fee2x']['combo']
    c = res['C_last500_short']['combo']
    res['PASS'] = {
        'B_short_sharpe_gt_1.0': b['sharpe'] > 1.0,
        'B_short_fee2x_gt_0.8': bf['sharpe'] > 0.8,
        'C_short_gt_1.5': c['sharpe'] > 1.5,
    }
    res['PASS']['overall'] = all(res['PASS'].values())
    res['config'] = {'formula': FORMULA, 'best': {k: list(v[:3]) + [v[3]] for k, v in BEST.items()},
                     'weights': {'ETC': 0.5, 'TRX': 0.5}, 'lev': 2.0, 'side': 'short',
                     'fund': FUND, 'fee': BASE_FEE, 'fee2x': FEE2X,
                     'full_n': n, 'H2frozen': [H2_LO, H2_HI]}

    for k, v in res.items():
        if k in ('PASS', 'config'):
            continue
        print(k + ' ' + json.dumps(v), flush=True)
    print('PASS ' + json.dumps(res['PASS']), flush=True)
    open('results/backtest_S1_shortgate.json', 'w').write(json.dumps(res, indent=1))
    print('saved results/backtest_S1_shortgate.json', flush=True)

if __name__ == '__main__':
    main()
