"""Y1 ensemble ladder: X5cd x X6sl/off x X4ts x X3-weights.
Locked engine mirrors research/run_x5.py exactly: FORMULA [3,2,7,2,7,11,15,4,4,6,6,10],
aster perp 2x fund 0.0005 fee base 0.0004 / 2x 0.0008, S2 q=0.3 long-leg quantile gate,
leg_series + quantile_mask_long + combo mirrored from run_x5.py.
X5 best: shared sth 0.12, ETC-cd18/slNone + TRX-cd6/sl0.05, 50/50.
Ladder (all q=0.3 long-gate):
  (a) X5 alone; (b) X5+X4 ts24; (c) X5+X6 shared sl0.02 + TRX-long-off;
  (d) full stack X5cd+X6sl/off+X4ts 50/50; (e) full stack wETC0.1.
Segments mirror run_x5.py: H2 = 2nd half of last 15%% OOS; B = last 200; C = last 500.
Per row: H2/B/C sharpe-ann-mdd-tr + B/C-fee2x.
Best = argmax(B_sharpe + C_sharpe). PASS: best B>2.0 AND C>2.5 AND H2>2.0 AND fee2x-B>1.0.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json, math
import torch
from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest

FORMULA = [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10]
COINS = ['ETC', 'TRX']
BASE_FEE = 0.0004
FEE2X = 0.0008
FUND = 0.0005
Q = 0.3
STH = 0.12

def load_bars(coin):
    import csv
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

def quantile_mask_long(sig, q):
    if q is None:
        return None
    a = sig.detach().float().abs().reshape(-1)
    k = max(1, int(len(a) * float(q)))
    thr = torch.topk(a, k).values.min()
    return (sig.detach().float().abs() >= thr).float()

def leg_series(raw, rets_t, sig, lth, sth, cd, sl, fee=None, fund=None,
               side='both', q=Q, time_stop=0, take_profit=None):
    kw = dict(venue='aster', leverage=2.0, short_enabled=True, long_th=lth,
              short_th=sth, cooldown_bars=cd, bars_per_year=2190.0, stop_loss=sl,
              time_stop=time_stop, take_profit=take_profit)
    kw['funding_override'] = FUND if fund is None else fund
    if fee is not None:
        kw['fee_override'] = fee
    bt = MemeBacktest(**kw)
    signal = torch.sigmoid(sig)
    is_safe = (raw['liquidity'] > bt.min_liq).float()
    lp = (signal > bt.long_th).float() * is_safe
    sp = (signal < bt.short_th).float() * is_safe
    mask = quantile_mask_long(sig, q)
    if mask is not None:
        lp = lp * mask
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
    return {'sharpe': round(sharpe, 3), 'ann': round(ann, 4), 'mdd': round(mdd, 4),
            'cum': round(cum, 4), 'n': n}

def combo(ser_list, weights=None):
    m = min(len(s) for s in ser_list)
    k = len(ser_list)
    w = weights or [1.0 / k] * k
    sw = sum(w)
    return [sum(ser_list[i][t] * w[i] / sw for i in range(k)) for t in range(m)]

# per-coin spec: (lth, sth, cd, sl, ts, tp, side); X5 best sth=0.12 shared both coins.
LADDER = [
    {'name': 'a_X5_alone', 'w': [0.5, 0.5],
     'ETC': (0.88, STH, 18, None, 0, None, 'both'),
     'TRX': (0.85, STH, 6, 0.05, 0, None, 'both')},
    {'name': 'b_X5_X4ts24', 'w': [0.5, 0.5],
     'ETC': (0.88, STH, 18, None, 24, None, 'both'),
     'TRX': (0.85, STH, 6, 0.05, 24, None, 'both')},
    {'name': 'c_X5_X6sl002_trxoff', 'w': [0.5, 0.5],
     'ETC': (0.88, STH, 18, 0.02, 0, None, 'both'),
     'TRX': (0.85, STH, 6, 0.02, 0, None, 'short')},
    {'name': 'd_full_5050', 'w': [0.5, 0.5],
     'ETC': (0.88, STH, 18, 0.02, 24, None, 'both'),
     'TRX': (0.85, STH, 6, 0.02, 24, None, 'short')},
    {'name': 'e_full_wETC01', 'w': [0.1, 0.9],
     'ETC': (0.88, STH, 18, 0.02, 24, None, 'both'),
     'TRX': (0.85, STH, 6, 0.02, 24, None, 'short')},
]

def eval_row(mats_seg, fee, spec):
    legs, tr = [], []
    for c in COINS:
        raw, rt, sg = mats_seg[c]
        lth, sth, cd, sl, ts, tp, side = spec[c]
        net, t = leg_series(raw, rt, sg, lth, sth, cd, sl, fee=fee,
                            side=side, q=Q, time_stop=ts, take_profit=tp)
        legs.append(net)
        tr.append(t)
    cb = combo(legs, spec['w'])
    s = stats(cb)
    s['trades'] = sum(tr)
    s['trades_by'] = {c: tr[i] for i, c in enumerate(COINS)}
    return s

def main():
    full = {c: load_bars(c) for c in COINS}
    n = min(len(b) for b in full.values())
    print('full 4h bars n=' + str(n), flush=True)
    cut = int(n * 0.85)
    oos = {c: full[c][cut:n] for c in COINS}
    h = len(oos['ETC']) // 2
    segH2 = {c: build_mats(oos[c][h:]) for c in COINS}
    segB = {c: build_mats(full[c][n - 200:n]) for c in COINS}
    segC = {c: build_mats(full[c][n - 500:n]) for c in COINS}
    print('H2 n=%d | B tail200 | C last500 | q=%.1f sth=%.2f' % (len(oos['ETC'][h:]), Q, STH), flush=True)
    rows = []
    for spec in LADDER:
        row = {'name': spec['name'], 'w': spec['w'],
               'ETC': list(spec['ETC']), 'TRX': list(spec['TRX'])}
        row['H2'] = eval_row(segH2, BASE_FEE, spec)
        row['B'] = eval_row(segB, BASE_FEE, spec)
        row['C'] = eval_row(segC, BASE_FEE, spec)
        row['B_fee2x'] = eval_row(segB, FEE2X, spec)
        row['C_fee2x'] = eval_row(segC, FEE2X, spec)
        rows.append(row)
        print('%s H2=%s B=%s C=%s B2x=%s C2x=%s' % (
            spec['name'], row['H2']['sharpe'], row['B']['sharpe'], row['C']['sharpe'],
            row['B_fee2x']['sharpe'], row['C_fee2x']['sharpe']), flush=True)
    best = max(range(len(rows)), key=lambda i: (rows[i]['B']['sharpe'] + rows[i]['C']['sharpe']))
    b = rows[best]
    res = {'config': {'formula': FORMULA, 'engine': 'StackVM+MemeBacktest (mirrors run_x5.py)',
                     'venue': 'aster', 'lev': 2.0, 'fund': FUND, 'fee': BASE_FEE, 'fee2x': FEE2X,
                     'q': Q, 'sth_shared': STH, 'full_n': n, 'H2_n': len(oos['ETC'][h:]),
                     'segs': {'H2': [cut + h, n], 'B': [n - 200, n], 'C': [n - 500, n]},
                     'ladder': [s['name'] for s in LADDER],
                     'best_rule': 'argmax(B_sharpe + C_sharpe)',
                     'filter': 'quantile top-0.3 on |logit|, long leg only'}}
    res['rows'] = rows
    res['best'] = b
    res['PASS'] = {'best': b['name'],
                   'B_gt_2': bool(b['B']['sharpe'] > 2.0),
                   'C_gt_2.5': bool(b['C']['sharpe'] > 2.5),
                   'H2_gt_2': bool(b['H2']['sharpe'] > 2.0),
                   'fee2xB_gt_1': bool(b['B_fee2x']['sharpe'] > 1.0),
                   'overall': bool(b['B']['sharpe'] > 2.0 and b['C']['sharpe'] > 2.5
                                      and b['H2']['sharpe'] > 2.0 and b['B_fee2x']['sharpe'] > 1.0)}
    open('results/backtest_Y1.json', 'w').write(json.dumps(res, indent=1))
    print('Y1_BEST ' + b['name'] + ' H2=%s B=%s C=%s B2x=%s' % (
        b['H2']['sharpe'], b['B']['sharpe'], b['C']['sharpe'], b['B_fee2x']['sharpe']), flush=True)
    print('Y1_PASS ' + json.dumps(res['PASS']), flush=True)
    print('saved results/backtest_Y1.json', flush=True)

if __name__ == '__main__':
    main()
