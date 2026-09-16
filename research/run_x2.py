"""X2 q_long sweep on Q1 ETC+TRX pair. Locked FORMULA [3,2,7,2,7,11,15,4,4,6,6,10],
aster perp 2x fund 0.0005 fee 0.0004. Q1 legs ETC(0.85/0.15/cd12/None) +
TRX(0.85/0.12/cd6/None), 50/50 both-legs. Quantile gate on LONG leg only
(mirrors research/run_s2.py leg_series + quantile_mask_long).
Segments (4h-bar slices of full n=6580): H2=[6077:6570], B=[6380:6580], C=[6080:6580].
Sweep q_long in [None,0.2,0.25,0.3,0.35,0.4,0.5]: per q H2/B/C + B-fee2x(0.0008) + C-fee2x.
Best q = argmax(B_sharpe + C_sharpe). PASS: best q has B>1.0 AND C>1.5 AND fee2x-B>0.5.
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
BEST = {'ETC': (0.85, 0.15, 12, None), 'TRX': (0.85, 0.12, 6, None)}
COINS = ['ETC', 'TRX']
BASE_FEE = 0.0004
FEE2X = 0.0008
FUND = 0.0005
QS = [None, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
SEGS = {'H2': (6077, 6570), 'B': (6380, 6580), 'C': (6080, 6580)}

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
    sig = StackVM(use_advanced=False).execute(FORMULA, FeatureEngineer.compute_features(raw, use_advanced=False))
    rets = [(bars[i + 1][3] - bars[i][3]) / bars[i][3] for i in range(n - 1)] + [0.0]
    return raw, torch.tensor([rets]), sig

def quantile_mask_long(sig, q):
    """Top-q fraction mask on |logit|, mirrors MemeBacktest._apply_quantile(side='long')."""
    if q is None:
        return None
    a = sig.detach().float().abs().reshape(-1)
    k = max(1, int(len(a) * float(q)))
    thr = torch.topk(a, k).values.min()
    return (sig.detach().float().abs() >= thr).float()

def leg_series(raw, rets_t, sig, lth, sth, cd, sl, fee=None, fund=None, side='both', q=None):
    kw = dict(venue='aster', leverage=2.0, short_enabled=True, long_th=lth,
              short_th=sth, cooldown_bars=cd, bars_per_year=2190.0, stop_loss=sl)
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

def eval_combo(mats, fee, q):
    legs, tr = [], []
    for c in COINS:
        raw, rt, sg = mats[c]
        lth, sth, cd, sl = BEST[c]
        net, t = leg_series(raw, rt, sg, lth, sth, cd, sl, fee=fee, side='both', q=q)
        legs.append(net)
        tr.append(t)
    cb = combo(legs, [0.5, 0.5])
    s = stats(cb)
    s['trades'] = sum(tr)
    s['trades_by'] = {c: tr[i] for i, c in enumerate(COINS)}
    return s

def main():
    full = {c: load_bars(c) for c in COINS}
    n = min(len(b) for b in full.values())
    print('full 4h bars n=' + str(n), flush=True)
    mats = {name: {c: build_mats(full[c][a:b]) for c in COINS} for name, (a, b) in SEGS.items()}
    for name, (a, b) in SEGS.items():
        print('%s [%d:%d] n=%d' % (name, a, b, b - a), flush=True)
    rows = []
    for q in QS:
        key = str(q)
        row = {'q': key}
        row['H2'] = eval_combo(mats['H2'], BASE_FEE, q)
        row['B'] = eval_combo(mats['B'], BASE_FEE, q)
        row['C'] = eval_combo(mats['C'], BASE_FEE, q)
        row['B_fee2x'] = eval_combo(mats['B'], FEE2X, q)
        row['C_fee2x'] = eval_combo(mats['C'], FEE2X, q)
        rows.append(row)
        print('q=%s H2_sh=%s B_sh=%s C_sh=%s B2x_sh=%s C2x_sh=%s trades(H2/B/C)=%d/%d/%d' % (
            key, row['H2']['sharpe'], row['B']['sharpe'], row['C']['sharpe'],
            row['B_fee2x']['sharpe'], row['C_fee2x']['sharpe'],
            row['H2']['trades'], row['B']['trades'], row['C']['trades']), flush=True)
    best = max(rows, key=lambda r: r['B']['sharpe'] + r['C']['sharpe'])
    bSh, cSh, b2Sh = best['B']['sharpe'], best['C']['sharpe'], best['B_fee2x']['sharpe']
    passed = bool(bSh > 1.0 and cSh > 1.5 and b2Sh > 0.5)
    res = {
        'config': {'formula': FORMULA,
                   'legs': {k: {'long_th': v[0], 'short_th': v[1], 'cooldown': v[2], 'stop_loss': v[3]} for k, v in BEST.items()},
                   'weights': {'ETC': 0.5, 'TRX': 0.5}, 'engine': 'StackVM+MemeBacktest',
                   'venue': 'aster', 'lev': 2.0, 'fund': FUND, 'fee': BASE_FEE, 'fee2x': FEE2X,
                   'full_n': n, 'segs': {k: list(v) for k, v in SEGS.items()},
                   'qs': [str(q) for q in QS],
                   'filter': 'quantile top-q on |logit|, long leg only; short untouched',
                   'best_rule': 'argmax(B_sharpe + C_sharpe)'},
        'rows': rows,
        'best': {'q': best['q'], 'H2': best['H2'], 'B': best['B'], 'C': best['C'],
                 'B_fee2x': best['B_fee2x'], 'C_fee2x': best['C_fee2x']},
        'PASS': {'best_q': best['q'], 'B_sharpe': bSh, 'C_sharpe': cSh, 'B_fee2x_sharpe': b2Sh,
                 'cond_B_gt_1': bool(bSh > 1.0), 'cond_C_gt_1_5': bool(cSh > 1.5),
                 'cond_fee2xB_gt_0_5': bool(b2Sh > 0.5), 'pass': passed},
    }
    open('results/backtest_X2.json', 'w').write(json.dumps(res, indent=1))
    print('BEST q=%s B=%s C=%s B_fee2x=%s' % (best['q'], bSh, cSh, b2Sh), flush=True)
    print('PASS ' + json.dumps(res['PASS']), flush=True)
    print('saved results/backtest_X2.json', flush=True)

if __name__ == '__main__':
    main()
