"""S2 long-leg filter: quantile-gated long (side='long'), short untouched.
Locked: FORMULA [3,2,7,2,7,11,15,4,4,6,6,10], ETC (0.88/0.12/cd12/None)
+ TRX (0.85/0.15/cd6/sl0.05), 50/50, aster perp 2x fund 0.0005 fee 0.0004.
Sweep q in [None,0.5,0.3,0.1] on LONG leg only. Segments: A=frozen H2
(2nd half of last 15% OOS), B=last 200 4h bars, C=last 500 4h bars.
Mirrors research/run_p2.py leg_series + model_core/backtest._apply_quantile(side='long').
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
BEST = {'ETC': (0.88, 0.12, 12, None), 'TRX': (0.85, 0.15, 6, 0.05)}
COINS = ['ETC', 'TRX']
BASE_FEE = 0.0004
FUND = 0.0005
QS = [None, 0.5, 0.3, 0.1]

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
    # S2: quantile gate on LONG leg only, short untouched
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

def eval_seg_q(mats_seg, fee, q):
    out = {}
    for side in ('both', 'long', 'short'):
        legs, tr = [], []
        for c in COINS:
            raw, rt, sg = mats_seg[c]
            lth, sth, cd, sl = BEST[c]
            net, t = leg_series(raw, rt, sg, lth, sth, cd, sl, fee=fee, side=side, q=q)
            legs.append(net)
            tr.append(t)
        cb = combo(legs, [0.5, 0.5])
        s = stats(cb)
        s['trades'] = sum(tr)
        s['trades_by'] = {c: tr[i] for i, c in enumerate(COINS)}
        out[side] = s
    return out

def main():
    full = {c: load_bars(c) for c in COINS}
    n = min(len(b) for b in full.values())
    print('full 4h bars n=' + str(n), flush=True)
    cut = int(n * 0.85)
    oos = {c: full[c][cut:n] for c in COINS}
    h = len(oos['ETC']) // 2
    segA = {c: build_mats(oos[c][h:]) for c in COINS}
    segB = {c: build_mats(full[c][n - 200:n]) for c in COINS}
    segC = {c: build_mats(full[c][n - 500:n]) for c in COINS}
    print('A(H2) n=%d | B(tail200) | C(last500)' % len(oos['ETC'][h:]), flush=True)
    res = {'qs': ['None', '0.5', '0.3', '0.1']}
    for q in QS:
        key = str(q)
        res['A_H2_q' + key] = eval_seg_q(segA, BASE_FEE, q)
        res['B_tail200_q' + key] = eval_seg_q(segB, BASE_FEE, q)
        res['C_last500_q' + key] = eval_seg_q(segC, BASE_FEE, q)
        b = res['B_tail200_q' + key]['both']
        bl = res['B_tail200_q' + key]['long']
        c = res['C_last500_q' + key]['both']
        print('q=%s B_both=%s B_long=%s C_both=%s' % (key, json.dumps(b), json.dumps(bl), json.dumps(c)), flush=True)
    # PASS per q on B
    res['PASS'] = {}
    for q in QS:
        key = str(q)
        b = res['B_tail200_q' + key]['both']
        bl = res['B_tail200_q' + key]['long']
        cond1 = b['sharpe'] > 1.0 and bl['trades'] <= 2
        cond2 = b['sharpe'] > 1.2 and bl['cum'] > 0
        res['PASS']['q' + key] = {'cond1_short_only_formally_both': bool(cond1),
                                  'cond2_long_positive': bool(cond2),
                                  'pass': bool(cond1 or cond2)}
    res['config'] = {'formula': FORMULA, 'best': {k: list(v[:3]) + [v[3]] for k, v in BEST.items()},
                     'weights': {'ETC': 0.5, 'TRX': 0.5}, 'lev': 2.0, 'fund': FUND,
                     'fee': BASE_FEE, 'full_n': n, 'A_n': len(oos['ETC'][h:]),
                     'filter': 'quantile top-q on |logit|, side=long only; short untouched'}
    open('results/backtest_S2_longfilter.json', 'w').write(json.dumps(res, indent=1))
    print('PASS ' + json.dumps(res['PASS']), flush=True)
    print('saved results/backtest_S2_longfilter.json', flush=True)

if __name__ == '__main__':
    main()
