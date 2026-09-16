"""X5 extended cooldown sweep on S2 q=0.3 variant.
Locked base: FORMULA [3,2,7,2,7,11,15,4,4,6,6,10], ETC (0.88/0.12/cd12/None)
+ TRX (0.85/0.15/cd6/sl0.05), 50/50, aster perp 2x fund 0.0005 fee 0.0004,
S2 long-leg quantile filter q=0.3 (side='long', short untouched).
T3: time_stop [0,12,24,48] x take_profit [None,0.08,0.15] = 12 rows, both-legs.
    SELECT on H2 top2, VERIFY B+C. PASS: selected row B>1.2 AND C>1.8.
X5: cd pairs (ETC,TRX) [(12,6),(18,6),(12,9),(18,9),(24,12)] x sth [0.12,0.15] = 10 rows,
    both-legs (long_th fixed, sl fixed, q=0.3, ts=0, tp=None). PASS: best B>1.2 AND C>1.8 with H2>1.5.
Segments: H2 = 2nd half of last 15% OOS (frozen); B = last 200 4h bars; C = last 500.
Mirrors research/run_s2.py leg_series + MemeBacktest._apply_stops/_apply_cooldown.
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
Q = 0.3

T3_TS = [0, 12, 24, 48]
T3_TP = [None, 0.08, 0.15]

X5_CD = [(12, 6), (18, 6), (12, 9), (18, 9), (24, 12)]
X5_STH = [0.12, 0.15]

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

def eval_both(mats_seg, fee, per_coin):
    # per_coin: {coin: (lth, sth, cd, sl, ts, tp)}
    legs, tr = [], []
    for c in COINS:
        raw, rt, sg = mats_seg[c]
        lth, sth, cd, sl, ts, tp = per_coin[c]
        net, t = leg_series(raw, rt, sg, lth, sth, cd, sl, fee=fee,
                            side='both', q=Q, time_stop=ts, take_profit=tp)
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
    cut = int(n * 0.85)
    oos = {c: full[c][cut:n] for c in COINS}
    h = len(oos['ETC']) // 2
    segH2 = {c: build_mats(oos[c][h:]) for c in COINS}
    segB = {c: build_mats(full[c][n - 200:n]) for c in COINS}
    segC = {c: build_mats(full[c][n - 500:n]) for c in COINS}
    print('H2 n=%d | B tail200 | C last500 | q=%.1f' % (len(oos['ETC'][h:]), Q), flush=True)

    res = {'q': Q}
    # ---- X5 ----
    x5 = []
    for (cd_e, cd_t) in X5_CD:
        for sth in X5_STH:
            pc = {}
            for c in COINS:
                lth, _, _, sl = BEST[c]
                cd = cd_e if c == 'ETC' else cd_t
                pc[c] = (lth, sth, cd, sl, 0, None)
            rH2 = eval_both(segH2, BASE_FEE, pc)
            rB = eval_both(segB, BASE_FEE, pc)
            rC = eval_both(segC, BASE_FEE, pc)
            row = {'cd_etc': cd_e, 'cd_trx': cd_t, 'sth': sth,
                   'H2': rH2, 'B': rB, 'C': rC}
            x5.append(row)
            print('X5 cd=(%d/%d) sth=%s H2=%s B=%s C=%s' % (
                cd_e, cd_t, sth, json.dumps(rH2), json.dumps(rB), json.dumps(rC)), flush=True)
    res['rows'] = x5
    best = max(range(len(x5)), key=lambda i: (x5[i]['B']['sharpe'], x5[i]['C']['sharpe']))
    res['best_idx'] = best
    res['best'] = x5[best]
    bB, bC, bH = x5[best]['B'], x5[best]['C'], x5[best]['H2']
    res['PASS'] = {'best': {'cd_etc': x5[best]['cd_etc'], 'cd_trx': x5[best]['cd_trx'], 'sth': x5[best]['sth']},
                   'B_gt_1.2': bool(bB['sharpe'] > 1.2),
                   'C_gt_1.8': bool(bC['sharpe'] > 1.8),
                   'H2_gt_1.5': bool(bH['sharpe'] > 1.5),
                   'overall': bool(bB['sharpe'] > 1.2 and bC['sharpe'] > 1.8 and bH['sharpe'] > 1.5)}
    res['config'] = {'formula': FORMULA,
                     'best': {k: list(v[:3]) + [v[3]] for k, v in BEST.items()},
                     'weights': {'ETC': 0.5, 'TRX': 0.5}, 'lev': 2.0,
                     'fund': FUND, 'fee': BASE_FEE, 'q': Q, 'full_n': n,
                     'H2_n': len(oos['ETC'][h:]),
                     'x5_grid': {'cd_pairs': [list(p) for p in X5_CD], 'sth': X5_STH},
                     'note': 'S2 q=0.3 long-quantile filter; sl locked (ETC None, TRX 0.05); X5 varies both-coin sth + per-coin cd pairs; ts=0 tp=None'}
    open('results/backtest_X5.json', 'w').write(json.dumps(res, indent=1))
    print('X5_PASS ' + json.dumps(res['PASS']), flush=True)
    print('saved results/backtest_X5.json', flush=True)

if __name__ == '__main__':
    main()
