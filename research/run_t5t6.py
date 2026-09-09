"""T5+T6: tail checks + weight sweep. Locked FORMULA, aster 2x fund 0.0005 fee 0.0004.
Per-leg params = Q1 H1-best per coin (results/backtest_Q1_5coin.json).
T5 combos (equal-weight, both-legs): BASE ETC+TRX, ETC+TRX+DOGE, ALL5, MAJORS(BTC+SOL+DOGE).
T6: weight sweep on base pair wETC in [0.2,0.3,0.5,0.7,0.8], plain + S2 q=0.3 long-gate variant.
Segments: H2 = 2nd half of last 15% OOS (frozen); B = last 200 4h bars; C = last 500 4h bars.
Method mirrors research/run_p2.py + research/run_s2.py (quantile gate on long only).
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
BASE_FEE = 0.0004
FUND = 0.0005
# Q1 H1-best per coin (from results/backtest_Q1_5coin.json)
BEST = {
    'BTC': (0.85, 0.12, 12, 0.05),
    'SOL': (0.85, 0.10, 3, None),
    'ETC': (0.85, 0.15, 12, None),
    'TRX': (0.85, 0.12, 6, None),
    'DOGE': (0.85, 0.15, 12, None),
}
COMBOS = {
    'BASE_ETC_TRX': ['ETC', 'TRX'],
    'ETC_TRX_DOGE': ['ETC', 'TRX', 'DOGE'],
    'ALL5': ['BTC', 'SOL', 'ETC', 'TRX', 'DOGE'],
    'MAJORS': ['BTC', 'SOL', 'DOGE'],
}
WEIGHTS = [0.2, 0.3, 0.5, 0.7, 0.8]
QVARS = [None, 0.3]

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

def qmask(sig, q):
    if q is None:
        return None
    a = sig.detach().float().abs().reshape(-1)
    k = max(1, int(len(a) * float(q)))
    thr = torch.topk(a, k).values.min()
    return (sig.detach().float().abs() >= thr).float()

def leg_series(raw, rets_t, sig, lth, sth, cd, sl, side='both', q=None):
    kw = dict(venue='aster', leverage=2.0, short_enabled=True, long_th=lth,
              short_th=sth, cooldown_bars=cd, bars_per_year=2190.0, stop_loss=sl,
              funding_override=FUND, fee_override=BASE_FEE)
    bt = MemeBacktest(**kw)
    signal = torch.sigmoid(sig)
    is_safe = (raw['liquidity'] > bt.min_liq).float()
    lp = (signal > bt.long_th).float() * is_safe
    sp = (signal < bt.short_th).float() * is_safe
    mask = qmask(sig, q)
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

def wcombo(ser_list, weights):
    m = min(len(s) for s in ser_list)
    sw = sum(weights)
    return [sum(ser_list[i][t] * weights[i] / sw for i in range(len(ser_list))) for t in range(m)]

def pcorr(a, b):
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((x - mb) ** 2 for x in b)
    if va <= 0 or vb <= 0:
        return 0.0
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    return round(cov / math.sqrt(va * vb), 3)

def main():
    coins5 = ['BTC', 'SOL', 'ETC', 'TRX', 'DOGE']
    full = {c: load_bars(c) for c in coins5}
    n = min(len(b) for b in full.values())
    print('full 4h bars n=' + str(n), flush=True)
    cut = int(n * 0.85)
    oos = {c: full[c][cut:n] for c in coins5}
    h = len(oos['ETC']) // 2
    segs = {
        'H2': {c: build_mats(oos[c][h:]) for c in coins5},
        'B': {c: build_mats(full[c][n - 200:n]) for c in coins5},
        'C': {c: build_mats(full[c][n - 500:n]) for c in coins5},
    }
    print('H2 n=%d | B tail200 | C last500' % len(oos['ETC'][h:]), flush=True)

    # per-coin both-leg series per seg (plain + q0.3 long-gate)
    legs = {}
    for seg, mats in segs.items():
        legs[seg] = {}
        for q in QVARS:
            legs[seg][str(q)] = {}
            for c in coins5:
                raw, rt, sg = mats[c]
                lth, sth, cd, sl = BEST[c]
                net, t = leg_series(raw, rt, sg, lth, sth, cd, sl, side='both', q=q)
                legs[seg][str(q)][c] = {'net': net, 'trades': t}

    res = {'T5': {}, 'T6': {}}
    # ---- T5: equal-weight combos, both-legs, plain ----
    for name, cl in COMBOS.items():
        res['T5'][name] = {'coins': cl}
        for seg in ('H2', 'B', 'C'):
            sl = [legs[seg]['None'][c]['net'] for c in cl]
            cb = wcombo(sl, [1.0] * len(cl))
            s = stats(cb)
            s['trades'] = sum(legs[seg]['None'][c]['trades'] for c in cl)
            s['trades_by'] = {c: legs[seg]['None'][c]['trades'] for c in cl}
            res['T5'][name][seg] = s
        # B leg-corr (pairwise)
        corr = {}
        bl = {c: legs['B']['None'][c]['net'] for c in cl}
        for i in range(len(cl)):
            for j in range(i + 1, len(cl)):
                corr['%s_%s' % (cl[i], cl[j])] = pcorr(bl[cl[i]], bl[cl[j]])
        res['T5'][name]['B_leg_corr'] = corr
        print('T5 %s H2=%s B=%s C=%s' % (name, json.dumps(res['T5'][name]['H2']),
              json.dumps(res['T5'][name]['B']), json.dumps(res['T5'][name]['C'])), flush=True)
    bB = res['T5']['BASE_ETC_TRX']['B']['sharpe']
    bC = res['T5']['BASE_ETC_TRX']['C']['sharpe']
    t5pass = {}
    for name in COMBOS:
        if name == 'BASE_ETC_TRX':
            continue
        t5pass[name] = bool(res['T5'][name]['B']['sharpe'] > bB and res['T5'][name]['C']['sharpe'] > bC)
    res['T5']['PASS'] = {'base_B_sharpe': bB, 'base_C_sharpe': bC,
                         'beats_base_on_B_and_C': t5pass,
                         'overall': bool(any(t5pass.values()))}

    # ---- T6: weight sweep on base pair, plain + q0.3 ----
    for q in QVARS:
        key = 'q' + str(q)
        res['T6'][key] = {}
        for w in WEIGHTS:
            res['T6'][key]['wETC_' + str(w)] = {}
            for seg in ('H2', 'B', 'C'):
                e = legs[seg][str(q)]['ETC']['net']
                t = legs[seg][str(q)]['TRX']['net']
                cb = wcombo([e, t], [w, 1 - w])
                s = stats(cb)
                s['trades'] = legs[seg][str(q)]['ETC']['trades'] + legs[seg][str(q)]['TRX']['trades']
                s['trades_by'] = {'ETC': legs[seg][str(q)]['ETC']['trades'],
                                  'TRX': legs[seg][str(q)]['TRX']['trades']}
                res['T6'][key]['wETC_' + str(w)][seg] = s
            print('T6 %s wETC=%s H2=%s B=%s C=%s' % (key, w,
                  json.dumps(res['T6'][key]['wETC_' + str(w)]['H2']['sharpe']),
                  json.dumps(res['T6'][key]['wETC_' + str(w)]['B']['sharpe']),
                  json.dumps(res['T6'][key]['wETC_' + str(w)]['C']['sharpe'])), flush=True)
    t6pass = {}
    for q in QVARS:
        key = 'q' + str(q)
        refB = res['T6'][key]['wETC_0.5']['B']['sharpe']
        refC = res['T6'][key]['wETC_0.5']['C']['sharpe']
        best = None
        for w in WEIGHTS:
            if w == 0.5:
                continue
            dB = res['T6'][key]['wETC_' + str(w)]['B']['sharpe'] - refB
            dC = res['T6'][key]['wETC_' + str(w)]['C']['sharpe'] - refC
            if dB > 0.2 and dC > 0.2 and (best is None or (dB + dC) > best[1]):
                best = (w, dB + dC, round(dB, 3), round(dC, 3))
        t6pass[key] = {'ref50_B': refB, 'ref50_C': refC, 'best': best, 'pass': best is not None}
    res['T6']['PASS'] = t6pass
    res['T6']['PASS']['overall'] = bool(any(v['pass'] for v in t6pass.values()))

    res['config'] = {'formula': FORMULA, 'best_Q1_H1': {k: list(v[:3]) + [v[3]] for k, v in BEST.items()},
                     'fee': BASE_FEE, 'fund': FUND, 'lev': 2.0, 'full_n': n,
                     'H2_n': len(oos['ETC'][h:]), 'B_n': 200, 'C_n': 500,
                     'note': 'Q1 H1-best per-leg params (TRX sl None, sth 0.12); differs from P2/S2 params (TRX 0.85/0.15/cd6/sl0.05)'}
    open('results/backtest_T5T6.json', 'w').write(json.dumps(res, indent=1))
    print('T5 PASS ' + json.dumps(res['T5']['PASS']), flush=True)
    print('T6 PASS ' + json.dumps(res['T6']['PASS']), flush=True)
    print('saved results/backtest_T5T6.json', flush=True)

if __name__ == '__main__':
    main()
