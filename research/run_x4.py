"""X4 time-stop/take-profit extend on S2 q0.3 base. Locked FORMULA [3,2,7,2,7,11,15,4,4,6,6,10],
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
BEST = {'ETC': (0.88, 0.12, 12, None), 'TRX': (0.85, 0.15, 6, 0.05)}
TS = [24, 36, 60, 96]
TPS = [None, 0.08]
COINS = ['ETC', 'TRX']
BASE_FEE = 0.0004
FEE2X = 0.0008
FUND = 0.0005
Q = 0.3
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
    sig = StackVM().execute(FORMULA, FeatureEngineer.compute_features(raw))
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

def leg_series(raw, rets_t, sig, lth, sth, cd, sl, fee=None, fund=None, side='both', q=Q, time_stop=0, take_profit=None):
    kw = dict(venue='aster', leverage=2.0, short_enabled=True, long_th=lth,
              short_th=sth, cooldown_bars=cd, bars_per_year=2190.0, stop_loss=sl)
    kw['funding_override'] = FUND if fund is None else fund
    if fee is not None:
        kw['fee_override'] = fee
    kw['time_stop'] = time_stop
    kw['take_profit'] = take_profit
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
        net, t = leg_series(raw, rt, sg, lth, sth, cd, sl, fee=fee, side='both', q=q, time_stop=ts, take_profit=tp)
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
    rows = []
    for ts in TS:
        for tp in TPS:
            row = {'ts': ts, 'tp': str(tp)}
            # inline eval with ts/tp
            legsH, trH = [], []
            legsB, trB = [], []
            legsC, trC = [], []
            for c in COINS:
                lth, sth, cd, sl = BEST[c]
                nH, tH = leg_series(*mats['H2'][c], lth, sth, cd, sl, fee=BASE_FEE, side='both', q=Q, time_stop=ts, take_profit=tp)
                nB, tB = leg_series(*mats['B'][c], lth, sth, cd, sl, fee=BASE_FEE, side='both', q=Q, time_stop=ts, take_profit=tp)
                nC, tC = leg_series(*mats['C'][c], lth, sth, cd, sl, fee=BASE_FEE, side='both', q=Q, time_stop=ts, take_profit=tp)
                legsH.append(nH); trH.append(tH); legsB.append(nB); trB.append(tB); legsC.append(nC); trC.append(tC)
            for key, lt in [('H2', (legsH, trH)), ('B', (legsB, trB)), ('C', (legsC, trC))]:
                legs, trs = lt
                cb = combo(legs, [0.5, 0.5]); s = stats(cb)
                s['trades'] = sum(trs); s['trades_by'] = {c: trs[i] for i, c in enumerate(COINS)}
                row[key] = s
            rows.append(row)
            print('ts=%s tp=%s H2=%s B=%s C=%s' % (ts, str(tp), row['H2']['sharpe'], row['B']['sharpe'], row['C']['sharpe']), flush=True)
    ref = [r for r in rows if r['ts'] == 24 and r['tp'] == 'None'][0]
    best = max(rows, key=lambda r: r['B']['sharpe'] + r['C']['sharpe'])
    passed = bool(best['B']['sharpe'] > 1.2 and best['C']['sharpe'] > 1.8)
    res = {'config': {'formula': FORMULA, 'ts': TS, 'tps': [str(t) for t in TPS], 'q': Q,
                      'ref_ts24': {'B': ref['B']['sharpe'], 'C': ref['C']['sharpe']}},
           'rows': rows,
           'best': {'ts': best['ts'], 'tp': best['tp'], 'H2': best['H2'], 'B': best['B'], 'C': best['C']},
           'PASS': {'B_gt_12': bool(best['B']['sharpe'] > 1.2), 'C_gt_18': bool(best['C']['sharpe'] > 1.8), 'pass': passed}}
    open('results/backtest_X4.json', 'w').write(json.dumps(res, indent=1))
    print('BEST ts=%s tp=%s B=%s C=%s' % (best['ts'], best['tp'], best['B']['sharpe'], best['C']['sharpe']), flush=True)
    print('PASS ' + json.dumps(res['PASS']), flush=True)
    print('saved results/backtest_X4.json', flush=True)

if __name__ == '__main__':
    main()
