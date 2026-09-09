"""P2 paper-gate: locked ETC+TRX baseline on fresh tail + fee stress.
Locked config: FORMULA [3,2,7,2,7,11,15,4,4,6,6,10], ETC (0.88/0.12/cd12/None)
+ TRX (0.85/0.15/cd6/sl0.05), 50/50 equal-weight, aster perp 2x fund 0.0005 fee 0.0004.
Segments: A = frozen OOS H2 (2nd half of last 15% OOS); B = last 200 4h bars;
C = last 500 4h bars. Fee 2x (0.0008) stress on B and C.
Method mirrors run_grouptest678910.py (net_series_side + combo_stats).
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

def leg_series(raw, rets_t, sig, lth, sth, cd, sl, fee=None, fund=None, side='both'):
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
    # trades = entries: transitions from flat (0) to non-flat (+1/-1)
    trades = sum(1 for t in range(len(pos)) if pos[t] != 0.0 and (t == 0 or pos[t - 1] == 0.0))
    # round-trip exits (entries that later close) ~ entries; report entries
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
    cb = [sum(ser_list[i][t] * w[i] / sw for i in range(k)) for t in range(m)]
    return cb

def eval_seg(mats_seg, fee, sides=('both', 'long', 'short')):
    out = {}
    for side in sides:
        legs, tr = [], []
        for c in COINS:
            raw, rt, sg = mats_seg[c]
            lth, sth, cd, sl = BEST[c]
            net, t = leg_series(raw, rt, sg, lth, sth, cd, sl, fee=fee, side=side)
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
    segA = {c: build_mats(oos[c][h:]) for c in COINS}       # frozen H2
    segB = {c: build_mats(full[c][n - 200:n]) for c in COINS}  # fresh tail 200
    segC = {c: build_mats(full[c][n - 500:n]) for c in COINS}  # last 500
    print('A(H2) n=%d | B(tail200) | C(last500)' % len(oos['ETC'][h:]), flush=True)

    res = {}
    res['A_H2_both'] = eval_seg(segA, fee=BASE_FEE, sides=('both',))
    res['B_tail200'] = eval_seg(segB, fee=BASE_FEE)
    res['C_last500'] = eval_seg(segC, fee=BASE_FEE)
    res['B_tail200_fee2x'] = eval_seg(segB, fee=0.0008)
    res['C_last500_fee2x'] = eval_seg(segC, fee=0.0008)

    b = res['B_tail200']['both']
    bf = res['B_tail200_fee2x']['both']
    res['PASS'] = {'B_both_sharpe_gt_1': b['sharpe'] > 1.0,
                   'B_trades_ge_5': b['trades'] >= 5,
                   'B_fee2x_sharpe_gt_0.5': bf['sharpe'] > 0.5}
    res['PASS']['overall'] = all(res['PASS'].values())
    res['config'] = {'formula': FORMULA, 'best': {k: list(v[:3]) + [v[3]] for k, v in BEST.items()},
                     'weights': {'ETC': 0.5, 'TRX': 0.5}, 'lev': 2.0,
                     'fund': FUND, 'fee': BASE_FEE, 'fee2x': 0.0008,
                     'full_n': n, 'A_n': len(oos['ETC'][h:])}
    for k, v in res.items():
        if k in ('PASS', 'config'):
            continue
        print(k + ' ' + json.dumps(v), flush=True)
    print('PASS ' + json.dumps(res['PASS']), flush=True)
    open('results/backtest_P2_papergate.json', 'w').write(json.dumps(res, indent=1))
    print('saved backtest_P2_papergate.json', flush=True)

if __name__ == '__main__':
    main()
