"""S3 leg-replacement sweep: TRX-leg diagnosis + DOGE replacement.
Locked per-leg params from Q1 H1-best, baseline formula.
FORMULA [3,2,7,2,7,11,15,4,4,6,6,10], aster perp 2x, fund 0.0005, fee base 0.0004.
Pairs (equal-weight): ETC+TRX (incumbent), ETC+DOGE, TRX+DOGE, ETC+TRX+DOGE (3-leg).
Segments: H1/H2 = frozen OOS halves (85% cut); B = last 200 4h bars; C = last 500.
Both-legs; fee 2x (0.0008) stress on B and C.
Method mirrors research/run_p2.py (leg_series + combo + stats).
PASS bar per pair: B_both > 1.0 AND C_both > 1.5 AND fee2x-B_both > 0.5.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json, math, statistics
import torch
from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest

FORMULA = [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10]
BEST = {'ETC': (0.85, 0.15, 12, None), 'TRX': (0.85, 0.12, 6, None), 'DOGE': (0.85, 0.15, 12, None)}
PAIRS = {'ETC_TRX': ['ETC', 'TRX'], 'ETC_DOGE': ['ETC', 'DOGE'],
         'TRX_DOGE': ['TRX', 'DOGE'], 'ETC_TRX_DOGE': ['ETC', 'TRX', 'DOGE']}
BASE_FEE = 0.0004
FEE2X = 0.0008
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

def leg_series(raw, rets_t, sig, lth, sth, cd, sl, fee=None, side='both'):
    kw = dict(venue='aster', leverage=2.0, short_enabled=True, long_th=lth,
              short_th=sth, cooldown_bars=cd, bars_per_year=2190.0, stop_loss=sl)
    kw['funding_override'] = FUND
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

def combo(ser_list):
    m = min(len(s) for s in ser_list)
    k = len(ser_list)
    return [sum(ser_list[i][t] for i in range(k)) / k for t in range(m)]

def eval_pair(mats_seg, coins, fee):
    legs, tr = [], []
    for c in coins:
        raw, rt, sg = mats_seg[c]
        lth, sth, cd, sl = BEST[c]
        net, t = leg_series(raw, rt, sg, lth, sth, cd, sl, fee=fee, side='both')
        legs.append(net)
        tr.append(t)
    s = stats(combo(legs))
    s['trades'] = sum(tr)
    s['trades_by'] = {c: tr[i] for i, c in enumerate(coins)}
    return s, legs

def main():
    coins_u = sorted({c for cl in PAIRS.values() for c in cl})
    full = {c: load_bars(c) for c in coins_u}
    n = min(len(b) for b in full.values())
    print('full 4h bars n=' + str(n), flush=True)
    cut = int(n * 0.85)
    oos = {c: full[c][cut:n] for c in coins_u}
    h = len(oos[coins_u[0]]) // 2
    segH1 = {c: build_mats(oos[c][:h]) for c in coins_u}
    segH2 = {c: build_mats(oos[c][h:]) for c in coins_u}
    segB = {c: build_mats(full[c][n - 200:n]) for c in coins_u}
    segC = {c: build_mats(full[c][n - 500:n]) for c in coins_u}
    print('H1 n=%d H2 n=%d B=200 C=500' % (h, len(oos[coins_u[0]][h:])), flush=True)
    segs = {'H1': segH1, 'H2': segH2, 'B_tail200': segB, 'C_last500': segC}

    res = {'pairs': {}, 'percoin': {}}
    # per-coin diagnosis: both/long/short on H2, B, C at base fee
    for c in coins_u:
        d = {}
        for sname, mats in [('H2', segH2), ('B_tail200', segB), ('C_last500', segC)]:
            dd = {}
            for side in ('both', 'long', 'short'):
                raw, rt, sg = mats[c]
                lth, sth, cd, sl = BEST[c]
                net, t = leg_series(raw, rt, sg, lth, sth, cd, sl, fee=BASE_FEE, side=side)
                s = stats(net)
                s['trades'] = t
                dd[side] = s
            d[sname] = dd
        res['percoin'][c] = d
        print('%s H2 both=%.3f B both=%.3f C both=%.3f' % (
            c, d['H2']['both']['sharpe'], d['B_tail200']['both']['sharpe'],
            d['C_last500']['both']['sharpe']), flush=True)

    for pname, cl in PAIRS.items():
        p = {}
        for sname in ('H1', 'H2', 'B_tail200', 'C_last500'):
            s, _ = eval_pair(segs[sname], cl, BASE_FEE)
            p[sname] = s
        for sname, mats in [('B_tail200', segB), ('C_last500', segC)]:
            s, _ = eval_pair(mats, cl, FEE2X)
            p[sname + '_fee2x'] = s
        # H2 leg corr (both-legs base fee series)
        legs = []
        for c in cl:
            raw, rt, sg = segH2[c]
            lth, sth, cd, sl = BEST[c]
            net, _ = leg_series(raw, rt, sg, lth, sth, cd, sl, fee=BASE_FEE, side='both')
            legs.append(net)
        corr = {}
        for i in range(len(cl)):
            for j in range(i + 1, len(cl)):
                try:
                    v = statistics.correlation(legs[i], legs[j])
                except Exception:
                    v = 0.0
                corr[cl[i] + '_' + cl[j]] = round(v, 3)
        p['H2_leg_corr'] = corr
        b, c_, bf = p['B_tail200']['sharpe'], p['C_last500']['sharpe'], p['B_tail200_fee2x']['sharpe']
        p['PASS'] = {'B_gt_1': b > 1.0, 'C_gt_1.5': c_ > 1.5, 'fee2xB_gt_0.5': bf > 0.5}
        p['PASS']['overall'] = all(p['PASS'].values())
        res['pairs'][pname] = p
        print('%s H1=%.3f H2=%.3f B=%.3f C=%.3f Bf2x=%.3f Cf2x=%.3f corr=%s PASS=%s' % (
            pname, p['H1']['sharpe'], p['H2']['sharpe'], b, c_, bf,
            p['C_last500_fee2x']['sharpe'], corr, p['PASS']['overall']), flush=True)

    order = sorted(res['pairs'], key=lambda k: res['pairs'][k]['C_last500']['sharpe'], reverse=True)
    res['rank_by_C'] = order
    res['config'] = {'formula': FORMULA, 'best': {k: list(v[:3]) + [v[3]] for k, v in BEST.items()},
                     'pairs': PAIRS, 'lev': 2.0, 'fund': FUND, 'fee': BASE_FEE, 'fee2x': FEE2X,
                     'full_n': n, 'cut': cut, 'H1_n': h, 'H2_n': len(oos[coins_u[0]][h:]),
                     'pass_bar': 'B>1.0 AND C>1.5 AND fee2x-B>0.5 (both-leg sharpe)'}
    open('results/backtest_S3_legreplace.json', 'w').write(json.dumps(res, indent=1))
    print('saved backtest_S3_legreplace.json rank=' + str(order), flush=True)

if __name__ == '__main__':
    main()
