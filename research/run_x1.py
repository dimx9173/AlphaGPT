"""X1 short-led lock worker.
Locked: StackVM FORMULA [3,2,7,2,7,11,15,4,4,6,6,10], aster perp 2x fund 0.0005 fee 0.0004.
Q1 legs: ETC (0.85/0.15/cd12/None) + TRX (0.85/0.12/cd6/None), equal-weight 50/50.
Segments (fixed 4h-bar index): H2[6077:6570] B[6380:6580] C[6080:6580].
Variants: (a) plain both, (b) q0.3 long-gate both, (c) ETC-both + TRX-short-only plain,
(d) ETC-both + TRX-short-only + q0.3, (e) full short-only both legs.
Mirrors research/run_s2.py + research/run_t5t6.py leg_series/quantile-gate exactly.
Each variant: H2/B/C sharpe-ann-mdd-tr + B-fee2x(0.0008) sharpe.
PASS bar: a variant with B>1.0 AND C>1.5 AND fee2x-B>0.5.
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
SEGS = {'H2': (6077, 6570), 'B': (6380, 6580), 'C': (6080, 6580)}
VARIANTS = {
    'a_plain_both': {'ETC': ('both', None), 'TRX': ('both', None)},
    'b_q03_both': {'ETC': ('both', 0.3), 'TRX': ('both', 0.3)},
    'c_etcBoth_trxShort_plain': {'ETC': ('both', None), 'TRX': ('short', None)},
    'd_etcBoth_trxShort_q03': {'ETC': ('both', 0.3), 'TRX': ('short', 0.3)},
    'e_shortOnly_both': {'ETC': ('short', None), 'TRX': ('short', None)},
}

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

def qmask(sig, q):
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
    kw['fee_override'] = BASE_FEE if fee is None else fee
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

def combo(ser_list, weights=None):
    m = min(len(s) for s in ser_list)
    k = len(ser_list)
    w = weights or [1.0 / k] * k
    sw = sum(w)
    return [sum(ser_list[i][t] * w[i] / sw for i in range(k)) for t in range(m)]

def eval_variant(mats, fee, spec):
    legs, tr = [], []
    by = {}
    for c in COINS:
        raw, rt, sg = mats[c]
        lth, sth, cd, sl = BEST[c]
        side, q = spec[c]
        net, t = leg_series(raw, rt, sg, lth, sth, cd, sl, fee=fee, side=side, q=q)
        legs.append(net)
        tr.append(t)
        by[c] = t
    cb = combo(legs, [0.5, 0.5])
    s = stats(cb)
    s['trades'] = sum(tr)
    s['trades_by'] = by
    return s

def main():
    full = {c: load_bars(c) for c in COINS}
    n = min(len(b) for b in full.values())
    print('full 4h bars n=' + str(n), flush=True)
    segbars = {}
    for name, (a, b) in SEGS.items():
        segbars[name] = {c: full[c][a:b] for c in COINS}
        print('%s [%d:%d] n=%d' % (name, a, b, len(segbars[name]['ETC'])), flush=True)
    segmats = {name: {c: build_mats(segbars[name][c]) for c in COINS} for name in SEGS}
    rows = {}
    for vname, spec in VARIANTS.items():
        rows[vname] = {}
        for seg in ('H2', 'B', 'C'):
            rows[vname][seg] = eval_variant(segmats[seg], BASE_FEE, spec)
        rows[vname]['B_fee2x'] = eval_variant(segmats['B'], FEE2X, spec)
        h, b, c = rows[vname]['H2'], rows[vname]['B'], rows[vname]['C']
        f2 = rows[vname]['B_fee2x']
        print('%s H2(sh=%.3f ann=%.4f mdd=%.4f tr=%d) B(sh=%.3f ann=%.4f mdd=%.4f tr=%d) C(sh=%.3f ann=%.4f mdd=%.4f tr=%d) Bfee2x(sh=%.3f)' % (
            vname, h['sharpe'], h['ann'], h['mdd'], h['trades'],
            b['sharpe'], b['ann'], b['mdd'], b['trades'],
            c['sharpe'], c['ann'], c['mdd'], c['trades'], f2['sharpe']), flush=True)
    res = {'config': {'formula': FORMULA,
                      'legs_Q1': {k: {'long_th': v[0], 'short_th': v[1], 'cooldown': v[2], 'stop_loss': v[3]} for k, v in BEST.items()},
                      'weights': {'ETC': 0.5, 'TRX': 0.5}, 'venue': 'aster', 'lev': 2.0,
                      'fund': FUND, 'fee': BASE_FEE, 'fee2x': FEE2X, 'full_n': n,
                      'segments': {k: {'slice': list(v), 'n': len(segbars[k]['ETC'])} for k, v in SEGS.items()},
                      'variants': {k: {c: list(vv) for c, vv in spec.items()} for k, spec in VARIANTS.items()},
                      'filter': 'quantile top-q on |logit| applied to long leg only (mirror run_s2/run_t5t6); short untouched'},
            'rows': rows}
    res['PASS'] = {}
    anypass = False
    for vname in VARIANTS:
        b = rows[vname]['B']['sharpe']
        c = rows[vname]['C']['sharpe']
        f2 = rows[vname]['B_fee2x']['sharpe']
        ok = bool(b > 1.0 and c > 1.5 and f2 > 0.5)
        res['PASS'][vname] = {'B': b, 'C': c, 'B_fee2x': f2, 'pass': ok}
        anypass = anypass or ok
    res['PASS']['overall'] = bool(anypass)
    open('results/backtest_X1.json', 'w').write(json.dumps(res, indent=1))
    print('PASS ' + json.dumps(res['PASS']), flush=True)
    print('saved results/backtest_X1.json', flush=True)

if __name__ == '__main__':
    main()
