"""Y2 vol-stack on X5 base (S2 q=0.3 + cd18/6 sth0.12, 50/50).
Base legs: ETC (0.88/0.12/cd18/None) + TRX (0.85/0.12/cd6/sl0.05),
quantile top-0.3 gate on LONG leg only, side='both', aster perp 2x,
fund 0.0005, fee base 0.0004 / 2x 0.0008.
Y1-best unknown -> X5 base only.
Grid: vol_target [None,0.01,0.015] x vol_window [12,24] = 6 rows,
per-leg scaling via MemeBacktest._vol_scale, quantile-gate exactly as X5/X8.
Each row: H2/B/C (sharpe-ann-mdd-tr-turnover) + B/C-fee2x.
Segments: H2 = 2nd half of last 15% OOS (frozen); B = last 200 4h bars; C = last 500.
PASS: best B>2.5 AND C>3.0 at turnover<0.15 AND fee2x-B>1.0.
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
# X5 base: S2 lth/sl locked, both-coin sth=0.12, per-coin cd 18/6
X5 = {'ETC': (0.88, 0.12, 18, None), 'TRX': (0.85, 0.12, 6, 0.05)}
COINS = ['ETC', 'TRX']
BASE_FEE = 0.0004
FEE2X = 0.0008
FUND = 0.0005
Q = 0.3
VT_GRID = [None, 0.01, 0.015]
VW_GRID = [12, 24]

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
               side='both', q=Q, vt=None, vw=24):
    kw = dict(venue='aster', leverage=2.0, short_enabled=True, long_th=lth,
              short_th=sth, cooldown_bars=cd, bars_per_year=2190.0, stop_loss=sl,
              vol_target=vt, vol_window=vw)
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
    scale = bt._vol_scale(rets_t)
    lp, sp = lp * scale, sp * scale
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
    turnl = turn[0].tolist()
    pos = (lp - sp)[0].tolist()
    trades = sum(1 for t in range(len(pos)) if pos[t] != 0.0 and (t == 0 or pos[t - 1] == 0.0))
    turnover = sum(turnl) / len(turnl)
    return net, trades, turnover

def stats(ser, trades=0, turnover=0.0):
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
            'cum': round(cum, 4), 'n': n, 'trades': trades,
            'turnover': round(turnover, 6)}

def combo(ser_list, weights=None):
    m = min(len(s) for s in ser_list)
    k = len(ser_list)
    w = weights or [1.0 / k] * k
    sw = sum(w)
    return [sum(ser_list[i][t] * w[i] / sw for i in range(k)) for t in range(m)]

def eval_both(mats_seg, fee, vt=None, vw=24):
    legs, tr, tos = [], [], []
    for c in COINS:
        raw, rt, sg = mats_seg[c]
        lth, sth, cd, sl = X5[c]
        net, t, to = leg_series(raw, rt, sg, lth, sth, cd, sl,
                                fee=fee, side='both', q=Q, vt=vt, vw=vw)
        legs.append(net)
        tr.append(t)
        tos.append(to)
    cb = combo(legs, [0.5, 0.5])
    s = stats(cb, trades=sum(tr), turnover=sum(tos) / len(tos))
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
    print('Y2 X5-base q=0.3 cd18/6 sth0.12 50/50 2x | H2 n=%d | B 200 | C 500' % len(oos['ETC'][h:]), flush=True)
    rows = []
    for vt in VT_GRID:
        for vw in VW_GRID:
            h2 = eval_both(segH2, fee=BASE_FEE, vt=vt, vw=vw)
            b = eval_both(segB, fee=BASE_FEE, vt=vt, vw=vw)
            c = eval_both(segC, fee=BASE_FEE, vt=vt, vw=vw)
            b2 = eval_both(segB, fee=FEE2X, vt=vt, vw=vw)
            c2 = eval_both(segC, fee=FEE2X, vt=vt, vw=vw)
            row = {'vol_target': vt, 'vol_window': vw,
                   'H2': h2, 'B': b, 'C': c,
                   'B_fee2x': b2, 'C_fee2x': c2}
            rows.append(row)
            print('Y2 vt=%s vw=%d H2(sh=%.3f ann=%.4f dd=%.4f tr=%d to=%.4f) B(sh=%.3f ann=%.4f dd=%.4f tr=%d to=%.4f B2x=%.3f) C(sh=%.3f ann=%.4f dd=%.4f tr=%d to=%.4f C2x=%.3f)' % (
                str(vt), vw,
                h2['sharpe'], h2['ann'], h2['mdd'], h2['trades'], h2['turnover'],
                b['sharpe'], b['ann'], b['mdd'], b['trades'], b['turnover'], b2['sharpe'],
                c['sharpe'], c['ann'], c['mdd'], c['trades'], c['turnover'], c2['sharpe']), flush=True)
    # vol-fee interaction: fee2x decay per row
    for r in rows:
        r['fee_decay_B'] = round(r['B']['sharpe'] - r['B_fee2x']['sharpe'], 3)
        r['fee_decay_C'] = round(r['C']['sharpe'] - r['C_fee2x']['sharpe'], 3)
    best = max(rows, key=lambda r: r['B']['sharpe'] + r['C']['sharpe'])
    best_to = max(best['B']['turnover'], best['C']['turnover'])
    ok = bool(best['B']['sharpe'] > 2.5 and best['C']['sharpe'] > 3.0
              and best_to < 0.15 and best['B_fee2x']['sharpe'] > 1.0)
    res = {'config': {'base': 'X5 (S2 q0.3 long-gate, ETC 0.88/0.12/cd18/None + TRX 0.85/0.12/cd6/sl0.05)',
                      'formula': FORMULA,
                      'x5': {k: list(v[:3]) + [v[3]] for k, v in X5.items()},
                      'weights': {'ETC': 0.5, 'TRX': 0.5}, 'lev': 2.0,
                      'q': Q, 'q_side': 'long', 'side': 'both',
                      'fund': FUND, 'fee': BASE_FEE, 'fee2x': FEE2X,
                      'full_n': n, 'H2_n': len(oos['ETC'][h:]),
                      'grid': 'vt[None,0.01,0.015]xvw[12,24] per-leg _vol_scale on X5 base; Y1-best unknown so X5 only',
                      'note': 'mirrors run_x5 leg_series (quantile long-only + cooldown + stops) + run_x8 _vol_scale placement (post-stops, pre-roll)'},
           'rows': rows,
           'best': {'vol_target': best['vol_target'], 'vol_window': best['vol_window'],
                    'H2': best['H2'], 'B': best['B'], 'C': best['C'],
                    'B_fee2x': best['B_fee2x'], 'C_fee2x': best['C_fee2x'],
                    'max_turnover': best_to,
                    'fee_decay_B': best['fee_decay_B'], 'fee_decay_C': best['fee_decay_C']},
           'PASS': {'best_vt': best['vol_target'], 'best_vw': best['vol_window'],
                    'B_gt_2.5': bool(best['B']['sharpe'] > 2.5),
                    'C_gt_3.0': bool(best['C']['sharpe'] > 3.0),
                    'turnover_lt_0.15': bool(best_to < 0.15),
                    'fee2x_B_gt_1.0': bool(best['B_fee2x']['sharpe'] > 1.0),
                    'overall': ok}}
    open('results/backtest_Y2.json', 'w').write(json.dumps(res, indent=1))
    print('BEST vt=%s vw=%s B=%.3f C=%.3f to=%.4f B2x=%.3f PASS=%s' % (
        str(best['vol_target']), str(best['vol_window']),
        best['B']['sharpe'], best['C']['sharpe'], best_to,
        best['B_fee2x']['sharpe'], str(ok)), flush=True)
    # vol-fee interaction flag
    base = [r for r in rows if r['vol_target'] is None][0]
    print('VOL-FEE: base(vt=None) to B/C=%.4f/%.4f decayB=%.3f | best(vt=%s) to B/C=%.4f/%.4f decayB=%.3f' % (
        base['B']['turnover'], base['C']['turnover'], base['fee_decay_B'],
        str(best['vol_target']), best['B']['turnover'], best['C']['turnover'], best['fee_decay_B']), flush=True)
    print('saved results/backtest_Y2.json', flush=True)

if __name__ == '__main__':
    main()
