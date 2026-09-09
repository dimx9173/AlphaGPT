"""X8 vol-threshold extension on S2 q=0.3 variant.
Base: FORMULA [3,2,7,2,7,11,15,4,4,6,6,10], ETC (0.88/0.12/cd12/None)
+ TRX (0.85/0.15/cd6/sl0.05), 50/50 equal-weight, aster perp 2x,
quantile top-0.3 gate on LONG leg only (S2), side='both'.
Segments: H2 = 2nd half of last 15%% OOS (frozen); B = last 200 4h bars;
C = last 500 4h bars.
T7: fund [0.0002,0.0005,0.001] x fee [0.0004,0.0008] = 6 rows on B+C both.
T8: vol_target [None,0.02,0.05] x vol_window [12,24] = 6 rows, per-leg
scaling via MemeBacktest._vol_scale, H2/B/C both, base fund/fee.
Mirrors research/run_s2.py (quantile long-only) + run_iterN7N8.py N8 (vol).
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
BASE_FUND = 0.0005
Q = 0.3

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
    kw['funding_override'] = BASE_FUND if fund is None else fund
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

def eval_both(mats_seg, fee, fund, vt=None, vw=24):
    legs, tr, tos = [], [], []
    for c in COINS:
        raw, rt, sg = mats_seg[c]
        lth, sth, cd, sl = BEST[c]
        net, t, to = leg_series(raw, rt, sg, lth, sth, cd, sl,
                                fee=fee, fund=fund, side='both', vt=vt, vw=vw)
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
    print('H2 n=%d | B tail200 | C last500 (q=0.3 long-gate, 2x)' % len(oos['ETC'][h:]), flush=True)
    rows = []
    for vt in [0.01, 0.015, 0.02, 0.03]:
        for vw in [12, 24]:
            h2 = eval_both(segH2, fee=BASE_FEE, fund=BASE_FUND, vt=vt, vw=vw)
            b = eval_both(segB, fee=BASE_FEE, fund=BASE_FUND, vt=vt, vw=vw)
            c = eval_both(segC, fee=BASE_FEE, fund=BASE_FUND, vt=vt, vw=vw)
            row = {'vol_target': vt, 'vol_window': vw, 'H2': h2, 'B': b, 'C': c}
            rows.append(row)
            print('X8 vt=%.3f vw=%d H2(sh=%.3f ann=%.4f dd=%.4f tr=%d to=%.4f) B(sh=%.3f ann=%.4f dd=%.4f tr=%d to=%.4f) C(sh=%.3f ann=%.4f dd=%.4f tr=%d to=%.4f)' % (
                vt, vw, h2['sharpe'], h2['ann'], h2['mdd'], h2['trades'], h2['turnover'],
                b['sharpe'], b['ann'], b['mdd'], b['trades'], b['turnover'],
                c['sharpe'], c['ann'], c['mdd'], c['trades'], c['turnover']), flush=True)
    best = max(rows, key=lambda r: r['B']['sharpe'] + r['C']['sharpe'])
    best_to = max(best['B']['turnover'], best['C']['turnover'])
    ok = bool(best['B']['sharpe'] > 1.5 and best['C']['sharpe'] > 2.0 and best_to < 0.15)
    res = {'config': {'formula': FORMULA,
                      'best': {k: list(v[:3]) + [v[3]] for k, v in BEST.items()},
                      'weights': {'ETC': 0.5, 'TRX': 0.5}, 'lev': 2.0,
                      'q': Q, 'q_side': 'long', 'side': 'both',
                      'base_fund': BASE_FUND, 'base_fee': BASE_FEE,
                      'full_n': n, 'H2_n': len(oos['ETC'][h:]),
                      'grid': 'vt[0.01,0.015,0.02,0.03]xvw[12,24] per-leg _vol_scale on H2/B/C'},
           'rows': rows,
           'best': {'vol_target': best['vol_target'], 'vol_window': best['vol_window'],
                    'B': best['B'], 'C': best['C'], 'H2': best['H2'],
                    'max_turnover': best_to},
           'PASS': ok}
    open('results/backtest_X8.json', 'w').write(json.dumps(res, indent=1))
    print('BEST vt=%s vw=%s B=%.3f C=%.3f to=%.4f PASS=%s' % (
        str(best['vol_target']), str(best['vol_window']),
        best['B']['sharpe'], best['C']['sharpe'], best_to, str(ok)), flush=True)
    print('saved results/backtest_X8.json', flush=True)

if __name__ == '__main__':
    main()
