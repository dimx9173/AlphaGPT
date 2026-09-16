"""X7 extended reality cost grid on S2 q=0.3 locked base.
Base: FORMULA [3,2,7,2,7,11,15,4,4,6,6,10], ETC (0.88/0.12/cd12/None)
+ TRX (0.85/0.15/cd6/sl0.05), 50/50 equal-weight, aster perp 2x,
quantile top-0.3 gate on LONG leg only (S2), side='both'.
Segments: B = last 200 4h bars; C = last 500 4h bars.
X7: fund [0.0003,0.0005,0.0007] x fee [0.0004,0.0006,0.0008] = 9 rows,
B+C both, sharpe/ann/mdd/trades/turnover + short-leg share.
Mirrors research/run_t7t8.py T7 leg_series/eval exactly, adds short-leg share.
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

def leg_series(raw, rets_t, sig, lth, sth, cd, sl, fee, fund,
               side='both', q=Q, vt=None, vw=24):
    kw = dict(venue='aster', leverage=2.0, short_enabled=True, long_th=lth,
              short_th=sth, cooldown_bars=cd, bars_per_year=2190.0, stop_loss=sl,
              vol_target=vt, vol_window=vw)
    kw['funding_override'] = fund
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
    lpl = lp[0].tolist(); spl = sp[0].tolist()
    pos = [(lpl[t] - spl[t]) for t in range(len(lpl))]
    trades = sum(1 for t in range(len(pos)) if pos[t] != 0.0 and (t == 0 or pos[t - 1] == 0.0))
    turnover = sum(turnl) / len(turnl)
    long_exp = sum(abs(v) for v in lpl)
    short_exp = sum(abs(v) for v in spl)
    tot = long_exp + short_exp
    short_share = short_exp / tot if tot > 0 else 0.0
    return net, trades, turnover, short_share

def stats(ser, trades=0, turnover=0.0, short_share=0.0):
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
            'turnover': round(turnover, 6), 'short_share': round(short_share, 4)}

def combo(ser_list, weights=None):
    m = min(len(s) for s in ser_list)
    k = len(ser_list)
    w = weights or [1.0 / k] * k
    sw = sum(w)
    return [sum(ser_list[i][t] * w[i] / sw for i in range(k)) for t in range(m)]

def eval_both(mats_seg, fee, fund, vt=None, vw=24):
    legs, tr, tos, sss = [], [], [], []
    for c in COINS:
        raw, rt, sg = mats_seg[c]
        lth, sth, cd, sl = BEST[c]
        net, t, to, ss = leg_series(raw, rt, sg, lth, sth, cd, sl,
                                    fee=fee, fund=fund, side='both', vt=vt, vw=vw)
        legs.append(net)
        tr.append(t)
        tos.append(to)
        sss.append(ss)
    cb = combo(legs, [0.5, 0.5])
    s = stats(cb, trades=sum(tr), turnover=sum(tos) / len(tos),
              short_share=sum(sss) / len(sss))
    s['trades_by'] = {c: tr[i] for i, c in enumerate(COINS)}
    s['short_share_by'] = {c: round(sss[i], 4) for i, c in enumerate(COINS)}
    return s

def main():
    full = {c: load_bars(c) for c in COINS}
    n = min(len(b) for b in full.values())
    print('full 4h bars n=' + str(n), flush=True)
    segB = {c: build_mats(full[c][n - 200:n]) for c in COINS}
    segC = {c: build_mats(full[c][n - 500:n]) for c in COINS}
    print('B tail200 | C last500 (S2 q=0.3 long-gate, both, 2x)', flush=True)
    rows = []
    for fund in [0.0003, 0.0005, 0.0007]:
        for fee in [0.0004, 0.0006, 0.0008]:
            b = eval_both(segB, fee=fee, fund=fund)
            c = eval_both(segC, fee=fee, fund=fund)
            row = {'fund': fund, 'fee': fee, 'B': b, 'C': c}
            rows.append(row)
            print('X7 fund=%.4f fee=%.4f B(sh=%.3f ann=%.4f mdd=%.4f tr=%d to=%.6f ss=%.3f) C(sh=%.3f ann=%.4f mdd=%.4f tr=%d to=%.6f ss=%.3f)' % (
                fund, fee, b['sharpe'], b['ann'], b['mdd'], b['trades'], b['turnover'], b['short_share'],
                c['sharpe'], c['ann'], c['mdd'], c['trades'], c['turnover'], c['short_share']), flush=True)
    worstB = min(rows, key=lambda r: r['B']['sharpe'])
    worstC = min(rows, key=lambda r: r['C']['sharpe'])
    passB = worstB['B']['sharpe'] > 0
    passC = worstC['C']['sharpe'] > 1.0
    # fund-driven inflation flag: C sharpe rises with fund at fixed fee?
    infl_flags = []
    for fee in [0.0004, 0.0006, 0.0008]:
        cs = [r['C']['sharpe'] for r in rows if r['fee'] == fee]
        if len(cs) == 3 and cs[2] > cs[0] + 0.3:
            infl_flags.append({'fee': fee, 'C_sharpes_by_fund': cs,
                               'note': 'C sharpe rises with funding (short-funding artifact suspected)'})
    res = {'config': {'formula': FORMULA,
                      'best': {k: list(v[:3]) + [v[3]] for k, v in BEST.items()},
                      'weights': {'ETC': 0.5, 'TRX': 0.5}, 'lev': 2.0,
                      'q': Q, 'q_side': 'long', 'side': 'both',
                      'grid': 'fund[0.0003,0.0005,0.0007]xfee[0.0004,0.0006,0.0008] on B+C',
                      'mirror': 'research/run_t7t8.py T7', 'full_n': n},
           'rows': rows,
           'worstB': {'fund': worstB['fund'], 'fee': worstB['fee'], 'B': worstB['B']},
           'worstC': {'fund': worstC['fund'], 'fee': worstC['fee'], 'C': worstC['C']},
           'fund_inflation_flags': infl_flags,
           'PASS': {'worst_B_gt_0': bool(passB), 'worst_C_gt_1': bool(passC),
                    'overall': bool(passB and passC)}}
    res['PASS']['detail'] = 'worstB fund=%.4f fee=%.4f sh=%.3f | worstC fund=%.4f fee=%.4f sh=%.3f' % (
        worstB['fund'], worstB['fee'], worstB['B']['sharpe'],
        worstC['fund'], worstC['fee'], worstC['C']['sharpe'])
    open('results/backtest_X7.json', 'w').write(json.dumps(res, indent=1))
    print('X7_PASS ' + json.dumps(res['PASS']), flush=True)
    print('INFL ' + json.dumps(infl_flags), flush=True)
    print('saved results/backtest_X7.json', flush=True)

if __name__ == '__main__':
    main()
