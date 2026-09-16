"""T9 final lock-in + T10 paper repro (S2 q=0.3 locked legs).

Locked config (S2 winner, no T1-T8 challenger on file -> contender (a) alone):
  FORMULA [3,2,7,2,7,11,15,4,4,6,6,10]
  ETC (0.88/0.12/cd12/None) + TRX (0.85/0.15/cd6/sl0.05), 50/50 equal-weight,
  aster perp 2x, fund 0.0005, fee base 0.0004, fee2x 0.0008.
  S2 q=0.3 long-leg quantile filter (top-30% |logit| keeps long entry;
  short leg untouched). Mask scope mirrors run_s2.py: per-segment/fold.

T9: 8-fold segmented walk-forward over full 6580 4h bars (fold bounds
  i*6580//8 -> sizes 822/823), both-legs, base fee. Per-fold sharpe/ann/mdd
  + pos-rate (fraction of bars with nonzero combo position) + trades.
  PASS: >=6/8 folds sharpe>0 AND min fold sharpe > -1.0. Plus fee2x on B+C
  (B=last-200 bars, C=last-500 bars).
T10: offline ledger simulation mirroring research/run_paper2.py exactly,
  plus the q=0.3 long-leg mask. Full-history equity curve, trades, final_x,
  sharpe, mdd. Compare vs paper2.log baseline (477 trades, 2.93x, 0.89).
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
FEE2X = 0.0008
FUND = 0.0005
LEV = 2.0
Q = 0.3
NFOLD = 8
BASELINE = {'trades': 477, 'final_x': 2.927, 'sharpe': 0.891, 'mdd': 0.7846}

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

def leg_series(raw, rets_t, sig, lth, sth, cd, sl, fee=None, q=None):
    kw = dict(venue='aster', leverage=LEV, short_enabled=True, long_th=lth,
              short_th=sth, cooldown_bars=cd, bars_per_year=2190.0, stop_loss=sl)
    kw['funding_override'] = FUND
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
    lp = lp.roll(1, dims=1); lp[:, 0] = 0
    sp = sp.roll(1, dims=1); sp[:, 0] = 0
    turn = (lp - lp.roll(1, dims=1)).abs() + (sp - sp.roll(1, dims=1)).abs()
    tx = turn * (bt.base_fee + torch.clamp(bt.trade_size / (raw['liquidity'] + 1e-9), 0.0, 0.05))
    gross = (lp - sp) * rets_t * bt.leverage
    fnd = (lp - sp) * bt.default_funding_rate * bt.leverage
    net = (gross - tx * bt.leverage - fnd)[0].tolist()
    pos = (lp - sp)[0].tolist()
    trades = sum(1 for t in range(len(pos)) if pos[t] != 0.0 and (t == 0 or pos[t - 1] == 0.0))
    return net, trades, pos

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

def eval_both(mats_seg, fee, q):
    legs, tr, pp = [], [], []
    for c in COINS:
        raw, rt, sg = mats_seg[c]
        lth, sth, cd, sl = BEST[c]
        net, t, pos = leg_series(raw, rt, sg, lth, sth, cd, sl, fee=fee, q=q)
        legs.append(net); tr.append(t); pp.append(pos)
    cb = combo(legs, [0.5, 0.5])
    s = stats(cb)
    s['trades'] = sum(tr)
    s['trades_by'] = {c: tr[i] for i, c in enumerate(COINS)}
    cp = [0.5 * pp[0][t] + 0.5 * pp[1][t] for t in range(len(cb))]
    s['pos_rate'] = round(sum(1 for v in cp if v != 0.0) / len(cp), 4)
    return s

def t10_ledger(full):
    n = min(len(b) for b in full.values())
    legs, poss = {}, {}
    for c in COINS:
        raw, rt, sg = build_mats(full[c][:n])
        lth, sth, cd, sl = BEST[c]
        kw = dict(venue='aster', leverage=LEV, short_enabled=True, long_th=lth,
                  short_th=sth, cooldown_bars=cd, bars_per_year=2190.0, stop_loss=sl)
        kw['funding_override'] = FUND
        bt = MemeBacktest(**kw)
        sgv = torch.sigmoid(sg)
        safe = (raw['liquidity'] > bt.min_liq).float()
        lp = (sgv > bt.long_th).float() * safe
        sp = (sgv < bt.short_th).float() * safe
        mask = quantile_mask_long(sg, Q)
        if mask is not None:
            lp = lp * mask
        lp, sp = bt._apply_cooldown(lp, sp)
        lp, sp = bt._apply_stops(lp, sp, rt)
        lp = lp.roll(1, dims=1); lp[:, 0] = 0
        sp = sp.roll(1, dims=1); sp[:, 0] = 0
        poss[c] = (lp - sp)[0].tolist()
    PORT = {'ETC': 0.5, 'TRX': 0.5}
    coins = sorted(PORT)
    eq = [1.0]
    ledger = []
    cur = {c: 0.0 for c in coins}
    ent = {c: 0.0 for c in coins}
    px = {c: [b[3] for b in full[c][:n]] for c in coins}
    for t in range(n):
        for c in coins:
            want = poss[c][t]
            want = 1.0 if want > 0.5 else (-1.0 if want < -0.5 else 0.0)
            if t > 0 and want != cur[c]:
                fill = px[c][t - 1]
                if cur[c] != 0.0:
                    move = (fill - ent[c]) / ent[c] * (1.0 if cur[c] > 0 else -1.0)
                    netp = move * LEV - BASE_FEE * LEV * 2.0 - FUND * LEV
                    eq.append(eq[-1] + eq[-1] * PORT[c] * netp)
                    ledger.append({'t': t, 'coin': c, 'side': int(cur[c]), 'fill': fill,
                                   'move': round(move, 6), 'eq': round(eq[-1], 4)})
                if want != 0.0:
                    ent[c] = fill
                cur[c] = want
        if len(eq) < t + 2:
            eq.append(eq[-1])
    eq = eq[:n]
    rets = [(eq[i + 1] - eq[i]) / eq[i] if eq[i] else 0.0 for i in range(len(eq) - 1)]
    mean = sum(rets) / max(len(rets), 1)
    var = sum((x - mean) ** 2 for x in rets) / max(len(rets) - 1, 1)
    sharpe = mean / math.sqrt(var) * math.sqrt(2190.0) if var > 0 else 0.0
    peak = eq[0]; mdd = 0.0
    for v in eq:
        peak = max(peak, v); mdd = max(mdd, (peak - v) / peak if peak else 0.0)
    by = {}
    for e in ledger:
        by[e['coin']] = by.get(e['coin'], 0) + 1
    return {'final_x': round(eq[-1], 4), 'sharpe': round(sharpe, 3),
            'mdd': round(mdd, 4), 'trades': len(ledger), 'by': by,
            'n': n, 'ledger': ledger, 'equity': [round(v, 4) for v in eq]}

def main():
    full = {c: load_bars(c) for c in COINS}
    n = min(len(b) for b in full.values())
    print('full 4h bars n=' + str(n), flush=True)

    # ---- T9: 8-fold walk-forward, both-legs, q=0.3, base fee ----
    bounds = [i * n // NFOLD for i in range(NFOLD + 1)]
    folds = []
    for i in range(NFOLD):
        a, b = bounds[i], bounds[i + 1]
        mats = {c: build_mats(full[c][a:b]) for c in COINS}
        s = eval_both(mats, fee=BASE_FEE, q=Q)
        s['fold'] = i
        s['bars'] = [a, b]
        folds.append(s)
        print('fold%d bars[%d:%d] n=%d sh=%.3f ann=%.4f mdd=%.4f pos=%.4f tr=%d' % (
            i, a, b, s['n'], s['sharpe'], s['ann'], s['mdd'], s['pos_rate'], s['trades']), flush=True)
    pos_folds = sum(1 for f in folds if f['sharpe'] > 0)
    min_sh = min(f['sharpe'] for f in folds)
    t9_pass = {'n_pos_ge_6': pos_folds >= 6, 'min_gt_neg1': min_sh > -1.0}
    t9_pass['overall'] = all(t9_pass.values())
    print('T9 folds positive %d/8, min sharpe %.3f, PASS=%s' % (pos_folds, min_sh, t9_pass['overall']), flush=True)

    # ---- fee2x on B (last-200) + C (last-500), both-legs, q=0.3 ----
    segB = {c: build_mats(full[c][n - 200:n]) for c in COINS}
    segC = {c: build_mats(full[c][n - 500:n]) for c in COINS}
    B = eval_both(segB, fee=BASE_FEE, q=Q)
    C = eval_both(segC, fee=BASE_FEE, q=Q)
    Bf = eval_both(segB, fee=FEE2X, q=Q)
    Cf = eval_both(segC, fee=FEE2X, q=Q)
    print('B base sh=%.3f ann=%.4f mdd=%.4f tr=%d | fee2x sh=%.3f' % (
        B['sharpe'], B['ann'], B['mdd'], B['trades'], Bf['sharpe']), flush=True)
    print('C base sh=%.3f ann=%.4f mdd=%.4f tr=%d | fee2x sh=%.3f' % (
        C['sharpe'], C['ann'], C['mdd'], C['trades'], Cf['sharpe']), flush=True)

    # ---- T10: paper ledger repro ----
    t10 = t10_ledger(full)
    print('T10 paper q=0.3: trades=%d final_x=%.4f sharpe=%.3f mdd=%.4f n=%d by=%s' % (
        t10['trades'], t10['final_x'], t10['sharpe'], t10['mdd'], t10['n'], t10['by']), flush=True)
    print('baseline paper2.log: trades=%d final_x=%.3f sharpe=%.3f mdd=%.4f' % (
        BASELINE['trades'], BASELINE['final_x'], BASELINE['sharpe'], BASELINE['mdd']), flush=True)

    res = {
        'config': {'formula': FORMULA, 'legs': {c: list(BEST[c]) for c in COINS},
                   'weights': [0.5, 0.5], 'venue': 'aster perp 2x', 'fund': FUND,
                   'fee_base': BASE_FEE, 'fee2x': FEE2X, 'q_longfilter': Q,
                   'challenger': None,
                   'challenger_note': 'no T1-T8 results on file; contender (a) S2 q=0.3 alone'},
        'T9_folds': folds,
        'T9_PASS': dict(t9_pass, n_pos=pos_folds, min_sharpe=min_sh),
        'B_base': B, 'C_base': C, 'B_fee2x': Bf, 'C_fee2x': Cf,
        'T10': {k: t10[k] for k in ('final_x', 'sharpe', 'mdd', 'trades', 'by', 'n')},
        'T10_baseline': BASELINE,
        'T10_drift': {'d_trades': t10['trades'] - BASELINE['trades'],
                      'd_final_x': round(t10['final_x'] - BASELINE['final_x'], 4),
                      'd_sharpe': round(t10['sharpe'] - BASELINE['sharpe'], 3)},
    }
    open('results/backtest_T9T10.json', 'w').write(json.dumps(res, indent=1))
    open('results/paper_T10.json', 'w').write(json.dumps(
        {'ledger': t10['ledger'], 'equity': t10['equity'],
         'stats': {k: t10[k] for k in ('final_x', 'sharpe', 'mdd', 'trades', 'by', 'n')},
         'config': res['config']}, indent=1))
    print('saved results/backtest_T9T10.json + results/paper_T10.json', flush=True)

if __name__ == '__main__':
    main()
