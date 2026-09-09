"""T1+T2 tail-regime follow-up.
T1: S2 q=0.3 paper-variant stress — fee sweep [0.0004,0.0008,0.0012] on B+C
    both-legs + time-OOS split C into C1[6080:6330]+C2[6330:6580].
    PASS: fee2x-B > 0.5 AND C1,C2 both > 0.5 (no half-dead).
T2: S4 rank2 (lth 0.85/sth 0.12/ETC cd12/TRX cd6/sl None) deep-dive —
    per-coin B breakdown (ETC vs TRX both/long/short on B) + SL variants
    [None,0.03,0.05,0.08] on rank2 params for B+C.
Locked base: FORMULA [3,2,7,2,7,11,15,4,4,6,6,10], aster perp 2x fund 0.0005.
Segments (n=6580): H1 [5584:6077], H2 [6077:6570], B [6380:6580],
C [6080:6580], C1 [6080:6330], C2 [6330:6580].
Method mirrors research/run_s2.py (quantile long-gate) + run_p2.py (trades).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import csv, json, math
import torch
from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest

FORMULA = [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10]
COINS = ['ETC', 'TRX']
S2_BEST = {'ETC': (0.88, 0.12, 12, None), 'TRX': (0.85, 0.15, 6, 0.05)}
T1_Q = 0.3
RANK2 = {'lth': 0.85, 'sth': 0.12, 'cd': {'ETC': 12, 'TRX': 6}, 'sl': None}
FUND = 0.0005
FEES = [0.0004, 0.0008, 0.0012]
SLS = [None, 0.03, 0.05, 0.08]
H1 = (5584, 6077); H2 = (6077, 6570)
B = (6380, 6580); C = (6080, 6580); C1 = (6080, 6330); C2 = (6330, 6580)

def load_bars(coin):
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

def leg_series(raw, rets_t, sig, lth, sth, cd, sl, fee, side='both', q=None):
    kw = dict(venue='aster', leverage=2.0, short_enabled=True, long_th=lth,
              short_th=sth, cooldown_bars=cd, bars_per_year=2190.0, stop_loss=sl,
              funding_override=FUND, fee_override=fee)
    bt = MemeBacktest(**kw)
    signal = torch.sigmoid(sig)
    is_safe = (raw['liquidity'] > bt.min_liq).float()
    lp = (signal > bt.long_th).float() * is_safe
    sp = (signal < bt.short_th).float() * is_safe
    if q is not None:  # S2-style: quantile gate on LONG leg only
        a = sig.detach().float().abs().reshape(-1)
        k = max(1, int(len(a) * float(q)))
        thr = torch.topk(a, k).values.min()
        lp = lp * (sig.detach().float().abs() >= thr).float()
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

def combo(legs):
    m = min(len(s) for s in legs)
    return [(legs[0][t] + legs[1][t]) / 2.0 for t in range(m)]

def t1_eval(mats_seg, fee):
    legs, tr = [], []
    for c in COINS:
        raw, rt, sg = mats_seg[c]
        lth, sth, cd, sl = S2_BEST[c]
        net, t = leg_series(raw, rt, sg, lth, sth, cd, sl, fee, side='both', q=T1_Q)
        legs.append(net); tr.append(t)
    s = stats(combo(legs))
    s['trades'] = sum(tr)
    s['trades_by'] = {c: tr[i] for i, c in enumerate(COINS)}
    return s

def t2_combo(mats_seg, sl, fee=0.0004):
    legs, tr = [], []
    by = {}
    for c in COINS:
        raw, rt, sg = mats_seg[c]
        net, t = leg_series(raw, rt, sg, RANK2['lth'], RANK2['sth'],
                            RANK2['cd'][c], sl, fee, side='both', q=None)
        legs.append(net); tr.append(t)
        by[c] = dict(stats(net), trades=t)
    s = stats(combo(legs))
    s['trades'] = sum(tr)
    s['trades_by'] = {c: tr[i] for i, c in enumerate(COINS)}
    return s, by

def t2_coin_sides(mats_seg, fee=0.0004):
    out = {}
    for c in COINS:
        raw, rt, sg = mats_seg[c]
        out[c] = {}
        for side in ('both', 'long', 'short'):
            net, t = leg_series(raw, rt, sg, RANK2['lth'], RANK2['sth'],
                                RANK2['cd'][c], RANK2['sl'], fee, side=side, q=None)
            out[c][side] = dict(stats(net), trades=t)
    return out

def main():
    full = {c: load_bars(c) for c in COINS}
    n = min(len(b) for b in full.values())
    assert n == 6580, n
    segs = {'H1': (H1, 'frozen H1'), 'H2': (H2, 'frozen H2'), 'B': (B, 'fresh tail200'),
            'C': (C, 'last500'), 'C1': (C1, 'C-first250'), 'C2': (C2, 'C-last250')}
    mats = {k: {c: build_mats(full[c][lo:hi]) for c in COINS} for k, ((lo, hi), _) in
            [ (k, segs[k]) for k in segs ]}
    print('n_full=%d' % n, {k: len(full['ETC'][v[0][0]:v[0][1]]) for k, v in segs.items()}, flush=True)

    res = {'config': {'formula': FORMULA, 'venue': 'aster', 'lev': 2.0, 'fund': FUND,
                      'weights': {'ETC': 0.5, 'TRX': 0.5}, 'full_n': n,
                      'segments': {k: list(v[0]) for k, v in segs.items()},
                      'T1': {'base': {k: list(v[:3]) + [v[3]] for k, v in S2_BEST.items()},
                             'q_long_only': T1_Q, 'fees': FEES},
                      'T2_rank2': {'lth': RANK2['lth'], 'sth': RANK2['sth'],
                                   'cd_etc': RANK2['cd']['ETC'], 'cd_trx': RANK2['cd']['TRX'],
                                   'sl': RANK2['sl'], 'sl_variants': ['None', 0.03, 0.05, 0.08]}}}
    # ---- T1: fee sweep on B+C both-legs ----
    res['T1_fee_sweep'] = {}
    for seg in ('B', 'C'):
        for fee in FEES:
            s = t1_eval(mats[seg], fee)
            res['T1_fee_sweep'][seg + '_fee' + str(fee)] = s
            print('T1 %s fee=%s both=%s' % (seg, fee, json.dumps(s)), flush=True)
    # ---- T1: time-OOS C1/C2 at base fee ----
    res['T1_time_oos'] = {}
    for seg in ('C1', 'C2'):
        s = t1_eval(mats[seg], 0.0004)
        res['T1_time_oos'][seg] = s
        print('T1 time-OOS %s base-fee both=%s' % (seg, json.dumps(s)), flush=True)
    b2x = res['T1_fee_sweep']['B_fee0.0008']['sharpe']
    c1 = res['T1_time_oos']['C1']['sharpe']
    c2 = res['T1_time_oos']['C2']['sharpe']
    res['T1_PASS'] = {'fee2x_B_gt_0.5': b2x > 0.5, 'C1_gt_0.5': c1 > 0.5, 'C2_gt_0.5': c2 > 0.5}
    res['T1_PASS']['overall'] = all(res['T1_PASS'].values())
    print('T1 PASS ' + json.dumps(res['T1_PASS']), flush=True)

    # ---- T2: per-coin B breakdown ----
    res['T2_B_bycoin_sides'] = t2_coin_sides(mats['B'])
    for c in COINS:
        print('T2 B %s %s' % (c, json.dumps(res['T2_B_bycoin_sides'][c])), flush=True)
    # ---- T2: SL variants on B+C ----
    res['T2_sl_variants'] = {}
    for sl in SLS:
        for seg in ('B', 'C'):
            s, by = t2_combo(mats[seg], sl)
            res['T2_sl_variants']['sl' + str(sl) + '_' + seg] = {'combo': s, 'bycoin': by}
            print('T2 sl=%s %s combo=%s ETC=%s TRX=%s' % (
                sl, seg, json.dumps(s),
                json.dumps(by['ETC']), json.dumps(by['TRX'])), flush=True)
    # which leg kills B: compare long vs short per coin on B
    bd = res['T2_B_bycoin_sides']
    res['T2_B_diagnosis'] = {
        c: {'long_sharpe': bd[c]['long']['sharpe'], 'short_sharpe': bd[c]['short']['sharpe'],
            'killer': 'short' if bd[c]['short']['sharpe'] < bd[c]['long']['sharpe'] else 'long'}
        for c in COINS}
    print('T2 B diagnosis ' + json.dumps(res['T2_B_diagnosis']), flush=True)
    open('results/backtest_T1T2.json', 'w').write(json.dumps(res, indent=1))
    print('saved results/backtest_T1T2.json', flush=True)

if __name__ == '__main__':
    main()
