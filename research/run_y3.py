"""Y3 gate worker: S2-base q0.3 full stack (Y1-unknown fallback X5+X6sl/off+X4ts 50/50)
vs plain ledger + shadow-mode gated proposal.

Fallback overlay stack (X5+X6sl/off+X4ts, q0.3, S2-base, 50/50):
  ETC (0.88/0.12/cd18/sl0.02) both-legs + TRX (0.85/0.12/cd6/sl0.02) short-only,
  q=0.3 long-leg quantile mask, time_stop=24.
Plain ledger: S2-base ETC (0.88/0.12/cd12/None) + TRX (0.85/0.15/cd6/sl0.05),
  both-legs, no q-gate, ts=0. Mirrors research/run_x10.py ledger accounting exactly:
  per-trade netp = move*LEV - FEE*LEV*2 - FUND*LEV, 50/50 portfolio split.
Gate (causal, trailing-200 bars, no lookahead): overlay active at bar t iff
  trailing-200 overlay-combo per-bar-net sharpe > 1.0 (gate computed on t-200:t).
  The task-example rule (plain trailing-200 sharpe < 1.0) is also backtested and
  rejected: LED-switch churn collapses B/C below plain.
Shadow = LED-switch ledger: per-bar position source = overlay when gate on,
  else plain. Segments H2=[6077:6570], B=[6380:6580], C=[6080:6580].
Verdict bar: gated shadow beats plain on BOTH B and C without full-history collapse.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import csv, json, math
import torch
from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest

FORMULA = [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10]
PLAIN = {'ETC': (0.88, 0.12, 12, None), 'TRX': (0.85, 0.15, 6, 0.05)}
OVER = {'ETC': (0.88, 0.12, 18, 0.02), 'TRX': (0.85, 0.12, 6, 0.02)}
OVER_SIDE = {'ETC': 'both', 'TRX': 'short'}
PORT = {'ETC': 0.5, 'TRX': 0.5}
COINS = ['ETC', 'TRX']
FEE = 0.0004
FUND = 0.0005
LEV = 2.0
Q = 0.3
TS_OVER = 24
SEGS = {'H2': (6077, 6570), 'B': (6380, 6580), 'C': (6080, 6580)}
GATE_WIN = 200
GATE_THRESH = 1.0

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

def leg_net_pos(bars, lth, sth, cd, sl, q=None, side='both', ts=0):
    n = len(bars)
    raw = {'open': torch.tensor([[x[0] for x in bars]]),
           'high': torch.tensor([[x[1] for x in bars]]),
           'low': torch.tensor([[x[2] for x in bars]]),
           'close': torch.tensor([[x[3] for x in bars]]),
           'volume': torch.tensor([[x[4] for x in bars]]),
           'liquidity': torch.full((1, n), 1e7),
           'fdv': torch.full((1, n), 1e8)}
    sig = StackVM().execute(FORMULA, FeatureEngineer.compute_features(raw))
    rets = [(bars[i + 1][3] - bars[i][3]) / bars[i][3] if i < n - 1 else 0.0 for i in range(n)]
    bt = MemeBacktest(venue='aster', leverage=LEV, short_enabled=True, long_th=lth,
                      short_th=sth, cooldown_bars=cd, bars_per_year=2190.0,
                      stop_loss=sl, time_stop=ts, funding_override=FUND)
    sg = torch.sigmoid(sig)
    safe = (raw['liquidity'] > bt.min_liq).float()
    lp = (sg > bt.long_th).float() * safe
    sp = (sg < bt.short_th).float() * safe
    if q is not None:
        a = sig.detach().float().abs().reshape(-1)
        k = max(1, int(len(a) * float(q)))
        thr = torch.topk(a, k).values.min()
        lp = lp * (sig.detach().float().abs() >= thr).float()
    lp, sp = bt._apply_cooldown(lp, sp)
    lp, sp = bt._apply_stops(lp, sp, torch.tensor([rets]))
    if side == 'long':
        sp = sp * 0.0
    if side == 'short':
        lp = lp * 0.0
    lp = lp.roll(1, dims=1); lp[:, 0] = 0
    sp = sp.roll(1, dims=1); sp[:, 0] = 0
    tgt = torch.tensor([rets])
    turn = (lp - lp.roll(1, dims=1)).abs() + (sp - sp.roll(1, dims=1)).abs()
    tx = turn * (bt.base_fee + torch.clamp(bt.trade_size / (raw['liquidity'] + 1e-9), 0.0, 0.05))
    gross = (lp - sp) * tgt * bt.leverage
    fnd = (lp - sp) * bt.default_funding_rate * bt.leverage
    net = (gross - tx * bt.leverage - fnd)[0].tolist()
    pos = (lp - sp)[0].tolist()
    return net, pos

def run_ledger(poss, px, n):
    coins = sorted(PORT)
    eq = [1.0]
    ledger = []
    cur = {c: 0.0 for c in coins}
    ent = {c: 0.0 for c in coins}
    for t in range(n):
        for c in coins:
            want = poss[c][t]
            want = 1.0 if want > 0.5 else (-1.0 if want < -0.5 else 0.0)
            if t > 0 and want != cur[c]:
                fill = px[c][t - 1]
                if cur[c] != 0.0:
                    move = (fill - ent[c]) / ent[c] * (1.0 if cur[c] > 0 else -1.0)
                    netp = move * LEV - FEE * LEV * 2.0 - FUND * LEV
                    eq.append(eq[-1] + eq[-1] * PORT[c] * netp)
                    ledger.append({'t': t, 'coin': c, 'side': int(cur[c]),
                                   'fill': fill, 'move': round(move, 6), 'eq': round(eq[-1], 4)})
                if want != 0.0:
                    ent[c] = fill
                cur[c] = want
        if len(eq) < t + 2:
            eq.append(eq[-1])
    return eq[:n], ledger

def eq_stats(eq, ledger):
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
    longs = sum(1 for e in ledger if e['side'] == 1)
    return {'final_x': round(eq[-1], 4), 'sharpe': round(sharpe, 3),
            'mdd': round(mdd, 4), 'trades': len(ledger), 'by': by, 'longs': longs}

def seg_stats(eq, ledger, a, b):
    seg = eq[a:b]
    rets = [(eq[i + 1] - eq[i]) / eq[i] if eq[i] else 0.0 for i in range(a, b - 1)]
    mean = sum(rets) / max(len(rets), 1)
    var = sum((x - mean) ** 2 for x in rets) / max(len(rets) - 1, 1)
    sharpe = mean / math.sqrt(var) * math.sqrt(2190.0) if var > 0 else 0.0
    peak = seg[0]; mdd = 0.0
    for v in seg:
        peak = max(peak, v); mdd = max(mdd, (peak - v) / peak if peak else 0.0)
    tr = sum(1 for e in ledger if a <= e['t'] < b)
    longs = sum(1 for e in ledger if a <= e['t'] < b and e['side'] == 1)
    return {'x': round(seg[-1] / seg[0], 4), 'sharpe': round(sharpe, 3),
            'mdd': round(mdd, 4), 'trades': tr, 'longs': longs}

def trailing_sharpe(ser, t, win):
    w = ser[t - win:t]
    mean = sum(w) / win
    var = sum((x - mean) ** 2 for x in w) / (win - 1)
    return mean / math.sqrt(var) * math.sqrt(2190.0) if var > 0 else 0.0

def main():
    bars = {c: load_bars(c) for c in COINS}
    n = min(len(b) for b in bars.values())
    print('full 4h bars n=' + str(n), flush=True)
    px = {c: [b[3] for b in bars[c][:n]] for c in COINS}
    pnet, ppos, onet, opos = {}, {}, {}, {}
    for c in COINS:
        lth, sth, cd, sl = PLAIN[c]
        net, pos = leg_net_pos(bars[c][:n], lth, sth, cd, sl, q=None, side='both', ts=0)
        pnet[c], ppos[c] = net, pos
        lth2, sth2, cd2, sl2 = OVER[c]
        net2, pos2 = leg_net_pos(bars[c][:n], lth2, sth2, cd2, sl2, q=Q,
                                 side=OVER_SIDE[c], ts=TS_OVER)
        onet[c], opos[c] = net2, pos2
    eq_plain, led_plain = run_ledger(ppos, px, n)
    eq_over, led_over = run_ledger(opos, px, n)
    s_plain, s_over = eq_stats(eq_plain, led_plain), eq_stats(eq_over, led_over)
    print('plain FULL final_x=%s sharpe=%s mdd=%s trades=%s by=%s longs=%s' % (
        s_plain['final_x'], s_plain['sharpe'], s_plain['mdd'],
        s_plain['trades'], s_plain['by'], s_plain['longs']), flush=True)
    print('over  FULL final_x=%s sharpe=%s mdd=%s trades=%s by=%s longs=%s' % (
        s_over['final_x'], s_over['sharpe'], s_over['mdd'],
        s_over['trades'], s_over['by'], s_over['longs']), flush=True)
    drift = {'d_trades': s_over['trades'] - s_plain['trades'],
             'd_final_x': round(s_over['final_x'] - s_plain['final_x'], 4),
             'd_sharpe': round(s_over['sharpe'] - s_plain['sharpe'], 3),
             'longs_filtered': s_plain['longs'] - s_over['longs']}
    print('drift(over-plain)=' + str(drift), flush=True)
    combo_plain = [0.5 * pnet['ETC'][t] + 0.5 * pnet['TRX'][t] for t in range(n)]
    combo_over = [0.5 * onet['ETC'][t] + 0.5 * onet['TRX'][t] for t in range(n)]
    gate_main = [False] * n
    for t in range(GATE_WIN, n):
        gate_main[t] = trailing_sharpe(combo_over, t, GATE_WIN) > GATE_THRESH
    gate_alt = [False] * n
    for t in range(GATE_WIN, n):
        gate_alt[t] = trailing_sharpe(combo_plain, t, GATE_WIN) < 1.0
    segs = {'FULL': (0, n), 'H2': SEGS['H2'], 'B': SEGS['B'], 'C': SEGS['C']}
    out_ledgers = {}
    for name, poss in (('plain', ppos), ('overlay', opos)):
        eq, led = (eq_plain, led_plain) if name == 'plain' else (eq_over, led_over)
        st = eq_stats(eq, led)
        st['segments'] = {k: seg_stats(eq, led, a, b) for k, (a, b) in segs.items()}
        out_ledgers[name] = st
    gate_out = {}
    for gname, g in (('main_over_gt1', gate_main), ('alt_plain_lt1', gate_alt)):
        spos = {c: [opos[c][t] if g[t] else ppos[c][t] for t in range(n)] for c in COINS}
        eq, led = run_ledger(spos, px, n)
        st = eq_stats(eq, led)
        st['segments'] = {k: seg_stats(eq, led, a, b) for k, (a, b) in segs.items()}
        st['coverage'] = round(sum(g) / n, 4)
        gate_out[gname] = st
        print('shadow[%s] cov=%s FULL x=%s sh=%s mdd=%s tr=%s | B=%s C=%s H2=%s' % (
            gname, st['coverage'], st['final_x'], st['sharpe'], st['mdd'],
            st['trades'], st['segments']['B'], st['segments']['C'],
            st['segments']['H2']), flush=True)
    main, alt, pl = gate_out['main_over_gt1'], gate_out['alt_plain_lt1'], out_ledgers['plain']
    verdict = {
        'main_beats_plain_B': bool(main['segments']['B']['sharpe'] > pl['segments']['B']['sharpe']),
        'main_beats_plain_C': bool(main['segments']['C']['sharpe'] > pl['segments']['C']['sharpe']),
        'main_full_no_collapse': bool(main['final_x'] > pl['final_x'] and main['sharpe'] > pl['sharpe']),
        'alt_beats_plain_B': bool(alt['segments']['B']['sharpe'] > pl['segments']['B']['sharpe']),
        'alt_beats_plain_C': bool(alt['segments']['C']['sharpe'] > pl['segments']['C']['sharpe']),
    }
    verdict['pass'] = bool(verdict['main_beats_plain_B'] and verdict['main_beats_plain_C']
                           and verdict['main_full_no_collapse'])
    verdict['decision'] = ('ADOPT shadow mode: paper ledger stays plain live, '
                           'overlay gated by trailing-200 overlay sharpe>1.0' if verdict['pass']
                           else 'REJECT: gated shadow fails bar, keep plain paper only')
    res = {
        'config': {'formula': FORMULA, 'fee': FEE, 'fund': FUND, 'lev': LEV, 'q': Q,
                   'plain_legs': {c: list(PLAIN[c]) for c in COINS},
                   'overlay_legs': {c: list(OVER[c]) for c in COINS},
                   'overlay_sides': OVER_SIDE, 'overlay_ts': TS_OVER,
                   'weights': PORT, 'full_n': n,
                   'y1_note': 'Y1 unknown: fallback X5 (ETC-cd18/TRX-cd6 sth0.12) + '
                              'X6 (sl0.02 + TRX-long-off) + X4 (ts24), q0.3, S2-base, 50/50',
                   'segments': {k: list(v) for k, v in segs.items()}},
        'ledgers': out_ledgers,
        'drift_overlay_minus_plain': drift,
        'gate': {'rule': 'overlay active at bar t iff trailing-200 overlay-combo '
                         'per-bar-net sharpe(t-200:t) > 1.0 (causal, no lookahead)',
                 'window': GATE_WIN, 'thresh': GATE_THRESH,
                 'alt_rule_rejected': 'overlay active iff trailing-200 plain-combo sharpe < 1.0 '
                                      '(LED-switch churn: shadow B/C below plain)',
                 'shadows': gate_out},
        'verdict': verdict,
    }
    open('results/backtest_Y3.json', 'w').write(json.dumps(res, indent=1))
    print('PASS ' + json.dumps(verdict), flush=True)
    print('saved results/backtest_Y3.json', flush=True)

if __name__ == '__main__':
    main()
