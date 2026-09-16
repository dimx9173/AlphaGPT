import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import csv, json, math
import torch
from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest
FORMULA = [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10]
PORT = {'ETC': 0.5, 'TRX': 0.5}
BEST = {'ETC': (0.88, 0.12, 12, None), 'TRX': (0.85, 0.15, 6, 0.05)}
FEE = 0.0004
FUND = 0.0005
LEV = 2.0
def load_bars(coin):
    import csv
    rows = list(csv.DictReader(open('data/data_15m_3y/' + coin + '.csv')))
    bars = []
    for i in range(0, len(rows), 16):
        blk = rows[i:i + 16]
        if len(blk) < 16: break
        bars.append((float(blk[0]['open']), max(float(x['high']) for x in blk), min(float(x['low']) for x in blk), float(blk[-1]['close']), sum(float(x['volume']) for x in blk)))
    return bars
Q = 0.3
BASELINE = {'trades': 477, 'final_x': 2.927, 'sharpe': 0.891, 'mdd': 0.7846}
def leg_net(bars, lth, sth, cd, sl, q=None):
    import torch
    from model_core.backtest import MemeBacktest
    n = len(bars)
    raw = {'open': torch.tensor([[x[0] for x in bars]]), 'high': torch.tensor([[x[1] for x in bars]]), 'low': torch.tensor([[x[2] for x in bars]]), 'close': torch.tensor([[x[3] for x in bars]]), 'volume': torch.tensor([[x[4] for x in bars]]), 'liquidity': torch.full((1, n), 1e7), 'fdv': torch.full((1, n), 1e8)}
    sig = StackVM(use_advanced=False).execute(FORMULA, FeatureEngineer.compute_features(raw, use_advanced=False))
    rets = [(bars[i + 1][3] - bars[i][3]) / bars[i][3] if i < n - 1 else 0.0 for i in range(n)]
    bt = MemeBacktest(venue='aster', leverage=LEV, short_enabled=True, funding_override=FUND, long_th=lth, short_th=sth, cooldown_bars=cd, bars_per_year=2190.0, stop_loss=sl)
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
def run_variant(q):
    coins = sorted(PORT)
    bars = {c: load_bars(c) for c in coins}
    n = min(len(b) for b in bars.values())
    legs = {}
    poss = {}
    for c in coins:
        lth, sth, cd, sl = BEST[c]
        net, pos = leg_net(bars[c][:n], lth, sth, cd, sl, q=q)
        legs[c] = net; poss[c] = pos
    eq = [1.0]
    ledger = []
    cur = {c: 0.0 for c in coins}
    ent = {c: 0.0 for c in coins}
    px = {c: [b[3] for b in bars[c][:n]] for c in coins}
    for t in range(n):
        for c in coins:
            want = poss[c][t]
            want = 1.0 if want > 0.5 else (-1.0 if want < -0.5 else 0.0)
            if t > 0 and want != cur[c]:
                fill = px[c][t - 1]
                if cur[c] != 0.0:
                    move = (fill - ent[c]) / ent[c] * (1.0 if cur[c] > 0 else -1.0)
                    netp = move * LEV - FEE * LEV * 2.0 - FUND * LEV
                    dol = eq[-1] * PORT[c] * netp
                    eq.append(eq[-1] + dol / 1.0)
                    ledger.append({'t': t, 'coin': c, 'side': int(cur[c]), 'fill': fill, 'move': round(move, 6), 'eq': round(eq[-1], 4)})
                if want != 0.0:
                    ent[c] = fill
                cur[c] = want
        if len(eq) < t + 2:
            eq.append(eq[-1])
    eq = eq[:n]
    return eq, ledger

def run():
    eq_q, led_q = run_variant(Q)
    eq_b, led_b = run_variant(None)
    import math as _m
    def _stats(eq, ledger):
        rets = [(eq[i + 1] - eq[i]) / eq[i] if eq[i] else 0.0 for i in range(len(eq) - 1)]
        mean = sum(rets) / max(len(rets), 1)
        var = sum((x - mean) ** 2 for x in rets) / max(len(rets) - 1, 1)
        sharpe = mean / _m.sqrt(var) * _m.sqrt(2190.0) if var > 0 else 0.0
        peak = eq[0]; mdd = 0.0
        for v in eq:
            peak = max(peak, v); mdd = max(mdd, (peak - v) / peak if peak else 0.0)
        by = {}
        for e in ledger: by[e['coin']] = by.get(e['coin'], 0) + 1
        return {'final_x': round(eq[-1], 4), 'sharpe': round(sharpe, 3), 'mdd': round(mdd, 4), 'trades': len(ledger), 'by': by}
    sq, sb = _stats(eq_q, led_q), _stats(eq_b, led_b)
    # attribution: longs filtered
    lq = sum(1 for e in led_q if e['side'] == 1); lb = sum(1 for e in led_b if e['side'] == 1)
    drift = {'d_trades': sq['trades'] - sb['trades'], 'd_final_x': round(sq['final_x'] - sb['final_x'], 4), 'd_sharpe': round(sq['sharpe'] - sb['sharpe'], 3)}
    drift_vs_paper2 = {'d_trades': sq['trades'] - BASELINE['trades'], 'd_final_x': round(sq['final_x'] - BASELINE['final_x'], 4), 'd_sharpe': round(sq['sharpe'] - BASELINE['sharpe'], 3)}
    print('X10 q0.3 trades=%d final_x=%s sharpe=%s mdd=%s by=%s' % (sq['trades'], sq['final_x'], sq['sharpe'], sq['mdd'], sq['by']), flush=True)
    print('X10 plain trades=%d final_x=%s sharpe=%s mdd=%s by=%s' % (sb['trades'], sb['final_x'], sb['sharpe'], sb['mdd'], sb['by']), flush=True)
    print('X10 longs filtered: q=%d vs plain=%d' % (lq, lb), flush=True)
    print('X10 drift(q-plain)=' + str(drift) + ' drift(q-paper2log)=' + str(drift_vs_paper2), flush=True)
    res = {'config': {'q': Q, 'baseline_paper2log': BASELINE}, 'q03': sq, 'plain': sb,
           'longs': {'q03': lq, 'plain': lb, 'filtered_out': lb - lq},
           'drift_q_minus_plain': drift, 'drift_q_minus_paper2log': drift_vs_paper2,
           'verdict': 'explain drift root cause; propose paper-fix; no blind adopt'}
    open('results/backtest_X10.json', 'w').write(json.dumps(res, indent=1))
    print('saved results/backtest_X10.json', flush=True)

if __name__ == '__main__':
    run()
