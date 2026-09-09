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
START_EQ = 10000.0
def load_bars(coin):
    import csv
    rows = list(csv.DictReader(open('data/data_15m_3y/' + coin + '.csv')))
    bars = []
    for i in range(0, len(rows), 16):
        blk = rows[i:i + 16]
        if len(blk) < 16: break
        bars.append((float(blk[0]['open']), max(float(x['high']) for x in blk), min(float(x['low']) for x in blk), float(blk[-1]['close']), sum(float(x['volume']) for x in blk)))
    return bars
def run_paper():
    import torch
    from model_core.backtest import MemeBacktest
    coins = sorted(PORT)
    bars = {c: load_bars(c) for c in coins}
    n = min(len(b) for b in bars.values())
    print('bars:', {c: len(bars[c]) for c in coins}, 'use_n:', n, flush=True)
    equity = START_EQ
    peak = START_EQ
    max_dd_pct = 0.0
    per_coin_state = {}
    for c in coins:
        b = bars[c][:n]
        raw = {'open': torch.tensor([[x[0] for x in b]]), 'high': torch.tensor([[x[1] for x in b]]), 'low': torch.tensor([[x[2] for x in b]]), 'close': torch.tensor([[x[3] for x in b]]), 'volume': torch.tensor([[x[4] for x in b]]), 'liquidity': torch.full((1, n), 1e7), 'fdv': torch.full((1, n), 1e8)}
        sig = StackVM().execute(FORMULA, FeatureEngineer.compute_features(raw))
        px = torch.tensor([x[3] for x in b])
        lth, sth, cd, sl = BEST[c]
        bt = MemeBacktest(venue='aster', leverage=2.0, short_enabled=True, funding_override=FUND, long_th=lth, short_th=sth, cooldown_bars=cd, bars_per_year=2190.0, stop_loss=sl)
        sg = torch.sigmoid(sig)
        safe = (raw['liquidity'] > bt.min_liq).float()
        lp = (sg > bt.long_th).float() * safe
        sp = (sg < bt.short_th).float() * safe
        lp, sp = bt._apply_cooldown(lp, sp)
        lp, sp = bt._apply_stops(lp, sp, torch.tensor([[(bars[c][i + 1][3] - bars[c][i][3]) / bars[c][i][3] if i < n - 1 else 0.0 for i in range(n)]]))
        per_coin_state[c] = {'lp': lp[0].tolist(), 'sp': sp[0].tolist(), 'px': px.tolist()}
    eq_curve = []
    ledger = []
    pos = {c: 0.0 for c in coins}
    entry = {c: 0.0 for c in coins}
    for t in range(n):
        for c in coins:
            px = per_coin_state[c]['px'][t]
            want = 1.0 if per_coin_state[c]['lp'][t] > 0.5 else (-1.0 if per_coin_state[c]['sp'][t] > 0.5 else 0.0)
            if t > 0 and want != pos[c]:
                prev = per_coin_state[c]['px'][t - 1]
                fill = prev
                if pos[c] != 0.0:
                    pnl_pct = (fill - entry[c]) / entry[c] * (1.0 if pos[c] > 0 else -1.0)
                    pnl_pct = pnl_pct * 2.0 - FEE * 2.0 * 2.0 - FUND * 2.0
                    w = PORT[c]
                    equity += START_EQ * w * pnl_pct
                    ledger.append({'t': t, 'coin': c, 'close': 1 if pos[c] > 0 else -1, 'fill': fill, 'pnl_pct': round(pnl_pct, 6), 'equity': round(equity, 2)})
                if want != 0.0:
                    entry[c] = fill
                pos[c] = want
        peak = max(peak, equity)
        dd = (peak - equity) / peak
        max_dd_pct = max(max_dd_pct, dd)
        eq_curve.append(equity)
    rets = [(eq_curve[i + 1] - eq_curve[i]) / eq_curve[i] if eq_curve[i] else 0.0 for i in range(len(eq_curve) - 1)]
    mean = sum(rets) / max(len(rets), 1)
    var = sum((x - mean) ** 2 for x in rets) / max(len(rets) - 1, 1)
    sharpe = mean / math.sqrt(var) * math.sqrt(2190.0) if var > 0 else 0.0
    tot = (equity - START_EQ) / START_EQ
    ratio = equity / START_EQ
    import math as _m
    ann = float(_m.exp(2190.0 / max(len(rets), 1) * _m.log(ratio)) - 1.0) if ratio > 0 else float('-inf')
    print('paper ETC+TRX 50/50 lev2x fee=' + str(FEE) + ' fund=' + str(FUND), flush=True)
    print('trades=' + str(len(ledger)) + ' final_eq=' + str(round(equity, 2)) + ' tot=' + str(round(tot, 4)) + ' ann=' + str(round(ann, 4)) + ' sharpe=' + str(round(sharpe, 3)) + ' maxdd=' + str(round(max_dd_pct, 4)), flush=True)
    by = {}
    for e in ledger:
        by[e['coin']] = by.get(e['coin'], 0) + 1
    print('trades_by_coin:', by, flush=True)
    open('results/paper_trades.json', 'w').write(json.dumps({'ledger': ledger, 'equity': eq_curve, 'stats': {'final_eq': equity, 'tot': tot, 'ann': ann, 'sharpe': sharpe, 'maxdd': max_dd_pct, 'trades': len(ledger), 'by': by}}))
    print('saved paper_trades.json', flush=True)
if __name__ == '__main__':
    run_paper()
