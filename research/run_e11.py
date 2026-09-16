import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import csv, math, json, pathlib
import torch
from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest

FORMULA = [3,2,7,2,7,11,15,4,4,6,6,10]
COINS = ['ETC','TRX']
Q = 0.3
VT = 0.012
VW = 12
TS = 24
FUND_BASE = 0.0005
FEE_BASE = 0.0004
FEE2X = 0.0008
N = 6580
FOLD_N = 6580 // 12

# E11: AA etc15 variants: sth[0.10,0.11,0.12] x etc_cd[15,18] x trx_cd[6,9] = 12
# Fixed: ETC lth 0.88 sl None, TRX lth 0.85 sl 0.05, ts24, q0.3, vt0.012, 50/50, aster 2x
STH_GRID = [0.10, 0.11, 0.12]
ETC_GRID = [15, 18]
TRX_GRID = [6, 9]

def load_bars(coin):
    rows = list(csv.DictReader(open(f'data/data_15m_3y/{coin}.csv')))
    bars = []
    for i in range(0, len(rows), 16):
        blk = rows[i:i+16]
        if len(blk) < 16:
            break
        bars.append((float(blk[0]['open']), max(float(x['high']) for x in blk), min(float(x['low']) for x in blk), float(blk[-1]['close']), sum(float(x['volume']) for x in blk)))
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
    rets = [(bars[i+1][3] - bars[i][3]) / bars[i][3] for i in range(n - 1)] + [0.0]
    return raw, torch.tensor([rets]), sig

def quantile_mask_long(sig, q):
    if q is None:
        return None
    a = sig.detach().float().abs().reshape(-1)
    k = max(1, int(len(a) * float(q)))
    thr = torch.topk(a, k).values.min()
    return (sig.detach().float().abs() >= thr).float()

def leg_series(raw, rets_t, sig, lth, sth, cd, sl, fee=None, fund=None, side='both', q=Q, vt=VT, vw=VW, ts=TS):
    kw = dict(venue='aster', leverage=2.0, short_enabled=True, long_th=lth, short_th=sth, cooldown_bars=cd, bars_per_year=2190.0, stop_loss=sl, time_stop=ts, vol_target=vt, vol_window=vw)
    kw['funding_override'] = FUND_BASE if fund is None else fund
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
    trades = sum(1 for t in range(len(pos)) if pos[t] != 0.0 and (t == 0 or pos[t-1] == 0.0))
    turnover = sum(turnl) / len(turnl) if turnl else 0.0
    return net, trades, turnover

def stats(ser, trades=0, turnover=0.0):
    n = len(ser)
    mean = sum(ser) / n if n else 0
    var = sum((x - mean) ** 2 for x in ser) / max(n - 1, 1) if n > 1 else 0
    sharpe = mean / math.sqrt(var) * math.sqrt(2190.0) if var > 0 else 0.0
    cum = sum(ser)
    ann = cum / n * 2190.0 if n else 0
    cs, peak, mdd = 0.0, -1e18, 0.0
    for x in ser:
        cs += x
        peak = max(peak, cs)
        mdd = max(mdd, peak - cs)
    return {'sharpe': round(sharpe, 3), 'ann': round(ann, 4), 'mdd': round(mdd, 4), 'cum': round(cum, 4), 'n': n, 'trades': trades, 'turnover': round(turnover, 6)}

def combo(ser_list, weights=None):
    m = min(len(s) for s in ser_list)
    k = len(ser_list)
    w = weights or [1.0 / k] * k
    sw = sum(w)
    return [sum(ser_list[i][t] * w[i] / sw for i in range(k)) for t in range(m)]

def eval_combo_on_bars(mats_by_coin, cfg, fee, fund):
    legs, trs, tos = [], [], []
    for coin, (lth, cd, sl) in [('ETC', (0.88, cfg['etc_cd'], None)), ('TRX', (0.85, cfg['trx_cd'], 0.05))]:
        raw, rt, sg = mats_by_coin[coin]
        net, t, to = leg_series(raw, rt, sg, lth, cfg['sth'], cd, sl, fee=fee, fund=fund, vt=cfg['vt'], vw=cfg['vw'])
        legs.append(net); trs.append(t); tos.append(to)
    cb = combo(legs, [0.5, 0.5])
    s = stats(cb, trades=sum(trs), turnover=sum(tos) / len(tos) if tos else 0)
    s['trades_by'] = {c: trs[i] for i, c in enumerate(COINS)}
    return s, cb

def walkforward_12fold(full, cfg, fee, fund):
    folds = []
    for i in range(12):
        a = i * FOLD_N
        b = (i + 1) * FOLD_N if i < 11 else N
        mats = {c: build_mats(full[c][a:b]) for c in COINS}
        s, _ = eval_combo_on_bars(mats, cfg, fee=fee, fund=fund)
        s['fold'] = i; s['range'] = [a, b]
        folds.append(s)
    return folds

def fold_summary(folds):
    sharpes = [f['sharpe'] for f in folds]
    n = len(sharpes)
    mean = sum(sharpes) / n if n else 0
    srt = sorted(sharpes)
    med = srt[n // 2] if n % 2 == 1 else (srt[n // 2 - 1] + srt[n // 2]) / 2
    n_pos = sum(1 for x in sharpes if x > 0)
    anns = [f['ann'] for f in folds]
    return {'sharpes': [round(x, 3) for x in sharpes], 'mean': round(mean, 3), 'median': round(med, 3),
            'min': round(min(sharpes), 3) if sharpes else 0, 'max': round(max(sharpes), 3) if sharpes else 0,
            'n_pos': int(n_pos), 'n_neg': int(n - n_pos),
            'max_dd': round(max(f['mdd'] for f in folds), 4) if folds else 0,
            'mean_ann': round(sum(anns) / len(anns), 4) if anns else 0}

def main():
    logp = pathlib.Path('logs/e11.log')
    logp.parent.mkdir(parents=True, exist_ok=True)
    open(logp, 'w').write('E11 start\n')
    def log(msg):
        print(msg, flush=True)
        with open(logp, 'a') as f:
            f.write(msg + '\n')
    full = {c: load_bars(c) for c in COINS}
    n = min(len(v) for v in full.values())
    log(f'full 4h bars n={n}')
    assert n == N, f'expected {N} got {n}'
    SEGS = {'H1': (5584, 6077), 'H2': (6077, 6570), 'B': (6380, 6580), 'C': (6080, 6580), 'FULL': (0, 6580)}
    mats = {seg: {c: build_mats(full[c][a:b]) for c in COINS} for seg, (a, b) in SEGS.items()}
    log('segments built')
    cfg0 = {'sth': 0.12, 'etc_cd': 18, 'trx_cd': 6, 'vt': None, 'vw': 12}
    y1b = {}
    for seg in SEGS:
        s, _ = eval_combo_on_bars(mats[seg], cfg0, fee=FEE_BASE, fund=FUND_BASE)
        y1b[seg] = s
    y1b_folds = walkforward_12fold(full, cfg0, FEE_BASE, FUND_BASE)
    y1b_summ = fold_summary(y1b_folds)
    log(f"Y1b H1={y1b['H1']['sharpe']:.3f} H2={y1b['H2']['sharpe']:.3f} FULL={y1b['FULL']['sharpe']:.3f} 12fold mean={y1b_summ['mean']} med={y1b_summ['median']} n_pos={y1b_summ['n_pos']}/12")
    rows = []
    for sth in STH_GRID:
        for etc_cd in ETC_GRID:
            for trx_cd in TRX_GRID:
                cfg = {'sth': sth, 'etc_cd': etc_cd, 'trx_cd': trx_cd, 'vt': VT, 'vw': VW}
                segs = {}
                for seg in SEGS:
                    s, _ = eval_combo_on_bars(mats[seg], cfg, fee=FEE_BASE, fund=FUND_BASE)
                    segs[seg] = s
                h2x, _ = eval_combo_on_bars(mats['H2'], cfg, fee=FEE2X, fund=FUND_BASE)
                folds = walkforward_12fold(full, cfg, FEE_BASE, FUND_BASE)
                summ = fold_summary(folds)
                g_med = summ['median'] >= 1.5
                g_mean = summ['mean'] > 1.7
                g_npos = summ['n_pos'] >= 9
                gates = {'median_ge1.5': bool(g_med), 'mean_gt1.7': bool(g_mean), 'npos_ge9': bool(g_npos)}
                row = {'sth': sth, 'etc_cd': etc_cd, 'trx_cd': trx_cd,
                       'H1': segs['H1'], 'H2': segs['H2'], 'B': segs['B'], 'C': segs['C'], 'FULL': segs['FULL'],
                       'H2_fee2x': h2x, 'fee_decay_H2': round(segs['H2']['sharpe'] - h2x['sharpe'], 3),
                       'wf12_summary': summ, 'gates': gates, 'gates_pass': bool(g_med and g_mean and g_npos)}
                rows.append(row)
                log(f"sth={sth:.2f} etc{etc_cd} trx{trx_cd} | H1={segs['H1']['sharpe']:.3f} H2={segs['H2']['sharpe']:.3f} H2_2x={h2x['sharpe']:.3f} B={segs['B']['sharpe']:.3f} C={segs['C']['sharpe']:.3f} FULL={segs['FULL']['sharpe']:.3f} | 12f mean={summ['mean']} med={summ['median']} n_pos={summ['n_pos']}/12 gates={gates} => {'PASS' if row['gates_pass'] else 'FAIL'}")
    h1_best = max(rows, key=lambda r: r['H1']['sharpe'])
    log(f"H1-best sth={h1_best['sth']} etc{h1_best['etc_cd']} trx{h1_best['trx_cd']} H1={h1_best['H1']['sharpe']:.3f} -> H2={h1_best['H2']['sharpe']:.3f} FULL={h1_best['FULL']['sharpe']:.3f} gates_pass={h1_best['gates_pass']}")
    hold = bool(h1_best['H2']['sharpe'] > 3.0)
    overall = bool(hold and h1_best['gates_pass'])
    verdict = 'PROMOTE sth=%.2f/etc%d/trx%d' % (h1_best['sth'], h1_best['etc_cd'], h1_best['trx_cd']) if overall else 'KEEP_Y1b'
    log(f"H2-hold H2>3.0: {h1_best['H2']['sharpe']:.3f} => {'PASS' if hold else 'FAIL'}; overall => {verdict}")
    passers = [r for r in rows if r['gates_pass']]
    out = {'config': {'formula': FORMULA, 'coins': COINS, 'N': N, 'fold_n': FOLD_N, 'q': Q, 'vt': VT, 'vw': VW, 'ts': TS,
                      'fund_base': FUND_BASE, 'fee_base': FEE_BASE, 'fee2x': FEE2X,
                      'grid': 'sth[0.10,0.11,0.12] x etc_cd[15,18] x trx_cd[6,9] = 12 variants, vt0.012, lth ETC0.88 TRX0.85 sl ETC None TRX0.05 ts24 q0.3 50/50 aster lev2',
                      'segments': {'H1': [5584, 6077], 'H2': [6077, 6570], 'B': [6380, 6580], 'C': [6080, 6580], 'FULL': [0, 6580]},
                      'engine': 'mirror run_e1.py/run_aa.py leg_series + quantile_mask_long(q0.3 long-only) + _vol_scale post-stops pre-roll + roll1',
                      'gates': 'H1-best -> H2>3.0 hold AND 12fold median>=1.5 AND mean>1.7 AND n_pos>=9'},
           'Y1b_baseline': {'H1': y1b['H1'], 'H2': y1b['H2'], 'B': y1b['B'], 'C': y1b['C'], 'FULL': y1b['FULL'], 'wf12_summary': y1b_summ, 'wf12_folds': y1b_folds},
           'variants': [{k: r[k] for k in ('sth', 'etc_cd', 'trx_cd', 'H1', 'H2', 'B', 'C', 'FULL', 'H2_fee2x', 'fee_decay_H2', 'wf12_summary', 'gates', 'gates_pass')} for r in rows],
           'h1_best': {'sth': h1_best['sth'], 'etc_cd': h1_best['etc_cd'], 'trx_cd': h1_best['trx_cd'], 'H1': h1_best['H1'], 'H2': h1_best['H2'], 'B': h1_best['B'], 'C': h1_best['C'], 'FULL': h1_best['FULL'], 'H2_fee2x': h1_best['H2_fee2x'], 'wf12_summary': h1_best['wf12_summary'], 'gates': h1_best['gates'], 'gates_pass': h1_best['gates_pass']},
           'decision': {'h2_hold_thr': 3.0, 'h2_hold': hold, 'n_gate_passers': len(passers),
                        'passers': [{'sth': r['sth'], 'etc_cd': r['etc_cd'], 'trx_cd': r['trx_cd'], 'H1': r['H1']['sharpe'], 'H2': r['H2']['sharpe'], 'wf12': r['wf12_summary']} for r in passers],
                        'verdict': verdict}}
    pathlib.Path('results/results_E11_walkforward.json').write_text(json.dumps(out, indent=2, ensure_ascii=False))
    log(f'Wrote results/results_E11_walkforward.json verdict={verdict}')
    print(json.dumps({'verdict': verdict, 'h1_best': out['h1_best'], 'Y1b_wf12': y1b_summ, 'n_passers': len(passers)}, indent=2))

if __name__ == '__main__':
    main()
