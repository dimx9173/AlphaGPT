"""H14 realized-funding check (1h native, Top5) - DIAGNOSTIC ONLY.

Engine mirrors research/run_weight_modes.py leg_net (E10 FORMULA
[3,2,7,2,7,11,15,4,4,6,6,10] via StackVM+FeatureEngineer; MemeBacktest
venue=aster lev2 short_enabled fund0.0005; quantile q0.3 long-only +
cooldown + stops + vol_scale(vt None->1.0) + roll1). 1h native: reads
data/data_1y/1h directly (no aggregation); cd/ts/vw x4 (4h-bar units
-> 1h-bar units); BPY=8760. Equal 0.2 weights. Top5 locked specs
ETC(0.88/0.12/cd18/None/ts24) TRX(0.85/0.12/cd6/0.05/ts24)
ATOM(0.85/0.15/cd6/0.05/ts24) APT(0.88/0.12/cd18/None/ts24)
KAS(0.88/0.12/cd6/None/ts24), q0.3.

Task: realized funding from Bybit demo venue vs assumed 0.0005, plus
funding PnL share per coin. Venue reads are PUBLIC-ONLY against
https://api-demo.bybit.com (tickers + funding/history); no signed
endpoints, no API keys, no orders, no position writes. Backtest
positions come from the locked engine above; realized venue rates are
applied arithmetically to those positions (no live change).

Per-coin signed-funding decomposition (long pays positive):
  gross = pos * rt * LEV; tx = turn * fee * LEV
  fnd_assumed = pos * 0.0005 * LEV; net_assumed = gross - tx - fnd_assumed
  fnd_real = pos * realized_mean * LEV; net_real = gross - tx - fnd_real
Funding PnL share per coin: share_net = fund_cum / net_cum,
share_gross = fund_cum / gross_cum (None when denominator is 0).

Verdict PENDING (P0-3 FAIL); no adoption; live FUND/FEE/LEV untouched.
Incremental dump: OUT rewritten after EACH unit (legs, venue coins,
realized evals); status partial->final so partial progress survives.
Smoke: ITER_H14_SMOKE=1 -> coins ETC,TRX, first 3000 bars, history limit 5.
ITER_H14_NO_VENUE=1 skips network (deterministic fallback).
OUT/LOG overridable via ITER_H14_OUT / ITER_H14_LOG.
"""

import csv
import json
import math
import os
import pathlib
import sys
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from model_core.factors import FeatureEngineer
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest
from strategy_manager.config import (
    FORMULA, LOCKED_ATOM, LOCKED_APT, LOCKED_ETC, LOCKED_KAS, LOCKED_TRX,
    LEV, FUND, FEE, FEE2X,
)

assert LEV == 2.0, 'LEV lock broken: %r' % LEV
assert abs(FUND - 0.0005) < 1e-12, 'FUND lock broken: %r' % FUND
assert abs(FEE2X - 2 * FEE) < 1e-12, 'FEE lock broken'
assert FORMULA == [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10], 'FORMULA lock broken'

COINS = ['ETC', 'TRX', 'ATOM', 'APT', 'KAS']
BASE_SPECS = {
    'ETC': dict(LOCKED_ETC),
    'TRX': dict(LOCKED_TRX),
    'ATOM': dict(LOCKED_ATOM),
    'APT': dict(LOCKED_APT),
    'KAS': dict(LOCKED_KAS),
}
SYMBOLS = {c: c + 'USDT' for c in COINS}
BPY = 8760.0
SCALE = 4
FEE_USE = FEE
FUND_ASSUMED = FUND
VENUE_BASE = 'https://api-demo.bybit.com'
OUT = pathlib.Path(os.getenv('ITER_H14_OUT', 'results/iter_H14_fundreal.json'))
LOG = pathlib.Path(os.getenv('ITER_H14_LOG', 'logs/iter_H14_fundreal.log'))
SMOKE = os.getenv('ITER_H14_SMOKE') == '1'
NO_VENUE = os.getenv('ITER_H14_NO_VENUE') == '1'


def log(m):
    print(m, flush=True)
    open(LOG, 'a').write(m + '\n')


def scaled_spec(base):
    s = dict(base)
    for k in ('cd', 'ts', 'vw'):
        if s.get(k) is not None:
            s[k] = int(s[k]) * SCALE
    return s


SPECS = {c: scaled_spec(b) for c, b in BASE_SPECS.items()}


def load1h(c):
    rows = list(csv.DictReader(open('data/data_1y/1h/%s.csv' % c)))
    return [(int(r['timestamp']), float(r['open']), float(r['high']),
             float(r['low']), float(r['close']), float(r['volume']))
            for r in rows]


def common_grid(coins):
    raw = {c: load1h(c) for c in coins}
    ts0 = set(r[0] for r in raw[coins[0]])
    for c in coins[1:]:
        ts0 &= set(r[0] for r in raw[c])
    ts_sorted = sorted(ts0)
    bars = {}
    for c in coins:
        m = {r[0]: r for r in raw[c]}
        bars[c] = [(m[t][1], m[t][2], m[t][3], m[t][4], m[t][5]) for t in ts_sorted]
    return ts_sorted, bars


def build_sig(bars):
    n = len(bars)
    raw = {
        'open': torch.tensor([[b[0] for b in bars]]),
        'high': torch.tensor([[b[1] for b in bars]]),
        'low': torch.tensor([[b[2] for b in bars]]),
        'close': torch.tensor([[b[3] for b in bars]]),
        'volume': torch.tensor([[b[4] for b in bars]]),
        'liquidity': torch.full((1, n), 1e7),
        'fdv': torch.full((1, n), 1e8),
    }
    sig = StackVM().execute(FORMULA, FeatureEngineer.compute_features(raw))
    rets = [(bars[i + 1][3] - bars[i][3]) / bars[i][3] for i in range(n - 1)] + [0.0]
    return raw, torch.tensor([rets]), sig


def qmask(sig, q):
    if q is None or float(q) <= 0:
        return None
    a = sig.detach().float().abs().reshape(-1)
    k = max(1, int(len(a) * float(q)))
    thr = torch.topk(a, k).values.min()
    return (sig.detach().float().abs() >= thr).float()


def leg_parts(raw, rt, sig, spec, fee):
    bt = MemeBacktest(
        venue='aster', leverage=LEV, short_enabled=True,
        funding_override=0.0, fee_override=fee,
        long_th=spec['lth'], short_th=spec['sth'],
        cooldown_bars=spec['cd'], bars_per_year=BPY,
        stop_loss=spec['sl'], time_stop=spec['ts'],
        vol_target=spec['vt'], vol_window=spec['vw'],
    )
    sg = torch.sigmoid(sig)
    safe = (raw['liquidity'] > bt.min_liq).float()
    lp = (sg > bt.long_th).float() * safe
    sp = (sg < bt.short_th).float() * safe
    mk = qmask(sig, spec['q'])
    if mk is not None:
        lp = lp * mk
    lp, sp = bt._apply_cooldown(lp, sp)
    lp, sp = bt._apply_stops(lp, sp, rt)
    sc = bt._vol_scale(rt)
    lp, sp = lp * sc, sp * sc
    lp = lp.roll(1, dims=1)
    lp[:, 0] = 0
    sp = sp.roll(1, dims=1)
    sp[:, 0] = 0
    turn = (lp - lp.roll(1, dims=1)).abs() + (sp - sp.roll(1, dims=1)).abs()
    pos = lp - sp
    gross = pos * rt * bt.leverage
    tx = turn * bt.base_fee * bt.leverage
    return pos[0].tolist(), turn[0].tolist(), gross[0].tolist(), rt[0].tolist()

def http_json(url, timeout=10):
    req = urllib.request.Request(url, headers={'User-Agent': 'AlphaGPT-H14/1.0'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        import json as _j
        return _j.loads(r.read().decode())


def venue_coin(symbol, hist_limit=200):
    out = {'symbol': symbol, 'latest_rate': None, 'latest_ts': None,
           'mean_rate': None, 'n_hist': 0, 'history': [], 'ok': False, 'error': None}
    if NO_VENUE:
        out['error'] = 'disabled via ITER_H14_NO_VENUE=1'
        return out
    try:
        tick = http_json(VENUE_BASE + '/v5/market/tickers?category=linear&symbol=' + symbol)
        rows = ((tick.get('result') or {}).get('list')) or []
        if rows:
            fr = rows[0].get('fundingRate')
            try:
                out['latest_rate'] = float(fr) if fr not in (None, '') else None
            except (TypeError, ValueError):
                out['latest_rate'] = None
            out['latest_ts'] = rows[0].get('fundingRateTimestamp') or rows[0].get('nextFundingTime')
    except Exception as e:
        out['error'] = 'tickers: ' + repr(e)[:160]
    try:
        hist = http_json(VENUE_BASE + '/v5/market/funding/history?category=linear&symbol=' + symbol + '&limit=' + str(hist_limit))
        rows = ((hist.get('result') or {}).get('list')) or []
        vals = []
        for r in rows:
            try:
                v = float(r.get('fundingRate'))
            except (TypeError, ValueError):
                continue
            vals.append(v)
        out['history'] = vals
        out['n_hist'] = len(vals)
        if vals:
            out['mean_rate'] = sum(vals) / len(vals)
            out['ok'] = True
    except Exception as e:
        err = 'history: ' + repr(e)[:160]
        out['error'] = (out['error'] + '; ' + err) if out['error'] else err
    return out


def seg(net, turn, a, b):
    s = net[a:b]
    t = turn[a:b]
    n = len(s)
    m = sum(s) / n if n else 0.0
    v = sum((x - m) ** 2 for x in s) / max(n - 1, 1) if n > 1 else 0.0
    import math as _m
    sh = m / _m.sqrt(v) * _m.sqrt(BPY) if v > 0 else 0.0
    cs = 0.0
    pk = 0.0
    md = 0.0
    for x in s:
        cs += x
        pk = max(pk, cs)
        md = max(md, pk - cs)
    cum = sum(s)
    return {'sharpe': round(sh, 3), 'ann': round(cum / n * BPY, 4) if n else 0.0,
            'mdd': round(md, 4), 'cum': round(cum, 4),
            'final_x': round(1.0 + cum, 4), 'n': n,
            'turnover': round(sum(t) / n, 6) if n else 0.0}


def safe_ratio(num, den):
    if den is None or abs(den) < 1e-12:
        return None
    return round(num / den, 4)


def dump(state):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix('.tmp')
    tmp.write_text(json.dumps(state, indent=1, ensure_ascii=False))
    os.replace(tmp, OUT)


def base_config(coins, n, hist_limit):
    return {'engine': 'mirror run_weight_modes leg_net + quantile q0.3 long-only + cooldown + stops + vol_scale(vt None->1.0) + roll1; static equal Top5; 1h native cd/ts/vw x4',
            'formula': list(FORMULA),
            'basket': {c: dict(BASE_SPECS[c]) for c in coins},
            'scaled_specs_1h': {c: dict(SPECS[c]) for c in coins},
            'scale_4h_to_1h': SCALE,
            'weights': {c: round(1.0 / len(coins), 6) for c in coins},
            'venue': 'aster', 'live_venue_reads': 'bybit-demo public tickers+funding/history (no keys, no orders)',
            'bybit_demo_base': VENUE_BASE, 'bybit_demo_symbols': {c: SYMBOLS[c] for c in coins},
            'venue_hist_limit': hist_limit, 'no_venue': NO_VENUE,
            'lev': LEV, 'fund_assumed': FUND_ASSUMED, 'fee': FEE_USE, 'fee2x': FEE2X,
            'grid': '1h', 'grid_bars': n, 'bpy': BPY, 'data_dir': 'data/data_1y/1h',
            'coins': list(coins), 'smoke': SMOKE,
            'funding_acct': 'signed long-pays-positive fnd = pos * rate * LEV',
            'share_def': 'share_net = fund_cum / net_cum; share_gross = fund_cum / gross_cum (None when denom 0)',
            'note': 'H14 realized-funding diagnostic; read-only; PENDING; live untouched'}

def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    open(LOG, 'w').write('iter_H14_fundreal start smoke=%s no_venue=%s\n' % (SMOKE, NO_VENUE))
    coins = ['ETC', 'TRX'] if SMOKE else list(COINS)
    hist_limit = 5 if SMOKE else 200
    ts, bars = common_grid(coins)
    n = len(ts)
    if SMOKE:
        n = min(n, 3000)
        bars = {c: bars[c][:n] for c in coins}
        ts = ts[:n]
    log('common 1h n=%d coins=%s smoke=%s scale=x%d fee=%.4f fund_assumed=%.4f' % (n, coins, SMOKE, SCALE, FEE_USE, FUND_ASSUMED))
    assert n > 2000, n
    w = 1.0 / len(coins)
    cfg = base_config(coins, n, hist_limit)
    dump({'status': 'partial', 'stage': 'loaded', 'config': cfg})
    poss, turns, gross, rts = {}, {}, {}, {}
    for c in coins:
        raw, rt, sig = build_sig(bars[c])
        poss[c], turns[c], gross[c], rts[c] = leg_parts(raw, rt, sig, SPECS[c], FEE_USE)
        dump({'status': 'partial', 'stage': 'leg_' + c, 'config': cfg, 'legs_done': sorted(poss)})
        log('leg %s done' % c)
    assumed_net, assumed_fnd = {}, {}
    for c in coins:
        assumed_fnd[c] = [poss[c][t] * FUND_ASSUMED * LEV for t in range(n)]
        assumed_net[c] = [gross[c][t] - turns[c][t] * FEE_USE * LEV - assumed_fnd[c][t] for t in range(n)]
    venue = {}
    for c in coins:
        venue[c] = venue_coin(SYMBOLS[c], hist_limit)
        dump({'status': 'partial', 'stage': 'venue_' + c, 'config': cfg,
              'legs_done': sorted(poss), 'venue_done': sorted(venue)})
        v = venue[c]
        log('venue %s ok=%s n_hist=%d mean=%s latest=%s' % (SYMBOLS[c], v['ok'], v['n_hist'], v['mean_rate'], v['latest_rate']))
    per_coin = {}
    for c in coins:
        v = venue[c]
        rmean = v['mean_rate'] if v['mean_rate'] is not None else FUND_ASSUMED
        rsrc = 'demo_mean' if v['mean_rate'] is not None else 'fallback_assumed'
        real_fnd = [poss[c][t] * rmean * LEV for t in range(n)]
        real_net = [gross[c][t] - turns[c][t] * FEE_USE * LEV - real_fnd[t] for t in range(n)]
        a_seg = seg(assumed_net[c], turns[c], 0, n)
        r_seg = seg(real_net, turns[c], 0, n)
        g_cum = round(sum(gross[c]), 6)
        a_fund_cum = round(sum(assumed_fnd[c]), 6)
        r_fund_cum = round(sum(real_fnd), 6)
        a_net_cum = round(sum(assumed_net[c]), 6)
        r_net_cum = round(sum(real_net), 6)
        d_sh = round(r_seg['sharpe'] - a_seg['sharpe'], 3)
        d_cum = round(r_net_cum - a_net_cum, 6)
        per_coin[c] = {'symbol': SYMBOLS[c], 'n': n,
                       'venue': {'ok': v['ok'], 'n_hist': v['n_hist'], 'mean_rate': v['mean_rate'],
                                 'latest_rate': v['latest_rate'], 'latest_ts': v['latest_ts'], 'error': v['error']},
                       'realized_rate_used': rmean, 'realized_rate_source': rsrc,
                       'rate_delta_vs_assumed': round(rmean - FUND_ASSUMED, 8),
                       'assumed': dict(a_seg, fund_cum=a_fund_cum, gross_cum=g_cum, net_cum=a_net_cum,
                                       share_net=safe_ratio(a_fund_cum, a_net_cum),
                                       share_gross=safe_ratio(a_fund_cum, g_cum)),
                       'realized': dict(r_seg, fund_cum=r_fund_cum, gross_cum=g_cum, net_cum=r_net_cum,
                                        share_net=safe_ratio(r_fund_cum, r_net_cum),
                                        share_gross=safe_ratio(r_fund_cum, g_cum)),
                       'delta': {'sharpe': d_sh, 'cum': d_cum, 'fund_cum': round(r_fund_cum - a_fund_cum, 6)}}
        dump({'status': 'partial', 'stage': 'eval_' + c, 'config': cfg,
              'venue': {k: venue[k] for k in sorted(venue)}, 'per_coin': per_coin})
        log('eval %s rmean=%.8f d_sh=%+.3f d_cum=%+.6f share_net_real=%s' % (c, rmean, d_sh, d_cum, per_coin[c]['realized']['share_net']))
    bw = 1.0 / len(coins)
    a_net = [sum(assumed_net[c][t] * bw for c in coins) for t in range(n)]
    a_fnd_b = [sum(assumed_fnd[c][t] * bw for c in coins) for t in range(n)]
    r_net = [sum((gross[c][t] - turns[c][t] * FEE_USE * LEV - poss[c][t] * (venue[c]['mean_rate'] if venue[c]['mean_rate'] is not None else FUND_ASSUMED) * LEV) * bw for c in coins) for t in range(n)]
    r_fnd_b = [sum(poss[c][t] * (venue[c]['mean_rate'] if venue[c]['mean_rate'] is not None else FUND_ASSUMED) * LEV * bw for c in coins) for t in range(n)]
    g_b = [sum(gross[c][t] * bw for c in coins) for t in range(n)]
    t_b = [sum(turns[c][t] * bw for c in coins) for t in range(n)]
    a_b = seg(a_net, t_b, 0, n)
    r_b = seg(r_net, t_b, 0, n)

    a_b_fund = round(sum(a_fnd_b), 6)
    r_b_fund = round(sum(r_fnd_b), 6)
    a_b_net = round(sum(a_net), 6)
    r_b_net = round(sum(r_net), 6)
    g_b_cum = round(sum(g_b), 6)
    basket = {'assumed': dict(a_b, fund_cum=a_b_fund, gross_cum=g_b_cum, net_cum=a_b_net,
                              share_net=safe_ratio(a_b_fund, a_b_net),
                              share_gross=safe_ratio(a_b_fund, g_b_cum)),
              'realized': dict(r_b, fund_cum=r_b_fund, gross_cum=g_b_cum, net_cum=r_b_net,
                               share_net=safe_ratio(r_b_fund, r_b_net),
                               share_gross=safe_ratio(r_b_fund, g_b_cum)),
              'delta': {'sharpe': round(r_b['sharpe'] - a_b['sharpe'], 3),
                        'cum': round(r_b_net - a_b_net, 6),
                        'fund_cum': round(r_b_fund - a_b_fund, 6)}}
    log('basket assumed sh=%.3f cum=%.4f fund=%.6f | realized sh=%.3f cum=%.4f fund=%.6f | d_sh=%+.3f' % (a_b['sharpe'], a_b_net, a_b_fund, r_b['sharpe'], r_b_net, r_b_fund, basket['delta']['sharpe']))
    res = {'status': 'final', 'config': cfg,
           'venue': {k: venue[k] for k in sorted(venue)},
           'per_coin': per_coin, 'basket': basket,
           'verdict': 'PENDING', 'decision': 'NO_ADOPTION',
           'decision_note': 'P0-3 permutation FAIL => H14 verdict PENDING; realized-funding diagnostic only, no adoption, live FUND/FEE/LEV untouched.',
           'conclusion': ('H14 1h Top5 realized-funding (bybit-demo public) vs assumed 0.0005: basket assumed sh=%.3f cum=%.4f fund=%.6f share_net=%s | realized sh=%.3f cum=%.4f fund=%.6f share_net=%s | d_sh=%+.3f d_cum=%+.6f. PENDING; no live change.' % (a_b['sharpe'], a_b_net, a_b_fund, basket['assumed']['share_net'], r_b['sharpe'], r_b_net, r_b_fund, basket['realized']['share_net'], basket['delta']['sharpe'], basket['delta']['cum']))}
    dump(res)
    log('wrote %s verdict=%s decision=%s' % (OUT, res['verdict'], res['decision']))


if __name__ == '__main__':
    main()
