"""R3 verification: Q2 candidate H1/H2 calendar purity + turnover/tradability audit.
Engine-exact convention (matches run_q2 / backtest_Q2_train.json):
features built on OOS 986 bars, FULL=OOS-full, H1/H2=493/493 halves.
Supplemental FULL6570 = features rebuilt on full 6570-bar sample (labeled).
Candidate [0,20,18,14,20,12,14,18,11,14,14,14], default 0.85/0.15/cd6,
aster perp 2x fund 0.0005 fee 0.0004. Baseline [3,2,7,2,7,11,15,4,4,6,6,10].
Output: backtest_R3_verify.json
"""
import json, math, csv, datetime, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import run_q2 as Q
from model_core.config import ModelConfig
from model_core.vm import StackVM
from model_core.backtest import MemeBacktest

CAND = [0,20,18,14,20,12,14,18,11,14,14,14]
BASE = [3,2,7,2,7,11,15,4,4,6,6,10]
COINS = ['BTC','SOL','ETC','TRX','DOGE']
LTH, STH, CD = 0.85, 0.15, 6
BPY = 2190.0
TRX_SWEEP = [0.80, 0.85, 0.88]
vm = StackVM()
device = ModelConfig.DEVICE

def iso(ms):
    return datetime.datetime.fromtimestamp(ms/1000, datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

def exec_positions(feats_slice, raw_slice, lth, sth, cd):
    """Mirror MemeBacktest.evaluate position path; return lp, sp lists + signal."""
    bt = MemeBacktest(venue='aster', leverage=2.0, short_enabled=True,
                      funding_override=0.0005, long_th=lth, short_th=sth,
                      cooldown_bars=cd, bars_per_year=BPY)
    fd = feats_slice.to(device)
    res = vm.execute(CAND if False else feats_slice_formula, fd)  # placeholder
    return None

# NOTE: positions must be derived per-formula; use explicit function below.
def positions_for(formula, feats_slice, raw_slice, lth, sth, cd):
    from model_core.factors import FeatureEngineer  # noqa (keep import surface stable)
    bt = MemeBacktest(venue='aster', leverage=2.0, short_enabled=True,
                      funding_override=0.0005, long_th=lth, short_th=sth,
                      cooldown_bars=cd, bars_per_year=BPY)
    res = vm.execute(formula, feats_slice.to(device))
    signal = torch.sigmoid(res)
    liq = raw_slice['liquidity'].to(device)
    is_safe = (liq > bt.min_liq).float()
    lp = (signal > bt.long_th).float() * is_safe
    sp = (signal < bt.short_th).float() * is_safe
    lp, sp = bt._apply_cooldown(lp, sp)
    lp = lp.roll(1, dims=1); lp[:, 0] = 0
    sp = sp.roll(1, dims=1); sp[:, 0] = 0
    return lp[0].tolist(), sp[0].tolist(), signal[0].tolist()

def diag(pos):
    n = len(pos)
    d = [abs(pos[t]-pos[t-1]) for t in range(1, n)]
    n_change = sum(1 for v in d if v > 0)
    entries = (1 if pos[0] != 0 else 0) + sum(1 for t in range(1, n) if pos[t] != 0 and pos[t] != pos[t-1])
    nz = sum(1 for v in pos if v != 0)
    return {'n': n, 'mean_abs_dpos': round(sum(d)/len(d), 5) if d else 0.0,
            'n_changes': n_change, 'flips_yr': round(n_change/n*BPY, 1),
            'entries': entries, 'nonzero_bars': nz,
            'long_bars': sum(1 for v in pos if v > 0),
            'short_bars': sum(1 for v in pos if v < 0),
            'avg_hold_bars': round(nz/entries, 2) if entries else 0.0}

def leg_diag(lp, sp):
    def one(s):
        e = (1 if s[0] > 0 else 0) + sum(1 for t in range(1, len(s)) if s[t] > 0 and s[t-1] == 0)
        nz = sum(1 for v in s if v > 0)
        return {'entries': e, 'bars': nz, 'avg_hold_bars': round(nz/e, 2) if e else 0.0}
    return {'long': one(lp), 'short': one(sp)}

def cut2(t, s, e):
    return t[:, s:e] if t.dim() == 2 else t[:, :, s:e]

def eng_perf(formula, feats, tgt, raw, lth, sth, cd, sl, s, e, side='both'):
    return Q.eval_segs(vm, formula, feats, tgt, raw, device, lth, sth, cd, sl,
                       side=side, segs=('X',)) if False else None

out = {'candidate': CAND, 'baseline': BASE,
       'params': {'lth': LTH, 'sth': STH, 'cd': CD, 'venue': 'aster',
                  'lev': 2.0, 'fund': 0.0005, 'fee': 0.0004,
                  'convention': 'engine-exact: feats on OOS986, FULL=OOS, H1/H2=493/493'},
       'gates': {'max_flips_yr': 300, 'min_hold_bars': 4, 'min_entries': 5},
       'calendar': {}, 'coins': {}, 'trx': {}}

# ---- calendar from raw CSVs ----
cal_ref = None
for coin in COINS:
    with open(f'data/data_15m_3y/{coin}.csv') as f:
        rows = list(csv.DictReader(f))
    ts = []
    for i in range(0, len(rows), 16):
        blk = rows[i:i+16]
        if len(blk) < 16: break
        ts.append((int(blk[0]['timestamp']), int(blk[-1]['timestamp'])))
    n = len(ts); cut = int(n*0.85); h = (n-cut)//2
    segs = {'FULL6570': (0, n), 'TRAIN': (0, cut), 'OOS_FULL': (cut, n),
            'H1': (cut, cut+h), 'H2': (cut+h, n)}
    c = {'n_4h': n, 'cut': cut, 'h': h, 'segs': {}}
    for k, (s, e) in segs.items():
        c['segs'][k] = {'idx': [s, e], 'start': iso(ts[s][0]), 'end': iso(ts[e-1][1])}
    out['calendar'][coin] = c
    key = json.dumps(c['segs'])
    if cal_ref is None: cal_ref = key
    c['aligned'] = (key == cal_ref)
out['calendar_aligned_all'] = all(v['aligned'] for v in out['calendar'].values())
print('calendar aligned:', out['calendar_aligned_all'], flush=True)
for coin in COINS:
    s = out['calendar'][coin]['segs']
    print(f"{coin} FULL6570 {s['FULL6570']['start']}->{s['FULL6570']['end']} "
          f"H1 {s['H1']['start']}->{s['H1']['end']} H2 {s['H2']['start']}->{s['H2']['end']}", flush=True)

# ---- engine-exact audit ----
all_bars = Q.load_4h(tuple(COINS))
for coin in COINS:
    bars = all_bars[coin]
    nn = len(bars); no = nn - int(nn*Q.TRAIN_FRAC)
    oos = bars[nn-no:]
    feats, tgt, raw = Q.build_single(oos)
    T = feats.shape[2] if feats.dim() == 3 else feats.shape[1]
    h = T//2
    bounds = {'FULL': (0, T), 'H1': (0, h), 'H2': (h, T)}
    ce = {}
    for tag, form in [('cand', CAND), ('base', BASE)]:
        te = {}
        for seg, (s, e) in bounds.items():
            if seg == 'H1' and tag in ('cand', 'base'):
                pass
            fsl = cut2(feats, s, e); tsl = cut2(tgt, s, e)
            rsl = {k: cut2(v, s, e) for k, v in raw.items()}
            # engine perf (single-seg evaluate -> same as eval_segs slice)
            bt = MemeBacktest(venue='aster', leverage=2.0, short_enabled=True,
                              funding_override=0.0005, long_th=LTH, short_th=STH,
                              cooldown_bars=CD, bars_per_year=BPY)
            res = vm.execute(form, fsl.to(device))
            rd = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in rsl.items()}
            fit, cum = bt.evaluate(res, rd, tsl.to(device))
            m = dict(bt.last_metrics)
            wlen = e - s
            ann = float(cum)*BPY/wlen
            perf = {'fitness': round(float(fit.item()) if isinstance(fit, torch.Tensor) else float(fit), 4),
                    'sharpe': round(m['sharpe'], 3), 'ann': round(ann, 4),
                    'mdd': round(m['max_dd'], 4),
                    'cum': round(float(cum), 4), 'n': wlen}
            lp, sp, sigv = positions_for(form, fsl, rsl, LTH, STH, CD)
            pos = [a-b for a, b in zip(lp, sp)]
            d = diag(pos); legs = leg_diag(lp, sp)
            d['pass_turnover'] = bool(d['flips_yr'] < 300 and (d['avg_hold_bars'] >= 4 or d['entries'] == 0))
            legs['long']['reject_near_flat'] = bool(legs['long']['entries'] < 5)
            legs['short']['reject_near_flat'] = bool(legs['short']['entries'] < 5)
            te[seg] = {'perf': perf, 'turnover': d, 'legs': legs}
            if seg in ('H2', 'FULL'):
                print(f"{coin} {tag} {seg} sh={perf['sharpe']} ann={perf['ann']} cum={perf['cum']} "
                      f"flips/yr={d['flips_yr']} entries={d['entries']} hold={d['avg_hold_bars']} "
                      f"L={legs['long']['entries']}/{legs['long']['bars']} S={legs['short']['entries']}/{legs['short']['bars']}", flush=True)
        ce[tag] = te
    # engine-exact signal-fire stats per segment (pre-cooldown, pre-lag raw signal)
    fire = {}
    res_full = vm.execute(CAND, feats.to(device))
    sigv = torch.sigmoid(res_full)[0].tolist()
    for seg, (s, e) in bounds.items():
        v = sigv[s:e]
        fire[seg] = {'max_sig': round(max(v), 4), 'mean_sig': round(sum(v)/len(v), 4),
                     'frac_gt_085': round(sum(1 for x in v if x > 0.85)/len(v), 4),
                     'frac_lt_015': round(sum(1 for x in v if x < 0.15)/len(v), 4)}
    ce['cand_signal_fire'] = fire
    print(coin, 'cand fire:', json.dumps(fire), flush=True)
    out['coins'][coin] = ce

# ---- TRX lth sweep on H2 (engine-exact slice) ----
bars = all_bars['TRX']; nn = len(bars); no = nn - int(nn*Q.TRAIN_FRAC)
feats, tgt, raw = Q.build_single(bars[nn-no:])
T = feats.shape[2]; h = T//2
s, e = h, T
fsl = cut2(feats, s, e); tsl = cut2(tgt, s, e); rsl = {k: cut2(v, s, e) for k, v in raw.items()}
res_sig = torch.sigmoid(vm.execute(CAND, fsl.to(device)))[0].tolist()
sweep = []
for lth in TRX_SWEEP:
    bt = MemeBacktest(venue='aster', leverage=2.0, short_enabled=True,
                      funding_override=0.0005, long_th=lth, short_th=STH,
                      cooldown_bars=CD, bars_per_year=BPY)
    res = vm.execute(CAND, fsl.to(device))
    rd = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in rsl.items()}
    fit, cum = bt.evaluate(res, rd, tsl.to(device))
    m = dict(bt.last_metrics)
    perf = {'fitness': round(float(fit), 4), 'sharpe': round(m['sharpe'], 3),
            'ann': round(float(cum)*BPY/(e-s), 4), 'mdd': round(m['max_dd'], 4),
            'cum': round(float(cum), 4), 'n': e-s}
    lp, sp, _ = positions_for(CAND, fsl, rsl, lth, STH, CD)
    pos = [a-b for a, b in zip(lp, sp)]
    d = diag(pos); legs = leg_diag(lp, sp)
    fire = {'frac_gt_lth': round(sum(1 for x in res_sig if x > lth)/len(res_sig), 4),
            'max_sig': round(max(res_sig), 4)}
    sweep.append({'lth': lth, 'perf': perf, 'turnover': d, 'legs': legs, 'fire': fire})
    print(f"TRX H2 lth={lth} sh={perf['sharpe']} ann={perf['ann']} cum={perf['cum']} "
          f"entries={d['entries']} flips/yr={d['flips_yr']} legs=L{legs['long']} S{legs['short']} fire={fire}", flush=True)
out['trx']['h2_lth_sweep'] = sweep

# ---- supplemental FULL6570 rebuild (different convention, labeled) ----
from model_core.factors import FeatureEngineer
supp = {}
for coin in COINS:
    with open(f'data/data_15m_3y/{coin}.csv') as f:
        rows = list(csv.DictReader(f))
    bars = []
    for i in range(0, len(rows), 16):
        blk = rows[i:i+16]
        if len(blk) < 16: break
        bars.append({'close': float(blk[-1]['close']), 'open': float(blk[0]['open']),
                     'high': max(float(x['high']) for x in blk),
                     'low': min(float(x['low']) for x in blk),
                     'volume': sum(float(x['volume']) for x in blk)})
    n = len(bars)
    rawm = {'open': torch.tensor([[b['open'] for b in bars]]),
            'high': torch.tensor([[b['high'] for b in bars]]),
            'low': torch.tensor([[b['low'] for b in bars]]),
            'close': torch.tensor([[b['close'] for b in bars]]),
            'volume': torch.tensor([[b['volume'] for b in bars]]),
            'liquidity': torch.full((1, n), 1e7), 'fdv': torch.full((1, n), 1e8)}
    feats = FeatureEngineer.compute_features(rawm)
    rets = torch.tensor([[(bars[i+1]['close']-bars[i]['close'])/bars[i]['close'] if i < n-1 else 0.0 for i in range(n)]])
    row = {}
    for tag, form in [('cand', CAND), ('base', BASE)]:
        bt = MemeBacktest(venue='aster', leverage=2.0, short_enabled=True,
                          funding_override=0.0005, long_th=LTH, short_th=STH,
                          cooldown_bars=CD, bars_per_year=BPY)
        res = vm.execute(form, feats.to(device))
        rd = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in rawm.items()}
        fit, cum = bt.evaluate(res, rd, rets.to(device))
        m = dict(bt.last_metrics)
        row[tag] = {'sharpe': round(m['sharpe'], 3), 'ann': round(float(cum)*BPY/n, 4),
                    'mdd': round(m['max_dd'], 4), 'cum': round(float(cum), 4), 'n': n}
    supp[coin] = row
    print(f"FULL6570 {coin} cand={row['cand']} base={row['base']}", flush=True)
out['supplemental_FULL6570_rebuild'] = {'note': 'NOT Q2 convention; feats rebuilt on full 6570 bars', 'coins': supp}
json.dump(out, open('results/backtest_R3_verify.json', 'w'), indent=1)
print('wrote backtest_R3_verify.json', flush=True)
