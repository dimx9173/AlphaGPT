"""E12 Shadow stability re-verify: Y1b_main_thr1.0_w200 + AA challengers parallel shadow + fee2x worst.

Mirror research/run_e9.py + run_w3.py ledger+gate engine (import + reuse, no copy drift):
  per-trade netp = move*LEV - FEE*LEV*2 - FUND*LEV, 50/50 ETC/TRX
  gate: causal trailing-200 per-bar-net sharpe
    main = overlay active iff trailing-200 overlay sharpe > thresh
    alt  = overlay active iff trailing-200 plain sharpe < thresh
  thresh in [0.8,1.0,1.5] x win200 x overlays{Z1,AA-H1,AA-H2} x {main,alt} = 18
  plain = Y1b vtNone (ETC 0.88/0.12/cd18/None/ts24 + TRX 0.85/0.12/cd6/0.05/ts24, q0.3)
  Z1 = same legs + vt0.012 vw12
  AA-H1 = H1-best idx01 sth0.10/etc15/trx9 + vt0.012 vw12
  AA-H2 = H2-best idx13 sth0.12/etc15/trx9 + vt0.012 vw12
Fee2x stress: same gate switches (base-fee signals), ledger recomputed at FEE=0.0008;
  worstB/worstC = min(base_fee, fee2x) sharpe on B/C.
Segments: FULL[0:6580]/H2[6077:6570]/B[6380:6580]/C[6080:6580]
PASS iff B_sh>plain_B_sh AND C_sh>plain_C_sh AND FULL no collapse (sharpe>0.5 & mdd<1.0).
Output: results/results_E12_shadow.json + logs/e12.log
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if os.path.dirname(os.path.abspath(__file__)) else ".")
import pathlib as _pl
_root = _pl.Path(__file__).resolve().parent.parent if "research" in str(_pl.Path(__file__).resolve()) else _pl.Path(".")
os.chdir(str(_root))
import json
import research.run_e9 as E9

FEE2X = 0.0008

def log(msg):
    print(msg, flush=True)
    try:
        with open("logs/e12.log", "a") as f:
            f.write(msg + "\n")
    except:
        pass

def main():
    open("logs/e12.log", "w").write("")
    log("E12 start: re-verify Y1b_main_thr1.0_w200 + AA challengers + fee2x worst (mirror E9/W3 engine)")
    E9.FEE = 0.0004
    full = {c: E9.load_bars(c) for c in E9.COINS}
    n = min(len(b) for b in full.values())
    log(f"full 4h bars n={n}")
    assert n == 6580, f"expected 6580 got {n}"
    px = {c: [b[3] for b in full[c][:n]] for c in E9.COINS}

    specs = {'plain': E9.PLAIN_SPEC, **E9.OVERLAYS}
    nets = {}; poss = {}
    for name, spec in specs.items():
        nets[name] = {}; poss[name] = {}
        for c in E9.COINS:
            lth, sth, cd, sl, ts, side, vt, vw = spec[c]
            net, pos = E9.leg_net_pos(full[c][:n], lth, sth, cd, sl, ts, side, E9.Q, vt, vw)
            nets[name][c] = net; poss[name][c] = pos
        log(f"leg done {name} q={E9.Q}")

    def ledger_all(fee):
        E9.FEE = fee
        out = {}
        for name in specs:
            eq, led = E9.run_ledger(poss[name], px, n)
            st = E9.eq_stats(eq, led)
            st['segments'] = {k: E9.seg_stats(eq, led, a, b) for k, (a, b) in E9.SEGS.items()}
            st['eq'] = eq; st['ledger'] = led
            out[name] = st
        return out

    base = ledger_all(0.0004)
    for name, st in base.items():
        log(f"{name} BASE FULL final_x={st['final_x']} sharpe={st['sharpe']} mdd={st['mdd']} trades={st['trades']} by={st['by']} longs={st['longs']}")
        for k in ['FULL', 'H2', 'B', 'C']:
            s = st['segments'][k]
            log(f"  {k} x={s['x']} sh={s['sharpe']} mdd={s['mdd']} tr={s['trades']}")
    fee2 = ledger_all(FEE2X)
    for name, st in fee2.items():
        log(f"{name} FEE2X FULL final_x={st['final_x']} sharpe={st['sharpe']} mdd={st['mdd']} trades={st['trades']}")
        for k in ['B', 'C']:
            s = st['segments'][k]
            log(f"  fee2x {k} x={s['x']} sh={s['sharpe']} mdd={s['mdd']}")

    plain_B = base['plain']['segments']['B']
    plain_C = base['plain']['segments']['C']

    combo_plain = [0.5 * nets['plain']['ETC'][t] + 0.5 * nets['plain']['TRX'][t] for t in range(n)]
    combo_over = {ov: [0.5 * nets[ov]['ETC'][t] + 0.5 * nets[ov]['TRX'][t] for t in range(n)] for ov in E9.OVERLAYS}

    shadows = []
    for ov in ['Z1', 'AA-H1', 'AA-H2']:
        for thresh in E9.THRESHES:
            for gtype in ['main', 'alt']:
                gate = [False] * n
                for t in range(E9.GATE_WIN, n):
                    if gtype == 'main':
                        gate[t] = E9.trailing_sharpe(combo_over[ov], t, E9.GATE_WIN) > thresh
                    else:
                        gate[t] = E9.trailing_sharpe(combo_plain, t, E9.GATE_WIN) < thresh
                spos = {c: [poss[ov][c][t] if gate[t] else poss['plain'][c][t] for t in range(n)] for c in E9.COINS}
                E9.FEE = 0.0004
                eq, led = E9.run_ledger(spos, px, n)
                st = E9.eq_stats(eq, led)
                st['segments'] = {k: E9.seg_stats(eq, led, a, b) for k, (a, b) in E9.SEGS.items()}
                E9.FEE = FEE2X
                eq2, led2 = E9.run_ledger(spos, px, n)
                st2 = E9.eq_stats(eq2, led2)
                st2['segments'] = {k: E9.seg_stats(eq2, led2, a, b) for k, (a, b) in E9.SEGS.items()}
                E9.FEE = 0.0004
                st['coverage'] = round(sum(gate) / n, 4)
                st['overlay'] = ov; st['thresh'] = thresh; st['gate'] = gtype; st['window'] = E9.GATE_WIN
                st['id'] = f"{ov}_{gtype}_thr{thresh}_w{E9.GATE_WIN}"
                b_sh = st['segments']['B']['sharpe']; c_sh = st['segments']['C']['sharpe']
                full_ok = st['sharpe'] > 0.5 and st['mdd'] < 1.0
                st['pass'] = bool((b_sh > plain_B['sharpe']) and (c_sh > plain_C['sharpe']) and full_ok)
                st['pass_x'] = bool(st['segments']['B']['x'] > plain_B['x'] and st['segments']['C']['x'] > plain_C['x'] and full_ok)
                st['fee2x'] = {'final_x': st2['final_x'], 'sharpe': st2['sharpe'], 'mdd': st2['mdd'],
                               'B_sh': st2['segments']['B']['sharpe'], 'B_x': st2['segments']['B']['x'],
                               'C_sh': st2['segments']['C']['sharpe'], 'C_x': st2['segments']['C']['x']}
                st['worstB'] = round(min(b_sh, st2['segments']['B']['sharpe']), 3)
                st['worstC'] = round(min(c_sh, st2['segments']['C']['sharpe']), 3)
                shadows.append(st)
                log(f"shadow {st['id']} cov={st['coverage']} FULL x={st['final_x']} sh={st['sharpe']} mdd={st['mdd']} tr={st['trades']} | B sh={b_sh} (2x {st['fee2x']['B_sh']}) vs plain {plain_B['sharpe']}/{plain_B['x']} | C sh={c_sh} (2x {st['fee2x']['C_sh']}) vs {plain_C['sharpe']}/{plain_C['x']} | worstB={st['worstB']} worstC={st['worstC']} | pass={st['pass']}")

    pass_sh = [s for s in shadows if s['pass']]
    pass_x = [s for s in shadows if s['pass_x']]
    log(f"SUMMARY pass_sharpe={len(pass_sh)}/18 pass_x={len(pass_x)}/18")
    for s in shadows:
        log(f"  {s['id']} cov={s['coverage']} FULL sh={s['sharpe']} mdd={s['mdd']} B_sh={s['segments']['B']['sharpe']} vs plain {plain_B['sharpe']} C_sh={s['segments']['C']['sharpe']} vs {plain_C['sharpe']} worstB={s['worstB']} worstC={s['worstC']} pass={s['pass']}")

    best = None
    if pass_sh:
        best = sorted(pass_sh, key=lambda s: (s['mdd'], -s['coverage']))[0]
        log(f"BEST pass_sh {best['id']}")
    elif pass_x:
        best = sorted(pass_x, key=lambda s: (s['mdd'], -s['coverage']))[0]
        log(f"BEST pass_x (fallback) {best['id']}")

    def clean_ledger(name):
        b = base[name]; f = fee2[name]
        seg = {}
        for k in E9.SEGS:
            seg[k] = {kk: b['segments'][k][kk] for kk in ['x', 'sharpe', 'mdd', 'trades', 'by', 'longs']}
            seg[k]['fee2x_sh'] = f['segments'][k]['sharpe']
            seg[k]['fee2x_x'] = f['segments'][k]['x']
        return {'final_x': b['final_x'], 'sharpe': b['sharpe'], 'mdd': b['mdd'], 'trades': b['trades'],
                'by': b['by'], 'longs': b['longs'],
                'fee2x': {'final_x': f['final_x'], 'sharpe': f['sharpe'], 'mdd': f['mdd']},
                'worstB': round(min(b['segments']['B']['sharpe'], f['segments']['B']['sharpe']), 3),
                'worstC': round(min(b['segments']['C']['sharpe'], f['segments']['C']['sharpe']), 3),
                'segments': seg}

    clean_ledgers = {name: clean_ledger(name) for name in specs}
    clean_shadows = []
    for s in shadows:
        clean_shadows.append({'id': s['id'], 'overlay': s['overlay'], 'gate': s['gate'], 'window': s['window'],
            'thresh': s['thresh'], 'coverage': s['coverage'], 'final_x': s['final_x'], 'sharpe': s['sharpe'],
            'mdd': s['mdd'], 'trades': s['trades'], 'by': s['by'], 'longs': s['longs'],
            'segments': {k: {kk: s['segments'][k][kk] for kk in ['x', 'sharpe', 'mdd', 'trades', 'by', 'longs']} for k in E9.SEGS},
            'pass': s['pass'], 'pass_x': s['pass_x'], 'fee2x': s['fee2x'], 'worstB': s['worstB'], 'worstC': s['worstC'],
            'vs_plain': {'B_plain_sh': plain_B['sharpe'], 'B_sh': s['segments']['B']['sharpe'], 'B_plain_x': plain_B['x'], 'B_x': s['segments']['B']['x'],
                         'C_plain_sh': plain_C['sharpe'], 'C_sh': s['segments']['C']['sharpe'], 'C_plain_x': plain_C['x'], 'C_x': s['segments']['C']['x']}})

    heat = {}
    for ov in E9.OVERLAYS:
        heat[ov] = {}
        for gtype in ['main', 'alt']:
            heat[ov][gtype] = {str(th): next(s for s in clean_shadows if s['overlay'] == ov and s['gate'] == gtype and s['thresh'] == th) for th in E9.THRESHES}

    res = {
        'config': {'formula': E9.FORMULA, 'fee': 0.0004, 'fee2x': FEE2X, 'fund': E9.FUND, 'lev': E9.LEV, 'q': E9.Q,
                   'weights': E9.PORT, 'full_n': n, 'segments': {k: list(v) for k, v in E9.SEGS.items()},
                   'plain_spec': {c: list(E9.PLAIN_SPEC[c]) for c in E9.COINS},
                   'overlays': {k: {c: list(v[c]) for c in E9.COINS} for k, v in E9.OVERLAYS.items()},
                   'gate_window': E9.GATE_WIN, 'gate_threshes': E9.THRESHES,
                   'gate_rules': {'main': 'overlay active iff trailing-200 overlay per-bar-net sharpe > thresh',
                                  'alt': 'overlay active iff trailing-200 plain sharpe < thresh'},
                   'fee2x_note': 'same gate switches (base-fee signals); ledger recomputed at fee=0.0008; worst=min(base,fee2x) sharpe',
                   'note': 'E12 re-verify Y1b_main_thr1.0_w200 + AA challengers parallel shadow, mirror E9/W3 ledger+gate engine.'},
        'ledgers': clean_ledgers,
        'shadows': clean_shadows,
        'heat': {ov: {gtype: {str(th): {'coverage': heat[ov][gtype][str(th)]['coverage'], 'sharpe': heat[ov][gtype][str(th)]['sharpe'],
                                        'mdd': heat[ov][gtype][str(th)]['mdd'], 'final_x': heat[ov][gtype][str(th)]['final_x'],
                                        'B_sh': heat[ov][gtype][str(th)]['segments']['B']['sharpe'],
                                        'C_sh': heat[ov][gtype][str(th)]['segments']['C']['sharpe'],
                                        'worstB': heat[ov][gtype][str(th)]['worstB'], 'worstC': heat[ov][gtype][str(th)]['worstC'],
                                        'fee2x': heat[ov][gtype][str(th)]['fee2x'],
                                        'pass': heat[ov][gtype][str(th)]['pass']} for th in E9.THRESHES} for gtype in ['main', 'alt']} for ov in E9.OVERLAYS},
        'summary': {'total': 18, 'pass_sharpe': len(pass_sh), 'pass_x': len(pass_x),
                    'pass_ids_sharpe': [s['id'] for s in pass_sh], 'pass_ids_x': [s['id'] for s in pass_x],
                    'plain_B': {kk: plain_B[kk] for kk in ['x', 'sharpe', 'mdd', 'trades', 'by', 'longs']},
                    'plain_C': {kk: plain_C[kk] for kk in ['x', 'sharpe', 'mdd', 'trades', 'by', 'longs']},
                    'plain_FULL': {'final_x': base['plain']['final_x'], 'sharpe': base['plain']['sharpe'], 'mdd': base['plain']['mdd']}},
        'best': ({'id': best['id'], 'overlay': best['overlay'], 'gate': best['gate'], 'thresh': best['thresh'], 'window': best['window'],
                  'coverage': best['coverage'], 'sharpe': best['sharpe'], 'mdd': best['mdd'], 'final_x': best['final_x'],
                  'B_sh': best['segments']['B']['sharpe'], 'C_sh': best['segments']['C']['sharpe'],
                  'B_x': best['segments']['B']['x'], 'C_x': best['segments']['C']['x'],
                  'worstB': best['worstB'], 'worstC': best['worstC']} if best else None),
        'decision': ('PASS adopt shadow ' + best['id'] if best and best['pass'] else ('REJECT no shadow passes B>C and full no collapse (keep plain)' if not pass_sh else 'REJECT')),
    }
    res['verdict'] = res['decision']
    open('results/results_E12_shadow.json', 'w').write(json.dumps(res, indent=2, ensure_ascii=False))
    log(f"saved results/results_E12_shadow.json best={best['id'] if best else None} decision={res['decision']}")
    log(f"FULL ledgers base: plain={base['plain']['final_x']}/{base['plain']['sharpe']} Z1={base['Z1']['final_x']}/{base['Z1']['sharpe']} AA-H1={base['AA-H1']['final_x']}/{base['AA-H1']['sharpe']} AA-H2={base['AA-H2']['final_x']}/{base['AA-H2']['sharpe']}")

if __name__ == '__main__':
    main()
