#!/usr/bin/env python3
"""Free diagnostics C1 + timescale sweep on the stored S0 batch.

Zero GA simulations. Recomputes pnet per (genome, fold) from the stored
champion formula, then scores S0/A1/FM/N0 fitness WITHOUT re-running
evolution. C1 re-ranks the 160 visited genomes -- approximated here by the
stored champion per fold plus S0 score; full 160 needs evolution logs which
are not stored, so C1 here = final-only re-rank across folds.

Usage:
  python3 tools/selection_free_diag_28c.py --dir results/f1h41_iter1000 --features 41 --limit 20
"""
from __future__ import annotations
import argparse, glob, json, math, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.ga_28c_30m_3y import (
    formula_signal, smooth_causal, POSITION_CAP, FEE, LEV, BAR_SIZES,
)
from research.universe_28c import COINS_28C
from research.accounting_28c import metrics, accounting_bar_returns, load_real_funding


def build_positions(formula, maps, returns, start, end, scale_end, bars, funding_by_coin, grammar='reduced'):
    leg_net = []
    for c in COINS_28C:
        sig = formula_signal(formula, maps, c, grammar=grammar)
        fit_scale = float(np.std(sig[:scale_end])) if scale_end > 0 else 0.0
        raw_pos = np.tanh(sig / (fit_scale + 1e-6)) if fit_scale > 1e-8 else np.zeros_like(sig)
        p = 0.25 * smooth_causal(raw_pos, 5)
        p = np.roll(p, 1); p[0] = 0
        r = returns[c][start:end]
        pos = p[start:end]
        turn = np.abs(np.diff(pos, prepend=0.0))
        fnd = np.asarray(funding_by_coin[c], dtype=np.float64)[start:end]
        net = accounting_bar_returns(pos, r, FEE, fnd, LEV)
        leg_net.append(net)
    pnet = np.mean(np.stack(leg_net), axis=0)
    return pnet


def sharpe_of(series, timestamps):
    try:
        return metrics(np.asarray(series, dtype=np.float64), timestamps)['sharpe']
    except Exception:
        return float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--features', type=int, default=41)
    ap.add_argument('--limit', type=int, default=0, help='0 = all seeds')
    ap.add_argument('--bars', default='1h', choices=['1h', '30m', '4h'])
    ap.add_argument('--scale-windows', action='store_true')
    args = ap.parse_args()

    import research.ga_28c_30m_3y as G
    G.BAR = args.bars
    G.DATA_DIR = G.ROOT / BAR_SIZES[args.bars]['data']
    G.CACHE_DIR = G.ROOT / BAR_SIZES[args.bars]['cache']
    G.N_FEATURES = args.features
    G.BPY = BAR_SIZES[args.bars]['bpy']
    if args.features in (36, 41):
        import research.grammar_reduced_36c as gr
        gr.with_feature_count(args.features)
        G.REDUCED_ALL_TOKENS = gr.REDUCED_TOKENS
        G.REDUCED_OP_ARITY = gr.REDUCED_ARITY
        G.FEATURE_COUNT = args.features
    ws = 0.5 if args.scale_windows else 1.0
    import research.causal_36f as _C36
    _C36.set_window_scale(ws)
    from research.splits_28c import configure as _cfg
    _cfg(step_ms=BAR_SIZES[args.bars]['step_ms'],
         expected_bars=G._expected_bars(), embargo_bars=G._embargo_bars())
    G.GRAMMAR = 'reduced'

    common, maps, returns, funding_mask = G.load_data(null='none')
    funding_by_coin = {c: load_real_funding(c, np.asarray(common, dtype=np.int64)) for c in COINS_28C}
    ts = np.asarray(common)

    files = sorted(glob.glob(str(Path(args.dir) / 'g10_real_s*.json')))
    if args.limit:
        files = files[:args.limit]
    print(f'{len(files)} real files', flush=True)

    # Per-seed: recompute train-window pnet per selection fold, score rules
    c1_rows = []
    ts_rows = []  # (seed, split_frac, agree, rankcorr)
    for fi, f in enumerate(files):
        j = json.loads(Path(f).read_text())
        if j.get('n_features') != args.features:
            continue
        seed = j.get('seed')
        for fr in j.get('fold_records', []):
            if fr.get('is_holdout'):
                continue
            tr_s, tr_e = fr['train']
            formula = fr['formula']
            T = tr_e - tr_s
            try:
                pnet = build_positions(formula, maps, returns, tr_s, tr_e,
                                       fr['train'][1], ts, funding_by_coin)
            except Exception as e:
                print(f'seed {seed} fold {fr["fold"]}: rebuild FAIL {e}', flush=True)
                continue
            # S0 fitness proxy: portfolio sharpe on train
            s_full = sharpe_of(pnet, ts[tr_s:tr_e])
            # A1 halves-min
            m = T // 2
            s_a = sharpe_of(pnet[:m], ts[tr_s:tr_s + m])
            s_b = sharpe_of(pnet[m:], ts[tr_s + m:tr_e])
            a1 = min(s_a, s_b) if np.isfinite(s_a) and np.isfinite(s_b) else -10.0
            # N0 odd/even min
            s_o = sharpe_of(pnet[0::2], ts[tr_s:tr_e][0::2])
            s_e = sharpe_of(pnet[1::2], ts[tr_s:tr_e][1::2])
            n0 = min(s_o, s_e) if np.isfinite(s_o) and np.isfinite(s_e) else -10.0
            # timescale sweep splits
            for frac in (1/3, 1/2, 2/3):
                k = int(T * frac)
                x = sharpe_of(pnet[:k], ts[tr_s:tr_s + k])
                y = sharpe_of(pnet[k:], ts[tr_s + k:tr_e])
                ts_rows.append((seed, fr['fold'], frac, x, y))
            c1_rows.append({'seed': seed, 'fold': fr['fold'], 's0': s_full, 'a1': a1, 'n0': n0,
                            'stored_train_sharpe': fr.get('train_sharpe')})
        if (fi + 1) % 10 == 0:
            print(f'  {fi+1}/{len(files)}', flush=True)

    import statistics as st
    print(f'\nC1 rows: {len(c1_rows)}')
    for key in ('s0', 'a1', 'n0'):
        xs = [r[key] for r in c1_rows if r[key] is not None and math.isfinite(r[key])]
        if xs:
            print(f'  {key}: mean {st.mean(xs):+.3f} med {st.median(xs):+.3f} n={len(xs)}')
    # S0 rebuild fidelity: recomputed vs stored train sharpe
    pairs = [(r['s0'], r['stored_train_sharpe']) for r in c1_rows
             if r['s0'] is not None and math.isfinite(r['s0'])
             and r['stored_train_sharpe'] is not None]
    if pairs:
        a = np.array([p[0] for p in pairs]); b = np.array([p[1] for p in pairs])
        print(f'  rebuild fidelity: corr(recomputed,stored) = {np.corrcoef(a,b)[0,1]:.4f} n={len(pairs)}')
    # timescale sweep: rank corr first-half vs second-half per frac
    print('\nTimescale sweep (first-half vs second-half Sharpe rank corr):')
    for frac in (1/3, 1/2, 2/3):
        sub = [(x, y) for (s, f, fr2, x, y) in ts_rows if fr2 == frac
               and math.isfinite(x) and math.isfinite(y)]
        if len(sub) > 5:
            from scipy import stats as sps
            r, p = sps.spearmanr([t[0] for t in sub], [t[1] for t in sub])
            print(f'  frac={frac:.2f}: spearman r={r:+.3f} p={p:.3g} n={len(sub)}')


if __name__ == '__main__':
    main()
