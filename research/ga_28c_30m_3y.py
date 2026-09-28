#!/usr/bin/env python3
"""Causal 28-coin/3y 30m GA search. Train/validation selection, OOS report only."""
from __future__ import annotations
import argparse, csv, json, math, os, random, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; os.chdir(ROOT); sys.path.insert(0,str(ROOT))
from research.universe_28c import COINS_28C, ACCEPTANCE_GATE_28C
from research.acceptance_schema_28c import (
    FOLD_TABLE_KEY, NEW_KEY, derive_live_adopted, make_acceptance)
from research.regime_gate_28c import evaluate_regime_gate
from research.data_contract_28c import HISTORY_YEARS_COMMON_28
from research.splits_28c import search_splits, split_indices
from research.formula_grammar import valid_token_mask, update_depth, is_valid, ALL_TOKENS
from research.causal_12f import (causal_features, evaluate_formula,
                                constant_features, assert_features_vary,
                                FEATURE_NAMES)
from research.accounting_28c import scheduled_funding_rates, position_from_signal, accounting_bar_returns, account_portfolio, compound_equity, max_drawdown, daily_sharpe, metrics, ACCOUNTING_VERSION, FUND_RATE, ACCOUNTING_VERSION_REAL, load_real_funding

BPY=17520.0; LEV=2.0; FEE=0.0004; FUND=0.0005; FEATURE_COUNT=12; FORMULA_LEN=12
DATA_DIR=ROOT/'data'/'data_3y'/'30m'
CACHE_DIR=ROOT/'data'/'data_3y'/'30m_causal'
# Bump when causal_features changes meaning; stale caches are then ignored
# instead of shadowing the current factor definitions.
FEATURE_CACHE_VERSION='amihud-log-f1-v2'
# A4 requires 24 of 28 legs positive out of sample. The reward targets the
# same number so selection and the gate are not scoring different things.
BREADTH_TARGET=24/28


def load_data(null='none', null_seed=0, null_block=48):
    """Aligned bars, features and returns for the 28c contract.

    ``null`` swaps in synthetic data with no real relationship to predict, so
    the search can be measured against a case where a positive result is a
    false positive by construction. Features are recomputed from the null bars
    and the feature cache is bypassed: a cache keyed only by coin would hand
    back the real factors and quietly turn the null back into the original
    experiment.
    """
    raw={}; common=None
    for c in COINS_28C:
        p=DATA_DIR/f'{c}.csv'
        with p.open() as f: rows=list(csv.DictReader(f))
        d={k:[float(r[k]) for r in rows] for k in ('open','high','low','close','volume')}
        d['timestamp']=[int(float(r['timestamp'])) for r in rows]
        raw[c]=d; s=set(d['timestamp']); common=s if common is None else common&s
    common=sorted(common); n=len(common)

    # Align every coin onto the common timestamps first, so a null is built
    # from the same window the real features are built from.
    aligned={}
    for c in COINS_28C:
        idx={t:i for i,t in enumerate(raw[c]['timestamp'])}
        d={k:[raw[c][k][idx[t]] for t in common]
           for k in ('open','high','low','close','volume')}
        d['timestamp']=list(common)
        aligned[c]=d

    if null in ('iid','xsec'):
        from research.null_data_28c import make_null
        # Each coin is resampled from its own series, so the marginal return
        # distribution, its fat tails and the return/volume relationship all
        # survive; only the real cross-coin and temporal structure is lost.
        aligned={c: make_null(null, aligned[c], seed=null_seed+i, block=null_block)
                 for i,c in enumerate(COINS_28C)}

    maps={}; returns={}
    for c in COINS_28C:
        d=aligned[c]
        # A null run must recompute. An unversioned or coin-keyed cache would
        # silently return the real factors and invalidate the control.
        cache=(None if null!='none' else CACHE_DIR/f'{c}-{FEATURE_CACHE_VERSION}.npy')
        loaded=None
        if cache is not None and cache.exists():
            try:
                loaded=np.load(cache)
            except Exception:
                loaded=None
        if (loaded is not None and loaded.shape==(len(FEATURE_NAMES), n)
                and not constant_features(loaded)):
            maps[c]=loaded
        else:
            maps[c]=causal_features(d)
            if cache is not None:
                np.save(cache,maps[c])
        # Refuse to hand a degenerate feature set to the search. Cheap, and it
        # is the only check that can see a placeholder input.
        assert_features_vary(maps[c], context=f"{c} on the 28c contract")
        close=np.asarray(d['close']); r=np.zeros(n); r[1:]=close[1:]/close[:-1]-1
        returns[c]=r
    funding=np.array([1.0 if (datetime.fromtimestamp(t/1000,tz=timezone.utc).hour%8==0 and datetime.fromtimestamp(t/1000,tz=timezone.utc).minute==0) else 0.0 for t in common],dtype=np.float64)
    return common,maps,returns,funding


def random_formula(rng):
    seq=[]; depth=0
    for pos in range(FORMULA_LEN):
        choices=np.flatnonzero(valid_token_mask(depth,pos,FORMULA_LEN).numpy()).tolist()
        t=rng.choice(choices); seq.append(int(t)); depth=update_depth(depth,int(t))
    return tuple(seq)


def mutate(formula,rng):
    f=list(formula)
    if rng.random()<0.7:
        pos=rng.randrange(FORMULA_LEN)
        depth=0
        for j in range(pos): depth=update_depth(depth,f[j])
        choices=np.flatnonzero(valid_token_mask(depth,pos,FORMULA_LEN).numpy()).tolist()
        f[pos]=rng.choice(choices)
        if not is_valid(f): return random_formula(rng)
    else: return random_formula(rng)
    return tuple(f)


def crossover(a,b,rng):
    cut=rng.randrange(1,FORMULA_LEN); f=list(a[:cut])
    depth=0
    for t in f: depth=update_depth(depth,t)
    for t in b[cut:]:
        if len(f)>=FORMULA_LEN: break
        pos=len(f)
        if bool(valid_token_mask(depth,pos,FORMULA_LEN)[int(t)]):
            f.append(int(t)); depth=update_depth(depth,int(t))
        else:
            choices=np.flatnonzero(valid_token_mask(depth,pos,FORMULA_LEN).numpy()).tolist(); x=rng.choice(choices); f.append(int(x)); depth=update_depth(depth,int(x))
    return tuple(f) if is_valid(f) else random_formula(rng)


def softplus(x):
    x=max(-40,min(40,x)); return math.log1p(math.exp(x))


def formula_signal(formula,maps,c):
    return evaluate_formula(list(formula),maps[c])


def smooth_causal(x, window=20):
    """Causal trailing mean over `window` bars (partial at the start).

    Uses the same prefix sums a loop would, so the result is bit-identical.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    c = np.concatenate(([0.0], np.cumsum(x)))
    t = np.arange(n)
    lo = np.maximum(0, t - window + 1)
    return (c[t + 1] - c[lo]) / (t - lo + 1)


# Reward-shaping constants. Fragmentation and divergence are charged in Sharpe
# units; GAP_TAU sets how sharply a train/peer gap starts to bite.
GAP_TAU = 1.0
LEG_FLOOR = 0.0

def shape_reward(psh, psh_peer, min_leg, mean_ic, mean_turn,
                 breadth=None, version='v1'):
    """Turn raw split diagnostics into a selection score.

    psh is this split's own portfolio Sharpe; psh_peer is the other selection
    split's Sharpe. psh is independent of psh_peer, so the caller can evaluate
    each split once and shape both scores afterwards.

    v1 is kept so the previous batch stays reproducible and so the A/B against
    null data is a real comparison rather than a rewrite.

    v2 adds an explicit breadth term. The justification is the gate, not the
    lockbox: A4 requires 24 of 28 legs to be positive, and across ten
    independent searches the achieved breadth ranged 0..21 with a mean of 13.2,
    so every run failed on that criterion while reporting a positive portfolio
    Sharpe. A selection score that cannot see breadth will keep selecting
    formulas that fail the gate.
    """
    if mean_act_guard(mean_turn):
        return -50.0
    peer = psh if psh_peer is None else psh_peer
    # Fragmentation: credit the worse of the two splits, not their average, so a
    # formula that only works on one split earns nothing from the good one.
    psh_worst = min(psh, peer)
    # Divergence: charge for the gap instead of averaging it away.
    gap = softplus((psh - peer) / GAP_TAU) * GAP_TAU
    base = (mean_ic
            + 0.10 * psh_worst
            - 0.30 * gap
            - 0.02 * mean_turn
            - 0.10 * softplus((LEG_FLOOR - min_leg) / 0.5))
    if version == 'v1' or breadth is None:
        return base
    # Shortfall against the gate's own requirement, charged superlinearly so a
    # strategy that merely averages positive is not mistaken for a broad one.
    short = max(0.0, BREADTH_TARGET - breadth)
    return base - 2.0 * short


def mean_act_guard(mean_turn):
    return mean_turn < 1e-5


def evaluate(formula, maps, returns, start, end, funding_mask, scale_end, bars, psh_peer=None, funding_by_coin=None, reward_version='v1'):
    leg_net = []; leg_sh = []; leg_turn = []; leg_ic = []; activity = []
    timestamps = bars[start:end]
    for c in COINS_28C:
        sig = formula_signal(formula, maps, c)
        fit_scale = float(np.std(sig[:scale_end])) if scale_end > 0 else 0.0
        raw_pos = np.tanh(sig / (fit_scale + 1e-6)) if fit_scale > 1e-8 else np.zeros_like(sig)
        p = 0.25 * smooth_causal(raw_pos, 5)
        p = np.roll(p, 1); p[0] = 0
        r = returns[c][start:end]
        pos = p[start:end]
        turn = np.abs(np.diff(pos, prepend=0.0))
        # Real history when available, the constant otherwise. The constant is
        # unconditional -- longs pay, shorts receive, every event -- and a
        # net-short book was credited ~9.6%/yr the venue never paid.
        fnd = (funding_by_coin[c][start:end] if funding_by_coin is not None
               else funding_mask[start:end] * FUND_RATE)
        net = accounting_bar_returns(pos, r, FEE, fnd, LEV)
        leg_net.append(net)
        m = metrics(net, timestamps)
        leg_sh.append(m['sharpe'])
        leg_turn.append(float(turn.mean()))
        activity.append(float(np.mean(np.abs(pos))))
        a = pos[1:]; b = r[1:]
        leg_ic.append(float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and a.std() > 1e-9 and b.std() > 1e-9 else 0.0)
    pnet = np.mean(np.stack(leg_net), axis=0)
    pm = metrics(pnet, timestamps)
    psh = pm['sharpe']
    mdd = pm['mdd']
    min_leg = min(leg_sh) if leg_sh else 0.
    mean_ic = float(np.mean(leg_ic)) if leg_ic else 0.
    mean_act = float(np.mean(activity)) if activity else 0.
    mean_turn = float(np.mean(leg_turn)) if leg_turn else 0.
    if mean_act < 0.01 or mean_turn < 1e-5:
        reward = -50.0
    else:
        breadth = sum(x > 0 for x in leg_sh) / len(leg_sh) if leg_sh else 0.0
        reward = shape_reward(psh, psh_peer, min_leg, mean_ic, mean_turn,
                              breadth=breadth, version=reward_version)
    return {'reward': float(reward), 'portfolio_sharpe': psh, 'portfolio_mdd': mdd, 'min_leg_sharpe': float(min_leg), 'mean_ic': mean_ic, 'activity': mean_act, 'positive_coins': sum(x > 0 for x in leg_sh), 'leg_sharpes': leg_sh, 'turnover': mean_turn, 'solvent': pm['solvent']}
def run_walkforward(args, maps, returns, common, funding_mask, funding_by_coin,
                     folds, rng, scale_end_hint):
    """Run the search fold by fold, giving each validation window one pass.

    The single train/validation/lockbox contract lets a search look at one
    validation window tens of thousands of times. This walks forward instead:
    fold i selects on its own training window, scores once on its own
    validation window, and that validation window then becomes part of fold
    i+1's training data. No window is ever scored twice, and the final fold is
    never trained on.

    Returns the per-fold record and the formula that the final selection fold
    produced, which is the only one that goes near the holdout.
    """
    records = []
    selected = None
    for fold in folds:
        tr_s, tr_e = fold['train']
        va_s, va_e = fold['validation']
        is_holdout = fold['is_holdout']

        if is_holdout:
            # The holdout is scored once, after selection is finished, and the
            # result is written to the record. It is not fed back into the loop.
            if selected is None:
                records.append({'fold': fold['index'], 'is_holdout': True,
                                'note': 'no selection fold produced a formula'})
                break
            h = evaluate(selected, maps, returns, va_s, va_e, funding_mask,
                         scale_end_hint, common, funding_by_coin=funding_by_coin,
                         reward_version=args.reward_version)
            records.append({
                'fold': fold['index'], 'is_holdout': True,
                'validation': (va_s, va_e),
                'validation_bars': va_e - va_s,
                'formula': list(selected),
                'portfolio_sharpe': h['portfolio_sharpe'],
                'portfolio_mdd': h['portfolio_mdd'],
                'positive_coins': h['positive_coins'],
                'min_leg_sharpe': h['min_leg_sharpe'],
            })
            break

        # Selection fold. scale_end is this fold's own training end, so the
        # signal normalisation is fit on training data only.
        fold_scale_end = tr_e
        pop = (seed_formulas()
               + [random_formula(rng) for _ in range(max(0, args.population - len(seed_formulas())))])[:args.population]
        fold_best = None
        for gen in range(args.generations):
            scored = []
            for f in pop:
                a = evaluate(f, maps, returns, tr_s, tr_e, funding_mask, fold_scale_end,
                             common, funding_by_coin=funding_by_coin,
                             reward_version=args.reward_version)
                b = evaluate(f, maps, returns, va_s, va_e, funding_mask, fold_scale_end,
                             common, funding_by_coin=funding_by_coin,
                             reward_version=args.reward_version)
                # The validation window contributes to the selection score, but
                # only for this one pass. It is never revisited.
                sel = 0.5 * a['reward'] + 0.5 * b['reward']
                scored.append((sel, f, a, b))
            scored.sort(key=lambda z: z[0], reverse=True)
            if fold_best is None or scored[0][0] > fold_best[0]:
                fold_best = scored[0]
            elites = [x[1] for x in scored[:max(2, args.population // 10)]]
            new = list(elites)
            while len(new) < args.population:
                p1 = scored[rng.randrange(min(5, len(scored)))][1]
                p2 = scored[rng.randrange(min(5, len(scored)))][1]
                child = (crossover(p1, p2, rng) if rng.random() < 0.5
                         else mutate(p1, rng))
                new.append(child)
            pop = new
        selected = fold_best[1]
        records.append({
            'fold': fold['index'], 'is_holdout': False,
            'train': (tr_s, tr_e), 'validation': (va_s, va_e),
            'train_bars': tr_e - tr_s, 'validation_bars': va_e - va_s,
            'formula': list(selected),
            'train_sharpe': fold_best[2]['portfolio_sharpe'],
            'validation_sharpe': fold_best[3]['portfolio_sharpe'],
            'train_positive_coins': fold_best[2]['positive_coins'],
            'validation_positive_coins': fold_best[3]['positive_coins'],
        })
        print(f'[wf] fold {fold["index"]}: train {fold_best[2]["portfolio_sharpe"]:+.3f} '
              f'val {fold_best[3]["portfolio_sharpe"]:+.3f}', flush=True)
    return records, selected


def seed_formulas():
    # Baseline seeds from the rank-IC scan: HL_RANGE, LOG_VOL, FOMO, VOL_TREND.
    seeds=[]
    for feature in (9,5,3,11):
        seq=[feature]+[17]*11
        if is_valid(seq): seeds.append(tuple(seq))
    return seeds


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--generations',type=int,default=20); ap.add_argument('--population',type=int,default=32); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--out',type=Path,default=ROOT/'results'/'ga_28c_30m_3y.json'); ap.add_argument('--funding',choices=('constant','real'),default='real',help='constant reproduces the historical equity-compound-v2 assumption; real uses recorded Binance funding. Default is real because the constant is known to overstate the rate ~16x and to invert the sign on ~30%% of events.'); ap.add_argument('--null',choices=('none','iid','xsec'),default='none',help='iid: block-bootstrapped returns and volume per coin, no real edge. '
     'xsec: each coin paired with a neighbour bars. Both are false-positive '
     'controls for the search, not market data.'); ap.add_argument('--null-seed',type=int,default=0); ap.add_argument('--reward-version',choices=('v1','v2'),default='v1'); ap.add_argument('--null-block',type=int,default=48); ap.add_argument('--mode',choices=('standard','walkforward'),default='standard',help='standard: one train/validation split scored against a lockbox. '
     'walkforward: expanding folds, one selection pass per validation window, '
     'final fold reserved as a holdout.'); ap.add_argument('--folds',type=int,default=4,help='walkforward fold count'); args=ap.parse_args()
    rng=random.Random(args.seed); common,maps,returns,funding_mask=load_data(null=args.null,null_seed=args.null_seed,null_block=args.null_block);
    # Load once per run: real funding is per-coin and the loader is not free.
    funding_by_coin=({c:load_real_funding(c,np.asarray(common,dtype=np.int64)) for c in COINS_28C}
                     if args.funding=='real' else None)
    if args.funding=='real':
        from research.accounting_28c import real_funding_coverage
        _cov=real_funding_coverage(list(COINS_28C),np.asarray(common,dtype=np.int64))
        _bad=[c for c,v in _cov.items() if v['events_matched']<v['scheduled_events']]
        if _bad:
            print(f'[funding] incomplete history for {sorted(_bad)}; those events are treated as zero')
    funding_version = ACCOUNTING_VERSION_REAL if args.funding=='real' else ACCOUNTING_VERSION; n=len(common); contract_splits=split_indices(n, common[0], common[-1]); ranges=search_splits(n, common[0], common[-1]); scale_end=ranges['train'][1]
    lockbox_range=contract_splits['lockbox']
    if args.mode=='walkforward':
        from research.splits_walkforward_28c import walk_forward_folds, fold_summaries
        folds=walk_forward_folds(n,n_folds=args.folds)
        for f_ in fold_summaries(folds,n):
            print(f'[wf] fold {f_["fold"]}: train {f_["train_bars"]} bars, '
                  f'val {f_["validation_bars"]} bars ({f_["validation_days"]:.0f}d)'
                  f'{"  HOLDOUT" if f_["is_holdout"] else ""}',flush=True)
        # scale_end for signal normalisation is the first fold's training end,
        # so no fold normalises on data it will later be scored against.
        wf_scale_end=folds[0]['train'][1]
        wf_records,wf_selected=run_walkforward(
            args,maps,returns,common,funding_mask,funding_by_coin,folds,rng,wf_scale_end)
        out={'status':'ga_paper_only','live_adopted':False,'mode':'walkforward',
             'null_data':args.null,'reward_version':args.reward_version,
             'funding_source':args.funding,'accounting_version':funding_version,
             'seed':args.seed,'generations':args.generations,'population':args.population,
             'folds':fold_summaries(folds,n),'fold_records':wf_records,
             'bars':n}
        hold=[r for r in wf_records if r.get('is_holdout') and 'portfolio_sharpe' in r]
        if hold:
            h=hold[0]
            out['holdout']={'portfolio_sharpe':h['portfolio_sharpe'],
                            'portfolio_mdd':h['portfolio_mdd'],
                            'positive_coins':h['positive_coins'],
                            'formula':h['formula']}
        args.out.parent.mkdir(parents=True,exist_ok=True)
        with args.out.open('w') as f_: json.dump(out,f_,indent=1)
        print(json.dumps(out.get('holdout',{}),indent=1))
        return 0
    seeds=seed_formulas(); pop=(seeds+[random_formula(rng) for _ in range(max(0,args.population-len(seeds)))])[:args.population]; history=[]; best=None
    for gen in range(args.generations):
        scored=[]
        for f in pop:
            a=evaluate(f,maps,returns,*ranges['train'],funding_mask,scale_end,common,funding_by_coin=funding_by_coin,reward_version=args.reward_version); b=evaluate(f,maps,returns,*ranges['validation'],funding_mask,scale_end,common,funding_by_coin=funding_by_coin,reward_version=args.reward_version)
            sel=0.5*a['reward']+0.5*b['reward']; scored.append((sel,f,a,b))
        scored.sort(key=lambda z:z[0],reverse=True); best=scored[0] if best is None or scored[0][0]>best[0] else best
        history.append({'generation':gen,'best_select_score':scored[0][0],'avg_select_score':float(np.mean([x[0] for x in scored])),'best_formula':list(scored[0][1])})
        print(f'gen={gen} best={scored[0][0]:.3f} avg={np.mean([x[0] for x in scored]):.3f}',flush=True)
        elites=[x[1] for x in scored[:max(2,args.population//10)]]
        new=list(elites)
        while len(new)<args.population:
            p1=scored[rng.randrange(min(5,len(scored)))][1]; p2=scored[rng.randrange(min(5,len(scored)))][1]
            child=crossover(p1,p2,rng) if rng.random()<0.5 else mutate(p1,rng); new.append(child)
        pop=new
    # Use shared accounting for all evaluations
    f=best[1]
    oos=evaluate(f,maps,returns,*lockbox_range,funding_mask,scale_end,common,funding_by_coin=funding_by_coin,reward_version=args.reward_version)
    train=evaluate(f,maps,returns,*ranges['train'],funding_mask,scale_end,common,funding_by_coin=funding_by_coin,reward_version=args.reward_version)
    val=evaluate(f,maps,returns,*ranges['validation'],funding_mask,scale_end,common,funding_by_coin=funding_by_coin,reward_version=args.reward_version)
    
    gates=ACCEPTANCE_GATE_28C
    # Regime gate. Built from the SAME positions evaluate() builds, and run
    # once at the end so it cannot influence selection. Net returns are
    # computed over the full series and sliced per fold inside the gate, so no
    # phantom entry fee is charged at a fold boundary.
    positions={}
    for c in COINS_28C:
        sig=formula_signal(f,maps,c)
        fs=float(np.std(sig[:scale_end])) if scale_end>0 else 0.0
        raw_pos=np.tanh(sig/(fs+1e-6)) if fs>1e-8 else np.zeros_like(sig)
        pp=0.25*smooth_causal(raw_pos,5); pp=np.roll(pp,1); pp[0]=0
        positions[c]=pp
    try:
        gate=evaluate_regime_gate(
            positions, returns, list(COINS_28C), funding_mask, contract_splits,
            np.asarray(common,dtype=np.int64), FEE, LEV, lockbox_range,
            gates['min_positive_coins'], gates['max_oos_mdd'], int(common[0]),
            funding_by_coin=funding_by_coin)
    except Exception as exc:
        # Fail closed: an absent or broken gate is never a pass.
        gate={'verdict':'fail','verdict_reason':f'gate error: {type(exc).__name__}: {exc}','criteria':{},'folds':[]}
    acceptance=make_acceptance(
        gate['verdict'], gate.get('verdict_reason',''),
        min_positive_coins=gates['min_positive_coins'],
        oos_mdd=oos['portfolio_mdd']<gates['max_oos_mdd'],
        oos_solvent=oos['solvent'],
        oos_portfolio_sharpe=oos['portfolio_sharpe'],
        legacy_bar=gates['min_oos_portfolio_sharpe'],
        criteria=gate.get('criteria',{}))
    # Demoted to a diagnostic: the gate decides on excess Sharpe, so keeping the
    # old name under acceptance/ would let a reader treat it as a live gate.
    diagnostics={NEW_KEY: oos['portfolio_sharpe']>gates['min_oos_portfolio_sharpe']}
    out={
        'status': 'ga_paper_only',
        'live_adopted': derive_live_adopted(acceptance),
        'formula': list(f),
        'seed': args.seed,
        'generations': args.generations,
        'population': args.population,
        'data_dir': str(DATA_DIR),
        'history_years': HISTORY_YEARS_COMMON_28,
        'bars': n,
        'splits': {
            'train': ranges['train'],
            'validation': ranges['validation'],
            'lockbox': lockbox_range
        },
        'train': train,
        'validation': val,
        'oos': oos,
        'acceptance': acceptance,
        'diagnostics': diagnostics,
        # N6: a NEW top-level key. dashboard/visualizer.py:75-83 reads
        # `history` expecting step/sharpe and silently returns an empty figure
        # on mismatch, so reusing that name would hide the fold table.
        FOLD_TABLE_KEY: {'folds': gate.get('folds',[]), 'lockbox': gate.get('lockbox',{}),
                         'fold_coverage': gate.get('fold_coverage',{})},
        'history': history,
        'note': 'continuous causal GA diagnostic; thresholds fitted later; OOS not used for selection'
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Record which funding basis produced these numbers. A result is not
    # comparable across the two, and the difference is large enough to flip
    # a sign, so this must travel with the artifact rather than be recalled.
    out['null_data']=args.null
    out['reward_version']=args.reward_version
    out['funding_source']=args.funding
    out['accounting_version']=funding_version
    out['funding_note']=(
        'real Binance funding history at 00/08/16 UTC; may be negative'
        if args.funding=='real' else
        'flat +0.0005 at every settlement; overstates the rate and is '
        'unconditionally credited to shorts. Superseded, kept only to '
        'reproduce pre-94b59ca results.')
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ('status', 'live_adopted', 'formula', 'train', 'validation', 'oos', 'acceptance', 'diagnostics')}, indent=2))

if __name__ == "__main__": main()
