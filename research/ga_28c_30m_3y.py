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


def load_data():
    raw={}; ts0=None; common=None
    for c in COINS_28C:
        p=DATA_DIR/f'{c}.csv'
        with p.open() as f: rows=list(csv.DictReader(f))
        d={k:[float(r[k]) for r in rows] for k in ('open','high','low','close','volume')}
        d['timestamp']=[int(float(r['timestamp'])) for r in rows]
        raw[c]=d; s=set(d['timestamp']); common=s if common is None else common&s
    common=sorted(common); n=len(common)
    maps={}; returns={}
    for c in COINS_28C:
        idx={t:i for i,t in enumerate(raw[c]['timestamp'])}
        d={k:[raw[c][k][idx[t]] for t in common] for k in ('open','high','low','close','volume')}
        # Cache is keyed by feature-contract version and validated by bar count.
        #
        # Two ways this used to go wrong. The length check called len() on a
        # (n_factors, n_bars) array, which returns n_factors, so it never matched
        # n_bars and the cache was recomputed on every run -- it was never
        # actually used, only paid for. Had that check been "fixed" without also
        # versioning, the stale files still on disk (f1 pinned at the old 0.4
        # placeholder) would have silently shadowed the current factor code.
        # Version the key and compare the right axis.
        cache=CACHE_DIR/f'{c}-{FEATURE_CACHE_VERSION}.npy'
        if cache.exists():
            try:
                loaded=np.load(cache)
            except Exception:
                loaded=None
            if loaded is not None and loaded.shape==(len(FEATURE_NAMES), n) \
               and not constant_features(loaded):
                maps[c]=loaded
            else:
                maps[c]=causal_features(d)
                np.save(cache,maps[c])
        else:
            maps[c]=causal_features(d)
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

def shape_reward(psh, psh_peer, min_leg, mean_ic, mean_turn):
    """Turn raw split diagnostics into a selection score.

    psh is this split's own portfolio Sharpe; psh_peer is the other selection
    split's Sharpe. psh is independent of psh_peer, so the caller can evaluate
    each split once and shape both scores afterwards.
    """
    if mean_act_guard(mean_turn):
        return -50.0
    peer = psh if psh_peer is None else psh_peer
    # Fragmentation: credit the worse of the two splits, not their average, so a
    # formula that only works on one split earns nothing from the good one.
    psh_worst = min(psh, peer)
    # Divergence: charge for the gap instead of averaging it away.
    gap = softplus((psh - peer) / GAP_TAU) * GAP_TAU
    return (mean_ic
            + 0.10 * psh_worst
            - 0.30 * gap
            - 0.02 * mean_turn
            - 0.10 * softplus((LEG_FLOOR - min_leg) / 0.5))


def mean_act_guard(mean_turn):
    return mean_turn < 1e-5


def evaluate(formula, maps, returns, start, end, funding_mask, scale_end, bars, psh_peer=None, funding_by_coin=None):
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
        reward = shape_reward(psh, psh_peer, min_leg, mean_ic, mean_turn)
    return {'reward': float(reward), 'portfolio_sharpe': psh, 'portfolio_mdd': mdd, 'min_leg_sharpe': float(min_leg), 'mean_ic': mean_ic, 'activity': mean_act, 'positive_coins': sum(x > 0 for x in leg_sh), 'leg_sharpes': leg_sh, 'turnover': mean_turn, 'solvent': pm['solvent']}
def seed_formulas():
    # Baseline seeds from the rank-IC scan: HL_RANGE, LOG_VOL, FOMO, VOL_TREND.
    seeds=[]
    for feature in (9,5,3,11):
        seq=[feature]+[17]*11
        if is_valid(seq): seeds.append(tuple(seq))
    return seeds


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--generations',type=int,default=20); ap.add_argument('--population',type=int,default=32); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--out',type=Path,default=ROOT/'results'/'ga_28c_30m_3y.json'); ap.add_argument('--funding',choices=('constant','real'),default='real',help='constant reproduces the historical equity-compound-v2 assumption; real uses recorded Binance funding. Default is real because the constant is known to overstate the rate ~16x and to invert the sign on ~30%% of events.'); args=ap.parse_args()
    rng=random.Random(args.seed); common,maps,returns,funding_mask=load_data();
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
    seeds=seed_formulas(); pop=(seeds+[random_formula(rng) for _ in range(max(0,args.population-len(seeds)))])[:args.population]; history=[]; best=None
    for gen in range(args.generations):
        scored=[]
        for f in pop:
            a=evaluate(f,maps,returns,*ranges['train'],funding_mask,scale_end,common,funding_by_coin=funding_by_coin); b=evaluate(f,maps,returns,*ranges['validation'],funding_mask,scale_end,common,funding_by_coin=funding_by_coin); a['reward']=shape_reward(a['portfolio_sharpe'],b['portfolio_sharpe'],a['min_leg_sharpe'],a['mean_ic'],a['turnover']); b['reward']=shape_reward(b['portfolio_sharpe'],a['portfolio_sharpe'],b['min_leg_sharpe'],b['mean_ic'],b['turnover'])
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
    oos=evaluate(f,maps,returns,*lockbox_range,funding_mask,scale_end,common,funding_by_coin=funding_by_coin)
    train=evaluate(f,maps,returns,*ranges['train'],funding_mask,scale_end,common,funding_by_coin=funding_by_coin)
    val=evaluate(f,maps,returns,*ranges['validation'],funding_mask,scale_end,common,funding_by_coin=funding_by_coin)
    
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
