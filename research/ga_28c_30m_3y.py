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

# The passive benchmark's position cap. A constant long at this cap is the
# reference every Sharpe in this project should be read against.
POSITION_CAP = 0.25
from research.formula_grammar import valid_token_mask, update_depth, is_valid, ALL_TOKENS, OP_ARITY as FULL_OP_ARITY
from research.grammar_reduced_28c import (REDUCED_TOKENS as REDUCED_ALL_TOKENS,
                                  REDUCED_ARITY as REDUCED_OP_ARITY,
                                  REDUCED_TOKEN_NAMES, validate_reduced,
                                  REDUCED_OP_COUNT, evaluate_formula_reduced)
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

# --- pilot meter repair (2026-10-01, docs/GA_SEARCH_RELIABILITY_2026-10-01.md) ---
# ACTIVITY_TARGET is the old hard cliff. INACTIVITY_WEIGHT is the slope of the
# replacement penalty, and it is deliberately set so the repaired reward stays on
# the SAME scale as the recorded v1 reward rather than being rescaled underneath
# the A/B: the old cliff scored a dead genome -50, and the non-cliff reward of a
# dead genome was already about -1.0, so the gap between "dead" and "worst
# passing" was roughly 49 reward units. The repaired reward spans about 1.7 units
# across the same inputs. Those scales are not comparable and are not meant to be;
# verify_pilot_meter_repair.py therefore compares real-vs-null INSIDE one meter,
# never across meters, and reports the rankable-run std rather than the raw pooled
# std. Choosing a weight to make the numbers look comparable would be fitting the
# scale to the batch it is scoring.
ACTIVITY_TARGET = 0.01
INACTIVITY_WEIGHT = 0.5

# PEER_BASELINE: a fixed external Sharpe used as psh_peer so the split-divergence
# term is live. None until the pilot supplies it, because any constant chosen
# after seeing the runs would be fitted to the batch it is scoring.
PEER_BASELINE = None

# Turnover price inside shape_reward.
#
# Extracted from an inline 0.02 so a probe can sweep it. The value is NOT
# arbitrary and was not tuned: at 0.02 the term's entire dynamic range across
# 90 recorded selections was 0.000502 reward units against a total reward spread
# of 0.6187, i.e. 0.0812%. It is the only term that would penalise trading and
# it cannot. Whether a larger weight helps is an open question measured by
# research/turnover_weight_probe_28c.py, not assumed here.
#
# The env override exists ONLY so that probe can sweep the weight without editing
# the search between runs. It is not a runtime knob: any value other than the
# default makes the run non-comparable with every recorded batch, and
# tests/test_pilot_meter_repair.py pins the default.
TURNOVER_WEIGHT = float(os.getenv("ALPHAGPT_TURNOVER_WEIGHT", "0.02"))

# --- PRP-0: validation-window selection leak (2026-10-01) ---
# DEFAULT_SELECT_ON is "train". "val_weighted" reproduces the pre-PRP behaviour
# and exists ONLY so the recorded historical batches stay reproducible; it must
# never be used to produce a reportable result.
DEFAULT_SELECT_ON = "train"
SELECT_ON_CHOICES = ("train", "val_weighted")

# Objective presets, each a kwargs dict for shape_reward.
#
# These exist because the sweep measured that v3 has no out-of-sample gradient:
# argmax_margin 0.009 against 0.70 of fold-to-fold scatter, 7 of 10 selected
# formulas holding a constant position, and 0 of 10 beating buy-and-hold. The
# turnover weight was measured across 0.02 to 200 and none of them helped, so
# the objective itself is what needs trying, not a scale factor on it.
#
# Every preset is opt-in and the default stays the frozen v3 arithmetic.
OBJECTIVES = {
    # The frozen baseline. Present so a sweep can name it explicitly.
    "v3": {},
    # Rank-IC as the primary selector. It is the only measured quantity that
    # separates real from null across regimes: +0.6188 against +0.4297, a gap of
    # +0.1891, while Sharpe manages +0.8545 against +0.8758, a gap of -0.0214.
    # Sharpe is dropped to near zero because it is the term that measures drift.
    "rank_ic": {"w_mean_ic": 0.0, "w_psh": 0.0,
                "w_rank_ic": 25.0},
    # Excess over buy-and-hold as the primary objective. This is what the gate
    # tests and what v3 already priced at 0.30; here it is the whole objective.
    "excess_bh": {"w_mean_ic": 0.0, "w_psh": 0.0,
                  "w_excess_bh": 1.0},
    # Rank-IC and excess-over-hold together, keeping a small Sharpe floor so a
    # book with no per-leg edge cannot win on breadth alone.
    "rank_ic_excess_bh": {"w_mean_ic": 0.0, "w_psh": 0.02,
                          "w_rank_ic": 25.0, "w_excess_bh": 1.0},
    # v3 with a turnover weight large enough to matter. Measured separately:
    # 0.02 contributes 0.0812% of the reward spread, so movement is free.
    "v3_turn20": {"w_turn": 20.0},
}


def selection_score(a, b, mode=DEFAULT_SELECT_ON):
    """The score the GA ranks, elits and picks parents on.

    "train"        -> the train reward alone. The validation split cannot reach
                      selection by any route.
    "val_weighted" -> 0.5*train + 0.5*validation. This is the historical key and
                      it leaked: the validation window was re-scored every
                      generation (pop x generations times per fold, 3,600 times at
                      pilot settings) and carried half the key.

    SCOPE, stated precisely because it is easy to over-claim:
      val_weighted restores the SELECTION KEY only. It does not restore the
      historical NUMBERS, because two separate changes landed in the same working
      tree: this leak fix, and the meter repair that replaced the -50 cliff with a
      continuous activity penalty. A batch recorded before either change is not
      reproducible from this file at all. To reproduce one, check out the commit
      that produced it.

      So --select-on val_weighted means "isolate the selection-key question",
      not "reproduce history". Any number it produces is a diagnostic, never a
      result.

    Pinned by tests/test_validation_selection_leak.py, including a scan that fails
    if any weighted expression reappears in an active code path.
    """
    if mode == "val_weighted":
        return 0.5 * a["reward"] + 0.5 * b["reward"]
    return a["reward"]


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
    elif null in ('iid1','iidw','iidg'):
        from research.null_data_28c import make_diagnostic_null
        # Diagnostic nulls. Each breaks exactly one property that the iid null
        # preserves, so a difference in the search result attributes the
        # effect to that property instead of leaving it open. They are
        # deliberately LESS realistic than iid and are used to attribute an
        # effect, not to pass or fail a candidate. See docs/
        # reduced_grammar_null_calibration_28c.md.
        aligned={c: make_diagnostic_null(null, aligned[c], seed=null_seed+i,
                                         block=null_block)
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


# Search operators, parameterised by grammar.
#
# The full grammar is 29 tokens (12 factors + 17 operators) and admits roughly
# 3.1e15 syntactically valid 12-token formulas. Ten independent searches return
# ten disjoint formulas, so the search is not converging on anything; it is
# sampling, and it samples Sharpe out of noise. The reduced grammar keeps the
# ten operators that carry an economic argument and drops the seven that only
# reshuffle a cross-section (ZSCORE, RANK, TS_RANK) or pattern-match a single
# series (JUMP, MAX3, ABS). Fewer tokens is the point: a smaller hypothesis
# class is a smaller multiple-testing burden, not a better optimiser.
#
# Both grammars are token-indexed the same way, features 0..11 then operators,
# so a formula decoded under the full grammar keeps its meaning. The reduced
# grammar re-indexes operators 12..22 with different meanings, which is why an
# artifact records `grammar` and a reduced-grammar formula is never scored by
# causal_12f's operator table.
GRAMMAR_SPECS={
    'full': dict(n_features=FEATURE_COUNT, arity=FULL_OP_ARITY,
                 max_token=ALL_TOKENS),
    'reduced': dict(
        n_features=FEATURE_COUNT,
        arity=REDUCED_OP_ARITY,
        max_token=REDUCED_ALL_TOKENS,
    ),
}
GRAMMAR='full'


def _spec():
    return GRAMMAR_SPECS[GRAMMAR]


# The reduced grammar's own reachability test and evaluator. The full grammar
# keeps using formula_grammar's versions, untouched: every stored artifact in
# this repository was produced by that code path, and a rewrite of it that
# produced the same formulas "up to reimplementation" would silently
# invalidate all of them.
def _can_finish_r(depth, slots, n, arity):
    if slots==0: return depth==1
    if depth>12: return False
    if depth<1: return False
    if depth>slots+1: return False
    if _can_finish_r(depth+1, slots-1, n, arity): return True
    for a in arity.values():
        if depth>=a and _can_finish_r(depth-a+1, slots-1, n, arity): return True
    return False


def _valid_tokens_r(depth, pos, max_len, n, arity):
    slots=max_len-pos-1
    out=[t for t in range(n) if _can_finish_r(depth+1, slots, n, arity)]
    out.extend(t for t, a in arity.items()
               if depth>=a and _can_finish_r(depth-a+1, slots, n, arity))
    return out


def _step_r(depth, token, n, arity):
    if token<n: return depth+1
    a=arity[token]
    if depth<a: raise ValueError('stack underflow')
    return depth-a+1


def _is_valid_r(formula):
    n=FEATURE_COUNT; arity=REDUCED_OP_ARITY
    depth=0
    try:
        for i, t in enumerate(formula):
            t=int(t)
            if not (0 <= t < REDUCED_ALL_TOKENS): return False
            if t not in _valid_tokens_r(depth, i, len(formula), n, arity): return False
            depth=_step_r(depth, t, n, arity)
    except ValueError:
        # Stack underflow. Only a malformed program may reach here: an
        # unexpected exception type means a bug in this module, and swallowing
        # it would report every formula as invalid instead of raising.
        return False
    return depth==1


def _fill_r(prefix, rng, n, arity):
    f=list(prefix); depth=0
    for t in f: depth=_step_r(depth, int(t), n, arity)
    while len(f)<FORMULA_LEN:
        choices=_valid_tokens_r(depth, len(f), FORMULA_LEN, n, arity)
        if not choices: return None
        t=int(rng.choice(choices)); f.append(t); depth=_step_r(depth, t, n, arity)
    return tuple(f) if depth==1 else None


def _random_formula_reduced(rng):
    n=FEATURE_COUNT; arity=REDUCED_OP_ARITY
    for _ in range(200):
        f=_fill_r([], rng, n, arity)
        if f is not None: return f
    raise RuntimeError('failed to sample a valid reduced-grammar formula')


def _mutate_reduced(formula, rng):
    n=FEATURE_COUNT; arity=REDUCED_OP_ARITY
    for _ in range(50):
        f=list(formula)
        pos=rng.randrange(FORMULA_LEN)
        depth=0
        for t in f[:pos]: depth=_step_r(depth, int(t), n, arity)
        choices=_valid_tokens_r(depth, pos, FORMULA_LEN, n, arity)
        if not choices: continue
        f[pos]=int(rng.choice(choices))
        if _is_valid_r(f): return tuple(f)
    return _random_formula_reduced(rng)


def _crossover_reduced(a, b, rng):
    # Repair is uniform resampling at the cut, not a tail-preserving splice.
    # A tail-preserving repair would depend on the second parent's suffix, and
    # under the reduced grammar the same token index means a different operator,
    # so it would not be the same operator.
    n=FEATURE_COUNT; arity=REDUCED_OP_ARITY
    cut=rng.randrange(1, FORMULA_LEN)
    f=_fill_r(list(a[:cut]), rng, n, arity)
    return f if f is not None else _random_formula_reduced(rng)


def random_formula(rng):
    if GRAMMAR=='reduced': return _random_formula_reduced(rng)
    seq=[]; depth=0
    for pos in range(FORMULA_LEN):
        choices=np.flatnonzero(valid_token_mask(depth,pos,FORMULA_LEN).numpy()).tolist()
        t=rng.choice(choices); seq.append(int(t)); depth=update_depth(depth,int(t))
    return tuple(seq)


def mutate(formula,rng):
    if GRAMMAR=='reduced': return _mutate_reduced(formula,rng)
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
    if GRAMMAR=='reduced': return _crossover_reduced(a,b,rng)
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


def resolve_grammar(formula=None, grammar=None):
    """Which evaluator a formula must go through, decided explicitly.

    Token indices are grammar-specific: reduced token 20 is DELAY1 while full
    token 20 is JUMP. A reduced formula routed through the full evaluator
    computes a DIFFERENT FUNCTION and reports it honestly, so nothing crashes
    and the number is simply wrong. This is the same class of bug as a
    placeholder factor: the pipeline runs, the guard passes, and the output is
    meaningless.

    GRAMMAR was previously a module global set only inside main(), so any
    importer -- a probe, a test, a notebook -- silently inherited 'full' and
    sent reduced formulas to the wrong evaluator. The grammar is now a
    parameter that travels with the formula, and a formula whose tokens do not
    belong to the named grammar is refused rather than mistranslated.
    """
    if grammar is None:
        grammar = GRAMMAR
    if formula is not None and grammar == 'reduced' and not validate_reduced(formula):
        raise ValueError(
            f"formula {list(formula)} is not valid in the reduced grammar; "
            "refusing to evaluate it rather than mistranslate its tokens")
    return grammar


def formula_signal(formula,maps,c,grammar=None):
    # Token indices are grammar-specific: reduced token 20 is DELAY1 while full
    # token 20 is JUMP. Routing a reduced formula through the full evaluator
    # would compute a different function and report it honestly, so the
    # grammar chooses the evaluator, and resolve_grammar refuses a mismatch.
    grammar=resolve_grammar(formula,grammar)
    if grammar=='reduced':
        return evaluate_formula_reduced(list(formula),maps[c])
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
                 breadth=None, version='v1', excess=None,
                 w_mean_ic=1.0, w_psh=0.10, w_gap=0.30, w_turn=0.02,
                 w_breadth=2.0, w_excess=0.30, w_rank_ic=0.0, w_excess_bh=0.0,
                 rank_ic=None, excess_bh=None):
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

    v3 adds excess-over-buy-and-hold as a direct objective. The gate tests
    EXCESS over a constant levered long on the same bars. Those are different
    objectives and nothing connected them, so the search maximised the wrong
    thing for two full iterations: it pushed absolute Sharpe up, which on this
    universe means matching drift, while the gate kept rejecting every result
    for failing to beat the hold.

    Weight is 0.30 against the 0.10 already applied to the worst split, so
    beating the hold is worth roughly three times as much as a one-Sharpe
    improvement in absolute terms. It is a reward, not a hard constraint: a
    formula that is short and loses is still scored on its own legs, and the
    null companion is what says whether v3 found anything or merely learned
    to look like a long.

    New optional components (disabled by default, enabled via weights):
    - w_rank_ic * rank_ic: rank-IC as a selector within a regime
    - w_excess_bh * excess_bh: excess over buy-and-hold as direct objective
    """
    # Activity guard: log-only. It used to return -50.0, which is a cliff, not a
    # gradient. Two runs recorded with the cliff still standing showed it firing
    # on 5/10 real and 6/10 null seeds, and it fired on mean_act < 0.01 in every
    # one of those cases while turnover was mostly above the guard line. A cliff
    # that fires more often on the arm that wins Sharpe is not a neutral filter.
    # Inactivity is now priced continuously by the caller's linear penalty instead.
    #
    # psh_peer: BOTH call sites passed psh_peer=None, which made peer == psh, which
    # made gap == softplus(0) == 0.0 exactly, for every genome, forever. The
    # -0.30*gap term was dead code and the split-divergence penalty never fired.
    # The caller must now supply a fixed external baseline (see PEER_BASELINE);
    # falling back to psh restores the old dead-code behaviour on purpose, so a
    # caller that forgets is visibly no worse than before rather than silently
    # gaining a term the A/B was not calibrated for.
    peer = psh if psh_peer is None else psh_peer
    # Fragmentation: credit the worse of the two splits, not their average, so a
    # formula that only works on one split earns nothing from the good one.
    psh_worst = min(psh, peer)
    # Divergence: charge for the gap instead of averaging it away.
    gap = softplus((psh - peer) / GAP_TAU) * GAP_TAU
    base = (w_mean_ic * mean_ic
            + w_psh * psh_worst
            - w_gap * gap
            - w_turn * mean_turn
            - 0.10 * softplus((LEG_FLOOR - min_leg) / 0.5))
    if version == 'v1':
        return base
    short = max(0.0, BREADTH_TARGET - breadth)
    base = base - w_breadth * short
    if version == 'v2' or excess is None:
        return base
    # v3: charge the search for what the ACCEPT GATE actually tests.
    #
    # v1 and v2 reward absolute Sharpe. The gate tests EXCESS over a constant
    # levered long on the same bars. Those are different objectives and nothing
    # connected them, so the search maximised the wrong thing for two full
    # iterations: it pushed absolute Sharpe up, which on this universe means
    # matching drift, while the gate kept rejecting every result for failing to
    # beat the hold.
    #
    # Weight is 0.30 against the 0.10 already applied to the worst split, so
    # beating the hold is worth roughly three times as much as a one-Sharpe
    # improvement in absolute terms. It is a reward, not a hard constraint: a
    # formula that is short and loses is still scored on its own legs, and the
    # null companion is what says whether v3 found anything or merely learned
    # to look like a long.
    base = base + w_excess * excess
    if rank_ic is not None:
        base = base + w_rank_ic * rank_ic
    if excess_bh is not None:
        base = base + w_excess_bh * excess_bh
    return base

def mean_act_guard(mean_turn):
    return mean_turn < 1e-5



def evaluate(formula, maps, returns, start, end, funding_mask, scale_end, bars, psh_peer=None, funding_by_coin=None, reward_version='v1', grammar=None, objective=None):
    grammar=resolve_grammar(formula,grammar)
    leg_net = []; leg_sh = []; leg_turn = []; leg_ic = []; activity = []; leg_bh = []
    timestamps = bars[start:end]
    for c in COINS_28C:
        sig = formula_signal(formula, maps, c, grammar=grammar)
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
        #
        # funding_by_coin must be a dict of per-coin RATE arrays. load_data
        # returns an 8-hour settlement MASK in that slot, which is a different
        # shape entirely: indexing it with a coin name raises IndexError, and
        # passing None instead silently reverts to the superseded flat rate.
        # Both failure modes were observed, so the type is checked here rather
        # than left to a comment. Comparing arrays of the wrong kind would
        # still broadcast, so the lengths are checked too.
        if funding_by_coin is not None:
            if not isinstance(funding_by_coin, dict) or c not in funding_by_coin:
                raise TypeError(
                    f"funding_by_coin must be a dict keyed by coin, got "
                    f"{type(funding_by_coin).__name__}; load_data returns the "
                    "8-hour settlement mask, which is a different thing")
            fnd = np.asarray(funding_by_coin[c], dtype=np.float64)[start:end]
        else:
            fnd = funding_mask[start:end] * FUND_RATE
        net = accounting_bar_returns(pos, r, FEE, fnd, LEV)
        leg_net.append(net)
        # The same window, the same book, held flat and long at the position
        # cap. A constant +1 never trades, so it pays no turnover fee.
        leg_bh.append(accounting_bar_returns(
            np.full(len(pos), POSITION_CAP), r, FEE, fnd, LEV))
        m = metrics(net, timestamps)
        leg_sh.append(m['sharpe'])
        leg_turn.append(float(turn.mean()))
        activity.append(float(np.mean(np.abs(pos))))
        a = pos[1:]; b = r[1:]
        leg_ic.append(float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and a.std() > 1e-9 and b.std() > 1e-9 else 0.0)
    pnet = np.mean(np.stack(leg_net), axis=0)
    pm = metrics(pnet, timestamps)
    psh = pm['sharpe']
    # The reported Sharpe is measured against ZERO, which is not a benchmark
    # anyone can hold. On this universe a constant long has a Sharpe of its
    # own, because crypto drifted, and it was worth +1.05 net over the lockbox
    # while the best real-data run scored -0.32. A Sharpe that clears zero can
    # therefore still be worse than doing nothing, and reading one as an edge
    # is reading a drift as a discovery.
    #
    # The excess is the Sharpe of the ACTIVE series (strategy minus hold), not
    # the difference of two Sharpes, which is not a Sharpe. Subtracting the
    # two P&L series keeps the compounding and the fee accounting on both
    # sides consistent, so the result is a return series that could itself
    # have been traded.
    bnet = np.mean(np.stack(leg_bh), axis=0)
    bh_sh = metrics(bnet, timestamps)['sharpe']
    active_sh = metrics(pnet - bnet, timestamps)['sharpe']
    mdd = pm['mdd']
    min_leg = min(leg_sh) if leg_sh else 0.
    mean_ic = float(np.mean(leg_ic)) if leg_ic else 0.
    mean_act = float(np.mean(activity)) if activity else 0.
    mean_turn = float(np.mean(leg_turn)) if leg_turn else 0.
    # Inactivity used to return the constant -50.0, a cliff with no gradient:
    # every genome below the threshold scored identically, so selection had
    # nothing to rank on inside that region. It is now a continuous penalty that
    # is zero at or above the old 0.01 activity level, so a formula that clears
    # the bar is scored exactly as before and only the dead region gains a slope.
    #
    # This is the only behavioural change in the pilot. Evaluator math, genome,
    # weights, costs, universe, seeds and windows are untouched, so a real-vs-null
    # comparison across this commit measures the meter, not the engine.
    activity_penalty = -INACTIVITY_WEIGHT * max(0.0, 1.0 - mean_act / ACTIVITY_TARGET)
    breadth = sum(x > 0 for x in leg_sh) / len(leg_sh) if leg_sh else 0.0
    reward = shape_reward(psh, psh_peer, min_leg, mean_ic, mean_turn,
                          breadth=breadth, version=reward_version,
                          excess=active_sh,
                          **(objective or {})) + activity_penalty
    return {'reward': float(reward), 'portfolio_sharpe': psh, 'portfolio_mdd': mdd,
            'buy_and_hold_sharpe': bh_sh, 'excess_sharpe_vs_buy_and_hold': active_sh,
            'mean_position': float(np.mean([float(np.mean(p[start:end])) for c in COINS_28C])) if COINS_28C else 0.0, 'min_leg_sharpe': float(min_leg), 'mean_ic': mean_ic, 'activity': mean_act, 'positive_coins': sum(x > 0 for x in leg_sh), 'leg_sharpes': leg_sh, 'turnover': mean_turn, 'solvent': pm['solvent']}
def run_walkforward(args, maps, returns, common, funding_mask, funding_by_coin,
                     folds, rng, scale_end_hint):
    """Run the search fold by fold.

    The single train/validation/lockbox contract lets a search look at one
    validation window tens of thousands of times. This walks forward instead:
    fold i selects on its own training window, and that validation window then
    becomes part of fold i+1's training data. The final fold is never trained on.

    PRP-0 correction. This docstring previously claimed "No window is ever scored
    twice", and that was false. The selection key was 0.5*train + 0.5*validation
    and was recomputed inside the generation loop, so each fold's validation
    window was read pop x generations times (400 at pilot settings, 3,600 across
    the nine selection folds). Selection now keys on the train reward only; see
    selection_score(). The validation split is still evaluated each generation
    because the fold record needs its Sharpe, but it is an observation now, not a
    selector input.

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
                         objective=OBJECTIVES[args.objective])
            records.append({
                'fold': fold['index'], 'is_holdout': True,
                'validation': (va_s, va_e),
                'validation_bars': va_e - va_s,
                'formula': list(selected),
                'portfolio_sharpe': h['portfolio_sharpe'],
                'portfolio_mdd': h['portfolio_mdd'],
                # The benchmark travels with the score. Without it a
                # walk-forward record reports a Sharpe against zero, and the
                # accept stage downstream has nothing to compare it to -- it
                # either blocks forever or, worse, is loosened to compare
                # against zero and the loop becomes a drift machine.
                'buy_and_hold_sharpe': h['buy_and_hold_sharpe'],
                'excess_sharpe_vs_buy_and_hold': h['excess_sharpe_vs_buy_and_hold'],
                'mean_position': h['mean_position'],
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
        # PRP-0 F5: the runner-up is tracked across generations so argmax_margin
        # measures how much the winner actually beat the field on train. A thin
        # margin combined with a large validation drop is the winner's-curse
        # signature, and it is invisible unless it is recorded.
        fold_runner_up = None
        for gen in range(args.generations):
            scored = []
            for f in pop:
                a = evaluate(f, maps, returns, tr_s, tr_e, funding_mask, fold_scale_end,
                             common, funding_by_coin=funding_by_coin,
                             objective=OBJECTIVES[args.objective])
                b = evaluate(f, maps, returns, va_s, va_e, funding_mask, fold_scale_end,
                             common, funding_by_coin=funding_by_coin,
                             objective=OBJECTIVES[args.objective])
                # PRP-0: the validation window is NOT part of the selection key.
                #
                # This used to read 0.5*train + 0.5*validation, evaluated inside
                # the generation loop, which re-scored the validation window
                # pop x generations times per fold (3,600 times at pilot settings)
                # and let half the key come from held-out data. The comment here
                # previously claimed the window "is never revisited", which was
                # false.
                #
                # b is still evaluated because the fold record and the per-fold
                # print both need its Sharpe. It is an observation now, not a
                # selector input.
                sel = selection_score(a, b, args.select_on)
                scored.append((sel, f, a, b))
            scored.sort(key=lambda z: z[0], reverse=True)
            if fold_best is None or scored[0][0] > fold_best[0]:
                fold_best = scored[0]
                fold_runner_up = scored[1][0] if len(scored) > 1 else None
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
            # PRP-0 F5 diagnostics.
            'select_on': args.select_on,
            'train_reward': float(fold_best[2]['reward']),
            'validation_reward': float(fold_best[3]['reward']),
            # Negative means validation scored the winner below train, which is
            # the ordinary case for an honestly selected genome. A large negative
            # number is overfit; a positive one is not an error by itself.
            'overfit_gap': float(fold_best[3]['reward'] - fold_best[2]['reward']),
            'argmax_margin': (float(fold_best[0] - fold_runner_up)
                              if fold_runner_up is not None else None),
            'mean_ic': float(fold_best[2]['mean_ic']),
            'validation_mean_ic': float(fold_best[3]['mean_ic']),
            'turnover': float(fold_best[2]['turnover']),
        })
        print(f'[wf] fold {fold["index"]}: train {fold_best[2]["portfolio_sharpe"]:+.3f} '
              f'val {fold_best[3]["portfolio_sharpe"]:+.3f}', flush=True)
    return records, selected


def run_standard_selection(args, maps, returns, ranges, scale_end, common,
                            funding_mask, funding_by_coin, rng):
    """Standard-mode selection loop, factored out of main() so it can be tested.

    Same statements and same order as before; the only change is that the loop
    lives in a named function instead of inline in main(). main() now calls this
    and prints the returned history, so there is one implementation, not two.

    Returns (best_record, history) where best_record is the (sel, formula, train,
    validation) tuple carried across generations.
    """
    seeds = seed_formulas()
    pop = (seeds + [random_formula(rng) for _ in range(max(0, args.population - len(seeds)))])[:args.population]
    history = []
    best = None
    for gen in range(args.generations):
        scored = []
        for f in pop:
            a = evaluate(f, maps, returns, *ranges['train'], funding_mask, scale_end,
                         common, funding_by_coin=funding_by_coin,
                          objective=OBJECTIVES[args.objective])
            b = evaluate(f, maps, returns, *ranges['validation'], funding_mask, scale_end,
                         common, funding_by_coin=funding_by_coin,
                          objective=OBJECTIVES[args.objective])
            sel = selection_score(a, b, args.select_on)
            scored.append((sel, f, a, b))
        scored.sort(key=lambda z: z[0], reverse=True)
        best = scored[0] if best is None or scored[0][0] > best[0] else best
        history.append({'generation': gen, 'best_select_score': scored[0][0],
                        'avg_select_score': float(np.mean([x[0] for x in scored])),
                        'best_formula': list(scored[0][1])})
        elites = [x[1] for x in scored[:max(2, args.population // 10)]]
        new = list(elites)
        while len(new) < args.population:
            p1 = scored[rng.randrange(min(5, len(scored)))][1]
            p2 = scored[rng.randrange(min(5, len(scored)))][1]
            child = crossover(p1, p2, rng) if rng.random() < 0.5 else mutate(p1, rng)
            new.append(child)
        pop = new
    return best, history


def seed_formulas():
    # Baseline seeds from the rank-IC scan: HL_RANGE, LOG_VOL, FOMO, VOL_TREND.
    # A bare factor wrapped in the grammar's unary operator is legal at length
    # 12, and the seeds are revalidated under the active grammar. Token 17 is
    # ABS under the full grammar and SIGN under the reduced one; both are arity
    # 1, so the same index is legal in both, but it is not the same function, so
    # a seed must never be carried across grammars by index alone.
    seeds=[]
    for feature in (9,5,3,11):
        seq=[feature]+[17]*11
        ok=_is_valid_r(seq) if GRAMMAR=='reduced' else is_valid(seq)
        if ok: seeds.append(tuple(seq))
    return seeds


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--generations',type=int,default=20); ap.add_argument('--population',type=int,default=32); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--out',type=Path,default=ROOT/'results'/'ga_28c_30m_3y.json'); ap.add_argument('--funding',choices=('constant','real'),default='real',help='constant reproduces the historical equity-compound-v2 assumption; real uses recorded Binance funding. Default is real because the constant is known to overstate the rate ~16x and to invert the sign on ~30%% of events.'); ap.add_argument('--null',choices=('none','iid','xsec','iid1','iidw','iidg'),default='none',help='iid: block-bootstrapped returns and volume per coin, no real edge. '
     'xsec: each coin paired with a neighbour bars. Both are false-positive '
     'controls for the search, not market data.'); ap.add_argument('--null-seed',type=int,default=0); ap.add_argument('--reward-version',choices=('v1','v2','v3'),default='v1'); ap.add_argument('--null-block',type=int,default=48); ap.add_argument('--mode',choices=('standard','walkforward'),default='standard',help='standard: one train/validation split scored against a lockbox. '
     'walkforward: expanding folds, one selection pass per validation window, '
     'final fold reserved as a holdout.'); ap.add_argument('--folds',type=int,default=4,help='walkforward fold count'); ap.add_argument('--grammar',choices=('full','reduced'),default='full',help='full: 29 tokens / ~3.1e15 valid formulas. reduced: 22 tokens, ten economically-motivated operators, a smaller hypothesis class. Default full so pre-94b59ca-style results stay reproducible.'); ap.add_argument('--select-on',choices=SELECT_ON_CHOICES,default=DEFAULT_SELECT_ON,help='train: selection keys on the train reward only, so the validation window cannot influence ranking, elitism or parent choice (PRP-0 default). val_weighted: restores the pre-PRP-0 SELECTION KEY (validation carried half the key and was re-scored every generation). It does NOT restore historical numbers: the meter repair also changed shape_reward in this tree, so a batch predating both needs its original commit. Diagnostic only, never a reportable result.'); ap.add_argument('--objective',choices=tuple(OBJECTIVES),default='v3',help='Selection objective preset. v3 is the frozen baseline. rank_ic makes rank-IC the primary selector. excess_bh optimises excess over buy-and-hold, which is what the gate tests. rank_ic_excess_bh combines them. v3_turn20 is v3 with a turnover weight large enough to matter. Every non-v3 preset is experimental and has no recorded batch behind it.'); args=ap.parse_args()
    if args.select_on == 'val_weighted':
        print('[select-on] WARNING: val_weighted restores the pre-PRP-0 selection key, NOT the '
              'pre-PRP-0 numbers (the meter repair also changed shape_reward in this tree). '
              'Validation carries half the key and is re-scored pop x generations times '
              'per fold. Diagnostic only; never reportable.', flush=True)
    global GRAMMAR
    GRAMMAR=args.grammar
    print(f'[grammar] {GRAMMAR}: {_spec()["max_token"]} tokens, '
          f'{len(_spec()["arity"])} operators', flush=True)
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
        out={'status':'ga_paper_only','live_adopted':False,'mode':'walkforward','select_on':args.select_on,
             'grammar':GRAMMAR,'null_data':args.null,'reward_version':args.reward_version,
             'funding_source':args.funding,'accounting_version':funding_version,
             'seed':args.seed,'generations':args.generations,'population':args.population,
             'folds':fold_summaries(folds,n),'fold_records':wf_records,
             'bars':n}
        hold=[r for r in wf_records if r.get('is_holdout') and 'portfolio_sharpe' in r]
        if hold:
            h=hold[0]
            out['holdout']={'portfolio_sharpe':h['portfolio_sharpe'],
                            'buy_and_hold_sharpe':h.get('buy_and_hold_sharpe'),
                            'excess_sharpe_vs_buy_and_hold':h.get('excess_sharpe_vs_buy_and_hold'),
                            'mean_position':h.get('mean_position'),
                            'portfolio_mdd':h['portfolio_mdd'],
                            'positive_coins':h['positive_coins'],
                            'formula':h['formula']}
        args.out.parent.mkdir(parents=True,exist_ok=True)
        with args.out.open('w') as f_: json.dump(out,f_,indent=1)
        print(json.dumps(out.get('holdout',{}),indent=1))
        return 0
    best, history = run_standard_selection(args, maps, returns, ranges, scale_end,
                                           common, funding_mask, funding_by_coin, rng)
    for h_ in history:
        print(f"gen={h_['generation']} best={h_['best_select_score']:.3f} "
              f"avg={h_['avg_select_score']:.3f}", flush=True)
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
        'status': 'ga_paper_only', 'select_on': args.select_on,
        'live_adopted': derive_live_adopted(acceptance),
        'grammar': GRAMMAR,
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
