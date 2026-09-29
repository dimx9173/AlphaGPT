"""Reduced operator set with economic rationale.

The full grammar has 29 tokens (12 factors + 17 operators), admitting roughly
3.1e15 syntactically valid 12-token programs. Ten independent searches over it
return ten disjoint formulas, and a block-bootstrap null over the same
protocol produces a *higher* median lockbox Sharpe than the real data, so the
full grammar's search space is large enough to manufacture the result it is
asked to find. This module keeps the ten operator families that carry an
explicit economic argument and drops the seven that only reshuffle a
cross-section or pattern-match a single series.

Kept, 11 operators in 10 families (economic rationale):
  ADD SUB MUL DIV     --  arithmetic combination of two factors (4 tokens)
  NEG SIGN            --  direction inversion, direction extraction
  GATE                --  regime-conditional logic (if A>0 then B else C)
  DECAY               --  exponential smoothing, i.e. an explicit mean-reversion
  DELAY1              --  one-bar causal lag
  DELTA               --  first difference, i.e. explicit change
  CORR                --  cross-factor rolling correlation

Removed (no clear economic prior):
  ABS        --  magnitude without direction is ambiguous
  JUMP       --  3-sigma jump detection, purely statistical
  MAX3       --  max of 3 lags, purely pattern matching
  ZSCORE     --  standardisation, factors already normalised
  RANK       --  rank transform, purely statistical
  TS_RANK    --  time-series rank, purely statistical

This reduces the operator count from 17 to 11, and total tokens from 29 to 23.
"""
from __future__ import annotations

import numpy as np

# Feature indices (0-11): same as the full contract
RET = 0
LIQ = 1
PRESSURE = 2
FOMO = 3
PUMP = 4
LOG_VOL = 5
VOL_CLUST = 6
MOM_REV = 7
RSI = 8
HL_RANGE = 9
CLOSE_POS = 10
VOL_TREND = 11

FEATURE_COUNT = 12

# Reduced operator set with economic rationale
REDUCED_OPS = {
    # Arithmetic (4)
    12: ("ADD", 2),
    13: ("SUB", 2),
    14: ("MUL", 2),
    15: ("DIV", 2),
    # Direction (2)
    16: ("NEG", 1),
    17: ("SIGN", 1),
    # Conditional (1)
    18: ("GATE", 3),
    # Temporal / mean-reversion (2)
    19: ("DECAY", 1),
    20: ("DELAY1", 1),
    # Change / correlation (2)
    21: ("DELTA", 1),
    22: ("CORR", 2),
}

REDUCED_TOKENS = FEATURE_COUNT + len(REDUCED_OPS)  # 12 + 11 = 23 (tokens 0..22)

# Map from reduced token ID to name (for decoding)
REDUCED_TOKEN_NAMES = {**{i: f"F{i}" for i in range(FEATURE_COUNT)},
                       **{k: v[0] for k, v in REDUCED_OPS.items()}}

# Arity for the reduced set
REDUCED_ARITY = {k: v[1] for k, v in REDUCED_OPS.items()}

# The full contract's token map, for decoding artifacts produced with the
# full grammar. Kept so old artifacts remain readable.
FULL_OPS = {
    12: ("ADD", 2), 13: ("SUB", 2), 14: ("MUL", 2), 15: ("DIV", 2),
    16: ("NEG", 1), 17: ("ABS", 1), 18: ("SIGN", 1), 19: ("GATE", 3),
    20: ("JUMP", 1), 21: ("DECAY", 1), 22: ("DELAY1", 1), 23: ("MAX3", 1),
    24: ("ZSCORE", 1), 25: ("RANK", 1), 26: ("TS_RANK", 1),
    27: ("DELTA", 1), 28: ("CORR", 2),
}
FULL_TOKEN_NAMES = {**{i: f"F{i}" for i in range(FEATURE_COUNT)},
                    **{k: v[0] for k, v in FULL_OPS.items()}}


def decode(formula, token_names=None):
    """Decode a formula sequence to human-readable RPN string."""
    names = token_names or REDUCED_TOKEN_NAMES
    return " ".join(names.get(int(t), f"UNK{t}") for t in formula)


def is_reduced_token(t: int) -> bool:
    """Check if a token belongs to the reduced grammar."""
    return 0 <= t < REDUCED_TOKENS


def validate_reduced(formula) -> bool:
    """RPN validity check against the reduced grammar."""
    depth = 0
    for t in formula:
        t = int(t)
        if not is_reduced_token(t):
            return False
        if t < FEATURE_COUNT:
            depth += 1
        else:
            arity = REDUCED_ARITY[t]
            if depth < arity:
                return False
            depth = depth - arity + 1
    return depth == 1


# Convenience: number of operators in the reduced set
REDUCED_OP_COUNT = len(REDUCED_OPS)


# ---------------------------------------------------------------------------
# Reduced-grammar evaluator
# ---------------------------------------------------------------------------
# The full evaluator in causal_12f dispatches on operator NAME, not on token
# index, so a reduced-grammar token would be silently evaluated as the wrong
# function if routed through it. This evaluator is separate to make that
# impossible.
from research.causal_12f import _delay, _corr  # noqa: E402  (import at module bottom to avoid cycle)


def _apply_reduced(op, args):
    if op == 'ADD':    return args[0] + args[1]
    if op == 'SUB':    return args[0] - args[1]
    if op == 'MUL':    return args[0] * args[1]
    if op == 'DIV':    return args[0] / (args[1] + 1e-6)
    if op == 'NEG':    return -args[0]
    if op == 'SIGN':   return np.sign(args[0])
    if op == 'GATE':   return np.where(args[0] > 0, args[1], args[2])
    if op == 'DECAY':  return args[0] + 0.8 * _delay(args[0], 1) + 0.6 * _delay(args[0], 2)
    if op == 'DELAY1': return _delay(args[0], 1)
    if op == 'DELTA':  return args[0] - _delay(args[0], 1)
    if op == 'CORR':   return _corr(args[0], args[1])
    raise ValueError(op)


def evaluate_formula_reduced(formula, features):
    """Evaluate a reduced-grammar formula on a feature matrix.

    Returns None on any illegal or out-of-grammar token, matching the full
    evaluator's contract.
    """
    stack = []
    for token in formula:
        token = int(token)
        if token < FEATURE_COUNT:
            stack.append(features[token])
        else:
            if token not in REDUCED_OPS:
                return None
            op, arity = REDUCED_OPS[token]
            if len(stack) < arity:
                return None
            args = [stack.pop() for _ in range(arity)][::-1]
            stack.append(_apply_reduced(op, args))
    if len(stack) != 1:
        return None
    return np.nan_to_num(stack[0], nan=0.0, posinf=5.0, neginf=-5.0)
