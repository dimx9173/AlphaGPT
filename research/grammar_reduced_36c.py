"""Reduced operator set indexed for a 36-factor feature space.

The reduced grammar in grammar_reduced_28c places its 11 operators at token
indices 12..22, immediately above the 12 factors. Those indices are absolute,
not relative, so a 36-factor feature space needs the same operators at 36..46.

Rather than duplicate the operator table, this module builds it from the 12-factor
one by shifting every operator key by (n_features - 12). The operator set, the
arity table, the evaluator and the validator are therefore identical to the
12-factor grammar by construction, and a formula that decodes the same way under
one is guaranteed to evaluate the same way under the other.

Token 0..11 keep their meaning. Token 12..35 are the 24 additions from
causal_36f.SHIPPED_24 in order. This is what makes the old artifacts decodable:
indices 0..11 are the original factors, so a formula recorded against the 12-factor
grammar still means the same thing here.
"""
from __future__ import annotations

import numpy as np

from research.grammar_reduced_28c import (
    REDUCED_OPS as _OPS_12,
    REDUCED_ARITY as _ARITY_12,
    _apply_reduced,
)
from research.causal_12f import _delay, _corr  # noqa: F401


FEATURE_COUNT = 36
_SHIFT = FEATURE_COUNT - 12

# Same 11 operators, same arities, re-indexed to sit above the 36 factors.
REDUCED_OPS = {k + _SHIFT: v for k, v in _OPS_12.items()}
REDUCED_ARITY = {k + _SHIFT: v for k, v in _ARITY_12.items()}
REDUCED_TOKENS = FEATURE_COUNT + len(REDUCED_OPS)          # 36 + 11 = 47
REDUCED_OP_COUNT = len(REDUCED_OPS)


def _names():
    from research.causal_12f import FEATURE_NAMES as OLD12
    from research.causal_36f import SHIPPED_24
    return list(OLD12) + list(SHIPPED_24)


FEATURE_NAMES_36 = _names()

REDUCED_TOKEN_NAMES = {**{i: FEATURE_NAMES_36[i] for i in range(FEATURE_COUNT)},
                       **{k: v[0] for k, v in REDUCED_OPS.items()}}


def decode(formula, token_names=None):
    names = token_names or REDUCED_TOKEN_NAMES
    return " ".join(names.get(int(t), "UNK%d" % int(t)) for t in formula)


def is_reduced_token(t: int) -> bool:
    return 0 <= t < REDUCED_TOKENS


def validate_reduced(formula) -> bool:
    depth = 0
    for t in formula:
        t = int(t)
        if not is_reduced_token(t):
            return False
        if t < FEATURE_COUNT:
            depth += 1
        else:
            arity = REDUCED_ARITY.get(t)
            if arity is None or depth < arity:
                return False
            depth = depth - arity + 1
    return depth == 1


def evaluate_formula_reduced(formula, features):
    """Evaluate on a (36, n_bars) feature matrix. Returns None on any illegal token."""
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
