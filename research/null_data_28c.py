#!/usr/bin/env python3
"""Null data for the 28c contract: same statistics, no real relationship to predict.

The point is to answer one question about the search itself, not about the
market. Ten real searches produced a median lockbox Sharpe of +0.78 and a
train-to-lockbox correlation of -0.82. Before tuning a reward function against
that, it is worth knowing whether the search reports an edge when none exists.
If it does, no reward change can be evaluated until the process is fixed,
because the measurement itself would be untrustworthy.

A null must be fair. If the synthetic series were obviously unrealistic the
search would trivially fail and the test would prove nothing. So each null
preserves what the strategy actually consumes and destroys only the
relationship that could be traded:

  iid    block-bootstrapped log returns and volume, drawn independently per
         coin. Preserves the marginal return distribution, its fat tails,
         volatility clustering, and the return/volume relationship that the
         FOMO and LOG_VOL factors read. Destroys cross-coin correlation and
         any real predictability.
  xsec   the real series for one coin paired with another coin's bars.
         Preserves each coin's own time series intact but breaks the
         cross-sectional relationship a multi-coin portfolio depends on.

  none   the real contract, for baseline comparison.

Everything downstream is untouched: same features, same grammar, same splits,
same accounting. Only the data changes, so a difference in results is
attributable to the data and not to a different code path.
"""
from __future__ import annotations

import numpy as np


def _block_starts(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Ordered block boundaries covering [0, n)."""
    if block <= 1:
        return np.arange(n)
    return np.arange(0, n, block)


def block_bootstrap_series(log_ret: np.ndarray, volume: np.ndarray,
                           block: int, rng: np.random.Generator):
    """Resample (return, volume) pairs in contiguous blocks.

    Returns and volumes are resampled together and contiguously, because the
    volume factors read the volume *at* a return, not a marginal one. Drawing
    them separately would break a relationship the search can see, and the
    result would flatter the real data by comparison.
    """
    n = len(log_ret)
    starts = _block_starts(n, block, rng)
    need = int(np.ceil(n / block))
    picked = starts[rng.integers(0, len(starts), need)]
    idx = np.concatenate([np.arange(s, min(s + block, n)) for s in picked])[:n]
    if len(idx) < n:                      # tail safety if the last block is short
        idx = np.concatenate([idx, np.arange(n - (n - len(idx)), n)])
    return log_ret[idx], volume[idx]


def make_null(kind: str, data: dict, seed: int, block: int = 48):
    """Rebuild one coin's bars from a null resample of its own series.

    ``data`` carries aligned open/high/low/close/volume arrays.
    Returns the same keys so callers cannot tell the difference.
    """
    rng = np.random.default_rng(seed)
    close = np.asarray(data['close'], dtype=np.float64)
    volume = np.asarray(data['volume'], dtype=np.float64)
    n = len(close)
    log_ret = np.diff(np.log(np.maximum(close, 1e-12)), prepend=np.log(close[0]))
    r, v = block_bootstrap_series(log_ret, volume, block, rng)
    new_close = close[0] * np.exp(np.cumsum(r))
    new_close = np.maximum(new_close, 1e-12)
    # Reconstruct bars around the new closes so high/low stay consistent.
    prev = np.concatenate(([new_close[0]], new_close[:-1]))
    open_ = np.where(rng.random(n) < 0.5, prev, new_close)
    hi = np.maximum(open_, new_close) * (1.0 + rng.random(n) * 0.004)
    lo = np.minimum(open_, new_close) * (1.0 - rng.random(n) * 0.004)
    return {'open': open_, 'high': hi, 'low': lo, 'close': new_close,
            'volume': np.maximum(v, 0.0), 'timestamp': list(data['timestamp'])}


def permute_cross_sectional(data_by_coin: dict, coins: list, seed: int) -> dict:
    """Keep every coin's own time series, rotate which coin it is paired with.

    Each coin's bars move to a different coin's slot. A single-coin factor
    still behaves normally; the cross-coin structure that a portfolio of 28
    positions depends on is gone.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(coins))
    out = {}
    for i, c in enumerate(coins):
        out[c] = data_by_coin[coins[order[i]]]
    return out
