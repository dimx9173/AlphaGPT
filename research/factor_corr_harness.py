#!/usr/bin/env python3
"""Measure factor-factor correlation for the 36-dimension expansion.

Round 2 selects factors on the argument that they are uncorrelated with the
existing 12 and with each other. That argument is testable, so it is tested
here rather than asserted: every shortlisted factor is computed on all 28 coins
and correlated against the existing 12 and against the rest of the shortlist.

The measurement is deliberately blunt. Spearman would be kinder to the heavy
tails that several of these factors have, and the gate is not asking whether a
factor is monotone in something useful, it is asking whether the factor is a
different thing. Pearson on a common mask is the version of that question that
cannot be talked around.

Pooling is across every coin and every bar. Per-coin correlation would hide the
exact failure this is looking for: a factor that is uncorrelated within each
coin but identical across coins is the same factor.

Everything is trailing-only. The correlation of a causal factor with a
non-causal one is a leak, not a correlation, so a causality guard runs over
each implementation before its correlations are read.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data" / "data_3y" / "30m"


def load_aligned():
    """Every coin on the common timestamp intersection, aligned as load_data does."""
    from research.universe_28c import COINS_28C
    raw, common = {}, None
    for c in COINS_28C:
        path = DATA_DIR / (c + ".csv")
        if not path.exists():
            continue
        with path.open() as f:
            rows = list(csv.DictReader(f))
        d = {k: [float(r[k]) for r in rows] for k in ("open", "high", "low", "close", "volume")}
        d["timestamp"] = [int(float(r["timestamp"])) for r in rows]
        raw[c] = d
        s = set(d["timestamp"])
        common = s if common is None else common & s
    common = sorted(common)
    out = {}
    for c, d in raw.items():
        idx = {t: i for i, t in enumerate(d["timestamp"])}
        import numpy as np
        import numpy as np
        out[c] = {k: np.array([d[k][idx[t]] for t in common], dtype=np.float64) for k in ("open", "high", "low", "close", "volume", "timestamp")}
    return out, common


def corr_matrix(series_by_coin):
    """Pearson correlation of every factor against every other, pooled.

    Factors are (n_coins, n_bars) float64 arrays. NaN is dropped pairwise
    rather than by a shared mask, because a shared mask over 24 factors with
    different warmup lengths throws away most of the sample and, worse, leaves
    the sample unevenly distributed across the later regime where several
    session-time factors carry all their variance.

    Returns (names, matrix, n_used) where n_used is the per-pair count.
    """
    names = sorted(series_by_coin)
    m = len(names)
    mat = np.full((m, m), np.nan)
    used = np.zeros((m, m), dtype=int)
    flat = {k: np.concatenate([series_by_coin[k][c] for c in sorted(series_by_coin[k])]) for k in names}
    for i in range(m):
        mat[i, i] = 1.0
        used[i, i] = flat[names[i]].size
        for j in range(i + 1, m):
            a, b = flat[names[i]], flat[names[j]]
            mask = np.isfinite(a) & np.isfinite(b)
            used[i, j] = used[j, i] = int(mask.sum())
            if used[i, j] < 500:
                continue
            x, y = a[mask], b[mask]
            sx, sy = x.std(), y.std()
            if sx < 1e-12 or sy < 1e-12:
                continue
            mat[i, j] = mat[j, i] = float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy))
    return names, mat, used


def causal_violation(fn, bars, n=4000, perturb_at=0.7):
    """Truncate the series and check the factor does not move.

    A causal factor computed on bars[:k] must equal the same factor computed on
    the full series and truncated, for every k past its warmup. A factor that
    peeks will change.
    """
    b = {k: v[:n] for k, v in bars.items()}
    full = np.asarray(fn(b), dtype=np.float64)
    k = int(perturb_at * n)
    b2 = dict(bars)
    b2 = {kk: (v[:k] if kk != "close" else v[:k]) for kk, v in bars.items()}
    b2 = {kk: v[:n] for kk, v in b2.items()}
    part = np.asarray(fn(b2), dtype=np.float64)
    n_common = min(len(full), len(part), n)
    diff = np.abs(full[:n_common] - part[:n_common])
    scale = float(np.nanstd(full[:n_common]))
    return bool(np.nanmax(diff) > 1e-9 * max(scale, 1e-12))
