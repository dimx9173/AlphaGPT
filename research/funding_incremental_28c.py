#!/usr/bin/env python3
"""Does funding carry anything the existing 12 factors do not already have?

funding_probe_28c.py showed FUND_X has an OOS rank-IC of the same order as RET
itself. That is not yet a reason to add a factor: RET is already in the set, so
a funding factor that merely re-expresses momentum adds nothing and costs a
feature slot.

This script settles it the honest way. For each coin and split it

  1. projects the funding factor onto the 12 existing causal factors,
  2. keeps only the orthogonal residual,
  3. measures the residual's IC against forward return.

A residual IC near zero means funding is redundant here and should not be added
to ADVANCED_FEATURE_NAMES. A residual IC that holds up out-of-sample means it is
genuinely new information.

Everything is fit on the training split only and then applied unchanged to
validation and oos, so the residual cannot borrow information from the future.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from research.universe_28c import COINS_28C  # noqa: E402
from research.causal_12f import causal_features  # noqa: E402
from data_pipeline.providers import funding_merge as fm  # noqa: E402
from research.funding_probe_28c import (  # noqa: E402
    funding_on_grid, load_prices, rolling_z, spearman, ic, EMBARGO,
)

DATA = ROOT / "data" / "data_3y" / "30m"
CACHE = ROOT / "data" / "data_3y" / "30m_causal"


def load_existing(coins, grid):
    """Existing 12 causal factors aligned to grid. Returns {coin: (12,n)}."""
    out = {}
    gridset = set(grid.tolist())
    for c in coins:
        cache = CACHE / f"{c}.npy"
        if cache.exists():
            # the causal cache is already stored as (n_features, n_bars)
            feat = np.load(cache)
            if feat.shape[0] != 12:
                feat = feat.T
        else:
            with (DATA / f"{c}.csv").open() as f:
                rows = list(csv.DictReader(f))
            d = {k: [float(r[k]) for r in rows] for k in ("open", "high", "low", "close", "volume")}
            feat = causal_features(d)
            if feat.shape[0] != 12:
                feat = feat.T
        with (DATA / f"{c}.csv").open() as f:
            rows = list(csv.DictReader(f))
        ts = np.array([int(r["timestamp"]) for r in rows], dtype=np.int64)
        close = np.array([float(r["close"]) for r in rows], dtype=np.float64)
        keep = np.array([t in gridset for t in ts])
        # feat rows and ts rows are 1:1 in causal_12f
        out[c] = (feat[:, keep].astype(np.float64), close[keep])
    return out


def build_funding_factors(coin, ts, close):
    fund, meta = funding_on_grid(coin, ts)
    ret = np.zeros(len(close))
    ret[1:] = close[1:] / close[:-1] - 1.0
    fund_z = rolling_z(fund, window=48)
    fund_x = np.nan_to_num(ret) * np.nan_to_num(fund)
    return {"FUND_RAW": fund, "FUND_Z": fund_z, "FUND_X": fund_x}, ret, meta


def residualise(f, X, tr_end):
    """Orthogonalize f against X (with intercept), fit on [0, tr_end).

    Returns the full-length residual, with bars before tr_end NaN where the fit
    is not applicable and bars where X is degenerate left untouched.
    """
    n = len(f)
    out = np.full(n, np.nan)
    m = ~np.isnan(f)
    if m.sum() < 50:
        return out
    # residualise expects X as (n_bars, n_features)
    X = np.asarray(X, dtype=np.float64)
    if X.shape[0] != n:
        X = X.T
    Xd = np.column_stack([np.ones(n), X])
    mtr = m.copy()
    mtr[tr_end:] = False
    if mtr.sum() < Xd.shape[1] + 10:
        return out
    try:
        coef, *_ = np.linalg.lstsq(Xd[mtr], f[mtr], rcond=None)
    except np.linalg.LinAlgError:
        return out
    out[m] = f[m] - Xd[m] @ coef
    return out


def main():
    coins = list(COINS_28C)
    prices, grid = load_prices(coins)
    n = len(grid)
    tr, va = int(n * 0.6), int(n * 0.8)
    splits = {
        "train": (0, tr - EMBARGO),
        "validation": (tr + EMBARGO, va - EMBARGO),
        "oos": (va + EMBARGO, n),
    }

    existing = load_existing(coins, grid)
    result = {
        "status": "probe_only",
        "bars": n,
        "question": "does funding add information beyond the existing 12 factors",
        "method": "fit on train split only, apply unchanged to validation and oos",
        "splits": {k: list(v) for k, v in splits.items()},
        "names": ["FUND_RAW", "FUND_Z", "FUND_X"],
        "raw": {},
        "residual": {},
    }

    raw_acc = {k: {s: {} for s in splits} for k in result["names"]}
    res_acc = {k: {s: {} for s in splits} for k in result["names"]}
    meta_by_coin = {}

    for c in coins:
        ts, close = prices[c]
        X, _ = existing[c]
        ff, ret, meta = build_funding_factors(c, ts, close)
        meta_by_coin[c] = meta
        for name, series in ff.items():
            resid = residualise(series, X, splits["train"][1])
            for s, (a, b) in splits.items():
                v = ic(series[a:b - 1], ret[a + 1:b])
                rv = resid[a:b - 1]
                raw_acc[name][s][c] = v
                res_acc[name][s][c] = ic(rv, ret[a + 1:b])

    for name in result["names"]:
        result["raw"][name] = {}
        result["residual"][name] = {}
        for s in splits:
            rv = raw_acc[name][s]
            sv = res_acc[name][s]
            result["raw"][name][s] = {
                "mean_ic": float(np.mean(list(rv.values()))) if rv else None,
                "mean_rank_ic": float(np.mean([spearman(*_pair(prices, c, name, splits, s, res=False)) for c in coins
                                              if spearman(*_pair(prices, c, name, splits, s, res=False)) is not None])),
            }
            result["residual"][name][s] = {
                "mean_ic": float(np.mean(list(sv.values()))) if sv else None,
                "median_ic": float(np.median(list(sv.values()))) if sv else None,
                "n_coins": len(sv),
                "per_coin": sv,
            }
        result["residual"][name]["r_squared_shared"] = {
            "note": "fraction of THIS factor's variance left unexplained by the existing 12, "
                    "pooled over coins and the train split. High means funding is largely "
                    "redundant with the current set; low means it carries new variation.",
            "train": _shared_r2(coins, prices, existing, splits["train"], (name,)),
        }

    result["verdict"] = {}
    for name in result["names"]:
        rv = result["raw"][name]
        sv = result["residual"][name]
        result["verdict"][name] = {
            "raw_oos_ic": rv["oos"]["mean_ic"],
            "residual_oos_ic": sv["oos"]["mean_ic"],
            "residual_oos_positive_coins": sum(1 for v in sv["oos"]["per_coin"].values() if v > 0),
            "residual_oos_n_coins": sv["oos"]["n_coins"],
            "add_to_vocab": bool(
                sv["oos"]["mean_ic"] is not None
                and abs(sv["oos"]["mean_ic"]) > 0.005
                and sum(1 for v in sv["oos"]["per_coin"].values() if v > 0) >= sv["oos"]["n_coins"] * 0.5
            ),
        }

    p = ROOT / "results" / "funding_incremental_28c.json"
    p.write_text(json.dumps(result, indent=2, default=float))
    print(f"bars={n}")
    print(f"{'factor':10} {'raw_oos_ic':>12} {'resid_oos_ic':>14} {'resid_pos':>10}")
    for name, v in result["verdict"].items():
        print(f"{name:10} {v['raw_oos_ic']:12.5f} {v['residual_oos_ic']:14.5f} "
              f"{v['residual_oos_positive_coins']:5}/{v['residual_oos_n_coins']:<4} "
              f"add={v['add_to_vocab']}")
    print()
    print("wrote", p)


def _pair(prices, coin, name, splits, s, res):
    ts, close = prices[coin]
    ff, ret, _ = build_funding_factors(coin, ts, close)
    a, b = splits[s]
    x = ff[name][a:b - 1]
    y = ret[a + 1:b]
    m = ~np.isnan(x) & ~np.isnan(y)
    return x[m], y[m]


def _shared_r2(coins, prices, existing, span, names=("FUND_RAW", "FUND_Z", "FUND_X")):
    """Pooled variance retained after projecting funding onto the existing 12.

    Near 1.0 means the existing factors already explain almost all of the
    funding factor's variation, i.e. funding is redundant. Near 0 means
    funding carries variation nothing in the current set explains.
    """
    kept, total = 0.0, 0.0
    a, b = span
    for c in coins:
        ts, close = prices[c]
        ff, _, _ = build_funding_factors(c, ts, close)
        X, _ = existing[c]
        for name in names:
            f = ff[name]
            # residualise on the full length so X stays aligned
            r = residualise(f, X, b)
            seg_f, seg_r = f[a:b], r[a:b]
            m = ~np.isnan(seg_f)
            mr = ~np.isnan(seg_r)
            if m.sum() < 50 or mr.sum() < 10:
                continue
            kept += float(np.var(seg_r[mr]))
            total += float(np.var(seg_f[m]))
    return float(kept / total) if total > 0 else None


if __name__ == "__main__":
    main()
