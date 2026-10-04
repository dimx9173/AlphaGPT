#!/usr/bin/env python3
"""Rank/IC probe for funding-rate factors on the 28-coin 3y universe.

What this measures
------------------
Funding is a perp-market quantity: longs pay shorts when it is positive. The
existing 12 price/volume factors have no funding input at all, so this asks one
question only: does funding carry forward-return information that the current
factors do not already have?

Three candidate transforms, each computed causally from the 8h settled series:

  FUND_RAW     the rate itself
  FUND_Z       its rolling z-score against its own trailing history
  FUND_X       price return x funding, i.e. funding-weighted momentum

The third is the one expected to matter: funding is a crowding gauge, and
crowding should modulate momentum rather than predict on its own.

Coins with no funding history are SKIPPED, not zero-filled. A coin that lacks
the factor is reported as absent and excluded from the mean, because zero is a
real funding rate (Binance funds flat fairly often) and pretending otherwise
would quietly invent data. This follows the project rule that absence of
evidence is not evidence of absence.

Splits mirror baseline_rank_ic_28c: 60/20/20 with a 200-bar embargo, so these
numbers are comparable with the existing factor screen.

Usage:
    python research/funding_probe_28c.py
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
from data_pipeline.providers import funding_merge as fm  # noqa: E402

DATA = ROOT / "data" / "data_3y" / "30m"
BAR_MS = 30 * 60 * 1000
FUND_HOURS = (0, 8, 16)
EMBARGO = 200

# Coins ByKaranteli has no derivatives row for. They still get Binance funding,
# so they are NOT excluded from the price study; only the OI/positioning/
# liquidation factors are unavailable for them.
NO_BYKAR = {"ETC", "ATOM", "KAS", "ICP", "POL", "RENDER"}


def load_prices(coins):
    """Return ({coin: ts_ms array}, {coin: close array}) on the common grid."""
    series = {}
    common = None
    for c in coins:
        with (DATA / f"{c}.csv").open() as f:
            rows = list(csv.DictReader(f))
        ts = np.array([int(r["timestamp"]) for r in rows], dtype=np.int64)
        close = np.array([float(r["close"]) for r in rows], dtype=np.float64)
        series[c] = (ts, close)
        s = set(ts.tolist())
        common = s if common is None else common & s
    common = sorted(common)
    out = {}
    for c in coins:
        ts, close = series[c]
        idx = np.searchsorted(ts, np.array(common, dtype=np.int64))
        out[c] = (ts[idx], close[idx])
    return out, np.array(common, dtype=np.int64)


def funding_on_grid(coin, grid_ms):
    """Align settled 8h funding onto the bar grid.

    Returns an array aligned to grid_ms, using the rate of the most recent
    settlement at or before each bar. Funding is a step function between
    settlements, so carrying the last value forward is the correct treatment,
    not a smoothing choice. Bars before the first settlement are NaN.
    """
    table, meta = fm.load_binance_funding(coin)
    if not table:
        return np.full(len(grid_ms), np.nan), meta
    ts = np.array(sorted(table), dtype=np.int64)
    vals = np.array([table[int(t)] for t in ts], dtype=np.float64)
    pos = np.searchsorted(ts, grid_ms, side="right") - 1
    out = np.full(len(grid_ms), np.nan)
    ok = pos >= 0
    out[ok] = vals[pos[ok]]
    meta = dict(meta, n_aligned=int(ok.sum()))
    return out, meta


def rolling_z(x, window):
    """Causal rolling z-score. Only uses the past `window` bars including the
    current one, and emits NaN until the window is full."""
    n = len(x)
    out = np.full(n, np.nan)
    if n < window:
        return out
    csum = np.cumsum(np.nan_to_num(x, nan=0.0))
    csum2 = np.cumsum(np.nan_to_num(x * x, nan=0.0))
    for i in range(window - 1, n):
        seg = x[i - window + 1: i + 1]
        if np.isnan(seg).any():
            continue
        m = (csum[i] - (csum[i - window] if i >= window else 0.0)) / window
        v = (csum2[i] - (csum2[i - window] if i >= window else 0.0)) / window - m * m
        sd = np.sqrt(max(v, 0.0))
        if sd > 1e-12:
            out[i] = (x[i] - m) / sd
    return out


def _paired(a, b):
    """Drop bars where either side is NaN, so a and b stay aligned."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    m = ~np.isnan(a) & ~np.isnan(b)
    return a[m], b[m]


def ic(a, b):
    """Pearson correlation of the factor against forward return."""
    a, b = _paired(a, b)
    if len(a) < 20 or a.std() < 1e-12 or b.std() < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b):
    """Rank IC. Funding is heavy-tailed and clips at exchange bounds, so the
    rank correlation is the more honest of the two numbers for this factor."""
    a, b = _paired(a, b)
    if len(a) < 20:
        return None
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def build_factors(coin, grid_ms, close):
    """Return {name: array} for one coin, plus its funding metadata."""
    fund, meta = funding_on_grid(coin, grid_ms)
    ret = np.zeros(len(close))
    ret[1:] = close[1:] / close[:-1] - 1.0

    fund_z = rolling_z(fund, window=48)          # 48 * 30m = 24h
    # funding-weighted momentum: same sign as momentum when funding is positive
    fund_x = np.nan_to_num(ret) * np.nan_to_num(fund)

    return {
        "FUND_RAW": fund,
        "FUND_Z": fund_z,
        "FUND_X": fund_x,
        "RET": ret,
    }, meta, fund


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

    factors = {}
    metas = {}
    coverage = {}
    for c in coins:
        ts, close = prices[c]
        f, meta, fund = build_factors(c, ts, close)
        factors[c] = f
        metas[c] = meta
        valid = ~np.isnan(fund)
        coverage[c] = {
            "bars": n,
            "funding_bars": int(valid.sum()),
            "coverage_frac": float(valid.sum() / n),
            "first_bar_ts": int(ts[valid][0]) if valid.any() else None,
            "last_bar_ts": int(ts[valid][-1]) if valid.any() else None,
            "bykaranteli": c not in NO_BYKAR,
        }

    out = {
        "status": "probe_only",
        "universe": len(coins),
        "bars": n,
        "grid_start": int(grid[0]),
        "grid_end": int(grid[-1]),
        "splits": {k: list(v) for k, v in splits.items()},
        "embargo": EMBARGO,
        "names": ["FUND_RAW", "FUND_Z", "FUND_X", "RET"],
        "coverage": coverage,
        "funding_meta": metas,
        "factors": {},
    }

    for name in out["names"]:
        row = {"per_coin": {}, "note": "coins with <20 valid bars in a split are excluded, not zero-filled"}
        for split, (a, b) in splits.items():
            vals, rvals = {}, {}
            for c in coins:
                x = factors[c][name][a:b - 1]
                y = factors[c]["RET"][a + 1:b]
                v = ic(x, y)
                rv = spearman(x, y)
                if v is not None:
                    vals[c] = v
                if rv is not None:
                    rvals[c] = rv
            if not vals:
                row[split] = {"mean_ic": None, "median_ic": None,
                              "positive_coins": 0, "n_coins": 0, "per_coin": {}}
                continue
            row[split] = {
                "mean_ic": float(np.mean(list(vals.values()))),
                "median_ic": float(np.median(list(vals.values()))),
                "mean_rank_ic": float(np.mean(list(rvals.values()))) if rvals else None,
                "median_rank_ic": float(np.median(list(rvals.values()))) if rvals else None,
                "positive_coins": sum(1 for v in vals.values() if v > 0),
                "n_coins": len(vals),
                "excluded_coins": sorted(set(coins) - set(vals)),
                "per_coin": vals,
                "per_coin_rank": rvals,
            }
        out["factors"][name] = row

    rank = sorted(
        [k for k in out["factors"]],
        key=lambda k: (out["factors"][k]["validation"]["mean_rank_ic"]
                       if out["factors"][k]["validation"]["mean_rank_ic"] is not None else -9),
        reverse=True,
    )
    out["ranking_by_validation_rank_ic"] = rank
    out["verdict"] = {
        name: {
            "validation_rank_ic": out["factors"][name]["validation"]["mean_rank_ic"],
            "oos_rank_ic": out["factors"][name]["oos"]["mean_rank_ic"],
            "oos_positive_coins": out["factors"][name]["oos"]["positive_coins"],
            "oos_n_coins": out["factors"][name]["oos"]["n_coins"],
        }
        for name in rank
    }

    p = ROOT / "results" / "funding_probe_28c.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"bars={n} grid={datetime.fromtimestamp(grid[0]/1000, tz=timezone.utc):%Y-%m-%d}"
          f"..{datetime.fromtimestamp(grid[-1]/1000, tz=timezone.utc):%Y-%m-%d}")
    cov = [v["coverage_frac"] for v in coverage.values()]
    print(f"funding coverage: min={min(cov):.3f} median={np.median(cov):.3f} max={max(cov):.3f}")
    print()
    print("ranking by validation rank-IC:", rank)
    print()
    print(json.dumps(out["verdict"], indent=2))
    print()
    print("wrote", p)


if __name__ == "__main__":
    main()
