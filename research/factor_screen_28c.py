#!/usr/bin/env python3
"""Do the 12 factors predict anything at all, once drift is removed?

The framework has never beaten a constant long, and the search is a drift
matching machine with a +5 to +13 Sharpe selection lift. Before spending
another search on top of it, test the building blocks. A search over a
grammar can only find what its features contain; if each feature is a dead
timing signal wearing a directional exposure, no search over that grammar can
produce edge, and the honest response is to change the factors rather than
the search.

The distinction that matters is between timing and direction. A factor that
is simply long more often when the market goes up has a positive correlation
with next returns and predicts nothing -- every long position in a drifting
market has that. The two are separated by centring the position:

    raw IC      = corr(p, r)
    demeaned IC = corr(p - mean(p), r)

The first credits being long. The second credits picking the right times.
Reporting only the first is how a factor gets mistaken for an edge.

Each factor is run through the live position path -- causal scale, tanh,
POSITION_CAP * smooth_causal(5), lagged one bar -- so the numbers are directly
comparable with every artifact in this project. Scoring is on train,
validation and lockbox separately, because a factor that works only in
sample is a fit.

The real data is scored against the frozen iid null, which is the control
that must not be beaten. A factor whose lockbox demeaned IC is the same on
both has found nothing: the iid null has no structure to find, so any match
is the null's own autocorrelation, not a forecast.
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import research.ga_28c_30m_3y as GA
from research.accounting_28c import (accounting_bar_returns, load_real_funding,
                                     metrics)
from research.causal_12f import FEATURE_NAMES
from research.splits_28c import search_splits, split_indices

POSITION_CAP = 0.25


def position_from_signal(sig, scale_end):
    fit = float(np.std(sig[:scale_end])) if scale_end > 0 else 0.0
    raw = np.tanh(sig / (fit + 1e-6)) if fit > 1e-8 else np.zeros_like(sig)
    p = POSITION_CAP * GA.smooth_causal(raw, 5)
    return np.roll(p, 1)


def score_factor(fi, null):
    common, maps, returns, mask = GA.load_data(null=null)
    funding = {c: load_real_funding(c, np.asarray(common, dtype=np.int64))
               for c in GA.COINS_28C}
    n = len(common)
    sp = search_splits(n, common[0], common[-1])
    lockbox = split_indices(n, common[0], common[-1])["lockbox"]
    scale_end = sp["train"][1]
    windows = {"train": sp["train"], "validation": sp["validation"],
               "lockbox": lockbox}

    out = {}
    for wname, (a, b) in windows.items():
        raw_ic, dem_ic, dem_t, nets, bh, mp = [], [], [], [], [], []
        for c in GA.COINS_28C:
            sig = np.asarray(maps[c][fi], dtype=np.float64)
            p = position_from_signal(sig, scale_end)
            r = np.asarray(returns[c], dtype=np.float64)
            pos, ret = p[a:b], r[a:b]
            if len(pos) < 50 or pos.std() < 1e-12 or ret.std() < 1e-12:
                continue
            raw_ic.append(np.corrcoef(pos, ret)[0, 1])
            # The correlation of the centred position, which is the one that
            # is not explained by holding a constant direction.
            dm = pos - pos.mean()
            if dm.std() > 1e-12:
                r_ic = np.corrcoef(dm, ret)[0, 1]
                dem_ic.append(r_ic)
                dem_t.append(r_ic * np.sqrt(len(dm) - 2))
            nets.append(accounting_bar_returns(pos, ret, GA.FEE,
                                              funding[c][a:b], GA.LEV))
            bh.append(accounting_bar_returns(
                np.full(len(pos), POSITION_CAP), ret, GA.FEE,
                funding[c][a:b], GA.LEV))
            mp.append(float(pos.mean()))
        if not nets:
            continue
        sh = metrics(np.mean(np.stack(nets), axis=0), common[a:b])["sharpe"]
        bsh = metrics(np.mean(np.stack(bh), axis=0), common[a:b])["sharpe"]
        out[wname] = {
            "raw_ic": float(np.nanmean(raw_ic)),
            "demeaned_ic": float(np.nanmean(dem_ic)) if dem_ic else 0.0,
            "demeaned_ic_t": float(np.nanmean(dem_t)) if dem_t else 0.0,
            "fractions_coins_ic_gt_0": float(np.mean(np.array(dem_ic) > 0)),
            "n_coins": len(nets),
            "sharpe": sh,
            "buy_and_hold_sharpe": bsh,
            "excess_vs_buy_and_hold": sh - bsh,
            "mean_position": float(np.mean(mp)),
        }
    return out


def main() -> int:
    modes = ["none", "iid"]
    data = {m: {f: score_factor(i, m)
                for i, f in enumerate(FEATURE_NAMES)} for m in modes}
    path = ROOT / "results" / "factor_screen.json"
    path.write_text(json.dumps(data, indent=1))

    for window in ("train", "validation", "lockbox"):
        print("\n=== %s ===" % window)
        print("%-12s %11s %11s %8s %10s %10s %8s %8s %8s %9s" % (
            "factor", "real rawIC", "real demIC", "real t", "iid demIC",
            "demIC gap", "real Sh", "B&H Sh", "excess", "%coins>0"))
        print("-" * 104)
        rows = []
        for f in FEATURE_NAMES:
            r = data["none"].get(f, {}).get(window)
            i = data["iid"].get(f, {}).get(window)
            if not r or not i:
                continue
            gap = r["demeaned_ic"] - i["demeaned_ic"]
            rows.append((f, r, i, gap))
            print("%-12s %+11.4f %+11.4f %+8.2f %+10.4f %+10.4f %+8.3f "
                  "%+8.3f %+8.3f %8.0f%%" % (
                      f, r["raw_ic"], r["demeaned_ic"], r["demeaned_ic_t"],
                      i["demeaned_ic"], gap, r["sharpe"],
                      r["buy_and_hold_sharpe"], r["excess_vs_buy_and_hold"],
                      100 * r["fractions_coins_ic_gt_0"]))
        if not rows:
            continue
        gaps = [g for _, _, _, g in rows]
        ts = [r["demeaned_ic_t"] for _, r, _, _ in rows]
        exc = [r["excess_vs_buy_and_hold"] for _, r, _, _ in rows]
        best = rows[int(np.argmax(gaps))]
        print("\n  demeaned IC gap real-minus-iid: median %+.4f  best %+.4f (%s)"
              "  sign-consistent %d/%d"
              % (st.median(gaps), max(gaps), best[0],
                 sum(1 for g in gaps if g > 0), len(gaps)))
        print("  |demeaned IC t| on real: max %.2f   (|t| > 1.96 per factor,"
              " plus a multiple-testing correction on top for %d factors)"
              % (max(abs(t) for t in ts), len(ts)))
        print("  excess over buy-and-hold: median %+.3f  best %+.3f  positive %d/%d"
              % (st.median(exc), max(exc), sum(1 for e in exc if e > 0), len(exc)))
    print("\nwrote %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
