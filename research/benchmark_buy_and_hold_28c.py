#!/usr/bin/env python3
"""Does the strategy beat simply holding a constant long position?

Every result this project has produced is measured against ZERO: a Sharpe of
+0.48 is called an edge, a Sharpe of -0.32 is called a failure, and the gate
screens on whether Sharpe clears 1.0. Zero is the wrong benchmark. A constant
levered long position has a Sharpe too, and on this universe over this window
it is a strong one, because crypto drifted.

The mechanism probe turned this up while chasing an unrelated puzzle -- why
the diagnostic nulls reported a HIGHER gross Sharpe (+3.03, +3.49, +5.07)
than the real data (+0.82). The answer is that the nulls' realized drift was
strongly positive over the lockbox and the search's formulas had collapsed to
a permanent long: 100% of bars long, mean position +0.17 to +0.22. What the
GA was reporting on those nulls was not a forecast. It was buy-and-hold.

That reframes everything. A search with no directional prior is a drift
matching machine: on a window where long paid, it finds long, and the result
is indistinguishable from holding. It is scored as a discovery.

So the correct benchmark is not zero, it is the constant long position the
same evaluation path would have produced. This script computes both and
reports the excess. A strategy that does not beat buy-and-hold has not found
anything, regardless of its absolute Sharpe.
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
from research.splits_28c import search_splits, split_indices

GRAMMAR = "reduced"
BATCHES = [
    ("gram_reduced_runs", "real_", "none"),
    ("gram_reduced_runs", "null_", "iid"),
    ("nullmode", "iid1", "iid1"),
    ("nullmode", "iidw", "iidw"),
    ("nullmode", "iidg", "iidg"),
]


def build(null):
    common, maps, returns, mask = GA.load_data(null=null)
    funding = {c: load_real_funding(c, np.asarray(common, dtype=np.int64))
               for c in GA.COINS_28C}
    n = len(common)
    lockbox = split_indices(n, common[0], common[-1])["lockbox"]
    scale_end = search_splits(n, common[0], common[-1])["train"][1]
    return common, maps, returns, lockbox, scale_end, funding


def position(formula, maps, coin, scale_end):
    sig = GA.formula_signal(formula, maps, coin, grammar=GRAMMAR)
    fit = float(np.std(sig[:scale_end])) if scale_end > 0 else 0.0
    raw = np.tanh(sig / (fit + 1e-6)) if fit > 1e-8 else np.zeros_like(sig)
    p = 0.25 * GA.smooth_causal(raw, 5)
    return np.roll(p, 1)          # p[0] becomes 0, matching the live path


def main() -> int:
    out = {}
    print("strategy vs a constant levered long, same bars, same fee, same funding")
    print()
    print(f"{'mode':6s} {'n':>3} {'B&H net':>9} {'strat net':>11} {'excess':>9} "
          f"{'meanPos':>9} {'%long':>7} {'%short':>8}  beats B&H?")
    print("-" * 82)
    for d, pre, null in BATCHES:
        bd = ROOT / "results" / d
        if not bd.exists():
            continue
        common, maps, returns, lockbox, scale_end, funding = build(null)
        a, b = lockbox
        ts = common[a:b]

        # Buy-and-hold: a constant long, levered identically. It never trades,
        # so it pays no turnover fee -- which is itself part of the point.
        bh_legs = []
        for c in GA.COINS_28C:
            r = returns[c][a:b]
            f = funding[c][a:b]
            p = np.ones(len(r))
            bh_legs.append(accounting_bar_returns(p, r, GA.FEE, f, GA.LEV))
        bh = metrics(np.mean(np.stack(bh_legs), axis=0), ts)["sharpe"]

        arts = [json.loads(f.read_text()) for f in
                sorted(bd.glob(f"{pre}*.json"))]
        nets, pos_mean, pct_long, pct_short = [], [], [], []
        for art in arts:
            legs = []
            for c in GA.COINS_28C:
                p = position(art["formula"], maps, c, scale_end)[a:b]
                legs.append((p, returns[c][a:b], funding[c][a:b]))
                pos_mean.append(float(p.mean()))
                pct_long.append(float(np.mean(p > 0.02)))
                pct_short.append(float(np.mean(p < -0.02)))
            nets.append(metrics(np.mean(np.stack([
                accounting_bar_returns(p, r, GA.FEE, f, GA.LEV)
                for p, r, f in legs]), axis=0), ts)["sharpe"])

        med = st.median(nets)
        excess = med - bh
        wins = sum(1 for x in nets if x > bh)
        out[null] = {
            "n": len(arts), "buy_and_hold_sharpe": bh,
            "strategy_median_sharpe": med, "excess": excess,
            "mean_position": st.mean(pos_mean),
            "pct_bars_long": st.mean(pct_long),
            "pct_bars_short": st.mean(pct_short),
            "runs_beating_buy_and_hold": f"{wins}/{len(arts)}",
        }
        verdict = f"{wins}/{len(arts)}" + (
            "  <-- at or below the trivial benchmark" if wins <= len(arts) / 2
            else "")
        print(f"{null:6s} {len(arts):3d} {bh:+9.3f} {med:+11.3f} {excess:+9.3f} "
              f"{st.mean(pos_mean):+9.4f} {100*st.mean(pct_long):6.1f}% "
              f"{100*st.mean(pct_short):7.1f}%  {verdict}")

    (ROOT / "results" / "buy_and_hold_benchmark.json").write_text(
        json.dumps(out, indent=1))
    print(f"\nwrote {ROOT/'results'/'buy_and_hold_benchmark.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
