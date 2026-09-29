#!/usr/bin/env python3
"""Is the search finding an edge, or is every formula positive on null?

The GA reports a median lockbox Sharpe of +0.82 on the iid null, where no
predictive relationship exists by construction. Two explanations fit that
number, and they imply opposite conclusions.

  A. The search is good at finding structure. The +0.82 is then a property of
     the search and the grammar, and the fix belongs in the search.
  B. The position construction is mildly profitable for almost any input.
     The search is then finding nothing: it picks the least bad member of a
     population whose members are all slightly positive, and the real-vs-null
     gap is a red herring about scoring rather than about edge.

The two are separated by evaluating RANDOM formulas, which carry no search,
no selection and no reward. If random formulas score positive on null, B
holds. If they sit at zero, A holds and the search really is manufacturing a
selection out of nothing.

This is the cheapest possible falsification of the whole framework, because
it reuses the exact evaluation path and costs no generations.

Usage:
    .venv2/bin/python research/probe_random_formulas_28c.py --n-formulas 300
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import research.ga_28c_30m_3y as GA
from research.splits_28c import search_splits, split_indices

# GRAMMAR is a module-level global that main() sets from --grammar. Token
# indices are grammar-specific, so a reduced formula evaluated with GRAMMAR
# left at its 'full' default is silently routed to the full evaluator and
# computes a different function. Set it explicitly rather than inheriting the
# import-time default.
GA.GRAMMAR = "reduced"


def probe(null: str, n_formulas: int, seed: int, null_seed: int) -> dict:
    # load_data returns (common, maps, returns, funding); funding is a
    # per-coin dict of recorded rates and doubles as the funding mask.
    common, maps, returns, funding_mask = GA.load_data(null=null, null_seed=null_seed)
    # load_data returns the 8-hour SETTLEMENT MASK, not per-coin rates. The
    # per-coin real rates are built separately; passing the mask where the
    # dict is expected raises an IndexError, and passing None silently falls
    # back to the superseded flat rate, which is the bug this project already
    # fixed once. Build them the way main() does.
    from research.accounting_28c import load_real_funding
    funding_by_coin = {c: load_real_funding(c, np.asarray(common, dtype=np.int64))
                       for c in GA.COINS_28C}
    n = len(common)
    contract = split_indices(n, common[0], common[-1])
    ranges = search_splits(n, common[0], common[-1])
    scale_end = ranges["train"][1]  # end index, not the (start, end) pair
    lockbox = contract["lockbox"]

    rng = random.Random(seed)
    vals, skipped = [], 0
    for _ in range(n_formulas):
        f = GA._random_formula_reduced(rng)
        if f is None:
            skipped += 1
            continue
        try:
            r = GA.evaluate(f, maps, returns, lockbox[0], lockbox[1],
                         funding_mask, scale_end, common,
                         funding_by_coin=funding_by_coin)
        except Exception:
            skipped += 1
            continue
        s = r.get("portfolio_sharpe")
        if s is None or not np.isfinite(s):
            skipped += 1
            continue
        vals.append(float(s))

    if not vals:
        return {"n_valid": 0, "n_skipped": skipped}
    pos = sum(v > 0 for v in vals)
    return {
        "n_valid": len(vals), "n_skipped": skipped,
        "median_sharpe": st.median(vals), "mean_sharpe": st.mean(vals),
        "sd": st.stdev(vals) if len(vals) > 1 else 0.0,
        "p10": float(np.quantile(vals, 0.10)),
        "p90": float(np.quantile(vals, 0.90)),
        "max": max(vals), "min": min(vals),
        "positive_rate": pos / len(vals),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-formulas", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--null-seed", type=int, default=0)
    ap.add_argument("--nulls", nargs="+", default=["none", "iid"])
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results" / "random_formula_probe.json")
    args = ap.parse_args()

    results = {}
    for null in args.nulls:
        d = probe(null, args.n_formulas, args.seed, args.null_seed)
        results[null] = d
        if not d["n_valid"]:
            print(f"{null:6s} no valid formulas")
            continue
        print(f"{null:6s} n={d['n_valid']:4d}  median={d['median_sharpe']:+.3f}  "
              f"mean={d['mean_sharpe']:+.3f}  sd={d['sd']:5.3f}  "
              f"p10={d['p10']:+.3f}  p90={d['p90']:+.3f}  "
              f"positive={d['positive_rate']*100:.0f}%")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
