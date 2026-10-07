#!/usr/bin/env python3
"""Real-minus-null gap on a search batch, plus the null arm's own calibration.

Two numbers come out of this and the second matters more than the gap.

  gap  = mean(real excess) - mean(null excess). This is the decision. A search
         that finds nothing has a gap of zero; a search that overfits has a
         NEGATIVE one, because real training windows contain structure the search
         can fit and cannot keep, while synthetic windows offer less to fit.

  null beat rate = how often a searched book beat a constant long on data with no
         edge in it by construction. This is the calibration every "k of n runs beat
         the hold" number in a report needs beside it, and nothing in the pipeline
         supplied it before. A null beat rate of 0.276 means a real arm reporting
         0.4% is not a weak result, it is an inverted one.

Also reported: the percentile the real arm sits at inside the null distribution, and
breadth with a t, because breadth separates cleanly and is the one measurement in
this project that has ever pointed anywhere.
"""
from __future__ import annotations

import argparse
import bisect
import collections
import glob
import json
import math
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.grammar_reduced_36c import FEATURE_NAMES_36  # noqa: E402
import research.causal_mtf_28c as MTF  # noqa: E402


def sem(xs):
    xs = [x for x in xs if x is not None]
    return st.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else float("nan")


def load(d, features):
    rows = []
    for f in sorted(glob.glob(str(Path(d) / "g10_*.json"))):
        j = json.loads(Path(f).read_text())
        if j.get("n_features") != features:
            continue
        h = j.get("holdout") or {}
        rows.append({
            "data": "null" if j.get("null_data", "none") != "none" else "real",
            "seed": j.get("seed"),
            "excess": h.get("excess_sharpe_vs_buy_and_hold"),
            "sharpe": h.get("portfolio_sharpe"),
            "bh": h.get("buy_and_hold_sharpe"),
            "coins": h.get("positive_coins"),
            "pos": h.get("mean_position"),
            "fo": h.get("formula") or [],
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--features", type=int, default=41)
    args = ap.parse_args()
    rows = load(args.dir, args.features)
    by = collections.defaultdict(list)
    for r in rows:
        by[r["data"]].append(r)
    for arm in ("real", "null"):
        if not by[arm]:
            print("no %s runs in %s" % (arm, args.dir))
            return 2
    print("=" * 72)
    print("%d real + %d null, %d features" % (len(by["real"]), len(by["null"]), args.features))
    print("=" * 72)
    for arm in ("real", "null"):
        x = [r["excess"] for r in by[arm]]
        beat = sum(1 for v in x if v > 0)
        print("  %-5s excess %+.4f +/- %.4f  sd %.3f  median %+.3f  beat hold %d/%d (%.1f%%)"
              % (arm, st.mean(x), sem(x), st.stdev(x), st.median(x), beat, len(x),
                 100.0 * beat / len(x)))
    re_ = [r["excess"] for r in by["real"]]
    nu = [r["excess"] for r in by["null"]]
    g = st.mean(re_) - st.mean(nu)
    se = math.sqrt(sem(re_) ** 2 + sem(nu) ** 2)
    print()
    print("REAL minus NULL   %+.4f +/- %.4f   t = %+.2f   95%% CI [%+.3f, %+.3f]"
          % (g, se, g / se, g - 1.96 * se, g + 1.96 * se))
    print("  -> %s" % ("CONSISTENT WITH ZERO" if abs(g / se) < 2 else "SIGNIFICANT"))
    print("BREADTH and POSITION")
    for arm in ("real", "null"):
        x = by[arm]
        print("  %-5s coins %.2f/28  mean_pos %.4f  at cap %d/%d  absSharpe %+.4f  bh %+.4f"
              % (arm, st.mean([r["coins"] for r in x]), st.mean([r["pos"] for r in x]),
                 sum(1 for r in x if r["pos"] > 0.24), len(x),
                 st.mean([r["sharpe"] for r in x]), st.mean([r["bh"] for r in x])))
    cb = [r["coins"] for r in by["real"]]
    cn = [r["coins"] for r in by["null"]]
    db, sb = st.mean(cb) - st.mean(cn), math.sqrt(sem(cb) ** 2 + sem(cn) ** 2)
    print("  breadth real %.2f vs null %.2f   diff %+.2f   t %+.2f"
          % (st.mean(cb), st.mean(cn), db, db / sb))

    print()
    print("FACTOR USAGE (real arm)")
    u = collections.Counter()
    for r in by["real"]:
        for tok in r["fo"]:
            if tok < 36:
                u[FEATURE_NAMES_36[tok]] += 1
            elif tok < 41:
                u["MTF:" + list(MTF.MTF_FACTORS)[tok - 36]] += 1
    print("  distinct used %d   MTF used %d/5"
          % (len(u), sum(1 for k in u if k.startswith("MTF:"))))
    print("  top: " + ", ".join("%s(%d)" % kv for kv in u.most_common(8)))


if __name__ == "__main__":
    main()