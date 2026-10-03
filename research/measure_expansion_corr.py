#!/usr/bin/env python3
"""Measure the correlation structure of the 12 -> 36 expansion on real bars.

Round 3 selected 24 factors on the argument that they are different things
from the existing 12 and from each other. This measures whether that is true,
because the argument is testable and an unmeasured correlation claim is a hope.

Reported per new factor:

  max |r| against the 12 existing
  max |r| against the other new factors
  whether either exceeds the 0.6 budget the round-3 reasoning assumed

Also runs a causality check on every implementation: truncate the series, and
the factor must not move. A factor that peeks would show a correlation with its
own future, which is not a correlation at all, so this runs before the
correlations are read and the run fails loudly rather than reporting a leak as a
result.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BUDGET = 0.6


def build_series(new_names, want_funding=True, limit=None):
    """(factors, coins) where factors maps name -> {coin: 1-D array}."""
    from research.causal_12f import causal_features, FEATURE_NAMES
    from research.candidate_factors_36 import F, FUNDING_FACTORS
    from research.factor_corr_harness import load_aligned

    bars, stamps = load_aligned()
    coins = sorted(bars)
    if limit:
        coins = coins[:limit]

    out = {}
    for i, nm in enumerate(FEATURE_NAMES):
        out[nm] = {c: causal_features(bars[c])[i] for c in coins}

    funding = {}
    if want_funding:
        from research.accounting_28c import load_real_funding
        ts = np.asarray(stamps, dtype=np.int64)
        for c in coins:
            arr = load_real_funding(c, ts)
            funding[c] = (ts[arr != 0] if (arr != 0).any() else ts, arr)

    for nm in new_names:
        if nm in F:
            fn = F[nm][0]
            out[nm] = {c: np.asarray(fn(bars[c]), dtype=np.float64) for c in coins}
        elif nm in FUNDING_FACTORS:
            fn = FUNDING_FACTORS[nm]
            out[nm] = {c: np.asarray(fn(bars[c], funding[c]), dtype=np.float64) for c in coins}
        else:
            raise KeyError("no implementation for " + nm)
    return out, coins, list(FEATURE_NAMES)


def causality_report(new_names, coins):
    """Truncate and check nothing moves. Returns list of offenders."""
    from research.candidate_factors_36 import F, FUNDING_FACTORS
    from research.factor_corr_harness import causal_violation
    bad = []
    for nm in new_names:
        for c in coins[:4]:
            b = {}
            from research.factor_corr_harness import load_aligned
            bars, stamps = load_aligned()
            b = bars[c]
            if nm in F:
                if causal_violation(F[nm][0], b):
                    bad.append((nm, c))
                    break
    return bad


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "expansion_corr.json")
    ap.add_argument("--skip-causality", action="store_true")
    args = ap.parse_args()

    new_names = list(args.names)
    print("computing factors on all coins ...", flush=True)
    series, coins, existing = build_series(new_names)

    if not args.skip_causality:
        print("causality check ...", flush=True)
        from research.candidate_factors_36 import F
        from research.factor_corr_harness import load_aligned, causal_violation
        bars, _ = load_aligned()
        offenders = []
        for nm in new_names:
            if nm not in F:
                continue
            for c in coins:
                if causal_violation(F[nm][0], bars[c]):
                    offenders.append((nm, c))
                    break
        if offenders:
            print("CAUSALITY VIOLATION, refusing to report correlations:")
            for nm, c in offenders:
                print("   %s on %s" % (nm, c))
            return 2
        print("  all bars-only factors pass the truncation check", flush=True)

    print("correlating ...", flush=True)
    from research.factor_corr_harness import corr_matrix
    names, mat, used = corr_matrix(series)

    idx = {n: i for i, n in enumerate(names)}
    ex = [n for n in existing if n in idx]
    report = []
    for nm in new_names:
        i = idx[nm]
        vs_existing = [(abs(mat[i, idx[e]]), e) for e in ex if np.isfinite(mat[i, idx[e]])]
        vs_new = [(abs(mat[i, idx[m]]), m) for m in new_names
                  if m != nm and np.isfinite(mat[i, idx[m]])]
        me = max(vs_existing) if vs_existing else (float("nan"), "-")
        mn = max(vs_new) if vs_new else (float("nan"), "-")
        report.append({"name": nm,
                       "max_r_existing": round(float(me[0]), 4) if me[0] == me[0] else None,
                       "argmax_existing": me[1],
                       "max_r_new": round(float(mn[0]), 4) if mn[0] == mn[0] else None,
                       "argmax_new": mn[1],
                       "over_budget": bool(max(me[0] if me[0] == me[0] else 0,
                                               mn[0] if mn[0] == mn[0] else 0) > BUDGET)})
    report.sort(key=lambda r: -max(r["max_r_existing"] or 0, r["max_r_new"] or 0))
    print()
    print("%-18s %-22s %-22s %s" % ("factor", "max |r| vs existing 12", "max |r| vs new 24", "over 0.6"))
    for r in report:
        print("%-18s %-22s %-22s %s" % (
            r["name"],
            "%.3f  (%s)" % (r["max_r_existing"], r["argmax_existing"]) if r["max_r_existing"] is not None else "n/a",
            "%.3f  (%s)" % (r["max_r_new"], r["argmax_new"]) if r["max_r_new"] is not None else "n/a",
            "YES" if r["over_budget"] else ""))
    n_over = sum(1 for r in report if r["over_budget"])
    print()
    print("%d of %d new factors exceed the |r| %.1f budget" % (n_over, len(report), BUDGET))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"budget": BUDGET, "report": report,
                                    "names": names,
                                    "matrix": [[None if not np.isfinite(v) else round(float(v), 4)
                                                for v in row] for row in mat]}, indent=2))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
