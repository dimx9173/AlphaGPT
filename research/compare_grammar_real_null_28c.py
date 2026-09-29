#!/usr/bin/env python3
"""Real-vs-null comparison for a GA batch, with the null's own variance named.

Three protocols have now been run against a block-bootstrap null and all three
agree: the null outscores the real data. Before that is read as "the real data
is worse", two things have to be separated, because they imply opposite next
steps.

1. Search freedom. Handed the same seeds, generations and population, does a
   smaller hypothesis class manufacture less Sharpe on structureless data?
   Compared full (29 tokens, ~3.1e15 valid programs) against reduced (23
   tokens, 1.18e13, a 262x reduction).

2. The null's own spread. Ten searches over ONE null dataset vary only by
   search seed. Ten searches over TEN null datasets vary by both. If the second
   spread is much wider, the ten null runs in a batch are not ten observations
   and any p-value computed over them understates the real uncertainty.

Reporting only the median gap hides both. So this tool prints the per-protocol
medians, the gap, a rank test, and, when a null-variance set is present, the
dataset-to-dataset spread against the search-to-search spread.

Usage:
    .venv2/bin/python research/compare_grammar_real_null_28c.py \
        --real results/gram_reduced_runs --null-prefix null_ \
        --real-prefix real_ [--null results/nullvar --label ns]
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _sharpe_of(d):
    """Lockbox Sharpe, whether the artifact is a standard or walkforward run."""
    if "oos" in d and isinstance(d["oos"], dict):
        return d["oos"].get("portfolio_sharpe")
    hold = d.get("holdout")
    if isinstance(hold, dict):
        return hold.get("portfolio_sharpe")
    return None


def load(directory: Path, prefix: str | None = None, key: str = "seed") -> dict:
    out = {}
    for path in sorted(directory.glob("*.json")):
        if prefix and not path.name.startswith(prefix):
            continue
        try:
            d = json.loads(path.read_text())
        except Exception:
            continue
        s = _sharpe_of(d)
        if s is None:
            continue
        # key is the caller's grouping: search seed for a single batch, or
        # "dataset/search" when the point is between-dataset variance.
        if key == "dataset_search" and path.stem.count("_") >= 1:
            k = path.stem
        else:
            k = d.get(key, path.stem)
        out[k] = {"sharpe": s, "file": path.name, "seed": d.get("seed"),
                  "formula": d.get("formula"),
                  "verdict": (d.get("acceptance") or {}).get("verdict")
                  or d.get("gate", {}).get("verdict"),
                  "breadth": (d.get("oos") or {}).get("positive_coins"),
                  "grammar": d.get("grammar", "full (artifact predates the flag)"),
                  "funding": d.get("funding_source", "constant (artifact predates the flag)")}
    return out


def mann_whitney(a: list, b: list) -> tuple:
    """Two-sided normal approximation with tie-corrected ranks."""
    allv = sorted([(x, 0) for x in a] + [(x, 1) for x in b])
    n = len(allv)
    ranks = [0.0] * n
    i = 0
    tie_term = 0.0
    while i < n:
        j = i
        while j + 1 < n and allv[j + 1][0] == allv[i][0]:
            j += 1
        m = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = m
        t = j - i + 1
        tie_term += t ** 3 - t
        i = j + 1
    n1, n2 = len(a), len(b)
    ra = sum(ranks[k] for k in range(n) if allv[k][1] == 0)
    u1 = ra - n1 * (n1 + 1) / 2.0
    mu = n1 * n2 / 2.0
    var = (n1 * n2 / 12.0) * ((n1 + n2 + 1) - tie_term / (n1 + n2 - 1))
    if var <= 0:
        return u1, 0.0, 1.0
    z = (u1 - mu) / math.sqrt(var)
    return u1, z, math.erfc(abs(z) / math.sqrt(2))


def _f_sf(x, d1, d2):
    """Upper tail of the F distribution, via the regularised incomplete beta."""
    from math import lgamma, log, exp

    def betacf(a, b, x):
        MAXIT, EPS, FPMIN = 300, 3e-16, 1e-300
        qab, qap, qam = a + b, a + 1.0, a - 1.0
        c, d = 1.0, 1.0 - qab * x / qap
        if abs(d) < FPMIN:
            d = FPMIN
        d = 1.0 / d
        h = d
        for mm in range(1, MAXIT + 1):
            m2 = 2 * mm
            aa = mm * (b - mm) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d
            if abs(d) < FPMIN:
                d = FPMIN
            c = 1.0 + aa / c
            if abs(c) < FPMIN:
                c = FPMIN
            d = 1.0 / d
            h *= d * c
            aa = -(a + mm) * (qab + mm) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d
            if abs(d) < FPMIN:
                d = FPMIN
            c = 1.0 + aa / c
            if abs(c) < FPMIN:
                c = FPMIN
            d = 1.0 / d
            de = d * c
            h *= de
            if abs(de - 1.0) < EPS:
                break
        return h

    def betai(a, b, x):
        if x <= 0:
            return 0.0
        if x >= 1:
            return 1.0
        bt = exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * log(x) + b * log(1 - x))
        if x < (a + 1) / (a + b + 2):
            return bt * betacf(a, b, x) / a
        return 1.0 - bt * betacf(b, a, 1 - x) / b

    return betai(d2 / 2.0, d1 / 2.0, d2 / (d2 + d1 * x))


def one_way_anova(groups: dict) -> dict:
    """Split the null spread into between-dataset and within-dataset variance.

    A batch of ten null runs over ONE null dataset measures only the
    within-dataset term. Whether that is enough to stand in for the
    between-dataset term is an empirical question, and the answer is a test
    rather than an assumption. The variance share is reported as a point
    estimate because with only a handful of datasets the F test is weak even
    when the share is far from zero.
    """
    groups = {k: v for k, v in groups.items() if len(v) > 1}
    m = len(groups)
    k = min(len(v) for v in groups.values())
    if m < 2 or k < 2:
        return {}
    grand = st.mean([x for v in groups.values() for x in v])
    ssb = sum(len(v) * (st.mean(v) - grand) ** 2 for v in groups.values())
    ssw = sum(sum((x - st.mean(v)) ** 2 for x in v) for v in groups.values())
    msb, msw = ssb / (m - 1), ssw / (m * k - 1)
    if msw <= 0:
        return {}
    F = msb / msw
    return {"F": F, "p": 1.0 - _f_sf(F, m - 1, m * k - 1),
            "ms_dataset": msb, "ms_search": msw,
            "variance_share": msb / (msb + msw),
            "between_sd": st.stdev([st.mean(v) for v in groups.values()]),
            "within_sd": st.mean([st.stdev(v) for v in groups.values()]),
            "n_datasets": m, "k_per_dataset": k}


def describe(label, vals):
    return (f"{label:28s} n={len(vals):3d}  median={st.median(vals):+7.3f}  "
            f"mean={st.mean(vals):+7.3f}  sd={st.stdev(vals) if len(vals) > 1 else 0.0:6.3f}  "
            f"min={min(vals):+7.3f}  max={max(vals):+7.3f}  "
            f"pos={sum(v > 0 for v in vals):2d}/{len(vals)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", type=Path, required=True)
    ap.add_argument("--null", type=Path, required=True)
    ap.add_argument("--real-prefix", default="real_")
    ap.add_argument("--null-prefix", default="null_")
    ap.add_argument("--nullvar", type=Path, default=None,
                    help="extra null runs over several null DATASETS, to "
                         "separate between-dataset from between-search spread")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    real = load(args.real, args.real_prefix)
    null = load(args.null, args.null_prefix)
    if not real or not null:
        print("need at least one real and one null run")
        return 1

    rv = [v["sharpe"] for v in real.values()]
    nv = [v["sharpe"] for v in null.values()]

    print(f"real : {args.real} ({args.real_prefix}*)")
    print(f"null : {args.null} ({args.null_prefix}*)")
    grammars = {v["grammar"] for v in list(real.values()) + list(null.values())}
    fundings = {v["funding"] for v in list(real.values()) + list(null.values())}
    print(f"grammar(s)={sorted(grammars)}  funding(s)={sorted(fundings)}")
    print()
    print(describe("real  OOS Sharpe", rv))
    print(describe("null  OOS Sharpe", nv))
    gap = st.median(rv) - st.median(nv)
    print(f"{'median gap (real - null)':28s} {gap:+7.3f}")
    u, z, p = mann_whitney(rv, nv)
    print(f"Mann-Whitney U={u:.1f} z={z:+.3f} p={p:.4f}"
          f"   {'(not significant at 0.05)' if p >= 0.05 else '(significant at 0.05)'}")
    rp = sum(1 for v in real.values() if v["verdict"] == "pass")
    npass = sum(1 for v in null.values() if v["verdict"] == "pass")
    print(f"gate pass rate: real {rp}/{len(real)}   null {npass}/{len(null)}")
    print()

    result = {
        "real_median": st.median(rv), "null_median": st.median(nv),
        "median_gap": gap, "mannwhitney_u": u, "z": z, "p": p,
        "n_real": len(rv), "n_null": len(nv),
        "gate_pass_real": rp, "gate_pass_null": npass,
        "real_sharpes": rv, "null_sharpes": nv,
        "grammars": sorted(grammars), "fundings": sorted(fundings),
    }

    # Config matching is checked rather than assumed. The null-variance set was
    # first run at 40 generations / population 16 while the real and null
    # batches used 100 / 32, which makes the two spreads incomparable and
    # would quietly change the answer.
    def _cfg(d, pat):
        for f in sorted(d.glob(pat + "*.json")):
            try:
                a = json.loads(f.read_text())
            except Exception:
                continue
            return (a.get("generations"), a.get("population"), a.get("grammar"))
        return (None, None, None)

    rc, nc = _cfg(args.real, args.real_prefix), _cfg(args.null, args.null_prefix)
    print(f"config  real={rc}  null={nc}"
          f"{'' if rc == nc else '   <-- MISMATCH, medians are not comparable'}")
    print()

    if args.nullvar and args.nullvar.exists():
        nvset = load(args.nullvar, None, key="dataset_search")
        by_ds = defaultdict(list)
        for k, v in nvset.items():
            by_ds[k.split("_gs")[0]].append(v["sharpe"])
        ds_medians = [st.median(v) for v in by_ds.values() if v]
        if len(ds_medians) > 1:
            print("Null variance decomposition (how many null observations do "
                  "10 runs really give?)")
            for k in sorted(by_ds):
                print(f"  null dataset {k}: n={len(by_ds[k])} "
                      f"median={st.median(by_ds[k]):+.3f} vals="
                      f"{[round(x, 3) for x in sorted(by_ds[k])]}")
            an = one_way_anova(by_ds)
            if an:
                print(f"  one-way ANOVA: F({an['n_datasets']-1},"
                      f"{an['n_datasets']*an['k_per_dataset']-1}) = {an['F']:.3f}"
                      f"  p = {an['p']:.4f}")
                print(f"  variance share of the DATASET component = "
                      f"{an['variance_share']:.3f}")
                print()
                if an["p"] < 0.05:
                    print("  => WHICH null dataset was drawn explains a real share "
                          "of the spread. Ten null runs on one dataset are closer "
                          "to one observation repeated ten times than to ten "
                          "observations, so the rank test above understates the "
                          "uncertainty and must not be read as the size of the "
                          "real-vs-null gap.")
                else:
                    print("  => at this many datasets, dataset identity is not "
                          "distinguishable from search randomness. The point "
                          f"estimate is still {an['variance_share']:.0%} of the "
                          "variance, so the test is underpowered rather than "
                          "evidence of independence; read the share, not only "
                          "the p-value.")
                print()
                result["null_variance"] = an
            result["null_dataset_medians"] = ds_medians

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=1))
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
