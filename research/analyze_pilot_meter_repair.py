#!/usr/bin/env python3
"""Analyse the meter-repair pilot: 10 real vs 10 null walkforward searches.

Answers one question: did repairing the reward meter make the search more
trustworthy, measured as real-separates-from-null on a holdout scored once?

Compares three batches, all at the same settings (walkforward, 25 generations,
population 16, 10 folds, real funding, reward v1), so the only intended
difference between the first two is the meter:

  wf_real / wf_null        pre-repair, -50 cliff active, gap term dead
  pilot_real / pilot_null  post-repair, cliff replaced by a continuous penalty

IMPORTANT ON COMPARABILITY
-------------------------
--null iid rewrites returns and volume, so the null arm's buy-and-hold Sharpe is
measured on synthetic bars, not real ones. excess_vs_buy_and_hold is therefore
NOT comparable across arms and is reported for completeness only. The comparable
statistic is raw portfolio Sharpe, and the comparison that matters is real-vs-null
WITHIN a batch, where the null is scored on the synthetic world it was selected on.

Usage:
    python research/analyze_pilot_meter_repair.py
"""
from __future__ import annotations

import glob
import json
import math
import os
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

BATCHES = {
    "pre_repair": {"real": "results/wf_real/*.json", "null": "results/wf_null/*.json"},
    "post_repair": {"real": "results/pilot_meter_repair/pilot_real_seed*.json",
                    "null": "results/pilot_meter_repair/pilot_null_seed*.json"},
}

# The four conditions the pilot was gated on, in docs/GA_SEARCH_RELIABILITY_2026-10-01.md
GATE = {
    "a_real_median_beats_null": "real median Sharpe > null median Sharpe",
    "b_sign_test": "one-sided Mann-Whitney real > null, p < 0.10 (pilot leniency)",
    "c_breadth": ">= 6/10 real runs with excess_vs_buy_and_hold > 0",
    "d_null_not_rising_faster": "null best-of-10 must not rise faster than real best-of-10",
}


def load(pattern):
    rows = []
    for f in sorted(glob.glob(str(ROOT / pattern))):
        j = json.loads(Path(f).read_text())
        h = j.get("holdout") or {}
        rows.append({
            "seed": j.get("seed"),
            "sharpe": h.get("portfolio_sharpe"),
            "excess": h.get("excess_sharpe_vs_buy_and_hold"),
            "bh": h.get("buy_and_hold_sharpe"),
            "positive_coins": h.get("positive_coins"),
            "mean_position": h.get("mean_position"),
            "mdd": h.get("portfolio_mdd"),
        })
    return rows


def mannwhitney(a, b):
    comb = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    ranks = [0.0] * len(comb)
    i = 0
    tie = 0.0
    while i < len(comb):
        j = i
        while j + 1 < len(comb) and comb[j + 1][0] == comb[i][0]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = r
        t = j - i + 1
        if t > 1:
            tie += t ** 3 - t
        i = j + 1
    R1 = sum(r for r, (v, g) in zip(ranks, comb) if g == 0)
    n1, n2 = len(a), len(b)
    N = n1 + n2
    U1 = R1 - n1 * (n1 + 1) / 2
    mu = n1 * n2 / 2
    sd = math.sqrt(n1 * n2 / 12 * ((N + 1) - tie / (N * (N - 1)))) if N > 1 else 0
    if sd == 0:
        return U1, 1.0
    z = (abs(U1 - mu) - 0.5) / sd
    return U1, 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))


def _med(xs):
    """Median over the non-None entries only; None when a batch omits the field."""
    v = [x for x in xs if x is not None]
    return st.median(v) if v else None


def _fmt(x, nd):
    return "n/a" if x is None else f"{x:.{nd}f}"


def sign_test(diffs):
    n = len(diffs)
    k = sum(1 for x in diffs if x > 0)
    p = min(1.0, sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n * 2)
    return k, n, p


def main():
    report = {"gates": GATE, "batches": {}, "note": (
        "excess_vs_buy_and_hold is NOT comparable across arms: --null iid rewrites "
        "returns, so the null arm's benchmark is measured on synthetic bars. "
        "Cross-arm comparison uses raw portfolio Sharpe only.")}

    for batch, pats in BATCHES.items():
        R, N = load(pats["real"]), load(pats["null"])
        if not R or not N:
            print(f"{batch}: missing runs")
            continue
        rs = [r["sharpe"] for r in R]
        ns = [r["sharpe"] for r in N]
        diffs = [a - b for a, b in zip(rs, ns)]
        k, n, sp = sign_test(diffs)
        U, mp = mannwhitney(rs, ns)
        rex = sum(1 for r in R if (r["excess"] or 0) > 0)
        gap = st.median(rs) - st.median(ns)
        gates = {
            "a_real_median_beats_null": gap > 0,
            # The Mann-Whitney statistic is TWO-SIDED: it detects that the arms
            # differ, not which one is higher. p=0.038 with null above real is a
            # significant difference in the WRONG direction, so the gate must
            # require the direction explicitly or it will report PASS on a result
            # that is actually the strongest possible evidence against the search.
            "b_sign_test": (mp < 0.10) and (gap > 0),
            "c_breadth": rex >= 6,
            "d_null_not_rising_faster": False,   # filled after both batches load
        }
        report["batches"][batch] = {
            "real_median_sharpe": st.median(rs), "null_median_sharpe": st.median(ns),
            "real_range": [min(rs), max(rs)], "null_range": [min(ns), max(ns)],
            "real_best": max(rs), "null_best": max(ns),
            "median_gap": gap,
            "real_wins": k, "n": n, "sign_test_p": sp,
            "mannwhitney_U": U, "mannwhitney_p": mp,
            "real_positive_excess": rex,
            "real_positive_coins_median": _med([r["positive_coins"] for r in R]),
            "null_positive_coins_median": _med([r["positive_coins"] for r in N]),
            "real_mean_position_median": _med([r["mean_position"] for r in R]),
            "null_mean_position_median": _med([r["mean_position"] for r in N]),
            "gates": gates,
        }

    if len(report["batches"]) == 2:
        pre = report["batches"]["pre_repair"]
        post = report["batches"]["post_repair"]
        for b in report["batches"].values():
            b["gates"]["d_null_not_rising_faster"] = (
                (post["null_best"] - pre["null_best"]) <= (post["real_best"] - pre["real_best"]))
        report["meter_effect"] = {
            "real_median_delta": post["real_median_sharpe"] - pre["real_median_sharpe"],
            "null_median_delta": post["null_median_sharpe"] - pre["null_median_sharpe"],
            "gap_before": pre["median_gap"], "gap_after": post["median_gap"],
            "gap_improved": post["median_gap"] > pre["median_gap"],
            "real_wins_before": pre["real_wins"], "real_wins_after": post["real_wins"],
        }

    print("=" * 78)
    print("METER-REPAIR PILOT  (walkforward, 25 gen x pop 16, 10 folds, seeds 101-110)")
    print("=" * 78)
    for batch, b in report["batches"].items():
        print(f"\n--- {batch} ---")
        print(f"  real sharpe median {b['real_median_sharpe']:+8.3f}  "
              f"range [{b['real_range'][0]:+.2f}, {b['real_range'][1]:+.2f}]")
        print(f"  null sharpe median {b['null_median_sharpe']:+8.3f}  "
              f"range [{b['null_range'][0]:+.2f}, {b['null_range'][1]:+.2f}]")
        print(f"  median gap        {b['median_gap']:+8.3f}      real wins {b['real_wins']}/{b['n']}")
        print(f"  Mann-Whitney p    {b['mannwhitney_p']:.3f}       sign-test p {b['sign_test_p']:.3f}")
        print(f"  positive_coins    real {_fmt(b['real_positive_coins_median'], 0)}  "
              f"null {_fmt(b['null_positive_coins_median'], 0)}")
        print(f"  mean_position     real {_fmt(b['real_mean_position_median'], 3)}  "
              f"null {_fmt(b['null_mean_position_median'], 3)}")

    if "meter_effect" in report:
        me = report["meter_effect"]
        print("\n" + "=" * 78)
        print("EFFECT OF THE METER REPAIR")
        print("=" * 78)
        print(f"  real median Sharpe {me['real_median_delta']:+.3f}")
        print(f"  null median Sharpe {me['null_median_delta']:+.3f}")
        print(f"  real-null gap      {me['gap_before']:+.3f} -> {me['gap_after']:+.3f}  "
              f"(improved: {me['gap_improved']})")
        print(f"  real wins          {me['real_wins_before']}/10 -> {me['real_wins_after']}/10")

    print("\nGATE CHECK (post-repair batch)")
    pb = report["batches"]["post_repair"]
    why = {
        "a_real_median_beats_null":
            f"gap {pb['median_gap']:+.3f} (needs > 0)",
        "b_sign_test":
            f"MW p={pb['mannwhitney_p']:.3f}, gap {pb['median_gap']:+.3f} "
            f"(needs p<0.10 AND gap>0)",
        "c_breadth":
            f"{pb['real_positive_excess']}/10 runs with excess>0 (needs >=6)",
        "d_null_not_rising_faster":
            f"real best {pb['real_best']:+.2f}, null best {pb['null_best']:+.2f} (vs pre-repair)",
    }
    for name, ok in pb["gates"].items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {GATE[name]}")
        print(f"         -> {why[name]}")
    print("\nNote: excess_vs_buy_and_hold is not comparable across arms; see the module docstring.")

    p = ROOT / "results" / "pilot_meter_repair_analysis.json"
    p.write_text(json.dumps(report, indent=2, default=float))
    print("\nwrote", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
