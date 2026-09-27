#!/usr/bin/env python3
"""Compare independent GA runs against each other, not just against a bar.

One search producing a good number proves nothing. A thousand formulas are
evaluated inside a single run and the best one is reported, so the winner's
score is a maximum over many draws rather than a single observation. Ten
independent runs expose that distribution: if the spread runs from clearly
negative to clearly positive, the search is selecting noise, and the headline
number describes the maximum of the noise rather than an edge.

Reported per run: seed, formula, lockbox Sharpe, gate verdict, and whether the
result rests on a placeholder. The funding basis is read from each artifact
rather than assumed, because a constant-funded score and a real-funded score
are not the same measurement and averaging them would be meaningless.

Usage:
    .venv2/bin/python research/compare_ga_runs_28c.py results/ga_iter_runs
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_runs(directory: Path) -> list:
    runs = []
    for path in sorted(directory.glob("*.json")):
        try:
            d = json.loads(path.read_text())
        except Exception:
            continue
        if "oos" not in d or "formula" not in d:
            continue
        runs.append({
            "file": path.name,
            "seed": d.get("seed"),
            "formula": d.get("formula"),
            "oos_sharpe": d["oos"].get("portfolio_sharpe"),
            "oos_mdd": d["oos"].get("portfolio_mdd"),
            "train_sharpe": d.get("train", {}).get("portfolio_sharpe"),
            "val_sharpe": d.get("validation", {}).get("portfolio_sharpe"),
            "verdict": d.get("acceptance", {}).get("verdict"),
            "funding": d.get("funding_source", "constant (assumed; artifact predates the flag)"),
            "accounting": d.get("accounting_version", "equity-compound-v2"),
        })
    return runs


def main() -> int:
    directory = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "ga_iter_runs"
    runs = load_runs(directory)
    if not runs:
        print(f"no run artifacts in {directory}")
        return 1

    bases = sorted({r["funding"] for r in runs})
    print(f"runs: {len(runs)}")
    print(f"funding bases present: {bases}")
    if len(bases) > 1:
        print("  NOTE: mixed bases below. These are not comparable; re-run on one basis.")
    print()
    print(f"{'file':26} {'seed':>5} {'train':>7} {'val':>7} {'OOS':>7} {'mdd':>7}  verdict")
    print("-" * 76)
    for r in runs:
        print(f"{r['file']:26} {str(r['seed']):>5} "
              f"{(r['train_sharpe'] or 0):+7.3f} {(r['val_sharpe'] or 0):+7.3f} "
              f"{(r['oos_sharpe'] or 0):+7.3f} {(r['oos_mdd'] or 0):7.4f}  {r['verdict']}")

    vals = [r["oos_sharpe"] for r in runs if r["oos_sharpe"] is not None]
    print()
    if len(vals) >= 2:
        print("=== lockbox Sharpe across independent searches ===")
        print(f"  n={len(vals)}  mean {statistics.mean(vals):+.3f}  "
              f"median {statistics.median(vals):+.3f}")
        print(f"  sd   {statistics.pstdev(vals):.3f}   min {min(vals):+.3f}  max {max(vals):+.3f}")
        pos = sum(1 for v in vals if v > 0)
        print(f"  positive: {pos}/{len(vals)}")
        print()
        # Selection bias: the best of N draws is not the typical draw.
        print("=== what selection does to the reported number ===")
        print(f"  best-of-{len(vals)} : {max(vals):+.3f}")
        print(f"  median      : {statistics.median(vals):+.3f}")
        print(f"  gap         : {max(vals) - statistics.median(vals):+.3f}")
        print("  A report that quotes only the best of N is quoting the top of this")
        print("  distribution, which is what selection buys whether or not edge exists.")

    verdicts = {}
    for r in runs:
        verdicts[r["verdict"]] = verdicts.get(r["verdict"], 0) + 1
    print()
    print("=== gate verdicts ===")
    for k, v in verdicts.items():
        print(f"  {k}: {v}/{len(runs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
