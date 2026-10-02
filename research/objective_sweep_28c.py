#!/usr/bin/env python3
"""Screen the objective presets for one that produces a usable gradient.

Why this exists
---------------
docs/P2_PREGATE_DIAGNOSIS.md measured that the frozen v3 objective has no
out-of-sample gradient. Over 90 real-arm selection folds:

    median argmax_margin   0.00903
    median |overfit_gap|   0.7019      78x larger than the margin
    train_reward > 0       20/90 folds, of which 0 survive validation
    7 of 10 selected formulas hold a constant position for the whole holdout
    0 of 10 beat buy-and-hold

A separate sweep of the turnover weight across 0.02 to 200 did not help, so the
problem is the objective rather than a mis-scaled term in it. This script tries
five objectives and reports the same four quantities for each.

The bar
-------
An objective is only interesting if it clears all three at once:

  1. argmax_margin rises materially above the v3 baseline. Without a gradient
     there is nothing for a wider search to exploit.
  2. the real arm beats the null arm. An objective that ranks noise well will
     rank synthetic data just as well.
  3. excess over buy-and-hold is positive on some seeds. Everything else is
     motion without progress.

Clearing 1 and failing 2 or 3 means the search got better at a task with no
signal, which is the same failure with a different number attached.

Usage:
    python research/objective_sweep_28c.py
    python research/objective_sweep_28c.py --objectives v3 rank_ic --seeds 101 102
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import research.ga_28c_30m_3y as ga  # noqa: E402

DEFAULT_OBJECTIVES = ["v3", "rank_ic", "excess_bh", "rank_ic_excess_bh", "v3_turn20"]


def run_one(objective, seed, arm, outdir, folds, generations, population):
    out = Path(outdir) / f"obj_{objective}_{arm}_{seed}.json"
    cmd = [sys.executable, "research/ga_28c_30m_3y.py",
           "--mode", "walkforward", "--folds", str(folds),
           "--generations", str(generations), "--population", str(population),
           "--seed", str(seed), "--null-seed", str(seed),
           "--objective", objective, "--funding", "real", "--out", str(out)]
    if arm == "null":
        cmd += ["--null", "iid"]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=14400)
    if r.returncode != 0 or not out.exists():
        return None, (r.stderr or "")[-300:]
    return json.loads(out.read_text()), None


def summarise(runs):
    if not runs:
        return None
    margins, gaps, pos_std, hold_sh, hold_ex = [], [], [], [], []
    for j in runs:
        for fr in j.get("fold_records", []):
            if fr.get("is_holdout"):
                continue
            if fr.get("argmax_margin") is not None:
                margins.append(abs(fr["argmax_margin"]))
            if fr.get("train_reward") is not None and fr.get("validation_reward") is not None:
                gaps.append(abs(fr["validation_reward"] - fr["train_reward"]))
        h = j.get("holdout") or {}
        if h.get("portfolio_sharpe") is not None:
            hold_sh.append(h["portfolio_sharpe"])
        if h.get("excess_sharpe_vs_buy_and_hold") is not None:
            hold_ex.append(h["excess_sharpe_vs_buy_and_hold"])
    return {
        "n_runs": len(runs),
        "argmax_margin_median": st.median(margins) if margins else None,
        "overfit_gap_median": st.median(gaps) if gaps else None,
        "holdout_sharpe_median": st.median(hold_sh) if hold_sh else None,
        "excess_vs_bh_median": st.median(hold_ex) if hold_ex else None,
        "excess_vs_bh_positive": sum(1 for x in hold_ex if x > 0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objectives", nargs="+", default=DEFAULT_OBJECTIVES)
    ap.add_argument("--seeds", type=int, nargs="+", default=[101, 102, 103])
    ap.add_argument("--folds", type=int, default=2)
    ap.add_argument("--generations", type=int, default=3)
    ap.add_argument("--population", type=int, default=12)
    ap.add_argument("--outdir", type=Path, default=Path("/tmp/obj_sweep"))
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)

    print("OBJECTIVE SWEEP")
    print("=" * 78)
    print(f"objectives {a.objectives}")
    print(f"seeds {a.seeds}  {a.folds} folds x {a.generations} gen x pop {a.population}")
    print("screen only; anything that clears the bar earns a full paired pilot")
    print()

    results = {}
    for obj in a.objectives:
        per_arm, failed = {}, []
        for arm in ("real", "null"):
            runs = []
            for s in a.seeds:
                j, err = run_one(obj, s, arm, a.outdir, a.folds, a.generations, a.population)
                if j is None:
                    failed.append(f"{arm}/{s}")
                else:
                    runs.append(j)
            per_arm[arm] = summarise(runs)
        real, null = per_arm["real"], per_arm["null"]
        gap = None
        if real and null and real["holdout_sharpe_median"] is not None \
                and null["holdout_sharpe_median"] is not None:
            gap = real["holdout_sharpe_median"] - null["holdout_sharpe_median"]
        results[obj] = {"real": real, "null": null, "real_minus_null": gap,
                        "failed": failed}

        def f(x, nd=5):
            return "n/a" if x is None else f"{x:.{nd}f}"

        print(f"{obj}")
        print(f"   argmax_margin  {f(real and real['argmax_margin_median'])}"
              f"   (v3 baseline 0.00903)")
        print(f"   overfit_gap    {f(real and real['overfit_gap_median'], 4)}")
        print(f"   holdout sharpe real {f(real and real['holdout_sharpe_median'], 3)}"
              f"  null {f(null and null['holdout_sharpe_median'], 3)}")
        print(f"   excess vs B&H  real {f(real and real['excess_vs_bh_median'], 3)}"
              f"  positive {real['excess_vs_bh_positive'] if real else 0}/{len(a.seeds)}")
        print(f"   real - null    {f(gap, 3)}")
        if failed:
            print(f"   FAILED {failed}")
        print()

    base = results.get("v3", {}).get("real") or {}
    base_margin = base.get("argmax_margin_median")
    print("=" * 78)
    print("THE BAR")
    print()
    for obj, r in results.items():
        real = r["real"]
        if not real:
            continue
        m = real["argmax_margin_median"]
        wider = (base_margin is not None and m is not None
                 and base_margin > 0 and m > base_margin * 1.5)
        beats_null = (r["real_minus_null"] is not None and r["real_minus_null"] > 0)
        beats_bh = real["excess_vs_bh_positive"] > 0
        verdict = "candidate" if (wider and beats_null and beats_bh) else "no"
        print(f"  {obj:22} margin {'wider' if wider else 'not wider':11} "
              f"real>null {'yes' if beats_null else 'no ':4} "
              f"beats B&H {'yes' if beats_bh else 'no'}  -> {verdict}")
    print()
    print("  A candidate has to clear all three. Clearing only the margin means")
    print("  the search improved at a task with no signal, which is the original")
    print("  failure wearing a different number.")

    p = ROOT / "results" / "objective_sweep_28c.json"
    p.write_text(json.dumps({"config": {k: str(v) for k, v in vars(a).items()},
                            "results": results}, indent=2, default=float))
    print()
    print("wrote", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
