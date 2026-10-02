#!/usr/bin/env python3
"""Micro-experiment: does a larger turnover weight give the search a gradient?

The question
------------
docs/P2_PREGATE_DIAGNOSIS.md section 9 measured that `shape_reward`'s turnover
term is the only part of the objective that would penalise trading, and that at
weight 0.02 its entire dynamic range is 0.000502 reward units against a total
reward spread of 0.6187, i.e. 0.0812%. Movement is therefore free, the argmax
margin is 0.009 against 0.70 of fold-to-fold scatter, and 7 of 10 selected
formulas hold a constant position.

The obvious hypothesis is that the weight is simply mis-scaled. This script
tests it rather than assuming it.

What it measures
----------------
For each candidate weight it runs a paired real/null search and reports:

  argmax_margin    how far the winner beat the runner-up on train. This is the
                   gradient. If the weight is the problem, raising it should
                   widen this.
  position_std     whether the selected book actually moves. A weight that only
                   produces turnover without information shows up here as a
                   position that swings and loses money.
  excess_vs_bh     whether it beats holding the same book passively.
  real-null gap    whether real data separates from synthetic at all.

Sizing
------
Deliberately small: 2 folds, 3 generations, population 12, 3 seeds per arm, 5
weights. This is a screen, not a promotion run. It answers "is there a gradient
worth chasing" and nothing more. A weight that looks promising here earns a full
paired pilot, not a deployment.

The null arm is not optional. A larger turnover weight can look good on real data
purely because trading costs money and the synthetic arm has a different
distribution of turnover. Only real beating null means something.

Usage:
    python research/turnover_weight_probe_28c.py
    python research/turnover_weight_probe_28c.py --weights 0.02 0.2 2.0
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

DEFAULT_WEIGHTS = [0.02, 0.2, 2.0, 20.0, 200.0]
DEFAULT_SEEDS = [101, 102, 103]


def run_one(weight, seed, arm, outdir, folds, generations, population):
    """One paired run with the turnover weight overridden.

    The weight is set by rewriting the module constant in a subprocess through
    an env var, so the search code is never edited between weights.
    """
    out = Path(outdir) / f"tw{weight:g}_{arm}_{seed}.json"
    env = dict(os.environ)
    env["ALPHAGPT_TURNOVER_WEIGHT"] = str(weight)
    cmd = [sys.executable, "research/ga_28c_30m_3y.py",
           "--mode", "walkforward", "--folds", str(folds),
           "--generations", str(generations), "--population", str(population),
           "--seed", str(seed), "--null-seed", str(seed),
           "--funding", "real", "--out", str(out)]
    if arm == "null":
        cmd += ["--null", "iid"]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env, timeout=14400)
    if r.returncode != 0 or not out.exists():
        return None, (r.stderr or "")[-300:]
    return json.loads(out.read_text()), None


def summarize(runs):
    """Pull the four quantities this screen is about out of a set of runs."""
    if not runs:
        return None
    margins, gaps, pos_std, hold_sh, hold_ex, real_null = [], [], [], [], [], []
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
    out = {
        "n_runs": len(runs),
        "argmax_margin_median": st.median(margins) if margins else None,
        "overfit_gap_median": st.median(gaps) if gaps else None,
        "holdout_sharpe_median": st.median(hold_sh) if hold_sh else None,
        "excess_vs_bh_median": st.median(hold_ex) if hold_ex else None,
        "excess_vs_bh_positive": sum(1 for x in hold_ex if x > 0),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=float, nargs="+", default=DEFAULT_WEIGHTS)
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--folds", type=int, default=2)
    ap.add_argument("--generations", type=int, default=3)
    ap.add_argument("--population", type=int, default=12)
    ap.add_argument("--outdir", type=Path, default=Path("/tmp/tw_probe"))
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)

    print("TURNOVER WEIGHT SCREEN")
    print("=" * 78)
    print(f"weights {a.weights}  seeds {a.seeds}  {a.folds} folds x "
          f"{a.generations} gen x pop {a.population}")
    print("screen only; a promising weight earns a full paired pilot, not deployment")
    print()

    results = {}
    for w in a.weights:
        per_arm = {}
        failed = []
        for arm in ("real", "null"):
            runs = []
            for s in a.seeds:
                j, err = run_one(w, s, arm, a.outdir, a.folds, a.generations, a.population)
                if j is None:
                    failed.append(f"{arm}/{s}: {err}")
                else:
                    runs.append(j)
            per_arm[arm] = summarize(runs)
        real, null = per_arm["real"], per_arm["null"]
        gap = None
        if real and null and real["holdout_sharpe_median"] is not None \
                and null["holdout_sharpe_median"] is not None:
            gap = real["holdout_sharpe_median"] - null["holdout_sharpe_median"]
        results[w] = {"real": real, "null": null, "real_minus_null": gap,
                      "failures": failed}

        def f(x, nd=5):
            return "n/a" if x is None else f"{x:.{nd}f}"

        print(f"weight {w:<8g}")
        print(f"   argmax_margin  real {f(real and real['argmax_margin'])}"
              f"   null {f(null and null['argmax_margin'])}")
        print(f"   overfit_gap    real {f(real and real['overfit_gap_median'], 4)}")
        print(f"   holdout sharpe real {f(real and real['holdout_sharpe_median'], 3)}"
              f"  null {f(null and null['holdout_sharpe_median'], 3)}")
        print(f"   excess vs B&H  real {f(real and real['excess_vs_bh_median'], 3)}"
              f"  positive {real['excess_vs_bh_positive'] if real else 0}/{len(a.seeds)}")
        print(f"   real - null    {f(gap, 3)}")
        if failed:
            print(f"   FAILED: {failed[0]}")
        print()

    # what would actually justify moving on
    print("=" * 78)
    print("READING THIS")
    print()
    best = None
    for w, r in results.items():
        if r["real"] and r["real"]["argmax_margin_median"] is not None:
            if best is None or r["real"]["argmax_margin_median"] > results[best]["real"]["argmax_margin_median"]:
                best = w
    print(f"  widest argmax margin at weight {best}")
    base = results[a.weights[0]]["real"]
    if best is not None and base and base["argmax_margin_median"]:
        ratio = results[best]["real"]["argmax_margin_median"] / base["argmax_margin_median"]
        print(f"  margin vs weight {a.weights[0]}: {ratio:.1f}x")
    print()
    print("  A weight is only interesting if the margin widens AND the real arm")
    print("  still beats the null arm. Widening the margin while real loses to")
    print("  null means the search got better at a task that has no signal, which")
    print("  is the same failure as before with a different number attached.")
    print()
    print("  If no weight widens the margin by a meaningful factor, the turnover")
    print("  term was not the binding constraint and PRP-1's precondition stands.")

    p = ROOT / "results" / "turnover_weight_probe_28c.json"
    p.write_text(json.dumps({"config": vars(a) and {k: str(v) for k, v in vars(a).items()},
                            "results": {str(k): v for k, v in results.items()}},
                           indent=2, default=float))
    print()
    print("wrote", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
