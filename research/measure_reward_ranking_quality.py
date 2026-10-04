#!/usr/bin/env python3
"""Does the reward rank genomes correctly, and does it rank BETTER on real data?

The critique this answers
-------------------------
Every acceptance criterion in the PRP line so far compares SELECTED MAXIMA across
re-runs: "real median Sharpe beats null", "positive_coins improved", "v5 beats v4
on 6 of 10 seeds". All of those are consistent with pure selection bias. A search
that ranks noise perfectly will produce a high winner on real data too, and a
search that ranks signal perfectly will produce a high winner on null data too.
Comparing winners cannot tell those cases apart.

The statistic that can is the rank correlation between the train reward and the
holdout outcome, computed over EVERY genome evaluated, not just the selected one.
Two distinct questions:

  Q1  Is the reward informative at all?
      Spearman(train_reward, holdout_net_sharpe) across all genomes of a run.
      If this is ~0 the search is not ranking anything and no amount of
      reward reshaping will help.

  Q2  Is the reward MORE informative on real data than on null data?
      The same statistic, real arm versus null arm. This is the one that tests
      whether the data carries signal the reward can see. If real and null give
      the same ranking quality, the search is measuring its own apparatus and the
      search-space work in PRP-1 is unjustified.

Why this is the right gate
-------------------------
Reward reshaping in PRP-2 or PRP-3 cannot, by construction, change the
information content of the data. It can only change which optimum the search
reaches. So if Q2 is zero, reshaping produces motion without progress, and if Q1
is zero the search has nothing to reshape. Both must be answered before any of
the three items is worth its compute.

Method
------
Runs a short walkforward search per arm, recording (formula, params-free train
reward, holdout net Sharpe) for every genome the GA evaluates, then reports the
per-run and pooled rank correlations. Short by design: the question is about
ranking behaviour, not about finding a good strategy, so 2 folds x 3
generations x 12 population is enough and keeps a full real+null sweep cheap.

Usage:
    python research/measure_reward_ranking_quality.py
    python research/measure_reward_ranking_quality.py --generations 5 --population 16
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import research.ga_28c_30m_3y as ga  # noqa: E402
from research.universe_28c import COINS_28C  # noqa: E402


def spearman(x, y):
    """Rank correlation with average ranks for ties. Returns None if degenerate."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    if len(x) < 8:
        return None
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


PROBE = r"""
# Injected ranking probe: wrap evaluate() so every genome the GA scores is
# recorded with its train reward and its holdout net Sharpe. Nothing about the
# search changes; this only observes.
import json as _json, os as _os, sys as _sys
sys.path.insert(0, {root!r})
import research.ga_28c_30m_3y as _ga

_ROWS = []
_orig_evaluate = _ga.evaluate
_HOLDOUT = [None]


def _probe(formula, maps, returns, start, end, funding_mask, scale_end, bars,
           psh_peer=None, funding_by_coin=None, reward_version='v1', grammar=None):
    out = _orig_evaluate(formula, maps, returns, start, end, funding_mask, scale_end,
                         bars, psh_peer=psh_peer, funding_by_coin=funding_by_coin,
                         reward_version=reward_version, grammar=grammar)
    _ROWS.append({{
        'formula': list(formula),
        'train_reward': float(out['reward']),
        'train_sharpe': float(out['portfolio_sharpe']),
        'train_ic': float(out['mean_ic']),
        'turnover': float(out['turnover']),
        'positive_coins': int(out['positive_coins']),
        'range': (int(start), int(end)),
    }})
    return out


def _install():
    _ga.evaluate = _probe
    _orig_run_wf = _ga.run_walkforward

    def run_wf(args, maps, returns, common, funding_mask, funding_by_coin,
               folds, rng, scale_end_hint):
        recs, selected = _orig_run_wf(args, maps, returns, common, funding_mask,
                                      funding_by_coin, folds, rng, scale_end_hint)
        # score every distinct genome on the HOLDOUT window so train reward and
        # holdout outcome are paired for all of them, not only the winner
        hold = [f for f in folds if f.get('is_holdout')]
        if hold:
            hs, he = hold[0]['validation']
            seen = set()
            for row in _ROWS:
                key = tuple(row['formula'])
                if key in seen:
                    continue
                seen.add(key)
                res = _orig_evaluate(tuple(row['formula']), maps, returns, hs, he,
                                     funding_mask, scale_end_hint, common,
                                     funding_by_coin=funding_by_coin,
                                     reward_version=args.reward_version)
                row['holdout_sharpe'] = float(res['portfolio_sharpe'])
                row['holdout_ic'] = float(res['mean_ic'])
                row['holdout_positive_coins'] = int(res['positive_coins'])
        _json.dump(_ROWS, open({dump!r}, 'w'))
        return recs, selected

    _ga.run_walkforward = run_wf
"""


def run_arm(arm, seed, generations, population, folds, tmpdir):
    """Run a short walkforward search with an injected probe that records every
    genome's train reward paired with its holdout outcome.

    The probe is injected via -c so repository source is never modified. It wraps
    evaluate() and run_walkforward, leaving the search itself untouched.
    """
    out = Path(tmpdir) / f"rq_{arm}_{seed}.json"
    dump = Path(tmpdir) / f"rq_rows_{arm}_{seed}.json"
    NL = chr(10)
    argv = ["--mode", "walkforward", "--folds", str(folds),
            "--generations", str(generations), "--population", str(population),
            "--seed", str(seed), "--null-seed", str(seed),
            "--funding", "real", "--out", str(out)]
    if arm == "null":
        argv += ["--null", "iid"]

    driver_lines = [
        "import json as _json, sys as _sys",
        "_sys.path.insert(0, %r)" % str(ROOT),
        "import research.ga_28c_30m_3y as _ga",
        "_ROWS = []",
        "_orig_evaluate = _ga.evaluate",
        "def _probe(formula, maps, returns, start, end, funding_mask, scale_end, bars,",
        "           psh_peer=None, funding_by_coin=None, reward_version='v1', grammar=None):",
        "    out = _orig_evaluate(formula, maps, returns, start, end, funding_mask, scale_end,",
        "                         bars, psh_peer=psh_peer, funding_by_coin=funding_by_coin,",
        "                         reward_version=reward_version, grammar=grammar)",
        "    _ROWS.append({'formula': list(formula),",
        "                  'train_reward': float(out['reward']),",
        "                  'train_sharpe': float(out['portfolio_sharpe']),",
        "                  'train_ic': float(out['mean_ic']),",
        "                  'train_positive_coins': int(out['positive_coins']),",
        "                  'turnover': float(out['turnover'])})",
        "    return out",
        "_ga.evaluate = _probe",
        "_orig_run_wf = _ga.run_walkforward",
        "def run_wf(args, maps, returns, common, funding_mask, funding_by_coin, folds, rng, scale_end_hint):",
        "    recs, selected = _orig_run_wf(args, maps, returns, common, funding_mask,",
        "                                      funding_by_coin, folds, rng, scale_end_hint)",
        "    hold = [f for f in folds if f.get('is_holdout')]",
        "    if hold:",
        "        hs, he = hold[0]['validation']",
        "        seen = set()",
        "        for row in _ROWS:",
        "            key = tuple(row['formula'])",
        "            if key in seen:",
        "                continue",
        "            seen.add(key)",
        "            res = _orig_evaluate(tuple(row['formula']), maps, returns, hs, he,",
        "                                 funding_mask, scale_end_hint, common,",
        "                                 funding_by_coin=funding_by_coin,",
        "                                 reward_version=args.reward_version)",
        "            row['holdout_sharpe'] = float(res['portfolio_sharpe'])",
        "            row['holdout_ic'] = float(res['mean_ic'])",
        "            row['holdout_positive_coins'] = int(res['positive_coins'])",
        "    _json.dump(_ROWS, open(%r, 'w'))" % str(dump),
        "    return recs, selected",
        "_ga.run_walkforward = run_wf",
        "import sys as _sys2; _sys2.argv = ['ga'] + %r" % (argv,),
        "_ga.main()",
    ]
    driver = NL.join(driver_lines)
    import subprocess
    r = subprocess.run([sys.executable, "-c", driver], cwd=ROOT,
                       capture_output=True, text=True, timeout=14400)
    if r.returncode != 0 or not dump.exists():
        return None, r.stderr[-600:]
    return json.loads(dump.read_text()), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=3)
    ap.add_argument("--population", type=int, default=12)
    ap.add_argument("--folds", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--tmpdir", type=Path, default=Path("/tmp/rq"))
    a = ap.parse_args()
    a.tmpdir.mkdir(parents=True, exist_ok=True)

    print("REWARD RANKING QUALITY")
    print("=" * 74)
    print("Q1: is the reward informative at all?")
    print("Q2: is it MORE informative on real data than on null data?")
    print()
    print(f"config: {a.folds} folds x {a.generations} generations x {a.population} pop, "
          f"{a.seeds} seeds per arm")
    print()

    # The run files do not carry per-genome records, so the statistic is computed
    # from what IS recorded: per fold, the selected genome's train reward and
    # validation Sharpe, plus the runner-up margin. Across folds and seeds this
    # gives a paired sample of (train reward, validation outcome) for SELECTED
    # genomes. That is weaker than all-genomes ranking and is labelled as such.
    arms = {}
    for arm in ("real", "null"):
        rows = []
        for i in range(a.seeds):
            seed = 101 + i
            rows_i, err = run_arm(arm, seed, a.generations, a.population, a.folds, a.tmpdir)
            if not rows_i:
                print(f"  {arm} seed{seed}: run failed ({str(err)[:120]})")
                continue
            rows.extend(rows_i)
            print(f"  {arm} seed{seed}: {len(rows_i)} genomes recorded")
        arms[arm] = rows

    print()
    out = {"config": {k: str(v) for k, v in vars(a).items()},
           "note": ("ranking quality over EVERY genome evaluated, not only the "
                    "selected winner, which is the statistic the PRP acceptances "
                    "were missing")}
    for arm, rows in arms.items():
        if not rows:
            continue
        tr = [r["train_reward"] for r in rows if "holdout_sharpe" in r]
        hs = [r["holdout_sharpe"] for r in rows if "holdout_sharpe" in r]
        ti = [r["train_ic"] for r in rows if "holdout_ic" in r]
        hi = [r["holdout_ic"] for r in rows if "holdout_ic" in r]
        ts = [r["train_sharpe"] for r in rows if "holdout_sharpe" in r]
        gaps = [r["holdout_sharpe"] - r["train_sharpe"] for r in rows if "holdout_sharpe" in r]
        rho_r = spearman(tr, hs)
        rho_s = spearman(ts, hs)
        rho_i = spearman(ti, hi)
        out[arm] = {
            "n_genomes": len(hs),
            "spearman_reward_vs_holdout": rho_r,
            "spearman_trainsharpe_vs_holdout": rho_s,
            "spearman_trainic_vs_holdout": rho_i,
            "mean_sharpe_decay_holdout_minus_train": st.mean(gaps) if gaps else None,
        }
        print(f"{arm.upper()}  genomes={len(hs)}")
        print(f"  Q1 spearman(train_reward,  holdout_sharpe) = {fmt(rho_r)}")
        print(f"     spearman(train_sharpe, holdout_sharpe) = {fmt(rho_s)}")
        print(f"     spearman(train_ic,     holdout_ic)      = {fmt(rho_i)}")
        print(f"     mean holdout_sharpe - train_sharpe      = {fmt(st.mean(gaps) if gaps else None)}")
        print()

    if "real" in out and "null" in out:
        dr = out["real"]["spearman_reward_vs_holdout"]
        dn = out["null"]["spearman_reward_vs_holdout"]
        print("=" * 74)
        print("Q2 ANSWER")
        print("=" * 74)
        gap = (dr or 0.0) - (dn or 0.0)
        print(f"  real ranking quality  = {fmt(dr)}")
        print(f"  null ranking quality  = {fmt(dn)}")
        print(f"  real - null           = {fmt(gap)}")
        print()
        if dr is None or dn is None:
            print("  INCONCLUSIVE: a ranking correlation could not be computed.")
        elif gap <= 0.0:
            print("  VERDICT: the reward is NOT more informative on real data.")
            print("  PRP-1, PRP-2 and PRP-3 are NO-GO. None of them can create")
            print("  information the data does not contain.")
        elif abs(dr) < 0.1:
            print("  VERDICT: real is nominally better but BOTH arms are near zero.")
            print("  The reward carries almost no ranking information on either, so")
            print("  reshaping it would be re-arranging noise. PRP-2 and PRP-3 NO-GO.")
        else:
            print("  VERDICT: real data ranks materially better. The reward carries")
            print("  information, so PRP-2/PRP-3 have something to work with and")
            print("  PRP-1's search expansion is justified.")
        out["q2_gap"] = gap

    p = ROOT / "results" / "reward_ranking_quality.json"
    p.write_text(json.dumps(out, indent=2, default=float))
    print()
    print("wrote", p)
    return 0


def fmt(x):
    return "n/a" if x is None else f"{x:+.4f}"


if __name__ == "__main__":
    sys.exit(main())
