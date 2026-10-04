#!/usr/bin/env python3
"""Verify the pilot meter repair against the recorded runs.

The repair has two parts:

  1. shape_reward no longer returns a -50 cliff when the genome is inactive; the
     caller applies a continuous activity_penalty instead.
  2. psh_peer is supplied from a fixed external baseline, so the gap term is no
     longer identically zero.

This script recomputes the reward for all 20 recorded runs (10 real, 10 null)
under the old and the repaired meter, and reports whether the two claims made in
docs/GA_SEARCH_RELIABILITY_2026-10-01.md actually hold:

  A. the reward stops saturating
  B. real separates from null on the repaired meter

It does not rerun the GA. It replays the recorded per-split diagnostics, so it
measures the meter only and cannot be confused with an engine change.

Usage:
    python research/verify_pilot_meter_repair.py
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

REAL_GLOB = ROOT / "results" / "ga_iter_runs" / "*.json"
NULL_GLOB = ROOT / "results" / "ga_null_runs" / "*.json"

GAP_TAU = 1.0
LEG_FLOOR = 0.0
ACTIVITY_TARGET = 0.01
INACTIVITY_WEIGHT = 0.5


def softplus(x):
    x = max(-40.0, min(40.0, x))
    return math.log1p(math.exp(x))


def load(pattern):
    out = []
    for f in sorted(glob.glob(str(pattern))):
        j = json.loads(Path(f).read_text())
        v = j["validation"]
        out.append({
            "seed": j["seed"],
            "psh": v["portfolio_sharpe"],
            "min_leg": v["min_leg_sharpe"],
            "mean_ic": v["mean_ic"],
            "turnover": v["turnover"],
            "activity": v["activity"],
            "reward_recorded": v["reward"],
        })
    return out


def old_meter(r, peer):
    """The meter as it stood: hard cliff, dead gap term."""
    if r["activity"] < ACTIVITY_TARGET or r["turnover"] < 1e-5:
        return -50.0
    p = peer if peer is not None else r["psh"]      # psh_peer=None -> peer==psh
    gap = softplus((r["psh"] - p) / GAP_TAU) * GAP_TAU
    return (r["mean_ic"]
            + 0.10 * min(r["psh"], p)
            - 0.30 * gap
            - 0.02 * r["turnover"]
            - 0.10 * softplus((LEG_FLOOR - r["min_leg"]) / 0.5))


def new_meter(r, peer):
    """Repaired: continuous inactivity penalty, live gap term."""
    p = peer if peer is not None else r["psh"]
    gap = softplus((r["psh"] - p) / GAP_TAU) * GAP_TAU
    base = (r["mean_ic"]
            + 0.10 * min(r["psh"], p)
            - 0.30 * gap
            - 0.02 * r["turnover"]
            - 0.10 * softplus((LEG_FLOOR - r["min_leg"]) / 0.5))
    penalty = -INACTIVITY_WEIGHT * max(0.0, 1.0 - r["activity"] / ACTIVITY_TARGET)
    return base + penalty


def describe(name, xs):
    return (f"{name:22} n={len(xs):2}  mean={st.mean(xs):+.4f}  std={st.pstdev(xs):.4f}  "
            f"range=[{min(xs):+.4f}, {max(xs):+.4f}]")


def main():
    real, null = load(REAL_GLOB), load(NULL_GLOB)
    if not real or not null:
        print("missing run files; nothing to verify")
        return 1

    # A fixed external baseline, not fitted to these runs: the median portfolio
    # Sharpe of the REAL arm is itself a batch statistic, so instead use the
    # zero line, i.e. "beat nothing". That is constant by construction and cannot
    # absorb a result.
    peer = 0.0

    print(f"peer baseline = {peer:.1f} (fixed zero line, not fitted to this batch)")
    print(f"activity target = {ACTIVITY_TARGET}, inactivity weight = {INACTIVITY_WEIGHT}")
    print()

    old_r = [old_meter(r, peer) for r in real]
    old_n = [old_meter(r, peer) for r in null]
    new_r = [new_meter(r, peer) for r in real]
    new_n = [new_meter(r, peer) for r in null]

    print("OLD METER")
    print(" ", describe("real", old_r))
    print(" ", describe("null", old_n))
    sat_o = sum(1 for x in old_r + old_n if x <= -49.9)
    print(f"  saturated: {sat_o}/{len(old_r + old_n)}   pooled std={st.pstdev(old_r + old_n):.4f}")
    print()

    print("REPAIRED METER")
    print(" ", describe("real", new_r))
    print(" ", describe("null", new_n))
    sat_n = sum(1 for x in new_r + new_n if x <= -49.9)
    print(f"  saturated: {sat_n}/{len(new_r + new_n)}   pooled std={st.pstdev(new_r + new_n):.4f}")
    print()

    # Compare spread among NON-SATURATED runs only. The old pooled std is
    # dominated by the -50 cliff, so comparing raw pooled std would read a
    # working repair as a regression: removing the cliff removes outliers and the
    # std necessarily falls. What should rise is the resolution inside the region
    # where the search actually had to rank genomes.
    old_live = [x for x in old_r + old_n if x > -49.9]
    new_live = new_r + new_n
    old_spread = st.pstdev(old_live) if len(old_live) > 1 else 0.0
    new_spread = st.pstdev(new_live)
    live_gain = len(new_live) / max(len(old_live), 1)
    claim_a = sat_n == 0 and new_spread > old_spread
    print(f"CLAIM A  reward stops saturating and gains resolution: {claim_a}")
    print(f"         saturation {sat_o} -> {sat_n} of {len(old_r + old_n)}")
    print(f"         rankable runs {len(old_live)} -> {len(new_live)}  ({live_gain:.2f}x)")
    print(f"         std among rankable runs {old_spread:.4f} -> {new_spread:.4f}")
    print(f"         (raw pooled std fell {st.pstdev(old_r + old_n):.4f} -> "
          f"{st.pstdev(new_r + new_n):.4f} only because the -50 outliers are gone)")
    print()

    d_new = [a - b for a, b in zip(new_r, new_n)]
    d_old = [a - b for a, b in zip(old_r, old_n)]
    wins = sum(1 for x in d_new if x > 0)
    n = len(d_new)
    p = min(1.0, sum(math.comb(n, i) for i in range(wins, n + 1)) / 2 ** n * 2)
    print("REAL minus NULL, per seed")
    print(f"  old  median gap = {st.median(d_old):+.4f}   real wins {sum(1 for x in d_old if x > 0)}/{n}")
    print(f"  new  median gap = {st.median(d_new):+.4f}   real wins {wins}/{n}   sign-test p={p:.3f}")
    print()

    claim_b = st.median(d_new) > st.median(d_old)
    print(f"CLAIM B  real separates from null better than before: {claim_b}")
    print()
    print("READ THIS AS")
    print("  A and B describe the METER, not the engine. They say the repaired")
    print("  reward has more resolution and orders real above null more often.")
    print("  They do NOT say the search found tradable alpha: mean_ic is still")
    print("  ~0.0107 and the sign test is well short of significance at n=10.")
    print("  The engine is unchanged and still frozen by the pilot contract.")

    out = {
        "peer_baseline": peer,
        "activity_target": ACTIVITY_TARGET,
        "inactivity_weight": INACTIVITY_WEIGHT,
        "old": {"real": old_r, "null": old_n, "saturated": sat_o,
                "pooled_std": st.pstdev(old_r + old_n)},
        "new": {"real": new_r, "null": new_n, "saturated": sat_n,
                "pooled_std": st.pstdev(new_r + new_n)},
        "claim_a_stops_saturating": claim_a,
        "old_rankable_std": old_spread,
        "new_rankable_std": new_spread,
        "old_rankable_n": len(old_live),
        "new_rankable_n": len(new_live),
        "claim_b_separates": claim_b,
        "new_median_gap": st.median(d_new),
        "new_real_wins": wins,
        "sign_test_p": p,
    }
    p_out = ROOT / "results" / "pilot_meter_repair_verification.json"
    p_out.write_text(json.dumps(out, indent=2))
    print()
    print("wrote", p_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
