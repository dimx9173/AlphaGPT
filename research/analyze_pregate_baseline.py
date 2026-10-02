#!/usr/bin/env python3
"""P2 pre-gate: is the instrument usable after PRP-0 plus the meter repair?

Question
--------
PRP-1, PRP-2 and PRP-3 all carry the same precondition: with formula-only
search, the repaired meter and the closed validation leak, does real data beat
synthetic data on a holdout scored exactly once?

Codex's R3 said this has to be answered BEFORE any search-space expansion is
paid for, because the previous pilot lost to null by -3.355 Sharpe with 0/10
positive-coin breadth. Spending the joint-search budget on an instrument that
already reports "null wins" would buy a noisier version of the same negative.

What changed since that pilot
-----------------------------
The validation-window selection leak is closed. Selection keyed on
0.5*train + 0.5*validation inside the generation loop, so each fold's validation
window was read 400 times and half the selection key came from held-out data.
Selection now keys on train only. That is exactly the correction that should
reduce the winner's curse, so this batch is the first fair test of the repaired
meter.

Decision rule (pre-registered, do not move after seeing the result)
------------------------------------------------------------------
  PASS  real median holdout Sharpe > null median, AND one-sided
        Mann-Whitney real > null at p < 0.10, AND real positive_coins median
        >= 14/28 on at least 6/10 seeds.
  FAIL  anything else.

  A two-sided p with the gap negative is a FAIL, not a near-miss: it means the
  arms separate in the wrong direction, which is the strongest evidence against
  the search, not a weak result.

  Power note carried from the PRP: at n=10 per arm, Mann-Whitney p<0.10 one-sided
  needs roughly 8/10 dominance, and power is about 38% even at a large true
  per-seed win rate of 0.7. A PASS is meaningful. A FAIL is weak evidence, and is
  recorded as such rather than used to justify widening the search.

Usage:
    python research/analyze_pregate_baseline.py [--dir results/p2_pregate_baseline]
"""
from __future__ import annotations

import argparse
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

DEFAULT_DIR = ROOT / "results" / "p2_pregate_baseline"

# pre-registered, before the batch was read
PASS_MIN_POSITIVE_COINS = 14      # of 28
PASS_MIN_COIN_SEEDS = 6           # of 10 seeds
PASS_ALPHA = 0.10                 # one-sided Mann-Whitney


def load(pattern):
    rows = []
    for f in sorted(glob.glob(str(pattern))):
        j = json.loads(Path(f).read_text())
        h = j.get("holdout") or {}
        rows.append({
            "seed": j.get("seed"),
            "select_on": j.get("select_on"),
            "sharpe": h.get("portfolio_sharpe"),
            "excess": h.get("excess_sharpe_vs_buy_and_hold"),
            "positive_coins": h.get("positive_coins"),
            "mean_position": h.get("mean_position"),
            "mdd": h.get("portfolio_mdd"), "bh": h.get("buy_and_hold_sharpe"),
        })
    return rows


def mannwhitney_one_sided(a, b):
    """U for a>b, plus a normal-approx one-sided p with tie correction.

    One-sided matters: the previous pilot produced a two-sided p=0.038 with the
    null ABOVE real, which is a significant result in the wrong direction and
    must never be read as a near-miss.
    """
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
    sd = math.sqrt(n1 * n2 / 12 * ((N + 1) - tie / (N * (N - 1)))) if N > 1 else 0.0
    if sd == 0:
        return U1, 1.0, 0.5
    z = (U1 - mu) / sd
    p = 1 - 0.5 * (1 + math.erf(z / math.sqrt(2)))
    return U1, p, z


def sign_test(diffs):
    n = len(diffs)
    k = sum(1 for x in diffs if x > 0)
    p = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return k, n, p


def describe(name, rows):
    sh = [r["sharpe"] for r in rows if r["sharpe"] is not None]
    pc = [r["positive_coins"] for r in rows if r["positive_coins"] is not None]
    mp = [abs(r["mean_position"]) for r in rows if r["mean_position"] is not None]
    out = {
        "n": len(sh),
        "sharpe_median": st.median(sh) if sh else None,
        "sharpe_mean": st.mean(sh) if sh else None,
        "sharpe_best": max(sh) if sh else None,
        "sharpe_worst": min(sh) if sh else None,
        "positive_coins_median": st.median(pc) if pc else None,
        "abs_mean_position_median": st.median(mp) if mp else None,
    }
    print(f"  {name:5} n={out['n']:2}  sharpe median {out['sharpe_median']:+7.3f}  "
          f"[{out['sharpe_worst']:+.2f}, {out['sharpe_best']:+.2f}]  "
          f"pos_coins median {out['positive_coins_median']}  "
          f"|pos| median {out['abs_mean_position_median']}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    a = ap.parse_args()

    R = load(a.dir / "pilot_real_seed*.json")
    N = load(a.dir / "pilot_null_seed*.json")
    if not R or not N:
        print(f"incomplete batch under {a.dir}: real={len(R)} null={len(N)}")
        print("the pre-gate needs all 10 + 10 before it is read")
        return 2

    print("=" * 74)
    print("P2 PRE-GATE  formula-only, repaired meter, closed validation leak")
    print(f"dir: {a.dir}")
    print("=" * 74)
    modes = {r.get("select_on") for r in R} | {n.get("select_on") for n in N}
    print(f"select_on across the batch: {modes}")
    print()

    rs = describe("real", R)
    ns = describe("null", N)
    print()

    gap = rs["sharpe_median"] - ns["sharpe_median"]
    diffs = [x["sharpe"] - y["sharpe"] for x, y in zip(R, N)]
    k, n, sp = sign_test(diffs)
    U, p, z = mannwhitney_one_sided([r["sharpe"] for r in R], [r["sharpe"] for r in N])
    breadth_ok = sum(1 for r in R if (r["positive_coins"] or 0) >= PASS_MIN_POSITIVE_COINS)
    dir_ok = sum(1 for r in R if abs(r["mean_position"] or 0.0) < 0.25)

    # BUY-AND-HOLD EXCESS is the criterion that matters most and it was missing from
    # the first version of this gate.
    #
    # The holdout here is 63 days of uptrend with a buy-and-hold Sharpe of +3.675.
    # A search that settles on a constant leveraged long will land near that number
    # without predicting anything. Measured across the real arm: portfolio Sharpe
    # ranged +0.00 to +3.67 while excess-over-buy-and-hold was NEGATIVE on every
    # seed, and corr(sharpe, |mean_position|) = +0.60. That is beta, not alpha.
    #
    # So a raw Sharpe comparison between arms is not sufficient: both arms can be
    # graded on how much exposure they took into the same rising tape.
    ex_r = [r["excess"] for r in R if r["excess"] is not None]
    ex_n = [r["excess"] for r in N if r["excess"] is not None]
    excess_pos = sum(1 for x in ex_r if x > 0)
    ex_median_r = st.median(ex_r) if ex_r else None
    ex_median_n = st.median(ex_n) if ex_n else None
    ex_gap = (ex_median_r - ex_median_n) if (ex_median_r is not None and ex_median_n is not None) else None

    criteria = {
        "real_median_beats_null": gap > 0,
        "mannwhitney_one_sided": p < PASS_ALPHA,
        "breadth": breadth_ok >= PASS_MIN_COIN_SEEDS,
        "direction": dir_ok >= PASS_MIN_COIN_SEEDS,
        "beats_buy_and_hold": excess_pos >= PASS_MIN_COIN_SEEDS,
    }
    passed = all(criteria.values())

    print("DECISION (pre-registered)")
    for name, ok in criteria.items():
        detail = {
            "real_median_beats_null": f"gap {gap:+.3f} (needs > 0)",
            "mannwhitney_one_sided": f"p={p:.3f} (needs < {PASS_ALPHA}, one-sided)",
            "breadth": f"{breadth_ok}/{len(R)} seeds with pos_coins >= {PASS_MIN_POSITIVE_COINS}"
                       f" (needs >= {PASS_MIN_COIN_SEEDS})",
            "direction": f"{dir_ok}/{len(R)} seeds with |mean_position| < 0.25"
                         f" (needs >= {PASS_MIN_COIN_SEEDS})",
        "beats_buy_and_hold": f"{excess_pos}/{len(R)} seeds with excess_vs_buy_and_hold > 0"
                              f" (needs >= {PASS_MIN_COIN_SEEDS}); real median {ex_median_r:+.3f}"
                              if ex_median_r is not None else "n/a",
        }[name]
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:26} {detail}")
    print()
    print(f"  real wins {k}/{n} seeds, sign-test one-sided p={sp:.3f}")
    print()
    print("BUY-AND-HOLD CHECK")
    if ex_r:
        print(f"  buy_and_hold sharpe on the holdout: {R[0]['bh']:+.3f}" if R and R[0].get("bh") is not None else "")
        print(f"  real excess median    {ex_median_r:+.3f}   ({excess_pos}/{len(ex_r)} seeds positive)")
        print(f"  null excess median    {ex_median_n:+.3f}")
        print(f"  real - null excess    {ex_gap:+.3f}" if ex_gap is not None else "")
        print("  A positive raw Sharpe on this window can be produced by exposure")
        print("  alone, so this check is not decorative.")
    print()
    print(f"VERDICT: {'PASS' if passed else 'FAIL'}")
    if not passed:
        print()
        print("A FAIL means PRP-1, PRP-2 and PRP-3 are NO-GO as specified: each")
        print("carries this precondition. Record it in docs/ rather than widening")
        print("the search to chase it. A FAIL is weak evidence at n=10; a PASS")
        print("would have been meaningful.")

    out = {
        "dir": str(a.dir),
        "select_on": sorted(x for x in modes if x),
        "real": rs, "null": ns,
        "median_gap": gap,
        "real_wins": k, "n": n, "sign_test_one_sided_p": sp,
        "mannwhitney_U": U, "mannwhitney_one_sided_p": p, "mannwhitney_z": z,
        "breadth_seeds": breadth_ok, "direction_seeds": dir_ok,
        "real_excess_median": ex_median_r, "null_excess_median": ex_median_n,
        "real_excess_positive_seeds": excess_pos,
        "criteria": criteria,
        "verdict": "PASS" if passed else "FAIL",
        "power_note": ("n=10 per arm; one-sided MW p<0.10 needs about 8/10 "
                       "dominance and has roughly 38% power at a true per-seed win "
                       "rate of 0.7, so a FAIL is weak evidence"),
    }
    p_out = ROOT / "results" / "p2_pregate_verdict.json"
    p_out.write_text(json.dumps(out, indent=2, default=float))
    print()
    print("wrote", p_out)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
