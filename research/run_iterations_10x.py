#!/usr/bin/env python3
"""Ten strategy iterations, paired as an A/B on the reward function.

The user asked for ten iterations. Running the same search ten times would
produce ten numbers that are all expected to be negative, and would answer
nothing. The one hypothesis in this project that has never been tested is the
objective mismatch: `shape_reward` v1 and v2 maximise ABSOLUTE Sharpe, while
the accept gate tests EXCESS over a constant levered long. Those are different
objectives and nothing connected them, so for two full iterations the search
was optimising something the gate cannot accept.

So the ten runs are paired:

    iterations 1-5   reward v1   absolute Sharpe      (control)
    iterations 6-10  reward v3   + excess vs hold     (treatment)

same seed, same generations, same population, same walk-forward folds, same
reduced grammar, same real funding, every run paired with a frozen-iid null
companion. If v3 helps, the treatment arm's excess should move; if the
factors are the binding constraint, both arms will be negative and the
difference will be noise, which is itself the answer.

The null companion is not optional. A run without one is a number, not a
result: the search carries a +5 to +13 Sharpe selection lift over a random
formula, so an absolute Sharpe carries no information about edge.

Paper-only and broker-free. `--live` is refused by the loop, and nothing here
adopts a strategy: a run that beats the hold would still have to clear the
regime gate A1-A6, which no run in this project has ever done.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv2" / "bin" / "python"
RESULTS = ROOT / "results"


def one(iteration, reward, seed, gens, pop, with_null, timeout):
    """Run one paired iteration through the project's own loop."""
    cmd = [str(PY), "research/run_loop_28c.py",
           "--stage", "iterate",
           "--mode", "walkforward",
           "--grammar", "reduced",
           "--generations", str(gens),
           "--population", str(pop),
           "--seed", str(seed),
           "--reward-version", reward,
           "--skip-tests"]
    if with_null:
        cmd.append("--with-null")
    t0 = time.time()
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    return {
        "iteration": iteration, "reward": reward, "seed": seed,
        "generations": gens, "population": pop,
        "rc": p.returncode,
        "seconds": round(time.time() - t0, 1),
        "stdout_tail": "\n".join(
            l for l in p.stdout.splitlines() if "warn" not in l.lower())[-1200:],
        "stderr_tail": p.stderr[-600:],
    }


def harvest(iteration, reward):
    """Pull the holdout numbers out of the artifacts this iteration produced."""
    out = {}
    for tag in ("real", "iid"):
        cands = sorted(RESULTS.glob(f"loop_*_c1_{tag}.json"), key=lambda p: p.stat().st_mtime)
        if not cands:
            continue
        art = json.loads(cands[-1].read_text())
        blk = art.get("holdout") or art.get("oos") or {}
        out[tag] = {
            "sharpe": blk.get("portfolio_sharpe"),
            "buy_and_hold": blk.get("buy_and_hold_sharpe"),
            "excess": blk.get("excess_sharpe_vs_buy_and_hold"),
            "mean_position": blk.get("mean_position"),
            "positive_coins": blk.get("positive_coins"),
            "gate": (art.get("gate") or {}).get("verdict") if isinstance(art.get("gate"), dict)
                    else art.get("gate_verdict"),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--generations", type=int, default=15)
    ap.add_argument("--population", type=int, default=20)
    ap.add_argument("--seed-base", type=int, default=20260930)
    ap.add_argument("--no-null", action="store_true")
    ap.add_argument("--stage-timeout", type=int, default=14400)
    ap.add_argument("--out", default=str(RESULTS / "ab_reward_10x.json"))
    a = ap.parse_args()

    with_null = not a.no_null
    log = []
    for i in range(1, a.iterations + 1):
        # paired: the control arm first, then the treatment, same seed per pair
        reward = "v1" if i <= a.iterations // 2 else "v3"
        seed = a.seed_base + i
        rec = one(i, reward, seed, a.generations, a.population, with_null, a.stage_timeout)
        rec["results"] = harvest(i, reward)
        log.append(rec)
        r = rec["results"].get("real", {})
        nl = rec["results"].get("iid", {})
        print(f"[{i}/{a.iterations}] reward={reward} seed={seed} rc={rec['rc']} "
              f"excess={r.get('excess')} bh={r.get('buy_and_hold')} "
              f"null_excess={nl.get('excess')} ({rec['seconds']}s)", flush=True)
        Path(a.out).write_text(json.dumps(log, indent=1))

    Path(a.out).write_text(json.dumps(log, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
