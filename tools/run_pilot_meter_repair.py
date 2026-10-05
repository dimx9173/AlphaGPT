#!/usr/bin/env python3
"""Driver for the meter-repair pilot: 10 real vs 10 null walkforward searches.

Why this exists
---------------
The pilot contract in docs/GA_SEARCH_RELIABILITY_2026-10-01.md freezes the
engine and changes only the meter. A real-vs-null comparison is therefore only
meaningful if BOTH arms run on the same commit with the same seeds and the same
fold layout, and if the only difference is the data.

The two arms are separate process invocations of ga_28c_30m_3y.py, not a flag on
one process, because load_data() seeds from --seed and --null-seed independently
and mixing them inside one process risks a shared RNG path.

Seeds 101-110 match the recorded ga_iter_runs / ga_null_runs batches so the
before/after comparison is paired rather than two independent draws.

Usage:
    python tools/run_pilot_meter_repair.py --workers 4
    python tools/run_pilot_meter_repair.py --dry-run
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GA = ROOT / "research" / "ga_28c_30m_3y.py"
PY = ROOT / ".venv2" / "bin" / "python"
SEEDS = list(range(101, 111))


def one_run(arm, seed, out_dir, generations, population, folds):
    """One search. Returns (arm, seed, ok, seconds, path_or_error)."""
    out = Path(out_dir) / f"pilot_{arm}_seed{seed}.json"
    cmd = [
        str(PY), str(GA),
        "--mode", "walkforward",
        "--folds", str(folds),
        "--generations", str(generations),
        "--population", str(population),
        "--seed", str(seed),
        "--null-seed", str(seed),
        "--funding", "real",
        "--reward-version", "v1",
        "--out", str(out),
    ]
    if arm == "null":
        cmd += ["--null", "iid"]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=14400)
    except subprocess.TimeoutExpired:
        return arm, seed, False, time.time() - t0, "timeout"
    dt = time.time() - t0
    if r.returncode != 0:
        return arm, seed, False, dt, r.stderr.strip()[-400:]
    return arm, seed, True, dt, str(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--generations", type=int, default=25)
    ap.add_argument("--population", type=int, default=16)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results" / "pilot_meter_repair")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    a.out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(arm, s) for arm in ("real", "null") for s in SEEDS]

    if a.dry_run:
        print(f"{len(jobs)} runs: {len(SEEDS)} real + {len(SEEDS)} null")
        print(f"each: mode=walkforward folds={a.folds} gen={a.generations} pop={a.population} "
              f"funding=real reward=v1 seeds={SEEDS[0]}-{SEEDS[-1]}")
        print(f"workers={a.workers}, out={a.out_dir}")
        print()
        print("sample command:")
        print(" ".join([str(PY), str(GA), "--mode", "walkforward", "--folds", str(a.folds),
                        "--generations", str(a.generations), "--population", str(a.population),
                        "--seed", "101", "--null-seed", "101", "--funding", "real",
                        "--reward-version", "v1", "--out", str(a.out_dir / "pilot_real_seed101.json")]))
        return 0

    print(f"launching {len(jobs)} runs on {a.workers} workers", flush=True)
    t0 = time.time()
    done = 0
    failed = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(one_run, arm, s, a.out_dir, a.generations, a.population, a.folds)
                for arm, s in jobs]
        for fut in futs:
            arm, seed, ok, dt, info = fut.result()
            done += 1
            if ok:
                print(f"[{done:2}/{len(jobs)}] {arm:5} seed{seed} OK {dt/60:.1f}m", flush=True)
            else:
                failed.append((arm, seed, info))
                print(f"[{done:2}/{len(jobs)}] {arm:5} seed{seed} FAIL {dt/60:.1f}m :: {info[:160]}", flush=True)

    print(f"\nfinished {done} runs in {(time.time()-t0)/60:.1f} min; {len(failed)} failed")
    if failed:
        print("failures:")
        for arm, seed, info in failed:
            print(f"  {arm} seed{seed}: {info[:200]}")
        return 1
    print("wrote", a.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
