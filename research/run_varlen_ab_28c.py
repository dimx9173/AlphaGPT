#!/usr/bin/env python3
"""A/B: fixed 12-token genomes vs variable 6..20, real data vs null data.

The question is NOT which arm scores higher on real data. A higher number
on real data only means the bigger search found more noise, and this code
path already documents that a positive Sharpe can be worse than doing
nothing. The question is whether each arm beats its OWN null arm. An arm
that gains on real and gains on null by the same amount has bought
multiple testing, not an edge.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def one(job):
    (arm, data, seed, outdir, gens, pop, folds) = job
    tag = f"{arm}_{data}_s{seed}"
    out = Path(outdir) / f"{tag}.json"
    null = "iid" if data == "null" else "none"
    cmd = [sys.executable, str(ROOT / "research" / "ga_28c_30m_3y.py"),
           "--mode", "walkforward", "--folds", str(folds),
           "--generations", str(gens), "--population", str(pop),
           "--grammar", "reduced", "--reward-version", "v3",
           "--objective", "v3", "--select-on", "train",
           "--formula-len", arm, "--seed", str(seed),
           "--null", null, "--null-seed", str(seed),
           "--out", str(out)]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return {"tag": tag, "arm": arm, "data": data, "seed": seed,
            "ok": r.returncode == 0 and out.exists(),
            "secs": round(time.time() - t0, 1),
            "err": "" if r.returncode == 0 else r.stderr[-800:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--start-seed", type=int, default=101)
    ap.add_argument("--generations", type=int, default=10)
    ap.add_argument("--population", type=int, default=16)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--outdir", type=Path,
                    default=ROOT / "results" / "varlen_ab")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    seeds = [args.start_seed + i for i in range(args.seeds)]
    jobs = [(arm, data, s, str(args.outdir), args.generations,
             args.population, args.folds)
            for arm in ("12", "varlen")
            for data in ("real", "null") for s in seeds]
    print(f"{len(jobs)} runs on {args.workers} workers: {args.generations} gen x pop {args.population}, {args.folds} folds")
    done = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(one, j): j for j in jobs}
        for k, fu in enumerate(as_completed(futs), 1):
            rec = fu.result()
            done.append(rec)
            flag = "ok " if rec["ok"] else "FAIL"
            print("[" + str(k) + "/" + str(len(jobs)) + "] " + flag + " " + rec["tag"] + " " + str(rec["secs"]) + "s", flush=True)
            if not rec["ok"]:
                print("      " + rec["err"].replace(chr(10), " ")[:400], flush=True)
    idx = args.outdir / "index.json"
    idx.write_text(json.dumps(done, indent=2))
    bad = [d["tag"] for d in done if not d["ok"]]
    print(f"\n{len(done)-len(bad)}/{len(done)} ok; index -> {idx}")
    if bad:
        print("failed:", bad)


if __name__ == "__main__":
    main()