#!/usr/bin/env python3
"""Search-effort ladder: hold the hypothesis class fixed and vary only how
hard the search is pushed.

This separates two failures that the variable-length A/B could not tell
apart. If holdout excess degrades monotonically with generations on BOTH
real and null, the search is overfitting and the fix is less search or
regularisation. If it degrades on null but not on real, there is real
structure being destroyed by effort. If it is flat on both, effort is not
the variable and the factor set has no conditional timing power to find.

The arm is fixed-12 on purpose: the variable-length arm already showed that
freeing the length degrades null performance, so it is not the clean
baseline for a dose-response curve.
"""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EFFORTS = (2, 5, 10, 20)


def one(job):
    (gens, data, seed, outdir, pop, folds, objective, factors) = job
    tag = "g%02d_%s_s%d" % (gens, data, seed)
    if objective != "v3":
        tag += "_" + objective
    if factors != "12":
        tag += "_f" + factors
    out = Path(outdir) / (tag + ".json")
    null = "iid" if data == "null" else "none"
    cmd = [sys.executable, str(ROOT / "research" / "ga_28c_30m_3y.py"),
           "--mode", "walkforward", "--folds", str(folds),
           "--generations", str(gens), "--population", str(pop),
           "--grammar", "reduced", "--reward-version", "v3",
           "--objective", objective, "--select-on", "train",
           "--formula-len", "12", "--factors", factors, "--seed", str(seed),
           "--null", null, "--null-seed", str(seed),
           "--out", str(out)]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return {"tag": tag, "gens": gens, "data": data, "seed": seed,
            "ok": r.returncode == 0 and out.exists(),
            "secs": round(time.time() - t0, 1),
            "err": "" if r.returncode == 0 else r.stderr[-800:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--start-seed", type=int, default=101)
    ap.add_argument("--population", type=int, default=16)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--efforts", type=int, nargs="+", default=list(EFFORTS))
    ap.add_argument("--outdir", type=Path, default=ROOT / "results" / "effort_ladder")
    ap.add_argument("--objective", default="v3", help="reward preset; v3 is the control")
    ap.add_argument("--factors", default="12", choices=["12", "36"], help="feature set")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    # The objective is a parameter here so the same harness can run the
    # reward-vs-gate experiment at one fixed effort level.
    if args.outdir == ROOT / "results" / "effort_ladder" and args.objective != "v3":
        args.outdir = ROOT / "results" / ("effort_ladder_" + args.objective)
    if args.factors != "12" and args.outdir == ROOT / "results" / "effort_ladder":
        args.outdir = ROOT / "results" / ("effort_ladder_f" + args.factors)
    seeds = [args.start_seed + i for i in range(args.seeds)]
    jobs = [(g, d, s, str(args.outdir), args.population, args.folds, args.objective, args.factors) for g in args.efforts for d in ("real", "null") for s in seeds]
    print(str(len(jobs)) + " runs on " + str(args.workers) + " workers")
    done = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(one, j) for j in jobs]
        for k, fu in enumerate(as_completed(futs), 1):
            rec = fu.result()
            done.append(rec)
            flag = "ok  " if rec["ok"] else "FAIL"
            print("[%d/%d] %s %-18s %6ss" % (k, len(jobs), flag, rec["tag"], rec["secs"]), flush=True)
            if not rec["ok"]:
                print("      " + rec["err"].replace(chr(10), " ")[:400], flush=True)
    idx = args.outdir / "index.json"
    idx.write_text(json.dumps(done, indent=2))
    bad = [d["tag"] for d in done if not d["ok"]]
    print()
    print(str(len(done) - len(bad)) + "/" + str(len(done)) + " ok; index -> " + str(idx))
    if bad:
        print("failed:", bad)


if __name__ == "__main__":
    main()