#!/usr/bin/env python3
"""Selection-rule batch harness: S0/A1/N0 arms, seeds 501-1000 paired.

One arm = 500 real + 500 null walkforward runs, 10gen/16pop, 41f/1h/sw,
--select-on {train,a1,n0}. Resumable via --skip-existing. Order per FINAL_SPEC:
FM, S0-gate, A1+N0, B1 pilot (FM/B1 need separate implementation).

Usage:
  python3 research/run_selection_rule_28c.py --arm s0 --workers 8
  python3 research/run_selection_rule_28c.py --arm a1 --workers 8 --skip-existing
"""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ARMS = {"s0": "train", "a1": "a1", "n0": "n0"}
DIRNAME = {"s0": "f1h41_s0_rerun", "a1": "f1h41_a1", "n0": "f1h41_n0"}


def one(job):
    (data, seed, outdir, select_on) = job
    tag = "%s_s%d" % (data, seed)
    out = Path(outdir) / (tag + ".json")
    null = "iid" if data == "null" else "none"
    cmd = [sys.executable, str(ROOT / "research" / "ga_28c_30m_3y.py"),
           "--mode", "walkforward", "--folds", "5",
           "--generations", "10", "--population", "16",
           "--grammar", "reduced", "--reward-version", "v3",
           "--objective", "v3", "--select-on", select_on,
           "--formula-len", "12", "--factors", "41", "--bars", "1h",
           "--scale-windows",
           "--seed", str(seed),
           "--null", null, "--null-seed", str(seed),
           "--out", str(out)]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return {"tag": tag, "data": data, "seed": seed,
            "ok": r.returncode == 0 and out.exists(),
            "secs": round(time.time() - t0, 1),
            "err": "" if r.returncode == 0 else r.stderr[-800:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--seeds", type=int, default=500)
    ap.add_argument("--start-seed", type=int, default=501)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()
    select_on = ARMS[args.arm]
    outdir = args.outdir or (ROOT / "results" / DIRNAME[args.arm])
    outdir.mkdir(parents=True, exist_ok=True)
    seeds = [args.start_seed + i for i in range(args.seeds)]
    jobs = [(d, s, str(outdir), select_on)
            for d in ("real", "null") for s in seeds]
    if args.skip_existing:
        jobs = [j for j in jobs if not (Path(j[2]) / ("%s_s%d.json" % (j[0], j[1]))).exists()]
    print(f"{len(jobs)} runs arm={args.arm} select-on={select_on} workers={args.workers} -> {outdir}", flush=True)
    done = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(one, j) for j in jobs]
        for k, fu in enumerate(as_completed(futs), 1):
            rec = fu.result()
            done.append(rec)
            flag = "ok  " if rec["ok"] else "FAIL"
            print("[%d/%d] %s %-14s %6ss" % (k, len(jobs), flag, rec["tag"], rec["secs"]), flush=True)
            if not rec["ok"]:
                print("      " + rec["err"].replace(chr(10), " ")[:400], flush=True)
    idx = outdir / "index.json"
    idx.write_text(json.dumps(done, indent=2))
    bad = [d["tag"] for d in done if not d["ok"]]
    print(f"\n{len(done)-len(bad)}/{len(done)} ok; index -> {idx}")
    if bad:
        print("failed:", bad)


if __name__ == "__main__":
    main()
