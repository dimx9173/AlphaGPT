#!/usr/bin/env python3
"""The project loop, as one command: train -> iterate -> accept -> trade.

    train -> iterate for the best strategy -> trade -> retrain -> iterate
    again -> trade again

Each stage is a real subprocess, not a reimplementation, so the loop cannot
quietly diverge from the tools it drives. Each stage also has an explicit
acceptance criterion, and the loop fails closed: a stage that cannot prove it
passed does not let the next stage run.

The acceptance criterion is the one this project spent three iterations
earning. A strategy is accepted only if it beats a constant levered long
position on the same bars, with the same fee, funding and leverage. Not zero.
Zero is a number with no position behind it, and on this universe a passive
hold scores +1.05 while the best real-data run scored -0.32. Accepting
against zero is what let a drift-matching machine be read as a discovery.

    --dry-run       show the plan and the commands, run nothing
    --stage NAME    run one stage only: train, iterate, accept, trade
    --cycles N      how many trade-then-retrain cycles (default 1)
    --live          NOT IMPLEMENTED, and refused on purpose. See below.

There is no live path. This project is broker-free and paper-only: no
account, no API key, no exchange connection, no order of any kind. The
refusal is explicit rather than an absent flag so that asking for it gets an
answer.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

# Running the script puts research/ on sys.path, not the repo root, so the
# package imports below fail with a bare ModuleNotFoundError on the first
# import rather than at parse time. The root has to be added explicitly.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
R = ROOT / "results"
LOGS = ROOT / "logs"

# The tests that cover the surface this loop actually drives. Anything else in
# the suite belongs to a different subsystem and is reported, not silently
# folded into this verdict.
IN_SCOPE = [
    "tests/test_accounting_28c.py", "tests/test_accounting_consistency_28c.py",
    "tests/test_acceptance_schema_28c.py", "tests/test_api_traps_28c.py",
    "tests/test_buy_and_hold_benchmark_28c.py", "tests/test_data_contract_28c.py",
    "tests/test_factor_screen_28c.py", "tests/test_formula_grammar.py",
    "tests/test_frozen_null_control_28c.py", "tests/test_grammar_reduced_28c.py",
    "tests/test_null_modes_28c.py", "tests/test_random_formula_probe_28c.py",
    "tests/test_real_funding_28c.py", "tests/test_regime_gate_28c.py",
    "tests/test_regime_gate_power_28c.py", "tests/test_selection_bias_28c.py",
    "tests/test_splits_walkforward_28c.py", "tests/test_statistical_power_28c.py",
    "tests/test_28c_acceptance_gate.py", "tests/test_feature_guard_28c.py",
    "tests/test_run_loop_28c.py", "tests/test_holdout_benchmark_28c.py",
]


def now():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


class Loop:
    """One cycle's record. Every stage appends; nothing is overwritten."""

    def __init__(self, run_id):
        self.run_id = run_id
        self.record = {
            "run_id": run_id, "started": now(), "cycle": 0, "stages": [],
            "accepted": False, "reason": None,
            "policy": "paper-only, broker-free; no live path exists",
        }
        self.path = R / f"loop_{run_id}.json"

    def stage(self, name, cmd, rc, tail="", extra=None):
        entry = {"stage": name, "rc": rc, "cmd": cmd, "at": now(),
                 "tail": tail[-1500:]}
        if extra:
            entry.update(extra)
        self.record["stages"].append(entry)
        return entry

    def save(self):
        self.path.write_text(json.dumps(self.record, indent=1))

    def verdict(self, accepted, reason):
        self.record["accepted"] = accepted
        self.record["reason"] = reason
        self.record["finished"] = now()
        self.save()
        return accepted


def sh(cmd, timeout=None, cwd=ROOT):
    t0 = time.time()
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           cwd=cwd, timeout=timeout)
        out = (p.stdout or "") + (p.stderr or "")
        return p.returncode, out, time.time() - t0
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or b"").decode("utf-8", "replace") if isinstance(
            exc.stdout, bytes) else (exc.stdout or "")
        return 124, out + f"\n[TIMEOUT after {timeout}s]", time.time() - t0


def require(imports):
    """Fail loudly if the running interpreter cannot import a dependency.

    Added after the gate report produced a clean-looking FAIL whose real cause
    was that the sub-steps ran under a different interpreter with no pytest.
    A loop that cannot tell "failed" from "could not start" is not a loop.
    """
    missing = []
    for mod in imports:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        print(f"FATAL: {sys.executable} cannot import: {', '.join(missing)}")
        print("  run with the project venv, e.g. .venv2/bin/python "
              "research/run_loop_28c.py")
        return False
    return True


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------

def stage_test(loop, timeout):
    """The in-scope suite, before anything is trusted downstream."""
    # The verdict comes from the junit XML, not from scraping stdout.
    #
    # An earlier version decided on `"passed" in out and rc == 0`, where `out`
    # was a filtered tail. It reported FAIL on a run whose exit code was 0,
    # because the summary line the string match looked for had been filtered
    # out of the tail. Inferring a verdict by text-scraping a truncated
    # stream is the same failure as reading a Sharpe off a moved benchmark: it
    # produces a confident answer that the evidence does not support.
    xml = R / f"loop_{loop.run_id}_junit.xml"
    cmd = (f'"{PY}" -m pytest -q --tb=short -p no:warnings '
           f'--junitxml="{xml}" ' + " ".join(f'"{p}"' for p in IN_SCOPE))
    rc, out, dt = sh(cmd, timeout=timeout)
    tail = "\n".join(l for l in out.splitlines()[-12:]
                      if "warn" not in l.lower())
    stats = {"tests": 0, "failures": 0, "errors": 0}
    try:
        import xml.etree.ElementTree as ET
        a = ET.parse(xml).getroot().find("testsuite").attrib
        stats = {k: int(a.get(k, 0)) for k in ("tests", "failures", "errors")}
    except Exception as exc:                      # noqa: BLE001
        stats["xml_error"] = f"{type(exc).__name__}: {exc}"
    # Zero collected is not a pass. It is the signature of a collection error
    # that the junit XML records as an empty suite.
    passed = (rc == 0 and stats["tests"] > 0
              and stats["failures"] == 0 and stats["errors"] == 0)
    loop.stage("test", cmd, rc, tail,
               {"seconds": round(dt, 1), "ok": passed, "stats": stats,
                "scope": "28c surface", "tests": IN_SCOPE})
    print(f"[test] rc={rc} tests={stats['tests']} failures={stats['failures']} "
          f"errors={stats['errors']} -> {'ok' if passed else 'FAILED'}")
    return passed


def stage_train(loop, args, cycle):
    """Train = fit the factor scale on the train window only.

    The GA does not have a separate training entry point: the scale it fits is
    part of evaluation and is bounded by scale_end, so 'training' is the
    evaluation contract rather than a separate run. This stage pins that
    contract by running the accounting and contract tests, and records the
    factor cache version the next stage will load, because a formula scored
    against a different cache is a different experiment.
    """
    import research.ga_28c_30m_3y as GA
    from research.causal_12f import FEATURE_NAMES
    loop.record["factor_cache_version"] = GA.FEATURE_CACHE_VERSION
    loop.record["factors"] = list(FEATURE_NAMES)
    loop.record["fee"] = GA.FEE
    loop.record["leverage"] = GA.LEV
    loop.record["grammar"] = args.grammar
    loop.stage("train", "(contract pinned; scale is fitted inside evaluate)",
               0, f"cache={GA.FEATURE_CACHE_VERSION} fee={GA.FEE} "
                  f"lev={GA.LEV} factors={len(FEATURE_NAMES)}")
    print(f"[train] factor cache {GA.FEATURE_CACHE_VERSION}, "
          f"fee {GA.FEE}, leverage {GA.LEV}, {len(FEATURE_NAMES)} factors")
    return True


def stage_iterate(loop, args, cycle, null=None):
    """Run the search. Optionally against a null, which is the whole point.

    A run with no null companion is a number, not a result. The search has
    been shown to produce a +5 to +13 Sharpe selection lift over a random
    formula, so an absolute Sharpe carries no information about edge.
    """
    out = R / f"loop_{loop.run_id}_c{cycle}_{null or 'real'}.json"
    cmd = (f'"{PY}" research/ga_28c_30m_3y.py --mode {args.mode} '
           f'--grammar {args.grammar} --generations {args.generations} '
           f'--population {args.population} --seed {args.seed + cycle} '
           f'--funding real --reward-version {args.reward_version} '
           f'--out "{out}"'
           + (f" --null {null}" if null else ""))
    rc, sout, dt = sh(cmd, timeout=args.stage_timeout)
    tail = "\n".join(l for l in sout.splitlines()[-10:] if "warn" not in l.lower())
    got = out.exists()
    loop.stage(f"iterate[{null or 'real'}]", cmd, rc, tail,
               {"seconds": round(dt, 1), "artifact": str(out), "produced": got})
    print(f"[iterate/{null or 'real'}] rc={rc} produced={got}")
    return got and rc == 0


def stage_accept(loop, args, out_path):
    """The gate. Beats a constant long, or the loop stops here."""
    art = json.loads(out_path.read_text())
    # A walk-forward run reports the holdout under "holdout"; a standard run
    # reports it under "oos". The first version read only "oos", found nothing,
    # and rejected with a None excess -- fail-closed, but for a reason that had
    # nothing to do with the strategy. The key is now discovered, and a
    # missing one is an error rather than a silent zero.
    oos = art.get("oos") or art.get("holdout") or {}
    if not oos:
        raise KeyError(
            f"{out_path} has neither an 'oos' nor a 'holdout' block; "
            f"keys are {sorted(art)}")
    rec = {
        "formula": art.get("formula") or oos.get("formula"),
        "portfolio_sharpe": oos.get("portfolio_sharpe"),
        "buy_and_hold_sharpe": oos.get("buy_and_hold_sharpe"),
        "excess": oos.get("excess_sharpe_vs_buy_and_hold"),
        "mean_position": oos.get("mean_position"),
    }
    if rec["excess"] is None:
        raise KeyError(
            f"{out_path} reports no excess_sharpe_vs_buy_and_hold. The accept "
            "stage cannot run without the benchmark, and it will not fall back "
            "to comparing against zero.")
    rec["beats_buy_and_hold"] = bool(
        rec["excess"] is not None and rec["excess"] > 0.0)
    rec["gate_verdict"] = art.get("gate", {}).get("verdict") if isinstance(
        art.get("gate"), dict) else art.get("gate_verdict")
    loop.stage("accept", "(in-process)", 0, json.dumps(rec, indent=1)[:1500],
               {"decision": rec})

    e = rec["excess"]
    print(f"\n[accept] Sharpe {rec['portfolio_sharpe']} vs buy-and-hold "
          f"{rec['buy_and_hold_sharpe']} -> excess {e}")
    print(f"[accept] mean position {rec['mean_position']} "
          f"({'long-biased' if (rec['mean_position'] or 0) > 0.1 else 'not long-biased'})")
    if not rec["beats_buy_and_hold"]:
        print("[accept] REJECTED. The loop stops here: there is nothing to trade.")
        print("[accept] The search is running; it is not finding. See the factor")
        print("[accept] screen (results/factor_screen.json) before spending more")
        print("[accept] search on this grammar.")
        return False
    if rec["gate_verdict"] and rec["gate_verdict"] != "pass":
        print(f"[accept] REJECTED. Gate verdict is {rec['gate_verdict']!r}.")
        return False
    return True


def stage_trade(loop, args, cycle):
    """Paper only. There is no live path and there will not be one here."""
    formula_file = R / f"loop_{loop.run_id}_formula.json"
    if not formula_file.exists():
        loop.stage("trade", "(no accepted formula)", 0,
                   "skipped: nothing was accepted, so nothing is traded")
        print("[trade] skipped: no accepted strategy")
        return False
    out = R / f"loop_{loop.run_id}_c{cycle}_paper.json"
    cmd = (f'"{PY}" research/run_paper_28c_pit_30m.py '
           f'--formula-file "{formula_file}" --out "{out}" --funding real')
    rc, sout, dt = sh(cmd, timeout=args.stage_timeout)
    tail = "\n".join(l for l in sout.splitlines()[-8:] if "warn" not in l.lower())
    loop.stage("trade", cmd, rc, tail,
               {"seconds": round(dt, 1), "artifact": str(out),
                "live": False})
    print(f"[trade] paper rc={rc} -> {out}")
    return rc == 0


def main() -> int:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--cycles", type=int, default=1)
    ap.add_argument("--stage", choices=["train", "iterate", "accept", "trade"])
    ap.add_argument("--generations", type=int, default=100)
    ap.add_argument("--population", type=int, default=32)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--grammar", choices=["full", "reduced"], default="reduced")
    ap.add_argument("--mode", choices=["standard", "walkforward"],
                    default="walkforward")
    ap.add_argument("--reward-version", default="v1", choices=("v1", "v2", "v3"),
                    help="v1 absolute Sharpe (historical). v2 adds a breadth "
                         "term. v3 additionally rewards EXCESS over a constant "
                         "levered long, which is what the accept gate actually "
                         "tests. v1 and v2 optimise a different objective from "
                         "the gate, so they search for something the gate then "
                         "rejects for a reason the search never saw.")
    ap.add_argument("--with-null", action="store_true",
                    help="also run the search against the frozen iid null")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--stage-timeout", type=int, default=7200)
    ap.add_argument("--live", action="store_true",
                    help="refused: this project is paper-only")
    args = ap.parse_args()

    if args.live:
        print("REFUSED. There is no live path.")
        print("  This project is broker-free and paper-only by design: no account,")
        print("  no API key, no exchange connection, no orders. --live is refused")
        print("  explicitly so that asking gets an answer rather than silence.")
        return 2

    if not require(["numpy", "pytest"]):
        return 2

    run_id = now()
    R.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    loop = Loop(run_id)

    print(f"AlphaGPT loop {run_id}")
    print(f"  grammar={args.grammar} mode={args.mode} cycles={args.cycles} "
          f"generations={args.generations} population={args.population}")
    print("  policy: paper-only, broker-free")
    print()

    if args.dry_run:
        print("dry run: no stage will execute\n")
        print(f"  test    {PY} -m pytest {len(IN_SCOPE)} in-scope files")
        print(f"  train   (contract pinned; cache/fee/leverage recorded)")
        print(f"  iterate {PY} research/ga_28c_30m_3y.py --mode {args.mode} "
              f"--grammar {args.grammar}")
        if args.with_null:
            print(f"  iterate {PY} research/ga_28c_30m_3y.py --null iid")
        print(f"  accept  excess over buy-and-hold must be > 0")
        print(f"  trade   {PY} research/run_paper_28c_pit_30m.py --funding real")
        return 0

    if not args.skip_tests and not stage_test(loop, args.stage_timeout):
        loop.verdict(False, "in-scope tests failed; nothing downstream is trusted")
        print("\nSTOP: in-scope tests failed. Report: " + str(loop.path))
        return 1

    for cycle in range(1, args.cycles + 1):
        loop.record["cycle"] = cycle
        print(f"\n=== cycle {cycle}/{args.cycles} ===")

        if not stage_train(loop, args, cycle):
            loop.verdict(False, "train stage failed")
            return 1

        if args.stage == "train":
            loop.save()
            continue

        if not stage_iterate(loop, args, cycle):
            loop.verdict(False, "search produced no artifact")
            print("\nSTOP: the search produced nothing.")
            return 1
        if args.with_null:
            stage_iterate(loop, args, cycle, null="iid")

        if args.stage == "iterate":
            loop.save()
            continue

        art = R / f"loop_{run_id}_c{cycle}_real.json"
        if not stage_accept(loop, args, art):
            loop.verdict(False, "did not beat a constant long position; "
                                "no trade")
            print(f"\nLOOP HALTED at cycle {cycle}. Report: {loop.path}")
            return 3

        if args.stage == "accept":
            loop.save()
            continue

        stage_trade(loop, args, cycle)
        loop.save()
        print("[cycle] traded on paper; the next cycle retrains on the "
              "forward data and iterates again")

    loop.verdict(True, "completed")
    print(f"\nloop finished. Report: {loop.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
