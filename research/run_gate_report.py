#!/usr/bin/env python3
"""run_gate_report.py — one-command Y1b gate summary (no orders, no keys).

Runs: pytest (count only) + run_y1b_verify 4-in-1 + E10 check + drift pins.
Writes results/gate_report.json (gitignored) and prints a compact summary.
Usage: python3 research/run_gate_report.py
"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)


def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr)[-3000:]


def main():
    rep = {"ok": True, "checks": {}}
    rc, out = sh("python3 -m pytest -q --tb=no -p no:warnings --junitxml=/tmp/gate_rep.xml 2>&1 | tail -n 3")
    try:
        import xml.etree.ElementTree as ET
        t = ET.parse("/tmp/gate_rep.xml").getroot().find("testsuite").attrib
        rep["checks"]["pytest"] = {"tests": int(t["tests"]), "failures": int(t["failures"]),
                                   "errors": int(t["errors"])}
        if int(t["failures"]) or int(t["errors"]):
            rep["ok"] = False
    except Exception as e:
        rep["checks"]["pytest"] = {"error": str(e)}
        rep["ok"] = False
    rc, out = sh("python3 research/run_y1b_verify.py 2>&1 | tail -n 6")
    rep["checks"]["y1b_verify"] = {"pass": "Y1B_VERIFY PASS" in out, "tail": out.strip().splitlines()[-6:]}
    if "Y1B_VERIFY PASS" not in out:
        rep["ok"] = False
    rc, out = sh("python3 research/run_e10.py --check 2>&1 | tail -n 2")
    rep["checks"]["e10"] = {"pass": "PASS" in out}
    if "PASS" not in out:
        rep["ok"] = False
    try:
        g = json.load(open("tests/fixtures/e10_gates_golden.json"))
        e10 = json.load(open("results/backtest_E10.json"))["locked_params"]
        drift = [k for k in [("FORMULA", g["FORMULA"], e10["FORMULA"]),
                             ("gate", g["gate_adopted_id"], e10["gate"]["adopted_id"])]
                 if k[1] != k[2]]
        rep["checks"]["drift"] = {"clean": not drift, "mismatch": [k[0] for k in drift]}
        if drift:
            rep["ok"] = False
    except Exception as e:
        rep["checks"]["drift"] = {"error": str(e)}
        rep["ok"] = False
    try:
        pj = json.load(open("results/paper_trades2.json"))["stats"]
        rep["checks"]["paper"] = {"trades": pj["trades"], "final_x": round(pj["final_x"], 4),
                                  "sharpe": round(pj["sharpe"], 3)}
    except Exception as e:
        rep["checks"]["paper"] = {"error": str(e)}
        rep["ok"] = False
    os.makedirs("results", exist_ok=True)
    json.dump(rep, open("results/gate_report.json", "w"), indent=1)
    print(json.dumps(rep, indent=1)[:2000])
    print("GATE_REPORT", "PASS" if rep["ok"] else "FAIL")
    return rep


if __name__ == "__main__":
    rep = main()
    sys.exit(0 if (rep or {}).get("ok") else 1)
