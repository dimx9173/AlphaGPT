#!/usr/bin/env python3
"""Shadow ledger: append-only daily stability rows (no orders, no keys)."""
import csv
import datetime
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
LEDGER = "results/shadow_ledger.jsonl"
BASE_TRADES, BASE_X, BASE_SH = 478, 2.98, 0.90


def paper_stats():
    r = subprocess.run([sys.executable, "research/run_paper2.py"],
                       capture_output=True, text=True, timeout=600)
    out = r.stdout + r.stderr
    import re
    t = int(re.search(r"trades=(\d+)", out).group(1))
    x = float(re.search(r"final_x=([0-9.]+)", out).group(1))
    s = float(re.search(r"sharpe=([0-9.\-]+)", out).group(1))
    m = re.search(r"mdd=([0-9.]+)", out)
    return {"trades": t, "final_x": round(x, 4), "sharpe": round(s, 3),
            "mdd": round(float(m.group(1)), 4) if m else None}


def data_rows():
    n = {}
    for c in ("ETC", "TRX"):
        with open(f"data/data_15m_3y/{c}.csv") as f:
            n[c] = sum(1 for _ in f) - 1
    return n


def main():
    st = paper_stats()
    row = {"date": datetime.date.today().isoformat(), **st, "rows": data_rows(),
           "drift": {"trades": abs(st["trades"] - BASE_TRADES) <= 5,
                     "final_x": abs(st["final_x"] - BASE_X) <= 0.15,
                     "sharpe": abs(st["sharpe"] - BASE_SH) <= 0.10}}
    row["pass"] = all(row["drift"].values())
    with open(LEDGER, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row, indent=1))
    print("SHADOW_LEDGER", "PASS" if row["pass"] else "DRIFT")


if __name__ == "__main__":
    main()
