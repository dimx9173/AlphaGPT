"""Shadow ledger tests: drift logic + ledger format, no paper rerun."""
import json
import os


def _drift(trades, x, sharpe):
    return {"trades": abs(trades - 478) <= 5,
            "final_x": abs(x - 2.98) <= 0.15,
            "sharpe": abs(sharpe - 0.90) <= 0.10}


def test_drift_logic_baseline_pass():
    d = _drift(478, 3.0782, 0.911)
    assert all(d.values())


def test_drift_logic_flags_blowup():
    d = _drift(478, 1.5, 0.911)
    assert d["trades"] and not d["final_x"] and d["sharpe"]


def test_ledger_rows_schema():
    p = "results/shadow_ledger.jsonl"
    assert os.path.exists(p)
    row = json.loads(open(p).read().strip().splitlines()[-1])
    for k in ("date", "trades", "final_x", "sharpe", "rows", "drift", "pass"):
        assert k in row, k
    assert row["pass"] == all(row["drift"].values())
