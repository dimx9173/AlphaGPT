"""P2-3: new-coin funnel smoke (offline)."""
import json
import pathlib
import subprocess
import sys


def _run(sym):
    r = subprocess.run([sys.executable, "scripts/new_coin_pipeline.py", sym],
                       capture_output=True, text=True, cwd=".")
    assert r.returncode == 0, r.stderr[-1500:]
    return json.loads(pathlib.Path(f"results/new_coin_{sym}.json").read_text())


def test_funnel_rejects_pepe():
    d = _run("PEPE")
    assert d["verdict"] == "REJECT_L1"
    assert d["L1"]["pass"] is False


def test_funnel_passes_atom():
    d = _run("ATOM")
    assert d["L0"]["pass"] is True
    assert d["L1"]["pass"] is True
    assert d["verdict"] in ("PASS_ALL_PENDING", "REJECT_L2")
    assert d["conclusion"] == "PENDING_P0-3-FAIL"


def test_report_schema():
    d = _run("ATOM")
    assert set(d) >= {"coin", "L0", "L1", "verdict", "conclusion"}
    assert d["L0"]["missing"] < 0.01
