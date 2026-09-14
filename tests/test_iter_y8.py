"""Y8 funding-regime sweep (15m native) tests: Top5 equal-weight FULL at fee2x."""
import json
import os
import pathlib
import subprocess
import sys

OUT = pathlib.Path("results/iter_Y8_fund.json")
FUND_GRID = [0.0001, 0.0005, 0.001, 0.002, 0.005]

def _load():
    assert OUT.exists(), "results/iter_Y8_fund.json missing; run research/run_iter_y8_fund.py"
    return json.loads(OUT.read_text())

def test_iter_y8_schema():
    d = _load()
    assert d["status"] == "done"
    assert d["config"]["fund_grid"] == FUND_GRID
    assert d["config"]["gate_fund"] == 0.001
    assert d["config"]["fail_sharpe"] == 0.05
    assert d["config"]["fee"] == 0.0008
    assert d["config"]["lev"] == 2.0
    assert d["config"]["lev_locked"] == 2.0
    assert d["config"]["grid"] == "15m"
    assert d["config"]["grid_bars"] == 35040
    assert d["config"]["bpy"] == 35040.0
    assert d["config"]["smoke"] is False
    assert set(d["config"]["weights"]) == {"ETC", "TRX", "ATOM", "APT", "KAS"}
    rows = d["curve"]["rows"]
    assert [r["fund"] for r in rows] == FUND_GRID
    for r in rows:
        for f in ("sharpe", "mdd", "final_x", "cum", "ann", "n", "turnover"):
            assert f in r, (r["fund"], f)
        assert r["n"] == d["config"]["grid_bars"]
    assert d["verdict"] in ("PASS", "FAIL")
    assert "gate_0001" in d and "downside" in d and "upside" in d
    assert "zero_cross_fund" in d["downside"] and "zero_cross_check" in d["downside"]

def test_iter_y8_curve_gate():
    d = _load()
    rows = d["curve"]["rows"]
    sh = [r["sharpe"] for r in rows]
    assert all(b >= a - 1e-9 for a, b in zip(sh, sh[1:])), sh
    assert d["curve"]["monotone_up"] is True
    assert d["curve"]["turnover_flat"] is True
    to = [r["turnover"] for r in rows]
    assert max(to) == min(to)
    g = d["gate_0001"]
    assert g["pass_sharpe_gt_0"] is True
    assert g["sharpe"] > 0
    assert g["margin_to_failure"] > 0
    assert g["reasonable"] is True
    assert d["verdict"] == "PASS"

def test_iter_y8_downside_bisect():
    d = _load()
    dd = d["downside"]
    assert dd["threshold"] == 0.05
    assert len(dd["bisection_rows"]) >= 10
    first = dd["bisection_rows"][0]
    assert first["sharpe"] < 0.05, first
    gate_row = next(r for r in d["curve"]["rows"] if abs(r["fund"] - 0.001) < 1e-12)
    assert gate_row["sharpe"] > 0.05
    zc = dd["zero_cross_fund"]
    assert zc < 0.001
    assert abs(dd["zero_cross_check"]["sharpe"] - 0.05) < 1.0

def test_iter_y8_script_runs_smoke():
    env = dict(os.environ, ITER_Y8_SMOKE="1", ITER_Y8_OUT="/tmp/iter_Y8_smoke.json",
               ITER_Y8_LOG="/tmp/iter_Y8_smoke.log")
    r = subprocess.run([sys.executable, "research/run_iter_y8_fund.py"],
                       capture_output=True, text=True, cwd=".", env=env, timeout=900)
    assert r.returncode == 0, r.stderr[-2000:]
    d = json.loads(pathlib.Path("/tmp/iter_Y8_smoke.json").read_text())
    assert d["status"] == "done"
    assert d["config"]["smoke"] is True
    assert [r_["fund"] for r_ in d["curve"]["rows"]] == FUND_GRID
    assert d["verdict"] in ("PASS", "FAIL")
