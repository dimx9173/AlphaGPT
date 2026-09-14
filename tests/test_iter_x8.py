"""X8 funding-regime test: schema + offline rerun (mirrors test_qsweep.py)."""
import json
import pathlib
import subprocess
import sys

GRID = [0.0001, 0.0005, 0.001, 0.002, 0.005]
SEG = {"sharpe", "ann", "mdd", "cum", "n", "turnover"}


def _load():
    p = pathlib.Path("results/iter_X8_fund.json")
    assert p.exists(), "results/iter_X8_fund.json missing; run research/run_iter_x8.py"
    return json.loads(p.read_text())


def test_iter_x8_schema():
    d = _load()
    assert d["config"]["fund_grid"] == GRID
    assert d["config"]["gate_fund"] == 0.001
    assert set(d["config"]["coins"]) == {"ETC", "TRX", "ATOM", "APT", "KAS"}
    rows = d["curve"]["rows"]
    assert [r["fund"] for r in rows] == GRID
    for r in rows:
        assert SEG <= set(r), r["fund"]
        assert r["n"] > 2000
    assert d["curve"]["monotone_up"] is True
    assert d["curve"]["turnover_flat"] is True
    assert d["upside"]["failure_in_grid"] is False
    assert d["upside"]["max_fund"] == 0.005
    assert d["downside"]["zero_cross_fund"] < 0.0001
    assert abs(d["downside"]["zero_cross_check"]["sharpe"]) < 0.05
    g = d["gate_0001"]
    assert g["pass_sharpe_gt_0"] is True
    assert g["sharpe"] > 0
    assert g["margin_to_failure"] > 0
    assert g["reasonable"] is True
    assert d["verdict"] == "PASS"


def test_iter_x8_matches_micro_overlap():
    d = _load()
    m = json.loads(pathlib.Path("results/micro.json").read_text())
    mrows = {r["fund"]: r["sharpe"] for r in m["funding"]["rows"]}
    for r in d["curve"]["rows"]:
        if r["fund"] in mrows:
            assert abs(r["sharpe"] - mrows[r["fund"]]) < 0.01, (r["fund"], r["sharpe"], mrows[r["fund"]])


def test_iter_x8_no_broker_in_engine():
    src = pathlib.Path("research/run_iter_x8.py").read_text()
    for bad in ("place_order", "submit_order", "api_key", "make_broker", "Y1B_LIVE"):
        assert bad.lower() not in src.lower(), bad


def test_iter_x8_script_runs_offline():
    r = subprocess.run([sys.executable, "research/run_iter_x8.py"],
                       capture_output=True, text=True, cwd=".")
    assert r.returncode == 0, r.stderr[-2000:]
    d = _load()
    assert [x["fund"] for x in d["curve"]["rows"]] == GRID
    assert d["verdict"] in ("PASS", "FAIL")
