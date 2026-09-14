"""X7 lev scan tests: Top5 equal-weight FULL at lev {1,2,3}. Recompute by script."""
import json
import pathlib
import subprocess
import sys

OUT = pathlib.Path("results/iter_X7_lev.json")

def _load():
    assert OUT.exists(), "results/iter_X7_lev.json missing; run research/run_iter_x7_lev.py"
    return json.loads(OUT.read_text())

def test_iter_x7_lev_lock():
    from strategy_manager.config import LEV
    assert LEV == 2.0

def test_iter_x7_lev_schema():
    d = _load()
    assert d["config"]["levs"] == [1, 2, 3]
    assert d["config"]["lev_locked"] == 2.0
    assert set(d["config"]["weights"]) == {"ETC", "TRX", "ATOM", "APT", "KAS"}
    assert set(d["rows"]) == {"1", "2", "3"}
    for k, v in d["rows"].items():
        for f in ("sharpe", "mdd", "final_x", "cum", "ann", "n", "turnover"):
            assert f in v["FULL"], (k, f)
        assert v["FULL"]["n"] == d["config"]["grid_bars"]
    assert d["verdict"] == "PENDING_KEEP_LEV2"
    assert d["decision"] == "KEEP_LEV_2.0"
    c = d["compare"]
    assert set(c["sharpe_by_lev"]) == {"1", "2", "3"}
    assert c["sharpe_invariant"] is True
    assert c["sharpe_spread"] < 0.01

def test_iter_x7_lev_dd_scales():
    d = _load()
    dd = d["compare"]["mdd_by_lev"]
    assert dd["1"] < dd["2"] < dd["3"], dd
    assert dd["2"] == d["rows"]["2"]["FULL"]["mdd"]
    assert d["compare"]["dd_3x_uncontrolled"] is True
    assert dd["3"] > 1.0  # 3x dd depth exceeds additive unit -> uncontrolled
    assert d["compare"]["lev2_optimal_sharpe"] is True

def test_iter_x7_lev_script_runs_offline():
    r = subprocess.run([sys.executable, "research/run_iter_x7_lev.py"], capture_output=True, text=True, cwd=".")
    assert r.returncode == 0, r.stderr[-2000:]
    d = _load()
    assert set(d["rows"]) == {"1", "2", "3"}
