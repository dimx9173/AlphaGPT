"""Y7 lev scan (15m native) tests: Top5 equal-weight FULL at lev {1,2,3}. Recompute by script."""
import json
import pathlib
import subprocess
import sys

OUT = pathlib.Path("results/iter_Y7_lev.json")

def _load():
    assert OUT.exists(), "results/iter_Y7_lev.json missing; run research/run_iter_y7_lev.py"
    return json.loads(OUT.read_text())

def test_iter_y7_lev_lock():
    from strategy_manager.config import LEV
    assert LEV == 2.0

def test_iter_y7_lev_schema():
    d = _load()
    assert d["config"]["levs"] == [1, 2, 3]
    assert d["config"]["lev_locked"] == 2.0
    assert d["config"]["grid"] == "15m"
    assert d["config"]["grid_bars"] == 35040
    assert d["config"]["bpy"] == 35040.0
    assert set(d["config"]["weights"]) == {"ETC", "TRX", "ATOM", "APT", "KAS"}
    assert set(d["rows"]) == {"1", "2", "3"}
    for k, v in d["rows"].items():
        for f in ("sharpe", "mdd", "final_x", "cum", "ann", "n", "turnover"):
            assert f in v["FULL"], (k, f)
        assert v["FULL"]["n"] == d["config"]["grid_bars"]
    assert d["verdict"] == "PENDING"
    assert d["decision"] == "KEEP_LEV_2.0"
    c = d["compare"]
    assert set(c["sharpe_by_lev"]) == {"1", "2", "3"}
    assert set(c["mdd_by_lev"]) == {"1", "2", "3"}
    assert set(c["final_x_by_lev"]) == {"1", "2", "3"}

def test_iter_y7_lev_sharpe_invariant():
    d = _load()
    c = d["compare"]
    assert c["sharpe_invariant"] is True
    assert c["sharpe_spread"] < 0.01
    sh = [d["rows"][k]["FULL"]["sharpe"] for k in ("1", "2", "3")]
    assert max(sh) - min(sh) < 0.01
    to = [d["rows"][k]["FULL"]["turnover"] for k in ("1", "2", "3")]
    assert max(to) == min(to)  # turnover invariant: positions identical, lev scales PnL only

def test_iter_y7_lev_dd_ordering():
    d = _load()
    dd = d["compare"]["mdd_by_lev"]
    assert dd["1"] < dd["2"] < dd["3"], dd
    assert dd["2"] == d["rows"]["2"]["FULL"]["mdd"]
    fx = d["compare"]["final_x_by_lev"]
    assert fx["1"] < fx["2"] < fx["3"], fx
    assert d["compare"]["dd_3x_uncontrolled"] is True

def test_iter_y7_lev_script_runs_offline():
    r = subprocess.run([sys.executable, "research/run_iter_y7_lev.py"], capture_output=True, text=True, cwd=".")
    assert r.returncode == 0, r.stderr[-2000:]
    d = _load()
    assert set(d["rows"]) == {"1", "2", "3"}
