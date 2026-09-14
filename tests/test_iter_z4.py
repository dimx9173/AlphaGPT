"""Z4 15m vol-window sweep tests: vw {6,12,24} x vt {None,0.006,0.012} FULL sharpe/mdd."""
import json
import os
import pathlib
import subprocess
import sys

OUT = pathlib.Path(os.getenv("ITER_Z4_OUT", "results/iter_Z4_vw.json"))
CELLS = ["vtNone_vw6", "vtNone_vw12", "vtNone_vw24",
         "vt0.006_vw6", "vt0.006_vw12", "vt0.006_vw24",
         "vt0.012_vw6", "vt0.012_vw12", "vt0.012_vw24"]


def _load():
    assert OUT.exists(), "results/iter_Z4_vw.json missing; run research/run_iter_z4_vw.py"
    return json.loads(OUT.read_text())


def test_iter_z4_lock():
    from strategy_manager.config import LEV
    assert LEV == 2.0


def test_iter_z4_schema():
    d = _load()
    assert d["status"] == "done"
    assert d["config"]["vw_grid"] == [6, 12, 24]
    assert d["config"]["vt_grid"] == [None, 0.006, 0.012]
    assert d["config"]["grid"] == "15m"
    assert d["config"]["grid_bars"] == 35040
    assert d["config"]["bpy"] == 35040.0
    assert set(d["config"]["weights"]) == {"ETC", "TRX", "ATOM", "APT", "KAS"}
    assert set(d["rows"]) == set(CELLS)
    for k in CELLS:
        full = d["rows"][k]["FULL"]
        for f in ("sharpe", "mdd", "ann", "cum", "final_x", "n", "turnover"):
            assert f in full, (k, f)
        assert full["n"] == d["config"]["grid_bars"]
    assert d["verdict"] == "PENDING"
    assert d["p03"] == "FAIL"
    assert d["decision"] == "NO_CHANGE"
    assert set(d["compare"]["sharpe_by_cell"]) == set(CELLS)
    assert set(d["compare"]["mdd_by_cell"]) == set(CELLS)


def test_iter_z4_vt_none_flat():
    d = _load()
    sh = d["compare"]["sharpe_by_cell"]
    none_vals = [sh["vtNone_vw%d" % vw] for vw in (6, 12, 24)]
    assert max(none_vals) - min(none_vals) < 1e-9, none_vals
    assert d["compare"]["vt_none_flat_across_vw"] is True
    # vw efficiency recorded per vt
    assert set(d["compare"]["vw_effect_by_vt"]) == {"None", "0.006", "0.012"}
    assert d["compare"]["vw_effect_by_vt"]["None"]["spread"] == 0.0
    fx = d["compare"]["vt_effect_at_vw12"]
    assert set(fx) == {"None", "0.006", "0.012"}


def test_iter_z4_best_consistent():
    d = _load()
    sh = d["compare"]["sharpe_by_cell"]
    best = d["compare"]["best_cell"]
    assert best in sh
    assert d["compare"]["best_sharpe"] == max(sh.values())
    assert d["compare"]["best_sharpe"] == d["rows"][best]["FULL"]["sharpe"]
    dd = d["compare"]["mdd_by_cell"]
    assert dd[best] == d["rows"][best]["FULL"]["mdd"]


def test_iter_z4_script_runs_offline():
    env = dict(os.environ, ITER_Z4_SMOKE="1", ITER_Z4_OUT="/tmp/iter_Z4_test_smoke.json", ITER_Z4_LOG="/tmp/iter_Z4_test_smoke.log")
    r = subprocess.run([sys.executable, "research/run_iter_z4_vw.py"], capture_output=True, text=True, cwd=".", env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    d = json.loads(pathlib.Path("/tmp/iter_Z4_test_smoke.json").read_text())
    assert set(d["rows"]) == set(CELLS)
    assert d["status"] == "done"
    assert d["verdict"] == "PENDING"
