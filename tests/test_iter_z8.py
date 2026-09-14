"""Z8 slip sensitivity (15m native) tests."""
import json
import os
import pathlib
import subprocess
import sys
import tempfile

OUT = pathlib.Path("results/iter_Z8_slip.json")
SRC = pathlib.Path("research/run_iter_z8_slip.py")

def _load(path=OUT):
    assert path.exists(), "%s missing; run research/run_iter_z8_slip.py" % path
    return json.loads(path.read_text())

def test_iter_z8_locks():
    from strategy_manager.config import LEV, FUND, FEE
    assert LEV == 2.0
    assert abs(FUND - 0.0005) < 1e-12
    assert abs(FEE - 0.0004) < 1e-12

def test_iter_z8_schema():
    d = _load()
    assert d["config"]["slip_bp_grid"] == [0, 2, 5, 10, 20]
    assert d["config"]["fee"] == 0.0004
    assert d["config"]["fund"] == 0.0005
    assert d["config"]["lev"] == 2.0
    assert d["config"]["lev_locked"] == 2.0
    assert d["config"]["grid"] == "15m"
    assert d["config"]["grid_bars"] == 35040
    assert d["config"]["bpy"] == 35040.0
    assert set(d["config"]["weights"]) == {"ETC", "TRX", "ATOM", "APT", "KAS"}
    rows = d["curve"]["rows"]
    assert [r["slip_bp"] for r in rows] == [0, 2, 5, 10, 20]
    for r in rows:
        for f in ("sharpe", "final_x", "mdd", "cum", "ann", "n", "turnover"):
            assert f in r, (r.get("slip_bp"), f)
        assert r["n"] == d["config"]["grid_bars"]
        assert abs(r["slip"] - r["slip_bp"] * 1e-4) < 1e-9
        assert abs(r["eff_fee"] - (r["fee"] + r["slip"])) < 1e-9
    assert d["verdict"] == "PENDING"
    assert d["status"] == "done"
    assert "sharpe_slope_per_bp" in d["curve"]
    assert d["curve"]["sharpe_slope_per_bp"] < 0

def test_iter_z8_slip_monotone_and_turnover_flat():
    d = _load()
    rows = d["curve"]["rows"]
    sh = [r["sharpe"] for r in rows]
    assert all(sh[i + 1] <= sh[i] + 1e-9 for i in range(len(sh) - 1)), sh
    assert d["curve"]["monotone_down"] is True
    tos = {r["turnover"] for r in rows}
    assert len(tos) == 1, tos
    assert d["curve"]["turnover_flat"] is True
    fx = [r["final_x"] for r in rows]
    assert all(fx[i + 1] <= fx[i] + 1e-9 for i in range(len(fx) - 1)), fx
    assert d["curve"]["final_x_monotone_down"] is True
    assert d["curve"]["sharpe_drop_0_to_20bp"] >= -1e-9

def test_iter_z8_native_grid_and_no_live():
    d = _load()
    b = d["config"]["basket"]
    assert b["ETC"] == {"lth": 0.88, "sth": 0.12, "cd": 18, "sl": None, "ts": 24, "vt": None, "vw": 12, "q": 0.3}
    assert b["TRX"] == {"lth": 0.85, "sth": 0.12, "cd": 6, "sl": 0.05, "ts": 24, "vt": None, "vw": 12, "q": 0.3}
    assert b["ATOM"] == {"lth": 0.85, "sth": 0.15, "cd": 6, "sl": 0.05, "ts": 24, "vt": None, "vw": 12, "q": 0.3}
    assert b["APT"] == {"lth": 0.88, "sth": 0.12, "cd": 18, "sl": None, "ts": 24, "vt": None, "vw": 12, "q": 0.3}
    assert b["KAS"] == {"lth": 0.88, "sth": 0.12, "cd": 6, "sl": None, "ts": 24, "vt": None, "vw": 12, "q": 0.3}
    src = SRC.read_text()
    assert "data/data_1y/15m" in src
    assert "35040" in src
    for bad in ("place_order", "submit_order", "api_key", "make_broker", "Y1B_LIVE"):
        assert bad.lower() not in src.lower(), bad

def test_iter_z8_script_runs_offline_smoke():
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "z8.json")
        log = os.path.join(td, "z8.log")
        env = dict(os.environ, ITER_Z8_SMOKE="1", ITER_Z8_OUT=out, ITER_Z8_LOG=log)
        r = subprocess.run([sys.executable, "research/run_iter_z8_slip.py"], capture_output=True, text=True, cwd=".", env=env)
        assert r.returncode == 0, r.stderr[-2000:]
        dd = json.loads(open(out).read())
        assert dd["config"]["smoke"] is True
        assert dd["config"]["coins"] == ["ETC", "TRX"]
        assert dd["config"]["grid_bars"] == 3000
        assert [x["slip_bp"] for x in dd["curve"]["rows"]] == [0, 2, 5, 10, 20]
        assert dd["verdict"] == "PENDING"
        assert dd["status"] == "done"
