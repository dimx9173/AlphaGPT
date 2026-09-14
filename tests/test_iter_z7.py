"""Z7 fee sensitivity (15m native) tests."""
import json
import os
import pathlib
import subprocess
import sys
import tempfile

OUT = pathlib.Path("results/iter_Z7_fee.json")

def _load(path=OUT):
    assert path.exists(), "%s missing; run research/run_iter_z7_fee.py" % path
    return json.loads(path.read_text())

def test_iter_z7_locks():
    from strategy_manager.config import LEV, FUND, FEE
    assert LEV == 2.0
    assert abs(FUND - 0.0005) < 1e-12
    assert abs(FEE - 0.0004) < 1e-12

def test_iter_z7_schema():
    d = _load()
    assert d["config"]["fee_grid"] == [0.0002, 0.0004, 0.0008, 0.0016]
    assert d["config"]["fund"] == 0.0005
    assert d["config"]["lev"] == 2.0
    assert d["config"]["lev_locked"] == 2.0
    assert d["config"]["grid"] == "15m"
    assert d["config"]["grid_bars"] == 35040
    assert d["config"]["bpy"] == 35040.0
    assert d["config"]["fail_sharpe"] == 0.05
    assert set(d["config"]["weights"]) == {"ETC", "TRX", "ATOM", "APT", "KAS"}
    rows = d["curve"]["rows"]
    assert [r["fee"] for r in rows] == [0.0002, 0.0004, 0.0008, 0.0016]
    for r in rows:
        for f in ("sharpe", "final_x", "mdd", "cum", "ann", "n", "turnover"):
            assert f in r, (r.get("fee"), f)
        assert r["n"] == d["config"]["grid_bars"]
    assert d["verdict"] == "PENDING"
    assert d["status"] == "done"
    be = d["breakeven"]
    assert be["threshold"] == 0.05
    assert be["breakeven_fee"] > 0
    assert "breakeven_check" in be
    assert abs(be["breakeven_check"]["sharpe"] - 0.05) < 0.5

def test_iter_z7_fee_monotone_and_turnover_flat():
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

def test_iter_z7_breakeven_bracket():
    d = _load()
    be = d["breakeven"]
    lo, hi = be["bracket_lo"], be["bracket_hi"]
    assert lo - 1e-6 <= be["breakeven_fee"] <= hi + 1e-6, be
    assert len(be["bisection_rows"]) >= 20
    fees = [r["fee"] for r in be["bisection_rows"]]
    assert min(fees) >= 0
    assert be["margin_from_grid_hi"] == round(be["breakeven_fee"] - 0.0016, 6)

def test_iter_z7_script_runs_offline_smoke():
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "z7.json")
        log = os.path.join(td, "z7.log")
        env = dict(os.environ, ITER_Z7_SMOKE="1", ITER_Z7_OUT=out, ITER_Z7_LOG=log)
        r = subprocess.run([sys.executable, "research/run_iter_z7_fee.py"], capture_output=True, text=True, cwd=".", env=env)
        assert r.returncode == 0, r.stderr[-2000:]
        d = json.loads(open(out).read())
        assert d["config"]["smoke"] is True
        assert d["config"]["coins"] == ["ETC", "TRX"]
        assert d["config"]["grid_bars"] == 3000
        assert len(d["curve"]["rows"]) == 4
        assert d["verdict"] == "PENDING"
        assert d["status"] == "done"
        assert len(d["breakeven"]["bisection_rows"]) >= 20
