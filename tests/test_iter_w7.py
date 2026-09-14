"""W7 entry-timing delay (15m native) tests."""
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tempfile

OUT = pathlib.Path("results/iter_W7_delay.json")
SRC = pathlib.Path("research/run_iter_w7_delay.py")

def _load(path=OUT):
    assert path.exists(), "%s missing; run research/run_iter_w7_delay.py" % path
    return json.loads(path.read_text())

def _mod():
    spec = importlib.util.spec_from_file_location("run_iter_w7_delay", str(SRC))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def test_iter_w7_locks():
    from strategy_manager.config import LEV, FUND, FEE
    assert LEV == 2.0
    assert abs(FUND - 0.0005) < 1e-12
    assert abs(FEE - 0.0004) < 1e-12

def test_iter_w7_schema():
    d = _load()
    assert d["config"]["delays"] == [0, 1, 2, 3]
    assert d["config"]["fee"] == 0.0004
    assert d["config"]["fund"] == 0.0005
    assert d["config"]["lev"] == 2.0
    assert d["config"]["lev_locked"] == 2.0
    assert d["config"]["grid"] == "15m"
    assert d["config"]["grid_bars"] == 35040
    assert d["config"]["bpy"] == 35040.0
    assert set(d["config"]["weights"]) == {"ETC", "TRX", "ATOM", "APT", "KAS"}
    rows = d["curve"]["rows"]
    assert [r["delay"] for r in rows] == [0, 1, 2, 3]
    for r in rows:
        for f in ("sharpe", "final_x", "mdd", "cum", "ann", "n", "turnover", "per_coin_sharpe"):
            assert f in r, (r.get("delay"), f)
        assert r["n"] == d["config"]["grid_bars"]
        assert set(r["per_coin_sharpe"]) == {"ETC", "TRX", "ATOM", "APT", "KAS"}
    assert d["verdict"] == "PENDING"
    assert d["status"] == "done"
    assert "sharpe_slope_per_bar" in d["curve"]
    assert "sharpe_drop_0_to_3" in d["curve"]

def test_iter_w7_delay_cost_bounded_and_turnover_stable():
    d = _load()
    rows = d["curve"]["rows"]
    sh = [r["sharpe"] for r in rows]
    assert d["curve"]["sharpe_drop_0_to_3"] >= -0.5, sh
    assert sh[-1] - sh[0] <= 0.5, sh
    tos = [r["turnover"] for r in rows]
    assert all(t > 0 for t in tos), tos
    assert max(tos) - min(tos) <= 0.01, tos
    assert len(d["curve"]["turnover_by_delay"]) == 4

def test_iter_w7_apply_delay_semantics():
    m = _mod()
    import torch
    lp_pre = torch.tensor([[1.0, 1.0, 0.0, 1.0, 0.0, 0.0]])
    sp_pre = torch.zeros((1, 6))
    rt = torch.tensor([[0.01, 0.02, 0.03, 0.04, 0.05, 0.06]])
    p0, t0, g0 = m.apply_delay(lp_pre, sp_pre, rt, 0)
    assert p0[0] == 0.0
    assert p0[1] == 1.0 and p0[2] == 1.0 and p0[3] == 0.0
    p2, t2, g2 = m.apply_delay(lp_pre, sp_pre, rt, 2)
    assert p2[0] == 0.0 and p2[1] == 0.0 and p2[2] == 0.0
    assert p2[3] == 1.0 and p2[4] == 1.0 and p2[5] == 0.0
    assert abs(g2[3] - p2[3] * 0.04 * 2.0) < 1e-6
    assert abs(g0[1] - p0[1] * 0.02 * 2.0) < 1e-6

def test_iter_w7_native_grid_and_no_live():
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

def test_iter_w7_script_runs_offline_smoke():
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "w7.json")
        log = os.path.join(td, "w7.log")
        env = dict(os.environ, ITER_W7_SMOKE="1", ITER_W7_OUT=out, ITER_W7_LOG=log)
        r = subprocess.run([sys.executable, "research/run_iter_w7_delay.py"], capture_output=True, text=True, cwd=".", env=env)
        assert r.returncode == 0, r.stderr[-2000:]
        dd = json.loads(open(out).read())
        assert dd["config"]["smoke"] is True
        assert dd["config"]["coins"] == ["ETC", "TRX"]
        assert dd["config"]["grid_bars"] == 3000
        assert [x["delay"] for x in dd["curve"]["rows"]] == [0, 1, 2, 3]
        assert dd["verdict"] == "PENDING"
        assert dd["status"] == "done"
