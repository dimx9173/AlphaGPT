"""Y3 thresh robustness tests (15m native, Top5). Schema-level only.

Full Y3 is base + 20 runs (smoke=base + 2 via ITER_Y3_SMOKE=1 to a temp
OUT so the committed FULL artifact is not clobbered). This test never
re-runs the full sweep: schema checks accept either artifact, and the
live subprocess check runs smoke mode only. Conclusion must stay PENDING
(\u5f85\u5b9a). Robustness only: no optimum pursuit, no parameter adoption.
"""
import json
import os
import pathlib
import subprocess
import sys
import tempfile

SEG = {"sharpe", "ann", "mdd", "cum", "final_x", "n", "turnover"}
ROW = {"coin", "param", "delta", "new_value", "FULL", "d_sharpe"}
COINS = {"ETC", "TRX", "ATOM", "APT", "KAS"}


def _load():
    p = pathlib.Path("results/iter_Y3_thresh.json")
    assert p.exists(), "results/iter_Y3_thresh.json missing; run research/run_iter_y3_thresh.py"
    return json.loads(p.read_text())


def test_iter_y3_schema():
    d = _load()
    cfg = d["config"]
    assert set(cfg["weights"]) == COINS
    assert cfg["delta"] == 0.02
    assert cfg["flat_tol"] == 0.15
    assert cfg["bpy"] == 35040.0
    assert cfg["scale"] == 16
    assert cfg["grid"] == "15m"
    assert cfg["grid_bars"] == 35040
    assert cfg["smoke"] in (True, False)
    rows = d["rows"]
    if not cfg["smoke"]:
        assert len(rows) == 20, len(rows)
        assert {(r["coin"], r["param"], r["delta"]) for r in rows} == {
            (c, p, s) for c in COINS for p in ("lth", "sth") for s in (0.02, -0.02)}
    else:
        assert len(rows) == 2, len(rows)
    assert SEG <= set(d["base_FULL"])
    assert d["base_FULL"]["n"] == cfg["grid_bars"]
    for r in rows:
        assert ROW <= set(r), r
        assert r["coin"] in COINS
        assert r["param"] in ("lth", "sth")
        assert r["delta"] in (0.02, -0.02)
        assert SEG <= set(r["FULL"])
        assert r["FULL"]["n"] == cfg["grid_bars"]
        assert abs(r["FULL"]["sharpe"] - d["base_FULL"]["sharpe"] - r["d_sharpe"]) < 1e-3
    ds = [abs(r["d_sharpe"]) for r in rows]
    assert ds == sorted(ds, reverse=True), "rows must be |d_sharpe|-desc"
    fc = d["flat_check"]
    assert fc["flat_verdict"] in ("flat(\u7a69\u5065)", "sensitive(\u654f\u611f)")
    assert fc["flat"] in (True, False)
    assert fc["flat"] == (fc["max_abs_d_sharpe"] <= fc["tol"])
    assert fc["n_runs"] == len(rows)
    assert d["verdict"] == "PENDING"
    assert d["decision"] == "KEEP_LOCKED"
    assert "\u5f85\u5b9a" in d["conclusion"]
    assert "no adoption" in d["conclusion"]


def test_iter_y3_no_broker():
    src = pathlib.Path("research/run_iter_y3_thresh.py").read_text()
    for bad in ("place_order", "submit_order", "api_key", "API_KEY",
                "make_broker", "Y1B_LIVE"):
        assert bad.lower() not in src.lower(), bad


def test_iter_y3_native_grid():
    d = _load()
    cfg = d["config"]
    assert cfg["basket"]["ETC"] == {"lth": 0.88, "sth": 0.12, "cd": 18, "sl": None, "ts": 24, "q": 0.3}
    assert cfg["basket"]["TRX"] == {"lth": 0.85, "sth": 0.12, "cd": 6, "sl": 0.05, "ts": 24, "q": 0.3}
    assert cfg["basket"]["ATOM"] == {"lth": 0.85, "sth": 0.15, "cd": 6, "sl": 0.05, "ts": 24, "q": 0.3}
    assert cfg["basket"]["APT"] == {"lth": 0.88, "sth": 0.12, "cd": 18, "sl": None, "ts": 24, "q": 0.3}
    assert cfg["basket"]["KAS"] == {"lth": 0.88, "sth": 0.12, "cd": 6, "sl": None, "ts": 24, "q": 0.3}
    src = pathlib.Path("research/run_iter_y3_thresh.py").read_text()
    assert "data/data_1y/15m" in src
    assert "35040" in src


def test_iter_y3_smoke_runs_offline():
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "iter_Y3_thresh.json")
        log = os.path.join(td, "iter_Y3_thresh.log")
        env = dict(os.environ, ITER_Y3_SMOKE="1", ITER_Y3_OUT=out, ITER_Y3_LOG=log)
        r = subprocess.run([sys.executable, "research/run_iter_y3_thresh.py"],
                           capture_output=True, text=True, cwd=".", env=env)
        assert r.returncode == 0, r.stderr[-2000:]
        d = json.loads(pathlib.Path(out).read_text())
        assert d["config"]["smoke"] is True
        assert len(d["rows"]) == 2
        assert d["verdict"] == "PENDING"
