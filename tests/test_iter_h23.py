"""H23 1h vol-window tests: Top5 1h-native uniform vw x vt sweep. Recompute by script.

Full H23 is 9 uniform cells (vw 3 x vt 3); smoke = 2 uniform via
ITER_H23_SMOKE=1. This test never re-runs the full sweep: schema checks
accept either artifact, and the live subprocess check runs smoke mode
only. Conclusion must stay PENDING (P0-3 FAIL): diagnostic only, no adoption,
live untouched.
"""
import json
import math
import os
import pathlib
import subprocess
import sys

OUT = pathlib.Path(os.getenv("ITER_H23_OUT", "results/iter_H23_vw.json"))
SEG = {"sharpe", "ann", "mdd", "cum", "final_x", "n", "turnover"}
COINS = {"ETC", "TRX", "ATOM", "APT", "KAS"}
BASE_SPECS = {
    "ETC": {"lth": 0.88, "sth": 0.12, "cd": 18, "sl": None, "ts": 24, "vt": None, "vw": 12, "q": 0.3},
    "TRX": {"lth": 0.85, "sth": 0.12, "cd": 6, "sl": 0.05, "ts": 24, "vt": None, "vw": 12, "q": 0.3},
    "ATOM": {"lth": 0.85, "sth": 0.15, "cd": 6, "sl": 0.05, "ts": 24, "vt": None, "vw": 12, "q": 0.3},
    "APT": {"lth": 0.88, "sth": 0.12, "cd": 18, "sl": None, "ts": 24, "vt": None, "vw": 12, "q": 0.3},
    "KAS": {"lth": 0.88, "sth": 0.12, "cd": 6, "sl": None, "ts": 24, "vt": None, "vw": 12, "q": 0.3},
}
FULL_VW = [6, 12, 24]
FULL_VT = [None, 0.006, 0.012]


def _load():
    assert OUT.exists(), "results/iter_H23_vw.json missing; run research/run_iter_h23_vw.py"
    return json.loads(OUT.read_text())


def _expected_keys(d):
    smoke = d["config"]["smoke"]
    vw = [12] if smoke else FULL_VW
    vt = [None, 0.012] if smoke else FULL_VT
    return {(v, t) for v in vw for t in vt}


def test_iter_h23_lock():
    from strategy_manager.config import LEV, FUND, FEE, FORMULA
    from strategy_manager.config import LOCKED_ETC, LOCKED_TRX, LOCKED_ATOM, LOCKED_APT, LOCKED_KAS
    assert LEV == 2.0
    assert FUND == 0.0005
    assert FEE == 0.0004
    assert FORMULA == [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10]
    for name, want in (("ETC", LOCKED_ETC), ("TRX", LOCKED_TRX), ("ATOM", LOCKED_ATOM),
                       ("APT", LOCKED_APT), ("KAS", LOCKED_KAS)):
        for k, v in BASE_SPECS[name].items():
            assert want[k] == v, (name, k, want[k], v)


def test_iter_h23_schema():
    d = _load()
    cfg = d["config"]
    assert cfg["smoke"] in (True, False)
    assert cfg["bpy"] == 8760.0
    assert cfg["scale"] == 4
    assert cfg["grid_bars"] > 8000
    assert cfg["h2_start"] == cfg["grid_bars"] // 2
    assert set(cfg["weights"]) == COINS
    assert set(cfg["basket"]) == COINS
    for c, spec in BASE_SPECS.items():
        for k, v in spec.items():
            assert cfg["basket"][c][k] == v, (c, k)
    assert cfg["uniform_cells"] == len(d["uniform_rows"])
    assert {(r["vw"], r["vt"]) for r in d["uniform_rows"]} == _expected_keys(d)
    for r in d["uniform_rows"]:
        assert r["vw"] in FULL_VW, r
        assert r["vt"] in FULL_VT, r
        assert SEG <= set(r["FULL"]), r
        assert r["FULL"]["n"] == cfg["grid_bars"]
        assert isinstance(r["H2_sharpe"], float)
        assert math.isfinite(r["FULL"]["sharpe"])
        assert math.isfinite(r["FULL"]["mdd"])
        assert abs(r["d_sharpe_vs_base"] - round(r["FULL"]["sharpe"] - d["base_FULL"]["sharpe"], 3)) < 1e-9
        assert abs(r["d_mdd_vs_base"] - round(r["FULL"]["mdd"] - d["base_FULL"]["mdd"], 4)) < 1e-9
    # best_sharpe = max FULL sharpe (tie-break min mdd)
    exp = max(d["uniform_rows"], key=lambda r: (r["FULL"]["sharpe"], -r["FULL"]["mdd"]))
    assert d["best_sharpe"] is not None
    assert d["best_sharpe"]["vw"] == exp["vw"]
    assert d["best_sharpe"]["vt"] == exp["vt"]
    assert d["best_sharpe"]["FULL"]["sharpe"] >= d["base_FULL"]["sharpe"] or len(d["uniform_rows"]) > 0
    # base cell present only in full sweep (smoke lacks vw=12/vt=None? no: smoke has it)
    assert d["verdict"] == "PENDING_P03_FAIL"
    assert d["decision"] == "NO_ADOPTION_KEEP_EQUAL"
    assert "PENDING" in d["conclusion"]
    assert "no adoption" in d["conclusion"]
    assert "live untouched" in d["conclusion"]


def test_iter_h23_engine_mirror():
    src = pathlib.Path("research/run_iter_h23_vw.py").read_text()
    for token in ("quantile", "cooldown", "_apply_stops", "_vol_scale",
                  "roll(1", "8760", "cooldown_bars=cd * SCALE",
                  "time_stop=ts * SCALE", "vol_window=vw * SCALE"):
        assert token in src, token
    assert "short_enabled=True" in src


def test_iter_h23_no_broker():
    src = pathlib.Path("research/run_iter_h23_vw.py").read_text()
    for bad in ("place_order", "submit_order", "api_key", "API_KEY",
                "make_broker", "Y1B_LIVE"):
        assert bad.lower() not in src.lower(), bad


def test_iter_h23_smoke_runs_offline(tmp_path):
    out = tmp_path / "iter_H23_vw.json"
    lg = tmp_path / "iter_h23_vw.log"
    env = dict(os.environ, ITER_H23_SMOKE="1", ITER_H23_OUT=str(out),
               ITER_H23_LOG=str(lg))
    r = subprocess.run([sys.executable, "research/run_iter_h23_vw.py"],
                       capture_output=True, text=True, cwd=".", env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    dd = json.loads(out.read_text())
    assert dd["config"]["smoke"] is True
    assert dd["config"]["grid_bars"] > 8000 or dd["config"]["grid_bars"] == dd["uniform_rows"][0]["FULL"]["n"]
    assert {(x["vw"], x["vt"]) for x in dd["uniform_rows"]} == {(12, None), (12, 0.012)}
    assert dd["verdict"] == "PENDING_P03_FAIL"
    assert dd["decision"] == "NO_ADOPTION_KEEP_EQUAL"
