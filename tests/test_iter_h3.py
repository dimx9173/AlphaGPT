"""H3 1h stops tests: Top5 1h-native uniform sl x ts sweep. Recompute by script.

Full H3 is 12 uniform cells (sl 4 x ts 3); smoke = 2 uniform via
ITER_H3_SMOKE=1. This test never re-runs the full sweep: schema checks
accept either artifact, and the live subprocess check runs smoke mode
only. Conclusion must stay PENDING (\u5f85\u5b9a): diagnostic only, no adoption,
live untouched.
"""
import json
import os
import pathlib
import subprocess
import sys

OUT = pathlib.Path("results/iter_H3_stops.json")
SEG = {"sharpe", "ann", "mdd", "cum", "final_x", "n", "turnover"}
COINS = {"ETC", "TRX", "ATOM", "APT", "KAS"}
BASE_SPECS = {
    "ETC": {"sl": None, "ts": 24},
    "TRX": {"sl": 0.05, "ts": 24},
    "ATOM": {"sl": 0.05, "ts": 24},
    "APT": {"sl": None, "ts": 24},
    "KAS": {"sl": None, "ts": 24},
}
FULL_SL = [None, 0.03, 0.05, 0.08]
FULL_TS = [12, 24, 36]


def _load():
    assert OUT.exists(), "results/iter_H3_stops.json missing; run research/run_iter_h3_stops.py"
    return json.loads(OUT.read_text())


def _expected_keys(d):
    smoke = d["config"]["smoke"]
    sl = [None, 0.05] if smoke else FULL_SL
    ts = [24] if smoke else FULL_TS
    return {(s, t) for s in sl for t in ts}


def test_iter_h3_schema():
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
        assert cfg["basket"][c]["sl"] == spec["sl"], c
        assert cfg["basket"][c]["ts"] == spec["ts"], c
        assert cfg["basket"][c]["q"] == 0.3, c
    assert cfg["uniform_cells"] == len(d["uniform_rows"])
    assert {(r["sl"], r["ts"]) for r in d["uniform_rows"]} == _expected_keys(d)
    for r in d["uniform_rows"]:
        assert SEG <= set(r["FULL"]), r
        assert r["FULL"]["n"] == cfg["grid_bars"]
        assert isinstance(r["H2_sharpe"], float)
        assert abs(r["d_sharpe_vs_base"] - round(r["FULL"]["sharpe"] - d["base_FULL"]["sharpe"], 3)) < 1e-9
        assert abs(r["d_mdd_vs_base"] - round(r["FULL"]["mdd"] - d["base_FULL"]["mdd"], 4)) < 1e-9
    # best_no_drop_dd_min = min mdd among uniform rows with sharpe >= base
    nodrop = [r for r in d["uniform_rows"] if r["FULL"]["sharpe"] >= d["base_FULL"]["sharpe"]]
    if nodrop:
        exp = min(nodrop, key=lambda r: r["FULL"]["mdd"])
        assert d["best_no_drop_dd_min"] is not None
        assert d["best_no_drop_dd_min"]["sl"] == exp["sl"]
        assert d["best_no_drop_dd_min"]["ts"] == exp["ts"]
        assert d["best_no_drop_dd_min"]["FULL"]["sharpe"] >= d["base_FULL"]["sharpe"]
    else:
        assert d["best_no_drop_dd_min"] is None
    assert d["verdict"] == "PENDING"
    assert "\u5f85\u5b9a" in d["conclusion"]
    assert "no adoption" in d["conclusion"]
    assert "live untouched" in d["conclusion"]


def test_iter_h3_no_broker():
    src = pathlib.Path("research/run_iter_h3_stops.py").read_text()
    for bad in ("place_order", "submit_order", "api_key", "API_KEY",
                "make_broker", "Y1B_LIVE"):
        assert bad.lower() not in src.lower(), bad


def test_iter_h3_engine_mirror():
    src = pathlib.Path("research/run_iter_h3_stops.py").read_text()
    for token in ("quantile", "cooldown", "_apply_stops", "_vol_scale",
                  "roll(1", "8760", "cooldown_bars=cd * SCALE",
                  "time_stop=ts * SCALE", "vol_window=vw * SCALE"):
        assert token in src, token
    assert "short_enabled=True" in src


def test_iter_h3_smoke_runs_offline(tmp_path):
    env = dict(os.environ, ITER_H3_SMOKE="1",
               ITER_H3_OUT=str(tmp_path / "h3.json"), ITER_H3_LOG=str(tmp_path / "h3.log"))
    r = subprocess.run([sys.executable, "research/run_iter_h3_stops.py"],
                       capture_output=True, text=True, cwd=".", env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    d = json.loads((tmp_path / "h3.json").read_text())
    assert d["config"]["smoke"] is True
    assert {(x["sl"], x["ts"]) for x in d["uniform_rows"]} == {(None, 24), (0.05, 24)}
