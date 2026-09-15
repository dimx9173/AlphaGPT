"""H21 1h take-profit tests: Top5 1h-native tp x trail sweep. Recompute by script.

Full H21 is 12 uniform cells (tp 4 x trail 3);
smoke = 4 cells (tp {None,0.10} x trail {None,0.03}) via ITER_H21_SMOKE=1.
This test never re-runs the full sweep: schema checks accept either
artifact, and the live subprocess check runs smoke mode only (restoring
the prior artifact afterwards so the committed 12-cell file is kept).
Conclusion must stay PENDING: diagnostic only, no adoption, live untouched.
"""
import json
import os
import pathlib
import subprocess
import sys

OUT = pathlib.Path("results/iter_H21_tp.json")
SEG = {"sharpe", "ann", "mdd", "cum", "final_x", "n", "turnover"}
COINS = {"ETC", "TRX", "ATOM", "APT", "KAS"}
BASE_SPECS = {
    "ETC": {"tp": None, "sl": None, "ts": 24},
    "TRX": {"tp": None, "sl": 0.05, "ts": 24},
    "ATOM": {"tp": None, "sl": 0.05, "ts": 24},
    "APT": {"tp": None, "sl": None, "ts": 24},
    "KAS": {"tp": None, "sl": None, "ts": 24},
}
FULL_TP = [None, 0.05, 0.10, 0.20]
FULL_TRAIL = [None, 0.03, 0.05]


def _load():
    assert OUT.exists(), "results/iter_H21_tp.json missing; run research/run_iter_h21_tp.py"
    return json.loads(OUT.read_text())


def _expected_keys(d):
    if d["config"]["smoke"]:
        return {(None, None), (None, 0.03), (0.10, None), (0.10, 0.03)}
    return {(t, r) for t in FULL_TP for r in FULL_TRAIL}


def test_iter_h21_schema():
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
    assert {(r["tp"], r["trail"]) for r in d["uniform_rows"]} == _expected_keys(d)
    for r in d["uniform_rows"]:
        assert SEG <= set(r["FULL"]), r
        assert r["FULL"]["n"] == cfg["grid_bars"]
        assert isinstance(r["H2_sharpe"], float)
        assert abs(r["d_sharpe_vs_base"] - round(r["FULL"]["sharpe"] - d["base_FULL"]["sharpe"], 3)) < 1e-9
        assert abs(r["d_mdd_vs_base"] - round(r["FULL"]["mdd"] - d["base_FULL"]["mdd"], 4)) < 1e-9
    # base cell (None, None) reproduces locked config
    base_row = next(r for r in d["uniform_rows"] if r["tp"] is None and r["trail"] is None)
    assert base_row["FULL"] == d["base_FULL"]
    # best_no_drop_dd_min = min mdd among uniform rows with sharpe >= base
    nodrop = [r for r in d["uniform_rows"] if r["FULL"]["sharpe"] >= d["base_FULL"]["sharpe"]]
    if nodrop:
        exp = min(nodrop, key=lambda r: r["FULL"]["mdd"])
        assert d["best_no_drop_dd_min"] is not None
        assert d["best_no_drop_dd_min"]["tp"] == exp["tp"]
        assert d["best_no_drop_dd_min"]["trail"] == exp["trail"]
        assert d["best_no_drop_dd_min"]["FULL"]["sharpe"] >= d["base_FULL"]["sharpe"]
    else:
        assert d["best_no_drop_dd_min"] is None
    assert d["verdict"] == "PENDING"
    assert "no adoption" in d["conclusion"]
    assert "live untouched" in d["conclusion"]


def test_iter_h21_no_broker():
    src = pathlib.Path("research/run_iter_h21_tp.py").read_text()
    for bad in ("place_order", "submit_order", "api_key", "API_KEY",
                "make_broker", "Y1B_LIVE"):
        assert bad.lower() not in src.lower(), bad


def test_iter_h21_engine_mirror():
    src = pathlib.Path("research/run_iter_h21_tp.py").read_text()
    for token in ("quantile", "cooldown", "_apply_stops", "_vol_scale",
                  "roll(1", "8760", "cooldown_bars=cd * SCALE",
                  "time_stop=ts * SCALE", "vol_window=vw * SCALE",
                  "take_profit=tp", "apply_trailing"):
        assert token in src, token
    assert "short_enabled=True" in src


def test_iter_h21_smoke_runs_offline():
    backup = OUT.read_text() if OUT.exists() else None
    try:
        env = dict(os.environ, ITER_H21_SMOKE="1")
        r = subprocess.run([sys.executable, "research/run_iter_h21_tp.py"],
                           capture_output=True, text=True, cwd=".", env=env)
        assert r.returncode == 0, r.stderr[-2000:]
        d = _load()
        assert d["config"]["smoke"] is True
        assert {(x["tp"], x["trail"]) for x in d["uniform_rows"]} == {
            (None, None), (None, 0.03), (0.10, None), (0.10, 0.03)}
    finally:
        if backup is not None:
            OUT.write_text(backup)
