"""H49 1h ATR-stop v2 tests (Top5): uniform atr_mult + per-coin best.

Schema accepts full or smoke artifact. Recompute-by-script only via
smoke subprocess; never re-runs the full sweep here. Verdict PENDING
(P0-3 FAIL), no adoption, live untouched.
"""
import json
import os
import pathlib
import subprocess
import sys

OUT = pathlib.Path("results/iter_H49_atr.json")
SRC = pathlib.Path("research/run_iter_h49_atr.py")
SEG = {"sharpe", "ann", "mdd", "cum", "final_x", "n", "turnover"}
COINS = ["ETC", "TRX", "ATOM", "APT", "KAS"]
FULL_ATR = [None, 2.0, 4.0, 6.0]
BASE_SPECS = {
    "ETC": {"sl": None, "ts": 24},
    "TRX": {"sl": 0.05, "ts": 24},
    "ATOM": {"sl": 0.05, "ts": 24},
    "APT": {"sl": None, "ts": 24},
    "KAS": {"sl": None, "ts": 24},
}


def _load():
    assert OUT.exists(), "results/iter_H49_atr.json missing; run research/run_iter_h49_atr.py"
    return json.loads(OUT.read_text())


def _uniform_keys(d):
    return [r["atr_mult"] for r in d["uniform_rows"]]


def test_iter_h49_schema():
    d = _load()
    cfg = d["config"]
    assert cfg["smoke"] in (True, False)
    assert cfg["bpy"] == 8760.0
    assert cfg["scale"] == 4
    assert cfg["grid"] == "1h"
    assert cfg["atr_window"] == 48
    coins = cfg.get("coins", COINS)
    assert set(coins) <= set(COINS)
    assert cfg["grid_bars"] >= (2500 if cfg["smoke"] else 8000)
    assert cfg["h2_start"] == cfg["grid_bars"] // 2
    assert set(cfg["basket"]) == set(coins)
    for c in coins:
        assert cfg["basket"][c]["sl"] == BASE_SPECS[c]["sl"], c
        assert cfg["basket"][c]["ts"] == BASE_SPECS[c]["ts"], c
        assert cfg["basket"][c]["q"] == 0.3, c
    if not cfg["smoke"]:
        assert _uniform_keys(d) == FULL_ATR
        assert cfg["uniform_cells"] == 4
        assert len(d["percoin_rows"]) == 20
        assert set(d["percoin_best"]) == set(COINS)
    else:
        assert _uniform_keys(d) == [None, 2.0]
    for r in d["uniform_rows"]:
        assert SEG <= set(r["FULL"]), r
        assert r["FULL"]["n"] == cfg["grid_bars"]
        assert isinstance(r["H2_sharpe"], float)
        assert abs(r["d_sharpe_vs_base"] - round(r["FULL"]["sharpe"] - d["base_FULL"]["sharpe"], 3)) < 1e-9
        assert abs(r["d_mdd_vs_base"] - round(r["FULL"]["mdd"] - d["base_FULL"]["mdd"], 4)) < 1e-9
    for r in d["percoin_rows"]:
        assert SEG <= set(r["FULL"]), r
        assert r["coin"] in coins
    for c in coins:
        crs = [r for r in d["percoin_rows"] if r["coin"] == c]
        assert {r["atr_mult"] for r in crs} == ({None, 2.0} if cfg["smoke"] else set(FULL_ATR))
        best = d["percoin_best"][c]
        exp = max(crs, key=lambda r: (r["FULL"]["sharpe"], -r["FULL"]["mdd"]))
        assert best["atr_mult"] == exp["atr_mult"], c
        assert best["FULL"]["sharpe"] == exp["FULL"]["sharpe"], c
    nodrop = [r for r in d["uniform_rows"] if r["FULL"]["sharpe"] >= d["base_FULL"]["sharpe"]]
    if nodrop:
        exp = min(nodrop, key=lambda r: r["FULL"]["mdd"])
        assert d["best_no_drop_dd_min"] is not None
        assert d["best_no_drop_dd_min"]["atr_mult"] == exp["atr_mult"]
    else:
        assert d["best_no_drop_dd_min"] is None
    assert d["verdict"] == "PENDING"
    assert "PENDING" in d["conclusion"]
    assert "no adoption" in d["conclusion"]
    assert "live untouched" in d["conclusion"]


def test_iter_h49_no_broker():
    src = SRC.read_text()
    for bad in ("place_order", "submit_order", "api_key", "make_broker", "Y1B_LIVE"):
        assert bad.lower() not in src.lower(), bad


def test_iter_h49_engine_mirror():
    src = SRC.read_text()
    for token in ("quantile", "cooldown", "_apply_stops", "_vol_scale",
                  "roll(1", "8760", "cooldown_bars=cd * SCALE",
                  "time_stop=ts * SCALE", "vol_window=vw * SCALE",
                  "atr_norm", "apply_atr_stop"):
        assert token in src, token
    assert "short_enabled=True" in src
    assert "ATR_WINDOW = 48" in src


def test_iter_h49_smoke_runs_offline():
    env = dict(os.environ, ITER_H49_SMOKE="1",
               ITER_H49_OUT="results/iter_H49_atr_smoke.json",
               ITER_H49_LOG="logs/iter_h49_atr_smoke.log")
    r = subprocess.run([sys.executable, "research/run_iter_h49_atr.py"],
                       capture_output=True, text=True, cwd=".", env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    d = json.loads(pathlib.Path("results/iter_H49_atr_smoke.json").read_text())
    assert d["config"]["smoke"] is True
    assert [x["atr_mult"] for x in d["uniform_rows"]] == [None, 2.0]
    assert d["verdict"] == "PENDING"
