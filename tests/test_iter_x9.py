"""X9 rolling-60 correlation monitor tests: pure math + artifact schema.

Recompute is done by research/run_iter_x9.py (offline, ~4s).
"""
import json
import os
import pathlib
import subprocess
import sys


def _load():
    p = pathlib.Path("results/iter_X9_corr.json")
    assert p.exists(), "results/iter_X9_corr.json missing; run research/run_iter_x9.py"
    return json.loads(p.read_text())


def test_corr_brake_map():
    sys.path.insert(0, ".")
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_iter_x9", "research/run_iter_x9.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert m.corr_brake_scale(0.5) == 1.0
    assert m.corr_brake_scale(0.65) == 1.0
    assert m.corr_brake_scale(0.65, 0.65) == 0.5
    assert m.corr_brake_scale(0.71) == 0.5
    assert m.corr_brake_scale(0.7) == 1.0
    assert m.corr_brake_scale(0.86) == 0.25
    assert m.corr_brake_scale(0.7, 0.9) == 0.5
    assert m.corr_brake_scale(None) == 1.0


def test_pearson_unit():
    sys.path.insert(0, ".")
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_iter_x9", "research/run_iter_x9.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert m.pearson([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0
    assert m.pearson([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == -1.0
    assert m.pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) == 0.0
    assert m.pearson([1.0], [2.0]) == 0.0


def test_iter_x9_schema():
    d = _load()
    assert d["verdict"] == "PENDING_P03_FAIL"
    assert d["decision"] == "NO_CHANGE"
    assert "\u5f85\u5b9a" in d["conclusion"] or "待定" in d["conclusion"]
    cfg = d["config"]
    assert cfg["window"] == 60
    assert cfg["thresholds"] == {"high": 0.6, "cut": 0.7, "halt": 0.85}
    assert set(cfg["coins"]) == {"ETC", "TRX", "ATOM", "APT", "KAS"}
    assert d["config"]["n_bars"] == 6135
    assert d["config"]["n_win"] == 6075
    assert len(d["pairs"]) == 10
    for k, v in d["pairs"].items():
        assert {"mean", "max", "share_gt_0.6"} <= set(v)
        assert -1.0 <= v["mean"] <= 1.0 and -1.0 <= v["max"] <= 1.0
        assert 0.0 <= v["share_gt_0.6"] <= 1.0
    agg = d["aggregate"]
    for k in ("avg_max", "max_max", "q50_max", "q75_max", "q90_max", "q95_max",
              "q99_max", "share_max_gt_0.6", "share_max_gt_0.7",
              "share_max_gt_0.85", "avg_mean", "max_mean",
              "share_mean_gt_0.6", "share_mean_gt_0.4"):
        assert k in agg, k
    assert agg["share_max_gt_0.6"] > 0.9
    assert agg["avg_max"] > 0.7
    assert d["diversification"]["verdict"] == "LIMITED"
    assert d["breaker"]["default"] == "OFF"
    assert "0.7" in d["breaker"]["rule"] and "0.85" in d["breaker"]["rule"]


def test_iter_x9_no_broker():
    src = pathlib.Path("research/run_iter_x9.py").read_text()
    for bad in ("place_order", "submit_order", "api_key", "API_KEY",
                "make_broker", "Y1B_LIVE"):
        assert bad.lower() not in src.lower(), bad


def test_iter_x9_script_runs_offline():
    r = subprocess.run([sys.executable, "research/run_iter_x9.py"],
                       capture_output=True, text=True, cwd=".")
    assert r.returncode == 0, r.stderr[-2000:]
    d = _load()
    assert d["verdict"] == "PENDING_P03_FAIL"
    assert len(d["pairs"]) == 10
