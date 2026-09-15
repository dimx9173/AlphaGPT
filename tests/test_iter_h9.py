"""H9 1h correlation monitor tests: pure math + artifact schema."""
import json
import os
import pathlib
import subprocess
import sys


def _load():
    p = pathlib.Path("results/iter_H9_corr.json")
    assert p.exists(), "results/iter_H9_corr.json missing; run research/run_iter_h9_corr.py"
    return json.loads(p.read_text())


def test_corr_brake_map():
    sys.path.insert(0, ".")
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_iter_h9_corr", "research/run_iter_h9_corr.py")
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
    spec = importlib.util.spec_from_file_location("run_iter_h9_corr", "research/run_iter_h9_corr.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert m.pearson([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0
    assert m.pearson([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == -1.0
    assert m.pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) == 0.0
    assert m.pearson([1.0], [2.0]) == 0.0


def test_roll_corr_unit():
    sys.path.insert(0, ".")
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_iter_h9_corr", "research/run_iter_h9_corr.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    a = [float(i) for i in range(10)]
    rc = m.roll_corr(a, list(a), 4)
    assert len(rc) == 7
    assert all(abs(v - 1.0) < 1e-9 for v in rc)
    b = [float(9 - i) for i in range(10)]
    rc2 = m.roll_corr(a, b, 4)
    assert all(abs(v + 1.0) < 1e-9 for v in rc2)


def test_iter_h9_schema():
    d = _load()
    assert d["verdict"] == "PENDING"
    cfg = d["config"]
    assert cfg["window"] == 240
    assert cfg["thresholds"] == {"high": 0.6, "cut": 0.7, "halt": 0.85}
    assert set(cfg["coins"]) == {"ETC", "TRX", "ATOM", "APT", "KAS"}
    assert cfg["grid"] == "1h"
    assert cfg["bpy"] == 8760.0
    assert cfg["n_bars"] == cfg["grid_bars"] > 8000
    assert cfg["n_win"] == cfg["n_bars"] - 240 + 1
    assert d["engine_done"] is True
    st = d["stats"]
    assert st["n_pairs"] == 10 and len(st["pairs"]) == 10
    for k, v in st["pairs"].items():
        assert {"max", "mean", "q50", "q95", "share_gt_0.6", "n_win"} <= set(v)
        assert -1.0 <= v["mean"] <= 1.0 and -1.0 <= v["max"] <= 1.0
        assert 0.0 <= v["share_gt_0.6"] <= 1.0
        assert v["n_win"] == cfg["n_win"]
    for k in ("avg_max", "max_max", "q50_max", "q75_max", "q90_max", "q95_max",
              "q99_max", "share_max_gt_0.6", "share_max_gt_0.7",
              "share_max_gt_0.85", "avg_mean", "max_mean",
              "share_mean_gt_0.6", "share_mean_gt_0.4"):
        assert k in st, k
    assert d["diversification"]["verdict"] in ("LIMITED", "WATCH", "CONCENTRATED", "DIVERSIFIED")
    assert d["breaker"]["default"] == "OFF"
    assert "0.7" in d["breaker"]["rule"] and "0.85" in d["breaker"]["rule"]


def test_iter_h9_no_broker():
    src = pathlib.Path("research/run_iter_h9_corr.py").read_text()
    for bad in ("place_order", "submit_order", "api_key", "API_KEY",
                "make_broker", "Y1B_LIVE"):
        assert bad.lower() not in src.lower(), bad


def test_iter_h9_script_runs_offline():
    r = subprocess.run([sys.executable, "research/run_iter_h9_corr.py"],
                       capture_output=True, text=True, cwd=".")
    assert r.returncode == 0, r.stderr[-2000:]
    d = _load()
    assert d["verdict"] == "PENDING"
    assert len(d["stats"]["pairs"]) == 10
