"""H78 1h Parabolic SAR filter (Top5): base + 3 SAR param cells, PENDING (no adoption)."""
import json
import os
import pathlib
import subprocess
import sys

OUT = pathlib.Path(os.getenv("ITER_H78_OUT", "results/iter_H78_sar.json"))
FULL_SEG = {"sharpe", "ann", "mdd", "cum", "final_x", "n", "turnover",
            "trades", "long_cov", "short_cov"}
COIN_SEG = {"sharpe", "ann", "mdd", "cum", "final_x", "n", "turnover",
            "trades", "long_cov", "short_cov"}


def _load():
    assert OUT.exists(), "results/iter_H78_sar.json missing; run research/run_iter_h78_sar.py"
    return json.loads(OUT.read_text())


def test_iter_h78_lock_and_grid():
    from strategy_manager.config import LEV
    assert LEV == 2.0
    d = _load()
    cfg = d["config"]
    assert cfg["formula"] == [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10]
    assert cfg["grid"] == "1h"
    assert cfg["bpy"] == 8760.0
    assert cfg["scale"] == 4
    assert cfg["lev"] == 2.0
    assert cfg["fund"] == 0.0005
    assert cfg["fee"] == 0.0004
    assert cfg["sar_params"] == [[0.01, 0.10], [0.02, 0.20], [0.04, 0.20]]
    assert cfg["grid_bars"] == 8760
    assert abs(sum(cfg["weights"].values()) - 1.0) < 1e-9
    assert d["verdict"] == "PENDING"
    assert d["decision"] == "NO_ADOPTION"
    assert d["complete"] is True
    assert "PENDING" in d["conclusion"]


def test_iter_h78_rows_and_sensitivity():
    d = _load()
    assert d["units_done"] == ["base", "sar_af0.01_max0.10",
                               "sar_af0.02_max0.20", "sar_af0.04_max0.20"]
    assert len(d["rows"]) == 4
    base = d["rows"][0]
    assert base["unit"] == "base" and base["sar"] is None
    assert FULL_SEG <= set(base["FULL"])
    assert base["FULL"]["n"] == d["config"]["grid_bars"]
    assert base["FULL"]["long_cov"] == 1.0
    assert base["FULL"]["short_cov"] == 1.0
    for r in d["rows"][1:]:
        assert r["sar"] in ([0.01, 0.10], [0.02, 0.20], [0.04, 0.20])
        assert FULL_SEG <= set(r["FULL"])
        assert r["FULL"]["n"] == d["config"]["grid_bars"]
        assert 0.0 < r["FULL"]["long_cov"] < 1.0
        assert 0.0 < r["FULL"]["short_cov"] < 1.0
        assert abs(r["FULL"]["long_cov"] + r["FULL"]["short_cov"] - 1.0) < 0.01
        assert r["FULL"]["trades"] >= 0
        assert r["FULL"]["turnover"] >= 0
        for c, pc in r["per_coin"].items():
            assert COIN_SEG <= set(pc), (c, sorted(pc))
            assert pc["n"] == d["config"]["grid_bars"]
            assert 0.0 < pc["long_cov"] < 1.0
            assert 0.0 < pc["short_cov"] < 1.0
            assert abs(pc["long_cov"] + pc["short_cov"] - 1.0) < 0.01
            assert pc["trades"] >= 0
            assert abs(pc["final_x"] - (1.0 + pc["cum"])) < 1e-3
        assert abs(r["FULL"]["final_x"] - (1.0 + r["FULL"]["cum"])) < 1e-3
    assert set(base["per_coin"]) == set(d["config"]["weights"])


def test_iter_h78_offline_and_live_untouched():
    src = pathlib.Path("research/run_iter_h78_sar.py").read_text()
    for bad in ("place_order", "submit_order", "api_key", "make_broker",
                "Y1B_LIVE", "y1b_executor", "y1b_basket"):
        assert bad.lower() not in src.lower(), bad
    assert "data/data_1y/1h" in src
    assert "SAR" in src
    live = pathlib.Path("strategy_manager/y1b_basket.py").read_text()
    assert "SAR" not in live and "sar" not in live.lower().replace("necessary", "")
    cfg_src = pathlib.Path("strategy_manager/config.py").read_text()
    assert "SAR" not in cfg_src


def test_iter_h78_smoke_runs_offline():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "iter_H78_sar.json")
        lg = os.path.join(td, "iter_H78_sar.log")
        env = dict(os.environ, ITER_H78_SMOKE="1", ITER_H78_OUT=out,
                   ITER_H78_LOG=lg)
        r = subprocess.run([sys.executable, "research/run_iter_h78_sar.py"],
                           capture_output=True, text=True, cwd=".", env=env)
        assert r.returncode == 0, r.stderr[-2000:]
        d = json.loads(pathlib.Path(out).read_text())
        assert d["config"]["smoke"] is True
        assert d["units_done"] == ["base", "sar_af0.02_max0.20"]
        assert len(d["rows"]) == 2
        assert d["verdict"] == "PENDING"
