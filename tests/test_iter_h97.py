"""H97 1h Ichimoku v2 filter (Top5): (18,52,104) long-above / short-below, PENDING (no adoption)."""
import json
import os
import pathlib
import subprocess
import sys

OUT = pathlib.Path(os.getenv("ITER_H97_OUT", "results/iter_H97_ichi.json"))
FULL_SEG = {"sharpe", "ann", "mdd", "cum", "final_x", "n", "turnover",
            "trades", "coverage_all", "coverage_long", "coverage_short"}
COIN_SEG = {"sharpe", "ann", "mdd", "cum", "final_x", "n", "turnover",
            "trades", "coverage", "coverage_long", "coverage_short"}


def _load():
    assert OUT.exists(), "results/iter_H97_ichi.json missing; run research/run_iter_h97_ichi.py"
    return json.loads(OUT.read_text())


def test_iter_h97_lock_and_grid():
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
    assert cfg["ichimoku"] == {"tenkan": 18, "kijun": 52, "senkou": 104,
                               "displacement": 52, "first_valid_bar": 155}
    assert cfg["modes"] == ["base", "ichi_long", "ichi_short", "ichi_both"]
    assert cfg["grid_bars"] == 8760
    assert abs(sum(cfg["weights"].values()) - 1.0) < 1e-9
    assert d["verdict"] == "PENDING"
    assert d["decision"] == "NO_ADOPTION"
    assert d["complete"] is True
    assert "PENDING" in d["conclusion"]


def test_iter_h97_rows_and_sensitivity():
    d = _load()
    assert d["units_done"] == ["base", "ichi_long", "ichi_short", "ichi_both"]
    assert len(d["rows"]) == 4
    base = d["rows"][0]
    assert base["unit"] == "base" and base["mode"] == "base"
    assert FULL_SEG <= set(base["FULL"])
    assert base["FULL"]["n"] == d["config"]["grid_bars"]
    assert base["FULL"]["coverage_all"] == 1.0
    assert base["FULL"]["coverage_long"] == 1.0
    assert base["FULL"]["coverage_short"] == 1.0
    for r in d["rows"][1:]:
        assert r["mode"] in ("ichi_long", "ichi_short", "ichi_both")
        assert FULL_SEG <= set(r["FULL"])
        assert r["FULL"]["n"] == d["config"]["grid_bars"]
        assert 0.0 < r["FULL"]["coverage_long"] <= 1.0
        assert 0.0 < r["FULL"]["coverage_short"] <= 1.0
        assert r["FULL"]["trades"] >= 0
        assert r["FULL"]["turnover"] >= 0
        for c, pc in r["per_coin"].items():
            assert COIN_SEG <= set(pc), (c, sorted(pc))
            assert pc["n"] == d["config"]["grid_bars"]
            assert 0.0 <= pc["coverage_long"] <= 1.0
            assert 0.0 <= pc["coverage_short"] <= 1.0
            assert pc["trades"] >= 0
            assert abs(pc["final_x"] - (1.0 + pc["cum"])) < 1e-3
        assert abs(r["FULL"]["final_x"] - (1.0 + r["FULL"]["cum"])) < 1e-3
    both = d["rows"][3]
    assert both["FULL"]["coverage_long"] < 1.0
    assert both["FULL"]["coverage_short"] < 1.0
    assert set(base["per_coin"]) == set(d["config"]["weights"])


def test_iter_h97_offline_and_live_untouched():
    src = pathlib.Path("research/run_iter_h97_ichi.py").read_text()
    for bad in ("place_order", "submit_order", "api_key", "make_broker",
                "Y1B_LIVE", "y1b_executor", "y1b_basket"):
        assert bad.lower() not in src.lower(), bad
    assert "data/data_1y/1h" in src
    assert "Ichimoku" in src or "ichimoku" in src.lower()
    assert "(18,52,104" in src or "[18, 52, 104" in src or "18, 52, 104" in src
    assert "155" in src
    live = pathlib.Path("strategy_manager/y1b_basket.py").read_text()
    assert "chimoku" not in live
    cfg_src = pathlib.Path("strategy_manager/config.py").read_text()
    assert "chimoku" not in cfg_src


def test_iter_h97_smoke_runs_offline():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "iter_H97_ichi.json")
        lg = os.path.join(td, "iter_H97_ichi.log")
        env = dict(os.environ, ITER_H97_SMOKE="1", ITER_H97_OUT=out,
                   ITER_H97_LOG=lg)
        r = subprocess.run([sys.executable, "research/run_iter_h97_ichi.py"],
                           capture_output=True, text=True, cwd=".", env=env)
        assert r.returncode == 0, r.stderr[-2000:]
        d = json.loads(pathlib.Path(out).read_text())
        assert d["config"]["smoke"] is True
        assert d["units_done"] == ["base", "ichi_both"]
        assert len(d["rows"]) == 2
        assert d["verdict"] == "PENDING"
