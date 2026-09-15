"""H88 1h ADX filter (Top5): base + ADX>20/25/30, PENDING (no adoption)."""
import json
import os
import pathlib
import subprocess
import sys

OUT = pathlib.Path(os.getenv("ITER_H88_OUT", "results/iter_H88_adx.json"))
FULL_SEG = {"sharpe", "ann", "mdd", "cum", "final_x", "n", "turnover",
            "trades", "coverage_all"}
COIN_SEG = {"sharpe", "ann", "mdd", "cum", "final_x", "n", "turnover",
            "trades", "coverage"}


def _load():
    assert OUT.exists(), "results/iter_H88_adx.json missing; run research/run_iter_h88_adx.py"
    return json.loads(OUT.read_text())


def test_iter_h88_lock_and_grid():
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
    assert cfg["adx_period"] == 224
    assert cfg["thresholds"] == [20, 25, 30]
    assert cfg["grid_bars"] == 8760
    assert abs(sum(cfg["weights"].values()) - 1.0) < 1e-9
    assert d["verdict"] == "PENDING"
    assert d["decision"] == "NO_ADOPTION"
    assert d["complete"] is True
    assert "待定" in d["decision_note"]


def test_iter_h88_rows_and_sensitivity():
    d = _load()
    assert d["units_done"] == ["base", "adx_gt_20", "adx_gt_25", "adx_gt_30"]
    assert len(d["rows"]) == 4
    base = d["rows"][0]
    assert base["unit"] == "base" and base["thr"] is None
    assert FULL_SEG <= set(base["FULL"])
    assert base["FULL"]["n"] == d["config"]["grid_bars"]
    covs = []
    for r in d["rows"][1:]:
        assert r["thr"] in (20, 25, 30)
        assert FULL_SEG <= set(r["FULL"])
        assert r["FULL"]["n"] == d["config"]["grid_bars"]
        assert 0.0 <= r["FULL"]["coverage_all"] < 1.0  # ADX224 joint (all-5) coverage can round to 0.0 at thr 25/30
        assert r["FULL"]["trades"] >= 0
        assert r["FULL"]["turnover"] >= 0
        covs.append(r["FULL"]["coverage_all"])
        for c, pc in r["per_coin"].items():
            assert COIN_SEG <= set(pc), (c, sorted(pc))
            assert pc["n"] == d["config"]["grid_bars"]
            assert 0.0 <= pc["coverage"] < 1.0  # ADX224 single-coin coverage can round to 0.0 at high thr
            assert pc["trades"] >= 0
            assert abs(pc["final_x"] - (1.0 + pc["cum"])) < 1e-3
        assert abs(r["FULL"]["final_x"] - (1.0 + r["FULL"]["cum"])) < 1e-3
    assert covs[0] >= covs[1] >= covs[2], "higher thr must not cover more bars"
    assert covs[0] < 1.0, "ADX224>20 must filter at least one bar"
    assert set(base["per_coin"]) == set(d["config"]["weights"])


def test_iter_h88_offline_and_live_untouched():
    src = pathlib.Path("research/run_iter_h88_adx.py").read_text()
    for bad in ("place_order", "submit_order", "api_key", "make_broker",
                "Y1B_LIVE", "y1b_executor", "y1b_basket"):
        assert bad.lower() not in src.lower(), bad
    assert "data/data_1y/1h" in src
    live = pathlib.Path("strategy_manager/y1b_basket.py").read_text()
    assert "ADX" not in live and "adx" not in live
    cfg_src = pathlib.Path("strategy_manager/config.py").read_text()
    assert "ADX" not in cfg_src


def test_iter_h88_smoke_runs_offline():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "iter_H88_adx.json")
        lg = os.path.join(td, "iter_H88_adx.log")
        env = dict(os.environ, ITER_H88_SMOKE="1", ITER_H88_OUT=out,
                   ITER_H88_LOG=lg)
        r = subprocess.run([sys.executable, "research/run_iter_h88_adx.py"],
                           capture_output=True, text=True, cwd=".", env=env)
        assert r.returncode == 0, r.stderr[-2000:]
        d = json.loads(pathlib.Path(out).read_text())
        assert d["config"]["smoke"] is True
        assert d["units_done"] == ["base", "adx_gt_25"]
        assert len(d["rows"]) == 2
        assert d["verdict"] == "PENDING"
