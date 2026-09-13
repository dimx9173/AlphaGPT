"""P2-2 coarse-grid test: schema-level only, recompute done by the script.

Full grid is 288 combos (smoke=4 via GRID_COARSE_SMOKE=1). This test never
re-runs the full grid: schema checks accept either artifact, and the live
subprocess check runs smoke mode only. Conclusion must stay PENDING (待定)
because P0-3 permutation FAILED.
"""
import json
import os
import pathlib
import subprocess
import sys

SEG = {"sharpe", "ann", "mdd", "cum", "n", "turnover"}
ROW = {"sth", "cd", "ts", "q", "FULL", "H2", "fold_median", "folds", "objective"}


def _load():
    p = pathlib.Path("results/grid_coarse.json")
    assert p.exists(), "results/grid_coarse.json missing; run research/run_grid_coarse.py"
    return json.loads(p.read_text())


def test_grid_coarse_schema():
    d = _load()
    cfg = d["config"]
    assert d["config"]["grid"]["sth"] == [round(0.08 + 0.01 * i, 2) for i in range(8)] or cfg["smoke"] is True
    assert set(cfg["weights"]) == {"ETC", "TRX", "ATOM", "APT", "KAS"}
    assert cfg["lam"] == 5.0 and cfg["mu"] == 1.0
    assert cfg["combos"] == len(d["rows"])
    if not cfg["smoke"]:
        assert cfg["combos"] == 288
        assert len(d["heatmap_sth_cd"]) == 8 * 4
        assert len(d["top5"]) == 5
    rows = d["rows"]
    objs = [r["objective"] for r in rows]
    assert objs == sorted(objs, reverse=True), "rows must be objective-desc"
    for r in rows:
        assert ROW <= set(r), r
        assert SEG <= set(r["FULL"]) and SEG <= set(r["H2"])
        assert len(r["folds"]) == cfg["n_fold"]
        assert r["FULL"]["n"] == cfg["grid_bars"]
        assert r["H2"]["n"] == cfg["h2_len"]
    assert len(d["top5"]) == min(5, len(rows))
    assert [t["objective"] for t in d["top5"]] == sorted(
        [t["objective"] for t in d["top5"]], reverse=True)
    hm = d["heatmap_sth_cd"]
    assert len(hm) == len(cfg["grid"]["sth"]) * len(cfg["grid"]["cd"])
    for c in hm:
        assert {"sth", "cd", "objective", "arg_ts", "arg_q",
                "fold_median", "FULL_sharpe"} <= set(c)
    assert d["plateau"]["verdict"] in ("plateau(\u9023\u7247)", "island(\u5b64\u5cf6)")
    assert d["plateau"]["connected"] in (True, False)
    assert d["wrc"]["gate_p_lt_0_05"] in (True, False)
    assert 0.0 <= d["wrc"]["p_value"] <= 1.0
    h = d["hand_config"]
    assert h["rank_vs_grid"] >= 1 and h["rank_vs_grid"] <= len(rows)
    assert h["in_plateau_box"] in (True, False)
    assert "\u5f85\u5b9a" in d["conclusion"]


def test_grid_coarse_no_broker():
    src = pathlib.Path("research/run_grid_coarse.py").read_text()
    for bad in ("place_order", "submit_order", "api_key", "API_KEY",
                "make_broker", "Y1B_LIVE"):
        assert bad.lower() not in src.lower(), bad


def test_grid_coarse_smoke_runs_offline():
    env = dict(os.environ, GRID_COARSE_SMOKE="1")
    r = subprocess.run([sys.executable, "research/run_grid_coarse.py"],
                       capture_output=True, text=True, cwd=".", env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    d = _load()
    assert d["config"]["smoke"] is True
    assert d["config"]["combos"] == 4
    assert len(d["rows"]) == 4
