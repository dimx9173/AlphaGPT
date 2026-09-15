"""H19 1h parameter-interaction tests: Top5 1h-native slxq + cdxts grids.

Full H19 is 8 uniform cells (sl 2 x q 2 + cd 2 x ts 2); smoke = 4 cells
(sl {None} x q {0.2,0.3} + cd {6} x ts {12,24}) via ITER_H19_SMOKE=1.
This test never re-runs the full sweep: schema checks accept either
artifact, and the live subprocess check runs smoke mode only.
Conclusion must stay PENDING: diagnostic only, no adoption, live untouched.
"""
import json
import os
import pathlib
import subprocess
import sys

OUT = pathlib.Path("results/iter_H19_interact.json")
SEG = {"sharpe", "ann", "mdd", "cum", "final_x", "n", "turnover"}
COINS = {"ETC", "TRX", "ATOM", "APT", "KAS"}
FULL_SL = [None, 0.03]
FULL_Q = [0.2, 0.3]
FULL_CD = [6, 12]
FULL_TS = [12, 24]


def _load():
    assert OUT.exists(), "results/iter_H19_interact.json missing; run research/run_iter_h19_interact.py"
    return json.loads(OUT.read_text())


def _expected(d):
    smoke = d["config"]["smoke"]
    if smoke:
        return {(None, 0.2), (None, 0.3)}, {(6, 12), (6, 24)}
    return {(s, q) for s in FULL_SL for q in FULL_Q}, {(c, t) for c in FULL_CD for t in FULL_TS}


def _check_rows(d, rows, keys):
    cfg = d["config"]
    for r in rows:
        assert SEG <= set(r["FULL"]), r
        assert r["FULL"]["n"] == cfg["grid_bars"]
        assert isinstance(r["H2_sharpe"], float)
        assert abs(r["d_sharpe_vs_base"] - round(r["FULL"]["sharpe"] - d["base_FULL"]["sharpe"], 3)) < 1e-9
        assert abs(r["d_mdd_vs_base"] - round(r["FULL"]["mdd"] - d["base_FULL"]["mdd"], 4)) < 1e-9
    assert {(r[keys[0]], r[keys[1]]) for r in rows} == _expected(d)[0 if keys[0] == "sl" else 1]


def test_iter_h19_schema():
    d = _load()
    cfg = d["config"]
    assert cfg["smoke"] in (True, False)
    assert cfg["bpy"] == 8760.0
    assert cfg["scale"] == 4
    assert cfg["grid_bars"] > 8000
    assert cfg["h2_start"] == cfg["grid_bars"] // 2
    assert set(cfg["weights"]) == COINS
    assert set(cfg["basket"]) == COINS
    for c in COINS:
        assert cfg["basket"][c]["q"] == 0.3, c
    _check_rows(d, d["sl_x_q_rows"], ("sl", "q"))
    _check_rows(d, d["cd_x_ts_rows"], ("cd", "ts"))
    # heatmaps consistent with rows (FULL sharpe only)
    for tag, rows in (("heatmap_sl_x_q", d["sl_x_q_rows"]), ("heatmap_cd_x_ts", d["cd_x_ts_rows"])):
        hm = d[tag]
        assert len(hm["sharpe"]) == len(hm["rows"]) == 2 or (cfg["smoke"] and tag == "heatmap_sl_x_q" and len(hm["rows"]) == 1)
        assert len(hm["sharpe"][0]) == len(hm["cols"]) == 2
        m = {(r["r"], r["c"]): r for r in rows}
        for i, rv in enumerate(hm["rows"]):
            for j, cv in enumerate(hm["cols"]):
                assert hm["sharpe"][i][j] == m[(rv, cv)]["FULL"]["sharpe"]
    # interaction = difference-in-differences of FULL sharpe
    for tag, key in (("heatmap_sl_x_q", "interaction_sl_x_q"), ("heatmap_cd_x_ts", "interaction_cd_x_ts")):
        sh = d[tag]["sharpe"]
        if len(sh) == 2 and len(sh[0]) == 2:
            exp = round(sh[1][1] - sh[1][0] - sh[0][1] + sh[0][0], 3)
            assert d[key] == exp, (key, d[key], exp)
        else:
            assert d[key] is None
    # best cells = max FULL sharpe rows
    assert (d["best_sl_x_q"]["sl"], d["best_sl_x_q"]["q"]) == (
        max(d["sl_x_q_rows"], key=lambda r: r["FULL"]["sharpe"])["sl"],
        max(d["sl_x_q_rows"], key=lambda r: r["FULL"]["sharpe"])["q"])
    assert (d["best_cd_x_ts"]["cd"], d["best_cd_x_ts"]["ts"]) == (
        max(d["cd_x_ts_rows"], key=lambda r: r["FULL"]["sharpe"])["cd"],
        max(d["cd_x_ts_rows"], key=lambda r: r["FULL"]["sharpe"])["ts"])
    assert d["verdict"] == "PENDING"
    assert "\u5f85\u5b9a" in d["conclusion"] or "待定" in d["conclusion"]
    assert "no adoption" in d["conclusion"]
    assert "live untouched" in d["conclusion"]


def test_iter_h19_no_broker():
    src = pathlib.Path("research/run_iter_h19_interact.py").read_text()
    for bad in ("place_order", "submit_order", "api_key", "API_KEY",
                "make_broker", "Y1B_LIVE"):
        assert bad.lower() not in src.lower(), bad


def test_iter_h19_engine_mirror():
    src = pathlib.Path("research/run_iter_h19_interact.py").read_text()
    for token in ("quantile", "cooldown", "_apply_stops", "_vol_scale",
                  "roll(1", "8760", "cooldown_bars=cd * SCALE",
                  "time_stop=ts * SCALE", "vol_window=vw * SCALE"):
        assert token in src, token
    assert "short_enabled=True" in src


def test_iter_h19_smoke_runs_offline():
    env = dict(os.environ, ITER_H19_SMOKE="1")
    r = subprocess.run([sys.executable, "research/run_iter_h19_interact.py"],
                       capture_output=True, text=True, cwd=".", env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    d = _load()
    assert d["config"]["smoke"] is True
    assert {(x["sl"], x["q"]) for x in d["sl_x_q_rows"]} == {(None, 0.2), (None, 0.3)}
    assert {(x["cd"], x["ts"]) for x in d["cd_x_ts_rows"]} == {(6, 12), (6, 24)}
    assert d["verdict"] == "PENDING"
