"""Z2 15m cooldown sweep tests: Top5 15m-native uniform cd + per-coin sensitivity.

Full Z2 is 4 uniform cells (cd {3,6,12,24}) + 20 per-coin runs
(5 coins x cd 4); smoke = 2 uniform + 4 per-coin via ITER_Z2_SMOKE=1
(coins {ETC,TRX}, first 3000 bars). This test never re-runs the full
sweep: schema checks accept either artifact, and the live subprocess
check runs smoke mode only. Conclusion must stay PENDING: diagnostic
only, no adoption, live untouched.
"""
import json
import os
import pathlib
import subprocess
import sys

OUT = pathlib.Path(os.getenv("ITER_Z2_OUT", "results/iter_Z2_cd.json"))
SEG = {"sharpe", "mdd", "turnover", "ann", "cum", "n", "trades"}
COINS5 = {"ETC", "TRX", "ATOM", "APT", "KAS"}
FULL_CD = [3, 6, 12, 24]
BASE_CD = {"ETC": 18, "TRX": 6, "ATOM": 6, "APT": 18, "KAS": 6}
KNEE_CUT = 0.20


def _load():
    assert OUT.exists(), "results/iter_Z2_cd.json missing; run research/run_iter_z2_cd.py"
    return json.loads(OUT.read_text())


def _expected_keys(d):
    smoke = d["config"]["smoke"]
    if smoke:
        return {6, 12}, {(c, t) for c in ("ETC", "TRX") for t in (6, 12)}
    return set(FULL_CD), {(c, t) for c in COINS5 for t in FULL_CD}


def test_iter_z2_lock():
    from strategy_manager.config import LEV, FORMULA, LOCKED_ETC, LOCKED_TRX, LOCKED_ATOM, LOCKED_APT, LOCKED_KAS
    assert LEV == 2.0
    assert FORMULA == [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10]
    assert (LOCKED_ETC["lth"], LOCKED_ETC["sth"], LOCKED_ETC["cd"], LOCKED_ETC["sl"], LOCKED_ETC["ts"]) == (0.88, 0.12, 18, None, 24)
    assert (LOCKED_TRX["lth"], LOCKED_TRX["sth"], LOCKED_TRX["cd"], LOCKED_TRX["sl"], LOCKED_TRX["ts"]) == (0.85, 0.12, 6, 0.05, 24)
    assert (LOCKED_ATOM["lth"], LOCKED_ATOM["sth"], LOCKED_ATOM["cd"], LOCKED_ATOM["sl"], LOCKED_ATOM["ts"]) == (0.85, 0.15, 6, 0.05, 24)
    assert (LOCKED_APT["lth"], LOCKED_APT["sth"], LOCKED_APT["cd"], LOCKED_APT["sl"], LOCKED_APT["ts"]) == (0.88, 0.12, 18, None, 24)
    assert (LOCKED_KAS["lth"], LOCKED_KAS["sth"], LOCKED_KAS["cd"], LOCKED_KAS["sl"], LOCKED_KAS["ts"]) == (0.88, 0.12, 6, None, 24)


def test_iter_z2_schema():
    d = _load()
    cfg = d["config"]
    assert cfg["smoke"] in (True, False)
    assert cfg["bpy"] == 35040.0
    assert cfg["scale"] == 16
    assert cfg["grid"] == "15m"
    assert cfg["grid_bars"] > 2000
    assert cfg["knee_cut"] == KNEE_CUT
    if not cfg["smoke"]:
        assert cfg["coins"] == ["ETC", "TRX", "ATOM", "APT", "KAS"]
        assert cfg["grid_bars"] == 35040
        assert set(cfg["weights"]) == COINS5
        for c in COINS5:
            assert cfg["basket"][c]["cd"] == BASE_CD[c], c
            assert cfg["basket"][c]["q"] == 0.3, c
    u_keys, p_keys = _expected_keys(d)
    assert {r["cd"] for r in d["uniform_rows"]} == u_keys
    assert {(r["coin"], r["cd"]) for r in d["percoin_rows"]} == p_keys
    for r in d["uniform_rows"]:
        assert SEG <= set(r["FULL"]), r
        assert r["FULL"]["n"] == cfg["grid_bars"]
        assert isinstance(r["H2_sharpe"], float)
        assert r["FULL"]["turnover"] >= 0
        assert r["FULL"]["trades"] >= 0
        assert isinstance(r["H2_trades"], int) and r["H2_trades"] >= 0
        assert r["H2_trades"] <= r["FULL"]["trades"]
        assert abs(r["d_sharpe_vs_base"] - round(r["FULL"]["sharpe"] - d["base_FULL"]["sharpe"], 3)) < 1e-9
        assert abs(r["d_mdd_vs_base"] - round(r["FULL"]["mdd"] - d["base_FULL"]["mdd"], 4)) < 1e-9
        assert abs(r["d_turnover_vs_base"] - round(r["FULL"]["turnover"] - d["base_FULL"]["turnover"], 6)) < 1e-9
        base_to = d["base_FULL"]["turnover"]
        exp_cut = round((base_to - r["FULL"]["turnover"]) / base_to, 4) if base_to > 0 else 0.0
        assert abs(r["turnover_cut_vs_base"] - exp_cut) < 1e-9
    for r in d["percoin_rows"]:
        assert SEG <= set(r["FULL"]), r
        assert r["FULL"]["n"] == cfg["grid_bars"]
        assert r["FULL"]["trades"] >= 0
        assert abs(r["d_sharpe_vs_base"] - round(r["FULL"]["sharpe"] - d["base_FULL"]["sharpe"], 3)) < 1e-9
    assert set(d["percoin_best"]) == set(cfg["coins"])
    for c, best in d["percoin_best"].items():
        crs = [r for r in d["percoin_rows"] if r["coin"] == c]
        exp = max(crs, key=lambda r: (r["FULL"]["sharpe"], -r["FULL"]["mdd"]))
        assert best["cd"] == exp["cd"], c
        assert best["FULL"] == exp["FULL"], c
    nodrop = [r for r in d["uniform_rows"] if r["FULL"]["sharpe"] >= d["base_FULL"]["sharpe"]]
    if nodrop:
        exp = min(nodrop, key=lambda r: r["FULL"]["mdd"])
        assert d["best_no_drop_dd_min"] is not None
        assert d["best_no_drop_dd_min"]["cd"] == exp["cd"]
        assert d["best_no_drop_dd_min"]["FULL"]["sharpe"] >= d["base_FULL"]["sharpe"]
    else:
        assert d["best_no_drop_dd_min"] is None
    # knee: first turnover cut >=20% vs base with sharpe no-drop, min turnover wins
    cand = [r for r in d["uniform_rows"]
            if r["turnover_cut_vs_base"] >= KNEE_CUT
            and r["FULL"]["sharpe"] >= d["base_FULL"]["sharpe"]]
    if cand:
        exp = min(cand, key=lambda r: (r["FULL"]["turnover"], -r["FULL"]["sharpe"]))
        assert d["knee"] is not None
        assert d["knee"]["cd"] == exp["cd"]
        assert d["knee"]["FULL"]["sharpe"] >= d["base_FULL"]["sharpe"]
    else:
        assert d["knee"] is None
    assert d["verdict"] == "PENDING"
    assert d["decision"] == "PENDING"
    assert "PENDING" in d["conclusion"]
    assert "no adoption" in d["conclusion"]
    assert "live untouched" in d["conclusion"]


def test_iter_z2_engine_mirror():
    src = pathlib.Path("research/run_iter_z2_cd.py").read_text()
    for token in ("quantile", "cooldown", "_apply_stops", "_vol_scale",
                  "roll(1", "35040", "cooldown_bars=cd * SCALE",
                  "time_stop=ts * SCALE", "vol_window=vw * SCALE"):
        assert token in src, token
    assert "short_enabled=True" in src


def test_iter_z2_no_broker():
    src = pathlib.Path("research/run_iter_z2_cd.py").read_text()
    for bad in ("place_order", "submit_order", "api_key", "API_KEY",
                "make_broker", "Y1B_LIVE"):
        assert bad.lower() not in src.lower(), bad


def test_iter_z2_smoke_runs_offline(tmp_path):
    out = tmp_path / "iter_Z2_cd.json"
    lg = tmp_path / "iter_z2_cd.log"
    env = dict(os.environ, ITER_Z2_SMOKE="1", ITER_Z2_OUT=str(out), ITER_Z2_LOG=str(lg))
    r = subprocess.run([sys.executable, "research/run_iter_z2_cd.py"],
                       capture_output=True, text=True, cwd=".", env=env, timeout=1200)
    assert r.returncode == 0, r.stderr[-2000:]
    dd = json.loads(out.read_text())
    assert dd["config"]["smoke"] is True
    assert {x["cd"] for x in dd["uniform_rows"]} == {6, 12}
    assert {(x["coin"], x["cd"]) for x in dd["percoin_rows"]} == {
        (c, t) for c in ("ETC", "TRX") for t in (6, 12)}
    assert dd["verdict"] == "PENDING"
    for x in dd["uniform_rows"]:
        assert x["FULL"]["trades"] >= 0
        assert x["H2_trades"] <= x["FULL"]["trades"]
