"""H17 1h max losing streak tests (Top5). Recompute by script."""
import importlib.util
import json
import os
import pathlib
import subprocess
import sys

STREAK_KEYS = {"start", "end", "start_ts", "end_ts", "length_bars", "length_days",
               "cum", "recovered", "recovery_bars", "recovery_days", "n_runs",
               "mean_run_len", "runner_up_bars", "vol_full_ann", "vol_streak_ann",
               "vol_ratio"}
OUT = pathlib.Path(os.getenv("ITER_H17_OUT", "results/iter_H17_streak.json"))


def _load():
    assert OUT.exists(), "results/iter_H17_streak.json missing; run research/run_iter_h17_streak.py"
    return json.loads(OUT.read_text())


def _mod():
    spec = importlib.util.spec_from_file_location(
        "h17mod", "research/run_iter_h17_streak.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["h17mod"] = m
    try:
        spec.loader.exec_module(m)
    except SystemExit:
        pass
    return m


def test_iter_h17_lock():
    from strategy_manager.config import (
        LEV, FUND, FEE, FORMULA,
        LOCKED_ETC, LOCKED_TRX, LOCKED_ATOM, LOCKED_APT, LOCKED_KAS)
    assert LEV == 2.0
    assert FUND == 0.0005
    assert FORMULA == [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10]
    assert (LOCKED_ETC["lth"], LOCKED_ETC["sth"], LOCKED_ETC["cd"],
            LOCKED_ETC["sl"], LOCKED_ETC["ts"]) == (0.88, 0.12, 18, None, 24)
    assert (LOCKED_TRX["lth"], LOCKED_TRX["sth"], LOCKED_TRX["cd"],
            LOCKED_TRX["sl"], LOCKED_TRX["ts"]) == (0.85, 0.12, 6, 0.05, 24)
    assert (LOCKED_ATOM["lth"], LOCKED_ATOM["sth"], LOCKED_ATOM["cd"],
            LOCKED_ATOM["sl"], LOCKED_ATOM["ts"]) == (0.85, 0.15, 6, 0.05, 24)
    assert (LOCKED_APT["lth"], LOCKED_APT["sth"], LOCKED_APT["cd"],
            LOCKED_APT["sl"], LOCKED_APT["ts"]) == (0.88, 0.12, 18, None, 24)
    assert (LOCKED_KAS["lth"], LOCKED_KAS["sth"], LOCKED_KAS["cd"],
            LOCKED_KAS["sl"], LOCKED_KAS["ts"]) == (0.88, 0.12, 6, None, 24)


def test_iter_h17_schema():
    d = _load()
    assert d.get("status") == "final"
    cfg = d["config"]
    assert cfg["grid"] == "1h"
    assert cfg["bpy"] == 8760.0
    assert cfg["scale_4h_to_1h"] == 4
    assert cfg["lev"] == 2.0
    assert cfg["grid_bars"] > 2000
    assert cfg["smoke"] in (True, False)
    coins = cfg["coins"]
    if not cfg["smoke"]:
        assert coins == ["ETC", "TRX", "ATOM", "APT", "KAS"]
    for c in coins:
        sc = cfg["scaled_specs_1h"][c]
        base = cfg["basket"][c]
        for k in ("cd", "ts", "vw"):
            if base[k] is None:
                assert sc[k] is None
            else:
                assert sc[k] == int(base[k]) * 4
    assert set(d["per_coin"]) == set(coins)
    for c, pc in d["per_coin"].items():
        assert set(pc["FULL"]) >= {"n", "cum", "sharpe", "mdd"}
        assert pc["FULL"]["n"] == cfg["grid_bars"]
        assert STREAK_KEYS <= set(pc["streak"]), (c, sorted(set(pc["streak"])))
    pf = d["portfolio"]
    assert STREAK_KEYS <= set(pf["streak"])
    assert pf["FULL"]["n"] == cfg["grid_bars"]
    assert set(pf["weights"]) == set(coins)
    n = cfg["grid_bars"]
    for c in coins:
        s = d["per_coin"][c]["streak"]
        assert 0 <= s["length_bars"] <= n
        assert s["cum"] <= 0.0
        assert abs(s["length_days"] - round(s["length_bars"] / 24.0, 4)) < 1e-9
        if s["length_bars"] == 0:
            assert s["start"] is None and s["end"] is None
            assert s["recovered"] is True and s["recovery_bars"] == 0
        else:
            assert 0 <= s["start"] <= s["end"] < n
            assert s["end"] - s["start"] + 1 == s["length_bars"]
            assert s["runner_up_bars"] <= s["length_bars"]
            if s["recovered"]:
                assert isinstance(s["recovery_bars"], int) and s["recovery_bars"] >= 1
                assert abs(s["recovery_days"] - round(s["recovery_bars"] / 24.0, 4)) < 1e-9
            else:
                assert s["recovery_bars"] is None and s["recovery_days"] is None
        assert s["vol_full_ann"] >= 0 and s["vol_streak_ann"] >= 0
    ps = pf["streak"]
    assert 0 <= ps["length_bars"] <= n
    assert ps["cum"] <= 0.0
    assert d["worst_leg"] in coins
    assert d["verdict"] == "PENDING_P03_FAIL"
    assert d["decision"] == "NO_ADOPTION_KEEP_EQUAL"
    assert "PENDING" in d["conclusion"] and "live untouched" in d["conclusion"]


def test_iter_h17_streak_helpers():
    m = _mod()
    assert m.losing_runs([0.01, -0.01, -0.02, 0.0, -0.05]) == [(1, 2, -0.03), (4, 4, -0.05)]
    assert m.losing_runs([0.0, 0.0]) == []
    assert m.losing_runs([-0.01, -0.02]) == [(0, 1, -0.03)]
    # zero breaks a run; longest tie-break = more negative cum
    assert m.longest_losing_run([-0.01, 0.0, -0.02], None)["length_bars"] == 1
    s = m.longest_losing_run([-0.01, -0.01, 0.0, -0.02, -0.02], None)
    assert (s["start"], s["end"], s["length_bars"]) == (3, 4, 2)
    assert s["cum"] < 0.0 and s["n_runs"] == 2
    # recovery: peak before run floored at 0
    rec, rb = m.recovery_after([0.05, -0.02, -0.02, 0.10], 1, 2)
    assert rec is True and rb == 1
    rec2, rb2 = m.recovery_after([0.05, -0.02, -0.02, 0.01], 1, 2)
    assert rec2 is False and rb2 is None
    # empty-net / no-loss edges
    e = m.longest_losing_run([], None)
    assert e["length_bars"] == 0 and e["recovered"] is True
    e2 = m.longest_losing_run([0.01, 0.0, 0.02], None)
    assert e2["length_bars"] == 0 and e2["cum"] == 0.0
    # vol + corr helpers
    assert m.ann_vol([], 8760.0) == 0.0
    assert m.ann_vol([0.0, 0.0, 0.0], 8760.0) == 0.0
    assert abs(m.pearson([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) - 1.0) < 1e-9
    assert m.pearson([1.0], [2.0]) == 0.0
    assert abs(m.longest_losing_run([-0.01, -0.02, 0.05], [10, 20, 30])["start_ts"] - 10) < 1e-9


def test_iter_h17_vol_contrast():
    d = _load()
    vc = d["vol_contrast"]
    cfg = d["config"]
    coins = cfg["coins"]
    assert set(vc["per_coin"]) == set(coins)
    for c in coins:
        v = vc["per_coin"][c]
        s = d["per_coin"][c]["streak"]
        assert abs(v["vol_full_ann"] - s["vol_full_ann"]) < 1e-9
        assert abs(v["vol_streak_ann"] - s["vol_streak_ann"]) < 1e-9
        assert abs(v["vol_ratio"] - s["vol_ratio"]) < 1e-9
        assert v["max_len_bars"] == s["length_bars"]
        assert v["vol_ratio"] >= 0.0
    assert vc["portfolio"]["max_len_bars"] == d["portfolio"]["streak"]["length_bars"]
    assert -1.0 <= vc["cross_coin_corr_len_vs_vol"] <= 1.0
    assert isinstance(vc["rule"], str) and len(vc["rule"]) > 0


def test_iter_h17_no_broker():
    src = pathlib.Path("research/run_iter_h17_streak.py").read_text().lower()
    for bad in ("place_order", "submit_order", "api_key", "make_broker",
                "y1b_live", "market_open"):
        assert bad not in src, bad


def test_iter_h17_smoke_runs_offline(tmp_path):
    out = tmp_path / "iter_H17_streak.json"
    lg = tmp_path / "iter_h17_streak.log"
    env = dict(os.environ, ITER_H17_SMOKE="1", ITER_H17_OUT=str(out),
               ITER_H17_LOG=str(lg))
    r = subprocess.run([sys.executable, "research/run_iter_h17_streak.py"],
                       capture_output=True, text=True, cwd=".", env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    dd = json.loads(out.read_text())
    assert dd["status"] == "final"
    assert dd["config"]["smoke"] is True
    assert dd["config"]["coins"] == ["ETC", "TRX"]
    assert set(dd["per_coin"]) == {"ETC", "TRX"}
    assert STREAK_KEYS <= set(dd["portfolio"]["streak"])
    assert dd["verdict"] == "PENDING_P03_FAIL"
