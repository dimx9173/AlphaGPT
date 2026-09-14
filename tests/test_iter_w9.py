"""W9 session split (15m native, Top5): Asia/EU/US per-coin sharpe/trades + FULL session-only arms."""
import json
import pathlib
import subprocess
import sys

OUT = pathlib.Path("results/iter_W9_session.json")
COINS = ["ETC", "TRX", "ATOM", "APT", "KAS"]
SESS = ["asia", "eu", "us"]


def _load():
    assert OUT.exists(), "results/iter_W9_session.json missing; run research/run_iter_w9_session.py"
    return json.loads(OUT.read_text())


def test_iter_w9_session_locks():
    from strategy_manager.config import LEV, FORMULA
    assert LEV == 2.0
    assert list(FORMULA) == [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10]


def test_iter_w9_session_schema():
    d = _load()
    assert d["config"]["grid"] == "15m"
    assert d["config"]["grid_bars"] == 35040
    assert d["config"]["bpy"] == 35040.0
    assert d["config"]["scale"] == 16
    assert set(d["config"]["weights"]) == set(COINS)
    assert d["config"]["sessions"] == {"asia": [0, 8], "eu": [8, 16], "us": [16, 24]}
    assert set(d["per_coin"]) == set(COINS)
    for c in COINS:
        for k in ("FULL", "asia", "eu", "us"):
            for f in ("sharpe", "mdd", "cum", "final_x", "ann", "n", "turnover", "trades"):
                assert f in d["per_coin"][c][k], (c, k, f)
        assert d["per_coin"][c]["FULL"]["n"] == 35040
        ssum = sum(d["per_coin"][c][s]["n"] for s in SESS)
        assert ssum == 35040, (c, ssum)
    for s in SESS:
        for k in ("FULL_calendar", "subset"):
            for f in ("sharpe", "mdd", "cum", "final_x", "ann", "n", "turnover", "trades"):
                assert f in d["arms"][s][k], (s, k, f)
    assert "basket_FULL" in d["arms"]
    assert d["verdict"] == "PENDING"
    assert d["decision"] == "KEEP_FULL_NO_SESSION_FILTER"
    assert d.get("partial") is False
    assert d["session_counts"] == {"asia": 11680, "eu": 11680, "us": 11680}


def test_iter_w9_session_trades_partition():
    d = _load()
    for c in COINS:
        tot = d["per_coin"][c]["FULL"]["trades"]
        psum = sum(d["per_coin"][c][s]["trades"] for s in SESS)
        assert psum == tot, (c, psum, tot)
    for s in SESS:
        expect = sum(d["per_coin"][c][s]["trades"] for c in COINS)
        assert d["arms"][s]["subset"]["trades"] == expect, (s,)
        assert d["arms"][s]["FULL_calendar"]["trades"] == expect, (s,)
        assert d["arms"][s]["FULL_calendar"]["cum"] == d["arms"][s]["subset"]["cum"], (s,)
    btot = sum(d["per_coin"][c]["FULL"]["trades"] for c in COINS)
    assert d["arms"]["basket_FULL"]["trades"] == btot
    bcum = round(sum(d["arms"][s]["subset"]["cum"] for s in SESS), 4)
    assert abs(bcum - d["arms"]["basket_FULL"]["cum"]) < 1e-3, (bcum,)


def test_iter_w9_session_compare():
    d = _load()
    c = d["compare"]
    assert set(c["best_session_by_coin"]) == set(COINS)
    for coin, s in c["best_session_by_coin"].items():
        assert s in SESS, (coin, s)
        sh = [d["per_coin"][coin][x]["sharpe"] for x in SESS]
        assert d["per_coin"][coin][s]["sharpe"] == max(sh), (coin,)
    assert set(c["arm_subset_sharpe"]) == set(SESS)
    for s in SESS:
        assert c["arm_subset_sharpe"][s] == d["arms"][s]["subset"]["sharpe"]
        assert c["arm_calendar_sharpe"][s] == d["arms"][s]["FULL_calendar"]["sharpe"]


def test_iter_w9_session_script_runs_offline():
    r = subprocess.run([sys.executable, "research/run_iter_w9_session.py"], capture_output=True, text=True, cwd=".")
    assert r.returncode == 0, r.stderr[-2000:]
    d = _load()
    assert set(d["per_coin"]) == set(COINS)
    assert set(d["arms"]) >= {"asia", "eu", "us", "basket_FULL"}
