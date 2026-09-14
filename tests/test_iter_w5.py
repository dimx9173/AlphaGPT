"""W5 15m turnover audit (Top5) tests: entries/flips/holds + scatter + min-hold.

Full W5 is Top5 15m-native (~35040 bars) with min-hold {0,2,4}; smoke =
coins {ETC,TRX}, first 3000 bars, min-hold {0,2} via ITER_W5_SMOKE=1.
This test never re-runs the full audit: schema checks accept either
artifact, and the live subprocess check runs smoke mode only. Conclusion
must stay PENDING: diagnostic only, no adoption, live untouched.
"""
import json
import os
import pathlib
import subprocess
import sys

OUT = pathlib.Path(os.getenv("ITER_W5_OUT", "results/iter_W5_turn.json"))
SEG = {"sharpe", "ann", "mdd", "cum", "n", "turnover"}
HOLD_KEYS = {"count", "mean", "median", "p90", "max", "min", "buckets"}
BUCKETS = ["1", "2_4", "5_8", "9_16", "17_32", "33_64", "65p"]
COINS5 = {"ETC", "TRX", "ATOM", "APT", "KAS"}
FULL_MH = [0, 2, 4]
BASE_CD = {"ETC": 18, "TRX": 6, "ATOM": 6, "APT": 18, "KAS": 6}


def _load():
    assert OUT.exists(), "results/iter_W5_turn.json missing; run research/run_iter_w5_turn.py"
    return json.loads(OUT.read_text())


def _expected(d):
    smoke = d["config"]["smoke"]
    if smoke:
        return ["ETC", "TRX"], [0, 2]
    return ["ETC", "TRX", "ATOM", "APT", "KAS"], FULL_MH


def test_iter_w5_lock():
    from strategy_manager.config import LEV, FORMULA, LOCKED_ETC, LOCKED_TRX, LOCKED_ATOM, LOCKED_APT, LOCKED_KAS
    assert LEV == 2.0
    assert FORMULA == [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10]
    assert (LOCKED_ETC["lth"], LOCKED_ETC["sth"], LOCKED_ETC["cd"], LOCKED_ETC["sl"], LOCKED_ETC["ts"]) == (0.88, 0.12, 18, None, 24)
    assert (LOCKED_TRX["lth"], LOCKED_TRX["sth"], LOCKED_TRX["cd"], LOCKED_TRX["sl"], LOCKED_TRX["ts"]) == (0.85, 0.12, 6, 0.05, 24)
    assert (LOCKED_ATOM["lth"], LOCKED_ATOM["sth"], LOCKED_ATOM["cd"], LOCKED_ATOM["sl"], LOCKED_ATOM["ts"]) == (0.85, 0.15, 6, 0.05, 24)
    assert (LOCKED_APT["lth"], LOCKED_APT["sth"], LOCKED_APT["cd"], LOCKED_APT["sl"], LOCKED_APT["ts"]) == (0.88, 0.12, 18, None, 24)
    assert (LOCKED_KAS["lth"], LOCKED_KAS["sth"], LOCKED_KAS["cd"], LOCKED_KAS["sl"], LOCKED_KAS["ts"]) == (0.88, 0.12, 6, None, 24)


def test_iter_w5_schema():
    d = _load()
    cfg = d["config"]
    coins, mh = _expected(d)
    assert cfg["smoke"] in (True, False)
    assert cfg["bpy"] == 35040.0
    assert cfg["scale"] == 16
    assert cfg["grid"] == "15m"
    assert cfg["grid_bars"] > 2000
    assert cfg["minhold_grid"] == mh
    assert d["partial"] is False
    if not cfg["smoke"]:
        assert cfg["coins"] == coins
        assert cfg["grid_bars"] == 35040
        assert set(cfg["weights"]) == COINS5
        for c in COINS5:
            assert cfg["basket"][c]["cd"] == BASE_CD[c], c
            assert cfg["basket"][c]["q"] == 0.3, c
    n = cfg["grid_bars"]
    h2a = cfg["h2_start"]
    assert SEG <= set(d["base_FULL"])
    assert d["base_FULL"]["n"] == n
    assert d["base_FULL"]["turnover"] >= 0
    assert d["base_FULL"]["trades"] >= 0
    assert SEG <= set(d["base_H2"])
    assert d["base_H2"]["n"] == n - h2a
    assert set(d["base_per_coin"]) == set(coins)
    for c, pc in d["base_per_coin"].items():
        assert SEG <= set(pc["FULL"]), c
        assert SEG <= set(pc["H2"]), c
        assert pc["FULL"]["n"] == n, c
        assert pc["H2"]["n"] == n - h2a, c
        assert pc["FULL"]["turnover"] >= 0, c
        assert pc["entries"] >= 0 and pc["flips"] >= 0 and pc["exits"] >= 0, c
        assert pc["trades"] == pc["entries"] + pc["flips"], c
        assert isinstance(pc["exits"], int) and pc["exits"] >= 0, c
        assert HOLD_KEYS <= set(pc["holds"]), c
        assert pc["holds"]["count"] >= 0, c
        assert pc["holds"]["max"] >= pc["holds"]["min"], c
        if pc["holds"]["count"]:
            assert pc["holds"]["max"] >= 1, c
            assert pc["holds"]["mean"] >= 1.0, c
        assert list(pc["holds"]["buckets"]) == BUCKETS, c
        assert sum(pc["holds"]["buckets"].values()) == pc["holds"]["count"], c
    assert [p["coin"] for p in d["scatter"]] == coins
    for p in d["scatter"]:
        pc = d["base_per_coin"][p["coin"]]
        assert p["turnover"] == pc["FULL"]["turnover"]
        assert p["entries"] == pc["entries"]
        assert p["flips"] == pc["flips"]
        assert p["trades"] == pc["trades"] == p["entries"] + p["flips"]
    assert [r["min_hold"] for r in d["minhold_rows"]] == mh
    for r in d["minhold_rows"]:
        assert SEG <= set(r["FULL"]), r
        assert r["FULL"]["n"] == n
        assert r["FULL"]["turnover"] >= 0
        assert r["FULL"]["trades"] >= 0
        assert isinstance(r["H2_sharpe"], float)
        assert isinstance(r["H2_trades"], int) and r["H2_trades"] >= 0
        assert r["H2_trades"] <= r["FULL"]["trades"]
        assert abs(r["d_sharpe_vs_base"] - round(r["FULL"]["sharpe"] - d["base_FULL"]["sharpe"], 3)) < 1e-9
        assert abs(r["d_turnover_vs_base"] - round(r["FULL"]["turnover"] - d["base_FULL"]["turnover"], 6)) < 1e-9
        base_to = d["base_FULL"]["turnover"]
        exp_cut = round((base_to - r["FULL"]["turnover"]) / base_to, 4) if base_to > 0 else 0.0
        assert abs(r["turnover_cut_vs_base"] - exp_cut) < 1e-9
    mh0 = next(r for r in d["minhold_rows"] if r["min_hold"] == 0)
    assert mh0["FULL"]["turnover"] >= d["base_FULL"]["turnover"] - 1e-12
    assert d["verdict"] == "PENDING"
    assert d["decision"] == "PENDING"
    assert "PENDING" in d["conclusion"] and "no live change" in d["conclusion"]


def test_iter_w5_hold_semantics():
    import importlib.util
    spec = importlib.util.spec_from_file_location("w5mod", "research/run_iter_w5_turn.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["w5mod"] = m
    try:
        spec.loader.exec_module(m)
    except SystemExit:
        pass
    assert m.hold_runs([0, 0, 0]) == []
    assert m.hold_runs([1, 1, 1, 0]) == [3]
    assert m.hold_runs([1, 1, -1, -1]) == [2, 2]
    st = m.trade_stats([1, 1, 0, 1, -1, -1, 0, 0, 1])
    assert (st["entries"], st["flips"], st["exits"], st["trades"]) == (3, 1, 2, 4)
    assert st["holds"]["count"] == 4
    assert sum(st["holds"]["buckets"].values()) == 4
    assert m.bucketize([1, 3, 7, 20, 70]) == {"1": 1, "2_4": 1, "5_8": 1, "9_16": 0, "17_32": 1, "33_64": 0, "65p": 1}


def test_iter_w5_no_broker():
    src = pathlib.Path("research/run_iter_w5_turn.py").read_text().lower()
    for bad in ("place_order", "submit_order", "api_key", "make_broker", "y1b_live", "market_open"):
        assert bad not in src, bad


def test_iter_w5_smoke_runs_offline(tmp_path):
    out = tmp_path / "iter_W5_turn.json"
    lg = tmp_path / "iter_w5_turn.log"
    env = dict(os.environ, ITER_W5_SMOKE="1", ITER_W5_OUT=str(out), ITER_W5_LOG=str(lg))
    r = subprocess.run([sys.executable, "research/run_iter_w5_turn.py"], capture_output=True, text=True, cwd=".", env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    dd = json.loads(out.read_text())
    assert dd["config"]["smoke"] is True
    assert dd["config"]["minhold_grid"] == [0, 2]
    assert dd["config"]["grid_bars"] == 3000
    assert dd["partial"] is False
    assert [rr["min_hold"] for rr in dd["minhold_rows"]] == [0, 2]
    assert dd["verdict"] == "PENDING"
