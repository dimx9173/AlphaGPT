"""Y9 15m rolling-960 correlation tests: Top5 leg-net pairwise corr. Recompute by script."""
import json
import pathlib

OUT = pathlib.Path("results/iter_Y9_corr.json")
COINS = ["ETC", "TRX", "ATOM", "APT", "KAS"]
BASE_SPECS = {
    "ETC": dict(lth=0.88, sth=0.12, cd=18, sl=None, ts=24, vt=None, vw=12, q=0.3),
    "TRX": dict(lth=0.85, sth=0.12, cd=6, sl=0.05, ts=24, vt=None, vw=12, q=0.3),
    "ATOM": dict(lth=0.85, sth=0.15, cd=6, sl=0.05, ts=24, vt=None, vw=12, q=0.3),
    "APT": dict(lth=0.88, sth=0.12, cd=18, sl=None, ts=24, vt=None, vw=12, q=0.3),
    "KAS": dict(lth=0.88, sth=0.12, cd=6, sl=None, ts=24, vt=None, vw=12, q=0.3),
}

def _load():
    assert OUT.exists(), "results/iter_Y9_corr.json missing; run research/run_iter_y9_corr.py"
    return json.loads(OUT.read_text())

def test_iter_y9_locks():
    from strategy_manager.config import FORMULA, LEV, FUND
    assert FORMULA == [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10]
    assert LEV == 2.0
    assert FUND == 0.0005
    d = _load()
    assert d["config"]["formula"] == FORMULA
    assert d["config"]["lev"] == 2.0
    assert d["config"]["fund"] == 0.0005
    assert d["config"]["venue"] == "aster"

def test_iter_y9_engine_mirror():
    d = _load()
    c = d["config"]
    assert c["grid"] == "15m"
    assert c["grid_bars"] == 35040
    assert c["bpy"] == 35040.0
    assert c["window"] == 960
    assert c["scale_15m"] == 16
    assert set(c["weights"]) == set(COINS)
    for coin in COINS:
        assert c["weights"][coin] == 0.2
    for coin in COINS:
        s15 = c["specs_15m"][coin]
        b = BASE_SPECS[coin]
        assert s15["lth"] == b["lth"] and s15["sth"] == b["sth"]
        assert s15["cd"] == b["cd"] * 16 and s15["ts"] == b["ts"] * 16 and s15["vw"] == b["vw"] * 16
        assert s15["sl"] == b["sl"] and s15["vt"] == b["vt"] and s15["q"] == b["q"]
    assert d["engine_done"] is True
    assert set(d["legs"]) == set(COINS)
    for coin in COINS:
        assert d["legs"][coin]["n"] == 35040

def test_iter_y9_schema():
    d = _load()
    s = d["stats"]
    assert s["n_pairs"] == 10
    assert set(s["pairs"]) == {
        "ETC-TRX", "ETC-ATOM", "ETC-APT", "ETC-KAS", "TRX-ATOM",
        "TRX-APT", "TRX-KAS", "ATOM-APT", "ATOM-KAS", "APT-KAS",
    }
    for k, v in s["pairs"].items():
        for f in ("max", "mean", "q50", "q95", "n_win"):
            assert f in v, (k, f)
        assert v["n_win"] == 35040 - 960 + 1
        assert -1.0 <= v["mean"] <= 1.0 and -1.0 <= v["q50"] <= 1.0
    for f in ("avg_max", "share_max_gt_0.6", "share_mean_gt_0.6",
              "share_max_gt_0.7", "share_max_gt_0.85"):
        assert f in s, f
    assert d["verdict"] == "PENDING"
    assert d["decision"] == "KEEP_NO_LIVE_CHANGE"
    assert d["diversification"]["verdict"] in ("DIVERSIFIED", "WATCH", "CONCENTRATED")
    assert d["brake"]["enabled"] is False
    assert d["brake"]["env"] == "Y1B_CORR_BRAKE"

def test_iter_y9_stats_consistency():
    d = _load()
    s = d["stats"]
    mx = [s["pairs"][k]["max"] for k in s["pairs"]]
    mn = [s["pairs"][k]["mean"] for k in s["pairs"]]
    assert abs(s["avg_max"] - sum(mx) / len(mx)) < 1e-3
    assert abs(s["share_max_gt_0.6"] - sum(1 for v in mx if v > 0.6) / len(mx)) < 1e-3
    assert abs(s["share_mean_gt_0.6"] - sum(1 for v in mn if v > 0.6) / len(mn)) < 1e-3
    assert abs(s["share_max_gt_0.7"] - sum(1 for v in mx if v > 0.7) / len(mx)) < 1e-3
    assert abs(s["share_max_gt_0.85"] - sum(1 for v in mx if v > 0.85) / len(mx)) < 1e-3

def test_iter_y9_brake_rule():
    d = _load()
    s = d["stats"]
    b = d["brake"]
    assert b["halve_triggers"] == sorted(k for k, v in s["pairs"].items() if v["max"] > 0.7)
    assert b["quarter_triggers"] == sorted(k for k, v in s["pairs"].items() if v["max"] > 0.85)
    if s["share_max_gt_0.85"] > 0:
        assert d["diversification"]["verdict"] == "CONCENTRATED"
    elif s["share_max_gt_0.7"] > 0:
        assert d["diversification"]["verdict"] == "WATCH"
    elif s["avg_max"] < 0.6:
        assert d["diversification"]["verdict"] == "DIVERSIFIED"
