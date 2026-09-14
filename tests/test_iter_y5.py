"""Y5 15m coin scan (28 coins): L0/L1 funnel, PENDING (no coins added)."""
import json
import os
import pathlib
import subprocess
import sys
import tempfile

SCAN = pathlib.Path(os.getenv("ITER_Y5_OUT", "results/iter_Y5_scan.json"))
UNIVERSE = ["ADA", "APT", "ARB", "ATOM", "AVAX", "BCH", "BNB", "BTC",
            "DOGE", "DOT", "ETC", "ETH", "HBAR", "ICP", "KAS", "LINK",
            "LTC", "NEAR", "PEPE", "POL", "RENDER", "SHIB", "SOL", "SUI",
            "TRX", "UNI", "XLM", "XRP"]
FOCUS6 = ["BTC", "ETH", "SOL", "BNB", "LINK", "LTC"]
FULL_SEG = {"sharpe", "ann", "mdd", "cum", "final_x", "n", "turnover"}


def _load():
    assert SCAN.exists(), "results/iter_Y5_scan.json missing; run research/run_iter_y5_scan.py"
    return json.loads(SCAN.read_text())


def test_iter_y5_lock_and_grid():
    from strategy_manager.config import LEV
    assert LEV == 2.0
    d = _load()
    cfg = d["config"]
    assert cfg["formula"] == [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10]
    assert cfg["grid"] == "15m"
    assert cfg["bpy"] == 35040.0
    assert cfg["scale"] == 16
    assert cfg["lev"] == 2.0
    assert cfg["fund"] == 0.0005
    assert cfg["fee"] == 0.0004
    assert cfg["fee2x"] == 0.0008
    assert cfg["l0_min_bars"] == 17280
    assert len(cfg["templates"]) == 3
    assert sorted(t["cd"] for t in cfg["templates"]) == [6, 12, 18]
    assert "data/data_1y/15m" in cfg["data"]


def test_iter_y5_universe_and_verdict():
    d = _load()
    assert d["iter"] == "Y5"
    assert d["verdict"] == "PENDING"
    assert d["decision"] == "NO_COINS_ADDED"
    assert d["offline"] is True
    assert d["universe"] == UNIVERSE
    assert set(d["universe"]) == set(UNIVERSE) and len(d["universe"]) == 28
    assert set(d["focus6"]) == set(FOCUS6)
    assert len(d["rows"]) == 28
    assert d["done"] == 28 and d["missing"] == []
    assert d["partial"] is False
    parts = set(d["addable_pending"]) | set(d["watch"]) | set(d["reject"])
    assert parts == set(UNIVERSE)
    assert "no coins added" in d["conclusion"]
    src = pathlib.Path("strategy_manager/y1b_basket.py").read_text()
    assert "Y5" not in src  # no live basket change


def test_iter_y5_gate_consistent():
    d = _load()
    for r in d["rows"]:
        assert r["L0_pass"] is True, r["coin"]
        assert r["L0"]["n"] >= 17280, r["coin"]
        assert r["L0"]["missing"] < 0.01, r["coin"]
        assert FULL_SEG <= set(r["FULL"]), r["coin"]
        assert r["FULL"]["n"] >= 17280, r["coin"]
        assert abs(r["FULL"]["final_x"] - (1.0 + r["FULL"]["cum"])) < 1e-3, r["coin"]
        if r["L1_pass"]:
            assert r["coin"] in d["addable_pending"], r["coin"]
            assert r["final_x"] > 1 and r["sharpe"] > 0 and r["fee2x"] > 0, r["coin"]
            assert r["verdict"] == "PASS_ALL_PENDING", r["coin"]
        else:
            assert r["coin"] in d["watch"] or r["coin"] in d["reject"], r["coin"]
            assert r["verdict"] in ("REJECT_L1", "REJECT_L0"), r["coin"]
        if r["coin"] in d["watch"]:
            assert (r["fee2x"] or 0) >= 0.8 or (r["sharpe"] or 0) >= 1.0, r["coin"]
        if r["coin"] in d["reject"]:
            assert not ((r["fee2x"] or 0) >= 0.8 or (r["sharpe"] or 0) >= 1.0), r["coin"]


def test_iter_y5_focus6_detail():
    d = _load()
    assert set(d["focus6"]) == set(FOCUS6)
    for c in FOCUS6:
        assert d["focus6"][c]["coin"] == c
        assert d["focus6"][c]["final_x"] == next(
            x for x in d["rows"] if x["coin"] == c)["final_x"]
        tmpls = d["focus6"][c]["templates"]
        assert len(tmpls) == 3
        assert sorted(t["template"]["cd"] for t in tmpls) == [6, 12, 18]
        for t in tmpls:
            assert FULL_SEG <= set(t["FULL"])
            assert "fee2x_sharpe" in t
        best = max(tmpls, key=lambda t: t["FULL"]["sharpe"])
        assert d["focus6"][c]["sharpe"] == best["FULL"]["sharpe"]
        assert d["focus6"][c]["best_template"]["cd"] == best["template"]["cd"]


def test_iter_y5_no_broker():
    src = pathlib.Path("research/run_iter_y5_scan.py").read_text()
    for bad in ("place_order", "submit_order", "api_key", "API_KEY",
                "make_broker", "Y1B_LIVE"):
        assert bad.lower() not in src.lower(), bad
    assert "data/data_1y/15m" in src


def test_iter_y5_smoke_runs_offline():
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "iter_Y5_scan.json")
        log = os.path.join(td, "iter_y5_scan.log")
        env = dict(os.environ, ITER_Y5_SMOKE="1", ITER_Y5_OUT=out, ITER_Y5_LOG=log)
        r = subprocess.run([sys.executable, "research/run_iter_y5_scan.py"],
                           capture_output=True, text=True, cwd=".", env=env)
        assert r.returncode == 0, r.stderr[-2000:]
        d = json.loads(pathlib.Path(out).read_text())
        assert d["config"]["smoke"] is True
        assert d["universe"] == ["BTC", "ETH"]
        assert d["done"] == 2
        assert d["partial"] is False
        assert d["verdict"] == "PENDING"
