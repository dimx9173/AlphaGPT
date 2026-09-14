"""X5 remaining-19 scan: L0/L1 funnel, PENDING (no coins added)."""
import json
import pathlib
import subprocess
import sys

SCAN = pathlib.Path("results/iter_X5_scan.json")
FOCUS6 = ["BTC", "ETH", "SOL", "BNB", "LINK", "LTC"]
REMAINING19 = ["ADA", "ARB", "AVAX", "BCH", "BNB", "BTC", "DOGE", "DOT", "ETH",
               "HBAR", "LINK", "LTC", "NEAR", "SHIB", "SOL", "SUI", "UNI",
               "XLM", "XRP"]


def _load():
    assert SCAN.exists(), "results/iter_X5_scan.json missing"
    return json.loads(SCAN.read_text())


def test_scan_schema():
    d = _load()
    assert d["iter"] == "X5"
    assert d["universe"] == REMAINING19
    assert set(d["focus6"]) == set(FOCUS6)
    assert set(d) >= {"addable_pending", "watch", "reject", "rows",
                      "conclusion", "offline"}
    assert d["offline"] is True
    assert "\u5f85\u5b9a" in d["conclusion"] or "待定" in d["conclusion"]
    parts = set(d["addable_pending"]) | set(d["watch"]) | set(d["reject"])
    assert parts == set(REMAINING19)
    assert len(d["rows"]) == 19


def test_l1_gate_consistent():
    d = _load()
    for r in d["rows"]:
        if r["L1_pass"]:
            assert r["coin"] in d["addable_pending"]
            assert r["final_x"] > 1 and r["sharpe"] > 0 and r["fee2x"] > 1
        else:
            assert r["coin"] in d["watch"] or r["coin"] in d["reject"]
        assert r["L0_pass"] is True  # all 19 have full history


def test_watch_is_near_miss():
    d = _load()
    for c in d["watch"]:
        r = next(x for x in d["rows"] if x["coin"] == c)
        assert (r["fee2x"] or 0) >= 0.8 or (r["sharpe"] or 0) >= 1.0
    for c in d["reject"]:
        r = next(x for x in d["rows"] if x["coin"] == c)
        assert not ((r["fee2x"] or 0) >= 0.8 or (r["sharpe"] or 0) >= 1.0)


def test_focus6_spot_btc_sol():
    d = _load()
    btc = d["focus6"]["BTC"]
    assert btc["verdict"] == "REJECT_L1" and btc["fee2x"] < 1
    sol = d["focus6"]["SOL"]
    assert sol["final_x"] < 1 and sol["sharpe"] < 0


def test_no_listing_change():
    src = pathlib.Path("strategy_manager/y1b_basket.py").read_text()
    for c in ("DOT", "ETH"):
        assert f'"{c}"' not in src
    d = _load()
    assert "no coins added" in d["conclusion"]
