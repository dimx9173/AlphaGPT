"""H63 1h VWAP filter v4 (Top5) tests: long above / short below 192-bar rolling VWAP."""
import json
import math
import pathlib
import subprocess
import sys

import pytest

OUT = pathlib.Path("results/iter_H63_vwap.json")
COINS = ["ETC", "TRX", "ATOM", "APT", "KAS"]


def _load():
    assert OUT.exists(), "results/iter_H63_vwap.json missing; run research/run_iter_h63_vwap.py"
    return json.loads(OUT.read_text())


def test_iter_h63_lock():
    from strategy_manager.config import LEV, FORMULA, LOCKED_ETC, LOCKED_TRX, LOCKED_ATOM, LOCKED_APT, LOCKED_KAS
    assert LEV == 2.0
    assert list(FORMULA) == [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10]
    d = _load()
    assert d["config"]["lev"] == 2.0
    assert d["config"]["formula"] == [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10]
    assert d["config"]["grid"] == "1h"
    assert d["config"]["bpy"] == 8760.0
    assert d["config"]["scale"] == 4
    assert d["config"]["vwap"]["window"] == 192
    assert d["config"]["vwap"]["typical"] == "(h+l+c)/3"
    assert d["config"]["vwap"]["long_gate"] == "close>vwap"
    assert d["config"]["vwap"]["short_gate"] == "close<vwap"
    for c, spec in (("ETC", LOCKED_ETC), ("TRX", LOCKED_TRX), ("ATOM", LOCKED_ATOM), ("APT", LOCKED_APT), ("KAS", LOCKED_KAS)):
        for k in ("lth", "sth", "cd", "sl", "ts", "vt", "vw", "q"):
            assert d["config"]["basket"][c][k] == spec[k], (c, k)
    assert d["verdict"] == "PENDING"
    assert d["decision"] == "NO_ADOPTION"
    assert d["status"] == "COMPLETE"


def test_iter_h63_schema():
    d = _load()
    assert set(d["coins"]) == set(COINS)
    for c in COINS:
        e = d["coins"][c]
        assert set(("FULL", "LONG", "SHORT", "BASE_FULL", "vwap")) <= set(e), c
        for k in ("FULL", "LONG", "SHORT", "BASE_FULL"):
            for f in ("sharpe", "cum", "ann", "mdd", "n", "turnover"):
                assert f in e[k], (c, k, f)
            assert e[k]["n"] == d["config"]["grid_bars"], (c, k)
        assert "trades" in e["FULL"], (c, e["FULL"])
        for k in ("LONG", "SHORT"):
            for f in ("pnl_share", "trades", "active_bars"):
                assert f in e[k], (c, k, f)
            assert e[k]["trades"] >= 0 and e[k]["active_bars"] >= 0, (c, k)
        assert e["FULL"]["trades"] == e["LONG"]["trades"] + e["SHORT"]["trades"], (c, e["FULL"])
        v = e["vwap"]
        for f in ("window", "mean", "min", "max", "n_long_open", "n_short_open", "frac_long_open", "frac_short_open"):
            assert f in v, (c, f)
        assert v["window"] == 192
        assert v["min"] <= v["max"], (c, v)
        assert v["n_long_open"] + v["n_short_open"] <= d["config"]["grid_bars"], (c, v)
        assert v["n_long_open"] > 0 and v["n_short_open"] > 0, (c, v)
    b = d["basket"]
    assert set(("FULL", "LONG", "SHORT", "coin_pnl_share")) <= set(b)
    assert set(b["coin_pnl_share"]) == set(COINS)
    assert "trades" in b["FULL"]
    assert b["FULL"]["trades"] == sum(d["coins"][c]["FULL"]["trades"] for c in COINS)
    assert set(d["baseline"]) >= set(("basket_FULL", "coin_FULL"))
    assert set(d["baseline"]["coin_FULL"]) == set(COINS)


def test_iter_h63_additivity():
    d = _load()
    for c in COINS:
        f = d["coins"][c]["FULL"]["cum"]
        l = d["coins"][c]["LONG"]["cum"]
        s = d["coins"][c]["SHORT"]["cum"]
        assert abs(f - (l + s)) < 1e-3, (c, f, l, s)
        assert abs(d["coins"][c]["LONG"]["pnl_share"] + d["coins"][c]["SHORT"]["pnl_share"] - 1.0) < 1e-3, c
    bf = d["basket"]["FULL"]["cum"]
    assert abs(bf - (d["basket"]["LONG"]["cum"] + d["basket"]["SHORT"]["cum"])) < 1e-3
    assert abs(d["basket"]["LONG"]["pnl_share"] + d["basket"]["SHORT"]["pnl_share"] - 1.0) < 1e-3
    assert abs(sum(d["basket"]["coin_pnl_share"].values()) - 1.0) < 1e-3


def test_iter_h63_vwap_math():
    import sys as _sys
    _sys.path.insert(0, ".")
    from research.run_iter_h63_vwap import rolling_vwap, vwap_gate_masks
    bars = [(100.0, 101.0, 99.0, 100.0, 1000.0)] * 30
    vw = rolling_vwap(bars, 48)
    assert len(vw) == 30
    assert all(abs(v - 100.0) < 1e-9 for v in vw)
    # zero-volume window falls back to close
    zb = [(100.0, 101.0, 99.0, 105.0, 0.0)] * 5
    vw2 = rolling_vwap(zb, 48)
    assert all(abs(v - 105.0) < 1e-9 for v in vw2)
    # rising prices: volume-weighted typical tracks typical
    rb = [(float(i), float(i) + 1.0, float(i) - 1.0, float(i), 100.0) for i in range(30)]
    vw3 = rolling_vwap(rb, 48)
    assert vw3[-1] < rb[-1][3]  # uptrend: close above trailing VWAP
    assert math.isnan(sum(vw3)) is False
    lok, sok = vwap_gate_masks([101.0, 99.0, 100.0], [100.0, 100.0, 100.0])
    assert lok == [1.0, 0.0, 0.0] and sok == [0.0, 1.0, 0.0]  # tie blocks both


def test_iter_h63_scale_and_nobroker():
    src = pathlib.Path("research/run_iter_h63_vwap.py").read_text()
    assert "BPY = 8760.0" in src
    assert "SCALE = 4" in src
    assert "VWAP_WIN = 192" in src
    assert "data/data_1y/1h/" in src
    for bad in ("place_order", "submit_order", "api_key", "API_KEY",
                "make_broker", "Y1B_LIVE"):
        assert bad.lower() not in src.lower(), bad
    d = _load()
    assert d["config"]["grid_bars"] > 8000
