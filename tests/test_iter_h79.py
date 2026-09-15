"""H79 1h Supertrend filter (Top5) tests: legs gated by supertrend(12,3) direction."""
import json
import math
import pathlib
import subprocess
import sys

import pytest

OUT = pathlib.Path("results/iter_H79_st.json")
COINS = ["ETC", "TRX", "ATOM", "APT", "KAS"]


def _load():
    assert OUT.exists(), "results/iter_H79_st.json missing; run research/run_iter_h79_st.py"
    return json.loads(OUT.read_text())


def test_iter_h79_lock():
    from strategy_manager.config import LEV, FORMULA, LOCKED_ETC, LOCKED_TRX, LOCKED_ATOM, LOCKED_APT, LOCKED_KAS
    assert LEV == 2.0
    assert list(FORMULA) == [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10]
    d = _load()
    assert d["config"]["lev"] == 2.0
    assert d["config"]["formula"] == [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10]
    assert d["config"]["grid"] == "1h"
    assert d["config"]["bpy"] == 8760.0
    assert d["config"]["scale"] == 4
    assert d["config"]["st"]["period"] == 12
    assert d["config"]["st"]["mult"] == 3.0
    assert d["config"]["st"]["long_gate"] == "dir>0"
    assert d["config"]["st"]["short_gate"] == "dir<0"
    assert d["config"]["st"]["warmup_bars"] == 12
    for c, spec in (("ETC", LOCKED_ETC), ("TRX", LOCKED_TRX), ("ATOM", LOCKED_ATOM), ("APT", LOCKED_APT), ("KAS", LOCKED_KAS)):
        for k in ("lth", "sth", "cd", "sl", "ts", "vt", "vw", "q"):
            assert d["config"]["basket"][c][k] == spec[k], (c, k)
    assert d["verdict"] == "PENDING"
    assert d["decision"] == "NO_ADOPTION"
    assert d["status"] == "COMPLETE"


def test_iter_h79_schema():
    d = _load()
    assert set(d["coins"]) == set(COINS)
    for c in COINS:
        e = d["coins"][c]
        assert set(("FULL", "LONG", "SHORT", "BASE_FULL", "st")) <= set(e), c
        for k in ("FULL", "LONG", "SHORT", "BASE_FULL"):
            for f in ("sharpe", "cum", "ann", "mdd", "n", "turnover"):
                assert f in e[k], (c, k, f)
            assert e[k]["n"] == d["config"]["grid_bars"], (c, k)
        for k in ("LONG", "SHORT"):
            for f in ("pnl_share", "trades", "active_bars"):
                assert f in e[k], (c, k, f)
            assert e[k]["trades"] >= 0 and e[k]["active_bars"] >= 0, (c, k)
        s = e["st"]
        for f in ("n_up", "n_down", "frac_up", "frac_down"):
            assert f in s, (c, f)
        assert s["n_up"] + s["n_down"] <= d["config"]["grid_bars"], (c, s)
        assert s["n_up"] > 0 and s["n_down"] > 0, (c, s)
        assert 0.0 <= s["frac_up"] <= 1.0 and 0.0 <= s["frac_down"] <= 1.0, (c, s)
    b = d["basket"]
    assert set(("FULL", "LONG", "SHORT", "coin_pnl_share")) <= set(b)
    assert set(b["coin_pnl_share"]) == set(COINS)
    assert set(d["baseline"]) >= set(("basket_FULL", "coin_FULL"))
    assert set(d["baseline"]["coin_FULL"]) == set(COINS)


def test_iter_h79_additivity():
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


def test_iter_h79_st_math():
    sys.path.insert(0, ".")
    from research.run_iter_h79_st import supertrend_direction
    n = 200
    flat_h = [101.0] * n
    flat_l = [99.0] * n
    flat_c = [100.0] * n
    d = supertrend_direction(flat_h, flat_l, flat_c)
    assert d[:12] == [0] * 12
    assert set(d[12:]) <= {1, -1}
    up_h = [100.0 + i for i in range(n)]
    up_l = [99.0 + i for i in range(n)]
    up_c = [99.5 + i for i in range(n)]
    du = supertrend_direction(up_h, up_l, up_c)
    assert du[:12] == [0] * 12
    assert du[-1] == 1
    dn_h = [100.0 - i for i in range(n)]
    dn_l = [99.0 - i for i in range(n)]
    dn_c = [99.5 - i for i in range(n)]
    dd = supertrend_direction(dn_h, dn_l, dn_c)
    assert dd[:12] == [0] * 12
    assert dd[-1] == -1
    assert len(du) == n and len(dd) == n


def test_iter_h79_scale_and_nobroker():
    src = pathlib.Path("research/run_iter_h79_st.py").read_text()
    assert "BPY = 8760.0" in src
    assert "SCALE = 4" in src
    assert "data/data_1y/1h/" in src
    for bad in ("place_order", "submit_order", "api_key", "API_KEY",
                "make_broker", "Y1B_LIVE"):
        assert bad.lower() not in src.lower(), bad
    d = _load()
    assert d["config"]["grid_bars"] > 8000
