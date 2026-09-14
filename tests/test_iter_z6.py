"""Z6 long/short attribution (15m native) tests: Top5 per-coin LONG vs SHORT FULL + bull/bear."""
import json
import pathlib
import subprocess
import sys

import pytest

OUT = pathlib.Path("results/iter_Z6_ls.json")
COINS = ["ETC", "TRX", "ATOM", "APT", "KAS"]

def _load():
    assert OUT.exists(), "results/iter_Z6_ls.json missing; run research/run_iter_z6_ls.py"
    return json.loads(OUT.read_text())

def test_iter_z6_lock():
    from strategy_manager.config import LEV, FORMULA, LOCKED_ETC, LOCKED_TRX, LOCKED_ATOM, LOCKED_APT, LOCKED_KAS
    assert LEV == 2.0
    assert list(FORMULA) == [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10]
    d = _load()
    assert d["config"]["lev"] == 2.0
    assert d["config"]["formula"] == [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10]
    assert d["config"]["grid"] == "15m"
    assert d["config"]["grid_bars"] == 35040
    assert d["config"]["bpy"] == 35040.0
    assert d["config"]["scale"] == 16
    assert d["config"]["weights"] == {c: 0.2 for c in COINS}
    for c, spec in (("ETC", LOCKED_ETC), ("TRX", LOCKED_TRX), ("ATOM", LOCKED_ATOM), ("APT", LOCKED_APT), ("KAS", LOCKED_KAS)):
        for k in ("lth", "sth", "cd", "sl", "ts", "vt", "vw", "q"):
            assert d["config"]["basket"][c][k] == spec[k], (c, k)
    assert d["verdict"] == "PENDING"
    assert d["decision"] == "NO_ADOPTION"

def test_iter_z6_schema():
    d = _load()
    assert set(d["coins"]) == set(COINS)
    for c in COINS:
        e = d["coins"][c]
        assert set(("FULL", "LONG", "SHORT", "bull", "bear")) <= set(e), c
        for k in ("FULL", "LONG", "SHORT"):
            for f in ("sharpe", "cum", "ann", "mdd", "n", "turnover"):
                assert f in e[k], (c, k, f)
            assert e[k]["n"] == d["config"]["grid_bars"], (c, k)
        for k in ("LONG", "SHORT"):
            for f in ("pnl_share", "trades", "active_bars"):
                assert f in e[k], (c, k, f)
            assert e[k]["trades"] >= 0 and e[k]["active_bars"] >= 0, (c, k)
        for name in ("bull", "bear"):
            assert set(("FULL", "LONG", "SHORT")) <= set(e[name]), (c, name)
            for k in ("FULL", "LONG", "SHORT"):
                for f in ("sharpe", "cum", "ann", "n", "cum_share"):
                    assert f in e[name][k], (c, name, k, f)
    b = d["basket"]
    assert set(("FULL", "LONG", "SHORT", "bull", "bear", "coin_pnl_share")) <= set(b)
    assert set(b["coin_pnl_share"]) == set(COINS)
    r = d["config"]["regime"]
    assert r["rule"] == "BTC 4h close>=MA200" and r["ma"] == 200
    assert r["bull_n"] + r["bear_n"] == d["config"]["grid_bars"]

def test_iter_z6_additivity():
    d = _load()
    for c in COINS:
        f = d["coins"][c]["FULL"]["cum"]
        l = d["coins"][c]["LONG"]["cum"]
        s = d["coins"][c]["SHORT"]["cum"]
        assert abs(f - (l + s)) < 1e-3, (c, f, l, s)
        assert abs(d["coins"][c]["LONG"]["pnl_share"] + d["coins"][c]["SHORT"]["pnl_share"] - 1.0) < 1e-3, c
        for k in ("FULL", "LONG", "SHORT"):
            tot = d["coins"][c][k]["cum"] if k == "FULL" else d["coins"][c][k]["cum"]
            bb = d["coins"][c]["bull"][k]["cum"] + d["coins"][c]["bear"][k]["cum"]
            assert abs(tot - bb) < 1e-3, (c, k, tot, bb)
    bf = d["basket"]["FULL"]["cum"]
    assert abs(bf - (d["basket"]["LONG"]["cum"] + d["basket"]["SHORT"]["cum"])) < 1e-3
    assert abs(d["basket"]["LONG"]["pnl_share"] + d["basket"]["SHORT"]["pnl_share"] - 1.0) < 1e-3
    assert abs(sum(d["basket"]["coin_pnl_share"].values()) - 1.0) < 1e-3

def test_iter_z6_basket_mirrors_y7():
    y7 = pathlib.Path("results/iter_Y7_lev.json")
    if not y7.exists():
        pytest.skip("results/iter_Y7_lev.json absent")
    ref = json.loads(y7.read_text())["rows"]["2"]["FULL"]
    d = _load()
    got = d["basket"]["FULL"]
    assert got["n"] == ref["n"] == 35040
    assert abs(got["sharpe"] - ref["sharpe"]) < 1e-6, (got, ref)
    assert abs(got["cum"] - ref["cum"]) < 1e-6
    assert abs(got["turnover"] - ref["turnover"]) < 1e-9

def test_iter_z6_script_runs_offline():
    r = subprocess.run([sys.executable, "research/run_iter_z6_ls.py"], capture_output=True, text=True, cwd=".")
    assert r.returncode == 0, r.stderr[-2000:]
    d = _load()
    assert set(d["coins"]) == set(COINS)
    assert d["verdict"] == "PENDING" and d["decision"] == "NO_ADOPTION"
