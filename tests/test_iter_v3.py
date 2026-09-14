"""V3 15m RSI filter (Top5) tests: long-only RSI(56)>50 / short-only RSI<50 gate on legs."""
import json
import math
import pathlib
import subprocess
import sys

import pytest

OUT = pathlib.Path("results/iter_V3_rsi.json")
COINS = ["ETC", "TRX", "ATOM", "APT", "KAS"]


def _load():
    assert OUT.exists(), "results/iter_V3_rsi.json missing; run research/run_iter_v3_rsi.py"
    return json.loads(OUT.read_text())


def test_iter_v3_lock():
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
    assert d["config"]["rsi"]["window"] == 56
    assert d["config"]["rsi"]["threshold"] == 50.0
    assert d["config"]["rsi"]["long_gate"] == "RSI>50"
    assert d["config"]["rsi"]["short_gate"] == "RSI<50"
    assert d["config"]["rsi"]["warmup_bars"] == 56
    for c, spec in (("ETC", LOCKED_ETC), ("TRX", LOCKED_TRX), ("ATOM", LOCKED_ATOM), ("APT", LOCKED_APT), ("KAS", LOCKED_KAS)):
        for k in ("lth", "sth", "cd", "sl", "ts", "vt", "vw", "q"):
            assert d["config"]["basket"][c][k] == spec[k], (c, k)
    assert d["verdict"] == "PENDING"
    assert d["decision"] == "NO_ADOPTION"
    assert d["status"] == "COMPLETE"


def test_iter_v3_schema():
    d = _load()
    assert set(d["coins"]) == set(COINS)
    for c in COINS:
        e = d["coins"][c]
        assert set(("FULL", "LONG", "SHORT", "BASE_FULL", "rsi")) <= set(e), c
        for k in ("FULL", "LONG", "SHORT", "BASE_FULL"):
            for f in ("sharpe", "cum", "ann", "mdd", "n", "turnover"):
                assert f in e[k], (c, k, f)
            assert e[k]["n"] == d["config"]["grid_bars"], (c, k)
        for k in ("LONG", "SHORT"):
            for f in ("pnl_share", "trades", "active_bars"):
                assert f in e[k], (c, k, f)
            assert e[k]["trades"] >= 0 and e[k]["active_bars"] >= 0, (c, k)
        r = e["rsi"]
        for f in ("mean", "min", "max", "n_long_open", "n_short_open", "frac_long_open", "frac_short_open"):
            assert f in r, (c, f)
        assert 0.0 <= r["min"] <= r["max"] <= 100.0, (c, r)
        assert r["n_long_open"] + r["n_short_open"] <= d["config"]["grid_bars"], (c, r)
        assert r["n_long_open"] > 0 and r["n_short_open"] > 0, (c, r)
    b = d["basket"]
    assert set(("FULL", "LONG", "SHORT", "coin_pnl_share")) <= set(b)
    assert set(b["coin_pnl_share"]) == set(COINS)
    assert set(d["baseline"]) >= set(("basket_FULL", "coin_FULL"))
    assert set(d["baseline"]["coin_FULL"]) == set(COINS)


def test_iter_v3_additivity():
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


def test_iter_v3_base_mirrors_z6():
    z6 = pathlib.Path("results/iter_Z6_ls.json")
    if not z6.exists():
        pytest.skip("results/iter_Z6_ls.json absent")
    ref = json.loads(z6.read_text())["basket"]["FULL"]
    d = _load()
    got = d["baseline"]["basket_FULL"]
    assert got["n"] == ref["n"] == 35040
    assert abs(got["sharpe"] - ref["sharpe"]) < 1e-6, (got, ref)
    assert abs(got["cum"] - ref["cum"]) < 1e-6


def test_iter_v3_rsi_math():
    sys.path.insert(0, ".")
    from research.run_iter_v3_rsi import wilder_rsi
    n = 200
    flat = [100.0] * n
    assert all(v == 50.0 for v in wilder_rsi(flat, 56))
    up = [100.0 + i for i in range(n)]
    r = wilder_rsi(up, 56)
    assert all(v == 50.0 for v in r[:56])
    assert r[56] == 100.0 and r[-1] == 100.0
    dn = [100.0 - i for i in range(n)]
    r2 = wilder_rsi(dn, 56)
    assert all(v == 50.0 for v in r2[:56])
    assert r2[56] == 0.0 and r2[-1] == 0.0
    assert math.isnan(sum(r)) is False


def test_iter_v3_script_runs_offline():
    r = subprocess.run([sys.executable, "research/run_iter_v3_rsi.py"], capture_output=True, text=True, cwd=".")
    assert r.returncode == 0, r.stderr[-2000:]
    d = _load()
    assert set(d["coins"]) == set(COINS)
    assert d["verdict"] == "PENDING" and d["decision"] == "NO_ADOPTION"
    assert d["status"] == "COMPLETE"
