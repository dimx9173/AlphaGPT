"""H75 1h CMF sign filter (Top5) tests: long CMF(24)>0 / short CMF(24)<0 gate on legs."""
import json
import math
import pathlib
import subprocess
import sys

import pytest

OUT = pathlib.Path("results/iter_H75_cmf.json")
COINS = ["ETC", "TRX", "ATOM", "APT", "KAS"]


def _load():
    assert OUT.exists(), "results/iter_H75_cmf.json missing; run research/run_iter_h75_cmf.py"
    return json.loads(OUT.read_text())


def test_iter_h75_lock():
    from strategy_manager.config import LEV, FORMULA, LOCKED_ETC, LOCKED_TRX, LOCKED_ATOM, LOCKED_APT, LOCKED_KAS
    assert LEV == 2.0
    assert list(FORMULA) == [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10]
    d = _load()
    assert d["config"]["lev"] == 2.0
    assert d["config"]["formula"] == [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10]
    assert d["config"]["grid"] == "1h"
    assert d["config"]["bpy"] == 8760.0
    assert d["config"]["scale"] == 4
    assert d["config"]["cmf"]["window"] == 24
    assert d["config"]["cmf"]["long_gate"] == "CMF>0"
    assert d["config"]["cmf"]["short_gate"] == "CMF<0"
    assert d["config"]["cmf"]["threshold"] == 0.0
    assert d["config"]["cmf"]["warmup_bars"] == 24
    for c, spec in (("ETC", LOCKED_ETC), ("TRX", LOCKED_TRX), ("ATOM", LOCKED_ATOM), ("APT", LOCKED_APT), ("KAS", LOCKED_KAS)):
        for k in ("lth", "sth", "cd", "sl", "ts", "vt", "vw", "q"):
            assert d["config"]["basket"][c][k] == spec[k], (c, k)
    assert d["verdict"] == "PENDING"
    assert d["decision"] == "NO_ADOPTION"
    assert d["status"] == "COMPLETE"


def test_iter_h75_schema():
    d = _load()
    assert set(d["coins"]) == set(COINS)
    for c in COINS:
        e = d["coins"][c]
        assert set(("FULL", "LONG", "SHORT", "BASE_FULL", "cmf")) <= set(e), c
        for k in ("FULL", "LONG", "SHORT", "BASE_FULL"):
            for f in ("sharpe", "cum", "ann", "mdd", "n", "turnover"):
                assert f in e[k], (c, k, f)
            assert e[k]["n"] == d["config"]["grid_bars"], (c, k)
        for k in ("LONG", "SHORT"):
            for f in ("pnl_share", "trades", "active_bars"):
                assert f in e[k], (c, k, f)
            assert e[k]["trades"] >= 0 and e[k]["active_bars"] >= 0, (c, k)
        r = e["cmf"]
        for f in ("mean", "min", "max", "n_long_open", "n_short_open", "frac_long_open", "frac_short_open"):
            assert f in r, (c, f)
        assert -1.0 <= r["min"] <= r["max"] <= 1.0, (c, r)
        assert r["n_long_open"] + r["n_short_open"] <= d["config"]["grid_bars"], (c, r)
        assert r["n_long_open"] > 0 and r["n_short_open"] > 0, (c, r)
    b = d["basket"]
    assert set(("FULL", "LONG", "SHORT", "coin_pnl_share")) <= set(b)
    assert set(b["coin_pnl_share"]) == set(COINS)
    assert set(d["baseline"]) >= set(("basket_FULL", "coin_FULL"))
    assert set(d["baseline"]["coin_FULL"]) == set(COINS)


def test_iter_h75_additivity():
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


def test_iter_h75_cmf_math():
    sys.path.insert(0, ".")
    from research.run_iter_h75_cmf import chaikin_cmf, cmf_gate_masks
    # flat bars (high==low): multiplier 0 -> CMF 0
    flat = [(100.0, 100.0, 100.0, 100.0, 1000.0)] * 40
    r = chaikin_cmf(flat, 24)
    assert len(r) == 40
    assert all(v == 0.0 for v in r)
    assert all(v == 0.0 for v in r[:24])
    # steady buying pressure: close == high every bar -> CMF +1 after warmup
    buy = [(99.0, 101.0, 99.0, 101.0, 100.0)] * 40
    rb = chaikin_cmf(buy, 24)
    assert all(v == 0.0 for v in rb[:24])
    assert abs(rb[-1] - 1.0) < 1e-9
    # steady selling pressure: close == low every bar -> CMF -1 after warmup
    sell = [(99.0, 101.0, 99.0, 99.0, 100.0)] * 40
    rs = chaikin_cmf(sell, 24)
    assert all(v == 0.0 for v in rs[:24])
    assert abs(rs[-1] - (-1.0)) < 1e-9
    # zero-volume window falls back to 0.0
    zb = [(99.0, 101.0, 99.0, 101.0, 0.0)] * 40
    rz = chaikin_cmf(zb, 24)
    assert all(v == 0.0 for v in rz)
    assert math.isnan(sum(rb)) is False
    lok, sok = cmf_gate_masks([0.5, -0.5, 0.0])
    assert lok == [1.0, 0.0, 0.0] and sok == [0.0, 1.0, 0.0]  # tie blocks both


def test_iter_h75_scale_and_nobroker():
    src = pathlib.Path("research/run_iter_h75_cmf.py").read_text()
    assert "BPY = 8760.0" in src
    assert "SCALE = 4" in src
    assert "CMF_WINDOW = 24" in src
    assert "data/data_1y/1h/" in src
    for bad in ("place_order", "submit_order", "api_key", "API_KEY",
                "make_broker", "Y1B_LIVE"):
        assert bad.lower() not in src.lower(), bad
    d = _load()
    assert d["config"]["grid_bars"] > 8000


def test_iter_h75_smoke_runs_offline():
    import tempfile, os as _os
    with tempfile.TemporaryDirectory() as td:
        out = _os.path.join(td, "h75.json")
        lg = _os.path.join(td, "h75.log")
        env = dict(_os.environ, ITER_H75_SMOKE="1", ITER_H75_OUT=out, ITER_H75_LOG=lg)
        r = subprocess.run([sys.executable, "research/run_iter_h75_cmf.py"],
                           capture_output=True, text=True, cwd=".", env=env)
        assert r.returncode == 0, r.stderr[-2000:]
        dd = json.loads(pathlib.Path(out).read_text())
        assert dd["config"]["smoke"] is True
        assert dd["config"]["coins"] == ["ETC", "TRX"]
        assert dd["config"]["grid_bars"] == 3000
        assert set(dd["coins"]) == {"ETC", "TRX"}
        assert dd["verdict"] == "PENDING"
        assert dd["decision"] == "NO_ADOPTION"
        assert dd["status"] == "COMPLETE"
