"""H25 long/short attribution (1h native) tests: Top5 per-coin LONG vs SHORT FULL + bull/bear."""
import json
import os
import pathlib
import subprocess
import sys
import tempfile

OUT = pathlib.Path(os.getenv("ITER_H25_OUT", "results/iter_H25_ls.json"))
SRC = pathlib.Path("research/run_iter_h25_ls.py")
COINS = ["ETC", "TRX", "ATOM", "APT", "KAS"]

def _load(path=OUT):
    assert path.exists(), "%s missing; run research/run_iter_h25_ls.py" % path
    return json.loads(path.read_text())

def test_iter_h25_lock():
    from strategy_manager.config import LEV, FUND, FEE, FORMULA, LOCKED_ETC, LOCKED_TRX, LOCKED_ATOM, LOCKED_APT, LOCKED_KAS
    assert LEV == 2.0
    assert abs(FUND - 0.0005) < 1e-12
    assert abs(FEE - 0.0004) < 1e-12
    assert list(FORMULA) == [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10]
    d = _load()
    assert d["config"]["lev"] == 2.0
    assert d["config"]["formula"] == [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10]
    assert d["config"]["grid"] == "1h"
    assert d["config"]["grid_bars"] == 8760
    assert d["config"]["bpy"] == 8760.0
    assert d["config"]["scale"] == 4
    assert d["config"]["weights"] == {c: 0.2 for c in COINS}
    for c, spec in (("ETC", LOCKED_ETC), ("TRX", LOCKED_TRX), ("ATOM", LOCKED_ATOM), ("APT", LOCKED_APT), ("KAS", LOCKED_KAS)):
        for k in ("lth", "sth", "cd", "sl", "ts", "vt", "vw", "q"):
            assert d["config"]["basket"][c][k] == spec[k], (c, k)
    assert d["config"]["smoke"] is False
    assert d["config"]["coins"] == COINS
    assert d["verdict"] == "PENDING"
    assert d["decision"] == "NO_ADOPTION"

def test_iter_h25_schema():
    d = _load()
    assert d.get("status") == "final"
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

def test_iter_h25_additivity():
    d = _load()
    for c in COINS:
        f = d["coins"][c]["FULL"]["cum"]
        l = d["coins"][c]["LONG"]["cum"]
        s = d["coins"][c]["SHORT"]["cum"]
        assert abs(f - (l + s)) < 1e-3, (c, f, l, s)
        assert abs(d["coins"][c]["LONG"]["pnl_share"] + d["coins"][c]["SHORT"]["pnl_share"] - 1.0) < 1e-3, c
        for k in ("FULL", "LONG", "SHORT"):
            tot = d["coins"][c][k]["cum"]
            bb = d["coins"][c]["bull"][k]["cum"] + d["coins"][c]["bear"][k]["cum"]
            assert abs(tot - bb) < 1e-3, (c, k, tot, bb)
    bf = d["basket"]["FULL"]["cum"]
    assert abs(bf - (d["basket"]["LONG"]["cum"] + d["basket"]["SHORT"]["cum"])) < 1e-3
    assert abs(d["basket"]["LONG"]["pnl_share"] + d["basket"]["SHORT"]["pnl_share"] - 1.0) < 1e-3
    assert abs(sum(d["basket"]["coin_pnl_share"].values()) - 1.0) < 1e-3

def test_iter_h25_native_grid_and_no_live():
    d = _load()
    assert d["config"]["grid_bars"] == 8760
    assert "data_1y/1h" in SRC.read_text()
    src = SRC.read_text().lower()
    for bad in ("place_order", "submit_order", "api_key", "make_broker", "y1b_live"):
        assert bad not in src, bad

def test_iter_h25_script_runs_offline_smoke():
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "h25.json")
        log = os.path.join(td, "h25.log")
        env = dict(os.environ, ITER_H25_SMOKE="1", ITER_H25_OUT=out, ITER_H25_LOG=log)
        r = subprocess.run([sys.executable, "research/run_iter_h25_ls.py"], capture_output=True, text=True, cwd=".", env=env)
        assert r.returncode == 0, r.stderr[-2000:]
        dd = json.loads(open(out).read())
        assert dd["config"]["smoke"] is True
        assert dd["config"]["coins"] == ["ETC", "TRX"]
        assert dd["config"]["grid_bars"] == 3000
        assert set(dd["coins"]) == {"ETC", "TRX"}
        assert dd["verdict"] == "PENDING"
        assert dd["decision"] == "NO_ADOPTION"
        assert dd["status"] == "final"
