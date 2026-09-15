"""H72 1h CCI filter (Top5) tests: entry only CCI(24)>+100 long / CCI(24)<-100 short."""
import json
import math
import pathlib
import subprocess
import sys

import pytest

OUT = pathlib.Path("results/iter_H72_cci.json")
COINS = ["ETC", "TRX", "ATOM", "APT", "KAS"]


def _load():
    assert OUT.exists(), "results/iter_H72_cci.json missing; run research/run_iter_h72_cci.py"
    return json.loads(OUT.read_text())


def test_iter_h72_lock():
    from strategy_manager.config import LEV, FORMULA, LOCKED_ETC, LOCKED_TRX, LOCKED_ATOM, LOCKED_APT, LOCKED_KAS
    assert LEV == 2.0
    assert list(FORMULA) == [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10]
    d = _load()
    assert d["config"]["lev"] == 2.0
    assert d["config"]["formula"] == [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10]
    assert d["config"]["grid"] == "1h"
    assert d["config"]["bpy"] == 8760.0
    assert d["config"]["scale"] == 4
    assert d["config"]["cci"]["window"] == 24
    assert d["config"]["cci"]["long_threshold"] == 100.0
    assert d["config"]["cci"]["short_threshold"] == -100.0
    assert d["config"]["cci"]["long_gate"] == "CCI>+100"
    assert d["config"]["cci"]["short_gate"] == "CCI<-100"
    assert d["config"]["cci"]["warmup_bars"] == 24
    for c, spec in (("ETC", LOCKED_ETC), ("TRX", LOCKED_TRX), ("ATOM", LOCKED_ATOM), ("APT", LOCKED_APT), ("KAS", LOCKED_KAS)):
        for k in ("lth", "sth", "cd", "sl", "ts", "vt", "vw", "q"):
            assert d["config"]["basket"][c][k] == spec[k], (c, k)
    assert d["verdict"] == "PENDING"
    assert d["decision"] == "NO_ADOPTION"
    assert d["status"] == "COMPLETE"


def test_iter_h72_schema():
    d = _load()
    assert set(d["coins"]) == set(COINS)
    for c in COINS:
        e = d["coins"][c]
        assert set(("FULL", "LONG", "SHORT", "BASE_FULL", "cci")) <= set(e), c
        for k in ("FULL", "LONG", "SHORT", "BASE_FULL"):
            for f in ("sharpe", "cum", "ann", "mdd", "n", "turnover"):
                assert f in e[k], (c, k, f)
            assert e[k]["n"] == d["config"]["grid_bars"], (c, k)
        for k in ("FULL", "LONG", "SHORT"):
            for f in ("trades", "active_bars"):
                assert f in e[k], (c, k, f)
            assert e[k]["trades"] >= 0 and e[k]["active_bars"] >= 0, (c, k)
        for k in ("LONG", "SHORT"):
            assert "pnl_share" in e[k], (c, k)
        r = e["cci"]
        for f in ("mean", "min", "max", "n_long_open", "n_short_open", "frac_long_open", "frac_short_open"):
            assert f in r, (c, f)
        assert r["min"] < -100.0 and r["max"] > 100.0, (c, r)
        assert r["n_long_open"] + r["n_short_open"] <= d["config"]["grid_bars"], (c, r)
        assert r["n_long_open"] > 0 and r["n_short_open"] > 0, (c, r)
    b = d["basket"]
    assert set(("FULL", "LONG", "SHORT", "coin_pnl_share")) <= set(b)
    assert set(b["coin_pnl_share"]) == set(COINS)
    assert set(d["baseline"]) >= set(("basket_FULL", "coin_FULL"))
    assert set(d["baseline"]["coin_FULL"]) == set(COINS)


def test_iter_h72_additivity():
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


def test_iter_h72_cci_math():
    sys.path.insert(0, ".")
    from research.run_iter_h72_cci import classic_cci
    mk = lambda closes: [(c, c, c, c, 0.0) for c in closes]
    n = 300
    flat = [100.0] * n
    assert all(v == 0.0 for v in classic_cci(mk(flat), 24))
    up = [100.0 + i for i in range(n)]
    r = classic_cci(mk(up), 24)
    assert all(v == 0.0 for v in r[:24])
    assert r[24] == pytest.approx(1150.0 / 9.0, rel=1e-6)
    assert r[-1] == pytest.approx(1150.0 / 9.0, rel=1e-6)
    dn = [1000.0 - i for i in range(n)]
    r2 = classic_cci(mk(dn), 24)
    assert all(v == 0.0 for v in r2[:24])
    assert r2[24] == pytest.approx(-1150.0 / 9.0, rel=1e-6)
    assert r2[-1] == pytest.approx(-1150.0 / 9.0, rel=1e-6)
    assert math.isnan(sum(r)) is False


def test_iter_h72_scale_and_nobroker():
    src = pathlib.Path("research/run_iter_h72_cci.py").read_text()
    assert "BPY = 8760.0" in src
    assert "SCALE = 4" in src
    assert "data/data_1y/1h/" in src
    for bad in ("place_order", "submit_order", "api_key", "API_KEY",
                "make_broker", "Y1B_LIVE"):
        assert bad.lower() not in src.lower(), bad
    d = _load()
    assert d["config"]["grid_bars"] > 8000
