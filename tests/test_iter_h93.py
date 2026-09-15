"""H93 1h Williams v2 %R filter (Top5) tests: entry-only %R(56)>-20 long / <-80 short gate."""
import json
import math
import pathlib
import sys

OUT = pathlib.Path("results/iter_H93_willr.json")
COINS = ["ETC", "TRX", "ATOM", "APT", "KAS"]


def _load():
    assert OUT.exists(), "results/iter_H93_willr.json missing; run research/run_iter_h93_willr.py"
    return json.loads(OUT.read_text())


def test_iter_h93_lock():
    from strategy_manager.config import LEV, FORMULA, LOCKED_ETC, LOCKED_TRX, LOCKED_ATOM, LOCKED_APT, LOCKED_KAS
    assert LEV == 2.0
    assert list(FORMULA) == [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10]
    d = _load()
    assert d["config"]["lev"] == 2.0
    assert d["config"]["formula"] == [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10]
    assert d["config"]["grid"] == "1h"
    assert d["config"]["grid_bars"] == 8760
    assert d["config"]["bpy"] == 8760.0
    assert d["config"]["scale"] == 4
    assert d["config"]["weights"] == {c: 0.2 for c in COINS}
    assert d["config"]["willr"]["window"] == 56
    assert d["config"]["willr"]["long_gate"] == "R>-20"
    assert d["config"]["willr"]["short_gate"] == "R<-80"
    assert d["config"]["willr"]["long_th"] == -20.0
    assert d["config"]["willr"]["short_th"] == -80.0
    assert d["config"]["willr"]["warmup_bars"] == 56
    for c, spec in (("ETC", LOCKED_ETC), ("TRX", LOCKED_TRX), ("ATOM", LOCKED_ATOM), ("APT", LOCKED_APT), ("KAS", LOCKED_KAS)):
        for k in ("lth", "sth", "cd", "sl", "ts", "vt", "vw", "q"):
            assert d["config"]["basket"][c][k] == spec[k], (c, k)
    assert d["verdict"] == "PENDING"
    assert d["decision"] == "NO_ADOPTION"
    assert d["status"] == "COMPLETE"
    assert "no adoption" in d["conclusion"]
    assert "live untouched" in d["conclusion"]


def test_iter_h93_schema():
    d = _load()
    assert set(d["coins"]) == set(COINS)
    for c in COINS:
        e = d["coins"][c]
        assert set(("FULL", "LONG", "SHORT", "BASE_FULL", "willr")) <= set(e), c
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
        wr = e["willr"]
        for f in ("mean", "min", "max", "n_long_open", "n_short_open",
                  "frac_long_open", "frac_short_open", "warmup_bars"):
            assert f in wr, (c, f)
        assert -100.0 <= wr["min"] <= wr["max"] <= 0.0, (c, wr)
        assert wr["n_long_open"] + wr["n_short_open"] <= d["config"]["grid_bars"], (c, wr)
        assert wr["n_long_open"] > 0 and wr["n_short_open"] > 0, (c, wr)
    b = d["basket"]
    assert set(("FULL", "LONG", "SHORT", "coin_pnl_share")) <= set(b)
    assert set(b["coin_pnl_share"]) == set(COINS)
    for f in ("sharpe", "trades", "cum", "mdd", "turnover"):
        assert f in b["FULL"], f
    assert set(d["baseline"]) >= set(("basket_FULL", "coin_FULL"))
    assert set(d["baseline"]["coin_FULL"]) == set(COINS)


def test_iter_h93_additivity():
    d = _load()
    for c in COINS:
        f = d["coins"][c]["FULL"]["cum"]
        l = d["coins"][c]["LONG"]["cum"]
        s = d["coins"][c]["SHORT"]["cum"]
        assert abs(f - (l + s)) < 1e-3, (c, f, l, s)
        assert abs(d["coins"][c]["LONG"]["pnl_share"] + d["coins"][c]["SHORT"]["pnl_share"] - 1.0) < 1e-3, c
        lf = d["coins"][c]["FULL"]
        assert abs(lf["trades"] - (d["coins"][c]["LONG"]["trades"] + d["coins"][c]["SHORT"]["trades"])) < 1e-9, c
    bf = d["basket"]["FULL"]["cum"]
    assert abs(bf - (d["basket"]["LONG"]["cum"] + d["basket"]["SHORT"]["cum"])) < 1e-3
    assert abs(d["basket"]["LONG"]["pnl_share"] + d["basket"]["SHORT"]["pnl_share"] - 1.0) < 1e-3
    assert abs(sum(d["basket"]["coin_pnl_share"].values()) - 1.0) < 1e-3


def test_iter_h93_willr_math():
    sys.path.insert(0, ".")
    from research.run_iter_h93_willr import willr, WILLR_WINDOW, WILLR_LONG, WILLR_SHORT
    assert WILLR_WINDOW == 56
    assert WILLR_LONG == -20.0
    assert WILLR_SHORT == -80.0
    n = 200
    flat = [100.0] * n
    R = willr(flat, flat, flat)
    assert all(v == -50.0 for v in R)
    # warmup: first 56 bars neutral even on trending data
    up = [100.0 + 0.01 * i * i for i in range(n)]
    h = [c + 1.0 for c in up]
    l = [c - 1.0 for c in up]
    Ru = willr(h, l, up)
    assert all(v == -50.0 for v in Ru[:56])
    # close pinned at bar high -> %R near 0 (long gate open)
    assert Ru[-1] > -20.0
    # close pinned at bar low -> %R near -100 (short gate open)
    dn = [10000.0 - 0.01 * i * i for i in range(n)]
    hd = [c + 1.0 for c in dn]
    ld = [c - 1.0 for c in dn]
    # shift closes down to the low edge of each bar window
    dn_low = [c - 1.0 for c in dn]
    Rd = willr(hd, ld, dn_low)
    assert all(v == -50.0 for v in Rd[:56])
    assert Rd[-1] < -80.0
    assert all(-100.0 <= v <= 0.0 for v in Ru)
    assert all(-100.0 <= v <= 0.0 for v in Rd)
    assert math.isnan(sum(Ru)) is False


def test_iter_h93_no_broker():
    src = pathlib.Path("research/run_iter_h93_willr.py").read_text().lower()
    for bad in ("place_order", "submit_order", "api_key", "make_broker", "y1b_live", "market_open"):
        assert bad not in src, bad


def test_iter_h93_engine_mirror():
    src = pathlib.Path("research/run_iter_h93_willr.py").read_text()
    for token in ("quantile", "qmask", "cooldown", "_apply_stops", "_vol_scale",
                  "roll(1", "8760", "cooldown_bars=int(spec", "* SCALE",
                  "time_stop=int(spec", "vol_window=int(spec", "vol_scale",
                  "short_enabled=True", "def willr", "rr > WILLR_LONG", "rr < WILLR_SHORT"):
        assert token in src, token
    assert "no adoption" in src
    assert "live untouched" in src
