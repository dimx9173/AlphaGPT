"""V7 15m Stoch filter (Top5) tests: entry-only Stoch(56,3) %K>%D long / %K<%D short gate."""
import json
import math
import pathlib
import subprocess
import sys

OUT = pathlib.Path("results/iter_V7_stoch.json")
COINS = ["ETC", "TRX", "ATOM", "APT", "KAS"]


def _load():
    assert OUT.exists(), "results/iter_V7_stoch.json missing; run research/run_iter_v7_stoch.py"
    return json.loads(OUT.read_text())


def test_iter_v7_lock():
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
    assert d["config"]["stoch"]["window"] == 56
    assert d["config"]["stoch"]["smooth"] == 3
    assert d["config"]["stoch"]["long_gate"] == "K>D"
    assert d["config"]["stoch"]["short_gate"] == "K<D"
    assert d["config"]["stoch"]["warmup_bars"] == 60
    for c, spec in (("ETC", LOCKED_ETC), ("TRX", LOCKED_TRX), ("ATOM", LOCKED_ATOM), ("APT", LOCKED_APT), ("KAS", LOCKED_KAS)):
        for k in ("lth", "sth", "cd", "sl", "ts", "vt", "vw", "q"):
            assert d["config"]["basket"][c][k] == spec[k], (c, k)
    assert d["verdict"] == "PENDING"
    assert d["decision"] == "NO_ADOPTION"
    assert d["status"] == "COMPLETE"
    assert "no adoption" in d["conclusion"]
    assert "live untouched" in d["conclusion"]


def test_iter_v7_schema():
    d = _load()
    assert set(d["coins"]) == set(COINS)
    for c in COINS:
        e = d["coins"][c]
        assert set(("FULL", "LONG", "SHORT", "BASE_FULL", "stoch")) <= set(e), c
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
        st = e["stoch"]
        for f in ("mean_k", "mean_d", "min_k", "max_k", "min_d", "max_d",
                  "n_long_open", "n_short_open", "frac_long_open", "frac_short_open", "warmup_bars"):
            assert f in st, (c, f)
        assert 0.0 <= st["min_k"] <= st["max_k"] <= 100.0, (c, st)
        assert 0.0 <= st["min_d"] <= st["max_d"] <= 100.0, (c, st)
        assert st["n_long_open"] + st["n_short_open"] <= d["config"]["grid_bars"], (c, st)
        assert st["n_long_open"] > 0 and st["n_short_open"] > 0, (c, st)
    b = d["basket"]
    assert set(("FULL", "LONG", "SHORT", "coin_pnl_share")) <= set(b)
    assert set(b["coin_pnl_share"]) == set(COINS)
    for f in ("trades", "active_bars"):
        assert f in b["FULL"], f
    assert set(d["baseline"]) >= set(("basket_FULL", "coin_FULL"))
    assert set(d["baseline"]["coin_FULL"]) == set(COINS)


def test_iter_v7_additivity():
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


def test_iter_v7_stoch_math():
    sys.path.insert(0, ".")
    from research.run_iter_v7_stoch import stoch_kd
    n = 200
    flat = [100.0] * n
    K, D = stoch_kd(flat, flat, flat)
    assert all(v == 50.0 for v in K)
    assert all(v == 50.0 for v in D)
    # warmup: first 60 bars neutral even on trending data
    up = [100.0 + 0.01 * i * i for i in range(n)]
    h = [c + 1.0 for c in up]
    l = [c - 1.0 for c in up]
    K, D2 = stoch_kd(h, l, up)
    assert all(v == 50.0 for v in K[:60])
    assert all(v == 50.0 for v in D2[:60])
    # accelerating uptrend: %K sits above %D at the tail
    assert K[-1] > D2[-1]
    assert K[-1] > 95.0
    dn = [10000.0 - 0.01 * i * i for i in range(n)]
    hd = [c + 1.0 for c in dn]
    ld = [c - 1.0 for c in dn]
    Kd, Dd = stoch_kd(hd, ld, dn)
    assert all(v == 50.0 for v in Kd[:60])
    assert Kd[-1] < Dd[-1]
    assert Kd[-1] < 5.0
    assert all(0.0 <= v <= 100.0 for v in K)
    assert all(0.0 <= v <= 100.0 for v in D2)
    assert math.isnan(sum(K)) is False


def test_iter_v7_no_broker():
    src = pathlib.Path("research/run_iter_v7_stoch.py").read_text().lower()
    for bad in ("place_order", "submit_order", "api_key", "make_broker", "y1b_live", "market_open"):
        assert bad not in src, bad


def test_iter_v7_engine_mirror():
    src = pathlib.Path("research/run_iter_v7_stoch.py").read_text()
    for token in ("quantile", "qmask", "cooldown", "_apply_stops", "_vol_scale",
                  "roll(1", "35040", "cooldown_bars=int(spec", "* SCALE",
                  "time_stop=int(spec", "vol_window=int(spec", "vol_scale",
                  "short_enabled=True", "stoch_kd", "rk > rd", "rk < rd"):
        assert token in src, token
    assert "no adoption" in src
    assert "live untouched" in src
