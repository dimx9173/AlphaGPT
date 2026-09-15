"""H76 1h Elder-ray filter (Top5) tests: long bull-power>0 / short bear-power<0 gate on legs."""
import json
import math
import pathlib
import subprocess
import sys

OUT = pathlib.Path("results/iter_H76_elder.json")
COINS = ["ETC", "TRX", "ATOM", "APT", "KAS"]


def _load():
    assert OUT.exists(), "results/iter_H76_elder.json missing; run research/run_iter_h76_elder.py"
    return json.loads(OUT.read_text())


def test_iter_h76_lock():
    from strategy_manager.config import LEV, FORMULA, LOCKED_ETC, LOCKED_TRX, LOCKED_ATOM, LOCKED_APT, LOCKED_KAS
    assert LEV == 2.0
    assert list(FORMULA) == [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10]
    d = _load()
    assert d["config"]["lev"] == 2.0
    assert d["config"]["formula"] == [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10]
    assert d["config"]["grid"] == "1h"
    assert d["config"]["bpy"] == 8760.0
    assert d["config"]["scale"] == 4
    assert d["config"]["elder"]["window"] == 13
    assert d["config"]["elder"]["long_gate"] == "bull-power>0"
    assert d["config"]["elder"]["short_gate"] == "bear-power<0"
    assert d["config"]["elder"]["threshold"] == 0.0
    assert d["config"]["elder"]["warmup_bars"] == 13
    for c, spec in (("ETC", LOCKED_ETC), ("TRX", LOCKED_TRX), ("ATOM", LOCKED_ATOM), ("APT", LOCKED_APT), ("KAS", LOCKED_KAS)):
        for k in ("lth", "sth", "cd", "sl", "ts", "vt", "vw", "q"):
            assert d["config"]["basket"][c][k] == spec[k], (c, k)
    assert d["verdict"] == "PENDING"
    assert d["decision"] == "NO_ADOPTION"
    assert d["status"] == "COMPLETE"


def test_iter_h76_schema():
    d = _load()
    assert set(d["coins"]) == set(COINS)
    for c in COINS:
        e = d["coins"][c]
        assert set(("FULL", "LONG", "SHORT", "BASE_FULL", "elder")) <= set(e), c
        for k in ("FULL", "LONG", "SHORT", "BASE_FULL"):
            for f in ("sharpe", "cum", "ann", "mdd", "n", "turnover"):
                assert f in e[k], (c, k, f)
            assert e[k]["n"] == d["config"]["grid_bars"], (c, k)
        for k in ("LONG", "SHORT"):
            for f in ("pnl_share", "trades", "active_bars"):
                assert f in e[k], (c, k, f)
            assert e[k]["trades"] >= 0 and e[k]["active_bars"] >= 0, (c, k)
        r = e["elder"]
        for f in ("bull_mean", "bull_min", "bull_max", "bear_mean", "bear_min", "bear_max",
                  "n_long_open", "n_short_open", "frac_long_open", "frac_short_open"):
            assert f in r, (c, f)
        assert r["bull_min"] <= r["bull_max"], (c, r)
        assert r["bear_min"] <= r["bear_max"], (c, r)
        assert r["n_long_open"] + r["n_short_open"] <= 2 * d["config"]["grid_bars"], (c, r)
        assert r["n_long_open"] > 0 and r["n_short_open"] > 0, (c, r)
    b = d["basket"]
    assert set(("FULL", "LONG", "SHORT", "coin_pnl_share")) <= set(b)
    assert set(b["coin_pnl_share"]) == set(COINS)
    assert set(d["baseline"]) >= set(("basket_FULL", "coin_FULL"))
    assert set(d["baseline"]["coin_FULL"]) == set(COINS)


def test_iter_h76_additivity():
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


def test_iter_h76_elder_math():
    sys.path.insert(0, ".")
    from research.run_iter_h76_elder import ema_series, elder_powers, elder_gate_masks
    # constant closes -> flat EMA == close
    e = ema_series([100.0] * 40, 13)
    assert len(e) == 40
    assert all(abs(v - 100.0) < 1e-9 for v in e)
    assert math.isnan(sum(e)) is False
    # flat bars: high==low==close -> powers exactly 0 after warmup too
    flat = [(100.0, 100.0, 100.0, 100.0, 1000.0)] * 40
    bull, bear = elder_powers(flat, 13)
    assert len(bull) == 40 and len(bear) == 40
    assert all(v == 0.0 for v in bull)
    assert all(v == 0.0 for v in bear)
    # steady uptrend (close==high): bull-power > 0 after warmup
    up = [(99.0 + i, 101.0 + i, 99.0 + i, 101.0 + i, 100.0) for i in range(40)]
    bu, be = elder_powers(up, 13)
    assert all(v == 0.0 for v in bu[:13])
    assert all(v == 0.0 for v in be[:13])
    assert all(v > 0 for v in bu[13:])
    # steady downtrend (close==low): bear-power < 0 after warmup
    dn = [(101.0 - i, 101.0 - i, 99.0 - i, 99.0 - i, 100.0) for i in range(40)]
    bu2, be2 = elder_powers(dn, 13)
    assert all(v == 0.0 for v in bu2[:13])
    assert all(v < 0 for v in be2[13:])
    # short series (n <= window) -> all neutral
    b3, e3 = elder_powers(flat[:10], 13)
    assert all(v == 0.0 for v in b3) and all(v == 0.0 for v in e3)
    lok, sok = elder_gate_masks([0.5, -0.5, 0.0], [0.5, -0.5, 0.0])
    assert lok == [1.0, 0.0, 0.0] and sok == [0.0, 1.0, 0.0]  # tie blocks both


def test_iter_h76_scale_and_nobroker():
    src = pathlib.Path("research/run_iter_h76_elder.py").read_text()
    assert "BPY = 8760.0" in src
    assert "SCALE = 4" in src
    assert "ELDER_WINDOW = 13" in src
    assert "data/data_1y/1h/" in src
    for bad in ("place_order", "submit_order", "api_key", "API_KEY",
                "make_broker", "Y1B_LIVE"):
        assert bad.lower() not in src.lower(), bad
    d = _load()
    assert d["config"]["grid_bars"] > 8000


def test_iter_h76_smoke_runs_offline():
    import tempfile, os as _os
    with tempfile.TemporaryDirectory() as td:
        out = _os.path.join(td, "h76.json")
        lg = _os.path.join(td, "h76.log")
        env = dict(_os.environ, ITER_H76_SMOKE="1", ITER_H76_OUT=out, ITER_H76_LOG=lg)
        r = subprocess.run([sys.executable, "research/run_iter_h76_elder.py"],
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
