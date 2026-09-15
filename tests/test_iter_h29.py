"""H29 1h MACD filter (Top5) tests: legs gated by MACD(12,26,9) sign agreement.

Full H29 = 5 coins 1h-native (~8760 bars); smoke = {ETC,TRX} x 2000 bars via
ITER_H29_SMOKE=1. Schema checks accept either artifact; the live subprocess
check runs smoke mode only. Conclusion stays PENDING: diagnostic only,
no adoption, live untouched.
"""
import json
import os
import pathlib
import subprocess
import sys

OUT = pathlib.Path("results/iter_H29_macd.json")
COINS_FULL = ["ETC", "TRX", "ATOM", "APT", "KAS"]
SEG = {"sharpe", "ann", "mdd", "cum", "final_x", "n", "turnover"}


def _load():
    assert OUT.exists(), "results/iter_H29_macd.json missing; run research/run_iter_h29_macd.py"
    return json.loads(OUT.read_text())


def test_iter_h29_lock():
    from strategy_manager.config import LEV, FORMULA, LOCKED_ETC, LOCKED_TRX, LOCKED_ATOM, LOCKED_APT, LOCKED_KAS
    assert LEV == 2.0
    assert list(FORMULA) == [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10]
    d = _load()
    assert d["config"]["lev"] == 2.0
    assert d["config"]["formula"] == [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10]
    assert d["config"]["grid"] == "1h"
    assert d["config"]["bpy"] == 8760.0
    assert d["config"]["scale"] == 4
    assert d["config"]["macd"]["fast"] == 12
    assert d["config"]["macd"]["slow"] == 26
    assert d["config"]["macd"]["signal"] == 9
    assert d["config"]["macd"]["warmup_bars"] == 35
    assert d["config"]["macd"]["long_gate"] == "macd>0 AND signal>0"
    assert d["config"]["macd"]["short_gate"] == "macd<0 AND signal<0"
    coins = d["config"]["coins"]
    locked = {"ETC": LOCKED_ETC, "TRX": LOCKED_TRX, "ATOM": LOCKED_ATOM, "APT": LOCKED_APT, "KAS": LOCKED_KAS}
    for c in coins:
        for k in ("lth", "sth", "cd", "sl", "ts", "vt", "vw", "q"):
            assert d["config"]["basket"][c][k] == locked[c][k], (c, k)
    assert d["verdict"] == "PENDING"
    assert d["decision"] == "NO_ADOPTION"
    assert d["status"] == "COMPLETE"


def test_iter_h29_schema():
    d = _load()
    cfg = d["config"]
    coins = cfg["coins"]
    assert set(d["coins"]) == set(coins)
    n = cfg["grid_bars"]
    assert n == cfg["h2_start"] * 2 or cfg["h2_start"] == n // 2
    for c in coins:
        e = d["coins"][c]
        assert {"FULL", "LONG", "SHORT", "BASE_FULL", "H2_FULL_sharpe", "macd"} <= set(e), c
        for k in ("FULL", "LONG", "SHORT", "BASE_FULL"):
            assert SEG <= set(e[k]), (c, k)
            assert e[k]["n"] == n, (c, k)
        assert isinstance(e["H2_FULL_sharpe"], float), c
        for k in ("LONG", "SHORT"):
            for f in ("pnl_share", "trades", "active_bars"):
                assert f in e[k], (c, k, f)
            assert e[k]["trades"] >= 0 and e[k]["active_bars"] >= 0, (c, k)
        m = e["macd"]
        for f in ("hist_mean", "hist_min", "hist_max", "n_long_open", "n_short_open", "frac_long_open", "frac_short_open"):
            assert f in m, (c, f)
        assert m["hist_min"] <= m["hist_mean"] <= m["hist_max"], (c, m)
        assert m["n_long_open"] + m["n_short_open"] <= n, (c, m)
        assert m["n_long_open"] > 0 and m["n_short_open"] > 0, (c, m)
        assert 0.0 < m["frac_long_open"] < 1.0 and 0.0 < m["frac_short_open"] < 1.0, (c, m)
    b = d["basket"]
    assert {"FULL", "LONG", "SHORT", "H2_FULL_sharpe", "coin_pnl_share"} <= set(b)
    assert set(b["coin_pnl_share"]) == set(coins)
    assert isinstance(b["H2_FULL_sharpe"], float)
    assert {"basket_FULL", "coin_FULL"} <= set(d["baseline"])
    assert set(d["baseline"]["coin_FULL"]) == set(coins)
    assert "\u5f85\u5b9a" in d["conclusion"] or "待定" in d["conclusion"]
    assert "no adoption" in d["conclusion"]
    assert "live untouched" in d["conclusion"]


def test_iter_h29_additivity():
    d = _load()
    coins = d["config"]["coins"]
    for c in coins:
        f = d["coins"][c]["FULL"]["cum"]
        l = d["coins"][c]["LONG"]["cum"]
        s = d["coins"][c]["SHORT"]["cum"]
        assert abs(f - (l + s)) < 1e-3, (c, f, l, s)
        assert abs(d["coins"][c]["LONG"]["pnl_share"] + d["coins"][c]["SHORT"]["pnl_share"] - 1.0) < 1e-3, c
    bf = d["basket"]["FULL"]["cum"]
    assert abs(bf - (d["basket"]["LONG"]["cum"] + d["basket"]["SHORT"]["cum"])) < 1e-3
    assert abs(d["basket"]["LONG"]["pnl_share"] + d["basket"]["SHORT"]["pnl_share"] - 1.0) < 1e-3
    assert abs(sum(d["basket"]["coin_pnl_share"].values()) - 1.0) < 1e-3


def test_iter_h29_macd_math():
    sys.path.insert(0, ".")
    from research.run_iter_h29_macd import macd_lines, gate_masks, WARMUP
    assert WARMUP == 35
    n = 300
    flat = [100.0] * n
    m, s, h = macd_lines(flat)
    assert all(v == 0.0 for v in m) and all(v == 0.0 for v in s) and all(v == 0.0 for v in h)
    lo, so = gate_masks(m, s)
    assert sum(lo) == 0 and sum(so) == 0
    up = [100.0 + i * 0.5 for i in range(n)]
    m2, s2, _ = macd_lines(up)
    lo2, so2 = gate_masks(m2, s2)
    assert m2[-1] > 0.0 and s2[-1] > 0.0
    assert sum(lo2) > 0 and sum(so2) == 0
    assert sum(lo2[:WARMUP]) == 0 and sum(so2[:WARMUP]) == 0
    dn = [100.0 - i * 0.5 for i in range(n)]
    m3, s3, _ = macd_lines(dn)
    lo3, so3 = gate_masks(m3, s3)
    assert m3[-1] < 0.0 and s3[-1] < 0.0
    assert sum(lo3) == 0 and sum(so3) > 0
    assert sum(lo3[:WARMUP]) == 0 and sum(so3[:WARMUP]) == 0


def test_iter_h29_engine_mirror():
    src = pathlib.Path("research/run_iter_h29_macd.py").read_text()
    for token in ("quantile", "cooldown", "_apply_stops", "_vol_scale",
                  "roll(1", "8760", "cooldown_bars=int(spec[\"cd\"]) * SCALE",
                  "time_stop=int(spec[\"ts\"]) * SCALE", "vol_window=int(spec[\"vw\"]) * SCALE"):
        assert token in src, token
    assert "short_enabled=True" in src
    assert "venue=\"aster\"" in src
    for bad in ("place_order", "submit_order", "api_key", "make_broker", "Y1B_LIVE"):
        assert bad.lower() not in src.lower(), bad


def test_iter_h29_smoke_runs_offline(tmp_path):
    # Smoke writes to a temp OUT so the committed full artifact is untouched.
    out = tmp_path / "smoke_h29.json"
    log = tmp_path / "smoke_h29.log"
    env = dict(os.environ, ITER_H29_SMOKE="1",
               ITER_H29_OUT=str(out), ITER_H29_LOG=str(log))
    r = subprocess.run([sys.executable, "research/run_iter_h29_macd.py"],
                       capture_output=True, text=True, cwd=".", env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    d = json.loads(out.read_text())
    assert d["config"]["smoke"] is True
    assert set(d["config"]["coins"]) == {"ETC", "TRX"}
    assert d["config"]["grid_bars"] == 2000
    assert d["verdict"] == "PENDING" and d["decision"] == "NO_ADOPTION"
    assert d["status"] == "COMPLETE"
