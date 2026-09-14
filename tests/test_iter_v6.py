"""V6 15m MACD filter (Top5) tests: legs gated by MACD(48,96,18) sign agreement."""
import json
import pathlib
import subprocess
import sys

OUT = pathlib.Path("results/iter_V6_macd.json")
COINS = ["ETC", "TRX", "ATOM", "APT", "KAS"]


def _load():
    assert OUT.exists(), "results/iter_V6_macd.json missing; run research/run_iter_v6_macd.py"
    return json.loads(OUT.read_text())


def test_iter_v6_lock():
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
    assert d["config"]["macd"]["fast"] == 48
    assert d["config"]["macd"]["slow"] == 96
    assert d["config"]["macd"]["signal"] == 18
    assert d["config"]["macd"]["warmup_bars"] == 114
    assert d["config"]["macd"]["long_gate"] == "macd>0 AND signal>0"
    assert d["config"]["macd"]["short_gate"] == "macd<0 AND signal<0"
    for c, spec in (("ETC", LOCKED_ETC), ("TRX", LOCKED_TRX), ("ATOM", LOCKED_ATOM), ("APT", LOCKED_APT), ("KAS", LOCKED_KAS)):
        for k in ("lth", "sth", "cd", "sl", "ts", "vt", "vw", "q"):
            assert d["config"]["basket"][c][k] == spec[k], (c, k)
    assert d["verdict"] == "PENDING"
    assert d["decision"] == "NO_ADOPTION"
    assert d["status"] == "COMPLETE"


def test_iter_v6_schema():
    d = _load()
    assert set(d["coins"]) == set(COINS)
    for c in COINS:
        e = d["coins"][c]
        assert set(("FULL", "LONG", "SHORT", "BASE_FULL", "macd")) <= set(e), c
        for k in ("FULL", "LONG", "SHORT", "BASE_FULL"):
            for f in ("sharpe", "cum", "ann", "mdd", "n", "turnover"):
                assert f in e[k], (c, k, f)
            assert e[k]["n"] == d["config"]["grid_bars"], (c, k)
        for k in ("LONG", "SHORT"):
            for f in ("pnl_share", "trades", "active_bars"):
                assert f in e[k], (c, k, f)
            assert e[k]["trades"] >= 0 and e[k]["active_bars"] >= 0, (c, k)
        m = e["macd"]
        for f in ("hist_mean", "hist_min", "hist_max", "n_long_open", "n_short_open", "frac_long_open", "frac_short_open"):
            assert f in m, (c, f)
        assert m["hist_min"] <= m["hist_mean"] <= m["hist_max"], (c, m)
        assert m["n_long_open"] + m["n_short_open"] <= d["config"]["grid_bars"], (c, m)
        assert m["n_long_open"] > 0 and m["n_short_open"] > 0, (c, m)
        assert 0.0 < m["frac_long_open"] < 1.0 and 0.0 < m["frac_short_open"] < 1.0, (c, m)
    b = d["basket"]
    assert set(("FULL", "LONG", "SHORT", "coin_pnl_share")) <= set(b)
    assert set(b["coin_pnl_share"]) == set(COINS)
    assert set(d["baseline"]) >= set(("basket_FULL", "coin_FULL"))
    assert set(d["baseline"]["coin_FULL"]) == set(COINS)


def test_iter_v6_additivity():
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


def test_iter_v6_macd_math():
    sys.path.insert(0, ".")
    from research.run_iter_v6_macd import macd_lines, gate_masks, WARMUP
    assert WARMUP == 114
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


def test_iter_v6_script_runs_offline():
    r = subprocess.run([sys.executable, "research/run_iter_v6_macd.py"], capture_output=True, text=True, cwd=".")
    assert r.returncode == 0, r.stderr[-2000:]
    d = _load()
    assert set(d["coins"]) == set(COINS)
    assert d["verdict"] == "PENDING" and d["decision"] == "NO_ADOPTION"
    assert d["status"] == "COMPLETE"
