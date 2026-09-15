"""H15 1h vol-regime split (Top5) tests. Recompute by script."""
import importlib.util
import json
import os
import pathlib
import subprocess
import sys

REG = ("low", "mid", "high")
OUT = pathlib.Path(os.getenv("ITER_H15_OUT", "results/iter_H15_volreg.json"))


def _load():
    assert OUT.exists(), "results/iter_H15_volreg.json missing; run research/run_iter_h15_volreg.py"
    return json.loads(OUT.read_text())


def _mod():
    spec = importlib.util.spec_from_file_location(
        "h15mod", "research/run_iter_h15_volreg.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["h15mod"] = m
    try:
        spec.loader.exec_module(m)
    except SystemExit:
        pass
    return m


def test_iter_h15_lock():
    from strategy_manager.config import (
        LEV, FEE, FUND, FEE2X, FORMULA,
        LOCKED_ETC, LOCKED_TRX, LOCKED_ATOM, LOCKED_APT, LOCKED_KAS)
    assert LEV == 2.0
    assert FEE2X == 2 * FEE
    assert FORMULA == [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10]
    assert (LOCKED_ETC["lth"], LOCKED_ETC["sth"], LOCKED_ETC["cd"],
            LOCKED_ETC["sl"], LOCKED_ETC["ts"]) == (0.88, 0.12, 18, None, 24)
    assert (LOCKED_TRX["lth"], LOCKED_TRX["sth"], LOCKED_TRX["cd"],
            LOCKED_TRX["sl"], LOCKED_TRX["ts"]) == (0.85, 0.12, 6, 0.05, 24)
    assert (LOCKED_ATOM["lth"], LOCKED_ATOM["sth"], LOCKED_ATOM["cd"],
            LOCKED_ATOM["sl"], LOCKED_ATOM["ts"]) == (0.85, 0.15, 6, 0.05, 24)
    assert (LOCKED_APT["lth"], LOCKED_APT["sth"], LOCKED_APT["cd"],
            LOCKED_APT["sl"], LOCKED_APT["ts"]) == (0.88, 0.12, 18, None, 24)
    assert (LOCKED_KAS["lth"], LOCKED_KAS["sth"], LOCKED_KAS["cd"],
            LOCKED_KAS["sl"], LOCKED_KAS["ts"]) == (0.88, 0.12, 6, None, 24)


def test_iter_h15_schema():
    d = _load()
    assert d.get("status") == "final"
    cfg = d["config"]
    assert cfg["grid"] == "1h"
    assert cfg["bpy"] == 8760.0
    assert cfg["scale_4h_to_1h"] == 4
    assert cfg["grid_bars"] > 2000
    assert cfg["vol_window"] == 60
    assert cfg["vt_grid"] == [0.006, 0.012]
    assert cfg["smoke"] in (True, False)
    assert cfg["regimes"] == ["low", "mid", "high"]
    if not cfg["smoke"]:
        assert cfg["coins"] == ["ETC", "TRX", "ATOM", "APT", "KAS"]
    n = cfg["grid_bars"]
    coins = cfg["coins"]
    for c in coins:
        sc = cfg["scaled_specs_1h"][c]
        base = cfg["basket"][c]
        for k in ("cd", "ts", "vw"):
            if base[k] is None:
                assert sc[k] is None
            else:
                assert sc[k] == int(base[k]) * 4
        assert d["vol_bounds"][c]["t1"] <= d["vol_bounds"][c]["t2"]
    assert set(d["per_coin"]) == set(coins)
    for c, pc in d["per_coin"].items():
        tot = 0
        for r in REG:
            assert {"sharpe", "ann", "mdd", "cum", "n", "turnover"} <= set(pc[r])
            assert pc[r]["n"] > 0
            tot += pc[r]["n"]
        assert tot == n, (c, tot, n)
        assert {"trades", "entries", "flips", "exits"} <= set(pc["FULL"])
        assert pc["FULL"]["n"] == n
        assert pc["FULL"]["trades"] == pc["FULL"]["entries"] + pc["FULL"]["flips"]
        assert abs(pc["FULL"]["final_x"] - round(1.0 + pc["FULL"]["cum"], 4)) < 1e-9
    b = d["basket"]
    tot = sum(b[r]["n"] for r in REG)
    assert tot == n, (tot, n)
    assert b["FULL"]["n"] == n
    assert abs(b["FULL"]["final_x"] - round(1.0 + b["FULL"]["cum"], 4)) < 1e-9
    assert b["FULL"]["trades"] == b["FULL"]["entries"] + b["FULL"]["flips"]
    assert d["basket_vol_bounds"]["t1"] <= d["basket_vol_bounds"]["t2"]
    vr = d["vt_recheck"]
    assert vr["base_FULL_sharpe"] == b["FULL"]["sharpe"]
    assert set(vr["rows"]) == {"0.006", "0.012"}
    for k, row in vr["rows"].items():
        assert row["FULL"]["n"] == n
        assert abs(row["gap_FULL_sharpe"] - round(row["FULL"]["sharpe"] - b["FULL"]["sharpe"], 3)) < 1e-9
        assert set(row["per_coin_FULL_sharpe"]) == set(coins)
    assert d["verdict"] == "PENDING_P03_FAIL"
    assert d["decision"] == "NO_ADOPTION_KEEP_EQUAL"
    assert "PENDING" in d["conclusion"] and "no live change" in d["conclusion"]


def test_iter_h15_helpers():
    m = _mod()
    v = m.trail_vol_series([100.0, 101.0, 102.0, 101.5, 103.0, 104.0])
    assert len(v) == 6
    assert v[0] == 0.0 and v[1] == 0.0
    assert all(x >= 0.0 for x in v)
    assert m.tercile_bounds([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])[0] <= m.tercile_bounds([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])[1]
    assert m.assign_regime([0.1, 0.5, 0.9], 0.3, 0.7) == [0, 1, 2]
    s = m.seg_idx([0.01, -0.02, 0.03, 0.01], [0.0, 1.0, 0.0, 2.0], [0, 2, 3])
    assert s["n"] == 3
    assert abs(s["cum"] - round(0.01 + 0.03 + 0.01, 4)) < 1e-9
    assert abs(s["turnover"] - round((0.0 + 0.0 + 2.0) / 3, 6)) < 1e-9
    full = m.seg_idx([0.01, -0.02, 0.03], [0.0, 0.0, 0.0], [0, 1, 2])
    assert full["n"] == 3
    assert m.count_trades([0.0, 1.0, 1.0, 0.0], 0, 4) == {
        "trades": 1, "entries": 1, "flips": 0, "exits": 1}


def test_iter_h15_no_broker():
    src = pathlib.Path("research/run_iter_h15_volreg.py").read_text().lower()
    for bad in ("place_order", "submit_order", "api_key", "make_broker",
                "y1b_live", "market_open"):
        assert bad not in src, bad


def test_iter_h15_smoke_runs_offline(tmp_path):
    out = tmp_path / "iter_H15_volreg.json"
    lg = tmp_path / "iter_H15_volreg.log"
    env = dict(os.environ, ITER_H15_SMOKE="1", ITER_H15_OUT=str(out),
               ITER_H15_LOG=str(lg))
    r = subprocess.run([sys.executable, "research/run_iter_h15_volreg.py"],
                       capture_output=True, text=True, cwd=".", env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    dd = json.loads(out.read_text())
    assert dd["status"] == "final"
    assert dd["config"]["smoke"] is True
    assert dd["config"]["coins"] == ["ETC", "TRX"]
    assert dd["config"]["grid_bars"] == 3000
    for c in ("ETC", "TRX"):
        assert sum(dd["per_coin"][c][x]["n"] for x in ("low", "mid", "high")) == 3000
    tot = sum(dd["basket"][x]["n"] for x in ("low", "mid", "high"))
    assert tot == 3000
    assert set(dd["vt_recheck"]["rows"]) == {"0.006", "0.012"}
    assert dd["verdict"] == "PENDING_P03_FAIL"
