"""H38 1h vol-target sweep (Top5) tests. Schema + gaps + smoke rerun."""
import importlib.util
import json
import os
import pathlib
import subprocess
import sys

OUT = pathlib.Path(os.getenv("ITER_H38_OUT", "results/iter_H38_vt.json"))
COINS = ["ETC", "TRX", "ATOM", "APT", "KAS"]
VT_KEYS = ["None", "0.003", "0.006", "0.012", "0.024"]


def _load():
    assert OUT.exists(), "results/iter_H38_vt.json missing; run research/run_iter_h38_vt.py"
    return json.loads(OUT.read_text())


def _mod():
    spec = importlib.util.spec_from_file_location(
        "h38mod", "research/run_iter_h38_vt.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["h38mod"] = m
    try:
        spec.loader.exec_module(m)
    except SystemExit:
        pass
    return m


def test_iter_h38_lock():
    from strategy_manager.config import (
        LEV, FEE, FUND, FORMULA,
        LOCKED_ETC, LOCKED_TRX, LOCKED_ATOM, LOCKED_APT, LOCKED_KAS)
    assert LEV == 2.0
    assert FUND == 0.0005
    assert FEE == 0.0004
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


def test_iter_h38_schema():
    d = _load()
    assert d.get("status") == "final"
    cfg = d["config"]
    assert cfg["grid"] == "1h"
    assert cfg["bpy"] == 8760.0
    assert cfg["scale_4h_to_1h"] == 4
    assert cfg["grid_bars"] == 8760
    assert cfg["vt_grid"] == VT_KEYS
    assert cfg["coins"] == COINS
    assert cfg["smoke"] is False
    n = cfg["grid_bars"]
    for c in COINS:
        sc = cfg["scaled_specs_1h"][c]
        base = cfg["basket"][c]
        for k in ("cd", "ts", "vw"):
            if base[k] is None:
                assert sc[k] is None, (c, k)
            else:
                assert sc[k] == int(base[k]) * 4, (c, k)
    assert set(d["rows"]) == set(VT_KEYS)
    for key in VT_KEYS:
        row = d["rows"][key]
        if key == "None":
            assert row["vt"] is None
        else:
            assert row["vt"] == float(key)
        assert set(row["per_coin"]) == set(COINS)
        for c in COINS:
            pc = row["per_coin"][c]
            assert {"sharpe", "ann", "mdd", "cum", "n", "turnover"} <= set(pc)
            assert pc["n"] == n
        b = row["basket"]
        assert {"sharpe", "ann", "mdd", "cum", "n", "turnover"} <= set(b)
        assert b["n"] == n
        assert b["mdd"] >= 0.0
        assert b["turnover"] >= 0.0
    assert d["base_vt"] == "None"
    assert d["verdict"] == "PENDING_P03_FAIL"
    assert d["decision"] == "NO_ADOPTION_KEEP_EQUAL"
    assert "PENDING" in d["conclusion"] and "no live change" in d["conclusion"]


def test_iter_h38_gaps_recompute():
    d = _load()
    base = d["rows"]["None"]["basket"]
    for key in VT_KEYS:
        row = d["rows"][key]
        b = row["basket"]
        assert abs(row["gap_vs_none_sharpe"] - round(b["sharpe"] - base["sharpe"], 3)) < 1e-9
        assert abs(row["gap_vs_none_mdd"] - round(b["mdd"] - base["mdd"], 4)) < 1e-9
        assert abs(row["gap_vs_none_turnover"] - round(b["turnover"] - base["turnover"], 6)) < 1e-9
    assert d["rows"]["None"]["gap_vs_none_sharpe"] == 0.0


def test_iter_h38_helpers():
    m = _mod()
    assert m.vt_key(None) == "None"
    assert m.vt_key(0.006) == "0.006"
    assert m.SPECS["ETC"]["cd"] == 18 * 4
    assert m.SPECS["TRX"]["cd"] == 6 * 4
    assert m.SPECS["ETC"]["ts"] == 24 * 4
    assert m.SPECS["ETC"]["vw"] == 12 * 4
    assert m.SPECS["ETC"]["vt"] is None
    s = m.seg([0.01, -0.02, 0.03, 0.01], [0.0, 1.0, 0.0, 2.0], 0, 4)
    assert s["n"] == 4
    assert abs(s["cum"] - round(0.01 - 0.02 + 0.03 + 0.01, 4)) < 1e-9
    assert abs(s["turnover"] - round(3.0 / 4, 6)) < 1e-9
    assert s["mdd"] >= 0.0
    src = pathlib.Path("research/run_iter_h38_vt.py").read_text().lower()
    for bad in ("place_order", "submit_order", "api_key", "make_broker",
                "y1b_live", "market_open"):
        assert bad not in src, bad


def test_iter_h38_smoke_runs_offline(tmp_path):
    out = tmp_path / "iter_H38_vt.json"
    lg = tmp_path / "iter_H38_vt.log"
    env = dict(os.environ, ITER_H38_SMOKE="1", ITER_H38_OUT=str(out),
               ITER_H38_LOG=str(lg))
    r = subprocess.run([sys.executable, "research/run_iter_h38_vt.py"],
                       capture_output=True, text=True, cwd=".", env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    dd = json.loads(out.read_text())
    assert dd["status"] == "final"
    assert dd["config"]["smoke"] is True
    assert dd["config"]["coins"] == ["ETC", "TRX"]
    assert dd["config"]["grid_bars"] == 3000
    assert set(dd["rows"]) == set(VT_KEYS)
    for key in VT_KEYS:
        assert dd["rows"][key]["basket"]["n"] == 3000
        assert set(dd["rows"][key]["per_coin"]) == {"ETC", "TRX"}
    assert dd["verdict"] == "PENDING_P03_FAIL"
