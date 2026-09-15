"""H82 1h OBV-slope v5 gate (Top5) tests: legs gated by OBV slope sign (384-bar) v5.

DIAGNOSTIC ONLY: P0-3 permutation FAILED => verdict PENDING, no adoption,
live untouched. Offline read-only (data/data_1y/1h only).
"""
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tempfile

OUT = pathlib.Path(os.getenv("ITER_H82_OUT", "results/iter_H82_obv.json"))
SRC = pathlib.Path("research/run_iter_h82_obv.py")
COINS = ["ETC", "TRX", "ATOM", "APT", "KAS"]


def _load(path=OUT):
    assert path.exists(), "%s missing; run research/run_iter_h82_obv.py" % path
    return json.loads(path.read_text())


def _mod():
    spec = importlib.util.spec_from_file_location("run_iter_h82_obv", str(SRC))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_iter_h82_locks():
    from strategy_manager.config import FORMULA, LEV, FUND, FEE
    assert FORMULA == [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10]
    assert LEV == 2.0
    assert abs(FUND - 0.0005) < 1e-12
    assert abs(FEE - 0.0004) < 1e-12


def test_iter_h82_schema():
    d = _load()
    assert d["status"] == "done"
    assert d["verdict"] == "PENDING"
    assert d["decision"] == "NO_ADOPTION"
    assert d["config"]["grid"] == "1h"
    assert d["config"]["grid_bars"] == 8760
    assert d["config"]["bpy"] == 8760.0
    assert d["config"]["scale"] == 4
    assert d["config"]["obv_win"] == 384
    assert d["config"]["q"] == 0.3
    assert d["config"]["coins"] == COINS
    assert set(d["config"]["weights"]) == set(COINS)
    assert set(d["per_coin"]) == set(COINS)
    for c in COINS:
        e = d["per_coin"][c]
        for leg in ("baseline", "gated"):
            for sgm in ("FULL", "LONG", "SHORT"):
                for f in ("sharpe", "ann", "mdd", "cum", "n", "turnover"):
                    assert f in e[leg][sgm], (c, leg, sgm, f)
                assert e[leg][sgm]["n"] == 8760
            assert e[leg]["FULL"]["n"] == 8760
            assert e[leg]["additivity_err"] < 1e-6, (c, leg, e[leg]["additivity_err"])
        assert e["gate"]["win"] == 384
        assert 0.0 <= e["gate"]["pass_rate"] <= 1.0
        assert e["gate"]["pass_bars"] <= 8760
        assert "d_sharpe_full" in e["delta"]
    for leg in ("baseline", "gated"):
        for sgm in ("FULL", "LONG", "SHORT"):
            assert d["basket"][leg][sgm]["n"] == 8760
        assert d["basket"][leg]["additivity_err"] < 1e-6
    assert "d_sharpe_full" in d["basket"]["delta"]


def test_iter_h82_obv_gate_unit():
    m = _mod()
    n = 600
    up_closes = [float(100 + i) for i in range(n)]
    vols = [1000.0] * n
    gate, _ = m.obv_gate(up_closes, vols, 384)
    assert gate[:384] == [1.0] * 384
    assert all(g == 1.0 for g in gate[384:]), "steady uptrend must pass"
    dn_closes = [float(300 - i) for i in range(n)]
    gate2, _ = m.obv_gate(dn_closes, vols, 384)
    assert gate2[:384] == [1.0] * 384
    assert all(g == 0.0 for g in gate2[384:]), "steady downtrend must block"
    flat_closes = [100.0] * n
    gate3, _ = m.obv_gate(flat_closes, vols, 384)
    assert all(g == 0.0 for g in gate3[384:]), "flat OBV slope (0) must block"


def test_iter_h82_script_runs_offline_smoke():
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "h82.json")
        log = os.path.join(td, "h82.log")
        env = dict(os.environ, ITER_H82_SMOKE="1", ITER_H82_OUT=out,
                   ITER_H82_LOG=log)
        r = subprocess.run([sys.executable, "research/run_iter_h82_obv.py"],
                           capture_output=True, text=True, cwd=".", env=env)
        assert r.returncode == 0, r.stderr[-2000:]
        dd = json.loads(open(out).read())
        assert dd["config"]["smoke"] is True
        assert dd["config"]["coins"] == ["ETC", "TRX"]
        assert dd["config"]["grid_bars"] == 3000
        assert set(dd["per_coin"]) == {"ETC", "TRX"}
        assert dd["status"] == "done"
        assert dd["verdict"] == "PENDING"
        assert dd["decision"] == "NO_ADOPTION"
        assert json.loads(open(out).read())["basket"]["gated"]["FULL"]["n"] == 3000
