"""H1 1h baseline anchor (Top5) tests. Recompute by script."""
import importlib.util
import json
import os
import pathlib
import subprocess
import sys

SEG = {"sharpe", "ann", "mdd", "cum", "final_x", "n", "turnover",
       "trades", "entries", "flips", "exits"}
OUT = pathlib.Path(os.getenv("ITER_H1_OUT", "results/iter_H1_base.json"))


def _load():
    assert OUT.exists(), "results/iter_H1_base.json missing; run research/run_iter_h1_base.py"
    return json.loads(OUT.read_text())


def _mod():
    spec = importlib.util.spec_from_file_location(
        "h1mod", "research/run_iter_h1_base.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["h1mod"] = m
    try:
        spec.loader.exec_module(m)
    except SystemExit:
        pass
    return m


def test_iter_h1_base_lock():
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


def test_iter_h1_base_schema():
    d = _load()
    assert d.get("status") == "final"
    cfg = d["config"]
    assert cfg["grid"] == "1h"
    assert cfg["bpy"] == 8760.0
    assert cfg["scale_4h_to_1h"] == 4
    assert cfg["h2_len"] == 730
    assert cfg["h2_start"] == cfg["grid_bars"] - 730
    assert cfg["grid_bars"] > 2000
    assert cfg["smoke"] in (True, False)
    if not cfg["smoke"]:
        assert cfg["coins"] == ["ETC", "TRX", "ATOM", "APT", "KAS"]
        assert set(cfg["weights"]) == {"ETC", "TRX", "ATOM", "APT", "KAS"}
        assert all(abs(w - 0.2) < 1e-12 for w in cfg["weights"].values())
    for c in cfg["coins"]:
        sc = cfg["scaled_specs_1h"][c]
        base = cfg["basket"][c]
        for k in ("cd", "ts", "vw"):
            if base[k] is None:
                assert sc[k] is None
            else:
                assert sc[k] == int(base[k]) * 4
    n = cfg["grid_bars"]
    assert set(d["per_coin"]) == set(cfg["coins"])
    for c, pc in d["per_coin"].items():
        assert SEG <= set(pc["FULL"])
        assert SEG <= set(pc["H2"])
        assert pc["FULL"]["n"] == n
        assert pc["H2"]["n"] == 730
        assert pc["FULL"]["turnover"] >= 0
        assert abs(pc["FULL"]["final_x"] - round(1.0 + pc["FULL"]["cum"], 4)) < 1e-9
        assert pc["FULL"]["trades"] == pc["FULL"]["entries"] + pc["FULL"]["flips"]
        assert pc["H2"]["trades"] == pc["H2"]["entries"] + pc["H2"]["flips"]
        assert pc["FULL"]["trades"] >= pc["H2"]["trades"]
    b = d["basket"]
    assert SEG <= set(b["FULL"])
    assert SEG <= set(b["H2"])
    assert b["FULL"]["n"] == n
    assert b["H2"]["n"] == 730
    assert abs(b["FULL"]["final_x"] - round(1.0 + b["FULL"]["cum"], 4)) < 1e-9
    assert b["FULL"]["trades"] == sum(d["per_coin"][c]["FULL"]["trades"] for c in cfg["coins"])
    assert b["H2"]["trades"] == sum(d["per_coin"][c]["H2"]["trades"] for c in cfg["coins"])
    fx = d["fee2x"]
    assert fx["basket"]["FULL"]["n"] == n
    assert fx["basket"]["H2"]["n"] == 730
    assert fx["basket"]["FULL"]["sharpe"] <= b["FULL"]["sharpe"] + 1e-9
    assert abs(fx["basket"]["gap_FULL_sharpe"] - round(fx["basket"]["FULL"]["sharpe"] - b["FULL"]["sharpe"], 3)) < 1e-9
    assert fx["basket"]["FULL"]["trades"] == b["FULL"]["trades"]
    assert set(fx["per_coin"]) == set(cfg["coins"])
    for c in cfg["coins"]:
        assert fx["per_coin"][c]["FULL"]["trades"] == d["per_coin"][c]["FULL"]["trades"]
    assert d["verdict"] == "PENDING_P03_FAIL"
    assert d["decision"] == "NO_ADOPTION_KEEP_EQUAL"
    assert "PENDING" in d["conclusion"] and "no live change" in d["conclusion"]


def test_iter_h1_base_trade_counts():
    m = _mod()
    assert m.count_trades([0.0, 1.0, 1.0, 0.0, -1.0, -1.0, 0.0], 0, 7) == {
        "trades": 2, "entries": 2, "flips": 0, "exits": 2}
    assert m.count_trades([1.0, -1.0, -1.0, 1.0, 0.0], 0, 5) == {
        "trades": 3, "entries": 1, "flips": 2, "exits": 1}
    assert m.count_trades([0.0, 0.0, 0.0], 0, 3) == {
        "trades": 0, "entries": 0, "flips": 0, "exits": 0}
    # slice-aware: position carried from pos[a-1]
    assert m.count_trades([1.0, 1.0, 1.0], 1, 3)["trades"] == 0
    assert m.count_trades([1.0, 0.0, 1.0], 2, 3)["trades"] == 1
    s = m.seg([0.01, -0.02, 0.03], [0.0, 2.0, 0.0], [0.0, 1.0, 1.0], 0, 3)
    assert s["n"] == 3 and s["trades"] == 1
    assert abs(s["final_x"] - round(1.0 + 0.02, 4)) < 1e-9
    assert s["turnover"] >= 0


def test_iter_h1_base_no_broker():
    src = pathlib.Path("research/run_iter_h1_base.py").read_text().lower()
    for bad in ("place_order", "submit_order", "api_key", "make_broker",
                "y1b_live", "market_open"):
        assert bad not in src, bad


def test_iter_h1_base_smoke_runs_offline(tmp_path):
    out = tmp_path / "iter_H1_base.json"
    lg = tmp_path / "iter_H1_base.log"
    env = dict(os.environ, ITER_H1_SMOKE="1", ITER_H1_OUT=str(out),
               ITER_H1_LOG=str(lg))
    r = subprocess.run([sys.executable, "research/run_iter_h1_base.py"],
                       capture_output=True, text=True, cwd=".", env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    dd = json.loads(out.read_text())
    assert dd["status"] == "final"
    assert dd["config"]["smoke"] is True
    assert dd["config"]["coins"] == ["ETC", "TRX"]
    assert dd["config"]["grid_bars"] == 3000
    assert dd["basket"]["FULL"]["n"] == 3000
    assert dd["basket"]["H2"]["n"] == 730
    assert dd["verdict"] == "PENDING_P03_FAIL"
