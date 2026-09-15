"""H39 1h session split (Top5) tests. Recompute by script."""
import importlib.util
import json
import os
import pathlib
import subprocess
import sys

SEG = {"sharpe", "ann", "mdd", "cum", "final_x", "n", "turnover",
       "trades", "entries", "flips", "exits"}
OUT = pathlib.Path(os.getenv("ITER_H39_OUT", "results/iter_H39_session.json"))


def _load():
    assert OUT.exists(), "results/iter_H39_session.json missing; run research/run_iter_h39_session.py"
    return json.loads(OUT.read_text())


def _mod():
    spec = importlib.util.spec_from_file_location(
        "h39mod", "research/run_iter_h39_session.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["h39mod"] = m
    try:
        spec.loader.exec_module(m)
    except SystemExit:
        pass
    return m


def test_iter_h39_lock():
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


def test_iter_h39_schema():
    d = _load()
    assert d.get("status") == "final"
    cfg = d["config"]
    assert cfg["grid"] == "1h"
    assert cfg["bpy"] == 8760.0
    assert cfg["bpy_session"] == 2920.0
    assert cfg["scale_4h_to_1h"] == 4
    assert cfg["grid_bars"] > 2000
    assert cfg["smoke"] in (True, False)
    assert cfg["sessions"] == {"asia": list(range(8)),
                               "eu": list(range(8, 16)),
                               "us": list(range(16, 24))}
    assert sum(cfg["session_bars"].values()) == cfg["grid_bars"]
    assert all(v > 100 for v in cfg["session_bars"].values())
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
        assert SEG <= set(pc["FULL"])
        for k in ("asia", "eu", "us"):
            assert SEG <= set(pc[k])
            assert pc[k]["n"] == cfg["session_bars"][k]
            assert pc[k]["trades"] == pc[k]["entries"] + pc[k]["flips"]
        assert pc["FULL"]["n"] == n
        assert abs(pc["FULL"]["final_x"] - round(1.0 + pc["FULL"]["cum"], 4)) < 1e-9
        assert pc["FULL"]["trades"] == pc["FULL"]["entries"] + pc["FULL"]["flips"]
        for fld in ("trades", "entries", "flips", "exits"):
            assert (pc["asia"][fld] + pc["eu"][fld] + pc["us"][fld]) == pc["FULL"][fld], (c, fld)
    assert set(d["arms"]) == {"asia", "eu", "us"}
    ncoins = len(cfg["coins"])
    for k, arm in d["arms"].items():
        assert SEG <= set(arm["basket"])
        assert arm["basket"]["n"] == n
        assert arm["basket"]["turnover"] >= 0
        assert abs(arm["basket"]["final_x"] - round(1.0 + arm["basket"]["cum"], 4)) < 1e-9
        assert set(arm["per_coin"]) == set(cfg["coins"])
        for c in cfg["coins"]:
            assert SEG <= set(arm["per_coin"][c])
            assert arm["per_coin"][c]["n"] == n
        assert arm["basket"]["trades"] == sum(
            arm["per_coin"][c]["trades"] for c in cfg["coins"])
    assert d["verdict"] == "PENDING_P03_FAIL"
    assert d["decision"] == "NO_ADOPTION_KEEP_EQUAL"
    assert "PENDING" in d["conclusion"] and "no live change" in d["conclusion"]


def test_iter_h39_session_helpers():
    m = _mod()
    assert m.session_of_hour(0) == "asia"
    assert m.session_of_hour(7) == "asia"
    assert m.session_of_hour(8) == "eu"
    assert m.session_of_hour(15) == "eu"
    assert m.session_of_hour(16) == "us"
    assert m.session_of_hour(23) == "us"
    # masked trade attribution sums to FULL, attributed by transition bar
    pos = [0.0, 1.0, 1.0, 0.0, -1.0, 0.0]
    idx_a = {0, 1, 2}
    idx_b = {3, 4, 5}
    fa = m.count_trades_masked(pos, idx_a)
    fb = m.count_trades_masked(pos, idx_b)
    full = m.count_trades(pos, 0, 6)
    for fld in ("trades", "entries", "flips", "exits"):
        assert fa[fld] + fb[fld] == full[fld]
    assert full == {"trades": 2, "entries": 2, "flips": 0, "exits": 2}
    # gated leg: zero outside session, boundary costs included
    net, turn, g = m.gated_leg([0.01, 0.02, 0.03], [1.0, 1.0, 1.0], {0, 2}, 0.0004, 0.0005)
    assert g == [1.0, 0.0, 1.0]
    assert turn[1] == 1.0 and turn[2] == 1.0
    assert abs(net[0] - (0.01 * 2.0 - 1.0 * 0.0004 * 2.0 - 1.0 * 0.0005 * 2.0)) < 1e-12
    # boundary exit at bar 1 still pays turnover fee on the flat bar
    assert abs(net[1] - (-1.0 * 0.0004 * 2.0)) < 1e-12
    assert abs(net[2] - (0.03 * 2.0 - 1.0 * 0.0004 * 2.0 - 1.0 * 0.0005 * 2.0)) < 1e-12
    # session slice annualisation uses BPY_SESS
    s = m.seg_subset([0.01, -0.02, 0.03], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0, 1, 2])
    assert s["n"] == 3
    assert abs(s["final_x"] - round(1.0 + 0.02, 4)) < 1e-9
    assert s["trades"] == s["entries"] + s["flips"]


def test_iter_h39_no_broker():
    src = pathlib.Path("research/run_iter_h39_session.py").read_text().lower()
    for bad in ("place_order", "submit_order", "api_key", "make_broker",
                "y1b_live", "market_open"):
        assert bad not in src, bad


def test_iter_h39_smoke_runs_offline(tmp_path):
    out = tmp_path / "iter_H39_session.json"
    lg = tmp_path / "iter_H39_session.log"
    env = dict(os.environ, ITER_H39_SMOKE="1", ITER_H39_OUT=str(out),
               ITER_H39_LOG=str(lg))
    r = subprocess.run([sys.executable, "research/run_iter_h39_session.py"],
                       capture_output=True, text=True, cwd=".", env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    dd = json.loads(out.read_text())
    assert dd["status"] == "final"
    assert dd["config"]["smoke"] is True
    assert dd["config"]["coins"] == ["ETC", "TRX"]
    assert dd["config"]["grid_bars"] == 3000
    assert sum(dd["config"]["session_bars"].values()) == 3000
    assert set(dd["arms"]) == {"asia", "eu", "us"}
    for c, pc in dd["per_coin"].items():
        for fld in ("trades", "entries", "flips", "exits"):
            assert pc["asia"][fld] + pc["eu"][fld] + pc["us"][fld] == pc["FULL"][fld]
    assert dd["verdict"] == "PENDING_P03_FAIL"
