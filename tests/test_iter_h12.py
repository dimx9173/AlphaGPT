"""H12 1h turnover decomposition tests. Recompute by script."""
import importlib.util
import json
import os
import pathlib
import subprocess
import sys

OUT = pathlib.Path(os.getenv("ITER_H12_OUT", "results/iter_H12_costsplit.json"))


def _load():
    assert OUT.exists(), "results/iter_H12_costsplit.json missing; run research/run_iter_h12_costsplit.py"
    return json.loads(OUT.read_text())


def _mod():
    spec = importlib.util.spec_from_file_location(
        "h12mod", "research/run_iter_h12_costsplit.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["h12mod"] = m
    try:
        spec.loader.exec_module(m)
    except SystemExit:
        pass
    return m


def test_iter_h12_lock():
    from strategy_manager.config import (
        LEV, FEE, FUND, FORMULA,
        LOCKED_ETC, LOCKED_TRX, LOCKED_ATOM, LOCKED_APT, LOCKED_KAS)
    assert LEV == 2.0
    assert abs(FUND - 0.0005) < 1e-12
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


def test_iter_h12_split_units():
    m = _mod()
    # bar 0 uses prev = pos[-1] (engine roll(1) wrap): all-zero tail pos
    # means no phantom exit; trailing nonzero pos[-1] counts bar-0 exit.
    assert [round(v, 9) for v in m.split_turnover([0.0, 1.0, 1.0, 0.0])[1]] == [0.0, 0.0, 0.0, 1.0]
    e, x, f, r = m.split_turnover([0.0, 1.0, 1.0, 0.0])
    assert (sum(e), sum(x), sum(f), sum(r)) == (1.0, 1.0, 0.0, 0.0)
    e, x, f, r = m.split_turnover([1.0, -1.0, -1.0, 1.0, 0.0])
    assert sum(e) == 1.0 and sum(f) == 4.0 and sum(x) == 1.0 and sum(r) == 0.0
    e, x, f, r = m.split_turnover([0.0, 0.0, 0.0])
    assert sum(e) + sum(x) + sum(f) + sum(r) == 0.0
    # split sums to position turnover exactly
    pos = [0.0, 1.0, 1.0, -1.0, 0.0, -1.0, -1.0, 0.0]
    e, x, f, r = m.split_turnover(pos)
    tot = sum(abs(pos[t] - (pos[t - 1] if t > 0 else 0.0)) for t in range(len(pos)))
    assert abs(sum(e) + sum(x) + sum(f) + sum(r) - tot) < 1e-9
    assert m.holding_runs([0.0, 1.0, 1.0, 0.0, -1.0]) == [2, 1]
    assert m.holding_runs([1.0, -1.0, -1.0]) == [1, 2]
    assert m.median([3.0, 1.0, 2.0]) == 2.0
    assert m.median([]) == 0.0


def test_iter_h12_schema():
    d = _load()
    assert d.get("status") == "final"
    cfg = d["config"]
    assert cfg["grid"] == "1h" and cfg["bpy"] == 8760.0
    assert cfg["scale_4h_to_1h"] == 4 and cfg["h2_len"] == 730
    assert cfg["h2_start"] == cfg["grid_bars"] - 730
    assert cfg["grid_bars"] > 2000
    assert abs(cfg["slip_assumed"] - 0.0005) < 1e-12
    assert cfg["smoke"] in (True, False)
    if not cfg["smoke"]:
        assert cfg["coins"] == ["ETC", "TRX", "ATOM", "APT", "KAS"]
    n = cfg["grid_bars"]
    assert set(d["per_coin"]) == set(cfg["coins"])
    for c, pc in d["per_coin"].items():
        for label, nn in (("FULL", n), ("H2", 730)):
            b = pc[label]
            assert b["seg"]["n"] == nn
            t = b["turnover"]
            assert abs(t["entry"] + t["exit"] + t["flip"] + t["resid"] - t["total"]) < 1e-3
            assert abs(t["check_split_eq_total"]) < 1e-3
            assert t["total"] >= 0
            co = b["cost"]
            assert co["total_abs"] >= 0
            assert abs(co["fee_cum"] + co["slip_cum"] + co["fund_cum_abs"] - co["total_abs"]) < 1e-3
            assert 0.0 <= co["share_fee"] <= 1.0
            assert 0.0 <= co["share_slip"] <= 1.0
            assert 0.0 <= co["share_fund_abs"] <= 1.0
            assert abs(co["share_fee"] + co["share_slip"] + co["share_fund_abs"] - 1.0) < 1e-3 or co["total_abs"] == 0
            assert 0.0 <= co["share_of_basket_abs"] <= 1.0
            assert 0.0 <= t["share_of_basket"] <= 1.0
            assert b["hold"]["n_runs"] >= 0 and b["hold"]["max"] >= 0
    b = d["basket"]
    for label, nn in (("FULL", n), ("H2", 730)):
        assert b[label]["n"] == nn
        ts = b[label]["turnover_split"]
        assert abs(ts["entry"] + ts["exit"] + ts["flip"] - ts["total"]) < 1e-3
        co = b[label]["cost"]
        assert abs(co["fee_cum"] + co["slip_cum"] + co["fund_cum_abs"] - co["total_abs"]) < 1e-3
    assert d["verdict"] == "PENDING"
    assert d["decision"] == "NO_ADOPTION_KEEP_EQUAL"
    assert "PENDING" in d["conclusion"] and "no live change" in d["conclusion"]


def test_iter_h12_no_broker():
    src = pathlib.Path("research/run_iter_h12_costsplit.py").read_text().lower()
    for bad in ("place_order", "submit_order", "api_key", "make_broker",
                "y1b_live", "market_open"):
        assert bad not in src, bad


def test_iter_h12_smoke_runs_offline(tmp_path):
    out = tmp_path / "iter_H12_costsplit.json"
    lg = tmp_path / "iter_H12_costsplit.log"
    env = dict(os.environ, ITER_H12_SMOKE="1", ITER_H12_OUT=str(out),
               ITER_H12_LOG=str(lg))
    r = subprocess.run([sys.executable, "research/run_iter_h12_costsplit.py"],
                       capture_output=True, text=True, cwd=".", env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    dd = json.loads(out.read_text())
    assert dd["status"] == "final"
    assert dd["config"]["smoke"] is True
    assert dd["config"]["coins"] == ["ETC", "TRX"]
    assert dd["config"]["grid_bars"] == 3000
    assert dd["basket"]["FULL"]["n"] == 3000
    assert dd["basket"]["H2"]["n"] == 730
    assert dd["verdict"] == "PENDING"
