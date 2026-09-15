"""H18 1h signal persistence tests: pure math + artifact schema + smoke run."""
import json, os, pathlib, subprocess, sys

OUT = pathlib.Path(os.getenv("ITER_H18_OUT", "results/iter_H18_persist.json"))

def _load():
    assert OUT.exists(), "results/iter_H18_persist.json missing; run research/run_iter_h18_persist.py"
    return json.loads(OUT.read_text())

def _mod():
    import importlib.util
    spec = importlib.util.spec_from_file_location("h18mod", "research/run_iter_h18_persist.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["h18mod"] = m
    try:
        spec.loader.exec_module(m)
    except SystemExit:
        pass
    return m

def test_iter_h18_lock():
    from strategy_manager.config import LEV, FORMULA, LOCKED_ETC, LOCKED_TRX, LOCKED_ATOM, LOCKED_APT, LOCKED_KAS
    assert LEV == 2.0
    assert FORMULA == [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10]
    assert (LOCKED_ETC["lth"], LOCKED_ETC["sth"], LOCKED_ETC["cd"], LOCKED_ETC["sl"], LOCKED_ETC["ts"]) == (0.88, 0.12, 18, None, 24)
    assert (LOCKED_TRX["lth"], LOCKED_TRX["sth"], LOCKED_TRX["cd"], LOCKED_TRX["sl"], LOCKED_TRX["ts"]) == (0.85, 0.12, 6, 0.05, 24)
    assert (LOCKED_ATOM["lth"], LOCKED_ATOM["sth"], LOCKED_ATOM["cd"], LOCKED_ATOM["sl"], LOCKED_ATOM["ts"]) == (0.85, 0.15, 6, 0.05, 24)
    assert (LOCKED_APT["lth"], LOCKED_APT["sth"], LOCKED_APT["cd"], LOCKED_APT["sl"], LOCKED_APT["ts"]) == (0.88, 0.12, 18, None, 24)
    assert (LOCKED_KAS["lth"], LOCKED_KAS["sth"], LOCKED_KAS["cd"], LOCKED_KAS["sl"], LOCKED_KAS["ts"]) == (0.88, 0.12, 6, None, 24)

def test_iter_h18_math_units():
    m = _mod()
    assert m.acf([1.0, 2.0, 3.0, 4.0], 1) > 0.9
    assert abs(m.acf([1.0, 1.0, 1.0], 1)) < 1e-9
    assert m.acf([1.0], 1) == 0.0
    assert m.acf([1.0, -1.0, 1.0, -1.0], 1) < -0.9
    assert m.holding_runs([1, 1, 0, -1, -1, -1, 0, 1]) == [2, 3, 1]
    assert m.holding_runs([0, 0, 0]) == []
    ev, nd = m.reversal_events([1, 1, -1, -1, 1])
    assert ev == [2, 4] and nd == 2
    evf, ndf = m.reversal_events([1, 0, -1])
    assert evf == [2] and ndf == 0
    ev2, nd2 = m.reversal_events([1, -1])
    assert ev2 == [1] and nd2 == 1
    fc = m.flip_cluster([10, 20, 130], gap=24, win=96)
    assert fc["n_reversals"] == 3 and fc["max_in_window"] == 2
    assert m.flip_cluster([10, 20, 100], gap=24, win=96)["max_in_window"] == 3
    tr = m.trade_segments([0, 1, 1, 0, -1, 0], [0, 0.1, 0.2, 0, 0.3, 0])
    assert len(tr) == 2 and tr[0][0] == 2 and abs(tr[0][1] - 0.3) < 1e-9
    assert m.median([3.0, 1.0, 2.0]) == 2.0
    assert m.median([]) == 0.0

def test_iter_h18_schema():
    d = _load()
    assert d["verdict"] == "PENDING"
    cfg = d["config"]
    assert cfg["grid"] == "1h" and cfg["bpy"] == 8760.0 and cfg["scale"] == 4
    assert cfg["grid_bars"] > 2000
    assert cfg["smoke"] in (True, False)
    if not cfg["smoke"]:
        assert cfg["coins"] == ["ETC", "TRX", "ATOM", "APT", "KAS"]
        assert cfg["grid_bars"] == 8760
    n = cfg["grid_bars"]
    assert d["status"] == "done"
    for c in cfg["coins"]:
        pc = d["per_coin"][c]
        assert len(pc["acf_1_96"]) == 96
        assert all(-1.0 <= v <= 1.0 for v in pc["acf_1_96"])
        assert abs(pc["acf_lag1"] - pc["acf_1_96"][0]) < 1e-9
        assert set(pc["sig_acf"]) == {"1", "2", "4", "6", "12", "24", "48", "72", "96"}
        h = pc["holding"]
        assert h["median_bars"] >= 0 and h["max_bars"] >= h["median_bars"] and h["n_runs"] >= 0
        f = pc["flips"]
        assert f["n_reversals"] >= f["n_direct"] and f["n_via_flat"] == f["n_reversals"] - f["n_direct"]
        assert 0.0 <= f["cluster_share_le24"] <= 1.0 and f["max_reversals_in_96b"] >= 0
        t = pc["trades"]
        assert t["n_hold_le24"] + t["n_hold_gt24"] == t["n"]
        assert -1.0 <= t["corr_hold_pnl"] <= 1.0
        assert pc["FULL"]["n"] == n
    assert d["portfolio"]["FULL"]["n"] == n
    assert -1.0 <= d["persistence_vs_pnl"]["corr_median_hold_vs_sharpe"] <= 1.0
    assert set(d["persistence_vs_pnl"]["by_coin"]) == set(cfg["coins"])

def test_iter_h18_no_broker():
    src = pathlib.Path("research/run_iter_h18_persist.py").read_text().lower()
    for bad in ("place_order", "submit_order", "api_key", "make_broker", "y1b_live", "market_open"):
        assert bad not in src, bad

def test_iter_h18_smoke_runs_offline(tmp_path):
    out = tmp_path / "iter_H18_persist.json"
    lg = tmp_path / "iter_h18_persist.log"
    env = dict(os.environ, ITER_H18_SMOKE="1", ITER_H18_OUT=str(out), ITER_H18_LOG=str(lg))
    r = subprocess.run([sys.executable, "research/run_iter_h18_persist.py"], capture_output=True, text=True, cwd=".", env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    dd = json.loads(out.read_text())
    assert dd["config"]["smoke"] is True
    assert dd["config"]["grid_bars"] == 3000
    assert set(dd["per_coin"]) == {"ETC", "TRX"}
    assert dd["verdict"] == "PENDING"
