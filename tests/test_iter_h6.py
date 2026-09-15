"""H6 1h frequency experiment (Top5) tests: native 1h vs 4h-thinned. Recompute by script."""
import json
import os
import pathlib
import subprocess
import sys

SEG = {"sharpe", "ann", "mdd", "cum", "n", "turnover"}
ARMS = {"native_1h", "thinned_4h"}
OUT = pathlib.Path(os.getenv("ITER_H6_OUT", "results/iter_H6_freq.json"))

def _load():
    assert OUT.exists(), "results/iter_H6_freq.json missing; run research/run_iter_h6_freq.py"
    return json.loads(OUT.read_text())

def test_iter_h6_freq_lock():
    from strategy_manager.config import LEV, FEE, FUND, FORMULA, LOCKED_ETC, LOCKED_TRX, LOCKED_ATOM, LOCKED_APT, LOCKED_KAS
    assert LEV == 2.0
    assert FORMULA == [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10]
    assert (LOCKED_ETC["lth"], LOCKED_ETC["sth"], LOCKED_ETC["cd"], LOCKED_ETC["sl"], LOCKED_ETC["ts"]) == (0.88, 0.12, 18, None, 24)
    assert (LOCKED_TRX["lth"], LOCKED_TRX["sth"], LOCKED_TRX["cd"], LOCKED_TRX["sl"], LOCKED_TRX["ts"]) == (0.85, 0.12, 6, 0.05, 24)
    assert (LOCKED_ATOM["lth"], LOCKED_ATOM["sth"], LOCKED_ATOM["cd"], LOCKED_ATOM["sl"], LOCKED_ATOM["ts"]) == (0.85, 0.15, 6, 0.05, 24)
    assert (LOCKED_APT["lth"], LOCKED_APT["sth"], LOCKED_APT["cd"], LOCKED_APT["sl"], LOCKED_APT["ts"]) == (0.88, 0.12, 18, None, 24)
    assert (LOCKED_KAS["lth"], LOCKED_KAS["sth"], LOCKED_KAS["cd"], LOCKED_KAS["sl"], LOCKED_KAS["ts"]) == (0.88, 0.12, 6, None, 24)

def test_iter_h6_freq_schema():
    d = _load()
    cfg = d["config"]
    assert cfg["grid"] == "1h"
    assert cfg["bpy"] == 8760.0
    assert cfg["scale"] == 4
    assert cfg["thin_every"] == 4
    assert cfg["grid_bars"] > 2000
    assert cfg["smoke"] in (True, False)
    if not cfg["smoke"]:
        assert cfg["coins"] == ["ETC", "TRX", "ATOM", "APT", "KAS"]
        assert set(cfg["weights"]) == {"ETC", "TRX", "ATOM", "APT", "KAS"}
        assert cfg["grid_bars"] == 8760
    assert set(d["arms"]) == ARMS
    n = cfg["grid_bars"]
    for a in ARMS:
        arm = d["arms"][a]
        assert SEG <= set(arm["FULL"])
        assert SEG <= set(arm["H2"])
        assert arm["FULL"]["n"] == n
        assert arm["H2"]["n"] == n - cfg["h2_start"]
        assert arm["FULL"]["turnover"] >= 0
        assert arm["flips"] >= 0 and arm["entries"] >= 0
        for coin, pc in arm["per_coin"].items():
            assert SEG <= set(pc["FULL"])
            assert SEG <= set(pc["H2"])
            assert pc["FULL"]["n"] == n
    assert set(d["fee2x"]) == ARMS
    for a in ARMS:
        fx = d["fee2x"][a]
        assert fx["FULL_sharpe"] <= d["arms"][a]["FULL"]["sharpe"] + 1e-9
        assert abs(fx["gap_FULL_sharpe"] - round(fx["FULL_sharpe"] - d["arms"][a]["FULL"]["sharpe"], 3)) < 1e-9
    cmp = d["compare"]
    fn, ft = d["arms"]["native_1h"]["FULL"], d["arms"]["thinned_4h"]["FULL"]
    assert abs(cmp["d_sharpe_FULL"] - round(fn["sharpe"] - ft["sharpe"], 3)) < 1e-9
    assert abs(cmp["d_turnover_FULL"] - round(fn["turnover"] - ft["turnover"], 6)) < 1e-9
    assert cmp["turnover_ratio_native_over_thinned"] >= 1.0
    assert cmp["d_flips"] == d["arms"]["native_1h"]["flips"] - d["arms"]["thinned_4h"]["flips"]
    assert cmp["d_entries"] == d["arms"]["native_1h"]["entries"] - d["arms"]["thinned_4h"]["entries"]
    rec = d["recommendation"]
    assert rec["Y1B_DECISION_ALIGN"] in ("0", "1")
    exp = "0" if cmp["d_sharpe_FULL"] >= 0.2 else "1"
    assert rec["Y1B_DECISION_ALIGN"] == exp
    assert "shelved" in rec["status"]
    assert d["verdict"] == "PENDING"
    assert d["decision"] == "PENDING"
    assert "PENDING" in d["conclusion"] and "live untouched" in d["conclusion"]

def test_iter_h6_freq_gate_semantics():
    import importlib.util
    spec = importlib.util.spec_from_file_location("h6mod", "research/run_iter_h6_freq.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["h6mod"] = m
    try:
        spec.loader.exec_module(m)
    except SystemExit:
        pass
    d = [1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 0.0, 0.0, 1.0]
    h = m.apply_gate(d, "native_1h")
    a = m.apply_gate(d, "thinned_4h")
    assert h == [0.0] + d[:-1]
    assert a[0] == 0.0 and a[1] == d[0] and a[4] == d[0] and a[5] == d[4]
    assert m.apply_gate([1.0, -1.0, 1.0, -1.0, 1.0], "thinned_4h")[2] == 1.0
    assert m.THIN_EVERY == 4
    assert m.SCALE == 4
    assert m.BPY == 8760.0

def test_iter_h6_freq_no_broker():
    src = pathlib.Path("research/run_iter_h6_freq.py").read_text().lower()
    for bad in ("place_order", "submit_order", "api_key", "make_broker", "y1b_live", "market_open"):
        assert bad not in src, bad

def test_iter_h6_freq_smoke_runs_offline(tmp_path):
    out = tmp_path / "iter_H6_freq.json"
    lg = tmp_path / "iter_h6_freq.log"
    env = dict(os.environ, ITER_H6_SMOKE="1", ITER_H6_OUT=str(out), ITER_H6_LOG=str(lg))
    r = subprocess.run([sys.executable, "research/run_iter_h6_freq.py"], capture_output=True, text=True, cwd=".", env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    dd = json.loads(out.read_text())
    assert dd["config"]["smoke"] is True
    assert set(dd["arms"]) == ARMS
    assert dd["config"]["grid_bars"] == 3000
