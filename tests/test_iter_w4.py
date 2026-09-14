"""W4 regime split (15m) tests: bull/bear via BTC close>=MA960 + FULL + funding split."""
import json
import os
import pathlib
import subprocess
import sys

OUT = pathlib.Path("results/iter_W4_regime.json")
COINS = ["ETC", "TRX", "ATOM", "APT", "KAS"]

def _load():
    assert OUT.exists(), "results/iter_W4_regime.json missing; run research/run_iter_w4_regime.py"
    return json.loads(OUT.read_text())

def test_iter_w4_schema():
    d = _load()
    assert d["status"] == "done"
    assert d["verdict"] == "PENDING"
    assert d["decision"] == "NO_ADOPTION"
    assert d["config"]["grid"] == "15m"
    assert d["config"]["grid_bars"] == 35039
    assert d["config"]["bpy"] == 35040.0
    assert d["config"]["scale"] == 16
    assert d["config"]["ma_window"] == 960
    assert d["config"]["lev"] == 2.0
    assert d["config"]["lev_locked"] == 2.0
    assert d["config"]["fee"] == 0.0004
    assert d["config"]["fund"] == 0.0005
    assert d["config"]["smoke"] is False
    assert set(d["config"]["weights"]) == set(COINS)
    assert set(d["per_coin"]) == set(COINS)
    for c in COINS:
        for k in ("FULL", "bull", "bear"):
            for f in ("sharpe", "ann", "mdd", "cum", "n", "coverage", "turnover"):
                assert f in d["per_coin"][c][k], (c, k, f)
    for k in ("FULL", "bull", "bear"):
        assert "sharpe" in d["basket"][k]
    for k in ("fund_pos_pay", "fund_neg_earn"):
        assert "sharpe" in d["funding_split"][k]
    assert "fund_zero_n" in d["funding_split"]

def test_iter_w4_regime_partition():
    d = _load()
    n = d["config"]["grid_bars"]
    r = d["regime"]
    assert r["bull_n"] + r["bear_n"] == n
    assert abs(r["bull_coverage"] + r["bear_coverage"] - 1.0) < 1e-6
    assert 0 < r["bull_n"] < n and 0 < r["bear_n"] < n
    for c in COINS:
        pc = d["per_coin"][c]
        assert pc["bull"]["n"] + pc["bear"]["n"] == n
        assert pc["FULL"]["n"] == n
        assert abs(pc["bull"]["coverage"] + pc["bear"]["coverage"] - 1.0) < 1e-6
    b = d["basket"]
    assert b["bull"]["n"] + b["bear"]["n"] == n
    assert b["FULL"]["n"] == n
    fs = d["funding_split"]
    assert fs["fund_pos_pay"]["n"] + fs["fund_neg_earn"]["n"] + fs["fund_zero_n"] == n
    assert fs["fund_neg_earn"]["n"] > 0  # pay side may be empty (book rarely net-long); only the partition sum is strict

def test_iter_w4_funding_consistency():
    d = _load()
    # basket FULL cum must equal mean of per-coin FULL cums under equal 0.2 weights
    mean_cum = sum(d["per_coin"][c]["FULL"]["cum"] for c in COINS) / len(COINS)
    assert abs(d["basket"]["FULL"]["cum"] - mean_cum) < 5e-4  # JSON round(4) on cum legs
    # full-year ann consistent with cum/n*bpy
    full = d["basket"]["FULL"]
    assert abs(full["ann"] - full["cum"] / full["n"] * 35040.0) < 1e-3

def test_iter_w4_script_runs_smoke():
    env = dict(os.environ, ITER_W4_SMOKE="1", ITER_W4_OUT="/tmp/iter_W4_smoke.json",
               ITER_W4_LOG="/tmp/iter_W4_smoke.log")
    r = subprocess.run([sys.executable, "research/run_iter_w4_regime.py"],
                       capture_output=True, text=True, cwd=".", env=env, timeout=1200)
    assert r.returncode == 0, r.stderr[-2000:]
    d = json.loads(pathlib.Path("/tmp/iter_W4_smoke.json").read_text())
    assert d["status"] == "done"
    assert d["config"]["smoke"] is True
    assert set(d["per_coin"]) == {"ETC", "TRX"}
    assert d["verdict"] == "PENDING"
