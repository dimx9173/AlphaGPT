"""X1 weight-optimization test: schema-level only, recompute by script.

Full X1 = 6 candidates (equal/kas10/invvol/invvol_cap/meanvar/riskpar);
smoke = equal + kas10 via ITER_X1_SMOKE=1. Schema checks accept either
artifact; the live subprocess check runs smoke mode only (full rerun is
~6s so it runs in-process too). Conclusion must stay PENDING (待定, P0-3
FAIL): no default change, no demo evidence.
"""
import json
import os
import pathlib
import subprocess
import sys

SEG = {"sharpe", "ann", "mdd", "cum", "n", "turnover"}
ROW = {"weights", "FULL", "H2", "FULL_fee2x", "fold12", "eligible", "method", "lev_scale"}
COINS = {"ETC", "TRX", "ATOM", "APT", "KAS"}
FULL6 = {"equal", "kas10", "invvol", "invvol_cap", "meanvar", "riskpar"}
SMOKE2 = {"equal", "kas10"}


def _load():
    p = pathlib.Path("results/iter_X1_weights.json")
    assert p.exists(), "results/iter_X1_weights.json missing; run research/run_iter_x1_weights.py"
    return json.loads(p.read_text())


def test_iter_x1_schema():
    d = _load()
    cfg = d["config"]
    assert set(cfg["coins"]) == COINS
    assert cfg["box"] == [0.05, 0.40]
    assert cfg["smoke"] in (True, False)
    rows = d["rows"]
    want = SMOKE2 if cfg["smoke"] else FULL6
    assert set(rows) == want, (set(rows), cfg["smoke"])
    for name, r in rows.items():
        assert ROW <= set(r), name
        assert set(r["weights"]) == COINS
        assert abs(sum(r["weights"].values()) - 1.0) < 1e-3, name
        for w in r["weights"].values():
            assert 0.05 - 1e-3 <= w <= 0.40 + 1e-3, (name, w)
        for k in ("FULL", "H2", "FULL_fee2x"):
            assert SEG <= set(r[k]), (name, k)
        assert r["FULL"]["n"] == cfg["grid_bars"], name
        assert r["FULL_fee2x"]["n"] == cfg["grid_bars"], name
        assert r["H2"]["n"] == cfg["h2_len"], name
        assert len(r["fold12"]["sharpes"]) == 12
        assert r["FULL"]["turnover"] < 0.16, name
        exp_elig = bool(r["FULL"]["turnover"] < 0.16 and r["fold12"]["median"] >= 1.5)
        assert r["eligible"] is exp_elig, name
    assert d["verdict"] == "PENDING_P03_FAIL"
    assert d["decision"] == "KEEP_equal"
    sel = d["selection"]
    if sel["eligible"]:
        assert sel["best"] in sel["eligible"]
        assert sel["best_FULL_sharpe"] == rows[sel["best"]]["FULL"]["sharpe"]
        best = max(sel["eligible"], key=lambda k: rows[k]["FULL"]["sharpe"])
        assert sel["best"] == best
    else:
        assert sel["best"] is None
    assert "待定" in d["decision_note"]


def test_iter_x1_kas10_fixed():
    d = _load()
    assert d["rows"]["kas10"]["weights"] == {
        "ETC": 0.225, "TRX": 0.225, "ATOM": 0.225, "APT": 0.225, "KAS": 0.1}


def test_iter_x1_full_candidates_present():
    d = _load()
    if d["config"]["smoke"]:
        import pytest
        pytest.skip("smoke artifact: meanvar/riskpar checked on full run")
    mv = d["rows"]["meanvar"]["weights"]
    rp = d["rows"]["riskpar"]["weights"]
    assert set(mv) == COINS and set(rp) == COINS
    for w in list(mv.values()) + list(rp.values()):
        assert 0.05 - 1e-3 <= w <= 0.40 + 1e-3
    assert d["rows"]["meanvar"]["method"] == "SLSQP-maxSharpe"
    assert d["rows"]["riskpar"]["method"] == "cyclical-sqrt"
    assert abs(sum(mv.values()) - 1.0) < 1e-3
    assert abs(sum(rp.values()) - 1.0) < 1e-3


def test_iter_x1_no_broker():
    src = pathlib.Path("research/run_iter_x1_weights.py").read_text()
    for bad in ("place_order", "submit_order", "api_key", "make_broker", "Y1B_LIVE"):
        assert bad.lower() not in src.lower(), bad


def test_iter_x1_script_runs_offline():
    p = pathlib.Path("results/iter_X1_weights.json")
    bak = p.read_text() if p.exists() else None
    try:
        r = subprocess.run([sys.executable, "research/run_iter_x1_weights.py"],
                           capture_output=True, text=True, cwd=".")
        assert r.returncode == 0, r.stderr[-2000:]
        d = _load()
        assert d["config"]["smoke"] is False
        assert set(d["rows"]) == FULL6
    finally:
        if bak is not None:
            p.write_text(bak)
