"""X2 stop-loss / time-stop grid test: schema-level only, recompute by script.

Full X2 is base + 72 runs (smoke=base + 2 via ITER_X2_SMOKE=1). This test
never re-runs the full sweep: schema checks accept either artifact, and
the live subprocess check runs smoke mode only. Conclusion must stay
PENDING (待定) because P0-3 permutation FAILED. Diagnostic only: no
optimum adoption, no live change.
"""
import json
import os
import pathlib
import subprocess
import sys

SEG = {"sharpe", "ann", "mdd", "cum", "n", "turnover"}
ROW = {"scope", "coin", "sl", "ts", "FULL", "H2",
       "d_sharpe", "d_mdd", "d_final"}
COINS = {"ETC", "TRX", "ATOM", "APT", "KAS"}
SL_GRID = [None, 0.03, 0.05, 0.08]
TS_GRID = [12, 24, 36]


def _load():
    p = pathlib.Path("results/iter_X2_stops.json")
    assert p.exists(), "results/iter_X2_stops.json missing; run research/run_iter_x2.py"
    return json.loads(p.read_text())


def test_iter_x2_schema():
    d = _load()
    cfg = d["config"]
    assert set(cfg["weights"]) == COINS
    assert cfg["sl_grid"] == SL_GRID
    assert cfg["ts_grid"] == TS_GRID
    assert cfg["smoke"] in (True, False)
    assert SEG <= set(d["base_FULL"])
    assert SEG <= set(d["base_H2"])
    rows = d["rows"]
    if not cfg["smoke"]:
        assert len(rows) == 72, len(rows)
        per = {(r["coin"], r["sl"], r["ts"]) for r in rows
               if r["scope"] == "per_coin"}
        assert per == {(c, sl, ts) for c in COINS
                       for sl in SL_GRID for ts in TS_GRID}, len(per)
        uni = {(r["sl"], r["ts"]) for r in rows if r["scope"] == "uniform"}
        assert uni == {(sl, ts) for sl in SL_GRID for ts in TS_GRID}, len(uni)
    else:
        assert len(rows) == 2, len(rows)
    for r in rows:
        assert ROW <= set(r), r
        assert r["scope"] in ("per_coin", "uniform")
        if r["scope"] == "per_coin":
            assert r["coin"] in COINS
        else:
            assert r["coin"] is None
        assert r["sl"] in SL_GRID
        assert r["ts"] in TS_GRID
        assert SEG <= set(r["FULL"])
        assert SEG <= set(r["H2"])
        assert r["FULL"]["n"] == cfg["grid_bars"]
        assert r["H2"]["n"] == cfg["h2_len"]
        assert abs(r["FULL"]["sharpe"] - d["base_FULL"]["sharpe"]
                   - r["d_sharpe"]) < 1e-3
        assert abs(r["FULL"]["mdd"] - d["base_FULL"]["mdd"]
                   - r["d_mdd"]) < 1e-3
        assert abs(r["FULL"]["cum"] - d["base_FULL"]["cum"]
                   - r["d_final"]) < 1e-3
    sel = d["selection"]
    assert sel["n_cells"] == len(rows)
    assert sel["n_eligible"] <= len(rows)
    assert sel["found_no_drop_dd_min"] == (d["best_overall"] is not None)
    if d["best_overall"] is not None:
        b = d["best_overall"]
        assert b["FULL"]["sharpe"] >= d["base_FULL"]["sharpe"]
        assert b["FULL"]["mdd"] == min(
            r["FULL"]["mdd"] for r in rows
            if r["FULL"]["sharpe"] >= d["base_FULL"]["sharpe"])
    assert set(d["per_coin_best"]) == COINS
    for c in COINS:
        pb = d["per_coin_best"][c]
        if pb is not None:
            assert pb["sl"] in SL_GRID and pb["ts"] in TS_GRID
            assert pb["FULL"]["sharpe"] >= d["base_FULL"]["sharpe"]
    assert d["decision"] == "PENDING (待定)"
    assert "待定" in d["conclusion"]
    assert "no adoption" in d["conclusion"]


def test_iter_x2_no_broker():
    src = pathlib.Path("research/run_iter_x2.py").read_text()
    for bad in ("place_order", "submit_order", "api_key", "API_KEY",
                "make_broker", "Y1B_LIVE"):
        assert bad.lower() not in src.lower(), bad


def test_iter_x2_smoke_runs_offline(tmp_path):
    out = tmp_path / "iter_X2_stops.json"
    log = tmp_path / "iter_x2.log"
    env = dict(os.environ, ITER_X2_SMOKE="1",
               ITER_X2_OUT=str(out), ITER_X2_LOG=str(log))
    r = subprocess.run([sys.executable, "research/run_iter_x2.py"],
                       capture_output=True, text=True, cwd=".", env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    d = json.loads(out.read_text())
    assert d["config"]["smoke"] is True
    assert len(d["rows"]) == 2
