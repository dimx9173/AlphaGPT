"""X3 threshold fine-tune test: schema-level only, recompute done by the script.

Full X3 is base + 20 runs (smoke=base + 2 via ITER_X3_SMOKE=1). This test
never re-runs the full sweep: schema checks accept either artifact, and
the live subprocess check runs smoke mode only. Conclusion must stay
PENDING (待定) because P0-3 permutation FAILED. Robustness only: no
optimum pursuit, no parameter adoption.
"""
import json
import os
import pathlib
import subprocess
import sys

SEG = {"sharpe", "ann", "mdd", "cum", "n", "turnover"}
ROW = {"coin", "param", "delta", "new_value", "FULL", "d_sharpe"}
COINS = {"ETC", "TRX", "ATOM", "APT", "KAS"}


def _load():
    p = pathlib.Path("results/iter_X3_thresh.json")
    assert p.exists(), "results/iter_X3_thresh.json missing; run research/run_iter_x3.py"
    return json.loads(p.read_text())


def test_iter_x3_schema():
    d = _load()
    cfg = d["config"]
    assert set(cfg["weights"]) == COINS
    assert cfg["delta"] == 0.02
    assert cfg["flat_tol"] == 0.15
    assert cfg["smoke"] in (True, False)
    rows = d["rows"]
    if not cfg["smoke"]:
        assert len(rows) == 20, len(rows)
        assert {(r["coin"], r["param"], r["delta"]) for r in rows} == {
            (c, p, s) for c in COINS for p in ("lth", "sth") for s in (0.02, -0.02)}
    else:
        assert len(rows) == 2, len(rows)
    assert SEG <= set(d["base_FULL"])
    for r in rows:
        assert ROW <= set(r), r
        assert r["coin"] in COINS
        assert r["param"] in ("lth", "sth")
        assert r["delta"] in (0.02, -0.02)
        assert SEG <= set(r["FULL"])
        assert r["FULL"]["n"] == cfg["grid_bars"]
        assert abs(r["FULL"]["sharpe"] - d["base_FULL"]["sharpe"] - r["d_sharpe"]) < 1e-3
    ds = sorted((abs(r["d_sharpe"]) for r in rows), reverse=True)
    assert [abs(r["d_sharpe"]) for r in rows] == ds, "rows must be |d_sharpe|-desc"
    v = d["verdict"]
    assert v["verdict"] in ("flat(穩健)", "sensitive(敏感)")
    assert v["flat"] in (True, False)
    assert v["flat"] == (v["max_abs_d_sharpe"] <= v["tol"])
    assert v["n_runs"] == len(rows)
    assert "待定" in d["conclusion"]
    assert "no adoption" in d["conclusion"]


def test_iter_x3_no_broker():
    src = pathlib.Path("research/run_iter_x3.py").read_text()
    for bad in ("place_order", "submit_order", "api_key", "API_KEY",
                "make_broker", "Y1B_LIVE"):
        assert bad.lower() not in src.lower(), bad


def test_iter_x3_smoke_runs_offline(tmp_path):
    env = dict(os.environ, ITER_X3_SMOKE="1",
               ITER_X3_OUT=str(tmp_path / "x3.json"), ITER_X3_LOG=str(tmp_path / "x3.log"))
    r = subprocess.run([sys.executable, "research/run_iter_x3.py"],
                       capture_output=True, text=True, cwd=".", env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    d = json.loads((tmp_path / "x3.json").read_text())
    assert d["config"]["smoke"] is True
    assert len(d["rows"]) == 2
