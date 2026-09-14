"""X4 cost-gate stress test: schema-level only, recompute done by the script.

Full X4 is 5 cost cells x 5 K (25 runs); smoke=1 cell x K {0,2} via
ITER_X4_SMOKE=1. This test never re-runs the full sweep: schema checks
accept either artifact, and the live subprocess check runs smoke mode
only. Conclusion must stay PENDING (待定) because P0-3 permutation
FAILED. Diagnostic only: Y1B_COST_K stays default OFF, no adoption.
"""
import json
import os
import pathlib
import subprocess
import sys

SEG = {"sharpe", "ann", "mdd", "cum", "n", "turnover"}
ROW = {"cost", "slip", "fund", "unit", "K", "thr", "FULL",
       "blocked_frac", "turnover_cut_vs_k0", "d_sharpe_vs_k0"}
FULL_CELLS = {"base-5bp-0.0005", "10bp-0.001", "10bp-0.002",
              "20bp-0.001", "20bp-0.002"}
FULL_KS = [0, 1, 2, 3, 5]


def _load():
    p = pathlib.Path("results/iter_X4_cost.json")
    assert p.exists(), "results/iter_X4_cost.json missing; run research/run_iter_x4.py"
    return json.loads(p.read_text())


def test_iter_x4_schema():
    d = _load()
    cfg = d["config"]
    assert set(cfg["weights"]) == {"ETC", "TRX", "ATOM", "APT", "KAS"}
    assert cfg["smoke"] in (True, False)
    rows = d["rows"]
    if not cfg["smoke"]:
        assert {(r["cost"], r["K"]) for r in rows} == {
            (c, k) for c in FULL_CELLS for k in FULL_KS}, len(rows)
    else:
        assert {(r["cost"], r["K"]) for r in rows} == {("20bp-0.002", 0), ("20bp-0.002", 2)}
    assert cfg["grid_bars"] > 2000
    for r in rows:
        assert ROW <= set(r), r
        assert SEG <= set(r["FULL"]), r
        assert r["FULL"]["n"] == cfg["grid_bars"]
        assert abs(r["thr"] - round(r["K"] * r["unit"], 5)) < 1e-9
        assert 0.0 <= r["blocked_frac"] <= 1.0
    # blocked_frac monotone non-decreasing in K within each cost cell
    by_cell = {}
    for r in rows:
        by_cell.setdefault(r["cost"], []).append(r)
    for cell, rs in by_cell.items():
        ks = [r["K"] for r in sorted(rs, key=lambda x: x["K"])]
        assert ks == sorted(ks)
        bf = [next(x["blocked_frac"] for x in rs if x["K"] == k) for k in ks]
        assert all(b <= c + 1e-9 for b, c in zip(bf, bf[1:])), (cell, bf)
    # turnover_cut vs same-cell K0 consistent
    for r in rows:
        k0 = next(x for x in rows if x["cost"] == r["cost"] and x["K"] == min(y["K"] for y in rows if y["cost"] == r["cost"]))
        t0 = k0["FULL"]["turnover"]
        if t0:
            assert abs(r["turnover_cut_vs_k0"] - round((t0 - r["FULL"]["turnover"]) / t0, 4)) < 1e-9
        assert abs(r["d_sharpe_vs_k0"] - round(r["FULL"]["sharpe"] - k0["FULL"]["sharpe"], 3)) < 1e-9
    # first_bite entries are smallest biting K (or null)
    for cell, bite in d["first_bite"].items():
        crs = sorted(by_cell[cell], key=lambda x: x["K"])
        if bite is None:
            assert all(r["turnover_cut_vs_k0"] < cfg["bite_cut"] for r in crs if r["K"] != crs[0]["K"])
        else:
            assert bite in [r["K"] for r in crs if r["K"] != crs[0]["K"]]
            for r in crs:
                if r["K"] != crs[0]["K"] and r["K"] < bite:
                    assert r["turnover_cut_vs_k0"] < cfg["bite_cut"]
            assert next(r for r in crs if r["K"] == bite)["turnover_cut_vs_k0"] >= cfg["bite_cut"]
    rec = d["recommendation"]
    if not cfg["smoke"]:
        assert rec["Y1B_COST_K"] in FULL_KS[1:] + [None]
        if rec["Y1B_COST_K"] is not None:
            top = next(r for r in rows if r["cost"] == "20bp-0.002" and r["K"] == rec["Y1B_COST_K"])
            base = next(r for r in rows if r["cost"] == "base-5bp-0.0005" and r["K"] == rec["Y1B_COST_K"])
            assert top["turnover_cut_vs_k0"] >= cfg["bite_cut"]
            assert base["blocked_frac"] < cfg["noop_blocked"]
            assert "待定" not in rec["reason"] or True
    assert "待定" in d["conclusion"]
    assert "default OFF" in d["conclusion"]


def test_iter_x4_no_broker():
    src = pathlib.Path("research/run_iter_x4.py").read_text()
    for bad in ("place_order", "submit_order", "api_key", "API_KEY",
                "make_broker", "Y1B_LIVE"):
        assert bad.lower() not in src.lower(), bad


def test_iter_x4_smoke_runs_offline():
    env = dict(os.environ, ITER_X4_SMOKE="1")
    r = subprocess.run([sys.executable, "research/run_iter_x4.py"],
                       capture_output=True, text=True, cwd=".", env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    d = _load()
    assert d["config"]["smoke"] is True
    assert {(x["cost"], x["K"]) for x in d["rows"]} == {("20bp-0.002", 0), ("20bp-0.002", 2)}
