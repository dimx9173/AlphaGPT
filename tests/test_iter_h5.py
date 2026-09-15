"""H5 cost-gate (1h native, Top5) tests. Recompute-cheap, never clobber FULL artifact."""
import json
import os
import pathlib
import subprocess
import sys
import tempfile

OUT = pathlib.Path("results/iter_H5_cost.json")
SRC = pathlib.Path("research/run_iter_h5_cost.py")
SEG = {"sharpe", "ann", "mdd", "cum", "final_x", "n", "turnover"}
COINS = {"ETC", "TRX", "ATOM", "APT", "KAS"}
FULL_KS = [0, 1, 2, 3, 5]
FULL_COSTS = ["base", "fee2x", "slip10", "slip20", "fundhi"]


def _load():
    assert OUT.exists(), "results/iter_H5_cost.json missing; run research/run_iter_h5_cost.py"
    return json.loads(OUT.read_text())


def _run_smoke():
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "iter_H5_cost.json")
        log = os.path.join(td, "iter_H5_cost.log")
        env = dict(os.environ, ITER_H5_SMOKE="1", ITER_H5_OUT=out, ITER_H5_LOG=log)
        r = subprocess.run([sys.executable, str(SRC)], capture_output=True,
                           text=True, cwd=".", env=env)
        assert r.returncode == 0, r.stderr[-2000:]
        return json.loads(pathlib.Path(out).read_text())


def test_iter_h5_schema():
    d = _load()
    cfg = d["config"]
    assert set(cfg["weights"]) == COINS
    assert cfg["bpy"] == 8760.0
    assert cfg["scale"] == 4
    assert cfg["grid"] == "1h"
    assert cfg["grid_bars"] == 8760
    assert cfg["top_cell"] == "fundhi"
    assert cfg["base_cell"] == "base"
    assert cfg["smoke"] in (True, False)
    assert cfg["Ks"] == ([0, 2] if cfg["smoke"] else FULL_KS)
    assert [c["label"] for c in cfg["costs"]] == (
        ["fundhi"] if cfg["smoke"] else FULL_COSTS)
    assert d["verdict"] == "PENDING"
    assert d["decision"] == "GATE_OFF"
    assert d["complete"] is True
    rows = d["rows"]
    assert len(rows) == len(cfg["costs"]) * len(cfg["Ks"])
    for r in rows:
        assert SEG <= set(r["FULL"]), r
        assert r["FULL"]["n"] == cfg["grid_bars"]
        for f in ("cost", "fee", "slip_bp", "slip", "fund", "unit", "K",
                  "thr", "blocked_frac", "turnover_cut_vs_k0",
                  "d_sharpe_vs_k0"):
            assert f in r, f
        assert abs(r["unit"] - (r["fee"] + r["slip"] + r["fund"])) < 1e-9
        assert abs(r["thr"] - r["K"] * r["unit"]) < 1e-9
        assert 0.0 <= r["blocked_frac"] <= 1.0
    by_cell = {}
    for r in rows:
        by_cell.setdefault(r["cost"], {})[r["K"]] = r
    for cell, rr in by_cell.items():
        assert rr[0]["turnover_cut_vs_k0"] == 0.0
        assert rr[0]["d_sharpe_vs_k0"] == 0.0
        assert rr[0]["blocked_frac"] == 0.0
    if not cfg["smoke"]:
        rec = d["recommendation"]["Y1B_COST_K"]
        top = by_cell["fundhi"]
        base = by_cell["base"]
        cands = [k for k in FULL_KS if k > 0
                 and top[k]["turnover_cut_vs_k0"] >= cfg["bite_cut"]
                 and base[k]["blocked_frac"] < cfg["noop_blocked"]]
        assert rec == (min(cands) if cands else None)
        assert set(d["first_bite"]) == set(FULL_COSTS)
        for cell, rr in by_cell.items():
            exp = next((k for k in FULL_KS if k > 0
                        and rr[k]["turnover_cut_vs_k0"] >= cfg["bite_cut"]),
                       None)
            assert d["first_bite"][cell] == exp, cell
    assert "\u5f85\u5b9a" in json.dumps(d["decision_note"], ensure_ascii=True) or "待定" in d["decision_note"]


def test_iter_h5_no_broker():
    src = SRC.read_text()
    assert "data/data_1y/1h" in src
    assert "8760" in src
    for bad in ("place_order", "submit_order", "api_key",
                "make_broker", "Y1B_LIVE"):
        assert bad.lower() not in src.lower(), bad


def test_iter_h5_native_grid():
    d = _load()
    b = d["config"]["basket"]
    assert b["ETC"] == {"lth": 0.88, "sth": 0.12, "cd": 18, "sl": None,
                        "ts": 24, "q": 0.3}
    assert b["TRX"] == {"lth": 0.85, "sth": 0.12, "cd": 6, "sl": 0.05,
                        "ts": 24, "q": 0.3}
    assert b["ATOM"] == {"lth": 0.85, "sth": 0.15, "cd": 6, "sl": 0.05,
                         "ts": 24, "q": 0.3}
    assert b["APT"] == {"lth": 0.88, "sth": 0.12, "cd": 18, "sl": None,
                        "ts": 24, "q": 0.3}
    assert b["KAS"] == {"lth": 0.88, "sth": 0.12, "cd": 6, "sl": None,
                        "ts": 24, "q": 0.3}


def test_iter_h5_incremental_dump():
    src = SRC.read_text()
    assert "cells_done" in src
    assert "complete" in src
    assert src.count("INCREMENTAL DUMP") >= 1
    assert "for cell in costs:" in src
    assert "after each unit" in src.lower()


def test_iter_h5_smoke_runs_offline():
    d = _run_smoke()
    assert d["config"]["smoke"] is True
    assert len(d["rows"]) == 2
    assert d["verdict"] == "PENDING"
    assert d["decision"] == "GATE_OFF"
    assert d["complete"] is True
    assert d["config"]["grid_bars"] == 8760
