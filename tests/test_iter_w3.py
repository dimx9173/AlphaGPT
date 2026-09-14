"""W3 vt sweep (15m native) tests."""
import json
import pathlib
import subprocess
import sys

OUT = pathlib.Path("results/iter_W3_vt.json")

def _load():
    assert OUT.exists(), "results/iter_W3_vt.json missing; run research/run_iter_w3_vt.py"
    return json.loads(OUT.read_text())

def test_iter_w3_locks():
    from strategy_manager.config import LEV, FORMULA
    assert LEV == 2.0
    assert list(FORMULA) == [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10]

def test_iter_w3_schema():
    d = _load()
    assert d["config"]["vts"] == ["None", 0.003, 0.006, 0.012, 0.024]
    assert d["config"]["grid"] == "15m"
    assert d["config"]["grid_bars"] == 35040
    assert d["config"]["bpy"] == 35040.0
    assert d["config"]["scale"] == 16
    assert set(d["config"]["weights"]) == {"ETC", "TRX", "ATOM", "APT", "KAS"}
    assert set(d["rows"]) == {"None", "0.003", "0.006", "0.012", "0.024"}
    for k, v in d["rows"].items():
        for f in ("sharpe", "mdd", "cum", "final_x", "ann", "n", "turnover"):
            assert f in v["FULL"], (k, f)
        assert v["FULL"]["n"] == d["config"]["grid_bars"]
    assert d["verdict"] == "PENDING"
    assert d["decision"] == "KEEP_VT_NONE"
    c = d["compare"]
    assert set(c["sharpe_by_vt"]) == {"None", "0.003", "0.006", "0.012", "0.024"}
    assert set(c["mdd_by_vt"]) == {"None", "0.003", "0.006", "0.012", "0.024"}
    assert set(c["turnover_by_vt"]) == {"None", "0.003", "0.006", "0.012", "0.024"}
    assert c["best_sharpe_vt"] in c["sharpe_by_vt"]

def test_iter_w3_vt_effect():
    d = _load()
    sh = [d["rows"][k]["FULL"]["sharpe"] for k in ("None", "0.003", "0.006", "0.012", "0.024")]
    assert all(isinstance(x, float) for x in sh)
    assert max(sh) - min(sh) == d["compare"]["sharpe_spread"]
    to_none = d["rows"]["None"]["FULL"]["turnover"]
    tos = [d["rows"][k]["FULL"]["turnover"] for k in ("0.003", "0.006", "0.012", "0.024")]
    assert any(abs(x - to_none) > 1e-9 for x in tos), (to_none, tos)
    for k, v in d["rows"].items():
        assert v["FULL"]["mdd"] >= 0.0, (k, v["FULL"]["mdd"])

def test_iter_w3_script_runs_offline():
    r = subprocess.run([sys.executable, "research/run_iter_w3_vt.py"], capture_output=True, text=True, cwd=".")
    assert r.returncode == 0, r.stderr[-2000:]
    d = _load()
    assert set(d["rows"]) == {"None", "0.003", "0.006", "0.012", "0.024"}
    assert d.get("partial") is not True
