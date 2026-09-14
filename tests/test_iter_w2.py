
"""W2 15m weight sweep fine (Top5) tests: ETC {0.1..0.3} + KAS cap {0.1,0.2}."""
import json
import os
import pathlib
import subprocess
import sys

OUT = pathlib.Path("results/iter_W2_wt.json")
ETC_GRID = [0.1, 0.15, 0.2, 0.25, 0.3]
KAS_CAPS = [0.1, 0.2]


def _load():
    assert OUT.exists(), "results/iter_W2_wt.json missing; run research/run_iter_w2_wt.py"
    return json.loads(OUT.read_text())


def test_iter_w2_weight_math():
    sys.path.insert(0, ".")
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_iter_w2_wt", "research/run_iter_w2_wt.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    for w in ETC_GRID:
        d = m.etc_weights(w, ["ETC", "TRX", "ATOM", "APT", "KAS"])
        assert abs(sum(d.values()) - 1.0) < 1e-9, d
        assert abs(d["ETC"] - w) < 1e-9, d
        rest = [(d[c]) for c in ("TRX", "ATOM", "APT", "KAS")]
        assert max(rest) - min(rest) < 1e-9, d
        assert abs(rest[0] - (1.0 - w) / 4) < 1e-9, d
    d = m.kas_cap_weights(0.2, ["ETC", "TRX", "ATOM", "APT", "KAS"])
    assert all(abs(v - 0.2) < 1e-9 for v in d.values()), d
    d = m.kas_cap_weights(0.1, ["ETC", "TRX", "ATOM", "APT", "KAS"])
    assert abs(d["KAS"] - 0.1) < 1e-9, d
    assert abs(sum(d.values()) - 1.0) < 1e-9, d
    rest = [d[c] for c in ("ETC", "TRX", "ATOM", "APT")]
    assert max(rest) - min(rest) < 1e-9 and abs(rest[0] - 0.225) < 1e-9, d


def test_iter_w2_schema():
    d = _load()
    assert d["verdict"] == "PENDING"
    assert d["decision"] == "KEEP_equal"
    assert d["config"]["grid"] == "15m"
    assert d["config"]["grid_bars"] == 35040
    assert d["config"]["bpy"] == 35040.0
    assert d["config"]["formula"] == [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10]
    assert d["config"]["etc_grid"] == ETC_GRID
    assert d["config"]["kas_caps"] == KAS_CAPS
    assert set(d["config"]["locked_4h"]) == {"ETC", "TRX", "ATOM", "APT", "KAS"}
    for k in ("cd", "ts", "vw"):
        for c in ("ETC", "TRX", "ATOM", "APT", "KAS"):
            assert d["config"]["specs_15m"][c][k] == d["config"]["locked_4h"][c][k] * 16, (c, k)
    assert [r["w_etc"] for r in d["etc_sweep"]] == ETC_GRID
    assert [r["kas_cap"] for r in d["kas_cap"]] == KAS_CAPS
    for r in d["etc_sweep"] + d["kas_cap"]:
        for f in ("sharpe", "mdd"):
            assert f in r["FULL"], (r["id"], f)
        assert abs(sum(r["weights"].values()) - 1.0) < 1e-6, (r["id"], r["weights"])
        assert len(r["fold12"]["sharpes"]) == 12
        assert r["FULL"]["n"] == d["config"]["grid_bars"]
    base = next(r for r in d["etc_sweep"] if abs(r["w_etc"] - 0.2) < 1e-9)
    assert all(abs(v - 0.2) < 1e-9 for v in base["weights"].values())
    for rid, v in d["compare"].items():
        f = d["arms"][rid]["FULL"]
        expect = bool(f["sharpe"] > base["FULL"]["sharpe"] and f["mdd"] < base["FULL"]["mdd"])
        assert v["beats_equal_on_both"] is expect, rid


def test_iter_w2_fee2x_and_h2_present():
    d = _load()
    for r in d["etc_sweep"] + d["kas_cap"]:
        assert "fee2x_FULL_sharpe" in r and isinstance(r["fee2x_FULL_sharpe"], float), r["id"]
        assert "H2_sharpe" in r and "H2_trades" in r, r["id"]
        assert "final_x" in r["FULL"] and "turnover" in r["FULL"], r["id"]


def test_iter_w2_script_runs_offline(tmp_path):
    out = tmp_path / "w2_wt.json"
    log = tmp_path / "w2_wt.log"
    env = dict(os.environ, ITER_W2_SMOKE="1", ITER_W2_OUT=str(out), ITER_W2_LOG=str(log))
    r = subprocess.run([sys.executable, "research/run_iter_w2_wt.py"],
                       capture_output=True, text=True, cwd=".", env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    d = json.loads(out.read_text())
    assert d["verdict"] == "PENDING" and d["decision"] == "KEEP_equal"
    assert len(d["etc_sweep"]) == 2 and d["kas_cap"] == []
    assert d["config"]["smoke"] is True
