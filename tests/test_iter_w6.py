"""W6 drawdown anatomy tests (smoke-isolated, offline)."""
import json
import os
import pathlib
import subprocess
import sys


def _run_smoke(tmp_path):
    out = tmp_path / "w6.json"
    log = tmp_path / "w6.log"
    env = dict(os.environ, ITER_W6_SMOKE="1",
               ITER_W6_OUT=str(out), ITER_W6_LOG=str(log))
    r = subprocess.run([sys.executable, "research/run_iter_w6_dd.py"],
                       capture_output=True, text=True, cwd=".", env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(out.read_text())


def test_iter_w6_smoke_schema(tmp_path):
    d = _run_smoke(tmp_path)
    assert d["config"]["smoke"] is True
    assert set(d["percoin"]) == {"ETC", "TRX", "ATOM", "APT", "KAS"}
    assert "portfolio" in d
    for coin, v in d["percoin"].items():
        assert len(v) == 1
        w = v[0]
        assert {"depth", "length_bars", "recovery_bars", "cum"} <= set(w)


def test_iter_w6_windows_nonoverlap(tmp_path):
    d = _run_smoke(tmp_path)
    for coin, v in d["percoin"].items():
        ws = v
        for i in range(len(ws)):
            for j in range(i + 1, len(ws)):
                a, b = ws[i], ws[j]
                assert a["trough"] <= b["peak"] or b["trough"] <= a["peak"]


def test_iter_w6_verdict_pending(tmp_path):
    d = _run_smoke(tmp_path)
    assert d["verdict"] == "PENDING"
