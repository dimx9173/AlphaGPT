"""train_12f tests: output routing isolation (smoke never clobbers real artifacts)."""
import json
import os
import pathlib
import subprocess
import sys


def test_train12f_smoke_writes_tmp_only():
    best = pathlib.Path("results/train_12f_30m_best.json")
    before = best.read_bytes() if best.exists() else None
    env = dict(os.environ, TRAIN_12F_SMOKE="1")
    r = subprocess.run([sys.executable, "research/train_12f_30m.py"],
                       capture_output=True, text=True, cwd=".", env=env, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "wrote /tmp/train_12f_30m_smoke.json" in (r.stdout + r.stderr), (r.stdout + r.stderr)[-1500:]
    out = json.loads(pathlib.Path("/tmp/train_12f_30m_smoke.json").read_text())
    assert out["vocab"] == "12f"
    assert out["floor"] == 0.0
    if before is not None:
        assert best.read_bytes() == before, "smoke clobbered results/train_12f_30m_best.json"


def test_train12f_floor_veto_contract():
    src = pathlib.Path("research/train_12f_30m.py").read_text()
    assert "TRAIN_12F_FLOOR" in src
    assert "train_12f_30m_floor_best.json" in src
