#!/usr/bin/env python3
"""Regression: legacy vs equity-compound-v2 accounting cross-check must pass."""
import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv2" / "bin" / "python"
SCRIPT = ROOT / "tools" / "crosscheck_accounting.py"


def test_crosscheck_accounting_passes():
    r = subprocess.run([str(PY), str(SCRIPT)], capture_output=True, text=True,
                       cwd=str(ROOT), timeout=300)
    assert r.returncode == 0, f"crosscheck failed:\n{r.stdout}\n{r.stderr}"
    assert "all cross-checks passed" in r.stdout
