"""H77 Ichimoku filter tests (schema + smoke, offline)."""
import json
import pathlib

OUT = pathlib.Path("results/iter_H77_ichi.json")

def _load():
    assert OUT.exists()
    return json.loads(OUT.read_text())

def test_iter_h77_schema():
    d = _load()
    assert d["verdict"] == "PENDING"
    assert d["decision"] == "NO_ADOPTION"

def test_iter_h77_arms():
    d = _load()
    base = [r for r in d["rows"] if r.get("unit") == "base"][0]
    assert abs(base["FULL"]["sharpe"] - 8.515) < 0.05

def test_iter_h77_no_adoption():
    d = _load()
    assert d["decision"] == "NO_ADOPTION"
