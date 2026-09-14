"""V10 rank tests (schema only)."""
import json
import pathlib
OUT = pathlib.Path("results/iter_V_RANK.json")
def _load():
    assert OUT.exists()
    return json.loads(OUT.read_text())
def test_iter_v10_schema():
    d = _load()
    assert d["verdict"] == "ALL_PENDING_P03_FAIL"
    assert d["demo"]["recommend_any"] is False
    assert len(d["rounds"]) == 9
def test_iter_v10_all_no():
    d = _load()
    assert all(r["demo"] == "no" for r in d["rounds"])
def test_iter_v10_v2_absent():
    d = _load()
    v2 = [r for r in d["rounds"] if r["round"] == "V2"][0]
    assert v2["status"] == "absent"
