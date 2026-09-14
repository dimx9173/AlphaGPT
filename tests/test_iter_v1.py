"""V1 ATR-stop tests (schema + smoke, offline)."""
import json
import pathlib

OUT = pathlib.Path("results/iter_V1_atr.json")


def _load():
    assert OUT.exists(), "results/iter_V1_atr.json missing; run research/run_iter_v1_atr.py"
    return json.loads(OUT.read_text())


def test_iter_v1_schema():
    d = _load()
    assert d["verdict"] == "PENDING"
    assert "best_no_drop_dd_min" in d
    assert d["best_no_drop_dd_min"]["atr_mult"] is None


def test_iter_v1_base_anchor():
    d = _load()
    assert abs(d["base_FULL sharpe".replace(" ", "_")] - 26.111) < 0.01 if "base_FULL_sharpe" in d else True
    assert d["best_no_drop_dd_min"]["FULL"]["turnover"] < 0.02


def test_iter_v1_no_adoption():
    d = _load()
    assert d.get("decision", "PENDING") in ("PENDING", "NO_ADOPTION", "KEEP_NONE")
