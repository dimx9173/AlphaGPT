"""validate-12f tests: 12fold artifact schema + gate contract (no recompute)."""
import json
import pathlib

OUT = pathlib.Path("results/validate_12f_floor_12fold.json")
BEST = pathlib.Path("results/train_12f_30m_floor_best.json")


def _load():
    assert OUT.exists(), "run research/validate_12f_floor_12fold.py"
    return json.loads(OUT.read_text())


def test_validate12f_formula_matches_best():
    d, b = _load(), json.loads(BEST.read_text())
    assert d["formula"] == b["formula"]
    assert len(d["folds"]) == 12


def test_validate12f_summary_gates():
    d = _load()
    s = d["summary"]
    assert s["pass_all"] is True
    assert s["gates"] == {"mean_gt_0": True, "median_ge_1_5": True, "n_pos_ge_8": True}
    assert s["mean"] > 0 and s["median"] >= 1.5 and s["n_pos"] >= 8
    for f in d["folds"]:
        assert set(f["per_coin"]) == {"ETC", "TRX", "ATOM", "APT", "KAS"}
        for c, m in f["per_coin"].items():
            assert {"sharpe", "mdd", "turnover"} <= set(m), (f["fold"], c)


def test_validate12f_pending_no_adoption():
    d = _load()
    assert d["verdict"] == "PENDING"
    assert d["decision"] == "NO_ADOPTION"
