"""C1 schema-migration tests.

The point of the rename is R6: an existing reader must not keep treating a
demoted diagnostic as a live gate. These tests pin that, plus fail-closed
behaviour and the derived `live_adopted`.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from research.acceptance_schema_28c import (  # noqa: E402
    FAIL, FOLD_TABLE_KEY, INSUFFICIENT, LEGACY_KEY, NEW_KEY, NEW_PATH, PASS,
    SCHEMA_VERSION, VALID_VERDICTS, derive_live_adopted, make_acceptance,
    read_acceptance,
)

OLD_PATH = "acceptance.portfolio_sharpe"


def _acc(**kw):
    base = dict(min_positive_coins=24, oos_mdd=True, oos_solvent=True,
                oos_portfolio_sharpe=0.478, legacy_bar=1.0,
                criteria={"A1": {}, "A4": {}})
    base.update(kw)
    return make_acceptance(base.pop("verdict", FAIL), base.pop("reason", "x"), **base)


# ------------------------------------------------------------------ writers

def test_written_record_has_no_legacy_key_anywhere():
    acc = _acc()
    assert LEGACY_KEY not in acc
    assert "verdict" in acc and "verdict_reason" in acc
    assert acc["schema_version"] == SCHEMA_VERSION


def test_criteria_and_reason_are_always_present():
    acc = _acc(verdict=FAIL, reason="failed: A4")
    assert acc["criteria"] != {}
    assert acc["verdict_reason"] == "failed: A4"


def test_blank_reason_fails_closed():
    acc = _acc(verdict=PASS, reason="")
    assert acc["verdict"] == FAIL
    assert "fail-closed" in acc["verdict_reason"]


def test_unrecognised_verdict_degrades_to_fail():
    acc = _acc(verdict="looks_good", reason="trust me")
    assert acc["verdict"] == FAIL
    assert "unrecognised" in acc["verdict_reason"]


def test_insufficient_regime_coverage_is_its_own_value_not_an_alias():
    assert INSUFFICIENT in VALID_VERDICTS
    assert INSUFFICIENT != FAIL
    acc = _acc(verdict=INSUFFICIENT, reason="up=0 down=0")
    assert acc["verdict"] == INSUFFICIENT
    # and it must not be treated as adoption
    assert derive_live_adopted(acc) is False


# ------------------------------------------------------------------ live_adopted

def test_live_adopted_is_derived_from_verdict_not_constant():
    assert derive_live_adopted(_acc(verdict=PASS, reason="ok")) is True
    assert derive_live_adopted(_acc(verdict=FAIL, reason="A4")) is False
    assert derive_live_adopted(_acc(verdict=INSUFFICIENT, reason="none")) is False


def test_live_adopted_is_false_for_every_verdict_except_pass():
    for v in VALID_VERDICTS:
        assert derive_live_adopted(_acc(verdict=v, reason="r")) is (v == PASS)


def test_source_does_not_hardcode_live_adopted_false():
    """The literal is what Codex flagged (ga_28c_30m_3y.py:212)."""
    src = open(os.path.join(ROOT, "research", "ga_28c_30m_3y.py")).read()
    assert "'live_adopted': False" not in src
    assert '"live_adopted": False' not in src
    assert "derive_live_adopted" in src


# ------------------------------------------------------------------ migration

def test_legacy_artifact_is_read_as_legacy_and_fail_closed():
    doc = {"acceptance": {"min_positive_coins": 24, LEGACY_KEY: True,
                          "mdd": True, "solvent": True}}
    acc, diag = read_acceptance(doc)
    assert acc["schema_version"] == 1
    assert acc["legacy"] is True
    # the crucial R6 property: it does NOT become a pass
    assert acc["verdict"] == FAIL
    assert derive_live_adopted(acc) is False
    # and the demoted criterion moved, with its provenance recorded
    assert diag[NEW_KEY] is True
    assert diag[NEW_KEY + "_source_path"] == OLD_PATH


def test_legacy_reader_does_not_expose_the_gate_named_key():
    doc = {"acceptance": {LEGACY_KEY: True}}
    acc, _ = read_acceptance(doc)
    assert LEGACY_KEY not in acc          # not at the old path any more
    assert NEW_PATH.split(".")[0] == "diagnostics"


def test_new_schema_artifact_round_trips_without_demotion():
    doc = {"acceptance": _acc(verdict=PASS, reason="all"),
           "diagnostics": {NEW_KEY: True}}
    acc, diag = read_acceptance(doc)
    assert acc["verdict"] == PASS
    assert acc["schema_version"] == SCHEMA_VERSION
    assert diag[NEW_KEY] is True
    assert "legacy" not in acc


def test_missing_acceptance_fails_closed():
    for doc in ({}, {"acceptance": None}, {"acceptance": "nonsense"}, None, 7):
        acc, _ = read_acceptance(doc)
        assert acc["verdict"] == FAIL
        assert acc["verdict_reason"]
        assert derive_live_adopted(acc) is False


# ------------------------------------------------------------------ N6 dashboard

def test_fold_table_key_cannot_collide_with_dashboard_keys():
    """dashboard/visualizer.py:75-83 reads `history` expecting step/sharpe.
    Reusing that name would silently render an empty figure."""
    assert FOLD_TABLE_KEY not in ("history", "plot_walk_forward")
    assert FOLD_TABLE_KEY == "regime_gate"


def test_shipped_ga_results_still_readable_under_the_migration():
    import glob
    import json
    paths = glob.glob(os.path.join(ROOT, "results", "ga_28c_*.json"))
    assert paths, "expected GA result artifacts"
    for p in paths:
        acc, diag = read_acceptance(json.load(open(p)))
        assert acc["verdict"] in VALID_VERDICTS
        if acc["schema_version"] == 1:
            assert acc["verdict"] == FAIL, p
            assert derive_live_adopted(acc) is False
