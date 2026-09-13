"""P0-3 freeze pins: cut index, file schemas, no-broker guard."""
import json
import pathlib


def test_freeze_file_schema():
    fz = json.load(open("research/oos_freeze.json"))
    assert fz["cut"]["cut_index"] == 6580
    assert fz["cut"]["meaning"] == "in-sample=[0:6580), frozen-OOS=[6580:N)"
    assert isinstance(fz["rules"], list) and len(fz["rules"]) >= 4
    assert any("OOS" in r for r in fz["rules"])
    assert fz["lock"]["lock_date"] == "2026-09-07"


def test_plateau_schema_and_verdict():
    p = json.load(open("results/plateau.json"))
    assert p["config"]["in_sample"] == [0, 6580]
    assert p["config"]["oos_role"].startswith("diagnostic")
    assert len(p["heatmap"]) == 25
    for c in p["heatmap"]:
        for k in ("sth_etc", "sth_trx", "sharpe", "ann", "mdd", "turnover"):
            assert k in c
    assert p["plateau"]["verdict"] in ("PASS", "FAIL")
    assert p["plateau"]["plateau_size"] >= 0
    assert p["locked_oos_diagnostic"]["diagnostic_only"] is True


def test_permutation_schema_and_gates():
    m = json.load(open("results/permutation.json"))
    assert m["config"]["in_sample"] == [0, 6580]
    assert m["config"]["n_perm"] == 200
    assert m["deflated_sharpe"]["trials"] == 10
    assert 0.0 <= m["deflated_sharpe"]["dsr"] <= 1.0
    for coin in ("ETC", "TRX"):
        v = m["per_coin"][coin]
        assert v["n_perm"] == 200
        assert 0.0 <= v["p_value"] <= 1.0
    assert isinstance(m["gates"]["perm_p_lt_0_05"], bool)
    assert isinstance(m["gates"]["dsr_gt_0_8"], bool)


def test_no_broker_in_plateau_engine():
    src = pathlib.Path("research/run_plateau.py").read_text()
    for bad in ("place_order", "submit_order", "api_key", "API_KEY", "make_broker", "Y1B_LIVE"):
        assert bad.lower() not in src.lower(), bad
