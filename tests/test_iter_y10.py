"""Y10 rank-table tests: schema-level only, no backtest recompute.

Reads results/iter_Y1..Y9 artifacts (produced by sibling rounds) plus
results/iter_Y_RANK.json (produced by Y10 aggregation, read-only).
Absent rounds (missing artifact) must be listed in meta.absent with a
stub round entry status=absent; done rounds must have existing files.
All conclusions stay PENDING (P0-3 FAIL): no demo recommendation.
"""
import json
import pathlib

OUT = pathlib.Path("results/iter_Y_RANK.json")
ROUNDS = [f"Y{i}" for i in range(1, 10)]
FILES = {
    "Y1": "results/iter_Y1_weights.json",
    "Y2": "results/iter_Y2_stops.json",
    "Y3": "results/iter_Y3_thresh.json",
    "Y4": "results/iter_Y4_cost.json",
    "Y5": "results/iter_Y5_scan.json",
    "Y6": "results/iter_Y6_freq.json",
    "Y7": "results/iter_Y7_lev.json",
    "Y8": "results/iter_Y8_fund.json",
    "Y9": "results/iter_Y9_corr.json",
}
DIM_KEYS = ("rank_sharpe", "rank_dd", "rank_robust")


def _load():
    assert OUT.exists(), "results/iter_Y_RANK.json missing; run Y10 aggregation"
    return json.loads(OUT.read_text())


def test_iter_y10_meta():
    d = _load()
    assert d["meta"]["iter"] == "Y10"
    assert d["meta"]["dims"] == ["sharpe_gain", "dd_cut_pct", "robustness"]
    assert isinstance(d["meta"]["absent"], list)
    assert d["meta"]["n_done"] + d["meta"]["n_absent"] == 9
    assert set(d["meta"]["absent"]) | {r["round"] for r in d["rounds"]
                                       if r["status"] == "done"} == set(ROUNDS)
    assert d["verdict"] == "ALL_PENDING_P03_FAIL"


def test_iter_y10_rounds_schema():
    d = _load()
    assert [r["round"] for r in d["rounds"]] == ROUNDS
    for r in d["rounds"]:
        for f in ("round", "file", "topic", "status", "sharpe_gain",
                  "dd_cut_pct", "robustness", "detail", "conclusion_zh",
                  "demo", "demo_reason"):
            assert f in r, (r["round"], f)
        assert isinstance(r["conclusion_zh"], str) and len(r["conclusion_zh"]) > 0
        assert r["demo"] == "no"
        assert "待定" in r["demo_reason"] or "PENDING" in r["demo_reason"]
        if r["status"] == "done":
            assert r["file"] == FILES[r["round"]]
            assert pathlib.Path(r["file"]).exists(), r["file"]
            assert isinstance(r["sharpe_gain"], (int, float))
            assert isinstance(r["dd_cut_pct"], (int, float))
            assert isinstance(r["robustness"], (int, float))
        else:
            assert r["round"] in d["meta"]["absent"]


def test_iter_y10_ranks_cover_all_rounds():
    d = _load()
    for k in DIM_KEYS + ("rank_composite",):
        assert len(d[k]) == 9, k
    for k in DIM_KEYS:
        assert {e["round"] for e in d[k]} == set(ROUNDS), k
        for e in d[k]:
            assert isinstance(e["rank"], int) and e["rank"] >= 1
            assert isinstance(e["value"], (int, float))
    comp = d["rank_composite"]
    assert {e["round"] for e in comp} == set(ROUNDS)
    by_round = {e["round"]: e for e in comp}
    dim_rank = {k: {e["round"]: e["rank"] for e in d[k]} for k in DIM_KEYS}
    sums = [e["rank_sum"] for e in comp]
    assert sums == sorted(sums), "composite must be ordered by rank_sum ascending"
    for e in comp:
        k = e["round"]
        expect = (dim_rank["rank_sharpe"][k] + dim_rank["rank_dd"][k]
                  + dim_rank["rank_robust"][k])
        assert e["rank_sum"] == expect, k
        assert e["ranks"] == {"sharpe": dim_rank["rank_sharpe"][k],
                              "dd": dim_rank["rank_dd"][k],
                              "robust": dim_rank["rank_robust"][k]}, k


def test_iter_y10_no_demo_recommendation():
    d = _load()
    assert d["demo"]["recommend_any"] is False
    assert d["demo"]["rounds_yes"] == []
    assert len(d["caveats"]) >= 1
    assert d["baseline"]["FULL_sharpe"] == 26.111
    assert d["baseline"]["FULL_mdd"] == 0.2901
