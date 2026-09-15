"""H60 rank-table tests: schema-level only, no backtest recompute.

Reads results/iter_H51..H59 artifacts (produced by sibling rounds) plus
results/iter_H60_RANK.json (produced by H60 aggregation, read-only).
Absent rounds (missing artifact) must be listed in meta.absent with a
stub round entry status=absent; done rounds must have existing files.
All conclusions stay PENDING (P0-3 FAIL): no demo recommendation.
"""
import json
import pathlib

OUT = pathlib.Path("results/iter_H60_RANK.json")
ROUNDS = [f"H{i}" for i in range(51, 60)]
FILES = {
    "H51": "results/iter_H51_donch.json",
    "H52": "results/iter_H52_obv.json",
    "H53": "results/iter_H53_vwap.json",
    "H54": "results/iter_H54_bb.json",
    "H55": "results/iter_H55_stoch.json",
    "H56": "results/iter_H56_macd.json",
    "H57": "results/iter_H57_rsi.json",
    "H58": "results/iter_H58_adx.json",
    "H59": "results/iter_H59_atr.json",
}
DIM_KEYS = ("rank_sharpe", "rank_dd", "rank_robust")


def _load():
    assert OUT.exists(), "results/iter_H60_RANK.json missing; run H60 aggregation"
    return json.loads(OUT.read_text())


def test_iter_h60_meta():
    d = _load()
    assert d["meta"]["iter"] == "H60"
    assert d["meta"]["dims"] == ["sharpe_gain", "dd_cut_pct", "robustness"]
    assert isinstance(d["meta"]["absent"], list)
    assert d["meta"]["n_done"] + d["meta"]["n_absent"] == 9
    assert set(d["meta"]["absent"]) | {r["round"] for r in d["rounds"]
                                       if r["status"] == "done"} == set(ROUNDS)
    assert d["verdict"] == "ALL_PENDING_P03_FAIL"


def test_iter_h60_rounds_schema():
    d = _load()
    assert [r["round"] for r in d["rounds"]] == ROUNDS
    for r in d["rounds"]:
        for f in ("round", "file", "topic", "status", "sharpe_gain",
                  "dd_cut_pct", "robustness", "detail", "conclusion_zh",
                  "demo", "demo_reason"):
            assert f in r, (r["round"], f)
        assert isinstance(r["conclusion_zh"], str) and len(r["conclusion_zh"]) > 0
        assert r["demo"] == "no"
        assert "PENDING" in r["demo_reason"]
        if r["status"] == "done":
            assert r["file"] == FILES[r["round"]]
            assert pathlib.Path(r["file"]).exists(), r["file"]
            assert isinstance(r["sharpe_gain"], (int, float))
            assert isinstance(r["dd_cut_pct"], (int, float))
            assert isinstance(r["robustness"], (int, float))
        else:
            assert r["round"] in d["meta"]["absent"]


def test_iter_h60_ranks_cover_all_rounds():
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


def test_iter_h60_no_demo_recommendation():
    d = _load()
    assert d["demo"]["recommend_any"] is False
    assert d["demo"]["rounds_yes"] == []
    assert len(d["caveats"]) >= 1
    assert d["baseline"]["FULL_sharpe"] == 8.514
    assert d["baseline"]["FULL_mdd"] == 0.1998
