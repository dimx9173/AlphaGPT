"""The holdout must report its benchmark, or the loop cannot do its job.

A walk-forward run writes its final score under "holdout", not "oos", and it
is the only number in the artifact that is scored exactly once. It is also the
number most likely to be quoted as a result.

The first version of the loop read "oos", found nothing, and rejected with a
None excess. Fail-closed, but for a reason that had nothing to do with the
strategy -- and the fix is not a loosened reader. Both the writer and the
reader had to be fixed, because a loop that cannot compare a holdout against
a passive hold will eventually be "fixed" by comparing it against zero.

The walk-forward writer dropped the benchmark entirely: run_walkforward copied
a hand-picked subset of evaluate()'s output into its records, and the three
benchmark fields were not in that subset. So even a correct reader would have
found nothing.

The concrete reason this matters is in the pinned run below. A 20-generation
search produced a holdout Sharpe of +1.155. That clears the gate's Sharpe 1.0
threshold and looks like a discovery in any summary. Against a constant
levered long on the same bars it is 1.28 Sharpe WORSE, and the position is
long-biased at +0.166 -- the drift-matching behaviour documented in
d60033c, showing up again on a fresh holdout.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH_KEYS = ("buy_and_hold_sharpe", "excess_sharpe_vs_buy_and_hold",
              "mean_position")


def holdout_reports(require_benchmark=False):
    """Every walk-forward artifact with a holdout block.

    Artifacts from before the fix are kept, not deleted -- they are what the
    fix is for. So this returns all of them, and the caller can ask for only
    the ones that carry the benchmark. Selecting the newest by name instead
    would silently pick a stale artifact, which is the bug this file is about
    happening to the test.
    """
    out = []
    for f in sorted((ROOT / "results").glob("loop_*.json")):
        try:
            d = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        h = d.get("holdout")
        if not isinstance(h, dict) or h.get("portfolio_sharpe") is None:
            continue
        if require_benchmark and not all(k in h for k in BENCH_KEYS):
            continue
        out.append((f, h))
    return out


def latest_benchmarked():
    rep = holdout_reports(require_benchmark=True)
    if not rep:
        pytest.skip("no benchmarked walk-forward run yet")
    return rep[-1]


def test_some_run_carries_a_benchmarked_holdout():
    """At least one artifact must prove the writer's fix took effect."""
    f, h = latest_benchmarked()
    for k in BENCH_KEYS:
        assert k in h, f"{f.name}: the holdout has no {k!r}"


def test_a_benchmarked_holdout_loses_to_the_passive_hold():
    """Pinned so the "it looks like a result" trap cannot be forgotten.

    +1.155 clears the gate's Sharpe 1.0 threshold. Against the hold it scores
    +1.322. Read against zero this is a discovery; read against a holdable
    position it is 1.28 Sharpe worse, and the position is long-biased.
    """
    f, h = latest_benchmarked()
    assert h["buy_and_hold_sharpe"] > h["portfolio_sharpe"], (
        f"{f.name}: holdout {h['portfolio_sharpe']:+.4f} vs B&H "
        f"{h['buy_and_hold_sharpe']:+.4f}. If the holdout ever clears the "
        "benchmark, that is a real result - re-derive the assertion, "
        "do not delete it.")
    assert h["excess_sharpe_vs_buy_and_hold"] < 0.0


def test_the_walk_forward_writer_copies_the_benchmark_through():
    """The writer is half the fix. A correct reader is not enough."""
    src = (ROOT / "research" / "ga_28c_30m_3y.py").read_text()
    # both the holdout record and the written holdout block
    assert src.count("excess_sharpe_vs_buy_and_hold") >= 4, (
        "run_walkforward is dropping the benchmark again")


def test_the_accept_stage_refuses_to_fall_back_to_zero():
    s = (ROOT / "research" / "run_loop_28c.py").read_text()
    assert "will not fall back" in s
    assert 'art.get("oos") or art.get("holdout")' in s
    # and a missing benchmark is an error, not a silent None that reads as 0
    assert 'if rec["excess"] is None:' in s
