"""The strategy has never beaten holding a constant long position.

Zero is not a benchmark. It is a number with no position behind it, and every
Sharpe this project has reported is measured against it. On this universe,
over this window, a constant long at the position cap earns a real Sharpe,
because crypto drifted. Reading a Sharpe of +0.48 as an edge while a levered
do-nothing earns +1.05 is reading a drift as a discovery.

This test pins the finding rather than the number. The absolute Sharpes will
move as the framework changes; the claim being protected is that no run in
the recorded batches has beaten the benchmark, so any future result that
claims to must be treated as a real change and investigated, not banked.

Mechanism, for the record. The diagnostic nulls reported a HIGHER gross
Sharpe than the real data, which looked like the search finding structure in
structureless data. It was the opposite: on those nulls the factors collapsed
to a constant sign and the position was long on 93-96% of bars with a mean
position of +0.17 to +0.22. The reported Sharpe was buy-and-hold, and the
null's own drift supplied it.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "results" / "buy_and_hold_benchmark.json"

pytestmark = pytest.mark.skipif(not ARTIFACT.exists(),
                                reason="benchmark not computed yet")


@pytest.fixture(scope="module")
def bench():
    return json.loads(ARTIFACT.read_text())


def test_every_recorded_batch_is_present(bench):
    assert set(bench) >= {"none", "iid", "iid1", "iidw", "iidg"}


@pytest.mark.parametrize("mode", ["none", "iid", "iid1", "iidw", "iidg"])
def test_no_batch_beats_buy_and_hold(bench, mode):
    b = bench[mode]
    assert b["excess"] < 0.0, (
        f"{mode} now beats buy-and-hold by {b['excess']:+.3f}. If this is a real "
        "improvement, say so explicitly and re-derive every conclusion that "
        "assumed the framework had never beaten a passive long. Do not delete "
        "this assertion to make a run pass.")


@pytest.mark.parametrize("mode", ["none", "iid", "iid1", "iidw", "iidg"])
def test_the_median_run_loses_to_the_benchmark(bench, mode):
    b = bench[mode]
    assert b["strategy_median_sharpe"] < b["buy_and_hold_sharpe"]


def test_the_real_data_result_is_the_worst(bench):
    """The headline number, and the reason the project is not shipping.

    Real data: strategy -0.32 against a benchmark of +1.05, and 9 of 10 runs
    lose to it. The gate has never passed on any reduced- or full-grammar
    artifact, and this is why that verdict was correct rather than cautious.
    """
    real = bench["none"]
    assert real["strategy_median_sharpe"] < 0.0
    assert real["excess"] < -1.0
    assert real["runs_beating_buy_and_hold"] in ("0/10", "1/10")


def test_the_diagnostic_nulls_are_permanent_long_exposure(bench):
    """The nulls were not found -- they were held.

    A position that is long nearly every bar is a drift position. On iidg the
    reported Sharpe was within noise of the benchmark itself, which is the
    clearest possible statement that the search had nothing to add.
    """
    for mode in ("iid1", "iidw", "iidg"):
        assert bench[mode]["pct_bars_long"] > 0.80
        assert bench[mode]["mean_position"] > 0.10
        assert bench[mode]["strategy_median_sharpe"] < bench[mode]["buy_and_hold_sharpe"]


def test_iidg_is_essentially_the_benchmark_itself(bench):
    b = bench["iidg"]
    gap = b["buy_and_hold_sharpe"] - b["strategy_median_sharpe"]
    assert gap < 0.5
