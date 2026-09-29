"""The random-formula probe, and what it means for every reported Sharpe.

Random formulas score strongly NEGATIVE, not mildly positive: the median
over 400 draws is -5.5 on the real contract and -6.9 on the iid null, with
only 4-6% positive. So the position construction is not generically
profitable, and the GA's reported +0.8 is a selection effect of roughly
+5 to +8 Sharpe units applied to a population centred near -6.

That is a property of the search, not of the data, and it is about five
times larger than any real-vs-null difference this contract can resolve. The
test below reads the recorded probe artifact rather than re-running the
400-formula sweep, so the claim stays cheap to verify while the numbers
themselves stay auditable.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "results" / "random_formula_probe.json"

pytestmark = pytest.mark.skipif(not PROBE.exists(),
                                reason="run research/probe_random_formulas_28c.py first")


@pytest.fixture(scope="module")
def probe():
    return json.loads(PROBE.read_text())


def test_random_formulas_are_negative_not_merely_bland(probe):
    """The falsifiable direction of the claim.

    If the position construction were mildly profitable for anything, the
    median would sit slightly above zero and the whole reading changes. It
    sits at -5.5, so "the search finds structure" survives and "everything
    is profitable" does not.
    """
    for name, d in probe.items():
        assert d["median_sharpe"] < -1.0, (
            f"{name}: random median {d['median_sharpe']} is not clearly "
            "negative; the probe's conclusion would need revisiting")


def test_random_formulas_are_rarely_positive(probe):
    for name, d in probe.items():
        assert d["positive_rate"] < 0.15, (
            f"{name}: {d['positive_rate']:.0%} of random formulas are "
            "positive, which is not the near-zero rate claimed")


def test_real_population_sits_above_the_nulls(probe):
    """Diagnostic nulls should make the search's job harder, not easier.

    Every diagnostic mode removes a property the iid null preserves, so each
    should push the random population DOWN. If a diagnostic null scored
    ABOVE iid it would be adding exploitable structure rather than removing
    it, and the attribution it exists to support would be void.
    """
    base = probe["iid"]["median_sharpe"]
    for name in ("iid1", "iidw", "iidg"):
        assert probe[name]["median_sharpe"] < base, (
            f"{name} median {probe[name]['median_sharpe']} is not below "
            f"iid {base}")
    assert probe["none"]["median_sharpe"] > base, (
        "the real contract should be the easiest population to beat, not "
        "the hardest")


def test_reported_sharpe_is_dominated_by_selection(probe):
    """Quantify the artifact against the GA's own recorded output.

    If the GA's reported median is within about one unit of the random
    population median, the search is not selecting and the probe says
    something different. It is several units above, so the selection effect
    is the dominant term in the reported number.
    """
    import statistics as st
    runs = ROOT / "results" / "gram_reduced_runs"
    if not runs.exists():
        pytest.skip("reduced-grammar batch not present")
    real = [json.loads(f.read_text())["oos"]["portfolio_sharpe"]
            for f in sorted(runs.glob("real_*.json"))]
    if not real:
        pytest.skip("no real runs recorded")
    lift = st.median(real) - probe["none"]["median_sharpe"]
    assert lift > 3.0, f"selection lift is only {lift:.2f}"
