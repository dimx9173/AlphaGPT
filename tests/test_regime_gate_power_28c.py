"""The gate verdict is a screening result, not statistical evidence.

These tests pin the distinction so a future change cannot quietly drop the
evidence class and let a "pass" be read as proof.
"""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.regime_gate_28c import statistical_power


def test_lockbox_cannot_resolve_the_gate_sharpe_threshold():
    """8833 bars is the real v3c lockbox: 184 daily observations."""
    p = statistical_power(8833)
    assert p["oos_days"] == 184
    assert p["sharpe_ci95_half_width"] > 3.0
    assert p["can_resolve_sharpe_1_0"] is False


def test_half_width_shrinks_with_length_but_stays_useless_here():
    short = statistical_power(8833)["sharpe_ci95_half_width"]
    long = statistical_power(8833 * 20)["sharpe_ci95_half_width"]
    assert long < short
    # even 20x the lockbox does not reach a usable interval
    assert long > 0.5


def test_achieved_t_statistic_tracks_the_observed_sharpe():
    p = statistical_power(8833, observed_sharpe=0.4789)
    assert p["achieved_t_statistic"] == pytest.approx(0.340, abs=0.01)
    assert p["achieved_t_statistic"] < p["t_required_for_95pct"]


def test_a_strong_sharpe_still_cannot_be_resolved():
    """Even Sharpe 2.0 on this lockbox is inside the noise band."""
    p = statistical_power(8833, observed_sharpe=2.0)
    assert p["can_resolve_sharpe_1_0"] is False
    assert p["sharpe_ci95_half_width"] > p["achieved_t_statistic"] / p["t_required_for_95pct"]


def test_note_says_criteria_are_screening_rules():
    assert "screening" in statistical_power(8833)["note"]
