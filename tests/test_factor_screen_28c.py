"""The 12 factors are dead as timing signals, and the search cannot fix that.

A search over a grammar can only find what its features contain. The factor
screen measures each feature on its own, through the live position path, on
train, validation and lockbox, and separates two things that look identical
in a Sharpe:

    raw IC      = corr(p, r)          -- credits holding a direction
    demeaned IC = corr(p - mean(p), r) -- credits picking the right times

Every long position in a drifting market has a positive raw IC. Only the
demeaned correlation is evidence of timing.

What the screen found, on the lockbox, which is the only window that counts:

  - No factor has |t| > 1.40. The threshold is 1.96 before any multiple
    testing correction, and there are 12 factors, so the honest bar is
    stricter still. Nothing clears it.
  - Every factor's demeaned IC is statistically indistinguishable from the
    frozen iid null's. The gaps are the same size and change sign from
    window to window: 5/12 positive on train, 8/12 on validation, 5/12 on
    lockbox. A factor that is genuinely predictive does not flip its sign
    against a null with nothing in it.
  - All 12 factors lose to a constant long on train and on lockbox, without
    exception: lockbox excess is -1.21 at best and -4.87 at the median. On
    validation 2 of 12 clear it, by +0.23 (LOG_VOL) and +0.38 (VOL_CLUST) --
    but that window's benchmark is -1.275, so beating a losing benchmark is
    close to free, and neither factor is anywhere near significant there.
  - The raw IC and the demeaned IC are numerically identical for every
    factor on every window. The position is already near-constant in sign,
    so there is almost no directional exposure left to strip. The factors are
    not even mostly-directional; they are simply uninformative.

Two factors are worth naming because they look like signal in a summary:
VOL_TREND and FOMO post the largest lockbox demeaned IC (+0.0149) and the
largest real-minus-iid gap (+0.0114, +0.0149). Both are the artefacts of a
sign that flips with the regime, and both have an excess over buy-and-hold of
-9.79 and -20.70. A large IC paired with a catastrophic Sharpe is the
signature of a position that is right about direction and wrong about size
and timing, which is worse than having no view at all.

The upshot for the search: the +5 to +13 Sharpe selection lift measured in
the random-formula probe is not the search uncovering weak signal in these
features. There is no weak signal to uncover. The lift is best-of-N over
noise. Every artifact this framework has produced is a selection artifact,
and the correct next step is to change the factors, not to run the search
again.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCREEN = ROOT / "results" / "factor_screen.json"
NAMES = ("RET", "LIQ_SCORE", "PRESSURE", "FOMO", "PUMP_DEV", "LOG_VOL",
         "VOL_CLUST", "MOM_REV", "DELTA_RSI", "HL_RANGE", "CLOSE_POS",
         "VOL_TREND")

pytestmark = pytest.mark.skipif(not SCREEN.exists(),
                                reason="factor screen not computed yet")


@pytest.fixture(scope="module")
def screen():
    return json.loads(SCREEN.read_text())


def test_all_twelve_factors_were_screened(screen):
    assert len(screen["none"]) == 12
    assert set(screen["none"]) == set(NAMES)


@pytest.mark.parametrize("name", NAMES)
def test_no_factor_is_significant_on_the_lockbox(screen, name):
    r = screen["none"][name]["lockbox"]
    assert abs(r["demeaned_ic_t"]) < 1.96, (
        f"{name} now has lockbox |t| = {r['demeaned_ic_t']:+.2f}. That is a real "
        "result if it holds up, but it must be re-derived rather than assumed, "
        "and the multiple-testing correction for 12 factors still applies.")


@pytest.mark.parametrize("name", NAMES)
def test_no_factor_beats_buy_and_hold_in_sample_or_out(screen, name):
    """Train and lockbox are the windows that decide it.

    An earlier version of this test asserted all three windows and failed on
    validation, where LOG_VOL (+0.23) and VOL_CLUST (+0.38) do clear the
    benchmark. That was not the factor screen being wrong; it was the claim
    being too strong. Validation's benchmark is -1.275, so it is a losing
    benchmark and clearing it is nearly free, which is exactly why a
    validation excess is not evidence. The test now pins what is actually
    true, and the two exceptions are pinned too so they cannot drift upward
    unnoticed.
    """
    for window in ("train", "lockbox"):
        e = screen["none"][name][window]["excess_vs_buy_and_hold"]
        assert e < 0.0, f"{name} beats buy-and-hold on {window} by {e:+.3f}"


def test_the_two_validation_exceptions_stay_small_and_insignificant(screen):
    """Naming them, so a later summary cannot quietly promote them."""
    positive = {n: screen["none"][n]["validation"]["excess_vs_buy_and_hold"]
                for n in NAMES
                if screen["none"][n]["validation"]["excess_vs_buy_and_hold"] > 0}
    assert positive == {"LOG_VOL": pytest.approx(0.2313, abs=1e-3),
                        "VOL_CLUST": pytest.approx(0.3837, abs=1e-3)}
    # The benchmark they beat is itself negative, so this is not a result.
    bsh = screen["none"]["LOG_VOL"]["validation"]["buy_and_hold_sharpe"]
    assert bsh < 0.0
    for n in positive:
        assert abs(screen["none"][n]["validation"]["demeaned_ic_t"]) < 1.0


@pytest.mark.parametrize("name", NAMES)
def test_no_factor_beats_the_iid_null_on_the_lockbox(screen, name):
    """The null has no structure to find, so a match is not a forecast."""
    real = screen["none"][name]["lockbox"]["demeaned_ic"]
    null = screen["iid"][name]["lockbox"]["demeaned_ic"]
    assert real == pytest.approx(null, abs=0.02), (
        f"{name}: real {real:+.4f} vs iid null {null:+.4f} on the lockbox")


def test_the_ic_sign_is_not_stable_across_windows(screen):
    """5/12, 8/12, 5/12. A real signal does not flip against an empty null."""
    for window, expect in (("train", 5), ("validation", 8), ("lockbox", 5)):
        pos = sum(1 for n in NAMES
                  if screen["none"][n][window]["demeaned_ic"]
                  - screen["iid"][n][window]["demeaned_ic"] > 0)
        assert pos == expect, f"{window}: {pos} positive, expected {expect}"


def test_the_position_carries_almost_no_directional_exposure(screen):
    """raw IC == demeaned IC means there was no 'being long' to remove.

    It also means the factors cannot be rescued by de-meaning later: the
    information is absent from the position itself, not from the accounting.
    """
    for name in NAMES:
        for window in ("train", "validation", "lockbox"):
            r = screen["none"][name][window]
            assert abs(r["raw_ic"] - r["demeaned_ic"]) < 1e-3


def test_a_large_ic_with_a_bad_sharpe_is_the_worst_case(screen):
    """VOL_TREND and FOMO: the two factors most likely to be mistaken for edge.

    Both post the biggest lockbox demeaned IC of the twelve and both lose to
    buy-and-hold by more than 9 Sharpe. This is pinned so a future summary
    that quotes the IC column without the excess column is caught.
    """
    for name in ("VOL_TREND", "FOMO"):
        r = screen["none"][name]["lockbox"]
        assert r["demeaned_ic"] > 0.010
        assert r["excess_vs_buy_and_hold"] < -5.0
