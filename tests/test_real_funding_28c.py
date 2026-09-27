#!/usr/bin/env python3
"""Real-funding loading: the semantics that are easy to get wrong.

The failure these guard against is not a crash. It is a coverage report that
looks like a data gap when the data is actually complete, which pushes someone
to "fix" it by imputing a constant and reintroducing the original bias.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.accounting_28c import (  # noqa: E402
    ACCOUNTING_VERSION,
    ACCOUNTING_VERSION_REAL,
    FUNDING_HOURS,
    load_real_funding,
    real_funding_coverage,
    scheduled_funding_rates,
)


def _grid(n_days: int = 40) -> np.ndarray:
    """A 30m grid that starts exactly on a UTC midnight."""
    start = 1726200000000 - (1726200000000 % 86400000)
    return np.arange(start, start + n_days * 86400000, 1800000, dtype=np.int64)


def test_real_funding_never_invents_a_positive_rate():
    """A signed history must be able to produce negative funding.

    The v2 constant was unconditionally positive, so shorts were credited at
    every event. If the loader cannot emit a negative value, the real data is
    not actually being used.
    """
    grid = _grid()
    rates = load_real_funding("BTC", grid)
    assert (rates < 0).any(), "real BTC funding must contain negative events"


def test_real_funding_magnitude_is_below_the_v2_constant():
    """The v2 constant overstated the typical rate by roughly an order of magnitude."""
    grid = _grid(n_days=400)
    modelled = scheduled_funding_rates(grid, 0.0005)
    modelled = modelled[modelled != 0]
    real = load_real_funding("BTC", grid)
    real = real[real != 0]
    assert np.median(np.abs(real)) < 0.2 * np.median(modelled)


def test_zero_rate_event_is_not_reported_as_missing():
    """A matched rate of exactly 0.0 is an observation, not an absence.

    BNB carries a genuine 0.0 rate on most of its events. Counting nonzero
    values as coverage reports BNB as half-missing and invites an imputation
    that would put the v2 constant straight back in.
    """
    # Start inside the stored history: the first contract bar predates the
    # funding download, so a grid anchored at the window start is short by one
    # event at each edge. That boundary shortfall is real and is reported as
    # such rather than being padded away.
    grid = _grid()[2:]
    cov = real_funding_coverage(["BNB"], grid)
    bnb = cov["BNB"]
    assert bnb["events_matched"] == bnb["scheduled_events"], "BNB record is complete"
    assert bnb["events_zero_rate"] > 0, "BNB should have genuine zero-rate events"
    assert bnb["has_history"] is True


def test_coverage_agrees_with_a_direct_count():
    """Coverage and the loaded series must not disagree about the same data."""
    grid = _grid()
    cov = real_funding_coverage(["BTC", "ETH"], grid)
    for coin, info in cov.items():
        nonzero = int(np.count_nonzero(load_real_funding(coin, grid)))
        assert nonzero == info["events_nonzero"]


def test_funding_lands_only_on_scheduled_bars():
    """Off-schedule bars must stay zero, including the widened match window."""
    grid = _grid()
    rates = load_real_funding("BNB", grid)
    scheduled = scheduled_funding_rates(grid, 1.0) != 0
    assert not np.count_nonzero(rates[~scheduled])


def test_missing_coin_returns_zeros_rather_than_raising():
    """Absence of data must not crash an evaluation mid-run."""
    grid = _grid()
    assert np.count_nonzero(load_real_funding("NOPE", grid)) == 0


def test_versions_are_distinct():
    """v2 and v3 must stay separately named or old results become ambiguous."""
    assert ACCOUNTING_VERSION != ACCOUNTING_VERSION_REAL
    assert ACCOUNTING_VERSION_REAL.endswith("real-funding")


def test_fetch_tool_knows_the_pepe_alias():
    """PEPE has no PEPEUSDT history; without the alias it silently gets zeros."""
    src = (ROOT / "tools" / "fetch_real_funding_28c.py").read_text()
    assert "1000PEPEUSDT" in src


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
