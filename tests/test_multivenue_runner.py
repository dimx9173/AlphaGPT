"""P4 Phase 4: multivenue runner injection + reconcile + deadman (no chain)."""
from unittest.mock import AsyncMock

import pytest

from execution.brokers.base import VenuePosition
from strategy_manager.portfolio import PortfolioManager


def _make_runner(monkeypatch, tmp_path):
    import strategy_manager.runner as R

    monkeypatch.setenv("STOP_SIGNAL_PATH", str(tmp_path / "STOP"))
    # avoid StrategyRunner.__init__ side effects (DataManager/Trader/file load)
    runner = R.StrategyRunner.__new__(R.StrategyRunner)
    runner.portfolio = PortfolioManager(
        state_file=str(tmp_path / "portfolio_state.json"))
    runner.brokers = {}
    import logging
    return runner, R


@pytest.mark.asyncio
async def test_reconcile_all_per_venue(monkeypatch, tmp_path):
    runner, _ = _make_runner(monkeypatch, tmp_path)
    runner.portfolio.add_position("HYPE", "HYPE", 25.0, 10.0, 0.0,
                                  venue="hyperliquid")
    runner.portfolio.add_position("TokenA", "TKA", 1.0, 5.0, 1.0,
                                  venue="solana")
    hl = AsyncMock()
    hl.get_position = AsyncMock(return_value=VenuePosition(
        venue="hyperliquid", symbol="HYPE", size=7.0))
    sol = AsyncMock()
    sol.get_position = AsyncMock(return_value=VenuePosition(
        venue="solana", symbol="TokenA", size=5.0))
    runner.brokers = {"hyperliquid": hl, "solana": sol}
    await runner._reconcile_all()
    assert runner.portfolio.positions["hyperliquid::HYPE"].amount_held == 7.0
    hl.get_position.assert_awaited_once_with("HYPE")


@pytest.mark.asyncio
async def test_reconcile_zero_closes_venue_position(monkeypatch, tmp_path):
    runner, _ = _make_runner(monkeypatch, tmp_path)
    runner.portfolio.add_position("ASTERUSDT", "ASTER", 0.75, 20.0, 0.0,
                                  venue="aster")
    ax = AsyncMock()
    ax.get_position = AsyncMock(return_value=None)
    runner.brokers = {"aster": ax}
    await runner._reconcile_all()
    assert "aster::ASTERUSDT" not in runner.portfolio.positions


@pytest.mark.asyncio
async def test_refresh_deadmen_skips_solana(monkeypatch, tmp_path):
    runner, _ = _make_runner(monkeypatch, tmp_path)
    hl = AsyncMock()
    hl.enable_deadman = AsyncMock(return_value=True)
    sol = AsyncMock()
    sol.enable_deadman = AsyncMock(return_value=True)
    runner.brokers = {"solana": sol, "hyperliquid": hl}
    await runner.refresh_deadmen(60)
    hl.enable_deadman.assert_awaited_once_with(60)
    sol.enable_deadman.assert_not_awaited()
