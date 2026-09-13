"""P0-2: swap gates (hyst / min-hold / cost)."""
import asyncio
from unittest.mock import AsyncMock

from strategy_manager.y1b_executor import (
    apply_swap_gates, cost_k, hyst_eps, min_hold_bars, run_once,
    swap_gates_on)
from strategy_manager.risk import RiskEngine


def test_gates_default_off(monkeypatch):
    for k in ("Y1B_HYST_EPS", "Y1B_MIN_HOLD_BARS", "Y1B_COST_K"):
        monkeypatch.delenv(k, raising=False)
    assert swap_gates_on() is False
    assert hyst_eps() == 0.0 and min_hold_bars() == 0 and cost_k() == 0.0


def test_hyst_blocks_weak_signal(monkeypatch):
    monkeypatch.setenv("Y1B_HYST_EPS", "0.1")
    w, n = apply_swap_gates(1.0, 0.55, "ETC")
    assert w == 0.0 and n == "hyst"
    w2, n2 = apply_swap_gates(-1.0, 0.2, "ETC")
    assert w2 == -1.0 and n2 == ""


def test_cost_blocks_thin_edge(monkeypatch):
    monkeypatch.setenv("Y1B_COST_K", "2")
    monkeypatch.delenv("Y1B_HYST_EPS", raising=False)
    w, n = apply_swap_gates(1.0, None, "ETC", edge_per_unit=0.0001,
                            fee2x=0.0008, slip_bp=5.0, funding=0.0005)
    assert w == 0.0 and n == "cost"
    w2, n2 = apply_swap_gates(1.0, None, "ETC", edge_per_unit=0.05,
                              fee2x=0.0008, slip_bp=5.0, funding=0.0005)
    assert w2 == 1.0 and n2 == ""


def _mock_broker(price=10.0):
    b = AsyncMock()
    b.get_price = AsyncMock(return_value=price)
    b.get_position = AsyncMock(return_value=None)
    b.market_open = AsyncMock(side_effect=AssertionError("must not open"))
    b.enable_deadman = AsyncMock(return_value=True)
    b.set_leverage = AsyncMock(return_value=True)
    b.venue = "bybit"
    return b


def test_min_hold_blocks_flip(tmp_path, monkeypatch):
    import time
    from strategy_manager.portfolio import PortfolioManager
    from strategy_manager.y1b_executor import sync_positions, Plan
    from execution.brokers.base import Side, VenuePosition, Venue
    monkeypatch.setenv("Y1B_MIN_HOLD_BARS", "2")
    pm = PortfolioManager(state_file=str(tmp_path / "y1b.json"))
    pm.add_position("ETCUSDT", "ETC", 20.0, 1.0, 0.0, venue="bybit",
                    side="LONG", leverage=2.0)
    b = _mock_broker(20.0)
    b.get_position = AsyncMock(return_value=VenuePosition(
        venue=Venue.BYBIT, symbol="ETCUSDT", side="LONG", size=1.0))
    b.market_open = AsyncMock(side_effect=AssertionError("min-hold must block flip"))
    plans = [Plan("ETC", "ETCUSDT", -1.0, Side.SELL, 1.0, 20.0, True, "")]
    acts = asyncio.run(sync_positions(b, plans, pm, True))
    assert any(a.get("reason") == "min-hold" for a in acts)
    b.market_open.assert_not_awaited()


def test_gates_off_zero_behavior_change(tmp_path, monkeypatch):
    for k in ("Y1B_HYST_EPS", "Y1B_MIN_HOLD_BARS", "Y1B_COST_K"):
        monkeypatch.delenv(k, raising=False)
    w, n = apply_swap_gates(-1.0, 0.51, "ETC", edge_per_unit=0.0)
    assert (w, n) == (-1.0, "")
