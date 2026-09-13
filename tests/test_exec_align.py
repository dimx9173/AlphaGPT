"""P0-1: execution alignment (4h decision vs 1h poll)."""
import asyncio
from unittest.mock import AsyncMock

from strategy_manager.y1b_executor import (
    build_plans, decision_aligned, run_once)
from strategy_manager.risk import RiskEngine


def _mock_broker(price=10.0):
    b = AsyncMock()
    b.get_price = AsyncMock(return_value=price)
    b.get_position = AsyncMock(return_value=None)
    b.market_open = AsyncMock(side_effect=AssertionError("must not order"))
    b.enable_deadman = AsyncMock(return_value=True)
    b.set_leverage = AsyncMock(return_value=True)
    b.venue = "bybit"
    return b


def test_decision_aligned_default_off(monkeypatch):
    monkeypatch.delenv("Y1B_DECISION_ALIGN", raising=False)
    assert decision_aligned(1) is True
    assert decision_aligned(3) is True


def test_decision_aligned_gate(monkeypatch):
    monkeypatch.setenv("Y1B_DECISION_ALIGN", "1")
    assert decision_aligned(0) is True
    assert decision_aligned(4) is True
    assert decision_aligned(1) is False
    assert decision_aligned(23) is False


def test_decision_only_zero_opens(tmp_path, monkeypatch):
    monkeypatch.setenv("Y1B_STATE", str(tmp_path / "y1b.json"))
    monkeypatch.setenv("Y1B_LIVE_ENABLED", "1")
    monkeypatch.delenv("PAPER_MODE", raising=False)
    b = _mock_broker(20.0)
    b.market_open = AsyncMock(return_value=type("R", (), {
        "ok": True, "oid": "x", "fill_price": 20.0, "reason": ""})())
    plans, res, sync = asyncio.run(run_once(
        broker=b, risk=RiskEngine(), notional=50.0, dry_run=False,
        decision_only=True))
    b.market_open.assert_not_awaited()
    assert all(r.get("reason") == "risk-only" for r in res)


def test_nonaligned_hour_zero_opens(tmp_path, monkeypatch):
    monkeypatch.setenv("Y1B_STATE", str(tmp_path / "y1b.json"))
    monkeypatch.setenv("Y1B_LIVE_ENABLED", "1")
    monkeypatch.setenv("Y1B_DECISION_ALIGN", "1")
    monkeypatch.delenv("PAPER_MODE", raising=False)
    b = _mock_broker(20.0)
    b.market_open = AsyncMock(side_effect=AssertionError("non-aligned must not open"))
    plans, res, sync = asyncio.run(run_once(
        broker=b, risk=RiskEngine(), notional=50.0, dry_run=False, hour=1))
    b.market_open.assert_not_awaited()


def test_signal_age_present():
    from strategy_manager.y1b_basket import basket_signals
    import unittest.mock as _m
    with _m.patch("strategy_manager.y1b_basket.load_bars_4h", return_value=[(1, 2, 0.5, 1.5, 100.0)] * 32):
        r = basket_signals()
    assert "signal_age_h" in r and isinstance(r["signal_age_h"], float)
