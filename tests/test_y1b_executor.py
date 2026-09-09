"""Y1b executor shadow test: dry-run only, mock broker, no orders."""
from unittest.mock import AsyncMock
from strategy_manager.y1b_executor import build_plans, run_once, SYMBOLS
from strategy_manager.risk import RiskEngine

def _mock_broker(price=10.0):
    b = AsyncMock()
    b.get_price = AsyncMock(return_value=price)
    b.get_position = AsyncMock(return_value=None)
    b.market_open = AsyncMock(side_effect=AssertionError("must not order in shadow"))
    return b

def test_build_plans_gated():
    import asyncio
    b = _mock_broker(20.0)
    plans = asyncio.run(build_plans(b, RiskEngine(), notional=50.0))
    assert {p.coin for p in plans} == {"ETC", "TRX"}
    for p in plans:
        assert p.symbol == SYMBOLS[p.coin]
        if p.want != 0:
            assert p.gate_ok is True
            assert abs(p.size * p.price - 50.0) < 1e-6

def test_run_once_dry_run(tmp_path, monkeypatch):
    import asyncio
    monkeypatch.setenv("Y1B_STATE", str(tmp_path / "y1b.json"))
    monkeypatch.delenv("Y1B_LIVE_ENABLED", raising=False)
    b = _mock_broker(20.0)
    plans, res, sync = asyncio.run(run_once(broker=b, notional=50.0))
    assert all(r.get("dry_run") for r in res)
    b.market_open.assert_not_awaited()

def test_run_once_returns_sync(tmp_path, monkeypatch):
    import asyncio
    monkeypatch.setenv("Y1B_STATE", str(tmp_path / "y1b.json"))
    b = _mock_broker(20.0)
    b.get_position = AsyncMock(return_value=None)
    plans, res, sync = asyncio.run(run_once(broker=b, notional=50.0))
    assert isinstance(sync, list)
    b.market_open.assert_not_awaited()

def test_stop_blocks_live(tmp_path, monkeypatch):
    import asyncio
    stop = tmp_path / "STOP"
    stop.write_text("STOP")
    monkeypatch.setenv("STOP_SIGNAL_PATH", str(stop))
    monkeypatch.setenv("Y1B_STATE", str(tmp_path / "y1b.json"))
    monkeypatch.setenv("Y1B_LIVE_ENABLED", "1")
    monkeypatch.delenv("PAPER_MODE", raising=False)
    b = _mock_broker(20.0)
    b.get_position = AsyncMock(return_value=None)
    b.enable_deadman = AsyncMock(return_value=True)
    plans, res, sync = asyncio.run(run_once(broker=b, notional=50.0, dry_run=False))
    assert res and res[0].get("blocked") is True
    b.market_open.assert_not_awaited()

def test_funding_gate_blocks_high_funding():
    import asyncio
    b = _mock_broker(20.0)
    b.get_funding_rate = AsyncMock(return_value=0.05)
    plans = asyncio.run(build_plans(b, RiskEngine(), notional=50.0))
    gated = [p for p in plans if p.want != 0]
    assert gated and all(p.gate_ok is False for p in gated)

def test_funding_mock_attr_ignored():
    import asyncio
    from unittest.mock import MagicMock
    b = _mock_broker(20.0)
    b.get_funding_rate = MagicMock(return_value=MagicMock())
    plans = asyncio.run(build_plans(b, RiskEngine(), notional=50.0))
    assert all(p.gate_ok is True for p in plans if p.want != 0)

def test_gate_failed_flip_closes_only(tmp_path, monkeypatch):
    import asyncio
    from execution.brokers.base import VenuePosition, Venue
    monkeypatch.setenv("Y1B_STATE", str(tmp_path / "y1b.json"))
    monkeypatch.setenv("Y1B_LIVE_ENABLED", "1")
    monkeypatch.delenv("PAPER_MODE", raising=False)
    b = _mock_broker(20.0)
    b.get_funding_rate = AsyncMock(return_value=0.05)  # force gate fail
    b.enable_deadman = AsyncMock(return_value=True)
    # held LONG vs want SHORT -> flip path, gate fails -> close-only
    b.get_position = AsyncMock(return_value=VenuePosition(
        venue=Venue.ASTER, symbol="ETCUSDT", side="LONG", size=1.0))
    b.market_open = AsyncMock(return_value=type("R", (), {
        "ok": True, "oid": "x1", "fill_price": 20.0, "reason": ""})())
    from strategy_manager.y1b_executor import run_once
    plans, res, sync = asyncio.run(run_once(broker=b, notional=50.0, dry_run=False))
    closes = [a for a in sync if a.get("closed_only_gate_fail")]
    flips = [a for a in sync if a.get("flipped")]
    assert closes and not flips
    b.market_open.assert_awaited()

def test_deadman_fail_blocks_live_no_orders(tmp_path, monkeypatch):
    import asyncio
    monkeypatch.setenv("Y1B_STATE", str(tmp_path / "y1b.json"))
    monkeypatch.setenv("Y1B_LIVE_ENABLED", "1")
    monkeypatch.delenv("PAPER_MODE", raising=False)
    b = _mock_broker(20.0)
    b.enable_deadman = AsyncMock(return_value=False)
    from strategy_manager.y1b_executor import run_once
    plans, res, sync = asyncio.run(run_once(broker=b, notional=50.0, dry_run=False))
    assert res == [{"blocked": True, "reason": "deadman FAILED"}]
    b.market_open.assert_not_awaited()

def test_leverage_reject_blocks_order_no_idempotent_skip(tmp_path, monkeypatch):
    import asyncio
    monkeypatch.setenv("Y1B_STATE", str(tmp_path / "y1b.json"))
    monkeypatch.setenv("Y1B_LIVE_ENABLED", "1")
    monkeypatch.delenv("PAPER_MODE", raising=False)
    b = _mock_broker(20.0)
    b.get_position = AsyncMock(return_value=None)
    b.enable_deadman = AsyncMock(return_value=True)
    b.set_leverage = AsyncMock(return_value=False)
    b.market_open = AsyncMock(side_effect=AssertionError("must not order on lev reject"))
    from strategy_manager.y1b_executor import run_once
    plans, res, sync = asyncio.run(run_once(broker=b, notional=50.0, dry_run=False))
    assert any(r.get("reason") == "leverage-rejected" for r in res)
    b.market_open.assert_not_awaited()

def test_perp_gate_rejects_oversize():
    import asyncio
    b = _mock_broker(1.0)
    plans = asyncio.run(build_plans(b, RiskEngine(), notional=99999.0))
    # capped at PERP_MAX_NOTIONAL_USDT=500 default
    for p in plans:
        if p.want != 0:
            assert p.size * p.price <= 500.0 + 1e-6
