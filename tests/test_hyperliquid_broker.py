"""P4 Phase 2: HyperliquidBroker unit tests (mocked SDK, no network)."""
from unittest.mock import MagicMock

import pytest

from execution.brokers.base import OrderStatus, Side, Venue
from execution.brokers.hyperliquid import HyperliquidBroker


def _broker(info=None, exchange=None):
    return HyperliquidBroker(
        info=info, exchange=exchange,
        base_url="https://api.hyperliquid-testnet.xyz",
        account_address="0x0000000000000000000000000000000000000001",
    )


@pytest.mark.asyncio
async def test_get_price_from_mids():
    info = MagicMock()
    info.all_mids.return_value = {"BTC": "67000.5", "HYPE": "25.1"}
    b = _broker(info=info)
    assert await b.get_price("BTC") == pytest.approx(67000.5)
    assert await b.get_price("UNKNOWN") == 0.0


@pytest.mark.asyncio
async def test_get_position_long_and_empty():
    info = MagicMock()
    info.user_state.return_value = {
        "marginSummary": {"accountValue": "1000"},
        "assetPositions": [{
            "position": {"coin": "HYPE", "szi": "10", "entryPx": "25",
                         "leverage": {"value": 3}},
        }],
    }
    b = _broker(info=info)
    pos = await b.get_position("HYPE")
    assert pos is not None and pos.side == "LONG"
    assert pos.size == pytest.approx(10.0) and pos.leverage == pytest.approx(3.0)
    assert await b.get_position("BTC") is None
    assert await b.get_balance() == pytest.approx(1000.0)


@pytest.mark.asyncio
async def test_market_open_ok_and_rejected():
    ok_resp = {"status": "ok",
               "response": {"data": {"statuses": [
                   {"filled": {"avgPx": "25.2", "totalSz": "10"}}]}}}
    ex = MagicMock()
    ex.market_open.return_value = ok_resp
    b = _broker(info=MagicMock(), exchange=ex)
    res = await b.market_open("HYPE", Side.BUY, 10.0)
    assert res.ok and res.fill_price == pytest.approx(25.2)
    assert res.venue == Venue.HYPERLIQUID

    rej_resp = {"status": "ok",
                "response": {"data": {"statuses": [{"error": "insufficient"}]}}}
    ex.market_open.return_value = rej_resp
    res2 = await b.market_open("HYPE", Side.BUY, 10.0)
    assert res2.status == OrderStatus.REJECTED


@pytest.mark.asyncio
async def test_market_open_exchange_error_maps_error():
    ex = MagicMock()
    ex.market_open.return_value = {"status": "error", "response": "bad"}
    b = _broker(info=MagicMock(), exchange=ex)
    res = await b.market_open("HYPE", Side.SELL, 1.0)
    assert not res.ok and res.status == OrderStatus.ERROR


@pytest.mark.asyncio
async def test_limit_cancel_leverage_deadman():
    ex = MagicMock()
    ex.order.return_value = {"status": "ok",
                             "response": {"data": {"statuses": [
                                 {"resting": {"oid": 123}}]}}}
    ex.cancel.return_value = {"status": "ok", "response": {}}
    ex.update_leverage.return_value = {"status": "ok", "response": {}}
    ex.schedule_cancel.return_value = {"status": "ok", "response": {}}
    info = MagicMock()
    info.name_to_asset.return_value = 1
    b = _broker(info=info, exchange=ex)
    lim = await b.limit_open("ETH", Side.BUY, 0.2, 3000.0)
    assert lim.ok and lim.oid == "123"
    assert await b.cancel("ETH", "123") is True
    assert await b.set_leverage("ETH", 3) is True
    assert await b.enable_deadman(60) is True


@pytest.mark.asyncio
async def test_missing_secret_maps_error(monkeypatch):
    from execution import config as cfg

    monkeypatch.setenv("HYPERLIQUID_SECRET_KEY", "")
    b = HyperliquidBroker(info=MagicMock(), exchange=None,
                          base_url="https://x", account_address="0xabc")
    # force exchange construction path with empty secret
    monkeypatch.setattr(cfg.HyperliquidConfig, "secret_key",
                        classmethod(lambda cls: ""), raising=False)
    res = await b.market_open("HYPE", Side.BUY, 1.0)
    assert not res.ok
