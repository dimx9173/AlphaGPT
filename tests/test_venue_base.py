"""P4 Phase 1: VenueBroker abstraction unit tests (no chain)."""
from unittest.mock import AsyncMock

import pytest

from execution.brokers.base import OrderStatus, Side, Venue, VenueBroker
from execution.brokers.solana import SolanaBroker


class FakeBroker(VenueBroker):
    @property
    def venue(self):
        return Venue.HYPERLIQUID

    async def get_price(self, symbol: str) -> float:
        return 1.0

    async def get_klines(self, symbol: str, interval: str = "1h", limit: int = 100) -> list:
        return []

    async def get_balance(self) -> float:
        return 10.0

    async def get_position(self, symbol: str):
        return None

    async def market_open(self, symbol, side, size, slippage_bps=500):
        from execution.brokers.base import OrderResult

        return OrderResult(status=OrderStatus.OK, venue=self.venue, symbol=symbol, side=side)

    async def limit_open(self, symbol, side, size, price):
        from execution.brokers.base import OrderResult

        return OrderResult(status=OrderStatus.OK, venue=self.venue, symbol=symbol, side=side)

    async def cancel(self, symbol: str, oid: str) -> bool:
        return True

    async def set_leverage(self, symbol: str, leverage: int, margin: str = "isolated") -> bool:
        return True

    async def enable_deadman(self, timeout_sec: int = 60) -> bool:
        return True

    async def close(self) -> None:
        return None


def test_interface_shape():
    b = FakeBroker()
    assert b.venue == Venue.HYPERLIQUID
    assert set(v.value for v in Venue) == {"solana", "hyperliquid", "aster"}
    assert set(s.value for s in Side) == {"buy", "sell"}


@pytest.mark.asyncio
async def test_fake_broker_market_ok():
    b = FakeBroker()
    res = await b.market_open("HYPE", Side.BUY, 1.0)
    assert res.ok and res.venue == Venue.HYPERLIQUID


@pytest.mark.asyncio
async def test_solana_adapter_buy_dispatch():
    trader = AsyncMock()
    trader.config.SOL_MINT = "So111"
    trader.buy = AsyncMock(return_value=True)
    trader.sell = AsyncMock(return_value=True)
    b = SolanaBroker(trader=trader)
    assert b.venue == Venue.SOLANA
    res = await b.market_open("MintAddr", Side.BUY, 0.5)
    assert res.ok
    trader.buy.assert_awaited_once()
    res2 = await b.market_open("MintAddr", Side.SELL, 1.0)
    assert res2.ok
    trader.sell.assert_awaited_once()


@pytest.mark.asyncio
async def test_solana_adapter_buy_failure_maps_error():
    trader = AsyncMock()
    trader.config.SOL_MINT = "So111"
    trader.buy = AsyncMock(return_value=False)
    b = SolanaBroker(trader=trader)
    res = await b.market_open("MintAddr", Side.BUY, 0.5)
    assert not res.ok and res.status == OrderStatus.ERROR


@pytest.mark.asyncio
async def test_solana_spot_perp_methods_raise():
    trader = AsyncMock()
    b = SolanaBroker(trader=trader)
    with pytest.raises(NotImplementedError):
        await b.set_leverage("MintAddr", 3)
    with pytest.raises(NotImplementedError):
        await b.enable_deadman(60)
    with pytest.raises(NotImplementedError):
        await b.limit_open("MintAddr", Side.BUY, 1.0, 1.0)
