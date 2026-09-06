"""Solana venue adapter (P4 Phase 1): wraps existing SolanaTrader, zero behavior change."""
from __future__ import annotations

from .base import (
    OrderResult,
    OrderStatus,
    Side,
    Venue,
    VenueBroker,
    VenuePosition,
)


class SolanaBroker(VenueBroker):
    """VenueBroker over the existing SolanaTrader (spot only)."""

    def __init__(self, trader=None):
        if trader is None:
            from ..trader import SolanaTrader

            trader = SolanaTrader()
        self._trader = trader

    @property
    def trader(self):
        return self._trader

    @property
    def venue(self) -> Venue:
        return Venue.SOLANA

    async def get_price(self, symbol: str, _jupiter=None) -> float:
        # symbol == Solana mint address. Quote cost of 1 unit in SOL.
        jup = _jupiter if _jupiter is not None else self._trader.jup
        from ..utils import get_mint_decimals

        decimals = await get_mint_decimals(symbol, self._trader.rpc.client)
        quote = await jup.get_quote(
            input_mint=symbol,
            output_mint=self._trader.config.SOL_MINT,
            amount_integer=10 ** decimals,
        )
        if not quote:
            return 0.0
        return int(quote["outAmount"]) / 1e9

    async def get_klines(self, symbol: str, interval: str = "1h", limit: int = 100) -> list:
        # Spot venue has no native klines; data comes from data_pipeline.
        raise NotImplementedError("spot venue: klines come from data_pipeline")

    async def get_balance(self) -> float:
        return await self._trader.rpc.get_balance()

    async def get_position(self, symbol: str) -> VenuePosition | None:
        raw = await self._trader.rpc.get_token_balance(symbol)
        if not raw:
            return None
        return VenuePosition(venue=Venue.SOLANA, symbol=symbol, size=float(raw))

    async def market_open(
        self, symbol: str, side: Side, size: float, slippage_bps: int = 500
    ) -> OrderResult:
        # For Solana spot: BUY size is SOL amount; SELL sells `size` as fraction when <=1 else full.
        if side == Side.BUY:
            ok = await self._trader.buy(symbol, size, slippage_bps=slippage_bps)
        else:
            pct = size if 0 < size <= 1 else 1.0
            ok = await self._trader.sell(symbol, percentage=pct, slippage_bps=slippage_bps)
        if ok is True or (isinstance(ok, str) and ok):
            oid = ok if isinstance(ok, str) else ""
            return OrderResult(
                status=OrderStatus.OK, venue=Venue.SOLANA, symbol=symbol, side=side, oid=oid
            )
        return OrderResult(
            status=OrderStatus.ERROR, venue=Venue.SOLANA, symbol=symbol, side=side,
            reason="solana buy/sell returned falsy",
        )

    async def limit_open(self, symbol: str, side: Side, size: float, price: float) -> OrderResult:
        raise NotImplementedError("spot venue: no limit orders via Jupiter wrapper")

    async def cancel(self, symbol: str, oid: str) -> bool:
        raise NotImplementedError("spot venue: no open-order cancel")

    async def set_leverage(self, symbol: str, leverage: int, margin: str = "isolated") -> bool:
        raise NotImplementedError("spot venue")

    async def enable_deadman(self, timeout_sec: int = 60) -> bool:
        raise NotImplementedError("spot venue")

    async def close(self) -> None:
        try:
            await self._trader.close()
        except Exception:
            pass
