"""Venue brokers: multi-exchange execution abstraction (P4)."""
from .base import Venue, Side, OrderType, OrderStatus, OrderResult, VenuePosition, VenueBroker

__all__ = ["Venue", "Side", "OrderType", "OrderStatus", "OrderResult", "VenuePosition", "VenueBroker"]


def make_venue_broker(name: str, **kwargs):
    """Create a venue broker by name (lazy import, no network)."""
    key = (name or "").strip().lower()
    if key == "binance":
        from .binance import BinanceBroker
        return BinanceBroker(**kwargs)
    if key == "bybit":
        from .bybit import BybitBroker
        return BybitBroker(**kwargs)
    if key == "okx":
        from .okx import OkxBroker
        return OkxBroker(**kwargs)
    if key == "aster":
        from .aster import AsterBroker
        return AsterBroker(**kwargs)
    if key == "hyperliquid":
        from .hyperliquid import HyperliquidBroker
        return HyperliquidBroker(**kwargs)
    if key == "solana":
        from .solana import SolanaBroker
        return SolanaBroker(**kwargs)
    raise ValueError(f"unknown venue: {name!r}")
