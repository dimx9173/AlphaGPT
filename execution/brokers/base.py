"""VenueBroker abstraction (P4 Phase 1).

Defines the minimal venue interface so Hyperliquid / Aster can plug in
without changing the Solana path. Shape-only: no network here.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Venue(str, Enum):
    SOLANA = "solana"
    HYPERLIQUID = "hyperliquid"
    ASTER = "aster"
    BINANCE = "binance"
    BYBIT = "bybit"
    OKX = "okx"


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(str, Enum):
    OK = "ok"
    ERROR = "error"
    REJECTED = "rejected"


@dataclass
class OrderResult:
    status: OrderStatus
    venue: Venue
    symbol: str
    side: Side
    oid: str = ""
    fill_price: float = 0.0
    fill_size: float = 0.0
    raw: Any = None
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.status == OrderStatus.OK


@dataclass
class VenuePosition:
    venue: Venue
    symbol: str
    side: str = "LONG"
    size: float = 0.0
    entry_price: float = 0.0
    leverage: float = 1.0
    notional: float = 0.0
    raw: Any = field(default=None)


class VenueBroker(ABC):
    """Minimal async venue interface (P4)."""

    @property
    @abstractmethod
    def venue(self) -> Venue:
        raise NotImplementedError

    @abstractmethod
    async def get_price(self, symbol: str) -> float:
        raise NotImplementedError

    @abstractmethod
    async def get_klines(self, symbol: str, interval: str = "1h", limit: int = 100) -> list:
        raise NotImplementedError

    @abstractmethod
    async def get_balance(self) -> float:
        raise NotImplementedError

    @abstractmethod
    async def get_position(self, symbol: str) -> VenuePosition | None:
        raise NotImplementedError

    @abstractmethod
    async def market_open(
        self, symbol: str, side: Side, size: float, slippage_bps: int = 500
    ) -> OrderResult:
        raise NotImplementedError

    @abstractmethod
    async def limit_open(
        self, symbol: str, side: Side, size: float, price: float
    ) -> OrderResult:
        raise NotImplementedError

    @abstractmethod
    async def cancel(self, symbol: str, oid: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def set_leverage(self, symbol: str, leverage: int, margin: str = "isolated") -> bool:
        raise NotImplementedError

    @abstractmethod
    async def enable_deadman(self, timeout_sec: int = 60) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def close(self) -> None:
        raise NotImplementedError
