"""Binance USDT-M futures venue broker.

Public reads: GET /fapi/v1/ticker/price, /fapi/v1/klines, /fapi/v1/premiumIndex.
Signed: GET /fapi/v2/balance, /fapi/v2/positionRisk; POST /fapi/v1/order,
/fapi/v1/leverage, /fapi/v1/marginType, /fapi/v1/countdownCancelAll;
DELETE /fapi/v1/order. Session injectable for unit tests.

Auth: HMAC-SHA256 of the query string (params + timestamp + recvWindow)
with the API secret (hex digest), header X-MBX-APIKEY.
No keys needed at construction; missing keys -> writes return
OrderResult ERROR (never raise), reads return defaults, other
mutations return False. No network at import.
"""
from __future__ import annotations

import hashlib
import hmac
import time
import urllib.parse

import aiohttp
from loguru import logger

from .base import OrderResult, OrderStatus, Side, Venue, VenueBroker, VenuePosition
from .cex_config import BinanceConfig

RECV_WINDOW = 5000
_TIMEOUT = 10


def _err(symbol: str, side: Side, reason: str, raw=None) -> OrderResult:
    return OrderResult(
        status=OrderStatus.ERROR, venue=Venue.BINANCE,
        symbol=symbol, side=side, reason=reason, raw=raw,
    )


class BinanceBroker(VenueBroker):
    def __init__(self, api_key: str | None = None,
                 api_secret: str | None = None,
                 base_url: str | None = None,
                 session: aiohttp.ClientSession | None = None):
        self._api_key = api_key
        self._api_secret = api_secret
        self._base = base_url
        self._session = session
        self._owns_session = False
        self._deadman_symbols: list[str] = []

    @property
    def venue(self) -> Venue:
        return Venue.BINANCE

    def _base_url(self) -> str:
        return self._base or BinanceConfig.base_url()

    def _keys(self) -> tuple[str, str]:
        return (self._api_key or BinanceConfig.api_key(),
                self._api_secret or BinanceConfig.api_secret())

    async def _session_or_create(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession()
            self._owns_session = True
        return self._session

    def _sign(self, secret: str, params: dict) -> tuple[str, str]:
        """Return (query_string, signature) for the given params."""
        query = urllib.parse.urlencode({
            k: v for k, v in params.items() if v is not None
        })
        sig = hmac.new(secret.encode(), query.encode(),
                       hashlib.sha256).hexdigest()
        return query, sig

    async def _public_get(self, path: str, params: dict | None = None):
        sess = await self._session_or_create()
        url = f"{self._base_url()}{path}"
        try:
            async with sess.get(url, params=params or {},
                                timeout=_TIMEOUT) as resp:
                if resp.status != 200:
                    logger.warning(f"[binance] GET {path} -> {resp.status}")
                    return None
                return await resp.json()
        except Exception as e:
            logger.error(f"[binance] GET {path} failed: {e}")
            return None

    async def _signed(self, method: str, path: str,
                      params: dict | None = None):
        """Signed request. Returns (data, err); data None on failure.

        Missing keys -> (None, reason) without touching the network.
        """
        key, secret = self._keys()
        if not key or not secret:
            return None, "missing BINANCE_API_KEY / BINANCE_API_SECRET"
        sess = await self._session_or_create()
        fields = dict(params or {})
        fields["timestamp"] = int(time.time() * 1000)
        fields["recvWindow"] = RECV_WINDOW
        query, sig = self._sign(secret, fields)
        url = f"{self._base_url()}{path}?{query}&signature={sig}"
        headers = {"X-MBX-APIKEY": key}
        try:
            async with sess.request(method, url, headers=headers,
                                    timeout=_TIMEOUT) as resp:
                try:
                    data = await resp.json()
                except Exception:
                    data = {"_text": await resp.text()}
                if resp.status != 200:
                    msg = data.get("msg") if isinstance(data, dict) else data
                    return None, f"http {resp.status}: {msg}"
                if isinstance(data, dict) and int(data.get("code", 0) or 0) != 0:
                    return None, f"code {data.get('code')}: {data.get('msg')}"
                return data, ""
        except Exception as e:
            logger.error(f"[binance] {method} {path} failed: {e}")
            return None, str(e)

    # -- reads --
    async def get_price(self, symbol: str) -> float:
        data = await self._public_get("/fapi/v1/ticker/price",
                                      {"symbol": symbol})
        if not data:
            return 0.0
        try:
            return float(data.get("price", 0) or 0)
        except (TypeError, ValueError):
            return 0.0

    async def get_klines(self, symbol: str, interval: str = "1h",
                         limit: int = 100) -> list:
        data = await self._public_get("/fapi/v1/klines", {
            "symbol": symbol, "interval": interval,
            "limit": max(1, min(limit, 1000)),
        })
        return data or []

    async def get_balance(self) -> float:
        data, err = await self._signed("GET", "/fapi/v2/balance", {})
        if data is None:
            logger.warning(f"[binance] balance failed: {err}")
            return 0.0
        try:
            rows = data if isinstance(data, list) else data.get("assets", [])
            total = 0.0
            for a in rows:
                if a.get("asset") in ("USDT", "USDC"):
                    total += float(a.get("crossWalletBalance",
                                         a.get("balance", 0)) or 0)
            return total
        except (TypeError, ValueError, AttributeError):
            return 0.0

    async def get_position(self, symbol: str) -> VenuePosition | None:
        data, err = await self._signed("GET", "/fapi/v2/positionRisk",
                                       {"symbol": symbol})
        if data is None:
            logger.warning(f"[binance] positionRisk failed: {err}")
            return None
        rows = data if isinstance(data, list) else [data]
        for r in rows:
            try:
                amt = float(r.get("positionAmt", 0) or 0)
            except (TypeError, ValueError):
                continue
            if amt == 0:
                continue
            try:
                entry = float(r.get("entryPrice", 0) or 0)
            except (TypeError, ValueError):
                entry = 0.0
            try:
                lev = float(r.get("leverage", 1) or 1)
            except (TypeError, ValueError):
                lev = 1.0
            return VenuePosition(
                venue=Venue.BINANCE, symbol=r.get("symbol", symbol),
                side="LONG" if amt > 0 else "SHORT",
                size=abs(amt), entry_price=entry, leverage=lev,
                notional=abs(amt) * entry, raw=r,
            )
        return None

    async def get_funding_rate(self, symbol: str) -> float | None:
        data = await self._public_get("/fapi/v1/premiumIndex",
                                      {"symbol": symbol})
        if not data:
            return None
        try:
            return float(data["lastFundingRate"])
        except (TypeError, ValueError, KeyError):
            return None

    # -- writes --
    async def market_open(self, symbol: str, side: Side, size: float,
                          slippage_bps: int = 500) -> OrderResult:
        data, err = await self._signed("POST", "/fapi/v1/order", {
            "symbol": symbol, "side": side.value.upper(),
            "type": "MARKET", "quantity": str(size),
        })
        if data is None:
            return _err(symbol, side, err or "order failed")
        try:
            oid = str(data.get("orderId", "") or "")
            px = float(data.get("avgPrice", 0) or data.get("price", 0) or 0)
            sz = float(data.get("executedQty", 0) or size)
        except (TypeError, ValueError):
            oid, px, sz = "", 0.0, size
        return OrderResult(
            status=OrderStatus.OK, venue=Venue.BINANCE, symbol=symbol,
            side=side, oid=oid, fill_price=px, fill_size=sz, raw=data,
        )

    async def limit_open(self, symbol: str, side: Side, size: float,
                         price: float) -> OrderResult:
        data, err = await self._signed("POST", "/fapi/v1/order", {
            "symbol": symbol, "side": side.value.upper(),
            "type": "LIMIT", "timeInForce": "GTC",
            "quantity": str(size), "price": str(price),
        })
        if data is None:
            return _err(symbol, side, err or "order failed")
        try:
            oid = str(data.get("orderId", "") or "")
        except Exception:
            oid = ""
        return OrderResult(
            status=OrderStatus.OK, venue=Venue.BINANCE, symbol=symbol,
            side=side, oid=oid, fill_price=price, fill_size=size, raw=data,
        )

    async def cancel(self, symbol: str, oid: str) -> bool:
        data, err = await self._signed("DELETE", "/fapi/v1/order", {
            "symbol": symbol, "orderId": oid,
        })
        if data is None:
            logger.warning(f"[binance] cancel failed: {err}")
            return False
        return True

    async def set_leverage(self, symbol: str, leverage: int,
                           margin: str = "isolated") -> bool:
        data, err = await self._signed("POST", "/fapi/v1/leverage", {
            "symbol": symbol, "leverage": int(leverage),
        })
        if data is None:
            logger.warning(f"[binance] leverage failed: {err}")
            return False
        mt, err2 = await self._signed("POST", "/fapi/v1/marginType", {
            "symbol": symbol,
            "marginType": ("ISOLATED" if margin.lower() == "isolated"
                           else "CROSSED"),
        })
        if mt is None:
            logger.warning(f"[binance] marginType note: {err2}")
        return True

    def set_deadman_symbols(self, symbols: list[str]) -> None:
        """Symbols armed by enable_deadman(timeout) (countdown is per-symbol)."""
        self._deadman_symbols = list(symbols or [])

    async def enable_deadman(self, timeout_sec: int = 60,
                             symbol: str | None = None) -> bool:
        # countdownCancelAll is per-symbol. Interface calls with timeout only;
        # arm all configured symbols (set via set_deadman_symbols or ctor).
        syms = [symbol] if symbol else list(getattr(self, "_deadman_symbols", []) or [])
        if not syms:
            logger.warning("[binance] enable_deadman needs a symbol")
            return False
        ok_all = True
        for sym in syms:
            data, err = await self._signed("POST",
                                           "/fapi/v1/countdownCancelAll", {
                                               "symbol": sym,
                                               "countdownTime": max(timeout_sec, 1) * 1000,
                                           })
            if data is None:
                logger.warning(f"[binance] countdownCancelAll {sym} failed: {err}")
                ok_all = False
        return ok_all

    async def close(self) -> None:
        if self._session is not None and self._owns_session:
            try:
                await self._session.close()
            except Exception:
                pass
            self._session = None
            self._owns_session = False
