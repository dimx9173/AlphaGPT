"""Aster venue broker (P4 Phase 3): V3 EIP-712 signed REST via aiohttp.

Reads: GET /fapi/v3/ticker/price, /fapi/v3/klines, /fapi/v3/accountWithJoinMargin,
       /fapi/v3/positionRisk. Writes: POST /fapi/v3/order, /fapi/v3/leverage,
       /fapi/v3/marginType; cancel via DELETE /fapi/v3/order; deadman via
       countdownTime. Session injectable for unit tests.
"""
from __future__ import annotations

import urllib.parse

import aiohttp
from loguru import logger

from ..config import AsterConfig
from .base import OrderResult, OrderStatus, Side, Venue, VenueBroker, VenuePosition

HEADERS = {"Content-Type": "application/x-www-form-urlencoded",
           "User-Agent": "AlphaGPT/1.0"}

# -5050: main wallet has not completed a deposit (auth writes blocked).
NEED_DEPOSIT = -5050


def _err(symbol: str, side: Side, reason: str, raw=None) -> OrderResult:
    return OrderResult(
        status=OrderStatus.ERROR, venue=Venue.ASTER,
        symbol=symbol, side=side, reason=reason, raw=raw,
    )


class AsterBroker(VenueBroker):
    def __init__(self, signer=None, session: aiohttp.ClientSession | None = None,
                 futures_url: str | None = None):
        self._signer = signer
        self._session = session
        self._futures = futures_url or AsterConfig.futures_url()
        self._owns_session = False

    @property
    def venue(self) -> Venue:
        return Venue.ASTER

    def _require_signer(self):
        if self._signer is None:
            self._signer = AsterConfig.make_signer()
        return self._signer

    async def _session_or_create(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession()
            self._owns_session = True
        return self._session

    async def _public_get(self, path: str, params: dict | None = None):
        sess = await self._session_or_create()
        url = f"{self._futures}{path}"
        async with sess.get(url, params=params or {}, timeout=10) as resp:
            if resp.status != 200:
                logger.warning(f"[aster] GET {path} -> {resp.status}")
                return None
            return await resp.json()

    async def _signed(self, method: str, path: str, fields: dict | None = None):
        try:
            signer = self._require_signer()
        except ValueError as e:
            return None, str(e)
        sess = await self._session_or_create()
        signed = signer.sign_fields(fields)
        url = f"{self._futures}{path}"
        body = urllib.parse.urlencode(signed)
        try:
            async with sess.request(method, url, data=body, headers=HEADERS,
                                    timeout=10) as resp:
                try:
                    data = await resp.json()
                except Exception:
                    data = {"_text": await resp.text()}
                if isinstance(data, dict) and data.get("code") == NEED_DEPOSIT:
                    return None, ("main wallet has not completed a deposit "
                                  "(code -5050); fund via bridge first")
                if resp.status != 200:
                    return None, f"http {resp.status}: {data}"
                if isinstance(data, dict) and data.get("code", 0) not in (0, None):
                    # Aster error envelope
                    if data.get("code") not in (None,):
                        return None, f"code {data.get('code')}: {data.get('msg')}"
                return data, ""
        except Exception as e:
            logger.error(f"[aster] {method} {path} failed: {e}")
            return None, str(e)

    # -- reads --
    async def get_price(self, symbol: str) -> float:
        data = await self._public_get("/fapi/v3/ticker/price", {"symbol": symbol})
        if not data:
            return 0.0
        try:
            return float(data.get("price", 0) or 0)
        except (TypeError, ValueError):
            return 0.0

    async def get_klines(self, symbol: str, interval: str = "1h", limit: int = 100) -> list:
        data = await self._public_get("/fapi/v3/klines", {
            "symbol": symbol, "interval": interval, "limit": min(limit, 1000),
        })
        return data or []

    async def get_balance(self) -> float:
        data, err = await self._signed("GET", "/fapi/v3/accountWithJoinMargin", {})
        if data is None:
            logger.warning(f"[aster] balance failed: {err}")
            return 0.0
        try:
            # envelope: {assets: [{asset, walletBalance...}], ...} variants
            for key in ("totalWalletBalance", "totalMarginBalance"):
                if key in data:
                    return float(data[key])
            assets = data.get("assets", [])
            total = 0.0
            for a in assets:
                if a.get("asset") in ("USDT", "USDC"):
                    total += float(a.get("walletBalance", 0) or 0)
            return total
        except (TypeError, ValueError, AttributeError):
            return 0.0

    async def get_position(self, symbol: str) -> VenuePosition | None:
        data, err = await self._signed("GET", "/fapi/v3/positionRisk", {"symbol": symbol})
        if data is None:
            logger.warning(f"[aster] positionRisk failed: {err}")
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
                venue=Venue.ASTER, symbol=r.get("symbol", symbol),
                side="LONG" if amt > 0 else "SHORT",
                size=abs(amt), entry_price=entry, leverage=lev,
                notional=abs(amt) * entry, raw=r,
            )
        return None

    # -- writes --
    async def market_open(self, symbol: str, side: Side, size: float,
                          slippage_bps: int = 500) -> OrderResult:
        data, err = await self._signed("POST", "/fapi/v3/order", {
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
            status=OrderStatus.OK, venue=Venue.ASTER, symbol=symbol,
            side=side, oid=oid, fill_price=px, fill_size=sz, raw=data,
        )

    async def limit_open(self, symbol: str, side: Side, size: float,
                         price: float) -> OrderResult:
        data, err = await self._signed("POST", "/fapi/v3/order", {
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
            status=OrderStatus.OK, venue=Venue.ASTER, symbol=symbol,
            side=side, oid=oid, fill_price=price, fill_size=size, raw=data,
        )

    async def cancel(self, symbol: str, oid: str) -> bool:
        # DELETE with signed query: use signer fields in URL.
        try:
            signer = self._require_signer()
        except ValueError as e:
            logger.warning(f"[aster] cancel no signer: {e}")
            return False
        sess = await self._session_or_create()
        url = signer.build_url(self._futures, "/fapi/v3/order",
                               {"symbol": symbol, "orderId": oid})
        try:
            async with sess.delete(url, headers=HEADERS, timeout=10) as resp:
                data = await resp.json()
                return resp.status == 200 and int(data.get("code", 200)) in (200, 0)
        except Exception as e:
            logger.error(f"[aster] cancel failed: {e}")
            return False

    async def set_leverage(self, symbol: str, leverage: int, margin: str = "isolated") -> bool:
        data, err = await self._signed("POST", "/fapi/v3/leverage", {
            "symbol": symbol, "leverage": str(int(leverage)),
        })
        if data is None:
            logger.warning(f"[aster] leverage failed: {err}")
            return False
        mt, err2 = await self._signed("POST", "/fapi/v3/marginType", {
            "symbol": symbol,
            "marginType": "ISOLATED" if margin.lower() == "isolated" else "CROSSED",
        })
        if mt is None:
            logger.warning(f"[aster] marginType note: {err2}")
        return True

    async def enable_deadman(self, timeout_sec: int = 60) -> bool:
        # countdownTime auto-cancels all open orders after timeout.
        data, err = await self._signed("POST", "/fapi/v3/cancelAllOpenOrders", {
            "symbol": "", "countdownTime": str(max(timeout_sec, 1) * 1000),
        })
        if data is None:
            # fall back: plain cancel-all without countdown still protects
            data2, _ = await self._signed("DELETE", "/fapi/v3/allOpenOrders", {})
            return data2 is not None
        return True

    async def close(self) -> None:
        if self._session is not None and self._owns_session:
            try:
                await self._session.close()
            except Exception:
                pass
            self._session = None
            self._owns_session = False
