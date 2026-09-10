"""Bybit V5 venue broker: linear USDT perp via aiohttp.

Reads: GET /v5/market/tickers, /v5/market/kline, /v5/market/funding/history,
       signed GET /v5/account/wallet-balance, /v5/position/list.
Writes: signed POST /v5/order/create, /v5/order/cancel, /v5/order/cancel-all,
        /v5/position/set-leverage. Session injectable for unit tests.

Auth (V5): HMAC-SHA256(api_secret, timestamp + apiKey + recvWindow +
queryString-or-body) in headers X-BAPI-API-KEY / X-BAPI-TIMESTAMP /
X-BAPI-RECV-WINDOW / X-BAPI-SIGN. No keys needed at construction; missing
keys make writes fail closed (OrderResult ERROR / False), reads fail soft.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import time
import urllib.parse

import aiohttp
from loguru import logger

from .base import OrderResult, OrderStatus, Side, Venue, VenueBroker, VenuePosition
from .cex_config import BybitConfig

RECV_WINDOW = "5000"
_CATEGORY = "linear"

# Default symbols armed by no-arg enable_deadman (demo cancel-all needs symbol).
_DEFAULT_DEADMAN_SYMBOLS = ("ETCUSDT", "TRXUSDT")

# Known qty steps (from instruments-info); fallback snaps conservatively.
_QTY_STEP = {"ETCUSDT": 0.1, "TRXUSDT": 1.0}


async def _qty_step(broker, symbol: str) -> float:
    known = _QTY_STEP.get(symbol)
    if known:
        return known
    try:
        data = await broker._public_get("/v5/market/instruments-info", {
            "category": _CATEGORY, "symbol": symbol})
        row = ((data or {}).get("result") or {}).get("list", [{}])[0]
        step = float((row.get("lotSizeFilter") or {}).get("qtyStep", 1) or 1)
        if step > 0:
            _QTY_STEP[symbol] = step
            return step
    except Exception:
        pass
    return 1.0


def _snap_qty(size: float, step: float) -> float:
    import math
    from decimal import Decimal, ROUND_DOWN
    if step <= 0:
        return max(size, 0.0)
    try:
        q = (Decimal(str(size)) // Decimal(str(step))) * Decimal(str(step))
        return float(q)
    except Exception:
        return math.floor(size / step) * step

# Bybit kline intervals; map common aliases, pass Bybit-native values through.
_INTERVAL_MAP = {
    "1m": "1", "3m": "3", "5m": "5", "15m": "15", "30m": "30",
    "1h": "60", "2h": "120", "4h": "240", "6h": "360", "12h": "720",
    "1d": "D", "1w": "W", "1M": "M",
}


def _sign(secret: str, timestamp: str, api_key: str, recv_window: str,
          payload: str) -> str:
    msg = (timestamp + api_key + recv_window + payload).encode()
    return hmac.new(secret.encode(), msg, hashlib.sha256).hexdigest()


def _err(symbol: str, side: Side, reason: str, raw=None) -> OrderResult:
    return OrderResult(
        status=OrderStatus.ERROR, venue=Venue.BYBIT,
        symbol=symbol, side=side, reason=reason, raw=raw,
    )


class BybitBroker(VenueBroker):
    def __init__(self, api_key: str = "", api_secret: str = "",
                 session: aiohttp.ClientSession | None = None,
                 base_url: str | None = None):
        # Keys optional here (tests / paper paths); resolved lazily per call.
        # Demo-only: ignore any custom base_url, always testnet.
        self._api_key = api_key
        self._api_secret = api_secret
        self._session = session
        self._base = BybitConfig.base_url()
        self._owns_session = False

    @property
    def venue(self) -> Venue:
        return Venue.BYBIT

    def _keys(self) -> tuple[str, str]:
        return (self._api_key or BybitConfig.api_key(),
                self._api_secret or BybitConfig.api_secret())

    async def _session_or_create(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession()
            self._owns_session = True
        return self._session

    def _auth_headers(self, query_or_body: str) -> dict | None:
        key, secret = self._keys()
        if not key or not secret:
            return None
        ts = str(int(time.time() * 1000))
        return {
            "X-BAPI-API-KEY": key,
            "X-BAPI-TIMESTAMP": ts,
            "X-BAPI-RECV-WINDOW": RECV_WINDOW,
            "X-BAPI-SIGN": _sign(secret, ts, key, RECV_WINDOW, query_or_body),
            "Content-Type": "application/json",
        }

    async def _public_get(self, path: str, params: dict | None = None):
        sess = await self._session_or_create()
        url = f"{self._base}{path}"
        try:
            async with sess.get(url, params=params or {}, timeout=10) as resp:
                if resp.status != 200:
                    logger.warning(f"[bybit] GET {path} -> {resp.status}")
                    return None
                return await resp.json()
        except Exception as e:
            logger.error(f"[bybit] GET {path} failed: {e}")
            return None

    async def _signed(self, method: str, path: str,
                      params: dict | None = None) -> tuple[dict | None, str]:
        """Signed V5 call. GET signs the sorted query string; POST signs the
        compact JSON body. Returns (result, error)."""
        if method == "GET":
            qs = urllib.parse.urlencode(sorted((params or {}).items()))
            headers = self._auth_headers(qs)
            if headers is None:
                return None, "missing BYBIT_API_KEY/BYBIT_API_SECRET"
            sess = await self._session_or_create()
            url = f"{self._base}{path}"
            try:
                async with sess.get(url, params=params or {}, headers=headers,
                                    timeout=10) as resp:
                    try:
                        data = await resp.json()
                    except Exception:
                        return None, f"http {resp.status}: non-JSON response"
                    if resp.status != 200 or data.get("retCode") != 0:
                        return None, (f"http {resp.status}: "
                                      f"{data.get('retCode')}: {data.get('retMsg')}")
                    return data.get("result") or {}, ""
            except Exception as e:
                logger.error(f"[bybit] GET {path} failed: {e}")
                return None, str(e)
        # POST: JSON body
        body = json.dumps(params or {}, separators=(",", ":"))
        headers = self._auth_headers(body)
        if headers is None:
            return None, "missing BYBIT_API_KEY/BYBIT_API_SECRET"
        sess = await self._session_or_create()
        url = f"{self._base}{path}"
        try:
            async with sess.post(url, data=body, headers=headers,
                                 timeout=10) as resp:
                try:
                    data = await resp.json()
                except Exception:
                    return None, f"http {resp.status}: non-JSON response"
                code = data.get("retCode")
                if resp.status != 200 or code != 0:
                    return None, (f"http {resp.status}: "
                                  f"{code}: {data.get('retMsg')}")
                return data.get("result") or {}, ""
        except Exception as e:
            logger.error(f"[bybit] POST {path} failed: {e}")
            return None, str(e)

    # -- reads (fail soft) --
    async def get_price(self, symbol: str) -> float:
        data = await self._public_get("/v5/market/tickers", {
            "category": _CATEGORY, "symbol": symbol,
        })
        if not data:
            return 0.0
        try:
            rows = (data.get("result") or {}).get("list") or []
            return float(rows[0].get("lastPrice", 0) or 0) if rows else 0.0
        except (TypeError, ValueError, IndexError, AttributeError):
            return 0.0

    async def get_klines(self, symbol: str, interval: str = "1h",
                         limit: int = 100) -> list:
        data = await self._public_get("/v5/market/kline", {
            "category": _CATEGORY, "symbol": symbol,
            "interval": _INTERVAL_MAP.get(interval, interval),
            "limit": min(max(limit, 1), 1000),
        })
        if not data:
            return []
        try:
            return (data.get("result") or {}).get("list") or []
        except AttributeError:
            return []

    async def get_balance(self) -> float:
        result, err = await self._signed("GET", "/v5/account/wallet-balance", {
            "accountType": "UNIFIED", "coin": "USDT",
        })
        if result is None:
            logger.warning(f"[bybit] balance failed: {err}")
            return 0.0
        try:
            wallets = result.get("list") or []
            if wallets and wallets[0].get("totalWalletBalance") not in (None, ""):
                return float(wallets[0]["totalWalletBalance"])
            total = 0.0
            for w in wallets:
                for c in w.get("coin") or []:
                    if c.get("coin") == "USDT":
                        total += float(c.get("walletBalance", 0) or 0)
            return total
        except (TypeError, ValueError, AttributeError):
            return 0.0

    async def get_position(self, symbol: str) -> VenuePosition | None:
        result, err = await self._signed("GET", "/v5/position/list", {
            "category": _CATEGORY, "symbol": symbol,
        })
        if result is None:
            logger.warning(f"[bybit] position list failed: {err}")
            return None
        for r in result.get("list") or []:
            try:
                size = float(r.get("size", 0) or 0)
            except (TypeError, ValueError):
                continue
            if size == 0:
                continue
            bybit_side = r.get("side", "")
            side = ("LONG" if bybit_side == "Buy"
                    else "SHORT" if bybit_side == "Sell" else bybit_side or "LONG")
            try:
                entry = float(r.get("avgPrice", 0) or 0)
            except (TypeError, ValueError):
                entry = 0.0
            try:
                lev = float(r.get("leverage", 1) or 1)
            except (TypeError, ValueError):
                lev = 1.0
            return VenuePosition(
                venue=Venue.BYBIT, symbol=r.get("symbol", symbol),
                side=side, size=size, entry_price=entry, leverage=lev,
                notional=size * entry, raw=r,
            )
        return None

    async def get_funding_rate(self, symbol: str) -> float | None:
        """Latest linear funding rate; None when unavailable (fail soft)."""
        data = await self._public_get("/v5/market/tickers", {
            "category": _CATEGORY, "symbol": symbol,
        })
        try:
            rows = (data.get("result") or {}).get("list") or []
            if rows and rows[0].get("fundingRate") not in (None, ""):
                return float(rows[0]["fundingRate"])
        except (TypeError, ValueError, AttributeError):
            pass
        hist = await self._public_get("/v5/market/funding/history", {
            "category": _CATEGORY, "symbol": symbol, "limit": 1,
        })
        try:
            rows = (hist.get("result") or {}).get("list") or []
            if rows and rows[0].get("fundingRate") not in (None, ""):
                return float(rows[0]["fundingRate"])
        except (TypeError, ValueError, AttributeError):
            pass
        return None

    # -- writes (fail closed) --
    def _side(self, side: Side) -> str:
        return "Buy" if side == Side.BUY else "Sell"

    async def market_open(self, symbol: str, side: Side, size: float,
                          slippage_bps: int = 500) -> OrderResult:
        qty = _snap_qty(size, await _qty_step(self, symbol))
        if qty <= 0:
            return _err(symbol, side, f"qty {size} below step minimum")
        result, err = await self._signed("POST", "/v5/order/create", {
            "category": _CATEGORY, "symbol": symbol,
            "side": self._side(side), "orderType": "Market",
            "qty": str(qty), "timeInForce": "GTC",
        })
        if result is None:
            return _err(symbol, side, err or "order failed")
        oid = str(result.get("orderId", "") or "")
        price = await self.get_price(symbol)
        return OrderResult(
            status=OrderStatus.OK, venue=Venue.BYBIT, symbol=symbol,
            side=side, oid=oid, fill_price=price, fill_size=qty, raw=result,
        )

    async def limit_open(self, symbol: str, side: Side, size: float,
                         price: float) -> OrderResult:
        qty = _snap_qty(size, await _qty_step(self, symbol))
        if qty <= 0:
            return _err(symbol, side, f"qty {size} below step minimum")
        result, err = await self._signed("POST", "/v5/order/create", {
            "category": _CATEGORY, "symbol": symbol,
            "side": self._side(side), "orderType": "Limit",
            "qty": str(qty), "price": str(price), "timeInForce": "GTC",
        })
        if result is None:
            return _err(symbol, side, err or "order failed")
        oid = str(result.get("orderId", "") or "")
        return OrderResult(
            status=OrderStatus.OK, venue=Venue.BYBIT, symbol=symbol,
            side=side, oid=oid, fill_price=price, fill_size=size, raw=result,
        )

    async def cancel(self, symbol: str, oid: str) -> bool:
        result, err = await self._signed("POST", "/v5/order/cancel", {
            "category": _CATEGORY, "symbol": symbol, "orderId": oid,
        })
        if result is None:
            logger.warning(f"[bybit] cancel failed: {err}")
            return False
        return True

    async def set_leverage(self, symbol: str, leverage: int,
                           margin: str = "isolated") -> bool:
        lev = str(int(leverage))
        result, err = await self._signed("POST", "/v5/position/set-leverage", {
            "category": _CATEGORY, "symbol": symbol, "leverage": lev,
            "buyLeverage": lev, "sellLeverage": lev,
        })
        if result is None:
            # 110043 = leverage not modified: already at target, treat as ok.
            if "110043" in err:
                return True
            logger.warning(f"[bybit] set-leverage failed: {err}")
            return False
        return True

    async def enable_deadman(self, timeout_sec: int = 60,
                             symbol: str = "") -> bool:
        """Cancel-all open linear orders (immediate deadman; Bybit V5 has no
        countdown timer). True on ok, False on any failure.
        Demo quirk: cancel-all without symbol returns params error, so the
        no-symbol call arms the default Y1b symbols one by one."""
        syms = [symbol] if symbol else list(_DEFAULT_DEADMAN_SYMBOLS)
        ok_all = True
        for _s in syms:
            body: dict = {"category": _CATEGORY, "symbol": _s}
            result, err = await self._signed("POST", "/v5/order/cancel-all", body)
            if result is None:
                logger.warning(f"[bybit] cancel-all {_s} failed: {err}")
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
