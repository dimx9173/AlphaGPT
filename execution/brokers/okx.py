"""OKX SWAP venue broker: USDT-margined perp REST via aiohttp.

Reads: GET /api/v5/market/ticker, /api/v5/market/candles,
       GET /api/v5/account/balance, /api/v5/account/positions (signed).
Writes: POST /api/v5/trade/order, /api/v5/trade/cancel-order,
        POST /api/v5/account/set-leverage (signed).
Deadman: POST /api/v5/trade/cancel-algos. Funding: public funding-rate.
Session injectable for unit tests. No network at import.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import urllib.parse
from datetime import datetime, timezone

import aiohttp
from loguru import logger

from .base import OrderResult, OrderStatus, Side, Venue, VenueBroker, VenuePosition
from .cex_config import OkxConfig

_QUOTES = ("USDT", "USDC", "USD")


def _to_inst(symbol: str) -> str:
    """Map e.g. ETCUSDT -> ETC-USDT-SWAP. Pass through OKX instIds."""
    s = (symbol or "").strip().upper()
    if not s:
        return s
    if "-" in s:
        return s
    for q in _QUOTES:
        if s.endswith(q) and len(s) > len(q):
            return f"{s[: -len(q)]}-{q}-SWAP"
    return f"{s}-USDT-SWAP"


def _to_bar(interval: str) -> str:
    iv = (interval or "1h").strip().lower()
    mapping = {"1h": "1H", "2h": "2H", "4h": "4H", "6h": "6H", "12h": "12H",
               "1d": "1D", "1w": "1W", "1m": "1m"}
    return mapping.get(iv, interval)


def _err(symbol: str, side: Side, reason: str, raw=None) -> OrderResult:
    return OrderResult(
        status=OrderStatus.ERROR, venue=Venue.OKX,
        symbol=symbol, side=side, reason=reason, raw=raw,
    )


class OkxBroker(VenueBroker):
    def __init__(self, api_key: str | None = None,
                 api_secret: str | None = None,
                 passphrase: str | None = None,
                 base_url: str | None = None,
                 use_demo: bool | None = None,
                 session: aiohttp.ClientSession | None = None):
        self._api_key = api_key
        self._api_secret = api_secret
        self._passphrase = passphrase
        self._base_url = base_url
        self._use_demo = use_demo
        self._session = session
        self._owns_session = False

    @property
    def venue(self) -> Venue:
        return Venue.OKX

    # -- config / auth --
    def _creds(self) -> tuple[str, str, str]:
        key = self._api_key if self._api_key is not None else OkxConfig.api_key()
        secret = self._api_secret if self._api_secret is not None else OkxConfig.api_secret()
        pp = self._passphrase if self._passphrase is not None else OkxConfig.passphrase()
        return key, secret, pp

    def _has_creds(self) -> bool:
        key, secret, pp = self._creds()
        return bool(key and secret and pp)

    def _base(self) -> str:
        return self._base_url or OkxConfig.base_url()

    def _demo(self) -> bool:
        return self._use_demo if self._use_demo is not None else OkxConfig.use_demo()

    def _auth_headers(self, method: str, request_path: str, body: str = "") -> dict:
        key, secret, pp = self._creds()
        ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        msg = ts + method.upper() + request_path + body
        mac = hmac.new(secret.encode(), msg.encode(), hashlib.sha256).digest()
        headers = {
            "Content-Type": "application/json",
            "OK-ACCESS-KEY": key,
            "OK-ACCESS-SIGN": base64.b64encode(mac).decode(),
            "OK-ACCESS-TIMESTAMP": ts,
            "OK-ACCESS-PASSPHRASE": pp,
        }
        if self._demo():
            headers["x-simulated-trading"] = "1"
        return headers

    async def _session_or_create(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession()
            self._owns_session = True
        return self._session

    async def _public_get(self, path: str, params: dict | None = None):
        sess = await self._session_or_create()
        url = f"{self._base()}{path}"
        try:
            async with sess.get(url, params=params or {}, timeout=10) as resp:
                if resp.status != 200:
                    logger.warning(f"[okx] GET {path} -> {resp.status}")
                    return None
                return await resp.json()
        except Exception as e:
            logger.warning(f"[okx] GET {path} failed: {e}")
            return None

    async def _signed(self, method: str, path: str,
                      params: dict | None = None, body: dict | None = None):
        if not self._has_creds():
            return None, "missing OKX API credentials (OKX_API_KEY/OKX_API_SECRET/OKX_PASSPHRASE)"
        sess = await self._session_or_create()
        query = ("?" + urllib.parse.urlencode(params)) if params else ""
        request_path = path + query
        body_str = json.dumps(body) if body is not None else ""
        headers = self._auth_headers(method, request_path, body_str)
        url = f"{self._base()}{request_path}"
        try:
            async with sess.request(method.upper(), url,
                                    data=body_str if body is not None else None,
                                    headers=headers, timeout=10) as resp:
                try:
                    data = await resp.json()
                except Exception:
                    data = {"_text": await resp.text()}
                if resp.status != 200:
                    return None, f"http {resp.status}: {data}"
                if not isinstance(data, dict) or data.get("code") != "0":
                    code = data.get("code") if isinstance(data, dict) else None
                    msg = data.get("msg") if isinstance(data, dict) else data
                    return None, f"code {code}: {msg}"
                return data, ""
        except Exception as e:
            logger.error(f"[okx] {method} {path} failed: {e}")
            return None, str(e)

    @staticmethod
    def _side_fields(side: Side) -> tuple[str, str]:
        if side == Side.BUY:
            return "buy", "long"
        return "sell", "short"

    # -- reads (fail-soft) --
    async def get_price(self, symbol: str) -> float:
        data = await self._public_get("/api/v5/market/ticker", {"instId": _to_inst(symbol)})
        if not data:
            return 0.0
        try:
            rows = data.get("data", [])
            return float(rows[0].get("lastPx", 0) or 0) if rows else 0.0
        except (TypeError, ValueError, AttributeError, IndexError):
            return 0.0

    async def get_klines(self, symbol: str, interval: str = "1h", limit: int = 100) -> list:
        data = await self._public_get("/api/v5/market/candles", {
            "instId": _to_inst(symbol), "bar": _to_bar(interval),
            "limit": str(min(limit, 300)),
        })
        if not data:
            return []
        rows = data.get("data", [])
        return rows if isinstance(rows, list) else []

    async def get_balance(self) -> float:
        data, err = await self._signed("GET", "/api/v5/account/balance", {"ccy": "USDT"})
        if data is None:
            logger.warning(f"[okx] balance failed: {err}")
            return 0.0
        try:
            rows = data.get("data", [])
            if not rows:
                return 0.0
            for d in rows[0].get("details", []):
                if d.get("ccy") == "USDT":
                    return float(d.get("availBal", 0) or 0)
            return float(rows[0].get("totalEq", 0) or 0)
        except (TypeError, ValueError, AttributeError):
            return 0.0

    async def get_position(self, symbol: str) -> VenuePosition | None:
        inst = _to_inst(symbol)
        data, err = await self._signed("GET", "/api/v5/account/positions",
                                       {"instType": "SWAP", "instId": inst})
        if data is None:
            logger.warning(f"[okx] positions failed: {err}")
            return None
        rows = data.get("data", []) if isinstance(data, dict) else []
        for r in rows:
            try:
                sz = float(r.get("pos", 0) or 0)
            except (TypeError, ValueError):
                continue
            if sz == 0:
                continue
            try:
                entry = float(r.get("avgPx", 0) or 0)
            except (TypeError, ValueError):
                entry = 0.0
            try:
                lev = float(r.get("lever", 1) or 1)
            except (TypeError, ValueError):
                lev = 1.0
            ps = (r.get("posSide") or "").lower()
            side = "LONG" if ps == "long" or (not ps and sz > 0) else "SHORT"
            return VenuePosition(
                venue=Venue.OKX, symbol=r.get("instId", inst),
                side=side, size=abs(sz), entry_price=entry, leverage=lev,
                notional=abs(sz) * entry, raw=r,
            )
        return None

    async def get_funding_rate(self, symbol: str) -> float | None:
        data = await self._public_get("/api/v5/public/funding-rate",
                                      {"instId": _to_inst(symbol)})
        if not data:
            return None
        try:
            rows = data.get("data", [])
            return float(rows[0].get("fundingRate")) if rows else None
        except (TypeError, ValueError, AttributeError, IndexError):
            return None

    # -- writes --
    async def _place(self, symbol: str, side: Side, size: float,
                     ord_type: str, price: float | None = None) -> OrderResult:
        if not self._has_creds():
            return _err(symbol, side, "missing OKX API credentials "
                        "(OKX_API_KEY/OKX_API_SECRET/OKX_PASSPHRASE)")
        okx_side, pos_side = self._side_fields(side)
        body: dict = {"instId": _to_inst(symbol), "tdMode": "cross",
                      "side": okx_side, "posSide": pos_side,
                      "ordType": ord_type, "sz": str(size)}
        if price is not None:
            body["px"] = str(price)
        data, err = await self._signed("POST", "/api/v5/trade/order", body=body)
        if data is None:
            return _err(symbol, side, err or "order failed")
        try:
            row = (data.get("data", []) or [{}])[0]
            oid = str(row.get("ordId", "") or "")
        except (AttributeError, IndexError):
            oid = ""
        fill_px = price or 0.0
        return OrderResult(
            status=OrderStatus.OK, venue=Venue.OKX, symbol=symbol,
            side=side, oid=oid, fill_price=fill_px, fill_size=size, raw=data,
        )

    async def market_open(self, symbol: str, side: Side, size: float,
                          slippage_bps: int = 500) -> OrderResult:
        return await self._place(symbol, side, size, "market")

    async def limit_open(self, symbol: str, side: Side, size: float,
                         price: float) -> OrderResult:
        return await self._place(symbol, side, size, "limit", price)

    async def cancel(self, symbol: str, oid: str) -> bool:
        if not self._has_creds():
            logger.warning("[okx] cancel without credentials")
            return False
        data, err = await self._signed("POST", "/api/v5/trade/cancel-order",
                                       body={"instId": _to_inst(symbol), "ordId": oid})
        if data is None:
            logger.warning(f"[okx] cancel failed: {err}")
            return False
        return True

    async def set_leverage(self, symbol: str, leverage: int, margin: str = "isolated") -> bool:
        if not self._has_creds():
            logger.warning("[okx] set_leverage without credentials")
            return False
        inst = _to_inst(symbol)
        lev = str(int(leverage))
        ok = True
        for ps in ("long", "short"):
            data, err = await self._signed(
                "POST", "/api/v5/account/set-leverage",
                body={"instId": inst, "lever": lev,
                      "mgnMode": "cross", "posSide": ps})
            if data is None:
                logger.warning(f"[okx] set_leverage {ps} failed: {err}")
                ok = False
        return ok

    async def enable_deadman(self, timeout_sec: int = 60) -> bool:
        if not self._has_creds():
            logger.warning("[okx] enable_deadman without credentials")
            return False
        data, err = await self._signed("POST", "/api/v5/trade/cancel-algos",
                                       body={"instType": "SWAP"})
        if data is None:
            logger.warning(f"[okx] enable_deadman failed: {err}")
            return False
        return True

    async def close(self) -> None:
        if self._session is not None and self._owns_session:
            try:
                await self._session.close()
            except Exception:
                pass
            self._session = None
            self._owns_session = False
