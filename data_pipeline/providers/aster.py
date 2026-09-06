"""Aster market-data provider (P4 Phase 3).

Read-only: klines + ticker/price + fundingRate. Start with a small
whitelist (ASTERUSDT/BTCUSDT/ETHUSDT); symbols are full USDT perp names.
"""
from __future__ import annotations

import aiohttp
from loguru import logger

from .base import DataProvider

FUTURES_MAIN = "https://fapi.asterdex.com"
FUTURES_TEST = "https://fapi.asterdex-testnet.com"
STARTER = ["ASTERUSDT", "BTCUSDT", "ETHUSDT"]


class AsterProvider(DataProvider):
    def __init__(self, base_url: str | None = None):
        if base_url is None:
            from execution.config import AsterConfig

            base_url = AsterConfig.futures_url()
        self.base_url = base_url

    async def _get(self, session: aiohttp.ClientSession, path: str, params: dict):
        url = f"{self.base_url}{path}"
        async with session.get(url, params=params, timeout=10) as resp:
            if resp.status != 200:
                logger.warning(f"[aster provider] {path} -> {resp.status}")
                return None
            return await resp.json()

    async def get_trending_tokens(self, limit: int = 50):
        async with aiohttp.ClientSession() as session:
            out = []
            for sym in STARTER[: max(limit, 0)]:
                px = await self._get(session, "/fapi/v3/ticker/price", {"symbol": sym})
                price = 0.0
                try:
                    price = float((px or {}).get("price", 0) or 0)
                except (TypeError, ValueError):
                    price = 0.0
                out.append({
                    "address": f"aster:perp:{sym}", "symbol": sym,
                    "name": sym.replace("USDT", ""), "decimals": 6,
                    "liquidity": 0.0, "fdv": 0.0, "price": price,
                })
            return out

    async def get_token_history(self, session, address: str, days: int = 7, **kwargs):
        sym = (address or "").split(":")[-1] or "BTCUSDT"
        interval = kwargs.get("interval", "1h")
        limit = min(max(int(days) * 24, 1), 1000)
        data = await self._get(session, "/fapi/v3/klines", {
            "symbol": sym, "interval": interval, "limit": limit,
        })
        if not data:
            return None
        from datetime import datetime

        out = []
        for k in data:
            # Aster klines mirror Binance: [openTime, o, h, l, c, v, ...]
            try:
                dt = datetime.fromtimestamp(int(k[0]) / 1000)
                out.append((
                    dt, address, float(k[1]), float(k[2]), float(k[3]),
                    float(k[4]), float(k[5]), 0.0,
                ))
            except (IndexError, TypeError, ValueError):
                continue
        return out or None
