"""Hyperliquid market-data provider (P4 Phase 2).

Read-only: allMids for prices, candles_snapshot for OHLCV history.
Start with a small whitelist (HYPE/BTC/ETH); testnet uses same coin names.
"""
from __future__ import annotations

import time

from loguru import logger

from .base import DataProvider

STARTER_COINS = ["HYPE", "BTC", "ETH"]


class HyperliquidProvider(DataProvider):
    def __init__(self, base_url: str | None = None):
        if base_url is None:
            from execution.config import HyperliquidConfig

            base_url = HyperliquidConfig.base_url()
        self.base_url = base_url
        self._info = None

    def _require_info(self):
        if self._info is None:
            from hyperliquid.info import Info

            self._info = Info(self.base_url, skip_ws=True)
        return self._info

    async def get_trending_tokens(self, limit: int = 50):
        import asyncio

        info = self._require_info()
        try:
            mids = await asyncio.to_thread(info.all_mids)
        except Exception as e:
            logger.error(f"[hl provider] all_mids failed: {e}")
            return []
        out = []
        for coin in STARTER_COINS[: max(limit, 0)]:
            if coin in (mids or {}):
                try:
                    price = float(mids[coin])
                except (TypeError, ValueError):
                    price = 0.0
                out.append({
                    "address": f"hl:perp:{coin}", "symbol": coin, "name": coin,
                    "decimals": 6, "liquidity": 0.0, "fdv": 0.0, "price": price,
                })
        # fill remaining slots with other perps by name order
        if len(out) < limit:
            for coin in sorted((mids or {}).keys()):
                if coin in STARTER_COINS or len(out) >= limit:
                    continue
                out.append({
                    "address": f"hl:perp:{coin}", "symbol": coin, "name": coin,
                    "decimals": 6, "liquidity": 0.0, "fdv": 0.0,
                    "price": float(mids[coin] or 0) if mids.get(coin) else 0.0,
                })
        return out

    async def get_token_history(self, session, address: str, days: int = 7, **kwargs):
        import asyncio
        from datetime import datetime

        coin = (address or "").split(":")[-1] or "BTC"
        interval = kwargs.get("interval", "1h")
        info = self._require_info()
        try:
            end_ms = int(time.time() * 1000)
            start_ms = end_ms - max(int(days), 1) * 24 * 3600 * 1000
            candles = await asyncio.to_thread(
                info.candles_snapshot, coin, interval, start_ms, end_ms,
            )
        except Exception as e:
            logger.error(f"[hl provider] candles failed for {coin}: {e}")
            return None
        if not candles:
            return None
        out = []
        for c in candles:
            try:
                dt = datetime.fromtimestamp(int(c["T"]) / 1000)
                out.append((
                    dt, address, float(c["o"]), float(c["h"]), float(c["l"]),
                    float(c["c"]), float(c["v"]), 0.0,
                ))
            except (KeyError, TypeError, ValueError):
                continue
        return out or None
