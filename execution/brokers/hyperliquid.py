"""Hyperliquid venue broker (P4 Phase 2): wraps official hyperliquid-python-sdk.

Reads: Info (all_mids, user_state, candles). Writes: Exchange
(market_open, order, cancel, update_leverage, scheduleCancel).
Queries use the MASTER account address; signing may use an agent key.
"""
from __future__ import annotations

import asyncio

from loguru import logger

from ..config import HyperliquidConfig
from .base import OrderResult, OrderStatus, Side, Venue, VenueBroker, VenuePosition


def _err(symbol: str, side: Side, reason: str, raw=None) -> OrderResult:
    return OrderResult(
        status=OrderStatus.ERROR, venue=Venue.HYPERLIQUID,
        symbol=symbol, side=side, reason=reason, raw=raw,
    )


class HyperliquidBroker(VenueBroker):
    def __init__(self, info=None, exchange=None, base_url: str | None = None,
                 account_address: str | None = None):
        self._base_url = base_url or HyperliquidConfig.base_url()
        self._account = account_address or HyperliquidConfig.account_address()
        self._info = info
        self._exchange = exchange

    @property
    def venue(self) -> Venue:
        return Venue.HYPERLIQUID

    def _require_info(self):
        if self._info is None:
            from hyperliquid.info import Info

            self._info = Info(self._base_url, skip_ws=True)
        return self._info

    def _require_exchange(self):
        if self._exchange is None:
            import eth_account
            from hyperliquid.exchange import Exchange

            secret = HyperliquidConfig.secret_key()
            if not secret:
                raise ValueError("HYPERLIQUID_SECRET_KEY is not set")
            acct = eth_account.Account.from_key(secret)
            account = self._account or HyperliquidConfig.account_address()
            self._exchange = Exchange(acct, self._base_url, account_address=account or None)
        return self._exchange

    def _run(self, fn, *args, **kwargs):
        # SDK is sync; run in thread so the runner loop stays async.
        async def _call():
            return await asyncio.to_thread(fn, *args, **kwargs)

        return _call()

    # -- reads --
    async def get_price(self, symbol: str) -> float:
        info = self._require_info()
        mids = await self._run(info.all_mids)
        try:
            return float(mids[symbol])
        except (KeyError, TypeError, ValueError):
            logger.warning(f"[hl] no mid for {symbol}")
            return 0.0

    async def get_klines(self, symbol: str, interval: str = "1h", limit: int = 100) -> list:
        info = self._require_info()
        end = None
        try:
            import time

            end = int(time.time() * 1000)
        except Exception:
            pass
        candles = await self._run(
            info.candles_snapshot, symbol, interval, 0, end or 0,
        )
        return candles or []

    async def get_balance(self) -> float:
        info = self._require_info()
        state = await self._run(info.user_state, self._account)
        try:
            return float(state["marginSummary"]["accountValue"])
        except (KeyError, TypeError, ValueError):
            return 0.0

    async def get_position(self, symbol: str) -> VenuePosition | None:
        info = self._require_info()
        state = await self._run(info.user_state, self._account)
        for ap in (state or {}).get("assetPositions", []):
            pos = ap.get("position", {})
            if pos.get("coin") == symbol:
                try:
                    szi = float(pos.get("szi", 0))
                except (TypeError, ValueError):
                    szi = 0.0
                if szi == 0:
                    return None
                try:
                    entry = float(pos.get("entryPx", 0) or 0)
                except (TypeError, ValueError):
                    entry = 0.0
                lev = pos.get("leverage", {}) or {}
                try:
                    lev_v = float(lev.get("value", 1) or 1)
                except (TypeError, ValueError):
                    lev_v = 1.0
                return VenuePosition(
                    venue=Venue.HYPERLIQUID, symbol=symbol,
                    side="LONG" if szi > 0 else "SHORT",
                    size=abs(szi), entry_price=entry, leverage=lev_v,
                    notional=abs(szi) * entry, raw=pos,
                )
        return None

    # -- writes --
    @staticmethod
    def _first_status(resp) -> dict:
        try:
            return (resp["response"]["data"]["statuses"] or [{}])[0]
        except (KeyError, TypeError, IndexError):
            return {}

    async def market_open(self, symbol: str, side: Side, size: float,
                          slippage_bps: int = 500) -> OrderResult:
        try:
            ex = self._require_exchange()
        except ValueError as e:
            return _err(symbol, side, str(e))
        try:
            slippage = max(slippage_bps, 0) / 10_000
            resp = await self._run(ex.market_open, symbol, side == Side.BUY, size,
                                   None, slippage)
        except Exception as e:
            logger.error(f"[hl] market_open failed: {e}")
            return _err(symbol, side, f"market_open exception: {e}")
        if not isinstance(resp, dict) or resp.get("status") != "ok":
            return _err(symbol, side, f"exchange rejected: {resp}", raw=resp)
        st = self._first_status(resp)
        if "error" in st:
            return OrderResult(
                status=OrderStatus.REJECTED, venue=Venue.HYPERLIQUID,
                symbol=symbol, side=side, reason=str(st["error"]), raw=resp,
            )
        filled = st.get("filled", {}) if isinstance(st, dict) else {}
        try:
            px = float(filled.get("avgPx", 0) or 0)
        except (TypeError, ValueError):
            px = 0.0
        try:
            sz = float(filled.get("totalSz", 0) or size)
        except (TypeError, ValueError):
            sz = size
        oid = ""
        try:
            oid = str(st.get("resting", {}).get("oid", "") or "")
        except Exception:
            oid = ""
        return OrderResult(
            status=OrderStatus.OK, venue=Venue.HYPERLIQUID, symbol=symbol,
            side=side, oid=oid, fill_price=px, fill_size=sz, raw=resp,
        )

    async def limit_open(self, symbol: str, side: Side, size: float,
                         price: float) -> OrderResult:
        try:
            ex = self._require_exchange()
        except ValueError as e:
            return _err(symbol, side, str(e))
        try:
            resp = await self._run(ex.order, symbol, side == Side.BUY, size, price,
                                   {"limit": {"tif": "Gtc"}})
        except Exception as e:
            logger.error(f"[hl] limit order failed: {e}")
            return _err(symbol, side, f"order exception: {e}")
        if not isinstance(resp, dict) or resp.get("status") != "ok":
            return _err(symbol, side, f"exchange rejected: {resp}", raw=resp)
        st = self._first_status(resp)
        if "error" in st:
            return OrderResult(
                status=OrderStatus.REJECTED, venue=Venue.HYPERLIQUID,
                symbol=symbol, side=side, reason=str(st["error"]), raw=resp,
            )
        oid = ""
        try:
            oid = str(st.get("resting", {}).get("oid", "") or "")
        except Exception:
            oid = ""
        return OrderResult(
            status=OrderStatus.OK, venue=Venue.HYPERLIQUID, symbol=symbol,
            side=side, oid=oid, fill_price=price, fill_size=size, raw=resp,
        )

    async def cancel(self, symbol: str, oid: str) -> bool:
        try:
            ex = self._require_exchange()
            resp = await self._run(ex.cancel, symbol, int(oid))
            return isinstance(resp, dict) and resp.get("status") == "ok"
        except Exception as e:
            logger.error(f"[hl] cancel failed: {e}")
            return False

    async def set_leverage(self, symbol: str, leverage: int, margin: str = "isolated") -> bool:
        try:
            ex = self._require_exchange()
            info = self._require_info()
            asset = await self._run(info.name_to_asset, symbol)
            is_cross = margin.lower() == "cross"
            resp = await self._run(ex.update_leverage, asset, is_cross, int(leverage))
            return isinstance(resp, dict) and resp.get("status") == "ok"
        except Exception as e:
            logger.error(f"[hl] set_leverage failed: {e}")
            return False

    async def enable_deadman(self, timeout_sec: int = 60) -> bool:
        try:
            ex = self._require_exchange()
            import time

            cancel_time = int(time.time() * 1000) + max(timeout_sec, 1) * 1000
            resp = await self._run(ex.schedule_cancel, cancel_time)
            return isinstance(resp, dict) and resp.get("status") == "ok"
        except Exception as e:
            logger.error(f"[hl] deadman failed: {e}")
            return False

    async def close(self) -> None:
        return None
