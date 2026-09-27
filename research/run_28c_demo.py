#!/usr/bin/env python3
"""Safe 28-coin Bybit-demo preflight. Read-only by construction.

The arm path is intentionally hard-closed. This program contains no order sink
and never calls market_open, limit_open, set_leverage, cancel, or deadman APIs.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from execution.brokers.bybit import BybitBroker
from research.causal_12f import causal_signal
from research.universe_28c import COINS_28C, OFFLINE_VENUE_SYMBOLS

FORMULA = [5, 5, 6, 26, 20, 17, 22, 13, 16, 14, 21, 21]
INTERVAL = "30m"
BAR_MS = 30 * 60 * 1000
KLINE_LIMIT = 300
LONG_THRESHOLD = 0.85
SHORT_THRESHOLD = 0.15
MAX_LEVERAGE = 2
MAX_NOTIONAL_PER_SYMBOL = float(os.getenv("DEMO_MAX_NOTIONAL_PER_SYMBOL", "10"))
MAX_TOTAL_NOTIONAL = float(os.getenv("DEMO_MAX_TOTAL_NOTIONAL", "50"))


def _row_to_dict(row) -> dict:
    if isinstance(row, dict):
        return {
            "timestamp": int(row.get("startTime", row.get("timestamp", 0))),
            "open": float(row["open"]), "high": float(row["high"]),
            "low": float(row["low"]), "close": float(row["close"]),
            "volume": float(row["volume"]),
        }
    if isinstance(row, (list, tuple)) and len(row) >= 6:
        return {
            "timestamp": int(row[0]), "open": float(row[1]),
            "high": float(row[2]), "low": float(row[3]),
            "close": float(row[4]), "volume": float(row[5]),
        }
    raise ValueError(f"unsupported Bybit kline row: {type(row).__name__}")


def parse_klines(rows: list) -> tuple[dict, int]:
    normalized = [_row_to_dict(x) for x in rows]
    normalized.sort(key=lambda x: x["timestamp"])
    # Drop the still-open candle. Never signal on an unfinished 30m bar.
    cutoff = int(time.time() * 1000) - BAR_MS
    normalized = [x for x in normalized if x["timestamp"] <= cutoff]
    if len(normalized) < 200:
        raise ValueError("insufficient closed klines: require >=200")
    timestamps = [x["timestamp"] for x in normalized]
    if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
        raise ValueError("kline timestamps must be strictly increasing")
    for x in normalized:
        values = (x["open"], x["high"], x["low"], x["close"], x["volume"])
        if not all(math.isfinite(float(v)) for v in values):
            raise ValueError("non-finite kline value")
        if min(x["open"], x["high"], x["low"], x["close"]) <= 0 or x["volume"] < 0:
            raise ValueError("invalid OHLCV value")
    if cutoff - timestamps[-1] > 2 * BAR_MS:
        raise ValueError("latest closed kline is stale")
    return {
        "timestamp": [x["timestamp"] for x in normalized],
        "open": [x["open"] for x in normalized],
        "high": [x["high"] for x in normalized],
        "low": [x["low"] for x in normalized],
        "close": [x["close"] for x in normalized],
        "volume": [x["volume"] for x in normalized],
    }, len(rows) - len(normalized)


def signal_for(data: dict) -> tuple[float, float, int, int]:
    raw_signal = float(causal_signal(FORMULA, data)[-1])
    prob = 1.0 / (1.0 + math.exp(max(-60.0, min(60.0, raw_signal))))
    want = 1 if prob > LONG_THRESHOLD else (-1 if prob < SHORT_THRESHOLD else 0)
    return raw_signal, prob, want, int(data["timestamp"][-1])


def stop_present() -> bool:
    raw = (os.getenv("STOP_SIGNAL_PATH", "STOP_SIGNAL") or "").strip()
    return bool(raw and (ROOT / raw).exists())


async def collect_preflight() -> dict:
    broker = BybitBroker()
    snapshot_started = time.time()
    out = {
        "status": "preflight", "mode": "read_only", "armed": False,
        "asof_utc": datetime.now(timezone.utc).isoformat(),
        "snapshot_started_epoch": snapshot_started,
        "venue": "bybit-demo", "base_url": "https://api-demo.bybit.com",
        "live_adopted": False, "orders_attempted": 0, "trading_state_mutated": False,
        "state_mutated": False, "formula": FORMULA, "coins": COINS_28C,
        "symbol_map": OFFLINE_VENUE_SYMBOLS, "interval": INTERVAL,
        "kline_limit": KLINE_LIMIT,
        "thresholds": {"long": LONG_THRESHOLD, "short": SHORT_THRESHOLD},
        "limits": {"max_leverage": MAX_LEVERAGE, "max_notional_per_symbol": MAX_NOTIONAL_PER_SYMBOL, "max_total_notional": MAX_TOTAL_NOTIONAL},
        "markets": {}, "positions": {}, "open_orders": [], "instrument_errors": {},
        "errors": [], "blockers": [], "account_readable": False,
        "position_read_ok": False, "open_order_read_ok": False,
    }
    if stop_present():
        out["blockers"].append("STOP signal is present")
    try:
        # One public instrument snapshot; no fallback symbols.
        info = await broker._public_get("/v5/market/instruments-info", {"category": "linear", "limit": 1000})
        rows = ((info or {}).get("result") or {}).get("list") or []
        instruments = {x.get("symbol"): x for x in rows if x.get("quoteCoin") == "USDT" and x.get("status") == "Trading"}
        out["instrument_count"] = len(instruments)
        for coin in COINS_28C:
            symbol = OFFLINE_VENUE_SYMBOLS[coin]
            inst = instruments.get(symbol)
            if not inst:
                out["instrument_errors"][coin] = f"missing trading linear USDT instrument: {symbol}"
                continue
            lot = inst.get("lotSizeFilter") or {}
            out["markets"][coin] = {
                "symbol": symbol, "instrument_ok": True,
                "qty_step": float(lot.get("qtyStep", 0) or 0),
                "min_order_qty": float(lot.get("minOrderQty", 0) or 0),
                "max_order_qty": float(lot.get("maxOrderQty", 0) or 0),
                "min_notional": float(lot.get("minNotionalValue", 0) or 0),
                "max_leverage": float((inst.get("leverageFilter") or {}).get("maxLeverage", 0) or 0),
            }
        # Market data and closed-bar signals.
        for coin in COINS_28C:
            symbol = OFFLINE_VENUE_SYMBOLS[coin]
            item = out["markets"].setdefault(coin, {"symbol": symbol, "instrument_ok": False})
            try:
                price = await broker.get_price(symbol)
                item["price"] = price
                rows = await broker.get_klines(symbol, INTERVAL, KLINE_LIMIT)
                data, dropped = parse_klines(rows)
                raw, prob, want, signal_ts = signal_for(data)
                item.update({"market_ok": price > 0, "bars": len(data["close"]), "dropped_open_bars": dropped,
                             "raw_signal": raw, "probability": prob, "want": want,
                             "signal_bar_ts": signal_ts})
            except Exception as exc:
                item.update({"market_ok": False, "error": f"{type(exc).__name__}: {exc}"})
        # Read-only plan sizing. No order sink exists in this program.
        active_coins = [c for c, v in out["markets"].items()
                        if v.get("market_ok") and v.get("want", 0) != 0]
        equal_notional = (MAX_TOTAL_NOTIONAL / len(active_coins)) if active_coins else 0.0
        plans = {}
        total_planned = 0.0
        infeasible = []
        for coin in COINS_28C:
            market = out["markets"].get(coin, {})
            want = int(market.get("want", 0) or 0)
            price = float(market.get("price", 0) or 0)
            step = float(market.get("qty_step", 0) or 0)
            min_notional = float(market.get("min_notional", 0) or 0)
            max_lev = float(market.get("max_leverage", 0) or 0)
            target = min(MAX_NOTIONAL_PER_SYMBOL, equal_notional) if want else 0.0
            raw_qty = target / price if price > 0 else 0.0
            qty = math.floor(raw_qty / step) * step if step > 0 and price > 0 else 0.0
            actual = qty * price
            reasons = []
            if want and max_lev < MAX_LEVERAGE: reasons.append("instrument leverage below cap")
            if want and step <= 0: reasons.append("invalid qty step")
            if want and actual + 1e-9 < min_notional: reasons.append("below exchange min notional")
            if want and not math.isfinite(target): reasons.append("non-finite target")
            plans[coin] = {
                "symbol": market.get("symbol"), "want": want,
                "side": "BUY" if want > 0 else ("SELL" if want < 0 else "FLAT"),
                "reference_price": price, "target_notional": target,
                "raw_qty": raw_qty, "qty_after_step": qty,
                "actual_notional": actual, "min_notional": min_notional,
                "qty_step": step, "max_leverage": max_lev,
                "ok": not reasons, "reasons": reasons,
            }
            if want and reasons: infeasible.append(coin)
            total_planned += actual
        out["plans"] = plans
        out["plan_summary"] = {
            "active_count": len(active_coins), "equal_notional_per_active": equal_notional,
            "total_planned_notional": total_planned, "max_total_notional": MAX_TOTAL_NOTIONAL,
            "infeasible": infeasible,
        }
        if infeasible: out["blockers"].append(f"plan infeasible: {','.join(infeasible)}")
        if total_planned > MAX_TOTAL_NOTIONAL + 1e-9: out["blockers"].append("aggregate notional cap exceeded")
        # Signed account reads: tri-state positions and open orders.
        wallet, wallet_err = await broker._signed("GET", "/v5/account/wallet-balance", {"accountType": "UNIFIED", "coin": "USDT"})
        if wallet is None:
            out["errors"].append(f"wallet read failed: {wallet_err}")
        else:
            wl = (wallet.get("list") or [{}])[0]
            out["account"] = {k: wl.get(k) for k in ("totalWalletBalance", "totalEquity", "totalAvailableBalance", "totalInitialMargin", "totalPerpUPL")}
            out["account_readable"] = True
        pos_rows, pos_err = await broker._signed("GET", "/v5/position/list", {"category": "linear", "settleCoin": "USDT"})
        if pos_rows is None:
            out["position_read_ok"] = False
            out["errors"].append(f"position read failed: {pos_err}")
            for coin in COINS_28C:
                out["positions"][coin] = {"state": "UNKNOWN", "position": None, "error": pos_err}
        else:
            out["position_read_ok"] = True
            by_symbol = {x.get("symbol"): x for x in (pos_rows.get("list") or []) if float(x.get("size", 0) or 0) != 0}
            for coin in COINS_28C:
                symbol = OFFLINE_VENUE_SYMBOLS[coin]
                row = by_symbol.get(symbol)
                if not row:
                    out["positions"][coin] = {"state": "FLAT", "position": None}
                else:
                    out["positions"][coin] = {"state": "OPEN", "position": {
                        "symbol": symbol, "side": row.get("side"), "size": float(row.get("size", 0) or 0),
                        "entry": float(row.get("avgPrice", 0) or 0), "leverage": float(row.get("leverage", 0) or 0),
                        "unrealised_pnl": float(row.get("unrealisedPnl", 0) or 0),
                    }}
        # _signed signs sorted query text; keep insertion order sorted too.
        orders, order_err = await broker._signed("GET", "/v5/order/realtime", {"category": "linear", "limit": "100", "settleCoin": "USDT"})
        if orders is None:
            out["open_order_read_ok"] = False
            out["errors"].append(f"open order read failed: {order_err}")
        else:
            out["open_order_read_ok"] = True
            out["open_orders"] = orders.get("list") or []
    finally:
        await broker.close()
    unknown_positions = [c for c, v in out["positions"].items() if v.get("state") == "UNKNOWN"]
    active_positions = [c for c, v in out["positions"].items() if v.get("state") == "OPEN"]
    unavailable = [c for c in COINS_28C if not out["markets"].get(c, {}).get("instrument_ok") or not out["markets"].get(c, {}).get("market_ok")]
    if unavailable: out["blockers"].append(f"market unavailable: {','.join(unavailable)}")
    if not out["account_readable"]: out["blockers"].append("demo account credentials unavailable")
    if unknown_positions: out["blockers"].append("position snapshot UNKNOWN")
    if active_positions: out["blockers"].append(f"existing positions: {','.join(active_positions)}")
    if not out["open_order_read_ok"]: out["blockers"].append("open-order snapshot UNKNOWN")
    if out["open_orders"]: out["blockers"].append("open orders present")
    invalid_caps = not all(math.isfinite(x) and x > 0 for x in (MAX_LEVERAGE, MAX_NOTIONAL_PER_SYMBOL, MAX_TOTAL_NOTIONAL))
    if invalid_caps: out["blockers"].append("invalid risk caps")
    out["active_position_count"] = len(active_positions)
    out["clean_account_for_arm"] = not active_positions and not unknown_positions and not out["open_orders"]
    out["preflight_pass"] = not out["blockers"]
    out["preflight_blockers"] = list(out["blockers"])
    out["snapshot_age_sec"] = time.time() - snapshot_started
    return out


def arm_gate() -> None:
    """Always closed until a separately reviewed order sink is implemented."""
    raise RuntimeError("arm gate hard-closed: no reviewed order sink is installed")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("preflight", "arm-demo"), default="preflight")
    ap.add_argument("--arm-demo", action="store_true")
    ap.add_argument("--i-understand-demo", action="store_true")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    result = asyncio.run(collect_preflight())
    if args.mode == "arm-demo":
        result["arm_attempted"] = True
        try:
            arm_gate()
        except Exception as exc:
            result["arm_result"] = f"blocked: {exc}"
    else:
        result["arm_attempted"] = False
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.out.with_suffix(args.out.suffix + ".tmp")
        with tmp.open("w") as f: json.dump(result, f, indent=2)
        os.replace(tmp, args.out)
    print(json.dumps(result, indent=2))
    if args.mode == "arm-demo":
        return 2
    return 0 if result.get("preflight_pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
