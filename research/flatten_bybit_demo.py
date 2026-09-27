#!/usr/bin/env python3
"""Explicit, reduce-only flatten for the Bybit demo account.

Default is dry-run. Writes require all three gates:
  --flatten-demo, --i-understand-demo, ALPHA_FLATTEN_ARM=1
No order is retried automatically. A final position snapshot must be flat.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from execution.brokers.bybit import BybitBroker, _qty_step, _snap_qty
from execution.brokers.cex_config import BybitConfig


def positions_from_rows(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        try:
            size = float(row.get("size", 0) or 0)
        except (TypeError, ValueError):
            continue
        if size <= 0:
            continue
        out.append({
            "symbol": row.get("symbol", ""), "side": row.get("side", ""),
            "size": size, "position_idx": int(row.get("positionIdx", 0) or 0),
            "entry": float(row.get("avgPrice", 0) or 0),
            "unrealised_pnl": float(row.get("unrealisedPnl", 0) or 0),
        })
    return out


async def snapshot(broker):
    rows, err = await broker._signed("GET", "/v5/position/list", {"category": "linear", "settleCoin": "USDT"})
    if rows is None:
        raise RuntimeError(f"position read failed: {err}")
    if not isinstance(rows.get("list"), list):
        raise RuntimeError("malformed position/list response")
    positions = positions_from_rows(rows["list"])
    symbols = [x["symbol"] for x in positions]
    if len(symbols) != len(set(symbols)):
        raise RuntimeError("duplicate hedge/position rows; manual reconciliation required")
    if any(x["position_idx"] != 0 for x in positions):
        raise RuntimeError("non-zero positionIdx/hedge mode; manual reconciliation required")
    orders, oerr = await broker._signed("GET", "/v5/order/realtime", {"category": "linear", "limit": "100", "settleCoin": "USDT"})
    if orders is None or not isinstance(orders.get("list"), list):
        raise RuntimeError(f"open-order read failed: {oerr}")
    if orders.get("nextPageCursor"):
        raise RuntimeError("open-order pagination required; refusing flatten")
    return positions, orders["list"]


async def collect():
    broker = BybitBroker()
    out = {
        "status": "flatten_dry_run", "venue": "bybit-demo",
        "base_url": BybitConfig.base_url(), "asof_utc": datetime.now(timezone.utc).isoformat(),
        "orders_attempted": 0, "state_mutated": False,
    }
    try:
        if out["base_url"] != "https://api-demo.bybit.com":
            raise RuntimeError(f"refusing non-demo endpoint: {out['base_url']}")
        positions, open_orders = await snapshot(broker)
        out["positions"] = positions
        out["open_orders"] = open_orders
        out["symbol_count"] = len({x["symbol"] for x in positions})
        out["would_flatten"] = [
            {"symbol": x["symbol"], "side": "Sell" if x["side"] == "Buy" else "Buy", "qty": x["size"]}
            for x in positions
        ]
        out["ready_for_flatten"] = True
    except Exception as exc:
        out["ready_for_flatten"] = False
        out["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        await broker.close()
    return out


async def flatten(preflight: dict):
    if not preflight.get("ready_for_flatten"):
        raise RuntimeError(preflight.get("error", "preflight failed"))
    if os.getenv("ALPHA_FLATTEN_ARM") != "1":
        raise RuntimeError("ALPHA_FLATTEN_ARM=1 required")
    broker = BybitBroker()
    result = {"status": "flattening", "orders_attempted": 0, "orders": [], "errors": []}
    try:
        if BybitConfig.base_url() != "https://api-demo.bybit.com":
            raise RuntimeError("refusing non-demo endpoint")
        positions, open_orders = await snapshot(broker)
        # Cancel only orders observed in the demo account, then re-snapshot.
        for symbol in sorted({x.get("symbol") for x in open_orders if x.get("symbol")}):
            res, err = await broker._signed("POST", "/v5/order/cancel-all", {"category": "linear", "symbol": symbol})
            if res is None:
                result["errors"].append(f"cancel-all {symbol}: {err}")
        if result["errors"]:
            raise RuntimeError("open-order cancellation failed; no positions closed")
        positions, open_orders = await snapshot(broker)
        if open_orders:
            raise RuntimeError("open orders remain after cancellation; no positions closed")
        for pos in positions:
            close_side = "Sell" if pos["side"] == "Buy" else "Buy"
            step = await _qty_step(broker, pos["symbol"])
            qty = _snap_qty(pos["size"], step)
            if qty <= 0 or not math.isfinite(qty):
                raise RuntimeError(f"invalid close qty for {pos['symbol']}")
            body = {
                "category": "linear", "symbol": pos["symbol"], "side": close_side,
                "orderType": "Market", "qty": str(qty), "timeInForce": "GTC",
                "reduceOnly": True, "positionIdx": pos["position_idx"],
            }
            res, err = await broker._signed("POST", "/v5/order/create", body)
            result["orders_attempted"] += 1
            result["orders"].append({"symbol": pos["symbol"], "side": close_side, "qty": qty, "ok": res is not None, "error": err if res is None else ""})
            if res is None:
                result["errors"].append(f"close {pos['symbol']}: {err}")
        final_positions, final_orders = await snapshot(broker)
        result["final_positions"] = final_positions
        result["final_open_orders"] = final_orders
        result["status"] = "flatten_complete" if not final_positions and not final_orders and not result["errors"] else "flatten_incomplete"
    except Exception as exc:
        result["status"] = "flatten_blocked"
        result["errors"].append(f"{type(exc).__name__}: {exc}")
    finally:
        await broker.close()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flatten-demo", action="store_true")
    ap.add_argument("--i-understand-demo", action="store_true")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    pre = asyncio.run(collect())
    if args.flatten_demo and args.i_understand_demo:
        pre["flatten"] = asyncio.run(flatten(pre))
    else:
        pre["flatten"] = {"status": "not_requested"}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as f: json.dump(pre, f, indent=2)
    print(json.dumps(pre, indent=2))
    if pre.get("flatten", {}).get("status") == "flatten_complete":
        return 0
    return 2 if args.flatten_demo else 0

if __name__ == "__main__":
    raise SystemExit(main())
