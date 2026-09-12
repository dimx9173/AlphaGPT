#!/usr/bin/env python3
"""run_y1b_hourly.py — hourly Y1b cycle on Bybit demo (cron entry).

Live arms AFTER imports (execution.config re-applies .env at import).
Appends JSONL rows to results/y1b_hourly.jsonl (gitignored via results/).
Exit 0 always unless catastrophic (cron noise control); errors logged in row.
Usage: Y1B_NOTIONAL_USDT=10 /path/python research/run_y1b_hourly.py
"""
import asyncio
import datetime
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv(".env")

import tests.conftest  # noqa: F401,E402  (solders stub)
from strategy_manager.y1b_executor import (  # noqa: E402
    make_broker, venue_name, run_once, stop_requested)
from strategy_manager.risk import RiskEngine  # noqa: E402

LOG = "results/y1b_hourly.jsonl"


def _log(row: dict):
    os.makedirs("results", exist_ok=True)
    with open(LOG, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")


async def _cycle(notional: float):
    os.environ["Y1B_LIVE_ENABLED"] = "1"
    os.environ["PAPER_MODE"] = ""
    os.environ["Y1B_VENUE"] = "bybit"
    os.environ["Y1B_TOP5"] = "1"
    broker = make_broker("bybit")
    out = {"venue": venue_name(broker)}
    try:
        plans, res, sync = await run_once(
            broker=broker, risk=RiskEngine(), notional=notional, dry_run=False)
        out["plans"] = [{"coin": p.coin, "symbol": p.symbol, "want": p.want,
                         "size": round(p.size, 6), "price": p.price,
                         "gate_ok": p.gate_ok, "reason": p.reason} for p in plans]
        out["results"] = res
        out["sync"] = sync
        poss = {}
        for p in plans:
            try:
                v = await broker.get_position(p.symbol)
                poss[p.symbol] = ({"side": v.side, "size": v.size,
                                   "entry": v.entry_price, "lev": v.leverage,
                                   "notional": round(v.notional, 2)}
                                  if v else None)
            except Exception as e:
                poss[p.symbol] = {"error": str(e)[:200]}
        out["positions"] = poss
        try:
            out["balance"] = await broker.get_balance()
        except Exception as e:
            out["balance_err"] = str(e)[:200]
    finally:
        try:
            await broker.close()
        except Exception:
            pass
    return out


def main():
    t0 = datetime.datetime.now(datetime.timezone.utc)
    try:
        notional = float(os.getenv("Y1B_NOTIONAL_USDT", "10"))
    except (TypeError, ValueError):
        notional = 10.0
    row = {"ts": t0.isoformat(), "notional": notional}
    if stop_requested():
        row.update({"skipped": True, "reason": "stop_signal"})
        _log(row)
        print(json.dumps(row, indent=1))
        return 0
    try:
        out = asyncio.run(_cycle(notional))
        row.update(out)
        row["ok"] = True
    except Exception as e:
        row.update({"ok": False, "error": str(e)[:500],
                    "trace": traceback.format_exc()[-1500:]})
    _log(row)
    print(json.dumps(row, indent=1)[:3000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
