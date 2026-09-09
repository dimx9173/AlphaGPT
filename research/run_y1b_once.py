#!/usr/bin/env python3
"""run_y1b_once.py — Y1b single cycle CLI (E10 lock).

Default dry-run (no orders, no keys). Live ONLY with --live AND
Y1B_LIVE_ENABLED=1 AND PAPER_MODE unset AND STOP absent AND circuit
closed AND deadman ok AND perp gate pass.
Usage:
  python3 research/run_y1b_once.py [--notional=50] [--live]
"""
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tests.conftest  # noqa: F401,E402  (solders stub for RiskEngine import chain)
from execution.brokers.aster import AsterBroker  # noqa: E402
from strategy_manager.risk import RiskEngine  # noqa: E402
from strategy_manager.y1b_executor import run_once  # noqa: E402


def main(argv):
    live = "--live" in argv
    notional = None
    for a in argv:
        if a.startswith("--notional="):
            try:
                notional = float(a.split("=", 1)[1])
            except ValueError:
                pass
    broker = AsterBroker()
    try:
        plans, res, sync = asyncio.run(run_once(
            broker=broker, risk=RiskEngine(), notional=notional,
            dry_run=False if live else None))
    finally:
        try:
            asyncio.run(broker.close())
        except Exception:
            pass
    print(json.dumps({"plans": [
        {"coin": p.coin, "symbol": p.symbol, "want": p.want,
         "size": round(p.size, 6), "price": p.price, "gate_ok": p.gate_ok,
         "reason": p.reason} for p in plans],
        "results": res, "sync": sync}, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1:])
