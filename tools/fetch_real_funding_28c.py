#!/usr/bin/env python3
"""Fetch real Binance perpetual funding rates for the 28c universe.

Why this exists
---------------
The 28c accounting contract originally modelled funding as a CONSTANT
+0.0005 at every 00/08/16 UTC event. That is not a small modelling choice: a
long pays it and a short RECEIVES it, unconditionally, every event, forever.

Measured against actual history for the same window, the constant is wrong in
three separate ways:

  magnitude   median real rate 0.000031  vs  assumed 0.000500  (~16x too high)
  sign        ~29% of real events are NEGATIVE, i.e. shorts PAY
  timing      real rates respond to positioning; a constant cannot

For a net-short book this is not a rounding error. Re-running the v3c lockbox
with real funding moves Sharpe from +0.479 to -0.624. A strategy whose sign
depends on its own cost assumption is not a strategy.

What this tool does
-------------------
Downloads public funding history. Read-only: no account, no API key, no
orders, no position. Binance's fundingRate endpoint is public market data.

Some symbols are listed under a different contract name -- PEPE trades as
1000PEPEUSDT and has no PEPEUSDT history at all. SYMBOL_ALIASES handles that;
without it PEPE silently ends up with no funding and is treated as zero, which
would flatter a short book.
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.universe_28c import COINS_28C  # noqa: E402

ENDPOINT = "https://fapi.binance.com/fapi/v1/fundingRate"
OUT = ROOT / "data" / "funding_binance"

# Binance lists some perpetuals only under a 1000x contract name.
SYMBOL_ALIASES = {"PEPE": "1000PEPEUSDT"}

WINDOW_START_MS = 1726200000000   # 2024-09-13, the 28c common-window start
WINDOW_END_MS = 1790222400000     # 2026-09-24, the dataset end

# Funding events are stamped a few ms after the UTC boundary.
MATCH_TOLERANCE_MS = 60_000


def fetch(symbol: str, start: int, end: int, pause: float = 0.12) -> list[dict]:
    rows, cur = [], start
    while cur < end and len(rows) < 5000:
        url = (f"{ENDPOINT}?symbol={symbol}&startTime={cur}"
               f"&endTime={end}&limit=1000")
        with urllib.request.urlopen(url, timeout=30) as fh:
            batch = json.loads(fh.read())
        if not batch:
            break
        rows += batch
        cur = int(batch[-1]["fundingTime"]) + 1
        time.sleep(pause)
    return rows


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    log = {}
    for coin in COINS_28C:
        symbol = SYMBOL_ALIASES.get(coin, f"{coin}USDT")
        try:
            rows = fetch(symbol, WINDOW_START_MS, WINDOW_END_MS)
            if not rows:
                log[coin] = {"error": f"no history for {symbol}"}
                print(f"  {coin:7} NO HISTORY for {symbol}", flush=True)
                continue
            seen = {int(r["fundingTime"]): float(r["fundingRate"]) for r in rows}
            ts = sorted(seen)
            rates = [seen[t] for t in ts]
            (OUT / f"{coin}.json").write_text(json.dumps(
                {"symbol": symbol, "timestamps": ts, "rates": rates}))
            pos = sum(1 for v in rates if v > 0)
            log[coin] = {"symbol": symbol, "n": len(rates),
                         "first": ts[0], "last": ts[-1],
                         "pos_frac": pos / len(rates),
                         "mean": sum(rates) / len(rates)}
            print(f"  {coin:7} n={len(rates):5}  pos={pos/len(rates):.0%}  "
                  f"mean={sum(rates)/len(rates):+.6f}", flush=True)
        except Exception as exc:                      # network shape varies
            log[coin] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"  {coin:7} FAILED {type(exc).__name__}", flush=True)
    (OUT / "_log.json").write_text(json.dumps(log, indent=1))
    (OUT / "_provenance.json").write_text(json.dumps({
        "source": ENDPOINT,
        "auth": "none (public market data)",
        "window_ms": [WINDOW_START_MS, WINDOW_END_MS],
        "symbol_aliases": SYMBOL_ALIASES,
        "coins_with_data": sorted(k for k, v in log.items() if "error" not in v),
        "coins_missing": sorted(k for k, v in log.items() if "error" in v),
        "note": ("Per-coin fundingRate at 00/08/16 UTC. Read-only; no account, "
                 "no keys, no orders."),
    }, indent=1))
    ok = sum(1 for v in log.values() if "error" not in v)
    print(f"\n{ok}/{len(COINS_28C)} coins written to {OUT}")
    return 0 if ok == len(COINS_28C) else 1


if __name__ == "__main__":
    raise SystemExit(main())
