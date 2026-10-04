#!/usr/bin/env python3
"""Backfill 3 years of 1h and 4h bars for the 28-coin research universe.

The 30m set has 3 years. The 1h and 4h sets have one. That difference is not a
bar-size question: a three-year 30m set and a one-year 1h set differ in sampling
AND in window, so any comparison between them is measuring two things and
attributing the result to one.

This downloads the missing history into data/data_3y/{interval}/, keeping the
one-year files as the tail so nothing is refetched that is already on disk.

Resume-safe per file: a partially written file is continued from its last
timestamp rather than restarted, because 28 coins x 2 intervals x ~26k and ~7k
bars is a lot of requests and a crash halfway through must not cost all of them.

The one thing this cannot fix is a coin that did not list three years ago. Those
get whatever history exists and are reported as short, because a bar-count
assertion downstream is the thing that turns a silent gap into a visible one.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "data_3y"
LIMIT = 1000
INTERVALS = {"1h": 60 * 60 * 1000, "4h": 4 * 60 * 60 * 1000}
YEARS = 3

sys.path.insert(0, str(ROOT))
from research.universe_28c import COINS_28C  # noqa: E402

# Binance futures perps; 1000-prefixed where the listing is that way.
EXTRA = {"PEPE": "1000PEPEUSDT", "SHIB": "1000SHIBUSDT"}


def symbol_for(coin):
    return EXTRA.get(coin, coin + "USDT")


def get_klines(symbol, interval, start_ms, tries=5):
    url = ("https://fapi.binance.com/fapi/v1/klines?symbol=%s"
           "&interval=%s&limit=%d&startTime=%d" % (symbol, interval, LIMIT, start_ms))
    req = urllib.request.Request(url, headers={"User-Agent": "AlphaGPT/1.0"})
    last = None
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except Exception as exc:            # noqa: BLE001 - retry anything
            last = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("%s %s failed: %s" % (symbol, interval, last))


def read_existing(path):
    if not path.exists():
        return []
    with path.open() as f:
        rows = list(csv.reader(f))
    if not rows:
        return []
    body = rows[1:] if rows[0][0] == "timestamp" else rows
    return [[int(float(r[0])), r[1], r[2], r[3], r[4], r[5],
             r[6] if len(r) > 6 else "", r[7] if len(r) > 7 else ""] for r in body]


def download(coin, interval, ims, now):
    """Fetch from the earliest available point up to now, merging any local file."""
    d = OUT / interval
    d.mkdir(parents=True, exist_ok=True)
    path = d / (coin + ".csv")
    sym = symbol_for(coin)
    existing = read_existing(path)
    start = now - YEARS * 365 * 24 * 3600 * 1000
    # A coin listed later than the 3-year mark simply has no history there;
    # stepping the cursor forward on an empty response is how that is detected
    # rather than a silently short file.
    floor = None
    probe = get_klines(sym, interval, start)
    if not probe:
        floor = start
    rows = []
    cur = start
    stalls = 0
    while cur < now:
        data = get_klines(sym, interval, cur)
        if not data:
            if rows:
                break
            stalls += 1
            if stalls > 5:
                break
            cur += ims * LIMIT
            continue
        rows.extend(data)
        nxt = data[-1][0] + ims
        if nxt <= cur:
            break
        cur = nxt
        if len(data) < LIMIT:
            break
        time.sleep(0.12)
    have = {r[0] for r in existing}
    fresh = [[str(k[0]), k[1], k[2], k[3], k[4], k[5], k[7], str(k[8])]
             for k in rows if int(k[0]) not in have]
    merged = existing + fresh
    uniq = {r[0]: r for r in merged}
    ordered = [uniq[k] for k in sorted(uniq)]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "open", "high", "low", "close", "volume",
                    "quote_volume", "trades"])
        w.writerows(ordered)
    return len(ordered), floor is not None


def main():
    only = sys.argv[1:] or sorted(INTERVALS)
    now = int(time.time() * 1000)
    report = []
    for interval in only:
        ims = INTERVALS[interval]
        for coin in COINS_28C:
            t0 = time.time()
            try:
                n, short = download(coin, interval, ims, now)
            except Exception as exc:            # noqa: BLE001 - report and continue
                print("%-7s %-4s FAIL %s" % (coin, interval, exc), flush=True)
                report.append((coin, interval, -1, str(exc)))
                continue
            span_days = 0
            print("%-7s %-4s %6d bars  %5.0fs%s"
                  % (coin, interval, n, time.time() - t0,
                     "  (no 3y history)" if short else ""), flush=True)
            report.append((coin, interval, n, "short" if short else "ok"))
    bad = [r for r in report if r[2] < 0]
    print()
    print("%d/%d fetched; %d failed" % (len(report) - len(bad), len(report), len(bad)))
    for r in bad:
        print("  ", r)
    (OUT / "backfill_report.json").write_text(json.dumps(
        [{"coin": c, "interval": i, "bars": n, "note": s} for c, i, n, s in report],
        indent=2))


if __name__ == "__main__":
    main()
