#!/usr/bin/env python3
"""Validate the 1h and 4h backfill before any search runs on it.

The 30m set was validated when it was built. The 1h and 4h sets are new and a
search over a series with a hole in it produces a number that looks like a
result. This checks, per interval:

  - every one of the 28 coins present
  - bar spacing exactly equal to the interval, with the gap count reported
  - the common 28-coin timestamp intersection and what bounds it
  - whether the intersection is a sane multiple of the interval, i.e. aligned
  - OHLC sanity: high >= low, no non-finite, no non-positive prices

A gap is reported rather than rejected. A coin that listed late is a fact about
the universe, not a defect in the download, and the intersection is the honest
consequence. What must be zero is the number of intervals in the intersection
that is not a multiple of the step, because that means the grids disagree.
"""
from __future__ import annotations

import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.universe_28c import COINS_28C  # noqa: E402

STEPS = {"30m": 30 * 60 * 1000, "1h": 60 * 60 * 1000, "4h": 4 * 60 * 60 * 1000}
DIRS = {"30m": "data_3y/30m", "1h": "data_3y/1h", "4h": "data_3y/4h"}


def load(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def check(interval):
    step = STEPS[interval]
    d = ROOT / "data" / DIRS[interval]
    print("=" * 72)
    print(interval, "->", d)
    print("=" * 72)
    if not d.is_dir():
        print("  MISSING DIRECTORY")
        return None
    missing = [c for c in COINS_28C if not (d / (c + ".csv")).exists()]
    present = [c for c in COINS_28C if (d / (c + ".csv")).exists()]
    print("coins: %d/28%s" % (len(present), "" if not missing else "  missing " + ",".join(missing)))
    if not present:
        return None
    common = None
    bad_price = 0
    total_gaps = 0
    for c in present:
        rows = load(d / (c + ".csv"))
        ts = [int(float(r["timestamp"])) for r in rows]
        s = set(ts)
        common = s if common is None else (common & s)
        gaps = sum(1 for i in range(len(ts) - 1) if ts[i + 1] - ts[i] != step)
        total_gaps += gaps
        for r in rows:
            try:
                o, h, l, cl = (float(r["open"]), float(r["high"]),
                               float(r["low"]), float(r["close"]))
                v = float(r["volume"])
            except Exception:          # noqa: BLE001
                bad_price += 1
                continue
            if not (h >= l and o > 0 and cl > 0 and l > 0 and v >= 0):
                bad_price += 1
    print("total off-grid gaps across coins: %d" % total_gaps)
    print("rows with bad or unparseable OHLCV: %d" % bad_price)
    if common is None:
        print("no common timestamps")
        return None
    common = sorted(common)
    a = datetime.fromtimestamp(common[0] / 1000, tz=timezone.utc)
    b = datetime.fromtimestamp(common[-1] / 1000, tz=timezone.utc)
    misaligned = sum(1 for t in common if t % step != 0)
    print("common intersection: %d bars  %s -> %s" % (len(common), a.date(), b.date()))
    print("  years: %.2f" % (len(common) * step / (365 * 24 * 3600 * 1000)))
    print("  timestamps not on the %s grid: %d" % (interval, misaligned))
    print("  shortest coin bounds it: %s" % min(
        (c for c in present), key=lambda c: len(load(d / (c + ".csv")))))
    return len(common)


if __name__ == "__main__":
    out = {}
    for iv in (sys.argv[1:] or sorted(STEPS)):
        n = check(iv)
        if n:
            out[iv] = n
        print()
    print("summary:", out)
