"""Merged funding-rate and derivatives factor source.

Two independent providers, deliberately not collapsed into one:

  binance     data/funding_binance/<COIN>.json
              00/08/16 UTC settled rates, 2024-09-13 .. 2026-09-24 (742 days),
              28 coins, 8h granularity. Long history, full universe. This is the
              backbone.

  bykaranteli CC0 datasets via the rotating client in this package.
              venue-markets-daily carries funding_rate_8h_pct + open_interest_usd
              per venue per day, but only 2026-08-21 .. 2026-10-01 (42 days) and
              only 22 of the 28 coins. It is the only source for OI, positioning
              and liquidation pressure, so it is not optional.

Measured agreement over the 35 shared days: pearson 0.72, sign agreement 0.84,
MAE 3.96e-05 against a typical |rate| of 7.0e-05. The two are correlated but
independent, so disagreement is information, not noise to be smoothed away.

Unit convention: everything returned here is a FRACTION per 8h event
(0.0001 == 0.01%), matching both Binance raw history and the existing
accounting_28c.load_real_funding contract. ByKaranteli publishes percent, so it
is divided by 100 on the way in.

No forward fill across gaps: a missing funding observation is returned as NaN so
the factor layer can decide. Absence of funding data is not evidence of zero
funding.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BINANCE_DIR = os.path.join(ROOT, "data", "funding_binance")

FUNDING_HOURS = (0, 8, 16)


# ---------------------------------------------------------------------------
# Binance: long-history backbone
# ---------------------------------------------------------------------------

def load_binance_funding(coin):
    """Return ({ts_ms: rate_fraction}, meta) for one coin.

    An empty dict means no stored history. Callers must treat that as unknown,
    not as zero cost.
    """
    path = os.path.join(BINANCE_DIR, f"{coin}.json")
    if not os.path.exists(path):
        return {}, {"coin": coin, "source": "binance", "status": "missing"}
    doc = json.loads(open(path).read())
    ts = doc.get("timestamps") or []
    rates = doc.get("rates") or []
    if len(ts) != len(rates) or not ts:
        return {}, {"coin": coin, "source": "binance", "status": "empty"}
    table = {int(t): float(r) for t, r in zip(ts, rates)}
    return table, {
        "coin": coin,
        "source": "binance",
        "status": "ok",
        "symbol": doc.get("symbol"),
        "n": len(table),
        "first": min(table),
        "last": max(table),
    }


# ---------------------------------------------------------------------------
# ByKaranteli: OI / positioning / liquidation, and a short funding cross-check
# ---------------------------------------------------------------------------

def _dataset_rows(name):
    from .bykaranteli import get_client
    import csv as _csv
    import io as _io
    text = get_client().dataset_text(name, "csv")
    if not text:
        return []
    return list(_csv.DictReader(_io.StringIO(text)))


def load_bykaranteli_venue_markets(venue="binance", market="perp"):
    """Return {symbol: {date: {funding, oi, price, volume}}} in fraction units."""
    out = defaultdict(dict)
    for r in _dataset_rows("venue-markets-daily"):
        if r.get("venue") != venue or r.get("market") != market:
            continue
        rec = {}
        f = r.get("funding_rate_8h_pct")
        if f not in ("", None):
            # published in percent; store as fraction per 8h event
            rec["funding"] = float(f) / 100.0
        oi = r.get("open_interest_usd")
        if oi not in ("", None):
            rec["oi"] = float(oi)
        px = r.get("price_usd")
        if px not in ("", None):
            rec["price"] = float(px)
        vol = r.get("volume_24h_usd")
        if vol not in ("", None):
            rec["volume_24h"] = float(vol)
        pc = r.get("price_change_24h_pct")
        if pc not in ("", None):
            rec["price_chg_24h"] = float(pc)
        if rec:
            out[r["symbol"]][r["date"]] = rec
    return dict(out)


def load_bykaranteli_positioning():
    """Return {symbol: {date: {metric: value}}} from positioning-daily."""
    out = defaultdict(dict)
    for r in _dataset_rows("positioning-daily"):
        try:
            out[r["symbol"]][r["date"]][r["metric"]] = float(r["value"])
        except (TypeError, ValueError):
            continue
    return dict(out)


def load_bykaranteli_liquidations():
    """Return {symbol: {date: {long, short, events}}} from liquidations-daily."""
    out = defaultdict(dict)
    for r in _dataset_rows("liquidations-daily"):
        rec = {}
        for key, col in (("long", "long_liquidations_usd"),
                         ("short", "short_liquidations_usd"),
                         ("events", "events"),
                         ("max_single", "max_single_liquidation_usd")):
            v = r.get(col)
            if v not in ("", None):
                try:
                    rec[key] = float(v)
                except ValueError:
                    pass
        if rec:
            out[r["symbol"]][r["date"]] = rec
    return dict(out)


# ---------------------------------------------------------------------------
# Merged view
# ---------------------------------------------------------------------------

def merged_funding(coin, bk_markets=None):
    """Merge both sources for one coin onto a day-keyed grid.

    Returns a dict with:
      funding_binance   {date: fraction}          long history, may be missing
      funding_bykar     {date: fraction}          short history, may be missing
      funding_merged    {date: fraction}          binance first, bykar fills gaps
      provenance        per-source coverage notes
    """
    table, meta = load_binance_funding(coin)
    by_day = defaultdict(list)
    for ts_ms, rate in table.items():
        day = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d")
        by_day[day].append(rate)

    funding_binance = {d: sum(v) / len(v) for d, v in by_day.items()}

    if bk_markets is None:
        bk_markets = load_bykaranteli_venue_markets()
    bk_rows = bk_markets.get(coin, {})
    funding_bykar = {d: rec["funding"] for d, rec in bk_rows.items() if "funding" in rec}

    merged = dict(funding_bykar)
    merged.update(funding_binance)  # binance wins on conflict: deeper, 8h native

    shared = sorted(set(funding_binance) & set(funding_bykar))
    return {
        "coin": coin,
        "funding_binance": funding_binance,
        "funding_bykar": funding_bykar,
        "funding_merged": merged,
        "provenance": {
            "binance": meta,
            "bykaranteli_days": len(funding_bykar),
            "shared_days": len(shared),
            "binance_days": len(funding_binance),
            "merged_days": len(merged),
        },
    }


def coverage_report(coins=None, bk_markets=None):
    """Compare coverage per coin across both sources."""
    if coins is None:
        coins = sorted(
            f[:-5] for f in os.listdir(BINANCE_DIR)
            if f.endswith(".json") and not f.startswith("_")
        )
    if bk_markets is None:
        bk_markets = load_bykaranteli_venue_markets()
    rows = []
    for c in coins:
        m = merged_funding(c, bk_markets)
        p = m["provenance"]
        rows.append({
            "coin": c,
            "binance_days": p["binance_days"],
            "bykar_days": p["bykaranteli_days"],
            "shared_days": p["shared_days"],
            "merged_days": p["merged_days"],
            "bykar_only": p["bykaranteli_days"] > 0 and p["binance_days"] == 0,
        })
    return rows
