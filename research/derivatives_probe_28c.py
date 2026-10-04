#!/usr/bin/env python3
"""Directional probe for OI, positioning and liquidation factors.

Why this is shaped differently from funding_probe_28c.py
-------------------------------------------------------
Funding has 742 days of history, so it could be tested on the full 3y grid with a
real 60/20/20 split. The ByKaranteli derivatives fields do not:

    open interest   42 days  (2026-08-21 .. 2026-10-01)
    positioning     29 days  (2026-09-03 .. 2026-10-01)
    liquidations    63 days  (2026-07-30 .. 2026-09-30)

That is 5.7% of the 35,553-bar grid, and it sits entirely at the tail. A
60/20/20 split of that window would leave roughly 500 bars per segment, which
cannot support a promotion decision. So this script does NOT claim a promotion
verdict. It reports direction and effect size, and states the power ceiling.

Two statistics, because they answer different questions:

  time-series IC   does the factor predict the next return in THIS coin
  cross-sectional  does ranking coins by the factor rank next return across coins

The cross-sectional one is the more informative of the two here: it uses every
coin on every day, so it gets 40-60 daily observations rather than one per coin.

Splits inside the window are first-half / second-half. That is a stability check,
not an out-of-sample test, and the output says so in those words.

Usage:
    python research/derivatives_probe_28c.py
"""
from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from research.universe_28c import COINS_28C  # noqa: E402
from data_pipeline.providers import funding_merge as fm  # noqa: E402

DATA = ROOT / "data" / "data_3y" / "30m"


def load_daily_close(coins):
    """{coin: (date_str -> close)} plus the sorted union of dates."""
    out = {}
    alld = None
    for c in coins:
        with (DATA / f"{c}.csv").open() as f:
            rows = list(csv.DictReader(f))
        d = {}
        for r in rows:
            day = datetime.fromtimestamp(int(r["timestamp"]) / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
            d[day] = float(r["close"])
        out[c] = d
        s = set(d)
        alld = s if alld is None else alld & s
    return out, sorted(alld)


def bare(sym):
    return sym[:-4] if sym.endswith("USDT") else sym


# ---------------------------------------------------------------------------
# Factor construction, all on a daily grid
# ---------------------------------------------------------------------------

def build_oi(coins, dates):
    """OI level and OI change. Binance perp only, so the series is comparable
    across coins without mixing a thin venue into the level."""
    rows = fm._dataset_rows("venue-markets-daily")
    lvl = {}
    for r in rows:
        if r.get("venue") != "binance" or r.get("market") != "perp":
            continue
        oi = r.get("open_interest_usd")
        if oi in ("", None):
            continue
        lvl.setdefault(bare(r["symbol"]), {})[r["date"]] = float(oi)
    out = {}
    for c in coins:
        src = lvl.get(c, {})
        s = [src.get(d, np.nan) for d in dates]
        a = np.array(s, dtype=np.float64)
        # OI change over 1d, expressed as a fraction of the level
        chg = np.full(len(a), np.nan)
        chg[1:] = (a[1:] - a[:-1]) / np.where(a[:-1] > 0, a[:-1], np.nan)
        out[c] = {"OI_LOG": np.log(np.where(a > 0, a, np.nan)), "OI_CHG": chg}
    return out


def build_positioning(coins, dates):
    """Long/short ratios and taker imbalance, summed across venues per day."""
    rows = fm._dataset_rows("positioning-daily")
    acc = {}
    for r in rows:
        try:
            v = float(r["value"])
        except (TypeError, ValueError):
            continue
        acc.setdefault((bare(r["symbol"]), r["date"]), []).append((r["exchange"], r["metric"], v))

    out = {}
    for c in coins:
        gls, tbr, top = [], [], []
        for d in dates:
            rec = acc.get((c, d), [])
            bym = {}
            for ex, m, v in rec:
                bym.setdefault(m, []).append(v)
            g = bym.get("global_ls_ratio")
            t = bym.get("taker_buy_sell_ratio")
            p = bym.get("top_position_ls_ratio") or bym.get("top_account_ls_ratio")
            gls.append(float(np.mean(g)) if g else np.nan)
            tbr.append(float(np.mean(t)) if t else np.nan)
            top.append(float(np.mean(p)) if p else np.nan)
        out[c] = {
            "LS_RATIO": np.array(gls, dtype=np.float64),
            "TAKER_IMB": np.array(tbr, dtype=np.float64),
            "TOP_LS": np.array(top, dtype=np.float64),
        }
    return out


def build_liquidations(coins, dates):
    """Long/short liquidation split and total intensity, summed across venues."""
    rows = fm._dataset_rows("liquidations-daily")
    acc = {}
    for r in rows:
        c = bare(r["symbol"])
        try:
            lo = float(r.get("long_liquidations_usd") or 0.0)
            sh = float(r.get("short_liquidations_usd") or 0.0)
            ev = float(r.get("events") or 0.0)
        except ValueError:
            continue
        k = (c, r["date"])
        a = acc.setdefault(k, [0.0, 0.0, 0.0])
        a[0] += lo; a[1] += sh; a[2] += ev

    out = {}
    for c in coins:
        L, S, E, imb = [], [], [], []
        for d in dates:
            a = acc.get((c, d), [0.0, 0.0, 0.0])
            lo, sh, ev = a
            tot = lo + sh
            L.append(lo); S.append(sh); E.append(ev)
            imb.append((lo - sh) / tot if tot > 0 else np.nan)
        L = np.array(L); S = np.array(S); E = np.array(E)
        tot = L + S
        out[c] = {
            "LIQ_IMB": np.array(imb, dtype=np.float64),
            "LIQ_LOG": np.log(np.where(tot > 0, tot, np.nan)),
        }
    return out


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def fwd_ret(close_map, dates, i):
    """Next-day simple return, NaN at the end of the grid."""
    out = {}
    for c, d in close_map.items():
        a = np.array([d.get(x, np.nan) for x in dates], dtype=np.float64)
        out[c] = a[i + 1] / a[i] - 1.0 if i + 1 < len(a) else np.nan
    return out


def ts_ic(x, y):
    m = ~np.isnan(x) & ~np.isnan(y)
    if m.sum() < 10 or x[m].std() < 1e-14 or y[m].std() < 1e-14:
        return None
    return float(np.corrcoef(x[m], y[m])[0, 1])


def cs_ic(factors, rets, coins):
    """Rank correlation across coins, per day, averaged over days.

    factors and rets are {coin: array} already sliced to the same segment, so
    index i is a position inside the segment, not an absolute grid index.
    """
    vals = []
    n = min(len(rets[c]) for c in coins)
    for i in range(n):
        x = np.array([factors[c][i] for c in coins], dtype=np.float64)
        y = np.array([rets[c][i] for c in coins], dtype=np.float64)
        m = ~np.isnan(x) & ~np.isnan(y)
        if m.sum() < 8:
            continue
        x, y = x[m], y[m]
        if x.std() < 1e-14 or y.std() < 1e-14:
            continue
        rx = np.argsort(np.argsort(x)).astype(float)
        ry = np.argsort(np.argsort(y)).astype(float)
        vals.append(float(np.corrcoef(rx, ry)[0, 1]))
    if not vals:
        return None, 0
    return float(np.mean(vals)), len(vals)


def main():
    coins = list(COINS_28C)
    close_map, dates = load_daily_close(coins)
    print(f"price grid {dates[0]} .. {dates[-1]} ({len(dates)} days)")

    oi = build_oi(coins, dates)
    pos = build_positioning(coins, dates)
    liq = build_liquidations(coins, dates)

    # builders return {coin: {factor_name: array}}, so transpose to
    # {factor_name: {coin: array}} which is the axis the scan below walks
    factors = {}
    for src in (oi, pos, liq):
        for coin, per_factor in src.items():
            for factor_name, arr in per_factor.items():
                factors.setdefault(factor_name, {})[coin] = arr

    # restrict every factor to the window where data actually exists
    spans = {}
    for name, per_coin in factors.items():
        arrs = [per_coin[c] for c in coins if c in per_coin]
        if not arrs:
            continue
        firsts = [next((i for i, v in enumerate(a) if not np.isnan(v)), None) for a in arrs]
        lasts = [max((i for i, v in enumerate(a) if not np.isnan(v)), default=None) for a in arrs]
        f_ = [x for x in firsts if x is not None]
        l_ = [x for x in lasts if x is not None]
        if f_ and l_:
            spans[name] = (min(f_), max(l_))

    coverage = {}
    for name, per_coin in factors.items():
        if name not in spans:
            continue
        lo, hi = spans[name]
        have = [c for c in coins if c in factors[name]]
        covered = [c for c in have
                   if any(not np.isnan(factors[name][c][i]) for i in range(lo, hi + 1))]
        cov = {}
        for c in have:
            span = factors[name][c][lo:hi + 1]
            cov[c] = float((~np.isnan(span)).sum() / max(len(span), 1))
        coverage[name] = {
            "window": f"{dates[lo]} .. {dates[hi]}",
            "days": hi - lo + 1,
            "n_coins_with_data": len(covered),
            "coins_with_data": sorted(covered),
            "coins_missing": sorted(set(coins) - set(covered)),
            "mean_coverage_within_window": float(np.mean(list(cov.values()))) if cov else 0.0,
        }

    # first-half / second-half within each factor's own window
    results = {}
    for name, per_coin in factors.items():
        if name not in spans or name not in coverage:
            continue
        lo, hi = spans[name]
        if hi - lo < 8:
            continue
        mid = lo + (hi - lo) // 2
        row = {"window": coverage[name]["window"], "days": coverage[name]["days"],
               "n_coins_with_data": coverage[name]["n_coins_with_data"],
               "segments": {}}
        for seg, (a, b) in (("first_half", (lo, mid)), ("second_half", (mid, hi))):
            seg_coins = [c for c in coins
                     if c in per_coin and (~np.isnan(per_coin[c][a:b + 1])).sum() >= 5]
            ic_vals, cs_vals = {}, []
            for c in seg_coins:
                x = per_coin[c][a:b]
                # forward return for each bar in [a, b), i.e. next day from that bar
                y_all = np.array(
                    [fwd_ret(close_map, dates, i).get(c, np.nan) for i in range(a, b)],
                    dtype=np.float64,
                )
                v = ts_ic(x, y_all)
                if v is not None:
                    ic_vals[c] = v
            # cross-sectional over the whole segment
            # factor over bars [a, b); matching forward return for each of those bars
            xa = {c: per_coin[c][a:b] for c in seg_coins}
            rets = {}
            for c in seg_coins:
                rets[c] = np.array(
                    [fwd_ret(close_map, dates, i).get(c, np.nan) for i in range(a, b)],
                    dtype=np.float64,
                )
            cv, cn = cs_ic(xa, rets, seg_coins)
            row["segments"][seg] = {
                "n_coins": len(seg_coins),
                "ts_ic_mean": float(np.mean(list(ic_vals.values()))) if ic_vals else None,
                "ts_ic_median": float(np.median(list(ic_vals.values()))) if ic_vals else None,
                "ts_ic_positive": sum(1 for v in ic_vals.values() if v > 0),
                "ts_ic_per_coin": ic_vals,
                "cs_rank_ic_mean": cv,
                "cs_days": cn,
            }
        results[name] = row

    out = {
        "status": "directional_probe_only",
        "power_ceiling": (
            "open interest 42 days, positioning 29 days, liquidations 63 days. "
            "The longest window is 63 days, which cannot support a promotion "
            "decision or an out-of-sample claim. first_half/second_half is a "
            "stability check, NOT an out-of-sample split."
        ),
        "bars_used": "daily close, aligned to the ByKaranteli window",
        "factors": results,
        "coverage": coverage,
    }

    p = ROOT / "results" / "derivatives_probe_28c.json"
    p.write_text(json.dumps(out, indent=2, default=float))

    for name, row in results.items():
        f, s = row["segments"]["first_half"], row["segments"]["second_half"]
        print()
        print(f"=== {name}  [{row['window']}]  {row['days']}d  "
              f"{row['n_coins_with_data']}/{len(coins)} coins")
        print(f"    ts_ic   first {f['ts_ic_mean'] if f['ts_ic_mean'] is None else round(f['ts_ic_mean'],4)}"
              f"   second {s['ts_ic_mean'] if s['ts_ic_mean'] is None else round(s['ts_ic_mean'],4)}")
        print(f"    cs_rank first {f['cs_rank_ic_mean'] if f['cs_rank_ic_mean'] is None else round(f['cs_rank_ic_mean'],4)}"
              f"   second {s['cs_rank_ic_mean'] if s['cs_rank_ic_mean'] is None else round(s['cs_rank_ic_mean'],4)}")
        print(f"    coins missing: {', '.join(coverage[name]['coins_missing']) or '(none)'}")
    print()
    print("wrote", p)


if __name__ == "__main__":
    main()
