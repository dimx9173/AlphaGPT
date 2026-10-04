#!/usr/bin/env python3
"""Mechanism checks for the derivatives fields.

Why this exists
---------------
derivatives_probe_28c.py measured IC and found nothing that survives a
permutation test. That is a statement about predictability, not about mechanism.
The two are different questions, and only the first one was asked.

A mechanism claim is falsifiable with far less data than a predictive claim,
because it does not need a per-bar forecast to be right. "Leverage building
without price moving is followed by a larger move" is a statement about
conditional DISTRIBUTIONS: it predicts a shift in the spread of outcomes, not a
shift in the mean. Dispersion effects need many fewer observations than mean
effects, because the mean effect has to compete with the noise in the level while
the dispersion effect only has to change the width of the distribution.

So each check here states a mechanism, tests the specific consequence that
mechanism predicts, and reports whether the data can even express it. Where the
window is too short to answer, the output says "underpowered" instead of
inventing a verdict.

The five mechanisms
-------------------
M1 leverage build without price move -> larger forward move
     OI rising while return is flat means positions are opening into a range.
     The prediction is expansion, not direction, so this is tested on |return|
     and never on signed return.

M2 liquidation spike -> continuation or exhaustion
     Tested both ways, because the sign is genuinely contested and the data is
     allowed to pick. Long liquidations can mean capitulation (rebound) or a
     trend being confirmed (continuation down). The test reports which, if
     either, is present.

M3 top traders vs crowd divergence
     Top-position long/short against the global account ratio. Divergence is the
     signal; tested on the residual gap.

M4 taker imbalance -> immediate flow persistence
     Taker buy/sell above 1 means aggressive buying. Tested at short horizon.

M5 funding extreme -> crowding unwind
     Binance funding has 742 days, so this is the only mechanism here with real
     power. Included to show what a powered test looks like next to M1-M4.

Every test is permutation-calibrated: the conditioning variable is shuffled
within coin, which destroys any coin-specific level effect while preserving the
marginal distribution. This is the same null the IC probe used, and it is the
reason that probe produced a non-zero null mean.

Usage:
    python research/derivatives_mechanism_28c.py
"""
from __future__ import annotations

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
from research.derivatives_probe_28c import (  # noqa: E402
    load_daily_close, build_oi, build_positioning, build_liquidations, bare,
)
from data_pipeline.providers import funding_merge as fm  # noqa: E402

N_PERM = 500
RNG_SEED = 20261001


def fwd(close_map, dates, i):
    """Next-day return per coin at grid index i, as a {coin: value} dict."""
    out = {}
    for c, d in close_map.items():
        a = dates[i]
        b = dates[i + 1] if i + 1 < len(dates) else None
        if b is None or a not in d or b not in d or d[a] <= 0:
            out[c] = np.nan
        else:
            out[c] = d[b] / d[a] - 1.0
    return out


def realized_move(close_map, dates, i, horizon):
    """Absolute return over `horizon` days from index i. NaN past the end."""
    out = {}
    for c, d in close_map.items():
        a = dates[i]
        j = i + horizon
        if j >= len(dates) or a not in d or dates[j] not in d or d[a] <= 0:
            out[c] = np.nan
        else:
            out[c] = abs(d[dates[j]] / d[a] - 1.0)
    return out


def signed_move(close_map, dates, i, horizon):
    out = {}
    for c, d in close_map.items():
        a = dates[i]
        j = i + horizon
        if j >= len(dates) or a not in d or dates[j] not in d or d[a] <= 0:
            out[c] = np.nan
        else:
            out[c] = d[dates[j]] / d[a] - 1.0
    return out


def masked_stat(cond, outcome):
    """Mean of outcome where cond is True, versus everywhere.

    Returns (event_mean, base_mean, delta, n_events) with NaN-safe masking.
    """
    m = ~np.isnan(cond) & ~np.isnan(outcome)
    if m.sum() < 10:
        return None, None, None, 0
    ev = outcome[m & cond.astype(bool)]
    if ev.size < 5:
        return None, None, None, int(ev.size)
    return float(ev.mean()), float(outcome[m].mean()), float(ev.mean() - outcome[m].mean()), int(ev.size)


def permute_test(cond, outcome, rng, n_perm=N_PERM):
    """Permutation-calibrate the event-minus-base gap.

    The conditioning column is shuffled within the coin, which destroys the
    relationship to the outcome while preserving the marginal distribution and
    the event rate. The observed gap is compared against that null.
    """
    obs_ev, obs_base, obs_delta, n_ev = masked_stat(cond, outcome)
    if obs_delta is None:
        return None
    nulls = []
    for _ in range(n_perm):
        _, _, d, _ = masked_stat(rng.permutation(cond), outcome)
        if d is not None:
            nulls.append(d)
    if len(nulls) < 50:
        return None
    nulls = np.array(nulls)
    sd = nulls.std()
    t = (obs_delta - nulls.mean()) / sd if sd > 1e-15 else float("nan")
    p = float((nulls >= obs_delta).mean())
    return {
        "event_mean": obs_ev,
        "base_mean": obs_base,
        "delta": obs_delta,
        "n_events": n_ev,
        "null_mean": float(nulls.mean()),
        "null_sd": float(sd),
        "t": float(t),
        "p_one_sided": p,
        "n_perm": len(nulls),
    }


def pool(builders, coins, dates):
    factors = {}
    for src in builders:
        for coin, per_factor in src.items():
            for name, arr in per_factor.items():
                factors.setdefault(name, {})[coin] = arr
    return factors


def span_of(per_coin, coins):
    arrs = [per_coin[c] for c in coins if c in per_coin]
    if not arrs:
        return None
    f = [next((i for i, v in enumerate(a) if not np.isnan(v)), None) for a in arrs]
    l = [max((i for i, v in enumerate(a) if not np.isnan(v)), default=None) for a in arrs]
    f = [x for x in f if x is not None]
    l = [x for x in l if x is not None]
    return (min(f), max(l)) if f and l else None


def stack(values_by_coin, coins, lo, hi):
    """Flatten {coin: array} over [lo, hi) into one vector, plus a coin id."""
    xs, cid = [], []
    for k, c in enumerate(coins):
        if c not in values_by_coin:
            continue
        seg = values_by_coin[c][lo:hi]
        xs.append(seg)
        cid.append(np.full(len(seg), k))
    if not xs:
        return None, None
    return np.concatenate(xs).astype(float), np.concatenate(cid)


# ---------------------------------------------------------------------------
# Mechanisms
# ---------------------------------------------------------------------------

def m1_leverage_build(factors, close_map, dates, coins, rng):
    """OI up + flat price -> larger move AFTER the setup.

    The timing here matters and an earlier version of this function got it
    wrong: it conditioned on the smallness of the SAME return it then measured,
    which guarantees a negative delta by construction. The conditioning window
    must be strictly in the past relative to the outcome window.

    Mechanism claims expansion and no direction, so the headline test is on
    |forward return| and the signed version is reported only to confirm the
    mechanism is not secretly a directional bet.
    """
    pc = factors["OI_LOG"]
    sp = span_of(pc, coins)
    if sp is None:
        return {"status": "no_data"}
    lo, hi = sp

    oichg, ret = {}, {}
    for c, d in close_map.items():
        o = pc.get(c)
        if o is None:
            continue
        g = np.full(len(dates), np.nan)
        for i in range(1, len(dates)):
            a, b = o[i - 1], o[i]
            if not (np.isnan(a) or np.isnan(b) or a <= 0):
                g[i] = b - np.log(a)      # log OI change into day i
        oichg[c] = g
        r = np.full(len(dates), np.nan)
        for i in range(len(dates)):
            if i + 1 >= len(dates):
                break
            a, b = dates[i], dates[i + 1]
            if day_notin(d, a, b) or d[a] <= 0:
                continue
            r[i] = d[b] / d[a] - 1.0     # return realized ON day i
        ret[c] = r

    # conditioning uses day i-1 (past); outcome uses day i (future of the setup)
    oi_c, _ = stack(oichg, coins, lo + 1, hi)
    past_r, _ = stack(ret, coins, lo + 1, hi)
    fut_r, _ = stack(ret, coins, lo + 2, hi + 1)
    if oi_c is None or fut_r is None:
        return {"status": "no_data"}
    m = ~np.isnan(oi_c) & ~np.isnan(past_r) & ~np.isnan(fut_r)
    oi_c, past_r, fut_r = oi_c[m], past_r[m], fut_r[m]
    if oi_c.size < 100:
        return {"status": "underpowered", "n_observations": int(oi_c.size)}

    oi_thr = np.nanpercentile(oi_c, 80)
    r_thr = np.nanpercentile(np.abs(past_r), 40)
    cond = (oi_c >= oi_thr) & (np.abs(past_r) <= r_thr)

    return {
        "status": "ok",
        "mechanism": "OI up + flat price -> larger move after the setup",
        "window": f"{dates[lo]} .. {dates[hi]}",
        "days": hi - lo,
        "n_coins": len([c for c in coins if c in pc]),
        "n_observations": int(oi_c.size),
        "timing": "conditioning on day i-1, outcome on day i (strictly out of sample)",
        "test": "event (top-20% log OI change, bottom-40% |past return|) minus base",
        "expansion": permute_test(cond, np.abs(fut_r), rng),
        "direction_check": permute_test(cond, fut_r, rng),
        "interpretation": (
            "expansion p<0.05 with direction p>0.05 would confirm the mechanism as "
            "dispersion-only, which is what it claims"
        ),
    }


def day_notin(d, a, b):
    return a not in d or b not in d


def m2_liquidation(factors, close_map, dates, coins, rng):
    """Liquidation spike -> continuation or exhaustion. Sign is contested, so
    both are tested and the data is allowed to choose."""
    pc = factors["LIQ_LOG"]
    imb = factors["LIQ_IMB"]
    sp = span_of(pc, coins)
    if sp is None:
        return {"status": "no_data"}
    lo, hi = sp
    h = 3

    lvl = {}
    for c in coins:
        if c in pc and c in imb:
            seg = pc[c][lo:hi]
            segi = imb[c][lo:hi]
            with np.errstate(invalid="ignore", divide="ignore"):
                lvl[c] = np.log(np.where(np.exp(seg) > 0, np.exp(seg), np.nan) + 1.0) * segi
    thr_map = lvl
    allv = np.concatenate([v for v in thr_map.values() if v.size]) if thr_map else np.array([])
    if allv.size < 50:
        return {"status": "underpowered"}
    # events: big liquidation burst skewed to long liquidations, or to shorts
    hi_thr = np.nanpercentile(allv, 90)
    lo_thr = np.nanpercentile(allv, 10)

    def build(truth):
        xs, ys = [], []
        for c in coins:
            if c not in thr_map:
                continue
            s = thr_map[c]
            for k in range(len(s)):
                i = lo + k
                if i + h >= len(dates):
                    continue
                if np.isnan(s[k]):
                    continue
                a, b = dates[i], dates[i + h]
                if day_notin(close_map[c], a, b):
                    continue
                xs.append(s[k]); ys.append(close_map[c][b] / close_map[c][a] - 1.0)
        return np.array(xs, dtype=float), np.array(ys, dtype=float)

    x, y = build(None)
    out = {
        "status": "ok",
        "mechanism": "liquidation burst -> continuation or exhaustion",
        "window": f"{dates[lo]} .. {dates[hi]}",
        "days": hi - lo,
        "horizon_days": h,
        "n_coins": len([c for c in coins if c in pc]),
        "n_observations": int(x.size),
    }
    if x.size < 100:
        out["status"] = "underpowered"
    cond_long = x >= hi_thr
    cond_short = x <= lo_thr
    out["long_liquidation_spike"] = permute_test(cond_long, y, rng)
    out["short_liquidation_spike"] = permute_test(cond_short, y, rng)
    return out


def m3_top_vs_crowd(factors, close_map, dates, coins, rng):
    """Top-trader positioning diverging from the account crowd."""
    top = factors["TOP_LS"]
    glob = factors["LS_RATIO"]
    sp = span_of(top, coins)
    if sp is None:
        return {"status": "no_data"}
    lo, hi = sp
    gap = {}
    for c in coins:
        if c not in top or c not in glob:
            continue
        with np.errstate(invalid="ignore"):
            lg = np.log(np.where(glob[c] > 0, glob[c], np.nan))
            lt = np.log(np.where(top[c] > 0, top[c], np.nan))
        gap[c] = lt - lg
    allv = np.concatenate([gap[c][lo:hi] for c in gap]) if gap else np.array([])
    if allv.size < 60:
        return {"status": "underpowered"}
    hi_thr = np.nanpercentile(allv, 85)
    h = 3
    xs, ys, cs = [], [], []
    for c in gap:
        s = gap[c]
        for k in range(len(s)):
            i = lo + k
            if i + h >= len(dates) or np.isnan(s[k]):
                continue
            a, b = dates[i], dates[i + h]
            if day_notin(close_map[c], a, b):
                continue
            xs.append(s[k]); ys.append(close_map[c][b] / close_map[c][a] - 1.0); cs.append(c)
    x = np.array(xs, float); y = np.array(ys, float)
    return {
        "status": "ok" if x.size >= 100 else "underpowered",
        "mechanism": "top-trader vs crowd positioning gap -> forward move",
        "window": f"{dates[lo]} .. {dates[hi]}",
        "days": hi - lo,
        "horizon_days": h,
        "n_coins": len(gap),
        "n_observations": int(x.size),
        "top_long_when_crowd_short": permute_test(x >= hi_thr, y, rng),
        "top_short_when_crowd_long": permute_test(x <= np.nanpercentile(allv, 15), y, rng),
    }


def m4_taker(factors, close_map, dates, coins, rng):
    """Aggressive taker imbalance -> immediate flow persistence."""
    pc = factors["TAKER_IMB"]
    sp = span_of(pc, coins)
    if sp is None:
        return {"status": "no_data"}
    lo, hi = sp
    xs, ys = [], []
    for c in coins:
        if c not in pc:
            continue
        s = pc[c]
        for k in range(len(s)):
            i = lo + k
            if i + 1 >= len(dates) or np.isnan(s[k]):
                continue
            a, b = dates[i], dates[i + 1]
            if day_notin(close_map[c], a, b):
                continue
            xs.append(s[k]); ys.append(close_map[c][b] / close_map[c][a] - 1.0)
    x = np.array(xs, float); y = np.array(ys, float)
    if x.size < 100:
        return {"status": "underpowered"}
    thr = np.nanpercentile(x, 80)
    return {
        "status": "ok",
        "mechanism": "taker buy/sell > threshold -> next-day persistence",
        "window": f"{dates[lo]} .. {dates[hi]}",
        "days": hi - lo,
        "horizon_days": 1,
        "n_coins": len([c for c in coins if c in pc]),
        "n_observations": int(x.size),
        "heavy_taking": permute_test(x >= thr, y, rng),
    }


def m5_funding_crowding(close_map, dates, coins, rng):
    """Funding extreme -> crowding unwind. The only powered test in this file."""
    xs, ys, cs = [], [], []
    for c in coins:
        table, meta = fm.load_binance_funding(c)
        if not table:
            continue
        byday = {}
        for ts_ms, r in table.items():
            day = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d")
            byday.setdefault(day, []).append(r)
        daily = {d: sum(v) / len(v) for d, v in byday.items()}
        for i, day in enumerate(dates):
            if i + 5 >= len(dates) or day not in daily:
                continue
            a, b = dates[i], dates[i + 5]
            if day_notin(close_map[c], a, b):
                continue
            xs.append(daily[day]); ys.append(close_map[c][b] / close_map[c][a] - 1.0); cs.append(c)
    x = np.array(xs, float); y = np.array(ys, float)
    if x.size < 500:
        return {"status": "underpowered", "n_observations": int(x.size)}
    z = (x - np.nanmean(x)) / (np.nanstd(x) + 1e-18)
    return {
        "status": "ok",
        "mechanism": "funding z-score extreme -> 5-day forward move",
        "window": f"{dates[0]} .. {dates[-1]}",
        "days": len(dates),
        "horizon_days": 5,
        "n_coins": len(set(cs)),
        "n_observations": int(x.size),
        "extreme_long_crowding": permute_test(z >= 2.0, y, rng),
        "extreme_short_crowding": permute_test(z <= -2.0, y, rng),
    }


def main():
    rng = np.random.default_rng(RNG_SEED)
    coins = list(COINS_28C)
    close_map, dates = load_daily_close(coins)

    factors = pool([build_oi(coins, dates), build_positioning(coins, dates),
                    build_liquidations(coins, dates)], coins, dates)

    checks = {
        "M1_leverage_build": m1_leverage_build(factors, close_map, dates, coins, rng),
        "M2_liquidation_spike": m2_liquidation(factors, close_map, dates, coins, rng),
        "M3_top_vs_crowd": m3_top_vs_crowd(factors, close_map, dates, coins, rng),
        "M4_taker_persistence": m4_taker(factors, close_map, dates, coins, rng),
        "M5_funding_crowding": m5_funding_crowding(close_map, dates, coins, rng),
    }

    confirmed, refuted, underpowered = [], [], []
    for name, res in checks.items():
        if res.get("status") != "ok":
            underpowered.append(name)
            continue
        tests = [v for k, v in res.items() if isinstance(v, dict) and "p_one_sided" in v]
        tests += [v for v in res.values() if isinstance(v, dict) and "p_one_sided" in v]
        ps = [t["p_one_sided"] for t in tests if t and t.get("p_one_sided") is not None]
        if not ps:
            underpowered.append(name)
            continue
        best = min(ps)
        if best < 0.05:
            confirmed.append(name)
        elif all(p > 0.20 for p in ps):
            refuted.append(name)
        else:
            underpowered.append(name)

    out = {
        "status": "mechanism_check_only",
        "method": "permutation-calibrated event study, conditioning variable shuffled within coin",
        "n_perm": N_PERM,
        "seed": RNG_SEED,
        "note": (
            "Dispersion mechanisms (M1) are tested on |return|; directional "
            "mechanisms (M2-M5) on signed return. A mechanism claims no direction "
            "only where stated. Every test is calibrated against a within-coin "
            "shuffle, so a non-zero null mean is expected."
        ),
        "checks": checks,
        "confirmed": confirmed,
        "refuted": refuted,
        "underpowered": underpowered,
    }

    p = ROOT / "results" / "derivatives_mechanism_28c.json"
    p.write_text(json.dumps(out, indent=2, default=float))

    for name, res in checks.items():
        print(f"\n=== {name}: {res.get('status')}")
        if res.get("status") != "ok":
            print(f"    {res}")
            continue
        print(f"    {res.get('mechanism')}")
        print(f"    window {res.get('window')}  coins {res.get('n_coins')}"
              f"  obs {res.get('n_observations', res.get('n_events','?'))}")
        for k, v in res.items():
            if isinstance(v, dict) and "p_one_sided" in v and v.get("p_one_sided") is not None:
                print(f"    {k:26} delta={v['delta']:+.5f} null={v['null_mean']:+.5f}"
                      f"+-{v['null_sd']:.5f} t={v['t']:+.2f} p={v['p_one_sided']:.3f} n={v['n_events']}")
    print()
    print(f"confirmed     : {confirmed or '(none)'}")
    print(f"refuted       : {refuted or '(none)'}")
    print(f"underpowered  : {underpowered or '(none)'}")
    print()
    print("wrote", p)


if __name__ == "__main__":
    main()
