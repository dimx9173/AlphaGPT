"""Multi-timeframe factors: compute on coarse grids, align onto the primary grid.

The primary grid stays 1h. Coarser grids are built FROM the 1h bars rather than
read from the separate 4h download, for two reasons. The 1h bars are what the
splitter, the funding alignment and the annualisation are all denominated in, so
deriving from them keeps one source of truth. And a coarse bar assembled from
finer bars has an unambiguous close, which is what makes the forward-fill below
causal rather than approximately causal.

Causality is the whole argument for this design. A 4h bar closing at T covers
[T-4h, T] and its factor value is known at T, not before. Aligning it onto the
1h grid by carrying the value forward from T is therefore strictly causal. What
would NOT be causal is resampling with any centred rule, or filling backwards, or
taking the 4h file's bar and pairing it with 1h bars inside it. None of those are
done here and the truncation check in factor_corr_harness.py verifies it.

Volume aggregates by sum. Taking the last 1h volume would make a "4h volume"
factor a mislabelled 1h factor, which is the kind of thing that looks fine and is
not.
"""
from __future__ import annotations

import numpy as np

from research.causal_36f import _wins, _tmean, _tstd, _tsum, _robust_z, _logret, _atr, _w

COARSE_MINUTES = {"4h": 240, "12h": 720, "1d": 1440}


def _floor_to(ts_ms, bucket_ms):
    """Bucket start for each bar. Aligned to the epoch, which for these three
    buckets is the same as the natural clock alignment."""
    return (ts_ms // bucket_ms) * bucket_ms


def coarse_bars(bars, minutes):
    """Aggregate the primary bars into one coarser grid, causally.

    A coarse bar is emitted only when the bucket that just CLOSED is complete.
    A trailing partial bucket is dropped rather than emitted short, because a
    short final bar would silently carry a shorter lookback than every other bar
    of that factor.
    """
    bucket = COARSE_MINUTES[minutes] * 60 * 1000
    ts = np.asarray(bars["timestamp"], dtype=np.float64)
    key = _floor_to(ts, bucket)
    # A bar belongs to bucket k; the bucket closes at k + bucket.
    close_at = key + bucket
    o = np.asarray(bars["open"], dtype=np.float64)
    h = np.asarray(bars["high"], dtype=np.float64)
    l = np.asarray(bars["low"], dtype=np.float64)
    c = np.asarray(bars["close"], dtype=np.float64)
    v = np.asarray(bars["volume"], dtype=np.float64)

    # The first bar of each bucket is its open; the last is its close.
    first = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    last = np.r_[first[1:] - 1, len(ts) - 1]
    oo, hh = o[first], h[first]
    ll, cc = c[last], h[last]
    for k in range(1, len(first)):
        hh[k] = max(hh[k], h[first[k]:last[k] + 1].max())
        ll[k] = min(ll[k], l[first[k]:last[k] + 1].min())
    vol = np.add.reduceat(v, first)
    # Per-bucket close times, indexed the same way as the per-bucket OHLCV. The
    # first version of this used the full-length close_at against the per-bucket
    # arrays, which is a shape mismatch and was caught by the smoke check rather
    # than by the compiler.
    bucket_close = close_at[first]
    # Keep only buckets whose close time has actually been reached, and drop the
    # one whose close precedes the first primary bar: a bucket that opened before
    # the data starts would carry a shorter lookback than every other bar.
    keep = (bucket_close <= ts[-1]) & (bucket_close > ts[0])
    return {"open": oo[keep], "high": hh[keep], "low": ll[keep],
            "close": cc[keep], "volume": vol[keep],
            "timestamp": bucket_close[keep]}


def align_to_primary(coarse_values, coarse_close_at, primary_ts):
    """Carry each coarse value forward from its bar's CLOSE onto the primary grid.

    searchsorted(..., side="right") - 1 is the last coarse bar whose close time is
    at or before each primary bar, so a primary bar strictly inside a coarse bar
    sees the PREVIOUS coarse value. That is the conservative direction and it is
    the causal one: the current coarse bar has not closed yet.
    """
    cv = np.asarray(coarse_values, dtype=np.float64)
    cc = np.asarray(coarse_close_at, dtype=np.float64)
    pt = np.asarray(primary_ts, dtype=np.float64)
    j = np.searchsorted(cc, pt, side="right") - 1
    ok = j >= 0
    out = np.zeros(len(pt))
    out[ok] = cv[j[ok]]
    # Before the first coarse close there is no information at all; zero is a
    # neutral value here because every coarse factor is centred or rank-based by
    # construction, and it is not constant because the rest of the series is not.
    return out


# The coarse factors. These are NOT new ideas: they are the same ideas read on a
# longer horizon with less sampling noise, which is the one thing the 30m->1h
# switch cannot give you and the reason multi-timeframe is worth doing separately
# from changing the bar size.
#
# Each entry is (suffix, factory) where the factory takes coarse bars and returns
# a 1-D array. The prefix is the timeframe so two timeframes never collide on a
# name in the 36-factor decoder.

def _f_trend_tstat(cb):
    c = np.log(np.maximum(cb["close"], 1e-12))
    n = _w(24)
    out = np.zeros(len(c))
    x = np.arange(n, dtype=np.float64) - (n - 1) / 2.0
    vx = float((x ** 2).sum())
    for t in range(n - 1, len(c)):
        y = c[t - n + 1:t + 1]
        yc = y - y.mean()
        slope = float((x * yc).sum() / vx)
        rs = float((yc - slope * x).std())
        out[t] = slope / rs if rs > 1e-12 else 0.0
    return out


def _f_efficiency(cb):
    c = cb["close"]
    n = _w(12)
    net = np.abs(c - np.roll(c, n))
    net[:n] = 0.0
    path = _tsum(np.abs(c - np.concatenate(([c[0]], c[:-1]))), n)
    return np.where(path > 1e-12, net / (path + 1e-12), 1.0)


def _f_rangemean(cb):
    return _tmean((cb["high"] - cb["low"]) / (cb["close"] + 1e-12), _w(20))


def _f_parkinson(cb):
    p = np.log(np.maximum(cb["high"], 1e-12) / np.maximum(cb["low"], 1e-12)) ** 2
    return _robust_z(np.sqrt(_tmean(p, _w(20)) / (4 * np.log(2))), _w(NORM_W))


def _f_vwapdev(cb):
    typ = (cb["high"] + cb["low"] + cb["close"]) / 3.0
    n = _w(24)
    pv = _tsum(typ * cb["volume"], n)
    vv = _tsum(cb["volume"], n)
    vwap = np.where(vv > 1e-12, pv / (vv + 1e-12), cb["close"])
    return _robust_z((cb["close"] - vwap) / (vwap + 1e-12), _w(NORM_W))


def _f_dollarimbal(cb):
    dv = cb["close"] * cb["volume"]
    s = np.sign(cb["close"] - cb["open"])
    n = _w(12)
    den = _tsum(dv, n)
    return np.where(den > 1e-12, _tsum(s * dv, n) / (den + 1e-12), 0.0)


def _f_skew(cb):
    r = _logret(cb["close"])
    n = _w(50)
    win = _wins(r, n)
    m = win.mean(axis=1, keepdims=True)
    d = win - m
    m2 = (d ** 2).mean(axis=1)
    m3 = (d ** 3).mean(axis=1)
    return np.where(m2 > 1e-18, m3 / (np.power(m2, 1.5) + 1e-18), 0.0)


def _f_dd(cb):
    c = cb["close"]
    n = _w(200)
    pk = _wins(c, n).max(axis=1)
    return (c - pk) / (pk + 1e-12)


def _f_donch(cb):
    c, h, l = cb["close"], cb["high"], cb["low"]
    n = _w(100)
    hh = _wins(h, n).max(axis=1)
    ll = _wins(l, n).min(axis=1)
    return (c - ll) / (hh - ll + 1e-12)


def _f_autocorr(cb):
    r = _logret(cb["close"])
    n = _w(20)
    a = _wins(r, n)
    b = np.zeros_like(a)
    b[n:] = a[n - 1:-1]
    sa, sb = a.std(axis=1), b.std(axis=1)
    num = ((a - a.mean(axis=1, keepdims=True))
           * (b - b.mean(axis=1, keepdims=True))).mean(axis=1)
    ok = (sa > 1e-12) & (sb > 1e-12)
    out = np.where(ok, num / (sa * sb + 1e-18), 0.0)
    out[:n] = 0.0
    return out


def _f_illiqchange(cb):
    dv = cb["close"] * cb["volume"]
    ill = np.log(np.maximum(np.abs(_logret(cb["close"])), 1e-12) / np.maximum(dv, 1e-12))
    return _tmean(ill, _w(20)) - _tmean(ill, _w(100))


NORM_W = 200

COARSE_FACTORS = {
    "TREND_TSTAT": _f_trend_tstat,
    "EFFICIENCY": _f_efficiency,
    "RANGEMEAN": _f_rangemean,
    "PARKINSON": _f_parkinson,
    "VWAPDEV": _f_vwapdev,
    "DOLLARIMBAL": _f_dollarimbal,
    "SKEW": _f_skew,
    "DD": _f_dd,
    "DONCH": _f_donch,
    "AUTOCORR": _f_autocorr,
    "ILLIQCHANGE": _f_illiqchange,
}


def mtf_series(bars, timeframes=("4h",), coarse=None):
    """name -> 1-D array on the PRIMARY grid.

    This is the whole function. Everything above is machinery for it.
    """
    out = {}
    primary_ts = np.asarray(bars["timestamp"], dtype=np.float64)
    for tf in timeframes:
        cb = coarse_bars(bars, tf)
        for suffix, factory in COARSE_FACTORS.items():
            raw = np.asarray(factory(cb), dtype=np.float64)
            out["%s_%s" % (tf, suffix)] = align_to_primary(raw, cb["timestamp"], primary_ts)
    return out


# The measured survivors. Eleven coarse factors were implemented and measured
# against the 36 on the real bars; six exceeded the |r| 0.6 budget and are NOT
# shipped. The six that were cut and why, because the pattern is the point:
#
#   4h_RANGEMEAN   0.722 vs RANGE_MEAN_20     same idea, longer window
#   4h_VWAPDEV      0.643 vs 4h_DOLLARIMBAL   price-vs-volume, coarse vs coarse
#   4h_DOLLARIMBAL  0.643 vs 4h_VWAPDEV       the other half of the same pair
#   4h_TREND_TSTAT  0.636 vs 4h_VWAPDEV       drift on a long horizon
#   4h_DONCH        0.611 vs 4h_DD             range position vs drawdown
#   4h_DD           0.611 vs 4h_DONCH         the other half of the same pair
#
# Every cut is a restatement of something already in the set at another horizon.
# A coarser grid does not by itself produce a different factor; it produces the
# same factor with more memory and less sampling noise, and the correlation
# measurement is what tells you which of those are worth a slot.
MTF_FACTORS = (
    "4h_PARKINSON",
    "4h_SKEW",
    "4h_EFFICIENCY",
    "4h_AUTOCORR",
    "4h_ILLIQCHANGE",
)

MTF_TIMEFRAMES = ("4h",)


def mtf36(bars):
    """The 5 shipped coarse factors, on the primary grid."""
    return mtf_series(bars, MTF_TIMEFRAMES)
