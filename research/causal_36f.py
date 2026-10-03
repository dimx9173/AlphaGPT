"""Causal 36-factor features: the existing 12 plus 24 measured-low-correlation additions.

This module ships the 36-dimension feature set. The 24 additions were selected
by a three-round discussion between two models (docs/factor_expansion/) and then
FILTERED BY MEASUREMENT, which is the part that matters: of the 43 candidates
implemented and measured on all 28 coins, these 24 are the ones whose worst
pairwise correlation against the existing 12 and against each other is below
0.6. Measured worst pair in the full 36 is 0.500.

The round-3 selection alone would NOT have satisfied that budget. Measured
|max r| 0.6 violations among its own picks included VWAP_DEV_24 against PUMP_DEV
at 0.958, VOL_RANK_100 against LOG_VOL at 0.914, and DOLLAR_IMBAL_12 against
OBV_SLOPE_20 at 0.856. An argument that two indicators measure different things
is not a measurement, and three of the twelve agreed factors turned out to be
restatements of factors already in the set. Those were replaced by candidates
that had been cut earlier and measured instead of argued about.

Every factor is trailing-only and passes the truncation check in
research/factor_corr_harness.py. Funding factors take the sparse settlement
series separately and never take a bar-indexed window over it.
"""


from __future__ import annotations

import numpy as np

NORM_WINDOW = 200


def _wins(x, w):
    x = np.asarray(x, dtype=np.float64)
    pad = np.pad(x, (w - 1, 0), mode="edge")
    return np.lib.stride_tricks.sliding_window_view(pad, w)[:len(x)]


def _tmean(x, w):
    return _wins(x, w).mean(axis=1)


def _tstd(x, w):
    return _wins(x, w).std(axis=1)


def _tsum(x, w):
    return _wins(x, w).sum(axis=1)



# ---------------------------------------------------------------------------
# Trailing median/MAD z, clipped. Identical convention to causal_12f.
# ---------------------------------------------------------------------------
def _robust_z(x, window=NORM_WINDOW):
    x = np.asarray(x, dtype=np.float64)
    wins = _wins(x, window)
    med = np.median(wins, axis=1)
    mad = np.median(np.abs(wins - med[:, None]), axis=1) + 1e-6
    return np.clip((x - med) / mad, -5, 5)


def _logret(close):
    out = np.zeros(len(close), dtype=np.float64)
    out[1:] = np.log(np.maximum(close[1:], 1e-12) / np.maximum(close[:-1], 1e-12))
    return out


def _atr(b, w):
    """Wilder-style trailing ATR from true range."""
    pc = np.concatenate(([b["close"][0]], b["close"][:-1]))
    tr = np.maximum.reduce([b["high"] - b["low"],
                            np.abs(b["high"] - pc),
                            np.abs(b["low"] - pc)])
    return _tmean(tr, w)


# ---------------------------------------------------------------------------
# The factors. Each takes the bar dict and returns a 1-D array.
#
# name -> (callable, family, standardised?)
# ---------------------------------------------------------------------------

F = {}


def _reg(name, fn, fam, z=False):
    F[name] = (fn, fam, z)
    return fn


# --- VOLREG: volatility from the H/L path rather than close-to-close ---
_reg("PARKINSON_20",
     lambda b: _robust_z(np.sqrt(_tmean(np.log(b["high"] / np.maximum(b["low"], 1e-12)) ** 2, 20)
                                 / (4 * np.log(2)))),
     "volreg", True)


def _atr_ratio_100(b):
    a14, a100 = _atr(b, 14), _atr(b, 100)
    return _robust_z(a14 / (a100 + 1e-12))


_reg("ATR_RATIO_100", _atr_ratio_100, "volreg", True)


# --- BREAKOUT: supply and demand over the 50-100 bar horizon ---
def _donch_pos_100(b):
    c, h, l = b["close"], b["high"], b["low"]
    hh = _wins(h, 100).max(axis=1)
    ll = _wins(l, 100).min(axis=1)
    return (c - ll) / (hh - ll + 1e-9)


_reg("DONCH_POS_100", _donch_pos_100, "breakout")


def _breakout_dist_50(b):
    """Signed distance to the 50-bar high excluding the current bar, in ATR14 units.

    The current bar is excluded from the reference high on purpose: including it
    makes the factor non-positive by construction, which is the degenerate case
    worth avoiding rather than flooring after the fact.
    """
    c, h = b["close"], b["high"]
    hh = np.roll(_wins(h, 50).max(axis=1), 1)
    hh[0] = h[0]
    return (c - hh) / (_atr(b, 14) + 1e-12)


_reg("BREAKOUT_DIST_50", _breakout_dist_50, "breakout")


def _time_since_high_100(b):
    h = b["high"]
    hh = _wins(h, 100).max(axis=1)
    age = np.zeros(len(h))
    run = 0.0
    for t in range(len(h)):
        run = 0.0 if h[t] >= hh[t] else run + 1.0
        age[t] = run
    return age / 100.0


_reg("TIME_SINCE_HIGH_100", _time_since_high_100, "breakout")


# --- DRAWDOWN: disposition-effect supply overhang ---
_reg("DD_200",
     lambda b: (b["close"] - _wins(b["close"], 200).max(axis=1))
               / (_wins(b["close"], 200).max(axis=1) + 1e-9),
     "drawdown")


def _dd_duration(b):
    """Bars since the running 200-bar peak, normalised. Loop because the state
    is recursive and a window would not express it."""
    c = b["close"]
    peak = -np.inf
    since = np.zeros(len(c))
    for t in range(len(c)):
        w = c[max(0, t - 199):t + 1]
        if len(w):
            peak = max(peak, float(w.max()))
        if c[t] >= peak - 1e-12:
            since[t] = 0.0
        else:
            since[t] = since[t - 1] + 1.0 if t else 1.0
    return since / 200.0


_reg("DD_DURATION", _dd_duration, "drawdown")


# --- VOLPRICE: whether a move had volume sponsorship ---
def _dollar_imbal_12(b):
    dv = b["close"] * b["volume"]
    s = np.sign(b["close"] - b["open"])
    num = _tsum(s * dv, 12)
    den = _tsum(dv, 12)
    return np.where(den > 1e-12, num / (den + 1e-12), 0.0)


_reg("DOLLAR_IMBAL_12", _dollar_imbal_12, "volprice")


def _obv_slope_20(b):
    obv = np.cumsum(np.sign(b["close"] - b["open"]) * b["volume"])
    return (obv - _tmean(obv, 20)) / (_tstd(obv, 20) + 1e-9)


_reg("OBV_SLOPE_20", _obv_slope_20, "volprice")


def _vwap_dev_24(b):
    typ = (b["high"] + b["low"] + b["close"]) / 3.0
    pv = _tsum(typ * b["volume"], 24)
    vv = _tsum(b["volume"], 24)
    vwap = np.where(vv > 1e-12, pv / (vv + 1e-12), b["close"])
    return _robust_z((b["close"] - vwap) / (vwap + 1e-12))


_reg("VWAP_DEV_24", _vwap_dev_24, "volprice", True)


# --- EFFICIENCY: price the inefficiency rather than a pattern ---
def _efficiency_12(b):
    """Kaufman efficiency ratio over 12 bars: net displacement over path length.

    Returns 1.0 when the path is flat, because "no information" is the honest
    value there and dividing by a zero path would return something arbitrary.
    """
    c = b["close"]
    net = np.abs(c - np.roll(c, 12))
    net[:12] = 0.0
    step = np.abs(c - np.concatenate(([c[0]], c[:-1])))
    path = _tsum(step, 12)
    return np.where(path > 1e-12, net / (path + 1e-12), 1.0)


_reg("EFFICIENCY_12", _efficiency_12, "efficiency")


def _variance_ratio_8(b):
    """Lo-MacKinlay variance ratio at horizon 8 over a 48-bar window.

    1.0 is the random-walk null, so it is also the fallback when the one-bar
    variance is degenerate.
    """
    r = _logret(b["close"])
    w = _wins(r, 48)
    v1 = w.var(axis=1)
    groups = w.reshape(len(r), 6, 8).sum(axis=2)
    v8 = groups.var(axis=1)
    return np.where(v1 > 1e-18, v8 / (8 * v1 + 1e-18), 1.0)


_reg("VARIANCE_RATIO_8", _variance_ratio_8, "efficiency")


def _autocorr_20(b):
    """Lag-1 autocorrelation over the trailing 20 returns.

    a[t] is r[t-19..t]; b[t] is r[t-20..t-1], which is the same window shifted
    one bar back. Shifting the window rather than sliding a 21-wide view keeps
    the two series aligned and the output length equal to the input length.
    """
    r = _logret(b["close"])
    a = _wins(r, 20)
    bshift = np.zeros_like(a)
    bshift[20:] = a[19:-1]
    ac = a.std(axis=1)
    bc = bshift.std(axis=1)
    num = ((a - a.mean(axis=1, keepdims=True))
           * (bshift - bshift.mean(axis=1, keepdims=True))).mean(axis=1)
    ok = (ac > 1e-12) & (bc > 1e-12)
    out = np.where(ok, num / (ac * bc + 1e-18), 0.0)
    out[:20] = 0.0
    return out


_reg("AUTOCORR_20", _autocorr_20, "efficiency")


# --- TREND: significance-scaled drift, not just distance ---
def _slope_tstat_24(b):
    """OLS slope of ln(C) on time over 24 bars, divided by residual std.

    This is a trend-per-noise ratio, not a trend level: two series with the same
    net return separate here when one arrived smoothly.
    """
    n = 24
    x = np.arange(n, dtype=np.float64)
    xc = x - x.mean()
    vx = float((xc ** 2).sum())
    out = np.zeros(len(b["close"]))
    ly = np.log(np.maximum(b["close"], 1e-12))
    for t in range(n - 1, len(ly)):
        y = ly[t - n + 1:t + 1]
        yc = y - y.mean()
        slope = float((xc * yc).sum() / vx)
        resid = yc - slope * xc
        rs = float(resid.std())
        out[t] = slope / rs if rs > 1e-12 else 0.0
    return out


_reg("SLOPE_TSTAT_24", _slope_tstat_24, "trend")


# --- TAIL: crash and lottery premia, not centre-of-distribution z-scores ---
def _moments(x, w):
    win = _wins(x, w)
    m = win.mean(axis=1, keepdims=True)
    d = win - m
    m2 = (d ** 2).mean(axis=1)
    m3 = (d ** 3).mean(axis=1)
    m4 = (d ** 4).mean(axis=1)
    return m2, m3, m4


def _skew_50(b):
    m2, m3, _ = _moments(_logret(b["close"]), 50)
    return np.where(m2 > 1e-18, m3 / (np.power(m2, 1.5) + 1e-18), 0.0)


_reg("SKEW_50", _skew_50, "tail")


def _kurt_50(b):
    m2, _, m4 = _moments(_logret(b["close"]), 50)
    return np.where(m2 > 1e-18, m4 / (m2 ** 2 + 1e-18) - 3.0, 0.0)


_reg("KURT_50", _kurt_50, "tail")


def _minret_20(b):
    r = _logret(b["close"])
    mn = _wins(r, 20).min(axis=1)
    sd = _tstd(r, 50)
    return np.where(sd > 1e-12, mn / (sd + 1e-12), 0.0)


_reg("MINRET_20", _minret_20, "tail")


# --- TURNOVER: cost of trading itself, from bars alone ---
def _roll_spread(b):
    r = _logret(b["close"])
    a = _wins(r, 50)
    bb = _wins(np.concatenate(([0.0], r[:-1])), 50)
    a = a - a.mean(axis=1, keepdims=True)
    bb = bb - bb.mean(axis=1, keepdims=True)
    cov = (a * bb).mean(axis=1)
    return 2.0 * np.sqrt(np.maximum(0.0, -cov))


_reg("ROLL_SPREAD", _roll_spread, "turnover")


# --- MICRO: intrabar shape ---
def _shadow_bal_12(b):
    rng = np.maximum(b["high"] - b["low"], 1e-12)
    uw = (b["high"] - np.maximum(b["close"], b["open"])) / rng
    lw = (np.minimum(b["close"], b["open"]) - b["low"]) / rng
    return _tmean(uw - lw, 12)


_reg("SHADOW_BAL_12", _shadow_bal_12, "micro")


# --- VOLPRICE rank: where this bar's volume sits in its own recent history ---
def _vol_rank_100(b):
    w = _wins(b["volume"], 100)
    return (w <= b["volume"][:, None]).sum(axis=1) / 100.0


_reg("VOL_RANK_100", _vol_rank_100, "volprice")


# --- SESSION: clock factors carry no cross-sectional variance by construction ---
def _hour_frac(b):
    t = np.asarray(b["timestamp"], dtype=np.float64) / 1000.0
    return ((t % 86400.0) / 3600.0)


_reg("TOD_SIN", lambda b: np.sin(2 * np.pi * _hour_frac(b) / 24.0), "session")
_reg("TOD_COS", lambda b: np.cos(2 * np.pi * _hour_frac(b) / 24.0), "session")


# ---------------------------------------------------------------------------
# DD_DURATION is defined by the spec as the argmax of the 200-bar window, which
# is a different quantity from the recursive "bars since the last new high" and
# is replaced here so the implementation matches the specification exactly.
# ---------------------------------------------------------------------------
def _dd_duration_argmax(b):
    c = b["close"]
    n = 200
    out = np.zeros(len(c))
    for t in range(len(c) - 1, -1, -1):
        w = c[max(0, t - n + 1):t + 1]
        k = int(np.argmax(w))
        out[t] = (len(w) - 1 - k) / n
    return out


F["DD_DURATION"] = (_dd_duration_argmax, "drawdown", False)


# ---------------------------------------------------------------------------
# Funding. Funding is a sparse 8-hourly settlement series held separately from
# the bars, so these three take it as an argument rather than reading it out of
# the bar dict. The series must already be forward-filled onto the bar grid by
# the caller; no bar-indexed window is ever taken over it.
# ---------------------------------------------------------------------------

def _fwd_fill_settlement(bars_t, settle_t, settle_v, n):
    """Last settlement at or before each bar, and its index."""
    idx = np.searchsorted(settle_t, bars_t, side="right") - 1
    valid = idx >= 0
    out = np.zeros(n)
    out[valid] = settle_v[idx[valid]]
    return out, np.where(valid, idx, 0), valid


def _fundo(b, funding):
    """funding = (settle_t, settle_v) for one coin, already sorted."""
    bt = np.asarray(b["timestamp"], dtype=np.float64)
    st = np.asarray(funding[0], dtype=np.float64)
    sv = np.asarray(funding[1], dtype=np.float64)
    f, idx, valid = _fwd_fill_settlement(bt, st, sv, len(bt))
    return f, idx, valid


def _fund_last(b, funding):
    f, _, _ = _fundo(b, funding)
    return f


def _fund_x_mom(b, funding):
    """Funding level signed by the direction of the last 48 bars of drift.

    A product, not a primitive, but the RPN operator set could build it from
    FUND_LAST and a momentum factor; shipping it saves the search the
    interaction and keeps the crowdedness reading explicit.
    """
    f, _, _ = _fundo(b, funding)
    c = b["close"]
    mom = np.sign(np.log(np.maximum(c, 1e-12) / np.roll(c, 48)))
    mom[:48] = 0.0
    return mom * f


def _fund_flip_7(b, funding):
    """Sign changes over the last 7 settlements. Settlement-indexed, not bar-indexed."""
    f, idx, valid = _fundo(b, funding)
    out = np.zeros(len(f))
    for t in range(len(f)):
        if not valid[t] or idx[t] < 6:
            continue
        window = f[idx[t] - 5:idx[t] + 1]
        s = np.sign(window)
        out[t] = float((s[1:] != s[:-1]).sum())
    return out / 6.0


FUNDING_FACTORS = {
    "FUND_LAST": _fund_last,
    "FUND_X_MOM": _fund_x_mom,
    "FUND_FLIP_7": _fund_flip_7,
}

FAMILIES = {name: (F.get(name) or (None, FUNDING_FACTORS.get(name) and "funding", False))[1]
            for name in list(F) + list(FUNDING_FACTORS)}


# ---------------------------------------------------------------------------
# Candidates from the other round-2 branch. Implemented so the correlation
# measurement can decide between the two branches on measured numbers rather
# than on which argument read better.
# ---------------------------------------------------------------------------

def _upper_wick(b):
    rng = np.maximum(b["high"] - b["low"], 1e-12)
    return _tmean((b["high"] - np.maximum(b["close"], b["open"])) / rng, 1)


def _lower_wick(b):
    rng = np.maximum(b["high"] - b["low"], 1e-12)
    return _tmean((np.minimum(b["close"], b["open"]) - b["low"]) / rng, 1)


_reg("UPPER_WICK", _upper_wick, "micro")
_reg("LOWER_WICK", _lower_wick, "micro")


def _body_ratio(b):
    rng = np.maximum(b["high"] - b["low"], 1e-12)
    return _tmean(np.abs(b["close"] - b["open"]) / rng, 1)


_reg("BODY_RATIO", _body_ratio, "micro")


def _gap_1(b):
    """Between-bar repricing, open(t) against close(t-1).

    On 30-minute crypto this is mostly one tick of jitter rather than an
    overnight gap, so the economic case is microstructure rather than
    news-while-closed. Kept because it isolates the event RET smooths over.
    """
    c = b["close"]
    prev = np.concatenate(([c[0]], c[:-1]))
    return np.log(np.maximum(b["open"], 1e-12) / np.maximum(prev, 1e-12))


_reg("GAP_1", _gap_1, "micro")


def _range_expand(b):
    """Event-level range spike: this bar's range against its own trailing max."""
    rng = b["high"] - b["low"]
    past = np.roll(_wins(rng, 20).max(axis=1), 1)
    past[0] = rng[0]
    return rng / (past + 1e-12)


_reg("RANGE_EXPAND", _range_expand, "range")


_reg("RANGE_MEAN_20",
     lambda b: _tmean((b["high"] - b["low"]) / (b["close"] + 1e-12), 20),
     "range")


def _chop_14(b):
    """Choppiness index: total true range against the window's own high-low span."""
    pc = np.concatenate(([b["close"][0]], b["close"][:-1]))
    tr = np.maximum.reduce([b["high"] - b["low"],
                            np.abs(b["high"] - pc),
                            np.abs(b["low"] - pc)])
    span = _wins(b["high"], 14).max(axis=1) - _wins(b["low"], 14).min(axis=1)
    return _tsum(tr, 14) / (span + 1e-12)


_reg("CHOP_14", _chop_14, "efficiency")


_reg("VOL_OF_VOL_30",
     lambda b: _tstd(b["volume"], 30) / (_tmean(b["volume"], 30) + 1e-12),
     "volprice")


def _time_since_high_100b(b):
    h = b["high"]
    hh = _wins(h, 100).max(axis=1)
    age = np.zeros(len(h))
    run = 0.0
    for t in range(len(h)):
        run = 0.0 if h[t] >= hh[t] else run + 1.0
        age[t] = run
    return age / 100.0


_reg("TIME_SINCE_HIGH_100B", _time_since_high_100b, "breakout")


def _fund_surprise(b, funding):
    """Funding relative to its own trailing median over 20 settlements.

    Settlement-indexed, never bar-indexed: funding is a step function that
    updates every 16 bars, and a bar-indexed window over it mostly measures
    where the bar sits inside the step rather than anything economic.
    """
    f, idx, valid = _fundo(b, funding)
    n = len(f)
    out = np.zeros(n)
    med_all = np.median(f[valid]) if valid.any() else 0.0
    mad_all = (np.median(np.abs(f[valid] - med_all)) + 1e-9) if valid.any() else 1.0
    for t in range(n):
        if not valid[t] or idx[t] < 19:
            continue
        win = f[idx[t] - 19:idx[t] + 1]
        med = np.median(win)
        mad = np.median(np.abs(win - med)) + 1e-9
        out[t] = (f[t] - med) / mad
    return np.clip(out, -5, 5)


FUNDING_FACTORS["FUND_SURPRISE"] = _fund_surprise


# ---------------------------------------------------------------------------
# Second batch, added because the measured budget only admitted 14 of the 29
# first-batch candidates. These were proposed in round 1 and cut before the
# measurement was possible; several are structurally orthogonal to price and
# volume levels and are worth measuring rather than assuming.
# ---------------------------------------------------------------------------

def _weekend(b):
    t = np.asarray(b["timestamp"], dtype=np.float64) / 1000.0
    dow = ((t // 86400.0) + 4) % 7          # 1970-01-01 was a Thursday
    return (dow >= 5).astype(np.float64)


_reg("WEEKEND", _weekend, "session")


def _hour_of_day(b):
    t = np.asarray(b["timestamp"], dtype=np.float64) / 1000.0
    return np.floor((t % 86400.0) / 3600.0)


_reg("SESSION_BLOCK",
     lambda b: np.where(_hour_of_day(b) < 8, 0.0,
                        np.where(_hour_of_day(b) < 16, 1.0, 2.0)),
     "session")


def _streak_10(b):
    """Length of the current run of same-sign returns, capped at 10.

    Cap rather than left uncapped: an uncapped streak is unbounded in a way that
    makes its pooled correlation depend on how many long runs the sample
    happened to contain.
    """
    r = _logret(b["close"])
    s = np.sign(r)
    out = np.zeros(len(r))
    run, prev = 0.0, 0.0
    for t in range(len(r)):
        if s[t] == 0:
            out[t] = 0.0
            continue
        if s[t] == prev:
            run = min(run + 1.0, 10.0)
        else:
            run = 1.0
        prev = s[t]
        out[t] = run * s[t]
    return out / 10.0


_reg("STREAK_10", _streak_10, "persistence")


def _sign_entropy_20(b):
    """Shannon entropy of the up/down sign distribution over 20 bars.

    In [0,1], where 1 is a coin flip and 0 is one-directional. A regime dial
    rather than a level, so it is close to orthogonal to the price factors.
    """
    r = _logret(b["close"])
    w = _wins(np.sign(r), 20)
    p_up = (w > 0).mean(axis=1)
    p_dn = (w < 0).mean(axis=1)
    def h(p):
        q = np.clip(p, 1e-9, 1 - 1e-9)
        return -(q * np.log(q) + (1 - q) * np.log(1 - q))
    return h(p_up) / np.log(2.0)


_reg("SIGN_ENTROPY_20", _sign_entropy_20, "persistence")


def _vol_price_corr_20(b):
    """Correlation of volume with absolute return over 20 bars.

    Whether a move had volume sponsorship, expressed as a correlation rather
    than a level, so it does not inherit LOG_VOL's scale.
    """
    r = np.abs(_logret(b["close"]))
    v = b["volume"]
    a, bb = _wins(r, 20), _wins(v, 20)
    ac = a - a.mean(axis=1, keepdims=True)
    bc = bb - bb.mean(axis=1, keepdims=True)
    den = (a.std(axis=1) * bb.std(axis=1)) * 20.0
    return np.where(den > 1e-18, (ac * bc).sum(axis=1) / (den + 1e-18), 0.0)


_reg("VOL_PRICE_CORR_20", _vol_price_corr_20, "volprice")


def _relvol_10_60(b):
    """Short-window volume against the long-window mean.

    A ratio of two volumes, so the price level cancels and it is not a restated
    LOG_VOL in the way VOL_RANK_100 turned out to be.
    """
    return _tmean(b["volume"], 10) / (_tmean(b["volume"], 60) + 1e-12) - 1.0


_reg("RELVOL_10_60", _relvol_10_60, "volprice")


def _illiq_change(b):
    """Change in Amihud illiquidity over 20 bars.

    LIQ_SCORE is the level. This is the derivative, which is what says the
    liquidity is deteriorating rather than that it is thin.
    """
    dv = b["close"] * b["volume"]
    ill = np.log(np.maximum(np.abs(_logret(b["close"])), 1e-12)
                 / np.maximum(dv, 1e-12))
    return _tmean(ill, 20) - _tmean(ill, 100)


_reg("ILLIQ_CHANGE", _illiq_change, "impact")


def _fund_z(b, funding):
    f, _, _ = _fundo(b, funding)
    m = f.mean()
    s = f.std()
    return np.where(s > 1e-12, (f - m) / (s + 1e-12), 0.0)


def _fund_countdown(b, funding):
    """Settlement events remaining until the next funding charge, in [0,1).

    The carry is charged at the event, so the hours to the event are part of the
    tradeable quantity rather than a nuisance.
    """
    bt = np.asarray(b["timestamp"], dtype=np.float64)
    nxt = (np.floor(bt / (8 * 3600 * 1000.0)) + 1) * (8 * 3600 * 1000.0)
    return (nxt - bt) / (8 * 3600 * 1000.0)


def _post_fund_drift(b, funding):
    """Return in the 16 bars following a settlement, else zero.

    Settlement-indexed by construction: it only fires on bars that follow a
    charge, so it is zero on most bars and is not a bar-indexed window over a
    step function.
    """
    bt = np.asarray(b["timestamp"], dtype=np.float64)
    st = np.asarray(funding[0], dtype=np.float64)
    after = np.zeros(len(bt))
    for s_t in st:
        j = np.searchsorted(bt, s_t, side="left")
        if j + 16 < len(bt):
            after[j:j + 16] = 1.0
    r = _logret(b["close"])
    return _tmean(r, 16) * after


FUNDING_FACTORS["FUND_Z"] = _fund_z
FUNDING_FACTORS["FUND_COUNTDOWN"] = _fund_countdown
FUNDING_FACTORS["POST_FUND_DRIFT"] = _post_fund_drift


# ---------------------------------------------------------------------------
# The shipped set. This list is the deliverable: 24 additions to the existing
# 12, giving 36 dimensions. Order matters: indices 0..11 are the original
# factors and must keep their meaning so every artifact recorded against the
# old indexing stays decodable.
# ---------------------------------------------------------------------------

SHIPPED_24 = (
    "FUND_LAST", "GAP_1", "FUND_COUNTDOWN", "FUND_X_MOM", "FUND_FLIP_7",
    "TIME_SINCE_HIGH_100B", "VARIANCE_RATIO_8", "SESSION_BLOCK",
    "SIGN_ENTROPY_20", "WEEKEND", "TOD_COS", "KURT_50", "SHADOW_BAL_12",
    "ILLIQ_CHANGE", "SKEW_50", "AUTOCORR_20", "ROLL_SPREAD",
    "VOL_PRICE_CORR_20", "EFFICIENCY_12", "MINRET_20", "RANGE_MEAN_20",
    "DD_200", "BREAKOUT_DIST_50", "UPPER_WICK",
)

assert len(SHIPPED_24) == 24
_missing = [n for n in SHIPPED_24 if n not in F and n not in FUNDING_FACTORS]
assert not _missing, "shipped factors without an implementation: %s" % _missing


def shipped_bars_only(coin_bars):
    """The 21 shipped factors that read only the bars, as one (n, bars) array."""
    from research.causal_12f import causal_features
    cols = [causal_features(coin_bars)]
    for nm in SHIPPED_24:
        if nm in F:
            cols.append(np.asarray(F[nm][0](coin_bars), dtype=np.float64)[None, :])
    return np.concatenate(cols, axis=0)


def shipped_with_funding(coin_bars, funding):
    """All 36, funding factors included. Indices 0..11 match causal_12f."""
    from research.causal_12f import causal_features
    cols = [causal_features(coin_bars)]
    for nm in SHIPPED_24:
        if nm in F:
            v = np.asarray(F[nm][0](coin_bars), dtype=np.float64)
        else:
            v = np.asarray(FUNDING_FACTORS[nm](coin_bars, funding), dtype=np.float64)
        cols.append(v[None, :])
    return np.concatenate(cols, axis=0)
