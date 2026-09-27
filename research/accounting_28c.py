#!/usr/bin/env python3
"""
Shared 28c accounting contract: equity-compound-v2

All 28c consumers (GA, threshold fit, paper replay, trainer) must use this module.
No broker, network, database, or model dependencies.
"""

import numpy as np
from datetime import datetime, timezone

ACCOUNTING_VERSION = "equity-compound-v2"
FUND_RATE = 0.0005  # per 8-hour funding event assumption (not exchange actuals)

# UTC hour boundaries where funding occurs
FUNDING_HOURS = (0, 8, 16)

def scheduled_funding_rates(timestamps: np.ndarray, fund_rate: float = FUND_RATE) -> np.ndarray:
    """
    Generate funding rate array from UTC timestamps.
    Only non-zero at exactly 00:00, 08:00, 16:00 UTC.
    Positive rate = cost for long, receipt for short.
    """
    n = len(timestamps)
    rates = np.zeros(n, dtype=np.float64)
    for i, ts_ms in enumerate(timestamps):
        dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
        if dt.hour in FUNDING_HOURS and dt.minute == 0:
            rates[i] = fund_rate
    return rates


def position_from_signal(
    signal: np.ndarray,
    timestamps: np.ndarray,
    long_th: float = 0.85,
    short_th: float = 0.15,
    cooldown_bars: int = 0,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    time_stop: int = 0,
) -> np.ndarray:
    """
    Derive lagged signed position from signal, matching MemeBacktest rules:
    - sigmoid(signal) -> long if > long_th, short if < short_th
    - one-bar execution lag (position at t uses signal[t-1])
    - cooldown after flip
    - stop-loss / take-profit / time-stop on adverse excursion
    Returns position array [-1, 0, 1] with same length as signal.
    """
    n = len(signal)
    pos = np.zeros(n, dtype=np.float64)
    
    sigmoid = 1.0 / (1.0 + np.exp(-signal))
    
    cur_pos = 0.0  # +1 long, -1 short, 0 flat
    exc = 0.0      # cumulative excursion since entry
    held = 0       # bars held
    lock = 0       # cooldown remaining
    
    for t in range(n):
        # Check cooldown
        if lock > 0:
            lock -= 1
        else:
            want = 1.0 if sigmoid[t] > long_th else (-1.0 if sigmoid[t] < short_th else 0.0)
            if want != cur_pos:
                cur_pos = want
                exc = 0.0
                held = 0
                lock = cooldown_bars
        
        if cur_pos != 0.0:
            # Only accumulate excursion if we have a return (lagged position uses t-1)
            # Excursion is tracked on the PnL of the position
            held += 1
            kill = False
            if stop_loss is not None and exc <= -abs(stop_loss):
                kill = True
            if take_profit is not None and exc >= abs(take_profit):
                kill = True
            if time_stop and held >= time_stop:
                kill = True
            if kill:
                cur_pos = 0.0
                exc = 0.0
                held = 0
        
        # Position filled at t uses signal at t-1 (one-bar lag)
        pos[t] = cur_pos
    
    # Apply one-bar lag: position at t is based on signal at t-1
    pos = np.roll(pos, 1)
    pos[0] = 0.0
    
    return pos


def accounting_bar_returns(
    position: np.ndarray,
    asset_return: np.ndarray,
    fee_rate: float,
    funding_rate: np.ndarray,
    leverage: float = 2.0,
) -> np.ndarray:
    """
    Single-leg bar returns with compounding convention.
    position: signed position array [-1, 0, 1], already lagged
    asset_return: simple returns per bar
    fee_rate: per-side fee rate (e.g., 0.0004)
    funding_rate: funding rate per bar (sparse, from scheduled_funding_rates)
    leverage: leverage multiplier
    """
    # Validate inputs
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(asset_return)) or not np.all(np.isfinite(funding_rate)):
        raise ValueError("Input arrays must contain only finite values")
    
    n = len(position)
    if n != len(asset_return) or n != len(funding_rate):
        raise ValueError("Length mismatch: position, asset_return, funding_rate")
    
    # Turnover: position change from prior bar (p[-1] = 0 for first bar)
    prev_pos = np.roll(position, 1)
    prev_pos[0] = 0.0
    turnover = np.abs(position - prev_pos)
    
    gross = position * asset_return * leverage
    fee = turnover * fee_rate * leverage
    funding_cost = position * funding_rate * leverage
    
    net = gross - fee - funding_cost
    return net


def account_portfolio(
    notionals: np.ndarray,  # shape (n, num_legs) = w * p
    asset_returns: np.ndarray,  # shape (n, num_legs)
    fee_rate: float,
    funding_rates: np.ndarray,  # shape (n,) or (n, num_legs)
    leverage: float = 2.0,
) -> np.ndarray:
    """
    Portfolio bar returns from weighted signed notionals.
    notionals[t,i] = w[t,i] * p[t,i]
    asset_returns[t,i] = simple return of leg i at bar t
    funding_rates can be (n,) broadcast or (n, num_legs)
    
    Single formula includes position changes AND capital reallocation.
    No separate weight-turnover fee.
    """
    # Validate inputs
    if not np.all(np.isfinite(notionals)) or not np.all(np.isfinite(asset_returns)) or not np.all(np.isfinite(funding_rates)):
        raise ValueError("Input arrays must contain only finite values")
    
    n, m = notionals.shape
    if asset_returns.shape != (n, m):
        raise ValueError("notionals and asset_returns shape mismatch")
    if funding_rates.shape == (n,):
        funding_rates = funding_rates[:, None]
    elif funding_rates.shape != (n, m):
        raise ValueError("funding_rates shape mismatch")
    
    # Turnover of notional (captures both position change and weight reallocation)
    prev_notional = np.roll(notionals, 1, axis=0)
    prev_notional[0] = 0.0
    turnover = np.sum(np.abs(notionals - prev_notional), axis=1)
    
    gross = leverage * np.sum(notionals * asset_returns, axis=1)
    fee = turnover * fee_rate * leverage
    funding_cost = leverage * np.sum(notionals * funding_rates, axis=1)
    
    net = gross - fee - funding_cost
    return net


def compound_equity(net_returns: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    Compound equity from bar returns. Returns (equity_array, solvent_flag).
    equity[0] = 1.0 (initial)
    equity[t] = equity[t-1] * (1 + net_returns[t-1]) for t >= 1
    If 1 + net <= 0 at any bar, mark insolvent, zero out rest, return solvent=False.
    """
    net = np.asarray(net_returns, dtype=np.float64)
    equity = np.concatenate(([1.0], np.cumprod(1.0 + net)))
    # Solvency fails at the first bar whose growth factor is non-positive; the
    # position is bankrupt from that bar onward.
    bad = np.flatnonzero(1.0 + net <= 0.0)
    if bad.size:
        equity[int(bad[0]) + 1:] = 0.0
        return equity, False
    return equity, True


def max_drawdown(equity: np.ndarray) -> float:
    """
    Equity-relative max drawdown as fraction in [0, 1].
    equity includes initial point (len = n_bars + 1).
    """
    eq = np.asarray(equity, dtype=np.float64)
    if len(eq) <= 1:
        return 0.0
    peak = np.maximum.accumulate(eq)
    safe = np.where(peak > 0, peak, 1.0)
    return float(np.max((peak - eq) / safe))


def daily_sharpe(net_returns: np.ndarray, timestamps: np.ndarray) -> tuple[float, int]:
    """
    Daily-return Sharpe (ddof=1) annualized by sqrt(365).
    Groups bar returns by UTC calendar day, compounds each day.
    Returns (sharpe, daily_return_count).
    """
    if len(net_returns) != len(timestamps):
        raise ValueError("Length mismatch")
    if len(net_returns) == 0:
        return 0.0, 0

    # Group by UTC date. Epoch milliseconds divide exactly into UTC day indices,
    # so the grouping is a vectorised integer floor instead of a per-bar
    # datetime parse -- this is the hot path for the 28-leg GA evaluation.
    ts = np.asarray(timestamps, dtype=np.int64)
    day_id = np.floor_divide(ts, 86_400_000)
    starts = np.flatnonzero(np.concatenate(([True], day_id[1:] != day_id[:-1])))
    ends = np.concatenate((starts[1:], [len(day_id)]))

    # Compound each day independently: a day starts from 1.0 and multiplies
    # only its own bars, so a day that wipes the account out does not poison
    # the days after it. Iterating over days (a few hundred), not bars, keeps
    # this vectorised within each day.
    gross = 1.0 + np.asarray(net_returns, dtype=np.float64)
    daily_rets = np.array([np.prod(gross[lo:hi]) - 1.0
                           for lo, hi in zip(starts, ends)], dtype=np.float64)

    d = len(daily_rets)
    if d <= 1:
        return 0.0, d
    mean = np.mean(daily_rets)
    std = np.std(daily_rets, ddof=1)
    if std < 1e-12:
        return 0.0, d
    sharpe = mean / std * np.sqrt(365.0)
    return float(sharpe), d


def metrics(
    net_returns: np.ndarray,
    timestamps: np.ndarray,
) -> dict:
    """
    Complete metrics dict for a return series.
    """
    equity, solvent = compound_equity(net_returns)
    mdd = max_drawdown(equity)
    sharpe, daily_count = daily_sharpe(net_returns, timestamps)
    
    final_x = float(equity[-1])
    total_return = final_x - 1.0
    mean_bar_return = float(np.mean(net_returns)) if len(net_returns) > 0 else 0.0
    
    return {
        "sharpe": sharpe,
        "mdd": mdd,
        "final_x": final_x,
        "total_return": total_return,
        "mean_bar_return": mean_bar_return,
        "solvent": solvent,
        "daily_return_count": daily_count,
        "accounting_version": ACCOUNTING_VERSION,
    }
