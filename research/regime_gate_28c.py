"""Regime-neutral acceptance gate for 28c strategies.

Two separate reference series, deliberately NOT the same object:

* ``market_regime_proxy`` - 1x gross equal-weight market, no leverage, no fee,
  no funding. Used ONLY to label a segment's regime, so the label measures what
  the market did rather than what it cost to hold. (Codex B7: reading the label
  from a 2x cost-inclusive benchmark makes funding drag dominate and forces every
  fold to "down".)
* ``passive_benchmark_net`` - a passive equal-weight hold at the strategy's own
  position cap, carrying the same cap, leverage, fee and scheduled funding as
  the strategy leg. Used for the excess comparison, so exposure and cost match
  by construction rather than by convention. (Codex B8: a benchmark 4x larger
  than the strategy's achievable exposure makes A1 nearly free.)

Subtracting two Sharpes is never done. Excess is the Sharpe of the difference
series.
"""
from __future__ import annotations

import numpy as np

from research.accounting_28c import (
    FUND_RATE,
    accounting_bar_returns,
    compound_equity,
    daily_sharpe,
    max_drawdown,
)

# Must track the position cap applied in ga_28c_30m_3y.evaluate():
#     p = 0.25 * smooth_causal(np.tanh(...), 5)
# so max |p| = POSITION_CAP and max gross notional = POSITION_CAP * LEVERAGE.
POSITION_CAP = 0.25
BARS_PER_DAY = 48


# --------------------------------------------------------------------------
# reference series
# --------------------------------------------------------------------------

def market_regime_proxy(returns_map, coins, start, end):
    """1x gross equal-weight market over [start, end). No cost, no leverage.

    Equal weight is rebalanced every bar, matching the strategy portfolio's own
    aggregation in ``evaluate()`` (mean of per-leg net returns).
    """
    if end <= start:
        return np.zeros(0, dtype=np.float64)
    legs = [np.asarray(returns_map[c][start:end], dtype=np.float64) for c in coins]
    if not legs:
        raise ValueError("market_regime_proxy requires at least one coin")
    return np.mean(np.stack(legs), axis=0)


def passive_benchmark_net(returns_map, coins, start, end, funding_mask,
                           fee_rate, leverage):
    """Passive equal-weight long at the strategy's position cap.

    Runs through the same ``accounting_bar_returns`` as a strategy leg, so the
    cap, leverage, fee and scheduled funding are identical by construction.
    A constant +1 signal that saturates the cap therefore has excess ~0.
    """
    if end <= start:
        return np.zeros(0, dtype=np.float64)
    n = end - start
    pos = np.full(n, POSITION_CAP, dtype=np.float64)
    fnd = np.asarray(funding_mask[start:end], dtype=np.float64)
    legs = [
        accounting_bar_returns(pos, np.asarray(returns_map[c][start:end], dtype=np.float64),
                               fee_rate, fnd, leverage)
        for c in coins
    ]
    return np.mean(np.stack(legs), axis=0)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def regime_label(proxy_net, timestamps):
    """Label a segment up/down from the gross market proxy only.

    Deterministic at exactly 0.0 (-> 'down'). No tolerance band: a band would
    mislabel a flat segment as 'up'.
    """
    proxy_net = np.asarray(proxy_net, dtype=np.float64)
    if len(proxy_net) == 0:
        return "down", 0.0, 0.0, 0.0
    equity, _ = compound_equity(proxy_net)
    compounded = float(equity[-1] - 1.0)
    days = len(proxy_net) / BARS_PER_DAY
    annual = float((1.0 + compounded) ** (365.0 / days) - 1.0) if days > 0 else 0.0
    sharpe, _ = daily_sharpe(proxy_net, timestamps)
    return ("up" if compounded > 0.0 else "down"), annual, sharpe, max_drawdown(equity)


def excess_metrics(strategy_net, bench_net, timestamps):
    """Sharpe of the active return series. Never a difference of Sharpes."""
    strategy_net = np.asarray(strategy_net, dtype=np.float64)
    bench_net = np.asarray(bench_net, dtype=np.float64)
    if len(strategy_net) != len(bench_net):
        raise ValueError("strategy_net and bench_net must have equal length")
    active = strategy_net - bench_net
    sharpe, daily_count = daily_sharpe(active, timestamps)
    s_sharpe, _ = daily_sharpe(strategy_net, timestamps)
    b_sharpe, _ = daily_sharpe(bench_net, timestamps)
    return {
        "excess_sharpe": float(sharpe),
        "strategy_sharpe": float(s_sharpe),
        "benchmark_sharpe": float(b_sharpe),
        "active_daily_count": int(daily_count),
    }


# --------------------------------------------------------------------------
# folds
# --------------------------------------------------------------------------

def midnight_aligned(bar_index, series_start_ts):
    """True when ``bar_index`` lands on 00:00 UTC."""
    import datetime as _dt
    ts = series_start_ts + int(bar_index) * 1_800_000
    d = _dt.datetime.fromtimestamp(ts / 1000.0, tz=_dt.timezone.utc)
    return d.hour == 0 and d.minute == 0


def _snap(lo, hi, series_start_ts):
    """Snap [lo, hi) inward to UTC-midnight-aligned bounds."""
    while lo < hi and not midnight_aligned(lo, series_start_ts):
        lo += 1
    while hi > lo and not midnight_aligned(hi, series_start_ts):
        hi -= 1
    return lo, hi


def walk_forward_folds_28c(splits, series_start_ts):
    """Four folds inside train+validation, two per contiguous block.

    The 400-bar embargo between train and validation may never fall inside a
    fold, so the region is cut per block rather than as one span. Bounds come
    from the split dict, never from a computed ``end``, so the lockbox is
    unreachable by construction.
    """
    tr = tuple(splits["train"])
    va = tuple(splits["validation"])
    bounds = [tr[0], tr[1], va[0], va[1]]
    folds = []
    for i in (0, 2):
        lo, hi = bounds[i], bounds[i + 1]
        mid = lo + (hi - lo) // 2
        folds.append(_snap(lo, mid, series_start_ts))
        folds.append(_snap(mid, hi, series_start_ts))
    folds = [(lo, hi) for lo, hi in folds if hi - lo > 2 * BARS_PER_DAY]
    assert max(hi for _, hi in folds) <= va[1], "fold escaped past validation[1]"
    return folds


def coverage_report(folds, splits):
    """Assert the coverage invariant in its satisfiable form."""
    tr = set(range(*splits["train"]))
    va = set(range(*splits["validation"]))
    embargo = set(range(splits["train"][1], splits["validation"][0]))
    covered = set()
    overlap = False
    for lo, hi in folds:
        block = set(range(lo, hi))
        overlap = overlap or bool(covered & block)
        covered |= block
    return {
        "overlap": overlap,
        "embargo_leak": sorted(embargo & covered),
        "uncovered_train": len(tr - covered),
        "uncovered_validation": len(va - covered),
        "max_hi": max(hi for _, hi in folds),
        "validation_end": splits["validation"][1],
    }


# --------------------------------------------------------------------------
# gate
# --------------------------------------------------------------------------

CRITERIA = ("A1", "A2p", "A2pp", "A3", "A4", "A5", "A6")


def statistical_power(lockbox_bars: int, bar_minutes: int = 30,
                      observed_sharpe: float | None = None) -> dict:
    """How much of a Sharpe estimate can this lockbox actually resolve?

    A gate that requires Sharpe >= 1.0 is only meaningful if the confidence
    interval around the estimate is narrow enough to distinguish 1.0 from 0.
    With a 30-minute bar and roughly independent daily returns,

        t = mean / (sd / sqrt(n_days)),  Sharpe_annual = t / sqrt(n_years)

    This reports the 95% CI half-width of the Sharpe estimate and the
    t-statistic actually achieved. It is a property of the sample, not of the
    strategy, and it is the reason a Sharpe threshold cannot be treated as
    evidence on this data.

    Measured serial correlation of the v3c daily returns is approximately zero
    (sum of lags 1..3 = -0.026), so no autocorrelation correction is applied;
    the interval is already close to the best this sample can give.
    """
    n_days = max(1, int(lockbox_bars * bar_minutes / (60 * 24)))
    years = n_days / 365.25
    # variance of the mean of n approximately independent daily returns
    half_width = 1.96 * np.sqrt((1.0 + 0.5 ** 2) / years)
    achieved_t = None
    if observed_sharpe is not None:
        achieved_t = observed_sharpe * np.sqrt(years)
    return {
        "lockbox_bars": int(lockbox_bars),
        "oos_days": n_days,
        "oos_years": round(years, 4),
        "sharpe_ci95_half_width": round(float(half_width), 4),
        "achieved_t_statistic": None if achieved_t is None else round(float(achieved_t), 4),
        "t_required_for_95pct": 1.96,
        "can_resolve_sharpe_1_0": bool(half_width < 1.0),
        "note": ("A Sharpe threshold is not testable when this half-width is "
                 "larger than the threshold. The criteria below are screening "
                 "rules, not statistical evidence."),
    }


def evaluate_regime_gate(positions, returns_map, coins, funding_mask, splits,
                         timestamps, fee_rate, leverage, lockbox_range,
                         min_positive_coins, max_oos_mdd, series_start_ts):
    """Per-fold table plus the A1..A6 criteria and one verdict.

    ``positions`` maps coin -> full-series position array. Net returns are
    computed ONCE over the full aligned series and sliced per fold, so no
    phantom entry cost is charged at a fold boundary (Codex B5: the existing
    evaluate() path resets prev_pos[0]=0 and charges turnover*fee*leverage,
    measured at 0.0008 per fold head).

    ``lockbox_range`` is used only for A3 and A4, the pre-existing lockbox
    gates. It never participates in fold construction or in A1/A2p/A2pp/A5/A6.
    """
    n = len(timestamps)
    # funding arrives as a 0/1 event mask; accounting needs the rate.
    fnd_full = np.asarray(funding_mask, dtype=np.float64) * FUND_RATE
    coin_list = list(coins)

    # Full-series net per leg, computed once.
    leg_net = {
        c: accounting_bar_returns(
            np.asarray(positions[c], dtype=np.float64),
            np.asarray(returns_map[c], dtype=np.float64),
            fee_rate, fnd_full, leverage)
        for c in coin_list
    }
    strat_full = np.mean(np.stack([leg_net[c] for c in coin_list]), axis=0)
    bench_full = passive_benchmark_net(returns_map, coin_list, 0, n,
                                       fnd_full, fee_rate, leverage)

    folds = walk_forward_folds_28c(splits, series_start_ts)
    table = []
    for idx, (lo, hi) in enumerate(folds, 1):
        ts = timestamps[lo:hi]
        proxy = market_regime_proxy(returns_map, coin_list, lo, hi)
        label, bench_ann, bench_sh, bench_mdd = regime_label(proxy, ts)
        s_net = strat_full[lo:hi]
        b_net = bench_full[lo:hi]
        ex = excess_metrics(s_net, b_net, ts)
        s_eq, s_sol = compound_equity(s_net)
        pos_legs = sum(
            1 for c in coin_list
            if daily_sharpe(leg_net[c][lo:hi], ts)[0] > 0.0)
        gross = float(np.mean([np.mean(np.abs(np.asarray(positions[c])[lo:hi]))
                               for c in coin_list])) * leverage
        table.append({
            "fold": idx,
            "start": int(lo), "end": int(hi),
            "bars": int(hi - lo), "days": round((hi - lo) / BARS_PER_DAY, 1),
            "regime": label,
            "benchmark_annual": bench_ann,
            "benchmark_sharpe": bench_sh,
            "benchmark_max_dd": bench_mdd,
            "strategy_sharpe": ex["strategy_sharpe"],
            "excess_sharpe": ex["excess_sharpe"],
            "strategy_mdd": max_drawdown(s_eq),
            "positive_legs": int(pos_legs),
            "solvent": bool(s_sol),
            "mean_gross_notional": gross,
        })

    lo, hi = lockbox_range
    ts = timestamps[lo:hi]
    oos_eq, oos_sol = compound_equity(strat_full[lo:hi])
    oos_mdd = max_drawdown(oos_eq)
    oos_leg_sh = [daily_sharpe(leg_net[c][lo:hi], ts)[0] for c in coin_list]
    oos_positive = int(sum(1 for v in oos_leg_sh if v > 0.0))

    a1_folds = [r["fold"] for r in table if r["excess_sharpe"] > 0.0]
    a2p_folds = [r["fold"] for r in table if r["strategy_sharpe"] > 0.0]
    a2pp = float(np.mean([r["excess_sharpe"] for r in table])) if table else 0.0
    up = sum(1 for r in table if r["regime"] == "up")
    down = sum(1 for r in table if r["regime"] == "down")

    coverage = coverage_report(folds, splits)
    criteria = {
        "A1": {"folds_with_positive_excess": len(a1_folds), "fold_ids": a1_folds},
        "A2p": {"folds_with_positive_absolute": len(a2p_folds), "fold_ids": a2p_folds},
        "A2pp": {"mean_excess_sharpe": a2pp},
        "A3": {"oos_mdd": oos_mdd, "limit": max_oos_mdd},
        "A4": {"oos_positive_legs": oos_positive, "required": min_positive_coins},
        "A5": {"all_folds_solvent": all(r["solvent"] for r in table),
               "oos_solvent": bool(oos_sol)},
        "A6": {"up_folds": up, "down_folds": down},
    }

    # A6 is evaluated first: insufficient evidence must not be misreported as
    # either a pass or a failure of the skill criteria.
    if up < 1 or down < 1:
        verdict = "insufficient_regime_coverage"
        reason = f"regime coverage up={up} down={down}; need >=1 of each"
    else:
        checks = {
            "A1": len(a1_folds) >= 3,
            "A2p": len(a2p_folds) >= 3,
            "A2pp": a2pp > 0.0,
            "A3": oos_mdd < max_oos_mdd,
            "A4": oos_positive >= min_positive_coins,
            "A5": all(r["solvent"] for r in table) and bool(oos_sol),
        }
        failed = sorted(k for k, v in checks.items() if not v)
        verdict = "pass" if not failed else "fail"
        reason = "all criteria satisfied" if not failed else "failed: " + ",".join(failed)

    # Statistical power is reported next to, not folded into, the verdict.
    # A candidate can satisfy every screening rule and still be statistically
    # indistinguishable from a random strategy, because the lockbox is too short
    # to resolve the difference. Reporting only "pass" would overstate what the
    # data can support.
    # Lockbox Sharpe of the strategy itself, on the same equity-compound-v2
    # accounting the criteria use.
    lb_sharpe, lb_days = daily_sharpe(strat_full[lo:hi], timestamps[lo:hi])
    power = statistical_power(int(hi - lo), observed_sharpe=lb_sharpe)
    power["oos_daily_observations"] = int(lb_days)
    evidence = (
        "screening_only_not_statistically_resolved"
        if not power["can_resolve_sharpe_1_0"] else "resolvable"
    )

    return {
        "folds": table,
        "criteria": criteria,
        "verdict": verdict,
        "verdict_reason": reason,
        "statistical_power": power,
        "evidence_class": evidence,
        "evidence_caveat": (
            "This verdict is a SCREENING result, not proof of edge. The OOS "
            "window yields a Sharpe confidence interval of +/-"
            f"{power['sharpe_ci95_half_width']:.2f}, which is wider than the "
            "gate's own Sharpe threshold. A candidate marked 'pass' here may "
            "still be statistically indistinguishable from random. Do not "
            "describe such a result as evidence that the strategy works."
        ) if evidence == "screening_only_not_statistically_resolved" else None,
        "fold_coverage": coverage,
        "lockbox": {"start": int(lo), "end": int(hi), "max_dd": oos_mdd,
                    "positive_legs": oos_positive, "solvent": bool(oos_sol),
                    "per_leg_excess": {
                        c: round(daily_sharpe(leg_net[c][lo:hi], ts)[0]
                                 - daily_sharpe(
                                     accounting_bar_returns(
                                         np.full(hi - lo, POSITION_CAP),
                                         np.asarray(returns_map[c][lo:hi]),
                                         fee_rate, fnd_full[lo:hi], leverage), ts)[0], 4)
                        for c in coin_list}},
    }
