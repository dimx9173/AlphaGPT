#!/usr/bin/env python3
"""28-coin point-in-time paper diagnostic for the latest 12-factor formula.

Offline only: no broker imports, API keys, network calls, or orders.
The universe is selected from trailing quote volume, with a warm-up period.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
os.environ["USE_ADVANCED"] = "1"
sys.path.insert(0, str(ROOT))

from model_core.backtest import MemeBacktest
from research.causal_12f import causal_signal
from research.universe_28c import COINS_28C, COMMON_THRESHOLDS_30M, MANIFEST_28C
from research.data_contract_28c import HISTORY_YEARS_COMMON_28
from research.splits_28c import split_indices
from research.accounting_28c import scheduled_funding_rates, position_from_signal, accounting_bar_returns, account_portfolio, compound_equity, max_drawdown, daily_sharpe, metrics, ACCOUNTING_VERSION, FUND_RATE

COINS = list(COINS_28C)
LEDGER_CAP = int(os.environ.get("PAPER_LEDGER_CAP", "500"))
FORMULA_FILE = ROOT / "results" / "train_12f_30m_28c_floor_best.json"
DEFAULT_OUT = ROOT / "results" / "paper_28c_pit_30m.json"
DATA_DIR = ROOT / "data" / "data_3y" / "30m"
BPY = 17520.0
LEV = 2.0
FEE = 0.0004
FUND = 0.0005
TARGET_N = 28
VOLUME_WINDOW = 30 * 48       # 30 days of 30m bars
VOL_WINDOW = 60
VOL_TARGET = 0.70
VOL_SCALE_MIN = 0.25
VOL_SCALE_MAX = 1.50
# Capital weights sum to at most 1.0. Leg PnL already includes LEV=2,
# so effective gross exposure is at most 2.0x; do not cap the already-leveraged
# legs at 2.0 again.
MAX_CAPITAL_WEIGHT = 1.0
# No per-coin thresholds were fitted for the 23 newly added symbols. Use one
# common, explicit transfer setting rather than silently inheriting five-coin
# tuned parameters.
COMMON_THRESHOLDS = (
    COMMON_THRESHOLDS_30M["long"], COMMON_THRESHOLDS_30M["short"],
    COMMON_THRESHOLDS_30M["cooldown_bars"], COMMON_THRESHOLDS_30M["stop_loss"],
)


def load_formula(path: Path) -> tuple[list[int], dict]:
    with path.open() as f:
        source = json.load(f)
    formula = source.get("formula")
    if not isinstance(formula, list) or len(formula) != 12:
        raise ValueError(f"invalid formula in {path}: {formula!r}")
    if set(source.get("coins", [])) != set(COINS):
        raise ValueError("paper source is not the unified 28-coin training artifact")
    if source.get("universe_mode") != "28c_common_transfer":
        raise ValueError("paper source universe_mode must be 28c_common_transfer")
    if abs(float(source.get("history_years", 0)) - HISTORY_YEARS_COMMON_28) > 0.01:
        raise ValueError("paper source history_years does not match common 28c contract")
    return [int(x) for x in formula], source


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_coin(coin: str, data_dir: Path, limit: int | None = None) -> dict:
    path = data_dir / f"{coin}.csv"
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if limit is not None:
        rows = rows[-limit:]
    if not rows:
        raise ValueError(f"empty data: {path}")
    return {
        "timestamp": [int(x["timestamp"]) for x in rows],
        "open": [float(x["open"]) for x in rows],
        "high": [float(x["high"]) for x in rows],
        "low": [float(x["low"]) for x in rows],
        "close": [float(x["close"]) for x in rows],
        "volume": [float(x["volume"]) for x in rows],
        "quote_volume": [float(x["quote_volume"]) for x in rows],
    }


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=np.float64)
    if len(values) < window:
        return out
    csum = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    out[window - 1:] = (csum[window:] - csum[:-window]) / window
    return out


def rolling_vol(close: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(close), np.nan, dtype=np.float64)
    if len(close) < window + 1:
        return out
    ret = np.zeros(len(close), dtype=np.float64)
    ret[1:] = np.diff(np.log(np.maximum(close, 1e-12)))
    c1 = np.concatenate(([0.0], np.cumsum(ret)))
    c2 = np.concatenate(([0.0], np.cumsum(ret * ret)))
    for t in range(window, len(close)):
        # Use returns ending at t-1; no current-bar information.
        mean = (c1[t] - c1[t - window]) / window
        var = (c2[t] - c2[t - window]) / window - mean * mean
        out[t] = math.sqrt(max(var, 0.0)) * math.sqrt(BPY)
    return out


def make_raw(bars: dict) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    n = len(bars["close"])
    raw = {k: torch.tensor([bars[k]], dtype=torch.float32) for k in ("open", "high", "low", "close", "volume")}
    raw["liquidity"] = torch.full((1, n), 1e7, dtype=torch.float32)
    raw["fdv"] = torch.full((1, n), 1e8, dtype=torch.float32)
    close = bars["close"]
    target = torch.tensor([[(close[i + 1] - close[i]) / close[i] if i < n - 1 else 0.0 for i in range(n)]], dtype=torch.float32)
    return raw, target


def signal_and_net(bars: dict, formula: list[int], coin: str) -> tuple[np.ndarray, np.ndarray, dict]:
    raw, target = make_raw(bars)
    # Causal feature/operator path: no full-sample median or global JUMP/ZSCORE.
    signal_np = causal_signal(formula, bars)
    factors = torch.tensor(signal_np, dtype=torch.float32).unsqueeze(0)
    lth, sth, cd, sl = COMMON_THRESHOLDS
    bt = MemeBacktest(
        venue="aster", leverage=LEV, short_enabled=True,
        fee_override=FEE, funding_override=FUND,
        long_th=lth, short_th=sth, cooldown_bars=cd,
        bars_per_year=BPY, stop_loss=sl,
    )
    sig = torch.sigmoid(factors)
    safe = (raw["liquidity"] > bt.min_liq).float()
    lp = (sig > bt.long_th).float() * safe
    sp = (sig < bt.short_th).float() * safe
    lp, sp = bt._apply_cooldown(lp, sp)
    lp, sp = bt._apply_stops(lp, sp, target)
    lp = lp.roll(1, dims=1); lp[:, 0] = 0
    sp = sp.roll(1, dims=1); sp[:, 0] = 0
    prev_lp = lp.roll(1, dims=1); prev_lp[:, 0] = 0
    prev_sp = sp.roll(1, dims=1); prev_sp[:, 0] = 0
    turnover = (lp - prev_lp).abs() + (sp - prev_sp).abs()
    impact = torch.clamp(bt.trade_size / (raw["liquidity"] + 1e-9), 0.0, 0.05)
    tx_cost = turnover * (bt.base_fee + impact)
    gross = (lp - sp) * target * bt.leverage
    funding = (lp - sp) * bt.default_funding_rate * bt.leverage
    net = gross - tx_cost * bt.leverage - funding
    position = lp - sp
    score, mean_cum = bt.evaluate(factors, raw, target)
    metrics = dict(bt.last_metrics)
    metrics["score"] = float(score.item()) if isinstance(score, torch.Tensor) else float(score)
    metrics["mean_cum"] = float(mean_cum)
    return net[0].numpy().astype(np.float64), position[0].numpy().astype(np.float64), metrics


def metrics(returns: np.ndarray, timestamps: np.ndarray) -> dict:
    """Use shared accounting metrics for consistency."""
    from research.accounting_28c import metrics as acct_metrics
    return acct_metrics(returns, timestamps)


def add_fold_stats(portfolio_net: np.ndarray, timestamps: np.ndarray, start: int, end: int) -> dict:
    returns = portfolio_net[start:end]
    ts_slice = timestamps[start:end]
    m = metrics(returns, ts_slice)
    return {"start": start, "end": end, "bars": end - start, **m}


def run(formula_file: Path, out_file: Path, data_dir: Path, limit: int | None = None) -> dict:
    formula, source = load_formula(formula_file)
    raw_data = {coin: load_coin(coin, data_dir, limit=None) for coin in COINS}
    # A few venue exports are offset by one 30m bar. Align on the actual
    # timestamp intersection instead of assuming row positions match.
    common_ts = set(raw_data[COINS[0]]["timestamp"])
    for coin in COINS[1:]:
        common_ts &= set(raw_data[coin]["timestamp"])
    timestamps = sorted(common_ts)
    if limit is not None:
        timestamps = timestamps[-limit:]
    if not timestamps:
        raise ValueError("no common timestamps across universe")
    data = {}
    for coin in COINS:
        index = {ts: i for i, ts in enumerate(raw_data[coin]["timestamp"])}
        data[coin] = {k: [raw_data[coin][k][index[ts]] for ts in timestamps]
                     for k in ("timestamp", "open", "high", "low", "close", "volume", "quote_volume")}
    n = len(timestamps)
    if limit is not None:
        # Smoke slices are diagnostic only; do not apply the full contract split
        # to a truncated sample.
        eval_start, eval_end = max(0, n - VOLUME_WINDOW), n
        contract_splits = {"lockbox": (eval_start, eval_end), "lockbox_start_ms": None}
    else:
        contract_splits = split_indices(n, timestamps[0], timestamps[-1])
        eval_start, eval_end = contract_splits["lockbox"]

    quote = np.vstack([np.asarray(data[c]["quote_volume"], dtype=np.float64) for c in COINS])
    close = np.vstack([np.asarray(data[c]["close"], dtype=np.float64) for c in COINS])
    avg_quote = np.vstack([rolling_mean(quote[i], VOLUME_WINDOW) for i in range(len(COINS))])
    vol = np.vstack([rolling_vol(close[i], VOL_WINDOW) for i in range(len(COINS))]).T
    selected = np.zeros((n, len(COINS)), dtype=bool)
    for t in range(n):
        valid = np.isfinite(avg_quote[:, t]) & (avg_quote[:, t] > 0)
        if not valid.any():
            continue
        # Only information available before bar t is used.
        idx = np.flatnonzero(valid)
        order = idx[np.argsort(-avg_quote[idx, t], kind="stable")]
        selected[t, order[:TARGET_N]] = True

    legs = {}
    positions = {}
    leg_metrics = {}
    for c in COINS:
        net, pos, m = signal_and_net(data[c], formula, c)
        legs[c] = net
        positions[c] = pos
        leg_metrics[c] = m
        print(f"leg {c}: score={m['score']:.3f} sharpe={m['sharpe']:.3f}", flush=True)

    scale = np.ones_like(vol)
    valid_vol = np.isfinite(vol) & (vol > 1e-8)
    scale[valid_vol] = np.clip(VOL_TARGET / vol[valid_vol], VOL_SCALE_MIN, VOL_SCALE_MAX)
    target_weights = selected.astype(np.float64) * (scale / len(COINS))
    capital_gross = target_weights.sum(axis=1)
    normalizer = np.ones(n, dtype=np.float64)
    too_high = capital_gross > MAX_CAPITAL_WEIGHT
    normalizer[too_high] = MAX_CAPITAL_WEIGHT / capital_gross[too_high]
    target_weights *= normalizer[:, None]
    # Universe/vol decisions made at bar t apply to the next executable bar.
    # Leg PnL is already execution-lagged inside MemeBacktest.
    weights = np.vstack((np.zeros((1, len(COINS))), target_weights[:-1]))
    portfolio_net = np.sum(np.vstack([weights[:, i] * legs[c] for i, c in enumerate(COINS)]), axis=0)
    # Keep the initial equity point plus one point per bar for ledger indexing.
    # portfolio_net holds per-bar RETURN RATES, not PnL amounts, so equity must
    # compound. Additive cumsum overstated growth hugely (it added 28 legs'
    # returns without reinvesting), which is what produced the +2578% paper
    # result and a Sharpe inconsistent with its own equity curve.
    _eq, _solv = compound_equity(portfolio_net)
    equity = _eq
    returns = portfolio_net

    ledger = []
    prev = {c: 0.0 for c in COINS}
    for t in range(eval_start, eval_end):
        for i, c in enumerate(COINS):
            want = 0.0
            if weights[t, i] > 0:
                p = positions[c][t]
                want = 1.0 if p > 0.5 else (-1.0 if p < -0.5 else 0.0)
            if want != prev[c]:
                action = "enter" if prev[c] == 0 else ("exit" if want == 0 else "flip")
                ledger.append({"bar": t, "ts_ms": timestamps[t], "coin": c, "action": action,
                               "from": int(prev[c]), "to": int(want),
                               "weight": round(float(weights[t, i]), 8),
                               "close": round(float(data[c]["close"][t]), 10),
                               "equity_x": round(float(equity[t + 1]), 8)})
                prev[c] = want

    active_counts = selected.sum(axis=1)
    fold_start = eval_start
    folds = []
    if eval_start < eval_end:
        edges = np.linspace(eval_start, eval_end, 5, dtype=int)
        for a, b in zip(edges[:-1], edges[1:]):
            # One-bar embargo between report segments. The formula is frozen;
            # this is an OOS evaluation, not per-fold parameter fitting.
            a2, b2 = a + 1, b
            if b2 > a2:
                folds.append(add_fold_stats(portfolio_net, timestamps, int(a2), int(b2)))
    active_returns = portfolio_net[eval_start:eval_end]
    _act_eq, _act_solv = compound_equity(active_returns)
    active_equity = _act_eq
    weight_sum = weights.sum(axis=1)
    weight_change = np.abs(np.diff(weights, axis=0)).sum(axis=1) if n > 1 else np.array([0.0])
    by = {}
    for row in ledger:
        by[row["coin"]] = by.get(row["coin"], 0) + 1
    ledger_cap = max(0, int(LEDGER_CAP))
    if ledger_cap <= 0 or len(ledger) <= ledger_cap:
        persisted_ledger = ledger
    else:
        # keep the opening and the most recent, so the sample still shows
        # the first entries and the current state, not just the tail
        head = ledger_cap // 2
        persisted_ledger = ledger[:head] + ledger[-(ledger_cap - head):]
    # A run that never opens a position is not a performance result. It means the
    # formula's signal does not clear this runner's fixed thresholds
    # (position_from_signal: sigmoid, long>0.85, short<0.15). Saying so prevents
    # a flat 1.0x / 0 trades from being read as "the strategy did not make
    # money" when the truth is "this signal model cannot express this formula".
    all_flat = bool(np.all(np.abs(portfolio_net) < 1e-15)) and not ledger
    if all_flat:
        print(
            "WARNING: no position was ever opened and net P&L is identically zero. "
            "This is NOT a performance result -- the formula's signal never "
            "crossed the paper runner's thresholds. Treat it as a framework "
            "incompatibility, not a strategy outcome.",
            flush=True,
        )
    result = {
        "status": "paper_only",
        "mode": "transfer_diagnostic",
        "live_adopted": False,
        "broker_imported": False,
        "network_access": False,
        "orders_attempted": 0,
        "state_mutated": False,
        "reset": True,
        "asof_utc": datetime.now(timezone.utc).isoformat(),
        "formula": formula,
        "decode": source.get("decode"),
        "source_result": str(formula_file),
        "source_sha256": sha256_file(formula_file),
        "no_position_ever_opened": all_flat,
        "flat_result_caveat": (
            "No position was opened and net P&L is identically zero. This is a "
            "framework incompatibility, NOT a performance outcome: the signal "
            "never crossed position_from_signal's thresholds. Do not cite the "
            "0.0 sharpe as evidence about the strategy."
        ) if all_flat else None,
        "source_score": source.get("score"),
        "source_worst_leg": source.get("worst_leg"),
        "formula_training_coins": source.get("coins"),
        "universe": COINS,
        "target_universe_size": TARGET_N,
        "available_symbols": len(COINS),
        "data_dir": str(data_dir),
        "history_years": source.get("history_years", HISTORY_YEARS_COMMON_28),
        "contract_version": "data-contract-28c-v1",
        "lockbox_boundary": (datetime.fromtimestamp(contract_splits["lockbox_start_ms"]/1000, tz=timezone.utc).isoformat() if contract_splits.get("lockbox_start_ms") is not None else None),
        "timestamp_start_ms": timestamps[0],
        "timestamp_end_ms": timestamps[-1],
        "evaluation_start_ms": timestamps[eval_start],
        "evaluation_end_ms": timestamps[eval_end - 1] if eval_end > eval_start else timestamps[eval_start],
        "bars": n,
        "bar_interval": "30m",
        "warmup_bars": VOLUME_WINDOW,
        "volume_window_bars": VOLUME_WINDOW,
        "selection": {"point_in_time": True, "uses_current_bar_for_next_bar_execution": True, "execution_lag_bars": 1, "average_active_after_warmup": float(active_counts[fold_start:].mean()) if fold_start < n else 0.0, "membership": "fixed 28-name local candidate set; volume rank is diagnostic"},
        "vol_scaling": {"window_bars": VOL_WINDOW, "target_annualized": VOL_TARGET, "min": VOL_SCALE_MIN, "max": VOL_SCALE_MAX, "max_capital_weight": MAX_CAPITAL_WEIGHT, "effective_max_gross_with_leg_leverage": LEV * MAX_CAPITAL_WEIGHT},
        "thresholds": {"long": COMMON_THRESHOLDS[0], "short": COMMON_THRESHOLDS[1], "cooldown_bars": COMMON_THRESHOLDS[2], "stop_loss": COMMON_THRESHOLDS[3], "scope": "common transfer default; not fitted per coin"},
        "manifest": MANIFEST_28C,
        "synthetic_liquidity": True,
        "execution_model": "signal_at_t_close_to_close_return_t_plus_1; no next-open claim",
        "funding_model": "fixed per-bar transfer assumption; not venue funding events",
        "accounting": ACCOUNTING_VERSION,
        "leverage": LEV, "fee": FEE, "funding": FUND, "bpy": BPY,
        "max_abs_bar_return": float(np.max(np.abs(active_returns))) if len(active_returns) else 0.0,
        "large_bar_return_count": int(np.sum(np.abs(active_returns) > 0.10)),
        "weight_summary": {"mean_capital_weight": float(weight_sum[fold_start:].mean()) if fold_start < n else 0.0, "max_capital_weight": float(weight_sum[fold_start:].max()) if fold_start < n else 0.0, "mean_bar_weight_turnover": float(weight_change[fold_start:].mean()) if fold_start < n else 0.0},
        "stats": metrics(active_returns, np.array(timestamps[eval_start:eval_end])),
        "segments": {"kind": "frozen_formula_lockbox_only", "fit_per_fold": False, "fold_boundary_exclusion_bars": 1, "selection_embargo_bars": 200, "range": (eval_start, eval_end), "folds": folds, "fold_count": len(folds)},
        "events": len(ledger),
        "trades": sum(x["action"] in ("exit", "flip") for x in ledger),
        "by": by,
        # The full ledger is ~6MB and is identical on every hourly rerun
        # (the formula is frozen), so persisting all of it each hour grew
        # results/paper_runs to 584MB of duplicates. Counts stay exact;
        # only the retained sample is capped.
        "ledger_retained": len(persisted_ledger),
        "ledger_truncated": len(ledger) > len(persisted_ledger),
        "per_leg": {c: {"score": leg_metrics[c]["score"], "engine_sharpe": leg_metrics[c]["sharpe"], "engine_mdd": leg_metrics[c]["max_dd"], "net_sum": float(np.sum(legs[c]))} for c in COINS},
        "ledger": persisted_ledger, "equity": equity.tolist(),
    }
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w") as f:
        json.dump(result, f, indent=1)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--formula-file", type=Path, default=FORMULA_FILE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    result = run(args.formula_file, args.out, args.data_dir, args.limit)
    print(json.dumps({k: result[k] for k in ("status", "mode", "live_adopted", "universe", "bars", "stats", "segments", "trades", "events", "large_bar_return_count")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
