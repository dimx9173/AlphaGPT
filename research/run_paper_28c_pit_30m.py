#!/usr/bin/env python3
"""28-coin portfolio-level paper trade matching GA evaluation exactly.

This is a true paper trade: every computation step mirrors the GA search
(ga_28c_30m_3y.py) — same features, same formula evaluation, same accounting,
same funding, same splits, same cost model. The ONLY difference is that
capital is simulated rather than real.

If the GA accepts a formula, this paper trade should produce comparable
portfolio-level metrics. If it does not, the discrepancy is a framework bug
to fix, not a strategy property to explain away.
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

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from research.causal_12f import (
    causal_features, evaluate_formula, FEATURE_NAMES, constant_features,
    assert_features_vary
)
from research.universe_28c import COINS_28C
from research.data_contract_28c import HISTORY_YEARS_COMMON_28
from research.splits_28c import split_indices
from research.accounting_28c import (
    scheduled_funding_rates, accounting_bar_returns, compound_equity,
    max_drawdown, daily_sharpe, metrics as acct_metrics,
    ACCOUNTING_VERSION, ACCOUNTING_VERSION_REAL, load_real_funding,
    real_funding_coverage
)
from research.formula_grammar import is_valid
from research.ga_28c_30m_3y import load_data, shape_reward, softplus, FEATURE_CACHE_VERSION

FORMULA_FILE = ROOT / "results" / "train_12f_30m_28c_floor_best.json"
DEFAULT_OUT = ROOT / "results" / "paper_28c_pit_30m.json"
DATA_DIR = ROOT / "data" / "data_3y" / "30m"

# GA evaluation parameters (must match ga_28c_30m_3y.py)
LEG_FLOOR = 0.0
GAP_TAU = 1.0
BREADTH_TARGET = 24/28


def load_formula(path: Path) -> tuple[list[int], dict]:
    with path.open() as f:
        source = json.load(f)
    formula = source.get("formula")
    if not isinstance(formula, list) or len(formula) != 12:
        raise ValueError(f"invalid formula in {path}: {formula!r}")
    if set(source.get("coins", [])) != set(COINS_28C):
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


def evaluate_portfolio(formula: list[int], maps: dict, returns: dict,
                       funding_mask: np.ndarray, funding_by_coin: dict,
                       common_ts: list, start: int, end: int, scale_end: int, bars: int,
                       reward_version: str = 'v1') -> dict:
    """Mirror ga_28c_30m_3y.py evaluate() exactly."""

    n = end - start
    leg_sharpes = {}
    leg_returns = {}
    leg_mdds = {}
    leg_turnovers = {}
    leg_ics = {}
    ts_slice = np.array(common_ts[start:end])

    for c in COINS_28C:
        sig = evaluate_formula(formula, maps[c][:, start:end])
        # Leg returns with funding (same as GA)
        leg_ret = accounting_bar_returns(
            sig, returns[c][start:end],
            0.0004,
            funding_mask[start:end] if funding_by_coin is None else funding_by_coin[c][start:end],
            2.0
        )
        leg_returns[c] = leg_ret
        eq, _ = compound_equity(leg_ret)
        leg_sharpes[c] = daily_sharpe(leg_ret, ts_slice)[0]
        leg_mdds[c] = max_drawdown(eq)
        # Turnover proxy: mean absolute change in signal
        leg_turnovers[c] = float(np.mean(np.abs(np.diff(sig)))) if len(sig) > 1 else 0.0
        # IC proxy: correlation of signal with next-bar return
        if len(sig) > 2 and np.std(sig[:-1]) > 1e-12 and np.std(returns[c][start+1:end]) > 1e-12:
            leg_ics[c] = float(np.corrcoef(sig[:-1], returns[c][start+1:end])[0,1])
        else:
            leg_ics[c] = 0.0

    # Portfolio: equal-weight of legs (GA uses equal-weight in evaluation)
    port_ret = np.zeros(n)
    for c in COINS_28C:
        port_ret += leg_returns[c]
    port_ret /= len(COINS_28C)

    # Portfolio metrics
    port_eq, _ = compound_equity(port_ret)
    port_sharpe = daily_sharpe(port_ret, ts_slice)[0]
    port_mdd = max_drawdown(port_eq)

    # Aggregate diagnostics
    min_leg = min(leg_sharpes.values()) if leg_sharpes else -1.0
    mean_ic = float(np.mean(list(leg_ics.values()))) if leg_ics else 0.0
    mean_turn = float(np.mean(list(leg_turnovers.values()))) if leg_turnovers else 0.0
    breadth = sum(1 for v in leg_sharpes.values() if v > 0) / len(leg_sharpes) if leg_sharpes else 0.0

    # GA's reward (same function, same version)
    reward = shape_reward(
        port_sharpe, None, min_leg, mean_ic, mean_turn,
        breadth=breadth, version=reward_version
    )

    return {
        'portfolio_sharpe': port_sharpe,
        'portfolio_mdd': port_mdd,
        'portfolio_return': float(port_eq[-1] - 1.0),
        'leg_sharpes': leg_sharpes,
        'leg_returns': {c: float(np.sum(v)) for c, v in leg_returns.items()},
        'leg_mdds': leg_mdds,
        'leg_turnovers': leg_turnovers,
        'leg_ics': leg_ics,
        'min_leg': min_leg,
        'mean_ic': mean_ic,
        'mean_turn': mean_turn,
        'breadth': breadth,
        'reward': reward,
        'n_bars': n,
    }


def run(formula_file: Path, out_file: Path, data_dir: Path,
        funding_mode: str = 'real', reward_version: str = 'v1') -> dict:
    formula, source = load_formula(formula_file)

    # Load data exactly as GA does (same cache, same factors, same splits)
    common, maps, returns, funding_mask = load_data()

    # Funding: use real Binance funding if requested (per coin)
    funding_by_coin = None
    if funding_mode == 'real':
        funding_by_coin = {c: load_real_funding(c, np.array(common)) for c in COINS_28C}

    # Contract splits (same as GA)
    n = len(common)
    contract_splits = split_indices(n, common[0], common[-1])

    # Evaluate each fold exactly as GA does
    train_start, train_end = contract_splits['train']
    val_start, val_end = contract_splits['validation']
    lock_start, lock_end = contract_splits['lockbox']

    train = evaluate_portfolio(formula, maps, returns, funding_mask, funding_by_coin, common,
                               train_start, train_end, train_end, n, reward_version)
    validation = evaluate_portfolio(formula, maps, returns, funding_mask, funding_by_coin, common,
                                    val_start, val_end, val_end, n, reward_version)
    lockbox = evaluate_portfolio(formula, maps, returns, funding_mask, funding_by_coin, common,
                                 lock_start, lock_end, lock_end, n, reward_version)

    # Gate (same as GA)
    from research.regime_gate_28c import evaluate_regime_gate
    from research.ga_28c_30m_3y import FEE, LEV, smooth_causal
    from research.universe_28c import ACCEPTANCE_GATE_28C as gates
    # Build positions for each coin (same tanh smoothing as GA)
    positions = {}
    for c in COINS_28C:
        sig = evaluate_formula(formula, maps[c])
        fs = float(np.std(sig[:len(common)])) if len(common) > 0 else 0.0
        raw_pos = np.tanh(sig / (fs + 1e-6)) if fs > 1e-8 else np.zeros_like(sig)
        pp = 0.25 * smooth_causal(raw_pos, 5) if 'smooth_causal' in dir() else 0.25 * np.roll(np.convolve(raw_pos, np.ones(5)/5, mode='same'), 1)
        pp = np.roll(pp, 1); pp[0] = 0
        positions[c] = pp
    gate_result = evaluate_regime_gate(
        positions, returns, list(COINS_28C), funding_mask, contract_splits,
        np.array(common, dtype=np.int64), FEE, LEV,
        contract_splits['lockbox'],
        gates['min_positive_coins'], gates['max_oos_mdd'], int(common[0]),
        funding_by_coin=funding_by_coin
    )

    # Full lockbox equity curve for output
    lock_start, lock_end = contract_splits['lockbox']
    port_ret = np.zeros(lock_end - lock_start)
    for c in COINS_28C:
        sig = evaluate_formula(formula, maps[c][:, lock_start:lock_end])
        leg_ret = accounting_bar_returns(
            sig, returns[c][lock_start:lock_end],
            0.0004,
            funding_mask[lock_start:lock_end] if funding_by_coin is None else funding_by_coin[c][lock_start:lock_end],
            2.0
        )
        port_ret += leg_ret
    port_ret /= len(COINS_28C)
    port_eq, _ = compound_equity(port_ret)

    # Timestamp range for lockbox
    lock_ts = [common[i] for i in range(lock_start, lock_end)]

    result = {
        "status": "paper_only",
        "mode": "portfolio_paper_trade_ga_matched",
        "live_adopted": False,
        "broker_imported": False,
        "network_access": False,
        "orders_attempted": 0,
        "state_mutated": False,
        "asof_utc": datetime.now(timezone.utc).isoformat(),
        "formula": formula,
        "decode": source.get("decode"),
        "source_result": str(formula_file),
        "source_sha256": sha256_file(formula_file),
        "funding_mode": funding_mode,
        "funding_source": funding_mode if funding_mode != 'real' else 'binance_public_history',
        "accounting_version": ACCOUNTING_VERSION_REAL if funding_mode == 'real' else ACCOUNTING_VERSION,
        "reward_version": reward_version,
        "universe": COINS_28C,
        "history_years": HISTORY_YEARS_COMMON_28,
        "contract_version": "data-contract-28c-v1",
        "lockbox_boundary_utc": datetime.fromtimestamp(common[lock_start]/1000, tz=timezone.utc).isoformat(),
        "bars_total": n,
        "bars_lockbox": lock_end - lock_start,
        "bar_interval": "30m",
        "train": train,
        "validation": validation,
        "lockbox": lockbox,
        "gate": gate_result,
        "equity_curve_lockbox": port_eq.tolist(),
        "timestamps_lockbox_ms": lock_ts,
    }

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w") as f:
        json.dump(result, f, indent=1)
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="Portfolio paper trade matching GA evaluation")
    ap.add_argument("--formula-file", type=Path, default=FORMULA_FILE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--funding", choices=('constant','real'), default='real',
                    help="Funding model: constant +0.0005 or real Binance history")
    ap.add_argument("--reward-version", choices=('v1','v2'), default='v1',
                    help="Reward version for diagnostics (does not affect paper P&L)")
    args = ap.parse_args()

    result = run(args.formula_file, args.out, args.data_dir,
                 funding_mode=args.funding, reward_version=args.reward_version)

    # Print summary
    print(json.dumps({
        "status": result["status"],
        "mode": result["mode"],
        "live_adopted": result["live_adopted"],
        "funding": result["funding_mode"],
        "reward_version": result["reward_version"],
        "gate_verdict": result["gate"].get("verdict"),
        "gate_criteria": result["gate"].get("criteria"),
        "train_sharpe": round(result["train"]["portfolio_sharpe"], 3),
        "val_sharpe": round(result["validation"]["portfolio_sharpe"], 3),
        "lockbox_sharpe": round(result["lockbox"]["portfolio_sharpe"], 3),
        "lockbox_mdd": round(result["lockbox"]["portfolio_mdd"], 4),
        "lockbox_breadth": f"{int(result['lockbox']['breadth']*28)}/28",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
