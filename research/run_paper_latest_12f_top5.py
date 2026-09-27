#!/usr/bin/env python3
"""Offline paper profile for the latest AlphaGPT 12-factor strategy.

This never imports a broker, reads API keys, or places orders. It resets only
its own output artifact and uses the same 30m/Top5 universe as training.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
os.environ["USE_ADVANCED"] = "1"
sys.path.insert(0, str(ROOT))

import torch

from model_core.backtest import MemeBacktest
from model_core.factors import FeatureEngineer
from model_core.vm import StackVM

DEFAULT_FORMULA_FILE = ROOT / "results" / "train_12f_30m_floor_best.json"
DEFAULT_OUT = ROOT / "results" / "paper_latest_12f_top5_30m.json"
COINS = ["ETC", "TRX", "ATOM", "APT", "KAS"]
WEIGHTS = {coin: 0.2 for coin in COINS}
# Thresholds and accounting match train_12f_30m.py's Top5 reward setup.
THRESHOLDS = {
    "ETC": (0.88, 0.12, 18, None),
    "TRX": (0.85, 0.12, 6, 0.05),
    "ATOM": (0.85, 0.15, 6, 0.05),
    "APT": (0.88, 0.12, 18, None),
    "KAS": (0.88, 0.12, 6, None),
}
BPY = 17520.0
LEV = 2.0
FEE = 0.0004
FUND = 0.0005


def load_formula(path: Path) -> tuple[list[int], dict]:
    with path.open() as f:
        result = json.load(f)
    formula = result.get("formula")
    if not isinstance(formula, list) or len(formula) != 12:
        raise ValueError(f"invalid 12-factor formula in {path}: {formula!r}")
    return [int(x) for x in formula], result


def load_bars(coin: str, data_dir: Path, limit: int | None = None) -> list[dict[str, float]]:
    path = data_dir / f"{coin}.csv"
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if limit is not None:
        rows = rows[-limit:]
    if not rows:
        raise ValueError(f"no bars in {path}")
    return [
        {key: float(row[key]) for key in ("open", "high", "low", "close", "volume")}
        for row in rows
    ]


def make_raw(bars: list[dict[str, float]]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    n = len(bars)
    raw = {
        key: torch.tensor([[float(row[key]) for row in bars]], dtype=torch.float32)
        for key in ("open", "high", "low", "close", "volume")
    }
    raw["liquidity"] = torch.full((1, n), 1e7, dtype=torch.float32)
    raw["fdv"] = torch.full((1, n), 1e8, dtype=torch.float32)
    close = raw["close"][0].tolist()
    target = torch.tensor(
        [[(close[i + 1] - close[i]) / close[i] if i < n - 1 else 0.0 for i in range(n)]],
        dtype=torch.float32,
    )
    return raw, target


def leg_series(
    bars: list[dict[str, float]],
    formula: list[int],
    coin: str,
) -> tuple[list[float], list[float], dict]:
    raw, target = make_raw(bars)
    features = FeatureEngineer.compute_features(raw, use_advanced=True)
    if features.shape[1] != 12:
        raise ValueError(f"{coin}: expected 12 features, got {tuple(features.shape)}")
    factors = StackVM(use_advanced=True).execute(formula, features)
    if factors is None:
        raise ValueError(f"{coin}: formula execution returned None")
    lth, sth, cd, sl = THRESHOLDS[coin]
    bt = MemeBacktest(
        venue="aster",
        leverage=LEV,
        short_enabled=True,
        fee_override=FEE,
        funding_override=FUND,
        long_th=lth,
        short_th=sth,
        cooldown_bars=cd,
        bars_per_year=BPY,
        stop_loss=sl,
    )
    # Keep the same signal/cooldown/stop/execution-lag path as MemeBacktest.evaluate.
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
    pos = lp - sp
    # Ask the engine for its score/metrics as an audit of the same setup.
    score, mean_cum = bt.evaluate(factors, raw, target)
    score_value = float(score.item()) if isinstance(score, torch.Tensor) else float(score)
    metrics = dict(bt.last_metrics)
    metrics.update({"score": score_value, "mean_cum": float(mean_cum)})
    return net[0].tolist(), pos[0].tolist(), metrics


def stats(returns: list[float], equity: list[float]) -> dict[str, float]:
    n = len(returns)
    mean = sum(returns) / n if n else 0.0
    var = sum((x - mean) ** 2 for x in returns) / max(n - 1, 1) if n else 0.0
    sharpe = mean / math.sqrt(var) * math.sqrt(BPY) if var > 1e-18 else 0.0
    peak = equity[0] if equity else 1.0
    mdd = 0.0
    for value in equity:
        peak = max(peak, value)
        mdd = max(mdd, (peak - value) / peak if peak else 0.0)
    return {
        "sharpe": sharpe,
        "mdd": mdd,
        "total_return": equity[-1] - 1.0 if equity else 0.0,
        "final_x": equity[-1] if equity else 1.0,
        "mean_bar_return": mean,
    }


def run(formula_file: Path, out_file: Path, data_dir: Path, limit: int | None = None) -> dict:
    formula, source = load_formula(formula_file)
    bars = {coin: load_bars(coin, data_dir, limit=limit) for coin in COINS}
    n = min(len(rows) for rows in bars.values())
    bars = {coin: rows[:n] for coin, rows in bars.items()}
    legs = {}
    positions = {}
    leg_metrics = {}
    for coin in COINS:
        net, pos, metrics = leg_series(bars[coin], formula, coin)
        if len(net) != n or len(pos) != n:
            raise AssertionError(f"{coin}: series length mismatch")
        legs[coin] = net
        positions[coin] = pos
        leg_metrics[coin] = metrics
        print(f"leg {coin}: score={metrics['score']:.4f} sharpe={metrics['sharpe']:.4f} mdd={metrics['max_dd']:.4f}", flush=True)

    portfolio_net = [sum(WEIGHTS[c] * legs[c][t] for c in COINS) for t in range(n)]
    # Match the existing paper2 convention: cumulative additive P&L, not
    # per-bar compounding. This avoids turning a single data outlier into a
    # fictional millions-fold equity curve.
    equity = [1.0]
    for value in portfolio_net:
        equity.append(equity[-1] + value)
    # Record executed position transitions for an auditable paper ledger.
    ledger = []
    current = {coin: 0.0 for coin in COINS}
    for t in range(n):
        for coin in COINS:
            want = positions[coin][t]
            want = 1.0 if want > 0.5 else (-1.0 if want < -0.5 else 0.0)
            if want != current[coin]:
                action = "enter" if current[coin] == 0 else ("exit" if want == 0 else "flip")
                ledger.append({
                    "bar": t,
                    "coin": coin,
                    "action": action,
                    "from": int(current[coin]),
                    "to": int(want),
                    "fill": round(float(bars[coin][t]["close"]), 10),
                    "equity_x": round(equity[t + 1], 8),
                })
                current[coin] = want
    by = {}
    for row in ledger:
        by[row["coin"]] = by.get(row["coin"], 0) + 1
    result = {
        "status": "paper_only",
        "live_adopted": False,
        "reset": True,
        "formula": formula,
        "decode": source.get("decode"),
        "source_result": str(formula_file),
        "source_score": source.get("score"),
        "source_worst_leg": source.get("worst_leg"),
        "coins": COINS,
        "weights": WEIGHTS,
        "thresholds": {k: list(v) for k, v in THRESHOLDS.items()},
        "data_dir": str(data_dir),
        "bars": n,
        "bar_interval": "30m",
        "bpy": BPY,
        "leverage": LEV,
        "fee": FEE,
        "funding": FUND,
        "accounting": "additive_cumulative_pnl_paper2_compatible",
        "max_abs_bar_return": max(abs(x) for x in portfolio_net),
        "large_bar_return_count": sum(abs(x) > 0.10 for x in portfolio_net),
        "stats": stats([(equity[i + 1] - equity[i]) / equity[i] if equity[i] else 0.0 for i in range(n)], equity),
        "trades": len([x for x in ledger if x["action"] in ("exit", "flip")]),
        "events": len(ledger),
        "by": by,
        "per_leg": {},
        "ledger": ledger,
        "equity": equity,
    }
    # Populate per-leg metrics without retaining large tensors.
    # Re-run only the cheap summary values already available from the engine output
    # by deriving them from the net series; engine scores are recorded separately.
    for coin in COINS:
        leg_eq = [1.0]
        for x in legs[coin]:
            leg_eq.append(leg_eq[-1] + x)
        s = stats([(leg_eq[i + 1] - leg_eq[i]) / leg_eq[i] if leg_eq[i] else 0.0 for i in range(n)], leg_eq)
        result["per_leg"][coin] = {
            "score": leg_metrics[coin]["score"],
            "engine_sharpe": leg_metrics[coin]["sharpe"],
            "engine_mdd": leg_metrics[coin]["max_dd"],
            "sharpe": s["sharpe"],
            "mdd": s["mdd"],
            "net_sum": sum(legs[coin]),
        }
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w") as f:
        json.dump(result, f, indent=1)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formula-file", type=Path, default=DEFAULT_FORMULA_FILE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "data_1y" / "30m")
    parser.add_argument("--limit", type=int, default=None, help="use only the last N bars (smoke)")
    args = parser.parse_args()
    result = run(args.formula_file, args.out, args.data_dir, args.limit)
    print(json.dumps({
        "status": result["status"],
        "out": str(args.out),
        "coins": result["coins"],
        "bars": result["bars"],
        "stats": result["stats"],
        "trades": result["trades"],
        "events": result["events"],
        "by": result["by"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
