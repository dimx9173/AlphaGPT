#!/usr/bin/env python3
"""How much of a result is a funding assumption rather than an edge?

Run this before believing any Sharpe figure that came out of a constant funding
rate. The v2 accounting contract charged a flat +0.0005 at every 00/08/16 UTC
event: longs paid it, shorts received it, unconditionally, in every regime.

Actual Binance history over the same window says otherwise:

    modelled constant     +0.000500
    median real rate      +0.000031     (~16x smaller)
    share of real events that are NEGATIVE    ~30%

For a net-short book those three facts are decisive. The v3c lockbox is short
0.184 against long 0.009, so the constant pays it about 9.6%/year that the
venue never did.

Usage:
    .venv2/bin/python research/funding_sensitivity_28c.py \
        --artifact results/ga_28c_100gen_v3c_seed42.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research import ga_28c_30m_3y as G          # noqa: E402
from research.accounting_28c import (              # noqa: E402
    FUND_RATE,
    accounting_bar_returns,
    load_real_funding,
    metrics,
    scheduled_funding_rates,
)
from research.splits_28c import search_splits       # noqa: E402


def build_positions(formula, maps, train_end, n_bars):
    """Reconstruct the searched positions exactly as the GA did."""
    pos = {}
    for coin in G.COINS_28C:
        sig = G.formula_signal(formula, maps, coin)
        scale = float(np.std(sig[:train_end]))
        raw = np.tanh(sig / (scale + 1e-6)) if scale > 1e-8 else np.zeros_like(sig)
        p = 0.25 * G.smooth_causal(raw, 5)
        p = np.roll(p, 1)
        p[0] = 0.0
        pos[coin] = p
    return pos


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact",
                    default="results/ga_28c_100gen_v3c_seed42.json")
    args = ap.parse_args()

    art = json.loads((ROOT / args.artifact).read_text())
    formula = art["formula"]
    lo, hi = art["splits"]["lockbox"]

    common, maps, returns, _ = G.load_data()
    ts = np.asarray(common, dtype=np.int64)
    train_end = search_splits(len(common), common[0], common[-1])["train"][1]
    pos = build_positions(formula, maps, train_end, len(common))

    real = {c: load_real_funding(c, ts) for c in G.COINS_28C}

    scenarios = [
        ("modelled constant", "flat", FUND_RATE),
        ("zero funding", "flat", 0.0),
        ("sign flipped", "flat", -FUND_RATE),
        ("0.2x magnitude", "flat", 0.0001),
        ("5x magnitude", "flat", 0.0025),
        ("REAL Binance history", "series", None),
    ]

    rows = []
    print(f"{'scenario':22} {'sharpe':>8} {'return':>9} {'mdd':>8}")
    print("-" * 50)
    for label, kind, rate in scenarios:
        legs = []
        for c in G.COINS_28C:
            f = (real[c] if kind == "series"
                 else scheduled_funding_rates(ts, abs(rate)) * np.sign(rate))
            legs.append(accounting_bar_returns(pos[c], returns[c], G.FEE, f, G.LEV))
        net = np.mean(np.stack(legs), axis=0)
        m = metrics(net[lo:hi], ts[lo:hi])
        rows.append({"scenario": label, "kind": kind, "rate": rate, **m})
        print(f"{label:22} {m['sharpe']:+8.3f} {m['total_return']:+9.4f} {m['mdd']:8.4f}")

    by = {r["scenario"]: r for r in rows}
    mod, real_r, zero = by["modelled constant"], by["REAL Binance history"], by["zero funding"]
    print()
    print("=== what the assumption was doing ===")
    credit = mod["total_return"] - zero["total_return"]
    print(f"  funding credit            {credit:+.4f}")
    print(f"  trading alpha, no funding {zero['total_return']:+.4f} "
          f"(sharpe {zero['sharpe']:+.3f})")
    print(f"  sharpe under real history {real_r['sharpe']:+.3f} "
          f"vs modelled {mod['sharpe']:+.3f} "
          f"({real_r['sharpe'] - mod['sharpe']:+.3f})")
    if mod["sharpe"] > 0 > real_r["sharpe"]:
        print("  VERDICT: the sign of the result depends on the cost assumption.")

    out = ROOT / "results" / "funding_sensitivity_v3c.json"
    out.write_text(json.dumps(rows, indent=1))
    print(f"\nwritten: {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
