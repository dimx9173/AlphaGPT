#!/usr/bin/env python3
"""Split the reported Sharpe into signal, funding and fees.

The cost attribution showed that the diagnostic nulls report a HIGHER gross
Sharpe than the real data (iid1 +2.99, iidg +4.94, against real +0.81) at
roughly a third of the turnover. That is backwards for a search that is
supposed to find nothing on structureless data, and there are two very
different reasons it could happen:

  A. The search is genuinely better on those nulls. Implausible, but it is
     what the number says if taken at face value.
  B. The nulls produce SMOOTHER signals. A smoother position turns over less,
     so it pays fewer fees, and the fee saving is booked as Sharpe. The
     diagnostic nulls broke the properties that make a factor noisy (fat
     tails, volatility clustering, and in iidg the whole tail shape), so the
     rebuilt factors are smoother than the real ones and the position holds
     still. Nothing was forecast; the book just traded less.

B is testable without touching the search, because accounting_bar_returns
already decomposes exactly:

    net = position*return*lev  -  |d position|*fee*lev  -  position*funding*lev

Scoring the SAME recorded formulas three ways -- gross only, gross minus
funding, and the full net -- separates the three contributions. If the nulls'
advantage survives into the gross-only column, it is (A). If it appears only
once funding and fees are switched on and off, it is (B).

Usage:
    .venv2/bin/python research/decompose_sharpe_28c.py
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import research.ga_28c_30m_3y as GA
from research.accounting_28c import (accounting_bar_returns, load_real_funding,
                                     metrics, FUND_RATE)
from research.splits_28c import search_splits, split_indices

GRAMMAR = "reduced"

# fee, funding_enabled -- the three columns that isolate the contributions
VARIANTS = {
    "gross":         (0.0,     False),
    "gross_funded":  (0.0,     True),
    "net":           (GA.FEE,  True),
    "net_nofund":    (GA.FEE,  False),
}


def _legs(formula, maps, returns, lockbox, scale_end, common, funding_by_coin):
    """Rebuild the per-coin position path and its funding/turnover pieces."""
    start, end = lockbox
    ts = common[start:end]
    out = []
    for c in GA.COINS_28C:
        sig = GA.formula_signal(formula, maps, c, grammar=GRAMMAR)
        fit_scale = float(np.std(sig[:scale_end])) if scale_end > 0 else 0.0
        raw = np.tanh(sig / (fit_scale + 1e-6)) if fit_scale > 1e-8 else np.zeros_like(sig)
        p = 0.25 * GA.smooth_causal(raw, 5)
        p = np.roll(p, 1); p[0] = 0.0
        pos = p[start:end]
        r = returns[c][start:end]
        fnd = (funding_by_coin[c][start:end] if funding_by_coin is not None
               else np.zeros(len(pos)))
        out.append((pos, r, fnd, ts))
    return out


def score(formula, maps, returns, lockbox, scale_end, common, funding_by_coin):
    legs = _legs(formula, maps, returns, lockbox, scale_end, common,
                 funding_by_coin)
    res = {}
    for name, (fee, use_funding) in VARIANTS.items():
        nets = []
        for pos, r, fnd, ts in legs:
            f = fnd if use_funding else np.zeros_like(fnd)
            nets.append(accounting_bar_returns(pos, r, fee, f, GA.LEV))
        res[name] = metrics(np.mean(np.stack(nets), axis=0), ts)["sharpe"]
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", nargs="+", default=[
        "results/gram_reduced_runs:real_:none",
        "results/gram_reduced_runs:null_:iid",
        "results/nullmode:iid1:iid1",
        "results/nullmode:iidw:iidw",
        "results/nullmode:iidg:iidg"])
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results" / "sharpe_decomposition.json")
    args = ap.parse_args()

    cache, out = {}, {}
    print(f"{'batch':16s} {'n':>3} {'gross':>8} {'+funded':>9} {'net':>8} "
          f"{'net_nofund':>11} {'FEE effect':>11} {'FUND effect':>12}")
    print("  (gross = no fee, no funding | net_nofund = fee, no funding | net = both)")
    print("-" * 84)
    for spec in args.batches:
        rel, pre, null = spec.split(":")
        d = ROOT / rel
        if not d.exists():
            print(f"{pre:16s} (missing {rel})")
            continue
        if null not in cache:
            common, maps, returns, mask = GA.load_data(null=null)
            funding = {c: load_real_funding(c, np.asarray(common, dtype=np.int64))
                       for c in GA.COINS_28C}
            n = len(common)
            cache[null] = (common, maps, returns,
                           split_indices(n, common[0], common[-1])["lockbox"],
                           search_splits(n, common[0], common[-1])["train"][1],
                           funding)
        common, maps, returns, lockbox, scale_end, funding = cache[null]

        rows = []
        for f in sorted(d.glob(f"{pre}*.json")):
            a = json.loads(f.read_text())
            s = score(a["formula"], maps, returns, lockbox, scale_end, common,
                      funding)
            s["recorded"] = a["oos"]["portfolio_sharpe"]
            rows.append(s)
        if not rows:
            continue
        m = {k: st.median([r[k] for r in rows]) for k in VARIANTS}
        # The two effects are isolated by which switch differs, not by the
        # order the columns happen to be printed in. An earlier version of this
        # script subtracted net from net_nofund and called it the fee effect;
        # both of those columns have fees ON, so that difference is the FUNDING
        # effect and the fee effect went unreported entirely.
        #
        #   fee   = fees on vs off, both unfunded   (net_nofund - gross)
        #   fund  = funding on vs off, both at fee   (net - net_nofund)
        fee_eff = m["net_nofund"] - m["gross"]
        fund_eff = m["net"] - m["net_nofund"]
        out[null] = {k: m[k] for k in VARIANTS}
        out[null]["fee_effect"] = fee_eff
        out[null]["funding_effect"] = fund_eff
        out[null]["gross_minus_real_gross"] = m["gross"] - out.get("_real_gross", m["gross"])
        out[null]["n"] = len(rows)
        out[null]["recorded_median"] = st.median([r["recorded"] for r in rows])
        out[null]["gross_median"] = m["gross"]
        if null == "none":
            out["_real_gross"] = m["gross"]
        print(f"{null:16s} {len(rows):3d} {m['gross']:+8.3f} {m['gross_funded']:+9.3f} "
              f"{m['net']:+8.3f} {m['net_nofund']:+11.3f} {fee_eff:+11.3f} {fund_eff:+12.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
