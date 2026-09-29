#!/usr/bin/env python3
"""Is the reported Sharpe a fee effect rather than an edge effect?

The diagnostic nulls produce LOWER turnover than the real data (median 0.0019
to 0.0023 against 0.0059) and their reported Sharpe is HIGHER. At 5bp per
side and 17520 half-hourly bars, those turnovers are 3.4%/yr of drag against
10.2%/yr on the real data, a 6.9%/yr difference that is plausibly the whole
gap.

That matters because it inverts the reading. If Sharpe here is mostly a
function of how much the position moves, then the search is not being
rewarded for forecasting; it is being rewarded for holding still, and a
null that produces smooth factors is a null it scores well on for free.

The test re-scores each recorded formula twice, once at the real fee and once
at zero, holding the signal fixed. The difference between the two is the
cost of that particular signal, isolated from its gross performance. If the
gross ordering and the net ordering disagree, the headline number is
measuring turnover.

Usage:
    .venv2/bin/python research/attribute_sharpe_to_costs_28c.py --runs results/gram_reduced_runs
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
from research.splits_28c import search_splits, split_indices

# The grammar travels with the call. A reduced formula evaluated with the
# module global left at 'full' is routed to the full evaluator and computes a
# different function; evaluate() now takes the grammar explicitly and refuses
# a formula that does not belong to it.
GRAMMAR = "reduced"

from research.accounting_28c import load_real_funding


def score_all(runs_dir: Path, prefix: str, fees=None, null="none") -> list[dict]:
    # Read the live constants rather than hardcoding them. FEE is 0.0004 and
    # LEV is 2.0 in this tree; an earlier version of this script assumed 0.0005
    # and 1.0 and produced a "net" column that disagreed with the recorded
    # artifacts, which is the fastest way to lose trust in a cost analysis.
    if fees is None:
        fees = (GA.FEE, 0.0)
    real_fee, zero_fee = fees
    # A formula found on a null must be re-scored on THAT null. Re-scoring it
    # on the real contract measures a different function's costs, and the net
    # column then disagrees with the recorded artifact instead of reproducing
    # it, which is the check that catches exactly this mistake.
    common, maps, returns, funding_mask = GA.load_data(null=null)
    funding_by_coin = {c: load_real_funding(c, np.asarray(common, dtype=np.int64))
                       for c in GA.COINS_28C}
    n = len(common)
    contract = split_indices(n, common[0], common[-1])
    scale_end = search_splits(n, common[0], common[-1])["train"][1]
    lb = contract["lockbox"]

    rows = []
    for f in sorted(runs_dir.glob(f"{prefix}*.json")):
        d = json.loads(f.read_text())
        formula = d.get("formula")
        if not formula:
            continue
        row = {"file": f.name, "recorded_sharpe": d["oos"]["portfolio_sharpe"],
               "recorded_turnover": d["oos"].get("turnover")}
        for fee in (real_fee, zero_fee):
            GA.FEE = fee
            r = GA.evaluate(formula, maps, returns, lb[0], lb[1], funding_mask,
                            scale_end, common, funding_by_coin=funding_by_coin,
                            grammar=GRAMMAR)
            key = "gross" if fee == zero_fee else "net"
            row[f"{key}_sharpe"] = r["portfolio_sharpe"]
            row["turnover"] = r["turnover"]
        rows.append(row)
    GA.FEE = real_fee
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=ROOT / "results" / "gram_reduced_runs")
    ap.add_argument("--prefixes", nargs="+", default=["real_", "null_"])
    ap.add_argument("--null", default="none",
                    help="which dataset these formulas were FOUND on; a null "
                         "formula must be re-scored on its own null")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results" / "sharpe_cost_attribution.json")
    args = ap.parse_args()

    # The re-scored net column must reproduce each recorded artifact. It
    # agrees to about 1e-7 rather than bit-exactly, because re-running the
    # accumulation in a different order changes the last few ulps and a
    # 1.85 Sharpe is then visibly different in the 6th decimal. A tolerance
    # of 1e-9 flagged that as a MISMATCH, which is why two batches were
    # reported as unreproducible when they were simply reproduced to float
    # precision. 1e-5 is still far tighter than any difference that matters.
    REPRODUCE_ATOL = 1e-5

    out = {"_constants": {"fee": GA.FEE, "lev": GA.LEV, "reproduce_atol": REPRODUCE_ATOL}}
    print(f"fee={GA.FEE} lev={GA.LEV} (read live, not hardcoded)")
    for pre in args.prefixes:
        rows = score_all(args.runs, pre, null=args.null)
        if not rows:
            continue
        out[pre] = rows
        net = [r["net_sharpe"] for r in rows]
        gross = [r["gross_sharpe"] for r in rows]
        turn = [r["turnover"] for r in rows]
        rec = [json.loads(f.read_text())["oos"]["portfolio_sharpe"]
               for f in sorted(args.runs.glob(f"{pre}*.json"))]
        ok = all(abs(a - b) <= REPRODUCE_ATOL for a, b in zip(net, rec))
        # Spearman between turnover and Sharpe, computed by rank
        def _rank(v):
            s = sorted(range(len(v)), key=lambda i: v[i]); rk = [0] * len(v)
            for pos, i in enumerate(s):
                rk[i] = pos + 1
            return rk
        def _spearman(a, b):
            ra, rb = _rank(a), _rank(b); nn = len(a)
            d2 = sum((ra[i] - rb[i]) ** 2 for i in range(nn))
            return 1 - 6 * d2 / (nn * (nn * nn - 1))
        print(f"{pre:8s} n={len(rows):3d}  net median={st.median(net):+7.3f}  "
              f"gross median={st.median(gross):+7.3f}  "
              f"turnover median={st.median(turn):.5f}  "
              f"rho(turnover, net sharpe)={_spearman(turn, net):+.3f}")
        print(f"         cost of the median signal: "
              f"{st.median([g - t for g, t in zip(gross, net)]):+.3f} Sharpe")
        print(f"         reproduces recorded artifacts: {ok}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
