"""Cost attribution for a 28-coin backtest, computed AFTER the fact.

Why this is a separate module
-----------------------------
The first implementation put the attribution inside `evaluate()` in
`research/ga_28c_30m_3y.py`. That was neutral in behaviour, and it was still the
wrong shape: `evaluate()` is where `reward` is built, so the fields sat one line
away from the only value selection depends on, and keeping them apart relied on a
scan of the source to prove they never touched. Six review rounds found one
bypass after another in that scan, which is the expected outcome for any check
whose job is to prove a negative about code paths.

The fix is structural rather than adversarial. `evaluate()` is now byte-identical
to its pre-PRP-4 form and knows nothing about attribution. This module re-runs the
SAME bar returns through the SAME accounting function and derives the drags
afterwards. The reward has already been computed and fixed by then, so there is
no path from these numbers back into selection, and no scan is needed to say so.

Convention
----------
Sharpe is nonlinear, so "fee drag" has no canonical meaning until an order is
fixed. The order here is fee first, then funding, matching the
`gross - fee - funding_cost` order inside `accounting_bar_returns`. The two drags
then sum back to gross minus net by construction.

    gross_sharpe - fee_drag - funding_drag == net_sharpe
"""
from __future__ import annotations

import numpy as np

from .accounting_28c import accounting_bar_returns, metrics


def decompose_leg(position, asset_return, fee_rate, funding_rate, leverage):
    """Per-leg decomposition. Returns (net, gross, fee, funding) arrays.

    The net returned here is produced by the accounting itself, not recomputed, so
    the gross / fee / funding parts are exactly the ones that produced it.
    """
    comp: dict = {}
    net = accounting_bar_returns(position, asset_return, fee_rate, funding_rate,
                                 leverage, out=comp)
    return net, comp["gross"], comp["fee"], comp["funding_cost"]


def attribute(net_series, gross_series, fee_series, funding_series, timestamps):
    """Chained drags for a portfolio-level return series.

    Returns a dict with gross_sharpe, fee_drag, funding_drag and net_sharpe.
    """
    net_sharpe = metrics(net_series, timestamps)["sharpe"]
    gross_sharpe = metrics(gross_series, timestamps)["sharpe"]
    mid_sharpe = metrics(gross_series - fee_series, timestamps)["sharpe"]
    fee_drag = gross_sharpe - mid_sharpe
    funding_drag = mid_sharpe - net_sharpe
    return {
        "gross_sharpe": float(gross_sharpe),
        "fee_drag": float(fee_drag),
        "funding_drag": float(funding_drag),
        "net_sharpe": float(net_sharpe),
    }


def attribute_legs(legs, timestamps):
    """Aggregate per-leg decompositions into a portfolio attribution.

    `legs` is a sequence of (position, asset_return, fee_rate, funding_rate,
    leverage) tuples, one per coin. The portfolio series is the mean of the legs,
    matching how evaluate() builds `pnet`.
    """
    nets, grosses, fees, fundings = [], [], [], []
    for position, asset_return, fee_rate, funding_rate, leverage in legs:
        n, g, f, u = decompose_leg(position, asset_return, fee_rate,
                                    funding_rate, leverage)
        nets.append(n)
        grosses.append(g)
        fees.append(f)
        fundings.append(u)
    if not nets:
        return None
    return attribute(
        np.mean(np.stack(nets), axis=0),
        np.mean(np.stack(grosses), axis=0),
        np.mean(np.stack(fees), axis=0),
        np.mean(np.stack(fundings), axis=0),
        timestamps,
    )
