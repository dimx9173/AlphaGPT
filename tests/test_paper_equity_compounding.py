"""The paper runner must compound, and must label its accounting honestly.

Defect: `run_paper_28c_pit_30m.py` computed correct per-bar weighted *return
rates* into `portfolio_net`, then built the equity curve with
`1.0 + np.cumsum(portfolio_net)`. That is additive PnL on return rates, so it
added 28 legs' returns without reinvesting. It reported equity 1.00 -> 26.78
(+2578%) and a Sharpe inconsistent with its own equity curve.
"""
import ast
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from research.accounting_28c import compound_equity  # noqa: E402

RUNNER = os.path.join(ROOT, "research", "run_paper_28c_pit_30m.py")


def _src():
    return open(RUNNER).read()


def test_equity_curve_is_not_additive_cumsum():
    src = _src()
    assert "np.cumsum(portfolio_net)" not in src, \
        "equity rebuilt from an additive cumsum of return rates"
    assert "np.cumsum(active_returns)" not in src
    assert "compound_equity(portfolio_net)" in src


def test_accounting_label_is_not_a_hardcoded_legacy_string():
    src = _src()
    assert "additive_cumulative_pnl" not in src, \
        "accounting label still claims the legacy additive convention"
    assert '"accounting": ACCOUNTING_VERSION' in src


def test_compounding_changes_the_answer_materially():
    """Compounding and cumsum must diverge on a realistic return stream,
    which is why the old number was wrong."""
    rng = np.random.default_rng(7)
    r = rng.normal(0.0005, 0.002, 20000)
    comp, _ = compound_equity(r)          # n+1, starts at 1.0
    add = np.concatenate(([1.0], 1.0 + np.cumsum(r)))   # same convention
    assert len(comp) == len(add)
    assert not np.allclose(comp, add)
    assert comp[-1] > 0


def test_runner_parses():
    ast.parse(_src())
