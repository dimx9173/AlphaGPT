# CODE REPORT — PRP-4 cost attribution logging

## Changes

research/accounting_28c.py
- `accounting_bar_returns` gains an optional `out` dict. When it is None the
  executed statements and their order are exactly as before.
- After the existing `net = gross - fee - funding_cost` line, `out` is filled
  with gross / fee / funding_cost. That line itself is untouched.
- `account_portfolio` was accidentally given the same block during editing and
  restored; it has no `out` and its body is byte-identical to before. Caught by
  tests/test_accounting_28c.py::TestAccountPortfolio, which failed with
  NameError before the restore.

research/ga_28c_30m_3y.py
- `evaluate` initialises leg_gross, leg_fee, leg_fund (additive).
- Per leg, calls accounting_bar_returns with out=comp and appends the three
  components. leg_net.append(net) and everything after is unchanged.
- After `psh = pm['sharpe']`, aggregates pgross/pfee/pfund and computes the
  chained drags. Convention fixed and documented: fee first, then funding,
  matching the arithmetic order inside the accounting.
- Returns three additive keys: gross_sharpe, fee_drag, funding_drag.
  shape_reward does not read them and its signature is unchanged.

tests/test_cost_attribution.py — 9 tests, new.

## Acceptance criteria

C1 drag reconciliation — MET. Residual on a real smoke run:
    train       gross=+0.9689 fee=+0.9334 funding=+0.0521 net=-0.0166  resid=4.9e-17
    validation  gross=-1.5689 fee=+1.0320 funding=-0.0370 net=-2.5639  resid=0.0
    oos         gross=+1.5939 fee=+1.3831 funding=+0.0198 net=+0.1910  resid=-5.6e-17

C2 selection neutrality — MET, bit-identical on a fixed seed (standard mode,
seed 101, 2 generations, population 8), recorded before the change existed:

    formula            [9,17,17,17,17,17,17,17,17,17,17,17]  both
    train_sharpe       -0.01659567834827822                    both
    train_reward       -0.4189917095596396                     both
    validation_sharpe  -2.563864091924182                      both

C3 suite green — MET, with pre-existing failures unchanged:
    tests/test_e10_gates.py::test_paper_baseline_within_tolerance (552 vs 478)
    tests/test_perp_risk.py (4) and tests/test_risk.py (2) — solders import,
    verified by stashing the change and re-running.

## What this already shows

The oos window has gross Sharpe +1.5939 and net +0.1910: a fee drag of 1.3831
turns a strongly positive pre-cost result into a marginal one. The train window
is gross +0.9689 into net -0.0166. That is a different diagnosis from "no
signal", and it was not visible anywhere before this PRP.

## Deviations

The delegated code phase stalled twice and produced no code, so the author
implemented T1-T5 directly. The design followed PLAN.md unchanged.
