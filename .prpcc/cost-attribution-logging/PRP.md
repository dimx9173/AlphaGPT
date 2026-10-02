# PRP-4: Cost Attribution Logging (read-only, no selection claim)

## 1. Problem Statement

Every other PRP in this line was refused on its preconditions. This one survives
because it makes no selection claim at all.

`accounting_bar_returns` in `research/accounting_28c.py` returns
`net = gross - fee - funding_cost`, and `evaluate()` builds every leg from that
net series, so the `portfolio_sharpe` the GA sees is already net of costs. There
is no gross Sharpe anywhere in the search path.

The consequence is that every recorded Sharpe is unexplained: the repo cannot say
how much of a run's performance is price movement and how much is the accounting.
That is the question the ranking-quality measurement ran into, and it is
answerable only by logging the decomposition.

## 2. Requirements

- [ ] F1. Record, per selection fold and for the holdout, the gross Sharpe, the
      net Sharpe, the fee drag and the funding drag, plus turnover, so that
      gross - fee_drag - funding_drag reconciles with net.
- [ ] F2. Purely additive to the result JSON. No change to evaluate(), no change
      to shape_reward, no new reward version, no change to selection.
- [ ] F3. The decomposition is computed from the same bar returns the accounting
      already produces, not re-derived.

## 3. Preconditions

- [ ] P1. PRP-0 landed and verified. Satisfied (commit 9c14fd3).
- [ ] P2. No dependency on the pre-gate result. This is why it can proceed.

## 4. Acceptance Criteria

- [ ] C1. For a smoke run, |gross - fee_drag - funding_drag - net| < 1e-9 for every
      recorded fold and the holdout.
- [ ] C2. The change is provably selection-neutral: a smoke run under a fixed seed
      selects the same formula before and after.
- [ ] C3. The full test suite is green and the known pre-existing failure is
      unchanged.

## 5. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Logging changes float ordering and shifts a selected formula | low | med | C2 pins selection-neutrality on a fixed seed |
| The decomposition exposes that most of the Sharpe is cost drag | high | none | that is the point; it is information, not a regression |

## 6. Rollback

Remove the added fields. No behavioural code is touched, so rollback is a revert.
