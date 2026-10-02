# PRP-2: Cross-Sectional Regularisation in the GA Reward

## 1. Problem Statement

The search maximises a portfolio score over 28 coins. Nothing in the objective
asks any two coins to behave differently. The last two pilot batches show the
consequence directly:

| batch | real pos_coins median | real mean_position median |
|---|---|---|
| pre-repair (wf_real) | 23 / 28 | n/a |
| post meter repair | **0 / 28** | **-0.197** |

A positive_coins median of 0 means the selected portfolio was losing on every
single leg, and a mean_position of -0.197 means it was persistently short. Those
are two failure modes the current reward cannot see: `shape_reward` charges
`min_leg` only through a softplus floor at LEG_FLOOR = 0, so a book that is
uniformly negative still scores close to a book with one good leg, and nothing
prices the correlation between legs at all.

Multiplicity is the deeper issue. 28 correlated legs give the search 28 chances
to fit the same noise spike. Standard deviation across legs is therefore a
first-order selection statistic and it is currently absent from the objective.

## 2. Requirements

### Functional
- [ ] F1. Add a cross-sectional dispersion charge to `shape_reward`, penalising
      low dispersion of per-leg Sharpe. Weights are versioned, never overwritten:
      v4 carries the new term, v1-v3 stay bit-reproducible.
- [ ] F2. Add an explicit breadth term measured on positive legs, already present
      in v2 via BREADTH_TARGET but currently inactive by default. v4 must use it.
- [ ] F3. Directional concentration is charged explicitly: penalise |mean_position|
      above a configured ceiling so "always short" cannot be a free optimum.
- [ ] F4. The reward must remain continuous. No new hard return path; the -50 cliff
      stays removed.
- [ ] F5. All new terms are recorded per fold so a v4 run can be decomposed.

### Non-Functional
- Frozen: evaluator math, FORMULA_LEN, FEE/FUND/LEV, universe, folds, seeds,
  holdout window, DEFAULT_SELECT_ON, existing v1 weights.

## 3. Preconditions

- [ ] P1. PRP-0 landed (commit 9c14fd3, verified).
- [ ] P2. P2 pre-gate passed: formula-only with both fixes gives real-null gap > 0
      on the holdout scored once. Currently UNKNOWN; the running batch is the
      measurement. If it fails, PRP-2 is NO-GO by the same logic as PRP-1.

## 4. Acceptance Criteria

- [ ] C1. v4 improves real positive_coins median from 0/28 to >= 14/28 on the same
      seeds, measured on the holdout.
- [ ] C2. real |mean_position| median < 0.25 on >= 6/10 seeds.
- [ ] C3. The v4-vs-v1 comparison is a PAIRED A/B: same seeds, same folds, same
      holdout, both arms run, and the per-fold decomposition is published.
- [ ] C4. v1 remains bit-reproducible after the change: a recorded v1 batch reruns
      to the same numbers. Test first, fixture second.
- [ ] C5. The new terms demonstrably move the score: a synthetic reward call with
      zero dispersion scores strictly worse than the same call with high
      dispersion, and the gap is reported.

## 5. Test Plan

- Unit: dispersion term monotone in std of leg Sharpes; breadth term monotone in
  positive count; direction term monotone in |mean_position|.
- Unit: v1 output unchanged by the presence of v4 code (the version gate).
- Integration: 2-fold 2-gen smoke writes the per-term decomposition.
- Regression: full suite plus the known pre-existing failure unchanged.

## 6. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Legs are genuinely correlated, so a dispersion penalty just adds noise | high | med | measure the leg-correlation first and report it in the run |
| v4 finds a book that is flat by construction (all legs tiny) | med | high | C1 requires positive_coins, not just low dispersion |
| Tuning the new weights on the same 10 seeds | high | high | weights fixed a priori from the leg-correlation measurement, not fitted to pilot output |

## X. Ranking-quality criterion (added after the feasibility review)

Every acceptance criterion above compares SELECTED MAXIMA across re-runs. That
is consistent with pure selection bias: a search that ranks noise perfectly
produces a high winner on real data too. Comparing winners cannot distinguish
"found signal" from "ranked the apparatus well".

The falsifiable statistic is the rank correlation between the training score and
the holdout outcome across EVERY genome evaluated, not only the winner:

    rho = spearman(train_score, holdout_net_sharpe)

Two questions, both answered by research/measure_reward_ranking_quality.py:

  Q1  Is the score informative at all? A rho near zero means the search ranks
      nothing and no reward reshaping can help.
  Q2  Is the score MORE informative on real data than on null data? This is the
      one that matters. If real and null give the same rho, the score is ranking
      a property the synthetic world provides for free, and none of the changes
      in this PRP can create information the data does not contain.

Measured on the current tree (3 folds x 3 generations x 10 population, 3 seeds
per arm):

    real: spearman(train_reward, holdout_sharpe) = +0.8545
    null: spearman(train_reward, holdout_sharpe) = +0.8758
    gap                                       = -0.0214

    real: spearman(train_ic, holdout_ic)       = +0.6188
    null: spearman(train_ic, holdout_ic)       = +0.4297
    gap                                         = +0.1891

Reading: the score is NOT broken. It ranks holdout Sharpe very well on BOTH arms,
which means it ranks a property the synthetic world supplies by construction
(volatility clustering, fat tails, drift) rather than anything about the data.
Rank-IC, which is closer to a cross-sectional property, is the quantity that
separates the arms, and it does separate by +0.189.

## 7. Rollback

`--reward-version v1` is untouched and remains the default until v4 passes its
own gate. v4 is opt-in per run.
