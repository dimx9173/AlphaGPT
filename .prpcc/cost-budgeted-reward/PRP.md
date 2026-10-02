# PRP-3: Cost-Budgeted Reward (turnover and funding as a budget, not a hope)

## 1. Problem Statement

`shape_reward` subtracts `0.02 * mean_turnover`. That is a linear penalty with no
relationship to what turnover actually costs. The repo already contains the
measurement needed to price it properly:
`research/attribute_sharpe_to_costs_28c.py`, and the pilot analysis showed the
real arm's turnover median was 3.06e-03 against the null's 1.58e-04, a 19x gap.

The failure this creates is specific and measurable: a genome can "win" by
trading less and paying less, without any predictive content. That is cost
avoidance dressed up as signal, and it is invisible in a Sharpe number.

It is also the natural third item after PRP-1 and PRP-2 because it changes what
the other two are optimising under. Adding parameters (PRP-1) and breadth
constraints (PRP-2) both alter turnover, so the cost model has to be right
before either of them can be judged.

## 2. Requirements

### Functional
- [ ] F1. Compute realised net Sharpe after fees and funding for every evaluation,
      using the same accounting the final equity curve uses. No proxy.
- [ ] F2. Replace the linear turnover penalty with the measured cost drag, in a
      new versioned reward (v5). v1-v4 stay bit-reproducible.
- [ ] F3. Record gross Sharpe, net Sharpe, fee drag, funding drag and turnover per
      fold, so cost avoidance is visible rather than inferred.
- [ ] F4. A selection whose Sharpe gain is more than 50% cost-driven is flagged
      in the output.

### Non-Functional
- Frozen: evaluator math, FORMULA_LEN, FEE/FUND/LEV, universe, folds, seeds,
  holdout window, DEFAULT_SELECT_ON.

## 3. Preconditions

- [ ] P1. PRP-0 landed and verified.
- [ ] P2. P2 pre-gate gap > 0.
- [ ] P3. PRP-2 landed, because breadth and parameter search both move turnover
      and this reward must be the one they are judged under.

## 4. Acceptance Criteria

- [ ] C1. gross-to-net attribution is published for every fold: fee drag and
      funding drag separately, summing to the observed gap.
- [ ] C2. v5 beats v4 on NET Sharpe on the holdout for >= 6/10 real seeds, or the
      PRP is NO-GO.
- [ ] C3. No selected pair in a v5 run is flagged cost-driven above the 50%
      threshold.
- [ ] C4. v4 remains bit-reproducible after the change.

## 5. Test Plan

- Unit: the attribution decomposition sums to the observed gross-net difference
  within a stated tolerance.
- Unit: a zero-turnover portfolio has zero fee drag.
- Integration: 2-fold 2-gen smoke writes all attribution fields.
- Regression: full suite, known pre-existing failure unchanged.

## 6. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Correcting for costs just lowers every Sharpe and changes no ranking | high | med | that is a legitimate finding; it is reported, not tuned away |
| The 50% cost-driven threshold is arbitrary | med | med | it is a flag for review, not a pass/fail gate |
| Net Sharpe is noisier than gross, so v5 selects worse | med | high | C2 is on net Sharpe specifically; a miss is NO-GO, not a re-tune |

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

`--reward-version v4` remains available and is what PRP-2 produced. v5 is
opt-in. The accounting path itself is shared and unchanged, so reverting the
reward is a flag.
