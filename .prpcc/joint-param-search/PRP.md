# PRP: Joint Parameter Search (execution layer into the GA search space)

## 1. Problem Statement

The GA in `research/ga_28c_30m_3y.py` searches only formula token sequences. The
eight execution parameters that `MemeBacktest` exposes (lth/sth/cd/sl/ts/vt/vw/q)
are fixed constants in that evaluator, and they are not even representable there:
`evaluate()` builds positions as `0.25 * smooth_causal(tanh(sig/scale), 5)` with no
threshold, no cooldown and no stop.

The project's own iteration record (`docs/STRATEGY_ITER_REPORT.md` round 2) shows
threshold and cooldown tuning alone moved Sharpe 0.70 -> 1.27 on the 4h BTC engine.
That effect was never available to the GA.

Empirical context, all measured in this repo:
- standard mode, real vs null(iid): Sharpe gap **-1.166** (p=0.001, null wins)
- walkforward pre-repair: gap **-0.434**
- walkforward post meter-repair: gap **-3.355**, real positive_coins **0/10**
  (see docs/GA_SEARCH_RELIABILITY_2026-10-01.md)
- rank-IC is the only statistic that separates real from null: +0.01071 vs
  +0.00363, real wins 8/10

## 2. Requirements

### Functional
- [ ] F1. A threshold/cooldown execution model exists inside the 28c accounting
      path so the GA evaluator can apply it. Reuse the semantics in
      `MemeBacktest._apply_cooldown` / `_apply_stops`, not a second implementation.
- [ ] F2. Position sizing remains continuous (tanh) with an entry threshold and an
      optional cooldown, so a genome that never triggers still scores rather than
      hitting a cliff. No new hard -50 return path.
- [ ] F3. The eight parameters are searched as a JOINT space with the formula, with
      each genome carrying `(formula, params)`.
- [ ] F4. Params are tuned on TRAIN only. The validation window scores the frozen
      pair exactly once per fold. No max-over-grid on validation. This depends on
      PRP-0 removing the `0.5*train + 0.5*val` selection key at BOTH sites
      (`run_walkforward` and the standard loop in `main()`); a test must FAIL if any
      validation-weighted selection survives.
- [ ] F5. Every reported number is real-minus-null, never an absolute Sharpe.

### Non-Functional
- Search cost per genome must stay within 2x the current cost, or the pilot
  configuration (25 gen x pop 16 x 10 folds x 20 runs) does not fit the budget.
- Frozen during the pilot: evaluator tanh/smooth/roll, genome length, FEE/FUND/LEV,
  universe, folds, seeds 101-110, holdout window, reward weights.

## 3. Architecture

```
evaluate(formula, params, split) -> {reward, sharpe, turnover, ...}
   |
   +-- signal = StackVM(formula, features)          (unchanged)
   +-- base_pos = 0.25 * smooth_causal(tanh(sig/scale), 5)
   +-- apply entry threshold:  long_th / short_th on sigmoid(base_pos)
   +-- apply cooldown:          cooldown_bars minimum hold
   +-- optional stop / time stop / vol target
   +-- roll(1)                  execution lag, unchanged
   +-- existing accounting: fees, funding, equity, Sharpe
```

The parameter vector is a small grid, not a continuous optimiser, so that every
point is reproducible and enumerable for the null calibration.

## 3a. Preconditions (must hold before any joint-search budget is spent)

- [x] P1. **PRP-0 landed** (commit 9c14fd3, Codex PASS, verify 6/6): validation-window selection leak closed. Currently the
      validation split is re-scored every generation and carries 50% of the
      selection key, 3,600 times per run at pilot settings. Until that is fixed a
      joint space cannot be evaluated, because the instrument is biased regardless
      of what changed. See `.prpcc/fix-validation-selection-leak/`.
- [ ] P2. **Formula-only pre-gate.** With the repaired meter AND the closed leak,
      run formula-only on seeds 101-110. Proceed only if the real-null gap is > 0.
- [ ] P3. **Null calibration for execution params.** `--null iid` destroys the
      autocorrelation that cooldown and hold-time parameters exploit, so the iid
      arm may not calibrate this search space at all. Add a second
      autocorrelation-preserving arm (block bootstrap or phase shuffle) and report
      both. Writing down why iid alone suffices is also acceptable if it can be
      argued; it cannot be assumed.

## 4. Acceptance Criteria

- [ ] C1. REPORT ONLY, not a gate. Real median Sharpe minus null median Sharpe on
      the holdout. Under the null this has ~50% false-pass rate, so it is recorded
      and never asserted. Listed because the earlier pilot was misread through it.
- [ ] C2. THE GATE. One-sided Mann-Whitney real > null at p < 0.10 on the holdout,
      direction asserted explicitly. Pre-registered seeds 101-110, no reseeding, no
      dropping seeds, no adding seeds after seeing data. Power at n=10 is ~38% for a
      per-seed win rate of 0.7 and needs 8/10 dominance, so a MISS is weak evidence
      and the pilot should report the power alongside the verdict.
- [ ] C3. real mean_position magnitude is below 0.25 on at least 6/10 seeds, i.e.
      the search is not winning by always betting one direction.
- [ ] C4. The repaired reward shows zero saturation and rankable-run std greater
      than pre-repair, measured by `verify_pilot_meter_repair.py`.
- [ ] C5. Per-fold diagnostics record the selected params, `scale_end`, the
      empirical trigger rate, the per-fold overfit gap (train reward of the
      selected pair minus its frozen validation reward), and the argmax margin
      (how much the winner beat the runner-up on train). Thin margin plus large
      val drop is the winner's-curse signature and must be visible.
- [ ] C6. BREADTH GATE. >= 6/10 real runs have positive_coins >= 14. The last
      pilot had 0/10, which means the selector was finding degenerate portfolios;
      adding freedom on top of that is how the search learns to gate into a
      one-direction drift.
- [ ] C7. COST GATE. For every selected pair, report turnover and the fee/funding
      attribution. Flag any pair whose Sharpe delta versus the default params is
      more than 50% cost-driven, since a threshold can "win" purely by trading
      less.
- [ ] C8. ABLATION GATE. Score the selected params with a random formula, and the
      selected formula with default params. If params-only matches joint on the
      holdout, the formula search is decoration and C2 is measuring execution fit,
      not alpha. Recorded as a gate because that outcome invalidates the whole PRP.

## 5. Test Plan

### Unit
- threshold/cooldown/stop applied to a hand-built signal matches the MemeBacktest
  semantics on the same inputs
- a genome that never crosses the threshold scores continuously, never -50
- params tuned on train do not leak into the validation score (assert the
  validation call receives the train-selected params)

### Integration
- a 2-fold, 2-generation smoke run completes and writes all five C5 fields
- the paired pilot driver runs both arms with matched seeds

### Chaos
- a null arm where the grid is wider than the formula space must not let the null
  win more than the real arm; if it does, the grid is manufacturing selection

## 6. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Joint space multiplies selection bias (winner's curse) | high | high | F4: tune on train, score frozen pair on val once. Never max on val. |
| Cost per genome rises too much | med | med | small grid (3 x 3 x 2), measured against the 2x budget in Non-Functional |
| New execution code diverges from MemeBacktest | med | high | F1: reuse, plus a parity unit test against the same inputs |
| Params dominate and formula becomes irrelevant | med | med | C5 records both; report per-fold whether formula or params moved the score |
| Real still loses to null | high | high | that is a legitimate NO-GO outcome; record it, do not widen the search to chase it |

## 7. Rollback Plan

A CLI flag is NOT sufficient here, because F1 edits the shared `evaluate()` path and
flag-off is only bit-identical if every default, signature and float ordering is
unchanged. A silent reshuffle of arithmetic would shift every historical number while
the flag still claims "off means current".

Therefore:
- the new execution logic lives in a SEPARATE wrapper function; the existing
  `evaluate()` body is not interleaved with branches
- a parity test asserts flag-off output is BITWISE equal to the pre-change output on
  reference seeds
- the flag defaults off and the pilot opts in per run

## 7a. Determinism rules

- Grid ties (equal train reward) resolve to the lowest grid index, identically in
  both arms.
- Both arms enumerate the grid in the same order.
- Formula RNG and parameter RNG are separate streams, so adding or removing a grid
  point does not shift the formula sequence.
- Grid-jitter robustness: shift the grid by half a step on train and confirm the
  selected region is stable. A winner that moves when the grid moves was luck.

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

## 8. Sign-offs

- Planner: pending
- Architect (Claude): pending
- Reviewer (Codex): pending
