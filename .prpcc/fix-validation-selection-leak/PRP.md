# PRP-0: Fix the validation-window selection leak

## 1. Problem Statement

`research/ga_28c_30m_3y.py` selects on `sel = 0.5*train_reward + 0.5*val_reward`
and ranks/elites on `sel`, inside the generation loop. The validation window is
therefore re-scored every generation, and the code comment claiming it "is never
revisited" is false as written.

Measured reuse at the pilot configuration (pop 16, gen 25, 10 folds):

| mode | validation-window reads |
|---|---|
| standard | 16 x 25 = **400** on one window |
| walkforward | 9 selection folds x 400 = **3,600** |

Every other number produced by this search is inflated by a winner's curse of
that multiplicity. This is the precondition for any further search work: until
the leak is closed, a change to the search space cannot be evaluated, because
the instrument reports a biased score regardless of what changed.

This defect was found by Codex during the PRP-1 feasibility review
(`.prpcc/joint-param-search/REVIEW_PLAN.md` R1).

## 2. Requirements

### Functional
- [ ] F1. Selection ranks on the TRAIN reward only. The validation split may be
      evaluated once per generation for reporting, and MUST NOT enter any
      selection, elitism, parent or tie-break decision.
- [ ] F2. The best genome per fold is chosen on train reward alone.
- [ ] F3. The final holdout fold is scored exactly once, after selection, and
      is never used to choose anything.
- [ ] F4. `--select-on` flag, default `train`, with the legacy `val_weighted`
      mode available ONLY for reproducing the historical batches. It must print a
      loud warning when used.
- [ ] F5. Per-fold diagnostics record: train reward of the selected genome, its
      validation reward, the difference (the overfit gap), and the argmax margin
      over the runner-up on train.

### Non-Functional
- Frozen: evaluator math, genome, FEE/FUND/LEV, universe, folds, seeds, holdout
  window, reward weights. Nothing else changes in this PRP.

## 3. Acceptance Criteria

- [ ] C1. No expression containing the validation reward influences `scored`
      ordering, `elites`, `p1`/`p2`, or `fold_best`. Proven by a test that
      perturbs the validation reward to a huge value and asserts the selected
      genome is unchanged.
- [ ] C2. `--select-on train` output for the selected formula matches
      `--select-on val_weighted` when the validation reward is forced constant
      (a degenerate case that isolates the ranking key).
- [ ] C3. The per-fold overfit gap and argmax margin are present in the result
      JSON for every fold.
- [ ] C4. Regression: a 2-fold, 2-generation smoke run completes and writes C5/C3
      fields.
- [ ] C5. Historical batches remain reproducible via `--select-on val_weighted`,
      asserted by a test on one recorded seed.

## 4. Test Plan

### Unit
- rank isolation: monkeypatch the validation reward to +/-1e9 for all genomes;
  the selected formula must not change.
- elitism isolation: same perturbation; the elite set must not change.
- tie-break: with all train rewards equal, selection is deterministic and
  documented (lowest index).

### Integration
- 2-fold, 2-gen smoke writes `select_on`, per-fold overfit gap, argmax margin.
- `--select-on val_weighted` on seed 101 reproduces the recorded formula.

### Chaos
- a fold whose validation window is pure noise must still select the train-best
  genome.

## 5. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Historical batches become unreproducible | med | med | F4/F5 keep the legacy path behind a flag, tested |
| A stray val reference survives the edit | med | high | C1 perturbation test fails loudly |
| New numbers look worse than old ones | high | low | expected and desired: the old numbers were inflated |

## 6. Rollback

The change is confined to the selection key. `--select-on val_weighted` restores
the prior SELECTION KEY.

It does NOT restore prior NUMBERS, and this is stated in the flag help rather
than papered over: two separate changes live in this working tree, this leak fix
and the meter repair that replaced the -50 cliff with a continuous activity
penalty. A batch recorded before either one is not reproducible from this file
and requires its original commit.

Splitting them into two artificial commits was rejected as worse than the honest
statement: the two were developed and tested together, and a synthetic split
would make `git bisect` point at a state that never existed.

Reverting the key itself is still a flag and needs no code change.

## 7. Sign-offs

- Planner: pending
- Architect (Claude): pending
- Reviewer (Codex): pending
