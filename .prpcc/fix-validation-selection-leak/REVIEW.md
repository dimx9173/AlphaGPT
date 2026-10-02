# REVIEW: PRP-0 fix-validation-selection-leak (Reviewer: Codex)

## Verdict: CONDITIONAL

The leak itself is closed in both loops. Merge is blocked on the freeze
violation (B1) and the test gaps (B2, B3), which mean the closure is not
pinned and historical reproducibility is unproven.

## Blocking issues

### B1. Pilot freeze violated: evaluator math moved in the same change
- The working-tree diff (`/tmp/prpcc_p0.diff`) contains, alongside the
  selection fix, the pilot-meter-repair edit: `shape_reward` activity cliff
  (`return -50.0` / `mean_act_guard`) removed and replaced with a continuous
  penalty, plus `ACTIVITY_TARGET` / `INACTIVITY_WEIGHT` / `PEER_BASELINE`.
- PRP-0 §Non-Functional freezes evaluator math. This breaks the §6 rollback
  story: `--select-on val_weighted` reproduces the old *selection key*, but
  NOT old batch numbers, because the reward underneath it changed.
- `results/wf_real/*/seed101.json` does not exist in the tree, so C5
  ("reproduces the recorded formula") is asserted nowhere and provable nowhere.
- Fix: split the meter repair out of this change, or amend the PRP to
  explicitly re-baseline the freeze with before/after reward dumps. Either
  way, add the seed-101 repro test (C5) against a checked-in fixture.

### B2. Tests pin the helper, not the wiring (C1/C2/C4 unproven)
- Every selection test calls `ga.selection_score` directly or re-implements
  the sort in the test body. Nothing runs `run_walkforward` or the standard
  loop, so a future edit that bypasses `selection_score` (e.g. inlines a
  weighted key at the call site) passes all tests while re-opening the leak.
- C1 demands a perturbation test on the *selected genome*; the tests perturb
  only the helper's return value. C2 (train == val_weighted under constant
  validation reward) and C4 (2-fold/2-gen smoke writing C3 fields) have no
  test at all.
- Fix: add a loop-level test that monkeypatches `evaluate` (or injects
  absurd `b['reward']` values) and asserts `fold_best`/elite/parent sets are
  unchanged, plus the C2 degenerate-case test and a C4 smoke run.

### B3. The regex scan is vacuous as written
- `test_no_val_weighted_selection_in_active_code` strips whitespace and
  looks for `0.5*a['reward']+0.5*b['reward']` (single quotes). The sanctioned
  legacy expression inside `selection_score` uses double quotes
  (`0.5 * a["reward"] + 0.5 * b["reward"]`), which the pattern never matches.
- Consequences: (a) the test passes with a weighted expression present, so it
  proves nothing about the active paths; (b) a re-introduced leak spelled with
  double quotes, reversed operands, or `(a+b)/2` also passes.
- Fix: allowlist the `selection_score` body explicitly and fail on any other
  `reward`-mixing expression, or drop the scan and rely on the B2
  loop-level perturbation test.

## Non-blocking notes

- N1. F5 arithmetic is correct: `overfit_gap = val_reward - train_reward`
  (negative = ordinary), `argmax_margin = winner_sel - runner_up_sel`.
- N2. The cross-generation worry is a non-issue: `fold_best` and
  `fold_runner_up` are updated together in one `if`, so both come from the
  generation that crowned the winner. Residual staleness (a later generation
  with a closer runner-up but no new winner keeps the old margin, strict `>`
  keeps first-wins ties) is acceptable and documented behavior.
- N3. Under `--select-on val_weighted`, `argmax_margin` is in weighted units,
  not train units, despite the record calling it the same field. Harmless for
  a repro-only mode, but label it or note it.
- N4. `TestTieBreakIsDeterministic` asserts a property of Python's stable
  sort, not of the project's code. Keep as documentation, not evidence.
- N5. `TestPilotFreezeStillHolds` does not cover what actually moved: it
  checks `FORMULA_LEN`/`FEE`/`FUND`/`LEV`/evaluator substrings but not the
  `shape_reward` cliff or reward scale, so B1 slipped through it. Extend the
  freeze test to the reward function if the freeze is to mean anything.
- N6. `b` is still evaluated every generation in both loops (400 reads/fold
  at pilot settings). That is observation, not selection, and F1 permits it —
  but the "never revisited" comment lineage should stay dead; the replacement
  comments do this correctly.

## What was verified

- `run_walkforward`: `scored` ordering, `elites`, `p1`/`p2` (top-5 of
  `scored`), and `fold_best` all key off `selection_score(a, b, args.select_on)`
  = `a['reward']` under default. No other `b['reward']` read feeds a decision.
- Standard loop: same, via `selection_score(a,b,args.select_on)`; final
  lockbox/train/val re-evaluations and the regime gate run once, post-search.
- Holdout fold is scored exactly once after selection, never fed back.
- `--select-on val_weighted` restores the exact old key expression and prints
  the loud warning; `select_on` is recorded in walkforward records and both
  `out` dicts.
