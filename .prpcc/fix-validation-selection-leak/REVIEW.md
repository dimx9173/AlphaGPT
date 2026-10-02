# REVIEW (FINAL): PRP-0 fix-validation-selection-leak (Reviewer: Codex, round 3)

## Verdict: PASS

Commit `9c14fd3` genuinely resolves R1 and R2 from the round-2 CONDITIONAL.
No remaining acceptance criterion is unmet and no new defect was introduced.

## R1 — resolved, not restated

- PRP.md C5 is amended to key-arithmetic-only: `--select-on val_weighted`
  reproduces the legacy selection KEY via a unit test on the helper, and the
  recorded-batch fixture is explicitly WAIVED with the reason recorded
  (meter repair also changed `shape_reward`; no seed101 fixture exists in
  `results/wf_real`; full numerical reproduction requires the original
  commit, as honestly stated in PRP section 6).
- The amendment matches the code reality: `test_legacy_mode_still_reproduces_the_old_formula`
  pins the key arithmetic, and `test_helper_still_contains_the_legacy_key_for_reproducibility`
  pins that the legacy key survives. The waiver closes the loop instead of
  leaving C5 silently unproven.

## R2 — resolved, not restated

- The standard-mode loop is factored out of `main()` into `run_standard_selection()`,
  and `TestStandardModeWiring` drives it with a stubbed `evaluate()`: it perturbs
  only the validation reward (`-BIG*tr` vs `+BIG*tr`) and asserts the selection does
  not move under `train` mode, with a `val_weighted` control that must move.
- The pin is behavioural, not textual: the commit message records mutation checks
  where the standard loop re-spelled as `(a+b)/2` fails 2 tests — the exact case
  the regex scan could never catch. The walkforward call site has the equivalent
  `TestLoopLevelWiring` pin.

## Factoring fidelity — standard-mode behaviour unchanged

Compared statement-by-statement against `edd2625:research/ga_28c_30m_3y.py`
(the inline loop in old `main()`): seed/pop init, per-generation
evaluate-train/evaluate-validation/`selection_score`/append, sort, best-carry,
history append, elite slice, refill loop, `pop = new` — all present in
`run_standard_selection()` in the same order with the same expressions,
including the sort key and elite-count formula. No RNG or `evaluate` call
was added, removed, or reordered.

Per-generation print: old code printed `gen=... best=... avg=...` inside the
loop; `main()` now prints the same fields off the returned history. The values
are the identical objects (`scored[0][0]`, same `np.mean` inputs) and the
format string renders byte-identical lines. The only difference is streaming
timing (all lines emit after the loop instead of interleaved); content is
identical on any completed run. History and holdout writes are untouched.

## Acceptance criteria status

- C1 (no validation influence on ordering/elites/parents): met at both call
  sites by loop-level perturbation tests with a responding legacy control.
- C2 (train == val_weighted under constant validation): met by implication —
  the invariance tests prove a strictly stronger property (arbitrary, not just
  constant, validation perturbation leaves selection unchanged).
- C3 (overfit gap + argmax margin in every fold's JSON): met,
  `test_fold_records_carry_the_f5_diagnostics` passes.
- C4 (2-fold, 2-generation smoke run writes C5/C3 fields): met — the
  `TestLoopLevelWiring` harness runs exactly that shape through the real
  `run_walkforward`.
- C5 (amended, key arithmetic only): met; batch fixture waived with reason.

All 17 tests in `tests/test_validation_selection_leak.py` pass.

## New defects introduced

None. Two cosmetic nits, neither blocking: the `_run_standard` docstring still
says "the loop is reproduced here" (it now calls the helper — the comment is
stale but harmless), and it carries an unused `pathlib.Path` import.

## Remaining blocking issues

None.
