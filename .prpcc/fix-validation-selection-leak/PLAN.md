# PLAN: fix-validation-selection-leak (PRP-0)

Architect review complete. Verified against `research/ga_28c_30m_3y.py` @ 779 lines.

## Verified facts

- **Claim 1 TRUE**: `run_walkforward` computes `sel = 0.5*a['reward'] + 0.5*b['reward']` at **line 581**, inside `for gen in range(args.generations)` (line 570), with the validation eval `b` at lines 576-578. Ranking (583), elites (586), parents (589-590) and `fold_best` (584-585) all key off `sel`. The comment at 579-580 ("never revisited") and the docstring at 519-521 ("No window is ever scored twice") are both false.
- **Standard mode has the identical defect**: `sel=0.5*a['reward']+0.5*b['reward']` at **line 685**, inside `for gen` at line 681; `best` (686) and elitism/parents (689-692) key off it.
- **PRP §6 correction**: there is no `run_iterations` function anywhere in the repo. The standard-mode loop is inline in `main()` (lines 681-694). PRP §6 must say `run_walkforward` / the standard loop in `main()`.
- **`b` staleness**: today `b` is bound fresh each inner iteration (lines 576-578 / 684) — not stale. It *becomes* stale/undefined only if the fix deletes the `b=evaluate(...)` call while leaving `scored.append((sel,f,a,b))` (NameError on the first genome; previous genome's val thereafter). In walkforward `b` is required anyway for the record's `validation_sharpe` (line 602) and the print (607).

## Minimal edit

Change the selection key expression only, at both sites; keep the `b` evaluation and the `(sel, f, a, b)` tuple shape:

- line 581 and line 685: `sel = a['reward'] if select_on=='train' else 0.5*a['reward'] + 0.5*b['reward']`
- new `--select-on {train,val_weighted}` argparse flag, default `train`; loud warning on `val_weighted`
- correct the false comment (579-580) and docstring (515-522)
- PRP §6: replace `run_iterations` with the standard loop in `main()`

Nothing else in the evaluator, folds, or reward math moves.

---

## Tasks (test-first)

### T1 — Red test: selection isolation
- **Files**: `tests/test_validation_selection_leak.py` (new)
- **done_when**: Two tests exist and FAIL against current code:
  1. walkforward: monkeypatch `evaluate` so the validation-window call returns `reward=±1e9`; assert the selected formula, elite set, and parent pool are byte-identical to the unpatched run (C1).
  2. standard mode (`main` loop extracted or driven via `--generations 2 --population 8`): same perturbation, same assertion on `best[1]` and elites.

### T2 — Close the leak: `--select-on`, train-only key at both sites
- **Files**: `research/ga_28c_30m_3y.py`, `.prpcc/fix-validation-selection-leak/PRP.md`
- **done_when**: T1 passes. `--select-on train` (default) keys `sel` on `a['reward']` only at lines 581 and 685; `b` evaluation and tuple shape untouched (no stale-`b` trap); `--select-on val_weighted` prints a loud warning; false comment (579-580) and docstring (515-522) corrected; PRP §6 no longer names the nonexistent `run_iterations`.

### T3 — Degenerate-equivalence test (C2) + legacy formula test (C5)
- **Files**: `tests/test_validation_selection_leak.py`
- **done_when**: Tests pass asserting: with validation reward forced constant, `--select-on train` and `--select-on val_weighted` produce identical `scored` ordering; and under `val_weighted` the sel key equals `0.5*a+0.5*b` on synthetic evals (the legacy formula is pinned, so historical batches stay reproducible).

### T4 — F5 diagnostics: overfit gap + argmax margin
- **Files**: `research/ga_28c_30m_3y.py`, `tests/test_validation_selection_leak.py`
- **done_when**: Each selection-fold record carries `select_on`, `overfit_gap` (selected genome's val reward − train reward), and `argmax_margin` (selected train reward − best train reward of any other genome across generations); a test asserts all three fields exist and `overfit_gap` equals the perturbed-value arithmetic on a 2-gen synthetic run (C3).

### T5 — Regression smoke: both modes, 2 folds × 2 gens
- **Files**: `tests/test_validation_selection_leak.py` (or a subprocess smoke test)
- **done_when**: `--mode walkforward --folds 2 --generations 2 --population 8` and the equivalent standard-mode run both exit 0 and write JSON containing `select_on`, per-fold `overfit_gap`/`argmax_margin`, and an untouched holdout record (C4). Holdout fold still scored exactly once, after selection (F3).

## Out of scope / follow-up note
Evaluating `b` in-loop under train mode is now dead weight for selection but harmless (still feeds `validation_sharpe`). Optional follow-up: evaluate validation once per fold after selection to make "one pass" literally true and cut pop×gens×folds evals. Not required for this PRP.
