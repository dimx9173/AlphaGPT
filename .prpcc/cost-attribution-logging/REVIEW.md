# Review 7 — PRP-4 commit af9187a (structural fix)

Scope: `git show HEAD` (af9187a), `research/ga_28c_30m_3y.py`,
`research/accounting_28c.py`, `research/cost_attribution_28c.py`,
`tests/test_cost_attribution.py`. All 12 tests pass.

## 1. Is the leak now structurally impossible?

Yes, for the shipped code. Verified, not asserted:

- `git diff 8110505 HEAD -- research/ga_28c_30m_3y.py` is empty: `evaluate()`
  is byte-identical to its pre-attribution form (AST equality also confirmed
  locally). It returns only `reward`, `portfolio_sharpe`, diagnostics — none
  of `gross_sharpe` / `fee_drag` / `funding_drag` appear anywhere in the module.
- `evaluate()` calls `accounting_bar_returns(pos, r, FEE, fnd, LEV)` with no
  `out` argument, so the component dict is never materialised on the search
  path. The `out` write is guarded by `if out is not None`; when `None` the
  statements and order are unchanged (contract pinned by bit-identical test).
- `selection_score` reads only `a["reward"]` / `b["reward"]` (exact-equality
  test, both signs/magnitudes — no epsilon room). `shape_reward` signature
  contains none of the attribution keys.
- `cost_attribution_28c.py` is never imported by the GA module (no import,
  no callers outside tests). Data flows one way: accounting → fixed reward →
  post-hoc attribution. Reversing that requires editing `evaluate()` itself,
  which the AST test trips.

Residual scope note: "structurally impossible" holds for this tree, not as a
language-level property — Python cannot prevent a future edit from wiring the
new module back into selection. What makes that edit non-silent is the
test tripwire (item 2), which is the correct shape for this guarantee.

## 2. Is the AST comparison sound, and what does it cost?

Sound as a tripwire, with understood limits:

- Sound: `ast.dump` comparison ignores formatting/comments, catches every
  semantic change including neutral reorders (mutation-tested). Does not
  depend on the dead scan helpers — they are gone, and good riddance.
- Limits: pins `evaluate()` only, not `shape_reward` / `selection_score` /
  accounting — but those have their own tests (signature, exact selection,
  bit-identical accounting). `ast.dump` defaults exclude line numbers, so
  no false churn from offsets. Depends on git history containing `8110505`
  and a git binary at test time (breaks in shallow clones / history rewrites).
- Maintenance cost: ANY legitimate `evaluate()` change fails the suite until
  the pin is re-baselined with its own review. That friction is intentional
  and stated in the failure message. The substring `key not in src` check is
  fail-closed (flags even comments) — conservative, acceptable.

## 3. New coupling or cost in cost_attribution_28c.py?

No. Imports only `accounting_bar_returns` and `metrics` — the same pure,
dependency-free accounting the evaluator uses. No broker/network/db/model
surface. `attribute_legs` mean-of-legs matches `evaluate()`'s `pnet`
construction; `decompose_leg` returns the accounting's own `net` rather than
re-deriving it (pinned by test), so accounting changes cannot silently
desynchronise attribution. Re-running accounting per leg duplicates work, but
the module has no production-path callers — it is post-hoc only, so the cost
is pay-as-used. Minor: package-relative import (`from .accounting_28c`) vs
the GA's absolute style; works under the tested import path, not worth churn.

## 4. Anything that should not ship?

Nothing blocking. Notes:

- The accounting docstring edit (dropping the caller-count sentence) is
  comment-only; the `out` behaviour it describes is test-pinned either way.
- `research/attribute_sharpe_to_costs_28c.py` (pre-existing diagnostic) still
  exists alongside the new module with overlapping purpose. Non-blocking;
  suggest consolidating or cross-referencing in a follow-up, not this PR.
- This file overwrites the review-6 `REVIEW.md`; prior round history lives in
  git history for this path.

## 5. Verdict: PASS

The six bypass families (concat folding, read callables, approx tolerance,
PRODUCERS exemption, dunder/getitem, iteration/membership, key constructors)
are moot: the fields no longer exist in the evaluator, so there is no scan to
bypass. No blocking issues.
