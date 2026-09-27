# PRP Plan: Regime-Neutral Acceptance Gate (revision 2)

Cycle: `regime-gate`. Spec: `PRP.md` rev 2. Anti-gaming clause PRP section 2 binding.
Addresses Codex B1-B6. All tasks test-first.

## Task 1 — Slice-based accounting entry point (fixes B5)
- `research/accounting_28c.py`: add `net_series_full_then_slice(maps, returns, coins,
  positions, start, end, funding_mask, scale_end, common)` that computes bar returns
  ONCE over the full aligned series and returns a slice. It must NOT reset
  `prev_pos[0] = 0.0` at the slice head.
- Also add `passive_benchmark_net(returns_map, coins, start, end, funding_mask)`:
  constant +1 position, same LEV/FEE/funding, so leverage and costs match by
  construction (fixes B3).
- Tests first: boundary-cost equivalence (test 7), benchmark leverage/cost parity
  (test 1), and the leverage-parity regression that a static 2x hold yields excess
  Sharpe ~ 0 (test 2).

## Task 2 — Benchmark / regime / excess primitives
- `regime_label(bench_net, timestamps)` -> `("up"|"down", annual, sharpe, max_dd)`,
  deterministic at exactly 0.0 -> `down`, no tolerance band (test 4).
- `excess_metrics(strategy_net, bench_net, timestamps)` -> active Sharpe + diagnostics,
  computed on the difference series only (test 3).

## Task 3 — Fold construction (fixes B1, B2)
- `research/splits_28c.py`: `walk_forward_folds_28c(n, split_indices_result, k=4)`.
- Returns exactly the four literal ranges in PRP F4:
  `[0,10566) [10566,21131) [21531,24025) [24025,26520)`.
- Takes the split dict, never a raw `end`, so the lockbox is unreachable by
  construction. Assert `max(fold_hi) <= validation[1]`.
- Tests first: exact ranges (test 5), coverage invariant and embargo exclusion
  (test 6).

## Task 4 — Gate evaluation
- `research/regime_gate_28c.py`: `evaluate_regime_gate(...)` returns the per-fold table
  plus A1, A2', A2'', A3, A4, A5, A6 and a single verdict in
  `pass | fail | insufficient_regime_coverage`.
- A2 (median) is deleted and must not appear.
- Tests first: A2'' independence (test 8), zero-signal degeneracy (test 9),
  verdict default on absence/exception (test 10).

## Task 5 — Wire into the GA final report (fixes B6)
- `research/ga_28c_30m_3y.py`:
  - compute the walk-forward table over train+validation only
  - lockbox per-leg excess diagnostic (F6) under a NEW top-level key
  - move `portfolio_sharpe` to `diagnostics.oos_portfolio_sharpe_exceeds_legacy_bar`
  - `acceptance.verdict` set from the gate; `live_adopted = (verdict == "pass")`
    DERIVED at the call site, not hardcoded
  - do not reuse the `history` key (N6)
- Tests first: derived `live_adopted` (test 11).

## Task 6 — Verification
- Full 28c suite green (test 15); `ACCEPTANCE_GATE_28C` consumers intact (test 16).
- Run the gate on the v3c winner; emit the per-fold table. **Report the verdict as an
  output. Do not adjust thresholds.** Tests 2/8/9 must pass first, since they are what
  make the verdict trustworthy.
- Run twice; assert byte-identical JSON (test 13).
- Codex read-only re-review of rev 2, then post-code review. Nothing is adopted unless
  the verdict is `pass` AND Codex returns PASS.
