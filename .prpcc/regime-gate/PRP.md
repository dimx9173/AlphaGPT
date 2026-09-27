# PRP: Regime-Neutral Acceptance Gate for 28c Strategies

Status: **REVISION 2** — written after Codex plan review returned CHANGES REQUIRED
Owner: Product Owner (root agent)
Date: 2026-09-25
Supersedes: revision 1 (CHANGES REQUIRED, 6 blocking findings B1-B6, 8 clarifications)

---

## 0. What changed in revision 2, and why

Codex found that revision 1 was unsatisfiable and, worse, that its central criterion
was unfalsifiable. Every fix below is a response to a specific finding. Where a fix
touches a frozen number, the number changed on a *principle stated here*, before any
new result was computed. No result under the new gate has been observed.

| finding | defect in rev 1 | fix in rev 2 |
|---|---|---|
| B1 | fold set was ambiguous; two readings disagreed; the stated coverage invariant was unsatisfiable | §4 F4 pins literal index ranges, 2 folds per contiguous block, embargo never inside a fold |
| B2 | fold duration arithmetic was wrong by 3x (said ~184 days, actual 552.5) | §4 F4 states real durations 220/220/52/52 days and re-justifies K=4 on that basis |
| B3 | **a 2x buy-and-hold scored active Sharpe 11.011 and passed A1 and A2 in every regime** | §3 F1 rebuilds the benchmark to be like-for-like at `LEV=2.0` with the same fee and funding model; §4 adds A5 requiring positive absolute Sharpe |
| B4 | A2 was provably implied by A1 (0 counterexamples in 200k draws) and passed on all-zero folds | A2 is **deleted**; replaced by A2' (absolute-Sharpe sign, duration-independent) and A2'' (duration-pooled excess) |
| B5 | N4 was unsatisfiable: `evaluate()` charges a phantom entry fee at every fold boundary (measured 0.00032) | §5 N4 permits a slice-based accounting entry point that bypasses the boundary charge |
| B6 | fail-closed was asserted but not specified; `live_adopted` was a hardcoded constant | §6 pins the verdict contract, the default, the JSON path, and requires `live_adopted` to be derived |

---

## 1. Problem Statement

The current gate is not measurable in a regime-neutral way, and it produced a
diagnosis that was wrong.

**v3c winner** `[7,3,12,7,12,5,27,1,9,19,22,28]` fails the gate: OOS Sharpe 0.478
(needs 1.0), 19/28 positive legs (needs 24). The first reading was overfitting.

**Equal-weight 1x passive hold, no cost, from the same loaded arrays:**

| split     | dates              | bench Sharpe | bench annual | coins positive |
|-----------|--------------------|--------------|--------------|----------------|
| train     | 2024-09 .. 2025-11 | +0.567        | +15.1%       | 68%            |
| validation| 2025-12 .. 2026-03 | **-1.311**    | **-66.9%**   | 7%             |
| lockbox   | 2026-03 .. 2026-09 | **+1.120**    | **+55.1%**   | 89%            |

The winner is a **~91% short** strategy (4.8% of gross notional long, stable across all
three splits). It wins structurally in the crash and lags in the rally: on the lockbox
it returned +8.8% against a +55.1% passive hold. The 10 negative lockbox legs are
exactly the majors that rallied.

`min_oos_portfolio_sharpe: 1.0` is an **absolute** bar. In the lockbox the market itself
scored 1.120, so the gate silently required a 91%-short strategy to beat a passive hold
in a rally. It measures regime, not skill.

---

## 2. Anti-Gaming Clause (binding)

1. Thresholds in §4 are frozen at spec time. Any change requires a new revision plus
   Codex re-review, justified by a principle stated independently of any result.
2. The new gate is explicitly permitted to **fail** v3c. A gate v3c passes is not
   evidence the gate is right.
3. v3c's verdict is reported as an **output**, with the full per-fold table. Thresholds
   are not adjusted afterward.
4. The lockbox is read once, after selection, and never participates in fold
   construction or in any blocking criterion other than A3 and A4, which are the
   pre-existing lockbox gates.
5. **Leverage-parity rule.** Any benchmark used by a blocking criterion must carry the
   same leverage and the same cost model as the strategy leg. Codex demonstrated that
   violating this admits a leveraged passive hold as a passing strategy (B3). This rule
   is binding and is testable.

---

## 3. Requirements

### F1 — Benchmark, like-for-like (replaces rev 1 F1)

The benchmark is a **passive equal-weight 2x hold over the same 28 coins**, run through
the *same* accounting path as the strategy: `LEV=2.0`, the same `FEE`, the same
scheduled funding at 00/08/16 UTC, the same position-extraction cost model, with
positions held constant at +1 for every bar (full investment, no signal).

Consequences, all intended:
- A static 2x buy-and-hold produces an active series of approximately zero and
  therefore **fails** A1. This is the specific defect Codex found in rev 1.
- A flat/cash "strategy" produces an active series of approximately minus the
  benchmark, which passes A1 in a down regime — and is then caught by A5, which
  requires positive absolute Sharpe.
- The benchmark therefore prices both leverage and the cost of being invested. It is
  no longer a free lunch.

Implementation must reuse `accounting_bar_returns` with a constant position array so
the cost and funding treatment is identical by construction, not by convention.

### F2 — Excess (active) return series (unchanged, confirmed sound by Codex)

`active[t] = strategy_portfolio_net[t] - bench_net[t]`, both from the shared
`equity-compound-v2` path. Metric is `daily_sharpe(active, ts)` with UTC-day grouping
and `sqrt(365)` annualization.

**Subtracting two Sharpes is forbidden.** It is not mathematically defined. Codex
verified no such subtraction exists in the current 28c path; that must stay true.

### F3 — Regime label (unchanged)

`up` if the benchmark's compounded return over the segment is positive, else `down`.
Depends only on the benchmark. Record `bench_annual`, `bench_sharpe`, `bench_max_dd`.
Codex flagged float-sign flapping as a hazard: at a benchmark return of exactly 0.0
the label must be `down` and must be deterministic, with no tolerance band.

### F4 — Walk-forward folds (fixed per B1, B2)

The search region is **not contiguous**. From `research/splits_28c.py`:
`train=(0, 21131)`, `validation=(21531, 26520)`. The gap is **400 bars**
(`EMBARGO_BARS=200` trimmed from each side), and a *second* 200-bar gap at
`[26520, 26720)` separates validation from the lockbox and is outside the region.

**The embargo may never appear inside a fold.** Therefore the only consistent reading
is two folds per contiguous block, four folds total:

| fold | bars         | duration | days |
|------|--------------|----------|------|
| 1    | `[0, 10566)` | 10566    | ~220 |
| 2    | `[10566, 21131)` | 10565 | ~220 |
| 3    | `[21531, 24025)` | 2494  | ~52  |
| 4    | `[24025, 26520)` | 2495  | ~52  |

**Coverage invariant (satisfiable form):** every bar in
`train ∪ validation` appears in exactly one fold, and no bar in
`[21131, 21531)` appears in any fold. `max(fold_hi) <= validation[1]` is asserted by
test, so the lockbox cannot be reached even if the function is later reused.

**K=4 re-justification (corrects B2):** the region is 26,520 bars = 552.5 days at
48 bars/day, not the ~184 days rev 1 claimed. The block split gives 220/220/52/52 days.
The 52-day folds yield ~52 daily observations — enough for `daily_sharpe`, but with wide
confidence, which is precisely why A2'' (duration-pooled) and A6 (regime coverage) exist
rather than relying on the short folds alone. K=4 is retained on that basis.

**Duration weighting is explicit and reported.** A median over four folds treats a
220-day fold and a 52-day fold as equal. That is why A2'' pools by bars, and why the
per-fold table must report each fold's bar count.

### F5 — Per-fold report (unchanged, plus duration)

Per fold emit: index, bar range, start/end dates, bar count, regime, strategy
absolute Sharpe, benchmark Sharpe, excess Sharpe, strategy MDD, positive leg count,
solvent, plus the strategy's net long/short notional share so short bias is visible.

### F6 — Per-leg lockbox excess (diagnostic only)

Per coin on the lockbox: Sharpe and excess vs benchmark, so the majors-concentration
failure mode is visible. Diagnostic only; gates nothing.

---

## 4. Acceptance Criteria (revision 2)

| id | criterion | threshold | rationale |
|----|-----------|-----------|-----------|
| A1 | folds with strictly positive excess Sharpe | **>= 3 of 4** | `min_positive_walk_forward_folds` already exists at 3; now implementable |
| A2'' | **duration-pooled** excess Sharpe over all four folds | **> 0.0** | not implied by A1 (duration weighting + one dominant negative fold can break it), and kills the all-zero degeneracy Codex demonstrated |
| A2' | folds with strictly positive **absolute** strategy Sharpe | **>= 3 of 4** | independent of A1; blocks the "flat while the market crashes" degenerate pass; supplies the absolute-skill evidence that A1 lacks |
| A3 | lockbox max drawdown | **< 0.25** | `max_oos_mdd`, unchanged |
| A4 | lockbox positive leg count | **>= 24** | `min_positive_coins`, unchanged |
| A5 | portfolio solvent on every fold and on the lockbox | **True** | existing solvency requirement, now checked per fold |
| A6 | regime coverage: at least 1 `up` and 1 `down` fold | else `insufficient_regime_coverage`, adoption blocked | guards against validating on a single regime |

**A2 (median excess Sharpe >= 0.0) is deleted.** Codex proved it implied by A1 and
satisfied by four dead folds. It is no longer cited as a defense for anything.

`min_oos_portfolio_sharpe: 1.0` becomes a **reported diagnostic**, not blocking.
Codex judged this defensible in principle but a disguised loosening as executed in
rev 1, because the replacement criteria could not separate skill from leveraged beta.
With F1 (leverage parity) and A2'/A2'' the replacement set can. This is a deliberate
spec change and requires Codex re-review.

Verdict vocabulary: `pass` | `fail` | `insufficient_regime_coverage`. Only `pass`
permits adoption.

---

## 5. Non-Functional

- **N1 Determinism.** No randomness. Byte-identical output across runs.
- **N2 No new data.** Benchmark derives from arrays already loaded.
- **N3 No gate tuning after observation.** §2.
- **N4 Accounting path (rewritten per B5).** `evaluate()` **may not** be used to build
  folds. It slices `pos[start:end]` and then calls `accounting_bar_returns`, which sets
  `prev_pos[0] = 0.0`, charging a phantom entry fee at every fold head — Codex measured
  the discrepancy at `0.00032`, exactly one `turnover * fee * leverage`. The
  implementation must instead compute net returns **once over the full aligned series**
  and slice, which is what `accounting_bar_returns` already supports when given the
  full-series `prev_pos`. A slice-based entry point must be added and unit-tested to
  prove the boundary charge is absent.
- **N5 Backward compatibility.** `ACCEPTANCE_GATE_28C` keys are preserved; new keys are
  additive. Three existing result JSONs already embed the gate dict
  (`results/train_12f_30m_28c_floor_best.json` and siblings) and
  `research/train_12f_30m.py` emits it verbatim — both must keep working.
- **N6 Dashboard safety (new, from Codex finding f).**
  `dashboard/visualizer.py:75-83` reads `plot_walk_forward` from a `history` dict with
  `step`/`sharpe` keys and returns an empty `go.Figure()` on missing keys rather than
  raising. The per-fold table must therefore use a **new top-level JSON key**, never
  `history`, and never a `plot_walk_forward` key.

---

## 6. Verdict Contract (new, fixes B6)

- The verdict is computed in exactly one place, `evaluate_regime_gate()` in
  `research/regime_gate_28c.py`, and surfaced at JSON path `acceptance.verdict`.
- **Default is fail-closed.** If the gate function is absent, raises, or returns an
  incomplete criteria set, the verdict is `fail` and the reason is recorded in
  `acceptance.verdict_reason`. It is never silently omitted.
- `live_adopted` **must be derived**: `live_adopted = (verdict == "pass")`, computed at
  the call site. A hardcoded `False` is not acceptable — it is fail-closed by accident,
  not by construction, and Codex flagged the current `ga_28c_30m_3y.py:212` constant for
  exactly this.
- `insufficient_regime_coverage` is a distinct value, not an alias for `fail`. It means
  the evidence is insufficient; it does not mean the strategy is bad, and it must not
  be reported as either pass or fail.
- **Key rename (Codex clarification 8).** The existing boolean
  `acceptance.portfolio_sharpe` is renamed to `diagnostics.oos_portfolio_sharpe_exceeds_legacy_bar`
  and moved under a `diagnostics` sub-object. Keeping the old key name while demoting
  the criterion would let existing readers interpret a diagnostic as a passing gate.

---

## 7. Test Plan

Unit
1. Benchmark is leverage- and cost-matched: a constant +1 position through
   `accounting_bar_returns` at `LEV=2.0` reproduces a hand-computed 2x net series
   including fee and scheduled funding.
2. **Leverage-parity regression (the B3 test).** A synthetic static 2x buy-and-hold
   strategy must produce `excess_sharpe` within tolerance of 0 on every fold, and must
   therefore fail A1. If this test passes with a large positive excess, F1 is broken.
3. Excess uses the difference series, not Sharpe subtraction.
4. Regime label at exactly 0.0 (must be `down`), tiny positive, tiny negative; no
   tolerance band; deterministic.
5. Fold set equals the four literal ranges in F4 exactly.
6. Coverage invariant: every train/validation bar in exactly one fold; no embargo bar
   in any fold; `max(fold_hi) <= validation[1]`.
7. **Boundary-cost test (B5).** Slicing a fold from a full-series net equals computing
   that fold standalone, for a synthetic case with no position change at the boundary.
8. **A2'' independence test.** A synthetic fold set with 3 positive and 1 large
   negative excess must satisfy A1 while failing A2''. Proves A2'' is not vacuous.
9. **Degeneracy test.** Four all-zero excess folds must fail A2''.
10. Verdict default: gate absent or raising yields `fail` plus a reason.
11. `live_adopted` is derived, not constant.

Integration
12. Run the gate on the v3c winner; emit the full per-fold table. Headline deliverable.
    Report the verdict as an output. Do not adjust thresholds.
13. Run twice; assert byte-identical JSON.
14. New top-level key, not `history`; dashboard import unaffected.

Regression
15. Existing 28c suite green: `test_accounting_28c`, `test_accounting_crosscheck`,
    `test_data_contract_28c`, `test_28c_acceptance_gate`, `test_paper_28c_pit_30m`.
16. `ACCEPTANCE_GATE_28C` consumers still work: `train_12f_30m.py` emission, the three
    existing result JSONs.

---

## 8. Risks

| # | risk | likelihood | impact | mitigation |
|---|------|-----------|--------|------------|
| R1 | gate reverse-engineered to admit v3c | medium | critical | §2; thresholds frozen; Codex re-review required; test 2 makes the leverage trap detectable |
| R2 | like-for-like benchmark is *too* strict and no strategy ever beats 2x passive | medium | medium | the gate is allowed to fail everything; a universal fail is a finding, not a reason to loosen |
| R3 | 52-day folds give wide Sharpe confidence | high | medium | A2'' pools by duration; per-fold bar counts reported; A6 blocks single-regime evidence |
| R4 | fold-boundary artifact | low | high | N4 slice-based path; test 7 |
| R5 | `live_adopted` stays accidentally hardcoded | medium | critical | §6 requires derivation; test 11 |
| R6 | existing readers misread a demoted diagnostic as a gate | medium | high | §6 key rename under `diagnostics`; test 16 |

---

## 9. Rollback

Pure added code plus one additive gate key and one derived `live_adopted`. Rollback is
deleting the new module and call site and restoring the single hardcoded `False`. No
data, split, strategy, or existing-JSON change is involved, so rollback is complete.

---

## 10. Open Questions for Review

1. Is 2x passive with full costs the right benchmark, or should the benchmark be a
   volatility-matched rather than nominal-leverage-matched hold? A vol-matched
   benchmark would be stricter still.
2. A2' requires positive *absolute* Sharpe in 3 of 4 folds. For a deliberately short
   strategy this is a real bar in a rally. Is it the right absolute anchor, or should
   the absolute criterion also be excess-based?
3. Should A6's `insufficient_regime_coverage` block adoption, or merely force a
   longer data requirement? Currently it blocks.

---

## 11. Sign-offs

- Planner (Owner): revision 2 drafted, all six Codex findings addressed
- Architect (Claude): pending
- Feasibility (Codex): **revision 1 CHANGES REQUIRED**; revision 2 re-review required
- Reviewer (Codex, post-code): pending
