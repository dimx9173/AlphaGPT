# PRP `regime-gate` — Plan Review Outcome

Date: 2026-09-25
Phase reached: **plan** (code phase NOT entered)
Outcome: **STOPPED at plan review.** Two independent reviewers converged on a defect
that makes the requested goal unachievable with the current dataset.

## Reviews run
- Codex plan review (rev 1) -> **CHANGES REQUIRED**, 6 blocking findings B1-B6
- Architect review (rev 2) -> **CHANGES REQUIRED**, 5 blocking items
- Codex re-review (rev 2) -> **CHANGES REQUIRED**, B1-B4/B5/B6 resolved, new B7 + B8 + B9

## Resolution status of rev-1 findings
| id | status |
|----|--------|
| B1 fold geometry ambiguous | RESOLVED - four literal ranges, embargo excluded, invariant satisfiable |
| B2 fold duration arithmetic (3x wrong) | RESOLVED - 220/220/52/52 days at 48 bars/day |
| B3 2x buy&hold scored active Sharpe 11.011 | RESOLVED as specified - constant 1.0 long now yields excess exactly 0.000 on all folds, A1 fails 0/4 |
| B4 A2 vacuous (implied by A1, passed on zeros) | RESOLVED - A2 deleted, A2' and A2'' added, both proven non-vacuous |
| B5 N4 unsatisfiable (phantom fold-boundary fee 0.0008) | RESOLVED - slice-based path specified |
| B6 fail-closed unspecified | RESOLVED - verdict contract, JSON path, derived `live_adopted` |

## The blocking discovery (B7, independently confirmed by the architect)

**The gate cannot return `pass` for any strategy, ever.**

A6 requires at least one `up` and one `down` fold. The regime label reads the F1
benchmark, which is 2x with scheduled funding. At `LEV=2.0` the funding drag is
`3 x 0.0005 x 2 = 0.30%/day`, compounding to **-48.3% over a 220-day fold**. A market
must rise more than 48% across 220 days to earn an `up` label.

| fold | gross only | funding drag | net 2x | labelled |
|------|-----------|--------------|--------|----------|
| 1    | -0.0039   | +0.4833      | -0.4852| down     |
| 2    | -0.4632   | +0.4838      | -0.7229| down     |
| 3    | -0.1717   | +0.1445      | -0.2914| down     |
| 4    | -0.4333   | +0.1437      | -0.5148| down     |

Fold 1 is the clearest defect: the market was **flat (-0.4% gross) over 220 days** and
is labelled a *down regime* purely because the benchmark paid 48% in funding. The label
measures **leverage-induced cost, not regime** - the identical error the gate exists to
remove, reintroduced one level down through the F1 benchmark change.

The architect reached the same conclusion by a different route: across `LEV` 1.0, 2.0
and 3.0, **0 of 32 sampled 220-day windows in the search region are `up`**.

Under the original 1x no-cost benchmark the same folds give `[up, down, down, down]`,
so A6 was satisfiable and did real work. **The F1 change, not the fold change, broke it.**

## Secondary findings not yet addressed
- **B8 / architect item 2** — the F1 "like-for-like" benchmark is **5.24x larger** than
  the strategy's achievable exposure. `evaluate()` caps positions at
  `0.25 * smooth_causal(tanh(...))`, so max gross is `0.25 x 2.0 = 0.5x` equity, while
  the benchmark holds `1.0` = `2.0x`. Letter-compliant, substance-violating. It makes A1
  nearly free for low exposure: every beta in [0.05, 0.35] scores A1 = 3/4 with no skill.
  The full gate still rejects them (A2' 1/4, A4 10/28), so it is not independently
  exploitable - but A1's discriminating power is near zero.
- **B9** — test 2 uses a static 2x hold; the realistic trap in this framework is a
  constant `0.25` hold (the position cap).
- **Architect item 4** — no fold boundary is UTC-day aligned; fold 3 contains a **1-bar
  "day"** fed into `std(ddof=1)`.
- **Architect item 5** — pure-beta short is rejected only by A4, and flat/cash *passes*
  A2'' at +0.207 (it is minus the benchmark, not zero), so the rev-1 degeneracy test
  targets the wrong criterion; A2' is what kills flat.
- **C1** — the `portfolio_sharpe` rename splits the JSON schema across generations
  (13 existing `results/*28c*.json` files carry the old shape). No in-repo consumer
  reads the key today, so the practical risk is low, but the policy must be stated.

## What the reviewers found about v3c itself
Architect disclosure (it computed v3c's per-fold table before writing the review, and
proposed no threshold change): v3c **passes A1 (3/4), A2' (4/4), A2'' (+0.365), A3 and
A5**, and is failed only by A6 (unsatisfiable) and A4 (the pre-existing lockbox leg count
19 < 24). So the skill criteria do not reject v3c; the unsatisfiable one and the
pre-existing one do.

## Why this cycle stops here rather than revising to rev 3
Both reviewers independently flagged the same hazard, and the PRP's own section 2
forbids it: **choosing the benchmark scale after being shown that fold 1 flips sign
(-0.171 -> +0.249 under exposure matching) is exactly the observation-driven revision
the anti-gaming clause prohibits.** The architect raised this explicitly and offered to
accept whatever number results from a principled choice.

The honest position is that the requested artefact - a *regime-neutral* gate for this
strategy family - requires evidence the current dataset does not contain: the search
region is a sustained bear market with no up-regime fold to validate against. That is a
data limitation, not a specification bug, and no amount of further spec revision changes it.

## Options for the owner
1. **Accept the limitation, keep the gate as a wall.** Implement rev 2 with A6 demoted to
   a reported diagnostic. The gate then discriminates on A1/A2'/A2''/A3/A4/A5, which
   Codex's adversarial sweep shows is genuinely non-trivial. Honest, but the name
   "regime-neutral" must change: it would be an excess-return gate on single-regime data.
2. **Fix the label, keep A6 blocking.** Define the regime label on a 1x gross no-cost
   market proxy so it measures what the market did, not what leverage cost. Both
   reviewers recommend this. Risk: the architect measured that even at LEV 1.0 no
   220-day window in the region is `up`, so A6 may remain unsatisfiable anyway.
3. **Scope new data.** No valid regime-neutral verdict exists for this strategy family
   until a holdout spans an up regime. POL bounds the 28-coin span to 2024-09-13, so
   this means waiting, or reducing the universe to buy history.
4. **Fix only the unambiguous defects now** (B8 exposure parity, B9 test, fold/day
   alignment, C1 schema policy) and defer the regime question to whoever owns the data
   decision.

No code was written. No thresholds were changed. No strategy result was used to select
a threshold.


---

# ADDENDUM 2026-09-25 — the A6 blocker was WRONG. Gate implemented and working.

The plan review concluded that A6 was unsatisfiable and the gate could never
return `pass`. **That conclusion was wrong**, and it was traced to a specific
cause. This addendum records the correction so the earlier record is not taken
as the final word.

## What was actually true
The architect reported: "0 of 32 sampled 220-day windows in the region are
`up`; the best-aligned 220d window is still -16.1%", at LEV 1.0, 2.0 and 3.0.
Measured directly on the search region with the exact `load_data()` path:

    SEARCH REGION  2024-09-13 -> 2026-03-20
      333 non-overlapping 220d windows: 128 up (38.4%)  best=+56.6%  median=-14.1%
    FULL DATA      2024-09-13 -> 2026-09-24
      521 non-overlapping 220d windows: 140 up (26.9%)  best=+56.6%

The search region is **not** a uniform bear market. It contains 38% up windows.
The architect's measurement appears to have read the regime off the
cost-inclusive leveraged series — which is exactly defect **B7**, the one Codex
identified independently. Under a 1x gross proxy the four folds label:

| fold | compounded | regime |
|------|-----------|--------|
| 1    | +25.2%    | up     |
| 2    |  -5.4%    | down   |
| 3    |  -6.8%    | down   |
| 4    | -21.7%    | down   |
| lockbox | +24.9% | up     |

**A6 is satisfied: 1 up, 3 down.** Option 3 of the original decision (scope new
data) was therefore unnecessary. The defect was definitional, not empirical.

## Robustness of the label
Checked that the label does not depend on the aggregation choice. For fold 1:

| aggregation | value | label |
|---|---|---|
| mean of per-bar returns, compounded | +25.2% | up |
| mean of per-coin compounded         | +29.2% | up |
| median of per-coin compounded       |  +3.9% | up |
| winsorised 99% bar-mean             | +36.2% | up |
| winsorised 95% bar-mean             | +95.8% | up |

The one divergent figure, -80.4%, comes from taking a cross-coin median *per bar*
then compounding — that is not an equal-weight portfolio (it switches coin every
bar) and is not a meaningful aggregation. The label is robust to every
economically sensible choice.

## What was implemented (option 4 — the unambiguous defects)
- `research/regime_gate_28c.py` — `market_regime_proxy` (1x gross, label only),
  `passive_benchmark_net` (cap-matched, cost-matched), `regime_label`,
  `excess_metrics`, `walk_forward_folds_28c`, `coverage_report`,
  `evaluate_regime_gate`.
- `tests/test_regime_gate_28c.py` — 10 tests including the two that were vacuous
  before: B9 constant-cap-hold leverage parity, B8 exposure parity, midnight
  alignment, embargo exclusion, lockbox unreachability.
- **B7 fix**: regime label decoupled from the cost-inclusive benchmark.
- **B8 fix**: benchmark position set to `POSITION_CAP = 0.25`, matching the cap
  in `evaluate()`, so benchmark max gross is 0.5x not 2.0x. Derived from the
  strategy's own cap constant, not tuned.
- **Fold/day alignment fix**: fold bounds snap inward to 00:00 UTC. Cost: 200
  bars (0.77%) dropped from coverage — `uncovered_train: 107`,
  `uncovered_validation: 93`. Reported rather than hidden.
- **B5 fix**: net returns computed once over the full series and sliced, so no
  fold-boundary entry cost.

## Two real bugs found while implementing
1. **In my own earlier vectorisation of `daily_sharpe`** (not introduced by this
   cycle but uncovered by it): `cum[end-1]/before` yields `0/0 -> NaN` once
   equity is wiped out. The original loop reset `f=1.0` each day, so a wipeout
   did not poison later days. Rewritten to compound each UTC day independently
   from 1.0. Verified against the original implementation on 6 cases including
   exact wipeout, `r < -1`, and repeated wipeouts — all match exactly, and the
   real-data GA numbers are unchanged to 10 decimals.
2. **Funding mask vs rate**: the first gate run passed the 0/1 event mask into
   `accounting_bar_returns` instead of `mask * FUND_RATE`, inflating funding cost
   2000x. Caught because the output was absurd (strategy Sharpe 40.8 vs the
   GA's 1.01); fixed, and the gate now reproduces the GA's OOS MDD to 1e-15 and
   its leg count exactly.

## Result on the v3c winner
Formula `[7,3,12,7,12,5,27,1,9,19,22,28]`, 57s to evaluate.

| fold | days | regime | mkt ann | strat sh | excess sh | mdd | pos legs |
|------|------|--------|---------|----------|-----------|-----|----------|
| 1    | 219  | up     | +31.2%  | 0.820    | +0.302    | 0.259 | 21/28 |
| 2    | 219  | down   |  -4.9%  | 1.341    | +0.659    | 0.145 | 23/28 |
| 3    | 51   | down   | -42.4%  | 2.758    | +2.134    | 0.060 | 27/28 |
| 4    | 51   | down   | -84.1%  | 3.552    | +3.132    | 0.067 | 27/28 |

    A1  4/4 folds positive excess        PASS
    A2p 4/4 folds positive absolute      PASS
    A2pp mean excess +1.557              PASS
    A3  OOS MDD 0.0968 < 0.25            PASS
    A4  OOS 19/28 < 24 required          FAIL   <- pre-existing gate
    A5  all folds + OOS solvent          PASS
    A6  1 up, 3 down                     PASS

    VERDICT: fail  (A4 only)

The excess Sharpe rises monotonically as the market falls (+0.30 in a +31% rally
to +3.13 in an -84% crash). That is the 91% short bias showing up cleanly, and
the gate measures it as excess rather than being fooled by the regime — which
was the entire point of the rebuild.

## Standing conclusion
v3c fails the new gate on the **pre-existing** A4 leg-count criterion, not on
any of the new excess-return criteria. The new criteria pass it. No threshold
was changed after seeing this result, and the architect's disclosed pre-computed
per-fold table was not used to choose the benchmark variant — the cap was taken
from the strategy's own `POSITION_CAP` constant in `evaluate()`.


---

# ADDENDUM 2026-09-26 — gate wired into the GA, C1 schema migration complete

## C1: JSON schema migration
New module `research/acceptance_schema_28c.py` (schema_version 2).

    acceptance.portfolio_sharpe
      -> diagnostics.oos_portfolio_sharpe_exceeds_legacy_bar

Policy — no dual-write, no deprecated alias (a kept alias is exactly risk R6):
- **Writers** emit the new schema only.
- **Readers** use `read_acceptance(doc)`, which understands v1 and v2, never
  guesses, and is fail-closed: a v1 artifact is marked `legacy=True`,
  `schema_version=1`, verdict `fail`; a missing acceptance record is `fail`
  with a reason; a non-dict document is `fail`.
- `insufficient_regime_coverage` is a distinct verdict value, not an alias for
  `fail`, and never counts as adoption.
- An unrecognised verdict string degrades to `fail`. A writer cannot invent a
  passing state by typo.
- The fold table uses a NEW top-level key `regime_gate`, never `history` and
  never `plot_walk_forward` (N6: dashboard/visualizer.py:75-83 reads `history`
  expecting `step`/`sharpe` and silently returns an empty figure on mismatch, so
  reusing that name would hide the fold table rather than display it).

All shipped `results/ga_28c_*.json` still read cleanly and all read as
fail-closed legacy.

## GA wiring
- `live_adopted` is now **derived**: `derive_live_adopted(acceptance)`, i.e.
  `verdict == "pass"`. The hardcoded `'live_adopted': False` literal is gone.
  A source test asserts the literal is absent, so it cannot come back.
- The gate runs **once, at the end**, from the same `positions` dict `evaluate()`
  constructs. It cannot influence selection, and the lockbox reaches it only
  through the pre-existing A3/A4 criteria.
- Fail-closed by construction: the gate call is wrapped so any exception yields
  `verdict="fail"` with the reason recorded, `criteria={}`, `folds=[]`.
  Verified by injecting a `RuntimeError`.

## Verification
| check | result |
|---|---|
| 1-gen GA end-to-end | schema v2, legacy key gone, `diagnostics` present, `regime_gate` 4 folds, `live_adopted` False (derived) |
| determinism, two runs | byte-identical JSON |
| injected gate failure | `verdict=fail`, reason recorded, no pass |
| `history` key | untouched, dashboard unaffected |
| 28c suite | 70 tests pass |

`tests/test_acceptance_schema_28c.py` (15 tests) covers the rename, the
fail-closed default, the derived `live_adopted`, the R6 legacy-reader property,
and the N6 key collision.

## Pre-existing failure, NOT a regression
`tests/test_e10_gates.py::test_paper_baseline_within_tolerance` fails
(`552 != 478`). It reads a static **untracked** artifact
`results/paper_trades2.json` dated 2026-09-22, four days before this work. It
touches none of the modules changed here. Legacy E10 artifact drift; left
untouched rather than silently "fixed".


---

# ADDENDUM 2026-09-26 (2) — two silent accounting defects fixed

Both were silent: neither raised, neither produced an implausible number, and
neither was caught by the 43-70 test suite. They were found by reading the call
sites rather than running them.

## 1. `threshold_fit_28c_3y.py` modelled a funding-free perpetual
Line 57 passed `np.zeros_like(pos)` as the funding array:

    net = accounting_bar_returns(pos, r, FEE, np.zeros_like(pos), LEV)

`accounting_bar_returns` accepts a zero array without complaint, so every
threshold was fitted against **no funding cost at all** — 3 missed 8-hour
payments per day, every day. `scheduled_funding_rates` was already imported at
line 10 and never used, which is what made it easy to miss.

Fixed to `scheduled_funding_rates(timestamps[start:end], FUND)`.

Measured effect on one factor/grid point (HL_RANGE, train split):

| | Sharpe | MDD | net_sum |
|---|---:|---:|---:|
| funding ON (corrected) | -5.452781 | 0.999607 | -6.5254 |
| funding OFF (old)      | -5.942433 | 0.999786 | -7.1467 |

Not a rounding difference — a different strategy. Confirmed the schedule fires
on exactly 1321 train bars with values in `{0, 0.0005}`.

## 2. `train_12f_30m.py` reported a gate verdict it never computed
It emitted bare literals:

    "acceptance_passed": False,
    "live_adopted": False,

A reader sees "a gate ran and did not pass". Nothing ran. Worse, deriving
`live_adopted = (verdict == "pass")` here would be *semantically wrong*: this
trainer's reward still goes through `MemeBacktest`, which uses additive equity
and bar-frequency Sharpe, not `equity-compound-v2`. Its `score` is not a v2
metric, so no gate verdict can be derived from it.

Fixed to emit a structured fail-closed record from the C1 module, with
`acceptance_gate_run: False` and a `verdict_reason` naming the real cause.
Verified by smoke run: `acceptance_passed=False`, `live_adopted=False`,
`verdict=fail`, reason present, `schema_version=2`.

The MemeBacktest-to-`equity-compound-v2` migration itself is **not** done. It
is an architectural change to a separate RL trainer, not a defect fix, and it is
left explicitly open rather than half-done.

## Regression guards
`tests/test_accounting_consistency_28c.py` (6 tests) pins both, including
`test_funding_changes_the_result_it_is_applied_to`, which asserts a funded long
is strictly worse than the same leg unfunded. That is the assertion whose
absence let the zero-funding defect through.

28c-relevant suite: 76 tests pass.


---

# ADDENDUM 2026-09-27 — remote paper P&L was an accounting artifact. Real result is a loss.

## What the remote paper run was reporting
`frapc` `brian@178.105.232.221` ran `5 * * * *` hourly, writing
`results/paper_latest.json`. It reported:

    equity 1.000 -> 26.784   total_return +2578%   sharpe 12.433   mdd 0.188
    trades 19,990   accounting "additive_cumulative_pnl_paper2_compatible"

## That number was not real, and the tell was in the data
- All 28 legs had `net_sum` between **26.18 and 31.24** — a 17.8% spread.
  28 independent assets do not all gain ~2600% at the same magnitude.
- The portfolio total (+25.78) equalled **one whole leg**, not the mean of 28
  and not 1/28 of one. That is additive PnL double-counted across legs.
- The reported Sharpe (12.433) did not match the Sharpe implied by its own
  equity curve (7.233) — a 72% discrepancy inside one file.
- The source model's training score was **-4.557**. A negative-score model
  cannot produce +2578%.

## Root cause (local, not just remote)
`research/run_paper_28c_pit_30m.py` computed correct per-bar weighted **return
rates** into `portfolio_net`, then built equity with
`1.0 + np.cumsum(portfolio_net)` — additive accumulation of return rates, so 28
legs' returns were summed without reinvesting. The stats call had already been
migrated to the shared `metrics()`, so the file reported a compounded Sharpe
next to an additive equity curve. This was my own integration gap.

Fixed: equity now via `compound_equity(...)`, and the hardcoded
`"accounting": "additive_cumulative_pnl_paper2_compatible"` string replaced by
`ACCOUNTING_VERSION`. Guards in `tests/test_paper_equity_compounding.py`.

## The corrected result is a loss
    accounting    equity-compound-v2
    sharpe       -13.644
    total_return  -87.84%
    final_x        0.1216
    mdd            0.883
    per-leg        0/28 positive engine Sharpe, net_sum mean -7.85, spread 112%
    folds          -13.0 / -14.2 / -17.2 / -10.9

The 112% cross-coin spread replaces the old fake 17.8% uniformity. This agrees
in sign and magnitude with the source model's -4.557 training score.

**This artifact evaluates `train_12f_30m_28c_floor_best.json`, a candidate that
is not adopted and fails acceptance.** The authoritative 28c result remains the
v3c GA run (OOS Sharpe 0.478, 19/28 positive legs, verdict `fail` on A4).

## Deployment
- Backups: `/tmp/alphagpt_evidence/` (crontab, code, pre-fix artifact).
- Synced to remote: `accounting_28c.py`, `data_contract_28c.py`,
  `splits_28c.py`, `acceptance_schema_28c.py`, `causal_12f.py`,
  `universe_28c.py`, `run_paper_28c_pit_30m.py`, and
  `results/train_12f_30m_28c_floor_best.json`. Remote had been missing
  `accounting_28c.py` entirely and lacked the features 3/4/10/11 in
  `causal_12f.py`, so its runner could not even execute the current code.
- Verified identical to local across Python 3.14.4/numpy 2.5.3 (remote) and
  Python 3.11/numpy 2.x (local): sharpe -13.643801719249284 both, total_return
  -0.8783920668031533 both.
- Cron re-fired at 03:05 and produced the corrected numbers, confirmed by an
  independent read of `results/paper_latest.json`.

## Storage defect fixed
The full 6.3 MB ledger and 0.67 MB equity array were rewritten every hour with
identical content (frozen formula), reaching **584 MB** in `paper_runs/`. The
ledger is now capped to 500 retained rows (head+tail, so the sample still shows
the opening and the current state) while `trades`/`events` counts stay exact.
Per-run size 8.4 MB -> 0.91 MB. Pre-fix artifacts moved to
`results/paper_runs_pre_additive_bug/` (kept, not deleted — the +2578% number
should stay auditable).

## Safety
`live_adopted=False`, `orders_attempted=0`, `broker_imported=False`,
`network_access=False`, `state_mutated=False` throughout.
