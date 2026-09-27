# ARCHITECT REVIEW - PRP `regime-gate` revision 2

Reviewer: Architect (Claude). Scope: feasibility + hidden edge cases. I did not edit files.
I re-derived every empirical claim in the PRP from the repo data (`data/data_3y/30m`,
35553 bars, 2024-09-13 12:00 UTC .. 2026-09-24 04:00 UTC).

**Verdict: CHANGES REQUIRED.** Revision 2 fixed the six Codex findings, and F2/F5/F6/N6
are sound as written. But it inherits the *same class* of defect Codex flagged in B1: a
criterion that is **unsatisfiable in practice**. A6 can never be satisfied by any
strategy on this dataset, and F1's central "like-for-like" claim is false by a factor of
5.24. Both are load-bearing. Details below, with the numbers.

---

## 0. What I independently confirmed (the PRP's honest claims hold)

Reproduced exactly, no discrepancy:

| claim | PRP | mine |
|---|---|---|
| train bench Sharpe (1x, no cost) | +0.567 | +0.567 |
| validation | -1.311 | -1.311 |
| lockbox | +1.120 | +1.120 |
| lockbox annual | +55.1% | +55.1% |
| lockbox coins positive | 25/28 | 25/28 |
| v3c OOS portfolio Sharpe | 0.478 | 0.479 |
| v3c gross long notional share | 4.8% | 4.6% (95.4% short) |
| train/validation/lockbox index ranges | (0,21131)(21531,26520)(26720,35553) | identical |
| fold index ranges F4 | 4 literal ranges | reproduce; they tile train+validation exactly (26120 bars, 400-bar embargo excluded) |
| fold durations | 220/220/52/52 d | 220.1/220.1/52.0/52.0 d |
| B5 boundary charge | 0.00032 | = 1.0 x 0.0004 x 2.0, confirmed |

F4 is genuinely fixed. The coverage invariant is satisfiable, the embargo never lands
inside a fold, and the 3x duration error is gone. B1/B2/B5 are correctly resolved.

And the headline finding is confirmed, and it is worse than stated: **v3c passes A1, A2'
and A2'', and would be failed only by A6.**

```
FOLD  bars  regime  strat_sh  bench_sh  excess_sh  strat_mdd  solvent
   1  10566    down     0.705     0.098    -0.171      0.259  True
   2  10565    down     1.417    -0.613     0.119      0.145  True
   3   2494    down     2.601    -1.650     1.675      0.060  True
   4   2495    down     3.519    -2.494     2.729      0.067  True

A1  excess>0       : 3/4    PASS  (needs 3)
A2' abs sharpe>0   : 4/4    PASS  (needs 3)
A2'' pooled excess : +0.365 PASS  (needs > 0)
A3 lockbox mdd     : 0.097  PASS
A5 lockbox solvent : True   PASS
A4 lockbox leg+    : 19/28  FAIL  (needs 24)   <- pre-existing, unchanged
A6 regime coverage : 0 up / 4 down -> insufficient_regime_coverage
```

So the 2.2 clause ("the gate is explicitly permitted to fail v3c") resolves not because
the new skill criteria reject v3c, but because a **structural** criterion does. The skill
criteria the PRP spent B3/B4 fixing are, on this data, satisfied by the very strategy the
gate is meant to scrutinise. That is the finding the Product Owner needs.

---

## 1. Feasibility of F1-F6 and Tasks 1-6

| item | feasible? | note |
|---|---|---|
| F1 benchmark | **mechanically yes, semantically NO** | see 3.3. 5.24x exposure mismatch. |
| F2 excess series | **yes** | correct; I confirm no Sharpe-subtraction exists in the 28c path. |
| F3 regime label | **yes, but wrong target** | see 4.3 - label is a function of the cost model. |
| F4 folds | **yes** | verified above. |
| F5 per-fold report | **yes** | add `mean_abs_notional` (see 2.1). |
| F6 per-leg lockbox | yes | diagnostic only, fine. |
| Task 1 | yes | but ship both entry points, see 2.2. |
| Task 2 | yes | |
| Task 3 | yes | |
| Task 4 | yes | |
| Task 5 | yes | `ga_28c_30m_3y.py:212` hardcoded `False` confirmed. |
| Task 6 | yes | |

Nothing here is a "this cannot be built" finding. All of Tasks 1-6 are implementable in a
normal afternoon. The problems are in *what they compute*.

---

## 2. Recommended signatures and implementation order

### 2.1 Signatures

```python
# research/accounting_28c.py  (Task 1)
def net_series_full_then_slice(
    returns: dict[str, np.ndarray],
    coins: list[str],
    positions: np.ndarray,          # (n,) per-leg position, ALREADY lagged, FULL length
    start: int, end: int,
    funding_mask: np.ndarray,       # (n,) full length
    scale_end: int,                 # kept for signature parity; assert unused
    common: np.ndarray,             # (n,) full timestamps
) -> np.ndarray:
    # Net bar returns for [start,end) computed from the FULL-series prev_pos.
    # MUST NOT call accounting_bar_returns on the slice.
    # Assert len(positions) == len(common).

def passive_benchmark_net(
    returns: np.ndarray,            # (28, n)
    coins: list[str],
    start: int, end: int,
    funding_mask: np.ndarray,
    gross_notional: float = 1.0,    # <-- see 3.3; do NOT hardcode pos = +1
    fee_rate: float = FEE,
    leverage: float = LEV,
) -> np.ndarray:
    # Equal-weight passive hold at a MATCHED gross notional, same fee/funding path.
```

The `gross_notional` parameter is the whole fix for 3.3.

```python
# research/regime_gate_28c.py  (Tasks 2 and 4)
def regime_label(seg_net: np.ndarray) -> str:
    # 'up' iff float(np.prod(1.0 + seg_net)) > 1.0 else 'down'. Exact, no tolerance.

def excess_metrics(strategy_net, bench_net, ts) -> dict:
    # active = strategy_net - bench_net; daily_sharpe(active, ts). NEVER Sharpe subtraction.

def walk_forward_folds_28c(n: int, splits: dict, k: int = 4) -> list[tuple[int, int]]:
    # Literal F4 ranges. Takes the splits dict, asserts max(hi) <= splits['validation'][1].

def evaluate_regime_gate(
    strategy_net: np.ndarray,       # (n,) full aligned series
    bench_net: np.ndarray,          # (n,) full aligned series
    timestamps: np.ndarray,         # (n,)
    leg_nets: list[np.ndarray],     # 28 x (n,), for per-leg diagnostics
    coins: list[str],
    splits: dict,                   # from split_indices(); NEVER a raw end index
    folds: list[tuple[int, int]],
    gate: dict = ACCEPTANCE_GATE_28C,
) -> dict:
    # Returns {folds: [...], criteria: {A1, A2_prime, A2_dprime, A3, A4, A5, A6},
    #          verdict, verdict_reason, diagnostics: {...}}
```

Rule for `verdict`: compute it in exactly one place, and evaluate **A6 first** so that
`insufficient_regime_coverage` short-circuits. Otherwise a strategy that fails A4 *and*
lacks regime coverage reports a reason that misattributes the cause.

### 2.2 Implementation order (one change at a time, each independently revertible)

1. **Task 3 first, alone** (folds only, no metrics). Pure index arithmetic, no dependency
   on accounting, and tests 5/6 fail loudly if F4 drifts. Doing it first means the
   accounting changes land on a frozen, already-verified fold set.
2. **Task 1**, both entry points, tests 1/7 green. Keep `accounting_bar_returns` untouched
   and mark it *slice-only, deprecated for gate use* - that is what stops a future caller
   reintroducing B5.
3. **Task 2** primitives.
4. **Task 4** gate.
5. **Task 5** wiring - **last**, because it is the only task touching a file other modules read.

Do **not** merge Task 5 in the same commit as Tasks 1-4. `live_adopted` going from
hardcoded `False` to derived is the highest-risk edit in the cycle (R5) and should be
reviewable in isolation.

---

## 3. Mathematical correctness

### 3.1 Excess formulation (F2) - CORRECT

`active[t] = strategy_net[t] - bench_net[t]`, then `daily_sharpe(active, ts)`. This is
right. Differencing the **net return series** (not the equities) is the correct primitive
and it commutes correctly with the `equity-compound-v2` path. The prohibition on
subtracting Sharpes is correct, and I verified no such subtraction exists in the 28c path.

One caveat the spec does not state: because both legs are net of *their own* fees and
funding, the active series charges the strategy's costs **and** the benchmark's costs.
That is correct and intended - it prices the cost of being invested, per F1's own words.

**A2'' pooling is under-specified.** "duration-pooled" is not a formula. I implemented the
literal reading (concatenate the four excess slices, then one `daily_sharpe`) and got
**+0.3647**; pooling the *daily returns* and taking mean/std across all 547 daily
observations gives **+0.3656**. Close here, but the two definitions diverge in general and
the spec must name one. Recommend: pool daily returns, then mean/std. It is
duration-weighted by construction, immune to the concatenation seam, and needs no
monotonicity assumption.

### 3.2 The regime label (F3) - definition is fragile, and A6 is unsatisfiable

Two separate problems.

**(a) The label is a function of the cost model, not of the market.** The PRP's own
headline table is a **1x, no-cost** benchmark (+55.1% on the lockbox). F1's benchmark is
**2x, with fees and funding**. Different objects, and the regime label differs between
them on the *same data*:

```
lockbox 1x NO COST    total = +0.2490  -> up
lockbox 1x WITH cost  total = -0.0523  -> down
```

Costs alone flip the lockbox from `up` to `down`. F3 says the label "depends only on the
benchmark" - true, but the benchmark is cost-laden, so the "regime" is partly a measure of
how expensive the assumed trading was. This is the same conflation the PRP set out to
remove, one level down.

**(b) DECISIVE: A6 is unsatisfiable. All four folds are `down`. No strategy can pass.**

```
LEV  fold1 fold2 fold3 fold4
1.0  down  down  down  down
2.0  down  down  down  down
3.0  down  down  down  down      (tried 1.0 .. 3.0, and 1x-no-cost)
```

The search region (2024-09-13 .. 2026-03-06) is a **sustained bear market** for this
28-coin equal-weight basket. I searched for *any* window that could have been `up`:
- 220-day windows sampled every 500 bars across the whole region: **0 of 32 are up**. The
  best-aligned 220-day window anywhere in the region still returns **-16.1%**.
- 52-day windows: 27 of 97 up - but the two 52-day folds the PRP pins (3 and 4) are both
  down, and the two 220-day folds are both down.

So `A6 -> insufficient_regime_coverage` is a **constant of the dataset**, not a property
of any strategy. Every strategy, including a deliberately long one, gets
`insufficient_regime_coverage`. The gate can never return `pass` on this data.

This is architecturally identical to Codex's B1 finding against rev 1 ("the stated
coverage invariant was unsatisfiable"). Rev 2 fixed the *index* unsatisfiability and
inherited an *evidence* unsatisfiability.

Consequence for 2.2: the anti-gaming clause says the gate is allowed to fail v3c. But a
gate that fails **everything** is not a gate, it is a wall. R2 pre-registers "a universal
fail is a finding, not a reason to loosen" - I agree with that principle and I am **not**
asking to loosen a threshold. I am asking that A6 be **removed or re-derived**, because as
written it encodes a false claim ("this data spans two regimes") rather than a difficulty
bar. Those are different kinds of change, and only the first one is honest.

**Recommended (a design change, not a threshold change; must go to rev 3 + Codex):**
- Drop A6 as a *blocking* criterion. Record regime composition as a **reported property
  of the evidence** (`regime_coverage: {up: 0, down: 4}`), with an explicit note that this
  dataset cannot support a two-regime claim. Any conclusion drawn from it is a
  single-regime conclusion. That is a real, statable limitation.
- If a blocking regime criterion is genuinely wanted, it has to come from *new* data
  spanning an up leg. That is a data-contract change, not a gate change, and should be
  scoped as its own piece of work.

### 3.3 Is the leverage-parity benchmark the right construction? **NO.** Most serious finding.

2.5 states the leverage-parity rule as binding. F1 implements it as *"run the benchmark
through the same accounting path at `LEV=2.0`, with positions held constant at +1."*

Matching `LEV` is **not** matching leverage. Leverage in this codebase is
`position * LEV`, and the strategy never holds +1. From `ga_28c_30m_3y.evaluate()`:

```python
raw_pos = np.tanh(sig / fit_scale)        # in [-1, 1]
p = 0.25 * smooth_causal(raw_pos, 5)     # in [-0.25, 0.25]   <-- the 0.25 cap
p = np.roll(p, 1)
net = accounting_bar_returns(p, r, FEE, fnd, LEV)   # gross = p * r * LEV
```

The `0.25` scaling is inside the strategy and is not part of `LEV`. So:

| | per-leg notional | x LEV | gross |
|---|---|---|---|
| strategy (v3c measured `mean abs pos` = 0.1907) | 0.1907 | 2.0 | **0.381** |
| F1 benchmark | 1.0000 | 2.0 | **2.000** |

**The benchmark carries 5.24x the gross exposure of the strategy it is compared against.**
F1's claim that "a static 2x buy-and-hold produces an active series of approximately zero"
only holds if the *strategy* is also a static 2x buy-and-hold. The real strategy is not,
and cannot be - it is capped at 0.25 pre-leverage.

This matters because the excess series is then dominated by the benchmark's size, not by
the strategy's skill. In a bear region a heavily-leveraged long benchmark bleeds, so
*anything* less long looks like alpha. I measured the distortion by substituting an
exposure-matched benchmark (passive long at the strategy's own measured `mean abs pos`):

```
fold1: excess vs 2x-full bench  -0.171 | vs exposure-matched  +0.249   (delta +0.42)
fold2: excess vs 2x-full bench  +0.119 | vs exposure-matched  +0.825   (delta +0.71)
fold3: excess vs 2x-full bench  +1.675 | vs exposure-matched  +2.080   (delta +0.41)
fold4: excess vs 2x-full bench  +2.729 | vs exposure-matched  +3.102   (delta +0.37)
```

Every fold moves, and fold 1 **flips sign** (-0.171 -> +0.249). That is A1's deciding
fold. The choice of benchmark scale is not cosmetic - it changes which folds count as
positive. Under current F1, A1's 3/4 is decided partly by an artifact of the benchmark's
5.24x over-sizing.

**The correct construction is exposure parity, not `LEV` parity.** Same gross notional,
same cost model, same holding period. That makes the comparison a statement about
*direction and selection*, which is the strategy's actual job, rather than about *size*,
which is a modelling choice the PRP never asked the strategy to make.

This is exactly Open Question 1, and my answer is: **yes, move to a volatility-matched or
exposure-matched benchmark - the nominal-leverage-matched one is wrong, demonstrably so.**
A vol-matched benchmark is the better of the two (it normalises the risk budget, not just
the notional). Note this makes the gate *stricter* in some folds and looser in others;
2.1 forbids picking the variant after seeing which one helps v3c. So: **the Product Owner
must choose the variant, state the principle in rev 3, and freeze it before any number is
recomputed.** I flag that choosing it now, after I have shown fold 1 flips, is
uncomfortably close to the thing 2 prohibits. The clean path is to choose on principle
(vol-matching is standard practice for benchmark construction) and accept whatever the
number does.

---

## 4. Hidden edge cases

### 4.1 Every fold boundary is misaligned to a UTC day - and this is not cosmetic

Index 0 is 12:00 UTC, so UTC days begin at index 24 (mod 48). The four fold starts are 0,
10566, 21531, 24025, giving offsets 0, 6, 27, 25 (mod 48) against the required 24. **All
four folds open and close mid-day.** `daily_sharpe` groups by UTC date, so each fold
carries partial days that enter the mean and the std as if they were complete days:

```
fold1: 2 partial days, 54 bars (0.5% of fold); sizes [24, 30]
fold2: 2 partial days, 53 bars (0.5% of fold); sizes [18, 35]
fold3: 2 partial days, 46 bars (1.8% of fold); sizes [ 1, 45]   <-- a 1-BAR day
fold4: 1 partial day,  47 bars (1.9% of fold); sizes [47]
```

Fold 3 contains a **single-bar "day"**. A one-bar return has a completely different
variance profile from a 48-bar compounded return, and `daily_sharpe` puts it straight into
the same `np.std(ddof=1)`. On a 53-observation fold this is not a rounding detail.

I also checked the N4 boundary-fee fix: it is real and correct, but it is **~0.024 sigma
of a single daily return**. It matters for byte-identical accounting (test 13); it does
**not** matter for any verdict. R4 rates the fold-boundary artifact "low likelihood / high
impact" - the fee part is low impact, and the day-alignment part, which the PRP does not
mention at all, is the one that actually bites.

**Recommendation:** snap fold boundaries to UTC day boundaries. Fold 3's 1-bar day should
not exist. This changes fold bar counts and therefore F4's literal ranges, so it is a rev
3 change and must be re-reviewed - but it is a correctness fix, not a threshold change.

### 4.2 The 400-bar embargo hole

The hole is real and F4 handles it correctly (verified: the four ranges tile
train+validation exactly, 26120 bars, zero overlap, `[21131,21531)` untouched). Two
residuals:

- **The hole makes the folds non-contiguous, so "walk-forward" is a misnomer.** Folds 1-2
  and 3-4 are contiguous; folds 2->3 are separated by 8.33 days. Calling this walk-forward
  implies each fold is trained on all prior folds. Nothing in the spec re-fits between
  folds, so these are really **four disjoint evaluation blocks**. Say so in the report; the
  name sets an expectation the implementation does not meet.
- **The A2'' concatenation seam sits exactly on the embargo.** `np.concatenate` of the
  four slices puts bar 21130 adjacent to bar 21531 - 400 bars and 8.33 days apart. I
  checked: they are *not* the same UTC day, and the concatenated timestamps remain
  monotonic, so `daily_sharpe` does not compound across the seam today. **This is
  accidental safety, not a guarantee.** It holds only because the embargo happens to exceed
  the intra-day offset. If 4.1 day-snapping moves a boundary, it could stop holding.
  Pooling daily returns (recommended in 3.1) removes the failure mode entirely rather than
  relying on it.

### 4.3 Float-sign flapping at exactly zero

F3 pins "at exactly 0.0 the label must be `down`, deterministic, no tolerance band." That
is right and is implementable as `float(np.prod(1.0 + seg)) > 1.0`. Note that testing
`v > 0` on a *return* is the wrong comparison and gives `up` at 0.0 - the test plan should
assert the specific expression, not just the outcome.

Real residual risk, in order:
1. **Cost-driven sign flip (4.3a)** - dominates. Not a float issue, but the same
   "which side of zero" fragility, and it is data-dependent rather than float-dependent.
2. **Underflow.** `np.prod` over a long losing segment is multiplicative and can reach
   denormal/zero. I could not reach exactly 0.0 within 35553 bars (worst case ~2.5e-321 at
   1e6 bars), and underflow yields 0.0 -> `down`, which is the *correct* direction for a
   losing fold. So underflow is benign here. Do not add a tolerance band: a band would
   mislabel a genuinely flat fold as `up`, which is the failure that matters.
3. **Exact zero from a -100% bar.** `np.prod` of a series containing a single -1.0 return
   is exactly 0.0 -> `down`. Correct and deterministic. Fine.

Verdict: F3's rule is sound; the danger is the *benchmark definition* feeding it, not the
comparison. Fix 4.3a, keep the comparison exact.

### 4.4 Short-fold statistical confidence - real, but mis-stated

R3 rates this "high likelihood / medium impact". The actual numbers:

- Folds 3 and 4 give **53 and 52 daily observations**. `daily_sharpe` needs `d > 1`, so
  they compute, but the standard error of a Sharpe from N daily returns scales roughly as
  `sqrt((1+SR^2/2)/N)`. At N~52, a measured Sharpe of 2.7 has a standard error of roughly
  **0.85**, i.e. **+/-1.7 at two sigma**. A fold measured at +1.675 is not distinguishable
  from +0.0.
- A1 is a **3-of-4 sign test on noisy quantities**. Under the null (no skill) each fold
  has ~50% chance of positive excess, so a coin-flip strategy scores >=3/4 about 31% of
  the time. Three-of-four is a weak bar.
- The two 220-day folds have 221 daily observations each, so ~5x the precision. The
  effective sample is dominated by 2 of 4 folds.

So the short folds are not merely imprecise, they are **close to uninformative**, and A1
counts them with the same weight as the long ones. A2'' pooling is the right mitigation
and the PRP is right to include it - but note A2'' pools by bar count, and folds 1+2 =
21131 bars (**80.9%**) against folds 3+4 = 4989 bars (**19.1%**). A2'' is therefore ~81% a
measurement of the two long folds, and A2' likewise. The short folds can flip the A1 count
while barely moving A2''. That asymmetry is not stated anywhere and should be.

**Recommendation:** report a per-fold Sharpe standard error (or block-bootstrap CI) in the
F5 table. Cheap, honest for 52-day folds, and it changes no threshold. Note this
*weakens* the case for v3c, which is the right direction for an anti-gaming review.

### 4.5 Solvency of the benchmark itself

Not in the spec. I checked: the 2x benchmark is solvent on all four folds, but fold 2
contains a **-58.01% single bar** and ends at `final_x = 0.2771` (a 72% loss), with MDD
0.867. A benchmark that loses 87% peak-to-trough is a fragile yardstick, and the excess
series is measured against it. A5 checks the *strategy's* solvency per fold but says
nothing about the benchmark's. Add a benchmark-solvency assertion - free, and it would
catch a future regime where 2x is simply uninvestable.

---

## 5. Tests I would insist on

The PRP's 16 are a good list. These are the ones I would add, and two I would **replace**.

**Must add**

17. **Exposure parity (the 3.3 test).** Assert `mean abs bench gross notional` is within
    tolerance of `mean abs strategy gross notional` on every fold. Today this fails by
    **5.24x**. This is the test that would have caught F1. Without it, the 2.5 claim that
    the "leverage-parity rule is testable" is false.
18. **Benchmark scale is not free.** Fix a strategy, perturb the benchmark
    `gross_notional` by +/-25%, assert the *verdict* does not change. Today fold 1's excess
    Sharpe moves by 0.42 and **flips sign**, so this fails today and passes after a
    principled fix.
19. **A6 satisfiability.** Assert that at least one fold of the search region is `up`
    under the frozen benchmark, or that the gate reports `insufficient_regime_coverage`
    with a machine-checkable reason naming the region. A test asserting a property of the
    *data* is unusual, but here it is the only thing that catches "A6 can never pass."
20. **Fold/day alignment.** Assert every fold start and end lands on a UTC day boundary
    (`(index - 24) % 48 == 0`). Fails today on all four folds.
21. **No 1-bar days.** Assert every daily group inside a fold has >= 40 bars, or that
    partial days are excluded from the Sharpe. Fails today on fold 3.
22. **Regime label vs cost model.** Assert the label is unchanged when `FEE` is set to 0.
    This *documents* the 4.3a dependence rather than hiding it. If the Product Owner
    decides the label should be cost-invariant, this test pins that decision.
23. **Lockbox reachability by construction.** Assert the gate raises (or returns
    `insufficient_regime_coverage`) if handed splits whose `validation[1] > lockbox[0]`.
    F4 asserts `max(fold_hi) <= validation[1]`; also assert the benchmark and strategy
    series passed in are not longer than `validation[1]`.
24. **A4 must not be silently satisfied by the per-fold count.** Assert the A4 leg count is
    computed on the lockbox only. With 19/28 it fails today; a wiring bug that
    accidentally evaluated it on train (19/28 there too, coincidentally) would be invisible.
    Use a fixture where the two differ.

**Must replace**

- **Test 2 is vacuous as written.** "A synthetic static 2x buy-and-hold strategy must
  produce excess Sharpe ~0" - I built it and the max absolute difference from the benchmark
  is **exactly 0.0**, because under F1 the synthetic 2x hold *is* the benchmark. It
  compares the benchmark to itself. It cannot fail for any reason. Codex's B3 scenario was
  **a 2x strategy against a 1x benchmark**. Write that: build a 2x-hold strategy, evaluate
  it against a 1x benchmark, assert it is *rejected*. (With F1 fixed the 2x-vs-1x case is
  rejected - I confirmed a static hold fails A4 at 0/28 legs and A1 at 2/4.)
- **Test 9 (four all-zero excess folds must fail A2'')** is the right *shape* but the wrong
  *degeneracy*. I ran the real degenerate case: **flat/cash PASSES A2'' at +0.207.** Flat is
  not "all-zero excess" - it is exactly minus the benchmark, which in a bear region is
  strongly positive. The spec's own F1 bullet predicts this. A2'' alone does not kill
  flat; **A2' does (0/4).** Keep test 9 as a unit test of the arithmetic, but add a named
  test that a **flat strategy is rejected by the full criterion set**, and assert *which*
  criterion catches it, so a future reordering cannot silently disarm it.

**Also insist on:** the v3c per-fold table is committed as a **checked-in golden file**,
not just printed. 2.3 says thresholds are not adjusted afterward; a golden file makes that
mechanically enforceable - any threshold edit, fold change or accounting change shows up as
a diff someone has to justify in review.

---

## 6. Remaining ways this spec can be gamed

Ordered by how likely they are to actually be used.

1. **Benchmark-scale selection (highest risk, and it is live right now).** Because the
   benchmark is 5.24x oversized and the region is a bear market, the excess series is
   substantially "I am less long than a very leveraged long". Anyone choosing the benchmark
   variant *after* seeing v3c's per-fold table is choosing the yardstick that makes the
   contestant look good. 2.5 tries to prevent this by pinning the rule, but the rule it
   pins (`LEV=2.0` + `pos=+1`) is the wrong rule, and "revise the rule" is exactly the move
   2.1 constrains. **This is the hole.** Close it by fixing the construction on principle in
   rev 3, before recomputing anything, and by having Codex re-review the *construction*
   rather than the number.
2. **Single-regime evidence presented as regime-neutral.** With 0 up folds, any claim from
   this gate is a bear-market claim. A3/A4 (the lockbox, a +55% rally) are the only up-leg
   evidence and they are unchanged pre-existing criteria. The report must say "bear-market
   folds only" in the headline, not a footnote. A6 currently "protects" against this by
   blocking everything, which is not protection.
3. **A1's 3-of-4 sign test is coin-flip-grade.** A strategy with zero skill clears it ~31%
   of the time. A strategy that is *deliberately short* and nothing else - I ran
   `pos = -0.25` constant, no signal whatsoever - scores **A1 2/4, A2' 3/4, A2'' +0.150,
   A3 0.218**, and is stopped **only** by A4 (0/28 positive legs on the lockbox). It is
   stopped by a criterion that has nothing to do with the fold design. If the lockbox
   happened to be a bear market too, a pure-beta constant-short would pass the entire
   gate. **That is the gaming vector.** A pure-beta strategy must be rejected *by the
   walk-forward criteria*, not rescued from falling outside them.
4. **Leverage as a free parameter.** A strategy that reduces its own notional shrinks its
   absolute Sharpe, but the *sign* of its excess against a fixed oversized benchmark is
   unchanged. So there is no penalty for de-risking within the fold criteria. If `LEV` or
   the `0.25` cap is ever a tuned knob, the gate cannot see it.
5. **The `0.25` position cap is invisible to the gate.** F5 asks for "net long/short
   notional share" but not the *size*. A strategy that quietly scales to 5% notional has
   the same long/short share, the same sign pattern and a lower absolute Sharpe - so A2'
   would catch it, but nothing reports it. Add `mean_abs_notional` to F5 (2.1).
6. **Per-fold report as a soft launch surface.** A 12-column per-fold table invites a future
   reader to quote the best fold. Mitigation: the verdict is the only field anything should
   branch on, and `insufficient_regime_coverage` must not be rendered as a near-miss
   "fail 3.5/4" - it means *no verdict exists*, not *a bad verdict*.

---

## 7. Bottom line

**Sound and should be kept as written:** F2 (excess series), F4 (folds), F5, F6, N1, N2, N4
(the boundary fix - correct, though its impact is ~0.02 sigma, not the "high impact" R4
claims), N5, N6, 6 (the verdict contract; the `live_adopted` derivation requirement is
exactly right), and the whole of 2 as a discipline.

**Must change, in order:**

1. **A6 is unsatisfiable on this data** (3.2b). 0 of 32 sampled 220-day windows in the
   search region are up; the best-aligned one is still -16%. The gate can never return
   `pass`. Remove A6 as a blocker and report regime composition as a limitation of the
   evidence, or scope new data. This is not a threshold change and does not violate 2.
2. **F1's leverage-parity rule is the wrong rule** (3.3). `LEV=2.0` with `pos=+1` gives
   the benchmark **5.24x** the strategy's gross exposure, and fold 1's excess Sharpe
   **flips sign** when the benchmark is matched correctly. Move to exposure- or
   volatility-matching, chosen on principle and frozen in rev 3 before recomputation.
3. **Test 2 is vacuous** - it compares the benchmark to itself and cannot fail (5). Rewrite
   it as Codex's actual B3 scenario: 2x strategy against a 1x benchmark.
4. **Fold boundaries are not UTC-day aligned** (4.1), producing a 1-bar "day" in fold 3 that
   enters `daily_sharpe`'s std as a full observation. Snap to day boundaries in rev 3.
5. **A pure constant-short beta strategy is rejected only by A4**, not by any walk-forward
   criterion (6.3). It scores A2' 3/4 and A2'' +0.150 on folds alone. The fold criteria
   need to reject pure beta on their own, or the gate's integrity depends on the lockbox
   happening to be a rally.

**Explicitly not asking for:** any threshold to move. A1 >=3/4, A2' >=3/4, A2'' >0, A3
<0.25, A4 >=24 all stand. I am asking that the *evidence* be able to support them.

**One process note.** I computed v3c's per-fold numbers before writing this review, and I
am disclosing that: I could not assess feasibility of A1/A2'/A2'' without running them. Per
2.3 I have not proposed any threshold change on the basis of those numbers, and my five
items above are all *structural* (unsatisfiable criterion, wrong benchmark construction,
vacuous test, day-alignment bug, pure-beta hole) rather than numeric. The Product Owner
should read items 1, 2 and 3 as having been derived independently of the v3c result - items
1 and 4 are dataset and arithmetic facts that hold for any strategy, and item 2's 5.24x is
a code-reading fact from `evaluate()`. If the Product Owner wants these frozen without
v3c's table in view, the cleanest path is to have Codex re-review rev 3's *construction*
first, then run v3c once.

Per 2.2, whatever the verdict turns out to be, that is the answer - including `pass`. I
want to be explicit that if items 1-5 are fixed properly and v3c then passes, that would
be a genuine pass and should be accepted. My concern is the *construction*, not the
verdict.
