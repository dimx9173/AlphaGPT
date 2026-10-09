## Round 3: settle the four disagreements, then freeze the spec

Both round-2 lists agree on the substance: temporal-stability (A), refit-agreement (B), style taxes (C), and the A1–B1 difference as the leakage meter. The four disagreements are real and decidable. Decisions below, each with the reason and the exact implementation.

### D1: include the negative control — N0, odd/even min

**Decision: in.** Without it, any gap-narrowing under a min-rule is uninterpretable: `min ≤ mean` always lowers levels, and the null arm has 3× the variance of real (1.86 vs 0.65), so a variance-sensitive penalty compresses the gap mechanically through the null falling faster. The control must add the *same min-strictness* with *no contiguous-regime robustness requirement*, so that co-movement under control and test diagnoses artefact rather than signal.

**N0 — "interleaved odd/even min (no-contiguity control)", axis A:**

Per (genome g, fold f), from the single shared train path used by S0/A1:

```
r      = stored train per-bar net returns, length T        # train-only, PRP-0
r_odd  = r[0::2] ;  r_even = r[1::2]
s_odd  = Sharpe(r_odd) ; s_even = Sharpe(r_even)           # same Sharpe() as S0
fit_N0(g,f) = min(s_odd, s_even)
fitness_N0(g) = mean over 5 folds of fit_N0(g,f)
```

Same slicing machinery as A1, same ~1.05x cost, but temporal contiguity is destroyed by construction: a genome whose edge is persistence across contiguous regimes passes A1 and fails N0; a genome that is merely low-variance at bar level passes both. TRUE predicts N0 does **not** compress the gap asymmetrically (gap stays within ±0.20 of −1.64, beat-rate ratio stays ≫20x); if N0 compresses alongside A1, the verdict is shared min-filter confound, not signal. Length-penalty was considered for this slot and rejected: it is a style tax (C) that changes *what wins* (short programs), so co-movement under it is confounded through the breadth/turnover channel; N0 keeps style fixed and isolates exactly the contiguity requirement.

### D2: reuse S0 conditionally — gate first, then pair

**Decision: reuse the 1000-run batch as the paired S0 control, subject to a pre-registered reproducibility gate.** Blind reuse risks comparing across code paths; blind re-run wastes 1× a demonstrated batch for nothing.

Gate, run before any arm:

1. Under the new tree, re-run S0 fitness on 40 pre-registered seeds (20 real + 20 null, first 20 per arm).
2. Paired difference `d = holdout-excess-Sharpe(new) − holdout-excess-Sharpe(stored)` per seed.
3. Reuse iff `|mean(d)| < 0.044` (0.5× the S0 gap SE of 0.088) **and** `sd(d) < 0.05` **and** paired `|t| < 2`. This is bit-reproducibility up to floating tolerance.
4. If the gate fails, re-run full S0 (adds 1.0×) and say so; all comparisons then run against re-run S0, still paired by seed.

If the gate passes, every comparison is paired by seed/fold/worker config (12 workers, 10 gens, pop 16, 5 folds, seeds 501–1000), which is the cheapest variance reduction available — worth more than any extra 50 seeds.

### D3: timescale — out of the evolution loop, in as free logging

**Decision: no horizon-sweep arms (scope stays), but the sweep is logged for free, and the license of a clean null is explicitly restricted.**

Rationale: a midpoint split tests persistence at half-train horizon only. A genuine edge living at, say, quarter-train regimes could fail A1 while existing. But each extra horizon as an evolved arm costs ~1× for a correlated verdict. The compromise: A1 (half) plus the fourth rule (cross-fold, see D4) already span two distinct horizons/scales, and every A1/N0 run additionally logs quarter-block Sharpes `q1..q4` from the same single path at ~0x. Offline, emulate finer horizons (min over 2, min/mean-sd over 4) on stored champion/top-8 paths. A clean all-null result is then licensed to mean exactly: **"no evidence of persistence at half-train to cross-fold horizons under these aggregations"** — it is *not* licensed to mean "no edge at any timescale." If the offline emulation shows separation growing monotonically with fineness, a horizon sweep becomes the pre-registered follow-up rather than today's cost.

### D4: the fourth rule — fold-median (FM) over four-block mean-minus-sd (A2)

**Decision: run FM; relegate A2 to a logged statistic.**

Both cost ~1×, so this is decided on orthogonality, not budget:

- A2 (four-block mean−sd) tests the *same fitted object* as A1 at a finer grain (T/4 vs T/2). Its verdict is correlated with A1's by construction, and its mean/sd decomposition can be recovered offline from the `q1..q4` logged under A1 at 0x. A separate A2 arm buys a correlated confirmation of a channel already measured.
- FM (median over the 5 fold-Sharpes) tests a channel **no A-rule can see**: cross-window agreement across refits. Each fold already refits scale/warm-up on its own train span, so FM is B-flavoured at 1.0× with zero new sims — it convicts exactly the "one lucky fold" vehicle, which matters because S0's aggregation (`mean` over folds) is maximally exploitable by single-fold luck.

What each would have distinguished that the other could not: A2 would have separated variance-driven from level-driven failure *within one fit*; FM separates fold-luck from within-window burst *across refits*. Given Fact 2 (real = broad books, not hot bars) and a 5-fold walkforward already in the design, fold-luck is the live untested channel. FM's per-generation effect cannot be recovered offline (evolution under median explores different genomes); A2's can. So FM gets the arm, A2 gets the log line.

---

## Final spec: 4 rules (S0 is the gated control, not counted)

Notation, fixed for all rules: per (g,f), one `evaluate()` yields train net-return vector `r` (length T, train-only). `Sharpe(·)` is the exact S0 function. `b(g,f)` = positive-coin count, `turn(g,f)` = turnover; logged, never in fitness (no breadth tax anywhere — any breadth co-movement is mechanism evidence, not fiat). Cross-fold aggregation is mean unless the rule says median. Selection operators (tournament/rank) unchanged; only the fitness scalar is replaced.

| # | name | exact fitness | axis | true cost | seeds/arm | TRUE predicts | FALSE predicts |
|---|---|---|---|---|---|---|---|
| A1 | Split-half min, shared path | per fold: `m=floor(T/2); s1=Sharpe(r[:m]); s2=Sharpe(r[m:]); fit=min(s1,s2)`; genome `mean_f(fit)`. Log `s1,s2,s=Sharpe(r),q1..q4`. | A | ~1.05x (one extra `metrics()` per half) | 500 real + 500 null, seeds 501–1000 | Real flat/up vs S0, null down; gap −1.64 compresses ≥0.40; beat-rate 70x→<20x | Both arms fall (`min≤mean`); gap within ±0.20 of −1.64; ordering real<null intact |
| B1 | Split-half min, scale-refit (pilot) | Same split, but per half recompute `scale_h=std(signal_h)` with own `scale_end_h`, reset smoother state at half start, re-apply tanh/cap-0.25/smooth(5)/lag-1, discard first 50 bars of each half; `fit=min(s1',s2')`; genome `mean_f`. | B (same genome, refit vehicle) | ~1.4–1.6x (features computed once; position/accounting legs redone per half — confirm on 10 seeds first) | 150+150 pilot, pre-registered first-150 seeds/arm; promote to 500×2 iff paired A1–B1 diff has `\|t\|>2` | B1≈A1 or stronger if A1's gain is real persistence; B1<A1 isolates the leakage portion. Either way A1−B1 is the leakage meter | Both fall further than A1; gap preserved or wider (real burstier) |
| N0 | Odd/even min control | per fold: `fit=min(Sharpe(r[0::2]),Sharpe(r[1::2]))`; genome `mean_f`. | A, no-contiguity | ~1.05x | 500+500, same seeds | **Must not** compress: gap within ±0.20, ratio ≫20x | Same (both fall symmetrically) — the diagnostic is co-movement with A1, see decision rule |
| FM | Fold-median | per fold `s_f=Sharpe(r)` as S0 (no slicing); `fitness(g)=median_f(s_f)`. | B-flavoured (cross-refit agreement) | 1.0x (re-aggregation of the same 5 numbers) | 500+500, same seeds | Null's one-fold-luck winners punished → null down, real flat; gap compresses; breadth logged to check vehicle | Symmetric fall; gap preserved |

Free offline reads (0x, not rules, from numbers logged above): A2-emulation (mean−sd over `q1..q4`) and F1 final-only re-rank (evolve under S0, re-rank stored top-8 by A1/FM score) to locate evolution-vs-final-pick bite.

Cap: nothing in the generation loop re-simulates a full path except B1's per-half position/accounting legs. Disjoint evolution stays out of scope.

## Decision rule (pre-registered, before any number)

Primary statistic is the **gap** (real-minus-null holdout excess Sharpe) with 95% CI, plus beat-rate ratio and breadth gap — never raw real level, since every penalty weakly lowers levels.

- **Recovering signal:** A1 and/or FM compress the gap by ≥0.40 (≈25% of −1.64, ≈4.5× gap SE) **with asymmetry** (real flat or up, null down) **and** beat-rate ratio compresses (70x toward <20x) **and** N0 does **not** compress (gap within ±0.20 of S0). B1≈A1 confirms negligible leakage; B1<A1 with residual (A1−N0) compression still counts as partial signal with leakage quantified. Breadth may co-move (no tax imposed); turnover audit must show the Sharpe move is not a pure cost-channel artefact.
- **Shared confound:** N0 compresses alongside A1/FM, or all penalties move the gap in proportion to arm variance without real-up/null-down asymmetry. Verdict: variance/min-filter artefact, not signal — regardless of how many rules "work."
- **Inconclusive:** (i) all rules within ±0.20 of −1.64 → licensed claim is only "no persistence at half-train to cross-fold horizons," not "no edge"; (ii) A1 moves but B1 pilot is underpowered (`|t|≤2`, wide CI) → leakage unseparated, promote B1 before concluding; (iii) per-generation vs final-only re-rank disagree → bite location ambiguous, follow-up required.

## Budget

Per full arm: 1000 runs × (10 gens × 16 pop × 5 folds) = 800k `evaluate()`s ≈ 1 S0-batch. Committed: A1 (1.05x) + N0 (1.05x) + FM (1.0x) + B1 pilot (0.3 × ~1.5x ≈ 0.45x) ≈ **3.55 S0-batches** (~2.8M eval-equivalents). If the D2 gate fails, add 1.0x for S0 re-run (≈4.55x). This does **not** fit one sitting at 12 workers if the S0 batch itself filled one — run it as three resumable chunks by arm (FM first at exactly 1.0x, then A1+N0, then B1 pilot), each checkpointed by seed.

Skipped: reading current fitness aggregation code to confirm S0=mean-over-folds (assumed above — verify at implement time). Risk: if S0 aggregates differently, FM's contrast changes meaning.
