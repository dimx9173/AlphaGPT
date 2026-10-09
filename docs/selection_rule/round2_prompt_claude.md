# Selection-rule experiment: shared context for the discussion

## Why this is the next question

Seven variables have been eliminated by experiment rather than by argument.
Search space (variable-length A/B), search effort (effort ladder), reward-vs-gate
alignment (pure excess), factor set (36f x 100), multi-timeframe (41f x 100), and
bar size (1h x 1000). None of them was the binding constraint.

The 1000-run batch is the one to start from. 500 real + 500 iid-null walkforward
runs, seeds 501-1000, 1h bars, 41 factors, reduced grammar, reward v3, train-only
selection, 10 generations, population 16, 5 folds, real Binance funding,
--scale-windows so factor horizons stay at their 30m values.

| arm | holdout excess Sharpe | sd | median | beat the hold |
|---|---|---|---|---|
| real (n=500) | -2.7429 +/- 0.0291 | 0.651 | -2.816 | 2 / 500 (0.4%) |
| null (n=500) | -1.1003 +/- 0.0832 | 1.860 | -0.992 | 138 / 500 (27.6%) |

Real minus null: **-1.6426 +/- 0.0881, t = -18.64**, 95% CI [-1.815, -1.470].

The median real run sits at the p16 of the observed null distribution. On data
with no edge by construction, this search beats a constant long 27.6% of the time;
on the real market it beats it 0.4% of the time. The ordering is inverted: more
structure in the data made the search worse.

## The three facts that do not fit "no edge"

1. Real data performs significantly WORSE than synthetic (t = -18.64).
2. Real data produces significantly BROADER books: 21.04 positive coins against
   13.96 on null, +7.08, **t = +24.47**.
3. Real data's beat-the-hold rate is 70x below the no-edge calibration.

A factor set with nothing in it would produce all three with the null's sign.
Instead the real arm responds to real structure, the response is breadth, and the
breadth does not survive contact with the hold.

## The hypothesis being tested

Selection is argmax on a single training window. A genome that fits that window's
noise is numerically indistinguishable from a genome that fits a real pattern, and
argmax prefers whichever fits the noise more tightly, because there is more noise
to fit. The negative gap is what that would produce.

The test: hold the factors, the grammar and the GA fixed, replace single-window
argmax with a selection rule that is harder to fool, and ask whether the gap moves
toward zero. If it does, the factors had something in them. If it does not, they
did not, and the negative gap is the honest answer.

## What is fixed and cannot move

- Universe: 28 Binance perps, the common intersection, 1h bars, 2.06 years
  (2024-09-13 to 2026-10-04), 18026 bars. Buy-and-hold on the holdout is +3.106.
- Features: 41, exactly as shipped. Indices 0..11 must keep their meaning.
- Grammar: the reduced set, 11 operators at tokens 41..51.
- Evaluation: the live position path, causal scale, tanh, POSITION_CAP 0.25,
  smooth_causal(5), lagged one bar, same accounting including real funding.
- Selection data: train-only. The validation and lockbox windows stay observations
  and must never reach the selector (PRP-0, already enforced).

## What may change

Only how a genome's training fitness is computed and ranked. Concretely, the
candidates below. The evaluation above is the expensive part and stays fixed, so
every rule is scored on the same folds.

## Data available

Per coin per bar: timestamp, open, high, low, close, volume. Sparse 8-hourly
settlement funding, loaded separately. No order book, no open interest, no
cross-asset series, no point-in-time fundamentals. A factor needing any of those
is a silent placeholder and has bitten this project before.

## Your task

Design the selection-rule experiment. Three rounds, as before: propose
generously, then vet, then decide.

Round 1: propose candidate selection rules. For each, state the exact fitness
computation, what it costs relative to the current one, and the specific failure
mode it is supposed to prevent. Include the rules that are known to be bad and say
why, so they are not re-proposed. Be concrete about how each interacts with a
population-based GA, because a rule that needs per-genome cross-validation inside
a generation loop may be too expensive to run at this scale, and that is a real
constraint rather than an implementation detail.

Answer in prose and tables. Do not run code. Do not edit files.

---

# The other participant's round 1

**Selection Rules - Round 1 Proposals**

Current baseline for reference: per genome, fitness = train-window excess Sharpe of the live position path, computed on train bars only, averaged (?) over 5 folds, then GA does tournament / argmax every generation for 10 generations x population 16, and final champion = argmax train fitness. Everything below keeps factors, grammar, accounting, folds fixed. Only the scalar that the GA ranks changes.

## Candidate Rules

| ID | Name | Exact fitness computation | Cost vs current | Failure mode it prevents |
|---|---|---|---|---|
| R1 | **Train-Halves Min** | Split each train window into first-half / second-half by bars. Compute train Sharpe `S1`, `S2` on identical position path restricted to each half. Fitness = `min(S1, S2)`. Final champion ranked by same. | **2x** train evals. Same path, two aggregations; no extra simulation if Sharpe is computed from stored bar PnL. ~2x aggregation, ~1x simulation. | Winner's curse on single-regime fit. Prevents genome that is great on months 1-6 and blows up on months 7-12 from winning. |
| R2 | **Train-Halves Harmonic / Rank-Product** | Same `S1`, `S2` as R1. Fitness = `2*S1*S2/(S1+S2)` if both > threshold, else `-inf` / large penalty. Variant: rank genomes by `S1` and `S2` separately, fitness = product of ranks. | **2x**, same as R1. Rank-product adds only a sort per generation. | Same as R1 but softer. Min is brittle when one half is flat market; harmonic / rank-product still requires both halves decent without letting one outlier half dominate. |
| R3 | **Interleaved Odd-Even Stability** | Split train bars by even / odd index (or by week parity). Compute `So`, `Se`. Fitness = `mean(So,Se) - lambda*abs(So-Se)`, `lambda=1.0` fixed a priori. | **2x**. | Prevents bar-timing luck: genome that exploits 2-3 specific events (funding spikes, one liquidation cascade) scores high on contiguous split by accident. Interleaved split destroys event-localization; only persistent cross-bar patterns survive. |
| R4 | **Fold-Median + Dispersion Penalty** | Current design already has 5 folds. Instead of `mean(S_fold)`, fitness = `median(S_fold) - k*IQR(S_fold)` or `median - k*sd`, `k=0.5` fixed. Rank on that. | **1x**. No extra evals, only changes aggregation over the 5 existing fold scores. Cheapest rule. | Prevents fold-cherry-picking: genome that is +3 on one fold and -2 on four folds currently averages out to mediocre-to-good; median+penalty kills it. Directly tests whether 5 folds are being used or wasted. |
| R5 | **Breadth-Capped Fitness** | Compute standard train Sharpe `S` plus `Npos` = number of coins with positive train contribution. Fitness = `S - c*max(0, Npos - Ncap)`, e.g. `Ncap=8`, `c=0.2` Sharpe points per excess coin. Or hard cap: genomes with `Npos > 12` ineligible for parent selection. | **1x**. `Npos` is free byproduct of existing per-coin accounting. | Prevents breadth mining: the observed real-arm pathology (+7.08 coins, t=+24.47). Tests whether breadth is the mechanism of the negative gap. If breadth is how noise is fit, penalizing it should move the gap. |
| R6 | **Turnover / Churn Penalty** | Fitness = `S_train - d*Turnover`, where `Turnover` = mean absolute bar-to-bar position change after `smooth_causal(5)` and lag, averaged over coins. `d` calibrated once from null arm so median penalty ~= 0.3 Sharpe points. | **1x** plus trivial arithmetic on already-materialized position path. | Prevents high-churn noise harvesting: genome that flips fast to catch microstructure noise looks like Sharpe on train before costs/funding fully bite. Distinct from breadth; attacks the time dimension rather than cross-section. |
| R7 | **Two-Stage Shortlist Rerank** | Stage 1: rank generation by plain `S_train`, keep top-4 of 16. Stage 2: rerank only those 4 by `min(S1,S2)` (R1) or fold-IQR. Parents drawn from Stage-2 order. Final champion from Stage-2 winner. | **~1.25x**. Only 4/16 genomes pay the second eval per generation. `(12*1 + 4*2)/16 = 1.25x`. | Prevents paying stability cost on hopeless genomes while still blocking a brittle champion. Practical compromise between R1 cost and R4 cheapness. Tests whether selection pressure needs to act all generation or only at the top. |
| R8 | **Final-Pick-Only Stability (Search Fixed)** | GA generational selection unchanged (plain `S_train`). Only the final champion choice among the 160 visited genomes uses `min(S1,S2)` or fold-median. No change to evolution trajectory. | **~1x + epsilon**. Only ~160 stored paths per seed rescored once at end. Negligible. | Isolates mechanism: distinguishes "search was steered into bad region" from "search found good region but picked wrong champion." If R1 moves gap but R8 does not, the damage happens during evolution, not at final argmax. Essential control. |
| R9 | **Bootstrap-Median Fitness** | Within train only, draw `B=5` block-bootstraps (contiguous 4-week blocks resampled). Fitness = median Sharpe over bootstraps. | **6x** (1 original + 5 resamples). Most expensive honest rule. | Prevents single-path dependence: the most direct estimate of "would this genome have been picked on a slightly different history." Gold standard for selection-bias measurement, priced accordingly. |

Notes on GA interaction, common to all:

- Rules R1-R4, R7 change **per-generation parent selection**, so they change the evolutionary trajectory, not just the trophy. Expect slower convergence and lower train fitness by construction. That is the point; do not compare train fitness across rules.
- Rules R5-R6 are **penalties on free byproducts** (`Npos`, turnover). They act as regularizers inside tournament selection with zero extra simulations. They are the only rules that directly target the two observed real-arm symptoms (breadth, and by hypothesis churn).
- Rule R8 deliberately does **not** touch per-generation selection. It is the cheap ablation that tells whether final-argmax alone explains the -1.64 gap.
- Rule R9 is evaluated per genome per generation, so its 6x multiplies the innermost loop. All others reuse the already-simulated position path and only re-aggregate.

## Known-Bad and Close-To-Useless Rules

- **Train+validation blend, validation rerank, or any lockbox peek.** Breaks PRP-0. Guaranteed to raise Sharpe on this batch and prove nothing. Includes "select on train, break ties on validation" and "filter champions by validation Sharpe > 0." Do not propose again.
- **Increasing generations 10 -> 50 or population 16 -> 64 under same argmax.** More argmax draws on the same window. Under the stated hypothesis this widens the negative gap, at higher cost. It tests effort, which the effort ladder already eliminated.
- **Plain profit, return, or win-rate fitness.** Strictly noisier than Sharpe on the same bars, more outlier-driven, easier to fool. Moves gap the wrong way.
- **Complexity / expression-size penalty alone.** Grammar is already reduced to 11 operators. Size is not the observed pathology; breadth is. A parsimony penalty without a breadth term will shrink trees while keeping 21-coin books intact.
- **Averaging fitness across seeds.** Seeds are independent searches, not genomes in one population. Cross-seed averaging is an analysis step, not a GA selection rule, and cannot be computed inside a generation loop.
- **Per-genome iid-null calibration (simulate null bars per genome, keep Sharpe minus null mean).** Correct in theory, ~50-100x cost in practice, and redundant: the 500-run null arm already provides the calibration at the batch level. Batch-level null comparison dominates per-genome nulls on cost.
- **Ensemble-of-population as champion.** Averaging positions across top-k genomes changes the evaluation operator, which is fixed. It may help in production but does not test the selection hypothesis on the same folds.
- **Reward v4 / new cost model as "selection rule."** Reward is fixed at v3 for this experiment. Changing accounting confounds selection with evaluation.

## Affordability at 500 Seeds x 12 Workers

Budget logic: current cost per seed ~= `16 pop x 10 gen x 5 folds = 800` train evals. Per arm (500 seeds) ~= 400k evals. Two arms per rule (real + null, mandatory, otherwise gap is uninterpretable) ~= 800k evals. That is one **unit**.

- **Affordable without discussion:** R4, R5, R6, R8 at **1x**. Same wall time as the 1000-run batch. Run all four.
- **Affordable, the default workhorse:** R1, R2, R3 at **2x**, R7 at **1.25x**. Each is 1-2M evals for both arms. On 12 workers this is roughly 1-2x the wall time of the reference batch. Run R1 and R7 for sure; R2/R3 pick one (R3 is more informative because it is orthogonal to R1).
- **Borderline, run once if budget allows:** R9 at **6x** (~4.8M evals for both arms). Only affordable as a small-seed pilot (e.g. 100+100 seeds) to calibrate whether R1/R4 already capture most of the bootstrap benefit. Not affordable at full 500x2 without displacing everything else.
- **Not affordable, do not propose for full scale:** anything requiring per-genome cross-validation inside the generation loop with `K >= 3` inner folds (9x+), per-genome null simulation (50x+), or nested GA restarts. Those belong in a 50-seed pilot at most.

Recommended Round-2 shortlist on cost grounds: R4 + R5 as the 1x tier, R1 + R7 as the 2x tier, R8 as the control. Five rules x 2 arms, with R8 nearly free.

## What Each Outcome Means

The estimand is always the **real-minus-null gap** on holdout excess Sharpe, plus the two secondary signatures (breadth `Npos`, beat-the-hold rate). Train Sharpe is not an outcome; every stability rule lowers it by design.

| Rule | If hypothesis TRUE (factors carry real effect, argmax hides it) | If hypothesis FALSE (no real effect, negative gap is honest) |
|---|---|---|
| R1 / R2 / R3 (split stability) | Gap moves toward zero (-1.64 -> e.g. -0.8 to -0.2). Real-arm Sharpe rises, null-arm Sharpe flat or falls (null has no persistent pattern to be consistent across halves). Breadth falls on real, beat-hold rises from 0.4% toward low single digits. R3 moves less than R1 if effect is regime-localized. | Gap stays ~= -1.6 or goes more negative. Both arms fall together or real falls more (stability filter just adds selection noise on top of no signal). Breadth may fall but Sharpe does not rise with it. Beat-hold stays near 0.4% real vs ~20%+ null. |
| R4 (fold-median) | Smallest positive move. Gap narrows modestly. Discriminates weakly because 5 folds already exist; if folds overlap heavily in time, median ~= mean and little changes, which itself is informative. | No move. Cheapest falsification: if even free aggregation change does nothing, argmax-over-folds is not the lever. |
| R5 (breadth cap) | Gap narrows **and** breadth gap (+7.08) collapses, with real Sharpe rising. That combination is the smoking gun that breadth was the noise-fitting channel. | Breadth gap collapses but real Sharpe flat or worse. Proves breadth was load-bearing for train Sharpe with no holdout value. Negative gap persists at lower breadth. |
| R6 (turnover penalty) | Gap narrows only if the real effect is slow (multi-hour persistence). Null arm, which harvests churn noise freely, falls more than real. | Both arms fall equally or real falls more. Turnover was not the channel; cross-sectional breadth was. |
| R7 (shortlist rerank) | Captures ~70-80% of R1's gap movement at 1.25x cost. If R7 ~= R1, damage is done at the top of the population (champion choice), not throughout evolution. | R7 ~= baseline. Same conclusion as R1-false but cheaper to establish. |
| R8 (final-pick-only) | Gap moves about half as far as R1. Difference `R1 - R8` measures how much evolution itself was mis-steered vs. just the final trophy. If `R8 ~= R1`, the search found good genomes and picked badly; if `R8 ~= baseline << R1`, the search never visited good genomes under argmax pressure. | No move. Confirms neither trajectory nor final pick contains recoverable signal. |
| R9 (bootstrap median) | Largest gap movement, upper bound on what any stability rule can recover. Sets ceiling for R1/R4. | No move at 6x cost. Definitive: even the gold-standard stability rule cannot manufacture edge. Stop the selection line of inquiry. |

Decision rule for Round 3: a rule **supports** the hypothesis only if the real-minus-null gap moves toward zero by a statistically clear margin (t on the gap difference, same +-0.09 scale as the reference batch) **with** the move coming from the real arm rising and not just the null arm falling, and breadth / beat-hold moving in the predicted direction. A rule that lowers both arms equally, or narrows the gap only by dragging null down, does not support it.

---

# A measurement from the source code, found after round 1 closed

## A measurement that changes the cost model, found while round 1 ran

`evaluate()` computes `pnet = np.mean(np.stack(leg_net), axis=0)` ONCE per
(genome, fold) and then calls `metrics(pnet, timestamps)` to get a Sharpe. The
per-bar portfolio return series is already in hand.

So a selection rule that splits the TRAIN window by time and scores each piece is
not 2x. It is close to 1x: the expensive part is the simulation, and the
simulation is shared. Splitting costs a second `metrics()` call on a slice.

That reclassifies most of the round-1 proposals from "2x, think about the budget"
to "affordable at full scale". It does not make R1-R3 free, because the slices
must be scored and scored Sharpes are not independent, but the multiplier is
dominated by the simulation that is not repeated.

## The subtlety this exposes, which matters more than the cost

Scoring sub-slices of one shared path is NOT cross-validation.

The position path is `POSITION_CAP * smooth_causal(tanh(sig / scale), 5)` lagged
one bar, where `scale` is the standard deviation of the signal fitted over the
whole train window up to `scale_end`. A genome's path on sub-slice 2 was therefore
computed with knowledge of sub-slice 1 through the fitted scale and through the
smoothing that carries state across the boundary.

What a time-split rule actually measures is therefore **temporal stability of one
fitted object**: does the same genome, fitted once on the whole training span,
still perform on each part of it. That is a real and useful property, and it does
penalise a genome that only works in one regime. It is not the same thing as
refitting per window and asking whether the refits agree, and a proposal that
claims the latter while only doing the former is overstating it.

The distinction matters for interpretation. Refit-agreement is the stronger test
and it costs a genuine multiplier, because the simulation must be repeated per
window. If the experiment is meant to distinguish "the factors had a real effect"
from "argmax fit one window's noise", the stronger test is the one that answers it.

---

# ROUND 2 (respond to this)

You have seen the other model's round-1 list. Two things changed since round 1:

1. The cost model above. Most time-split rules are NOT 2x.
2. The distinction between scoring sub-slices of one shared path and genuinely
   refitting per window. Say which of your candidates is which, and correct any
   claim in your round 1 that overstates it.

Do this:

## 1. Map your candidates onto theirs, and onto the two tests

For each of your candidates, name the closest candidate in the OTHER list, and
whether it is the same test under a different name, a stricter version of the
same test, or genuinely different. Collapsing 17 candidates that are 4 ideas
matters more than choosing among 17.

Then place every candidate on one axis:

    A: scores sub-slices of one shared path   (temporal stability of one fit)
    B: refits per window and requires agreement (true out-of-sample consistency)

Say which axis each is on, and which candidates straddle both.

## 2. Cross-audit

Name, specifically, any candidate in the other list that you think is (a) a
duplicate of one of yours, (b) already covered by the null calibration, (c)
dominated by another candidate, or (d) going to move the gap for a reason that has
nothing to do with overfitting.

That last one is the important one. Any rule that changes the gap by changing what
the fitness rewards, rather than by making the fitness harder to fool, will look
like a result and mean nothing. Say which candidates are at risk of that.

## 3. Which 3-4 rules to actually run, and the design

Pick at most four. For each: the exact fitness, its true cost multiplier under the
measurement above, and what it predicts under TRUE and under FALSE.

Then state the design: how many seeds, which arms, and what the decision rule is.
Be concrete about how many rules can be afforded at full 500x2 scale and which
must be a pilot.

## 4. The failure mode of the whole experiment

If every rule moves the gap, what does that mean? If no rule moves it, what does
that mean? State the interpretation that would make this experiment uninformative
even if it runs cleanly, and say what would have to be added to catch it.

Do not run code. Do not edit files. Tables and prose only.
