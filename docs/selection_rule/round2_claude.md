## 0. Collapse first: 17 candidates are 4 ideas

The other list's R1–R9 are not 9 ideas. They are:

- **Idea I — temporal consistency:** R1, R2, R3, R9 (all split train time and demand agreement).
- **Idea II — cross-fold aggregation:** R4 (change mean to median over the 5 existing folds).
- **Idea III — penalise an observed pathology:** R5 (breadth), R6 (turnover). These change *what* is rewarded.
- **Idea IV — scope of application:** R7 (stability only at top-4), R8 (stability only at final pick). These are not new fitnesses, they are *where* Idea I/II is applied.

My Round-1-equivalent set, stated now so the mapping is explicit:

| Mine | Content | Closest other |
|---|---|---|
| M1 halves-min | per fold, split train in half, fitness = min(half Sharpes) | R1 Train-Halves Min — **same test, same name** |
| M2 fold-median | fitness = median over 5 folds, no penalty term | R4 Fold-Median — **same test minus the penalty** (I argue the `-k·IQR` should go) |
| M3 refit-halves | same as M1 but refit scale + reset smoother per half, re-simulate | R1 as *claimed* ("would this genome have been picked on a slightly different history") — **stricter version of R1** |
| M4 final-only control | run search on baseline, apply M1 only to pick champion from visited set | R8 — **same test** |
| M5 pathology penalty (rejected) | breadth or turnover penalty | R5 / R6 — **same test, and I reject it** (see §2d) |

R2 (harmonic/rank-product), R3 (odd-even), R7 (shortlist), R9 (bootstrap) have no separate counterpart in mine because they are dominated variants of M1/M3 — details in §2c.

## 1. Map onto the two axes

```
A: scores sub-slices of one shared path (temporal stability of one fit)
B: refits per window and requires agreement (true out-of-sample consistency)
```

| Other candidate | Axis as written | Why |
|---|---|---|
| R1 Train-Halves Min | **A** | Path simulated once with scale fit on the whole train span; `S1`,`S2` are two `metrics()` calls on slices. The proposal text claims refit-agreement ("would this have been picked"), which overstates it. As specified ("identical position path restricted to each half") it is A. |
| R2 Harmonic / rank-product | **A** | Same inputs as R1, different aggregator. Same axis, softer. |
| R3 Odd-even | **A, and a broken A** | Same shared path, sliced by parity. Additionally destroys the smoother/lag structure the evaluation operator assumes, so the slices are not valid counterfactual histories. |
| R4 Fold-median + dispersion | **B (across folds)** | The 5 folds are already 5 independent simulations, each with its own scale fit. Changing mean→median is a genuine B-type change at the fold level — provided folds are temporally non-overlapping enough that median ≠ mean. This is the cheapest true-B test in the list. The `-k·IQR` tail, however, is not B; it is a reward change (see §2d). |
| R5 Breadth cap | **neither** | 1× penalty on a free byproduct. Does not split time or folds. Changes the objective. |
| R6 Turnover penalty | **neither** | Same structural status as R5, on the time axis instead of the cross-section. |
| R7 Shortlist rerank | **straddles A/B depending on Stage-2 scorer** | Scope wrapper around R1/R4. As written with R1 Stage-2, it is A with 1.25× cost accounting that is now wrong — under the new measurement it is ~1× + epsilon, because the 4/16 rescores are slice re-aggregations. |
| R8 Final-pick-only | **straddles, and the straddle matters** | As written (rescore stored paths) it is **A at ~1× + epsilon**. If it is upgraded to rescore with refit halves it becomes **B at ~3× per seed** (160 visited genomes × 5 folds × 2 halves = 1600 extra sims vs 800 for the search). The other list quotes "~1× + epsilon" without stating this only holds for the A version. |
| R9 Bootstrap-median | **ambiguous as written, A by default** | "Median Sharpe over bootstraps" from one simulated path is A. To become B it must refit scale and re-simulate per resample, at which point the 6× multiplier is real (5 resamples + original, all simulations). The list prices B while describing something closer to A. |

Corrections to overstatements in the other list:

1. R1/R2/R3 do not estimate "would this genome have been picked on a slightly different history." Nothing is repicked; one fitted object is sliced. The correct claim is "does the same fitted genome perform on both parts of its own train span."
2. R4's strength is understated, not overstated: it is the *only* candidate that is already B without extra simulations, because folds re-simulate. If folds overlap heavily in time, median ≈ mean and the test is weak — but that is an informative null, not a cost problem.
3. R7's 1.25× and R8's "epsilon" only hold for slice scoring. Refit versions cost genuine simulation multipliers (§3 table).

## 2. Cross-audit of the other list

**(a) Duplicates of mine:** R1 = M1 (A version); R4-core = M2; R8 = M4 (A version). R2 is a duplicate of R1 with a different aggregator — min, harmonic-mean, and rank-product are three parameterisations of "require both halves good." Keep min (strictest, no tuning constant, no per-generation sort semantics); drop the other two.

**(b) Already covered by the null calibration:** per-genome iid-null simulation (correctly listed as known-bad) — the 500-run null arm already gives the batch-level calibration at 1/50th the cost. Also "averaging fitness across seeds" — correctly listed as known-bad; seeds are independent searches, not a population.

**(c) Dominated:** R2 dominated by R1 (same inputs, extra fragility: harmonic blows up near zero, rank-product discards magnitudes). R9 dominated by M2+M3 jointly: fold-median already gives cross-window B at 1×, refit-halves gives temporal B at 2×; block-bootstrap at 6× buys little once both are in hand, except as a ceiling estimate — relegate to pilot at most. R7 dominated by R8 + R1: once slice scoring is ~1×, the "save compute by reranking only top-4" motive evaporates; the remaining question (does pressure need to act during evolution or only at the trophy) is answered more cleanly by R1-full vs R8-final-only, without the arbitrary top-4 cutoff.

**(d) Moves the gap for a reason with nothing to do with overfitting — the important one.** Three candidates are at risk:

| Candidate | How it moves the gap without testing selection |
|---|---|
| **R5 breadth cap** | Directly selects for fewer-coin books. Real books are broader (+7.08 coins). Any breadth cap mechanically compresses the breadth gap and changes holdout Sharpe through position sizing (fewer legs → different portfolio variance → different Sharpe even with zero edge). A gap move under R5 cannot distinguish "breadth was the noise channel" from "Sharpe rewards concentrated books on this holdout." It changes the estimand. |
| **R6 turnover penalty** | Same structure on the time axis. Null books may harvest churn; penalising churn drags null down more than real regardless of whether the real effect is slow. Gap narrows by differential shrinkage, not by harder-to-fool ranking. Also introduces a free constant `d` calibrated on the null arm — tuning the penalty on the comparator it is then tested against. |
| **R3 odd-even** | Parity slicing breaks `smooth_causal(5)` state and the 1-bar lag: adjacent-bar dependence is destroyed in each slice but present in the live path. A genome relying on any 2–5 bar persistence (real or spurious) scores differently on parity slices than on live data for purely mechanical reasons. Gap moves because the fitness measures an unphysical object, not because selection got stricter. Contiguous halves (R1) preserve the operator's assumptions; parity does not. |

R4's `-k·IQR` tail carries the same risk in miniature: pure median keeps the estimand (Sharpe) and hardens aggregation; subtracting dispersion introduces a reward trade-off (`k=0.5` is arbitrary) that selects for flat-across-folds genomes even when flat means flat-zero. Run median first; report IQR as a diagnostic, not a penalty.

## 3. Which 3–4 rules to run, and the design

### The shortlist (3 at full scale, 1 pilot)

| # | Rule (exact fitness) | True cost under new measurement | Prediction if TRUE (factors carry effect, argmax hides it) | Prediction if FALSE (no effect, −1.64 honest) |
|---|---|---|---|---|
| **F1** Fold-median (M2, stripped R4) | **1×**. `fitness = median_f S_f` over existing 5 folds. No extra sims, no `k`. | Gap narrows modestly (−1.64 → ≈−1.2 to −0.9): real flat-to-up, null flat-to-down. Small because folds may overlap; a small move is still informative. | No move in gap; both arms flat. Cheapest falsification of "argmax-over-folds is the lever." |
| **A1** Halves-min, slice (M1, honest R1-A) | **~1× + epsilon**. Per fold, split train bars at midpoint; reuse shared `pnet`; `S1f,S2f = metrics` on slices; `fitness = mean_f min(S1f,S2f)`. One extra `metrics()` per half; zero extra sims. | Gap narrows (−1.64 → ≈−0.8 to −0.3): real rises, breadth falls, beat-hold rises from 0.4% toward low single digits. Train Sharpe falls by construction — ignore it. | Both arms fall together or real falls more; breadth may fall with no Sharpe recovery; beat-hold stays ≈0.4% vs ≈20%+. |
| **C1** Final-pick-only slice (M4/R8-A, control) | **~1× + epsilon**. Search on baseline `mean_f S_f`; among the 160 visited genomes pick `argmax mean_f min(S1f,S2f)` using stored paths. | Moves ≈half as far as A1. `A1 − C1` = damage done *during* evolution vs at the trophy. `C1 ≈ A1` → search visited good genomes and picked badly; `C1 ≈ baseline ≪ A1` → argmax pressure steered evolution away from good regions entirely. | No move. Neither trajectory nor trophy contains recoverable signal. |
| **B1 (pilot)** Halves-min, refit (M3, strict R1-B) | **2× sims** (per fold: refit scale per half, reset smoother at boundary, re-simulate two half-paths; `fitness = mean_f min(S1f^refit,S2f^refit)`). Full 500×2 = 2 units — **pilot at 100×2 only (0.4 units)**. | Largest honest move; sets ceiling for A1/F1. `B1 − A1` = leakage optimism from shared-scale/smoother carryover. If TRUE with a regime-localised effect, B1 exceeds A1 clearly. | No move at 2×. Definitive for the temporal channel: even genuine refit-agreement cannot manufacture edge. |

Deliberately excluded: R2 (duplicate), R3 (unphysical slices), R5/R6 (change the estimand), R7 (motive gone at ~1×; C1 answers its question cleanly), R9-full (dominated; B1-pilot + F1 cover its information at 1/15th the cost).

### Design

- **Arms:** every rule runs **real + null**, otherwise the gap is uninterpretable. No validation rerank, no lockbox peek, reward stays v3 (PRP-0).
- **Seeds:** F1, A1, C1 at full **500×2, seeds 501–1000 paired with the reference batch** (same seed list → paired deltas, variance reduction; report paired t on the gap difference at the known ±0.09 scale). B1 as **100×2 pilot on seeds 501–600**; promote to full scale only if pilot `B1 − A1` is statistically clear.
- **Affordability:** 1 unit = 400k evals/arm-pair baseline (500×2×800). F1 + A1 + C1 ≈ 3×1 units (all ~1×); B1-pilot ≈ 0.4 units. Total ≈ **3.4 units**, ~3× the reference batch wall time on 12 workers. R9-full alone would be 6 units — excluded on cost. Nothing requiring K≥3 inner folds or per-genome nulls is in the design.
- **Decision rule (per rule):** supports the hypothesis only if (i) real-minus-null gap moves toward zero by a statistically clear margin on the paired t, **and** (ii) the move comes from the **real arm rising**, not just null falling, **and** (iii) breadth and beat-hold move in the predicted direction (breadth gap compresses, real beat-hold rises). A rule that drags both arms down equally, or narrows the gap purely by crushing null, does not support it. Train Sharpe is not an outcome.
- **Mandatory diagnostics per rule:** train Sharpe drop vs baseline (expect drop — confirms pressure applied); rank correlation of `S1f` vs `S2f` on real vs null (TRUE predicts higher on real); per-fold `S_f` dispersion on real vs null; breadth `Npos` distribution; beat-hold rates. These distinguish "no signal" from "wrong timescale" in §4.

## 4. The failure mode of the whole experiment

- **If every rule moves the gap:** that is evidence of a shared confound, not of recovered edge — most likely differential fragility (null books are easier to break by any added strictness, so every rule drags null down and the gap "narrows" with real flat). The decision rule's clause (ii) is the tripwire: uniform gap-narrowing with **real flat and null down** across F1/A1/C1/B1 means the experiment measured "stability filters hurt fragile books" rather than "factors carry signal." The required addition is a **negative-control rule that should not move the gap under the hypothesis** — e.g. final-pick among visited genomes by `mean_f S_f + injected noise`, or first-half-only selection. If the negative control also "moves the gap," all positives are discounted.
- **If no rule moves it:** two readings survive — (a) FALSE (no real effect; −1.64 honest), or (b) TRUE but the stability horizon is mismatched (true effect timescale ≠ half-window; folds overlap so median ≈ mean; scale-leakage makes A-versions toothless and the B-pilot underpowered at n=100). The diagnostics separate them: (b) predicts `S1f–S2f` rank correlation above null on real with no Sharpe recovery, and train Sharpe collapsing while ordering is preserved. To catch (b) the experiment needs what is currently missing: a **timescale sweep** (halves at 2–3 horizons, not one midpoint) or, equivalently, promoting B1 beyond pilot. Without that, a clean all-null result is ambiguous between "no signal" and "asked about consistency at the wrong horizon."
- **The uninformative clean run:** A1 moves, B1-pilot does not (or vice versa) with no diagnostic separation — i.e. slice-vs-refit disagree and `B1 − A1` is noise. That outcome means the scale/smoother leakage identified in the measurement note is the same order as the effect being hunted, and every A-version result is uninterpretable. The fix must be built in now: **B1-pilot is not optional** — it is the leakage meter without which A1/F1/C1 cannot be read.

Skipped: code/source verification of fold overlap and scale fitting — taken as described; risk is that fold structure differs and cost/axis claims need re-derivation.
