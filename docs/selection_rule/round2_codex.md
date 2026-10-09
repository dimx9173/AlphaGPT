**Round 2: vet, collapse, and choose**

## 1. Map: 17 candidates that are 4 ideas

### Our round-1 set, for the record

| # | name | one-line fitness |
|---|---|---|
| R0 | Train Sharpe (baseline) | `Sharpe(B_train)` |
| R1 | Split-half min, shared path | `min(s_1,s_2)` from one path |
| R2 | Quarter-block mean-minus-sd | `mean(s_k)-1.0*sd(s_k)`, K=4, one path |
| R3 | Cross-half coin-rank agreement | `s(g) * rankcorr(coinPnL_H1, coinPnL_H2)` |
| R4 | Split-half min, scale-refit per half | `min(s_1',s_2')`, scale re-fit inside each half |
| R5 | Disjoint-evolution overlap | evolve on H1 and H2 separately, keep intersection |
| R6 | Cost-haircut min | `min(s_base, s_2x)` |
| R7 | Length penalty | `s(g) - c*n(g)` |

### Collapse onto their list

| Ours | Closest theirs | Relation |
|---|---|---|
| R0 | S0 | Same test, same name. Keep one. |
| R1 | S1 | Same test under a different name. Both: one path, slice `r_t` in two, take min. |
| R2 | S2 / S3 | Same idea as S2 (soft) and S3 (hard gate). Three names for one idea: temporal-dispersion penalty. S2 and R2 identical; S3 is the hard-threshold version of the same. |
| R3 | — (none) | Genuinely different. Cross-sectional stability (do the *same coins* win in both halves?) rather than temporal level stability. No counterpart in S-list. |
| R4 | S1 (claimed) | Stricter version of the same test. S1 scores slices of one fitted object; R4 refits the scale and warm-up per half, then scores. Same split, stronger claim. |
| R5 | — (none) | Genuinely different and genuinely B. Nothing in S-list refits the search itself. |
| R6 | S6 | Same test, same name. |
| R7 | S5 | Same test, same name. |

Net: R0=R-S0, R1=S1, R2=S2=S3, R6=S6, R7=S5. That leaves 4 distinct ideas across both lists: (a) temporal-level stability, (b) cross-sectional/breadth regularisation (S4, R3), (c) cost/friction robustness (S6), (d) resampling robustness (S7). Plus the meta-idea (S8): *when* in the loop the rule is applied.

Correction owed on our round 1: R1 and R2 were described as "out-of-sample consistency." They are not. Under the measurement above they are **A**, not B. One `evaluate()` produces one `pnet`; slicing `r_t` and calling `metrics()` again does not refit anything. The scale was fit over the whole train span, the smoother carries state across the cut. What R1/R2/S1/S2/S3/S7 measure is *temporal stability of one fitted object*. R4 and R5 are the only B candidates either side proposed. S8 is A-at-one-point.

### Axis placement, all candidates

```
A: scores sub-slices of one shared path (temporal stability of one fit)
B: refits per window and requires agreement (true out-of-sample consistency)
C: changes what is rewarded (style regulariser, neither A nor B)
```

| Candidate | Axis |
|---|---|
| S1 split-half min, shared path | A |
| S2 quarter mean-minus-sd | A |
| S3 consistency gate | A (same as S2, hard form) |
| S4 breadth penalty | C (cross-sectional tax, no time split) |
| S5 / R7 length penalty | C |
| S6 / R6 cost-stressed min | C with A-like computation (one path, two accountings) |
| S7 block-bootstrap median | A (within-window resampling of one path) |
| S8 final-only re-rank | A, applied once instead of per generation |
| R3 coin-rank agreement | A+C hybrid (two slices, but cross-sectional statistic) |
| R4 scale-refit min | Straddles A/B: same genome, but scale and smoother state refit per half |
| R5 disjoint evolution | Pure B |

Nothing else straddles. The S-list, as written, contains **zero pure-B rules** — that is the gap in it.

## 2. Cross-audit

**(a) Duplicates.** S2 and S3 are the same test; S3's gate is S2 with infinite penalty below threshold. Keep one. R1=S1, R2=S2, R6=S6, R7=S5 as above. Running all of both lists would run the same regression 4 times.

**(b) Already covered by the null calibration.** S7 (block-bootstrap median). The 500-run iid-null arm already calibrates "what does this search do on structureless data." A bootstrap *inside* fitness re-estimates, per genome per generation, the sampling noise the null arm measures once at batch level, at 200 Sharpe calls per genome, to test the bar-outlier theory — which Fact 2 already argues against (the real signature is broad books, not a few hot bars). S7 is the most expensive A-rule for the least discriminating theory. Do it offline on stored paths if at all.

**(c) Dominated.** S3 is dominated by S2 for per-generation use: the hard gate creates a two-tier population with flat gradients in early generations (all-fail gens select on penalty + noise), while S2 preserves rank spread with the same 4 numbers logged. S5/R7 is dominated as an experiment: grammar already reduced to 11 operators, length is unlikely to be the binding degree of freedom, and its only role is "any-penalty control." A control that cannot move under either hypothesis adds no information beyond what S1/S2's level-drop already shows. Run it offline as a re-rank, not as a 500-seed arm.

**(d) Moves the gap by changing the reward — the dangerous ones.** S4, S6, and to a lesser extent S5. Each changes *what kind of genome wins* (narrow books, low turnover, short programs), not *how hard the fitness is to fool*. S4 is the sharpest risk: taxing `b(g)` narrows real books by construction, so the breadth gap (Fact 2) closes by fiat, and any Sharpe movement rides on forcing a different portfolio, not on finding signal. Same for S6 (selects low-turnover genomes; turnover correlates with breadth and with costs, so the gap can move through the cost channel alone). A gap movement under S4/S6 without the joint condition — breadth down *and* real Sharpe up *and* null flat, plus turnover audit — looks like a result and means nothing. S8 has the opposite risk profile: it cannot change evolution at all, so a null S8 result is ambiguous between "final pick is not the problem" and "pool already converged to noise."

## 3. Which 3–4 to run, and the design

Run four: one A-level rule, its B-strict pair, one A-variance rule only if cheap, and the free ablation. Everything else goes offline.

| # | Rule (exact fitness, train-only) | True cost under the measurement | TRUE predicts | FALSE predicts |
|---|---|---|---|---|
| A1 | Split-half min, shared path. One `evaluate()` per (genome, fold); slice stored `r_t` at train midpoint; `fitness=min(s_1,s_2)`. Log `s_1,s_2,s`. | ~1.05x (one extra `metrics()` on slices) | Real holds or rises vs S0, null falls (null noise rarely spans halves); gap −1.64 compresses toward zero; 70x beat-rate ratio compresses. | Both arms fall (min ≤ mean); gap preserved, ordering real<null preserved. |
| B1 | Split-half min, scale-refit. Same split, but recompute `scale=std(sig)` inside each half with its own `scale_end`, re-apply tanh/cap/smooth(5)/lag-1 per half with 50-bar warmup discarded at each half start; `fitness=min(s_1',s_2')`. | ~1.4–1.6x (signal reused, position/accounting legs redone per half; confirm by timing 10 seeds before full commit) | If A1's gain is real persistence, B1 ≈ A1 or stronger (leakage removal hurts noise more than signal). If A1's gain was scale-leakage artefact, B1 < A1 and may return to S0 gap. Either way the A1–B1 *difference* is the leakage measure. | Both arms fall further than A1; gap preserved or wider (real burstier). A1≈B1 in levels means leakage was negligible and A1 stands. |
| A2 | Four-block mean-minus-sd. Same single path, K=4 contiguous blocks; `fitness=mean(s_k)-1.0*sd(s_k)`; log mean and sd separately. Rank/tournament selection only. | ~1.05x | Same direction as A1 via variance channel: real sd falls, null sd stays high; gap narrows. Smaller expected effect than A1. | Both means fall by ~their sds; gap preserved. |
| F1 | Final-only re-rank (S8). Evolve 10 gens under S0 unchanged; re-rank stored top-8 by A2 score; champion = robust argmax. Log S0-vs-robust disagreement rate. | ~1.0x + epsilon (no new sims) | If damage is final-pick luck: S0<F1≤A2. If damage accumulates in evolution: F1≈S0<A2. Both outcomes locate the bite. | F1≈S0 in both arms. |

Dropped to offline/pilot, with reason: S3 (duplicate of A2, worse gradients — re-tune A2's logged mean/sd offline to emulate it); S4/R3 (reward-changers — re-rank stored S0 paths across a λ sweep for free instead of evolving under them; promote to full arms only if offline sweep shows negative breadth–Sharpe slope on real and flat on null); S5/R7 (control — same offline treatment); S6 (reward-changer + costliest — run final-only on stored paths first; promote to per-generation only if accounting vectorisation confirmed *and* offline shows turnover-mediated effect); S7 (covered by null calibration — log 10th pct offline if desired).

**Design.** Reuse the 1000-run batch as the S0 control (same seeds 501–1000, same folds, same accounting) — do not re-run S0. Run A1, A2, F1 at full 500-real + 500-null on identical seeds/folds/workers (12 workers, 10 gens, pop 16, 5 folds). Run B1 as a 150+150 pilot on a pre-registered seed subset (e.g. first 150 seeds per arm) because of the 1.4–1.6x multiplier; promote to full 500×2 iff pilot A1–B1 level difference exceeds pilot noise (paired t on the difference, |t|>2). Total committed ≈ 3×1.05x + pilot 0.3×1.5x ≈ 3.6× the demonstrated batch — affordable by the round-1 budget anchor. Cap: nothing in the generation loop may re-simulate a full path except B1's per-half position/accounting legs; R5-style disjoint evolution is out of scope at this scale.

**Decision rule, pre-registered.** Primary statistic is the *gap* (real-minus-null holdout excess Sharpe), never raw real level (all penalties weakly lower levels). TRUE requires *asymmetric* movement: gap compresses by ≥0.40 (≈25% of −1.64, ≈4.5× the S0 gap SE of 0.088) *and* beat-rate ratio compresses (70x toward <20x) *and* breadth gap does not merely track a direct breadth tax (moot for A1/A2/F1/B1 since none tax breadth — any breadth co-movement is mechanism evidence, not fiat). FALSE is symmetric movement: both arms fall, gap within ±0.20 of −1.64, ordering preserved, pass-rate/beat-rate ratio intact. Report per-arm mean ± SE, gap with 95% CI, beat-rate ratio, breadth gap with t, and for A2 the mean/sd decomposition so a mean-drop with sd-drop is not misread.

## 4. The failure mode of the whole experiment

If every rule moves the gap, that means nothing by itself: every penalty lowers levels, null has 3× the variance of real (sd 1.86 vs 0.65), so any variance-sensitive penalty compresses the gap mechanically through the null arm falling faster — a statistical artefact, not signal found. The pre-registered asymmetry condition (real up/flat while null down) is the only guard; without it "everything works" is the expected outcome of penalising a high-variance arm.

If no rule moves it, that is equally ambiguous between "no edge" and "all A-rules share the same fitted object": A1/A2/F1 all score slices of one path whose scale and smoother state were fit over the whole span. If the overfit lives *in* that shared fit (scale absorbs the train regime; smoother bridges the cut), every A-rule inherits it together and fails together. That joint failure would look like FALSE while proving only that A cannot see the vehicle.

The interpretation that makes this experiment uninformative even run cleanly: **common-mode leakage across all A-rules.** Same path, same scale, same smoother → correlated verdicts mistaken for independent confirmations (if all move) or independent rejections (if none move).

What catches it is the B1 pair, which is why B1 is non-negotiable even as a pilot: A1 and B1 differ in exactly one thing (was the scale/smoother refit per window or shared?). A1≈B1 means leakage negligible and the A-verdict stands; A1≫B1 means A1's result was leakage artefact. If B1 pilot is cut for budget, add the cheapest substitute: offline scale-refit re-score of stored S0 champions (no re-evolution) — it tests the leakage channel at ~0x. Without one of those two, the experiment cannot distinguish "factors had nothing" from "the test shared the overfit it claimed to penalise."
