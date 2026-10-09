# Round 1: candidate selection rules

## Assumed status quo (to fix notation)

Let `B_train` = train-only bars. One genome evaluation = one full 28-coin position path over `B_train` (causal scale, tanh, cap 0.25, smooth(5), lag 1, real funding, costs), producing one per-bar portfolio excess-return series `r_t` (strategy minus constant-long, i.e. the same "excess" used on holdout).

Current rule **S0**: `fitness(g) = Sharpe(r_t, t in B_train)`, single number, argmax via the GA's per-generation selection, final pick = argmax of last generation. Cost = **1.0x** by definition. Everything below is a drop-in replacement for `fitness(g)` using train bars only. Validation/lockbox never enter.

Partition notation used below: split `B_train` into `K` contiguous time blocks (not shuffled — order matters). `s_k(g)` = excess Sharpe of `g` on block `k`, computed from the *same* `r_t` sliced by time. `s(g)` = whole-train Sharpe (S0). `n(g)` = genome length in tokens. `b(g)` = number of coins with positive train P&L ("book breadth", the Fact-2 variable).

---

## Candidates

### Overview

| # | name | fitness in one line | cost | failure mode it targets |
|---|---|---|---|---|
| S0 | Single-window argmax (baseline) | `Sharpe(B_train)` | 1.0x | — (calibration only) |
| S1 | Split-half worst-case | `min(s_1, s_2)` | ~1.05x | one-regime fit: genome fits first or second half of train only |
| S2 | Four-block stability-adjusted | `mean(s_k) - 1.0*sd(s_k)`, K=4 | ~1.05x | high-mean/high-variance genomes: one hot block drags argmax |
| S3 | Consistency-gated mean | `mean(s_k)` iff positive in ≥3/4 blocks, else heavily penalized | ~1.05x | same as S2 but hard gate instead of soft penalty; kills single-block wonders outright |
| S4 | Breadth-penalized | `s(g) - λ·b(g)` | ~1.1x | breadth-as-overfit: Fact 2 (+7 coins on real); argmax buys breadth because more coins = more noise to fit |
| S5 | Parsimony-penalized | `s(g) - c·n(g)` | ~1.0x | expression bloat: long genomes fit idiosyncratic train wiggles |
| S6 | Cost-stressed min | `min(s_base, s_2x-costs)` | ~1.3–2.0x | friction-fit: genomes whose train edge dies under slightly higher costs |
| S7 | Block-bootstrap median | `median(s*_1..s*_B)`, B≈200 block-resamples of `r_t` | ~1.1–1.3x | outlier-bar dependence: fitness hinging on a handful of train bars |
| S8 | Final-only Pareto re-rank | GA runs on S0; final elite picked from top-8 by S2/S3 | ~1.0x + epsilon | all of the above, but contained to one decision point instead of every generation |

### S1 — Split-half worst-case

- **Exact computation:** split `B_train` at its temporal midpoint into H1, H2. From the single position path, slice `r_t` into the two halves, compute `s_1, s_2`. `fitness = min(s_1, s_2)`. (Variant for sensitivity: harmonic mean `2/(1/s_1 + 1/s_2)` when both positive, else `min` — same cost, same intent.)
- **Cost:** ~1.05x. One path, two Sharpe aggregations. No re-simulation.
- **Prevents:** the cheapest form of single-window luck — a genome that memorises the dominant train regime (which lives mostly in one half) while losing in the other. S0 cannot see this; S1 makes it the fitness.
- **GA interaction:** clean per-generation drop-in. Preserves ranking spread (min still varies across population). Risk: halves are still long (~1yr each at 1h), so a genome fitting a 3-month noise episode inside one half still passes. That is what S2/S3 are for.

### S2 — Four-block stability-adjusted (soft)

- **Exact computation:** split `B_train` into K=4 contiguous quarter-blocks. `s_1..s_4` from the single path. `fitness = mean(s_k) − 1.0·sd(s_k)` (sample sd, `ddof=1`). Report `mean` and `sd` separately in logs so the penalty weight can be re-tuned offline without re-running.
- **Cost:** ~1.05x. One path, four aggregations.
- **Prevents:** high-variance genomes: one block at +4 Sharpe and three at −2 currently beat a steady +0.5 everywhere under S0. The `−sd` term prices temporal instability directly.
- **GA interaction:** good per-generation behaviour — smooth, differentiable-ish in rank space, no ties, selection pressure retained. Caution: `mean − sd` compresses the fitness range, so tournament pressure weakens slightly; if the GA uses fitness-proportionate selection this matters, if it uses tournament/rank selection it does not. Prefer rank/tournament with S2.

### S3 — Consistency-gated mean (hard)

- **Exact computation:** same 4 blocks. `m = mean(s_k)`, `p = count(s_k > 0)`. `fitness = m` if `p ≥ 3`, else `m − 5.0` (large fixed penalty; value logged, tunable offline). Alternative equivalent: `fitness = min(m, min_top3_avg − 0)` — same gate, fewer magic constants.
- **Cost:** ~1.05x.
- **Prevents:** the same failure as S2 but as a veto rather than a tax. A genome negative in two of four quarters cannot win no matter how hot its best quarter is.
- **GA interaction:** creates a two-tier population (pass/fail), which is fine for tournament selection but can flatten gradients early (whole population failing in gen 0 → all penalized → selection among penalties ≈ selection on `m` anyway, acceptable). Best paired with elitism so a passing genome is never lost to a failing one with higher raw `m`.

### S4 — Breadth-penalized Sharpe

- **Exact computation:** from the same path, decompose train P&L per coin (already available as the 28 legs of the portfolio sum), count `b(g)` = coins with positive train excess. `fitness = s(g) − λ·b(g)`, with λ pre-registered (e.g. λ = 0.05 Sharpe per coin) and `b(g)` logged so λ can be re-tuned offline. Report both terms.
- **Cost:** ~1.1x. One path plus per-coin aggregation (28 small Sharpe/ sums instead of 1). No re-simulation.
- **Prevents:** the Fact-2 signature directly: real-arm genomes buy +7 extra positive coins vs null. If breadth is how argmax harvests noise (each added coin is another degree of freedom), S4 taxes exactly that margin.
- **GA interaction:** per-generation safe, but changes the *direction* of evolution, not just its strictness — the GA will learn to write narrow books. That is the point, but it means S4 is the only candidate that could "fix" the gap by forbidding the response rather than by finding signal. Interpret S4 jointly with book-breadth on holdout (see discriminating-predictions table).

### S5 — Parsimony-penalized Sharpe

- **Exact computation:** `fitness = s(g) − c·n(g)`, `n(g)` = token length of the decoded genome, `c` pre-registered small (e.g. 0.01 Sharpe per token), `n(g)` logged for offline re-tuning. No bar-level work beyond S0.
- **Cost:** ~1.0x (free — length is known at decode).
- **Prevents:** expression bloat: longer RPN programs have more functional degrees of freedom and fit train noise more tightly. Classic anti-overfit pressure.
- **GA interaction:** the standard bloat-control role; interacts well with mutation/crossover every generation. Weakness specific to this project: the reduced grammar is already small (11 operators), so length may not be the binding degree of freedom — breadth (S4) and regime-fit (S1–S3) are more plausible. S5 is cheap enough to include as a control.

### S6 — Cost-stressed min

- **Exact computation:** run the accounting twice on the *same* position path: once with shipped costs, once with 2x costs (slippage + fees doubled; funding unchanged). `fitness = min(s_base, s_stressed)`.
- **Cost:** ~1.3–2.0x depending on whether accounting is vectorised over the path (cheap) or requires re-simulation per genome (full 2x). Position generation itself is not repeated — only the P&L accounting leg.
- **Prevents:** friction-fits: genomes whose train Sharpe lives inside the bid-ask spread / turnover margin. These look like edge under S0 and vanish net of slightly worse execution.
- **GA interaction:** per-generation feasible but the most expensive of the cheap set. Side effect: selects for low-turnover genomes, which is a real behavioural change (like S4). Log turnover so the mechanism is auditable.

### S7 — Block-bootstrap median

- **Exact computation:** from the single `r_t` series, draw B≈200 resamples by circular block bootstrap (block length ≈ 168 bars = 1 week at 1h, preserves autocorrelation), each resample concatenates sampled blocks to train length; compute Sharpe per resample; `fitness = median(s*_1..s*_B)`. (Also log 10th percentile for offline analysis.)
- **Cost:** ~1.1–1.3x. No re-simulation; 200 Sharpe computations on length-`T` vectors per genome. Vectorisednumpy makes this small next to path generation, but it is the largest constant among the aggregation-only rules.
- **Prevents:** outlier-bar dependence: fitness hinging on a few high-leverage bars (a crash, a funding spike). Median-over-resamples downweights genomes whose Sharpe evaporates when those bars are omitted.
- **GA interaction:** smooth and rank-preserving, good per-generation behaviour. Must fix the resample seed per generation (or per run) so genomes within one generation are compared on the same resamples — otherwise bootstrap noise injects ranking noise. Common seed per generation is the correct design.

### S8 — Final-only Pareto re-rank (two-stage)

- **Exact computation:** run all 10 generations under S0 (unchanged dynamics, unchanged cost). At the end, take the top-8 genomes by train Sharpe (the would-be elite pool), compute S2 (or S3) for those 8 only from their stored paths, pick the final champion by the robust score. Log both rankings so the S0-vs-robust disagreement rate is measurable.
- **Cost:** ~1.0x + epsilon (8 extra aggregations per seed, negligible).
- **Prevents:** nothing during evolution — that is deliberate. It isolates the *final argmax* step ("pick the luckiest of 16" × 10 generations) from the *evolutionary guidance* question. If the damage is done mostly at the final pick, S8 captures most of the benefit for ~zero cost.
- **GA interaction:** zero interference by construction — the GA never sees the robust score. This makes S8 the cleanest ablation: S8-vs-S0 difference = value of de-noising the final pick alone. Conversely, if the population itself converges to noise-fits by gen 10, the top-8 pool contains nothing worth re-ranking and S8 does nothing — in which case per-generation rules (S1–S3) are required. S8 should always be run alongside one per-generation rule for exactly this comparison.

---

## Known-bad / close-to-useless rules (name them so they stay dead)

| Rule | Why it fails here |
|---|---|
| Validation-window selection / early stopping on val | Violates PRP-0 (train-only selection). Converts the observation windows into second training windows; the lockbox design exists precisely to forbid this. Any gap movement would be uninterpretable. |
| Max-over-folds (`max(s_k)`) | Cherry-picking formalised. Strictly worse than S0 at the stated failure mode — selects the genome with the luckiest block. |
| Deflated Sharpe / White's Reality Check / Hansen SPA as in-loop fitness | Estimates the same selection bias it would need to remove, per genome, per generation. Statistically circular at this level and ~10–100x too expensive in-loop. These are *batch-level* diagnostics for the 500-seed analysis, not per-genome selectors. |
| AIC/BIC on Sharpe | No likelihood, no well-defined parameter count for RPN programs; the penalty has no units that match Sharpe. S5 (explicit `c·n`) is the honest version of this impulse. |
| Per-coin argmax then aggregate ("best coin per genome") | Maximises breadth-harvesting (Fact 2). Gives each genome 28 chances to be lucky instead of one. |
| Rank-IC / factor-IC fitness replacing path Sharpe | Reintroduces reward-vs-gate misalignment (already eliminated by experiment): selects genomes that predict well and trade badly under the real accounting. |
| Bigger population / more generations as the fix | Search effort was eliminated by the effort ladder; more search under a broken selector is more overfit, not less. |
| More factors / fewer factors as the fix | Factor set was eliminated (36f×100) and multi-timeframe was eliminated (41f×100). Do not re-litigate inside the selection experiment. |
| Shuffled-bar bootstrap (iid resample, no blocks) | Destroys autocorrelation and funding periodicity; manufactures a null the market does not live in. Block bootstrap (S7) exists to avoid exactly this. |
| Turnover / cost-unaware Sharpe with costs added only at the end | The eliminated reward-vs-gate misalignment. S6 is the allowed form (costs inside the min). |

---

## Affordability at 500 seeds × 12 workers

Budget anchor: the 1.0x batch (1000 runs: 500 real + 500 null) already ran on this machine, so **1.0–1.3x per rule is affordable by definition** — same seeds, same workers, ~same wall-clock. Multiples below assume per-genome cost dominates (10 gens × 16 genomes = 160 path evaluations per seed, plus fixed final holdout evals that do not change).

| Cost band | Rules in it | Verdict |
|---|---|---|
| ~1.0x | S0, S5, S8 | Trivially affordable. Run all three. |
| ~1.05x | S1, S2, S3 | Affordable. One path + K aggregations; aggregation is noise next to path generation. Run all three. |
| ~1.1x | S4 | Affordable. Per-coin aggregation over stored legs. |
| ~1.1–1.3x | S7 (B=200, vectorised, common seed per generation) | Affordable *if* vectorised and seeded per generation. If implemented as a Python loop per genome, re-estimate before committing — cap B at 200. |
| ~1.3–2.0x | S6 | Borderline-affordable: doubles only the accounting leg, not path generation. Affordable if accounting is vectorised; if each stress requires a full re-sim, treat as 2x and halve seeds (250+250) with pre-registered justification, or run S6 final-only like S8. |
| ≥5x (not proposed) | Per-genome K-fold re-simulation, mutant-neighbourhood averaging (1+m evals per genome per generation), nested null-calibration per genome | **Unaffordable** at 500×12 without cutting generations/seeds/coins — and cutting those changes the experiment. These belong, if anywhere, as final-only re-ranks on stored paths, never in the generation loop. |

Concrete recommendation for the affordable set: **S0 (re-run as control) + S1 + S2 + S3 + S4 + S5 + S7 + S8 = 8 arms**, each at 500 seeds or matched seed count to S0. S6 joins iff accounting-vectorisation is confirmed; otherwise S6-final-only. Total ≈ 8 × 1.1x ≈ one more batch of the size already demonstrated. Nothing in the generation loop may re-simulate a path.

GA-specific cost note: with pop 16, per-generation rules (S1–S5, S7) execute 16 fitness computations per generation — but each is one path plus aggregation, so generation wall-clock ≈ S0 generation wall-clock. The binding constraint is worker throughput over 500 seeds, not population size. Rules needing cross-genome information (e.g. novelty search, Pareto fronts across the population) would add synchronisation cost per generation; none of the proposed rules need it — every fitness above is a pure function of one genome's train path plus pre-registered constants.

---

## Discriminating predictions: what each rule says under TRUE vs FALSE

Hypothesis TRUE = factors carry a real (weak, diffuse?) effect and single-window argmax systematically prefers tighter noise-fits over it. Hypothesis FALSE = no edge; the −1.64 gap is the honest signature of fitting real-but-untradeable structure (or pure noise with fatter tails), and no train-only reweighting recovers anything.

| Rule | If TRUE (real effect exists) | If FALSE (no edge) | How to read it |
|---|---|---|---|
| S1 split-half min | Real arm improves, null arm flat-to-worse (null noise rarely spans both halves); gap moves toward zero from below. Beat-rate ratio (70x) compresses. | Both arms move down together (min ≤ mean always); gap stays negative, ordering preserved. | S1 is the sharpest single test: needs persistence across ~1yr halves, which noise rarely supplies. |
| S2 stability-adjusted | Same direction as S1, smoother: real sd falls (real patterns recur across quarters), null sd stays high; gap narrows. | Both means fall by ~their sds; gap roughly preserved (possibly wider if real is burstier). | S2 distinguishes via the *variance* channel rather than the level channel. Log mean and sd separately so a mean-drop with sd-drop is not misread as failure. |
| S3 consistency gate | Real pass-rate (≥3/4 positive quarters) exceeds null pass-rate; conditional-on-pass gap much narrower than unconditional S0 gap. | Pass-rates both low and similar (null 27.6% beat-rate suggests null passes *more* often); gap among passers stays negative. | Pre-register the pass-rate comparison — it is as informative as the Sharpe gap. |
| S4 breadth-penalized | Real breadth falls toward null breadth; if breadth was the vehicle of overfit, real Sharpe rises as breadth falls (a *negative* breadth–Sharpe slope across λ in offline re-tuning). | Breadth falls in both arms with no Sharpe recovery; gap preserved at lower breadth. Offline λ sweep shows flat lines in both arms. | S4's danger: it can narrow the breadth gap by fiat without finding signal. Require the joint outcome (breadth down *and* Sharpe up on real, Sharpe flat on null) before calling it a fix. |
| S5 parsimony | Weak or no movement under either hypothesis (grammar already small). If TRUE via complex interactions, slight worsening possible. | No movement. | S5 is a control: a null result here sharpens any positive result in S1–S3 (rules out "any penalty works"). |
| S6 cost-stressed | If TRUE and the effect is low-turnover, real holds up under 2x costs while null (high-turnover noise-fits) collapses; gap narrows. If TRUE but high-turnover, both fall. | Both fall together; gap preserved. | Log turnover by arm — the mechanism check is whether S6's gain (if any) rides on turnover reduction. |
| S7 bootstrap median | If TRUE, real median ≈ real mean (stable bars) while null median < null mean (outlier-dependent); gap narrows modestly. Expect smaller effect than S1–S3. | Medians below means in both arms; gap preserved. | S7 tests the *within-window outlier* theory specifically. A null S7 result with positive S1/S2 localises the overfit to regime-scale, not bar-scale. |
| S8 final-only re-rank | If TRUE and damage concentrates at the final pick, S8 recovers part of S2's gain for ~free (S0 < S8 ≤ S2). If damage accumulates during evolution, S8 ≈ S0 < S2. | S8 ≈ S0 in both arms. | S8-vs-S2 comparison locates *where* argmax bites: final pick only, or the whole evolutionary trajectory. Either outcome is informative for round 3 design. |

Across all rules, the hypothesis-TRUE signature is **asymmetric movement**: real improves or holds while null degrades or holds, so the −1.64 gap compresses toward zero and the 70x beat-rate ratio compresses with it. The hypothesis-FALSE signature is **symmetric movement**: every penalty lowers both arms together (any min/penalty must weakly lower levels), the ordering real < null survives, and the breadth gap only moves when directly taxed (S4) without Sharpe recovery. Pre-register that levels *will* fall under S1–S3/S7 in both arms — the test statistic is the *gap* (real − null) and the beat-rate ratio, never the raw real-arm level.

Skipped: exact train/val/lockbox bar indices and GA selector internals (tournament vs proportional) — assumed standard, verify before round 2. Risk: cost multiples assume vectorised aggregation; confirm accounting cost before including S6 per-generation.
