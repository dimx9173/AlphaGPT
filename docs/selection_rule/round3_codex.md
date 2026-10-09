## Settling the four disagreements

**Disagreement 1 — negative control: INCLUDED.**
The "differential fragility" confound is the single most dangerous reading of any positive: null books are churn-harvested and fragile, so *any* added strictness plausibly drags null down while real sits flat, and the gap "narrows" with nothing recovered. The decision rule already tries to trip on this via its real-must-rise clause, but a clause is not a control. A rule that applies the same slice machinery and the same data-halving strictness while demanding *zero* agreement separates "machinery/strictness artifact" from "agreement did work." Cost is ~1x, identical to `A1`. There is no budget reason to refuse it, and without it every positive is deniable.

- Exact fitness (`N1` first-half-only): per fold, split train bars at midpoint `m_f`, reuse the shared simulated `pnet` exactly as in `A1`; `S1f = Sharpe(pnet[0:m_f])`; `fitness = mean_f S1f`. Same non-finite handling as `A1` (below). It halves the selection data and keeps all slice-boundary artifacts, but requires no cross-half agreement. Under the hypothesis it MUST NOT recover signal: expected reading is gap flat or wider (noisier selection), real flat-to-down, null down-or-flat.

**Disagreement 2 — reuse vs re-run S0: RE-RUN.**
Reusing the old 1000-run batch as the comparator is paired and cheap, but the pair is only valid if the numbers are reproducible from the current tree — and the tree is being touched (slice `metrics()`, scale/smoother handling, the very A/B corrections that repriced every cost claim). The gap scale is ±0.09; silent drift of even a few hundredths in baseline fitness handling would masquerade as a treatment effect. The old batch is retained only as a sanity check (correlation of re-run S0 holdout Sharpes against stored values should be ~1.0); the comparator for every paired test is the fresh re-run on seeds 501–1000. 1 unit of insurance on a 4-unit experiment.

**Disagreement 3 — timescale: IN as a free diagnostic, OUT as full-evolution arms.**
Both sides are half-right. A single midpoint split genuinely cannot separate "no signal" from "signal lives at a different horizon than half-window" — but a full-evolution sweep over horizons (3 extra units) is scope creep that triples the bill before the flagship result exists. The resolution: horizons enter as zero-sim rescores on stored paths, not as evolution arms. On every stored S0 path, compute `min`-agreement and `S_first–S_second` rank correlation at splits `{1/3, 1/2, 2/3}`. That costs epsilon (slice re-aggregation only) and buys the interpretive key: a clean all-null *with* elevated real rank-correlation at some horizon means "wrong horizon, promote `B1`"; a clean all-null *with* flat correlations at all three horizons means "no slice-consistency signal at any tested horizon," which is licensed to lean FALSE. Without this diagnostic in the spec now, a null result is uninterpretable — so it is mandatory, but it is free.

**Disagreement 4 — fourth rule: fold-median wins, four-block loses.**
Four-block mean-minus-sd and fold-median are both 1x, so the choice is informational, not budgetary. Four-block entangles two changes at once (finer temporal slicing *plus* a dispersion penalty with an arbitrary `k`), so no outcome under it is attributable: a gap move could mean finer horizons matter, or variance-penalisation matters, or `k` happened to suit this holdout. Fold-median changes exactly one thing (mean→median over the 5 already-independent re-simulated folds), keeps the estimand (Sharpe, no penalty constant), and is the cheapest genuine B-type test in the whole candidate set. Its distinguishing outcome is one four-block cannot produce: `F1` moves the gap while `A1` does not → the lever is cross-fold aggregation, not within-train temporal consistency (and vice versa). Four-block's only unique prediction — "dispersion-penalised fine slices recover what coarse min-agreement cannot" — is a compound claim that should be decomposed *after* a positive, not tested first. Pick the simpler, Fal: fold-median.

## Final spec

Common quantities per seed per fold `f = 1..5`: train bars `T_f`, simulated position path `pnet_f` (scale fit on the full `T_f` unless the rule says refit), fold Sharpe `S_f = Sharpe(pnet_f)`. Non-finite half-Sharpe rule (applies everywhere): a non-finite half value sets that fold's `min(...)` to `-10` (hard discard, GA-numerics-safe; never silently averaged as 0). Holdout/validation handling unchanged: train-only selection, reward v3, PRP-0.

| Rule | Exact fitness | Axis | True cost | Seeds/arm |
|---|---|---|---|---|
| **S0** re-run control | `fitness = mean_f S_f` | — | 1x | 500×2, seeds 501–1000 |
| **F1** fold-median | `fitness = median(S_1..S_5)`, no penalty term; IQR recorded as diagnostic only | B (across folds) | 1x, zero extra sims | 500×2, same seeds |
| **A1** halves-min, slice | Per fold split `T_f` at `m_f = floor(\|T_f\|/2)`; reuse shared `pnet_f`; `S1f, S2f = metrics` on slices; `fitness = mean_f min(S1f, S2f)` | A | ~1x + epsilon (one extra `metrics()` per half, zero sims) | 500×2, same seeds |
| **N1** first-half-only (negative control) | Same slices as `A1`; `fitness = mean_f S1f`; no agreement term | A-machinery, zero robustness | ~1x + epsilon | 500×2, same seeds |
| **B1** halves-min, refit (pilot) | Per fold: fit `scale_1` on `T_f[0:m_f]`, reset smoother, simulate half-path 1; fit `scale_2` on `T_f[m_f:]`, reset smoother, simulate half-path 2; `fitness = mean_f min(S1f^refit, S2f^refit)` | B (temporal) | 2x sims; pilot 100×2 only (seeds 501–600), promote iff pilot `B1 − A1` separation is clear | 100×2 pilot |

Free rescore battery (epsilon, on stored S0 paths — not evolution arms, not counted in the four):
- **C1**: among the 160 visited genomes pick `argmax mean_f min(S1f,S2f)`; simulate holdout once for the pick. `A1 − C1` = damage done during evolution vs at the trophy.
- **Timescale diagnostic**: `min`-agreement and `S_first–S_second` rank correlation at splits `{1/3, 1/2, 2/3}`, real vs null.

Predictions:

| Rule | If TRUE (factors carry effect, argmax hides it) | If FALSE (−1.64 honest) |
|---|---|---|
| `F1` | Gap −1.64 → ≈−1.2 to −0.9; real flat-to-up, null flat-to-down; small move still informative | No gap move; both arms flat |
| `A1` | Gap → ≈−0.8 to −0.3; real rises, breadth falls, real beat-hold rises from 0.4% toward low single digits; train Sharpe falls by construction (ignore) | Both arms fall together or real falls more; breadth may fall with no Sharpe recovery; beat-hold stays ≈0.4% vs ≈20%+ |
| `N1` | Gap flat or wider; real flat-to-down (strictly worse selector, no agreement demanded) | Same: flat or wider — any `N1` gap-narrowing matching `A1` invalidates `A1`'s positive |
| `B1` pilot | Largest honest move; `B1 − A1` = leakage optimism from shared-scale/smoother carryover; exceeds `A1` clearly if the effect is regime-localised | No move at 2x — definitive for the temporal channel |

## Decision rule (pre-registered)

- **Recovered signal:** gap moves toward zero by a paired-t-clear margin on the ±0.09 scale, *and* the move comes from the **real arm rising** (not null-only collapse), *and* breadth gap compresses via real narrowing with real beat-hold rising, *and* `N1` does not reproduce the move, *and* `B1 − A1 ≥ 0` directionally (dose-response: refit ≥ slice).
- **Shared confound:** uniform gap-narrowing across `F1`/`A1`/`N1` with **real flat and null down**, or `N1 ≈ A1` in magnitude — read as "strictness hurts fragile books," discount all positives.
- **Inconclusive:** `A1` vs `B1`-pilot disagree beyond noise with no diagnostic separation (leakage same order as effect — A-versions unreadable, promote `B1` to full scale); or all-null *with* elevated real `S_first–S_second` rank correlation at some horizon (wrong-horizon, promote `B1` / re-split); all-null with flat correlations at all three horizons leans FALSE for slice-consistency at any tested horizon.

## Budget

- 1 unit = one 500×2 full-evolution arm-pair at ~1x (400k evals baseline).
- `S0` 1 + `F1` 1 + `A1` 1 + `N1` 1 + `B1`-pilot 0.4 + rescores ~0 = **≈4.4 units** (~3.5M evals).
- Fits this machine in one sitting at 12 workers at roughly 4–5× the reference-batch wall time; it is the ceiling, not headroom. If wall time binds, the pre-agreed cut is `N1` → rescore-only on stored S0 paths (saves 1 unit, weakens the confound check from full-pressure to pick-level); `B1`-pilot is never the cut, it is the leakage meter without which `A1` cannot be read.
