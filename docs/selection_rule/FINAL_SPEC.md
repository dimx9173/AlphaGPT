# Selection-rule experiment: final spec, pre-registered

Agreed by two models over three rounds, plus one source-code measurement that
reclassified the cost of every time-split rule. Written before any number is seen.

## Why

The 1000-run batch: real -2.7429 +/- 0.0291, null -1.1003 +/- 0.0832, gap **-1.6426
+/- 0.0881, t = -18.64**. On data with no edge by construction this search beats a
constant long 138 times in 500 (27.6%); on the real market, 2 times in 500 (0.4%).
Real data produces significantly broader books (+7.08 positive coins, t = +24.47)
and significantly worse ones. That combination is not the shape of an absent
effect. It is the shape of a selection rule that cannot tell a real effect from a
lucky fit.

Seven variables have been eliminated by experiment: search space, search effort,
reward-vs-gate alignment, factor set, multi-timeframe, bar size, and now search
magnitude at n=1000. The selection rule is what is left.

## What is fixed

Universe, features (41, indices 0..11 unchanged), reduced grammar (operators at
41..51), evaluation path, folds, accounting, real funding, reward v3, train-only
selection, PRP-0. Only the fitness scalar changes.

## Notation

Per (genome, fold f): the live position path `pnet_f` is computed exactly once by
`evaluate()`, and `Sharpe(.)` is the project's own `metrics()`. Cross-fold
aggregation is the mean unless stated. A non-finite half-Sharpe sets that fold's
`min(...)` to -10, a hard discard, never a silent zero.

## The rules

| rule | fitness | axis | cost | seeds/arm |
|---|---|---|---|---|
| **S0** | `mean_f S_f` | control | 1.00x | 500x2 |
| **A1** halves-min | `mean_f min(Sharpe(pnet_f[0:m]), Sharpe(pnet_f[m:]))`, `m = floor(T/2)` | A | 1.05x | 500x2 |
| **FM** fold-median | `median(S_1..S_5)`, IQR logged not penalised | B across folds | 1.00x | 500x2 |
| **N0** odd/even min | `mean_f min(Sharpe(pnet_f[0::2]), Sharpe(pnet_f[1::2]))` | A, unphysical by design | 1.05x | 500x2 |
| **B1** halves-min refit | as A1 but `scale` refitted inside each half, smoother reset, first 50 bars of each half discarded | B temporal | 1.4-1.6x | 150x2 pilot |
> **FM cost note (2026-10-29, implementation)**: FM as a true joint-evolution
> fitness (median over all selection folds per genome per generation) costs
> ~4x, not the 1.00x tabled here. The tabled 1.00x assumes the walkforward
> per-fold loop, in which a cross-fold median is undefined as a within-generation
> fitness. True FM needs a joint loop; FM-as-post-hoc-reselect needs the 160
> visited genomes per fold, which the stored batch does not keep (champion only).
> Decision deferred until the S0 gate passes: either pay ~4x for true FM, or
> downgrade FM to a free diagnostic on a future instrumented batch.

All full-scale rules run on the same seeds 501-1000, real and null, so every
comparison is paired.

### Why the costs are near 1x

`evaluate()` computes `pnet`, the per-bar portfolio return series, once and then
takes a Sharpe from it. A rule that splits the train window by time reuses that
series: one extra `metrics()` call, no second simulation. Round 1 priced these at
2x; that was wrong.

### Why N0 is the negative control

N0 applies the same machinery and the same `min()` strictness as A1 and changes
only that the two views interleave rather than sit adjacent in time. Its parity cut
breaks the smoother state and the lag-1 alignment, which is why it was rejected as
a treatment in round 2. As a control that defect is the point: a deliberately
unphysical aggregator that must NOT recover signal, because a control that is
itself a plausible treatment cannot separate a treatment effect from a control
effect.

### S0 is re-run, not reused

The code that computes fitness changes in this experiment, and the gap SE is 0.09.
Silent drift of a few hundredths in baseline handling would masquerade as a
treatment effect. The old batch is kept as a sanity check: re-run S0 holdout
Sharpes must correlate with the stored values, and the correlation is reported.

## Decision rule, fixed in advance

Primary statistic is the gap, with 95% CI, plus the beat-rate ratio and the breadth
gap. Never the raw real level, because every aggregation here weakly lowers
levels by construction.

**Signal recovered** requires ALL of:
1. A1 and/or FM compress the gap by >= 0.40 (about 25% of -1.64, roughly 4.5 gap SE)
2. with asymmetry: real flat or up, null down
3. the beat-rate ratio compresses from 70x toward <20x
4. N0 does NOT compress: its gap stays within +/-0.20 of S0
5. B1 within noise of A1, confirming negligible leakage. If B1 < A1 with a residual
   compression in (A1 - N0), that counts as partial signal with leakage quantified.

**Shared confound** if N0 compresses alongside A1 and FM, or if all rules move the
gap in proportion to arm variance without the asymmetry in (2). Verdict: a
min-filter artefact, not signal, however many rules appear to work.

**Inconclusive** in three cases: all rules within +/-0.20 of S0, which licenses only
"no persistence at half-train to cross-fold horizons" and not "no edge"; A1 moves
but the B1 pilot is underpowered, which leaves leakage unseparated; per-generation
versus final-only re-rank disagree, which leaves the bite location ambiguous.

## Free diagnostics, not evolution arms

Zero extra simulations, computed on stored paths from the S0 run:

- **C1** final-only re-rank: evolve under S0, pick the champion from the 160
  visited genomes by A1 or FM score. A1 - C1 locates whether the damage was done
  during evolution or only at the trophy.
- **Timescale sweep**: agreement and first-half/second-half rank correlation at
  splits 1/3, 1/2, 2/3, real against null. This is what separates "no signal" from
  "signal at a different horizon than a half window", and it costs nothing.

## Budget

| chunk | cost | note |
|---|---|---|
| S0 re-run | 1.00x | gate: must correlate ~1.0 with the stored batch |
| FM | 1.00x | run first, exactly 1.00x |
| A1 + N0 | 2.10x | |
| B1 pilot | 0.45x | 150 seeds/arm at ~1.5x |
| **total** | **~4.6 S0-batches** | ~3.6M evaluation-equivalents |

That does not fit one sitting. Run as resumable chunks by arm, checkpointed by
seed, in the order FM, S0-gate, A1+N0, B1 pilot. The S0 gate runs before the
expensive arms, because if the re-run does not reproduce, none of the paired
comparisons mean anything and there is no point paying for them.