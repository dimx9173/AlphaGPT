# Search-effort ladder: is more searching making it worse?

Date: 2026-10-03. Commit 4815006.

## Design

Fixed-12 arm, generations 2/5/10/20, real and iid null, 8 seeds each, 64 runs.
Everything else fixed: reduced grammar, reward v3, objective v3, selection keyed on
train only, population 16, 5 folds, real Binance funding.

Fixed-12 rather than variable length on purpose: the variable-length A/B already
showed freeing the length degrades null performance, so it is not a clean baseline
for a dose-response curve.

```
.venv2/bin/python research/run_effort_ladder_28c.py --seeds 8 --efforts 2 5 10 20 \
    --population 16 --folds 5 --workers 8 --outdir results/effort_ladder
.venv2/bin/python research/analyze_effort_ladder_28c.py --outdir results/effort_ladder
```

## Result 1: no dose-response on either arm

OLS of holdout excess Sharpe on log(generations), 4 levels x 8 seeds per arm.

| data | slope per log-generation | SE | t |
|---|---|---|---|
| real | -0.188 | 0.440 | -0.43 |
| null | +0.186 | 0.435 | +0.43 |

Both flat. The real-minus-null gap is negative at gen 2, positive at gen 5, most
negative at gen 10 and negative again at gen 20, with no ordering. Raw table:

| gens | real | null | gap | gap t |
|---|---|---|---|---|
| 2 | -2.210 +/- 0.325 | -1.249 +/- 0.375 | -0.961 | -1.94 |
| 5 | -1.657 +/- 0.556 | -2.448 +/- 0.510 | +0.791 | +1.05 |
| 10 | -3.192 +/- 0.880 | -1.333 +/- 0.398 | -1.860 | -1.93 |
| 20 | -2.201 +/- 0.389 | -1.044 +/- 0.558 | -1.157 | -1.70 |

No gap at any level is separable from zero. The oscillation is noise at n=8.

## Result 2: the ladder is measuring real variation, not a stuck search

A flat curve is only informative if the search actually changed its answer as effort
grew. Across the four effort levels, 0 of 48 seed-level formula pairs were identical.
Every level produced a different genome. The variation being measured is genuine; it
is simply not ordered by effort.

Reproducibility: all 8 gen-10 real seeds are bit-identical between this harness and
the variable-length A/B harness, which reach the same computation by different code
paths.

## Result 3: what this rules out

95% CI on the real-data slope is [-1.050, +0.674]. A dose-response larger than 0.86
excess-Sharpe per log-generation would have appeared. Across 2 to 20 generations
(4.3 log-generations) any total effect above 3.73 is excluded.

So large overfitting is ruled out and small overfitting is not. More importantly the
pre-registered shape test returns **both flat**, which was the branch meaning effort is
not the variable and the factor set has no conditional timing power for the search to
find.

## Result 4: the only number that has never moved

Across every real-data run in both this experiment and the variable-length A/B:

| | runs | beat buy-and-hold |
|---|---|---|
| effort ladder | 32 | 3 |
| variable-length A/B | 16 | 3 |
| combined | 48 | **6** |

Absolute Sharpe on real data reaches +1.33 while buy-and-hold on the same bars is
+1.42. The book tracks the drift and pays costs to do it.

## Conclusion

Two candidate explanations have now been eliminated by experiment rather than by
argument: the search space (variable-length A/B) and the search effort (this ladder).
Neither is the binding constraint.

What remains is the objective. The reward scores absolute Sharpe at w_psh = 0.10 and
excess over the hold at w_excess = 0.30, so a book that simply holds long is scored
positively before it has demonstrated any timing. The gate tests excess only. The
reward and the gate are not measuring the same thing, and the search is optimising the
one it is given.
## What was NOT concluded

The ladder has 8 seeds per cell. It cannot resolve an effect smaller than roughly 0.9
excess-Sharpe per log-generation. A small real effect, or a small real overfitting
effect, remains compatible with this data. The conclusion is that there is no LARGE
effect of effort, not that effort is exactly irrelevant.

## Next experiment this points to

Run the ladder again with the reward restricted to what the gate actually tests:
w_psh = 0, w_mean_ic = 0, w_excess = 1.0. The absolute-Sharpe and mean-IC terms are
the only routes by which "hold long" currently earns reward on a drifting universe.

The outcome is decisive in both directions:

- If holdout excess stays negative, the factors carry no conditional timing power and
  further search is wasted effort. Stop searching and either widen the factor set or
  stop.
- If holdout excess rises, the lever was the reward's disagreement with the gate, and
  that is now identified.

This is the same shape as the two experiments already run: hold everything fixed, vary
one thing, and let the null arm say whether the improvement is real.