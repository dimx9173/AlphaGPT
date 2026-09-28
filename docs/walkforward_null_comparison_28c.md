# Walk-forward did not close the real/null gap

Date: 2026-09-28
Commits: 5301017 (folds), b899d7c (walkforward mode)
Artifacts: results/wf_real/, results/wf_null/

## Question

The standard contract gave one training window, one validation window, and one
lockbox. A 100-generation genetic search sees that single validation window tens
of thousands of times and is not charged for it. Ten real searches and ten null
searches were indistinguishable (Mann-Whitney p=0.13).

If that gap is caused by evaluation reuse, removing the reuse should close it.
This run tests that directly.

## Protocol

Twenty matched runs, seeds 101-110, 25 generations, 16 population, 4 expanding
folds. Real and IID-null differ only in the data: the null is block-bootstrapped
per coin with cross-coin correlation destroyed (0.669 to 0.004), fat tails and
volatility clustering preserved.

Each fold selects on its own training window and scores its validation window
exactly once. That validation window then becomes training data for the next
fold. The final fold is a holdout, scored outside the selection loop, never
trained on. Signal normalisation is refit per fold on that fold's own training
end, so no fold normalises on data it will later be scored against.

## Result

| Protocol | Real median | Null median | Gap |
|----------|-------------|-------------|-----|
| Standard  | +0.782 | +1.812 | -1.03 |
| Walk-forward | +1.242 | +1.676 | -0.43 |

Mann-Whitney walk-forward real vs null: p=0.26, not significant.
Positive holdout Sharpe: 6/10 real, 8/10 null.

## What this rules out

The gap narrows in the right direction and real data moves up, but null still
wins. Evaluation reuse is a contributing factor, not the whole cause.

The null produces a positive holdout in 8 of 10 searches even with one
selection pass per window. The search itself finds something that looks
positive on data with no structure in it, under any evaluation protocol tried
so far.

## Fold-level detail (real run_01)

| Fold | Train Sharpe | Validation Sharpe |
|------|--------------|-------------------|
| 0 |  +1.92 | -0.23 |
| 1 |  -1.65 | +4.56 |
| 2 |  -1.19 | +3.09 |
| Holdout | -- | -2.46 |

The fold that produces the final formula (fold 2) trains on the worst data and
validates on the best. The holdout it is then scored on gives -2.46. The
protocol is working as designed; the answer is negative.

## Next

Remaining levers, in order of expected effect:

1. **Search freedom.** The grammar allows 12 tokens from 28 operators, a large
   space to search against ~35k bars. A smaller grammar, or an economically
   motivated restriction, reduces the number of things that can look good by
   chance.

2. **Selection-bias correction.** CSCV, PBO, or a selection-adjusted Sharpe
   that explicitly prices the fact that the best of many is reported.

3. **Accept the negative result.** If the search cannot find edge even under a
   correct protocol, the conclusion is that this factor set does not support an
   edge at this sample size, not that the search needs another knob.
