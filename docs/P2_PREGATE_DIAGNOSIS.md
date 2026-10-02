# Why the pre-gate fails, in numbers (2026-10-02)

Batch: 10/20 runs so far, real arm complete enough to read.
Source: results/p2_pregate_baseline/, and the per-fold diagnostics added by PRP-0.

## 1. The repair worked

Real-arm holdout Sharpe, pre-PRP-0 versus post-PRP-0:

| seed | pre  | post |
|---|---|---|
| 101 | -4.867 | +3.314 |
| 102 | -4.139 | +3.664 |
| 103 | +0.641 | +3.069 |
| 104 | -1.682 | +3.483 |

positive_coins median went 0/28 -> 26/28.
Closing the validation leak was worth roughly 8 Sharpe, so the earlier negative
batches were partly an artifact of selecting on a window read 3,600 times.

## 2. What the search actually found: beta

Buy-and-hold on this holdout is +3.675. The window is 63 days of
uptrend; BTC is +7.5% over it, and 8 of the 10 walkforward validation windows are
rising.

    portfolio_sharpe        +0.000 .. +3.674
    excess_vs_buy_and_hold  -5.783 .. -2.402
    seeds with excess > 0   0/10
    corr(sharpe, |position|) +0.646

Every real run underperforms holding the same book passively, and the highest
Sharpe belongs to the largest average position. Three seeds sit at the 0.25 cap
with 28/28 positive legs, which is a constant leveraged long on a rising tape.

## 3. Why: the selection is choosing from near-ties

The F5 diagnostics added by PRP-0 answer this directly, over 90 selection
folds:

    median |overfit_gap|      0.7019
    median argmax_margin      0.009027
    ratio                     78x
    overfit_gap < 0           69/90
    validation mean_ic = 0    20/90

The margin by which the winner beat the runner-up on train is about
78x smaller than the fold-to-fold scatter in the validation reward. The
search is picking from a field of near-ties, so the validation number it lands on
is scatter, not skill. Validation rank-IC collapses to exactly zero in
20/90 folds even though train rank-IC is positive in
79/90.

That is the signature of a search with no gradient to follow: it moves a
negligible amount of reward per generation and keeps moving.

## 4. Consequence for the three PRPs

The pre-registered rule compared real against null on raw Sharpe. On a window
where buy-and-hold scores +3.675, that comparison cannot separate
"found signal" from "took more exposure", because both arms are graded on the
same rising tape.

research/analyze_pregate_baseline.py now carries a fifth criterion,
`beats_buy_and_hold`, requiring 6/10 real seeds with positive excess. The real arm
is at 0/10.

PRP-1, PRP-2 and PRP-3 each carry this gate as a precondition. On the evidence
above they stay NO-GO, and the reason is now specific rather than generic: the
search does not have a signal to amplify, so widening the space, constraining
breadth, or repricing costs cannot produce one.

## 5. What would change this

Not a bigger search. The measurement says the binding constraint is that the
objective has no gradient: argmax_margin is ~0.009 against a 0.70 fold-to-fold
scatter. Two changes follow directly and neither is in the current three:

  - make the objective something the market actually varies. Rank-IC has a
    positive train value in 79/90 folds and a validation value of zero in
    20/90, so it is the quantity to look at, and PRP-2's own C2
    shows direction is not currently a reward term at all.
  - score against buy-and-hold rather than against zero. On this universe a
    constant long has a Sharpe of its own, and every number in the table above is
    a statement about exposure until it is differenced.


## 6. Control: the population has no good member to find

Twelve random genomes scored on the same training window, no search involved:

    train_reward, sorted:  -3.2161 -2.8257 -2.4760 -2.3560 -1.9550 -1.4546
                            -1.3694 -1.1613 -1.1003 -0.4854 -0.4768 -0.4410

    spread                     2.7751
    margin over the runner-up  0.0358   (1.3% of the spread)

Every one of the twelve is negative. The search is therefore not climbing toward
a good region and failing to arrive; the good region is not present in the
population it starts from, and the reward differences between members are small
enough that selection cannot tell them apart.

### Correction: the search works, the target does not hold

The twelve random genomes are worse than every one of the GA's 90 selections, so
the search is doing its job. 90/90 selected genomes beat the best of the twelve.
GA train rewards run -0.3595 to +0.2592 against the random sample's -3.2161 to
-0.4410.

The failure is therefore not a weak search. It is that a positive training reward
does not survive:

    folds where train_reward > 0        20/90
    of those, validation_reward > 0      0/20

Every one of the twenty flips negative on validation. And across all folds:

    corr(train_reward, validation_reward) = -0.0891
    corr(train_IC,     validation_IC)     = +0.2309

The reward is not merely flat in gradient, it is not pointing where validation
is. Rank-IC carries a small positive relationship (+0.23) while the composite
reward carries none (-0.09), which is the same split the earlier probe found:
mean-IC separates real from null and Sharpe does not.

So the three items are refused for a reason that is now specific. They all
re-weight or re-constrain the composite reward, and the composite reward has no
relationship with the out-of-sample objective. PRP-2's breadth term would force
`positive_coins` up on the same twenty folds whose validation reward is negative;
PRP-3's cost term would re-price a quantity that is already net; PRP-1's extra
freedom would buy more of the near-tie the argmax margin already measures.
