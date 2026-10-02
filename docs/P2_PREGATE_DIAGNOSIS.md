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

positive_coins median went 0/28 -> 26.5/28.
Closing the validation leak moved this holdout by roughly 8 Sharpe, seed for seed
on the same window.

What that number is and is not. It measures how much of the earlier negative
result was an artifact of selecting on a window that had been read 3,600 times.
It does NOT measure an improvement in strategy quality, and the fold-level
numbers say so:

    P2 validation Sharpe positive in             38/90 folds
    P2 holdout excess over buy-and-hold positive in  0/10

A pre/post comparison of the two walkforward batches is not usable here: they
differ in fold count (30 vs 90), window length (144 days vs 63 days) and fold
layout, and the post-fix batch has a worse median validation Sharpe (-1.033
against -0.115). That is confounded so no claim is made from it, but it is why
the 8 Sharpe should not be read as the strategy getting better.

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

The reward is flat rather than reversed. -0.09 at n=90 is indistinguishable
from no relationship.

Correction to an earlier claim in this file. I had written that the composite
reward "points the wrong way while rank-IC carries the signal". Testing that
directly inside the real regime does not support it. Over the same 90 folds:

    composite reward -> validation reward    -0.065
    composite reward -> validation mean_IC   +0.056
    mean_IC          -> validation mean_IC   +0.199
    mean_IC          -> validation reward    -0.254

and on the practical test, taking the top 9 of 90 by each key and looking at the
mean validation rank-IC of those folds:

    by composite reward   +0.00216
    by mean_IC           +0.00027
    random baseline      +0.00115

The composite reward selects the better set. The earlier statement conflated two
different questions. Rank-IC is more robust ACROSS data regimes, which is what
the real-versus-null probe measured; that is not the same as being a better
selector WITHIN one regime.

So the three items are refused for a reason that is now specific. They all
re-weight or re-constrain the composite reward, and the composite reward has no
relationship with the out-of-sample objective. PRP-2's breadth term would force
`positive_coins` up on the same twenty folds whose validation reward is negative;
PRP-3's cost term would re-price a quantity that is already net; PRP-1's extra
freedom would buy more of the near-tie the argmax margin already measures.


## 7. Direct check: what position does the search actually hold?

Rebuilt the portfolio position the evaluator would produce for each selected
formula, on the holdout window, across the 28 coins:

    seed    mean    std     min     max   frac long  frac active
    101   0.0714 0.0002  0.066  0.071     1.000       1.000
    102   0.2493 0.0005  0.247  0.250     1.000       1.000
    103   0.0405 0.1062 -0.215  0.231     0.650       1.000
    104   0.1262 0.0930 -0.135  0.241     0.865       1.000
    105   0.2497 0.0005  0.246  0.250     1.000       1.000
    106   0.2033 0.0534 -0.105  0.243     0.988       1.000
    107   0.0714 0.0004  0.068  0.071     1.000       1.000
    108   0.2500 0.0000  0.250  0.250     1.000       1.000
    109   0.2499 0.0003  0.246  0.250     1.000       1.000
    110   0.1429 0.0000  0.143  0.143     1.000       1.000

Seven of the ten selected "strategies" hold a fixed long position for every bar
of the holdout, with a standard deviation below 0.001. Three sit at the 0.25
position cap. These are not strategies; they are levered buy-and-holds whose
size the search chose.

Only three seeds vary at all (103, 104, 106), and they carry the three lowest
holdout Sharpe values in the batch. The relationship runs the wrong way: the
constant-position runs have a median position of 0.2493 and the varying ones
0.1262.

This is the clearest statement of the finding. It is not that the search picked a
bad formula. It is that seven times out of ten the search converged on holding,
and the only thing it optimised was how much.


## 8. The space can express timing; the search selects against it

250 random formulas, each evaluated into a portfolio position across the 28
coins through the same evaluator the search uses:

    position std        median 0.03206   max 0.12116
    std < 0.001          9 / 250   (3.6%)
    std < 0.01           84 / 250
    mean position       median +0.0003

Against the ten selected formulas:

    position std        median 0.00045
    std < 0.001          7 / 10   (70%)

This changes the diagnosis again, and in a way that matters. Ninety-six per cent
of random formulas produce a position that moves. The hypothesis space is not
incapable of expressing a timing strategy; variation is there for the taking.

The search picks a constant position nineteen times more often than a random draw
does. Whatever the reward is doing, it is not failing to find movement because
movement is scarce. It is selecting against movement.

That points at the objective rather than the space or the operators, and it is
consistent with everything else measured here: argmax_margin of 0.009 against a
0.70 fold-to-fold scatter, a validation rank-IC that collapses to zero in 32 of
90 folds, and a composite reward whose top-9-of-90 selection lands at +0.00216
against a +0.00115 random baseline.

It also explains why the three PRPs could not have worked. PRP-1 adds freedom to
move, against an objective that prefers not moving. PRP-2 penalises legs that
disagree, which flattens the position. PRP-3 reprices turnover, which is the
one quantity that distinguishes a moving position from a constant one.


## 9. The mechanism: the only term that could penalise trading is four orders too small

Section 8 said the search selects against movement. That is the right
observation but the wrong mechanism, and the follow-up corrects it.

Across the 90 real-arm selection folds:

    corr(position_std, reward)     +0.119   (weakly POSITIVE)
    corr(position_std, train_IC)   +0.531   (positive)
    corr(turnover,      train_IC)  +0.866   (strongly positive)

The reward does not penalise movement. If anything it mildly rewards it. So the
constant positions are not the reward's preference, and section 8's framing is
withdrawn.

The scale of each reward term is the mechanism:

    mean_ic term          spread 0.02570   weight 1.00
    0.02 * turnover       spread 0.000502  weight 0.02
    total reward          spread 0.6187

The turnover term's entire dynamic range is 0.0812% of the reward's total
spread. It is the only term in `shape_reward` that would penalise trading, and it
is roughly four orders of magnitude too small to do anything.

Movement is therefore free. Given movement is free and the argmax margin is
0.009 against a 0.70 fold-to-fold scatter, the selection has no reason to prefer
a moving position and lands on a constant one whenever the near-ties happen to
include one. The GA is not choosing to hold. It is choosing arbitrarily, and a
constant position is the modal shape of a near-tie in this space.

This is the precise reason all three items fail, and it is a single number rather
than a judgement:

  - PRP-1 adds freedom to move. The objective already permits it freely, so more
    freedom changes nothing.
  - PRP-2 penalises legs that disagree. Flattening the position is the same
    thing that is already happening, for free.
  - PRP-3 reprices turnover. This is the only one that touches the actual
    mechanism, and it was rejected because its premise was wrong: the Sharpe it
    reprices is already net of costs, and the `-0.02 * mean_turn` term it would
    have touched is the same negligibly weighted term identified here.

That last point is a real limitation of the PRP-3 review rather than of PRP-3
itself. The review was right that the stated premise was inverted, and the
rejection followed from the stated premise. The underlying observation, that
turnover is priced at a scale that cannot influence selection, survived the
review and is recorded above.
