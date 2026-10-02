# P2 pre-gate: partial real-arm reading (2026-10-02)

The batch is 9/20 (real arm only so far). Recorded here because one
finding is already decisive and would be lost if we waited for the null arm.

## What PRP-0 bought

Real-arm holdout Sharpe, pre-PRP-0 versus post-PRP-0:

| seed | pre  | post |
|---|---|---|
| 101 | -4.867 | +3.314 |
| 102 | -4.139 | +3.664 |
| 103 | +0.641 | +3.069 |
| 104 | -1.682 | +3.483 |

positive_coins went from a median of 0/28 to 27/28.

Closing the validation-window selection leak was worth roughly 8 Sharpe. The
earlier negative batches were partly an artifact of selecting on a window that
had been read 3,600 times.

## What it did not buy

Buy-and-hold Sharpe on this holdout is +3.675. It is 63 days of
uptrend, and every real run is scored against the same tape.

    portfolio_sharpe          +0.000 .. +3.674
    excess_vs_buy_and_hold    -5.783 .. -2.402
    seeds with excess > 0     0/9
    corr(sharpe, |mean_position|) = +0.599

Every single real run UNDERPERFORMS holding the same book passively, and the runs
with the highest Sharpe are the ones with the largest average position. Three
seeds sit at the 0.25 position cap with 28/28 positive legs, which is what a
constant leveraged long looks like on a rising tape.

So the search now behaves honestly and the honest answer is: it found beta.

## Consequence for the gate

The pre-registered rule compared real versus null on raw Sharpe. On a window
where buy-and-hold scores +3.675, that comparison cannot distinguish "found
signal" from "took more exposure", because both arms are graded on the same
rising tape.

research/analyze_pregate_baseline.py now carries a fifth criterion,
beats_buy_and_hold, requiring at least 6/10 real seeds to post positive excess.
On the current real arm that criterion is already failing at 0/9.

This is recorded rather than adjusted. The gate exists to catch exactly this.
