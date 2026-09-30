# F_B cross-sectional momentum: the +16.6 Sharpe was look-ahead

Date: 2026-09-30. Commit context: 712d71e.

## What happened

A cross-sectional momentum book on the 28 coins produced the best numbers in
the entire project:

    lockbox Sharpe        +16.565
    weekly block bootstrap 95% CI   [+14.824, +20.698]
    P(Sharpe <= 0)        0.0000
    IC                    +0.027, +0.027, +0.029 on train / validation / lockbox
    lift over turnover-matched null   +21.517
    coins with positive IC            26/28

Every check passed. The IC had the same sign in all three windows, which is
the exact consistency test the 12 existing factors fail (they flip 5/12 ->
8/12 -> 5/12). It beat a turnover-matched null. The block bootstrap excluded
zero by a wide margin.

**All of it was an artifact.** The result was wrong and has been withdrawn.

## The bug

The position was built directly from a trailing 288-bar window that ends at
bar t:

    M[i, win-1:] = rolling sum of returns over bars (t-win+1 .. t)
    pos[i] = tanh(sig[i] / sd) * 0.25          # no lag

So `pos[t]` contains `r[t]`, and the accounting then multiplied `pos[t]` by
`r[t]`. Every other position path in this repository lags by one bar first:

    p = np.roll(p, 1); p[0] = 0        # research/ga_28c_30m_3y.py:389

F_B did not. On iid data the same alignment manufactures a correlation of
about **+0.43**; with the one-bar lag applied the same synthetic series shows
**-0.0035**. The signal was the position reading its own outcome.

This was verified on synthetic data before being believed, and the corrected
run was compared against the uncorrected one on identical inputs:

    lookback  sign     train       val   lockbox
        48     -1    -3.581    -0.953    -7.385
        48     +1    -5.213   -10.516    -6.317
        96     -1    -2.811    -2.798    -7.234
        96     +1    -3.648    -6.044    -2.355
       288     -1    -0.872    -3.607    -4.325
       288     +1    -2.768    -1.963    -1.184

    best setting: lookback 288, sign +1, Sharpe -1.184
    (the uncorrected version of this same code reported +16.565)

The "consistent IC sign across windows" did not survive either: the corrected
per-coin IC is **+0.0006**, which is a coin flip. What looked like a robust
cross-sectional anomaly was the look-ahead repeating itself in each window
with the same sign, which is precisely what a definition error does.

## Why the narrow confidence interval did not protect us

The bootstrap gave [+14.8, +20.7] and P(Sharpe<=0) = 0.0000. A tight interval
around a wrong number is exactly what a look-ahead produces, because the
artifact is present in **every** block, so resampling cannot remove it. The
block bootstrap measures sampling uncertainty. It cannot measure whether the
estimator is the one you meant to compute.

The first version of the bootstrap script was also wrong in a different way: it
used a hand-rolled mean/std annualisation that did not match the production
`metrics()`, and reported a median Sharpe of +0.395 against a point estimate
of +16.565. Using the production function for both fixed that scale mismatch.
Two separate bugs in one result, both of which pointed the same way.

## Verdict

**F_B is rejected.** With the position lagged the book loses at every lookback
and both signs, on every window, and its per-coin IC is indistinguishable
from zero.

Cross-sectional momentum remains a plausible strategy in the literature. It
does not exist in this dataset, this window, at this frequency, once the
position is causal.

## F_A funding carry: re-checked, and the result stands

The same audit was applied to the funding-carry factor, which had a subtler
version of the alignment problem: funding is charged on the settlement bar
itself, and the F_A position was rebuilt at settlement bars using a carry sum
that included that same bar's rate.

    variant                       Sharpe      null      timing  ann_carry
    same-bar (look-ahead)         -1.110    -2.939      +1.829    -0.0411
    lagged 1 bar (causal)         -1.202    -2.962      +1.761    -0.0290

The timing lift barely moves (+1.829 -> +1.761), so F_A's positive timing
result is not an artifact of that alignment. It still fails on absolute terms,
but the reason it fails is now known to be the size of the carry, not a
bookkeeping error. See `docs/fund_carry_screen_28c.md`.

## The complete ledger

| Candidate | Result | Verdict |
|---|---|---|
| 12 existing factors | all \|t\| < 1.40, none beat B&H | rejected |
| 2 unused bar columns | all \|t\| < 1.45, none beat B&H | rejected |
| 30m reversal (gross) | +0.86 excess, -14.36 at 0.0004 fee | edge below the fee |
| F_A funding carry | real timing lift, best -1.18 vs +1.12 B&H | carry below the drift |
| F_B cross-sectional | +16.57 uncorrected, -1.18 corrected | look-ahead |
| **F_C vol-scaled long** | not yet tested | — |

Five independent structures, all measured. One real effect found, and it is
smaller than the cost of trading it.

## The lesson worth keeping

The framework guards against missing data, wrong accounting, and selection
bias. It did not guard against a position that was not lagged, because nothing
in the pipeline asserts causality of the position path -- it is a convention
followed by hand in each file.

That is a real gap and it is now the highest-value thing left to fix: an
automated causality assertion, so that any position path which reads bar t and
is applied to bar t fails loudly instead of producing a Sharpe of +16.

## Reproduce

    .venv2/bin/python tools/rel_rev_28c.py             # uncorrected, kept for audit
    .venv2/bin/python tools/rel_rev_causal_28c.py     # corrected, lagged position
    .venv2/bin/python tools/fund_carry_alignment_28c.py
