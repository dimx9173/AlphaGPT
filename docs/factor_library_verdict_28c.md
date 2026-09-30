# The factor library question, answered by measurement

Date: 2026-09-30. Commit context: 62dfe41.

## The question

The 12-factor gene pool is dead: no factor has timing signal, and across 50 GA
runs on 5 datasets not one beat a constant levered long. Two external reviews
(Codex, Kimi) were asked whether to extend the gene pool, what to replace it
with, whether the target was even right, and what the cheapest decisive test
would be. Both independently advised against extending it. This document
records what happened when the two structures they proposed were measured, and
the one structure neither of them could have known about.

## What both reviews got right

- **Do not extend the gene pool.** The 12 factors are 12 reparametrisations of
  one 30m OHLCV series. A 13th raises selection pressure, not the ceiling.
- **The causal normalisation caps what direction is even possible.**
  `_trailing_robust(x, 200)` maps everything into `[-5,5]` stationary wiggle, so
  LOG_VOL, VOL_CLUST and HL_RANGE can only say "vol is high versus its own last
  200 bars". Recombining stationary wiggles with ADD/MUL/SIGN/RANK cannot
  recreate direction. The volatility factors are risk signals, not direction
  signals, and no grammar over them will produce a directional edge.
- **The measurement floor binds.** A Sharpe CI of +/-2.76 on 184 OOS days and
  N_eff 1.7 mean a genuine Sharpe-0.4 edge is undetectable by construction. No
  factor fixes a denominator.
- **The target was wrong.** A per-coin 30m timing book shorts the only reliably
  positive term, pays turnover twice, and pays funding on the short leg.
  Beating a constant long with 30m timing is paying costs to fight drift.

## What was measured

| Candidate | Structure | Result | Verdict |
|---|---|---|---|
| 12 existing factors | same-bar OHLCV transforms | all \|t\| < 1.40, none beat B&H | rejected |
| quote_volume, trades | the 2 unused bar columns | all \|t\| < 1.45, none beat B&H | rejected |
| 30m reversal | same-bar, lagged correctly | gross +0.86 excess, **-14.36** at the 0.0004 fee | edge below the cost |
| F_A funding carry | new data axis, contractual cashflow | real timing lift, best **-1.110** vs +1.122 B&H | carry below the drift |
| F_B cross-sectional | cross-section, demeaned | **+16.565 uncorrected, -1.184 corrected** | look-ahead |
| F_C vol-scaled long | risk overlay | Sharpe **+1.110**, vol-shuffled null **+1.191** | timing adds nothing |

## The two findings worth keeping

### 1. A real edge exists, and it is smaller than the fee

The 30m reversal is genuine. A variance-ratio test rejected iid on signed
returns, and the autocorrelation function measured against an in-coin shuffled
null shows reversion of about **-0.03 at lag 1 bar and lag 48 bars**, clearing
the null at t_eff = -1.92 and -2.18. This is the only directional structure
found in the dataset by any method.

Gross of costs it beats the passive hold by **+0.86 Sharpe**. At the
framework's own fee of 0.0004 it is **-14.36**, because turnover is 0.1545 per
bar. Break-even sits between Binance's maker and taker tiers. The effect is
real and the venue will not let you have it.

### 2. The funding carry is real and too small

F_A is the factor with the best mechanism in this project: funding is a
contractual cashflow, not a price statistic, and a book short sustained
positive funding receives payment whether or not prices are predictable. It
was tested favourably -- rebalanced only at settlements, scale fitted on train
only, credited through the same code path that charges funding, swept over
both knobs.

The carry is real. Timing lift over a funding-shuffled null is **+1.76 to
+1.83** across settings, and every setting collects income rather than paying
it. The verdict is still rejection: best of 12 settings is **-1.110** against
**+1.122**, and beta-hedging makes it worse at **-1.388**.

The P&L decomposition explains why. The carry income is roughly **one fifth
the size of the price exposure** it is attached to, and the hedge cannot
separate them because the funding stream and the short position are the same
position. Unconditional funding on these 28 majors averages 0.000021 per
settlement, about half a percent a year, against a +1.1 Sharpe drift.

## F_B: the retraction

The most important result in this document is the one that was wrong.

A cross-sectional momentum book reported Sharpe **+16.565**, weekly block
bootstrap 95% CI **[14.8, 20.7]**, **P(Sharpe <= 0) = 0.0000**, IC of the same
sign in all three windows, **+21.5** over a turnover-matched null, and 26/28
coins positive. Every check passed. It was look-ahead: the signal was a
trailing window ending at bar t, and the position was never lagged, so `p[t]`
contained `r[t]`. Every other position path in this repository applies
`np.roll(pos, 1)` first; F_B did not.

Lagged correctly it is **-1.184**, and the consistent IC was the artifact
repeating itself in each window. The honest per-coin IC is **+0.0006**.

Two things are worth stating plainly:

- **A narrow confidence interval does not protect against a wrong
  definition.** The bootstrap CI was narrow because the defect was present in
  every resampled block. A block bootstrap measures sampling uncertainty; it
  cannot measure whether the estimator is the one you meant to compute.
- **The first version of the bootstrap script was also wrong**, in a different
  way: a hand-rolled annualisation that did not match the production
  `metrics()`, reporting a median of +0.395 against a point estimate of
  +16.565. Two independent bugs in one result, both pointing the same way.

`research/causality.py` now makes that class of bug fail loudly. It rebuilds a
signal against a replaced future, binary-searches for the first bar whose
value depends on data at or after it, and separately checks that a position
was lagged before being applied. On the real signals it localises the F_B peek
to bar 287 and catches the unlagged position.

## F_A was re-checked the same way, and survives

Funding is charged on the settlement bar itself, and F_A's position was
rebuilt at settlement bars using a carry sum containing that same bar's rate.
So F_A was audited for the identical defect:

    variant                       Sharpe      null      timing lift
    same-bar (look-ahead)         -1.110    -2.939            +1.829
    lagged 1 bar (causal)         -1.202    -2.962            +1.761

The lift barely moves. F_A's positive timing result is not an artifact, and it
fails for the reason stated, which is size rather than bookkeeping.

## F_C: the risk overlay that both reviews recommended

Both reviews said the honest target is volatility, not direction. F_C tests
that directly: a long-only book inversely scaled by trailing realised
volatility, renormalised to a fixed average exposure, against a constant long
of the same exposure and a vol-shuffled null.

    win   target    Sharpe      MDD    exposure   turnover    MDD vs flat
     48   0.125    +0.715   0.0988      0.125     0.0017      -0.0804
    288   0.125    +1.063   0.0934      0.125     0.0003      -0.0859
    288   0.250    +1.110   0.1622      0.228     0.0003      -0.0171
    576   0.250    +1.159   0.1622      0.229     0.0002      -0.0171

    constant long, lockbox:  Sharpe +1.122   MDD 0.1793
    vol-scaled (288/0.25):  Sharpe +1.110   MDD 0.1622
    vol-SHUFFLED NULL:       Sharpe +1.191   MDD 0.1594

**The null beats the real thing.** A book that sorts coins by a shuffled
version of their own volatility scores +1.191 against the real book's +1.110,
with the same exposure and the same turnover. The only difference is the
ordering, and the ordering is not informative. The apparent MDD improvement is
proportional to the exposure reduction, not to skill. The sign also flips
across windows: train +0.436, validation **-1.279**, lockbox +1.110.

F_C is rejected. Volatility on these bars is not forecastable by the one method
that could have worked.

## The answer to the question asked

**Do not expand the gene pool. There is no factor library to build.**

Six structures were measured, spanning every data axis available in this
project: same-bar price/volume transforms, the two unused bar columns, a
measured 30m reversal, funding, the cross-section, and a volatility overlay.
One real effect was found. It is smaller than the cost of trading it, and the
one structure that looked best was a look-ahead.

The remaining value of this project is the apparatus that established this:
real Binance funding, fail-closed accounting, the null calibration, the
selection-lift probe, the power ceiling, the benchmarked holdout, and now a
causality guard. That is a real asset and it is the part that survived contact
with the data.

## Reproduce

    .venv2/bin/python tools/variance_ratio_28c.py
    .venv2/bin/python tools/reversal_probe_28c.py
    .venv2/bin/python tools/fee_decomposition_28c.py
    .venv2/bin/python tools/unused_column_screen_28c.py
    .venv2/bin/python tools/fund_carry_sweep_28c.py
    .venv2/bin/python tools/fund_carry_alignment_28c.py
    .venv2/bin/python tools/rel_rev_causal_28c.py
    .venv2/bin/python tools/causality_guard_on_real_signals.py
    .venv2/bin/python tools/vol_scaled_long_28c.py
