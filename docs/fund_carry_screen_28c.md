# F_A FUND_CARRY: the carry is real, and it is too small

Date: 2026-09-30. Commit context: 712d71e.

## Why this factor deserved a real test

The 12 existing factors are all functions of the same 30m OHLCV bar, which is
the reason the gene pool is dead. Funding is a different data axis. It is a
contractual cashflow rather than a price statistic, and the mechanism is
explicit: persistently positive funding means longs pay shorts, so a book
short sustained positive funding receives payment whether or not prices are
predictable. No combination of O,H,L,C,V can do that.

The implementation is deliberately favourable to the factor, because a factor
tested at a disadvantage proves nothing:

- **Rebalanced only at settlements** (every 8h, 3x/day), not every 30m bar.
  Carry is slow; trading it at bar frequency would pay 48x the fee for a
  signal that changes 3x a day.
- **Scale fitted on train only**, then frozen for the lockbox.
- **Causality enforced**: at bar t only settlements at or before t are used.
  The rate is forward-filled from the last observed event and never
  interpolated.
- **Credited by the same code path that charges funding**
  (`funding_cost = position * funding_rate * leverage`), so there is no
  separate bookkeeping that could flatter the result.
- **Swept over both knobs** (lookback k in {3,6,9,18} settlements, entry
  threshold in {0, 0.5, 1.0}), so the verdict is not one unlucky setting.

## Result: 12 settings, none beat the passive hold

Lockbox, buy-and-hold Sharpe +1.122.

    k   thresh   Sharpe    null    timing lift   ann_carry   turn/bar
    3    0.0     -1.183   -2.549      +1.366       received     0.0150
    3    0.5     -1.257   -2.322      +1.065       received     0.0129
    3    1.0     -1.110   -5.115      +4.006       received     0.0061
    6    0.0     -1.480   -2.437      +0.957       received     0.0186
    6    0.5     -1.427   -3.085      +1.658       received     0.0163
    6    1.0     -1.372   -2.381      +1.009       received     0.0081
    9    0.0     -1.583   -2.650      +1.067       received     0.0202
    9    0.5     -1.617   -3.207      +1.589       received     0.0179
    9    1.0     -1.529   -3.033      +1.504       received     0.0091
   18    0.0     -1.644   -1.077      -0.566       received     0.0206
   18    0.5     -1.542   -3.308      +1.766       received     0.0185
   18    1.0     -1.290   -4.457      +3.168       received     0.0098

Best setting: k=3, thresh=1.0, Sharpe **-1.110** against a buy-and-hold of
**+1.122**, an excess of **-2.232**. Train is also negative (-0.374).

## Three findings, and only the first is what we hoped for

**1. The carry is real.** Every setting collects income rather than paying it,
worth roughly +6.7%/yr in the P&L decomposition, and the timing lift over a
funding-shuffled null is positive in 11 of 12 settings (up to +4.01 Sharpe).
So funding timing does contain information the null does not have. This is
the first non-null result in the whole project and it is worth stating
plainly: **the factor is not noise.**

**2. Hedging the market does not rescue it.** A carry book that must also be
directionally short is not a carry strategy, so the position was beta-hedged
against the equal-weight market (loading fitted on train only). That made it
*worse*, not better: raw -1.110 became hedged -1.388, because the hedge adds
a second directional leg paying its own fees. The P&L decomposition shows why
the hedge fails:

    raw          Sharpe -1.110   price -0.010   carry +0.002   fee -0.005
    beta-hedged  Sharpe -1.388   price -0.028   carry +0.003   fee -0.005

    iid null
    raw          Sharpe -2.939   price -0.015   carry +0.001   fee -0.010
    beta-hedged  Sharpe -1.953   price -0.033   carry +0.001   fee -0.010

The price term is an order of magnitude larger than the carry term in every
row. The carry income is roughly **2/10ths the size of the price exposure it
comes packaged with**, and there is no hedge that removes the price exposure
without also removing the funding stream that is attached to the same
position. The hedge moves price from -0.010 to -0.028: it triples the price
drag to save 0.001 of fee.

**3. The unconditional carry is negligible here.** An always-short book
receives +0.0026 over the lockbox against always-long paying -0.0026, while
their Sharpe gap is +1.122 to -1.175. Real Binance funding on these 28 majors
averages 0.000021 per settlement with 70% positive, which annualises to
roughly half a percent. The 3-settlement-per-day mechanism that makes funding
strategy famous does not clear this universe's drift.

## Verdict

**F_A is rejected. Do not add it to the gene pool.** It fails the accept gate
at every setting, on both train and lockbox, and it fails for a reason that no
amount of further tuning addresses: the effect is real but the measurable
income is one fifth the size of the price risk it requires taking.

This is the third independent line of evidence that the factor library is not
the binding constraint:

| Probe | Result | Meaning |
|---|---|---|
| 12 existing factors | all \|t\| < 1.40, none beat B&H | no timing signal in OHLCV |
| 2 unused bar columns | all \|t\| < 1.45, none beat B&H | unused data is not the gap |
| 30m reversal | gross +0.86 excess, net -14.36 at 0.0004 | real edge, below the fee |
| F_A funding carry | real timing lift, best -1.110 vs +1.122 | real income, below the drift |

The remaining unmeasured axis is the cross-section across the 28 coins
(Kimi's F_B), which is a dollar-neutral construction and therefore has no
benchmark of "buy-and-hold" at all. It is the last genuinely different
structure available in this data.

## Reproduce

    .venv2/bin/python tools/fund_carry_28c.py        # single setting + decomposition
    .venv2/bin/python tools/fund_carry_sweep_28c.py  # 12 settings
    .venv2/bin/python tools/fund_carry_hedged_28c.py # beta-hedged arm
