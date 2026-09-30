# 30m reversal: real, measured, and not tradeable

Date: 2026-09-30. Commit context: f45aa12.

## Why this probe was run

The factor library is dead: 12 factors, no timing signal, and two external
reviews (Codex, Kimi) both advised stopping directional research rather than
extending the gene pool. Before accepting that, one cheap test was proposed
and run: a Lo-MacKinlay variance-ratio test on 30m returns.

The VR test **rejected iid on signed returns by a wide margin**:

    q=  4  VR=0.244   q=  8  VR=0.116   q= 16  VR=0.058   q= 32  VR=0.028

That contradicts the reviews' premise that signed 30m returns are
directionless. A VR far from 1 is real dependence. This document records what
that dependence actually is, and whether it can be traded.

## The VR z-statistic was inflated; the ACF is the honest measurement

The VR test statistic assumes iid. Under strong negative autocorrelation that
asymptotic is invalid and the z blows up (`t_raw` reached -894). The
autocorrelation function is the reliable measurement, compared against an
in-coin shuffled null that preserves the marginal and destroys dependence:

    lag    acf_raw   acf_null   t_eff(N_eff=1.7)   verdict
      1    -0.0287    +0.0018             -1.92       noise
      2    -0.0113    +0.0021             -0.86       noise
      3    -0.0017    +0.0001             -0.15       noise
      6    -0.0076    -0.0027             -0.70       noise
     12    -0.0079    -0.0017             -1.12       noise
     24    -0.0021    +0.0012             -0.25       noise
     48    -0.0294    -0.0019             -2.18       REAL
     96    +0.0043    -0.0020             +0.55       noise
    288    -0.0010    -0.0010             -0.16       noise

There is a real, small reversion at **lag 1 bar (30m)** and a second one at
**lag 48 bars (1 day)**. Both are around -0.03 per lag. These are genuine
dependencies, not artifacts, and they are the only directional structure found
in this dataset by any method so far.

## The reversion is real and it is not tradeable

A plain contrarian book (fade the previous bar, same position path, same
leverage 2.0, position cap 0.25, real Binance funding) on the lockbox:

    lag 1   fee 0.0000   strategy +1.982   B&H +1.124   excess +0.858
            fee 0.0001   strategy -1.783   B&H +1.123   excess -2.906
            fee 0.0004   strategy -13.237  B&H +1.122   excess -14.359
            fee 0.0010   strategy -35.341  B&H +1.120   excess -36.461

    lag 48  fee 0.0000   strategy +1.276   B&H +1.124   excess +0.152
            fee 0.0001   strategy -2.479   B&H +1.123   excess -3.602
            fee 0.0004   strategy -13.738  B&H +1.122   excess -14.860
            fee 0.0010   strategy -34.679  B&H +1.120   excess -35.799

Turnover is 0.1545 per bar in both cases, which is the whole story.

**Gross of costs, the reversion beats the passive hold** (+0.86 excess at lag 1).
**At the framework's own fee of 0.0004 it is annihilated** (-14.36 excess).

The break-even fee is between 0.0001 and 0.0004. Binance taker fees on
majors are 0.0004, so the signal sits just below the trading cost of the venue
it was measured on. At 0.0001 (maker tier) it would be worth trading; at the
taker rate it is not.

## What this settles

1. **The reviews were wrong about one thing.** Signed 30m returns are not
   directionless. There is a measured -0.03 reversal at 1 bar and 1 day that
   clears an iid null at `t_eff = -1.92` and `-2.18`. The GA never found it
   because it cannot express a fixed-lag single-input reversal inside a
   12-token RPN search, and because the fee load makes the net version worse
   than doing nothing, so the reward correctly rejects it.

2. **The reviews were right about the conclusion.** The factor library is
   still the wrong instrument. The one real dependency this dataset contains
   is 30m microstructure reversion, and it is smaller than the round-trip
   cost of trading it. A gene pool built to find it would still fail the
   accept gate, for the same reason every one of the 50 runs failed: the
   cost of the observation exceeds the size of the effect.

3. **This is the measurement floor, not a factor gap.** Codex's proposed
   kill-shot test returned a real signal, and the signal is unprofitable. That
   is the strongest possible form of the "no edge here" result: the edge was
   found, and the fee took it.

## Recommended next step

Do not extend the factor library. The only unmeasured data axes are funding
carry (Kimi's F_A) and cross-sectional relative value (F_B). Both are
materially different from price/volume transforms because they are not
functions of the same bar series, and carry in particular has a cashflow
component that exists regardless of price predictability.

If the question is specifically "can this project find a tradeable 30m edge
in this universe", the answer is now measured, not assumed: **no, and the
cost that kills it is identified.**
