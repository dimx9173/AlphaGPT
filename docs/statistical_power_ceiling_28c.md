# Why the 28c framework cannot produce a statistically resolvable verdict

Date: 2026-09-27
Subject: v3c (`results/ga_28c_100gen_v3c_seed42.json`), 28-coin / 30m / 2.03y contract

This document records a measurement, not an opinion. It is the admission gate for
any future GA run: **do not start a new 28c search until someone has read this
and accepted that the framework cannot answer the question being asked.**

## The short version

The v3c lockbox Sharpe of +0.479 has a 95% confidence interval of roughly
[-1.7, +2.9]. The interval contains zero. Under this contract, a candidate
cannot be shown to differ from a random strategy, and no amount of additional
searching changes that.

## Measurements

### 1. The lockbox is too short

8,833 bars = 184 daily observations.

    point Sharpe                 : +0.479
    weekly block bootstrap 95% CI : [-1.738, +2.901]
    t-statistic for mean != 0    : +0.341   (1.96 required)
    P(Sharpe <= 0)               : 0.336

Serial correlation of the daily returns is approximately zero (sum of lags
1..3 = -0.026), so the statistical efficiency is already near the theoretical
maximum. This is not a resampling artefact.

### 2. The required sample does not exist

95% CI half-width on a Sharpe estimate, as a function of OOS length:

| OOS length | half-width |
|---|---|
| 185 days (current lockbox) | +/-2.92 |
| 300 days (all that remains after a 1y burn-in) | +/-2.29 |
| 1.2 years | +/-2.00 |
| 4.8 years | +/-1.00 |
| 19.2 years | +/-0.50 |

The contract contains 2.03 years in total. Resolving a Sharpe of 0.5 from zero
needs 19.2 years: a shortfall of 23x.

### 3. Diversification cannot close the gap

    cross-sectional pairwise correlation (coins)  : +0.639  -> N_eff 1.53
    strategy leg-return correlation              : +0.450  -> N_eff 2.13
    same-formula position correlation            : +0.498

The strategy runs one formula across 28 names, so the positions are more
synchronised than the prices. Even assuming a physically impossible
correlation of exactly zero:

| mean pairwise rho | N_eff | implied t |
|---|---|---|
| 0.639 (observed) | 1.53 | +0.174 |
| 0.200 | 4.38 | +0.294 |
| 0.000 (unreachable) | 28.0 | +0.744 |

The theoretical ceiling is t = 0.744. The requirement is 1.96.

### 4. Leverage is mathematically irrelevant here

The t-statistic is invariant to a common scale factor:

    x1 leverage: sharpe +0.289  t +0.205
    x2 leverage: sharpe +0.289  t +0.205
    x4 leverage: sharpe +0.289  t +0.205

Raising notional size scales the mean and the standard deviation together.
It changes neither Sharpe nor significance.

## What this rules out

| Direction | Ceiling | Verdict |
|---|---|---|
| Extend the OOS window | 19.2 years needed | data does not exist |
| Diversify further | t -> 0.744 at best | still far below 1.96 |
| Increase leverage | no effect | scale-invariant |

All three routes terminate at the same place: the only remaining lever is a
higher per-bet edge, which means a better formula, not a longer or wider sample.

## Why the v3c result is not a bug in the accounting

An earlier run reported a large drawdown and few positive legs on a
pre-contract period. Two follow-ups pinned the cause:

- The universe is not responsible. Removing POL/KAS/RENDER and re-running v3c
  on the contractual lockbox reproduces +0.477, against +0.478 for 28 coins.
- The period is responsible. On 2023-09..2024-09 the same 25 coins give
  Sharpe -0.006 and MDD 0.289, with position activity unchanged at 0.193.

So the 0.0968 lockbox MDD is not evidence of risk control. It is what a
0.5-year window looks like when it happens to be favourable, which is exactly
what a +/-2.29 interval predicts.

## What the gate now reports

`evaluate_regime_gate` emits a `statistical_power` block and an
`evidence_class`. When the OOS window cannot resolve the gate's own Sharpe
threshold, `evidence_class` is `screening_only_not_statistically_resolved` and
an `evidence_caveat` accompanies the verdict.

A candidate may pass every screening rule and still be indistinguishable from
random. **Do not describe a `pass` in this framework as evidence that the
strategy works.** `live_adopted` remains `False` and nothing in the gate may
authorise live trading.

## What would change the answer

Only one thing: real out-of-sample history at a scale this framework does not
have. Concretely, that means either

- a much longer contract window, which the 30m perpetual data does not support
  (the series begins 2023-09-25 and the 28-coin common window begins 2024-09-13
  because POL lists that day), or
- accepting that the evaluation is a screen, and stating that in every artifact.

Absent either, additional GA generations will produce candidates that look
better and remain equally unresolvable. That is the specific failure this
document exists to prevent.
