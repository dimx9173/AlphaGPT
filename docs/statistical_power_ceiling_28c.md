# Why the 28c framework cannot produce a statistically resolvable verdict

Date: 2026-09-27. Corrected 2026-09-28.
Subject: v3c (`results/ga_28c_100gen_v3c_seed42.json`), 28-coin / 30m / 2.03y contract

This document records a measurement, not an opinion. It is the admission gate for
any future GA run: **do not start a new 28c search until someone has read this
and accepted that the framework cannot answer the question being asked.**

## Provenance warning, read this first

The headline `+0.479` and the correlation figures originally in this document
were measured on the factor set as it stood **before** commit `de2e353`, when
`LIQ_SCORE` was the constant `0.4` on every bar of every coin. `de2e353`
replaced it with a causal log-Amihud proxy. v3c's formula is
`F7 F3 ADD F7 ADD F5 DELTA F1 F9 GATE DELAY1 CORR` and `F1` is exactly that
factor, so **v3c's artifact is no longer reproducible from the current tree**,
even on the constant-funding basis it was produced under.

Re-measured with today's factors and today's code, same 8,833-bar lockbox:

| funding basis | v3c lockbox Sharpe |
|---|---|
| as recorded in the artifact (constant +0.0005, old `LIQ_SCORE`) | +0.478 |
| constant +0.0005, current log-Amihud `LIQ_SCORE` | **+0.199** |
| real Binance funding, current log-Amihud `LIQ_SCORE` | **-0.897** |

So `+0.479` is a historical number that this tree cannot regenerate. The
conclusions below are unchanged, and the conclusion is in fact stronger when
stated against the current code: the reproducible Sharpe is smaller, and the
real-funding Sharpe is negative. Nothing here depends on the `+0.479` figure.

What *is* version-independent is the statistical-power arithmetic, which
depends only on the sample length. That section has been corrected: the CI
formula used a hardcoded per-period Sharpe of 0.5, which asserts an annualised
Sharpe of about 9.6. See "Correction to the CI formula" below.

## The short version

The v3c lockbox of 8,833 bars is 184 daily observations. On the recorded
`+0.479` that gives a 95% interval of about [-2.3, +3.2]; the weekly block
bootstrap gave [-1.738, +2.901]. Every interval contains zero. On the
current-tree value of `+0.199` it is wider still. Under this contract a
candidate cannot be shown to differ from a random strategy, and no amount of
additional searching changes that.

## Measurements

### 1. The lockbox is too short

8,833 bars = 184 daily observations. The units below are deliberately
different quantities and are not comparable with each other: a point estimate
is an annualised daily Sharpe, the bootstrap interval is a percentile interval
on the same annualised Sharpe, and `t` is a standardised mean of *daily* returns.

    point Sharpe                  : +0.479   (annualised, daily returns)
    weekly block bootstrap 95% CI : [-1.738, +2.901]   (same units as above)
    t-statistic for mean != 0     : +0.341   (1.96 required)
    P(Sharpe <= 0)                : 0.336

Measured again on the current factor set, the same lockbox gives:

    point Sharpe (current factors, constant funding) : +0.199
    t-statistic for mean != 0                        : +0.142

Serial correlation of the daily returns is approximately zero: the sum of
sample autocorrelations at lags 1..3 is -0.013 on the current factor set and
was -0.026 on the old one. So the statistical efficiency is already near the
theoretical maximum. This is not a resampling artefact, and re-running it will
not help.

### 2. Correction to the CI formula

`statistical_power` previously computed

    half_width = 1.96 * sqrt((1 + 0.5**2) / years)

The `0.5` is meant to be the per-period (non-annualised) Sharpe, per Lo (2002).
Hardcoding it asserts a per-period Sharpe of 0.5, i.e. an **annualised** Sharpe
of `0.5 * sqrt(365.25) ~= 9.6`. For any realistic observed Sharpe the term is
around `3e-4`, so the old formula overstated the variance by about 25%, which
is about 12% on the reported half-width, and it did so uniformly, so it never
flipped a verdict. The corrected form is

    half_width = 1.96 * sqrt((1 + 0.5 * (SR_annualised / sqrt(365.25))**2) / years)

and when no Sharpe is supplied the caller is told the kurtosis term is
unmeasured rather than being handed a plausible-looking number. The function now
also reports `observed_annualised_sharpe` and `observed_per_period_sharpe` so a
reader can check which Sharpe went in.

### 3. The required sample does not exist

95% CI half-width on the annualised Sharpe estimate, corrected formula,
evaluated at the recorded `+0.479`:

| OOS length | bars | days | half-width |
|---|---|---|---|
| 184 days (v3c lockbox) | 8,832 | 184 | +/-2.76 |
| 300 days (what remains after a 1y burn-in) | 14,400 | 300 | +/-2.16 |
| 1.2 years | 21,024 | 438 | +/-1.79 |
| 2.03 years (the entire contract) | 35,568 | 741 | +/-1.38 |
| 4.8 years | 84,144 | 1,753 | +/-0.89 |
| 19.2 years | 336,576 | 7,012 | +/-0.45 |

The whole contract is 2.03 years. To resolve an annualised Sharpe to the
precision the gate's own threshold needs:

| target half-width | years required | multiple of the 2.03y contract |
|---|---|---|
| +/-1.00 | 3.84 | 2x |
| +/-0.50 | 15.37 | 8x |
| +/-0.25 | 61.48 | 30x |

So even the entire 2.03-year contract, used in full and never touched by
selection, resolves the Sharpe only to about +/-1.38. Resolving 0.5 from zero
needs 15.4 years.

### 4. Diversification cannot close the gap

Recomputed on the current factor set over the same 8,833-bar lockbox. Each
figure is the mean of the 378 off-diagonal pairwise correlations:

    cross-sectional pairwise correlation (bar returns)  : +0.572  -> N_eff 1.70
    same-formula position correlation                  : +0.369  -> N_eff 2.56
    strategy leg-net-return correlation                : +0.392  -> N_eff 2.42

`N_eff = n / (1 + (n-1) * rho)` with n = 28.

Note the direction: the one formula's positions are **less** correlated across
coins (+0.369) than the raw bar returns are (+0.572). The original version of
this document claimed the opposite, that the positions were "more synchronised
than the prices", while quoting numbers in its own table that said otherwise
(+0.498 against +0.639). The measured claim is the second one, and it is the
weaker one, so the conclusion below is unchanged.

Even assuming a physically impossible correlation of exactly zero:

| mean pairwise rho | N_eff | implied t |
|---|---|---|
| 0.572 (observed, bar returns) | 1.70 | +0.185 |
| 0.200 | 4.38 | +0.310 |
| 0.000 (unreachable) | 28.0 | +0.744 |

The theoretical ceiling is `t = 0.185 * sqrt(28) = 0.744`. The requirement is
1.96. The `t` here is the current-tree value (+0.142 observed, +0.185
correlation-adjusted); on the old factor set the same ceiling was +0.744 from
an observed +0.341, and the ratio is the same either way: perfect
diversification across 28 names buys a factor of `sqrt(28) ~= 5.3`, and the gap
to significance is a factor of 8.8.

### 5. Leverage is mathematically irrelevant here

The t-statistic is invariant to a common scale factor:

    x1 leverage: sharpe +0.289  t +0.205
    x2 leverage: sharpe +0.289  t +0.205
    x4 leverage: sharpe +0.289  t +0.205

Raising notional size scales the mean and the standard deviation together.
It changes neither Sharpe nor significance.

## What this rules out

| Direction | Ceiling | Verdict |
|---|---|---|
| Extend the OOS window | 15.4 years needed for +/-0.50 | data does not exist |
| Diversify further | t -> 0.744 at best | still far below 1.96 |
| Increase leverage | no effect | scale-invariant |

All three routes terminate at the same place: the only remaining lever is a
higher per-bet edge, which means a genuinely better formula, not a longer or
wider sample. And the null-control work in
`docs/selection_bias_null_calibration_28c.md` shows that under this framework a
"better formula" is currently obtained more reliably from structureless data
than from real data, so even that lever is not yet trustworthy.

## Why the v3c result is not a bug in the accounting

An earlier run reported a large drawdown and few positive legs on a
pre-contract period. Two follow-ups pinned the cause:

- The universe is not responsible. Removing POL/KAS/RENDER and re-running v3c
  on the contractual lockbox reproduces +0.477, against +0.478 for 28 coins.
- The period is responsible. On 2023-09..2024-09 the same 25 coins give
  Sharpe -0.006 and MDD 0.289, with position activity unchanged at 0.193.

So the 0.0968 lockbox MDD is not evidence of risk control. It is what a
0.5-year window looks like when it happens to be favourable, which is exactly
what a +/-2.76 interval predicts.

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
