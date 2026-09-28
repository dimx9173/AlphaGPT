# Deflated Sharpe: the real batches collapse, and the null batches expose a limit in the adjustment

Date: 2026-09-28
Commit: d6b9010
Artifacts: results/ga_iter_runs, results/ga_null_runs, results/ga_null_v2_runs, results/wf_real, results/wf_null

## Method

DSR (Bailey & Lopez de Prado) prices two things a raw Sharpe ignores:

1. **Selection.** Reporting the best of N searches is selecting on the outcome.
   The bar is E[max SR] under the null, which grows with both N and the
   variance of the Sharpe across trials.
2. **Non-normality.** This universe runs excess kurtosis above 200, so the
   usual Sharpe standard error is wrong.

Inputs: cross-run Sharpe variance from the batch itself, N = 10, n_obs = 2000
daily observations, non-excess kurtosis = 200 (conservative tail value rather
than the point estimate, so the adjustment is not tuned to be favourable).

## Result

| Batch | Best reported | Selection bar | Deflated | P(deflated > 0) |
|-------|---------------|---------------|----------|-----------------|
| real, standard | +1.079 | +1.334 | **-0.255** | 0.000 |
| real, walk-forward | +1.368 | +2.318 | **-0.951** | 0.000 |
| null, standard v1 | +4.881 | +3.132 | +1.749 | 1.000 |
| null, standard v2 | +3.127 | +2.147 | +0.980 | 1.000 |
| null, walk-forward | +4.575 | +3.333 | +1.242 | 1.000 |

Both real batches deflate to negative. The best number each protocol produced
from real data is below what picking the best of ten would have produced by
chance. Walk-forward makes this worse, not better: its selection bar is higher
because fold-to-fold variance is larger, and the real holdout does not clear it.

## The null batches, and a limit in this adjustment

The null batches deflate *positive*, which is not a result about the null
data. It is a result about the adjustment.

DSR estimates trial dispersion from the sample variance of the observed
Sharpes and then assumes those trials are Gaussian. Here they are not: the
null's own Sharpe distribution is heavy-tailed, because a search over ~35k
bars produces occasional extreme fits even on structureless data. Estimating
the tail of a heavy-tailed quantity from ten draws understates its spread, so
the bar is set too low and the null clears a bar that was mis-specified.

The consequence is practical: **DSR is not usable here as a formal gate.**
It is usable as a description of the real batches, and that description is
consistent with the null control, but it cannot certify a null on its own. A
test that passes on fabricated data is not a test.

## What stands

The defensible conclusions, in order of strength:

1. Real and null are statistically indistinguishable (Mann-Whitney p = 0.13
   standard, p = 0.26 walk-forward). This is the primary result and it does not
   depend on DSR at all.

2. The best-of-N reported from real data does not exceed what selection alone
   produces. This is descriptive and consistent with (1).

3. No measurement available in this setup can certify an edge, including the
   one just added. That is now demonstrated rather than assumed.

## Next

The remaining lever is search freedom. The grammar admits 12 tokens from 28
operators against ~35k bars of a 28-coin universe whose effective sample size
is around 1.5 independent bets. A search of that size against data of that
size will find something, which is what the null demonstrates.

The honest options are to reduce the search space until the multiple-testing
problem is genuinely tractable, or to accept that this factor set does not
support a detectable edge at this sample size.
