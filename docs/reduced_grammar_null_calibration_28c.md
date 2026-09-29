# Reduced grammar, and what it did not fix

Date: 2026-09-28
Subject: 28-coin / 30m / 2.03y contract, real Binance funding
(`equity-compound-v3-real-funding`), reward v1, seeds 101-110

## What was tried

The full grammar is 29 tokens: 12 factors and 17 operators, admitting about
3.1e15 syntactically valid 12-token programs. Ten independent searches over it
returned ten *disjoint* formulas, so the search is not converging on anything.
It is sampling, and it samples Sharpe out of noise.

The obvious response is to make the hypothesis class smaller. A reduced grammar
keeps the eleven operators that carry an economic argument and drops the seven
that only reshape a cross-section or pattern-match one series:

| kept | why |
|---|---|
| ADD SUB MUL DIV | arithmetic combination of two factors |
| NEG SIGN | direction inversion, direction extraction |
| GATE | regime-conditional logic |
| DECAY | exponential smoothing, i.e. an explicit mean-reversion |
| DELAY1 | one-bar causal lag |
| DELTA | first difference, i.e. explicit change |
| CORR | cross-factor rolling correlation |

| dropped | why |
|---|---|
| ABS | magnitude without direction is ambiguous |
| JUMP, MAX3 | pattern-match a single series |
| ZSCORE, RANK, TS_RANK | reshape a cross-section that is already normalised |

Measured exactly, by dynamic programming over reachable stack states:

    full     : 29 tokens, ~3.1e15 valid programs   = 2^51.5
    reduced  : 23 tokens,  1.185e13 valid programs = 2^43.4
    reduction: 262x, i.e. 8.0 fewer bits

An artifact records `grammar`, because reduced token 20 is DELAY1 while full
token 20 is JUMP, and a reduced formula evaluated by the full evaluator
computes a different function and reports it honestly. The full-grammar code
path is left byte-identical to before; `random_formula`, `mutate`, `crossover`
dispatch to a parallel reduced implementation rather than sharing one.

## The result

Matched to the earlier batches in every respect except the grammar: same seeds
(101-110), same 100 generations, same population 32, same reward v1, same real
funding, same iid null construction with null-seed 0.

| protocol | real median | null median | gap | real > 0 | null > 0 | Mann-Whitney p |
|---|---|---|---|---|---|---|
| standard, full grammar | +0.782 | +1.812 | -1.030 | 7/10 | 8/10 | 0.131 |
| standard, full grammar, reward v2 | +0.782 | +1.407 | -0.625 | 7/10 | 8/10 | - |
| walk-forward, full grammar | +1.242 | +1.676 | -0.434 | 7/10 | 8/10 | 0.257 |
| **standard, reduced grammar** | **-0.317** | **+0.822** | **-1.139** | **4/10** | **7/10** | **0.227** |

Per seed, reduced grammar:

| seed | real OOS | null OOS | real breadth | null breadth | gate |
|---|---|---|---|---|---|
| 101 | -0.288 | +2.193 | 13/28 | 16/28 | fail / fail |
| 102 | +0.537 | +2.335 | 16/28 | 14/28 | fail / fail |
| 103 | +1.105 | +0.595 | 14/28 | 15/28 | fail / fail |
| 104 | -0.347 | -1.182 | 10/28 | 10/28 | fail / fail |
| 105 | -0.466 | +1.554 | 13/28 | 13/28 | fail / fail |
| 106 | -2.173 | -1.162 | 5/28 | 12/28 | fail / fail |
| 107 | +0.923 | +0.480 | 17/28 | 13/28 | fail / fail |
| 108 | -1.332 | +1.048 | 8/28 | 16/28 | fail / fail |
| 109 | +0.641 | +1.481 | 16/28 | 16/28 | fail / fail |
| 110 | -1.012 | -2.270 | 6/28 | 9/28 | fail / fail |

**Shrinking the hypothesis class 262x did not close the gap. It widened the
median gap from -1.030 to -1.139 and the null still outscores the real data.**

The mechanism is therefore not "the search had too many formulas to choose
from". A 262x reduction removes 8 bits of freedom and the null's median Sharpe
barely moved, from +1.812 to +0.822. Something other than search space is
producing the null's Sharpe.

## What the null actually preserves, and why that is the suspect

The iid null block-bootstraps each coin's returns and volume **per coin, in
isolation**, and then recomputes features on the resampled series. Verified on
the construction:

| property | real | null |
|---|---|---|
| excess kurtosis | 236 | 173 |
| vol-clustering AC(1) | -0.043 | -0.036 |
| p99/median return ratio | 8.5 | 8.5 |
| cross-coin return correlation | +0.669 | **+0.004** |

So the null keeps fat tails, keeps volatility clustering, and keeps the
per-coin marginal return distribution almost exactly. It removes only the
cross-coin co-movement and any genuine predictive relationship.

A factor that predicts a real, persistent per-coin effect will do worse on the
null, because that effect is destroyed. A factor that does **not** predict
anything, but merely keys off volatility clustering and fat tails, will do about
as well on the null as on the real data, because both are present in both.

The search reports +0.822 median on data where no predictive relationship
exists. Under the null, whatever the search is keying off is a property of
*noise* that the bootstrap faithfully preserves. Volatility clustering and fat
tails are exactly the properties that let a "signal" line up with future
returns without any causal mechanism. That is a more specific hypothesis than
"the search space is too big", and it points at the factors and the position
construction rather than at the GA.

## The power problem, stated honestly

With n = 10 per arm, the reduced-grammar comparison has **24% power** at the
observed effect size (Cohen's d = -0.56). Reaching 80% power at that d needs
about 50 runs per arm. So the p = 0.227 result means *this experiment could not
distinguish real from null*, not *real and null are identical*.

There is a second and larger limitation. The ten null runs in a batch share
`--null-seed 0`: they are ten searches over **one** null dataset, so they
measure search randomness, not null-dataset randomness. A rank test over those
ten treats them as ten observations and therefore understates the real
uncertainty. Pooling all four protocols gives a stratified permutation
p = 0.0017, but that number is not trustworthy either, because the same
dependence inflates it.

Both limits are reported rather than papered over. A between-dataset null
variance set is included in `research/compare_grammar_real_null_28c.py` to
separate the two spreads; see the tool's output for the ratio.

## What still holds

- **The gate is the only working defence.** 0 of 20 reduced-grammar runs passed
  (0/10 real, 0/10 null), and 0 of 20 across the earlier full-grammar batches.
  It has never once passed anything this framework produced.
- **`live_adopted` remains `False`** everywhere. No artifact from this work
  authorises trading.
- **v3c's `+0.479` is not reproducible** from the current tree. Its formula
  contains `LIQ_SCORE`, which `de2e353` replaced. Re-measured with today's
  factors: **+0.199** on the constant-funding basis it was produced under, and
  **-0.897** on real funding. See `docs/statistical_power_ceiling_28c.md`.
- **The 2.03-year contract cannot resolve the question.** Corrected half-width
  is +/-1.38 on the entire contract used in full. Resolving a Sharpe of 0.5
  from zero needs 15.4 years.

## What to do next, and what not to do

Do **not** shrink the grammar further. It is not the binding constraint, and a
262x reduction already failed to move the null.

The next discriminating experiment is to vary the null, because the null's
construction is now the leading suspect:

1. **Break the volatility clustering in the null.** Resample at the bar level
   rather than in blocks. If the null's Sharpe collapses toward zero, the
   search was keying off autocorrelation, and the honest conclusion is that
   the factor set and position construction are not measuring a causal
   relationship at all.
2. **Break the fat tails in the null.** Rank-transform or winsorise the
   resampled returns. Same question, different mechanism.
3. If the null stays positive under both, the position construction itself
   (`tanh` of a median/MAD-normalised factor, smoothed, then rolled) is
   generating signal from scale alone, and that is a bug to find rather than a
   property to exploit.

Each of these is one run of the existing null machinery with a different
`null_mode`. None of them needs a new search protocol, and all three are
cheaper than another GA batch.
