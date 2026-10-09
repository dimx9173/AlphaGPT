# Factor set expansion: shared context for the discussion

## The situation

AlphaGPT searches over RPN formulas built from 12 factors and 11 operators on
28 crypto perps at 30-minute bars over 3 years. Three consecutive experiments
have now eliminated every explanation except the factor set itself:

| experiment | varied | outcome |
|---|---|---|
| variable-length A/B | search space (12 tokens vs 6..20) | factor subsets reachable went 61.3% -> 99.7% of 2^12; no edge |
| effort ladder | generations 2/5/10/20 | no dose-response on real or null, slope t = -0.43 |
| reward vs gate | v3 reward vs pure excess reward | steering worked (position 0.218 -> 0.073, t -3.13), no holdout gain |

Across all of it, **6 of 48 real-data runs beat buy-and-hold**. Absolute
Sharpe reaches +1.33 while buy-and-hold on the same bars is +1.42. The book
tracks the drift and pays costs to do it.

The search space, the search effort, and the reward's agreement with the gate
are no longer the binding constraint. The factors are.

## The 12 existing factors

From research/causal_12f.py. All are strictly causal and trailing-only.

| # | name | definition |
|---|---|---|
| 0 | RET | log return, trailing median/MAD robust z-score over 200 |
| 1 | LIQ_SCORE | Amihud illiquidity = log(|ret| / dollar volume), sign-flipped so higher = more liquid |
| 2 | PRESSURE | tanh(3 * (close-open)/(high-low)) |
| 3 | FOMO | second difference of volume change |
| 4 | PUMP_DEV | (close - MA20) / MA20, robust z |
| 5 | LOG_VOL | log1p(volume), robust z |
| 6 | VOL_CLUST | sqrt(mean of squared returns over 10), robust z |
| 7 | MOM_REV | 1 if 5-bar momentum and its predecessor have opposite signs |
| 8 | DELTA_RSI | (RSI14 - 50) / 50, robust z |
| 9 | HL_RANGE | (high-low)/close, robust z |
| 10 | CLOSE_POS | (close-low)/(high-low), raw in [0,1] |
| 11 | VOL_TREND | (volume - prev_volume)/prev_volume, robust z |

## Data actually available

The 28c CSVs carry exactly five columns per bar, per coin:

    timestamp, open, high, low, close, volume

Real Binance funding history is loaded separately and IS available as a
per-coin rate series (research/accounting_28c.load_real_funding). It is sparse:
settlement events only, every 8 hours.

**Not available and not recoverable:** order book depth, trade-level data,
open interest, liquidations, on-chain flows, cross-asset series, any
point-in-time fundamentals. A factor needing any of these is a silent
placeholder, and the project has already been bitten by that: LIQ_SCORE used
to read 'liquidity' and 'fdv', fell back to fixed constants, and was a constant
0.4 on every bar of every coin while still passing every well-formedness check.

## Constraints a proposal must satisfy

1. **Strictly causal.** Trailing windows only, no peeking. There is a causality
   guard test (tests/test_causality_guard.py) that must keep passing.
2. **Only the data above.** Otherwise it is a placeholder.
3. **Actually varies.** A factor whose ptp or std is below 1e-9 is treated as
   constant and refuses to be searched (constant_features / assert_features_vary).
4. **An economic argument.** The reduced operator set was introduced precisely to
   cut hypothesis classes that carry no economic argument. Factors need the same
   standard: a reason to expect it prices something, not a reason to expect it
   measures something.
5. **Low correlation** with the 12 existing factors and with each other.

## The ask

Propose **at least 24 new factors**, bringing the set to 36 dimensions, such
that after measuring them the new factors have low absolute Pearson correlation
with the existing 12 and with each other.

This is ROUND 1 of 3. In round 1 the job is coverage, not selection: propose
the full candidate pool, generously, with an economic argument for each and the
exact bars-only formula. Be concrete. Do not worry yet about the correlation
budget, that is round 2's job.

## Deliverable for round 1

A table, one row per candidate:

    name | bars-only definition | economic argument | family

Then a short list of which families you think are most promising and which
candidates you already suspect are duplicates of an existing factor.
