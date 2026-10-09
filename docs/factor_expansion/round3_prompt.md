# ROUND 3 — resolve the contested set and produce an implementable spec

You have seen the context (12 existing factors, data available, constraints) and
the other model's round-2 reasoning.

## Where the two round-2 proposals stand

**Agreed by both models (12).** Both put these in the final 24. They are the
core of the expansion:

    ATR_RATIO_100, BREAKOUT_DIST_50, DD_200, DOLLAR_IMBAL_12, DONCH_POS_100, EFFICIENCY_12, OBV_SLOPE_20, PARKINSON_20, ROLL_SPREAD, SKEW_50, SLOPE_TSTAT_24, VWAP_DEV_24

**Contested (24), A proposes 12:**

    AMIHUD_20, BODY_RATIO, CHOP_14, FUND_SURPRISE, GAP_1, LOWER_WICK, RANGE_EXPAND, RANGE_MEAN_20, TIME_SINCE_HIGH_100, TREND_R2, UPPER_WICK, VOL_OF_VOL_30

**Contested (24), B proposes 12:**

    AUTOCORR_20, DD_DURATION, FUND_FLIP_7, FUND_LAST, FUND_X_MOM, KURT_50, MINRET_20, SHADOW_BAL_12, TOD_COS, TOD_SIN, VARIANCE_RATIO_8, VOL_RANK_100

Total contested: 24. We need exactly 12 of them.

## Your task

### 1. Adjudicate the 24 contested candidates

For each of the 24, say which of the three buckets it belongs in:

  KEEP    - it should be in the final 24
  CUT     - it should be dropped, and the one-line reason
  FOLD    - it is redundant with an already-agreed factor; say which, and
            whether the agreed one should be swapped out instead

Be decisive. The purpose of this round is to end, not to hedge. If you think
the agreement of the 12 core is itself wrong, say so explicitly and say which
core member should go.

### 2. The final 24

State the final 24 as a single numbered list. If your adjudication above
replaced any core member, that shows up here.

### 3. Implementable specification

This is the part that matters most, because it gets turned into code exactly as
written. For EACH of the final 24 give:

  name | exact bars-only formula | window lengths | output range or transform

Rules the formulas must obey, because the code will be checked against them:

- Only O, H, L, C, V and the timestamp of the current bar. Funding only if
  the factor is a funding factor, and funding is sparse 8-hourly: use the last
  settlement with timestamp <= t and never a bar-indexed window over it.
- Every window includes bar t and only past bars. No centred windows, no
  full-sample mean or std, no ewm with adjust=True, no rank over the whole
  series.
- Write the denominator guard explicitly. Where a denominator can be zero, say
  what the factor returns there.
- Say whether the factor is raw or trailing-robust-z normalised over 200 bars,
  and if raw, say why the range is bounded and comparable across coins.

Ambiguity here costs a round: if two readings of your formula give different
numbers, the wrong one gets implemented.

### 4. Expected correlation structure of the final 24

Which pairs do you expect to exceed |r| 0.6? This will be measured, and pairs
you flagged will be checked against what the measurement actually says.

## Output

Sections 1-4 as above. Tables and prose. No code execution, no file edits.
