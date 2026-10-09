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

---

# The other model's round-2 reasoning

# ROUND 2 — selection

## 1. Cross-audit of the other model's (B/Codex) list

B's list has 51 factor-slots (TOD_SIN/TOD_COS counted as two). I audited all 51 against the 12 existing factors, against A's list, and against the data/causality constraints.

### 1a. Duplicates of one of the 12 existing factors

| B candidate | Existing factor it duplicates | Why it is a duplicate |
|---|---|---|
| BB_POS_20 | PUMP_DEV (#4) | PUMP_DEV is `(C-MA20)/MA20` robust-z'd. BB_POS_20 is `(C-SMA20)/(2*STD20)`. Identical numerator `(C-MA20)`, only the scale differs. Same signal, different denominator. |
| BODY_RATIO | PRESSURE (#2) | PRESSURE is `tanh(3*(C-O)/(H-L))`. BODY_RATIO is `(C-O)/(H-L)` then SMA_5. Same single-bar quantity before a monotone tanh. Smoothing does not create new information. |
| DOLLARVOL_Z | LOG_VOL (#5) | LOG_VOL is `robust_z(log1p(V))`. DOLLARVOL_Z is `robust_z(log(DVOL))` with `log(DVOL)=log(V)+log(C)`. Within a coin `log(C)` is a slow drift that the robust-z absorbs, leaving a near-affine copy. Expect \|r\|>0.9. |
| AMIHUD_20 | LIQ_SCORE (#1) | LIQ_SCORE is single-bar `log(\|ret\|/dvol)` sign-flipped. AMIHUD_20 is its 20-bar mean (unflipped). Correlated by construction; the only difference is averaging and sign. One of the two must go, and the existing one cannot be dropped. |
| RSI2_2 | DELTA_RSI (#8) | Same RSI machinery, 2-period vs 14-period. Not an exact duplicate — the horizons genuinely differ — but it is the closest pair in the set. Flag and verify; keep only if measured \|r\| is low. |
| ROUNDTRIP_12 | RET (#0) | RET is 1-bar robust-z return. ROUNDTRIP_12 sums ~12 of them net of drift. Partially correlated by construction. Survivable only if measured low. |
| DIST_MED_100 | PUMP_DEV (#4) | Same idea (distance from center), 100-bar median vs 20-bar mean. Conceptual siblings. Longer memory should decorrelate, but verify. |
| DONCH_POS_100 | CLOSE_POS (#10) | CLOSE_POS is 1-bar `(C-L)/(H-L)`. DONCH_POS_100 is the same position concept over a 100-bar range. Different memory, same geometry. Expect moderate r, probably survivable. |
| VOL_RANGE_RATIO | LOG_VOL (#5) + HL_RANGE (#9) | It is literally the ratio of the two (`log(V/range)` robust-z'd). Will correlate with both parents. Keep only if it adds interaction information beyond the pair. |
| PARKINSON_20 / GARMAN_KLASS_20 / ROGERS_SATCHELL_20 | VOL_CLUST (#6) | All four estimate the same latent volatility — close-to-close vs three H/L-path variants. Expect 0.4–0.6 vs VOL_CLUST and >0.8 among the three range estimators themselves. |
| UP_DOWN_VOL_14 | CLOSE_POS (#10) | Both reward closes near highs; one weights by volume. Expect ~0.4–0.6. Probably survivable, flagged. |
| MAXRET_20 / MINRET_20 | VOL_CLUST (#6) | Big jumps mechanically inflate trailing vol. Expect 0.3–0.5. Conceptual overlap only; likely survivable. |

Verdict: four hard duplicates that must not ship alongside the existing factor — **BB_POS_20, BODY_RATIO, DOLLARVOL_Z, AMIHUD_20**. The rest in this table are flagged pairs to verify in measurement.

### 1b. Duplicates of a candidate in A's list (both names)

| B candidate | A candidate | Relationship |
|---|---|---|
| PARKINSON_20 | PARKINSON_20 | Exact coincidence (same name, same estimator). Ship once. |
| UPPER_WICK | UPPER_WICK | Exact coincidence. Ship once. |
| LOWER_WICK | LOWER_WICK | Exact coincidence. Ship once. |
| AMIHUD_20 | AMIHUD_20 | Exact coincidence — and per 1a, also a duplicate of existing LIQ_SCORE. Double reason to cut. |
| EFFICIENCY_12 | EFFICIENCY | Same Kaufman efficiency. Ship once. |
| UP_DOWN_VOL_14 | UP_DOWN_VOL | Same Granville-style factor, window pinned at 14 vs unpinned. Ship once. |
| SKEW_50 | SKEW_48 | Same skew, window 50 vs 48 — a trivially different window. Ship one. |
| KURT_50 | KURT_48 | Same kurtosis, window 50 vs 48. Ship one. |
| ATR_RATIO_100 | ATR_RATIO | Same short/long trading-cost ratio, window pinned vs unpinned. Ship once. |
| AUTOCORR_20 | AUTOCORR_5_20 | Same first-order tape-memory idea, different lag/window. Ship one. |
| SHADOW_BAL_12 | WICK_IMBAL | Both are smoothed (upper-wick-minus-lower-wick). Same footprint of a defending book. Ship one. |
| RANGE_TREND_RATIO | BODY_FRAC | `mean(\|C-O\|/(H-L))` vs body fraction — same single-bar quantity, absolute vs signed. Ship at most one. |
| TOD_SIN / TOD_COS | HOD_SIN / HOD_COS | Same circular time-of-day encoding under a different name. One pair, not two. |
| FUND_LAST | FUND_RATE | Both are the raw last-settled funding level `F_t`. Same series. Ship once. |
| FUND_SURPRISE | FUND_Z | Both are funding level net of its recent norm (median of prior ~21 settlements vs robust-z). Same shock concept. Ship at most one. |
| BB_POS_20 | BB_PCTB_20 | Both are Bollinger-position — and both duplicate PUMP_DEV per 1a. Ship neither. |
| RSI2_2 | RSI2 | Same 2-period snapback idea. Ship once (and per 1a, verify vs DELTA_RSI). |
| DD_200 | MAXDD_96 | Same drawdown-depth concept, window 200 vs 96. Ship one. |
| DONCH_POS_100 | DONCH_POS_48 | Same Donchian-position, window 100 vs 48. Ship one. |
| EMA_SPREAD_144 | EMA_SPREAD | Same medium-vs-long trend spread, window pinned vs unpinned. Ship at most one — and both are PUMP_DEV's long-memory cousins, so verify vs #4. |
| SLOPE_TSTAT_24 | TREND_10_60 / TREND_R2 | Same trend-strength family (slope vs distance/R²). Ship at most one trend-per-noise representative. |

### 1c. Not computable from OHLCV + sparse funding

**None.** All 51 are computable. The funding trio (FUND_LAST, FUND_SURPRISE, FUND_X_MOM) correctly uses strictly-before forward-fill `F_t` with settlement-indexed windows, so the sparsity is handled rather than violated. No candidate needs book, trades, OI, liquidations, on-chain, or cross-asset data. (A's list had the same clean record; neither side proposed a placeholder.)

### 1d. Not strictly causal as written

**None.** Every window is trailing-only or same-bar-at-close (`DONCH_POS_100` including `H_t/L_t` alongside `C_t` is causal — the whole bar is known at its close). Funding uses settlements strictly before bar `t`. No `t+1` leakage anywhere. The causality guard should pass on all 51.

Two non-causality defects that *are* present, classified honestly — they are degeneracy hazards, not peeking:

- GAP_FILL_6 divides by `(O_t − C_{t−1})`, which is one tick of jitter on 30-min crypto bars and frequently ~0. Needs the clipped/SMA form as written, and even then expect tiny variance — the constant-feature guard is most likely to fire here.
- FUND_LAST/FUND_SURPRISE are step functions flat for 16 bars between settlements. Any *bar-indexed* windowed stat over them measures position-inside-the-step, not economics. Both candidates as written use settlement-indexed or raw form, so they survive — but any future variant that puts an SMA/STD over raw `F_t` will not.

### 1e. Near-degenerate (scaled copy, sign flip, monotone transform)

- UPPER_WICK, LOWER_WICK, PARKINSON_20, AMIHUD_20: exact cross-list copies (§1b, rows 1–4).
- SKEW_50/SKEW_48, KURT_50/KURT_48: window 50 vs 48 is a distinction without a difference.
- BODY_RATIO vs PRESSURE: tanh is monotone; SMA_5 of the pre-image is not a new factor.
- BB_POS_20 vs PUMP_DEV (and vs BB_PCTB_20): same numerator, rescaled.
- DOLLARVOL_Z vs LOG_VOL: near-affine copy per §1a.
- FUND_LAST vs FUND_RATE; TOD pair vs HOD pair: renames, not new factors.
- PARKINSON_20 vs GARMAN_KLASS_20 vs ROGERS_SATCHELL_20: three estimators of one latent; pairwise \|r\| will likely exceed 0.8. Shipping more than one is spending dimensions on the estimator, not the phenomenon.

---

## 2. Shortlist of 30 (from the union, plus one justified addition)

Conventions: `ret_t=log(C_t/C_{t−1})`, trailing windows use bars `t−N+1..t`, `robust_z_N` over trailing N, `F_t` = last settlement strictly before `t`. One factor per duplicated concept (winner named in §1b).

| # | name | exact bars-only formula | family | economic argument (one line) |
|---|---|---|---|---|
| 1 | FUND_LAST | `F_t` raw (ffill, strictly-before) | FUNDING | Carry itself: the only exogenous P&L stream in the data — longs pay shorts, so level predicts contrarian harvest. |
| 2 | FUND_SURPRISE | `F_t − median(prior 21 settlements)` raw | FUNDING | Funding shock vs what holders budgeted: surprise hikes trigger long capitulation, cuts trigger short-cover. |
| 3 | FUND_X_MOM | `sign(log(C_t/C_{t−48}))·F_t` raw | FUNDING | Crowdedness-signed carry: high funding + up = crowded long to fade; high funding + down = squeeze fuel. |
| 4 | FUND_FLIP_7 ★added | count of sign changes of `F` over prior 7 settlements, integer 0..6 | FUNDING | Neither list has funding *sign dynamics*: choppy flips mark exhausted consensus where the carry trade keeps getting stopped. |
| 5 | VWAP_DEV_24 | `(C_t−VWAP_24)/VWAP_24`, `VWAP_24=Σ(typical·V)/ΣV`, `typical=(H+L+C)/3`, then `robust_z_200` | VOLPRICE | Distance from where volume actually traded: institutional benchmark pain invites reversion. |
| 6 | DOLLAR_IMBAL_12 | `Σ_12(sign(C−O)·DVOL)/Σ_12(DVOL)`, raw in [−1,1] | VOLPRICE | Who paid more over 12 bars, buyers or sellers: order-flow proxy without trade data. |
| 7 | OBV_SLOPE_20 | `(OBV_t−SMA_20(OBV))/STD_20(OBV)`, `OBV=cumsum(sign(C−O)·V)` | VOLPRICE | Separates price moves with volume sponsorship from those on air. |
| 8 | UP_DOWN_VOL_14 | `Σ(V on up bars,14)/Σ(V,14)` raw | VOLPRICE | Granville sponsorship: rallies on expanding volume are inventory-backed; rallies on air are not. |
| 9 | VOL_RANK_100 | rank(`V_t` in trailing 100)/100 raw | VOLPRICE | Nonparametric volume surprise: robust to distribution shifts that break z-scores; attention/liquidation flag. |
| 10 | VARIANCE_RATIO_8 | `Var(8-bar sums over 48)/(8·Var(1-bar ret over 48))` raw | EFFICIENCY | Lo–MacKinlay violation priced directly: VR<1 = snapback tape, VR>1 = underreaction drift. |
| 11 | EFFICIENCY_12 | `\|C_t−C_{t−12}\|/(Σ_12\|ΔC\|)` raw | EFFICIENCY | Path efficiency: chops charge trend-followers costs; trends pay them — regime gate for momentum. |
| 12 | AUTOCORR_20 | Pearson(`ret_{t−19..t}`, `ret_{t−20..t−1}`) raw | EFFICIENCY | Sign of tape memory switches the sign of the optimal response: continuation vs bounce regime. |
| 13 | SLOPE_TSTAT_24 | OLS slope of `log(C)` over 24 / residual std, raw | TREND | Trend strength per unit noise, not distance: distinguishes drift from noisy excursion. |
| 14 | SKEW_50 | skew(`ret`, 50) raw | TAIL | Lottery-demand (right skew) vs crash-risk (left skew) coins carry different premia. |
| 15 | KURT_50 | excess kurtosis(`ret`, 50) raw | TAIL | Fat-tail regime marks where stop-hunting and liquidation cascades concentrate. |
| 16 | MINRET_20 | `min_20(ret)/STD_50(ret)` raw (≤0) | TAIL | Largest down-jump in vol units: forced-sell washout marker — capitulation is the cheapest entry. |
| 17 | MAXRET_20 | `max_20(ret)/STD_50(ret)` raw (≥0) | TAIL | Largest upside jump in vol units: FOMO-inflow marker and blowoff-top warning. |
| 18 | JUMP_COUNT_50 | count(`\|ret\|>4·1.4826·MAD_50`) over 50, 0..50 | TAIL | Jump frequency not size: repeated-breaker tape signals an unstable book vs one-off news. |
| 19 | PARKINSON_20 | `sqrt(mean_20(log(H/L)²)/(4·log2))`, `robust_z_200` | VOLREG | H/L-path vol sees intrabar informed flow before close-to-close does. Sole range-estimator survivor. |
| 20 | ATR_RATIO_100 | `ATR_14/ATR_100`, `robust_z_200` | VOLREG | Short-vs-long trading-cost regime: spread expansion predicts reversal payoff and stop density. |
| 21 | DD_200 | `(C_t−max_200(C))/max_200(C)` raw in [−1,0] | DRAWDOWN | Disposition-effect overhang: deeper drawdown = heavier trapped-seller supply. Nothing in the 12 knows distance from peak. |
| 22 | DD_DURATION | `(t−argmax(C,200))/200` raw | DRAWDOWN | Time underwater: long drawdowns exhaust weak hands — the capitulation clock. |
| 23 | RECOVERY_20 | `(C_t−min_20(C))/(max_200(C)−min_20(C))` raw | DRAWDOWN | Bounce position inside the well: early-recovery momentum vs still-buried entry timing. |
| 24 | DONCH_POS_100 | `(C_t−LL_100)/(HH_100−LL_100)` raw | BREAKOUT | Position in 100-bar range: breakout vs range-bound supply overhang at 10× the memory of any existing factor. |
| 25 | BREAKOUT_DIST_50 | `(C_t−HH_50,t−1)/ATR_14`, floored at 0 for non-breakouts | BREAKOUT | Fresh-high supply absorption: stops/FOMO cluster at known highs; pays only when it prices. |
| 26 | TIME_SINCE_HIGH_100 | `(t−argmax(H,100))/100` raw | BREAKOUT | Age of the anchoring high: fresh highs are live supply, old highs are forgotten resistance. |
| 27 | ROLL_SPREAD | `2·sqrt(max(0,−cov(ret,lag(ret),50)))` raw | TURNOVER | Implied effective spread from return bounce alone: the cost of trading estimated with no book. |
| 28 | SHADOW_BAL_12 | `SMA_12(upper_wick − lower_wick)` raw in [−1,1] | MICRO | Persistent one-sided rejection: multi-bar footprint of a defending book. Sole wick survivor. |
| 29 | TOD_SIN | `sin(2π·hour_utc/24)` raw | SEASONAL | Time-of-day liquidity cycle (Asia/EU/US handover) is known ex ante and orthogonal to everything by construction. |
| 30 | TOD_COS | `cos(2π·hour_utc/24)` raw | SEASONAL | Cosine half of the circular pair — ships with SIN or not at all. |

Deliberate cuts to reach 30 (all justified by §1): BB_POS_20, BB_PCTB_20, BODY_RATIO, BODY_FRAC, DOLLARVOL_Z, AMIHUD_20, GARMAN_KLASS, ROGERS_SATCHELL, RSI2/RSI2_2, DIST_MED_100, EMA_SPREADs, TREND_R2/TREND_10_60, CHOP_14, HURST_PROXY_50, ROUNDTRIP_12, VOL_RANGE_RATIO, VWAP-adjacent duplicates, VPT_DEV_50, TURN_PERSIST_50, DOWNSIDE_VOL_RATIO, GAP_1, GAP_FILL_6, WEEKEND, HOD pair (merged into TOD), raw UPPER/LOWER_WICK (folded into SHADOW_BAL_12).

★ **Addition justification — FUND_FLIP_7.** A's funding set (RATE/Z/MOM/COUNTDOWN/POST_DRIFT) and B's (LAST/SURPRISE/X_MOM) cover *level*, *shock*, *interaction with momentum*, and *post-event drift* — but neither covers *sign persistence*: how often carry flips direction. A market flipping long-pays-short to short-pays-long 5 times in 7 settlements is a different animal (exhausted, whipsawed consensus) than one pinned positive all week (stable carry), at identical average level. Formula uses settlement-indexed counting only, so it dodges the step-function trap B identified; strictly causal (prior settlements only); integer 0–6 so it trivially passes the constant-feature guard while varying only when funding actually flips.

---

## 3. Proposed 24 to ship (and the 6 dropped, with reasons)

**Ship these 24:** FUND_LAST, FUND_X_MOM, FUND_FLIP_7, VWAP_DEV_24, DOLLAR_IMBAL_12, OBV_SLOPE_20, VOL_RANK_100, VARIANCE_RATIO_8, EFFICIENCY_12, AUTOCORR_20, SLOPE_TSTAT_24, SKEW_50, KURT_50, MINRET_20, PARKINSON_20, ATR_RATIO_100, DD_200, DD_DURATION, DONCH_POS_100, BREAKOUT_DIST_50, ROLL_SPREAD, SHADOW_BAL_12, TOD_SIN, TOD_COS.

| # | dropped | why |
|---|---|---|
| 1 | FUND_SURPRISE | Cheapest funding cut: surprise contains level, so it correlates ~0.5 with FUND_LAST by construction. Keep level (the tradeable carry) + crowdedness interaction + flip dynamics; surprise is the redundant third leg. |
| 2 | MAXRET_20 | Keep the capitulation side (MINRET — forced selling is the cheaper, better-documented entry) and cut its FOMO mirror; the pair correlates through VOL_CLUST and one tail-direction suffices. |
| 3 | RECOVERY_20 | Most derived of the DD trio — defined from `max_200` and `min_20`, so it correlates with both DD_200 and DONCH_POS_100 while adding no new primitive. Depth + duration already encode the overhang. |
| 4 | TIME_SINCE_HIGH_100 | Weakest of the breakout trio: argmax-age is steppy with long flat stretches (low variance, guard-bait), and DONCH_POS_100 + BREAKOUT_DIST_50 already span range-position and fresh-high distance. |
| 5 | JUMP_COUNT_50 | Overlaps KURT_50 (both count tail weight; jump-count is kurtosis binarized) and MINRET_20 (the jumps it counts are the extremes MIN/MAX already span). Third leg of a tail tripod — cut. |
| 6 | UP_DOWN_VOL_14 | Overlaps DOLLAR_IMBAL_12 (both are signed-volume shares; dollar-imbalance weights by size traded, which is the strictly more informative version). Keep the dollar one. |

Net effect: funding 4→3, volprice 5→4, tail 5→3, drawdown 3→2, breakout 3→2; efficiency 3, volreg 2, trend 1, turnover 1, micro 1, seasonal 2 preserved.

---

## 4. Correlation-budget reasoning (before the pooled measurement)

Expected to **exceed \|r\| 0.6** (do not ship both sides — already cut in §3, listed here so the measurement can confirm the cut was correct):

- DOLLARVOL_Z – LOG_VOL (>0.9 expected). Cut.
- AMIHUD_20 – LIQ_SCORE (high by construction). Cut.
- BB_POS_20 / BB_PCTB_20 – PUMP_DEV (same numerator). Cut both.
- BODY_RATIO / BODY_FRAC – PRESSURE (monotone + smooth). Cut all raws; only smoothed SHADOW_BAL_12 ships.
- PARKINSON_20 – GARMAN_KLASS_20 – ROGERS_SATCHELL_20 (pairwise >0.8 expected). Ship PARKINSON only.
- SKEW_50 – SKEW_48; KURT_50 – KURT_48 (window-trivial). One each ships.
- UP_DOWN_VOL_14 – UP_DOWN_VOL (same, pinned window). One ships in the 30, zero in the 24 (folded into DOLLAR_IMBAL_12).
- EFFICIENCY_12 – EFFICIENCY; FUND_LAST – FUND_RATE; TOD – HOD (renames). One of each ships.

Expected **0.3–0.6 (survivable, watch-list)** — keep but verify:

- PARKINSON_20 – VOL_CLUST (~0.4–0.6: same latent, different lens; the H/L lens is the point).
- MINRET_20 – VOL_CLUST (~0.3–0.5, mechanical inflation).
- DONCH_POS_100 – CLOSE_POS, UP_DOWN_VOL – CLOSE_POS, ROUNDTRIP_12 – RET (dropped), DIST_MED_100 – PUMP_DEV (dropped): same-geometry/different-memory pairs, expected moderate.
- FUND_LAST – FUND_SURPRISE (~0.5): resolved by dropping SURPRISE.
- VOL_RANGE_RATIO – LOG_VOL/HL_RANGE (dropped — a ratio of two existing factors cannot beat the pair plus an interaction operator).

Expected **~0 by construction (free diversification)**: TOD_SIN – TOD_COS (sine vs cosine orthogonality); TOD pair vs everything pooled within-coin (time-of-day has no cross-sectional variance, only time-series variance — contributes nothing to cross-sectional rank operators, which is a limitation to remember, not a correlation problem).

**Families to treat as mutually exclusive (at most one survivor):**

1. RANGE_VOL — {PARKINSON, GARMAN_KLASS, ROGERS_SATCHELL} → keep PARKINSON_20.
2. BOLLINGER_POSITION — {BB_POS_20, BB_PCTB_20} → keep neither (PUMP_DEV owns this geometry).
3. SIGNED_BODY — {PRESSURE (existing), BODY_RATIO, BODY_FRAC} → keep PRESSURE only; ships no new raw body.
4. VOLUME_LEVEL — {LOG_VOL (existing), DOLLARVOL_Z} → keep LOG_VOL; new volume info must be nonparametric (VOL_RANK_100) or signed (DOLLAR_IMBAL_12).
5. PRICE_IMPACT — {LIQ_SCORE (existing), AMIHUD_20} → keep LIQ_SCORE for now; new cost info comes from ROLL_SPREAD instead.
6. FUND_LEVEL — {FUND_LAST/FUND_RATE, FUND_SURPRISE/FUND_Z} → keep FUND_LAST + FUND_X_MOM + FUND_FLIP_7.
7. WICK — {UPPER_WICK, LOWER_WICK, SHADOW_BAL_12/WICK_IMBAL} → keep SHADOW_BAL_12 only.
8. SKEW_WINDOWS / KURT_WINDOWS / AUTOCORR_WINDOWS / DONCH_WINDOWS / DD_DEPTH_WINDOWS → one window each (50 / 50 / 20 / 100 / 200).
9. TOD_SIN/TOD_COS → one indivisible 2-dim group: ship both or neither.

---

## 5. What I would do differently — the most likely disguised duplicate

**DOLLARVOL_Z** — from either list, the single most likely disguised duplicate of something already in the set (existing LOG_VOL, §1a row 3).

The disguise is good: "dollar volume" *sounds* like different information from "share volume," and the VOLPRICE family story (size matters) makes it feel like an economic upgrade. It is not. `log(DVOL_t) = log(V_t) + log(C_t)`, and within a single coin `log(C_t)` moves orders of magnitude slower than `log(V_t)` — it is a slow drift plus the identical fast series. Both factors then pass through the same `robust_z_200` standardization, which absorbs exactly that slow drift. What remains is the same standardized volume-surprise series twice, destined to print \|r\| > 0.9 on most of the 28 coins while consuming a dimension, passing every well-formedness check, and contributing zero orthogonal information — the LIQ_SCORE-shaped trap (constant 0.4 that passed all checks) wearing a different mask. The honest dollar-volume factors are the ones that *interact* dollars with something else (DOLLAR_IMBAL_12 signs them; VWAP_DEV_24 benchmarks against them), not the one that merely re-logs them.

---

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
