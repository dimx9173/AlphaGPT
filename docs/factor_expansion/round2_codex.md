**Cross-Audit, Shortlist, And Correlation Budget**

## **Cross-Audit Of Other List**

Every name below is from the other model's 50. `A:X` = their candidate, `B:Y` = my pool, `E:Z` = one of the 12 existing.

- `A:TREND_10_60` duplicate of `E:PUMP_DEV`. `PUMP_DEV` is `(C-MA20)/MA20` robust-z'd; `TREND_10_60` is `(MA10-MA60)/MA60` robust-z'd. Same numerator class (price minus its own moving average, scaled by price). Also duplicate of `B:EMA_SPREAD_144` — both are MA/EMA-spread trend locators, SMA vs EMA only.
- `A:MOM_5_RAW` duplicate of `E:RET`. `RET` is trailing robust-z of log return over 200; `sum(r[-5:])` is the raw 5-bar slice of the same return series. Same primitive, standardization removed.
- `A:MOM_24_RAW`, `A:MOM_96_RAW` duplicate of `E:RET` by the same argument, different slice length. The three `MOM_*_RAW` are also nested sums of each other — `MOM_96` contains `MOM_24` contains `MOM_5`.
- `A:EMA_SPREAD` duplicate of `B:EMA_SPREAD_144`. `(EMA12-EMA48)/EMA48` vs EMA-spread with longer slow leg. Same indicator, parameter tweak. Also overlaps `E:PUMP_DEV` per above.
- `A:EFFICIENCY` duplicate of `B:EFFICIENCY_12`. Kaufman ER over 25 vs 12 bars. Identical formula, window differs by 2x. Keep one.
- `A:REV_SHORT` near-degenerate sign flip of `A:MOM_5_RAW`. `-sum(r[-6:])` vs `+sum(r[-5:])`. Same 3-hour return sum with opposite sign and one bar difference. Ship at most one of the two, and neither if `E:RET` stays.
- `A:BB_PCTB_20` duplicate of `B:BB_POS_20`. Both are `(C-LBB20)/(UBB20-LBB20)`. Identical. Keep one.
- `A:RSI2` duplicate of `B:RSI2_2` and of `E:DELTA_RSI`. `DELTA_RSI` is `(RSI14-50)/50` robust-z'd; `RSI2` is `RSI(2)` raw. Same RSI primitive, shorter lookback, no z-score. The A-vs-B pair is the same indicator twice.
- `A:ATR_RATIO` duplicate of `B:ATR_RATIO_100`. `ATR14/ATR60` vs ATR-ratio with 100-bar denominator. Same ratio, parameter tweak.
- `A:PARKINSON_20` exact duplicate of `B:PARKINSON_20`. Same name, same 20-bar Parkinson formula. One must go.
- `A:RANGE_MEAN_20` duplicate of `E:HL_RANGE`. `HL_RANGE` is `(H-L)/C` robust-z'd single-bar; `RANGE_MEAN_20` is its 20-bar mean robust-z'd. Smoothed copy of an existing factor.
- `A:UPPER_WICK`, `A:LOWER_WICK` exact duplicates of `B:UPPER_WICK`, `B:LOWER_WICK`. Same `(H-max(O,C))/(H-L)` and `(min(O,C)-L)/(H-L)` formulas. Keep one copy of each leg maximum.
- `A:WICK_IMBAL` duplicate of `B:SHADOW_BAL_12` and near-degenerate vs the wick legs. `UPPER-LOWER` single-bar vs 12-bar shadow balance. Also algebraically redundant with the `UPPER_WICK` plus `LOWER_WICK` pair jointly.
- `A:BODY_FRAC` duplicate of `B:BODY_RATIO`. `|C-O|/(H-L)` both. Identical.
- `A:DRIFT_OPEN` duplicate of `B:GAP_1`. `(O_t-C_{t-1})/C_{t-1}` both. Identical.
- `A:UP_DOWN_VOL` duplicate of `B:UP_DOWN_VOL_14`. Green-volume share over 20 vs 14 bars. Same formula, window tweak.
- `A:VOL_CV_20` duplicate of `B:VOL_OF_VOL_30`. `stdev(V)/mean(V)` over 20 vs 30. Same statistic, window tweak.
- `A:AMIHUD_20` exact duplicate of `B:AMIHUD_20`, and duplicate of `E:LIQ_SCORE`. `LIQ_SCORE` is single-bar `log(|ret|/dollarvol)` sign-flipped; `AMIHUD_20` is its 20-bar mean without the flip. Persistent version of an existing factor by construction.
- `A:CLOSE_LOC_5` duplicate of `E:CLOSE_POS`. `(C-LL5)/(HH5-LL5)` vs `(C-L)/(H-L)` raw. 5-bar channel position vs 1-bar channel position, same 0-1 geometry.
- `A:DONCH_POS_48` duplicate of `B:DONCH_POS_100`. Same Donchian `(C-LL)/(HH-LL)`, 48 vs 100 lookback. Keep one.
- `A:AUTOCORR_5_20` duplicate of `B:AUTOCORR_20`. Lag-5 autocorr over 48 vs generic autocorr over 20. Same persistence primitive.
- `A:SKEW_48` duplicate of `B:SKEW_50`. Sample skew of `r` over 48 vs 50. Correlation will be ~0.99. Keep one.
- `A:KURT_48` duplicate of `B:KURT_50`. Same, kurtosis over 48 vs 50. Keep one.
- `A:MAXDD_96` duplicate of `B:DD_200`. `(C-max(C,96))/max(C,96)` vs drawdown over 200. Same drawdown depth, window tweak.
- `A:FUND_RATE` duplicate of `B:FUND_LAST`. Last forward-filled settlement rate both. Identical. Keep one name.
- `A:FUND_Z`, `A:FUND_MOM` duplicate class of `B:FUND_SURPRISE`. Z-score vs level vs change-vs-7d-ago of the same step-function funding series. Level, standardized level, and first difference of one input.
- `A:DIST_MA60_ATR` near-duplicate of `E:PUMP_DEV`. `(C-MA60)/ATR14` vs `(C-MA20)/MA20`. Distance-from-own-MA overextension gauge; scaling (ATR vs percent) and lookback differ, numerator logic identical.
- `A:HH_DIST_ATR` / `A:LL_DIST_ATR` near-duplicate pair of `A:DONCH_POS_48` and `B:DONCH_POS_100` / `B:BREAKOUT_DIST_50`. Distance-to-20-bar-high in ATR units is a near-monotone transform of position-in-48-bar-range when ATR is stable. Do not ship all three legs.
- `A:VOL_RATIO`, `A:VOL_CHANGE` mechanically coupled to `E:VOL_CLUST`. Both contain the same 10-bar `stdev(r)` that `VOL_CLUST` levels in their numerator. The ratio/acceleration framing is real, but raw Pearson will be inflated by shared numerator.
- `A:RELVOL_10_60`, `A:VOL_SHOCK` overlap `E:LOG_VOL` / `E:VOL_TREND`. Mean-volume ratio and `V/max(V,60)` vs z-scored level and one-bar change. Same attention-spike trade, smoother window.
- Not computable from OHLCV plus sparse funding: none outright. The list is clean on this — no order-book, open-interest, or liquidation factors. Closest watch items are `A:HOD_SIN`, `A:HOD_COS`, `A:WEEKEND`, which read the bar timestamp, not OHLCV. Computable only if timestamps are UTC and stationary; if the CSV clock is venue-local or drifts, the seasonality is a clock error, not a factor.
- Not strictly causal as written: none structurally. Two need grid fixes to stay causal in code: `A:FUND_COUNTDOWN` as `(bars since last settlement)/16` assumes exactly 16 bars per 8h and breaks on missing bars — must be `(t - t_last_settlement)/8h` from timestamps. `A:POST_FUND_DRIFT` as written averages "the 2 bars after each of last 10 settlements" — only causal if the window closes at the last *completed* settlement strictly before `t`.
- Near-degenerate scaled copies / sign flips / monotone transforms: `A:REV_SHORT` vs `A:MOM_5_RAW` (sign flip), `A:WICK_IMBAL` vs the two wick legs (literal difference), `A:FUND_Z` vs `A:FUND_RATE` (z-score vs level of one series), `A:MOM_5/24/96_RAW` vs each other (nested sums), `A:BB_PCTB_20` vs `A:CLOSE_LOC_5` vs `A:DONCH_POS_48` vs `E:CLOSE_POS` (all position-in-interval, windows 1/5/20/48).

## **Shortlist Of 30**

Survivors picked from the union for orthogonality. Duplicated pairs collapsed to one entry with the better specification.

- `PARKINSON_20` | `sqrt(mean((ln(H/L))^2/(4 ln2),20))`, robust-z 200 | volatility | Efficient OHLC variance the close-only `VOL_CLUST` discards.
- `GARMAN_KLASS_20` | GK OHLC variance over 20 (`0.5 ln(H/L)^2-(2ln2-1) ln(C/O)^2`), robust-z 200 | volatility | Adds open-close drift path Parkinson ignores; keep one of the two estimators.
- `ATR_RATIO_100` | `ATR14/(ATR100+eps)`, robust-z 200 | volatility | Range-based regime-shift ratio, not vol level.
- `VOL_OF_VOL_30` | `stdev(V,30)/(mean(V,30)+eps)`, raw | volume | Lumpiness of participation; whale-block vs retail flow.
- `RANGE_MEAN_20` | `mean((H-L)/C,20)`, robust-z 200 | range-candle | Persistent wide-market regime; single-bar `HL_RANGE` sees spikes only.
- `RANGE_EXPAND` | `((H-L)/C)_t/(max((H-L)/C prior 20)+eps)`, raw | range-candle | Ignition bar that triggers stop cascades.
- `UPPER_WICK` | `(H-max(O,C))/(H-L+eps)`, raw | range-candle | Failed auction overhead supply.
- `LOWER_WICK` | `(min(O,C)-L)/(H-L+eps)`, raw | range-candle | Bid-defense floor absorption.
- `BODY_RATIO` | `|C-O|/(H-L+eps)`, raw | range-candle | Conviction vs indecision inside the bar.
- `GAP_1` | `(O_t-C_{t-1})/(C_{t-1}+eps)`, raw | range-candle | Between-bar repricing where funding/liquidation flow concentrates.
- `EFFICIENCY_12` | `|C_t-C_{t-12}|/(sum|C diffs,12|+eps)`, raw | trend | Straight-line travel vs path length; regime switch the operator set cannot see.
- `CHOP_14` | `sum(TR,14)/(HH14-LL14+eps)`, raw | trend | Choppiness index; complements efficiency with range normalization.
- `TREND_R2` | signed `R^2` of `ln(C)` on time over 48, raw | trend | Separates carried drift from noisy drift with identical net return.
- `SLOPE_TSTAT_24` | OLS slope t-stat of `ln(C)` on time over 24, raw | trend | Significance-scaled drift; kills big-but-noisy slopes.
- `DONCH_POS_100` | `(C-LL100)/(HH100-LL100+eps)`, raw | structure | 50-hour breakout location where systematic stop clusters live.
- `BREAKOUT_DIST_50` | `(C-HH50)/(ATR14+eps)`, raw | structure | Signed distance to 25-hour resistance in risk units.
- `TIME_SINCE_HIGH_100` | bars since `HH100` / 100, raw | structure | Age of the high; fresh highs vs stale compression.
- `VWAP_DEV_24` | `(C-VWAP24)/(ATR14+eps)`, `VWAP=sum(HLC3*V)/sum(V)` over 24, raw | impact-liquidity | Deviation from where volume actually traded; VWAP-reversion pressure.
- `ROLL_SPREAD` | `2*sqrt(max(-cov(r,C diff? use autocov of close changes,20),0))`, scaled by `C`, raw | impact-liquidity | Implied effective spread from negative autocovariance; Roll's transaction-cost proxy from bars alone.
- `AMIHUD_20` | `log(mean(|r|/(C*V+eps),20))`, robust-z 200 | impact-liquidity | Persistent Kyle's-lambda cost premium over 10 hours.
- `DOLLAR_IMBAL_12` | `sum(sign(C-O)*C*V,12)/sum(C*V,12)`, raw | volume | Signed dollar participation; who paid, not how much printed.
- `OBV_SLOPE_20` | OLS slope of trailing OBV over 20, scaled by mean volume, raw | volume | Cumulative signed-volume drift stripped of single-bar spikes.
- `UP_DOWN_VOL_14` | `sum(V|C>O,14)/(sum(V,14)+eps)`, raw | volume | Fraction of 7-hour trade on green bars; buying-pressure share.
- `SKEW_50` | sample skew of `r` over 50, raw | tail-risk | Crash/jump asymmetry; skew premium the 12 never measure.
- `JUMP_COUNT_50` | count of `|r|>4*MAD(r,50)` over 50, raw | tail-risk | Realized discontinuity count; fragile positioning footprint.
- `DD_200` | `(C-max(C,200))/(max(C,200)+eps)`, raw | tail-risk | 100-hour capitulation depth; exhaustion gauge.
- `HURST_PROXY_50` | `log(R50/S50)/log(50)` with `R` = range of cumulative demeaned `r`, `S` = stdev, raw | persistence | Mean-reversion vs trending exponent at 25 hours.
- `FUND_LAST` | last settlement funding rate forward-filled per coin, raw | funding | Crowded-long tax itself; only non-price series available.
- `FUND_SURPRISE` | `FUND_LAST - median(FUND_LAST,21 settlements)`, raw | funding | Abnormal crowding vs coin's own month; level lies, surprise bites.
- `FUND_X_MOM` | `FUND_LAST * sign(sum(r,24))`, raw | funding | Reflexive pile-on detector; crowding times direction is the squeeze configuration.

## **Proposed 24 And 6 Drops**

Ship: `PARKINSON_20`, `ATR_RATIO_100`, `VOL_OF_VOL_30`, `RANGE_MEAN_20`, `RANGE_EXPAND`, `UPPER_WICK`, `LOWER_WICK`, `BODY_RATIO`, `GAP_1`, `EFFICIENCY_12`, `CHOP_14`, `TREND_R2`, `SLOPE_TSTAT_24`, `DONCH_POS_100`, `BREAKOUT_DIST_50`, `TIME_SINCE_HIGH_100`, `VWAP_DEV_24`, `ROLL_SPREAD`, `AMIHUD_20`, `DOLLAR_IMBAL_12`, `OBV_SLOPE_20`, `SKEW_50`, `DD_200`, `FUND_SURPRISE`.

Drop from the 30:

- `GARMAN_KLASS_20` — same OHLC-variance estimator as `PARKINSON_20` with a drift term; pairwise |r| will exceed 0.9. Keep Parkinson as the cleaner standard.
- `JUMP_COUNT_50` — tail-count built from the same `|r|/MAD` exceedances that drive `SKEW_50` tails; redundant tail vote, weaker argument than skew itself.
- `UP_DOWN_VOL_14` — near-linear in `DOLLAR_IMBAL_12` (green-share vs signed-share of the same volume); dollar-signed version carries the better cost-flow argument.
- `HURST_PROXY_50` — R/S on 50 bars of 30m crypto returns is dominated by estimation noise and overlaps `EFFICIENCY_12` plus `CHOP_14` as a third regime dial. Two regime dials suffice.
- `FUND_LAST` — level and surprise of one step function; keep `FUND_SURPRISE` (abnormality) because raw level is dominated by cross-coin venue-regime offsets pooled across 28 coins.
- `FUND_X_MOM` — interaction term of funding with 12-hour momentum; clever but a product of two shipped signals the RPN operator set can rebuild itself (`MUL` exists). Spend the slot on a primitive, not an interaction.

## **Correlation Budget Reasoning**

Expected to exceed |r| 0.6 if measured pooled at bar grain on a common mask:

- `PARKINSON_20` vs `E:VOL_CLUST` (0.6-0.85, two realized-vol levels), vs dropped `GARMAN_KLASS_20` (>0.9).
- `RANGE_MEAN_20` vs `E:HL_RANGE` (0.5-0.7; mean vs instance of same primitive).
- `BODY_RATIO` vs `UPPER_WICK` / `LOWER_WICK` (mechanically negative; small body forces large wicks by identity `body+shadows=1`).
- `DONCH_POS_100` vs `BREAKOUT_DIST_50` (0.6-0.8 when ATR stable; position-in-range vs distance-to-edge).
- `EFFICIENCY_12` vs `CHOP_14` (strongly negative; straight travel vs chop of the same path).
- `AMIHUD_20` vs `E:LIQ_SCORE` (strongly negative by sign flip; 20-bar mean vs single-bar of same Amihud quantity).
- `SLOPE_TSTAT_24` vs `TREND_R2` (same 24-48-bar OLS fit, slope significance vs signed fit quality).
- Any funding level vs its own z-score/difference (`FUND_LAST` vs `FUND_SURPRISE`, ~0.5-0.7 at bar grain, higher at settlement grain).
- Mutually exclusive families where at most one should survive: OHLC variance estimators (`PARKINSON_20` vs `GARMAN_KLASS_20` vs `ROGERS_SATCHELL_20` vs `E:VOL_CLUST` — keep one new plus existing); wick decomposition (`UPPER_WICK` vs `LOWER_WICK` vs `WICK_IMBAL`/`SHADOW_BAL` — keep at most two, preferably one leg plus imbalance); Donchian geometry (`DONCH_POS` vs `HH_DIST`/`LL_DIST` vs `BREAKOUT_DIST` — keep at most two with different windows); raw momentum sums (`MOM_5` vs `MOM_24` vs `MOM_96` vs `E:RET` — keep at most one new horizon); funding transforms (level vs z vs surprise vs momentum-interaction — keep at most two); BB/Donchian/close position (all position-in-interval — keep at most one new alongside `E:CLOSE_POS`).

## **What I Would Do Differently**

- Most likely disguised duplicate: `A:AMIHUD_20`. It is the 20-bar mean of exactly the `|ret|/dollar-volume` quantity inside `E:LIQ_SCORE`, with only the sign flip removed. The economic text ("persistent premium vs single-bar") sounds like a new family, but mechanically it is `LIQ_SCORE` smoothed — same numerator, same denominator, same dollars. If round 2 keeps both and pooled |r| comes back under 0.6, distrust the measurement before celebrating novelty: the funding-style step smoothing plus robust-z restandardization attenuates the very correlation the smoothing creates. Ship it only against `LIQ_SCORE` head-to-head, never alongside.