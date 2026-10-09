# Round 1 candidate pool: 50 new bars-only factors

Notation used in every definition below. All windows are trailing-only, bar `t` uses bars `t-N+1..t` (or `t-N..t-1` where stated), never `t+1`. `ret_t = log(C_t/C_{t-1})`. `SMA_N`, `EMA_N`, `STD_N`, `median_N`, `MAD_N` are trailing. `robust_z_N(x) = (x_t - median_N(x)) / (1.4826*MAD_N + 1e-9)`. `z_N` is the standard-score version. `HH_N/LL_N` = trailing highest-high / lowest-low. `ATR_14` = trailing Wilder mean of `H-L` (no peeking). `DVOL_t = C_t*V_t`. `F_t` = last known funding settlement rate strictly before bar `t` (forward-filled, never interpolated from future settlements). `h(t)` = hour of day UTC, `d(t)` = day of week.

| name | bars-only definition | economic argument | family |
|---|---|---|---|
| EMA_SPREAD_144 | `(EMA_48(C)_t - EMA_144(C)_t)/EMA_144(C)_t`, then `robust_z_200` | Medium trend vs long anchor: prices underreaction and slow capital movement; different horizons than PUMP_DEV's MA20 | TREND |
| SLOPE_TSTAT_24 | OLS slope of `log(C)` over trailing 24 bars divided by residual std; raw value | Trend strength per unit noise, not just distance: distinguishes drift from noisy excursion | TREND |
| DONCH_POS_100 | `(C_t - LL_100,t)/(HH_100,t - LL_100,t + 1e-9)`, raw in [0,1] | Position in 100-bar trading range: breakout vs range-bound supply overhang; longer memory than any existing factor | BREAKOUT |
| BREAKOUT_DIST_50 | `(C_t - HH_50,t-1)/ATR_14,t`, floored at 0 for non-breakouts; else raw signed distance | Fresh-high supply absorption: stops and FOMO entry cluster at known highs; pays only when it prices | BREAKOUT |
| TIME_SINCE_HIGH_100 | `(t - argmax(H over t-99..t))/100`, raw in [0,1] | Age of the high prices anchoring bias; old highs are forgotten resistance, fresh highs are live supply | BREAKOUT |
| BB_POS_20 | `(C_t - SMA_20(C)_t)/(2*STD_20(C)_t + 1e-9)`, raw | Stretch relative to own recent noise band: overextension that dealer/market-maker inventory must fade | MEANREV |
| DIST_MED_100 | `(C_t - median_100(C))/(1.4826*MAD_100(C) + 1e-9)`, raw | Long-horizon anchor deviation: value-trader entry scales with distance from accepted price, MA20-independent | MEANREV |
| RSI2_2 | 2-bar RSI on `ret` (`avg gain2/(avg gain2+avg loss2)`), raw in [0,1] | Connors short-horizon snapback: 2-period RSI prices 1-3 bar inventory exhaustion, RSI14 prices weeks | MEANREV |
| ROUNDTRIP_12 | `(C_t - C_{t-12})/C_{t-12} - sum_{i=0}^{11} ret_{t-i} + drift_adj` simplified: `log(C_t/C_{t-12}) - 12*median_200(ret)`, `robust_z_200` | Drift-adjusted 12-bar excursion: mean-reversion signal net of the bull drift the book keeps tracking | MEANREV |
| PARKINSON_20 | `sqrt(mean_{20}(log(H/L)^2)/(4*log2))`, then `robust_z_200` | Range-based vol using H/L path, not close-to-close: informed-trade intensity shows first in intrabar excursion | VOLREG |
| GARMAN_KLASS_20 | `mean_{20}(0.5*log(H/L)^2 - (2*log2-1)*log(C/O)^2)`, sqrt, then `robust_z_200` | Uses O/C as well as H/L: separates drift (O-C) from noise (H-L); efficient vol when trend pollutes close vol | VOLREG |
| ROGERS_SATCHELL_20 | `mean_{20}(log(H/C)*log(H/O) + log(L/C)*log(L/O))`, sqrt, then `robust_z_200` | Drift-robust vol estimator: prices volatility regime without confusing the +1.42 drift for risk | VOLREG |
| ATR_RATIO_100 | `ATR_14,t / ATR_100,t`, then `robust_z_200` | Short vs long trading-cost regime: spread/impact expansion predicts reversal payoff and stop density | VOLREG |
| VOL_OF_VOL_30 | `STD_30(\|ret\|)/ (mean_30(\|ret\|)+1e-9)`, raw | Instability of volatility itself: vol-of-vol spikes mark deleveraging cascades and forced-book regime change | VOLREG |
| RANGE_TREND_RATIO | `mean_20(\|C-O\|/(H-L+1e-9))`, raw in [0,1] | Directional vs noisy bars: trending microstructure (large bodies) vs indecision (large shadows) regime flag | VOLREG |
| BODY_RATIO | `(C_t - O_t)/(H_t - L_t + 1e-9)`, then `SMA_5` smoothed, raw in [-1,1] | Signed conviction per unit excursion: close-near-high means buyers paid through the range, not just drifted | MICRO |
| UPPER_WICK | `(H_t - max(C_t,O_t))/(H_t - L_t + 1e-9)`, `SMA_12`, raw in [0,1] | Failed rally inventory: sellers absorbed the push; predicts short-horizon fade when persistent | MICRO |
| LOWER_WICK | `(min(C_t,O_t) - L_t)/(H_t - L_t + 1e-9)`, `SMA_12`, raw in [0,1] | Failed selloff absorption: bids held the dip; mirror-image dip-buying footprint | MICRO |
| SHADOW_BAL_12 | `SMA_12((upper_wick - lower_wick))`, raw in [-1,1] | Persistent one-sided rejection: multi-bar footprint of a defending book (support vs distribution) | MICRO |
| EFFICIENCY_12 | `\|C_t - C_{t-12}\|/(sum_{i=0}^{11}\|C_{t-i}-C_{t-i-1}\| + 1e-9)`, raw in [0,1] (Kaufman) | Path efficiency: trending vs choppy tape; chops charge costs to trend-followers, trends pay them | EFFICIENCY |
| CHOP_14 | `100*log10(sum_{14}\|H-L\|/(HH_14 - LL_14 + 1e-9))/log10(14)`, raw | Choppiness index: sideways chop vs explosive range expansion; regime gate for when momentum can be paid | EFFICIENCY |
| HURST_PROXY_50 | `log2(R_50/S_50)/log2(50)` where `R_50 = max(C_{50})-min(C_{50})`, `S_50 = STD of ret over 50 * sqrt(50)`-style rescaled range, raw | Mean-reversion vs trending memory: sub-0.5 = anti-persistent tape where fades are paid; super-0.5 = momentum tape | EFFICIENCY |
| VARIANCE_RATIO_8 | `Var(sum of 8-bar returns over trailing 48) / (8*Var(1-bar ret over 48) + 1e-9)`, raw | Lo-MacKinlay efficiency violation: VR<1 = microstructure snapback, VR>1 = underreaction drift; prices the inefficiency directly | EFFICIENCY |
| AUTOCORR_20 | Pearson corr of `ret_{t-19..t}` with `ret_{t-20..t-1}`, raw in [-1,1] | First-order tape memory sign: continuation vs bid-ask bounce regime; switches the sign of optimal response | EFFICIENCY |
| ROLL_SPREAD | `2*sqrt(max(0,-cov(ret, lag1(ret)) over trailing 50))`, raw | Roll implied effective spread from return bounce alone: trade-cost regime from bars only, no book needed | TURNOVER |
| VWAP_DEV_24 | `(C_t - VWAP_24,t)/VWAP_24,t` with `VWAP_24 = sum(typical*V)/sum(V)` over 24, `typical=(H+L+C)/3`; then `robust_z_200` | Distance from where volume actually traded: institutional benchmark pain; stretch from VWAP invites reversion | VOLPRICE |
| OBV_SLOPE_20 | `OBV_t = cumsum(sign(C-O)*V)`; factor = `(OBV_t - SMA_20(OBV)_t)/(STD_20(OBV)+1e-9)`, raw | Cumulative signed-volume conviction: separates price moves with vs without volume sponsorship | VOLPRICE |
| VPT_DEV_50 | `VPT_t = cumsum(ret_t * V_t)`; factor = `robust_z_50(VPT change)` i.e. robust z of `VPT_t - VPT_{t-1}` | Return-weighted accumulation: large-size informed flow leaves VPT footprints before price completes | VOLPRICE |
| DOLLAR_IMBAL_12 | `sum_{12}(sign(C-O)*DVOL)/ (sum_{12}(DVOL)+1e-9)`, raw in [-1,1] | Signed dollar majority over 12 bars: who paid more, buyers or sellers; order-flow proxy without trade data | VOLPRICE |
| VOL_RANK_100 | rank of `V_t` within trailing 100 volumes / 100, raw in [0,1] | Nonparametric volume surprise: robust to volume distribution shifts that break z-scores; attention/liquidation flag | VOLPRICE |
| VOL_RANGE_RATIO | `robust_z_200(log((V_t+1)/(H_t-L_t+1e-9)))`, i.e. log volume per unit range, robust z | Volume absorbed per unit price movement: high ratio = deep book absorbing flow; low ratio = thin tape, easy push | VOLPRICE |
| UP_DOWN_VOL_14 | `sum(V on up bars over 14)/(sum(V over 14)+1e-9)`, raw in [0,1] | Granville-style sponsorship: rallies on expanding volume vs on air; only the former is inventory-backed | VOLPRICE |
| SKEW_50 | skew of `ret` over trailing 50, raw | Asymmetric tail pricing: lottery-demand coins (right skew) vs crash-risk coins (left skew) carry different premia | TAIL |
| KURT_50 | excess kurtosis of `ret` over trailing 50, raw | Fat-tail regime: jump-prone tape where stop-hunting and liquidation cascades concentrate | TAIL |
| MAXRET_20 | `max_{20}(ret)/ (STD_50(ret)+1e-9)`, raw | Largest recent upside jump in vol units: FOMO-chasing inflow marker and blowoff-top warning | TAIL |
| MINRET_20 | `min_{20}(ret)/ (STD_50(ret)+1e-9)`, raw (negative) | Largest recent down-jump in vol units: forced-sell washout marker; capitulation is the cheapest entry | TAIL |
| JUMP_COUNT_50 | count of `\|ret\| > 4*1.4826*MAD_50(ret)` over trailing 50, integer 0..50 | Jump frequency, not size: repeated circuit-breaker tape signals unstable book vs one-off news | TAIL |
| DOWNSIDE_VOL_RATIO | `sqrt(mean_{20}(min(ret,0)^2)) / (STD_50(ret)+1e-9)`, raw | Downside-vol share: crash-risk pricing; coins with concentrated downside need a premium to hold | TAIL |
| DD_200 | `(C_t - max_{200}(C))/(max_{200}(C)+1e-9)`, raw in [-1,0] | Depth of drawdown from trailing peak: disposition-effect overhang; deeper DD = heavier trapped-seller supply | DRAWDOWN |
| DD_DURATION | `(t - argmax(C over t-199..t))/200`, raw in [0,1] | Time underwater: long drawdowns exhaust weak hands and reset the holder base; capitulation clock | DRAWDOWN |
| RECOVERY_20 | `(C_t - min_{20}(C))/(max_{200}(C) - min_{20}(C) + 1e-9)`, raw | Bounce position inside the drawdown well: early-recovery momentum vs still-buried; entry-timing within DD | DRAWDOWN |
| GAP_1 | `(O_t - C_{t-1})/(C_{t-1}+1e-9)`, then `robust_z_200` | Inter-bar repricing on no tape: 30-min gap is the crypto analog of overnight news; gaps fill or extend | GAP |
| GAP_FILL_6 | `(C_t - O_t)/((O_t - C_{t-1})+eps)` clipped to [-2,2], `SMA_6`; measures fill fraction | Gap-fill vs gap-go tendency of this coin: market-microstructure personality that persists per venue | GAP |
| AMIHUD_20 | `mean_{20}(\|ret_t\|/(DVOL_t+1))`, then `robust_z_200` | Averaged price-impact cost: Kyle-lambda regime; high-impact tape mean-reverts as liquidity providers demand premium | TURNOVER |
| DOLLARVOL_Z | `robust_z_200(log(DVOL_t+1))` | Dollar-volume attention regime: distinct from share-volume LOG_VOL when prices differ 100x across coins | TURNOVER |
| TURN_PERSIST_50 | lag-1 autocorr of `log(V+1)` over trailing 50, raw | Volume clustering persistence: sustained attention (fund-driven) vs one-bar wash (noise) | TURNOVER |
| FUND_LAST | `F_t` (last settled 8h rate, ffill), raw | Carry itself: positive funding = longs pay shorts; contrarian short-funding-harvest is the only exogenous P&L stream in the data | FUNDING |
| FUND_SURPRISE | `F_t - median of prior 21 settlements (≈7d)`, raw | Funding shock vs what holders budgeted: surprise hikes trigger long capitulation, cuts trigger short-cover | FUNDING |
| FUND_X_MOM | `sign(log(C_t/C_{t-48})) * F_t`, raw | Crowdedness-signed carry: high funding + up روند = crowded long (fade); high funding + down = squeeze fuel (chase) | FUNDING |
| TOD_SIN / TOD_COS | `sin(2*pi*hour_utc/24)`, `cos(2*pi*hour_utc/24)`, raw (two factors) | Time-of-day liquidity cycle: Asia/EU/US handover moves spreads and volatility predictably; known ex ante, no peeking | SEASONAL |

That is 51 rows counting `TOD_SIN`/`TOD_COS` as two. Intentional overshoot so round 2 can cut.

## 1. Families ranked most promising

- **1. FUNDING.** Ranked first because it is the only family fed by data orthogonal to OHLCV. Funding is a real transfer, a real incentive (longs pay shorts), and the only series in the building that is not a deterministic re-encoding of price and volume. Everything else refines measurement; funding adds information.
- **2. VOLPRICE.** `VWAP_DEV_24`, `DOLLAR_IMBAL_12`, `OBV_SLOPE_20`, `UP_DOWN_VOL_14` ask whether a move had volume sponsorship. Existing volume factors (`LOG_VOL`, `VOL_TREND`, `FOMO`) measure how much traded, never whether buyers or sellers paid. Highest information-per-correlation odds.
- **3. EFFICIENCY.** `VARIANCE_RATIO_8`, `EFFICIENCY_12`, `AUTOCORR_20`, `HURST_PROXY_50` price the inefficiency itself rather than a pattern. The reduced operator set wants economic arguments; "tape is anti-persistent, so fade" is an argument. Also structurally uncorrelated with level factors.
- **4. TAIL.** `SKEW_50`, `KURT_50`, `MINRET_20`, `JUMP_COUNT_50` price crash/lottery premia, a genuinely different risk than center-of-distribution z-scores. Crypto perps live in the tails; none of the 12 look there.
- **5. VOLREG (range-based).** `PARKINSON_20`, `GARMAN_KLASS_20`, `ATR_RATIO_100` extract vol from H/L path instead of close-to-close `VOL_CLUST`. Same phenomenon, different lens — but the H/L lens sees intrabar informed flow first.
- **6. DRAWDOWN.** `DD_200`, `DD_DURATION`, `RECOVERY_20` encode disposition-effect supply overhang. Nothing in the 12 knows distance from the peak; `PUMP_DEV` knows distance from MA20, which forgets in 10 hours.
- **7. BREAKOUT.** `DONCH_POS_100`, `BREAKOUT_DIST_50`, `TIME_SINCE_HIGH_100` carry the supply/demand argument (stops cluster at known highs) over 50-100 bars, an order of magnitude longer memory than existing factors.
- **8. TURNOVER (cost).** `ROLL_SPREAD`, `AMIHUD_20` are the only candidates that estimate the cost of trading itself from bars alone. If costs explain the +1.33-vs-+1.42 gap, cost-regime factors are the gate, not the alpha.
- **9. MICRO.** `BODY_RATIO`, `UPPER_WICK`, `SHADOW_BAL_12` are cheap and causal, but individually weak; kept as a family because 12-bar smoothed wicks may survive correlation cuts as a group.
- **10. SEASONAL + GAP.** Ranked last among the promising because the edge (if any) is small and periodic, but nearly guaranteed uncorrelated — time-of-day is orthogonal to everything by construction. Cheap diversification for the 36-dim set.

`MEANREV` variants (`BB_POS_20`, `RSI2_2`) are deliberately ranked out: they are the most likely to work and the most likely to be redundant (see below).

## 2. Suspected duplicates, named explicitly

- `DOLLARVOL_Z` vs existing `LOG_VOL`: same transform, dollar vs share volume. Expect |r| > 0.9 on most coins. Keep at most one; round 2 should test and drop the loser.
- `AMIHUD_20` vs existing `LIQ_SCORE`: `LIQ_SCORE` is single-bar `log(|ret|/dollarvol)`; `AMIHUD_20` is its 20-bar mean. Expect high correlation by construction. The averaged version is the keeper (less noise), the single-bar one is the legacy.
- `BB_POS_20` vs existing `PUMP_DEV`: both are `(C - MA20)/scale`. Different scale (STD vs robust z) but same numerator. Expect |r| > 0.7. Do not keep both.
- `RSI2_2` vs existing `DELTA_RSI`: same RSI machinery, 2-period vs 14-period. Short-horizon RSI2 is much less correlated than it looks, but flag the pair and check.
- `ROUNDTRIP_12` vs existing `RET`: `RET` is 1-bar robust-z return; `ROUNDTRIP_12` sums 12 of them net of drift. Partially correlated by construction; keep only if correlation comes in low.
- `DIST_MED_100` vs existing `PUMP_DEV`: same idea (distance from center), 100-bar median vs 20-bar mean. Longer memory should decorrelate, but they are conceptual siblings — verify.
- `VOL_RANGE_RATIO` vs existing `LOG_VOL` and `HL_RANGE`: it is literally the ratio of the two. Will correlate with both; keep only if it adds interaction information beyond the pair.
- `BODY_RATIO` (smoothed) vs existing `PRESSURE = tanh(3*(C-O)/(H-L))`: nearly the same single-bar quantity before the tanh. The 5-bar SMA version may decorrelate enough to survive, but the raw pair is a duplicate — do not keep raw `BODY_RATIO` alongside `PRESSURE`.
- `UP_DOWN_VOL_14` vs existing `CLOSE_POS`: both reward closes near highs; one weights by volume. Expect moderate correlation (~0.4-0.6). Probably survivable, but flagged.
- `MAXRET_20`/`MINRET_20` vs existing `VOL_CLUST`: big jumps inflate trailing vol mechanically, so expect 0.3-0.5 correlation. Conceptual overlap only; likely survivable.
- `TOD_SIN` vs `TOD_COS`: orthogonal by construction (sine vs cosine), but they are a matched pair encoding one circular variable — round 2 should treat them as one 2-dim group, not two independent draws.
- `FUND_LAST` vs `FUND_SURPRISE`: surprise contains level; expect ~0.5 correlation. Keep both only if the budget allows; `FUND_X_MOM` interacts either with momentum so it is the least redundant of the three.
- `PARKINSON_20` vs `GARMAN_KLASS_20` vs `ROGERS_SATCHELL_20` vs existing `VOL_CLUST`: all four estimate the same latent volatility. Pairwise |r| among the three range estimators will likely exceed 0.8. Round 2 should keep exactly one range estimator plus `VOL_CLUST`, not all three.

## 3. Traps in the data constraints

- **Funding forward-fill becomes a step function.** `F_t` updates every 16 bars and is flat between. Any windowed stat (SMA/STD/z-score) over raw `F_t` mostly measures where the bar sits inside the 8-hour step, not economics. Use settlement-indexed windows (prior N settlements), never bar-indexed windows, and expect near-zero variance inside some 200-bar estimation windows — the constant-feature guard will bite.
- **Zero-volume / zero-range bars divide by zero.** Illiquid coins and halted stretches print `H=L` or `V=0`. Every ratio with `(H-L)`, `DVOL`, or `V` in the denominator needs its `+1e-9`/`+1` guard and a defined fallback (0, not NaN), or the factor silently becomes NaN/constant on exactly the coins where liquidity matters most.
- **`log(V+1)` vs `log(DVOL+1)` vs `log1p(V)` inconsistency.** The existing `LOG_VOL` uses share volume; dollar volume differs by the coin price level, which varies 1000x across the 28 coins. Cross-coin pooling will mistake price level for liquidity unless each factor is standardized per coin (it is, via robust z) — but raw family members (`VOL_RANK_100`, funding levels) are not, so check cross-sectional comparability before trusting pooled correlations.
- **Warmup NaNs masquerade as signal.** 100-200 bar windows plus 200-bar robust-z standardization need ~400 bars of warmup (~8 days). The first 1-2% of each coin's series will be NaN or edge-artifact; correlation measured with pairwise deletion overweights the later regime. Require a common valid mask across all 12+50 factors before computing the round-2 correlation matrix.
- **Time-of-day factors are identical across coins.** `TOD_SIN/COS` and weekend flags have zero cross-sectional variance at any timestamp. A pooled correlation looks fine, but a per-bar cross-sectional rank of these is a constant — any cross-sectional gate or rank-based operator silently ignores them. They contribute only time-series variance.
- **Crypto has no close, so GAP_1 is thin.** `(O_t - C_{t-1})` on 30-min crypto bars is usually one tick of jitter, not an overnight gap. The economic argument (news while market shut) does not transfer; what remains is a bid-ask bounce proxy. Keep the factor but relabel expectations: it belongs to microstructure, and its variance will be tiny — the `1e-9` constant guard is most likely to fire here.
- **Survivorship and stablecoin contamination.** If any of the 28 perps are stablecoin-margined outliers or delisted/relisted series with gaps, range/vol factors collapse to constants on flat stretches while trend factors explode on repeg jumps. Check per-coin `ptp` before pooled correlation; one dead coin can fake a duplicate (all factors constant together) or fake independence (NaNs pairwise-deleted differently).