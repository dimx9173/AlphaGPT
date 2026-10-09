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

# Combined candidate pool after round 1

Claude proposed 49, Codex proposed 49, 4 names coincide: AMIHUD_20, LOWER_WICK, PARKINSON_20, UPPER_WICK.
Union is 94 candidates. Target is 24 of them.

## A's pool (Claude)

TREND_10_60, MOM_5_RAW, MOM_24_RAW, MOM_96_RAW, EMA_SPREAD, MACD_HIST, TREND_R2, EFFICIENCY, REV_SHORT, DIST_MA60_ATR, BB_PCTB_20, BB_WIDTH, RSI2, VOL_RATIO, VOL_CHANGE, ATR_RATIO, PARKINSON_20, RANGE_MEAN_20, UPPER_WICK, LOWER_WICK, WICK_IMBAL, BODY_FRAC, RANGE_EXPAND, DRIFT_OPEN, RELVOL_10_60, VOL_CV_20, VOL_PRICE_CORR, UP_DOWN_VOL, VOL_SHOCK, AMIHUD_20, ILLIQ_CHANGE, CLOSE_LOC_5, DONCH_POS_48, HH_DIST_ATR, LL_DIST_ATR, STREAK_10, AUTOCORR_5_20, SIGN_ENTROPY, SKEW_48, KURT_48, MAXDD_96, HOD_SIN, HOD_COS, FUND_COUNTDOWN, WEEKEND, FUND_RATE, FUND_Z, FUND_MOM, POST_FUND_DRIFT

## B's pool (Codex)

EMA_SPREAD_144, SLOPE_TSTAT_24, DONCH_POS_100, BREAKOUT_DIST_50, TIME_SINCE_HIGH_100, BB_POS_20, DIST_MED_100, RSI2_2, ROUNDTRIP_12, PARKINSON_20, GARMAN_KLASS_20, ROGERS_SATCHELL_20, ATR_RATIO_100, VOL_OF_VOL_30, RANGE_TREND_RATIO, BODY_RATIO, UPPER_WICK, LOWER_WICK, SHADOW_BAL_12, EFFICIENCY_12, CHOP_14, HURST_PROXY_50, VARIANCE_RATIO_8, AUTOCORR_20, ROLL_SPREAD, VWAP_DEV_24, OBV_SLOPE_20, VPT_DEV_50, DOLLAR_IMBAL_12, VOL_RANK_100, VOL_RANGE_RATIO, UP_DOWN_VOL_14, SKEW_50, KURT_50, MAXRET_20, MINRET_20, JUMP_COUNT_50, DOWNSIDE_VOL_RATIO, DD_200, DD_DURATION, RECOVERY_20, GAP_1, GAP_FILL_6, AMIHUD_20, DOLLARVOL_Z, TURN_PERSIST_50, FUND_LAST, FUND_SURPRISE, FUND_X_MOM

## Only A proposed

ATR_RATIO, AUTOCORR_5_20, BB_PCTB_20, BB_WIDTH, BODY_FRAC, CLOSE_LOC_5, DIST_MA60_ATR, DONCH_POS_48, DRIFT_OPEN, EFFICIENCY, EMA_SPREAD, FUND_COUNTDOWN, FUND_MOM, FUND_RATE, FUND_Z, HH_DIST_ATR, HOD_COS, HOD_SIN, ILLIQ_CHANGE, KURT_48, LL_DIST_ATR, MACD_HIST, MAXDD_96, MOM_24_RAW, MOM_5_RAW, MOM_96_RAW, POST_FUND_DRIFT, RANGE_EXPAND, RANGE_MEAN_20, RELVOL_10_60, REV_SHORT, RSI2, SIGN_ENTROPY, SKEW_48, STREAK_10, TREND_10_60, TREND_R2, UP_DOWN_VOL, VOL_CHANGE, VOL_CV_20, VOL_PRICE_CORR, VOL_RATIO, VOL_SHOCK, WEEKEND, WICK_IMBAL

## Only B proposed

ATR_RATIO_100, AUTOCORR_20, BB_POS_20, BODY_RATIO, BREAKOUT_DIST_50, CHOP_14, DD_200, DD_DURATION, DIST_MED_100, DOLLARVOL_Z, DOLLAR_IMBAL_12, DONCH_POS_100, DOWNSIDE_VOL_RATIO, EFFICIENCY_12, EMA_SPREAD_144, FUND_LAST, FUND_SURPRISE, FUND_X_MOM, GAP_1, GAP_FILL_6, GARMAN_KLASS_20, HURST_PROXY_50, JUMP_COUNT_50, KURT_50, MAXRET_20, MINRET_20, OBV_SLOPE_20, RANGE_TREND_RATIO, RECOVERY_20, ROGERS_SATCHELL_20, ROLL_SPREAD, ROUNDTRIP_12, RSI2_2, SHADOW_BAL_12, SKEW_50, SLOPE_TSTAT_24, TIME_SINCE_HIGH_100, TURN_PERSIST_50, UP_DOWN_VOL_14, VARIANCE_RATIO_8, VOL_OF_VOL_30, VOL_RANGE_RATIO, VOL_RANK_100, VPT_DEV_50, VWAP_DEV_24


---

# The other participant's round 1

The other model produced this independently in round 1. Read it. You are about
to be asked to reconcile with it.

# ROUND 1 — candidate factor pool (50 candidates)

Notation used in every definition: per coin, per 30m bar `t`: `O H L C V` = open/high/low/close/volume. `r_t = ln(C_t / C_{t-1})`. `MA_n` / `EMA_n` / `LL_n` / `HH_n` are trailing windows **including bar t only** (never forward). `TR_t = max(H_t−L_t, |H_t−C_{t-1}|, |L_t−C_{t-1}|)`, `ATR14 = mean(TR, 14)`. `robust_z(x; 200) = (x − median(x[-200:])) / (1.4826·MAD(x[-200:]) + eps)`. All divisions carry `+ eps` guards. All funding fields use the last settlement with timestamp `≤ t` (forward-fill, per coin). Timestamp fields use the current bar's own timestamp (known at decision time).

| name | bars-only definition | economic argument | family |
|---|---|---|---|
| TREND_10_60 | `(MA(C,10) − MA(C,60)) / MA(C,60)`, then robust_z over 200 | Medium-horizon CTA/cross-sectional trend flow is the most documented persistent price pressure; a 10-vs-60 spread sits between PUMP_DEV (20) and RET (200-z) and prices who is currently being carried vs stopped | trend |
| MOM_5_RAW | `sum(r[-5:])`, raw (no z-score) | 2.5-hour drift captures hot-money chasing without the 200-bar standardization of RET; raw scale preserves "big move = big signal," which standardization destroys | trend |
| MOM_24_RAW | `sum(r[-24:])` (~12h), raw | Half-day drift is the horizon of discretionary crypto momentum desks and funding-period positioning; prices persistence of directional order flow | trend |
| MOM_96_RAW | `sum(r[-96:])` (~48h), raw | Multi-day drift captures swing-fund and trend-follower inventory; mean-reversion at 5 bars and drift at 96 bars can coexist, and only this horizon sees the latter | trend |
| EMA_SPREAD | `(EMA(C,12) − EMA(48)) / EMA(48)`, robust_z 200 | EMA weighting reacts faster to fresh flow than the SMA spreads; prices recency-weighted consensus, i.e. what late entrants paid vs history | trend |
| MACD_HIST | `MACD = EMA12 − EMA26; sig = EMA(MACD,9); hist = MACD − sig`, robust_z 200 | The single most widely traded mechanical trigger; whether or not it "works," desks trade it, so its histogram turning prices coordinated flow | trend |
| TREND_R2 | signed R²: OLS of `ln(C)` on time over trailing 48; `value = sign(slope)·R²`, raw in [−1,1] | Separates trending drift from noisy drift with identical net return; high-R² trends attractPyramiding, low-R² chop gets faded — prices trend conviction, not direction | trend |
| EFFICIENCY | `|C_t − C_{t−25}| / (sum_{i=0..24} |C_{t−i} − C_{t−i−1}| + eps)`, raw in [0,1] | Kaufman efficiency ratio: straight-line travel vs path length. Choppy round-trips (low ER) mean mean-reversion regime; clean travel means momentum regime. Prices the regime the operator set cannot see | trend |
| REV_SHORT | `−sum(r[-6:])` (buy the 3-hour loser), raw | Market-maker inventory control mechanically fades short-horizon moves; the bounce-back after a 6-bar push is the dealer's edge, and this prices it | reversal |
| DIST_MA60_ATR | `(C_t − MA(C,60)) / (ATR14 + eps)`, robust_z 200 | Overextension measured in volatility units, not percent: a 5% deviation means nothing at 10% vol and everything at 1% vol. Liquidation cascades stretch this and snap back; prices stretch in risk units | reversal |
| BB_PCTB_20 | `(C − LBB20)/(UBB20 − LBB20 + eps)`, LBB/UBB = MA20 ± 2σ20, raw (clip 0–1 not required) | Position inside a volatility envelope, not a single bar (unlike CLOSE_POS). Envelope tags are where option dealers hedge and retail capitulates; prices statistically stretched location | reversal |
| BB_WIDTH | `(UBB20 − LBB20)/(MA20 + eps)`, robust_z 200 | The squeeze precedes the move: compressed envelopes forecast expansion as stops cluster tightly. Prices stored energy, direction-agnostic | reversal |
| RSI2 | `RSI(2)` on C, raw in [0,100] | Connors-style 2-period RSI measures the rubber band at its most stretched; RSI14-based DELTA_RSI cannot see a 1-hour panic. Prices immediate snap-back fuel | reversal |
| VOL_RATIO | `stdev(r,10) / (stdev(r,60) + eps)`, robust_z 200 | Level of vol (VOL_CLUST) vs *regime shift* in vol: a ratio >1 means vol just arrived. Market makers reprice spreads on the change, not the level; prices the repricing event | volatility |
| VOL_CHANGE | `(stdev(r,10) − stdev(r,10)[lag 20]) / (stdev(r,10)[lag 20] + eps)`, raw | Acceleration of realized vol; vol-of-vol momentum precedes quote withdrawal and stop runs. Prices fright arriving, not fright present | volatility |
| ATR_RATIO | `ATR14 / (ATR60 + eps)`, robust_z 200 | Range-based regime shift using full OHLC path, not close-only; ATR ratios detect the volatility expansion that close-only estimators miss on gap-and-hold bars | volatility |
| PARKINSON_20 | `sqrt(mean( (ln(H/L))² / (4·ln2) ) over 20)`, robust_z 200 | Parkinson estimator uses high-low path the close-only VOL_CLUST throws away; it is the more efficient realized-variance proxy. Prices true variance where hedging demand (options, variance swaps) actually lives | volatility |
| RANGE_MEAN_20 | `mean((H−L)/C over 20)`, robust_z 200 | HL_RANGE is a single-bar spike detector; this is the persistent wide-market regime (thin books, stressed liquidity). Spikes fade, regimes persist — prices the regime | range-candle |
| UPPER_WICK | `(H − max(O,C)) / (H − L + eps)`, raw in [0,1] | Long upper wicks are failed auctions: buyers pushed, sellers absorbed, price rejected. Stop-hunts and supply absorption leave exactly this signature; prices overhead supply | range-candle |
| LOWER_WICK | `(min(O,C) − L) / (H − L + eps)`, raw in [0,1] | Mirror image: dip-buy absorption and bid defense. Persistent lower wicks mark defended demand where shorts cover; prices the floor | range-candle |
| WICK_IMBAL | `UPPER_WICK − LOWER_WICK`, raw in [−1,1] | Net intrabar rejection sign; a green body with a huge upper wick is distribution, not strength. Prices who won the *inside* of the bar when the close lies | range-candle |
| BODY_FRAC | `|C − O| / (H − L + eps)`, raw in [0,1] | Small bodies = disagreement/indecision (doji regime); large bodies = conviction. Indecision clusters precede breakpoints as positioning is unresolved; prices conviction | range-candle |
| RANGE_EXPAND | `((H−L)/C)_t / (max((H−L)/C over prior 20) + eps)`, raw | A range breakout in relative terms: today's bar exceeds everything in 10 hours. Stop cascades and liquidation engines trigger on exactly this; prices the ignition bar | range-candle |
| DRIFT_OPEN | `(O_t − C_{t−1}) / (C_{t−1} + eps)`, raw | In 24/7 crypto there is no "overnight," but the between-bar repricing still concentrates settlement, funding, and liquidation flow. Prices the gap the continuous session still produces | range-candle |
| RELVOL_10_60 | `mean(V,10) / (mean(V,60) + eps)`, robust_z 200 | Relative volume: attention regime vs history. LOG_VOL sees level, VOL_TREND sees one bar; this sees the week-vs-day attention shift that precedes sustained moves | volume |
| VOL_CV_20 | `stdev(V,20) / (mean(V,20) + eps)`, raw | Lumpiness of participation: smooth retail flow vs whale-block flow. Lumpy volume means a few large players dominate — prices concentration of counterparties | volume |
| VOL_PRICE_CORR | `Pearson(C, V) over trailing 20`, raw in [−1,1] | Up-on-volume vs up-on-air: accumulation rises with volume, distribution rises against it. Prices whether volume confirms or betrays price | volume |
| UP_DOWN_VOL | `sum(V where C>O) / (sum(V over 20) + eps)`, raw in [0,1] | Volume-weighted buying pressure: what fraction of the last 10 hours of trading happened on green bars. Prices signed participation, not raw activity | volume |
| VOL_SHOCK | `V_t / (max(V over prior 60) + eps)`, raw | Historic attention spike: today's bar out-trades everything in 30 hours. Liquidation/news candles print exactly here; prices the anomaly bar for fade-or-follow logic | volume |
| AMIHUD_20 | `log(mean(|r| / (C·V + eps)) over 20)`, robust_z 200 | Persistent illiquidity premium over 10 hours, vs single-bar LIQ_SCORE. Assets that are *chronically* expensive to trade carry a return premium; prices the premium, not the tick | impact-liquidity |
| ILLIQ_CHANGE | `AMIHUD_20 / (AMIHUD_60 + eps) − 1`, raw | Liquidity withdrawal: market makers pulling quotes before a jump. The *change* in Kyle's-lambda-style cost predicts discontinuities that the level does not; prices vanishing depth | impact-liquidity |
| CLOSE_LOC_5 | `(C − LL5) / (HH5 − LL5 + eps)`, raw in [0,1] | 5-bar channel position: is price pressing the top or bottom of its 2.5-hour range. Short-horizon breakout proximity that single-bar CLOSE_POS cannot express | structure |
| DONCH_POS_48 | `(C − LL48) / (HH48 − LL48 + eps)`, raw in [0,1] | 24-hour Donchian position: Turtle-style breakout location. Daily-extreme tags are where systematic breakout funds enter and losers stop; prices proximity to the crowd's trigger | structure |
| HH_DIST_ATR | `(HH20 − C) / (ATR14 + eps)`, raw | Distance to 10-hour resistance in vol units: the supply overhang above. Near-zero with rising volume = absorption before breakout; large = no pressure; prices the ceiling | structure |
| LL_DIST_ATR | `(C − LL20) / (ATR14 + eps)`, raw | Mirror: distance from 10-hour support. Tests of the floor are where dip demand either shows or fails; prices the cushion | structure |
| STREAK_10 | `(#{r>0} − #{r<0}) over trailing 10 using sign(r)`, integer −10..+10, raw | Count-based persistence, not magnitude-based: ten small green bars herd differently than one big one. Prices herding participation breadth that return-sums wash out | persistence |
| AUTOCORR_5_20 | `lag-5 autocorrelation of r over trailing 48`, raw in [−1,1] | Direct measurement of the momentum-vs-reversal regime at 2.5-hour lag: positive = trend-following flow dominates, negative = dealer fade dominates. Prices the regime parameter itself | persistence |
| SIGN_ENTROPY | `−(p·ln p + (1−p)·ln(1−p))`, `p = fraction of up bars over 20`, raw in [0, ~0.69] | Certainty of direction: max entropy = coin-flip chop (wide spreads, fade everything), min entropy = one-sided tape (trend, don't fade). Prices tape certainty for spread/conviction scaling | persistence |
| SKEW_48 | sample skewness of `r` over 48, raw | Crash/jump asymmetry over 24h: negative skew = liquidation-tail market where puts and stop-protection are bid. Skew risk carries a premium; prices the asymmetry | tail-risk |
| KURT_48 | excess kurtosis of `r` over 48, raw | Fat-tail regime: clustered stop cascades make tails fat before they realize. Fragile positioning shows up as kurtosis before it shows up as returns; prices fragility | tail-risk |
| MAXDD_96 | `(C − max(C trailing 96)) / (max(C trailing 96) + eps)`, raw in (−1, 0] | 48-hour drawdown depth: capitulation gauge. Deep drawdowns exhaust disposition sellers and trigger value/buy-the-dip flow; prices exhaustion | tail-risk |
| HOD_SIN | `sin(2π·hour_UTC / 24)` from bar timestamp, raw in [−1,1] | Crypto has a documented intraday seasonality (Asia/EU/US handover, funding hours, equity-open spillover). The sine component lets formulas learn "risk-on hours" without a clock lookup | session-time |
| HOD_COS | `cos(2π·hour_UTC / 24)` from bar timestamp, raw in [−1,1] | Cosine partner of HOD_SIN: together they encode circular time so midnight doesn't look distant from 23:30. Prices the orthogonal phase of the same handover cycle | session-time |
| FUND_COUNTDOWN | `(bars since last settlement on 30m grid) / 16`, raw in [0,1) | Positioning into the 8-hour funding payment: longs/shorts adjust leverage ahead of the known transfer. Prices the predictable pre-settlement urgency cycle | session-time |
| WEEKEND | `1 if Sat/Sun UTC else 0` from bar timestamp | Weekend books are thinner with a different participant mix (less institutional flow, more retail/liquidation-driven moves). Prices the structurally different weekend microstructure | session-time |
| FUND_RATE | last observed settlement funding rate, forward-filled per coin, raw | The crowded-long tax itself: persistently positive funding = crowded longs paying shorts, the fuel and the warning of a long squeeze. The single most direct positioning proxy available | funding |
| FUND_Z | `robust_z(FUND_RATE; 90 settlements ≈ 30d)`, raw | Abnormal crowding vs the coin's own history: +3σ funding on a coin that normally sits flat is a qualitatively different signal than the same level on a chronically hot coin | funding |
| FUND_MOM | `FUND_RATE − FUND_RATE[lag 21 settlements ≈ 7d]`, raw | Crowding acceleration: latecomer leverage arriving. Rising funding into rising price = reflexive long pile-on, the classic pre-reversal configuration; prices the second derivative of positioning | funding |
| POST_FUND_DRIFT | `mean(r over the 2 bars after each of last 10 settlements)`, raw | Settlement microstructure has a repeatable post-payment drift (losers cut, winners press). A trailing average of it prices the coin-specific settlement footprint the formulas can fade or follow | funding |

## 1. Families ranked (most promising first)

1. **funding** — the only family with genuinely *new information* (a non-price series). Everything else is a transform of OHLCV the search already had; funding imports outside positioning data. FUND_RATE/FUND_Z/FUND_MOM directly measure the crowded-long tax. Highest prior.
2. **session-time** — timestamp seasonality is also information the 12 factors never touch (none of them read the clock). HOD_SIN/COS, WEEKEND, FUND_COUNTDOWN exploit predictable participant-mix cycles. Cheap, orthogonal by construction.
3. **impact-liquidity** — Kyle's-lambda/Amihud reasoning prices the *cost* of trading, which is the mechanism that turns flow into returns. AMIHUD_20 as a persistent premium and ILLIQ_CHANGE as withdrawal detection both carry textbook microstructure arguments.
4. **tail-risk** — skew/kurtosis/drawdown price crash-risk premia and capitulation exhaustion, the exact phenomena that decide whether a drift signal survives costs. Nothing in the 12 measures asymmetry or exhaustion.
5. **range-candle** — wicks, body fraction, and range expansion read the *inside* of the bar (auction failure, absorption, ignition), information the close-only factors discard. Every bar's H/L/O shape is currently used only by PRESSURE, CLOSE_POS, HL_RANGE.
6. **structure** — Donchian position and ATR-distance-to-extremes price where systematic breakout flow and stop clusters live. This is where the actual entry/stop triggers of other participants sit.
7. **persistence** — streak counts, autocorrelation, and entropy measure the momentum-vs-reversal *regime parameter* directly instead of betting on one side. AUTOCORR_5_20 in particular tells formulas when to be a trend follower at all.
8. **reversal** — economically sound (dealer inventory, envelope tags) but partially overlaps existing mean-reversion coverage (PUMP_DEV, DELTA_RSI, MOM_REV). Still earns its place via volatility-unit scaling and envelope framing.
9. **volume** — attention and signed participation matter, but three of the 12 already mine volume (LOG_VOL, VOL_TREND, FOMO), so marginal information per candidate is lower; RELVOL and VOL_PRICE_CORR are the best of the bunch.
10. **volatility** — weakest marginal case *for direction*: VOL_CLUST plus HL_RANGE already cover vol level, so only the regime-shift ratios (VOL_RATIO, VOL_CHANGE) and the OHLC estimators (PARKINSON_20) add anything, and vol predicts magnitude more than sign.

## 2. Suspected duplicates (name the pair explicitly)

- **AMIHUD_20 vs LIQ_SCORE** — highest-risk pair. Both are Amihud `|ret|/dollar-volume` transforms; AMIHUD_20 is the 20-bar mean of LIQ_SCORE's single-bar quantity. Expect strongly negative correlation (sign flip noted). Keep both into round 2 only so the correlation cut can verify; if |ρ|>0.7, kill one.
- **RANGE_MEAN_20 vs HL_RANGE** — same primitive `(H−L)/C`, single-bar vs 20-bar mean. The z-scored 20-bar mean will correlate with a smoothed HL_RANGE. Keep both, expect the cut to keep one.
- **RELVOl_10_60 (VOL family) vs LOG_VOL / VOL_TREND** — relative-volume level will correlate with LOG_VOL's z-scored level; VOL_SHOCK vs VOL_TREND (both spike detectors) likewise. Flagged triple: RELVOL_10_60–LOG_VOL, VOL_SHOCK–VOL_TREND.
- **DIST_MA60_ATR vs PUMP_DEV** — both are distance-from-moving-average overextension gauges (60 vs 20 lookback, ATR vs percent scaling). Expect moderate-high correlation in trending tapes.
- **BB_PCTB_20 vs CLOSE_POS** — both are "position within an interval" with the same 0–1 geometry; BB uses a 20-bar volatility envelope, CLOSE_POS a single bar, but in quiet markets they rhyme. Same concern for **CLOSE_LOC_5 vs CLOSE_POS** and **DONCH_POS_48 vs CLOSE_POS** (progressively safer as the window lengthens: 1 vs 5 vs 48).
- **HH_DIST_ATR vs DONCH_POS_48** — distance-to-high and position-in-48-bar-range are near-monotone transforms of each other when ATR is stable; expect high |ρ|. Similarly **LL_DIST_ATR vs DONCH_POS_48**.
- **UPPER_WICK vs LOWER_WICK vs WICK_IMBAL** — WICK_IMBAL is literally the difference of the other two, so it is algebraically redundant with the pair jointly. Keep the imbalance plus only one leg, or keep all three and let round 2's pairwise cut handle it.
- **PARKINSON_20 vs VOL_CLUST** — two realized-volatility levels over similar windows (20 vs 10), different estimators. Expect |ρ| in the 0.6–0.85 band; the economic case for keeping Parkinson is estimator efficiency, not novelty.
- **VOL_RATIO vs VOL_CLUST** — ratio contains the same 10-bar stdev in its numerator; mechanically coupled. The *information* is in the denominator normalization, so check partial correlation, not raw.
- **FUND_Z vs FUND_RATE** — z-score vs level of the same series; correlated by construction over short histories, diverging only as regimes shift. Keep both deliberately (level = tax now, z = abnormality), flag for the round-2 budget.
- **MOM_5_RAW vs MOM_REV / RET** — 5-bar return sum shares bars with RET's window and MOM_REV's 5-bar sign logic. Expect MOM_5_RAW–RET moderate correlation; the defense is raw-vs-standardized scaling.
- **HOD_SIN vs HOD_COS** — orthogonal by construction over full days (ρ≈0); listed here to state explicitly they are *not* duplicates despite sharing a family.
- **STREAK_10 vs MOM_5_RAW** — sign-count vs signed-sum over overlapping windows; they agree on clean trends and disagree exactly when bars are mixed (big-up/small-downs), which is the intended complementarity. Mild flag only.
- **POST_FUND_DRIFT vs FUND_COUNTDOWN** — both key off the settlement clock; drift is a return footprint, countdown is a deterministic clock, so correlation should be low, but they jointly span "the funding cycle" and round 2 should not keep both plus all three FUND_* without checking the family block correlation.

## 3. Traps in the data constraints

- **The funding forward-fill trap.** The funding series is settlement events every 8h forward-filled across sixteen 30m bars. Any funding factor is therefore a step function constant for 16 bars at a time. Its bar-level variance is ~1/16th what it looks like, its autocorrelation is ~0.94 mechanically, and any correlation measured at bar grain is attenuated. Round 2 must measure funding-factor correlations at *settlement grain* (or downsampled), or it will conclude funding factors are "low variance / low correlation" for purely mechanical reasons. Related: FUND_MOM with a 21-settlement lag needs ~7 days of settlement history — warmup must be in settlement count, not bar count, or early-sample values are garbage.
- **The settlement-grid assumption.** FUND_COUNTDOWN and POST_FUND_DRIFT assume settlements on an exact 8h UTC grid. If any of the 28 coins settles off-grid (or a venue changed schedule mid-sample), the countdown is phase-shifted and the drift average mixes pre- and post-settlement bars. Verify actual settlement timestamps per coin before trusting either factor; a one-bar phase error halves the drift signal.
- **Division by zero / flat bars.** `(H−L+eps)`, `(HH−LL+eps)`, `(C·V+eps)` guards are load-bearing, not cosmetic: dead hours print `H==L`, zero-volume bars print `V==0`, and ILLIQ_CHANGE-style ratios of two small numbers explode. Worse, on a flat stretch every range-normalized factor collapses to 0/eps noise that *passes* a variance check on paper while being pure artifact. The vary-check (ptp/std > 1e-9) must be evaluated on the winsorized factor, and denominators need flooring at a percentile of their own history, not just eps.
- **The timestamp-timezone trap.** HOD_SIN/COS and WEEKEND assume timestamps are UTC and stationary. If the CSV timestamps are venue-local, millisecond-shifted, or the sample spans a venue timezone change, the "seasonality" learned is a clock error. Confirm UTC and confirm the 24h sinusoid against per-hour returnMeans before promoting these.
- **The LIQ_SCORE lesson, restated.** Any formula that silently degrades to a constant (the `liquidity`/`fdv` incident: constant 0.4 passing all well-formedness checks) will do so again here via `max(V)==0` windows, all-flat `HH==LL` windows, or missing funding history per coin. Every candidate above needs the same treatment LIQ_SCORE eventually got: assert per-coin, per-window variance, not pooled variance — a factor that varies on 27 coins and is flat on one still feeds the 28th coin a constant.
- **Crypto has no close.** DRIFT_OPEN, BODY_FRAC, and the wick family assume `O_t` vs `C_{t-1}` and intrabar shape carry auction meaning. On venues where the "open" is just the previous close carried forward (or interpolated over a missing bar), DRIFT_OPEN is identically zero and wicks are synthetic. Check what fraction of bars have `O_t != C_{t-1}` before trusting the gap family.
- **Robust-z lookback vs factor lookback mismatch.** Several candidates stack a 200-bar robust-z on top of a 48–96-bar trailing statistic (e.g., TREND_R2, MOM_96_RAW if z-scored). The effective memory is ~300 bars (~6 days), which eats the start of every coin's history and smooths regime transitions the factor claims to detect. Prefer raw scaling where the definition already normalizes (EFFICIENCY, TREND_R2, entropy, correlations, positions in [0,1]) and reserve robust-z for unbounded quantities.
- **Correlation measured at the wrong grain, again.** Bar-level Pearson between a 96-bar-smooth factor (MOM_96_RAW, MAXDD_96) and a 1-bar factor (VOL_SHOCK, DRIFT_OPEN) is mechanically near zero even when the slow factor fully determines the fast factor's *regime*. Round 2's "low correlation" screen should be supplemented with regime-conditional checks (correlation of the fast factor within top/bottom terciles of the slow one), or it will pass pairs that are economically the same trade.


---

# ROUND 2 TASK (respond to this)

You have seen the other model's round-1 list. Now do the selection work. This
is the round that decides, so be harsh.

Produce, in this order:

## 1. Cross-audit of the other model's list

Name, explicitly, every candidate in the OTHER model's list that you think is:
  - a duplicate of one of the 12 existing factors, and which one
  - a duplicate of a candidate in your own list, naming both
  - not computable from OHLCV + sparse funding, and why
  - not strictly causal as written
  - near-degenerate (a scaled copy, a sign flip, or a monotone transform of
    something already in the set)

Be specific. "These two overlap" is not an answer; "X is (C-MA20)/MA20 and
PUMP_DEV is (C-MA20)/MA20 robust-z'd, same numerator" is.

## 2. Shortlist of 30

Pick 30 survivors from the union of both pools. You may add factors the other
model missed, but justify each addition against both lists.

For each: name, the exact bars-only formula, the family, and a one-line economic
argument.

## 3. Your proposed 24

From your 30, name the 24 you would actually ship, and for the 6 you are
dropping say why.

## 4. Correlation budget reasoning

We will measure absolute Pearson correlation of every shortlisted factor
against the 12 existing and against each other, pooled across all 28 coins and
all bars, on a common valid mask.

Given that measurement is coming, which pairs do you EXPECT to exceed |r| 0.6,
and which families should be treated as mutually exclusive so that at most one
survives? Name the families.

## 5. What you would do differently

Name the single candidate from either list that you think is most likely to be
a disguised duplicate of something already in the set, and explain.

Do not run code. Do not edit files. Tables and prose only.
