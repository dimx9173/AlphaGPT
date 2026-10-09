Round 3 — decisive. No hedging.

**1. Adjudication of 24 contested**

| candidate | bucket | reason / swap note |
|---|---|---|
| AMIHUD_20 | **CUT** | 20-bar mean of `|r|/dollar_vol` — same numerator/denom as E:LIQ_SCORE with sign-flip removed. Mechanical duplicate; cross-audit confirms. |
| AUTOCORR_20 | **CUT** | Lag-5 autocorr over 48 — persistence primitive already covered by EFFICIENCY_12 + CHOP_14 regime dials; weaker argument than the dials themselves. |
| BODY_RATIO | **KEEP** | `|C-O|/(H-L)` — body/concentration; structurally distinct from wick legs and from range-level factors. |
| CHOP_14 | **KEEP** | `sum(TR)/range` — regime dial; complements EFFICIENCY_12 (negative correlation expected, different geometry). |
| DD_DURATION | **CUT** | Duration of drawdown — DD_200 already in core covers capitulation depth; duration is noisier and less actionable at 30m. |
| FUND_FLIP_7 | **CUT** | Derived interaction; FUND_SURPRISE + SLOPE already give crowding + direction. RPN MUL can rebuild. |
| FUND_LAST | **CUT** | Raw funding level — cross-audit: dominated by venue/coin offset when pooled; SURPRISE is the correct primitive. |
| FUND_X_MOM | **CUT** | Product term `FUND * sign(sum r)`; spend slot on primitive. |
| FUND_SURPRISE | **KEEP** | Abnormality vs 21-settlement median; only defensible funding factor at bar grain (step-function, not index-windowed). |
| GAP_1 | **KEEP** | Between-bar repricing; isolates funding/liquidation-flow events that RET (close-to-close) smeared. |
| KURT_50 | **KEEP** | Sample kurtosis — complements core SKEW_50 (tail family, different moment; correlation ~0.3). |
| LOWER_WICK | **KEEP** | Bid-defense leg; pair with UPPER_WICK gives full wick decomposition. |
| MINRET_20 | **KEEP** | Worst 20-bar log-return; direct downside-tail gauge complementary to SKEW. |
| RANGE_EXPAND | **KEEP** | Ignition bar `curr/max(past 20)`; event-level range spike vs RANGE_MEAN_20's regime mean. |
| RANGE_MEAN_20 | **KEEP** | Persistent wide-market regime; mean of same primitive as E:HL_RANGE but regime vs single-bar. |
| SHADOW_BAL_12 | **CUT** | `UPPER-LOWER` over 12 bars — algebraically derived from UPPER_WICK + LOWER_WICK pair kept; redundant. |
| TIME_SINCE_HIGH_100 | **KEEP** | Age of HH100; geometric complement to DONCH_POS_100 (position in range vs time since edge). |
| TOD_COS / TOD_SIN | **CUT** both | Read timestamp, not OHLCV; risk is venue-local clock drift = factor becomes clock-error, not signal. |
| TREND_R2 | **CUT** | Signed R² over 48 — same OLS fit as SLOPE_TSTAT_24 (core); keep significance-dial, drop fit-quality dial. |
| UPPER_WICK | **KEEP** | Auction-overhead leg; paired with LOWER_WICK. |
| VARIANCE_RATIO_8 | **CUT** | Variance-ratio persistence; overlaps AUTOCORR + regime dials already kept. |
| VOL_OF_VOL_30 | **KEEP** | `stdev(V)/mean(V)`; participation lumpiness; distinct from LOG_VOL level and VOL_TREND change. |

No core member replaced. Final 24 = 12 agreed + 12 above kept.

**2. Final 24 (numbered)**

1. ATR_RATIO_100 · 2. BREAKOUT_DIST_50 · 3. DD_200 · 4. DOLLAR_IMBAL_12 · 5. DONCH_POS_100 · 6. EFFICIENCY_12 · 7. OBV_SLOPE_20 · 8. PARKINSON_20 · 9. ROLL_SPREAD · 10. SKEW_50 · 11. SLOPE_TSTAT_24 · 12. VWAP_DEV_24 · 13. BODY_RATIO · 14. CHOP_14 · 15. FUND_SURPRISE · 16. GAP_1 · 17. KURT_50 · 18. LOWER_WICK · 19. MINRET_20 · 20. RANGE_EXPAND · 21. RANGE_MEAN_20 · 22. TIME_SINCE_HIGH_100 · 23. UPPER_WICK · 24. VOL_OF_VOL_30

**3. Implementable spec (per factor)**

All windows include t and only past bars (t-i, i≥1). Funding factors use only `timestamp ≤ t` settlement with `t_last_settlement`; never an indexed window over the sparse series. Rob-z = median/MAD over trailing 200 bars; raw = comparable range explicitly bounded.

| # | name | exact bars-only formula | windows | guard / denom | raw / z | range / notes |
|---|---|---|---|---|---|---|
| 1 | ATR_RATIO_100 | `ATR14 / (ATR100 + eps)`; ATR = mean(TR,n), TR=max(H-L,|H-C_prev|,|L-C_prev|) using C_{t-1} | 14, 100 | `+eps` denom | robust-z 200 | vol ratio; coin-comparable after z |
| 2 | BREAKOUT_DIST_50 | `(C - HH50) / (ATR14 + eps)` | 50 (HH), 14 (ATR) | `+eps` | raw | signed ATR units; ~[-N, N] |
| 3 | DD_200 | `(C - max(C,200)) / (max(C,200) + eps)` | 200 | denom = max+eps, never 0 (max ≥ C > 0 for crypto) | raw | [-1, 0] |
| 4 | DOLLAR_IMBAL_12 | `sum(sign(C-O)*C*V,12) / (sum(C*V,12)+eps)` | 12 | `+eps` | raw | [-1, 1]; signed dollar participation |
| 5 | DONCH_POS_100 | `(C - LL100) / (HH100 - LL100 + eps)` | 100 | `+eps`; if HH=LL, return 0.5 (degenerate flat) | raw | [0, 1] |
| 6 | EFFICIENCY_12 | `abs(C_t - C_{t-12}) / (sum(abs(dC),12)+eps)` | 12 | `+eps` | raw | [0, 1]; straight-line vs path |
| 7 | OBV_SLOPE_20 | OLS slope of OBV over 20, `slope / (mean(V,20)+eps)`; OBV = cum(sum(sign(C-O)*V)) | 20 | `+eps` | robust-z 200 | scaled by vol; unbounded |
| 8 | PARKINSON_20 | `sqrt(mean( ln(H/L)^2 / (4*ln2), 20 ))`; robust-z 200 | 20 | ln(H/L) defined (H>L>0) | robust-z 200 | vol level; comparable after z |
| 9 | ROLL_SPREAD | `2 * sqrt(max(-autocov(r,20), 0)) / (C + eps)`; autocov of r=ln(C_t/C_{t-1}) over 20 | 20 | max(.,0) guards negative; `+eps` | robust-z 200 | spread estimate; scale via C |
| 10 | SKEW_50 | sample skew of r over 50: `mean((r-μ)^3) / (std(r)^3 + eps)` | 50 | `+eps` denom | robust-z 200 | unbounded |
| 11 | SLOPE_TSTAT_24 | OLS slope t-stat of ln(C) on time over 24: `slope / SE(slope)` | 24 | SE>0 by OLS (if constant C, slope=0, t=0) | robust-z 200 | unbounded; kills noisy big slopes |
| 12 | VWAP_DEV_24 | `(C - VWAP24) / (ATR14 + eps)`; VWAP = `sum(HLC3*V)/sum(V)` over 24 (HLC3 = (H+L+C)/3) | 24, 14 | `+eps` | raw | ATR units; comparable |
| 13 | BODY_RATIO | `abs(C-O) / (H-L + eps)` | 1 (bar) | `+eps`; if H=L, return 0 | raw | [0, 1]; conviction vs indecision |
| 14 | CHOP_14 | `sum(TR,14) / (HH14 - LL14 + eps)`; TR as above | 14 | `+eps`; if range=0, return 1 | raw | [1, ∞); chop index |
| 15 | FUND_SURPRISE | `FUND_LAST(t) - median(FUND_SETTLEMENT with ts≤t and >t-21*8h, 21 settlements)`; forward-fill last settlement rate; median over up to 21 settlement events | 21 settlement events (not bars) | median of ≥1 settlement; if 1, surprise=0 | raw | funding-rate units (%/8h); abnormality |
| 16 | GAP_1 | `(O_t - C_{t-1}) / (C_{t-1} + eps)` | 1 (prev close) | `+eps`; if C_prev missing (start), return 0 | raw | unbounded; repricing gap |
| 17 | KURT_50 | sample kurtosis of r over 50: `mean((r-μ)^4)/(std(r)^4 + eps)` | 50 | `+eps` | robust-z 200 | unbounded |
| 18 | LOWER_WICK | `(min(O,C)-L) / (H-L+eps)` | 1 | `+eps`; H=L → 0 | raw | [0, 1]; bid defense |
| 19 | MINRET_20 | `min(ln(C_t / C_{t-i}))` for i=1..20 (worst of 20 trailing single-bar returns) | 20 | none (min over finite set) | robust-z 200 | unbounded below; downside tail |
| 20 | RANGE_EXPAND | `(H-L)_t / (max(H-L over t-19..t) + eps)` | 20 (past range max) | `+eps`; at t=0 if first bar, denom=curr so =1 | raw | [0, ∞); ignition |
| 21 | RANGE_MEAN_20 | `mean((H-L)/C, 20)` | 20 | C>0 always; denom implicit via /C | robust-z 200 | positive; regime width |
| 22 | TIME_SINCE_HIGH_100 | `(bars since HH100) / 100`; HH100 = max(C,100) | 100 | if HH at t, 0 | raw | [0, 1]; fresh vs stale high |
| 23 | UPPER_WICK | `(H - max(O,C)) / (H-L+eps)` | 1 | `+eps`; H=L → 0 | raw | [0, 1]; failed auction |
| 24 | VOL_OF_VOL_30 | `stdev(V,30) / (mean(V,30)+eps)` | 30 | `+eps` | robust-z 200 | positive; participation lumpiness |

**4. Expected |r| > 0.6 pairs (check these when measured)**

- PARKINSON_20 vs E:VOL_CLUST (0.6–0.85) — both realized-vol levels.
- RANGE_MEAN_20 vs E:HL_RANGE (0.5–0.7) — mean vs instance of same primitive.
- BODY_RATIO vs UPPER_WICK / LOWER_WICK (mechanically negative, -0.7 to -0.9; identity body + wicks = 1).
- DONCH_POS_100 vs BREAKOUT_DIST_50 (0.6–0.8 when ATR stable; position-in-range vs distance-to-edge).
- EFFICIENCY_12 vs CHOP_14 (strongly negative, ~-0.8 to -0.9; straight travel vs chop of same path).
- RANGE_EXPAND vs RANGE_MEAN_20 (positive, ~0.5–0.7; mean predicts ignition).
- VWAP_DEV_24 vs E:CLOSE_POS (moderate positive; both measure deviation from reference price, different refs).
- KURT_50 vs SKEW_50 (moderate positive ~0.3–0.5 — *should not exceed 0.6*; flag only if it does, indicates tail dominance).

Pairs expected < 0.6 (not flagged; if they come back high, distrust measurement first): FUND_SURPRISE vs any existing price factor (step function, different time scale); GAP_1 vs RET (gap is a component, not duplicate); MINRET_20 vs SKEW_50 (tail vs asymmetry); VOL_OF_VOL_30 vs LOG_VOL / VOL_TREND (different stats on same series).

**Deliverable is implementable as written.** No ambiguity in guard notation; denominators explicit; funding uses settlement timestamp, never bar-indexed window; no centered windows, no full-series stats, no `adjust=True` EWM. Round 2's correlation budget is reserved for measurement; this is the spec that measurement will be judged against.

No code written — this is the spec round. When measured, the pairs above get checked first.
