**ROUND 3 — adjudication and implementable spec**

## 1. Adjudication of the 24 contested candidates

Core 12 stand. No core member is removed.

### A-proposed 12 — all CUT / FOLD, none ship as-written

| candidate | verdict | reason |
|---|---|---|
| `AMIHUD_20` | CUT | Duplicate of existing `LIQ_SCORE` (#1) by construction: 20-bar mean of the same `log(\|ret\|/dvol)` quantity. One must go; existing cannot go. |
| `BODY_RATIO` | CUT | Duplicate of existing `PRESSURE` (#2): same single-bar `(C-O)/(H-L)` before monotone `tanh`. `SMA_5` does not create new information. |
| `CHOP_14` | FOLD | Folds into agreed `EFFICIENCY_12`. Both are path-efficiency / chop gauges. Keep `EFFICIENCY_12`; do not swap. |
| `FUND_SURPRISE` | FOLD | Folds into `FUND_LAST`. Surprise contains level (`F_t − median(prior)`), correlates ~0.5 with level by construction. Keep level + interaction + flip dynamics instead. |
| `GAP_1` | CUT | Degenerate on 24/7 30-min crypto: `(O_t − C_{t−1})` is one tick of jitter, frequently ~0. Guard-bait, tiny variance. |
| `LOWER_WICK` | FOLD | Folds into `SHADOW_BAL_12`. Raw one-sided wick is subsumed by the smoothed signed imbalance. Keep the imbalance; do not ship raws. |
| `UPPER_WICK` | FOLD | Same as `LOWER_WICK`. Fold into `SHADOW_BAL_12`. |
| `RANGE_EXPAND` | FOLD | Folds into agreed `BREAKOUT_DIST_50` + `ATR_RATIO_100`. Range-vs-ATR expansion is already spanned by those two. |
| `RANGE_MEAN_20` | CUT | Duplicate of existing `HL_RANGE` (#9) smoothed, and of agreed `PARKINSON_20`. No new primitive. |
| `TIME_SINCE_HIGH_100` | CUT | Steppy argmax-age with long flats (low variance); range-position already covered by `DONCH_POS_100` + `BREAKOUT_DIST_50`. Weakest breakout leg. |
| `TREND_R2` | FOLD | Folds into agreed `SLOPE_TSTAT_24`. One trend-per-noise representative suffices. Keep `SLOPE_TSTAT_24`. |
| `VOL_OF_VOL_30` | CUT | Lowest marginal value: volume instability overlaps `VOL_RANK_100` (attention surprise) + `KURT_50` (tail weight) + `ATR_RATIO_100` (regime). Cut to fund seasonality. |

### B-proposed 12 — all KEEP

| candidate | verdict | reason |
|---|---|---|
| `FUND_LAST` | KEEP | Only exogenous P&L stream in the data. Level of carry. Irreplaceable. |
| `FUND_X_MOM` | KEEP | Only crowdedness interaction (carry × trend sign). Level alone cannot express squeeze-vs-fade. |
| `FUND_FLIP_7` | KEEP | Only sign-persistence / consensus-exhaustion gauge. Level, shock, and interaction all miss it. |
| `AUTOCORR_20` | KEEP | Only tape-memory sign switch (continuation vs bounce regime). |
| `VARIANCE_RATIO_8` | KEEP | Lo–MacKinlay violation priced directly; distinct math from Kaufman `EFFICIENCY_12`. |
| `VOL_RANK_100` | KEEP | Only nonparametric volume surprise; robust where z-scores break. |
| `KURT_50` | KEEP | Fat-tail / cascade regime; complements agreed `SKEW_50` (asymmetry vs tail weight). |
| `MINRET_20` | KEEP | Capitulation / forced-sell marker in vol units. Keeps cheap entry side; FOMO mirror cut. |
| `SHADOW_BAL_12` | KEEP | Sole microstructure survivor; multi-bar defending-book footprint. Sole wick representative. |
| `DD_DURATION` | KEEP | Capitulation clock (time underwater); depth alone (`DD_200`) cannot express exhaustion. |
| `TOD_SIN` | KEEP | Indivisible pair with `TOD_COS`. Only family orthogonal by construction; only ex-ante-known liquidity cycle. Both or neither. |
| `TOD_COS` | KEEP | Cosine half of the circular pair. Ships with `SIN` or not at all. |

Score: 12 KEEP (all B), 12 CUT/FOLD (all A, folded targets named above).

## 2. Final 24

Agreed core 12 plus the 12 keeps above:

1. `VWAP_DEV_24`
2. `DOLLAR_IMBAL_12`
3. `OBV_SLOPE_20`
4. `EFFICIENCY_12`
5. `VARIANCE_RATIO_8`
6. `AUTOCORR_20`
7. `SLOPE_TSTAT_24`
8. `SKEW_50`
9. `KURT_50`
10. `MINRET_20`
11. `PARKINSON_20`
12. `ATR_RATIO_100`
13. `DD_200`
14. `DD_DURATION`
15. `DONCH_POS_100`
16. `BREAKOUT_DIST_50`
17. `ROLL_SPREAD`
18. `SHADOW_BAL_12`
19. `FUND_LAST`
20. `FUND_X_MOM`
21. `FUND_FLIP_7`
22. `VOL_RANK_100`
23. `TOD_SIN`
24. `TOD_COS`

## 3. Implementable specification

**Global conventions.** Bar `t` has `O_t,H_t,L_t,C_t,V_t`, all `>0` except `V_t ≥ 0`; bar close time `T_t`. `ret_t = ln(C_t / C_{t−1})`, defined for `t ≥ 1`, else `0`. `DVOL_t = V_t · C_t`. `typical_t = (H_t+L_t+C_t)/3`. Trailing window `t−N+1..t` includes `t`, past bars only. `median_N`, `mean_N`, `std_N` (population, `ddof=0`), `min_N`, `max_N` are trailing. `robust_z_200(x_t) = (x_t − med) / (1.4826 · MAD)` where `med = median_200(x)`, `MAD = median_200(|x − med|)`; if `MAD < 1e-12` or fewer than 20 valid bars, return `0.0`. `SMA_N` / `STD_N` trailing, minimum 1 bar (short windows return the available mean; `STD` with `<2` bars returns `0`). Division guard notation `a / max(b, eps)` or explicit fallback stated per factor. No centered windows, no full-sample stats, no `ewm(adjust=True)`, no whole-series rank. Warmup: any factor returns `0.0` (or `0.5` for bounded position factors where stated) until its shortest trailing window is full. Funding: `F_t` = rate of the last settlement with settlement-timestamp `< T_t` (strictly before bar close), forward-filled; no bar-indexed window over funding; settlement-indexed windows only; if no prior settlement, `F_t = 0.0`.

| # | name | exact bars-only formula | windows | output / transform |
|---|---|---|---|---|
| 1 | `VWAP_DEV_24` | `vwap = Σ_{i=t−23..t}(typical_i·V_i) / max(Σ V_i, 1e-12)`; `d = (C_t − vwap)/max(vwap,1e-12)`; factor = `robust_z_200(d)`; if `ΣV_i==0` return `0` | 24, 200 | unbounded z, raw z-scored |
| 2 | `DOLLAR_IMBAL_12` | `s_i = sign(C_i − O_i)` (`0` if equal); factor = `Σ_{12}(s_i·DVOL_i) / max(Σ_{12}DVOL_i, 1e-12)`; if denom `0` return `0` | 12 | raw in `[−1,1]`, bounded so comparable; no z |
| 3 | `OBV_SLOPE_20` | `obv_t = Σ_{j≤t} sign(C_j−O_j)·V_j` (cumulative from bar 0); factor = `(obv_t − SMA_20(obv)) / max(STD_20(obv),1e-12)`; if `STD<1e-12` return `0` | 20 | unbounded standardized; raw (own window norm, no 200-z) |
| 4 | `EFFICIENCY_12` | `num = \|C_t − C_{t−12}\|`; `den = Σ_{i=t−11..t}\|C_i − C_{i−1}\|`; factor = `num / max(den,1e-12)`; if `den<1e-12` return `0` | 12 (+1 seed) | raw in `[0,1]`; no z |
| 5 | `VARIANCE_RATIO_8` | 48 one-bar rets `r = ret_{t−47..t}`; `v1 = var(r)`; six non-overlapping 8-bar sums `R_k = Σ_{8} r`; `v8 = var(R_1..R_6)`; factor = `v8 / max(8·v1, 1e-18)`; if `v1<1e-18` return `1.0` (random-walk null) | 8/48 | raw, ~`[0,8]` centered on 1; no z (ratio self-normalizes) |
| 6 | `AUTOCORR_20` | `a = ret_{t−19..t}`, `b = ret_{t−20..t−1}`; factor = Pearson(`a`,`b`) = `cov/max(std_a·std_b,1e-18)`; if either `std<1e-12` return `0` | 20 (+1 lag) | raw in `[−1,1]`; no z |
| 7 | `SLOPE_TSTAT_24` | OLS of `y_i = ln(C_i)` on `x = 0..23` over `i=t−23..t`; `slope = cov(x,y)/max(var(x),1e-18)`; `resid_std = std(y − slope·x − intercept)`; factor = `slope / max(resid_std,1e-12)`; if `resid_std<1e-12` return `0` | 24 | unbounded t-stat-like; raw, self-normalized |
| 8 | `SKEW_50` | moment skew of `ret_{t−49..t}`: `m3 / max(m2^1.5, 1e-18)` where `m2,m3` central moments; if `m2<1e-18` return `0` | 50 | raw, typically `[−5,5]`; no z (standardized by construction) |
| 9 | `KURT_50` | excess kurtosis: `m4 / max(m2²,1e-18) − 3`; if `m2<1e-18` return `0` (declare −3 convention avoided; flat tape → 0) | 50 | raw, `≥−3` typically `[−2,30]`; no z |
| 10 | `MINRET_20` | `mn = min_{t−19..t}(ret)`; `sd = std_50(ret)`; factor = `mn / max(sd,1e-12)`; if `sd<1e-12` return `0`; always `≤0` | 20/50 | raw in vol units, `≤0`; no z |
| 11 | `PARKINSON_20` | `p_i = ln(max(H_i,1e-12)/max(L_i,1e-12))² / (4·ln2)`; `pk = sqrt(max(mean_20(p),0))`; factor = `robust_z_200(pk)` | 20, 200 | z-scored |
| 12 | `ATR_RATIO_100` | `tr_i = max(H_i−L_i, \|H_i−C_{i−1}\|, \|L_i−C_{i−1}\|)`; `a14 = mean_14(tr)`, `a100 = mean_100(tr)`; `r = a14/max(a100,1e-12)`; factor = `robust_z_200(r)`; if `a100<1e-12` return `0` | 14/100, 200 | z-scored |
| 13 | `DD_200` | `pk = max_{t−199..t}(C)`; factor = `(C_t − pk)/max(pk,1e-12)`; if `pk<1e-12` return `0` | 200 | raw in `[−1,0]`; bounded, no z |
| 14 | `DD_DURATION` | `k = argmax_{t−199..t}(C)` (latest max on ties); factor = `(t − k)/200` | 200 | raw in `[0,0.995]`; bounded, no z |
| 15 | `DONCH_POS_100` | `hh = max_{t−99..t}(H)`, `ll = min_{t−99..t}(L)`; factor = `(C_t − ll)/max(hh − ll,1e-12)`; if `hh−ll<1e-12` return `0.5`; clip to `[0,1]` | 100 | raw in `[0,1]`; no z |
| 16 | `BREAKOUT_DIST_50` | `hh = max_{t−50..t−1}(H)` (excludes bar `t`); `atr14 = mean_14(tr)` (as above); `d = (C_t − hh)/max(atr14,1e-12)`; factor = `max(d, 0.0)`; if `atr14<1e-12` return `0` | 50/14 | raw censored `≥0`, usually `[0,5]`; no z (pays only on breakout) |
| 17 | `ROLL_SPREAD` | `c = cov(ret_{t−49..t}, ret_{t−50..t−1})`; factor = `2·sqrt(max(0,−c))`; per-bar returns so spread is in log-return units | 50 (+1 lag) | raw `≥0`; no z (units comparable across coins) |
| 18 | `SHADOW_BAL_12` | `rng_i = max(H_i − L_i, 1e-12)`; `uw_i = (H_i − max(C_i,O_i))/rng_i`, `lw_i = (min(C_i,O_i) − L_i)/rng_i`; factor = `mean_12(uw − lw)` | 12 | raw in `[−1,1]`; no z |
| 19 | `FUND_LAST` | `F_t` raw as defined above | settlement, strictly-before | raw rate (e.g. ±0.001); no z (level is the tradable quantity) |
| 20 | `FUND_X_MOM` | `m = sign(ln(C_t/max(C_{t−48},1e-12)))` (`0` if `C_t==C_{t−48}`); factor = `m · F_t` | 48 + settlement | raw signed rate; no z |
| 21 | `FUND_FLIP_7` | over prior 7 settlements `s_0..s_6` (all `< T_t`, `s_6` latest): `flips = Σ_{j=1..6} 1[sign(F_{s_j}) ≠ sign(F_{s_{j−1}})]` with `sign(0)=0`; if fewer than 7 prior settlements return `0` | 7 settlements | raw integer `0..6`; no z |
| 22 | `VOL_RANK_100` | `rk = #{i in t−99..t : V_i ≤ V_t}` (1..100, ties count); factor = `rk/100` | 100 | raw in `[0.01,1]`; uniform by construction, no z |
| 23 | `TOD_SIN` | `h = hour_utc(T_t) + minute_utc(T_t)/60`; factor = `sin(2πh/24)` | none (timestamp only) | raw in `[−1,1]`; no z |
| 24 | `TOD_COS` | same `h`; factor = `cos(2πh/24)` | none | raw in `[−1,1]`; no z |

Why raw vs z: z-scored are unbounded level/deviation quantities that drift across coins and regimes (`VWAP_DEV_24`, `PARKINSON_20`, `ATR_RATIO_100`). Raw are bounded (`[−1,1]`/`[0,1]`), self-normalized (t-stat, skew/kurt, VR, correlations, censored distances), unit-carrying (Roll, funding levels), or uniform (rank) — z-scoring them would destroy the bound or the tradable level.

## 4. Expected correlation structure of the final 24

No pair is expected to exceed `|r| 0.6` on pooled measurement after the §1 cuts. Pairs the measurement should check (expected 0.3–0.6, survivable — kept deliberately as different lenses on related phenomena):

- `DOLLAR_IMBAL_12` – `OBV_SLOPE_20` (~0.4–0.6): both sign volume by `sign(C−O)`; bounded share vs detrended cumulative slope. Highest-risk new–new pair; keep both only if measured `<0.6`.
- `PARKINSON_20` – existing `VOL_CLUST` (~0.4–0.6): same latent volatility, H/L-path vs close-to-close lens. The H/L lens is the point.
- `MINRET_20` – existing `VOL_CLUST` (~0.3–0.5): large jumps mechanically inflate trailing vol.
- `DD_200` – `DONCH_POS_100` (~0.3–0.5): depth-from-peak vs position-in-range share the anchoring high.
- `BREAKOUT_DIST_50` – `DONCH_POS_100` (~0.3–0.5): fresh-high distance vs range position.
- `EFFICIENCY_12` – `VARIANCE_RATIO_8` (~0.3–0.5): Kaufman path-efficiency vs Lo–MacKinlay variance ratio; different math, same chop-vs-trend phenomenon.
- `SLOPE_TSTAT_24` – `EFFICIENCY_12` (~0.3): trend-per-noise vs path efficiency.
- `FUND_LAST` – `FUND_X_MOM` (~0–0.3): level vs level×momentum-sign; decorrelates to the extent 48-bar momentum sign is balanced.
- `SKEW_50` – `KURT_50` (~0–0.3): asymmetry vs tail weight from the same 50 returns.

Expected ~0 by construction: `TOD_SIN` – `TOD_COS` (sine/cosine orthogonality); TOD pair vs everything pooled within-coin (clock carries no cross-sectional variance — contributes only time-series gating, which is a scope limitation, not a correlation problem); `FUND_FLIP_7` vs price-derived factors (settlement-counted integer vs bar statistics); `ROLL_SPREAD` vs directional factors (bounce magnitude vs signed flow); `AUTOCORR_20` vs level factors (memory sign vs surprise level).

Mutually-exclusive families already enforced (at most one survivor each): range-vol estimator → `PARKINSON_20` only; chop gauge → `EFFICIENCY_12` only; trend-per-noise → `SLOPE_TSTAT_24` only; wick → `SHADOW_BAL_12` only; funding level → `FUND_LAST` only; TOD → indivisible 2-dim group or neither.