# STRATEGY_ITER_N16_25 (N1+N2)

Baseline: FORMULA [3,2,7,2,7,11,15,4,4,6,6,10], aster perp 2x fund 0.0005. BEST ETC (0.88,0.12,cd12,None) TRX (0.85,0.15,cd6,0.05). Equal-weight ETC+TRX. OOS = last 15% of 4h bars, H1 = first half OOS (n=493), H2 = second half OOS (n=493), FULL n=6570.

## N1: cooldown/threshold micro-grid (81 rows, unbiased H1-select/H2-verify)

Top3 by H1 sharpe:

| rank | ETC | TRX | H1 sharpe/ann/mdd | H2 sharpe/ann/mdd | FULL sharpe/ann/mdd |
|---|---|---|---|---|---|
| 1 (H1-best) | 0.85,0.15,cd12,SL None | 0.88,0.12,cd12,SL 0.05 | 5.694 / 2.448 / 0.171 | 0.708 / 0.412 / 0.397 | 2.126 / 1.676 / 0.673 |
| 2 | 0.85,0.15,cd12,SL None | 0.80,0.20,cd12,SL 0.05 | 5.582 / 2.363 / 0.171 | 1.227 / 0.719 / 0.397 | 2.254 / 1.794 / 0.656 |
| 3 | 0.85,0.15,cd12,SL None | 0.88,0.12,cd6,SL 0.05 | 5.574 / 2.500 / 0.191 | 0.790 / 0.463 / 0.397 | 2.167 / 1.737 / 0.644 |

N1 H1-best verdict: H1-best FAILS H2 verify (H2 sharpe 0.708, worst of top3). ETC (0.85,0.15) overfits H1 across the grid — every ETC 0.85/0.15+cd12 row has H2 < 1.3 while FULL looks strong (>2.1). Do NOT adopt H1-best. The (0.88,0.12,cd12 / 0.80,0.20,cd12) row verifies better (H2 1.227) but was rank 2, and H2-cherry-picking is disallowed — keep baseline.

Robust pocket (high H2, not H1-selected): ETC cd18 + TRX cd12 rows reach H2 3.5-4.6 (e.g. ETC 0.90,0.10,cd18 / TRX 0.80,0.20,cd12: H1 3.775 H2 4.634 FULL 1.906). Flag for future pre-registered test only.

## N2: time-stop sweep on baseline (H1/H2/FULL)

| time_stop (bars) | H1 sharpe/ann/mdd | H2 sharpe/ann/mdd | FULL sharpe/ann/mdd |
|---|---|---|---|
| 0 (disabled) | 5.130 / 2.318 / 0.190 | 2.197 / 1.181 / 0.191 | 2.336 / 1.872 / 0.656 |
| 6 | 4.106 / 1.713 / 0.167 | 1.080 / 0.537 / 0.197 | 1.463 / 1.054 / 0.853 |
| 12 | 4.614 / 2.006 / 0.151 | 2.120 / 1.083 / 0.184 | 1.958 / 1.488 / 0.725 |
| 24 | 5.232 / 2.322 / 0.147 | 2.309 / 1.208 / 0.182 | 2.370 / 1.864 / 0.662 |
| 48 (H1-best) | 5.567 / 2.470 / 0.155 | 2.204 / 1.185 / 0.191 | 2.340 / 1.875 / 0.673 |

N2 H1-best: ts=48 (H1 5.567, H2 2.204, FULL 2.340). ts=24 verifies marginally better on H2 (2.309) and FULL (2.370) but H1-second — no cherry-pick; highlight only. Short stops (ts=6) hurt everywhere. ts=24/48 both preserve H2 ~2.2-2.3 vs 2.197 at ts=0. Effect is small; keep ts=0 (disabled) as locked default unless a follow-up pre-registers ts=24/48.

## Summary

- N1: no adoption. H1-best overfits; keep baseline thresholds/cooldowns.
- N2: no adoption. H1-best ts=48 ≈ baseline on H2/FULL; keep time_stop=0.
- Full JSON rows: backtest_iterN1N2.json (n1=81, n2=5). Script: run_iterN1N2.py.
## N5 — TRX signal-threshold asymmetry (ETC fixed 0.88/0.12/cd12/None; combo = equal-weight ETC+TRX; aster perp 2x fund 0.0005; H1 select / H2 verify + FULL)

### N5a — TRX short_th sweep, long_th=0.85 fixed (cd6, sl 0.05)
| TRX short_th | H1 sharpe / ann / mdd | H2 sharpe / ann / mdd | FULL sharpe / ann / mdd |
|---|---|---|---|
| 0.08 | 4.801 / 2.153 / 0.190 | 1.897 / 1.015 / 0.191 | 2.108 / 1.676 / 0.692 |
| 0.10 | 5.045 / 2.292 / 0.190 | 2.033 / 1.089 / 0.191 | 2.200 / 1.753 / 0.659 |
| 0.15 (base) | 5.130 / 2.318 / 0.190 | 2.197 / 1.181 / 0.191 | 2.336 / 1.872 / 0.656 |
| 0.20 | 5.209 / 2.363 / 0.194 | 2.286 / 1.236 / 0.191 | 2.373 / 1.907 / 0.653 |
| 0.25 | 5.245 / 2.384 / 0.194 | 2.203 / 1.195 / 0.191 | 2.454 / 1.975 / 0.662 |
H1-best short_th = 0.25 (monotone gain H1 and FULL; H2 peaks at 0.20 with 2.286 then dips to 2.203 at 0.25).

### N5b — TRX long_th sweep, short_th=0.25 fixed (H1-best)
| TRX long_th | H1 sharpe / ann / mdd | H2 sharpe / ann / mdd | FULL sharpe / ann / mdd |
|---|---|---|---|
| 0.80 | 4.605 / 2.095 / 0.194 | 2.371 / 1.286 / 0.191 | 2.364 / 1.912 / 0.662 |
| 0.85 | 5.245 / 2.384 / 0.194 | 2.203 / 1.195 / 0.191 | 2.454 / 1.975 / 0.662 |
| 0.88 | 5.245 / 2.384 / 0.194 | 2.251 / 1.221 / 0.191 | 2.475 / 1.993 / 0.651 |
| 0.92 | 5.144 / 2.294 / 0.194 | 2.194 / 1.191 / 0.191 | 2.488 / 2.004 / 0.651 |
H1-best long_th = 0.85 (tied 0.88 on H1); H2-best = 0.80 (2.371); FULL-best = 0.92 (2.488). No clean H1->H2 confirmation for raising long_th; 0.85 base holds.

## N6 — short-only vs both-legs vs long-only (baseline params; aster perp 2x fund 0.0005)
| item | seg | both sharpe/ann/mdd | short sharpe/ann/mdd | long sharpe/ann/mdd |
|---|---|---|---|---|
| ETC | H1 | 4.386 / 3.600 / 0.288 | 4.658 / 3.451 / 0.288 | 0.423 / 0.150 / 0.119 |
| ETC | H2 | 1.842 / 1.831 / 0.332 | 1.842 / 1.831 / 0.332 | 0.000 / 0.000 / 0.000 |
| ETC | FULL | 2.114 / 2.725 / 1.079 | 1.894 / 2.286 / 1.079 | 0.967 / 0.439 / 0.410 |
| TRX | H1 | 2.590 / 1.035 / 0.131 | 2.582 / 0.992 / 0.131 | 0.388 / 0.043 / 0.055 |
| TRX | H2 | 1.864 / 0.530 / 0.074 | 2.946 / 0.781 / 0.074 | -2.477 / -0.251 / 0.075 |
| TRX | FULL | 1.500 / 1.020 / 0.537 | 1.724 / 1.102 / 0.517 | -0.356 / -0.082 / 0.462 |
| ETC+TRX eq | H1 | 5.130 / 2.318 / 0.190 | 5.337 / 2.221 / 0.190 | 0.493 / 0.097 / 0.087 |
| ETC+TRX eq | H2 | 2.197 / 1.181 / 0.191 | 2.442 / 1.306 / 0.191 | -2.477 / -0.126 / 0.037 |
| ETC+TRX eq | FULL | 2.336 / 1.872 / 0.656 | 2.227 / 1.694 / 0.662 | 0.628 / 0.178 / 0.285 |
Trade counts (bars/entries): ETC H1 both LB=13 SB=321 LE=1 SE=14; H2 both LB=0 SB=312 LE=0 SE=14; FULL both LB=235 SB=4142 LE=18 SE=168. TRX H1 both LB=24 SB=332 LE=3 SE=20; H2 both LB=33 SB=328 LE=4 SE=15; FULL both LB=269 SB=4662 LE=38 SE=255. Combo FULL both LB=504 SB=8804 LE=56 SE=423 (short-only zeroes long leg).
Takeaway: edge is short-driven. Short-only beats both on H1 and H2 for the combo (H1 5.337 vs 5.130; H2 2.442 vs 2.197); long leg adds nothing OOS (H2 long-only 0 / negative) but helps FULL via early-sample longs. Keep both-legs default; short-only is a valid risk-off variant, not a replacement.

## N3: stop-loss / take-profit sweep, ETC+TRX equal-weight, lev 2x (H1 select / H2 verify + FULL)
- Grid: SL in [None,0.03,0.05,0.08] x TP in [None,0.08,0.15], applied to both legs via MemeBacktest(stop_loss, take_profit). Baseline FORMULA [3,2,7,2,7,11,15,4,4,6,6,10], aster perp 2x fund 0.0005, ETC (0.88,0.12,cd12,None) TRX (0.85,0.15,cd6,0.05).
- Full table (H1 sharpe/ann/mdd/to | H2 sharpe/ann/mdd/to | FULL sharpe/ann/mdd):
| SL | TP | H1 | H2 | FULL |
|---|---|---|---|---|
| None | None | 5.180 / 2.341 / 0.185 / 164 | 2.197 / 1.181 / 0.191 / 147 | 2.025 / 1.674 / 1.323 |
| None | 0.08 | 5.115 / 2.311 / 0.185 / 169 | 2.197 / 1.181 / 0.191 / 147 | 1.905 / 1.558 / 1.323 |
| None | 0.15 | 5.180 / 2.341 / 0.185 / 164 | 2.197 / 1.181 / 0.191 / 147 | 1.913 / 1.567 / 1.323 |
| 0.03 | None | 4.845 / 2.118 / 0.170 / 222 | 3.379 / 1.398 / 0.084 / 187 | 1.906 / 1.464 / 1.053 |
| 0.03 | 0.08 | 5.044 / 2.178 / 0.170 / 235 | 3.475 / 1.438 / 0.084 / 187 | 1.846 / 1.377 / 0.998 |
| 0.03 | 0.15 | 4.845 / 2.118 / 0.170 / 222 | 3.379 / 1.398 / 0.084 / 187 | 1.846 / 1.398 / 1.053 |
| 0.05 | None | 4.454 / 1.953 / 0.190 / 191 | 1.623 / 0.831 / 0.199 / 169 | 2.137 / 1.668 / 0.756 |
| 0.05 | 0.08 | 4.692 / 2.033 / 0.190 / 200 | 1.701 / 0.871 / 0.197 / 169 | 2.071 / 1.580 / 0.739 |
| 0.05 | 0.15 | 4.454 / 1.953 / 0.190 / 191 | 1.623 / 0.831 / 0.199 / 169 | 2.035 / 1.566 / 0.756 |
| 0.08 | None | 5.252 / 2.371 / 0.173 / 173 | 1.579 / 0.809 / 0.199 / 151 | 2.100 / 1.680 / 0.820 |
| 0.08 | 0.08 | 5.187 / 2.341 / 0.173 / 178 | 1.656 / 0.848 / 0.197 / 151 | 1.981 / 1.557 / 0.818 |
| 0.08 | 0.15 | 5.252 / 2.371 / 0.173 / 173 | 1.579 / 0.809 / 0.199 / 151 | 2.026 / 1.601 / 0.820 |
- Top3 by H1: (1) SL=0.08 TP=None H1 5.252 H2 1.579 FULL 2.100; (2) SL=0.08 TP=0.15 H1 5.252 H2 1.579 FULL 2.026; (3) SL=0.08 TP=0.08 H1 5.187 H2 1.656 FULL 1.981.
- Best by H2 (not selected, reported): SL=0.03 TP=0.08 H2 3.475 / ann 1.438 / dd 0.084 (H1 5.044, FULL 1.846, to 235/187). Tight 3% stop halves H2 drawdown (0.191 -> 0.084) at cost of higher turnover (~187 vs ~147/yr).
- Selection note: H1-select keeps SL=0.08 TP=None, but H2 favors SL=0.03 TP=0.08. H1/H2 disagree -> do not lock stops yet; carry both configs forward or keep baseline None/None (H2 2.197, FULL dd 1.323 worst). JSON: backtest_iterN3N4.json N3.

## N4: leverage sweep on H1-best N3 config (SL=0.08 TP=None, ETC+TRX eq)
- Lev in [1.0,1.5,2.0,2.5,3.0]. Turnover invariant (~173/151/167 per yr).
| lev | H1 sharpe/ann/mdd | H2 sharpe/ann/mdd | FULL sharpe/ann/mdd |
|---|---|---|---|
| 1.0 | 5.252 / 1.185 / 0.086 | 1.579 / 0.404 / 0.100 | 2.100 / 0.840 / 0.410 |
| 1.5 | 5.252 / 1.778 / 0.129 | 1.579 / 0.606 / 0.149 | 2.100 / 1.260 / 0.615 |
| 2.0 | 5.252 / 2.371 / 0.173 | 1.579 / 0.809 / 0.199 | 2.100 / 1.680 / 0.820 |
| 2.5 | 5.252 / 2.964 / 0.216 | 1.579 / 1.011 / 0.249 | 2.100 / 2.100 / 1.025 |
| 3.0 | 5.252 / 3.556 / 0.259 | 1.579 / 1.213 / 0.299 | 2.100 / 2.520 / 1.229 |
- Note: sharpe is leverage-invariant as expected (gross of costs scales linearly); ann and dd scale ~linearly. lev 2.0 keeps FULL dd 0.82 vs 1.23 at 3.0. Keep 2.0x default; 1.5x is the risk-off variant (FULL dd 0.61, H2 dd 0.15). JSON: backtest_iterN3N4.json N4.

## N9: cross-coin OOS guard — AVAX/SHIB/DOGE/SOL vs ETC+TRX pair (2026-09-05)

- Scope: single-leg baseline (FORMULA [3,2,7,2,7,11,15,4,4,6,6,10], aster perp 2x, fund 0.0005), per-coin BEST params, unbiased H1 (OOS half 1, n=493) / H2 (OOS half 2, n=493) / FULL (n=6570).
- Params: AVAX (0.85,0.15,cd12,sl0.03), SHIB (0.85,0.15,cd6,None), DOGE (0.85,0.15,cd6,sl0.05), SOL (0.9,0.1,cd3,sl0.03).

| coin | H1 sharpe | H2 sharpe | FULL sharpe |
|---|---|---|---|
| AVAX | 1.679 | 0.726 | 0.578 |
| SHIB | 2.014 | 0.867 | 0.962 |
| DOGE | -0.469 | 0.154 | 0.480 |
| SOL | 1.180 | 0.034 | 0.214 |
| ETC (ref) | 4.386 | 1.842 | 2.114 |
| TRX (ref) | 2.590 | 1.864 | 1.500 |
| **ETC+TRX 50/50** | **5.130** | **2.197** | **2.336** |

- Guard: best watchlist H2 = SHIB 0.867 < pair H2 2.197. **PASS — no single watchlist coin beats the pair on H2; pair is not cherry-picked over these four.**
- Note: H1 select / H2 verify: pair H1 5.13 -> H2 2.20 (decay but holds >2.0 target on unseen half).

## N10: FINAL lock-in — ETC+TRX 50/50 equal-weight release-candidate audit (2026-09-05)

- Config: no tweaks, baseline params per leg (ETC 0.88/0.12/cd12/None, TRX 0.85/0.15/cd6/sl0.05), aster perp 2x, fund 0.0005, default fee 0.0004.
- FULL (6570 bars 4h): sharpe **2.336**, ann **187.2%**, mdd 0.656, cum 5.617.
- 6-fold walk-forward (1095 bars/fold, sequential):

| fold | sharpe | ann | mdd |
|---|---|---|---|
| 0 (oldest) | -0.808 | -66.3% | 0.650 |
| 1 | 2.391 | 187.5% | 0.293 |
| 2 | 2.401 | 246.4% | 0.656 |
| 3 | 0.744 | 60.6% | 0.485 |
| 4 | 6.664 | 509.4% | 0.122 |
| 5 (newest) | 3.655 | 185.7% | 0.191 |

- 5/6 folds positive; fold0 (earliest ~6 months) negative — oldest-regime caveat, recent folds (4,5) strongest.
- Fee sweep (fund fixed 0.0005):

| fee | H2 sharpe | FULL sharpe | FULL ann |
|---|---|---|---|
| 0.0004 (baseline) | 2.197 | 2.336 | 187.2% |
| 0.0008 (2x stress) | 1.976 | 2.176 | 174.5% |

- Robust to 2x fee: sharpe stays >1.9 on H2, >2.1 FULL.
- Turnover estimate (mean |dPos|/bar, x2190): ETC 124.0/yr, TRX 195.3/yr, pair avg **~160 position-flips/yr per leg** (~1 flip per ~14 bars ≈ 2.3 days). Low-frequency 4h swing — fee-insensitive, consistent with fee sweep.
- Release verdict: **ETC+TRX 50/50 qualifies as release candidate** (H2 2.20 > 2.0, FULL 2.34, 5/6 folds +, fee-robust). Known risk: fold0 negative = earliest regime differs; recommend paper-trade gate before sizing.
- Artifacts: backtest_iterN9N10.json (N9 singles+pair, N10 folds+fees+turnover), run_iterN9N10.py (repro, reuses run_grouptest678910.py pattern).

## N7: holding-cap / rotation — max one leg at a time (stronger |sigmoid-0.5| wins) vs equal-weight both-legs

Method: per-leg lp/sp built with baseline params (ETC 0.88/0.12/cd12/SL None; TRX 0.85/0.15/cd6/SL 0.05), then each bar picks the leg with larger |sigmoid-0.5|, rolled +1 for execution lag. Turnover = mean |dpos|/bar; x2190 for per-yr.

| seg | EQ sharpe/ann/mdd/turnover-perbar | ROT sharpe/ann/mdd/turnover-perbar | ROT shareETC / both-active / both-flat |
|---|---|---|---|
| H1 (n=493) | 5.130 / 2.318 / 0.190 / 0.0771 (168.8/yr) | 8.425 / 5.604 / 0.122 / 0.1258 (275.4/yr) | 0.49 / 0.52 / 0.12 |
| H2 (n=493) | 2.197 / 1.181 / 0.191 / 0.0669 (146.6/yr) | 1.782 / 1.597 / 0.283 / 0.1542 (337.6/yr) | 0.57 / 0.49 / 0.13 |
| FULL (n=6570) | 2.336 / 1.872 / 0.656 / 0.0729 (159.7/yr) | 1.696 / 1.928 / 1.258 / 0.1279 (280.0/yr) | 0.49 / 0.52 / 0.11 |

N7 verdict: rotation WINS H1 big (sharpe 8.4 vs 5.1) but FAILS H2 verify (1.78 < 2.20) and FULL (1.70 < 2.34, dd 1.26 vs 0.66). Classic H1 overfit: both-active ~50% of bars in every segment, so rotation is not selecting a clean winner — it halves exposure ~half the time and churns (turnover ~1.7-2.3x EQ) for a small ann gain (+0.06 FULL) at ~2x drawdown. Do NOT adopt rotation; keep equal-weight both-legs.

## N8: vol-target sweep on ETC+TRX equal-weight (vol_window=24, per-leg scaling)

Note: prior vol-target failed on BTC single-leg (run_vol.py: vt in [None,0.005,0.01,0.02] x lev — recorded as failed). Test if it helps the diversified pair.

| vol_target | H1 sharpe/ann/mdd/to-perbar | H2 sharpe/ann/mdd/to-perbar | FULL sharpe/ann/mdd/to-perbar |
|---|---|---|---|
| None | 5.130 / 2.318 / 0.190 / 0.0771 (168.8/yr) | 2.197 / 1.181 / 0.191 / 0.0669 (146.6/yr) | 2.336 / 1.872 / 0.656 / 0.0729 (159.7/yr) |
| 0.02 | 5.139 / 4.167 / 0.380 / 0.1527 (334.5/yr) | 2.508 / 2.125 / 0.288 / 0.1315 (287.9/yr) | 2.280 / 2.607 / 0.878 / 0.1380 (302.1/yr) |
| 0.05 | 5.202 / 4.693 / 0.380 / 0.1542 (337.6/yr) | 2.357 / 2.464 / 0.383 / 0.1340 (293.5/yr) | 2.265 / 3.481 / 1.304 / 0.1454 (318.5/yr) |
| 0.10 | 5.202 / 4.693 / 0.380 / 0.1542 (337.6/yr) | 2.334 / 2.494 / 0.383 / 0.1339 (293.2/yr) | 2.322 / 3.707 / 1.313 / 0.1457 (319.0/yr) |

N8 verdict: MIXED — unlike BTC single-leg, vol-target helps H2 modestly (vt=0.02: H2 sharpe 2.51 vs 2.20, ann 2.13 vs 1.18) but at ~2x turnover and ~1.5x H2 dd. FULL sharpe is flat-to-down (2.28/2.27/2.32 vs 2.34) while FULL dd roughly doubles at vt>=0.05 (1.30 vs 0.66). vt=0.02 is the only setting that improves H2 without hurting FULL much (FULL 2.28 vs 2.34, dd 0.88 vs 0.66), but scale saturates (0.05/0.10 clamp at 2.0x, identical H1). No H1-select/H2-verify winner pre-registered; do NOT adopt as default. vt=0.02 is a flagged candidate for a pre-registered follow-up only. JSON: backtest_iterN7N8.json N8 (turnover per-bar + per-yr in table).

## Summary N7/N8

- N7: no adoption. Rotation overfits H1, fails H2/FULL, doubles dd. Keep equal-weight.
- N8: no adoption as default. vt=0.02 improves H2 (2.51 vs 2.20) with flat FULL (2.28 vs 2.34); vol-target is not dead on the pair, but cost is 2x turnover + higher dd. Flag vt=0.02 only.
- Artifacts: backtest_iterN7N8.json (N7 eq+rot with turnover/diagnostics, N8 sweep), run_iterN7N8.py (repro, reuses run_grouptest678910.py pattern).

## FINAL — 10輪 (N1~N10) 總結論 (2026-09-05)

| 輪 | 題目 | H1-best | H2驗證 | 採用? |
|---|---|---|---|---|
| N1 | 冷卻/閾值81格 | 5.694 | 0.708 | 否,維持基線 |
| N2 | 時間止損 | ts48 H1 5.567/H2 2.204 | ts24 H2 2.309次佳 | 否,ts=0 |
| N3 | SL/TP | SL0.08 H1 5.252 | H2 1.579;H2-best SL0.03 3.475分歧 | 否,不鎖止損 |
| N4 | 槓桿 | sharpe不變 | lev2x H2 ann0.809/dd0.199 | 維持2x,1.5x降風險備選 |
| N5 | TRX閾值不對稱 | short0.25 H1 5.245 | H2 2.20;FULL 2.45 | 候選TRX 0.85/0.25,未鎖定 |
| N6 | 多空確認 | both H1 5.13 | short H2 2.44>both 2.20 | 維持雙向,short-only作risk-off |
| N7 | 輪動單腿 | ROT H1 8.43 | H2 1.78<2.20,FULL dd翻倍 | 否 |
| N8 | vol-target | vt0.02 H1 5.14 | H2 2.51>FULL持平 | 否,vt0.02僅候選 |
| N9 | 跨幣檢查 | pair H1 5.13 | pair H2 2.197>SHIB 0.867 | 通過 |
| N10 | 最終鎖定 | FULL 2.336/ann187%/dd0.656 | 5/6折正,fee2x仍>1.9 | release candidate |

鎖定: ETC+TRX 50/50等權基線不動 (ETC 0.88/0.12/cd12/None, TRX 0.85/0.15/cd6/sl0.05, aster perp2x fund0.0005 fee0.0004). H2 2.20>2.0, FULL 2.34, 5/6折正, 費用穩健. 候選僅觀察: TRX 0.85/0.25, vt0.02, short-only risk-off, SL0.03/TP0.08分歧未解. 下一步: paper-trade gate.
