# Q4 long-only engine: 5-coin search+verify (BTC/SOL/ETC/TRX/DOGE, 4h, unbiased) — 2026-09-05

- Script: run_q4.py (per-coin seeds 71-75, len-12, 600-601 valid / ~304 tried per coin, long-only aster perp 2x fund 0.0005, lth 0.85/0.88, cd 6). Search+rank on H1 ONLY (OOS first half, 493 bars); top8 verified on H2 (493) + FULL (6570). Output: backtest_Q4_long.json.
- Baseline: grouptest formula long-leg, fresh recompute per coin/segment.

## Per-coin long #1 (H1-best) H1/H2/FULL + pass flag (H2 sh>1.0 AND FULL sh>0.5)
- BTC #1 f=[1, 4, 1, 18, 0, 21, 0, 4, 9, 1, 18, 11, 6, 6, 6, 6, 6] lth=0.85: H1 sh 1.861/ann 0.862/mdd 0.16 | H2 sh -1.889/ann -1.006 | FULL sh -0.651/ann -0.399 -> FAIL. Baseline long: H1/0.85 sh -5.728, H2/0.88 sh 2.869, FULL/0.88 sh -0.448.
- SOL #1 f=[3, 1, 0, 3, 14, 6, 9, 5, 14, 13, 5, 12, 6] lth=0.88: H1 sh 1.663/ann 0.529/mdd 0.078 | H2 sh 2.978/ann 1.826 | FULL sh 0.01/ann 0.007 -> FAIL. Baseline long: H1/0.85 sh -3.05, H2/0.88 sh 0.592, FULL/0.88 sh 0.466.
- ETC #1 f=[1, 4, 12, 22, 1, 2, 11, 0, 0, 8, 22, 20, 6, 6] lth=0.85: H1 sh 1.237/ann 0.175/mdd 0.036 | H2 sh 2.912/ann 0.46 | FULL sh -0.065/ann -0.002 -> FAIL. Baseline long: H1/0.85 sh -1.595, H2/0.88 sh -0.645, FULL/0.88 sh -0.768.
- TRX #1 f=[5, 3, 11, 22, 2, 5, 7, 18, 5, 14, 15, 1, 6, 6, 6] lth=0.88: H1 sh 1.008/ann 0.193/mdd 0.055 | H2 sh -2.911/ann -0.445 | FULL sh -1.651/ann -0.26 -> FAIL. Baseline long: H1/0.85 sh -1.44, H2/0.88 sh -2.856, FULL/0.88 sh -0.107.
- DOGE #1 f=[2, 5, 18, 17, 10, 2, 4, 2, 0, 5, 0, 18, 6, 6, 6, 6, 6, 6, 6] lth=0.85: H1 sh 1.652/ann 1.006/mdd 0.249 | H2 sh -3.619/ann -3.053 | FULL sh -0.785/ann -0.918 -> FAIL. Baseline long: H1/0.85 sh -1.698, H2/0.88 sh -0.129, FULL/0.88 sh -0.124.

## Pass tally
- 0/40 rows pass (0/8 per coin on every coin). No coin has long H2>1.0 AND FULL>0.5.
- Near-misses (H2 holds, FULL fails): SOL #1 (H2 sh 2.978/ann 1.826, FULL sh 0.010/ann 0.007 — H2 spike dies in full sample) and SOL #4 (H2 sh 3.370/ann 1.966, FULL sh -0.212); ETC #1/#2 same formula both thresholds (H2 sh 2.912/ann 0.460, FULL sh -0.065/ann -0.002 — near-flat full sample, threshold-insensitive sanity check).
- Regime flips: BTC #1 (H1 1.861 -> H2 -1.889); TRX top8 ALL negative on H1-ranked set with H2 sh -2.9..-6.0 (H1 search surface itself weak: #1 H1 sh only 1.008, ranks 2-8 H1 negative); DOGE top8 uniform flip (H1 +1.14..+1.65 -> H2 -2.3..-4.0).
- Baseline long-leg negative almost everywhere: BTC H1/0.85 -5.728, FULL/0.88 -0.448 (H2/0.88 +2.869 is the known near-flat spike, to 0.0041); SOL FULL/0.88 +0.466 only positive baseline; ETC/TRX/DOGE baselines negative all segments.

## Pair-test note (kept simple per spec)
- No per-coin long qualifies, so no long-short pair (best-long + ETC/TRX short edge) is constructed. Hypothetical pairs with SOL/ETC H2-spike longs would inherit FULL-sample failure (FULL sh ~0).

- Conclusion: no long-only candidate passes H1->H2->FULL. Long leg unexploitable in this engine across all 5 coins; keep long engine OFF / short-only bias. SOL #1 / ETC #1 H2 spikes are watch-only, not tradable (FULL ~0).

# Q1 baseline: 5-coin both-legs grid + side split (BTC/SOL/ETC/TRX/DOGE, 4h) — 2026-09-05

- Script: run_q1.py. Formula [3,2,7,2,7,11,15,4,4,6,6,10], aster perp 2x, fund 0.0005, fee 0.0004 (aster default).
- Each coin 6570 4h bars, 85% cut (5584 train), OOS 986 -> H1/H2 493/493. Grid lth[0.85,0.88] x sth[0.10,0.12,0.15] x cd[3,6,12] x sl[None,0.03,0.05] = 54 rows. H1-best by H1 sharpe, verify H2 + FULL. Output: backtest_Q1_5coin.json.

## Per-coin H1-best params + H1/H2/FULL (both/long/short)
- BTC best lth=0.85 sth=0.12 cd=12 sl=0.05: H1 both sh 2.922/ann 1.679/mdd 0.192 | long -2.715 | short 4.171 || H2 both sh -0.329/ann -0.210/mdd 0.442 | long 0.000 | short -0.329 || FULL both sh 0.913/ann 0.676/mdd 1.032 | long -0.159 | short 1.017
- SOL best lth=0.85 sth=0.1 cd=3 sl=None: H1 both sh 2.481/ann 2.113/mdd 0.358 | long -3.584 | short 4.037 || H2 both sh -0.869/ann -0.885/mdd 0.635 | long 1.045 | short -0.955 || FULL both sh 0.320/ann 0.471/mdd 3.209 | long 0.261 | short 0.251
- ETC best lth=0.85 sth=0.15 cd=12 sl=None: H1 both sh 4.702/ann 3.833/mdd 0.293 | long 0.423 | short 5.015 || H2 both sh 0.391/ann 0.426/mdd 0.743 | long 2.730 | short -0.012 || FULL both sh 1.878/ann 2.417/mdd 1.088 | long 0.560 | short 1.793
- TRX best lth=0.85 sth=0.12 cd=6 sl=None: H1 both sh 3.107/ann 1.214/mdd 0.103 | long 0.742 | short 3.029 || H2 both sh 1.635/ann 0.458/mdd 0.093 | long -2.790 | short 2.866 || FULL both sh 0.668/ann 0.526/mdd 2.123 | long -0.140 | short 0.745
- DOGE best lth=0.85 sth=0.15 cd=12 sl=None: H1 both sh 2.573/ann 2.217/mdd 0.250 | long -0.033 | short 2.848 || H2 both sh 0.507/ann 0.464/mdd 0.593 | long 0.494 | short 0.412 || FULL both sh 0.248/ann 0.381/mdd 3.390 | long 0.162 | short 0.208

## Combos (H1-best params, equal-weight) H1/H2/FULL sharpe/ann/mdd + side split
- ALL5 ['BTC', 'SOL', 'ETC', 'TRX', 'DOGE']: H1 both 4.665/2.211/0.125 long -1.559 short 5.460 || H2 both 0.084/0.051/0.425 long 1.117 short -0.025 || FULL both 1.092/0.894/1.233 long 0.288 short 1.019
- MAJORS ['BTC', 'SOL', 'DOGE']: H1 both 3.321/2.003/0.194 long -2.735 short 4.413 || H2 both -0.294/-0.210/0.517 long 0.817 short -0.373 || FULL both 0.517/0.509/1.864 long 0.194 short 0.469
- EDGE ['ETC', 'TRX']: H1 both 5.624/2.523/0.186 long 0.618 short 5.835 || H2 both 0.756/0.442/0.397 long 0.792 short 0.633 || FULL both 1.781/1.472/1.321 long 0.343 short 1.740
- ALL5exETC ['BTC', 'SOL', 'TRX', 'DOGE']: H1 both 3.849/1.806/0.160 long -2.377 short 4.902 || H2 both -0.078/-0.043/0.397 long -0.449 short -0.028 || FULL both 0.635/0.513/1.629 long 0.138 short 0.601

## H2 pairwise corr (both-leg)
- ALL5: {'BTC_SOL': 0.653, 'BTC_ETC': 0.428, 'BTC_TRX': 0.225, 'BTC_DOGE': 0.475, 'SOL_ETC': 0.48, 'SOL_TRX': 0.183, 'SOL_DOGE': 0.495, 'ETC_TRX': 0.174, 'ETC_DOGE': 0.56, 'TRX_DOGE': 0.145}
- MAJORS: {'BTC_SOL': 0.653, 'BTC_DOGE': 0.475, 'SOL_DOGE': 0.495}
- EDGE: {'ETC_TRX': 0.174}
- ALL5exETC: {'BTC_SOL': 0.653, 'BTC_TRX': 0.225, 'BTC_DOGE': 0.475, 'SOL_TRX': 0.183, 'SOL_DOGE': 0.495, 'TRX_DOGE': 0.145}

## Read: H2 both-leg holds only on TRX (1.635, short-driven 2.866) + DOGE (0.507); BTC (-0.329, long flat 0.0) and SOL (-0.869) fail H2. Best combo H2 is EDGE(ETC+TRX) both 0.756 (corr 0.174, most diversified pair). ALL5 H2 both 0.084.

# STRATEGY_Q5 — Q2: fresh BOTH-legged formula on BTC/SOL/ETC/TRX/DOGE

Date: 2026-09-05. Script: `run_q2.py` (train) + `run_q2_verify.py` (OOS verify).
Artifacts: `train_q5_best.json`, `backtest_Q2_train.json`, logs `q2_run.log`, `q2_verify.log`.

## Method
- Train: RL 200 steps x batch 64 on front-85% 4h bars (5584 bars), fitness = MEAN
  over 5 coins. Backtest: BOTH legs short_enabled=True, aster perp 2x,
  funding_override 0.0005, lth 0.85 / sth 0.15 / cd 6, no stop-loss.
- OOS: each top3 candidate on all 5 coins LAST-15% (986 bars: FULL + H1/H2 493/493),
  with default thresholds AND Q1 per-coin H1-best params
  (BTC 0.85/0.12/cd12/sl0.05, SOL 0.85/0.10/cd3, DOGE 0.85/0.15/cd12;
  ETC/TRX fall back to default). H2 side split both/long/short.
- Baseline: BSD-1 formula [3,2,7,2,7,11,15,4,4,6,6,10] on same segments.

## Train top3 (by mean-5-coin fitness)
| rank | train_fit | formula | decoded |
| 0 | 0.0706 | [0,20,18,14,20,12,14,18,11,14,14,14] | RET TS_RANK ZSCORE JUMP TS_RANK SIGN JUMP ZSCORE ABS JUMP JUMP JUMP |
| 1 | 0.0094 | [5,21,21,14,20,14,14,14,14,14,14,14] | LOG_VOL DELTA DELTA JUMP TS_RANK JUMP x7 |
| 2 | 0.0094 | [5,21,21,21,14,14,14,14,14,14,14,10] | LOG_VOL DELTA x3 JUMP x6 NEG |

## OOS H2 (default thresholds) — cand0 vs baseline per coin
| coin | cand0 fit/sh/ann/mdd | baseline fit/sh/ann/mdd |
| BTC | 0.1197 / 3.169 / 0.5316 / 0.0038 | -0.6154 / -0.747 / -0.5124 / 0.4523 |
| SOL | 0.0442 / 1.241 / 0.1964 / 0.0322 | -1.7683 / -1.163 / -1.1919 / 0.7109 |
| ETC | 0.1416 / 3.111 / 0.6292 / 0.0350 | -0.1102 / 1.626 / 1.7314 / 0.5590 |
| TRX | -0.0026 / -0.450 / -0.0114 / 0.0086 | 0.1080 / 1.712 / 0.4800 / 0.0928 |
| DOGE | 0.0401 / 0.438 / 0.1780 / 0.1402 | -0.1329 / 1.657 / 1.6306 / 0.5850 |
| AVG(5) | fit +0.0686, sharpe +1.502 | fit -0.5038, sharpe +0.617 |

H2 averages (default): cand0 +0.0686 fit / 1.502 sharpe; cand1 -2.0310 / -2.150;
cand2 -5.9933 / 0.617; baseline -0.5038 / 0.617.
H2 averages (Q1 params where avail): cand0 fit -1.8734 (SOL-Q1 inactivity drags the
mean; BTC-Q1 H2 fit 0.2236/sh 4.198, DOGE-Q1 H2 fit 0.2705/sh 2.526);
baseline -0.6873 / 0.273.

## OOS H2 side split (default thresholds, cand0 — H2 P&L is 100% long leg)
| coin | both fit | long fit | short fit |
| BTC | 0.1197 | 0.1197 | inactive (-10) |
| SOL | 0.0442 | 0.0442 | inactive (-10) |
| ETC | 0.1416 | 0.1416 | inactive (-10) |
| TRX | -0.0026 | -0.0026 | inactive (-10) |
| DOGE | 0.0401 | 0.0401 | inactive (-10) |
Baseline H2 side split differs: baseline short leg active on BTC/SOL/ETC/TRX
(negative on BTC/SOL, positive on ETC/TRX), long leg near-flat.
Note: cand0 H2 triggers only the long leg (short never fires); TRX H2 is
near-flat (-0.0026). Cand2 H2 is the only short-leg signal (ETC +0.0233,
TRX +0.0101) but is inactive elsewhere.

## Q1-params view (per-coin H1-best, H2)
- cand0/BTC-Q1: fit 0.2236, sh 4.198, ann 0.9935, mdd 0.0185 (best single cell).
- cand0/DOGE-Q1: fit 0.2705, sh 2.526, ann 1.2017, mdd 0.1116.
- cand0/SOL-Q1: inactive on fitness (cum +0.0223, sh 0.64) — cd3/sth0.10
  suppresses all H2 trades for this formula.
- ETC/TRX have no Q1 entry, use default.

## Verdict
- WIN on H2-average vs baseline under default thresholds (+0.0686 vs -0.5038 fit,
  1.502 vs 0.617 sharpe), driven by BTC/ETC.
- WEAKNESSES: (1) long-leg only — no short diversification; (2) TRX ~flat;
  (3) Q1-param fragility on SOL (inactivity); (4) absolute H2 edge is small
  (avg cum +0.069 over 493 bars). Cand1/cand2 are not viable (negative/inactive).
- Keep cand0 as Q5 candidate #1 under default thresholds; do NOT adopt Q1 params
  blindly (SOL breaks). Next: ensemble or short-leg complement.
