# STRATEGY E13 — Fee Hardening on AA-H1 (FORMULA locked)

## Objective
Push AA-H1 worst-B stress cell (fund 0.0003 x fee 0.0008 = 0.354, borderline)
above 1.0 without breaking the median.
Guards: cooldown +2, vt 0.012 -> 0.010 / 0.015, ts 24 -> 12 / 36,
q 0.3 -> 0.25 / 0.35. AA-H1 (0.10/15/9) only. FORMULA locked.

## Method (mirror)
- Fee curve mirrors `research/run_e3.py`: FULL fee scan
  [0.0002, 0.0004, 0.0008, 0.0012] + 12-fold median sharpe.
- Stress grid mirrors `research/run_aa.py`: fund [0.0003, 0.0005]
  x fee [0.0004, 0.0008] 4-cell, B/C worst sharpe.
- Engine: quantile_mask_long abs top-q long-only, vol_scale clamp 0.2-2
  roll-1 post-stops pre-roll, cooldown, stops, time_stop, tx fee+liquidity,
  funding, lev 2.0. Coins ETC/TRX, weights 0.5/0.5, venue aster.
- Repro: `.venv2/bin/python research/run_e13.py`
  -> `results/results_E13_fee.json`, log `logs/e13.log`.

## Constraint gates
- turnover (FULL) < 0.16, FULL sharpe > 2.0, H2 sharpe > 3.5.

## Base (AA-H1, no guard) — reproduced
- FULL sh 2.025 / to 0.142399 / H2 5.581 / H1 4.604 / med12 1.964.
- Stress: worstB 0.354 (fund 0.0003 x fee 0.0008), worstC 3.465.
- Fee curve FULL: 0.0002 -> 2.180, 0.0004 -> 2.025, 0.0008 -> 1.713, 0.0012 -> 1.402.

## Results (8 variants, ranked by gate-pass then worstB)

| variant | worstB | worstC | FULL | H2 | turnover | med12 | gate |
|---|---|---|---|---|---|---|---|
| cd+2 (etc15->17, trx9->11) | 3.065 | 4.573 | 2.010 | 6.612 | 0.136258 | 2.376 | PASS |
| ts36 | 0.954 | 4.202 | 2.076 | 5.924 | 0.128856 | 1.946 | PASS |
| vt0.010 | 0.373 | 3.312 | 2.011 | 5.537 | 0.132874 | 1.929 | PASS |
| base | 0.354 | 3.465 | 2.025 | 5.581 | 0.142399 | 1.964 | PASS |
| vt0.015 | 0.001 | 3.257 | 2.026 | 5.257 | 0.151157 | 2.033 | PASS |
| q0.35 | 0.623 | 3.465 | 1.964 | 5.581 | 0.142421 | 1.681 | FAIL (FULL<=2.0) |
| q0.25 | 0.253 | 3.465 | 1.991 | 5.581 | 0.142349 | 2.018 | FAIL (FULL<=2.0) |
| ts12 | -0.557 | 2.575 | 1.945 | 5.100 | 0.211197 | 2.155 | FAIL (turnover, FULL) |

Median check (no median break): base med12 1.964 -> cd+2 med12 2.376 (up).
FULL 2.025 -> 2.010 (flat). H2 5.581 -> 6.612 (up). Turnover 0.142 -> 0.136 (down).

## Best guard: cd+2
- Params: sth 0.10, etc_cd 17, trx_cd 11, vt 0.012, vw 12, ts 24, q 0.3.
- worstB 3.065 (was 0.354, +2.711), worstC 4.573 (was 3.465).
- turnover 0.136258, FULL 2.010, H2 6.612, med12 2.376.
- Segments: H1 3.893 / to 0.145809 / 34 trades; H2 6.612 / 0.156159 / 36;
  B 5.132 / 0.158032 / 15; C 6.488 / 0.153812 / 36;
  FULL 2.010 / 0.136258 / 476 (ETC 209, TRX 267).
- Stress 4-cell:
  fund0.0003 x fee0.0004 -> B 3.562 / C 5.026;
  fund0.0003 x fee0.0008 -> B 3.065 / C 4.573;
  fund0.0005 x fee0.0004 -> B 5.132 / C 6.488;
  fund0.0005 x fee0.0008 -> B 4.635 / C 6.038.
- Fee curve FULL: 0.0002 -> 2.162 (med 2.521); 0.0004 -> 2.010 (2.376);
  0.0008 -> 1.706 (2.085); 0.0012 -> 1.401 (1.793).

## Why cd+2 works
- Weak cell is low-funding + high-fee: churn cost dominates.
  +2 cooldown bars cut whipsaw entries (FULL trades 510 -> 476),
  turnover falls, worst-B lifts from 0.354 to 3.065.
- ts12 fails opposite: turnover 0.211 breaks the 0.16 cap, worstB -0.557.
- ts36 helps worstB (0.954) but less than cd+2.
- vt/q moves are near-neutral or break FULL>2.0 (q guards both FAIL).

## Verdict
- ADOPT cd+2 as E13 fee-hardened AA-H1: worstB 3.065 > 1.0 with margin,
  all gates PASS, median improved (2.376 > 1.964).
- Keep FORMULA locked. No further guard stacking in E13 (single-guard scope).
- Artifacts: `results/results_E13_fee.json`, `research/run_e13.py`, `logs/e13.log`.
