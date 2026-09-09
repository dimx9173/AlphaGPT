# STRATEGY E11 — Walk-Forward: AA etc15 branch H1→H2 median recovery check

Date: 2026-09-08 | Engine: mirror `research/run_e1.py` + `run_aa.py` (leg_series + quantile_mask_long q0.3 long-only + `_vol_scale` post-stops pre-roll + roll1)
Formula (global Y1b): [3,2,7,2,7,11,15,4,4,6,6,10] | ETC lth 0.88/sl None + TRX lth 0.85/sl 0.05 | ts24 | vt0.012 | 50/50 | aster perp 2x fund 0.0005 fee 0.0004/0.0008
Bars: n=6580 4h | H1 [5584:6077] (493) | H2 [6077:6570] (493) | B [6380:6580] | C [6080:6580] | FULL [0:6580] | 12-fold 548 bar

## Grid
12 AA etc15 variants: sth [0.10, 0.11, 0.12] × etc_cd [15, 18] × trx_cd [6, 9], all vt0.012.
Y1b baseline: sth 0.12 / etc18 / trx6 / vt None.

## Y1b baseline
- H1 3.364 | H2 4.091 | FULL 1.861
- 12-fold: mean 1.810 | median 3.178 | n_pos 9/12 | sharpes [-0.699, -2.914, 3.578, 3.352, -2.947, 3.048, 1.556, 0.168, 4.747, 4.971, 3.307, 3.558]

## Variant results (H1 | H2 | H2_2x | FULL | 12f mean/med/n_pos | gates)
| sth | etc | trx | H1 | H2 | H2_2x | FULL | 12f mean | 12f med | n_pos | PASS |
|-----|-----|-----|-----|-----|-------|------|----------|---------|-------|------|
| 0.10 | 15 | 6 | 4.062 | 5.285 | 4.805 | 1.760 | 1.608 | 1.581 | 8/12 | FAIL (mean, npos) |
| 0.10 | 15 | 9 | 4.604 | 5.581 | 5.121 | 2.025 | 1.644 | 0.847 | 7/12 | FAIL (all three) |
| 0.10 | 18 | 6 | 2.747 | 5.309 | 4.826 | 2.081 | 1.608 | 2.523 | 8/12 | FAIL (mean, npos) |
| 0.10 | 18 | 9 | 3.279 | 5.617 | 5.157 | 2.338 | 1.652 | 1.816 | 8/12 | FAIL (mean, npos) |
| 0.11 | 15 | 6 | 2.818 | 5.529 | 5.038 | 1.822 | 1.819 | 1.680 | 9/12 | PASS |
| 0.11 | 15 | 9 | 3.379 | 5.864 | 5.366 | 2.085 | 1.791 | 1.049 | 9/12 | FAIL (median) |
| 0.11 | 18 | 6 | 2.373 | 4.790 | 4.335 | 2.088 | 1.544 | 2.265 | 8/12 | FAIL (mean, npos) |
| 0.11 | 18 | 9 | 2.870 | 4.981 | 4.530 | 2.348 | 1.513 | 1.596 | 8/12 | FAIL (mean, npos) |
| 0.12 | 15 | 6 | 3.092 | 6.557 | 6.076 | 1.936 | 1.777 | 1.776 | 9/12 | PASS |
| 0.12 | 15 | 9 | 3.557 | 6.835 | 6.349 | 2.198 | 1.752 | 1.525 | 8/12 | FAIL (npos) |
| 0.12 | 18 | 6 | 2.655 | 5.013 | 4.561 | 2.064 | 1.655 | 2.479 | 9/12 | FAIL (mean) |
| 0.12 | 18 | 9 | 3.046 | 5.132 | 4.682 | 2.328 | 1.631 | 2.239 | 8/12 | FAIL (mean, npos) |

Gates: 12-fold median ≥ 1.5 AND mean > 1.7 AND n_pos ≥ 9.

## Unbiased H1-best check
- H1-best = sth 0.10 / etc15 / trx9 (H1 4.604) → H2 5.581: H2-hold (> 3.0) PASS.
- But its 12-fold (mean 1.644 / med 0.847 / n_pos 7) fails all three gates → overall FAIL.
- H1-best picks the variant with the worst 12-fold median (0.847): classic in-sample H1 overfit signature.

## Gate passers (not H1-best — watchlist only, promoting either would be cherry-picking)
1. sth 0.11 / etc15 / trx6: H1 2.818 → H2 5.529 (H2_2x 5.038) | 12f 1.819/1.680/9 | FULL 1.822
2. sth 0.12 / etc15 / trx6: H1 3.092 → H2 6.557 (H2_2x 6.076) | 12f 1.777/1.776/9 | FULL 1.936 — strongest H2 of grid.

Note: both passers use etc15/trx6; fee2x decay H2 ≈ 0.48–0.49, acceptable.

## Verdict: KEEP_Y1b
H1-best does not survive walk-forward (fails mean/median/n_pos). No promotion.
Y1b keeps the best 12-fold median (3.178 vs best variant 2.523) and tied-best n_pos (9/12).
Watch sth 0.12/etc15/trx6 for E12 (strongest H2 6.557, passes gates) — needs fresh-segment confirmation before any promotion.

## Artifacts
- `results/results_E11_walkforward.json` (config + Y1b baseline + 12 variants + h1_best + decision)
- `research/run_e11.py`, `logs/e11.log`
