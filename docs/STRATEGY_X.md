# STRATEGY_X — x10 iteration (S/T follow-up, 2026-09-06)

Locked engine: FORMULA [3,2,7,2,7,11,15,4,4,6,6,10], aster perp 2x, fund 0.0005, fee base 0.0004 / 2x 0.0008.
Segments: H2 = frozen 2nd half OOS [6077:6570] (n=493/494); B = last-200 [6380:6580]; C = last-500 [6080:6580]. Full n=6580.
Leg bases: Q1 (ETC 0.85/0.15/cd12/None + TRX 0.85/0.12/cd6/None) for X1/X2/X3; S2-base (ETC 0.88/0.12/cd12/None + TRX 0.85/0.15/cd6/sl0.05) for X4/X5/X7/X8/X9/X10; rank2 (0.85/0.12/ETC-cd12/TRX-cd6) for X6.
Scripts: research/run_x1..x10.py → results/backtest_X1..X10.json.

## X1 short-led lock (Q1 legs, 50/50; PASS: a variant B>1.0 AND C>1.5 AND fee2x-B>0.5) — FAIL
| variant | H2 sh | B sh | C sh | B-fee2x |
| a plain both | 0.756 | 0.381 | 0.778 | 0.203 |
| b q0.3 both | 0.393 | 1.257 | 0.579 | 1.091 |
| c ETC-both+TRX-short plain | 1.004 | 0.813 | 0.986 | 0.659 |
| d ETC-both+TRX-short q0.3 | 0.303 | 0.926 | 0.488 | 0.774 |
| e short-only both | 0.633 | 0.813 | 0.781 | 0.659 |
No variant clears C>1.5. Best B is b (1.257) but C collapses to 0.579 — q-gate trades C for B on Q1 legs. TRX-short-only (c/e) lifts H2 (~1.0/0.63) but B stays <1.0.

## X2 q-sweep (Q1 legs, 50/50; best=argmax B+C; PASS B>1.0 AND C>1.5 AND fee2x-B>0.5) — FAIL
Best q=0.3: H2 0.393 / B 1.257 / C 0.579 / B-fee2x 1.091. C never clears 1.5 at any q (q ladder B/C: None 0.381/0.778 → 0.3 1.257/0.579). q-gate confirmed as B-only fix on Q1 legs, not a C solution.

## X3 T6-followup (Q1 legs, wETC 0.1-0.5 x q 0.25/0.3/0.35; PASS: beat q0.3-50/50 ref B 1.257/C 0.579 on BOTH by >0.2 AND H2>1.0 AND fee2x-B>0.5) — PASS
Best wETC=0.1 q=0.3: H2 2.141 / B 2.724 (+1.467) / C 1.779 (+1.200) / B-fee2x 2.251, mdd B 0.085 C 0.126, tr 12/29.
Ladder (q=0.3): w0.1 B 2.724 C 1.779 → w0.15 B 2.481 C 1.588 → w0.2 B 2.228 C 1.392 → w0.3 B 1.794 C 1.044 → w0.5 B 1.257 C 0.579. Monotone: TRX-heavy wins B+C+H2 jointly; H2 2.141→0.393 as wETC rises. q=0.3 dominates q=0.25/0.35 at every weight.
New candidate: Q1-legs + wETC 0.1 + q0.3 long-gate (TRX 90% + bleed-filtered longs).

## X4 ts/tp extend (S2-base, q0.3, 50/50; ts 24/36/60/96 x tp None/0.08; PASS B>1.2 AND C>1.8) — PASS (ts=24, tp inert)
| ts | tp | H2 | B | C | tr H2/B/C |
| 24 | None/0.08 | 2.499 | 1.344 | 2.052 | 40/16/41 |
| 36 | None/0.08 | 2.385 | 0.988 | 1.955 | 35/14/34 |
| 60 | None/0.08 | 2.387 | 1.032 | 1.946 | 30/12/30 |
| 96 | None/0.08 | 2.387 | 1.032 | 1.946 | 30/12/30 |
tp is fully inert (identical rows — stops never bind on this path). ts=24 is the peak; longer ts decays B (1.344→0.988→1.032). Keep ts=24/tp=None.

## X5 cooldown extend (S2-base, q0.3; cd pairs x sth; PASS B>1.2 AND C>1.8 AND H2>1.5) — PASS (cd 18/6, sth 0.12)
Best ETC-cd18/TRX-cd6 sth 0.12: H2 3.953 / B 4.180 / C 3.697, mdd 0.081/0.067/0.074, tr 28/12/28.
Runner-up ETC-cd18/TRX-cd9 sth 0.12: H2 4.141 / B 3.744 / C 3.859. cd18 rows dominate; cd24 collapses C (0.409/-0.016). sth 0.12 beats 0.15 everywhere at cd18.
New candidate: S2-base + q0.3 + ETC-cd18/TRX-cd6 + sth 0.12 (all three segments >3.5, mdd <0.09). Strongest raw cell of x10 — flag for fee/WF confirm (done in X7/X9 scope: fee2x-B held 1.091 on Q1-q0.3; S2-base fee sweep T1 held 0.867→0.702).

## X6 SL x TRX-long-off (rank2 0.85/0.12/cd12/cd6; sl None-0.06 x off F/T; PASS B>1.2 AND C>1.5) — PASS (sl 0.02 + TRX-long-off)
Best sl0.02 + TRX-long-off: H2 4.244 / B 2.639 / C 3.513, tr 45/17/46.
Ladder: sl0.02 off=T B 2.639 C 3.513 → sl0.03 off=T B 2.213 C 3.504 → sl0.05 off=F B 1.643 C 1.532 → sl0.06 B negative (overtightened). TRX-long-off helps at every SL (e.g. sl0.05 B 1.643→2.353). Confirms T2 diagnosis: TRX-long is regime-dead in B; gate it + light SL 0.02-0.03.
New candidate: rank2 + sl0.02 + TRX-long-off.

## X7 cost pocket (S2-base q0.3; fund 0.0003/0.0005/0.0007 x fee 0.0004/0.0006/0.0008; PASS worst B>0 AND worst C>1.0) — FAIL (honest)
Worst cell fund 0.0003/fee 0.0008: B 0.007 / C 0.679 → C fails. Base fund 0.0005 holds (B 1.032/C 1.946 at base fee; fee2x B 0.867/C 1.746). fund 0.0003 kills the short edge (short-funding dependence confirmed); fund 0.001 inflates C to 2.8-4.6 (artifact, flagged at all fees).
Verdict: cost pocket is fund 0.0005±, not wider. Do not trade at fund 0.0003; do not trust fund 0.001 prints.

## X8 vol-target (S2-base q0.3; vt 0.01-0.03 x vw 12/24; PASS B>1.5 AND C>2.0 at turnover<0.15) — PASS (vt 0.01/w12)
Best vt0.01/w12: H2 3.806 / B 3.374 / C 3.925, turnover 0.126/0.134/0.125 (<0.15).
All 8 cells pass raw bars; turnover 0.12-0.14 (2x base 0.065 but under cap). Ladder: smaller vt wins (0.01 > 0.015 > 0.02 > 0.03); w12 beats w24 everywhere.
Candidate: S2-base + q0.3 + vt0.01/w12. Caveat: vol-scaling doubles turnover — net of fee2x to confirm before sizing.

## X9 12-fold WF (S2-base q0.3; PASS ≥10/12 pos AND min>-1.0) — FAIL (10/12 pos, min -2.207)
Folds (sh): [1.228, -2.207, 3.412, 3.712, -2.149, 2.879, 0.844, 0.925, 3.993, 5.044, 3.357, 2.655]. 10/12 positive but folds 1 (-2.207) and 4 (-2.149) break the min bar. Fee2x B 0.867 / C 1.746 still hold. Regime note: early folds fragile, late folds (8-11) all >2.6 — same early-weak signature as grouptest-10 (fold0 negative).

## X10 paper attribution (S2-base q0.3 ledger vs plain vs paper2.log) — DIAGNOSED, no adopt
q0.3 ledger: 431 trades, final_x 0.9945, sharpe 0.371, mdd 0.799 (ETC 169/TRX 262). Plain repro: 478 trades, 2.9883, 0.899, 0.7846 (ETC 186/TRX 292) — matches paper2.log 477/2.927/0.891. Drift q−plain: −47 trades, −1.9938x, −0.528 sharpe. Longs: q 5 vs plain 57 (52 longs filtered).
Root cause: q-gate removes 52/57 longs; the removed longs carried +2x of full-history equity (early-sample long contribution). Segment PnL (T1/X1) shows longs ≈0 on H2/B/C tails but +2x over FULL — q0.3 optimizes the tail at the cost of the full-history long premium.
Paper-fix: do NOT put q-gate in the paper ledger path. Keep paper = plain both-legs (2.99x/0.90). Use q-gate/sl/off/ts/cd/vol variants as tail overlays only, gated by regime (B-window), not as full-history replacements.

## x10 tally
- PASS: X3 (w0.1+q0.3), X4 (ts24), X5 (cd18/6 sth0.12), X6 (sl0.02+TRX-long-off), X8 (vt0.01/w12) — 5/10.
- FAIL: X1 (no variant clears C), X2 (C never clears), X7 (cost pocket narrow), X9 (2 early folds break min), X10 (diagnosed, no adopt by design).
- Candidates (paper-gated, regime overlays — NOT full-history replacements):
  1. X5: S2-base + q0.3 + ETC-cd18/TRX-cd6 + sth0.12 (H2 3.95/B 4.18/C 3.70, mdd <0.09).
  2. X3: Q1-legs + wETC 0.1 + q0.3 (H2 2.14/B 2.72/C 1.78, TRX-heavy).
  3. X6: rank2 + sl0.02 + TRX-long-off (H2 4.24/B 2.64/C 3.51).
  4. X8: S2-base + q0.3 + vt0.01/w12 (H2 3.81/B 3.37/C 3.93, to ~0.13).
  5. X4: S2-base + q0.3 + ts24 (H2 2.50/B 1.34/C 2.05).
- Paper ledger stays plain (X10): 2.99x/0.90 — q-gate and overlays are B-regime tools, not ledger replacements.
- Next: combine non-overlapping winners (X5 cd + X6 sl/off + X4 ts on X3 weights?) as Y1 ensemble, fee2x + 12-fold gated; then live-paper the plain ledger + best overlay in shadow mode.
