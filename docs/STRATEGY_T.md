# STRATEGY_T — tail checks + weight sweep (locked formula [3,2,7,2,7,11,15,4,4,6,6,10], aster 2x, fund 0.0005, fee 0.0004)

## T5+T6 (2026-09-06) — run: research/run_t5t6.py → results/backtest_T5T6.json
Per-leg params = Q1 H1-best per coin (TRX 0.85/0.12/cd6/slNone — differs from P2/S2 TRX 0.85/0.15/cd6/sl0.05).
Segments: H2 = frozen 2nd half of last-15% OOS (n=494); B = last 200 4h bars; C = last 500 4h bars. Full n=6580.

### T5 — combo tail check (equal-weight, both-legs) — VERDICT: FAIL (no combo beats base on BOTH B and C)
| combo | H2 sharpe/ann/mdd/tr | B sharpe/ann/mdd/tr | C sharpe/ann/mdd/tr |
| BASE ETC+TRX | 0.635 / 0.3743 / 0.3967 / 31 | 0.381 / 0.2624 / 0.1914 / 14 | 0.778 / 0.4488 / 0.3967 / 32 |
| ETC+TRX+DOGE | 0.516 / 0.3287 / 0.4449 / 44 | -2.206 / -1.6593 / 0.3081 / 18 | 0.695 / 0.4291 / 0.4449 / 45 |
| ALL5 | 0.766 / 0.4684 / 0.4245 / 82 | -4.228 / -2.91 / 0.3398 / 35 | 0.643 / 0.3865 / 0.4245 / 86 |
| MAJORS | 0.725 / 0.5312 / 0.5175 / 51 | -5.99 / -5.025 / 0.515 / 21 | 0.477 / 0.345 / 0.5175 / 54 |
B leg-corr: ETC_TRX 0.154 (low, diversifying); ETC_DOGE 0.518, BTC_DOGE 0.609, SOL_DOGE 0.588, BTC_SOL 0.60 (high);
TRX_DOGE 0.227, BTC_TRX 0.191, SOL_TRX 0.169. DOGE/BTC/SOL legs drag tail-B hard → keep base pair.

### T6 — weight sweep base pair (wETC, wTRX=1-w), plain + S2 q=0.3 long-gate — VERDICT: PASS (q0.3 only)
qNone (ref 50/50: B 0.381 / C 0.778): w0.2 B-0.430 C1.055; w0.3 B-0.038 C0.953; w0.7 B0.582 C0.667; w0.8 B0.647 C0.628 → no weight beats 50/50 on BOTH by >0.2 → FAIL.
q0.3 (ref 50/50: B 1.257 / C 0.579): w0.2 H2 1.228 / B 2.228 (+0.971) / C 1.392 (+0.813), mdd B0.1113 C0.1934, tr 12/29;
w0.3 B1.794 C1.044; w0.7 B0.971 C0.318; w0.8 B0.875 C0.230 → best wETC=0.2 beats 50/50 on BOTH by >0.2 → PASS.
Note: q0.3 gate lifts B sharply but H2 degrades as wETC rises (H2: 1.228→0.055 across w0.2→w0.8); TRX-heavy + long-gate is the tail winner.

## T1+T2 (2026-09-06) — run: research/run_t1t2.py → results/backtest_T1T2.json
Locked formula [3,2,7,2,7,11,15,4,4,6,6,10], aster perp 2x, fund 0.0005. Full n=6580.
Segments: H1 [5584:6077], H2 [6077:6570] (493/493), B [6380:6580] (200), C [6080:6580] (500), C1 [6080:6330] (250), C2 [6330:6580] (250).

### T1 — S2 q=0.3 (long-gate) paper-variant stress — VERDICT: PASS
Base per-leg: ETC 0.88/0.12/cd12/slNone + TRX 0.85/0.15/cd6/sl0.05, 50/50 eq, q=0.3 long-only gate. Repro: B 1.032 / C 1.946 (matches S2 PASS 1.03/1.95).
Fee sweep both-legs:
| seg | fee 0.0004 | fee 0.0008 (2x) | fee 0.0012 (3x) |
| B sharpe/ann/mdd/tr | 1.032 / 0.7119 / 0.2097 / 12 | 0.867 / 0.5981 / 0.2105 / 12 | 0.702 / 0.4842 / 0.2113 / 12 |
| C sharpe/ann/mdd/tr | 1.946 / 1.0599 / 0.2549 / 30 | 1.746 / 0.9513 / 0.2557 / 30 | 1.545 / 0.8427 / 0.2565 / 30 |
Time-OOS (base fee): C1 1.181 / 0.5184 / 0.0837 / 17; C2 2.066 / 1.3455 / 0.2549 / 15.
PASS: fee2x-B 0.867 > 0.5 AND C1 1.181 > 0.5 AND C2 2.066 > 0.5 → overall PASS (no half-dead; fee decay ~0.16-0.20 sharpe per +0.0004 fee).

### T2 — S4 rank2 (0.85/0.12/med:ETC-cd12/TRX-cd6/slNone) deep-dive — VERDICT: B-fixable at COMBO level via SL, TRX-long regime-dead in B
Per-coin B breakdown (both/long/short sharpe, trades):
| coin | both | long | short | killer leg |
| ETC | 0.428 (t5) | 0.0 (t0, flat — no long signals in B) | 0.428 (t5) | long (absent: contributes 0, combo carried by short) |
| TRX | -1.593 (t9) | -3.967 (t2) | +0.656 (t7) | long (2 long trades deeply negative, drag combo to 0.09) |
SL variants on rank2 params (shared SL, B+C combo sharpe | ETC | TRX):
| sl | B combo | B ETC | B TRX | C combo | C ETC | C TRX |
| None | 0.09 | 0.428 | -1.593 | 2.001 | 1.873 | 1.038 |
| 0.03 | 1.433 | 2.06 | -1.512 | 3.168 | 3.151 | 1.006 |
| 0.05 | 1.643 | 2.325 | -1.593 | 1.532 | 1.376 | 1.038 |
| 0.08 | -1.334 | -1.066 | -1.593 | 3.079 | 3.042 | 1.038 |
Diagnosis: SL fixes ETC-B (0.428→2.325 at sl0.05, mdd 0.369→0.129) but NEVER fixes TRX-B (stuck ≈-1.5/-1.6 at every SL; TRX SL legs inert — stops never trigger on TRX B path). Combo B crosses 1.0 purely on ETC repair (sl0.05 B 1.643, mdd 0.096). sl0.08 overtightens ETC (-1.066) → combo -1.334.
Verdict: B-fixable at combo level (adopt sl0.05 on rank2 params if trading B window), but TRX-long is regime-dead in B (long sharpe -3.967, only 2 trades). Cleaner paper fix: rank2 + sl0.05 AND TRX-long gate/off in B regime; else expect TRX-long tail risk to recur.
