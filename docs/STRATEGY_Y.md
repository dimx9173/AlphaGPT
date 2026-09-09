# STRATEGY_Y — Y1 ensemble + vol stack + shadow gate (2026-09-06)

Locked engine: FORMULA [3,2,7,2,7,11,15,4,4,6,6,10], aster perp 2x, fund 0.0005, fee base 0.0004 / 2x 0.0008.
Segments: H2 [6077:6570], B [6380:6580], C [6080:6580], FULL n=6580. Scripts research/run_y1..y3.py → results/backtest_Y1..Y3.json.

## Y1 ensemble ladder (all q0.3 long-gate; PASS best B>2.0 AND C>2.5 AND H2>2.0 AND fee2x-B>1.0) — PASS
| stack | H2 | B | C | B-fee2x | tr H2/B/C |
| a X5 alone (S2 cd18/6 sth0.12, 50/50) | 3.953 | 4.180 | 3.697 | 3.886 | 28/12/28 |
| b X5+X4 ts24 (WINNER) | 4.429 | 4.496 | 3.479 | 4.155 | 35/14/36 |
| c X5+X6 sl0.02/TRX-off | 3.598 | 3.277 | 3.356 | 2.937 | 38/15/38 |
| d full 50/50 (cd+sl/off+ts) | 4.090 | 3.337 | 3.058 | 2.974 | 41/16/42 |
| e full on X3 wETC0.1 | 2.349 | 1.634 | 1.889 | 0.989 | 41/16/42 |
Winner b: S2-base + q0.3 + ETC-cd18/TRX-cd6 sth0.12 + ts24, 50/50 — H2 4.429/B 4.496/C 3.479, mdd ≤0.08, fee2x-B 4.155.
Notes: sl/off (c/d) adds trades but dilutes peak — ts24 stacks cleanly, sl/off does not. X3 TRX-heavy weights (e) collapse on S2-base (B 1.634/C 1.889) — weights do not transfer across leg bases.

## Y2 vol stack (X5 base; vt None/0.01/0.015 x vw 12/24; PASS B>2.5 AND C>3.0 at to<0.15 AND fee2x-B>1.0) — PASS
Best vt0.01/w12 on X5: H2 5.121 / B 6.449 (fee2x 5.936) / C 4.722 (fee2x 4.330), turnover 0.120/0.130/0.119, mdd B 0.074 C 0.118.
Vol-fee: turnover 2x (0.065→0.130) but fee2x decay only 0.51/0.39 — fee-robust. Smaller vt wins; w12 beats w24.
Caution: vol numbers live on the tail; Y1-b (no vol) stays the lock, Y2-vol is a gated overlay candidate.

## Y3 shadow gate (plain ledger live + overlay gated; LED-switch, causal trailing-200) — ADOPT
- Plain ledger: 478 trades, 2.9883x, 0.899, mdd 0.7846 (longs 57) — matches paper2.log.
- Overlay (fallback X5+X6sl/off+X4ts since Y1 unknown at run time): 753 trades, 1993.09x, 4.79, mdd 0.1256, longs 2. Attribution: q-gate alone collapses (X10 0.99x); the +2x comes from cd/sth+sl/off+ts stack.
- Gate rule (adopted): overlay active iff trailing-200 overlay-combo sharpe > 1.0 (coverage 62.8%). Alt rule (plain trailing-200 < 1.0) REJECTED: churn gives shadow B 0.171/C 0.538 < plain.
- Shadow (main gate): FULL 54.42x/2.40/633tr (no collapse) | B 1.079x/3.738 vs plain 1.0659x/2.509 | C 1.2665x/3.852 vs plain 1.0929x/1.775 | H2 1.2823x/4.093 vs plain 1.0929x/1.788. Beats plain on B+C+H2, no full-history collapse.
- Overlay staleness note: Y3 overlay used sl0.02/TRX-off fallback legs, not the Y1-winner b (ts24, no sl/off). Re-run shadow with Y1-b legs before live sizing.

## Y decisions
1. LOCK Y1-b (S2 + q0.3 + cd18/6 sth0.12 + ts24, 50/50) as the tail overlay lock. Paper ledger stays plain.
2. Y2-vol (vt0.01/w12) is a gated candidate only — confirm fee2x + 12-fold on Y1-b+vol before sizing.
3. ADOPT Y3 shadow mode (trailing-200 overlay sharpe>1.0, coverage ~63%), but refresh overlay legs to Y1-b then shadow-paper live.
4. Next: Y4 = Y1-b + vol on 12-fold + fee2x; Y5 = shadow with Y1-b legs; then live shadow-paper.
