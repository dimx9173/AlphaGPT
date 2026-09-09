
# STRATEGY_S — short-led combo (ETC+TRX locked legs)

## S1 context (inherited)
- Locked legs: FORMULA [3,2,7,2,7,11,15,4,4,6,6,10], ETC (0.88/0.12/cd12/None) + TRX (0.85/0.15/cd6/sl0.05), 50/50, aster perp 2x, fund 0.0005, fee 0.0004.
- P2 baseline problem: B-tail200 both sharpe 0.158; long leg sharpe -3.967 (2 TRX trades, cum -0.0269) drags short leg (sharpe 0.588, 12 trades).

## S2 long-leg quantile filter (side='long', short untouched)
- Method: mirrors research/run_p2.py leg_series + model_core/backtest._apply_quantile(side='long'): keep long entries only where |logit| in top-q fraction; short leg untouched; 50/50 combo.
- Script: research/run_s2.py. Output: results/backtest_S2_longfilter.json.
- Sweep q in [None, 0.5, 0.3, 0.1]; segments A=frozen H2 (n=494), B=last-200, C=last-500.

| q | A-both sh/tr | B-both sh/ann/mdd/tr | B-long sh/cum/tr | B-short sh | C-both sh/ann/tr | C-long sh/cum/tr |
|---|---|---|---|---|---|---|
| None | 1.512/32 | 0.158/0.109/0.210/14 | -3.967/-0.0269/2 | 0.588 | 1.756/0.946/33 | -2.231/-0.0267/4 |
| 0.5 | 1.720/31 | 0.694/0.481/0.210/13 | +7.122/+0.0102/1 | 0.533 | 1.937/1.049/31 | +4.480/+0.0102/1 |
| 0.3 | 1.731/30 | 1.032/0.712/0.210/12 | +5.850/+0.0204/1 | 0.703 | 1.946/1.060/30 | +4.190/+0.0119/1 |
| 0.1 | 1.635/30 | 0.541/0.379/0.210/12 | 0.000/0.0000/0 | 0.541 | 1.850/1.008/30 | 0.000/0.0000/0 |

- Best-q: q=0.3. B-both sharpe 1.032 (ann 0.712, mdd 0.210, 12 trades; ETC 5 + TRX 7), B-long 1 trade sharpe +5.85 cum +0.0204 (the 1 losing TRX long filtered out; remaining long wins), B-short 0.703. C-both 1.946 (ann 1.060, 30 trades), C-long 1 trade +4.19/+0.0119. A-both 1.731 (30 trades).
- q=0.1 kills long fully (0 trades) but B-both drops to 0.541 — over-filtering removes the one good long; q=0.3 keeps the winner.
- Verdict: PASS via cond1 (B-both sharpe > 1.0 with long trades 1 <= 2 — effectively short-led but formally both). cond2 (B-both > 1.2 with long positive) not met; do not claim long edge, only bleed fixed. Fragility note: B-long rests on 1 trade — treat as bleed-removal, not long alpha. Recommend S3: fee2x stress on q=0.3 + time-OOS confirm before sizing.

## S3 TRX-leg diagnosis + DOGE replacement sweep (Q1 H1-best legs, fresh data n=6580)
- Locked legs (Q1 H1-best, NOTE differs slightly from P2/S1-S2 locks): FORMULA [3,2,7,2,7,11,15,4,4,6,6,10]; ETC (0.85/0.15/cd12/None), TRX (0.85/0.12/cd6/None), DOGE (0.85/0.15/cd12/None). Equal-weight, aster perp 2x, fund 0.0005, fee base 0.0004, fee2x 0.0008 on B/C.
- Segments: H1/H2 = frozen OOS halves (cut 5593, H1 n=493, H2 n=494); B = last-200 4h bars; C = last-500. Both-legs.
- Script: research/run_s3.py. Output: results/backtest_S3_legreplace.json.
- Per-coin diagnosis (both-legs, base fee): H2 / B / C sharpe — ETC 0.397/0.736/0.570; TRX 1.114/-1.593/1.038; DOGE 0.251/-4.603/0.418.
- TRX-leg diagnosis: TRX long leg is the drag everywhere (H2 long -2.245/4tr, B long -3.967/2tr, C long -2.231/4tr); TRX short leg is strong (H2 +2.114, C +2.046, B +0.656). B both-leg failure (-1.593) = long bleed outweighs short gain on the fresh tail.
- DOGE is worse, not a replacement: B both -4.603 (short leg -4.603, 4 trades, cum -0.5025 — fresh-tail short blowup), C both only 0.418. ETC is the only leg green on B (+0.736, short-only, 5 trades).

| pair | H1 sh/ann/mdd/tr | H2 sh/ann/mdd/tr | B sh/ann/mdd/tr | C sh/ann/mdd/tr | B-fee2x sh | C-fee2x sh | H2 leg corr | PASS |
|---|---|---|---|---|---|---|---|---|
| ETC+TRX (incumbent) | 5.901/2.658/0.186/33 | 0.635/0.374/0.397/31 | 0.381/0.262/0.191/14 | 0.778/0.449/0.397/32 | 0.203 | 0.565 | ETC_TRX 0.172 | FAIL |
| ETC+DOGE | 4.284/3.034/0.203/29 | 0.367/0.336/0.642/25 | -2.081/-2.270/0.437/9 | 0.563/0.501/0.642/25 | -2.152 | 0.445 | ETC_DOGE 0.612 | FAIL |
| TRX+DOGE | 3.039/1.457/0.144/36 | 0.531/0.275/0.318/32 | -4.619/-2.970/0.332/13 | 0.664/0.338/0.318/33 | -4.798 | 0.422 | TRX_DOGE 0.195 | FAIL |
| ETC+TRX+DOGE (3-leg) | 4.800/2.383/0.154/49 | 0.516/0.329/0.445/44 | -2.206/-1.659/0.308/18 | 0.695/0.429/0.445/45 | -2.344 | 0.506 | 0.172/0.612/0.195 | FAIL |

- PASS bar (B > 1.0 AND C > 1.5 AND fee2x-B > 0.5): all four pairs FAIL. Rank by C: ETC_TRX (0.778) > 3-leg (0.695) > TRX_DOGE (0.664) > ETC_DOGE (0.563).
- Corr note: ETC_TRX H2 corr 0.172 (diversifying); ETC_DOGE 0.612 (redundant — replacement adds no diversification).
- Verdict: FAIL — DOGE replacement rejected on all counts (worse B, worse C, high corr with ETC, tail short-blowup). Incumbent ETC+TRX still best but does NOT pass S3 bar either (B 0.381, C 0.778, fee2x-B 0.203). Do not swap legs. Open question for S4: S2-style long-leg quantile filter (q=0.3 passed B>1.0 on old vintage/legs) re-tested on Q1 legs + fee2x, or short-only variant given TRX-short/ETC-short carry all fresh-tail gains.
