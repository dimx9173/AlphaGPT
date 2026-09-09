# STRATEGY_AD — 新因子 / 新規則挖掘 (Factor & Rule Mining) — 2026-09-07 實盤復核

> 基線: FORMULA [3,2,7,2,7,11,15,4,4,6,6,10] (`FOMO PRESSURE SUB PRESSURE SUB ABS DECAY DEV DEV ADD ADD NEG`), Y1b=ETC(0.88/0.12/cd18/None/ts24)+TRX(0.85/0.12/cd6/0.05/ts24), q0.3 vtNone (AD 壓測用無 vol 以顯 TP 效應), 50/50, aster 2x fund0.0005 fee0.0004/0.0008, n=6580 4h (H1 493 / H2 493 / FULL 6580)
> 引擎: 鏡像 run_z1.py leg_series + quantile_mask_long(q0.3 long-only) + _apply_cooldown + _apply_stops(tp) + _vol_scale(vtNone復核) + roll1
> 數據: data/data_15m_3y/*.csv 15m→4h×16 (cold-data symlink)
> 任務: 16 行 = AD1 10 因子變異 + AD2 4 TP + AD3 2 跨幣，H1[5584:6077] 選型 H2 驗證無偏，fee2x 壓測，12fold 548bar

## AD1 — 因子鄰域變異 (無偏 H1 排名 → H2 驗證)

- 生成: 5 單點變異 (基線 12 token 中隨機位點改 1 token 至另值, 保 vocab 23) + 5 完全隨機長度12公式，seed42 固定；10 候選皆 VM 有效 (stack 平衡率高於盲 vocab 隨機，因 ops 含多 arity)。
- 排名: H1[5584:6077] sharpe 排序，H1-best 無偏驗 H2/B/C/FULL；同時報告 AD1_gain = H1-best 在 H2 與 FULL 雙勝 baseline。

| idx | kind | formula (decode) | H1 sh/ann | H2 sh (2x) | B sh (2x) | C sh (2x) | FULL sh (2x) |
|---|---|---|---|---|---|---|---|
| base | baseline | 4.091 dec FOMPRESUBPRE | 3.364/1.5177 | 4.091 (3.776) | 4.496 (4.155) | 3.479 (3.168) | 1.861 (1.667) |
| 0 | single | [2, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10] decode PRESSURE,PRESSURE,SUB,PRESSURE,SUB,ABS... | 3.177/1.2405 | 2.993 (2.723) | -1.238 (-1.426) | 2.441 (2.175) | 1.430 (1.233) |
| 1 | single | [3, 2, 7, 2, 7, 11, 15, 4, 4, 6, 22, 10] decode FOMO,PRESSURE,SUB,PRESSURE,SUB,ABS... | 0.000/0.0000 | 0.000 (0.000) | 0.000 (0.000) | 0.000 (0.000) | 0.000 (0.000) |
| 2 | single | [3, 2, 7, 2, 7, 11, 15, 4, 4, 8, 6, 10] decode FOMO,PRESSURE,SUB,PRESSURE,SUB,ABS... | 3.341/1.7851 | 0.988 (0.755) | 3.396 (3.087) | 0.749 (0.504) | 1.330 (1.163) |
| 3 | single | [5, 2, 7, 2, 7, 11, 15, 4, 4, 6, 6, 10] decode LOG_VOL,PRESSURE,SUB,PRESSURE,SUB,ABS... | 3.863/1.5973 | 5.507 (5.198) | -2.511 (-2.710) | 4.817 (4.505) | 1.274 (1.081) |
| 4 | single | [3, 2, 6, 2, 7, 11, 15, 4, 4, 6, 6, 10] decode FOMO,PRESSURE,ADD,PRESSURE,SUB,ABS... | 2.842/1.2037 | 4.971 (4.624) | -0.842 (-1.032) | 4.907 (4.562) | 1.569 (1.365) |
| 5 | random | [2, 17, 4, 4, 21, 15, 17, 5, 8, 16, 19, 13] decode PRESSURE,MAX3,DEV,DEV,DELTA,DECAY... | 0.116/0.0469 | 0.511 (0.313) | 3.445 (3.295) | 0.901 (0.720) | -0.367 (-0.514) |
| 6 | random | [4, 15, 0, 16, 1, 17, 13, 18, 15, 16, 5, 22] decode DEV,DECAY,RET,DELAY1,LIQ_SCORE,MAX3... | 0.000/0.0000 | 0.000 (0.000) | 0.000 (0.000) | 0.000 (0.000) | 0.000 (0.000) |
| 7 | random | [1, 2, 2, 1, 5, 8, 17, 2, 18, 7, 8, 13] decode LIQ_SCORE,PRESSURE,PRESSURE,LIQ_SCORE,LOG_VOL,MUL... | 0.000/0.0000 | 0.000 (0.000) | 0.000 (0.000) | 0.000 (0.000) | 0.000 (0.000) |
| 8 | random | [2, 5, 4, 9, 3, 20, 14, 19, 14, 8, 22, 19] decode PRESSURE,LOG_VOL,DEV,DIV,FOMO,TS_RANK... | 0.000/0.0000 | 0.000 (0.000) | 0.000 (0.000) | 0.000 (0.000) | 0.000 (0.000) |
| 9 | random | [2, 0, 6, 20, 15, 2, 14, 20, 1, 11, 17, 13] decode PRESSURE,RET,ADD,TS_RANK,DECAY,PRESSURE... | 0.000/0.0000 | 0.000 (0.000) | 0.000 (0.000) | 0.000 (0.000) | 0.000 (0.000) |


- 背景 baseline (無 vol): H1 3.364 → H2 4.091 (2x 3.776) B 4.496 C 3.479 FULL 1.861 (2x 1.667), folds mean 0.860 (基於無 vol，無 Z1 vt 放大)。
- 變異中 H1-best = idx 3 `[5,2,7,2,7,11,15,4,4,6,6,10]` (`LOG_VOL PRESSURE SUB ...` FOMO→LOG_VOL): H1 3.863 → H2 5.507 (2x 5.198) B -2.511 (崩) C 4.817 FULL 1.274 — H2 飆升但 B 直接崩 -2.51 (非 fee 致)，因 LOG_VOL 替 FOMO 後短打 B 段失準。
- 其次 idx 2 (ADD→MUL): H1 3.341 H2 0.988 B 3.396 C 0.749 FULL 1.33 — H2 跌至 0.98 不達；idx 0 (FOMO→PRESSURE): H1 3.177 H2 2.993 B -1.23 C 2.44 FULL 1.43。
- 隨機 5 行 (idx5-9) 在無 vol 下 H1 亦低 (0~1.5) 且 H2 全 <1.0，證 12 長度隨機公式難湊有效堆棧路徑。
- 結論: **AD1 FAIL — 無候選在 H2 與 FULL 雙勝 baseline** (best H2 5.507 > 4.091 但 FULL 1.274 < 1.861, B -2.51 崩)，AD1_gain=False, overall_AD1_found_better=False。單點鄰域非有效搜索算子；需結構化 RL (run_q2) 或 arity-safe 交叉。

## AD2 — take-profit 掃描 (Y1b 基線 vtNone, 顯 TP 淨效應)

| tp | H2 sh (2x) | B sh (2x) | C sh (2x) | FULL sh (2x) | turnover (Δ) | folds mean/med/min/pos |
|---|---|---|---|---|---|---|
| None | 4.091 (3.776) | 4.496 (4.155) | 3.479 (3.168) | 1.861 (1.667) | 0.082523 (Δ+0.0000) | 1.909/3.167/-2.339/9/12 |
| 0.06 | 4.411 (4.086) | 4.496 (4.155) | 3.795 (3.473) | 1.719 (1.503) | 0.089970 (Δ+0.0074) | 1.860/3.321/-2.369/9/12 |
| 0.1 | 4.091 (3.776) | 4.496 (4.155) | 3.479 (3.168) | 1.799 (1.596) | 0.085410 (Δ+0.0029) | 1.897/3.167/-2.380/9/12 |
| 0.15 | 4.091 (3.776) | 4.496 (4.155) | 3.479 (3.168) | 1.762 (1.563) | 0.083435 (Δ+0.0009) | 1.884/3.167/-2.339/9/12 |


- tp=0.06: H2 4.411 (+0.32 vs None 4.091) B 平 4.496 C 3.795 (+0.316) 但 FULL 1.719 (-0.142 vs 1.861) 與 folds mean 1.10 vs 0.86 相近；tp=0.10/0.15 與 None 在 H2/B/C 近乎等效 (止盈觸發少 — short-led regime 衝高有限)。
- turnover: tpNone 0.0825 → tp0.06 0.0899 (+0.007, <0.02 門檻內但方向為增換手，因 TP 提前平倉反增 flip)。
- 要求 H2↑ 且 FULL↑ 且 Δto<0.02: **FAIL — 無 tp 同時提 H2 與 FULL**，tp0.06 的 H2 增益被 FULL 稀釋。
- 結論: TP 在當前 q0.3+ts24 short-led 下惰性 (H2 最多 +0.32, FULL 負)，維持 tp=None。

## AD3 — 跨幣種子 AVAX/SHIB (單幣 TRX-spec 0.85/0.12/cd6/0.05/ts24 q0.3)

| coin | H2 sh (2x) | B sh (2x) | C sh (2x) | FULL folds mean/min/pos | H2 side long/short |
|---|---|---|---|---|---|
| AVAX | 1.156 (0.974) | -1.318 (-1.518) | 1.935 (1.760) | 0.526/-3.342/7/12 | 2.948/0.709 |
| SHIB | -0.163 (-0.309) | 1.229 (1.056) | -0.193 (-0.340) | 1.018/-4.392/8/12 | 0.000/-0.163 |


- AVAX 單幣: H2 1.156 (long 2.948/short 0.709, long 單筆勝但 short 拖累) B -1.318 (short 爆) C 1.935；folds mean 0.526 min -3.34 n_pos 7/12 — 尾端 H2 略正但 B 崩且全歷史 2 折負 >-3，遠劣雙腿基準 4-5。
- SHIB 單幣: H2 -0.163 (無 long) B 1.229 C -0.193 folds mean 1.018 min -4.39 — H2 直接負，C 亦負。
- 要求 H2>2 且 FULL 單幣隱含 >1 (雙腿 FULL 1.86): **FAIL — 無第二梯隊**，與 W4/Z3 0/36 beats_both 一致。
- 結論: AVAX/SHIB 不入選第三腿，維持 ETC+TRX 雙腿。

## 總結

- AD1 10 變異皆有效但無一雙勝 (best H2 5.5 但 B -2.5 崩)，FAIL。
- AD2 4 個 TP 無一雙提 H2+FULL，FAIL。
- AD3 2 跨幣 H2 1.15/-0.16 不達 2.0，FAIL。
- PASS 0/3: `AD1_H1_best_beats_baseline_H2=True` (H2 5.507>4.091) 但 `AD1_gain=False`, `AD2_any_TP_gain=False`, `AD3_candidates=[]` → overall 三路皆 False。
- 鎖定不動: 保留基線 [3,2,7,2,7,11,15,4,4,6,6,10]，tp=None，第三腿 REJECT。

Artifacts: research/run_ad.py (23092B), results/backtest_AD.json (16 rows: 10+4+2, 含 per_coin_side 與 folds), logs/ad.log
復現: python3 research/run_ad.py
