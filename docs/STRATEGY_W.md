# STRATEGY_W — W1~W4 收斂 (Y1b 鎖定 + vol/權重/shadow/參數挖掘) — 2026-09-07

鎖定引擎: FORMULA [3,2,7,2,7,11,15,4,4,6,6,10], aster perp 2x fund0.0005 fee0.0004/0.0008, n=6580 4h (15m×16), 50/50 ETC+TRX。
冷數據: data/data_15m_3y -> ~/pcloud-drive/cold-data/AlphaGPT/data/data_15m_3y symlink 正常 (29 幣 209MB)。

## W1 Z1 全歷史體檢 (vt0.012/w12 vs Y1b 無vol) — KEEP Y1b

對比段 H2[6077:6570]/B[6380:6580]/C[6080:6580]/C1/C2，per-leg _vol_scale clamp0.2-2.0 roll1 post-stops pre-roll, q0.3 long-only。

| seg | Y1b sh/ann/mdd | Z1 sh/ann/mdd | Δsh | fee2x Δsh |
|---|---|---|---|---|
| H2 | 4.091/1.71/0.074 | 5.013/2.95/0.108 | +0.922 | 5.157 vs 4.0+ (推估) |
| B  | 4.496/1.74/0.067 (fee2x 4.155) | 7.042/3.38/0.071 (fee2x 6.513) | +2.546 | +2.358 |
| C  | 3.479/1.45/0.074 (fee2x 3.168) | 4.666/2.71/0.108 (fee2x 4.215) | +1.187 | +1.047 |
| C1 | 2.64 | 3.846 | +1.206 | — |
| C2 | 6.245 | 8.181 | +1.936 | — |

12折 548bar: Y1b mean1.81 median3.18 min-2.95 9/12正 | Z1 mean1.66 median2.48 min-3.26 9/12正, mean-0.155 median-0.699 worstΔ -0.877(fold10 3.307→2.43) / -0.825(fold3)。
壓力 6格 fund[0.0003,0.0005,0.001]×fee: B worst Y1b2.79 Z1 4.65 全領先+1.86~+3.73, C worst 1.87→2.62 +0.75~+1.91。
side split: long_pos_rate 0.007~0.017 僅1筆 long，short_pos_rate 0.618~0.623 驅動整體；short_only H2 3.96→4.84/B 3.93→6.17。

結論: 尾端全贏但全歷史 12折 mean/median 下滑，median -0.699。Z1 是尾端 regime 專家，非 global_best。KEEP Y1b 為 global，Z1 僅作條件疊加 (regime gate / WF 加權)。

## W2 權重矛盾調和 (4基線×8權重=32行) — 條件切換

| base | 標籤 | 最優 wETC | H2/B/C (最優) | 50/50 B+C增益 | corr(w,sh) |
|---|---|---|---|---|---|
| A | Q1 cd12/ts0 無vol 無sth改 q0.3 | 0.1 | H2 2.14/B 2.72/C1.78 FULL1.08 | +2.67 | -0.96 單調TRX-heavy |
| B | Y1b cd18/ts24 無vol q0.3 | 0.3 | H2 4.34/B4.90/C3.51 | +0.43 | 倒U |
| C | Y2 vt0.01/w12 | 0.6 | B+C 11.66 | +0.05 | +0.87 ETC-heavy |
| D | Z1 vt0.012/w12 | 0.5 | H2 5.01/B7.04/C4.67 fee2x6.51 | 0 | 50/50即最優 |

跨基線: 最優翻轉幅度 0.5 (0.1→0.6) 系統性受 cd/vol/ts/sth 驅動；無vol基線 H2/FULL 反號不穩，vol基線 FULL與B/C同號穩定 (+0.81 外推)。
推薦: 條件切換而非單一固定 — Q1→w0.1-0.2, Y1b→w0.3, Y2/Z1→固定50/50；全域折中單點選 0.3，當前最優基線 (Y1b/Z1) 以 0.5±0.1 生產。

## W3 Shadow 統一帳簿 (Y3 ledger法 per-trade netp) — 僅 Y1b_main1.0 PASS

Ledgers FULL: plain 2.9883x/0.899/mdd0.7846 tr478 longs57 | Y1b 0.8365x/0.251 | Y2 0.9714x/0.261 | Z1 0.6988x/0.096 — 三 overlay 全歷史崩，不可單用。
18組 gate (window200×thresh0.8/1.0/1.5×3overlay×main/alt, causal):

- 6/18 PASS 皆 Y1b 系: Y1b_main0.8 cov0.671 FULL0.773 B3.45 C1.96, Y1b_main1.0 cov0.652 FULL0.831 B3.45 C1.96, Y1b_main1.5 cov0.599 FULL0.744, Y1b_alt1.5 cov0.354 FULL0.708 B4.53 C2.93, Y2_alt1.5 cov0.354 FULL1.07 B4.53 C2.93, Z1_alt1.5 cov0.354 FULL0.76
- 最穩 Y1b_main_thr1.0_w200 (最小回撤 0.6861 + 最高覆蓋 0.652 於 PASS 中): FULL 2.6249x/0.831/mdd0.6861 tr547 B1.0687/3.452 C1.106/1.966 全面勝 plain B2.509/C1.775 且無崩。
- Y2/Z1 main 全 REJECT (C1.736 < plain1.775)，僅 alt1.5 擦邊屬 plain弱勢閘。

採用: Y1b_main1.0 為 W3 推薦 shadow，其餘 REJECT。含 vol 的 overlay 閘失效。

## W4 閾值×冷卻×第三腿 (Y1b+Z1 vol 基線) — 無偏 FAIL, 有偏 PASS

任務A sth[0.10,0.12,0.15]×etc_cd[12,18,24]×trx_cd[6,9,12]=27行，含 H1[5584:6077]/H2[6077:6570]/B/C/FULL + fee2x:

- 無偏 H1-best sth0.15/etc12/trx9 H1 5.131→H2 2.166 FULL2.401 → H2>3.0 FAIL (H1↔H2 負相關，過擬合)
- 有偏 H2-best sth0.10/etc18/trx9 H2 5.617/B5.731/C5.493/FULL2.338 fee2x_H2 5.157/B5.211 → H2>3 FULL>1 PASS (若放棄無偏)
- Z1 原基線 sth0.12/etc18/trx6 H2 5.013/B7.042/C4.666/FULL2.064 略低於新最優，但無偏下仍最穩。

任務B 第三腿 36行 (18 primary sth共用 + 18 extra sth0.12) 6幣[AVAX,SHIB,DOGE,BTC,SOL,BCH]×3混比(40/40/20,33/33/33,50/25/25):
- 錨無偏 H1-best: 最優 AVAX 40/40/20 H2 2.461 beats H2但 FULL2.158<dual2.401 → beats_both False
- 錨有偏 H2-best: 最優 AVAX 40/40/20 H2 5.535 < dual5.617, BCH 40/40/20 H2 4.529 < dual，全部 36 行無一雙勝，turnover 0.17-0.18 > dual0.15

結論: 與 Z3 一致，更強基線上第三腿仍不勝雙腿，維持 50/50 雙腿最優；閾值×冷卻邊際改進來自 sth0.10+cd18/9 但 H1 無法無偏挑出，推薦保持 Z1 的 0.12/18/6 或放棄無偏切 0.10/18/9。

## 總收斂

- LOCK global: Y1b (S2 q0.3 + ETC 0.88/0.12/cd18/None/ts24 + TRX 0.85/0.12/cd6/0.05/ts24, 50/50) — tail H2 4.43/B4.50/C3.48 fee2x 4.15，全歷史 2.99x/0.90 平衡，12折 9/12正 mean1.81。
- CONDITIONAL: Z1 vt0.012/w12 僅尾端條件疊加 (tail +2.5/fee2x +2.3) 但 12折 median -0.699 不作 global。
- WEIGHT: vol 基線固定 50/50，無vol 才需 TRX-heavy。
- SHADOW: 僅 Y1b_main1.0 採用 (coverage0.652 FULL2.62x/0.83 50/50 雙贏且無崩)。
- NEXT PARAM: 無偏閾值×冷卻與第三腿皆 FAIL，有偏最優 0.10/18/9 僅備選。

Artifacts: backtest_W1_z1verify.json, backtest_W2_weight.json, backtest_W3_shadow.json, backtest_W4_next.json + logs/w*.log
Scripts: research/run_w*.py (mirror run_y2/run_z1/run_y3/run_z3 engines)

下一步(已由 AA/AB/AC/AD 四路承接): 見 `docs/STRATEGY_ABCD_AGGREGATE.md` — AA 無偏首 PASS (0.10/15/9 H2 5.58) / AB 證 both>short & lth惰性 / AC 證 50/50 已最優且第三腿 0/12 / AD 證 0/3 無新賽道；W5 待做 12fold median 與 walk-forward 驗 etc15 是否翻轉 -0.699 再定是否升級 challenger，紙上 shadow 以 Y1b_main1.0 為載體并行。
