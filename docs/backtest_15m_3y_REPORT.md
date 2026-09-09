# 30幣 × 3年 × 15m 回測報告（資深量化視角）

日期：2026-09-04 ｜ 數據：Binance 永續公開K線 ｜ 引擎：MemeBacktest（venue費率/槓桿/空頭/資金費）

## 1. 資料校驗（✅ 全綠）

- 28/28 幣零缺口、零重複、OHLC零異常、價格全正 → `validate_15m_3y.json`
- 25幣滿1095天（105121行）；KAS/POL/RENDER因上市晚自然偏短
- 2.87M根K線；TON/OM無Binance永續（缺席，非品質問題）
- PEPE/SHIB使用1000倍合約（價格需/1000還原現貨價）

## 2. 回測設計（誠實聲明）

- 每幣獨立全歷史單次通過（~105k根），動量因子=96根均值收益×放大＋噪聲
- 4配置：sol現貨1x只多 / HL perp2x多空 / Aster perp2x多空 / HL perp3x多空
- 3種子/幣/配置 → 共336行 → `backtest_15m_3y.jsonl`
- ⚠️ 測的是**框架相對強弱**，非某套Alpha很神；全負fitness＝動量在3年15m上不賺，排名才有意義

## 3. 幣種偏好（按aster2x fitness，由好到壞）

| 梯隊 | 幣 | aster2x | 特徵 |
|---|---|---|---|
| T1 抗跌 | BTC(-7.0) TRX(-8.9) BNB(-10.5) ETH(-10.5) | 最優 | 高市值、高流動、低噪聲 |
| T2 中段 | ATOM DOT LTC POL ETC AVAX SOL BCH ADA NEAR UNI LINK ARB | -11~-18 | 公鏈/DeFi主流，動量衰減中等 |
| T3 脆弱 | RENDER XLM XRP APT SHIB | -18~-20 | 敘事驅動、跳空多 |
| T4 毒藥 | HBAR SUI DOGE KAS ICP PEPE | -21~-41 | meme/新公鏈/超高波動；PEPE(-41)墊底 |

分類均值（aster2x）：majors(-10.5) < payments(-14.6) < defi_infra(-15.9) < alts(-19.6) < memes(-27.7)

**解讀**：策略偏好**低噪聲大市值幣**；meme幣15m噪聲淹沒動量訊號，不適合此類策略。

## 4.  venue/槓桿偏好

- 21/28幣最優=aster2x（費率0.04%最低）；7幣最優=sol現貨（RENDER/HBAR/SUI/DOGE/KAS/ICP/PEPE——全是T3/T4高噪幣，perp空頭被雙向打臉，不如現貨只多躺平）
- hl2x緊追aster2x（費率差0.00005）；hl3x 28幣全部最差（槓桿放大虧損，PEPE上-112.9）
- 風險均值：sol現貨MaxDD 74.7 / hl2x 26.7 / aster2x 24.3 / hl3x 40.1；perp多空反而MaxDD更低（空頭對沖了單邊熊市）
- Sharpe：aster2x(-0.59) > hl2x(-0.65) > sol(-2.42)；全負＝策略本體不賺，venue只能減傷

## 5. 週期偏好（BTC/ETH/PEPE × 15m/1h/4h，aster2x）

| 幣 | 15m fit | 1h fit | 4h fit | 趨勢 |
|---|---|---|---|---|
| BTC | -10.64 | -5.80 | -4.51 | 越粗越好 |
| ETH | -8.41 | -15.75 | -6.06 | 4h最優 |
| PEPE | -43.2 | -47.6 | -39.4(cum+0.6轉正) | 4h唯一轉正 |

窗口敏感度（BTC）：24根(-8.7) vs 96根(-10.6) vs 384根(-6.2) → 4天窗口最優
換手率隨週期放大單調降（15m .059 → 1h .057 → 4h .053）

**解讀**：15m動量=噪聲交易；4h動量=趨勢跟隨。策略偏好**4h及以上**；meme幣只有4h能勉強轉正。

## 6. 交易建議（按偏好排序）

1. 幣：BTC TRX BNB ETH ATOM DOT LTC（T1/T2前段）；禁PEPE/SUI/ICP/KAS/DOGE（此策略下）
2. venue：Aster 2x多空（費率最低）；高噪幣改現貨只多或直接剔除
3. 槓桿：≤2x；3x在虧損策略上只是加速自殺
4. 週期：4h主訊號，15m僅執行；96根(24h)窗口居中，384根(4天)更穩
5. 下一步：換真Alpha因子重跑；PEPE類需另配wildlife策略（breakout+波動率過濾），勿用動量均值回歸

## 交付

- `data_15m_3y/` 28幣CSV（2.87M根）
- `validate_15m_3y.json` 校驗報告
- `backtest_15m_3y*.jsonl` 336行明細＋`backtest_15m_3y.jsonl`合併版
- `run_backtest.py` 可重跑（單幣參數）
