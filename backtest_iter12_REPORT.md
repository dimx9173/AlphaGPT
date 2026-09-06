# 策略迭代 1+2 報告：真因子 × 4h

## 方法
- 15m CSV → 4h重採樣（16根合1，6570根/幣）→ 真6因子（RET/LIQ/PRESSURE/FOMO/DEV/LOG_VOL）
- F_MOM=[RET PRESSURE ADD DECAY]（平滑動量）；F_REV=[DEV FOMO SUB]（偏離減FOMO，反轉）
- StackVM執行 → next-bar 4h收益 → 4配置 × 3種子 → 672行（`backtest_iter12.jsonl`）

## 核心結論
1. **F_REV全面優勝**：28/28幣（除TRX）反轉公式好於動量（mean -428 vs -511）；4h上均值回歸>動量追漲
2. **排名高度穩定**：基線top5(BTC/TRX/BNB/ETH/ATOM) vs 迭代top5(TRX/BTC/BNB/ETH/POL)重合4個；worst5重合4個（SUI/KAS/ICP/PEPE＋DOGE/NEAR輪換）→ **幣種偏好是真實屬性，非因子假象**
3. **絕對值不可比**：迭代fitness量級（-100~-900）遠大於基線（-7~-41），因4h單根收益×全倉每根都計、6因子訊號更密集（turnover 0.5~0.6 vs 0.06）。看排名，不看絕對值
4. **TRX/BTC/BNB/ETH前四不動**：兩套因子、兩種週期下一致最優 → 核心持倉候選
5. **PEPE墊底不動**：兩套因子都最差 → 剔除或另配策略

## 下一步（建議迭代3）
- 用12因子（ADVANCED）重跑前10幣，看VOL_CLUSTER/REL_STRENGTH能否翻轉中期幣
- F_REV變體：[DEV FOMO SUB]→加DECAY平滑，或GATE門控（高FOMO才反轉）
- 訓練真AlphaGPT公式（engine.py）在4h數據上，替代手工兩式
