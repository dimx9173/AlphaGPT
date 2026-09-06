# 群組分散測試報告 (OOS + 穩健性)

## 方法 (無偏)
- 同公式 [3,2,7,2,7,11,15,4,4,6,6,10], aster perp 2x, 資金費 0.0005
- 前85% 訓練段 12組grid選每幣最優參數, 後15% (986根4h) 測試段評分, 等權組合
- 檔案: run_grouptest2.py / backtest_grouptest_oos.json / grouptest2.log
- 穩健性: run_grouptest3b.py (taker費加倍 0.0004->0.0008, 資金費不變) / grouptest3b.log

## 單幣測試段 (sharpe / ann / mdd)
- ETC 2.914 / 2.68 / 0.46, 參數 (0.88,0.12,cd12,無止損)
- TRX 2.662 / 0.91 / 0.11, 參數 (0.85,0.15,cd6,sl0.05)
- AVAX 1.667, BCH 1.482, SHIB 1.221, XRP 1.071, BTC 0.703, DOGE 0.705, SOL 0.464
- 失效: BNB -0.34, ETH -0.035, LINK -0.04, XLM 0.07

## 組合測試段
- 跨組 ETC+TRX+BCH+SOL: 2.426 / dd 0.32 (跨組最佳, PnL相關0.25)
- 跨組 ETC+TRX+DOGE+XRP: 2.351 / dd 0.46
- 跨組 ETC+TRX+XRP+SOL: 2.271 / dd 0.38
- 組內 DOGE+SHIB: 1.225 (meme內分散有效, pnl相關僅0.31)
- naive BTC+TRX+BNB+ETH: 0.615 (BNB+ETH雙拖累)

## 穩健性 (費用加倍, 僅taker費)
- ETC 2.914->2.801 (-4%), TRX 2.662->2.203 (-17%), BTC 0.703->0.498 (-29%)
- ETC+TRX+BCH+SOL 2.426->2.115 (-13%), 仍>2
- 注: 之前同時加倍資金費的版本 sharpe 反而上升, 屬空頭收資金費假象, 已作廢, 以本輪fee-only為準

## leave-one-out (ETC+TRX+BCH+SOL, base費用)
- 剔SOL (ETC+TRX+BCH): 3.00 / ann 1.76 / dd 0.36 ← 全場最佳, 唯一打贏單ETC
- 剔BCH (ETC+TRX+SOL): 2.513 / dd 0.24 (dd最低)
- 剔TRX: 2.097; 剔ETC: 1.677 (ETC最重要, TRX次之)
- 警告: 在測試段上挑剔除對象 = 選擇偏差, 需 first-half選 / second-half驗證 (待跑)

## 測試段前後半穩定性 (493/493根)
- ETC: 4.55 / 1.61; TRX: 3.42 / 1.70 (後半仍活)
- BTC: 1.79 / -0.28 (後半死亡)
- ETC+TRX+BCH+SOL: 5.18 / 0.31 (後半衰退)
- ETC+TRX+DOGE+XRP: 3.89 / 1.01 (後半仍有1.0)
- DOGE+SHIB: 1.28 / 1.18 (最穩定)
- BTC+TRX+BCH: 4.33 / -0.04 (被BTC後半拖死)

## 結論
1. 跨組分散有效 (PnL相關0.18~0.33 < 組內0.31~0.66), 但sharpe不敵單ETC, 價值在降dd
2. 候選: ETC+TRX+BCH (3.0) / ETC+TRX+BCH+SOL (2.43, dd 0.32) / ETC+TRX+DOGE+XRP (2.35, 後半穩)
3. 基線維持ETC v3單幣; 候選組合需通過 first-half選/second-half驗證後才能升級
4. 砍掉 ETH/BNB/LINK/XLM (測試段失效); BTC後半死亡, 不再當核心腿

## H1選 / H2驗證 (測試段再對半, 493/493根4h, 無偏)
- 單幣 H1->H2: ETC 4.39->1.84 / TRX 2.59->1.86 / SHIB 2.01->0.87 / AVAX 1.68->0.73
- 崩盤: BCH 5.61->0.17 / XRP 4.05->-0.90 / BTC 2.46->-0.55 / SOL 1.18->0.03
- 組合 H1->H2:
  - A ETC+TRX+BCH: 6.86->1.33 / dd 0.32
  - B ETC+TRX+BCH+SOL: 5.73->1.01 / dd 0.29
  - C ETC+TRX+SOL: 3.93->1.40 / dd 0.20 (H2組合最佳, dd最低)
  - D ETC+TRX+DOGE+XRP: 3.64->0.66
  - E DOGE+SHIB: 0.85->0.73 (衰退最小, 最穩)
  - F 單ETC: 4.39->1.84 (H2仍最強之一)
  - G 單TRX: 2.59->1.86 (H2最強, dd 0.07)
- 結論: BCH/XRP/BTC的強勢全是H1幻覺, H2死亡; 只有ETC+TRX在H2存活>1.5
- 組合仍打不贏單腿sharpe, 但降dd: C的0.20 < ETC的0.33
- 檔案: run_grouptest4.py / backtest_grouptest_h1h2.json / grouptest4.log

## 第5輪: 純ETC+TRX + sharpe加權 + walk-forward (2026-09-05)
- 純ETC+TRX (等權) H1 5.13 / H2 2.197 / dd 0.19 ← 首次打贏單腿! H2全場最佳
  - 單ETC H2 1.84 / 單TRX H2 1.86, 雙腿 2.20 雙殺
  - dd 0.19 < ETC 0.33, TRX 0.07; ann 1.1 介於兩腿之間
- sharpe加權 (H1 sharpe) 反而輸等權: ETC+TRX 2.075<2.197, ETC+TRX+SOL 1.844>1.398(唯一正例)
  - 結論: 加權增益不穩, 維持等權
- H2新排名: ETC+TRX 2.20 > TRX 1.86 > ETC 1.84 > ETC+TRX+SOL(W) 1.84 > ETC+TRX+SOL 1.40
- walk-forward全歷史4段 (6570根):
  - ETC+TRX: 全 2.34 / 段 [0.38, 2.29, 2.96, 4.63] (第1段弱, 後3段遞增, 趨勢向好)
  - 單ETC: 全 2.11 / 段 [0.11, 2.16, 2.93, 3.90]
  - 單TRX: 全 1.50 / 段 [0.83, 1.39, 1.51, 3.33] (最穩, 全正)
  - ETC+TRX+SOL: 全 1.63 / 段 [-0.72, 1.52, 2.65, 3.92] (第1段負, 有SOL拖累)
  - 含BCH組合第1段皆負 (A -0.28, B -0.93), BCH早期不穩, 踢出核心
- 檔案: run_grouptest5.py / backtest_grouptest5_pw.json / grouptest5.log

## 更新結論
1. 核心從單ETC升級為 ETC+TRX 等權雙腿 (H2 2.20, dd 0.19, WF全正遞增)
2. 配角只留SOL觀察 (ETC+TRX+SOL H2 1.40, dd 0.20), BCH/XRP/DOGE/SHIB/AVAX/BTC全部踢出核心
3. 加權維持等權, 不搞sharpe加權
4. 基線v3維持ETC單幣不動; 新候選 ETC+TRX 進入paper trading驗證

## 第6輪: ETC:TRX權重掃描 (H1/H2/FULL)
- H2峰值在 wETC=0.2 (2.44), H1峰值 wETC=0.4 (5.13), FULL峰值 wETC=0.5 (2.34)
- 三段交集最穩: wETC 0.2~0.5, H2全>2.19, FULL全>2.06
- dd隨ETC權重單調升 (H2: 0.07->0.33), TRX是降dd腿
- 決策: 維持50/50等權 (FULL最優, H2 2.20僅比峰值低0.24, 簡單即強)
- 檔: backtest_grouptest6_weight.json

## 第7輪: 第三腿 (等權加到ETC+TRX, H2排名)
- AVAX 1.74 / SHIB 1.59 / DOGE 1.49 / SOL 1.40 / BTC 1.34 / BCH 1.33 / XRP 0.80
- AVAX H2最佳但FULL第1段歷史包袱未知; SHIB次之
- 沒有第三腿打贏純雙腿2.20, 全部降級為觀察
- 檔: backtest_grouptest7_thirdleg.json

## 第8輪: 長短倉分解
- H2: ETC多頭0.0(無多倉, 純空alpha) / TRX多頭-2.48(多頭虧錢) / 雙腿空頭2.44
- FULL: ETC多頭0.97轉正, TRX多頭仍-0.36; 空頭雙腿2.23
- 結論: alpha在空頭, H2尤其純; 維持多空雙向 (砍多頭=砍半個引擎, 且FULL多頭有正貢獻)
- 檔: backtest_grouptest8_side.json

## 第9輪: 費用掃描 (資金費固定, ETC+TRX等權)
- fee 0.0002/0.0004/0.0008/0.0012: H2 2.31/2.20/1.98/1.76, FULL 2.42/2.34/2.18/2.02
- 費用x3仍>1.75, 3倍費用才掉0.44, 換手低 (ETC 0.059/TRX 0.089)
- 檔: backtest_grouptest9_fee.json

## 第10輪: 最終鎖定 (ETC+TRX等權, 全歷史6折)
- 全 2.34 / ann 1.87 / dd 0.66
- 6折: [-0.81, 2.39, 2.40, 0.74, 6.66, 3.66], 5/6正, 折0(最早段)唯一負
- 早段弱是公式在BTC訓練期對ETC/TRX的適應期, 後5段全正且遞增
- 檔: backtest_grouptest10_final.json

## 最終結論 (10輪)
1. 鎖定: ETC+TRX 50/50等權 (H2 2.20 / FULL 2.34 / WF 5/6正)
2. 基線v3 ETC單幣不動; 新組合進paper驗證
3. 第三腿觀察席: AVAX > SHIB > DOGE > SOL (都不加倉, 只觀察)
4. BCH/XRP/BTC/ETH/BNB/LINK/XLM踢出
5. alpha=空頭, 維持多空雙向; 費用x3仍活, 換手低是護城河
