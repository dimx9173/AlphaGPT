# PRP: P0~P2 策略迭代優化 + 驗收 (AlphaGPT Top5 demo)

Goal: 使用 PRP 流程,將 Oracle/Librarian 評審的 P0~P2 共 11 項完成並做驗收。
Repo: /home/brian/project/AlphaGPT, branch fork/brian. Python: /home/linuxbrew/.linuxbrew/bin/python3.
Baseline: 166 passed. Demo: Bybit api-demo Top5 (ETC/APT/KAS SHORT, TRX/ATOM flat), cron 每小時, heartbeat 12h.
約束: 不動 E10 FORMULA; 不動 live 安全鏈 (STOP/熔斷/deadman/gate); demo 名義維持每邊 10U, 2x; 所有改動預設 OFF, 用 env 開關。

## P0-1 執行對齊 (E08/E09)
背景: 4h 訊號每小時 poll = 同訊號重讀 4 次,回測與實盤是兩個系統。
任務:
1. hourly 腳本加決策門: 只有 UTC %4==0 的整點才允許開/翻倉, 其餘整點只做風控 (sync 平倉 + gate 檢查, 不開新倉)。用 env Y1B_DECISION_ALIGN=1 開啟, 預設 OFF。
2. hourly JSONL 每筆記 signal_age 小時數 (訊號 bar 距今)。
3. 回測對照: 同訊號源跑 X(4h對齊) vs Y(現狀1h全量), 比較 net sharpe/turnover/滑點敏感性 (2/5/10bp)。
檔案: research/run_y1b_hourly.py, strategy_manager/y1b_executor.py (run_once 加 decision_only 參數), tests/test_exec_align.py (新)。
驗收: pytest 全綠; 對照報告寫入 results/exec_align.json (含 turnover 差 + slippage 斜率); decision_only 非對齊整點零開倉 (mock broker market_open 零呼叫)。

## P0-2 換倉門檻 (E14/E15)
背景: 訊號翻轉就調 = 假設交易免費; sigmoid 邊界抖動會連翻。
任務:
1. executor 加三門 (env 各自開關, 預設全 OFF): Y1B_HYST_EPS (如 0.1, sigmoid 需超閾且持續 1 根才翻)、Y1B_MIN_HOLD_BARS (如 2 根 4h, 風控止損不受限)、Y1B_COST_K (如 2, 預期邊際收益 > k*(fee2x+5bp+funding) 才換)。
2. 回測 frontier: epsilon {0,0.05,0.1,0.15,0.2} x 最小持有 {0,1,2根}, fee2x+5bp+funding 實值成本, 畫 turnover vs net sharpe 取 knee。
檔案: strategy_manager/y1b_executor.py, research/run_swap_frontier.py (新), tests/test_swap_gate.py (新)。
驗收: pytest 全綠; results/swap_frontier.json 含 knee 點; knee 處 turnover 較現狀 -20% 以上且 net sharpe 不跌; 滑點斜率減半。

## P0-3 凍結 OOS + 參數高原 (E01/E02)
背景: 全樣本選參再同樣本驗, 有洩漏; 5 組手配可能是 noise peak。
任務:
1. 定 OOS 起點: 以 git 取得 E10 lock 日 (2026-09-07) 對應 bar index, 之後數據為凍結 OOS, 選參禁用。寫成 research/oos_freeze.json (含 cut index + 規則)。
2. 高原掃描: ETC/TRX 閾值 ±20% 各 5 點網格, 畫 sharpe 熱圖, 判孤立峰 vs 成片高原。
3. permutation: 每幣 200 次 signal 打亂重算, 實測 sharpe 在 null 分布分位 + Deflated Sharpe (trials=5x幣數)。
檔案: research/run_plateau.py (新), research/oos_freeze.json (新), tests/test_oos_freeze.py (新)。
驗收: pytest 全綠; results/plateau.json (高原判定 PASS/FAIL + 熱圖數據); results/permutation.json (p<0.05, DSR>0.8); 若任一 FAIL 則 P1/P2 暫停, 先修。

## P1-1 逆 vol 加權 (E06/E07)
背景: 等權讓高 vol 幣主導組合; 固定 2x 使實質槓桿隨 vol 漂。
任務 (需 P0-3 PASS 才啟動):
1. 加權模式 env Y1B_WEIGHT_MODE=equal|invvol|invvol_cap, 預設 equal。invvol 用過去 60 根 4h realized vol 倒數, 單幣 clip 10~35%; invvol_cap 再加組合年化 vol target (env Y1B_VOL_TARGET, 預設 0.35) 反推總槓桿。
2. 三版對照 A/B/C: FULL/H2 sharpe、maxDD、dd 貢獻分解、turnover 增量 (扣成本比 net); 再跑 fee2x+funding 四格 + 12fold。
檔案: strategy_manager/y1b_basket.py (WEIGHTS 計算), research/run_weight_modes.py (新), tests/test_weight_modes.py (新)。
驗收: pytest 全綠; results/weight_modes.json; 若 B/C 全面勝 A (sharpe+ 且 maxDD-20%+) 則切換預設為勝者, 否則維持 equal 並記錄已知次優。

## P1-2 回撤三層剎車 (E10/E11)
背景: 2x + 無 vol scaling + long bias + 慢 DECAY = 反轉段滿倉吃滿; dd 1-4。
任務 (需 P0-3 PASS):
1. B1 單幣慢速 vol targeting (60 根 4h, env Y1B_VOL_TS, 預設 OFF); B2 組合 vol cap (超標等比降倉, env Y1B_PORT_CAP); B3 time-stop/組合 trailing (訊號停滯 N 根無利潤降半倉)。
2. dd 歸因: Top5 歷史最大 3 次 dd 區間, 每幣 PnL 貢獻 + 當時 vol + 區間 corr 矩陣。
檔案: strategy_manager/y1b_basket.py, strategy_manager/y1b_executor.py, research/run_dd_brake.py (新), tests/test_dd_brake.py (新)。
驗收: pytest 全綠; results/dd_brake.json: maxDD -30% 以上、sharpe 不跌、final 損失 <20%; 歸因表含單幣貢獻>50% 或區間 corr>0.6 的確診。

## P1-3 q-sweep (E04/E05)
背景: q0.3 可能是截掉熊段的 beta 假象。
任務: q {0,0.1,0.2,0.3,0.4,0.5} 全掃, 每 q 報 FULL/H2 sharpe、turnover、long/short 筆數與 PnL 佔比、牛熊分段 sharpe (BTC 200d 分段); fee2x 下 net sharpe 曲線找 knee。
檔案: research/run_qsweep.py (新), tests/test_qsweep.py (新, 輕量: 只驗腳本輸出 schema, 重算由腳本做)。
驗收: pytest 全綠; results/qsweep.json; 若 q0.3 是孤立峰則改取 knee 點並更新 config 註記, 否則維持並記錄差異<15% 穩健。

## P2-1 納幣白名單 (E12/E13)
背景: PEPE/ICP 失效證明是體制策略; 需先分類再納幣。
任務: 每幣 4h 算 Hurst、lag1~4 自相關、trend SNR、Amihud illiq、funding vol、跳空率; 有效組 vs 失效組找分開指標, 定門檻; PEPE 最優參數套回 ETC/TRX 反證。
檔案: research/run_suitability.py (新), results/suitability.json (新, 含門檻 + 每幣 PASS/FAIL), tests/test_suitability.py (新)。
驗收: pytest 全綠; suitability.json 門檻能分開有效/失效組 (無重疊); 反證顯示參數體制專屬。

## P2-2 全網格粗掃 (Librarian 3a)
背景: 手配 5 組可能是運氣; E13 證明 cd+2 單步就能修 worstB, 粗掃 ROI 最高。
任務 (需 P0-3 PASS): sth 0.08-0.15 step0.01 x cd 6-24 x ts 12/24/36 x q 0.25/0.3/0.35 約 200-500 組, 目標函數=12fold median - λ*turnover - μ*max_dd; 畫 sth-cd 熱力圖; White Reality Check p<0.05。
檔案: research/run_grid_coarse.py (新), tests/test_grid_coarse.py (新)。
驗收: pytest 全綠; results/grid_coarse.json: top10% 參數連片 (高原) 或孤島 (需貝葉斯) 的判定; 手配是否在高原上的結論。

## P2-3 新幣自動漏斗 (Librarian 5)
背景: 加幣靠人工研究, 天級; 需變小時級管線。
任務: scripts/new_coin_pipeline.py, 輸入 SYMBOL, 輸出 L0 (資格: 上市>180天、缺失<1%、日均名義、funding 可得、非黑名單) /L1 (3 套模板 cd6/12/18 跑分, final_x>1/sharpe>0/fee2x B>0 C>1)/L2 (逐幣加入 basket, FULL/H2 不降、mean_corr<0.4、turnover<0.16、12fold median 不破 1.5) 三頁報告 + 與 Top5 corr 行。
檔案: scripts/new_coin_pipeline.py (新), tests/test_new_coin_pipeline.py (新)。
驗收: pytest 全綠; 對 PEPE 跑出 L1 拒收, 對 ATOM 跑出 L2 通過; 每月全市場掃一次的 cron 行 (先寫入 plans, 不啟用)。

## P2-4 Demo→live 晉級 + 微結構門 (Librarian 2/4/6)
背景: shadow 14 天只有 6-7 筆, 噪音; 缺微結構與營運門。
任務:
1. 晉級標準寫成 tests/test_promotion_gate.py: 30 天 + 20 筆平倉 + 覆蓋 funding>0.001 區間 + demo 淨 sharpe>0.5 + dd < H2 mdd 1.5x + slippage ≤ 假設 1.5x + 對帳零差異 + 熔斷/deadman 各演練一次。
2. 微結構門 research/run_micro.py: 滑點 ±2/5bps 重算、拒單率仿真、deadman 空窗、venue 可攜 (aster/bybit 雙口徑 fee2x)。
3. Deflated Sharpe (trials≈50-100) 寫進 Gate2 並列條件 DSR>0.8。
檔案: tests/test_promotion_gate.py (新), research/run_micro.py (新)。
驗收: pytest 全綠; 當前 demo 狀態跑晉級測試預期 FAIL (筆數不夠) 並明確列出缺口; 微結構報告 results/micro.json。

## 全局驗收 (PRP exit)
1. pytest 全綠 (166+ 新增)。
2. 每步 results/*.json 齊全且 gate 欄位明確 PASS/FAIL。
3. P0-3 FAIL 則 P1/P2 結論全部標「待定」, 不得上 demo。
4. docs/LIVE_CHECKLIST.md 追加 P0P2 驗收章節, git commit 每步一顆。
5. 任何時刻 echo STOP > STOP_SIGNAL 整點跳過; demo 名義不變。
