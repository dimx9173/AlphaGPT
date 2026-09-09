# LIVE_CHECKLIST_E10_PATCH.md — E10 補丁：paper → shadow → live 三階段 Gate 與監控 (合併至 LIVE_CHECKLIST.md) — 2026-09-07

> 本文件為 `docs/LIVE_CHECKLIST.md` 的 **E10 補丁段**。合併方式：將下節「8. E10 收斂門檻 (paper→shadow→live)」整段追加至 `LIVE_CHECKLIST.md` 末尾作為唯一操作源，原 1-7 節不動。`strategy_manager/config.py` 鎖定參數以 `results/backtest_E10.json:locked_params` 為機器事實源，本文件為操作事實源，`docs/STRATEGY_E10.md` 為策略事實源。

---

## 8. E10 收斂門檻 (paper → shadow → live) — 2026-09-07 新增

### 8.0 鎖定 (AB/AC/AD 已證不切 short、不轉權重、不換公式)

- **FORMULA** `[3,2,7,2,7,11,15,4,4,6,6,10]` (`FOMO PRESSURE SUB PRESSURE SUB ABS DECAY DEV DEV ADD ADD NEG`) 不換。
- **每幣** `ETC 0.88/0.12/cd18/None/ts24` + `TRX 0.85/0.12/cd6/0.05/ts24`，`q0.3 long-only` (`quantile_mask_long side=both gated on long`)，`50/50`，`aster perp 2x fund0.0005 fee0.0004/0.0008`，`n=6580 4h`。
- **條件疊加** `Z1 vt0.012/w12` (ETC/TRX 同) 僅 tail，不作 global；**AA challengers** `0.10/15/9` (unbiased) / `0.12/15/9` (biased) 僅 shadow。
- **權重** Z1 生產固定 `50/50` (AC 7點細網 gain 0, `1/vol` 劣 `B-0.28`)；**第三腿/新因子/TP** 全 REJECT。

### 8.1 監控指標 (必檢門檻)

| 指標 | 門檻 | 判定 |
|---|---|---|
| **12fold mean sharpe** (12×548bar) | `>0` 且 `vs Y1b delta > -0.30` | `Y1b 1.81 vs Z1 1.655 delta -0.155 PASS watch` |
| **12fold median sharpe** | `>1.5` | `Y1b 3.178 vs Z1 2.479 delta -0.699` → 維持 Y1b 成因 |
| **fee2x B worst** (`fund[0.0003,0.0005]×fee[0.0004,0.0008] 4格` 最小) | `>0` (優 `>1`) | AA_H1 `0.354` borderline / AA_H2 `2.58` robust |
| **fee2x C worst** | `>1` | `3.46/3.62 PASS` |
| **turnover** | `<0.16` | Y1b 0.082 / Z1 0.152 / AA 0.15 `PASS`; 第三腿 0.17-0.18 `REJECT` |
| **coverage (shadow)** | `0.5-0.7` | 採用 `Y1b_main_thr1.0_w200 0.652 PASS`; `alt1.5 0.354` 太低 |
| **trailing sharpe (window 200)** | `>1` 啟 overlay | `main` 規則唯一採用；`alt` 已 REJECT |
| **mdd FULL** | `<1.0` | `Y1b 0.89 Z1 0.90 shadow 0.68` |
| **corr ETC_TRX (H2)** | `<0.20` | `0.07-0.15` 分散有效 |

### 8.2 paper 階段 (offline，無下單)

```
python3 research/run_paper2.py                # 基線
python3 research/run_e10.py                   # E10 彙整校驗 (純聚合，無重算)
```

- [ ] `pytest -q` 全綠 (2026-09-06: 112 passed)
- [ ] `paper n=6580 trades 478±5 final_x 2.98±0.15 sharpe 0.90±0.10` (實測 478/2.9883/0.899)
- [ ] `fee2x final_x>1 sharpe>0` (實測 2.0386/0.721)
- [ ] `turnover<0.16` / `12fold mean>0 median>1.5 n_pos≥8` / `corr<0.20`
- [ ] 1 週無未捕獲異常，drift 超限則 re-baseline 原因記錄

### 8.3 shadow 階段 (唯一載體 `Y1b_main_thr1.0_w200`)

- [ ] `coverage 0.5-0.7` (基準 `0.652`)，外溢即 gate 閾失配
- [ ] shadow `FULL 2.62x/0.831/mdd0.686 tr547 B3.45 C1.96` 全面勝 plain `B2.509/C1.775` 且無崩 (`plain` 單用 Y1b overlay 崩至 0.83x 故以 gate 後勝為判)
- [ ] 切換判 `trailing-200 overlay sharpe>1` (`W3 main`)，`alt` 規則禁行
- [ ] `fee2x worstB>0 worstC>1` 壓力必過
- [ ] challengers `AA H1/H2` 僅并行觀測；**升級 global 需同時** `12fold median≥1.5` + `walk-forward H1→H2 holds` + `fee2x worstB>1`

### 8.4 live 階段 (testnet → mainnet)

- [ ] `VENUES_ENABLED=aster ASTER_TESTNET=true` 小名義 `refresh_deadmen(60)` + `STOP_SIGNAL` + `reconcile` 皆 PASS
- [ ] 風控鎖定 `RISK_DAILY_LOSS_PCT=0.10 RISK_MAX_DRAWDOWN_PCT=0.15 MAX_SINGLE_EXPOSURE=1.0 VELOCITY 3/60 COOLDOWN 300 PERP_LEV 2 PERP_NOTIONAL 500 FUNDING 0.001`
- [ ] shadow 連續 2 週無異常 + `walk-forward` etc15 分支中位修復證明後，人批才 `ASTER_TESTNET=false` 以最小名義始動

### 8.5 告警與熔斷

| 觸發 | 閾 | 動作 |
|---|---|---|
| 日損 | `daily_pnl < -10%` | `CIRCUIT OPEN 300s` 阻新開倉 |
| 回撤 | `drawdown>15%` | 熔斷 + 查 `peak_equity` |
| 單筆敞口 | `notional>500 USDT` 或 `>1 SOL` | `check_perp/check_safety` 拒單 |
| 速率 | `>3/60s` per symbol | `Velocity limit` 拒單 |
| 費率 | `|funding|>0.1%` | `funding` 拒單 |
| turnover 漂移 | `>0.16` | 參數漂移告警，重檢 `vt/cd` |
| 12fold 漂移 | `mean<0 or median<1.5 or n_pos<8` | 全歷史崩潰告警，降倉或回 Y1b |
| coverage 漂移 | `<0.5 or >0.7` | gate 失配告警，切回 plain |

### 8.6 合併指令

```bash
cat docs/LIVE_CHECKLIST_E10_PATCH.md >> docs/LIVE_CHECKLIST.md   # 人工覆核後執行
# 或由 E10 腳本任務自動附至 LIVE_CHECKLIST.md §8
```

***End of E10 patch. Sources: `results/backtest_E10.json` (machine truth) + `docs/STRATEGY_E10.md` (strategy truth).***
