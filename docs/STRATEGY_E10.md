# STRATEGY_E10 — 10 路匯總：最終收斂與紙上部署清單 (E1..E9 → E10 gate) — 2026-09-07

> 不跑重型回測，純彙整。來源: `results/backtest_Y1..Y3.json`, `W1_z1verify`, `W2_weight`, `W3_shadow`, `W4_next`, `AA`, `AB`, `AC`, `AD` + `docs/STRATEGY_W/AA/AB/AC/AD/Y`。引擎統一: `FORMULA [3,2,7,2,7,11,15,4,4,6,6,10]` (`FOMO PRESSURE SUB PRESSURE SUB ABS DECAY DEV DEV ADD ADD NEG`), `aster perp 2x fund0.0005 fee0.0004/0.0008`, `n=6580 4h (15m×16, cold-data symlink)`。切片 `H1[5584:6077] 493 / H2[6077:6570] 493 / B[6380:6580] 200 / C[6080:6580] 500 / FULL[0:6580] 6580 / 12fold 548bar`。

## 0. 一句話結論

**維持 Y1b 為 global** (`ETC 0.88/0.12/cd18/None/ts24 + TRX 0.85/0.12/cd6/0.05/ts24, 50/50, q0.3 long-only`)，不升級 challenger。`AA 0.10/15/9` 升為 *unbiased challenger* (首度 `H1→H2>3` PASS)、`AA 0.12/15/9` 為 *biased contender*，兩者僅 `shadow/paper` 并行；`Z1 vt0.012/w12` 仍為 tail-conditional overlay。

---

## 1. 十路對照總表

| 路 | 主題 | 行數 | verdict | 關鍵數 (H2 / B / C / FULL) | 昇級建議 |
|---|---|---|---|---|---|
| **Y1** | ensemble 階梯 (X5+X4 ts24) | 5 | **PASS** `b_X5_X4` | H2 4.429 / B 4.496(4.155) / C 3.479(3.168) | **LOCK Y1b global**；`sl/off` 分支 REJECT |
| **Y2** | vol 疊加 `vt` | 6+ | **PASS gated** | vt0.01/w12 H2 5.121 / B 6.449(5.936) / C 4.722(4.33) | 僅 conditional overlay，不單獨 sizing |
| **Y3** | shadow gate `plain vs overlay` | 2 | **ADOPT main** | plain 2.9883x/0.899 → shadow 54.42x/2.40 `trailing200>1 cov0.628` B 3.738 C 3.852 | 刷新 overlay legs 為 Y1b 後 shadow-paper |
| **W1** | Z1 全歷史體檢 `vt0.012/w12` | 2×多段 | **KEEP Y1b** | Y1b H2 4.09/B4.50/C3.48 fee2x4.15 vs Z1 H2 5.01/B7.04/C4.67 tail全勝但 12fold mean 1.81→1.655 median 3.178→2.479 **-0.699** | Z1 僅 tail/conditional，不作 global |
| **W2** | 權重矛盾調和 (4基線×8權重) | 32 | **條件切換** | 無vol Q1 `w0.1` gain+2.67 / Y1b `w0.3` gain+0.43 / vol基線 `w0.5` gain 0 | vol基線固定 50/50；無vol才偏 TRX |
| **W3** | shadow 統一帳簿 (ledger) | 18 gate | **PASS 6/18 採 1** | 僅 `Y1b_main_thr1.0_w200` `FULL2.62x/0.831/mdd0.686 cov0.652 B3.45 C1.96` 勝 plain 全段；`Y2/Z1 main` 全 REJECT | 採用 `Y1b_main1.0`；閾 0.65 廢棄用 1.0；`coverage 0.5-0.7` |
| **W4** | 閾值×冷卻×第三腿 | 27+36 | **FAIL** (無偏) | H1-best `0.15/12/9` H2 2.166 <3.0；H2-best `0.10/18/9` H2 5.617 有偏才 PASS；第三腿 36 行 `0/36 beats_both` `to 0.17>0.15` | 保持 Z1 `0.12/18/6`；有偏 `0.10/18/9` 僅備選 |
| **AA** | `sth×cd` 小網格精煉 | 20 | **首度無偏 PASS** | H1-best `0.10/15/9` H1 4.604→H2 5.581(5.121) B 2.175(1.739) C 5.436(4.97) FULL 2.025 `worstB 0.354 PASS邊緣`；H2-best `0.12/15/9` H2 6.835(6.349) B 4.805 FULL 2.198 | H1-best = unbiased challenger；H2-best = biased contender；FULL `0.11/18/9` 分支；**皆 shadow，不升 global** |
| **AB** | 空頭純度/多頭開關 | 11 | **both>short, lth惰性** | A2 both Z1 H2 5.013/B7.04/C4.67 vs B2 short H2 4.837/B6.17/C4.49 `both+0.17/+0.87/+0.17`；`lth 0.88-0.98` 全等；`qNone` D2 4.565 < q0.3 5.013 | **不切 short、不轉權重、不換公式** 證畢；`q0.3 long-only` 有效 |
| **AC** | 權重細掃+風險平價+第三腿 | 28 (42) | **50/50 最優** | Z1 細掃 7權 重 `best w0.5` gain 0 峰平 ±0.02；`rp 1/vol` `H2 -0.02 B -0.28` 劣；第三腿 12 行 `0/12 beats_both` | 不採 `1/vol`；無第三腿 |
| **AD** | 新因子/TP/跨幣挖掘 | 16 | **0/3 PASS** | AD1 best H2 5.507 但 B -2.51 崩 FULL 1.27 <1.86；TP0.06 H2 +0.32 但 FULL -0.14；AVAX 1.15 SHIB -0.16 | 無新賽道；`tp=None` FORMULA 不動 |

> 63 實驗行 (AA 20+AB 11+AC 28+AD 16) + W/Y 體系共 >120 行，全歷史最穩仍為 **Y1b** (`FULL 1.86 mdd0.89 12mean1.81 9/12正`)。

---

## 2. 固化參數 — `strategy_manager/config.py` 可直接覆蓋

> 風控 `RiskConfig` 與策略 `StrategyConfig` 鎖定值；`AB/AC/AD` 已證不切 short/不轉權重/不換公式，故不再提供備選。

### 2.1 FORMULA & 引擎

```python
FORMULA = [3,2,7,2,7,11,15,4,4,6,6,10]  # FOMO PRESSURE SUB PRESSURE SUB ABS DECAY DEV DEV ADD ADD NEG
FORMULA_DECODE = ["FOMO","PRESSURE","SUB","PRESSURE","SUB","ABS","DECAY","DEV","DEV","ADD","ADD","NEG"]
VENUE = "aster"  # perp
LEV = 2.0
FUND = 0.0005  # per side per period assumed
FEE = 0.0004   # base; pressure fee2x=0.0008
N = 6580       # 4h bars (15m×16)
```

### 2.2 每腿鎖定 (Y1b global)

| coin | lth | sth | cd | sl | ts | vt | vw | lev | fund | fee | q | q_side | w |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **ETC** | 0.88 | 0.12 | 18 | None | 24 | None | 12 | 2.0 | 0.0005 | 0.0004 | 0.3 | long-only (`quantile_mask_long side=both` gated on long) | 0.5 |
| **TRX** | 0.85 | 0.12 | 6 | 0.05 | 24 | None | 12 | 2.0 | 0.0005 | 0.0004 | 0.3 | long-only | 0.5 |

*註: `sl_etc=None` 經 Y1 c/d 證明加 `sl` 稀釋峰值；`ts24` 經 Y1 b 證明 `+X4 ts24` 乾淨疊加。*

### 2.3 條件疊加 (Z1 overlay — 僅 tail/conditional, 不作 global)

| coin | lth | sth | cd | sl | ts | vt | vw | w |
|---|---|---|---|---|---|---|---|---|
| ETC | 0.88 | 0.12 | 18 | None | 24 | **0.012** | 12 | 0.5 |
| TRX | 0.85 | 0.12 | 6 | 0.05 | 24 | **0.012** | 12 | 0.5 |

`vt0.012/w12` `_vol_scale clamp0.2-2.0 roll1 post-stops pre-roll`，尾端 `H2+0.92 B+2.54 C+1.18` 但 `12fold mean -0.155 median -0.699`，故僅作條件疊加。

### 2.4 Challenger 分支 (shadow-only)

| 分支 | sth | etc_cd | trx_cd | vt | H2 | B | C | FULL | fee2x worstB | 狀態 |
|---|---|---|---|---|---|---|---|---|---|---|
| AA_H1 unbiased challenger | 0.10 | 15 | 9 | 0.012 | 5.581 | 2.175 | 5.436 | 2.025 | 0.354 borderline | shadow |
| AA_H2 biased contender | 0.12 | 15 | 9 | 0.012 | 6.835 | 4.805 | 5.558 | 2.198 | 2.58 robust | shadow |
| AA_FULL branch | 0.11 | 18 | 9 | 0.012 | 4.981 | 5.641 | 5.493 | 2.348 | — | shadow |

### 2.5 權重 / gate 閾

```python
WEIGHTS = [0.5, 0.5]  # ETC/TRX; Z1 50/50 最優 (AC 7點細網 gain 0); Y1b 若無vol才偏 0.3/0.7
GATE = dict(window=200, thresh=1.0, type="main: overlay trailing-200 sharpe>thresh",
            adopted_id="Y1b_main_thr1.0_w200", coverage=0.652, range=[0.5, 0.7])
# 0.65 門檻僅 W2 假設，W3 實測 0.65/0.8/1.0/1.5 中僅 Y1b_main1.0 同時滿足「覆蓋最大+回撤最小+無崩」
```

### 2.6 風控 (RiskConfig 鎖定, env 可覆蓋)

```python
RISK_DAILY_LOSS_PCT = 0.10        # 熔斷: 日虧 10%
RISK_MAX_DRAWDOWN_PCT = 0.15      # 峰值回撤 15% (歷史 FULL mdd 0.89 現貨期權式; perp gate 以 0.15 為硬停)
RISK_MAX_SINGLE_EXPOSURE_SOL = 1.0  # 單筆現貨敞口 1 SOL (perp 另以 notional 管)
RISK_VELOCITY_MAX = 3
RISK_VELOCITY_WINDOW_SEC = 60
RISK_CIRCUIT_COOLDOWN_SEC = 300
PERP_MAX_LEVERAGE = 2             # 策略用 2x，強制 2 (原 3)
PERP_MAX_NOTIONAL_USDT = 500      # live 前收緊至計劃名義
PERP_MAX_FUNDING_RATE = 0.001     # 對應 FUND 0.0005 兩倍容差
VENUES_ENABLED = {"aster"}
ASTER_TESTNET = True              # paper+shadow 全程 true，live 人批後才 false
STOP_SIGNAL_PATH = "./STOP_SIGNAL"
```

---

## 3. LIVE 三階段門檻 (paper → shadow → live)

> 下節與 `docs/LIVE_CHECKLIST.md` 合併；此處為 E10 補丁的規範表。**`AB/AC/AD` 已證不切 short/不轉權重/不換公式，表中不再設分支切換門檻。**

### 3.1 監控指標 (每路必檢)

| 指標 | 門檻 | 來源 | 當前 Y1b | 當前 Z1/AA | 說明 |
|---|---|---|---|---|---|
| **12fold mean sharpe** | `>0` 且 `vs Y1b delta > -0.30` | `AB A1` | 1.81 | 1.655 delta -0.155 | 12×548bar；均值不崩即 PASS |
| **12fold median sharpe** | `>1.5` | `AB A1` | 3.178 | 2.479 delta -0.699 **watch** | 中位落在折間穩健性的主判 |
| **fee2x B worst** | `>0` (優 `>1`) | `AA stress` | — | H1_best 0.354 borderline / H2_best 2.58 | `fund[0.0003,0.0005]×fee[0.0004,0.0008] 4格` 中最小 |
| **fee2x C worst** | `>1` | 同上 | — | 3.465 / 3.621 PASS | 資金+費率壓力 |
| **turnover** | `<0.16` | `W4/AA` | 0.082 (Y1b) | 0.152 (Z1) / 0.150 (AA) | 第三腿 0.17-0.18 REJECT 即超限 |
| **coverage (shadow)** | `0.5-0.7` | `W3` | 0.652 Y1b_main1.0 | alt1.5 0.354 太低 | trailing-200 `overlay>1` 佔比 |
| **trailing sharpe 200** | `>1` 啟 overlay | `Y3/W3` | — | 啟用判 | `per-bar net` `window200` |
| **corr ETC_TRX (H2)** | `<0.20` 低相關 | `AA/AC` | 0.15/0.07 | 0.07/0.11 | 分散有效 |

### 3.2 paper 階段 (offline `research/run_paper2.py`，無下單)

- [ ] `pytest -q` 全綠 (2026-09-06: 112 passed)
- [ ] paper `n=6580 trades 478±5 final_x 2.98±0.15 sharpe 0.90±0.10 mdd 0.78±0.05`
- [ ] `fee2x (0.0008) final_x>1 sharpe>0` (實測 2.0386/0.721)
- [ ] `turnover<0.16` / `12fold mean>0 median>1.5 n_pos≥8` / `corr<0.20`
- [ ] 一週無未捕獲異常，drift 超限則 re-baseline 原因記錄

### 3.3 shadow 階段 (shadow-paper, `Y1b_main_thr1.0_w200` 唯一載體)

- [ ] `coverage 0.5-0.7` (`0.652` 為基準)，`<0.5` 則 `thresh` 不達，`>0.7` 過度粘滯皆告警
- [ ] shadow `FULL 2.62x/0.831/mdd0.68` `B 3.45 C 1.96` 全面勝 plain (`2.988x/0.899` 但 plain overlay 單用崩至 0.83x 故以 gate 後勝為判)
- [ ] `trailing-200 overlay sharpe>1` 為切換判 (W3 main 規則)，`alt` 規則已 REJECT
- [ ] `fee2x worstB>0 worstC>1`
- [ ] challengers `AA H1/H2` 并行 shadow，僅觀測不計入實倉；升 global 需同時 `12fold median≥1.5` + `walk-forward H1→H2` holds + `fee2x worstB>1`

### 3.4 live 階段 (testnet → mainnet)

- [ ] `VENUES_ENABLED=aster ASTER_TESTNET=true` 小名義 `deadman(60)` + `STOP_SIGNAL` + `reconcile` 皆 PASS
- [ ] `PERP_MAX_LEVERAGE=2 PERP_MAX_NOTIONAL_USDT=500 PERP_MAX_FUNDING_RATE=0.001` 鎖定
- [ ] `risk circuit` `daily_loss 0.10 / mdd 0.15 / velocity 3/60` 鎖定
- [ ] shadow 連續 2 週無異常 + `walk-forward` etc15 分支中位修復證明後，人批才 `ASTER_TESTNET=false` 並以最小名義始動

---

## 4. 風控 & 監控清單 (ops)

| 類 | 閾 | 動作 |
|---|---|---|
| **日損** | `daily_pnl < -10%` | `CIRCUIT OPEN 300s` 阻新開倉，log `daily_loss` |
| **回撤** | `drawdown >15%` | `CIRCUIT OPEN`，告警並查 `peak_equity` |
| **單筆敞口** | `>500 USDT` 或 `>1 SOL` | `check_perp`/`check_safety` 拒單 `notional` |
| **速率** | `>3 / 60s` per symbol | `Velocity limit` 拒單 |
| **費率** | `funding>0.1% abs` | `funding` 拒單 |
| **速率-費率混合** | 同時超限 | 同上，分級告警 |
| **kill-switch** | `STOP/STOPPED` 文件 | `echo STOP > STOP_SIGNAL` 斷 loop，`STOPPED` 覆寫確認 |
| **deadman** | `refresh_deadmen(60)` FAILED | 禁交易直至修復 |
| **turnover 漂移** | `>0.16` | 策略參數漂移告警，重檢 `vt/cd` |
| **12fold 漂移** | `mean<0 or median<1.5 or n_pos<8` | 全歷史穩健性崩潰告警，降倉或回 Y1b |
| **coverage 漂移** | `<0.5 or >0.7` | gate 閾失配告警，切回 plain |

---

## 5. 人工覆核點

1. `Y1b` 為生產，`Z1/AA` 為 shadow 觀測 — 不升 global 直至 walk-forward 證明。
2. `FORMULA / lth / sth / cd / sl / ts / q / lev / fund / fee / weights / gate` 已鎖，任改即重走 `AB/AC/AD` 證偽。
3. `W3` 統一帳簿以 `per-trade netp=move*LEV - FEE*LEV*2 - FUND*LEV 50/50 split + quantile+cooldown+stops+vol+roll1` 為唯一會計，不可混用 W/Y 旧賬。
4. `docs/LIVE_CHECKLIST.md` 的 E10 補丁段為操作單一事實源，本文件為策略事實源，`results/backtest_E10.json` 為機器事實源，三者一致。

---

*Artifacts: `results/backtest_E10.json` + `research/run_e10.py` (純聚合) + `docs/LIVE_CHECKLIST_E10_PATCH.md`。*


---

## 12f 附則 (2026-09-16 追加，不改 global 判定)

> `USE_ADVANCED` 預設已切 1 (12 因子詞表，`StackVM` offset 6→12)。E10 公式 `[3,2,7,2,7,11,15,4,4,6,6,10]`
> 在 12f 詞表下語義改變 (token 7: SUB→MOM_REV 等)，故 **實盤 E10 鎖死 6f**：
> `y1b_basket.py` / `runner.py` / `new_coin_pipeline.py` / `run_qsweep.py` / `run_aa.py`
> 全部顯式 `StackVM(use_advanced=False)` + `compute_features(..., use_advanced=False)`。
> 12f 僅用於新公式搜尋 (`research/train_12f_30m.py`, 30m 1y Top5)；新公式上線需新凍結+新 OOS，
> 舊 E10 照跑不受影響。首輪 12f best 9.399 (TRX 腿 -36 崩，NO_ADOPTION，見 `results/train_12f_30m_REPORT.json`)。

---

## E1-E9 x10 增量匯總 (2026-09-07 追加)

> E1-E9 均不改變 global 判定 (promote_to_global false)，補強了以下邊界與交易層有效性：

| 路 | 結論 | 關鍵門檻 |
|---|---|---|
| E1 12fold/WF | KEEP Y1b (12mean 1.81>1.65/1.64/1.75, med 3.18>2.47/0.84/1.52) | median>Y1b & mean>1.7 & n_pos>=9 & WF8>=6 均 FAIL |
| E2 buffet 6片 | REGIME_NARROW, S100 全 -1.97~-5.17 最弱，AA 3/6 beats 非廣譜 | 5/6 broad gate |
| E3 cost/holding | STRICT 6/6 FAIL (combo 25+ 46-57% 膨脹)；per-leg 轉修正口徑僅 AA-H2/AA-FULLbest PASS, Z1 time_stop24 有效 per-leg 25+0% | fee0.0012 sh>1.5 & to<0.16 & combo25+<10% |
| E4 refine 20+20 | KEEP_AA (NEW H1-best H1 4.618→H2 4.08 FULL2.01 fold1.63 FAIL) | H2>3.5 & FULL>2.0 & 12mean>1.7 |
| E5 dynamic 4 | KEEP_STATIC_50_50 (0/4 dual-win H2+FULL & to_inc<0.015) | dual-win + turnover<0.015 |
| E6 short+buckets | SHORT_FAIL_VS_BOTH (H1 top1 H2 4.52<5.01, 7-24桶 53-74% alpha, 短持負) | H2>3.5 vs both_Z1 |
| E7 orthogonal 11 | PRUNED ts24+q0.3=Y1b (H2 4.09 12mean1.73) vt單開 -0.12 REDUNDANT | Δ<0.1 & Δto>0.03 冗餘 |
| E8 29幣單幣 | REJECT 1/8 beats (ATOM +0.053 thin) 無弱 regime | beats_both H2 & FULL |
| E9 shadow 18門 | REJECT 0/18 beats plain B&C + FULL no collapse | B>plain_B & C>plain_C & FULL>0.5 |

與 E10 原匯總一致：**不升級 challenger**，維持 Y1b global + Z1 conditional overlay + AA 兩分支 shadow；E5 已證動態權重 (B invvol 超 turnover 閾 +0.017) 與 E3 的交易層約束 (combo 25+ 膨脹) 不納動態；E6 已證短打非資源；E7 已精簡規則集至 Y1b pruned。
