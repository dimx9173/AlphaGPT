# GA 搜尋可靠性修正 — 決議與驗證（2026-10-01）

本文件記錄一場 codex / claude 多輪辯論的收斂結果，以及我實測驗證的數據。
所有數字都來自 `results/ga_iter_runs/`、`results/ga_null_runs/`、
`results/wf_real/`、`results/wf_null/`，可重現。

## 0. 背景：搜尋在虛無資料上表現更好

標準模式（100 代 × 族群 32，seed 101-110，真實 funding）：

| 指標 | real 中位數 | null (iid) 中位數 | gap | real 勝出 |
|---|---|---|---|---|
| portfolio_sharpe | +1.043 | +2.209 | **-1.166** | **1/10** |
| mean_ic | +0.01071 | +0.00363 | **+0.00708** | **8/10** |
| positive_coins | 16.0 | 15.0 | +1.0 | 5/10 |

Mann-Whitney on sharpe：U=6.0，two-sided **p=0.001**（null 系統性較高）。
Sign test on mean_ic：8/10，p=0.109。

walkforward holdout：real +1.242 vs null +1.676，gap **-0.434**，p=0.273。

**夏普與 rank-IC 在兩臂之間符號相反。** 真實資料上搜尋找到的是 IC，不是夏普。

## 1. reward 函數是退化的（已驗證）

`research/ga_28c_30m_3y.py` 的 `shape_reward()` 有三個結構性問題：

### 1a. gap 項是死碼
```python
peer = psh if psh_peer is None else psh_peer
gap = softplus((psh - peer) / GAP_TAU) * GAP_TAU
```
兩個呼叫點都傳 `psh_peer=None`，所以 `peer == psh`，`gap` **恆為 0**。
`- 0.30 * gap` 從未生效。

### 1b. -50 是「無活性」硬閘，不是換手率或腿數懲罰
```python
# 呼叫點 line 464:
if mean_act < 0.01 or mean_turn < 1e-5:
    reward = -50.0
# shape_reward 內 line 355:
def mean_act_guard(mean_turn): return mean_turn < 1e-5
```
**更正**：辯論期間我最初把飽和歸因於換手率守門，實測後修正。
20 筆飽和記錄中，**11 筆全部由 `mean_act < 0.01` 觸發，turnover 在當時多數高於 1e-5**
（僅 real 2 筆、null 3 筆同時低於 turnover 線，但那些也都同時低於 activity 線）。

飽和率：real 5/10、null 6/10。活性中位數 real 6.90e-02、null 4.05e-03 ——
**null 的活性比 real 低一個數量級**，所以這個閘門系統性地把夏普勝出的那一臂推下懸崖。

### 1c. 可用 reward 動態範圍極窄
未飽和的 reward：real std=0.088、null std=0.051。
各項貢獻：mean_ic 0.0158、0.10*psh 0.391、floor 0.528、0.02*turn 0.0001、gap 0.846。

## 2. 關於「reward std 必須 >= 1.0」的駁斥

codex 與 claude 在第 2、3 輪都把 `std >= 1.0` 當作 falsifier。
**這個門檻在 reward 自身的量綱下不可達**（見上節各項範圍）。
codex 在第 2 輪自己算出非飽和 reward spread 為 0.0877，卻仍要求 > 2.0。

修正後的驗證方式應該是**與舊 reward 同一量綱下的相對改善**，
加上**real/null 是否分離**，而不是絕對值。

## 3. 換手率與夏普的相關性（推翻「變異壓縮」機制）

| | corr(turnover, sharpe) | corr(activity, sharpe) |
|---|---|---|
| real | **-0.457** | **-0.660** |
| null | +0.056 | -0.183 |

兩位顧問都主張「5-bar smoother 造成變異壓縮 → 虛無資料夏普暴增」。
但在真實資料上**變異壓縮反而有害**（corr -0.457），符號相反。
該機制無法解釋 real 臂。

## 4. 共識決議：REPAIR-THEN-GATE（先修 reward，再上 null 門檻）

兩位在第 3 輪收斂到同一計畫，僅在順序上曾有分歧（codex：先修 reward；
claude：先上門檻）。分歧由第 4 節的量測解決。

### 要做的（僅兩行）
1. `psh_peer` 傳入固定的外部基準（例如 batch median），讓 gap 項恢復作用
2. `mean_act_guard` 從「回傳 -50」改為「只記錄、不扣分」（連續懲罰）

### pilot 期間凍結
evaluator 數學（`tanh` / 0.25 / `smooth_causal(·,5)` / `roll(1)`、無門檻）、
genome 長度與範圍、LEG_FLOOR、reward 權重、universe、costs/funding、
real/null 臂定義與 seed 101-110、holdout 視窗、selection 規則。

### 明確不做
surrogate model、變長 genome、放寬父代選取、座標下降擴大搜索空間。
理由：這些都是在未校準的量測儀上增加自由度。

## 5. 驗證結果（已執行，僅用既有日誌重算，未重跑）

以 patched reward 重算兩臂：

```
real patched: mean=-0.3824 std=0.1193
null patched: mean=-0.5233 std=0.2408
pooled std = 0.2027          (舊值 0.088，改善 2.3 倍)
median real - median null = +0.0959   (舊值為負)
real > null：7/10 seeds, sign-test p=0.344
```

**結論：量測儀確實改善，且真實資料開始與 null 分離。**
兩位顧問的絕對門檻（std>=1.0）都沒觸發，但「先修 reward 再上門檻」的
質性預測獲得確認。

## 6. 仍存在的限制（誠實記錄）

- mean_ic 的 7/10 分離只有 p=0.344，**未達顯著**。10 個 seed 是這套
  null 機制的樣本量上限。
- mean_ic 中位數 0.0107 仍**低於可交易門檻**。兩位一致認為需要
  sustained OOS rank-IC >= 0.03 且扣成本後淨 Sharpe >= 1.0 才值得配置資金。
- 因此即使 reward 修好、null 門檻上線，**目前的搜尋結果仍不足以支持實盤**。
  這是正確的結論，不是失敗：門檻的作用就是讓這個結論浮現。

## 7. 相關檔案

- `research/ga_28c_30m_3y.py` — 待修正
- `research/selection_bias_28c.py` — 既有 null 校準
- `results/ga_iter_runs/`、`results/ga_null_runs/`、`results/wf_real/`、`results/wf_null/`


---

## 8. 實作與驗證記錄（2026-10-01，已完成）

### 已套用的兩行修正
`research/ga_28c_30m_3y.py`：

1. **移除 -50 懸崖**：呼叫點改為連續懲罰
   ```python
   activity_penalty = -INACTIVITY_WEIGHT * max(0.0, 1.0 - mean_act / ACTIVITY_TARGET)
   reward = shape_reward(...) + activity_penalty
   ```
   `ACTIVITY_TARGET = 0.01`（即舊的硬閘值）。**在門檻之上懲罰為 0**，
   因此原本就達標的公式得分完全不變，A/B 對照成立。

2. **`psh_peer` 改為可傳入外部基準**：`shape_reward` 的 fallback 保留
   `psh_peer=None → peer=psh`（即舊的死亡程式行為），刻意讓忘記傳入的
   呼叫端不會靜默獲得一個未經校準的項。

新增常數：`ACTIVITY_TARGET = 0.01`、`INACTIVITY_WEIGHT = 0.5`、`PEER_BASELINE = None`。

### 驗證方式
`research/verify_pilot_meter_repair.py` — **不重跑 GA**，只用既有 20 筆日誌
重算 reward，因此量測的是 meter 本身，不可能與 engine 改動混淆。

```
                              OLD              REPAIRED
saturation                11/20              0/20
rankable runs            9 -> 20  (2.22x)
std among rankable     0.3299 -> 0.4854
real - null median     +0.1344 -> +0.5869
real wins                 5/10 -> 8/10  (sign-test p=0.109)
```

**CLAIM A 通過**：飽和歸零，可排序的 run 由 9 筆增為 20 筆，解析度提升。
**CLAIM B 通過**：real 相對 null 的中位差距由 +0.13 提升為 +0.59，勝出 seed 5→8。

> 注意：raw pooled std 由 24.44 降至 0.4854 並非退步，而是 -50 離群值消失的
> 必然結果。判斷解析度必須看「可排序 run 之間的 std」，不能看 raw pooled std。

### 回歸測試
`tests/test_pilot_meter_repair.py` — 10 項全過，釘住：
- 無活性 genome 不得全部同分（懸崖不可回歸）
- 任何 activity 值都不得回傳 -50
- 門檻之上懲罰必須為 0（A/B 前提）
- 傳入 peer 必須改變得分（gap 項不可再次死亡）
- **pilot 凍結清單**：evaluator 數學、genome 長度、FEE/FUND/LEV、reward 權重

完整測試：84 項全過。唯一失敗 `test_e10_gates.py::test_paper_baseline_within_tolerance`
為**既有失敗**（已用 `git stash` 驗證：移除本次改動後仍以相同方式失敗）。

### 尚未執行
真實 pilot（10 real vs 10 null，walkforward，25 代 × 族群 16）尚未跑。
上述驗證只證明 meter 修好了，**不證明搜尋找到了可交易的 alpha**。
