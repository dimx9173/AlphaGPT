# STRATEGY E8 — 跨幣與 Regime 診斷（單幣 H2 排名挑第三腿壓力 + SMA50 趨勢分桶）

> 基線: FORMULA [3,2,7,2,7,11,15,4,4,6,6,10] (`FOMO PRESSURE SUB ...`), Y1b=ETC(0.88/0.12/cd18/None/ts24)+TRX(0.85/0.12/cd6/0.05/ts24), q0.3 vtNone, 50/50, aster 2x fund0.0005 fee0.0004/0.0008, 4h n=6580 (H1 493 / H2 493 / B 200 / C 500 / FULL 6580)
> 引擎: 鏡像 run_ad/run_ac — `MemeBacktest(aster lev2 short_enabled quantile_mask_long q0.3 `_apply_cooldown` `_apply_stops` `_vol_scale(vtNone)` roll1) + funding/tx` — 單幣參數統一復用 TRX-spec (0.85/0.12/cd6/0.05/ts24) vtNone both，以暴露純單幣真相
> 數據: data/data_15m_3y/*.csv 15m→4h×16 聚合（29幣全集，最短 ASTER 2111 / RENDER 4621 / POL 4327 / KAS 6135，其餘 6570-6580）
> 切片: H1[5584:6077] n493 / H2[6077:6570] n493 / B[6380:6580] n200 / C[6080:6580] n500 / FULL[0:6580] n6580
> 試驗: E8-1 全29幣單幣 H2/B/C 排名 + E8-2 併集第三腿 50/25/25 壓力 + E8-3 SMA50 趨勢三桶 regime

---

## 方法

### E8-1 單幣排名（TRX-spec, q0.3 vtNone both）

- 對 29 幣各自以 `TRX(0.85/0.12/cd6/0.05/ts24) q0.3 vtNone both` 單腿回測 H2/B/C/FULL（`both-leg` 同時計 long/short，不經 vol 放縮）。
- 每段輸出 `sharpe/ann/mdd/cum/n/trades/turnover` + `fee2x(0.0008) sharpe` + `both/long/short side split`。
- 排名依 `sharpe` 降序，取 `H2 top5` 與 `B top5` 與 `C top5` 併集作為第三腿候選（overlap 高時併集 8-12 幣）。

### E8-2 第三腿併入壓力（50/25/25）

- 候選併集每幣以 `50/25/25 (ETC/TRX/third)` 權合成（third 同樣 TRX-spec），回測 `H2/B/C/FULL` 全段 + `fee2x` + `turnover`。
- 基準 dual 為 Y1b `ETC+TRX 50/50` 同引擎同 fee 的 `H2/B/C/FULL` sharpe。
- 判定 `beats_both = (H2_sharpe > dual_H2) && (FULL_sharpe > dual_FULL)` 雙勝才算；否則稀釋。
- 報告 `delta_H2/FULL`、`fee_decay`、及 `turnover` 增量。

### E8-3 Regime 分桶（FULL 內 50bar SMA 斜率三桶）

- 以 BTC close 的 50bar SMA 為市場趨勢代理：`SMA50[t]=mean(close[t-49:t])`, `slope[t]=SMA50[t]-SMA50[t-1]`（首 bar 0）。
- FULL 6580 bars 的 `slope` 分布取 `tercile p33/p67` 為閾，分為 `down(≤p33)` / `flat(p33<p≤p67)` / `up(>p67)` 三桶（各約 33%），同時報告 `0.3σ` 平帶作為對照。
- 在每桶時間索引上切 dual `ETC+TRX 50/50 vtNone q0.3` 的 per-bar net 序列，計算桶內 `sharpe/ann/cum/mdd/占比%`；`sharpe<0` 標記弱 regime。

---

## 結果

### 單幣 H2/B/C 排名

**H2 top5 [6077:6570] n493 (TRX-spec vtNone):**

| rank | coin | sharpe | fee2x | ann | mdd | trades | turnover |
|------|------|--------|-------|-----|-----|--------|----------|
| 1 | ATOM | 3.464 | 3.256 | 2.9964 | 0.4455 | 25 | 0.1014 |
| 2 | NEAR | 3.383 | 3.196 | 4.2946 | 0.3822 | 33 | 0.1339 |
| 3 | TRX | 3.201 | 2.612 | 0.8763 | 0.0709 | 21 | 0.0892 |
| 4 | APT | 2.915 | 2.727 | 3.0022 | 0.4089 | 27 | 0.1095 |
| 5 | ETC | 2.836 | 2.613 | 2.3057 | 0.2749 | 25 | 0.1014 |

> 次優: DOT 2.717 / KAS 2.592(n58 小樣本) / HBAR 1.976 / ICP 1.862 / AVAX 1.156

**B top5 [6380:6580] n200:**

| rank | coin | sharpe | fee2x | ann | mdd |
|------|------|--------|-------|-----|-----|
| 1 | NEAR | 5.957 | 5.696 | 6.8867 | 0.2702 |
| 2 | TRX | 3.737 | 3.445 | 0.9549 | 0.0566 |
| 3 | APT | 3.307 | 3.076 | 3.8339 | 0.2685 |
| 4 | DOT | 2.036 | 1.845 | 2.2911 | 0.4730 |
| 5 | ETC | 1.444 | 1.291 | 1.2465 | 0.3195 |

> 次優: SHIB 1.229 / ICP 1.227 / XLM 0.779 / ATOM 0.701 — B 段僅 200 bars，排序方差大。

**C top5 [6080:6580] n500:**

| rank | coin | sharpe | fee2x |
|------|------|--------|-------|
| 1 | NEAR | 3.659 | 3.472 |
| 2 | APT | 3.611 | 3.423 |
| 3 | ATOM | 3.580 | 3.372 |
| 4 | ICP | 2.702 | 2.489 |
| 5 | HBAR | 2.659 | 2.443 |

> 次優: DOT 2.464 / ETC 2.263 / TRX 2.149 / AVAX 1.935

**併集 8 幣:** `ATOM, NEAR, TRX, APT, ETC, DOT, ICP, HBAR`（H2 top5 ∪ B top5 ∪ C top5；高度重疊故 8 <12，APT/NEAR 跨三段重複，ETC/TRX 基線自入）。

**FULL 參考（單幣 TRX-spec, 過濾無關成立性）:**

| coin | FULL sharpe | FULL fee2x | 註 |
|------|-------------|------------|----|
| KAS | 1.723 | — | n6135 短樣本，不入併集但值高 |
| POL | 1.711 | — | n4327 未覆 H2，單看 FULL 偏倖存 |
| TRX | 1.492 | 1.301 | 基線腿 |
| ATOM | 1.493 | 1.306 | 併集 |
| APT | 1.306 | 1.134 | 併集 |
| BTC | 0.954 | 0.742 | 中位 |

### 第三腿 50/25/25 併入壓力

**Dual 基線 Y1b 50/50 vtNone:**

| seg | sharpe | fee2x | ann | mdd | trades | turnover |
|-----|--------|-------|-----|-----|--------|----------|
| H2 | 4.091 | 3.776 | 1.7118 | 0.0744 | 36 | 0.0751 |
| B | 4.496 | 4.155 | 1.7376 | 0.0666 | 14 | 0.0750 |
| C | 3.479 | 3.168 | 1.4493 | 0.0744 | 36 | 0.0740 |
| FULL | 1.861 | 1.667 | 1.3918 | 0.8913 | 540 | 0.0825 |

**併集逐幣 50/25/25 (ETC/TRX/third) 表:**

| third | H2 sh (Δ \| fee2x) | B sh | C sh | FULL sh (Δ \| fee2x) | H2 turnover | beats_both |
|-------|-------------------|------|------|----------------------|-------------|------------|
| **ATOM** | **4.483 (+0.392 \| 4.210)** | 4.578 | 4.574 | **1.914 (+0.053 \| 1.735)** | 0.0838 | **True** |
| NEAR | 4.381 (+0.290 \| 4.122) | 7.047 | 4.588 | 1.428 (−0.433 \| 1.255) | 0.0947 | False |
| TRX | 4.091 (0.000 \| 3.776) | 4.496 | 3.479 | 1.861 (0.000 \| 1.667) | 0.0798 | False (基線複製) |
| APT | 3.974 (−0.117 \| 3.724) | 5.506 | 4.304 | 1.891 (+0.030 \| 1.714) | 0.0865 | False |
| ETC | 3.773 (−0.318 \| 3.521) | 3.489 | 3.226 | 1.512 (−0.349 \| 1.345) | 0.0838 | False |
| DOT | 3.873 (−0.218 \| 3.616) | 4.820 | 3.808 | 1.727 (−0.134 \| 1.554) | 0.0892 | False |
| ICP | 3.738 (−0.353 \| 3.466) | 5.036 | 4.175 | 1.208 (−0.653 \| 1.033) | 0.0865 | False |
| HBAR | 3.692 (−0.399 \| 3.423) | 3.873 | 4.043 | 1.484 (−0.377 \| 1.311) | 0.0852 | False |

- 併集 8 幣中 **1/8 beats_both (ATOM)**，7/8 在 H2 或 FULL 至少一端被稀釋；ATOM 雖雙勝但 FULL 僅 +0.053 且 B/C 雖高於 dual 但幅度有限，turnover 由 0.0751→0.0838 (+0.0087) 合格，fee2x 在 H2/FULL 均仍正但 FULL 衰至 1.735。
- 若按任務書擴為 12 幣口徑計則 **1/12**（4 空位視為未試 beats=False）；按已試 8 幣計 **1/8 ≈12.5%**。
- `NEAR` H2 雖 +0.29 但 FULL 大幅稀釋 −0.433；`APT` FULL 微增 +0.03 但 H2 稀釋；其餘全雙端或 FULL 稀釋。
- **以「H2 與 FULL 雙勝」嚴格標準，8 幣中僅 1 幣且增益薄弱，整體第三腿不成立。**

### Regime 三桶（BTC SMA50 斜率 tercile，FULL 6580）

- 斜率: `p33=−25.02`, `p67=42.74`, `σ=103.66`；對照 `0.3σ` 平帶 `|slope|≤31.10` 有 2095 bars(31.9%)。
- 三桶均佔約 33%：down 2172(33.0%) / flat 2236(34.0%) / up 2172(33.0%)。

| regime | 占比 | n bars | sharpe | ann | cum | mdd |
|--------|------|--------|--------|-----|-----|-----|
| down (≤p33) | 33.01% | 2172 | 2.344 | 1.7498 | 1.7354 | 0.5373 |
| flat (p33, p67] | 33.98% | 2236 | 2.120 | 1.4531 | 1.4836 | 0.6232 |
| up (>p67) | 33.01% | 2172 | 1.201 | 0.9708 | 0.9628 | 0.5857 |

- **無弱 regime**：三桶 sharpe 全 >0，均未落至負值。
- 相對最弱為 **up 上漲桶** `sharpe 1.201`（仍 >1），較 down/flat 低約 49%/43%，但 ann 仍 0.97，mdd 0.586 並非失控。
- 解讀: Y1b 雙腿在上漲趨勢中 edge 略降（空頭腿逆勢），但未失效；此與 C 段多空交織表現一致。

---

## 判定

| 判定項 | 結果 | 口徑 |
|--------|------|------|
| 第三腿併集 beats_both | **1/8 已試 (1/12 擴口徑)**，僅 ATOM (+0.392 H2 / +0.053 FULL) | 50/25/25 vs dual 50/50 |
| 第三腿是否 REJECT | **再確 REJECT**：併集 8 幣中僅 1 幣雙勝且 FULL 增益 0.053 屬噪聲級；其餘顯著稀釋 | 任務書「0/12 REJECT」標準下，實際 1/12 仍不達通過閾（需多幣可複製雙勝） |
| 弱 regime | **無**：down/flat/up 均 sharpe>0，最弱 up 1.20 仍 >0 | BTC SMA50 tercile |

**最佳併入者:** `ATOM` — H2 4.483 (+0.392) fee2x 4.210，FULL 1.914 (+0.053) fee2x 1.735，B 4.578 / C 4.574，turnover 0.0838（Δ+0.0087）。但 FULL 邊際太薄且 NEAR/APT 等其餘候選均 FULL 稀釋，整體不推薦。

**Regime 弱桶:** 無；若硬標相對弱桶為 `up`（1.201），需在後續 Z1 vol 加持後再複測是否補足。

---

## 文件

- `research/run_e8.py` — 鏡像 AD/AC 引擎，TRX-spec 單幣排名 + 50/25/25 三腿壓力 + SMA50 三桶
- `results/backtest_E8.json` — 全量數值（single_coin / rankings / dual / triple / regime / verdict）
- `logs/e8.log` — 運行日誌（29幣單幣 + 8三腿 + 3桶 regime）
- `docs/STRATEGY_E8.md` — 本文件

## 復現

```bash
.venv2/bin/python research/run_e8.py
```
引擎參數: FORMULA [3,2,7,2,7,11,15,4,4,6,6,10], TRX-spec 0.85/0.12/cd6/0.05/ts24, q0.3 vtNone both, fee0.0004/0.0008, fund0.0005, 50bar SMA tercile.
