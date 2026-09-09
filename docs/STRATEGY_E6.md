# STRATEGY_E6 — 持仓时长细分 + 空头专精 (Short-Only × Time-Stop 交互) — 2026-09-07

> 任务 E6：镜像 `research/run_ab.py` 引擎，3 持仓桶 `[1-6bar,7-24bar,25+bar]` 对 Y1b/Z1/AA-H1 三基线在 FULL 上分解每桶 cum/占比/trades/avg bars_held；另对 short-only 分支做专精网 `sth[0.08,0.10,0.12]×trx_cd[6,9,12]×ts[12,24,36]=27` 行于 Z1 vt0.012 short-only 上，每行 H2/B/C/FULL+fee2x+turnover，H1 无偏取 top1 验 H2 要求 H2>3.5 才算 short 专精选胜 both。

## 0. 一句话结论

**持仓桶：中段 7-24bar 为绝对主导（52.9%-74.0% cum 贡献），25+bar 次之（29.6%-54.9%），1-6bar 轻微亏损（-0.7% 至 -4.7%）；短持 vs 长持贡献比 1-6/25+ ≈ -0.08 (Y1b/Z1) 至 -0.02 (AA-H1)，长持稳赚短持拖累。Short 专精：H1 无偏 top1 (0.08/12/36) H2=4.527 虽过 3.5 门槛但不胜 both_Z1(5.013) → FAIL_VS_BOTH；H1 次优 (0.08/9/36 H2 5.31) 与 (0.10/9/24 H2 5.40) 虽胜 both 但为 rank2/3 非无偏首选，不计胜。**

## 1. 引擎与基线

- 镜像 `run_ab.py`：`leg_series + quantile_mask_long(abs top-q long-only q0.3) + _apply_cooldown(cd) + _apply_stops(time_stop) + _vol_scale(vt clamp0.2-2.0 roll1 post-stops pre-roll) + roll1`，`turnover/tx/funding` 净收益，`lev2`，`50/50 ETC/TRX`，`fund0.0005 fee0.0004/fee2x0.0008`，`n=6580 4h (15m×16冷数据 symlink)`。
- 3 基线 FULL 规格：
  - **Y1b** `ETC 0.88/0.12/cd18/None/ts24 + TRX 0.85/0.12/cd6/0.05/ts24` `both` `vtNone` `q0.3`
  - **Z1** 同上 `vt0.012/w12` (AB A2 等价)
  - **AA-H1** `ETC 0.88/0.10/cd15/None/ts24 + TRX 0.85/0.10/cd9/0.05/ts24` `both` `vt0.012/w12` (AA H1-best unbiased)
- 切片 `H1[5584:6077] n493 / H2[6077:6570] n493 / B[6380:6580] n200 / C[6080:6580] n500 / FULL[0:6580] n6580`；脚步 `research/run_e6.py → results/backtest_E6.json + logs/e6.log`。

## 2. 持仓时长桶分解（FULL，combo position 按重叠后 sign-aware Episode 划分）

combo position = 0.5*ETC_pos + 0.5*TRX_pos，重叠时合并为连续持有；episode 按 sign 分段即一次完整多/空持仓。`time_stop=24` 单腿上限 24bar，但 combo 重叠可延至 25+bar（腿错位连续）。

### 2.1 Combo 层桶表（主表，portfolio 视角）

| 基线 | 桶 | cum | 占比 | trades | avg bars_held | 说明 |
|---|---|---|---|---|---|---|
| **Y1b** | 1-6 | -0.1602 | -3.8% | 14 | 2.93 | 极短持亏损 |
|  | 7-24 | 2.4970 | 59.7% | 74 | 14.88 | 主导 |
|  | 25+ | 1.8908 | 45.2% | 107 | 41.89 | 重叠长持次优 |
|  | **FULL** | 4.1814 | 100% | 195 | — | `sh 1.861 pos 0.62 (ETC+TRX sum 540 trades per-leg)` |
| **Z1** | 1-6 | -0.2347 | -4.7% | 15 | 2.73 |  |
|  | 7-24 | 2.6631 | 52.9% | 75 | 14.59 | 主导 |
|  | 25+ | 2.7607 | 54.9% | 108 | 41.89 | 长持反超中段 (vt 放大尾部波动) |
|  | **FULL** | 5.0321 | 100% | 198 | — | `sh 2.064` |
| **AA-H1** | 1-6 | -0.0364 | -0.7% | 16 | 3.31 |  |
|  | 7-24 | 3.6177 | 74.0% | 79 | 17.82 | 极强中段主导 |
|  | 25+ | 1.4479 | 29.6% | 93 | 42.72 | 长持占比骤降 (cd9/15使重叠缩短) |
|  | **FULL** | 4.8869 | 100% | 188 | — | `sh 2.025` |

- **短持 vs 长持贡献比 (1-6/25+ cum 比)**：Y1b -0.085 / Z1 -0.085 / AA-H1 -0.025。1-6bar 均为负贡献，长持稳正且为第二大来源；AA-H1 短持亏损最小（sth0.10+cd15/9 使短打过滤更好），但长持占比也最低。
- **中段 7-24bar** 在 AA-H1 达 74% peak，说明 `sth0.10+cd15/9+ts24` 把收益压缩至 time_stop 边界内高密度区间。

### 2.2 Per-leg 桶表（诊断，单腿视角，无 25+因 ts=24 硬上限）

| 基线 | leg | 1-6 cum/trades/avg | 7-24 cum/trades/avg | 25+ |
|---|---|---|---|---|
| Y1b | ETC | 0.6773/21/2.81 | 5.1171/192/20.04 | 0 |
|  | TRX | -0.3963/34/3.38 | 3.5055/296/14.97 | 0 |
| Z1 | ETC | 0.6489/21/2.81 | 5.7860/192/20.04 | 0 |
|  | TRX | -0.3939/34/3.38 | 4.8055/296/14.97 | 0 |
| AA-H1 | ETC | 1.1416/22/3.45 | 4.3427/203/19.07 | 0 |
|  | TRX | 0.0285/31/3.39 | 4.9820/256/16.09 | 0 |

- 单腿 25+ 恒 0（符合 time_stop=24 硬截断与冷却逻辑），combo 25+ 源于双腿错位重叠连续持有。
- TRX 1-6bar 在 Y1b/Z1 均亏损（-0.39），AA-H1 收敛至接近 0（0.02），印证 AA-H1 的 `sth0.10+cd9` 对 TRX 短打有净化作用。
- ETC 单腿 7-24bar 始终为最大收益池（Z1达 5.78）。

## 3. Short-Only 专精网 27 行（Z1 vt0.012 short-only，ETC cd18 固定/TRX cd变、sth/ts 双变）

固定：`ETC 0.88/sth/cd18/None/ts + TRX 0.85/sth/cd[6,9,12]/0.05/ts` `side short` `q0.3` `vt0.012/w12` `50/50` `lev2`。

每行 `H1/H2/B/C/FULL (base fee0.0004)` + `H2/B/C/FULL fee2x` + `turnover`；无偏以 H1 sharpe 排序。

### 3.1 H1 排序 Top3（无偏候选）与 H2 验证

| rank | idx | sth | trx_cd | ts | H1 sh | H2 sh | H2 fee2x | B sh (2x) | C sh (2x) | FULL sh (2x) | turnover FULL | 验证 vs both_Z1 5.013 & gate 3.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **1** | 08 | 0.08 | 12 | 36 | 3.961 | **4.527** | 4.157 | -0.119(-0.396) | 5.950(5.565) | 1.863(1.601) | 0.1150 | gate PASS 但 **FAIL vs both** (4.527<5.013) |
| 2 | 05 | 0.08 | 9 | 36 | 3.835 | **5.310** | 4.920 | 0.150(-0.152) | 6.632(6.226) | 2.141(1.864) | 0.1231 | 胜 both 但 rank2 非首选 |
| 3 | 13 | 0.10 | 9 | 24 | 3.767 | **5.400** | 4.950 | 5.358(4.868) | 5.275(4.829) | 2.344(2.037) | 0.1372 | 胜 both 但 rank3 |

- **无偏 verdict**：H1-top1 H2=4.527 >3.5 true 但 vs both_Z1 5.013 false → `SHORT_SPECIALIST_FAIL_VS_BOTH`。
- 按规则「H2>3.5 才算 short 专精选胜 both」，即使放宽至 3.5 亦不胜 both；仅当以有偏 H2 排序时 `idx14 (0.10/9/36) H2 5.527` / `idx13 H2 5.400` / `idx05 H2 5.310` 三行可胜 both (分别 +0.514/+0.387/+0.297)，但属有偏偷看，**无偏不计胜**。

### 3.2 全 27 行要点

- **sth 维度**：`0.08` 在 `trx_cd12/ts36` 上 H1 最高(3.961)，但 H2/B 偏弱(B -0.119 崩)；`0.12` 传统阈值在 `trx_cd6/ts24` 复现 Z1 both等价 short分支 H2 4.837/B 6.172(近 both) 但仍略逊。
- **trx_cd 维度**：`trx_cd9` 为甜点，3个胜 both的短线行皆 cd9；`cd12` 过钝导致 H1 虚高H2 不稳，`cd6` 在 sth0.08上 B段全崩（B -0.06~0.12）。
- **ts 维度**：`ts36` 普遍优于 `ts12`(全网 B 负 vs B正 对比)，`ts24`为折中；但 `ts36` 改善以 turnover 下降为代价 (0.23→0.11)。
- **fee2x 抗压**：短线网 fee2x 衰减约 0.35-0.45，优于 long 网；但 B段在 `sth0.08` 全网极差(-0.6~0.15)，说明小 sth 短打在 B(window末 200bar 空头回撤段)失效，`sth0.10/0.12 cd9 ts24/36` 的 B 5.3~6.6 才稳。
- **turnover**：短网 FULL turnover 0.11-0.22，低于 both Z1 0.15 的仅 ts36 低换手行，ts12 高换手行反而 0.20+。

### 3.3 与 Both 对比

Both_Z1 H2 5.013 / Both_Y1b H2 4.091；短线无偏 top1 4.527 介于两者之间（胜 Y1b 负 Z1），有偏 top H2 5.527 略胜 Z1 0.51 但 B 5.90 vs both 7.04 仍差 1.14，中段稳度不足。

## 4. 结论与建议

- **持仓桶**：绝不追短——1-6bar 负贡献，25+虽正但依赖腿重叠而非单腿能力；**7-24bar 为 alpha 核心**，AA-H1 将 74% 收益压缩至此时段说明 `sth0.10+cd15/9+ts24` 已近最优中段富集。
- **Short 专精**：无偏不胜 both，**维持 both (AB 结论) 不切 short**；短线仅作 regime 对冲影子，`sth0.10/trx_cd9/ts24` 若作影子 short sleeve 可考虑但须另起独立资金与风控，不并入主账户。
- **下一步**：持仓桶分解建议配合 E3 的 fee曲线与止损压力，25+重叠长持的 tail risk 需以 time_stop=24 硬核 + vt 为组合回撤控制。

## 5. 产物

- `research/run_e6.py` 镜像 AB 引擎，buckets `[1-6,7-24,25+]` 分解 + 27 行 short 网。
- `results/backtest_E6.json` 含 `bucket_decomposition (Y1b/Z1/AA-H1)` + `short_grid 27 rows` + `short_top3_H1/H2` + `unbiased_verdict` + `both_H2_ref`。
- `logs/e6.log` 全量日志。
- 本文档 `docs/STRATEGY_E6.md`。

---
*注：bucket trades 为 combo层合并后 trades，per-leg sum ≈ 540 与 `stats trades`一致；pct 为 cum口径贡献占比。*
