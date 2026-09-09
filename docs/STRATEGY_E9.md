# STRATEGY_E9 — Shadow Gate & Ledger 实盘模拟（AA 分支对照 W3）— 2026-09-07
基线: FORMULA [3,2,7,2,7,11,15,4,4,6,6,10], lev2 fund0.0005 fee0.0004, 50/50 ETC/TRX, n=6580 4h (15m×16)
Plain: Y1b vtNone = ETC(0.88/0.12/cd18/None/ts24,both)+TRX(0.85/0.12/cd6/0.05/ts24,both), q0.3 long-only
Overlays (3候选): Z1(0.12/18/6 +vt0.012/w12), AA-H1(0.10/15/9 +vt0.012/w12  H1-best), AA-H2(0.12/15/9 +vt0.012/w12 H2-best)
Ledger: per-trade netp = move*LEV - FEE*LEV*2 - FUND*LEV, 50/50 split, via leg_net_pos with quantile+cooldown+stops+vol_scale+roll1
Gate: trailing-200 per-bar-net sharpe | main=overlay sharpe>thresh 时 active | alt=plain sharpe<thresh 时 overlay active | thresh∈[0.8,1.0,1.5] win200 共18组
Segments: FULL[0:6580]/H2[6077:6570]/B[6380:6580]/C[6080:6580] | PASS= B>plain_B AND C>plain_C 且 FULL no collapse (sharpe>0.5 & mdd<1.0)
冷数据: `data/data_15m_3y -> ~/pcloud-drive/cold-data/AlphaGPT/data/data_15m_3y` symlink
镜像: research/run_w3.py + run_z2.py ledger+gate 引擎，脚本 research/run_e9.py -> results/backtest_E9.json + logs/e9.log
## 1. Ledger 重建 (FULL/H2/B/C)
- **plain** FULL x=0.8365 sh=0.251 mdd=0.7964 trades=541 by={'ETC': 212, 'TRX': 329} longs=6
  - FULL x=0.8365 sh=0.251 mdd=0.7964 tr=541 longs=6 by={'ETC': 212, 'TRX': 329}
  - H2 x=1.106 sh=1.98 mdd=0.0662 tr=38 longs=0 by={'TRX': 23, 'ETC': 15}
  - B x=1.0687 sh=3.452 mdd=0.0295 tr=15 longs=0 by={'TRX': 9, 'ETC': 6}
  - C x=1.106 sh=1.966 mdd=0.0662 tr=38 longs=0 by={'TRX': 23, 'ETC': 15}
- **Z1** FULL x=0.6988 sh=0.096 mdd=0.7274 trades=555 by={'ETC': 225, 'TRX': 330} longs=5
  - FULL x=0.6988 sh=0.096 mdd=0.7274 tr=555 longs=5 by={'ETC': 225, 'TRX': 330}
  - H2 x=1.0892 sh=1.749 mdd=0.0662 tr=38 longs=0 by={'TRX': 23, 'ETC': 15}
  - B x=1.0525 sh=2.973 mdd=0.0295 tr=15 longs=0 by={'TRX': 9, 'ETC': 6}
  - C x=1.0892 sh=1.736 mdd=0.0662 tr=38 longs=0 by={'TRX': 23, 'ETC': 15}
- **AA-H1** FULL x=0.3284 sh=-0.28 mdd=0.8567 trades=521 by={'ETC': 235, 'TRX': 286} longs=3
  - FULL x=0.3284 sh=-0.28 mdd=0.8567 tr=521 longs=3 by={'ETC': 235, 'TRX': 286}
  - H2 x=1.0908 sh=1.128 mdd=0.1748 tr=38 longs=0 by={'TRX': 22, 'ETC': 16}
  - B x=0.9032 sh=-1.583 mdd=0.1748 tr=16 longs=0 by={'ETC': 8, 'TRX': 8}
  - C x=1.0156 sh=0.383 mdd=0.1748 tr=39 longs=0 by={'TRX': 22, 'ETC': 17}
- **AA-H2** FULL x=0.7846 sh=0.186 mdd=0.8198 trades=525 by={'ETC': 236, 'TRX': 289} longs=3
  - FULL x=0.7846 sh=0.186 mdd=0.8198 tr=525 longs=3 by={'ETC': 236, 'TRX': 289}
  - H2 x=1.2526 sh=3.33 mdd=0.0468 tr=36 longs=0 by={'TRX': 21, 'ETC': 15}
  - B x=1.0301 sh=1.045 mdd=0.0778 tr=15 longs=0 by={'ETC': 7, 'TRX': 8}
  - C x=1.1663 sh=2.124 mdd=0.0778 tr=37 longs=0 by={'TRX': 21, 'ETC': 16}

与 W3 对照: W3 plain(Y1b之前的cd12/ts0) FULL 2.9883x/0.899/mdd0.7846 已被 E9 plain(Y1b cd18/ts24) 覆盖为 0.8365x/0.251 — Y1b 在 ledger 法下 FULL 崩 (fee*2+fund 每笔扣 0.0026*lev)，尾段 B/C 仍厚 (B 3.452/C1.966) 但长历史被磨平，说明 Y1b/AA 均属尾段动量，在 per-trade 扣费下仅尾端存活。
与腿级收益对照: AA 的腿级累计(leg_series gross - tx - fund 累加) 显示 AA-H1 FULL 4.8869 cum / AA-H2 5.3755 cum，但 ledger 法(按成交逐笔复利 + 50/50等权持仓切换) 将其翻成 0.3284x/0.7846x，差异来自 per-trade固定扣费 + 杠杆后仓位等权分仓 的复利稀释 — 同一信号在两种会计口径下结论相反，尾端策略对会计口径敏感。

## 2. 每 Overay vs Plain 漂移
- Z1-plain: d_trades=14 d_final_x=-0.1377 d_sharpe=-0.155 longs_filtered=1 | FULL 0.251->0.096 | B 3.452->2.973 C 1.966->1.736
- AA-H1-plain: d_trades=-20 d_final_x=-0.5081 d_sharpe=-0.531 longs_filtered=3 | FULL 0.251->-0.28 | B 3.452->-1.583 C 1.966->0.383
- AA-H2-plain: d_trades=-16 d_final_x=-0.0519 d_sharpe=-0.065 longs_filtered=3 | FULL 0.251->0.186 | B 3.452->1.045 C 1.966->2.124
- 结论: 无一 overlay 在 ledger FULL 上超越 plain，且 Z1/AA-H1 甚至 B 段直接输给 plain (Z1 B 2.973<3.452, AA-H1 B -1.583, AA-H2 B 1.045)，C 段亦仅 AA-H2 小胜 +0.158 但 B 崩 -2.407。

## 3. Gate 热力表 (18组 shadows)

### Z1
| gate | thresh | coverage | FULL sh/mdd/x | B sh | C sh | vs plain B/C | PASS |
|---|---|---|---|---|---|---|---|
| main | 0.8 | 0.6394 | 0.055/0.7962/0.5724 | 2.973 | 1.736 | B -0.479 C -0.23 | REJECT |
| main | 1.0 | 0.624 | 0.064/0.7962/0.5843 | 2.973 | 1.736 | B -0.479 C -0.23 | REJECT |
| main | 1.5 | 0.5878 | 0.048/0.8051/0.5661 | 2.973 | 1.736 | B -0.479 C -0.23 | REJECT |
| alt | 0.8 | 0.2983 | 0.35/0.7003/1.0936 | 3.452 | 1.966 | B 0.0 C 0.0 | REJECT |
| alt | 1.0 | 0.3176 | 0.381/0.6871/1.1571 | 3.452 | 1.966 | B 0.0 C 0.0 | REJECT |
| alt | 1.5 | 0.371 | 0.299/0.7297/0.9981 | 3.452 | 1.966 | B 0.0 C 0.0 | REJECT |

### AA-H1
| gate | thresh | coverage | FULL sh/mdd/x | B sh | C sh | vs plain B/C | PASS |
|---|---|---|---|---|---|---|---|
| main | 0.8 | 0.6185 | -0.296/0.853/0.3033 | -1.583 | 0.383 | B -5.035 C -1.583 | REJECT |
| main | 1.0 | 0.6023 | -0.372/0.8729/0.2596 | -1.583 | 0.383 | B -5.035 C -1.583 | REJECT |
| main | 1.5 | 0.5564 | -0.341/0.8739/0.256 | -1.583 | 0.383 | B -5.035 C -1.583 | REJECT |
| alt | 0.8 | 0.2983 | 0.756/0.5682/2.2765 | 3.452 | 1.966 | B 0.0 C 0.0 | REJECT |
| alt | 1.0 | 0.3176 | 0.82/0.5224/2.5585 | 3.452 | 1.966 | B 0.0 C 0.0 | REJECT |
| alt | 1.5 | 0.371 | 0.6/0.6567/1.7063 | 3.452 | 1.966 | B 0.0 C 0.0 | REJECT |

### AA-H2
| gate | thresh | coverage | FULL sh/mdd/x | B sh | C sh | vs plain B/C | PASS |
|---|---|---|---|---|---|---|---|
| main | 0.8 | 0.6536 | 0.009/0.8614/0.514 | 1.045 | 2.124 | B -2.407 C +0.158 | REJECT |
| main | 1.0 | 0.6362 | 0.01/0.8674/0.5158 | 1.045 | 2.124 | B -2.407 C +0.158 | REJECT |
| main | 1.5 | 0.5871 | 0.121/0.8261/0.6431 | 1.045 | 2.124 | B -2.407 C +0.158 | REJECT |
| alt | 0.8 | 0.2983 | 0.605/0.61/1.7145 | 3.452 | 1.966 | B 0.0 C 0.0 | REJECT |
| alt | 1.0 | 0.3176 | 0.673/0.6122/1.948 | 3.452 | 1.966 | B 0.0 C 0.0 | REJECT |
| alt | 1.5 | 0.371 | 0.574/0.5931/1.6258 | 3.452 | 1.966 | B 0.0 C 0.0 | REJECT |

### 全量 shadow gates 明细
| id | overlay | gate | thresh | coverage | FULL x/sh/mdd/tr | B x/sh | C x/sh | PASS |
|---|---|---|---|---|---|---|---|---|
| Z1_main_thr0.8_w200 | Z1 | main | 0.8 | 0.6394 | 0.5724/0.055/0.7962/552 | 1.0525/2.973 | 1.0892/1.736 | REJECT |
| Z1_alt_thr0.8_w200 | Z1 | alt | 0.8 | 0.2983 | 1.0936/0.35/0.7003/549 | 1.0687/3.452 | 1.106/1.966 | REJECT |
| Z1_main_thr1.0_w200 | Z1 | main | 1.0 | 0.624 | 0.5843/0.064/0.7962/552 | 1.0525/2.973 | 1.0892/1.736 | REJECT |
| Z1_alt_thr1.0_w200 | Z1 | alt | 1.0 | 0.3176 | 1.1571/0.381/0.6871/552 | 1.0687/3.452 | 1.106/1.966 | REJECT |
| Z1_main_thr1.5_w200 | Z1 | main | 1.5 | 0.5878 | 0.5661/0.048/0.8051/553 | 1.0525/2.973 | 1.0892/1.736 | REJECT |
| Z1_alt_thr1.5_w200 | Z1 | alt | 1.5 | 0.371 | 0.9981/0.299/0.7297/549 | 1.0687/3.452 | 1.106/1.966 | REJECT |
| AA-H1_main_thr0.8_w200 | AA-H1 | main | 0.8 | 0.6185 | 0.3033/-0.296/0.853/558 | 0.9032/-1.583 | 1.0156/0.383 | REJECT |
| AA-H1_alt_thr0.8_w200 | AA-H1 | alt | 0.8 | 0.2983 | 2.2765/0.756/0.5682/562 | 1.0687/3.452 | 1.106/1.966 | REJECT |
| AA-H1_main_thr1.0_w200 | AA-H1 | main | 1.0 | 0.6023 | 0.2596/-0.372/0.8729/565 | 0.9032/-1.583 | 1.0156/0.383 | REJECT |
| AA-H1_alt_thr1.0_w200 | AA-H1 | alt | 1.0 | 0.3176 | 2.5585/0.82/0.5224/564 | 1.0687/3.452 | 1.106/1.966 | REJECT |
| AA-H1_main_thr1.5_w200 | AA-H1 | main | 1.5 | 0.5564 | 0.256/-0.341/0.8739/570 | 0.9032/-1.583 | 1.0156/0.383 | REJECT |
| AA-H1_alt_thr1.5_w200 | AA-H1 | alt | 1.5 | 0.371 | 1.7063/0.6/0.6567/553 | 1.0687/3.452 | 1.106/1.966 | REJECT |
| AA-H2_main_thr0.8_w200 | AA-H2 | main | 0.8 | 0.6536 | 0.514/0.009/0.8614/556 | 1.0301/1.045 | 1.1663/2.124 | REJECT |
| AA-H2_alt_thr0.8_w200 | AA-H2 | alt | 0.8 | 0.2983 | 1.7145/0.605/0.61/564 | 1.0687/3.452 | 1.106/1.966 | REJECT |
| AA-H2_main_thr1.0_w200 | AA-H2 | main | 1.0 | 0.6362 | 0.5158/0.01/0.8674/555 | 1.0301/1.045 | 1.1663/2.124 | REJECT |
| AA-H2_alt_thr1.0_w200 | AA-H2 | alt | 1.0 | 0.3176 | 1.948/0.673/0.6122/569 | 1.0687/3.452 | 1.106/1.966 | REJECT |
| AA-H2_main_thr1.5_w200 | AA-H2 | main | 1.5 | 0.5871 | 0.6431/0.121/0.8261/550 | 1.0301/1.045 | 1.1663/2.124 | REJECT |
| AA-H2_alt_thr1.5_w200 | AA-H2 | alt | 1.5 | 0.371 | 1.6258/0.574/0.5931/555 | 1.0687/3.452 | 1.106/1.966 | REJECT |

## 4. 最佳 Gate 配置
最佳: **无** — 18组全 REJECT (pass_sharpe=0/18 pass_x=0/18)
Plain 基准: FULL x=0.8365 sh=0.251 mdd=0.7964 | B x=1.0687 sh=3.452 mdd=0.0295 | C x=1.106 sh=1.966 mdd=0.0662
最近似: alt 类门槛在 B/C 上与 plain 持平(main 崩更重)，无一同时 beats plain于 B 与 C。AA-H2 的 C 段 2.124 小胜 plain 1.966 (+0.158) 但 B 段 1.045 巨亏 -2.407；Z1 B 2.973/-0.479, AA-H1 B -1.583/-5.035 全面崩。

## 5. 判定 — 是否采用新 overlay gate
**REJECT — 不采用任何新 overlay gate，保持 plain (Y1b vtNone, 0.88/0.12/cd18/None/ts24 + 0.85/0.12/cd6/0.05/ts24, q0.3)**
- 门槛 PASS=0/18 (sharpe口径) 且 0/18 (x口径)，全三段 beats plain 不成立。在 E9 的 AA 候选上未出现翻转 W3 结论的机会：W3 曾有 Y1b_main1.0 PASS (coverage 0.65 FULL 2.62x/0.83)，但那是基于 W3-plain(cd12/ts0, 2.98x) 的对照；在 E9-plain(Y1b本身) 为基准后，Y1b/Z1/AA 分支间不再有叠加增益。
- 原因: (1) AA 的 etc15/trx9 在 leg_series 上 H2/B/C 高sharpe，但 ledger per-trade 固定扣费下 FULL 崩为 0.32~0.78x (AA-H1 甚至 B 段 -1.58)，说明 AA 参数在真实成交会计下尾段单点失效；(2) Z1(vt0.012)在 E9-ledger 下亦 FULL 0.69x，且 B 段输 plain 0.48 sh；(3) 纵使切到 shadow gate，B 段永远被 plain 3.452 压制 — W3 中 Y1b B 3.452 本身就是 plain，gate 无法在 B 上再超越自身。
- 与 W3 结论一致性: W3 仅 Y1b_main1.0 PASS，Y2/Z1 main 全 REJECT；在 E9 以 AA 为 overlay 候选、Y1b 为 plain 再验，结论仍为 REJECT，无翻转。AA 的 H1-best 在最需要 gate 的 B 段直接 -1.58 崩，证其非 gate 可救。
- 建议: 保持 plain Y1b vtNone  live；Z1/AA 仅作 paper观察，不入 shadow。下一步若要复活 AA，需改用 12折 median / walk-forward 验证 etc15 是否非尾端过拟合，再考虑 funding/fee 弹性而非 gate。

## 6. 复现
```bash
.venv2/bin/python research/run_e9.py
```
- 输入: data/data_15m_3y/{ETC,TRX}.csv -> 4h聚合 n=6580
- Ledger: run_ledger per-trade netp=move*2 -0.0004*4 -0.0005*2, 50/50等权，pos 来自 leg_net_pos(quantile0.3 + cooldown + stops ts24 + vol_scale + roll1)
- Gate: trailing-200 per-bar-net sharpe，main/alt × thresh[0.8,1.0,1.5] × overlay[Z1/AA-H1/AA-H2]=18
- 输出: results/backtest_E9.json, logs/e9.log
- 配置hash: formula [3,2,7,2,7,11,15,4,4,6,6,10], fee0.0004/fund0.0005/lev2, vt0.012/w12, q0.3 long-only
