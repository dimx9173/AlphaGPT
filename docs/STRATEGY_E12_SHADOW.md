# STRATEGY_E12_SHADOW — Y1b_main_thr1.0_w200 re-verify + AA challengers parallel shadow — 2026-09-08

基線: FORMULA [3,2,7,2,7,11,15,4,4,6,6,10], lev2 fund0.0005 fee0.0004/fee2x0.0008, 50/50 ETC/TRX, n=6580 4h (15m×16)
Plain: Y1b vtNone = ETC(0.88/0.12/cd18/None/ts24,both)+TRX(0.85/0.12/cd6/0.05/ts24,both), q0.3
Overlays (3候選): Z1(0.12/18/6 +vt0.012/w12), AA-H1 H1-best無偏(0.10/15/9 +vt0.012/w12, idx01), AA-H2 H2-best有偏(0.12/15/9 +vt0.012/w12, idx13)
Ledger: per-trade netp = move*LEV - FEE*LEV*2 - FUND*LEV, 50/50 split, leg_net_pos quantile+cooldown+stops+vol_scale+roll1 (鏡像 run_e9.py + run_w3.py, import重用零漂移)
Gate: causal trailing-200 per-bar-net sharpe | main=overlay sharpe>thresh 時 active | alt=plain sharpe<thresh 時 overlay active | thresh∈[0.8,1.0,1.5] win200 共18組
Fee2x: 相同 gate 切換 (base-fee 信號), ledger 以 fee=0.0008 重算; worstB/worstC = min(base,fee2x) sharpe
Segments: FULL[0:6580]/H2[6077:6570]/B[6380:6580]/C[6080:6580] | PASS = B_sh>plain_B AND C_sh>plain_C 且 FULL no collapse (sharpe>0.5 & mdd<1.0)
冷數據: `data/data_15m_3y -> ~/pcloud-drive/cold-data/AlphaGPT/data/data_15m_3y` symlink
腳本: research/run_e12.py (import research.run_e9 引擎) -> results/results_E12_shadow.json + logs/e12.log

## 0. W3 Y1b_main_thr1.0_w200 參照 (adopted shadow, 非本輪重算口徑)

- W3 採用: `Y1b_main_thr1.0_w200` coverage=0.652 FULL 2.6249x/sh=0.831/mdd=0.6861 tr=547 | B 1.0687/3.452 | C 1.106/1.966, 全面勝 W3-plain (B 2.509/C 1.775) 且無崩 (來源 results/backtest_W3_shadow.json)。
- 口徑差異: W3 的 plain = Q1舊基線 (cd12/ts0/qNone) + Y1b 作 overlay; E9/E12 的 plain = Y1b 本身, overlay = Z1/AA-H1/AA-H2。故 W3 的 Y1b_main 在 E12 網格中無對應 id, 本輪以 E12 並行 shadow 驗其穩定性主張: Y1b-plain 尾段 (B 3.452/C 1.966) 是否仍不可被 challenger 超越。
- 結論: E12  bestätigt — Y1b-plain B/C 仍為全場最高 (見§1), 18組 shadow 全 REJECT, Y1b_main_thr1.0_w200 作為 W3 紙上載體維持有效, 無需替換為 AA challenger。

## 1. Ledger 重建 (FULL/H2/B/C + fee2x)

- **plain (Y1b)** FULL x=0.8365 sh=0.251 mdd=0.7964 trades=541 by={'ETC': 212, 'TRX': 329} longs=6 | fee2x x=0.5422 sh=0.032 | worstB=2.879 worstC=1.416
  - FULL x=0.8365 sh=0.251 mdd=0.7964 tr=541 longs=6 (fee2x sh=0.032 x=0.5422)
  - H2 x=1.106 sh=1.98 mdd=0.0662 tr=38 longs=0 (fee2x sh=1.426)
  - B x=1.0687 sh=3.452 mdd=0.0295 tr=15 longs=0 by={'TRX': 9, 'ETC': 6} (fee2x sh=2.879 x=1.056)
  - C x=1.106 sh=1.966 mdd=0.0662 tr=38 longs=0 by={'TRX': 23, 'ETC': 15} (fee2x sh=1.416 x=1.0729)
- **Z1** FULL x=0.6988 sh=0.096 mdd=0.7274 trades=555 by={'ETC': 225, 'TRX': 330} longs=5 | fee2x x=0.4478 sh=-0.155 | worstB=2.322 worstC=1.165
  - FULL x=0.6988 sh=0.096 mdd=0.7274 tr=555 longs=5 (fee2x sh=-0.155 x=0.4478)
  - H2 x=1.0892 sh=1.749 mdd=0.0662 tr=38 longs=0 (fee2x sh=1.173)
  - B x=1.0525 sh=2.973 mdd=0.0295 tr=15 longs=0 (fee2x sh=2.322 x=1.04)
  - C x=1.0892 sh=1.736 mdd=0.0662 tr=38 longs=0 (fee2x sh=1.165 x=1.0566)
- **AA-H1** FULL x=0.3284 sh=-0.28 mdd=0.8567 trades=521 by={'ETC': 235, 'TRX': 286} longs=3 | fee2x x=0.2161 sh=-0.504 | worstB=-1.817 worstC=0.076
  - FULL x=0.3284 sh=-0.28 mdd=0.8567 tr=521 longs=3 (fee2x sh=-0.504 x=0.2161)
  - H2 x=1.0908 sh=1.128 mdd=0.1748 tr=38 longs=0 (fee2x sh=0.812)
  - B x=0.9032 sh=-1.583 mdd=0.1748 tr=16 longs=0 by={'ETC': 8, 'TRX': 8} (fee2x sh=-1.817 x=0.8916)
  - C x=1.0156 sh=0.383 mdd=0.1748 tr=39 longs=0 (fee2x sh=0.076 x=0.9844)
- **AA-H2** FULL x=0.7846 sh=0.186 mdd=0.8198 trades=525 by={'ETC': 236, 'TRX': 289} longs=3 | fee2x x=0.5151 sh=-0.04 | worstB=0.701 worstC=1.756
  - FULL x=0.7846 sh=0.186 mdd=0.8198 tr=525 longs=3 (fee2x sh=-0.04 x=0.5151)
  - H2 x=1.2526 sh=3.33 mdd=0.0468 tr=36 longs=0 (fee2x sh=2.946 x=1.2172)
  - B x=1.0301 sh=1.045 mdd=0.0778 tr=15 longs=0 (fee2x sh=0.701 x=1.0179)
  - C x=1.1663 sh=2.124 mdd=0.0778 tr=37 longs=0 (fee2x sh=1.756 x=1.1324)

與 E9 對照: base-fee ledger 四本帳與 results/backtest_E9.json 逐值一致 (4/4 MATCH, shadow 18/18 MATCH), 證引擎重用零漂移。
Fee2x 讀數: plain 在 fee2x 下 FULL 轉平 (0.032) 但 B/C 仍厚 (worstB 2.879/worstC 1.416); Z1 fee2x FULL 轉負 (-0.155); AA-H1 fee2x B/C 崩 (worstB -1.817/worstC 0.076, B2x x=0.8916 跌破1); AA-H2 僅 C 段抗費 (worstC 1.756) 但 B 崩至 0.701。

## 2. 每 Overlay vs Plain 漂移 (base fee)

- Z1-plain: d_trades=+14 d_final_x=-0.1377 d_sharpe=-0.155 longs_filtered=1 | FULL 0.251->0.096 | B 3.452->2.973 C 1.966->1.736
- AA-H1-plain: d_trades=-20 d_final_x=-0.5081 d_sharpe=-0.531 longs_filtered=3 | FULL 0.251->-0.28 | B 3.452->-1.583 C 1.966->0.383
- AA-H2-plain: d_trades=-16 d_final_x=-0.0519 d_sharpe=-0.065 longs_filtered=3 | FULL 0.251->0.186 | B 3.452->1.045 C 1.966->2.124
- 結論: 無一 overlay 在 ledger FULL 上超越 plain; AA-H2 僅 C 段小勝 +0.158 但 B 崩 -2.407; AA-H1 B 段轉負。

## 3. Gate 熱力表 (18組 shadows, base + fee2x worst)

### Z1 (plain B 3.452 / C 1.966)
| gate | thresh | coverage | FULL sh/mdd/x | B sh (2x/worst) | C sh (2x/worst) | vs plain B/C | PASS |
|---|---|---|---|---|---|---|---|
| main | 0.8 | 0.6394 | 0.055/0.7962/0.5724 | 2.973 (2.322/2.322) | 1.736 (1.165/1.165) | B -0.479 C -0.23 | REJECT |
| main | 1.0 | 0.624 | 0.064/0.7962/0.5843 | 2.973 (2.322/2.322) | 1.736 (1.165/1.165) | B -0.479 C -0.23 | REJECT |
| main | 1.5 | 0.5878 | 0.048/0.8051/0.5661 | 2.973 (2.322/2.322) | 1.736 (1.165/1.165) | B -0.479 C -0.23 | REJECT |
| alt | 0.8 | 0.2983 | 0.35/0.7003/1.0936 | 3.452 (2.879/2.879) | 1.966 (1.416/1.416) | B 0.0 C 0.0 | REJECT |
| alt | 1.0 | 0.3176 | 0.381/0.6871/1.1571 | 3.452 (2.879/2.879) | 1.966 (1.416/1.416) | B 0.0 C 0.0 | REJECT |
| alt | 1.5 | 0.371 | 0.299/0.7297/0.9981 | 3.452 (2.879/2.879) | 1.966 (1.416/1.416) | B 0.0 C 0.0 | REJECT |

### AA-H1 (plain B 3.452 / C 1.966)
| gate | thresh | coverage | FULL sh/mdd/x | B sh (2x/worst) | C sh (2x/worst) | vs plain B/C | PASS |
|---|---|---|---|---|---|---|---|
| main | 0.8 | 0.6185 | -0.296/0.853/0.3033 | -1.583 (-1.817/-1.817) | 0.383 (0.076/0.076) | B -5.035 C -1.583 | REJECT |
| main | 1.0 | 0.6023 | -0.372/0.8729/0.2596 | -1.583 (-1.817/-1.817) | 0.383 (0.076/0.076) | B -5.035 C -1.583 | REJECT |
| main | 1.5 | 0.5564 | -0.341/0.8739/0.256 | -1.583 (-1.817/-1.817) | 0.383 (0.076/0.076) | B -5.035 C -1.583 | REJECT |
| alt | 0.8 | 0.2983 | 0.756/0.5682/2.2765 | 3.452 (2.879/2.879) | 1.966 (1.416/1.416) | B 0.0 C 0.0 | REJECT |
| alt | 1.0 | 0.3176 | 0.82/0.5224/2.5585 | 3.452 (2.879/2.879) | 1.966 (1.416/1.416) | B 0.0 C 0.0 | REJECT |
| alt | 1.5 | 0.371 | 0.6/0.6567/1.7063 | 3.452 (2.879/2.879) | 1.966 (1.416/1.416) | B 0.0 C 0.0 | REJECT |

### AA-H2 (plain B 3.452 / C 1.966)
| gate | thresh | coverage | FULL sh/mdd/x | B sh (2x/worst) | C sh (2x/worst) | vs plain B/C | PASS |
|---|---|---|---|---|---|---|---|
| main | 0.8 | 0.6536 | 0.009/0.8614/0.514 | 1.045 (0.701/0.701) | 2.124 (1.756/1.756) | B -2.407 C +0.158 | REJECT |
| main | 1.0 | 0.6362 | 0.01/0.8674/0.5158 | 1.045 (0.701/0.701) | 2.124 (1.756/1.756) | B -2.407 C +0.158 | REJECT |
| main | 1.5 | 0.5871 | 0.121/0.8261/0.6431 | 1.045 (0.701/0.701) | 2.124 (1.756/1.756) | B -2.407 C +0.158 | REJECT |
| alt | 0.8 | 0.2983 | 0.605/0.61/1.7145 | 3.452 (2.879/2.879) | 1.966 (1.416/1.416) | B 0.0 C 0.0 | REJECT |
| alt | 1.0 | 0.3176 | 0.673/0.6122/1.948 | 3.452 (2.879/2.879) | 1.966 (1.416/1.416) | B 0.0 C 0.0 | REJECT |
| alt | 1.5 | 0.371 | 0.574/0.5931/1.6258 | 3.452 (2.879/2.879) | 1.966 (1.416/1.416) | B 0.0 C 0.0 | REJECT |

### 全量 shadow gates 明細
| id | overlay | gate | thresh | coverage | FULL x/sh/mdd/tr | B x/sh (worst) | C x/sh (worst) | PASS |
|---|---|---|---|---|---|---|---|---|
| Z1_main_thr0.8_w200 | Z1 | main | 0.8 | 0.6394 | 0.5724/0.055/0.7962/552 | 1.0525/2.973 (2.322) | 1.0892/1.736 (1.165) | REJECT |
| Z1_main_thr1.0_w200 | Z1 | main | 1.0 | 0.624 | 0.5843/0.064/0.7962/552 | 1.0525/2.973 (2.322) | 1.0892/1.736 (1.165) | REJECT |
| Z1_main_thr1.5_w200 | Z1 | main | 1.5 | 0.5878 | 0.5661/0.048/0.8051/553 | 1.0525/2.973 (2.322) | 1.0892/1.736 (1.165) | REJECT |
| Z1_alt_thr0.8_w200 | Z1 | alt | 0.8 | 0.2983 | 1.0936/0.35/0.7003/549 | 1.0687/3.452 (2.879) | 1.106/1.966 (1.416) | REJECT |
| Z1_alt_thr1.0_w200 | Z1 | alt | 1.0 | 0.3176 | 1.1571/0.381/0.6871/552 | 1.0687/3.452 (2.879) | 1.106/1.966 (1.416) | REJECT |
| Z1_alt_thr1.5_w200 | Z1 | alt | 1.5 | 0.371 | 0.9981/0.299/0.7297/549 | 1.0687/3.452 (2.879) | 1.106/1.966 (1.416) | REJECT |
| AA-H1_main_thr0.8_w200 | AA-H1 | main | 0.8 | 0.6185 | 0.3033/-0.296/0.853/558 | 0.9032/-1.583 (-1.817) | 1.0156/0.383 (0.076) | REJECT |
| AA-H1_main_thr1.0_w200 | AA-H1 | main | 1.0 | 0.6023 | 0.2596/-0.372/0.8729/565 | 0.9032/-1.583 (-1.817) | 1.0156/0.383 (0.076) | REJECT |
| AA-H1_main_thr1.5_w200 | AA-H1 | main | 1.5 | 0.5564 | 0.256/-0.341/0.8739/570 | 0.9032/-1.583 (-1.817) | 1.0156/0.383 (0.076) | REJECT |
| AA-H1_alt_thr0.8_w200 | AA-H1 | alt | 0.8 | 0.2983 | 2.2765/0.756/0.5682/562 | 1.0687/3.452 (2.879) | 1.106/1.966 (1.416) | REJECT |
| AA-H1_alt_thr1.0_w200 | AA-H1 | alt | 1.0 | 0.3176 | 2.5585/0.82/0.5224/564 | 1.0687/3.452 (2.879) | 1.106/1.966 (1.416) | REJECT |
| AA-H1_alt_thr1.5_w200 | AA-H1 | alt | 1.5 | 0.371 | 1.7063/0.6/0.6567/553 | 1.0687/3.452 (2.879) | 1.106/1.966 (1.416) | REJECT |
| AA-H2_main_thr0.8_w200 | AA-H2 | main | 0.8 | 0.6536 | 0.514/0.009/0.8614/556 | 1.0301/1.045 (0.701) | 1.1663/2.124 (1.756) | REJECT |
| AA-H2_main_thr1.0_w200 | AA-H2 | main | 1.0 | 0.6362 | 0.5158/0.01/0.8674/555 | 1.0301/1.045 (0.701) | 1.1663/2.124 (1.756) | REJECT |
| AA-H2_main_thr1.5_w200 | AA-H2 | main | 1.5 | 0.5871 | 0.6431/0.121/0.8261/550 | 1.0301/1.045 (0.701) | 1.1663/2.124 (1.756) | REJECT |
| AA-H2_alt_thr0.8_w200 | AA-H2 | alt | 0.8 | 0.2983 | 1.7145/0.605/0.61/564 | 1.0687/3.452 (2.879) | 1.106/1.966 (1.416) | REJECT |
| AA-H2_alt_thr1.0_w200 | AA-H2 | alt | 1.0 | 0.3176 | 1.948/0.673/0.6122/569 | 1.0687/3.452 (2.879) | 1.106/1.966 (1.416) | REJECT |
| AA-H2_alt_thr1.5_w200 | AA-H2 | alt | 1.5 | 0.371 | 1.6258/0.574/0.5931/555 | 1.0687/3.452 (2.879) | 1.106/1.966 (1.416) | REJECT |

註: alt 系列 B/C 與 plain 完全相等 (B 3.452/C 1.966) — gate 在 B/C 段內幾乎全切回 plain, shadow 無增益, 故判 REJECT (需嚴格大於)。

## 4. 最佳 Gate 配置

最佳: **無** — 18組全 REJECT (pass_sharpe=0/18 pass_x=0/18)。
- main 系列: Z1/AA-H1/AA-H2 在 overlay-active 期間 B 段落後 plain (Z1 -0.479, AA-H1 -5.035, AA-H2 -2.407), C 段僅 AA-H2 小勝 +0.158 不足以雙贏。
- alt 系列: B/C 與 plain 持平 (Δ=0), 不滿足嚴格大於。
- fee2x 下: main 系列 worst 進一步惡化 (AA-H1 worstB -1.817; AA-H2 worstB 0.701), 無一組 worstB>plain-worstB 且 worstC>plain-worstC。
- Plain 基準: FULL x=0.8365 sh=0.251 mdd=0.7964; fee2x FULL x=0.5422 sh=0.032 mdd=0.8305。

## 5. 決策

- **decision = REJECT no shadow passes B>C and full no collapse (keep plain)** — 與 E9 一致, AA challengers 並行 shadow 無一通過。
- Y1b_main_thr1.0_w200 (W3 採用) 維持為紙上 shadow 載體: 其口徑 (Q1-plain + Y1b-overlay) 下 FULL 2.6249x/sh 0.831 仍成立, E12 證 Y1b-plain 尾段不可被 Z1/AA 取代, 故無替換理由。
- AA-H2 僅作觀察: C 段 2.124>1.966 (fee2x 1.756>1.416) 是唯一亮點, 但 B 段 1.045<<3.452 且 fee2x worstB 0.701, 不滿足雙段門檻。
- Artifacts: results/results_E12_shadow.json + logs/e12.log (+ logs/e12_run.out); 腳本 research/run_e12.py。
