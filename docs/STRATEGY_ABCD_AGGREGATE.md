# 策略迭代匯總 — AA/AB/AC/AD 四路并行 (2026-09-07)

> 延續 W 收斂後的下一輪四路并行。基線不動: FORMULA [3,2,7,2,7,11,15,4,4,6,6,10] (`FOMO PRESSURE SUB PRESSURE SUB ABS DECAY DEV DEV ADD ADD NEG`), Y1b=ETC(0.88/0.12/cd18/None/ts24)+TRX(0.85/0.12/cd6/0.05/ts24), q0.3 long-only, 50/50, aster perp 2x fund0.0005 fee0.0004/0.0008, n=6580 4h (15m×16, cold-data symlink)。
> 引擎統一: `StackVM.execute` → `sigmoid` → `quantile_mask_long(q0.3 side=long)` → `_apply_cooldown(cd)` → `_apply_stops(ts24/sl/tp)` → `_vol_scale(vt)` → `roll1` → `turnover/tx/funding` net `lev2`。切片 H1[5584:6077] n493 / H2[6077:6570] n493 / B[6380:6580] n200 / C[6080:6580] n500 / FULL[0:6580] n6580 / 12fold 548bar。
> 四路共 63 實驗行: AA 20 + AB 11 + AC 28(42明細) + AD 16(10+4+2)，全量落地 `results/backtest_AX.json` + `docs/STRATEGY_AX.md` + `research/run_ax.py`。

---

## AA — 參數精煉 (sth×cd 小網格, 圍繞 Z1 無偏首度 PASS)

- 網: `sth [0.10,0.11,0.12] × etc_cd [15,18,21] × trx_cd [6,9]=18` + `sth0.09×2ctrl=2` → 20 行；固定 `lth ETC0.88 TRX0.85 sl ET C None/TRX0.05 ts24 vt0.012/w12 q0.3 50/50`。
- 無偏門檻 `H1-best H2>3.0` 才採納，否則 REJECT；壓力 `fund[0.0003,0.0005]×fee[0.0004,0.0008]=4` 格於 B/C，要求 `worst B>0 且 worst C>1` 才 PASS。

| 冠 | sth/etc/trx | H1 sh | H2 sh (2x) | B sh (2x) | C sh (2x) | FULL sh |
|---|---|---|---|---|---|---|
| H1-best (無偏冠) idx01 | 0.10/15/9 | 4.604 | 5.581 (5.121) | 2.175 (1.739) | 5.436 (4.974) | 2.025 |
| H2-best (有偏冠) idx13 | 0.12/15/9 | 3.557 | 6.835 (6.349) | 4.805 (4.274) | 5.558 (5.114) | 2.198 |
| Z1 等價 0.12/18/6 (AA內復現) | 0.12/18/6 | 2.655 | 5.013 (4.561) | 7.042 (6.513) | 4.666 (4.215) | 2.064 |
| FULL-best | 0.11/18/9 | 2.870 | 4.981 (4.530) | 5.641 (5.128) | 5.493 (5.036) | 2.348 |

- 壓力: H1-best `worst B0.354 C3.465 PASS` (貼閾, fund0.0003 B 0.79/0.35)；H2-best `worst B2.580 C3.621 PASS` (厚實)；ctrl `sth0.09` B 1.07/0.20 (2x -0.14) 崩證 0.10 為下界。
- 判定: **首度無偏 PASS** (W4 為 5.13→2.16 FAIL)，但 **保持 Z1 0.12/18/6 為 global**，H1-best 升為 unbiased challenger，H2-best 為 biased contender，FULL-best 0.11/18/9 作長歷史分支，三者并行 shadow/paper，下一步以 12fold median 與 walk-forward 驗 etc15 是否翻轉 12fold -0.699 隱憂再轉正。

## AB — 空頭純度 / 多頭開關 (11 行 both vs short, vt 拆解, lth 掃描)

- 對照: A both q0.3 / B short q0.3 / C short qNone / D both qNone × vtNone/vt0.012 + E lth [0.88,0.92,0.98] 在 both q0.3 vt0.012 上。

| 組 | H2 | B | C | FULL | 12mean/med | lpr/spr |
|---|---|---|---|---|---|---|
| A1 both q0.3 vtNone (Y1b) | 4.091 | 4.496 | 3.479 | 1.861 | 1.810/3.178 | 0.0071/0.6187 |
| A2 both q0.3 vt0.012 (Z1) | 5.013 | 7.042 | 4.666 | 2.064 | 1.655/2.479 | 0.0071/0.6187 |
| B1 short q0.3 vtNone | 3.966 | 3.932 | 3.355 | 1.843 | 1.78/3.038 | 0/0.6187 |
| B2 short q0.3 vt0.012 | 4.837 | 6.172 | 4.49 | 2.062 | 1.646/2.514 | 0/0.6187 |
| E lth sweep 0.88/0.92/0.98 | 5.013 全等 | 7.042 全等 | 4.666 全等 | 2.064 全等 | 1.655 | 0.0071 惰性 |

- 尾端 `both > short`: A2 vs B2 H2 +0.176 B +0.87 C +0.176 FULL +0.002 — 雖 long 僅 1 筆 (0.7% 持倉)，該筆為正貢獻，切 short 反降銳度。
- vt 僅尾端: A2 vs A1 H2 +0.92 B +2.54 C +1.18 但 12mean -0.155 med -0.699 (復現 W1)。
- q0.3 尾端有效: D2 both qNone vt0.012 H2 4.565/B4.464/C4.21 vs A2 5.013/7.042/4.666 → q0.3 尾端 +0.45/+2.57/+0.45；FULL +0.146。
- lth 惰性: q0.3 後 long 候選僅 1 筆極端 |logit|，0.88→0.98 全等，無需調參；真關多頭用 `side='short'`。
- 判定: **尾端不切 short-only (B代價-0.87)**；**不調 lth**；**全歷史保留 both q0.3**；short-only 僅作風險備用。

## AC — 權重與分散再探 (28 行: AC1 14 + AC2 2rp + AC3 12)

- AC1 雙基線 × 7 權重 `wETC 0.2-0.8`：
  - Z1 基線 `best_BC=w0.5 (B7.042 C4.666 B+C11.708)` gain vs50=0.000 — **50/50 即最優**，peak 平坦 ±0.02 (H2 pk w0.4 5.030 +0.017, FULL pk w0.6 2.072 +0.008)。
  - Y1b 無vol `best_BC=w0.3 (B4.899 C3.506 B+C8.405) gain+0.430` — 驗證權重隨基線翻轉 0.5 (Q1 0.1 → Y1b 0.3 → Z1 0.5)。
  - corr 0.119 恆定，分散有效。
- AC2 風險平價 (Z1): H2 vol ETC0.020964 TRX0.011686 → wETC_rp0.3579；FULL vol 0.0238/0.0193 →0.4478。H2-rp 全面劣於50/50 (B-0.287 C-0.252)；FULL-rp 差<0.06 — **不採 1/vol，維持 50/50**。
- AC3 第三腿 (Z1 50/30/20+60/30/10, AVAX/SHIB/DOGE/BTC/SOL/BCH各2): **0/12 beats_both**，最佳 BCH 60/30/10 H2 4.653 Δ-0.36 FULL2.085 Δ+0.021 未雙勝，turnover 0.158-0.168>dual0.15 — **REJECT**，與 W4/Z3 0/36 一致。
- 判定: **維持 50/50**；不採 rp；第三腿 REJECT。

## AD — 新因子 / 新規則挖掘 (16 行: AD1 10 + AD2 4 + AD3 2)

- AD1 單點變異 5 + 隨機 5 (seed42, vocab23): H1-best idx03 `[5,2,7,2,7,11,15,4,4,6,6,10]` (FOMO→LOG_VOL) H1 3.863→H2 5.507 但 **B -2.511 崩** C 4.817 FULL 1.274 <base 1.861，**FAIL 雙勝**；隨機 5 行 H1 全 <1.5 且 H2 <1.0；**overall_AD1_found_better=False**。
- AD2 take_profit [None,0.06,0.10,0.15] on Y1b vtNone: tp0.06 H2 +0.32 B平 C+0.32 但 FULL -0.14，tp0.10/0.15 幾乎等效 None (少觸發)，turnover +0.007；**overall_TP_effective=False**，維持 tp=None。
- AD3 AVAX/SHIB 單幣 TRX-spec q0.3: AVAX H2 1.156 B -1.318 C 1.935, SHIB H2 -0.163 B 1.229 C -0.193 — 皆 H2<2.0，**overall_third_leg=False**。
- 判定: **AD 0/3 PASS**，無新賽道，鎖定不動 ([3,2,7,2,7,11,15,4,4,6,6,10] + tpNone)。

---

## 全局收斂 (LOCK)

| 角色 | 配置 | 尾端 H2/B/C (fee2x) | FULL | 12fold mean/med | 定位 |
|---|---|---|---|---|---|
| **LOCK global** | **Y1b both q0.3 vtNone ETC0.88/0.12/cd18/None/ts24 + TRX0.85/0.12/cd6/0.05/ts24 50/50** | 4.091/4.496/3.479 (3.77/4.15/3.16) | 1.861/4.18x mdd0.89 | mean1.81 med3.18 9/12正 | 生產全歷史平衡鎖 |
| **COND tail overlay** | **Z1 vt0.012/w12 同參 (原鎖)** | 5.013/7.042/4.666 (4.11/6.51/4.21) | 2.064/5.03x | mean1.655 med2.48 | 尾端 +2.5sh，僅條件疊加 (shadow gate, 不作 global) |
| AA challenger (unbiased) | 0.10/15/9 vt0.012 | 5.581/2.175/5.436 (5.12/1.73/4.97) | 2.025 | - | 無偏首 PASS，B 貼閾，需 12fold 復核 |
| AA contender (biased) | 0.12/15/9 vt0.012 | 6.835/4.805/5.558 (6.34/4.27/5.11) | 2.198 | - | 有偏峰，需 H1 獨立前瞻驗 |
| AA FULL-best | 0.11/18/9 vt0.012 | 4.981/5.641/5.493 | 2.348 | - | 長歷史分支 |
| WEIGHT | 50/50 on Z1 | - | - | - | AC 證平坦，rp 與第三腿 REJECT |
| SHORT | both > short (AB) | - | - | - | 不切 short-only，lth 惰性 |
| FACTOR/TP/COIN | 基線不動 | - | - | - | AD 0/3，不開新賽道 |
| SHADOW | Y1b_main1.0 gate (W3) | B3.45 C1.96 FULL2.62x/0.83 cov0.65 | - | - | 唯一 PASS gate，plain 平衡器 |

## 下一步

1. AA 三分支的 12fold median 與 walk-forward (W5): 驗 etc15 是否翻轉 Z1 的 12fold -0.699 隱憂，若 median 回升則可轉正 challenger。
2. 紙上 shadow 以 Y1b_main1.0 為載體，并行跑「Y1b global + Z1/AA challenger/contender 條件疊加」的 gated 雙帳 (coverage 0.65 路線)，不直接替換 global。
3. 若 W5 復核 AA 12fold 仍跌，則回歸 W 結論: **Y1b 為 global，Z1 為 tail 條件疊加，AA 作研究分支**；若 AA median 回升>3.0 且 worst 改善，則升級 challenger 為新 global 候選。
4. 不再擴第三腿 / 權重動態 / TP / 隨機公式盲搜；下一輪若開新賽道走結構化 RL (run_q2 路線) 或 LLM arity-safe 變異。

Artifacts: AA `research/run_aa.py → results/backtest_AA.json (20) / docs/STRATEGY_AA.md / logs/aa.log`; AB `run_ab.py → backtest_AB.json (11) / STRATEGY_AB.md`; AC `run_ac.py → backtest_AC.json (28→42明細) / STRATEGY_AC.md`; AD `run_ad.py → backtest_AD.json (16) / STRATEGY_AD.md`; W 收斂 `docs/STRATEGY_W.md`。
復現: `python3 research/run_aa.py && python3 research/run_ab.py && python3 research/run_ac.py && python3 research/run_ad.py`。
