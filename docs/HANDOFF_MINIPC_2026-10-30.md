# Handoff: AlphaGPT selection-rule → minipc OMP agent

> 寫於 2026-10-30，ubuntu-32gb session 關閉前。minipc 接手後讀這份 + CONVERGENCE 即可開工。

## 1. 現況（一句話）

S0/A1/N0 跑完，shared confound，min 家族雙死，NO-GO。主戰場已切 minipc `~/Projects/AlphaGPT`，commit 9be9969。

## 2. 關鍵數字（判讀用，不用重算）

| arm | gap | beat-hold real/null | coins real/null |
|---|---|---|---|
| S0 re-run | -1.5203±0.0881 t=-17.25 | 0.4% / 25.0% | 21.04 / 13.59 |
| A1 halves-min | -2.1367±0.0806 t=-26.52 | 0.0% / 27.6% | 13.37 / 13.70 |
| N0 odd/even | -1.7843±0.0812 t=-21.97 | 0.0% / 22.8% | 15.86 / 13.28 |

S0-gate: corr(re-run,stored)=1.0000 bit-identical，paired comparison 有效。
Verdict 邏輯：A1 擴大反方向（-1.52→-2.14），N0 跟著動（→-1.78，超 ±0.20 容限），按 FINAL_SPEC §Decision rule = shared confound，不是 partial win。

## 3. minipc 環境（已驗）

- repo: `~/Projects/AlphaGPT` @ 9be9969，四個 results 子目錄各 1000 json
- venv: `.venv2`，torch/numpy/scipy ok
- data: `data/data_3y/1h/` 28 幣全到，`data/funding_binance/` ok
- smoke: `selection_free_diag_28c.py --limit 5` fidelity 1.0000（15 秒）

## 4. 待辦（按序，細節見 CONVERGENCE §8 + FINAL_SPEC FM cost note）

1. **FM 決策**：真 joint-evolution median fitness（~4x 成本，每代每基因跑全 4 selection 折）還是降級為診斷？min 家族雙死，median 同類聚合，翻案概率極低。需要主人拍板才燒錢。
2. **B1 pilot**（150 seeds/arm）：`evaluate()` 加 `refit_halves` 參數，每半重 fit scale + smoother 重啟 + 丟前 50 bars，成本 1.4-1.6x。
3. **不做的事**：不加因子、不重訓上線、不談 Kelly（見 CONVERGENCE §8）。

## 5. 預熱指令（接手 agent 第一條跑這個）

```bash
cd ~/Projects/AlphaGPT && git log --oneline -1 && ./.venv2/bin/python tools/selection_free_diag_28c.py --dir results/f1h41_iter1000 --features 41 --bars 1h --scale-windows --limit 5
```

期望：5 個文件，C1 rows 20，fidelity 1.0000，~15 秒。過了環境就是熱的。

## 6. 禁止

- 不許動 `CATREADME.md` / `plans/P0P2-PRP.md`（舊時代文件，歸檔不管）。
- 不許改 `results/f1h41_*` 已有 batch（只讀）。
- 新 batch 寫新目錄，不許覆蓋。
- 任何實盤/上線話題直接拒絕，引用 CONVERGENCE §8。
