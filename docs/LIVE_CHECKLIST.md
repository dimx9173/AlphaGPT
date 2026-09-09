# LIVE_CHECKLIST.md — AlphaGPT P3 ops: paper dry-run -> live gates

Date: 2026-09-06 (UTC). Scope: ETC+TRX aster perp 2x basket (run_paper2.py).
Live trading is NOT enabled by this checklist. No orders were placed.

## 1. What a paper run needs (no orders placed)

Paper repro (`run_paper2.py`) is offline backtest-style simulation:
- Inputs: `data/data_15m_3y/ETC.csv`, `data/data_15m_3y/TRX.csv` (15m bars, aggregated x16 blocks).
- Params: FORMULA=[3,2,7,2,7,11,15,4,4,6,6,10], PORT={ETC:0.5,TRX:0.5},
  BEST={ETC:(0.88,0.12,12,None), TRX:(0.85,0.15,6,0.05)}, FEE=0.0004, FUND=0.0005, LEV=2.0.
- Needs: NO env keys, NO venue flags, NO network. Run: `python3 research/run_paper2.py`.
  Output: stdout stats line + `results/paper_trades2.json`.
- Runner paper mode: there is NO dedicated paper/dry-run flag in `strategy_manager/runner.py`.
  `run_loop()` always calls live brokers (`SolanaTrader`, venue brokers). Paper safety today =
  only run `run_paper2.py` offline. Do NOT start `StrategyRunner.run_loop()` for paper.

Live runner (`strategy_manager/runner.py`) needs (only when going live):
- `ExecutionConfig.validate_env()`: `QUICKNODE_RPC_URL`, `SOLANA_PRIVATE_KEY` (real, non-placeholder).
- Venue brokers wired in `StrategyRunner(brokers=...)`; default = solana only.
- Risk: `strategy_manager/config.py` StrategyConfig + RiskConfig (env overrides below).
- Kill-switch file: `STOP_SIGNAL_PATH` (default `./STOP_SIGNAL`); SIGTERM/SIGINT writes STOP.

## 2. Env keys needed

Copy first, never commit: `cp .env.example .env` (.env is gitignored).

| Key | Needed for | Value for paper | Value for live (ETC+TRX aster perp 2x) |
|---|---|---|---|
| QUICKNODE_RPC_URL | solana live init | unset/placeholder OK (paper needs nothing) | real RPC URL |
| SOLANA_PRIVATE_KEY | solana live init | unset/placeholder OK | real base58 or JSON-array key |
| VENUES_ENABLED | venue gating | `solana` or unset (paper ignores) | `aster` (or `aster,...` subset) |
| ASTER_TESTNET | aster endpoint | `true` (default) | `true` for testnet paper/live-test; `false` ONLY for real live |
| ASTER_USER_ADDRESS | aster validate_env (when aster enabled) | testnet main wallet | main wallet (mainnet when ASTER_TESTNET=false) |
| ASTER_SIGNER_ADDRESS | aster validate_env | testnet agent wallet | agent wallet |
| ASTER_SIGNER_PRIVATE_KEY | aster validate_env | testnet agent key | agent key (API-wallet, small scope) |
| HYPERLIQUID_* | only if hyperliquid enabled | leave blank | leave blank unless HL enabled |
| RISK_DAILY_LOSS_PCT | circuit | default 0.10 | 0.10 |
| RISK_MAX_DRAWDOWN_PCT | circuit | default 0.15 | 0.15 |
| RISK_MAX_SINGLE_EXPOSURE_SOL | spot size cap | default 1.0 | 1.0 (spot path only) |
| RISK_BLACKLIST | blacklist | empty | empty (add bad symbols on incident) |
| RISK_VELOCITY_MAX / RISK_VELOCITY_WINDOW_SEC | velocity | 3 / 60 | 3 / 60 |
| RISK_CIRCUIT_COOLDOWN_SEC | cooldown | 300 | 300 |
| PERP_MAX_LEVERAGE | perp gate | default 3 | **2** (strategy uses 2x; set 2 to enforce) |
| PERP_MAX_NOTIONAL_USDT | perp gate | default 500 | 500 (tighten to planned size before live) |
| PERP_MAX_FUNDING_RATE | perp gate | default 0.001 | 0.001 (matches FUND=0.0005 assumption w/ headroom) |
| STOP_SIGNAL_PATH | kill-switch file | default STOP_SIGNAL | default STOP_SIGNAL (know its abs path) |

`AsterConfig.validate_env()` raises if aster enabled but any of the 3 ASTER_* keys empty.
`HyperliquidConfig.validate_env()` same for HL keys. Both no-op when venue not in VENUES_ENABLED.

## 3. Venue / risk settings for ETC+TRX aster perp 2x

- Venue: `VENUES_ENABLED=aster`, `ASTER_TESTNET=true` until live approval; futures endpoint
  `https://fapi.asterdex-testnet.com` (test) vs `https://fapi.asterdex.com` (live).
- Broker: `execution/brokers/aster.py` (V3 EIP-712 via AsterSigner; deadman supported).
- On start: call `refresh_deadmen(timeout_sec=60)`; do not trade if deadman returns FAILED.
- Risk gates per open (`RiskEngine.check_perp`): leverage<=PERP_MAX_LEVERAGE (set 2),
  notional<=PERP_MAX_NOTIONAL_USDT, |funding|<=PERP_MAX_FUNDING_RATE, symbol not blacklisted.
- Position sizing (spot legacy): ENTRY_AMOUNT_SOL=2.0 capped by RISK_MAX_SINGLE_EXPOSURE_SOL;
  perp notional must be set explicitly before live (no auto-size for aster in runner yet).
- Strategy thresholds (from run_paper2.py BEST): ETC long 0.88 / short 0.12 / cd 12 / no SL;
  TRX long 0.85 / short 0.15 / cd 6 / SL 0.05. PORT 50/50, LEV 2.0, FEE 0.0004, FUND 0.0005.

## 4. Paper command (safe, offline)

```
cd /home/brian/project/AlphaGPT
python3 research/run_paper2.py
# expect (2026-09-06 data): trades=478 final_x=2.9883 sharpe=0.899 mdd=0.7846 n=6580
# committed baseline (paper2.log): trades=477 final_x=2.927 sharpe=0.891 mdd=0.7846 n=6570
```

## 5. Promote-to-live gates (ALL must pass)

1. P2 pass: full `pytest -q` green (2026-09-06: 112 passed).
2. Paper 1 week no-exceptions: run paper loop daily on fresh data, zero unhandled exceptions,
   stats drift within tolerance (trades +/-5, final_x +/-0.15, sharpe +/-0.10) or re-baseline with reason.
3. Fee2x pass (pre-check 2026-09-06: FEE=0.0008 -> trades=478 final_x=2.0386 sharpe=0.721 mdd=0.8161; PASS final_x>1, sharpe>0): re-run paper with FEE doubled (0.0008) and confirm final_x > 1.0 and sharpe > 0;
   record numbers in this file before approval.
4. Testnet live-test: VENUES_ENABLED=aster + ASTER_TESTNET=true with small notional, deadman ok,
   STOP file works (create STOP_SIGNAL -> loop exits), reconcile ok.
5. Human approval + set ASTER_TESTNET=false only at promotion; start with min notional.

## 6. Kill-switch (STOP_SIGNAL / risk circuit)

- File: `$STOP_SIGNAL_PATH` (default `./STOP_SIGNAL`). Runner checks every loop stage
  (`_handle_stop_signal`); content STOP/STOPPED/empty (or file with that text) breaks loop.
  SIGTERM/SIGINT auto-writes STOP. Consumed runs rewrite STOPPED.
- Immediate stop: `echo STOP > STOP_SIGNAL` (abs path per env) from repo root.
- Risk circuit: `RiskEngine.check_circuit` blocks new entries on daily_loss < -RISK_DAILY_LOSS_PCT
  or drawdown > RISK_MAX_DRAWDOWN_PCT; cooldown RISK_CIRCUIT_COOLDOWN_SEC (300s). Check logs for
  `CIRCUIT OPEN (reason)`.
- Venue deadman: `refresh_deadmen(60)` per venue; exchange auto-cancels if runner dies.
- After stop: verify flat (reconcile), cancel open orders on Aster UI/API, then investigate.

## 7. This-run results (2026-09-06)

- pytest: 112 passed, 0 failed (`python3 -m pytest tests/ -q`; torch jit DeprecationWarning only).
- Paper repro: MATCH with small drift (data refresh: n 6570 -> 6580, +10 bars).
  Baseline paper2.log: trades=477 final_x=2.927 sharpe=0.891 mdd=0.7846.
  Re-run: trades=478 (+1) final_x=2.9883 (+0.061) sharpe=0.899 (+0.008) mdd=0.7846 (same).
  By-coin: ETC 185->186 (+1), TRX 292->292. Within tolerance. No live keys touched, no orders placed.
- Note: re-run overwrote `results/paper_trades2.json` (git status shows only pre-existing STRATEGY_Q5.md
  modification + untracked run_r*.py / run_p2.py; paper_trades2.json appears gitignored or untracked-unchanged).
  `git log` on data CSVs/paper2.log empty (data files untracked or history shallow).

## 7b. This-run results (2026-09-08, E10 gates re-verify)

- pytest: 112 passed, 0 failed, 0 errors (`python3 -m pytest --tb=no -p no:warnings -q --junitxml=/tmp/pt.xml`; torch jit DeprecationWarning suppressed).
- E10 check: `python3 research/run_e10.py --check` PASS (global=Y1b, gate=1.0, no promotion, docs present).
- paper baseline (fee 0.0004): trades=478 final_x=2.9883 sharpe=0.899 mdd=0.7846 n=6580 by={ETC:186, TRX:292}. Gate `478±5 / 2.98±0.15 / 0.90±0.10` PASS.
- fee2x (fee 0.0008): trades=478 final_x=2.0386 sharpe=0.721 mdd=0.8161. Gate `final_x>1 sharpe>0` PASS.
- turnover: Y1b 0.082523 (E3 FULL) < 0.16 PASS.
- 12fold Y1b: mean 1.81 > 0, median 3.178 > 1.5, n_pos 9 >= 8 PASS.
- shadow adopted `Y1b_main_thr1.0_w200`: coverage 0.652 in [0.5,0.7], FULL 2.6249x/0.831/mdd 0.6861, B_sh 3.452 > plain 2.509, C_sh 1.966 > plain 1.775 PASS.
- challenger stress: AA-H1 worstB 0.354 > 0 borderline, worstC 3.465 > 1 PASS; AA-H2 worstB 2.58, worstC 3.621 PASS.
- corr note: gate uses leg-net-return corr (AA 0.069/0.065, AC Z1 0.119/Y1b 0.155 < 0.20 PASS), NOT spot-price corr (H2 spot 0.28/FULL 0.43 for reference only).
- No live keys touched, no orders placed; baseline file restored after fee2x run.

## 7c. This-run results (2026-09-09, Y1b wiring)

- `best_meme_strategy.json` created (Y1b FORMULA, source E10 locked_params); runner falls back to `config.FORMULA` if missing.
- `PAPER_MODE=1` gate added: `run_loop()` refuses live trading, points to `research/run_paper2.py`.
- New `strategy_manager/y1b_basket.py`: offline Y1b ETC+TRX signal generator mirroring paper2 engine; live check `latest_signals() -> {ETC: -1.0, TRX: -1.0}`.
- New `tests/test_y1b_basket.py` (3 tests): locked params + discrete signals + paper refusal. Full suite: 115 passed, 0 failed.
- pytest 115 passed (112 + 3 new), E10 check PASS, paper baseline re-verified 478/2.9883/0.899.
- Runner still legacy meme-scan path for live entries; Y1b basket signals available for wiring but NOT auto-trading. No orders placed.

## 7d. This-run results (2026-09-09, Y1b executor shadow)

- New `strategy_manager/y1b_executor.py`: signals -> perp gate -> broker. Default dry-run; live only if `Y1B_LIVE_ENABLED=1` + no PAPER_MODE + explicit `dry_run=False`.
- Size = min(Y1B_NOTIONAL_USDT=50, PERP_MAX_NOTIONAL_USDT) / price; lev 2; symbols ETCUSDT/TRXUSDT.
- New `tests/test_y1b_executor.py` (3 tests): gated plans + dry-run no-order + oversize cap.
- Shadow mock run (price 20, notional 50): ETC -1.0 x2.5 / TRX -1.0 x2.5, gate True, all dry_run, `market_open` never called.
- Full suite: 118 passed, 0 failed. No live keys, no orders.

## 7e. This-run results (2026-09-09, executor safety close)

- `run_once` now returns (plans, results, sync); preflight = STOP absent + circuit closed + deadman ok.
- `sync_positions`: want==0 flattens venue position (live) / reports dry_run_close (shadow); opposite-side flip reported.
- New tests: 3-tuple sync + STOP-blocks-live (market_open zero-call). Full suite: 120 passed, 0 failed.
- paper 478/2.9883/0.899 + E10 check PASS re-verified. Live still default-off; no orders.

## 7g. This-run results (2026-09-09, gates-as-tests + 125 suite)

- New `tests/test_e10_gates.py` (3 tests): E10 lock-vs-config + paper tolerance + artifact gates (12fold/turnover/coverage/fee2x).
- Full suite: 125 passed (122+3), 0 failed. `run_y1b_verify.py` 3-in-1 PASS.
- No orders, no keys. Live still default-off; human testnet gates pending (§5/§8.4).

## 7h. This-run results (2026-09-09, post-commit re-verify)

- HEAD `aee13f3` clean (9 files, 554+/5-). Full suite 125 passed, verify 3-in-1 PASS, E10 check PASS.
- No orders, no keys. Live default-off; human testnet gates pending (§5/§8.4).

## 7f. This-run results (2026-09-09, runner Y1b wiring)

- `StrategyRunner.run_y1b_once()`: delegates to y1b_executor with aster broker + runner risk; legacy meme path untouched.
- Runner-level shadow test passes (ETC+TRX dry_run, market_open zero-call). Full suite: 121 passed, 0 failed.
- Untracked wiring files: best_meme_strategy.json, y1b_basket.py, y1b_executor.py, test_y1b_basket.py, test_y1b_executor.py. No orders.
- `research/run_y1b_verify.py`: permanent 3-in-1 wiring check (shadow + STOP-block + paper 478/2.9883) PASS.


---

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
