# PRP Plan: 28c Accounting Correction

## Task 1 — Shared accounting module and unit tests
- Add `research/accounting_28c.py` with `scheduled_funding_rates`, `position_from_signal`, `accounting_bar_returns`, `account_portfolio`, and `metrics`.
- Add `tests/test_accounting_28c.py` covering event cadence, signed funding, notional turnover, compounding, MDD, daily Sharpe, validation, boundary policy, and account-level insolvency.

## Task 2 — Migrate GA and threshold fit
- Replace local event masks, additive MDD, and per-bar Sharpe in both scripts.
- Add fixed `equity-compound-v2` metadata and portfolio-level `solvent` acceptance fields.
- Recalibrate daily-Sharpe reward constants explicitly; compute net once over the full aligned series and slice train/validation/lockbox so no artificial split-boundary entry fee is charged.

## Task 3 — Migrate paper replay
- Replace portfolio equity/metrics and leg accounting.
- Compute costs from weighted signed notionals; do not add a second weight-turnover fee.
- Keep broker-free/offline safeguards and JSON compatibility.

## Task 4 — Migrate 28c trainer reward
- Reuse the same `position_from_signal` and accounting helpers for `TRAIN_UNIVERSE_28C=1`; thread UTC timestamps through the trainer.
- Leave legacy five-coin behavior unchanged.
- Add a smoke test for the 28c accounting path.

## Task 5 — Verification and review
- Run unit, integration, validator, and smoke checks.
- Run one-generation GA and paper replay.
- Run Codex read-only review; do not adopt results unless verdict is PASS.
