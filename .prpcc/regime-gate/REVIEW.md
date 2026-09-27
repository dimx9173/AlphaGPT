# Codex Review: Regime-Neutral Acceptance Gate

Status: pending

## Verdict
Pending.

## Checklist
- [ ] Benchmark is derived from already-loaded arrays, no new data source.
- [ ] Excess Sharpe is computed from the difference series, not by subtracting Sharpes.
- [ ] Regime label depends only on the benchmark.
- [ ] Walk-forward folds are contiguous, non-overlapping, and exclude the embargo.
- [ ] Net returns are computed once over the full series and sliced per fold.
- [ ] Verdict vocabulary is fail-closed; only `pass` allows adoption.
- [ ] A1/A2/A6 block adoption; `min_oos_portfolio_sharpe` is diagnostic only.
- [ ] Lockbox is read once, after selection, and is not used to build folds.
- [ ] Thresholds in PRP section 4 are unchanged from spec time.
- [ ] Existing 28c suite still green.
