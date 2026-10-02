# PLAN: Cost Attribution Logging (cost-attribution-logging)

Architect verdict: PRP is accurate on the facts. `accounting_bar_returns`
returns net only, and `evaluate()` derives `portfolio_sharpe` from the net
series, so there is no gross Sharpe anywhere in the search path. The design
below keeps it that way: additive logging only, selection-neutral by
construction.

## 1. Confirmed facts (read, not assumed)

`accounting_bar_returns` (`research/accounting_28c.py:125-135`) is exactly:

```python
prev_pos = np.roll(position, 1)
prev_pos[0] = 0.0
turnover = np.abs(position - prev_pos)

gross = position * asset_return * leverage
fee = turnover * fee_rate * leverage
funding_cost = position * funding_rate * leverage

net = gross - fee - funding_cost
return net
```

The components exist as locals and are discarded; only `net` leaves the
function. `evaluate()` (`research/ga_28c_30m_3y.py:498`) calls it per leg,
stacks leg nets (`pnet = mean(leg_net)`, line 510), and `metrics(pnet)` ->
`daily_sharpe` yields `psh` = `portfolio_sharpe` (lines 511-512). `shape_reward`
(line 544) scores `psh`. Confirmed: every Sharpe on the selection path is net
of costs; no gross Sharpe exists today.

There are ~20 callers of `accounting_bar_returns` across `research/`,
`tools/`, and `tests/`. Any change to its return shape breaks all of them.

## 2. Design: selection-neutral by construction

Three hard constraints drive every choice: no change to `evaluate()`'s
arithmetic order, no change to `shape_reward`, no new reward version. The
selection keys (`reward`, `portfolio_sharpe`) must be bit-identical before
and after.

### 2a. Capturing components: optional out-parameter, NOT a tuple

A tuple return (`return net, gross, fee, funding_cost`) breaks ~20 callers
and is rejected. A sibling function (`decompose_bar_returns`) that
recomputes the components duplicates the arithmetic, which violates PRP F3
("computed from the same bar returns the accounting already produces, not
re-derived") and creates a drift risk between two copies of the same formula.

The minimal option is an optional out-parameter:

```python
# research/accounting_28c.py — only appended lines, existing lines untouched
def accounting_bar_returns(position, asset_return, fee_rate, funding_rate,
                           leverage=2.0, out=None):
    ...
    net = gross - fee - funding_cost
    if out is not None:
        out["gross"] = gross
        out["fee"] = fee
        out["funding_cost"] = funding_cost
    return net
```

Callers that pass nothing execute the identical code path and get the
identical array back. The four arithmetic lines above the `if` are not
moved, reordered, or re-spelled, so float results are bit-identical. This
satisfies F3: the components come out of the same call that produced the
net the search actually used.

### 2b. Where the decomposition lives in evaluate()

Inside the existing per-leg loop, immediately after line 498, one additive
statement per leg (no reordering of surrounding lines):

```python
comp = {}
net = accounting_bar_returns(pos, r, FEE, fnd, LEV, out=comp)
# ... existing leg_net.append(net), leg_bh, metrics calls unchanged ...
leg_gross.append(comp["gross"]); leg_fee.append(comp["fee"]); leg_fund.append(comp["funding_cost"])
```

After `pnet` (line 510), aggregate the same way `pnet` is built:

```python
pgross = np.mean(np.stack(leg_gross), axis=0)
pfee   = np.mean(np.stack(leg_fee), axis=0)
pfund  = np.mean(np.stack(leg_fund), axis=0)
```

Scope note: strategy legs only. The buy-and-hold legs (`leg_bh`) stay
untouched; their Sharpe (`bh_sh`) is a benchmark diagnostic, not a cost
object, and decomposing it adds nothing the PRP asks for.

### 2c. Drag definition (the only convention choice)

Sharpe is nonlinear, so "fee drag" has no canonical meaning until defined.
Use the chained decomposition, which reconciles by construction:

```python
m = lambda s: metrics(s, timestamps)["sharpe"]
gross_sharpe = m(pgross)
mid_sharpe   = m(pgross - pfee)
net_sharpe   = m(pnet)          # == pm['sharpe'], already computed; reuse it
fee_drag     = gross_sharpe - mid_sharpe
funding_drag = mid_sharpe - net_sharpe
```

Then `gross - fee_drag - funding_drag - net = 0` up to float rounding, which
is what C1 checks with `< 1e-9`. Ordering convention is fee-first, then
funding (matching the `gross - fee - funding_cost` order in the accounting).
Document it in one comment; do not parameterise it.

`metrics()` takes any return series, so no new Sharpe code is needed --
`daily_sharpe` is reused as-is.

### 2d. Result JSON: three additive keys

```python
return {...,   # every existing key byte-identical
        'gross_sharpe': gross_sharpe,
        'fee_drag': fee_drag,
        'funding_drag': funding_drag}
```

Turnover is already reported (`'turnover': mean_turn`). The fold-record
writers (`run_walkforward` selection records ~line 656, holdout records
~line 588) may copy the three keys through; that is presentation, not
selection, and `selection_score` never sees them. `shape_reward` and
`reward_version` are untouched -- no v4 exists after this change.

### What is deliberately NOT built

- No tuple return, no sibling recomputation, no new Sharpe helper.
- No per-leg decomposition in the result JSON (PRP asks per-fold/holdout).
- No buy-and-hold decomposition.
- No logging framework, no extra file: ~15 lines across two files.

## 3. Test-first task list

- [ ] T1. Component capture reconciles. Synthetic positions/returns/fees with
      a fixed seed: call `accounting_bar_returns(..., out=comp)`, assert
      `comp["gross"] - comp["fee"] - comp["funding_cost"] == net` exactly and
      that the returned `net` equals a call without `out` elementwise.
      (Red: `out` kwarg does not exist.)
- [ ] T2. Caller-compatibility pin. Assert the default call returns a plain
      `np.ndarray` (not a tuple) and that the existing accounting test files
      pass unmodified -- they exercise the no-`out` path and must not be
      edited to accommodate this change.
- [ ] T3. Drag reconciliation (C1 at unit level). Fixed-seed synthetic
      28-leg panel: build `pgross/pfee/pfund` by mean-stacking, compute the
      chained drags per 2c, assert
      `|gross - fee_drag - funding_drag - net| < 1e-9`. Include a mixed
      long/short panel so funding changes sign across legs.
- [ ] T4. Selection-neutrality pin (C2). Fixed-seed smoke test through
      `run_standard_selection` (or a 1-fold walkforward) with tiny
      population/generations on fixture data: assert the selected formula
      equals the value recorded before the change, and assert the selection
      keys (`reward`, `portfolio_sharpe`) in every `evaluate()` result are
      bit-identical with the new keys stripped versus `git stash` baseline.
      If this test fails, the change does not land -- per the PRP risk
      table this is the one mitigation that matters.
- [ ] T5. Full suite green (C3). Run the repo test suite; the known
      pre-existing failure must be unchanged (same test, same failure, no
      new failures).
