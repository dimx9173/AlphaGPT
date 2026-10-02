# PLAN: Joint Parameter Search (joint-param-search)

Architect verdict: PRP architecture section is wrong in three places
(reuse target, threshold input, cost math). This plan implements the
corrected version. Sign-off after PRP corrections in T0.

## Corrected architecture (replaces PRP section 3)

```
evaluate(formula, params, split) -> {reward, sharpe, turnover, ...}
   |
   +-- signal = StackVM(formula, features)          (unchanged)
   +-- z = sig / fit_scale                           (pre-tanh, unbounded)
   +-- magnitude = 0.25 * smooth_causal(tanh(z), 5)  (unchanged)
   +-- gate: direction = long if sigmoid(z) > long_th,
   |         short if sigmoid(z) < short_th, else flat;
   |         survivors keep their continuous magnitude, flat gets 0
   +-- cooldown: min-hold after a direction flip (event timing only)
   +-- roll(1)                  execution lag, unchanged
   +-- existing accounting: fees, funding, equity, Sharpe (unchanged)
```

Reuse target is `research/accounting_28c.py` (numpy, tested), NOT
`model_core/backtest.py` (torch, single-book engine, liquidity-aware).
The new gate lives in `accounting_28c` next to `position_from_signal`;
the GA file owns signal construction, reward, and grid wiring only.
Stops / vol-target / quantile are OUT of the grid — see T0 for why.

## Concrete signatures

```python
# research/accounting_28c.py (appended; position_from_signal untouched)

def apply_threshold_cooldown(
    z: np.ndarray,               # pre-tanh signal / fit_scale, unbounded
    magnitude: np.ndarray,       # 0.25 * smooth_causal(tanh(z), 5), same length
    long_th: float = 0.85,       # on sigmoid(z)
    short_th: float = 0.15,      # on sigmoid(z)
    cooldown_bars: int = 0,      # min hold after a direction flip
) -> np.ndarray:
    """Direction gate + cooldown over a continuous-magnitude position.
    Returns the PRE-LAG series; the caller applies roll(1).
    Direction timing must match position_from_signal on the same z."""

# research/ga_28c_30m_3y.py

class ExecParams(NamedTuple):   # frozen by construction
    long_th: float
    short_th: float
    cooldown_bars: int

EXEC_GRID: tuple[ExecParams, ...]   # 3 x 3 x 2 = 18 points; data, not code

def evaluate(formula, maps, returns, start, end, funding_mask, scale_end,
             bars, psh_peer=None, funding_by_coin=None, reward_version='v1',
             grammar=None, params: ExecParams | None = None) -> dict:
    # params=None or DEFAULT_PARAMS reproduces today's evaluator exactly
    # (flag off == current behaviour; default is pass-through:
    #  long_th=0.0, short_th=1.0, cooldown_bars=0, gate never closes)

def select_params(formula, maps, returns, train_range, funding_mask,
                  scale_end, bars, funding_by_coin, reward_version,
                  grammar, grid=EXEC_GRID) -> ExecParams:
    # argmax of train-only reward; frozen before the validation call

def evaluate(frozen_formula, frozen_params, maps, returns, val_range, ...)
```

Fold record gains five fields: `fold`, `formula`, `params{long_th,
short_th, cooldown_bars}`, `train_sharpe`, `validation_sharpe`.

## Tasks (test-first order)

### T0. Correct the PRP record
- **files:** `.prpcc/joint-param-search/PRP.md`
- **done_when:**
  - F1 reuse target changed from `MemeBacktest._apply_cooldown/_apply_stops`
    to `accounting_28c` (numpy), reuse requirement stated as **event-timing
    parity only**, with the divergence reason recorded (torch tensors,
    liquidity gate, per-book sizing vs numpy per-coin GA legs)
  - section 3 diagram: threshold input changed from `sigmoid(base_pos)` to
    `sigmoid(z)` pre-tanh, with the reason (sigmoid(base_pos) lives in
    [-0.52, +0.52] on this evaluator, so long_th=0.85 can literally never
    fire and short_th=0.15 almost never — the diagram as written would
    search a dead threshold axis)
  - cost section states the real 9.5x multiplier (see assessment) and its
    consequence for the 2x budget: grid limited to the 9 train points, or
    params fitted only at the final selection fold (see T4 decision)
  - grid named explicitly `long_th x short_th x cooldown_bars`; sl / ts /
    vt / vw / q recorded as deliberately excluded from the grid: vt
    reopens the closed F_C vol-scaling question (commit 15f7a50), and
    stop-loss/take-profit/time-stop inside `position_from_signal`
    (accounting_28c.py:81-90) are dead code — `exc` is never updated, so
    a stop never fires (accounting_28c.py:76-79)

### T1. Unit tests for the continuous gate (RED first)
- **files:** `tests/test_exec_gate_28c.py` (new)
- **done_when:**
  - direction-flip bars of `apply_threshold_cooldown(z, mag, ...)` match
    the discrete events `position_from_signal(z, ts, ...)` produces on the
    same z (event-timing parity, tested against the semantic reference)
  - cooldown_bars=2 holds a flip for exactly 2 bars, mirroring
    `tests/test_accounting_28c.py::TestPositionFromSignal::test_cooldown`
  - a z series that never crosses threshold returns all-zero positions,
    and a genome scored through it gets a finite, rankable reward —
    never a cliff, never -50
  - survivors keep their continuous magnitude (not snapped to +/-1)
  - all of the above FAIL before T2 exists

### T2. Implement the gate in accounting_28c
- **files:** `research/accounting_28c.py` (append-only; `position_from_signal`
  untouched — its tests pin its behaviour)
- **done_when:**
  - `apply_threshold_cooldown(z, magnitude, long_th, short_th,
    cooldown_bars)` exists per the signature above: numpy-only, no torch
  - threshold on `sigmoid(z)`; magnitude passes through for survivors,
    0.0 for flat; cooldown freezes direction after a flip, then releases
  - returns PRE-LAG; caller applies `roll(1)` exactly where the current
    evaluator does (ga_28c_30m_3y.py:437)
  - T1 green; existing `tests/test_accounting_28c.py` still green

### T3. Flag-off equivalence test (RED first)
- **files:** `tests/test_ga_params_search_28c.py` (new)
- **done_when:**
  - `evaluate(..., params=DEFAULT_PARAMS)` is numerically identical to
    `evaluate(...)` on a fixed small fixture (2 coins, few hundred bars)
  - flag-off path scores bit-for-bit the current evaluator's output
  - test FAILS before T4 exists

### T4. Wire params into the evaluator, walkforward, and gate rebuild
- **files:** `research/ga_28c_30m_3y.py` (evaluate path ~428-512,
  `run_walkforward` ~513-608, regime-gate position rebuild ~706-712,
  CLI args ~627-631)
- **done_when:**
  - `ExecParams` + `EXEC_GRID` (18 points) exist; `--params-search` flag,
    default off; flag off reproduces current scores exactly (T3 green)
  - selection: all grid points scored on TRAIN only; `select_params`
    freezes the argmax; exactly ONE validation `evaluate` with the frozen
    pair (F4) — asserted in code, not by comment
  - selection-fold records carry all five C5 fields
  - the regime-gate position rebuild (which duplicates the position math
    at lines 706-712) applies the identical gate, so the gate scores the
    same book `evaluate` scored
  - decision recorded in the artifact: grid on train only (9 points, see
    T0) vs params fitted only at the final selection fold — whichever
    keeps the measured cost inside the corrected budget

### T5. Smoke run (integration)
- **files:** no new module; existing GA entry point with smoke flags
- **done_when:**
  - 2-fold, 2-generation, `--params-search on` run completes on real data
    and every selection-fold record contains all five C5 fields
  - wall-clock cost per genome (on vs off) logged and compared against the
    predicted multiplier before the pilot burns seeds

### T6. Paired pilot driver + real-minus-null report
- **files:** small pilot driver script (research/ or tools/), results note
- **done_when:**
  - 10 real vs 10 null (seeds 101-110, walkforward, meter-repair pilot
    settings), matched seeds across arms
  - every reported number is real-minus-null, never an absolute Sharpe (F5)
  - C1/C2/C3 as stated: direction-asserted one-sided test; a two-sided p
    on a negative gap is recorded as FAIL
  - either GO on C1-C3 or a recorded NO-GO — a loss is not followed by
    widening the grid
