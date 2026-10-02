# REVIEW PLAN — Joint Parameter Search PRP (Codex, feasibility reviewer)

## Verdict up front
RECOMMEND **NO-GO as written** (revise-and-resubmit). The PRP adds a large
new selection surface on top of a search that currently loses to null by
-3.35 Sharpe with 0/10 positive-coin breadth, on the basis of a meter
repair whose benefit was measured only by log-recalculation and was then
contradicted by the actual re-run. Three of the five acceptance criteria
do not gate what they claim to gate, and the rollback is not a rollback
of the shared accounting path.

## 1. Joint search and selection bias — "tune on train" does not fix it

1a. **Current code selects on validation every generation.** `run_walkforward`
(`research/ga_28c_30m_3y.py`, selection fold) scores each genome as
`sel = 0.5 * train_reward + 0.5 * val_reward` and selects/elites on `sel`.
F4 ("tune on TRAIN only, score frozen pair on val once") therefore requires
rewriting the selection loop, not adding a branch. The PRP never names this
line, and its unit test ("validation call receives the train-selected
params") pins the new contract without pinning the deletion of the old one.
A partial implementation that keeps any val-weighted selection silently
preserves the leak F4 claims to close.

1b. **Train-only tuning shrinks but does not remove the winner's curse.**
Selecting max over (formula x param-grid) on train still overfits train;
the frozen-pair val score is unbiased for *that* pair conditional on
selection, but the reported number is max-over-grid-then-scored-once, whose
expectation is still upward-biased by ~sqrt(2 log K) in noise units, with
K multiplied by the grid size (3x3x2 = 18x per formula). The PRP's
mitigation column treats F4 as reducing this risk to zero; it reduces the
val-overfit term only.

1c. **Walkforward compounds selection across folds.** Each selection fold
re-tunes (formula, params); only the last fold's pick reaches the holdout,
but every fold's val window becomes fold i+1's training data — data that was
already used for selection in fold i. That is not test-set leakage in the
narrow sense, but it is repeated selection on overlapping data with 10+
selection events per seed, and the PRP reports only the terminal holdout
number. The per-fold overfit gap (train-selected reward minus frozen val
reward) is the diagnostic that would reveal this, and C5 does not require
recording it — only the selected params.

1d. **The null arm may not calibrate execution params at all.** The null
rewrites returns as iid, which destroys exactly the autocorrelation structure
that cooldown/hold-time params exploit. A cooldown grid can therefore look
"real" for purely mechanical reasons (any hold rule behaves differently on
autocorrelated vs iid bars), in either direction. The chaos test (wider grid
must not let null win more) checks grid-size pathology, not this
autocorrelation mismatch. Missing: an autocorrelation-preserving null
(block bootstrap / phase shuffle) as a second arm, or a stated reason why
iid suffices for threshold/cooldown calibration.

## 2. C1/C2 — C1 is a coin flip, C2 has low power, and their conjunction is ambiguous

2a. **C1 (real median > null median, n=10 vs 10, no test) has ~50% false-pass
rate under the null.** With identical continuous distributions, P(real median
> null median) is ~0.5 minus ties. As a listed acceptance checkbox it is a
wish, not a gate: half of all pure-noise pilots check it. It is redundant
given C2 and dangerous without it, because "C1 passed, C2 narrowly missed"
will be read as promise.

2b. **C2 (one-sided Mann-Whitney p < 0.10, n=10) needs ~8/10 directional
dominance and still has weak power.** Sign-test arithmetic: 8/10 one-sided
p = 0.055, 9/10 p = 0.011; MW at n=10/10, alpha 0.10 one-sided needs
U ≈ 66+ (≈ pairwise dominance at the same 8/10 level). Power via the sign
approximation: if the true per-seed P(real > null) = 0.7 (a large effect),
P(X ≥ 8/10) ≈ 0.38. For 80% power the per-seed win rate must be ~0.85+.
Against the measured baseline (gap -3.35, real wins 2/10) the pilot must
reverse a 6-seed deficit just to reach the gate — possible, but the PRP
should state that C2 is powered only for very large effects and that a
C2-miss is therefore weak evidence either way.

2c. **Two-sided-p discipline is good (C2's direction clause) but incomplete.**
The meter-repair pilot already produced a "significant" MW p = 0.038 in the
WRONG direction (post_repair U=22). Any pilot-history pooling, seed
replacement, or "one more seed" after seeing the data invalidates the
p-value; the PRP needs a pre-registered seed list (it has: 101-110) plus a
no-reseed / no-drop rule, which is currently absent.

## 3. The meter-repair pilot predicts this PRP will fail — say so: it does

Post-repair numbers (`results/pilot_meter_repair_analysis.json`):
real median -3.72 vs null median -0.36, gap **-3.355** (worse than pre-repair
-0.434), real wins 2/10, MW significant in the wrong direction (p=0.038),
real positive_coins median **0.0** (0/10 breadth), real range entirely
negative except one run (+0.64 vs null best +3.25).

Three consequences for this PRP:

3a. The reliability doc's optimistic §5 (+0.0959 gap, 7/10 wins) was a
**log-recalculation with the patched reward, never a re-run**. The actual
re-run contradicts it (gap -3.35, 2/10). The PRP's C4 ("repaired reward
shows zero saturation and rankable-run std greater than pre-repair") treats
the meter as fixed, but the only post-fix search evidence says the fixed
meter made Sharpe separation worse. C4 measures the meter; nothing in C1-C5
requires the search to beat null *with the repaired meter before* the joint
space is added — that pre-gate is missing and should be a precondition.

3b. **0/10 positive-coin breadth means the real-arm selector currently finds
degenerate portfolios.** Adding threshold/cooldown gives the optimizer more
degrees of freedom to exploit exactly this degeneracy (e.g. thresholds that
gate into a one-direction drift fit). C3 (mean_position < 0.25 on 6/10) does
not catch this: a portfolio can have small mean position while its Sharpe
comes from a drift leg plus costs asymmetry. Breadth (positive_coins,
C-gate "c_breadth" from the pilot: 0/10 both batches) should be a joint
accept gate, not an unlisted observation.

3c. The base rate says the null wins. Joint search multiplies K by ~18x;
under the null both arms gain winner's-curse lift, and there is no reason
given to believe the real arm gains more — the pilot says the real arm's
selection is currently worse at converting freedom into holdout Sharpe.
Expected outcome: gap stays negative, possibly more negative. That is a
legitimate NO-GO per the PRP's own risk table, but the pilot budget
(25 x 16 x 10 x 20) is then spent to re-confirm a negative with a noisier
instrument. Cheaper pre-gate: re-run the *current* (formula-only) search
with the repaired meter on the same 10 seeds first; only proceed to joint
search if gap ≥ ~0.

## 4. What is missing — failures the PRP would not see

4a. **Overfit-gap blindness.** No diagnostic records train-selected reward vs
frozen val reward per fold. If the joint space overfits train harder, the
val score just looks "a bit worse" — indistinguishable from regime shift.
Require per-fold: train reward of selected pair, frozen val reward,
their difference, and the argmax margin (how much the winner beat the
runner-up on train; thin margins + large val drops = curse signature).

4b. **Turnover/cost confounding.** Thresholds change turnover by orders of
magnitude; fees/funding then dominate Sharpe differences between grid
points. A "params win" may be a cost-avoidance artefact (trade less, lose
less to costs), not signal. Missing: report turnover and cost attribution
per selected pair (the repo already has `attribute_sharpe_to_costs_28c.py`;
wire it into C5 fields), and a guard that flags selections whose Sharpe
delta is >50% cost-driven.

4c. **Scale/threshold interaction.** Thresholds apply to sigmoid(base_pos)
where base_pos normalization is fit with scale_end on training data. If the
grid is tuned on train but scale drifts per fold, a threshold means
different things per fold — C5 records the params but not the scale they
were selected under, so a "reproducible" record may not reproduce. Record
scale_end and the empirical trigger rate per fold.

4d. **Formula-irrelevance has no falsifier.** Risk table admits params may
dominate; mitigation is "report whether formula or params moved the score"
with no threshold and no gate. Concrete version: ablate — score selected
params with a random formula and selected formula with default params; if
params-only ≥ joint on val, the formula search is decoration and C1/C2 are
measuring execution-fit, not alpha.

4e. **Grid-resolution luck.** A 3x3x2 grid samples 18 points of a continuous
response; the winner's exact position in the grid is luck-dominated, and
null calibration with the same grid inherits the same luck rather than
cancelling it (correlated luck is not zero luck at n=10). Missing: a
grid-jitter robustness check (shift grid by half-step on train, confirm the
selected region is stable) or acknowledge-and-accept.

4f. **Seed/tie-break non-determinism.** No rule for grid ties (equal train
reward → which params?), RNG streams for formula vs params sampling, or
whether the null arm uses identical grid order. Ties broken differently per
arm = uncalibrated selection. Pin tie-breaking (e.g. lowest index) and
shared grid enumeration in both arms.

## 5. Rollback — a CLI flag is not a rollback of a shared accounting path

F1 puts the execution model "inside the 28c accounting path so the GA
evaluator can apply it," i.e. it edits `evaluate()` and helpers shared with
the baseline. `--params-search off` then restores the old evaluator only if
the off-path is bit-identical to today's code — including default argument
values, import-time constants, reshaped signatures (e.g. threading params
or psh_peer through), and floating-point order (sigmoid(threshold=0) is not
bitwise the identity of no-threshold code). One stray default or refactored
helper and every historical number silently shifts while the flag claims
"off = exactly current." Real rollback requires: (i) the off-path be the
unmodified function (new logic in a separate wrapper, not interleaved
branches); (ii) a parity test asserting flag-off output equals pre-change
output bitwise on reference seeds; (iii) the flag defaulting off with the
pilot explicitly opting in per-run (not a global default flip). As specified
— "reverting is a single flag because no accounting math changes when the
flag is off" — the premise is asserted, not tested. C4's meter check does
not cover it.

## Required changes (resubmit checklist)
- [ ] R1. Name and delete the `0.5*train + 0.5*val` selection line; F4 test
      must fail if any val-weighted selection survives.
- [ ] R2. Demote C1 to a report-only statistic or delete it; keep C2 as the
      gate, add no-reseed/no-drop pre-registration, and state the power
      calculation (≈38% power at per-seed win rate 0.7; 8/10 needed).
- [ ] R3. Add a pre-gate: formula-only + repaired meter must reach gap ≥ ~0
      on seeds 101-110 before any joint-search budget is spent.
- [ ] R4. Add per-fold overfit-gap + argmax-margin diagnostics to C5; add
      turnover/cost attribution per selected pair; add the params-vs-formula
      ablation as a gate (params-only must not match joint).
- [ ] R5. Specify null calibration for execution params (autocorrelation
      issue): second null arm or written justification that iid suffices.
- [ ] R6. Make rollback structural: separate wrapper, bitwise parity test on
      flag-off, default-off with per-run opt-in. Plus tie-break, grid order,
      and scale_end/trigger-rate recording rules.
