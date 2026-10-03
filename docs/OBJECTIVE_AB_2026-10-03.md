# Reward vs gate: a pure excess objective does not find an edge

Date: 2026-10-03. Commit b4392f5.

## Design

The v3 reward scores absolute Sharpe at w_psh 0.10 and mean-IC at 1.00. The acceptance
gate tests excess over buy-and-hold only. On a universe whose buy-and-hold is +1.42
Sharpe, a book that simply holds long earns reward before it has demonstrated any
timing, and 6 of 8 v3 runs finished sitting at the 0.25 position cap.

The treatment zeroes both drift terms and leaves excess as the whole objective:
`--objective excess_bh`, which is w_mean_ic 0, w_psh 0, w_excess_bh 1.0. `excess` and
`excess_bh` are the same quantity in `evaluate()`, so total weight on excess is 1.30.

Control is the existing gen-10 v3 run: same 8 seeds, same 5 folds, same harness, same
everything except the reward. Only the reward varies.

## Result 1: the reward demonstrably changed the strategy

| real data | v3 | excess_bh | diff | t |
|---|---|---|---|---|
| mean position | 0.218 | 0.073 | -0.144 | -3.13 |
| positive coins | 21.75 | 15.75 | -6.00 | -3.54 |
| at the 0.25 cap | 6/8 | 1/8 | | |

Both significant. The book stopped being long. That part of the hypothesis is
confirmed: the reward was steering the strategy and the steering works.

## Result 2: the steering produced no holdout gain

| reward | real | null | real-null gap |
|---|---|---|---|
| v3 (control) | -3.192 +/- 0.880 | -1.333 +/- 0.398 | -1.860 +/- 0.966 |
| excess_bh | -1.537 +/- 0.187 | -1.066 +/- 0.557 | -0.471 +/- 0.587 |

Real-data excess beats the hold in **0 of 8** treatment runs, against 1 of 8 under
v3. The gap moves from -1.860 to -0.471, a change of +1.389 +/- 1.131, **t = +1.23**.
That is not distinguishable from zero.

## A correction to the analysis script

The first version of `analyze_objective_ab_28c.py` declared a verdict using an
arbitrary 0.5 threshold on the gap change and reported this run as "the pure-excess
reward WIDENS the separation". On t = +1.23 that was wrong. The script now tests
significance and prints the interval, and the comment in the source records why.

This is the same error the null arm exists to catch, one level up: reading a
direction off a noisy number.

## Conclusion

The reward WAS steering the search, and removing the drift terms removed the long
book. It did not produce a strategy that beats buy-and-hold.

Three things are now eliminated by experiment rather than by argument: the search
space (variable-length A/B), the search effort (effort ladder), and the reward's
disagreement with the gate (this run). What is left is the factor set itself.

The honest reading of the full sequence is that these twelve factors, on this universe,
do not carry conditional timing power that this search can reach. Continuing to vary
the engine is now low-value. The two remaining options are to widen the factor set
with features that have an economic argument for timing, or to stop.