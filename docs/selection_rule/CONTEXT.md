# Selection-rule experiment: shared context for the discussion

## Why this is the next question

Seven variables have been eliminated by experiment rather than by argument.
Search space (variable-length A/B), search effort (effort ladder), reward-vs-gate
alignment (pure excess), factor set (36f x 100), multi-timeframe (41f x 100), and
bar size (1h x 1000). None of them was the binding constraint.

The 1000-run batch is the one to start from. 500 real + 500 iid-null walkforward
runs, seeds 501-1000, 1h bars, 41 factors, reduced grammar, reward v3, train-only
selection, 10 generations, population 16, 5 folds, real Binance funding,
--scale-windows so factor horizons stay at their 30m values.

| arm | holdout excess Sharpe | sd | median | beat the hold |
|---|---|---|---|---|
| real (n=500) | -2.7429 +/- 0.0291 | 0.651 | -2.816 | 2 / 500 (0.4%) |
| null (n=500) | -1.1003 +/- 0.0832 | 1.860 | -0.992 | 138 / 500 (27.6%) |

Real minus null: **-1.6426 +/- 0.0881, t = -18.64**, 95% CI [-1.815, -1.470].

The median real run sits at the p16 of the observed null distribution. On data
with no edge by construction, this search beats a constant long 27.6% of the time;
on the real market it beats it 0.4% of the time. The ordering is inverted: more
structure in the data made the search worse.

## The three facts that do not fit "no edge"

1. Real data performs significantly WORSE than synthetic (t = -18.64).
2. Real data produces significantly BROADER books: 21.04 positive coins against
   13.96 on null, +7.08, **t = +24.47**.
3. Real data's beat-the-hold rate is 70x below the no-edge calibration.

A factor set with nothing in it would produce all three with the null's sign.
Instead the real arm responds to real structure, the response is breadth, and the
breadth does not survive contact with the hold.

## The hypothesis being tested

Selection is argmax on a single training window. A genome that fits that window's
noise is numerically indistinguishable from a genome that fits a real pattern, and
argmax prefers whichever fits the noise more tightly, because there is more noise
to fit. The negative gap is what that would produce.

The test: hold the factors, the grammar and the GA fixed, replace single-window
argmax with a selection rule that is harder to fool, and ask whether the gap moves
toward zero. If it does, the factors had something in them. If it does not, they
did not, and the negative gap is the honest answer.

## What is fixed and cannot move

- Universe: 28 Binance perps, the common intersection, 1h bars, 2.06 years
  (2024-09-13 to 2026-10-04), 18026 bars. Buy-and-hold on the holdout is +3.106.
- Features: 41, exactly as shipped. Indices 0..11 must keep their meaning.
- Grammar: the reduced set, 11 operators at tokens 41..51.
- Evaluation: the live position path, causal scale, tanh, POSITION_CAP 0.25,
  smooth_causal(5), lagged one bar, same accounting including real funding.
- Selection data: train-only. The validation and lockbox windows stay observations
  and must never reach the selector (PRP-0, already enforced).

## What may change

Only how a genome's training fitness is computed and ranked. Concretely, the
candidates below. The evaluation above is the expensive part and stays fixed, so
every rule is scored on the same folds.

## Data available

Per coin per bar: timestamp, open, high, low, close, volume. Sparse 8-hourly
settlement funding, loaded separately. No order book, no open interest, no
cross-asset series, no point-in-time fundamentals. A factor needing any of those
is a silent placeholder and has bitten this project before.

## Your task

Design the selection-rule experiment. Three rounds, as before: propose
generously, then vet, then decide.

Round 1: propose candidate selection rules. For each, state the exact fitness
computation, what it costs relative to the current one, and the specific failure
mode it is supposed to prevent. Include the rules that are known to be bad and say
why, so they are not re-proposed. Be concrete about how each interacts with a
population-based GA, because a rule that needs per-genome cross-validation inside
a generation loop may be too expensive to run at this scale, and that is a real
constraint rather than an implementation detail.

Answer in prose and tables. Do not run code. Do not edit files.
