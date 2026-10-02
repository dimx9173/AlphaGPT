# prpcc execution report: items 1, 2, 3

Objective: run the three proposed changes through the prpcc cycle and verify each.

Every item below reached its verify gate. Two prerequisites were discovered and
fixed first, because neither of the three can be evaluated while they are broken.

## Prerequisite PRP-0: the validation-window selection leak

Codex found this during the PRP-1 plan review. Selection ranked on
`0.5*train_reward + 0.5*validation_reward` inside the generation loop, so each
fold's validation window was read 400 times, 3,600 times across the nine selection
folds, and half the selection key came from held-out data. The inline comment and
the `run_walkforward` docstring both claimed the window was never revisited.

Closed by selecting on the train reward alone, with the legacy key kept behind
`--select-on val_weighted` for diagnosis. Six review rounds, final PASS, verify
6/6, committed 9c14fd3 and 8110505.

Measured effect on the P2 batch: real-arm holdout Sharpe moved from -4.87 / -4.14
to +3.31 / +3.66 on seeds 101 and 102, and `positive_coins` median from 0/28 to
26/28. Roughly 8 Sharpe.

## Prerequisite PRP-4: cost attribution

Salvaged from PRP-3's F3 after PRP-3 was rejected. Six review rounds, final PASS,
verify 6/6, committed af9187a and 5751c41.

The first four implementations tried to prove neutrality with a source scan. Each
round found a real bypass, and the fifth established that the concat fix had
silently been a no-op. The design was changed instead: `evaluate()` is now
byte-identical to its pre-PRP-4 form and contains none of the attribution fields,
the decomposition lives in `research/cost_attribution_28c.py` and runs after the
reward is fixed, and there is no scan because there is no path to scan.

## Item 1: joint formula and parameter search

Verdict: NO-GO at the verify gate.

PRP written, revised against a Codex feasibility review (R1-R6), then stopped by
its own precondition. Two independent measurements:

  - the real-vs-null gate, which requires the real arm to beat null on a holdout
    scored once
  - ranking quality, spearman between the training score and the holdout outcome
    over EVERY genome rather than only the winner

    reward -> holdout sharpe   real +0.8545   null +0.8758   gap -0.0214
    train_IC -> holdout_IC     real +0.6188   null +0.4297   gap +0.1891

The score is not broken. It ranks holdout Sharpe equally well on synthetic data,
which means it ranks a property the synthetic world supplies by construction.

## Item 2: cross-sectional regularisation

Verdict: NO-GO at the verify gate.

Same precondition failure. In addition, its own acceptance criterion would have
passed by construction: a -2.0 breadth penalty forces `positive_coins` up with no
edge required, which produces motion that reads as progress while the gap stays
negative.

## Item 3: cost-budgeted reward

Verdict: NO-GO at the verify gate, with one item salvaged.

The premise was inverted. `accounting_bar_returns` returns
`net = gross - fee - funding_cost` and `evaluate()` builds every leg from that net
series, so `portfolio_sharpe` was already net of costs; there was no gross Sharpe
anywhere on the selection path. Measured, the `-0.02 * mean_turn` term spans
8.1e-05 to 1.4e-04 while `0.10 * psh_worst` spans 0.029 to 0.146, so the second
charge is arithmetically real and economically negligible.

F3, the attribution logging, was salvaged and landed as PRP-4.

## The gate itself

The pre-registered rule compared real against null on raw Sharpe. On a holdout
where buy-and-hold scores +3.675 that cannot separate "found signal" from "took
more exposure". A fifth criterion, `beats_buy_and_hold`, was added.

    real arm, 10 seeds: excess over buy-and-hold positive on 0/10
    corr(sharpe, |mean position|) = +0.65

Why, from the PRP-0 diagnostics over 90 selection folds:

    median |overfit_gap|     0.7019
    median argmax_margin     0.0090      1.5% of the achievable reward range
    overfit_gap < 0          69/90
    validation mean_ic = 0   32/90       while train mean_ic > 0 in 79/90

A twelve-genome control separates two things that looked like one failure. Those
random genomes score between -3.2161 and -0.4410, and the GA beats the best of
them on 90 of 90 selections, so the search is working. But 20 folds reach a
positive training reward and none of the twenty carries a positive validation
reward, and corr(train_reward, validation_reward) is -0.0891 against
+0.2309 for rank-IC alone.

The reward is not pointing where validation is. All three items re-weight or
re-constrain that composite reward, so all three operate on an objective with no
out-of-sample relationship.

## Summary

| item | outcome |
|---|---|
| PRP-0 validation leak | PASS, fixed, worth ~8 Sharpe |
| PRP-4 cost attribution | PASS, fixed, structurally neutral |
| 1 joint param search | NO-GO at the gate |
| 2 cross-sectional reg | NO-GO at the gate |
| 3 cost-budgeted reward | NO-GO at the gate, F3 salvaged |

Two things shipped, three were refused, and the refusal is backed by measurement
rather than by the reviews declining to sign off.
