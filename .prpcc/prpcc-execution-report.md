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
26.5/28. Roughly 8 Sharpe.

That figure measures how much of the earlier negative result was an artifact of
selecting on a window read 3,600 times. It is not an improvement in strategy
quality: in the same batch, validation Sharpe is positive in 38/90 folds and
excess over buy-and-hold on the holdout is positive in 0/10. A pre/post
comparison against the old walkforward batch is confounded (different fold count
and window length) and its post-fix median validation Sharpe is worse, so no
claim is made from it.

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
    validation mean_ic = 0   20/90       while train mean_ic > 0 in 79/90

A twelve-genome control separates two things that looked like one failure. Those
random genomes score between -3.2161 and -0.4410, and the GA beats the best of
them on 90 of 90 selections, so the search is working. But 20 folds reach a
positive training reward and none of the twenty carries a positive validation
reward, and corr(train_reward, validation_reward) is -0.0891 against
+0.2309 for rank-IC alone.

The reward is flat rather than reversed. A further test, run over the same 90
real-arm folds, shows the composite reward is not worse than rank-IC as a
selector within this regime: taking the top 9 of 90 by each key, the composite
gives a mean validation rank-IC of +0.00216 against rank-IC's +0.00027 and a
random baseline of +0.00115. An earlier claim in the diagnosis that rank-IC
"carries the signal while the reward points the wrong way" conflated robustness
across data regimes with selection quality within one regime, and does not
survive that test.

So all three items re-weight or re-constrain a reward that has no measurable
out-of-sample relationship, and the best available selector is barely above
chance.

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


## The clearest single measurement

Rebuilding the portfolio position each selected formula produces on the holdout:

    7 of 10 seeds hold a FIXED long position for every bar (std < 0.001),
    three of them at the 0.25 cap.
    3 of 10 vary, and they carry the three lowest holdout Sharpe values.

Seven times out of ten the search converged on holding, and the only thing it
optimised was how much. That is the finding behind all three NO-GOs, and it is
visible without any statistical test.


## Why the space is not the constraint

250 random formulas through the same evaluator:

    position std   median 0.03206
    std < 0.001     9/250  (3.6%)

The ten selected formulas:

    position std   median 0.00045
    std < 0.001     7/10   (70%)

Ninety-six per cent of random formulas move. The space can express a timing
strategy, and the search picks a constant position nineteen times more often than
a random draw. The constraint is the objective, not the space and not the
operators, which is also why PRP-1 (more freedom), PRP-2 (flatter legs) and
PRP-3 (reprice turnover) could not have helped.


## The mechanism, in one number

    mean_ic term      spread 0.02570   weight 1.00
    0.02 * turnover   spread 0.000502  weight 0.02
    total reward      spread 0.6187

The turnover term's whole dynamic range is 0.0812% of the reward's total spread.
It is the only part of `shape_reward` that would penalise trading and it is far
too small to do so. Movement is free, and with an argmax margin of 0.009 against
0.70 of fold-to-fold scatter the selection has no reason to prefer it.

Note on the PRP-3 review: it was right that the stated premise was inverted, and
the rejection followed from that premise. The underlying observation, that
turnover is priced at a scale incapable of influencing selection, was not what
the review examined and is recorded here as a limitation rather than a vindication.
