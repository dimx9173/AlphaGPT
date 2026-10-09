
## A measurement that changes the cost model, found while round 1 ran

`evaluate()` computes `pnet = np.mean(np.stack(leg_net), axis=0)` ONCE per
(genome, fold) and then calls `metrics(pnet, timestamps)` to get a Sharpe. The
per-bar portfolio return series is already in hand.

So a selection rule that splits the TRAIN window by time and scores each piece is
not 2x. It is close to 1x: the expensive part is the simulation, and the
simulation is shared. Splitting costs a second `metrics()` call on a slice.

That reclassifies most of the round-1 proposals from "2x, think about the budget"
to "affordable at full scale". It does not make R1-R3 free, because the slices
must be scored and scored Sharpes are not independent, but the multiplier is
dominated by the simulation that is not repeated.

## The subtlety this exposes, which matters more than the cost

Scoring sub-slices of one shared path is NOT cross-validation.

The position path is `POSITION_CAP * smooth_causal(tanh(sig / scale), 5)` lagged
one bar, where `scale` is the standard deviation of the signal fitted over the
whole train window up to `scale_end`. A genome's path on sub-slice 2 was therefore
computed with knowledge of sub-slice 1 through the fitted scale and through the
smoothing that carries state across the boundary.

What a time-split rule actually measures is therefore **temporal stability of one
fitted object**: does the same genome, fitted once on the whole training span,
still perform on each part of it. That is a real and useful property, and it does
penalise a genome that only works in one regime. It is not the same thing as
refitting per window and asking whether the refits agree, and a proposal that
claims the latter while only doing the former is overstating it.

The distinction matters for interpretation. Refit-agreement is the stronger test
and it costs a genuine multiplier, because the simulation must be repeated per
window. If the experiment is meant to distinguish "the factors had a real effect"
from "argmax fit one window's noise", the stronger test is the one that answers it.
