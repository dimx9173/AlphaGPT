# Negative-control design: two proposals, one adopted

Both round-3 answers insisted a negative control is mandatory, and both were
right to. The differential-fragility confound is the most dangerous reading of
any positive result here: null books are churn-harvested and fragile, so any added
strictness plausibly drags null down while real sits flat, and the gap "narrows"
with nothing recovered. Without a control, every positive is deniable.

They propose different controls and the difference matters.

    N1  first-half-only   fitness = mean_f Sharpe(pnet[0:m])
        Same slicing machinery as A1, no agreement requirement.

    N0  odd/even min      fitness = mean_f min(Sharpe(pnet[0::2]), Sharpe(pnet[1::2]))
        Same slicing machinery, same min() strictness, but agreement is demanded
        across INTERLEAVED rather than CONTIGUOUS halves.

N1 is the weaker control. Halving the selection data and dropping the agreement
term change two things at once. If N1 moves the gap it is not clear whether the
cause is the machinery, the reduced sample, or the absence of agreement, and N1
therefore fails at the job a control exists to do.

N0 holds the machinery and the strictness fixed and changes only what the two
views are. Both take half the bars; both are scored with min(); the only
difference is whether they are adjacent in time. A genome that fits the first
half's noise and collapses in the second half gets punished by N0 exactly as it
does by A1. A genome whose edge is genuinely present throughout the window looks
the same under both. So if A1 compresses the gap and N0 does not, agreement across
time is doing the work. If both compress, the compression is a property of taking
min() over two views of anything, not of temporal consistency, and the A1 result is
an artefact of the aggregator.

N0 also has a physical defect worth naming rather than hiding: interleaving bars
by parity breaks the smoother state and the lag-1 alignment, so the two views are
not valid counterfactual histories. That was the objection that killed R3 in round
2, where parity was proposed as a *treatment*. As a *control* the defect is
acceptable and in fact useful: it makes N0 a deliberately unphysical aggregator,
which is exactly what a negative control should be. A control that is itself a
plausible treatment cannot distinguish a treatment effect from a control effect.

## Adopted: N0, odd/even min, as the negative control
