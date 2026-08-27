# METHODS — hierarchical cell-level companion to the epoch-modulation analysis

*Methods text for the cell-level companion analysis (`caban/epoch_modulation.py`,
`run_hierarchical_cell_lane`). It is a companion to the mouse-level epoch-modulation analysis
described in `epoch_modulation_methods.md` and reopens none of it: every model, contrast, family,
omnibus test, table and figure of that analysis is unchanged, and it remains the primary analysis.*

## What this analysis adds, and what it does not

The primary analysis summarises each animal by the mean modulation index across its cells and
compares those animal-level means. That is a valid hierarchical analysis and it stays the primary
one, but it discards **within-animal cellular coherence**: how consistently an animal's individual
neurons modulate, and how precisely each animal's own cells estimate its mean. This companion
retains that information by modelling every cell explicitly.

It does **not** buy treatment-level precision. The standard error of a group contrast is a
between-animal quantity; it is floored by between-animal variance and no number of cells reduces
it. Nor is it a test of the *shape* of the cellular distribution: a manipulation that raised
modulation in some cells and lowered it in others at constant mean would be invisible to it, as it
is to the primary analysis.

## The model

For cell *c* of animal *m* in epoch *e*, with the modulation index defined exactly as in the
primary analysis (a within-trial, within-cell difference of window means of a single per-cell
standardised trace, taken as the median across that animal's retained trials):

    index ~ group * epoch + (1 | animal) + (0 + epoch || animal) + (1 | animal:cell)

Epoch levels are tone (reference), trace, shock and post-shock; the pre-tone window is the
construction baseline of the index and is not a level of the model. The response is a signed
standardised difference and is analysed untransformed.

**The animal-level epoch random effect is required, not optional.** Between-animal variation in the
epoch effect is the very thing being compared across groups; without a term to absorb it, that
variation has nowhere to go and the cells act as replicates for the epoch contrast, producing
intervals that are too narrow. The term is independent (uncorrelated) across epochs with a separate
variance for each, because the between-animal variance of the shock epoch is roughly an order of
magnitude larger than that of the tone epoch in these data.

**The estimand is unconditional.** Every cell is defined in every window — the index is a window
mean of a continuous trace, so there is no event-count dropout, no eligibility criterion and no
conditional estimand to qualify the estimate with.

## Inference: mouse-label randomisation

**The animal is the unit of randomisation and of inference.** Treatment was assigned to *n* = 17
animals, not to cells. Every reported *P*-value is obtained by permuting the 17 **animal** labels
with every cell fixed to its own animal, and recomputing the statistic by refitting the model under
each relabelling. Cell counts are descriptive and are never reported as a sample size. The fitted
model's own asymptotic *P*-values, which treat cell rows as the unit, are written to the output
under a column name marking them as not for inference, and enter no family and no figure.

**Within-epoch comparisons are exact.** For each epoch, each treatment-versus-control comparison is
the corresponding contrast of the three-group model — the group coefficient at the reference epoch,
and the group coefficient combined with that epoch's interaction term elsewhere. Exchangeability is
restricted to the two groups being compared: the third group's animals keep their true labels and
remain in every fit, so the restriction is on the randomisation and not on the data. With this
cohort those restricted spaces hold C(11,5) = 462 and C(12,6) = 924 relabellings, so every one is
enumerated and the *P*-values are exact, with no Monte Carlo error and no +1/+1 correction — the
observed labelling is itself one of the enumerated draws.

**Multiplicity follows the primary analysis.** Within each epoch, the two treatment-versus-control
comparisons are Holm-corrected together: four independent two-member families. The comparison
between the two DREADD groups is computed and tabulated but belongs to no family and carries no
figure annotation, exactly as in the primary analysis.

**Epoch dependence is assessed once**, by a joint Wald test of all six group × epoch interaction
coefficients, permuted under the global null. The global space holds 5,717,712 relabellings and
cannot be enumerated, so this test is Monte Carlo at a frozen seed; the seed and the number of
draws are recorded in the output. A significant interaction establishes that the group effect is
not uniform across epochs; it does not identify which epochs differ, and no epoch-specific claim is
made from one epoch being significant and another not.

**A design-based sensitivity test accompanies every comparison** and belongs to no family: the same
restricted enumeration applied to the difference of group means of the per-animal mean index. It is
model-free and therefore a genuine check, but it collapses each animal to a single number before
testing and so carries no cellular hierarchy. No numerical equivalence between it and the
model-based statistic is claimed or expected.

## Computation

Refitting the full cell-level model under every relabelling is not feasible: a single fit takes
minutes, and the analysis requires tens of thousands of refits. Because the design is completely
balanced within cell — every cell contributes to every retained trial and every epoch — the
fixed-effect design is constant within each (animal, epoch) block of cells, and the model's
coefficients and their covariance depend on the data only through the per-(animal, epoch) cell
means and the per-animal cell counts. The within-animal variance components are estimated from
within-animal variation alone and are invariant to any relabelling of a between-animal factor. The
permuted statistic is therefore computed from an algebraically equivalent representation on the
animal × epoch cell means, with the animal-level variance components refit by REML under every
relabelling. This is a reorganisation of the arithmetic, not a reduction of the model: the reported
estimates and intervals come from the full cell-level fit, and the equivalence of the two is
asserted numerically on the observed labelling on every run, the analysis stopping if it does not
hold.

## Reporting

Where this companion supports a conclusion, the supportable wording is that *a hierarchical
cell-level analysis, retaining within-animal cellular variation while preserving animal-level
treatment assignment, provided additional evidence that …*. The sample size in every such statement
is the number of animals.
