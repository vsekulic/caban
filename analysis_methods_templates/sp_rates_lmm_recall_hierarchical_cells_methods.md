# METHODS — hierarchical cell-level companion analysis of drug-free recall

*Methods text for the **additive hierarchical cell-level companion / sensitivity analysis** of a
drug-free recall session (`Test_B` at 48 h or `Test_B_1wk` at 1 week). Each session is analysed
independently and the two are never compared — a copy of this file sits beside each session's own
output. The paper-facing recall analysis this accompanies is described in
`sp_rates_lmm_recall_paper_methods.md`, copied into the parent output folder; the full statistical
architecture and decision record are in `sp_rates_lmm_methods.md` and `docs/sp_rates_lmm.md`
§A.7.2.*

**This analysis replaces nothing.** The mouse-level recall models, their contrasts, their Holm
families, their group × epoch omnibus tests, the pre→post modulation decomposition and every
existing figure and table are unchanged and remain the paper-facing analysis. Everything described
here is additive evidence written into its own subdirectory.

## The question

The paper-facing recall analysis summarises each animal to **one scalar per epoch** before fitting.
That is a valid hierarchical analysis and it stays the primary one. It is not the only valid one,
and it discards two things:

- **Within-cell pairing.** An animal's pre-tone amplitude is a mean over the cells active during
  pre-tone and its post-tone amplitude is a mean over the cells active during post-tone — different
  cell sets — so cell-identity variance never cancels out of the change score.
- **Within-mouse cellular heterogeneity.** Whether an animal's shift is coherent across its
  recorded population or carried by a handful of cells.

This companion analysis asks whether the apparent pre→post modulation pattern occurs coherently
across the cellular population within animals, with the cell hierarchy explicitly modelled rather
than collapsed before fitting — while treatment assignment stays where it actually is, at the
**mouse**.

## Measurement — inherited unchanged

Event detection, threshold, the two duration-matched 20 s windows, the retained tone trials, the
animal cohort, the event-run amplitude definition, the zero-event-cell convention and the
source-extracted ROI set are **identical** to the paper-facing recall analysis and were not
re-tuned. They are inherited by construction: this analysis is computed from that lane's own
retained-trial event table, not rebuilt from the recordings.

## Paired within-cell amplitude modulation

For each animal and each cell, events were pooled across the retained trials within each epoch and
the cell's mean event-run integral was log-transformed, exactly as in the primary lane. A cell is
**eligible** when amplitude is defined in both epochs, and its paired modulation is

    delta_log_amplitude_cell = log(mean event amplitude, post-tone)
                             − log(mean event amplitude, pre-tone).

Per-event amplitude is conditional on an event by construction, so a cell with no events in an
epoch has no defined amplitude there. **No value was imputed and no pseudocount was added**; a
constant inside a logarithm would have determined the answer by its size.

### The estimand is conditional, and is reported as such

This quantity estimates the pre→post amplitude modulation **of neurons with measurable event
amplitude in both epochs**. It is not an estimate over all detected cells and is not generalised to
them. The conditioning is unavoidable for a paired quantity, but it is a genuine restriction — the
eligible set is selected on activity in both windows, and that selection could itself differ by
group. Per-animal eligibility counts and fractions are therefore reported as a result
(`stats/hierarchical_cell_eligibility.csv`) and are quoted wherever the estimate is quoted. **The
analysis that includes zero-event cells is the rate model below**, which is one of the reasons it
exists.

## The primary amplitude estimator

The paired modulation values were fit with a linear mixed model

    delta_log_amplitude_cell ~ group + (1 | mouse),

reference group mCherry. This is the **primary estimator**, not a diagnostic: its group
coefficients are the reported effect estimates, and the same coefficients are the statistics the
randomisation test permutes, so estimate, interval and *P* all describe one model specification.
Cells enter through the model's own within- and between-animal variance partition rather than
through an average taken before the model.

**The model's asymptotic `P>|z|` values are not reported as inference.** Sixteen animals were
randomised, not *N* cells. They appear in the model summary file, marked as such, and enter no
multiplicity family and no figure.

## Inference: mouse-label randomisation

Group labels were permuted **across animals**, never across cells; every cell remained attached to
its own animal and the cell table was never altered.

### Pairwise comparisons (primary)

For each of the three comparisons the statistic is the corresponding contrast of the **full
three-group** model — β(hM3D), β(hM4D), or β(hM3D) − β(hM4D), the last formed as a linear contrast
using the coefficients' covariance rather than by subtracting two standard errors. The model is
**refit under every relabelling**; the three-group specification is refit each time, never a
two-group subset, so the permuted statistic and the reported interval come from the same model.

Exchangeability was restricted to the two groups being compared: the third group's animals keep
their true labels and remain in every fit. **The restriction is on the randomisation, not on the
data.** With the observed 6/5/5 cohort these restricted spaces contain C(11,6) = 462, C(11,6) = 462
and C(10,5) = 252 relabellings, so every relabelling was enumerated and the resulting *P*-values
are **exact randomisation *P*-values** with no Monte Carlo error. Space sizes are computed from the
observed group sizes at run time and checked against the enumerator's output.

**Holm correction was applied across exactly these three exact *P*-values**, and that family is
what the figure's significance brackets report.

### Omnibus

The three-group omnibus permutes a **2-df model-based statistic** — the joint Wald test of both
group coefficients of the same model — under the global null across all animals. Enumerating the
2,018,016 global relabellings with a mixed-model refit per draw is not feasible, so this test is
Monte Carlo at a fixed seed and a fixed, reported number of draws. The null hypothesis is that
those coefficients are **jointly zero**: it is a test of the model's group coefficients, not of the
shape, spread or tails of the cellular distribution.

### Design-based sensitivity

A model-free randomisation of the difference in group means of the **per-animal mean paired-cell
delta**, enumerated exactly over the same restricted spaces, is reported alongside — together with
a cell-weighted variant and global-null versions. These are labelled sensitivity analyses and are
in no multiplicity family. They are a genuine check on the model-based result, but each collapses
an animal to one number before testing and therefore carries no cellular hierarchy. No numerical
equivalence between them and the model-based statistic is claimed: the model's generalised-least-
squares weighting and equal-animal weighting coincide only at equal cluster sizes, and cells per
animal vary here. Restricted-null values are compared only with restricted-null values.

### Interval provenance

The model contrast is reported with its own Wald interval; the design-based estimate is reported
with an interval computed over the same 16 animal means. Neither interval is ever attached to the
other estimate, and the randomisation *P*-values carry no interval of their own.

## Amplitude sensitivity model over all epoch-defined cells

To ask whether conditioning on the paired intersection drives the result, a long-format model was
also fit over the union of epoch-defined cells, each cell contributing only the epochs in which its
amplitude is defined:

    log(mean event amplitude)_{cell, epoch} ~ group * epoch
        + (1 | mouse) + (0 + post-tone | mouse) + (1 | mouse:cell).

The **animal-level epoch random slope** is required here because this is a `group * epoch` model:
between-animal variation in the pre→post change — the very effect being compared across groups —
would otherwise have nowhere to go and the group × epoch terms would receive intervals that are too
narrow, with cells effectively serving as replicates for the epoch contrast. The primary model
needs no such term, being fit on the within-cell difference directly, where the animal random
intercept already *is* the animal-specific modulation effect. The slope is independent
(uncorrelated) of the animal intercept, and there is no cell-level epoch slope. This model is
estimation and structural sensitivity only; its asymptotic *P*-values are in no family.

## Hierarchical cell-level event-rate model

A cell × epoch count table was built from the same retained trials, containing **all detected
cells, zero-event cells included**. Counts were not log-transformed and no pseudocount was added.
Counts were fit with a hierarchical negative-binomial model

    n_events ~ group * epoch + offset(log exposure seconds)
        + (1 | mouse) + (0 + post-tone | mouse) + (1 | mouse:cell),

reference group mCherry, reference epoch pre-tone. The **animal-level epoch random slope** is part
of the specified structure for the reason above: animal and cell intercepts model heterogeneity in
*baseline* rate only, while the quantity compared across groups is the pre→post *change*, in which
animals also differ. Omitting it would push that between-animal variation into the cell level and
narrow the group × epoch posterior intervals — the count-model form of the same pseudoreplication
the amplitude path guards against. The slope is independent of the animal intercept; there is no
cell-level epoch slope.

### Priors

Priors are part of this model and were **specified and frozen before the model was fit to the
data**, rather than left to the software's data-dependent defaults. On the log-rate scale (the
exposure offset makes the intercept a log events per second): intercept Normal(0, 5); all fixed
effects Normal(0, 2.5); each random-effect standard deviation Half-Normal(1); negative-binomial
dispersion Gamma(2, 0.5). The dispersion parameterisation was **confirmed empirically against the
installed sampler** rather than assumed, because a prior that is weakly informative under
Var = μ + μ²/α is strongly informative under its reciprocal. The exact priors, and the confirmed
parameterisation, are written into the model's own output file.

### Prior- and posterior-predictive checks

A **prior-predictive simulation was run before the inferential fit** and the implied per-cell event
rates and counts checked for biological plausibility; had they been implausible the priors would
have been revised and re-frozen at that point — on prior simulations only, never after seeing a
posterior.

After fitting, **posterior-predictive checks** compared observed and replicated counts **by group ×
epoch**: the count distribution, the **fraction of zero-event cells**, and overdispersion
(variance-to-mean). These are part of model adequacy, not an optional extra — convergence
diagnostics establish only that the sampler explored this model's posterior, not that the model can
reproduce the observed counts. A model failing these checks is reported as inadequate alongside its
estimates.

### Reported quantities

Two distinct quantities are reported and are never conflated:

- the **pre→post modulation ratio-of-ratios**, which is the group × epoch interaction alone —
  exp(β_interaction) for each treatment group against control, and exp(β_interaction[hM3D] −
  β_interaction[hM4D]) between them;
- the **absolute post-tone rate ratio** against control, exp(β_group + β_interaction).

Both were formed **draw by draw**, preserving posterior covariance between coefficients, and are
reported as posterior medians with 95% highest-density intervals. **No frequentist *P*-value was
derived from this posterior**; it contributes no significance annotation to any figure and alters
no existing result.

### Convergence

The hard convergence criteria (r̂ ≤ 1.01, bulk and tail effective sample size ≥ 400) were applied,
with scope fixed in advance, to the population-level fixed effects, the dispersion parameter, every
random-effect scale hyperparameter, the animal-level intercepts and slopes, and the derived
modulation contrasts. Divergent transitions were required to be zero and this was **not** scoped —
a divergence is a property of the sampler's trajectory rather than of any one parameter. Diagnostics
for the several thousand individual cell-level intercepts are reported in full, including how many
exceed the thresholds, but do not define model failure, since a handful of poorly identified cells
would otherwise condemn a model that is adequate for every quantity being reported.

## Fixed analysis flow

This suite is an **explicitly invoked companion/sensitivity analysis**: it is not part of a routine
pass of the parent analysis, and is run deliberately when this sensitivity question is being asked
(`run_hierarchical_cell_analysis=True`). The mouse-level recall analysis is the primary,
paper-facing one and is complete without it.

When it is invoked it is **one predefined analysis suite**. Every component described here —
including both predictive checks and the declared sensitivity analyses — runs. Nothing is activated
or omitted because another result was or was not significant. If a specified
model fails to fit, or fails its predefined adequacy criteria, the analysis **stops and reports the
failure with its diagnostics**; no simpler model, pooled-cell test, Gaussian approximation or
reduced random-effect structure is substituted. Outputs are staged and promoted only once every
component has succeeded, so a partially failed run cannot be mistaken for a complete result.

## Verification against known truth

Both paths were validated on simulated data with known planted structure. **These are
implementation checks of the analysis code, run separately from any analysis of the experimental
data**; they validate the software, not the experiment, and form no part of the reported
analysis.

For the amplitude path: a true null must not be rejected; a shift planted coherently across the
cells of every animal in a group must be detected; **a large effect confined to a single animal
must not reach significance**, since a cell-level test would call it overwhelming while a
mouse-label randomisation cannot; and increasing cells per animal at a fixed number of animals must
improve within-animal estimation without buying treatment-level precision. That last design checks
both halves and prints them: over a 100-fold increase in cells per animal under a true null the
mean within-animal standard error must fall, while the **group-contrast standard error must not** —
it is a between-animal quantity, bounded below by between-animal variance, which no number of cells
reduces — and the randomisation *P* must not drift toward zero.

For the count path: a null interaction must give modulation intervals covering 1; a planted
group-specific modulation must be recovered; planted animal-to-animal slope heterogeneity must be
recovered and must **widen** the group × epoch interval relative to a matched simulation without
it; and the posterior-predictive zero fraction must match a simulated truth.

## Sample size and interpretation

**n = 16 mice.** That is the number of independently assigned experimental units and it does not
change because cells were modelled. Cell counts are reported descriptively and never as a sample
size. Cells contribute information about each animal's cellular response distribution while the
model and the randomisation preserve animal-level treatment assignment; treating cells as
independently randomised treatment replicates would be pseudoreplication and is precisely what the
randomisation scheme prevents.

Where this analysis provides stronger evidence than the collapsed animal-level analysis, the
supportable statement is that *a hierarchical cell-level analysis, retaining within-animal cellular
variation while preserving animal-level treatment assignment, provided additional evidence that
pre-to-post amplitude modulation differed among groups* — not that the sample size was larger. A
smaller *P* here is not automatically the better answer and is not grounds for replacing the
animal-level result.

**No CNO was present at recall.** All recall measurements are drug-free.
