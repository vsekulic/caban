# METHODS — cellular calcium event amplitude and rate during trace fear conditioning

*Paper-facing methods text. This is the short form intended for a manuscript. The complete
statistical architecture — every epoch definition, sensitivity analysis, multiplicity family and
design decision, with the reasoning behind each — is in `sp_rates_lmm_methods.md`, copied into
this same folder, and in `docs/sp_rates_lmm.md`. Nothing here contradicts those; it selects from
them.*

*Values written `{{ABnn}}` are placeholders to be filled from the run's output files; the lookup
table mapping each code to its file, row and column is in `docs/sp_rates_lmm.md` §M.3.*

## Event detection and measurement

Calcium events were detected on the deconvolved spike-inference trace (`S`) of each pyramidal
cell. **One event is one contiguous supra-threshold run of `S`**, and its amplitude is the
integral of `S` over that run. This differs from the more common convention of counting one event
per local maximum and taking the value at the peak frame: summed over a window, that quantity is
exactly rate × mean peak height and so cannot separate "larger events" from "more events". Under
the run definition, temporally clustered peaks merge into a single wider, larger-integral event
(median run length 4 frames; 11.7% of runs are single-frame), which is why the two definitions
give different event counts and must never be mixed.

Throughout, the measurement is the **integral of a deconvolved event run**; "burst" is an
interpretation of that measurement, not a synonym for it.

Cells with zero events were retained in the population event-rate denominator — a manipulation
that silences cells must appear as a reduced population rate, which only happens if silenced cells
remain in the denominator. Amplitude is undefined for a cell with no events, so such cells are
necessarily absent from amplitude analyses; this is a definitional exclusion, not missing data.

## Behavioural windows

Trial structure was measured per animal from each recording rather than taken from the nominal
protocol constants, since recordings are ragged and trial counts vary. Analyses use three
duration-matched 20 s windows: a **pre-tone baseline** (the 20 s ending at tone onset), the
**trace interval** (tone offset to shock onset), and a **post-shock** window (20 s from shock
offset). The 2 s shock itself is excluded from all models: at the observed event rates the
expected count in a 2 s window is small enough that per-cell rate estimates there are dominated
by counting noise.

Because the trace interval on the first trial is 15 s rather than 20 s, analyses were restricted
for each animal to trials in which all three windows were present at the full 20 s duration,
determined from the measured exposure of each window rather than from a trial index. **Both
outcomes were computed from that same set of animal × trial windows**, so the two are directly
parallel; amplitude is a per-event quantity and is not itself duration-sensitive, but using one
trial set for both removes any hidden difference in the input data.

## Statistical analysis

**The animal is the unit of inference.** Cells and trials contribute precision, not replication;
no analysis treats cells as independent experimental units. Figures showing per-cell distributions
display those cells for description only — all statistics are computed from the per-animal values
(SuperPlot convention, Lord et al. 2020, *J Cell Biol* 219:e202001064). The inferential dataset
contained one value per animal per epoch ({{AB00}} rows: 17 animals × 3 epochs).

For each animal and epoch, **per-event amplitude** was summarised by first calculating each active
cell's mean event-run integral across the retained trials, log-transforming this value, and then
averaging across cells within the animal. Because the response is a within-animal mean of
cell-level logarithms, an exponentiated group contrast is a **ratio of geometric means** of the
cell-level mean event amplitudes, not a ratio of pooled arithmetic means. **Population event
rate** was calculated for the same trials as the total number of detected events divided by the
corresponding total cell-seconds across all detected pyramidal cells, including cells with zero
events, and was log-transformed before analysis. No animal × epoch had a zero population rate and
no pseudocount was applied.

Per-event amplitude and population event rate were analysed separately using the **same linear
mixed-effects model**, with DREADD group (mCherry, hM3D or hM4D), epoch (pre-tone, trace or
post-shock), and their interaction as fixed effects, and animal as a random intercept
(`log(metric) ~ group * epoch + (1|animal)`; reference levels mCherry and pre-tone, set
explicitly). **Within each epoch, hM3D and hM4D were compared with mCherry using linear contrasts
of the fitted model** — at the reference epoch the group coefficient alone, and at the trace and
post-shock epochs that coefficient plus the corresponding group × epoch interaction coefficient,
with their covariance — **with Holm correction across the two treatment-versus-control comparisons
in that epoch**. Amplitude and rate form separate correction families; the direct
hM3D-versus-hM4D comparison was not part of either. Group × epoch interactions were assessed by
joint Wald tests of the four interaction coefficients.

All Wald and contrast inference used an **animal-level denominator degrees of freedom of
n_animals − 1 = 16**, applied consistently to both outcomes; no Satterthwaite or Kenward–Roger
approximation was applied.

Effects are reported as exponentiated model contrasts (fold change or rate ratio) with 95%
confidence intervals — for amplitude the ratio of animal-level geometric mean event amplitudes,
for population rate a rate ratio. Population-rate effects are additionally reported as observed
absolute differences in events s⁻¹ per cell, which are descriptive summaries of the animal-level
means and carry no separate test. *P* < 0.05 after Holm correction was considered statistically
significant.

As a conservative **sensitivity analysis**, the six treatment-versus-control simple effects
spanning all three epochs were additionally corrected together by the Holm procedure within each
outcome. This sensitivity correction did not determine the figure annotations or any significance
statement in the Results.

Locomotion and freezing were **not** covaried: both are post-treatment variables, and conditioning
on them would remove part of the effect being estimated. One cross-registration cell set is
primary; the others are sensitivity analyses, not replications.

Analyses used Python 3.11.15 with statsmodels 0.14.6 (`MixedLM`, REML, L-BFGS), scipy 1.17.1,
numpy 2.4.5 and pandas 3.0.2.

## Supplementary methods — sensitivity analyses

A **negative-binomial mixed-effects count model** with a `log(total cell-seconds)` exposure
offset, random intercepts for animal and for animal × trial, and jointly estimated dispersion
(Bambi/PyMC) was used as a distribution-aware sensitivity analysis of the population-rate result.
Additional sensitivity analyses — a cell-level amplitude model, mouse-label permutation tests of
both the group contrasts and the group × epoch interaction, event-detection threshold sensitivity,
event run-structure measures, a group × trial photobleaching control, and cross-registration
subset variants — are described in `sp_rates_lmm_methods.md`. None of them contributes to the
statistics reported above.

## Interpretive constraints

The analysis above was finalised after substantial prior inspection of this dataset. It is
described as the **main statistical analysis**, not as prospectively preregistered or
prospectively confirmatory. That is why the conservative six-comparison correction is retained and
reported as a sensitivity analysis rather than dropped. The post-hoc procedure is identical in all
three displayed epochs; the trace interval's biological importance is argued in the Introduction
and Results and is not encoded in the statistical structure.

With n = 5/6/6 animals the design has 80% power only for very large standardized effects
(Cohen's *d* ≈ 1.8–2.0 for a pairwise comparison). **Every null is therefore reported with its
interval and with what that interval still admits**; none is presented as evidence of absence.

A non-significant group × epoch interaction means there was **no evidence that the treatment
effect differed across the pre-tone, trace and post-shock epochs**. It is not evidence that the
effect is identical, global, tonic, or equivalent across them. Conversely, a significant
treatment-control comparison within one epoch does not imply that the treatment effect differs
from another epoch; epoch dependence was tested directly by the interaction.

Region-of-interest inclusion is activity-dependent, so cells that are silent throughout a session
may not be detected at all. Measured session-to-session detection dropout was comparable across
treatment groups (77–82%), which constrains but does not eliminate this as a source of bias.
