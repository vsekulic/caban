# METHODS — cellular calcium event amplitude and rate during trace fear conditioning

*Paper-facing methods text. This is the short form intended for a manuscript. The complete
statistical architecture — every epoch definition, sensitivity analysis, multiplicity family and
design decision, with the reasoning behind each — is in `sp_rates_lmm_methods.md`, copied into
this same folder, and in `docs/sp_rates_lmm.md`. Nothing here contradicts those; it selects from
them.*

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

Cells with zero events were retained in every rate and fraction-active denominator — a
manipulation that silences cells must appear as a reduced population rate, which only happens if
silenced cells remain in the denominator. Amplitude is undefined for a cell with no events, so
such cells are necessarily absent from amplitude analyses; this is a definitional exclusion, not
missing data.

## Behavioural windows

Trial structure was measured per animal from each recording rather than taken from the nominal
protocol constants, since recordings are ragged and trial counts vary. Analyses use three
duration-matched 20 s windows: a **pre-tone baseline** (the 20 s ending at tone onset), the
**trace interval** (tone offset to shock onset), and a **post-shock** window (20 s from shock
offset). The 2 s shock itself is excluded from all models: at the observed event rates the
expected count in a 2 s window is small enough that per-cell rate estimates there are dominated
by counting noise.

Because the trace interval on the first trial is 15 s rather than 20 s, comparisons that are
sensitive to window duration (fraction active, event rate) were restricted to (animal, trial)
pairs in which all three windows were present at the full 20 s, determined from the measured
exposure of each window rather than from a trial index.

## Statistical analysis

**The animal is the unit of inference throughout.** Cells and trials contribute precision, not
replication; no analysis treats cells as independent experimental units. Figures showing per-cell
distributions display those cells for description only — all statistics are computed from the
per-animal values (SuperPlot convention, Lord et al. 2020, *J Cell Biol* 219:e202001064).

Per-event amplitude was analysed with linear mixed-effects models on log amplitude with treatment
group as a fixed effect and animal as a random intercept. Group omnibus effects were assessed by
joint Wald tests with animal-level denominator degrees of freedom; treatment-versus-control
contrasts are reported as equal-animal-weighted estimates with 95% confidence intervals. Because
cluster-robust standard errors are anti-conservative with 17 clusters, every reported p-value was
corroborated by a **mouse-label permutation test** in which treatment labels were shuffled across
animals (never across cells) and the contrast recomputed.

A small **confirmatory family** of three prespecified tests — trace-interval amplitude, and the
within-cell elevation of the trace and post-shock windows over baseline — was corrected by the
Holm procedure at α = 0.05. Epoch contrasts were formed **within cell** (each cell's mean log
amplitude in one window minus its own baseline), which removes the cell-level random effect by
construction. All remaining frequentist tests form a declared secondary family controlled by the
Benjamini–Hochberg procedure at α = 0.05. Analyses outside both families are reported as effect
estimates with intervals and are never described as significant.

Whether the group effect **differs between windows** was addressed by one joint group × epoch
test per component, using the same mouse-label permutation scheme with each animal retaining its
complete profile across windows, rather than by comparing per-window p-values against one another.

Event **counts** were analysed with a negative-binomial mixed-effects model at the animal ×
trial × window level, with a `log(total cell-seconds)` exposure offset, random intercepts for
animal and for animal × trial, and dispersion estimated jointly with the remaining parameters
(Bambi/PyMC). Rate results are reported as posterior rate ratios with highest-density intervals;
this endpoint is secondary and carries no confirmatory α.

Locomotion and freezing were **not** covaried in the primary estimand: both are post-treatment
variables, and conditioning on them would remove part of the effect being estimated. One
cross-registration cell set is primary; the others are sensitivity analyses, not replications.

## Interpretive constraints

This is a **locked confirmatory reanalysis**, not a prospective preregistration: the endpoint was
chosen after prior inspection of this dataset, and is described as confirmatory only in the sense
that the family was fixed before the reported models were fit.

With n = 5/6/6 animals the design has 80% power only for very large standardized effects
(Cohen's *d* ≈ 1.8–2.0 for a pairwise comparison). **Every null is therefore reported with its
interval and with what that interval still admits**; none is presented as evidence of absence.

Region-of-interest inclusion is activity-dependent, so cells that are silent throughout a session
may not be detected at all. Measured session-to-session detection dropout was comparable across
treatment groups (77–82%), which constrains but does not eliminate this as a source of bias.
