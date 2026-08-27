# METHODS — per-cell epoch modulation during trace fear conditioning

*Methods text for the single-cell block of the conditioning figure (`caban/epoch_modulation.py`).
This analysis is a companion to the population-level conditioning analysis described in
`sp_rates_lmm_paper_methods.md`; it reopens none of it.*

## What this analysis asks

The population analysis quantifies how much calcium activity each group shows in each conditioning
epoch. This analysis asks a single-cell question instead: relative to its **own pre-tone baseline on
the same trial**, how far does each cell's activity depart during the tone, the trace interval, the
shock period and the post-shock window — and does chemogenetic manipulation of SST interneurons
change the **average cellular modulation profile** across those epochs?

Because the primary mouse-level endpoint is the mean across that mouse's cells, this is a question
about the average cellular modulation profile. It is **not** a test of whether treatment shifts the
full cell distribution.

## Signals

Two signals are analysed with a fixed division of labour; this is not an open-ended comparison.

**YrA is primary.** YrA is the demixed observed fluorescence trace produced by Minian, loaded with
its own unit ids. It is primary because the deconvolved-and-denoised calcium trace `C` is the
deconvolution-constrained fit and is exactly zero between fitted transients: in a 2 s shock window a
cell with no fitted transient contributes an identically flat value, whereas YrA still carries graded
fluorescence. For each cell, the session **median** of YrA is subtracted before standardization,
matching this codebase's existing convention for that signal. This subtraction makes the trace
interpretable as a fluorescence change; it does not alter the modulation index, because the index is
a difference of two window means of a standardized trace and a per-cell additive constant cancels.

**C is a confirmatory replicate.** The identical pipeline is re-run on `C`. Agreement between the two
signals indicates the result is not carried by residual neuropil or noise that CNMF-E assigns to
background, nor by the deconvolution model. Agreement is pre-specified as **matching contrast sign
with overlapping 95% confidence intervals**. Disagreement is reported; no signal is selected on the
basis of its result.

**The deconvolved trace `S` is not used.** At this cohort's population event rates (0.05–0.1 events
s⁻¹ cell⁻¹) a 20 s window contains approximately one to two detected events and a 2 s window
essentially none, so a per-cell, per-trial, per-epoch quantity built on `S` would be effectively
0/1/2 — too coarse for a continuous index.

Cells were matched to traces **by unit identifier rather than by row position**, because the
deconvolved and observed-fluorescence matrices are exported independently and do not hold identical
unit sets: across the conditioning cohort each recording contained an equal *number* of cells in
both, but in 8 of 17 animals the unit sets differed by one to four cells, and where they differ the
row correspondence shears for the remainder of the matrix. Positional indexing would therefore have
paired most cells with another cell's trace.

The analysed population is accordingly the set of cells carrying a **usable** trace in **both**
signals. Two conditions exclude a cell. First, a cell absent from one signal's unit set: 14 of 9,531
detected conditioning cells (0.15%), across 8 animals. Second, a cell whose trace is constant across
the analysis window and therefore has no defined standardized index: 6 cells (0.06%), across 4
animals, all of them constant in the denoised calcium trace and none in the observed fluorescence
trace. The latter is not a quality-control failure but the extreme case of the denoised trace's
sparsity — it is the deconvolution fit, and is identically zero across a window in which no
transient was fitted — and is the same property, at the level of whole cells, that makes the
observed fluorescence trace the primary signal here.

Both signals were restricted to the same resulting set, so that the confirmatory comparison is
computed over identical cells; per-animal counts of each exclusion are reported with the analysis
output. Together these exclusions retain 99.79% of detected cells and make this analysis's cell
denominators differ marginally from those of the population analysis, which uses the
deconvolved-trace unit set.

## The modulation index

Each cell's trace is standardized **once**, over the whole session:

    z_c(t) = ( x_c(t) − mean_over_session(x_c) ) / sd_over_session(x_c)

Window means are then taken **of that single standardized trace**:

    index(cell, trial, epoch) = mean over the epoch window − mean over that trial's pre-tone window

There is exactly one standardization and it precedes all windowing, so the epoch mean and the
pre-tone mean share one denominator (that cell's session standard deviation). Windows are never
standardized separately.

The standardization is **per cell**, never pooled across a mouse's cells: pooling would leave
per-cell scale differences (footprint amplitude, expression level, depth of field) inside the index
and allow the brightest cells to dominate the mouse-level mean. A consequence of per-cell
standardization is that quiet cells, having a small standard deviation, are amplified.

The index therefore expresses modulation **in units of each cell's own session-wide variability**.
It is not an absolute response magnitude and is not comparable to the ΔF/F- or event-amplitude-scaled
quantities of the population analysis. The same numerical index also means different physical things
on the two signals, since YrA's per-cell standard deviation is dominated by measurement noise and
C's by fitted transients; the signal comparison therefore concerns agreement of conclusions, not
equality of magnitudes.

Because the index is defined from window means of a continuous trace, every detected cell contributes
to every retained trial and epoch. There is no zero-event dropout, no eligibility criterion and no
conditional estimand.

## Epochs

Four response epochs are analysed, each against the same trial's 20 s pre-tone baseline: the 20 s
**tone**, the 20 s **trace** interval from tone offset to shock onset, the measured 2 s **shock**
period, and the 20 s **post-shock** window beginning at shock offset.

The pre-tone window is the **construction baseline** of every index and is not an epoch level of the
statistical model; as a level it would be identically zero.

Unequal epoch durations are acceptable because the endpoint is a window *mean* rather than an event
count or rate, and so is not exposure-scaled. Two qualifications are stated explicitly. First, the
shock index averages roughly 40 frames at 20 Hz and is therefore estimated from far less data than
the 20 s epochs; **the short window may contribute to greater dispersion** of the shock index, so
dispersion there should not be read as biological heterogeneity without accounting for window length,
and equally is not attributed to window length alone. Second, because of the calcium indicator's
kinetics, **shock-evoked activity can extend into the separate post-shock window**; the two epochs
are therefore not independent readouts of distinct processes, and the post-shock window is kept
separate from the shock period rather than merged with it.

## Trials

A trial was retained only when every analysis window was completely present: the 20 s pre-tone
baseline, the tone, a **full 20 s** trace interval, the shock, and a complete 20 s post-shock window.
This retains three to four of the five conditioning trials per animal.

Trace-interval length was taken from the protocol's declared trial times — shock onset minus tone
offset — which are exact: 15 s on the first trial and 20 s on the remainder. Each declared time is
converted to a frame index by matching it to the nearest imaging timestamp, so a declared 20 s
interval may span a frame or two more or fewer than 20 s of frames. That quantization is immaterial
here, because the endpoint is a window mean rather than a count or a rate and is therefore not
exposure-scaled. Whether the post-shock window was fully observed does depend on the recording, since
the final trial ends with the recording rather than with a subsequent tone, and was determined from
the frames actually available after shock offset.

The first conditioning trial is excluded as a consequence of that rule, its trace interval being 15 s
rather than 20 s. It is deliberately not recovered by truncating every trace to a common 15 s: the
first trial is an acquisition trial preceding the first unconditioned stimulus, and the final 5 s of
the later trace intervals, immediately preceding shock onset, is where an anticipatory signal would
be expected.

The index is computed per trial and the trial dimension is preserved until summarization. Each cell's
value is the **median across its retained trials**, so that a single anomalous conditioning trial
cannot define a neuron's response.

## Statistical analysis

The **mouse is the experimental unit**. Cell-level values are displayed to show within-animal
distributions and do not constitute independent replicates. Each mouse contributes one value per
epoch: the mean modulation index across its cells. Each mouse's standard error across cells is
reported alongside the between-mouse spread, so that the negligible contribution of within-mouse
sampling variance to the mouse mean is demonstrated on these data rather than assumed.

Per-cell epoch modulation was analysed with a linear mixed-effects model

    index ~ group × epoch + (1|mouse)

with group (Ctl/mCherry, Exc/hM3D, Inh/hM4D), epoch (tone, trace, shock, post-shock) and their
interaction as fixed effects and mouse as a random intercept. Ctl and the tone epoch are the
reference levels. The index is a signed standardized difference and is analysed **untransformed**;
effects are reported as differences in modulation index with 95% confidence intervals, with no log
transformation and no exponentiation, and no claim is made about the shape or symmetry of its
distribution.

Within each epoch, Exc and Inh were each compared with Ctl using linear contrasts of the fitted
model. At the reference epoch the contrast is the group coefficient; at the remaining epochs the
group coefficient is combined with its group × epoch interaction term using the fitted covariance
matrix. The two treatment-versus-control contrasts within each epoch were **Holm-corrected
together**. Contrasts and Wald tests use an animal-level denominator degrees of freedom of
n_mouse − 1.

Epoch dependence was assessed once by a **joint Wald test of all six interaction coefficients**. This
omnibus establishes whether the group effect is **non-uniform across epochs**; it does not itself
identify which epochs differ from which. The within-epoch contrasts describe where an effect sits and
do not, on their own, support an epoch-specificity claim. No epoch-specific treatment effect is
inferred from one epoch being significant and another not.

## What this analysis does not do

Cells are **not** classified as responsive or non-responsive. The continuous modulation index carries
the analysis directly, and no cell is selected using a statistic that is then reported as a treatment
effect for those same selected cells.

## Figure panels

Two panels are produced, from the primary signal and again from the confirmatory one, so that the two
lanes can be inspected identically; the figure reports the primary signal. The first shows
tone-onset-aligned, trial-averaged
per-cell activity as one heatmap per group on a shared colour scale, with cells sorted by their trace
modulation — the same quantity the statistics are computed on — and displayed on the standardized
scale used throughout, with no further re-scaling within the alignment window. The second shows the
modulation index by epoch and group, with cells as a visual cloud and each mouse's mean overlaid;
every annotated statistic is computed from the per-mouse means and reports the model-derived,
Holm-corrected within-epoch contrasts described above, with the group × epoch omnibus shown above the
panel. Group display order is mCherry, Exc, Inh throughout.
