# METHODS — event-proximal companion to the epoch-modulation analysis

*Methods text for the short-window companion analysis (`caban/epoch_modulation.py`,
`run_event_proximal_lane`). It is a companion to the epoch-modulation analysis described in
`epoch_modulation_methods.md` and reopens none of it: every window, model, contrast, family,
omnibus test, table and figure of that analysis is unchanged, and it remains the primary analysis.*

## What this analysis asks

The primary analysis averages activity over each conditioning epoch, three of which are 20 s long.
A response locked to an event *onset* — a transient lasting a second or two — contributes only a
small fraction of such a mean and can be diluted below detectability. This companion asks whether
that is what is happening, by recomputing the identical modulation index over a **3 s window
beginning at each event onset**: tone onset, trace onset (tone offset), shock onset, and post-shock
onset.

## The measurement

For cell *c* on trial *t*, with the trace standardised **once** per cell over the whole session
exactly as in the primary analysis:

    index(c, t, event) = mean over [onset, onset + 3 s) − mean over that trial's pre-tone window

- The **window is the only thing that differs** from the primary analysis. The standardisation, the
  trial-retention rule, the resolved cell set, the two signals and the summarisation are identical,
  so the two lanes are a **paired** comparison on the same cells and the same trials rather than two
  separate analyses.
- The **baseline remains the full-length (20 s) trial-matched pre-tone window**. The index is a
  difference of two window means and is not exposure-scaled, so a longer baseline is simply a more
  precisely estimated one; shortening it to match the response window would add noise to every
  index without changing what is measured.
- The window length is **uniform across all four events**, so every window contains the same number
  of frames and the four-level event factor is a clean comparison.
- The **trial dimension is preserved** until summarisation: one index per cell per trial per event,
  then each cell's value is the median across its retained trials, and each animal's value is the
  mean across its cells.
- Every cell is defined in every window, so the design is balanced within cell and the estimand is
  **unconditional** — no eligibility set, no conditional estimand.

## Two properties of the design, stated rather than corrected

**The shock window is the one that was never diluted.** The primary analysis's shock epoch is
already the 2 s unconditioned stimulus, so the two lanes measure nearly the same quantity there.
Agreement in that window is an internal consistency check on the windowing, not independent
evidence.

**The shock and post-shock windows overlap by 1 s**, because the US is 2 s long and the window is
3 s. Those two estimates are correlated by construction; no event-specificity claim is made across
that boundary.

**Deconvolved traces are constant inside a short window more often than inside an epoch.** The
deconvolution-constrained signal is exactly zero between fitted transients, so a cell with no fitted
transient inside a 3 s window contributes a deterministic constant rather than a measurement. The
proportion of (cell × trial × event) values affected is reported per animal. The raw fluorescence
residual, which carries graded signal everywhere, is the primary signal for this reason, and the
argument for that choice is stronger here than in the primary analysis.

## Statistical analysis

**The animal is the unit of inference** (*n* = 17), as throughout. The reported model is the
hierarchical cell-level model described in `epoch_modulation_hierarchical_cells_methods.md`, applied
unchanged to the short-window index:

    index ~ group * event + (1 | animal) + (0 + event || animal) + (1 | animal:cell)

with an **independent between-animal variance for each event window**, mouse-label randomisation
with every cell fixed to its own animal, exact enumeration for the within-window
treatment-versus-control comparisons, Holm correction across the two such comparisons within each
window, and a Monte Carlo permuted joint Wald test of all six group × event interaction
coefficients.

**The per-window variance is not a refinement but a requirement.** A model with a single residual
variance shared across windows constrains the between-animal spread of the shock response to equal
that of the tone response; in these data those differ by roughly two orders of magnitude, which
makes the quiet windows' intervals too wide and the shock window's too narrow. A mouse-level model
with that shared-variance structure is fitted and reported alongside as a labelled **cross-check**
only; it annotates no figure and licenses no claim.

Both signals are analysed identically, and agreement between them is judged on whether they license
the same conclusion: both intervals containing zero, or estimates of matching sign with overlapping
intervals.

## Reporting

Results of this lane are reported as a companion. Where the short windows reveal modulation the
epoch means did not, the supportable wording is that *an analysis restricted to the 3 s following
each event onset, using the same index and the same animals, showed …*. Where they do not, the
conclusion is bounded by the event-aligned time courses: a null on short windows in the absence of
any visible onset-locked transient establishes that there is little brief event-locked modulation to
compare, which is a different statement from establishing that groups do not differ in it.
