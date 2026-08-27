# epoch_modulation — analysis reference and results report

Per-cell **epoch modulation** of dorsal CA1 pyramidal neurons during trace fear conditioning: the
single-cell block of Figure 2, and the companion to the population-level amplitude/rate analysis in
[sp_rates_lmm.md](sp_rates_lmm.md).

**The headline result of the run reported here is a null.** Neither DREADD alters the average
cellular modulation profile across tone, trace, shock and post-shock; the group × epoch interaction
is far from significance on both signals and no within-epoch contrast survives correction. §R
reports every number. §R.6 explains why the null should be read alongside the fact that
epoch modulation is itself marginal in *every* group, which limits what the null can be said to
rule out.

**A second null, and a warning about this document's own figure.** Panel K appeared to show
sequential, time-cell-like activity. It does not survive cross-validation (§R.9): the apparent
sequence is an artifact of sorting cells by a statistic computed from the data being displayed, and
the same display shows a near-perfect diagonal when given pure noise. **Do not read structure off a
sorted heatmap in this analysis without the held-out check.**

**The hierarchical cell-level companion (§O.1) is now implemented** — §A.8 for the design, §R.10 for
its results. It reaches the same null, and its one substantive contribution is a caution: the shock
epoch's between-animal variance is an order of magnitude larger than the other epochs', which the
mouse-level model's single shared residual cannot express, so §R.2's one named contrast is weaker
than that table makes it look (§R.10.1).

| | |
|---|---|
| Module | [caban/epoch_modulation.py](../caban/epoch_modulation.py) (2,722 lines) |
| Entry point | `caban.sections.run_epoch_modulation(ds, cfg)`, gated on `cfg.plot_epoch_modulation` |
| METHODS template | [epoch_modulation_methods.md](../analysis_methods_templates/epoch_modulation_methods.md) |
| Notebook | `run_pipeline.ipynb`, "Epoch modulation" section (immediately after `run_event_locked_responsiveness`) |
| Output | `PLOTS_DIR/epoch_modulation/{YrA,C}/` + `signal_comparison.{txt,csv}` |
| Companion | cross-validated sequence test (§R.9): [caban/epoch_sequence.py](../caban/epoch_sequence.py), `run_epoch_sequence(ds, cfg)`, output `PLOTS_DIR/epoch_sequence/` |
| Companion | hierarchical cell-level lane (§A.8, §R.10), same module, gated on `cfg.epoch_modulation_hierarchical_cells` (**on**), output `<signal>/hierarchical_cells/` |
| Companion | event-proximal short-window lane (§A.9), same module, gated on `cfg.epoch_modulation_event_proximal` (**on**), output `<signal>/event_proximal/` — **implemented, results pending** |
| Design | 17 mice — hM3D (Exc) n=5, hM4D (Inh) n=6, mCherry (Ctl) n=6 |
| Run reported here | §R.1–§R.8: **2026-08-26, 12:24:51 → 12:28:09** (3 min 18 s). §R.10: **2026-08-27, 10:16 → 10:21** (3 min 0 s), which re-ran the mouse-level lane as well and reproduced every §R.2 number exactly. `PLOTS_DIR = .../plots/CURRENT` now holds the 08-27 output. |
| Repo state at run | working tree on `feat/sp-rates-axis-labels`, last commit `6cf7ee6` |

---

# Part A — what was implemented

## A.1 The question this asks, and how it differs from sp_rates_lmm

`sp_rates_lmm` asks how much calcium activity each group shows in each conditioning epoch. This
analysis asks a single-cell question instead: relative to **its own pre-tone baseline on the same
trial**, how far does each cell's activity depart during the tone, trace, shock and post-shock
windows, and does chemogenetic manipulation change the **average cellular modulation profile**?

Because the primary mouse-level endpoint is the mean across a mouse's cells, this is a question
about the *average* cellular modulation profile. It is **not** a test of whether treatment reshapes
the cell distribution — a manipulation that raised modulation in some cells and lowered it in others
at constant mean would be invisible here (§A.6).

## A.2 The modulation index

Standardize **once**, per cell, over the **whole session**:

```
z_c(t) = ( x_c(t) − mean_over_session(x_c) ) / sd_over_session(x_c)
```

Then take window means **of that single standardized trace**:

```
index(cell, trial, epoch) = mean over the epoch window − mean over that trial's pre-tone window
```

**There is exactly one standardization and it precedes all windowing.** The epoch mean and the
pre-tone mean are two averages of the *same* standardized trace and share one denominator (that
cell's session SD). Windows are never standardized separately: per-window z-scoring would force each
window to unit variance by its own SD, so the difference of the two means would carry no amplitude
information at all. `_standardized_trace` is the only place in the module that ever computes a SD —
**any change that computes a SD inside a window is wrong.**

The standardization is **per cell**, never pooled across a mouse's cells. Pooling would leave
per-cell scale differences (footprint amplitude, expression level, depth in the FOV) inside the index
and let the brightest cells dominate the mouse mean. Known consequence, stated rather than corrected:
a quiet cell has a small SD in the denominator, so per-cell standardization amplifies quiet cells.

**What the index is:** modulation in units of that cell's **own session-wide variability**. It is not
an absolute response magnitude and is not comparable to the ΔF/F- or event-amplitude-scaled
quantities of the population analysis. The same numerical index also means different physical things
on the two signals of §A.4, since YrA's per-cell SD is dominated by measurement noise and C's by
fitted transients — so the signal comparison concerns **agreement of conclusions, not equality of
magnitudes**.

Every detected cell contributes to every retained trial and epoch: the index is defined from window
means of a continuous trace, so unlike the event-based analyses there is **no zero-event dropout, no
eligibility set and no conditional estimand**.

## A.3 Epochs and trials

**Baseline:** `pre_tone_matched`, 20 s ending at that trial's tone onset. It is the within-trial
**construction baseline** of every index and is therefore **not an epoch level of the model** —
as a level it would be identically zero.

| Epoch | Window | Note |
|---|---|---|
| `tone` | 20 s, tone onset → offset | |
| `trace` | tone offset → shock onset | full-length trials only |
| `shock` | measured 2 s US | lower-precision, see §A.5 |
| `post_shock` | 20 s from shock offset | kept separate from `shock` |

Unequal durations are acceptable because the endpoint is a window **mean**, not a count or rate, and
is therefore not exposure-scaled.

### Trial retention uses the DECLARED protocol times, and needs no tolerance

`TraceFearCondSession.__init__` declares the timing outright — `tone_onsets_def = [185, 420, 660,
900, 1140]`, `tone_duration = 20`, `shock_onsets_def = [220, 460, 700, 940, 1180]`,
`shock_duration = 2`. So the trace interval is exactly

```
nominal_trace_seconds = shock_onsets_def[t] − (tone_onsets_def[t] + tone_duration)
```

= **15 s on trial 1** and **20 s on trials 2–5**, as integers.

`find_exp_boundaries` then converts each declared time to a frame by snapping it to the nearest
miniscope timestamp. **That snapping is why a declared 20 s interval spans 396 frames (19.8 s), not
400.** Asking the *frames* how long the trace is therefore poses a question the definitions already
answer exactly, and forces a tolerance to absorb the resulting jitter. Asking the *definitions* needs
none. This still reads the session's own timing rather than hard-coding a trial index, so it stays
correct if the protocol changes.

> **Do not reintroduce a duration tolerance here.** An earlier version required a measured 20.0 s and
> rejected every trial of every mouse; the fix after that copied `sp_rates_lmm`'s `tol_seconds=0.5`,
> which worked but was absorbing jitter of its own making. That function works from measured
> `exposure_seconds` in an event table and has no choice. Here the declarations are available
> directly.

Everything else is a **presence** check, never a length check: `tone` and `shock` have fixed declared
durations, `pre_tone_matched` returns `None` exactly when it does not fit, and `post_shock` is
checked through the **`iti`** window. That indirection is load-bearing — `get_epoch_frames` *raises*
for `post_shock` when the window runs past the end of a trial (by design; the population analysis
requires every trial to supply it), so querying it directly aborts the run on a truncated final trial
before that trial can be dropped. Since `post_shock_onsets[i] == shock_offsets[i]`
(`sessions.py:1337,1340`), "the 20 s post-shock window fits" is precisely "the ITI is ≥ 20 s".

Trial 1 is excluded as a consequence of the rule, not as a special case. It is deliberately **not**
recovered by truncating every trace to a common 15 s: it is an acquisition trial preceding the first
US, and the final 5 s of the later trace intervals — immediately before shock onset — is where an
anticipatory signal would live.

The index is computed per trial and the trial dimension is preserved until summarization; each cell's
value is the **median across its retained trials**, so one anomalous trial cannot define a neuron.

## A.4 Signals: YrA primary, C confirmatory

**S is excluded, for a concrete reason.** At this cohort's population event rates (0.05–0.1 events
s⁻¹ cell⁻¹) a 20 s window holds ~1–2 detected events and a 2 s window essentially none, so a
per-cell-per-trial-per-epoch S quantity would be effectively 0/1/2 — too coarse for a continuous
index. An S-based single-cell analysis would also largely restate Figure 2E–G.

**YrA is primary** because C's sparsity is a specific liability for short, event-locked windows: C is
the deconvolution-constrained fit and is exactly zero between fitted transients, so on a 2 s shock
window a cell with no fitted transient contributes an identically flat value, whereas YrA still
carries graded fluorescence. **§R.3 shows this argument being borne out by the data.**

**Per-cell baseline subtraction for YrA:** `F0` = that cell's session median (matching
`analysis.py:2323`), analysed trace `YrA − F0`. Stated plainly rather than included silently: because
the index is a *difference* of two window means of a *standardized* trace, a per-cell additive
constant cancels and this **does not change the index**. Verified empirically — two signals differing
only by a DC offset give indices identical to 4×10⁻¹⁵. It is retained because it makes the trace
interpretable as a fluorescence change and is what panel K displays, not because it does statistical
work.

**C is a confirmatory replicate**: the whole pipeline is re-run on C. Agreement shows the result is
not carried by residual neuropil/noise that CNMF-E assigns to background, nor by the deconvolution
model. Both lanes produce a full panel set so the output directories are structurally identical.

### Cells are matched to traces BY UNIT ID, never by row position

`S.zarr` and `YrA.zarr` are exported independently and **do not hold the same units on this dataset.**
Every mouse has an equal *count* in both — so a length check passes — but in 8 of 17 mice the id
*sets* differ by 1–4 cells, and where they differ the row alignment shears for the rest of the matrix.
For G05, `S_idx` holds unit 71 where `YrA_idx` holds unit 70, and **37 of 570 row positions
thereafter refer to different cells.** Indexing YrA by S-derived row positions would pair most cells
with another cell's trace and produce a confident, entirely wrong result.

This is why `get_mapping_signal` is **not** used for the trace matrices here: it resolves cells
through `get_S_indeces`, i.e. positions in `S_idx` — correct for C (`C_idx == S_idx` exactly) and
wrong for YrA.

`resolve_shared_cells` returns the mapping's cells that carry a **usable** trace in *both* signals.
Two conditions exclude a cell:

1. **absent from one signal's unit set** — 14 of 9,531 cells (0.15%), across 8 mice;
2. **constant across the analysis window**, so the standardization would divide by zero — 2 cells,
   both in C, none in YrA.

The second is not a QC failure but the extreme of C's sparsity, at the level of whole cells: a cell
for which CNMF-E fitted no transient inside the window is identically zero across it. (G18's case has
C activity only in the final 46 frames of the recording, outside the experiment window, while its YrA
SD is 18.73 — a real, perfectly measurable cell.)

Both signals are restricted to the **same** resulting set, so the pre-specified agreement criterion
compares identical cells.

## A.5 The mouse-level model

Mouse is the treatment-level unit. Cells are displayed, never counted as replicates.

Per mouse × epoch the endpoint is the **mean modulation index across that mouse's cells**, fit with

```
index ~ C(group, Treatment("mCherry")) * C(epoch, Treatment("tone")) + (1 | mouse)
```

- Epoch levels `tone` (reference), `trace`, `shock`, `post_shock` **only**; pre-tone does not appear.
- 17 mice × 4 epochs = **68 observations**; denominator df = n_mouse − 1 = **16**.
- The index is a **signed standardized difference**, analysed **untransformed** — no log, no
  exponentiation. Effects are differences in modulation index with 95% CIs. No claim is made about
  the shape or symmetry of its distribution.
- Within each epoch, Exc-vs-Ctl and Inh-vs-Ctl are **Holm-corrected together**. At the reference
  epoch the contrast *is* the group coefficient; at the others it is the group coefficient combined
  with that epoch's interaction term through the fitted covariance — a contrast, not a coefficient,
  so it cannot be read off the model summary.
- Epoch dependence is assessed **once**, by a joint Wald test of all **six** interaction
  coefficients.

> **What the omnibus does and does not do.** A significant interaction establishes that the group
> effect is **not uniform across epochs**. It does **not** identify which epochs differ. The
> within-epoch contrasts describe where an effect sits and do not, on their own, license an
> epoch-specificity claim. No epoch-specific effect is inferred from one epoch being significant and
> another not.

**Optimizer note (numerical, not statistical).** On this frame shape — 68 rows, 4 per mouse, 17 mice
— statsmodels' default `lbfgs` raises `LinAlgError: Singular matrix` inside `MixedLM.fit`, which
`fit_mixed_model` reports as failed convergence and answers with mouse-clustered OLS. `bfgs`, `cg`,
`powell` and `nm` all reach the *same* interior optimum (mouse variance 0.006746 to six figures), so
the escalation starts at `bfgs`. A list *beginning* with `lbfgs` does not help — the exception escapes
statsmodels' own escalation. The clustered-OLS fallback remains available for genuine degeneracy.

## A.6 What is deliberately not done

- **No binary responsive/non-responsive gate.** The continuous index carries the analysis. In
  particular this module does **not** reproduce the `event_locked_responsiveness` pattern of
  *shuffle → FDR-select responders → compare the selecting statistic's magnitude across groups*,
  which selects cells on a noisy statistic and then reports that same statistic as a treatment
  effect.
- **No cell-level treatment model *in the paper-facing lane*.** Averaging cells within a mouse costs
  essentially no treatment-level power — Var(mouse mean) = σ²_between + σ²_within/n_cells, and the
  second term is negligible at hundreds of cells per mouse (**demonstrated on these data in §R.5**).
  It does discard distribution *shape*, which the panels display. An additive hierarchical
  cell-level companion mirroring `sp_rates_lmm`'s recall lane (§A.7.2 there) is now **implemented**
  and described in §A.8; it changes nothing here, and the mouse-level analysis remains the
  paper-facing one.
- **No third signal, no per-epoch signal selection, no sensitivity suite.**

## A.7 Panels

**Panel K** — tone-onset-aligned, trial-averaged per-cell activity, one heatmap per group on a shared
colour scale, cells sorted by their **trace** index (the same quantity the statistics use), spanning
−20 s to +65 s with epoch boundaries and a shaded US band. Displayed on the standardized scale used
throughout, with **no re-scaling inside the alignment window** — `event_locked_responsiveness._peri_event_matrix`
z-scores across the window, which would be a second standardization and is therefore not reused.

**Panel L** — modulation index by epoch × group; cells as a visual cloud, each mouse's mean overlaid.
`yscale='linear'` is passed deliberately: the index is a signed standardized difference taking
negative values, so `draw_superplot_triplet`'s heavy-tailed-positive default does not apply.
Brackets report **this analysis's model-derived Holm-corrected contrasts**, not the helper's default
Welch test, via a `stat_fn` adapter — so the panel and the stats file cannot disagree. Exc-vs-Inh is
returned as NaN (not in the correction family) and is skipped rather than drawn.

Group display order is **mCherry, hM3D, hM4D** throughout, per CLAUDE.md.

Both panels are produced for **both** signals; the figure reports the primary signal.

## A.8 The hierarchical cell-level companion (§O.1, now implemented)

> **Part of a routine pass, and written into this analysis's own plots tree.** Gated on
> `cfg.epoch_modulation_hierarchical_cells`, which is **`True`** like every other switch in
> `Config`, so `run_epoch_modulation(ds, cfg)` produces it under
> `<PLOTS_DIR>/epoch_modulation/<signal>/hierarchical_cells/`, beside panels K and L.
>
> **It dominates this analysis's runtime** — ~8 min (YrA) and ~17 min (C) against seconds for
> everything else, most of it the single full statsmodels fit retained as an independent
> cross-check (§A.8.2). Set the flag `False` for a quick pass.
>
> It is a **companion**: everything in §A.1–§A.7 and every number in §R is unchanged and remains
> the paper-facing analysis.

**What it adds.** §A.5's endpoint is one scalar per mouse per epoch. That is a valid hierarchical
analysis and it stays primary, but it discards **within-mouse cellular coherence**. This lane
retains it:

```
index ~ group*epoch + (1|mouse) + (0+epoch||mouse) + (1|mouse:cell)
```

The animal-level epoch random effect is **required, not optional** — between-animal variation in
the epoch effect is the thing being compared across groups, and without a term to absorb it the
cells become replicates for the epoch contrast. It is **independent across epochs with its own
variance per epoch**, because the fitted between-animal shock variance is ~0.034 against tone's
~0.00009: a single shared epoch variance would be the wrong constraint.

**One genuine improvement over the recall lane.** `sp_rates_lmm`'s companion needed a *conditional*
estimand, because pre and post amplitudes came from different sets of active cells. Here the index
is within-cell and within-trial and every cell is defined in every window, so the estimand is
**unconditional** — no eligibility set, no eligibility table, no fraction-eligible caveat.

### A.8.1 Inference, and the multiplicity rule

Mouse-label randomisation, with every cell fixed to its own animal. **n = 17 mice** is the sample
size in every statement; cell counts are descriptive, and the fitted model's own asymptotic
*P*-values are written out under a column named so they cannot enter a family or a figure.

- **Within-epoch comparisons are exact.** For each epoch, each comparison is the corresponding
  contrast of the three-group model, with exchangeability restricted to the two groups compared —
  the third group's mice keep their true labels and stay in every fit. The spaces hold
  C(11,5) = 462 (Exc–Ctl, Exc–Inh) and C(12,6) = 924 (Inh–Ctl), all enumerated, so the *P*-values
  are exact with no `+1/+1` correction.
- **Holm follows §A.5, not the recall lane.** Within each epoch the two treatment-versus-control
  comparisons are corrected together: four independent two-member families. Exc-vs-Inh is computed
  and tabulated but is in **no** family and carries no bracket, matching panel L.
- **Omnibus:** the joint Wald *F* on all six interaction coefficients, permuted under the global
  null. 17!/(5!6!6!) = 5,717,712 relabellings forces Monte Carlo; seed and draw count are written
  into the output.
- **Design-based sensitivity**, always run and in no family: the same restricted enumeration on the
  difference of group means of the per-mouse mean index. Model-free, but it collapses each animal
  to one number and carries no cellular hierarchy.

### A.8.2 How it is made affordable, and why that is not a reduced structure

A literal implementation is not a computation anyone runs. **Measured on this dataset:** one full
cell-level fit (38,060 rows) takes **220–300 s**, and the lane needs ~27,000 refits per signal —
**60 to 150 hours**. That is §O.1's flagged risk, confirmed rather than predicted.

The way out is algebraic, and it turns on §A.2's guarantee that **every cell is defined in every
epoch**. Each mouse's data then splits orthogonally into the four cell means and the cell-contrast
space, the covariance is block-diagonal with respect to that split, and the fixed-effect design
lies entirely in the first block. So the REML objective is a sum of two pieces: a cell-contrast
piece with two chi-square terms per mouse (the spread of cell means, variance
4σ²_cell + σ²_resid on n−1 df; and the two-way residual, variance σ²_resid on 3(n−1) df), and a
cell-mean piece in the 68 numbers ȳ_{m,e} with

```
Var(ȳ_m·) = (σ²_mouse + σ²_cell/n_m)·J + diag(σ²_epoch) + (σ²_resid/n_m)·I
```

**Nothing is discarded and nothing is held fixed.** All seven variance parameters and β are refit
by REML under every relabelling; the cellular hierarchy enters both as the two sufficient
statistics and as the `/n_m` weights — which is how a mouse with 89 cells comes to carry less
weight than one with 1,047. The four cell-contrast sufficient statistics are **invariant under
every relabelling** (group is a between-mouse factor, so no relabelling can move a within-block
residual), so they are computed once. A refit costs **~10 ms**, and the whole randomisation runs in
**~5 min per signal**.

**The identity is proven on every run, not argued.** `assert_reml_objective_identity` evaluates the
collapsed objective and a deliberately naive full-covariance reference at several unrelated
parameter vectors and requires their difference to be the same constant. On this dataset the offset
varies by **8×10⁻¹³** across probes and the GLS coefficients agree to **3×10⁻¹⁵**. `_require_balanced`
catches a violation of the balance assumption earlier and more legibly. This check is **mandatory
and costs under a second**; it is what establishes that the reported fit optimises the full model's
REML objective.

**The optimiser is given an analytic gradient**, `_neg2_reml_and_grad`. Every dV/dφ in this model is
`J`, a single diagonal entry, or `I`, so each trace the gradient needs collapses onto quantities the
objective has already formed. This is not a modelling choice — the objective and its optimum are
unchanged — but it takes a permuted refit from **33 ms to 10 ms**, and the lane does ~9,400 of them
per signal. Design 0 of the synthetic suite checks it against finite differences, because a wrong
gradient would not raise: it would quietly return a worse optimum.

**Which fit is reported.** The estimate, its interval and its *P*-value all come from the collapsed
fit — the same likelihood, the fit the randomisation actually refits, so all three describe one
model.

**The full statsmodels refit is an optional cross-check, off by default**
(`HIERARCHICAL_CROSS_CHECK_FULL_FIT`). It refits the whole model on all 38,060 rows through a second
implementation and cost **246 s (YrA) and 807 s (C)** — 17.5 of the lane's original 21 minutes — to
produce a number that is neither reported nor gating. It is also the *weaker* of the two checks:
statsmodels converges to a **worse** point on the shared objective than the collapsed fit does
(it stops short along the flat between-mouse variance directions that 17 groups leave poorly
determined), so where they disagree the cross-check is what needs explaining. When it is enabled,
`assert_collapse_equivalence` scores both on the shared objective and accepts a disagreement **only**
when the collapsed fit attains a strictly better value; its contrast is written into the
`*_full_fit_crosscheck` columns. Turn it on after changing the model structure, with
`cross_check_full_fit=True`. The synthetic suite runs it on every design, where the frames are small
and it costs seconds.

### A.8.3 Validation and output

`verify_hierarchical_epoch_synthetic` is a **hand-run development tool**, never called by a real
run (the §A.7.2 rule). Five planted designs, each checking a property the lane rests on: the
analytic gradient matches finite differences; a group × epoch effect is recovered; **a huge effect
confined to one animal must not reach significance** (the pseudoreplication guard); more cells at
fixed n_mice sharpen the per-animal estimate while the group-contrast SE does not move; and the
collapse identity plus the statsmodels equivalence hold on independently simulated data.

Output is **staged and promoted** — written to `hierarchical_cells__staging/` and swapped in
wholesale only after every component succeeds, leaving `FAILED.txt` and re-raising otherwise.
There is no per-component switch, no result-dependent branching, and no statistical fallback:
`fit_mixed_model`'s clustered-OLS fallback is explicitly barred here, since clustered OLS over
38,060 cell rows with 17 clusters is the inference this lane exists to avoid.

The lane's own two-signal comparison uses the **fixed** agreement criterion (both CIs contain zero,
*or* matching sign with overlapping CIs) rather than §R.7's, which is defective near zero. That is a
choice for new code, not a change to the existing criterion — see §O.3.

## A.9 The event-proximal companion (short windows) — implemented, not yet run

> **Gated on `cfg.epoch_modulation_event_proximal`** (**`True`**), run from inside
> `run_epoch_modulation`'s per-signal loop, output
> `<PLOTS_DIR>/epoch_modulation/<signal>/event_proximal/` — beside `hierarchical_cells/`.
> It carries a hierarchical lane of its own, so it roughly **doubles** this analysis's runtime.
>
> It is a **companion**: §A.1–§A.8 and every number in §R are unchanged. **Results are not in this
> document yet** — §R.11 is written after the first run.

**What it asks.** §R.6's honest limit on the main null is that epoch modulation is marginal in
*every* group, so there is little room for a group difference in it. One concrete explanation is
**temporal dilution**: three of the four windows are 20 s, and a transient locked to an event
*onset* contributes only a few percent of such a mean. This lane tests that directly by recomputing
the same index over **3 s from each event onset** — tone, trace, shock, post-shock — and changing
nothing else.

```
index(cell, trial, event) = mean over [onset, onset + 3 s) − mean over that trial's 20 s pre-tone window
```

- **One window parameter, and it is the only difference.** `build_modulation_table(window_seconds=…)`
  — `None` is §A.2's full-epoch analysis, byte-for-byte; a float takes the short window from the
  same onset frame `get_epoch_frames` already returns. Standardization, retention rule, resolved
  cell set, summarization and both signals are shared code, so the two lanes are a **paired**
  comparison on identical cells and trials rather than two analyses.
- **The baseline stays the full 20 s pre-tone window.** The index is a mean against a mean and is
  not exposure-scaled, so a longer baseline is a *more precise* one; matching it to 3 s would only
  add noise. Every index still carries the trial-matched construction baseline of §A.3.
- **Uniform 3 s across all four events**, so the four windows hold equal frames and the design stays
  balanced within cell — which is also what the hierarchical lane's collapse identity requires
  (§A.8.2).

**Two properties stated rather than engineered around.**

1. **Shock is the one event that was never diluted** — its epoch is already the 2 s US, so the two
   lanes measure nearly the same thing there. That row is an internal check on the windowing, not a
   result. The 3 s shock window also **overlaps post-shock by 1 s**, so those two are correlated by
   construction and no event-specificity claim crosses that boundary.
2. **C's sparsity bites harder at 60 frames.** A cell with no fitted transient inside the window is
   identically zero across it, so its index is exactly −(its pre-tone mean) — a deterministic
   constant, not a measurement. `resolve_shared_cells` excludes only cells constant over the *whole*
   session (§A.4), so these stay in; the per-mouse **flat-window fraction** is written to the
   coverage table and the report instead. This is §A.4's argument for YrA-primary, stronger.

**Inference is the hierarchical lane, not the mouse-level one.** §R.10.4 showed that a single
residual shared across epochs borrows precision from the quiet epochs and lends it to the loud one;
short windows do not repair that and are unlikely to shrink the gap. So `run_hierarchical_cell_lane`
is reused unchanged on the short-window `df_cell` — per-event between-animal variance, exact
mouse-label randomisation, the same Holm rule (§A.8.1) — and panel N's brackets come from it. The
mouse-level model is still fit, because it is the hierarchical lane's comparison arm, and is written
to `stats/mouse_level_crosscheck.txt` under that name. **It brackets nothing.**

**Panels.**

- **Panel M — event-aligned group-mean traces**, −5 s to +10 s around each of the four onsets, mean
  ± SEM **across mice**, the analysed 3 s shaded. Standardized once per cell and baseline-subtracted
  per trial, with **no re-scaling inside the alignment window** (§A.7's rule). *This is the panel the
  lane rests on*: a null on short windows means something different depending on whether an
  onset-locked transient is visible here at all, and §R.11 has to say which.
- **Panel N** — the index by event × group, `plot_epoch_modulation_superplot` reused with this
  lane's wording and the hierarchical brackets.
- **Dilution panel + table** — per-mouse short-window index against the full-epoch index, four
  facets, paired. The premise, quantified.

One known cosmetic side effect on the existing lane: `contrasts.txt`'s heading underline is now
generated from the heading and is 46 characters rather than 45. No number changes.

---

# Part R — results of the 2026-08-26 12:24 run

All numbers below are from `PLOTS_DIR/epoch_modulation/` as written at 12:27–12:28 on 2026-08-26.

## R.1 Cohort and coverage

9,515 cells analysed (Exc 3,488 / Inh 3,119 / Ctl 2,908); 89–1,047 per mouse, median 520.
16 of 17 mice retained 4 trials; G09 retained 3 (it has only 4 total). **Trial 1 excluded in every
mouse**, as designed. 14 cells dropped for a missing trace, 2 for a constant trace — **99.79% of
detected cells retained**.

| Mouse | Group | Cells | Dropped: no trace | Dropped: constant | Trials | Retained |
|---|---|---|---|---|---|---|
| G08 | Ctl | 494 | 1 | 0 | 4 | 1,2,3,4 |
| G09 | Ctl | 455 | 0 | 0 | 3 | 1,2,3 |
| G12 | Ctl | 520 | 1 | 0 | 4 | 1,2,3,4 |
| G13 | Ctl | 89 | 0 | 0 | 4 | 1,2,3,4 |
| G16 | Ctl | 656 | 0 | 0 | 4 | 1,2,3,4 |
| G17 | Ctl | 694 | 3 | 0 | 4 | 1,2,3,4 |
| G05 | Exc | 569 | 1 | 0 | 4 | 1,2,3,4 |
| G10 | Exc | 797 | 2 | 0 | 4 | 1,2,3,4 |
| G11 | Exc | 737 | 0 | 0 | 4 | 1,2,3,4 |
| G18 | Exc | 1047 | 1 | 1 | 4 | 1,2,3,4 |
| G19 | Exc | 338 | 0 | 0 | 4 | 1,2,3,4 |
| G06 | Inh | 413 | 0 | 0 | 4 | 1,2,3,4 |
| G07 | Inh | 534 | 0 | 0 | 4 | 1,2,3,4 |
| G14 | Inh | 392 | 1 | 1 | 4 | 1,2,3,4 |
| G15 | Inh | 417 | 0 | 0 | 4 | 1,2,3,4 |
| G20 | Inh | 846 | 4 | 0 | 4 | 1,2,3,4 |
| G21 | Inh | 517 | 0 | 0 | 4 | 1,2,3,4 |

Model fit: `mixedlm` (not the OLS fallback) on both signals, 68 observations, 17 groups, REML,
scale 0.0099, mouse variance 0.001 (YrA).

## R.2 The primary result — no group × epoch effect

| Signal | Group × epoch joint Wald | P |
|---|---|---|
| **YrA (primary)** | F(6,16) = **0.459** | **0.828** |
| C (confirmatory) | F(6,16) = **0.647** | **0.692** |

**No within-epoch contrast survives Holm correction on either signal.** Full contrast table
(difference in modulation index, 95% CI, t(16), raw P, Holm-adjusted P):

### YrA — primary

| Epoch | Contrast | Estimate | 95% CI | t(16) | P | P_holm |
|---|---|---|---|---|---|---|
| tone | Exc vs Ctl | −0.0299 | [−0.1616, +0.1017] | −0.482 | 0.636 | 1 |
| tone | Inh vs Ctl | −0.0138 | [−0.1393, +0.1117] | −0.233 | 0.819 | 1 |
| trace | Exc vs Ctl | −0.0319 | [−0.1635, +0.0998] | −0.513 | 0.615 | 1 |
| trace | Inh vs Ctl | −0.0346 | [−0.1601, +0.0909] | −0.584 | 0.567 | 1 |
| shock | Exc vs Ctl | −0.0255 | [−0.1571, +0.1062] | −0.410 | 0.687 | 0.687 |
| shock | Inh vs Ctl | −0.0972 | [−0.2227, +0.0283] | −1.642 | 0.120 | 0.240 |
| post_shock | Exc vs Ctl | +0.0463 | [−0.0853, +0.1780] | +0.746 | 0.466 | 0.933 |
| post_shock | Inh vs Ctl | −0.0124 | [−0.1379, +0.1131] | −0.209 | 0.837 | 0.933 |

### C — confirmatory

| Epoch | Contrast | Estimate | 95% CI | t(16) | P | P_holm |
|---|---|---|---|---|---|---|
| tone | Exc vs Ctl | +0.0031 | [−0.0963, +0.1024] | +0.065 | 0.949 | 1 |
| tone | Inh vs Ctl | −0.0115 | [−0.1062, +0.0832] | −0.257 | 0.801 | 1 |
| trace | Exc vs Ctl | +0.0059 | [−0.0934, +0.1052] | +0.126 | 0.901 | 1 |
| trace | Inh vs Ctl | −0.0142 | [−0.1089, +0.0805] | −0.317 | 0.755 | 1 |
| shock | Exc vs Ctl | −0.0199 | [−0.1192, +0.0795] | −0.424 | 0.677 | 0.677 |
| shock | Inh vs Ctl | **−0.0951** | **[−0.1898, −0.0004]** | −2.129 | **0.0491** | **0.0983** |
| post_shock | Exc vs Ctl | +0.0232 | [−0.0762, +0.1225] | +0.494 | 0.628 | 1 |
| post_shock | Inh vs Ctl | −0.0269 | [−0.1216, +0.0678] | −0.603 | 0.555 | 1 |

Every estimate lies between 0.003 and 0.10 SD units with CIs spanning roughly ±0.15. This is not an
underpowered near-miss — the estimates sit on zero.

**The one contrast worth naming, and not claiming.** Inh-vs-Ctl at shock is the largest effect and is
consistent across signals (YrA −0.097, C −0.095). On C its raw P = 0.049, but **Holm-adjusted
P = 0.098** — it does not survive correction, and the omnibus is non-significant, so by this
analysis's own rules (§A.5) no epoch-specific claim is licensed. As a *direction* it coheres with the
population-level Inh rate reduction in Figure 2E/2G — a blunted US response. **Treat it as a
hypothesis, not a result** — and note that **§R.10.1 weakens it further**: the shock epoch's
between-animal variance is an order of magnitude larger than the other epochs', which this model's
single shared residual cannot express, and the hierarchical companion puts the same estimate at
exact P = 0.159 on C.

## R.3 Descriptive profile — the US response, and the sparsity argument confirmed

Per-mouse modulation index, mean ± SD across mice:

| Epoch | | Ctl (n=6) | Exc (n=5) | Inh (n=6) |
|---|---|---|---|---|
| tone | YrA | −0.0095 ± 0.0431 | −0.0395 ± 0.0461 | −0.0233 ± 0.0599 |
| trace | YrA | −0.0119 ± 0.0502 | −0.0437 ± 0.0902 | −0.0464 ± 0.0789 |
| **shock** | YrA | **+0.1269 ± 0.1600** | **+0.1014 ± 0.2450** | **+0.0296 ± 0.0972** |
| post_shock | YrA | −0.0075 ± 0.0572 | +0.0389 ± 0.0671 | −0.0199 ± 0.0739 |
| tone | C | −0.0226 ± 0.0574 | −0.0196 ± 0.0289 | −0.0341 ± 0.0599 |
| trace | C | −0.0256 ± 0.0685 | −0.0197 ± 0.0469 | −0.0398 ± 0.0692 |
| **shock** | C | **+0.0637 ± 0.1133** | **+0.0438 ± 0.1484** | **−0.0315 ± 0.0983** |
| post_shock | C | +0.0205 ± 0.0384 | +0.0436 ± 0.0382 | −0.0065 ± 0.0741 |

**The shock epoch is the only one with positive modulation, in all three groups, on both signals.**
Tone and trace are slightly *negative*. A hard-wired US producing the largest response is the
biological sanity check passing — and note it was explicitly **not required** to be largest (§A.5's
design admits several reasons it might not be: the 2 s window, indicator kinetics pushing signal into
post-shock, shock-period motion or saturation).

**The US response is much clearer on YrA (+0.085) than on C (+0.024).** That is §A.4's sparsity
argument corroborating itself on the data: a 2 s window frequently contains no fitted C transient.
It is independent support for making YrA primary, arrived at after the choice rather than before it.

## R.4 How much modulation is there at all? (descriptive, not pre-specified)

One-sample t-tests of the 17 per-mouse values against zero, pooling groups. **These were computed
during results review and are not part of the pre-specified analysis** — they are reported because
they bound what the §R.2 null can be read to mean.

| Epoch | YrA mean [95% CI] | t(16) | P | C mean [95% CI] | t(16) | P |
|---|---|---|---|---|---|---|
| tone | −0.0232 [−0.0483, +0.0019] | −1.96 | 0.068 | −0.0258 [−0.0510, −0.0006] | −2.17 | 0.046 |
| trace | −0.0334 [−0.0699, +0.0031] | −1.94 | 0.070 | −0.0289 [−0.0597, +0.0019] | −1.99 | 0.064 |
| shock | +0.0851 [−0.0007, +0.1708] | +2.10 | 0.052 | +0.0242 [−0.0375, +0.0859] | +0.83 | 0.417 |
| post_shock | +0.0018 [−0.0327, +0.0362] | +0.11 | 0.914 | +0.0178 [−0.0103, +0.0458] | +1.34 | 0.198 |

No epoch is convincingly different from its own pre-tone baseline. See §R.6.

## R.5 The variance diagnostic, on real data

`Var(mouse mean) = σ²_between + σ²_within/n_cells`. Averaging cells removes the second term.

| Epoch | YrA within-mouse SEM | YrA between-mouse SD | ratio | C ratio |
|---|---|---|---|---|
| tone | 0.0063 | 0.0489 | **0.128** | 0.136 |
| trace | 0.0073 | 0.0710 | **0.103** | 0.127 |
| shock | 0.0181 | 0.1669 | **0.108** | 0.151 |
| post_shock | 0.0089 | 0.0670 | **0.133** | 0.192 |

At a median of 520 cells per mouse, between-mouse variance dominates by roughly **8–10×**. Collapsing
cells to a mouse mean therefore costs essentially no treatment-level precision — the §A.6 claim
demonstrated on these data rather than asserted. What it discards is distribution *shape*, which is a
different matter from power.

## R.6 What this null does and does not establish

**Does:** there is no detectable difference between groups in the average cellular modulation profile
across tone, trace, shock and post-shock, on either signal, at n=17 mice.

**Does not:** establish that SST manipulation leaves single-cell conditioning responses unchanged.
The null sits inside a regime where **epoch modulation is itself marginal in every group** (§R.4):
the largest cohort-wide effect is the US at +0.085 SD (P = 0.052) and tone/trace sit at −0.02 to
−0.03. A group difference in epoch modulation is hard to detect when the epoch modulation being
compared is ~0.03–0.09 SD in all three groups. The honest statement is that **this measurement finds
little epoch-locked modulation to begin with**, and consequently little room for a group difference
in it — not that no such difference exists.

This is also why the mean-based endpoint's blind spot (§A.6) matters more than usual here: a
manipulation that reorganised *which* cells respond without moving the mean would be invisible to
every number above. The hierarchical companion (§A.8, results in §R.10) closes part of that gap —
it retains within-mouse cellular coherence and reaches the same null — but **not** that part: it
models the cellular hierarchy, not the shape of the cell distribution, so a reorganisation at
constant mean is invisible to it too. Q2 (§O.2) is what would address it.

## R.7 The signal comparison, and a flaw in its criterion

The pre-specified criterion is **matching sign AND overlapping 95% CIs**. It reports **2 of 8
contrasts disagreeing** — tone and trace Exc-vs-Ctl:

| Contrast | YrA | C | Flag |
|---|---|---|---|
| tone Exc vs Ctl | −0.0299 [−0.1616, +0.1017] | +0.0031 [−0.0963, +0.1024] | DISAGREE |
| trace Exc vs Ctl | −0.0319 [−0.1635, +0.0998] | +0.0059 [−0.0934, +0.1052] | DISAGREE |

**These flags are the criterion misfiring, not a signal discrepancy.** The CIs overlap almost
entirely and both span zero; only the point estimates fall on opposite sides of it. Requiring a
matching sign is informative for a non-null effect but **degenerate near zero, where the sign of an
estimate is noise**. The substantive reading is that the two signals agree completely: neither finds
anything distinguishable from zero.

This is a defect in the criterion as pre-specified, not something the data did. A caveat to this
effect is now printed in `signal_comparison.txt`; the criterion itself is unchanged (§O.3).

## R.8 Reading panel K

Panel K is legible but **washed out at the default `vmax=1.0`** — trial-averaged values are mostly
well under 1 SD, so most of the map sits near the middle of the colour scale. Dropping `vmax` to
~0.3–0.4 shows real structure.

One caveat when reading it: because cells are sorted by trace index and the index is *epoch minus
pre-tone*, **the sort induces an anticorrelated pre-tone band by construction**. The apparent
pre-tone contrast at the extreme rows is partly a sorting artifact, not a finding.

That caveat was subsequently put to a direct test, because the panel also appeared to show
sequential, time-cell-like structure. **It does not survive cross-validation — see §R.9.**

## R.9 The panel-K sorting artifact, tested directly

Panel K's C lane appeared to contain sequential / time-cell-like activity, most visibly during the
tone period in Exc and Inh. **The panel cannot settle that question**, for a structural reason: its
row order is computed from the very data it displays, and sorting any matrix by a statistic derived
from it produces a clean diagonal — including a matrix of pure noise. So the question was answered
with independent data instead.

**Module:** [caban/epoch_sequence.py](../caban/epoch_sequence.py); entry point
`caban.sections.run_epoch_sequence(ds, cfg)`, gated on `cfg.plot_epoch_sequence`; notebook cell
immediately after this analysis's own. Output `PLOTS_DIR/epoch_sequence/`. It is **exploratory and
additive — it changes nothing above.**

**Design.** Per mouse, on C, over the 0–40 s tone→shock window: split the retained trials into
disjoint halves (A = 1,3; B = 2,4), trial-average each separately, smooth 0.5 s, and take each
cell's peak latency as the argmax **independently in each half**. The statistic is the Spearman
correlation between the two halves' latencies **across the cells of one mouse** — one value per
mouse. Because the halves share no trials, the null is 0 by construction; **no shuffle or
permutation framework is used.**

Cells whose windowed trace is **constant** in either half are excluded: `argmax` silently returns
index 0 for them, so they would be recorded as peaking at exactly tone onset in *both* halves and
would contribute a **perfect** correlation. This is a *definability* requirement, not a
responsiveness filter — any non-constant cell is kept however weak, and an untuned cell then
dilutes the correlation toward zero. The controls below show this is not a hypothetical concern.

**C rather than YrA, deliberately, and for the opposite reason to §A.4.** There the endpoint is a
window mean, where C's sparsity is a liability. Here it is a *timing* statistic, where YrA's
sample-to-sample noise destabilises the argmax while C's denoised transients give a clean peak.

### Result: the apparent sequence is a sorting artifact

Run **2026-08-26 13:51**, same cohort and retained trials as §R.1.

| | Sort half (A) | Held-out half (B) |
|---|---|---|
| Ctl / Exc / Inh heatmaps | crisp diagonals in all three | **diagonal gone; essentially featureless** |

| Quantity | Value |
|---|---|
| Mean split-half ρ (17 mice) | **+0.039** |
| Mouse-level test of Fisher-z vs 0 | **t(16) = +1.13, P = 0.274** |
| Per-mouse ρ range | −0.257 … +0.368 |
| Per-group mean ρ | Ctl −0.031 (n=6), Exc +0.043 (n=5), Inh +0.107 (n=6) |
| Cells with a defined peak latency | 6,511 / 9,515 (**68.4%**); per mouse 40.4–83.3%, median 66.3% |

**The structure in panel K does not survive on trials that did not define it.** This matches the
noise control almost exactly (below), which is the strongest available evidence that what panel K
shows is the ordering rather than the biology.

### Controls, run on planted data

| Control | Result | What it establishes |
|---|---|---|
| Planted staggered sequence | ρ = **+1.00**, latencies span 40 s | a real sequence *is* recovered |
| **Pure noise** | sort half ρ = **+0.999**, held out ρ = **+0.043** | **a sorted noise matrix looks perfectly sequential** — the artifact, demonstrated |
| Silent cells (half the cells flat in-window) | ρ = **+0.86 without** the definability rule, **+0.08 with** it | the exclusion prevents a strong spurious sequence built out of silence |
| Perfect synchrony | ρ = **+0.01** | synchrony does *not* inflate ρ — identical latencies are ties |
| Two response clusters | ρ = **+0.79**, only ~4 distinct latencies | **a high ρ can occur with no tiling** — see below |

### Two things this result does not mean

**A high ρ would not have established a sequence.** The failure mode is specific, and the obvious
guess about it is wrong: perfect synchrony gives ρ ≈ 0 (ties), whereas *a few reproducible response
times* give ρ ≈ +0.8 with a handful of distinct latencies. A sequence additionally requires the
peaks to **tile** the interval, and only the held-out latency histogram shows that. It is on the
main panel for this reason.

**Peak latency is not the only form of temporal structure.** 20.5% of defined cells have their
held-out peak at a window edge (13.0% at 0 s, 7.5% at 40 s), meaning no interior peak — a drifting
rather than peaking trace, for which "peak latency" is a weak descriptor. Note this cuts *against*
a false null: cells pinned at the same edge in both halves would push ρ **up**, so the null is if
anything conservative. Still, the test rules out reproducible *peak timing*, not slower structure
such as sustained ramping across the trace interval (§O.7).

## R.10 The hierarchical cell-level companion (§A.8), both signals

Run **2026-08-27, 10:16 → 10:21** (3 min 0 s over both signals), `PLOTS_DIR = .../plots/CURRENT`.
Same cohort, same cells and same retained trials as §R.1: **9,515 cells × 4 epochs = 38,060 rows,
17 mice**, and the same `df_cell` table the mouse-level lane was fit on — the two analyses cannot
be describing different cells or different trials. Settings as frozen: 2,000 omnibus draws at seed
0, exact pairwise enumerations, statsmodels cross-check off.

**The conclusion is the same null, on both signals**, and it is a *stronger* null than §R.2's in
one specific sense and a *weaker* one in another — §R.10.4 is the part of this section that
matters.

### R.10.1 The model as fitted — and why the epoch random effect earns its place

Between-animal and within-animal variance components, REML, both signals:

| Component | YrA | C |
|---|---|---|
| mouse intercept | 0.002309 | 0.002705 |
| epoch — tone | **0.000091** | **0.000000** † |
| epoch — trace | 0.001710 | 0.000218 |
| epoch — **shock** | **0.033986** | **0.015548** |
| epoch — post_shock | 0.004752 | 0.001316 |
| cell (within mouse) | 0.012811 | 0.018614 |
| residual | 0.055756 | 0.065255 |

† a **boundary** estimate. That is a statement about the data — the between-animal spread of the
tone-epoch effect on C is not distinguishable from zero — reported as such rather than nudged off
the bound.

**These are not equal, and that is the whole point of the term.** On YrA the shock epoch's
between-animal variance is **373×** the tone epoch's; on C it is 71× the trace epoch's. §A.5's
mouse-level model has a *single residual variance shared across all four epochs* and therefore
cannot represent this. §R.10.4 shows what that costs.

The cell component is real but modest — 0.0128 (YrA) against a 0.0558 residual — so most
within-animal variation is trial-to-trial rather than stable cell identity. That is consistent
with §R.9's finding that per-cell response timing does not reproduce across trial halves.

### R.10.2 The omnibus — no group × epoch effect

| Signal | Hierarchical: mouse-label randomisation of the 6-df joint Wald | Mouse-level (§R.2) |
|---|---|---|
| YrA | F = 0.900, **P = 0.568** (2,000 MC draws, seed 0) | F(6,16) = 0.459, P = 0.828 |
| C | F = 0.604, **P = 0.650** (2,000 MC draws, seed 0) | F(6,16) = 0.647, P = 0.692 |

Both far from significance. Monte Carlo standard error at these values is ~0.011, so neither
number is near any threshold that the draw count could move it across.

### R.10.3 Within-epoch comparisons, in full

Difference in modulation index, analysed untransformed; 95% Wald interval on df = 16; exact
mouse-label randomisation *P* over the restricted space; Holm across the **two**
treatment-versus-control comparisons within each epoch. Exc-vs-Inh is computed and tabulated but
is **in no Holm family** and carries no bracket, exactly as on panel L.

#### YrA — primary

| Epoch | Comparison | Estimate | 95% CI | Exact *P* | *n* relabellings | *P*_holm | Design-based *P* |
|---|---|---|---|---|---|---|---|
| tone | Exc vs Ctl | −0.0281 | [−0.0930, +0.0368] | 0.3485 | 462 | 0.697 | 0.3312 |
| tone | Inh vs Ctl | −0.0120 | [−0.0741, +0.0501] | 0.7013 | 924 | 0.7013 | 0.6732 |
| tone | *Exc vs Inh* | −0.0161 | [−0.0806, +0.0484] | 0.6190 | 462 | — | 0.6169 |
| trace | Exc vs Ctl | −0.0319 | [−0.1148, +0.0511] | 0.5260 | 462 | 0.7814 | 0.5130 |
| trace | Inh vs Ctl | −0.0344 | [−0.1137, +0.0449] | 0.3907 | 924 | 0.7814 | 0.3874 |
| trace | *Exc vs Inh* | +0.0025 | [−0.0801, +0.0852] | 0.9719 | 462 | — | 0.9740 |
| shock | Exc vs Ctl | −0.0257 | [−0.2708, +0.2194] | 0.8377 | 462 | 0.8377 | 0.8377 |
| shock | **Inh vs Ctl** | **−0.0974** | [−0.3312, +0.1364] | **0.2294** | 924 | 0.4589 | 0.2294 |
| shock | *Exc vs Inh* | +0.0717 | [−0.1733, +0.3167] | 0.5325 | 462 | — | 0.5325 |
| post_shock | Exc vs Ctl | +0.0453 | [−0.0638, +0.1544] | 0.2468 | 462 | 0.4935 | 0.2446 |
| post_shock | Inh vs Ctl | −0.0134 | [−0.1176, +0.0908] | 0.7165 | 924 | 0.7165 | 0.7489 |
| post_shock | *Exc vs Inh* | +0.0587 | [−0.0501, +0.1675] | 0.1991 | 462 | — | 0.1991 |

#### C — confirmatory

| Epoch | Comparison | Estimate | 95% CI | Exact *P* | *n* relabellings | *P*_holm | Design-based *P* |
|---|---|---|---|---|---|---|---|
| tone | Exc vs Ctl | +0.0054 | [−0.0635, +0.0743] | 0.8766 | 462 | 1 | 0.9329 |
| tone | Inh vs Ctl | −0.0095 | [−0.0755, +0.0565] | 0.6180 | 924 | 1 | 0.5628 |
| tone | *Exc vs Inh* | +0.0149 | [−0.0537, +0.0835] | 0.7771 | 462 | — | 0.7900 |
| trace | Exc vs Ctl | +0.0059 | [−0.0656, +0.0774] | 0.8810 | 462 | 1 | 0.8810 |
| trace | Inh vs Ctl | −0.0127 | [−0.0811, +0.0558] | 0.7630 | 924 | 1 | 0.7143 |
| trace | *Exc vs Inh* | +0.0186 | [−0.0526, +0.0898] | 0.6710 | 462 | — | 0.6667 |
| shock | Exc vs Ctl | −0.0202 | [−0.1946, +0.1541] | 0.8052 | 462 | 0.8052 | 0.8074 |
| shock | **Inh vs Ctl** | **−0.0954** | [−0.2618, +0.0709] | **0.1591** | 924 | 0.3182 | 0.1602 |
| shock | *Exc vs Inh* | +0.0752 | [−0.0990, +0.2493] | 0.3658 | 462 | — | 0.3615 |
| post_shock | Exc vs Ctl | +0.0208 | [−0.0625, +0.1042] | 0.3723 | 462 | 0.7446 | 0.3333 |
| post_shock | Inh vs Ctl | −0.0294 | [−0.1090, +0.0503] | 0.4621 | 924 | 0.7446 | 0.4957 |
| post_shock | *Exc vs Inh* | +0.0502 | [−0.0327, +0.1332] | 0.2078 | 462 | — | 0.2100 |

**Nothing is significant, adjusted or unadjusted, on either signal.** The smallest raw *P* in the
whole table is C shock Inh-vs-Ctl at 0.159.

**The point estimates are the mouse-level ones**, to three decimals — YrA trace Exc-vs-Ctl −0.0319
against §R.2's −0.0319, shock Inh-vs-Ctl −0.0974 against −0.0972. That is by construction, not
corroboration: the design is balanced within cell, so the cell-level coefficients are a
cell-count-weighted version of the mouse-level ones (§A.8.2). **The estimates were never the
question — the intervals are.**

### R.10.4 What this lane actually changes: the mouse-level intervals are the wrong shape

The mouse-level model gives every epoch the *same* interval width, because it has one residual
variance for all four. The hierarchical model estimates each epoch's between-animal variance
separately, and the widths come apart. YrA, Exc-vs-Ctl, against the mouse-level ±0.1317 that
§R.2 reports at every epoch:

| Epoch | Hierarchical half-width | Ratio to mouse-level | Between-animal variance |
|---|---|---|---|
| tone | ±0.0649 | **0.49×** | 0.000091 |
| trace | ±0.0830 | 0.63× | 0.001710 |
| **shock** | **±0.2451** | **1.86×** | 0.033986 |
| post_shock | ±0.1091 | 0.83× | 0.004752 |

The same pattern on C (0.69× / 0.72× / **1.76×** / 0.84× against its ±0.099), and the same ratios
again for Inh-vs-Ctl on both signals — the widths are a property of the epoch, not of the
comparison. **Pooling one residual across epochs was borrowing precision from the quiet epochs and
lending it to the loud one.** The tone and trace intervals were about twice as wide as they needed
to be; the shock interval was about half as wide as it should have been.

**This lands squarely on the one contrast §R.2 singled out.** "Inh-vs-Ctl at shock" — the largest
effect, consistent across signals, C raw *P* = 0.049 — is the comparison sitting in the epoch whose
uncertainty was most understated:

| | Mouse-level raw *P* | Mouse-level Holm | Hierarchical exact *P* | Hierarchical Holm |
|---|---|---|---|---|
| YrA shock Inh vs Ctl | 0.120 | 0.240 | 0.229 | 0.459 |
| **C shock Inh vs Ctl** | **0.049** | 0.098 | **0.159** | 0.318 |

The **estimate is unchanged** (−0.095 on C, either way). What changed is the interval around it,
and it changed in the direction the variance components say it should. §R.2's instruction to
*treat it as a hypothesis, not a result* is **reinforced**, and the C raw *P* = 0.049 should not be
quoted without this alongside it.

Conversely, the tone and trace nulls are now on a **narrower** interval than §R.2 reported — the
same conclusion, better supported.

### R.10.5 The two sensitivity columns, and what they say

**The design-based statistic tracks the model-based one almost exactly.** Across all 24
comparisons the largest divergence is at C tone Exc-vs-Ctl (0.877 model, 0.933 design-based), and
most agree to within 0.02 — YrA shock Inh-vs-Ctl is 0.2294 against 0.2294. No equivalence was
claimed (§A.8.1: GLS and equal-animal weighting coincide only at equal cluster sizes, and cells per
mouse run 89–1,047 here), and none is needed. It simply means the model-based result is not an
artefact of the GLS weighting.

**The asymptotic column is not uniformly smaller than the randomisation column**, which is worth
stating because the naive expectation is that a cell-level model inflates significance. It does not
here, for the reason `sp_rates_lmm` §4.3 records: `linear_contrast_test` already uses
df = n_mice − 1, the animal-level convention, so the asymptotic *P* is not a cell-count-inflated
quantity to begin with. On YrA shock Inh-vs-Ctl the asymptotic *P* is **0.390** against the
randomisation's **0.229** — the randomisation is the *smaller* one. Neither is paper-facing; the
asymptotic column is written out under a name that says so.

### R.10.6 Signal agreement

**All 12 comparisons agree between YrA and C**, under §A.8.3's fixed criterion (both CIs contain
zero, *or* matching sign with overlapping CIs). Three of the twelve — tone Exc-vs-Ctl
(−0.0281 / +0.0054), tone Exc-vs-Inh (−0.0161 / +0.0149) and trace Exc-vs-Ctl (−0.0319 / +0.0059) —
have estimates of *opposite sign* on the two signals and would have been flagged DISAGREE by the
mouse-level criterion. They are the same near-zero sign flips §R.7 describes: every one of the six
intervals involved spans zero. This lane's criterion calls them what they are.

### R.10.7 Verification, on this run

Both signals passed the mandatory exactness check with no cross-check enabled:

| | YrA | C |
|---|---|---|
| REML objective identity, offset spread across 4 probes | **1.02×10⁻¹²** (relative 6.1×10⁻¹⁵) | **7.67×10⁻¹³** (relative 4.5×10⁻¹⁵) |
| collapsed GLS vs full-covariance GLS, max \|Δβ\| | **3.82×10⁻¹⁵** | **2.55×10⁻¹⁵** |

That is what licenses the collapsed fit as the model (§A.8.2). The optional statsmodels refit was
**not** run — `hierarchical_model_summary.txt` is correspondingly absent from the output, and the
`*_full_fit_crosscheck` columns of the permutation CSV are empty. An earlier pass with it enabled
agreed to 4×10⁻⁶ on coefficients and 0.2% on standard errors.

`verify_hierarchical_epoch_synthetic` — the hand-run development tool, not part of this run —
passed all five planted designs, including the pseudoreplication guard (a +6.0 effect confined to
one animal gives *P* = 0.584, not significance) and the cell-scaling check (within-animal SEM falls
10.0× over a 100× cell increase while the group-contrast SE moves 0.92×).

### R.10.8 Cost

**1.6 min (YrA) + 1.4 min (C) = 3.0 min.** The first working version of this lane took **53 min**,
and none of the difference is a weakened analysis — same model, same enumerations, same estimand:

| | was | now |
|---|---|---|
| statsmodels full refit, mandatory | 246 s + 807 s = **17.5 min** | **off by default** — a corroboration, not the guarantee (§A.8.2) |
| `HIERARCHICAL_N_PERM_OMNIBUS` | 20,000 ≈ 13 min/signal | **2,000**, `sp_rates_lmm`'s frozen value |
| optimiser gradient | finite-difference, **33 ms**/refit | analytic, **10 ms**/refit, same optimum |

Raise the draw count deliberately if an omnibus ever lands near 0.05, where Monte Carlo error would
be load-bearing. An earlier pass at 20,000 draws and a finite-difference optimiser gave omnibus
*P* = 0.580 (YrA) and 0.681 (C), and agreed with the tables above to four decimals on every
estimate; one exact *P* moved by 1/924 (YrA shock Inh-vs-Ctl, 0.2284 → 0.2294) as a single
borderline relabelling flipped under the tighter optimum.

### R.10.9 What this null does and does not establish

**Does:** retaining within-animal cellular coherence and weighting animals by how precisely their
own cells estimate them, there is still no detectable group difference in the epoch modulation
profile, on either signal, at *n* = 17 mice — and the epoch-specific intervals are now the right
shape.

**Does not:** anything §R.6 did not already exclude. This lane models the cellular *hierarchy*, not
the *shape* of the cellular distribution: a manipulation that raised modulation in some cells and
lowered it in others at constant mean is invisible to it exactly as it is to the mouse-level lane.
It also buys **no treatment-level power** — the contrast standard error is a between-animal
quantity, floored by between-animal variance, and no number of cells reduces it (demonstrated on
planted data, §A.8.3). Q2 (§O.2) remains the thing that would address distribution shape.

---

# Part O — open items

**O.1 The hierarchical cell-level companion — DONE.** Implemented as specified; see **§A.8** for the
design and **§R.10** for its results. The flagged tractability risk was real and was measured (one
full fit is 220–300 s, so a literal implementation is 60–150 h per signal); it was resolved
algebraically rather than by reducing the structure, using the exact orthogonal split that §A.2's
within-cell balance guarantees, with the objective identity asserted numerically on every run
(§A.8.2). Nothing above it changed.

**O.2 Q2 (trial-to-trial stability) is now answerable and was left contingent.** Retained trials are
4 for 16/17 mice, so an odd/even split is 2-vs-2 (2-vs-1 for G09) — the thinnest defensible
split-half. It would measure the stability of *relative* cell-specific response profiles, and must be
framed that way: **not** as a "stable cell-identity code". Whether it is worth reporting given §R.2 is
a judgement call.

**O.3 The signal-agreement criterion should be fixed** (§R.7). Agreement should be judged on whether
the two signals license the same conclusion — at minimum, a sign flip between two estimates whose
CIs both contain zero should count as agreement. **The fixed criterion is now written and running**,
in `compare_hierarchical_signals` (§A.8.3) — new code, so it was not built with a known defect. The
mouse-level `compare_signals` is deliberately **unchanged**, because its numbers are already
reported in §R.7; adopting the fixed criterion there is still the open item.

**O.4 Panel K `vmax` should probably default lower** than 1.0 (§R.8).

**O.5 The YrA/S unit-set mismatch is an upstream provenance issue** (§A.4). Re-exporting `YrA.zarr`
over the same unit set as `S.zarr` would remove the cell restriction entirely. Worth understanding
before publication regardless, since anything else reading YrA positionally is silently affected.

**O.6 Runtime.** Both lanes now produce panels, roughly doubling panel-generation time; the full run
took 3 min 18 s. `resolve_shared_cells` also recomputes per-cell SDs for both signals once per signal
pass — computing the shared set once and reusing it would remove that duplication.

**O.7 Slower temporal structure is untested.** §R.9 rules out reproducible *peak timing*; it does
not address sustained ramping across the trace interval, which an argmax cannot capture. The same
split-half logic would answer it with a different per-cell statistic (e.g. the slope over the
interval) and would need no new framework. Not run, and not obviously worth running given that the
held-out heatmaps are featureless — a strong ramp should have left visible structure there.

**O.8 Panel K's sort could be made honest by construction.** Since `epoch_sequence` already computes
a held-out ordering, panel K could be sorted on one trial half and displayed on the other, which
would remove the artifact from the figure itself rather than documenting it in a caveat. That is a
change to a paper-facing panel, so it is flagged rather than made.

---

# File map

| Path | Contents |
|---|---|
| `caban/epoch_modulation.py` | the analysis |
| `caban/sections.py::run_epoch_modulation` | `(ds, cfg)` entry point, gated on `cfg.plot_epoch_modulation` |
| `analysis_methods_templates/epoch_modulation_methods.md` | manuscript METHODS text, copied into the output dir at runtime |
| `<out>/{YrA,C}/panel_K_tone_aligned_heatmaps.{png,svg}` | panel K |
| `<out>/{YrA,C}/panel_L_epoch_modulation.{png,svg}` | panel L |
| `<out>/{YrA,C}/stats/contrasts.txt` | contrasts, omnibus, variance diagnostic |
| `<out>/{YrA,C}/stats/model_summary.txt` | raw fitted model summary |
| `<out>/{YrA,C}/tables/per_cell_modulation.csv` | per (mouse, cell, epoch) index, median over trials |
| `<out>/{YrA,C}/tables/per_mouse_modulation.csv` | per (mouse, epoch) mean, n_cells, sd_cells, sem_cells |
| `<out>/{YrA,C}/tables/trial_coverage.csv` | retained/dropped trials and cells, realised window durations |
| `<out>/{YrA,C}/tables/model_contrasts.csv` | machine-readable contrast table |
| `<out>/signal_comparison.{txt,csv}` | YrA-vs-C side by side, agreement verdicts, criterion caveat |

Hierarchical cell-level companion (§A.8), on by default — `cfg.epoch_modulation_hierarchical_cells`:

| Path | Contents |
|---|---|
| `<out>/{YrA,C}/hierarchical_cells/hierarchical_epoch_modulation.{png,svg}` | the modulation panel, brackets from this lane's exact randomisation |
| `<out>/{YrA,C}/hierarchical_cells/tables/hierarchical_permutation.csv` | every comparison: model contrast, CI, exact P, Holm P, design-based sensitivity, full-fit cross-check |
| `<out>/{YrA,C}/hierarchical_cells/stats/hierarchical_contrasts.txt` | the report, including both equivalence assertions and the fitted variance components |
| `<out>/{YrA,C}/hierarchical_cells/stats/hierarchical_model_summary.txt` | the full cell-level statsmodels fit (cross-check; its P>\|z\| is not paper-facing) |
| `<out>/{YrA,C}/hierarchical_cells/stats/hierarchical_vs_mouse_level_summary.md` | the two analyses side by side |
| `<out>/hierarchical_cells/signal_comparison.{txt,csv}` | YrA-vs-C for this lane, fixed agreement criterion |
| `analysis_methods_templates/epoch_modulation_hierarchical_cells_methods.md` | its METHODS text, copied into the output dir at runtime |

Event-proximal companion (§A.9), on by default — `cfg.epoch_modulation_event_proximal`:

| Path | Contents |
|---|---|
| `<out>/{YrA,C}/event_proximal/panel_M_event_aligned_traces.{png,svg}` | group-mean activity around each event onset, analysed window shaded |
| `<out>/{YrA,C}/event_proximal/panel_N_event_proximal_index.{png,svg}` | the index by event × group, brackets from this lane's hierarchical randomisation |
| `<out>/{YrA,C}/event_proximal/dilution_vs_full_epoch.{png,svg}` | per-mouse short-window index against the full-epoch one, paired |
| `<out>/{YrA,C}/event_proximal/tables/` | `per_cell`, `per_mouse`, `trial_coverage` (incl. flat-window fraction), `dilution_vs_full_epoch`, `model_contrasts_crosscheck` |
| `<out>/{YrA,C}/event_proximal/stats/event_proximal_report.txt` | the window, the paired dilution summary, the flat-window audit |
| `<out>/{YrA,C}/event_proximal/stats/mouse_level_crosscheck.txt` | the shared-variance mouse-level fit, labelled as a cross-check |
| `<out>/{YrA,C}/event_proximal/hierarchical_cells/` | the reported inference, produced by the same shared lane |
| `<out>/event_proximal/signal_comparison.{txt,csv}` | YrA-vs-C for this lane, fixed agreement criterion |
| `analysis_methods_templates/epoch_modulation_event_proximal_methods.md` | its METHODS text, copied into the output dir at runtime |

Cross-validated sequence test (§R.9), exploratory and additive:

| Path | Contents |
|---|---|
| `caban/epoch_sequence.py` | the sequence test |
| `caban/sections.py::run_epoch_sequence` | `(ds, cfg)` entry point, gated on `cfg.plot_epoch_sequence` |
| `<PLOTS_DIR>/epoch_sequence/cross_validated_sequence.{png,svg}` | sort half vs held-out half per group, with held-out latency histogram |
| `<PLOTS_DIR>/epoch_sequence/split_half_correlations.{png,svg}` | per-mouse ρ by group |
| `<PLOTS_DIR>/epoch_sequence/tables/per_cell_peak_latency.csv` | both halves' latencies + `defined` flag |
| `<PLOTS_DIR>/epoch_sequence/tables/per_mouse_correlations.csv` | ρ, Fisher-z, retained cell counts and fraction |
| `<PLOTS_DIR>/epoch_sequence/stats/sequence_report.txt` | the above plus the mouse-level test |
