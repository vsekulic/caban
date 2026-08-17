# Cell-level pyramidal event amplitude across DREADD groups (trace fear conditioning)

## Overview

- This analysis replaces the ~205 uncorrected three-group ANOVAs of the earlier `sp_rates`
  analysis (session x behaviour period x cross-registration subset x metric) with a small,
  pre-declared confirmatory family built on **per-event deconvolved amplitude at the cell
  level**. No step in this analysis collapses cells to a per-mouse scalar before testing;
  amplitude/rate/fraction-active are computed per cell and modelled with a mouse random
  intercept, so hundreds of cells per animal inform the estimate without inflating the effective
  n of independent DREADD assignments (still 5 hM3D / 6 hM4D / 6 mCherry mice).
- **The scientific claim under test is that hM3D increases the per-event deconvolved run integral**
  ("larger calcium events") during conditioning; **bursting is the stated MECHANISTIC
  INTERPRETATION of that measurement, not the measurement itself** — see "Event definition" below
  for exactly what is and is not directly observed. **Event amplitude is therefore the PRIMARY
  endpoint; event rate is SECONDARY.** This is a deliberate departure from the usual convention
  (and from an external methodological review consulted while designing this analysis, which
  recommended rate as primary) — the physiological claim here is specifically about the magnitude
  of the deconvolved signal per detected run, and rate is reported in every panel alongside
  amplitude with a full confidence interval, never hidden or dropped, so "secondary" means it
  carries no confirmatory alpha, not that it is de-emphasized.
- **This is a LOCKED CONFIRMATORY REANALYSIS, not a prospective preregistration.** The
  amplitude-primary decision followed prior inspection of this dataset (the earlier ~205-test
  `sp_rates` sweep). Locking the confirmatory family (this document, before the fixes below were
  applied) and holding it fixed across re-runs is what earns the term "confirmatory" here — it is
  not a claim that the hypothesis was specified before any data were seen.
- **A post-hoc methodological review of an earlier version of this module found three correctness
  bugs and two overstated designs**, all fixed in the current version (see each numbered section
  below for the fix and rationale): (1) the omnibus test's denominator degrees of freedom used an
  observation-level, not cluster-level, count; (2) the rate model's exposure offset was missing a
  factor of `n_cells`; (3) a recall-window figure carried a hardcoded trace-period title; (4) the
  co-primary epoch model understated its own uncertainty by not accounting for within-cell
  repetition; (5) the primary contrast's point estimate was implicitly cell-weighted despite cell
  count being group-correlated and post-treatment. The underlying biology (hM3D shows larger
  per-event amplitude during conditioning; hM4D is near-null on amplitude but reduced on rate)
  survived all five corrections; several previously-reported *numbers* did not and must not be
  reused from before this pass.

## Event definition: per-event integral, not peak amplitude

- `find_spikes_ca()` / `find_spikes_ca_S()` (`caban/utilities.py`) detect one event per LOCAL
  MAXIMUM of the thresholded S trace and report the S value AT THAT PEAK FRAME. Summed over a
  window and divided by duration, this exactly equals `rate x mean peak height` — mathematically
  incapable of separating "bigger events" from "more events."
- This analysis instead uses `find_event_runs_ca()` / `find_event_runs_ca_S()`
  (`caban/utilities.py`): one event per CONTIGUOUS supra-threshold run of S, with amplitude
  defined as the PER-EVENT INTEGRAL — `sum(S[run])` over the whole run, not the peak value. A
  wider/taller run (more bursting) therefore contributes a larger amplitude even at the same peak
  height, which the legacy peak-only measure cannot express.
- A run containing two local maxima (per `find_spikes_ca()`'s own peak detector, reused inside
  `find_event_runs_ca()` purely as a diagnostic count — see `n_local_maxima` in its return value)
  is one event here, not two. This changes event *counts* relative to the legacy convention, so
  the frequency/rate endpoints in this analysis are recomputed under this SAME event definition,
  not mixed with legacy `find_spikes_ca()` counts. **This is also why "bursting" is an
  interpretation, not the measurement**: because temporally clustered peaks MERGE into one wider
  run under this event definition, part of the observed "high amplitude + low rate among active
  cells" pattern in hM3D is a mechanical consequence of the merging itself, not independent
  evidence for it. The run-structure evidence below (run width, local maxima per run, fraction of
  multi-peak runs) is what distinguishes "genuinely burst-like" from "a threshold/merging
  artifact of the event definition" — a coupling that must be stated, not silently relied on.
- **Zero-event cells are kept in every rate/fraction-active computation** (the codebase's
  existing convention: a manipulation that silences cells must show up as a reduced population
  rate, which only happens if silenced cells remain in the denominator). Amplitude is undefined
  for a cell with zero events by construction — those cells are excluded from amplitude
  computations, not imputed to zero. This is definitional, not a missing-data problem.

## Exact decomposition: rate and amplitude are never separate analyses

For a fixed exposure window at the cell level:

```
summed S per second  =  event rate  x  mean per-event amplitude
```

which is additive on the log scale: `log(total S/sec) = log(rate) + log(mean amplitude)`. All
three quantities are reported from the same panel/figure, and where both the rate and amplitude
models are fit for the same contrast, the reported coefficients should sum consistently
(`caban/sp_rates_lmm.py`'s `report_decomposition_additivity()` performs this arithmetic
check — it is not itself a fitted model or a significance test, since a proper standard error
for the sum requires the joint covariance of two separately-fit models, which is not available).

Aggregated across cells, this identity holds with **event-weighted**, not cell-weighted, mean
amplitude (`sum_amplitude / n_events` over the pooled cells, not the average of each cell's own
mean). The per-cell amplitude MODELS in this analysis use cell-weighted means (one row per
cell — each cell one vote, consistent with the cell-level unit of observation this analysis
otherwise commits to); only the additivity-check identity itself is event-weighted.

## Epoch structure

- TFC_cond epochs used by the confirmatory model: `pre_tone` (35 s baseline window, a chosen
  default, not protocol-derived), `tone` (20 s), `trace` (`tone_offsets[i]` to
  `shock_onsets[i]`), `post_shock`. **Shock (2 s) is excluded**: at 0.05-0.2 Hz a 2 s window
  yields ~0-1 events per cell, dominated by quantization, and the window carries motion artifact.
  Handled separately elsewhere via YrA/C, not here.
- **Trace duration is NOT constant across the five trials.** From `TraceFearCondSession`'s own
  timing arrays (`tone_onsets_def = [185, 420, 660, 900, 1140]`, `tone_duration = 20`,
  `shock_onsets_def = [220, 460, 700, 940, 1180]`): trace durations are `[15, 20, 20, 20, 20]`
  seconds. Trial 1 is short because `tone_onsets_def[0] = 185` rather than an intended 180 s
  (flagged as an accident in the source comment — at 180 the first trace would also be 20 s).
  This needs no special handling in the models: every window's exposure (`exposure_seconds`) is
  computed from that trial's ACTUAL onset/offset frames, not an assumed constant duration, so the
  unequal first trial is already correctly absorbed wherever an exposure offset is used.
- Test_B / Test_B_1wk post-tone window is pinned to **20 s** explicitly (via
  `get_testb_epoch_frames(..., post_tone_duration_s=20.0)`), matching the representative TFC
  trace duration (4 of 5 trials). Neither of the two existing alternatives is the correct recall
  analog of trace: `get_testb_epoch_frames`'s own default is 35 s, and the sp_rates-native
  `TestBSession.post_tone_onsets/offsets` extends all the way to the next tone onset
  (~200+ s inter-trial interval, not a matched window at all).

## Statistics

### Denominator degrees of freedom (applies to every joint Wald test below)

- `caban.single_unit_common.joint_wald_test` REQUIRES an explicit `n_groups` (mouse count) and
  uses `df2 = n_groups - 1` for every omnibus F-test in this module, for both MixedLM and
  cluster-robust OLS fits. **This replaced an observation-level `nobs - n_fixed` denominator**
  that treated each cell (or cell x trial x epoch row) as an independent unit of information —
  correct for i.i.d. rows, badly wrong when hundreds to thousands of rows are nested inside only
  17 mice. Verified exactly on the primary trace model: the old denominator reported `F(2, 7407)`
  → p = 0.0016; the corrected `F(2, 16)` → p = 0.0089, matching statsmodels' own
  `Prob (F-statistic)` for the equivalent cluster-robust OLS fit to the digit.
- For a **between-mouse** contrast (group does not vary within a mouse — the primary and
  co-primary tests below), `n_groups - 1` is the standard small-G cluster-robust choice (Cameron &
  Miller 2015) and is exactly appropriate, not merely conservative.
- For a **within-mouse** factor (the group x trial photobleaching check, where trial varies within
  a mouse), `n_groups - 1` is used as a DOCUMENTED CONSERVATIVE approximation: a
  Satterthwaite/Kenward-Roger denominator informed by the mouse x trial residual would in
  principle afford more power, but this codebase implements no such correction. This affects
  power, not Type I error control.

### Confirmatory family (Holm-corrected across exactly these two tests)

- **Primary endpoint**: `log(mean per-event amplitude) ~ C(group, Treatment('mCherry'))`,
  cell-level, trace epoch pooled (summed, not averaged) across the five trials, random intercept
  on mouse. Falls back to mouse-clustered OLS if the mixed model does not converge or converges
  to a degenerate boundary fit (`caban.single_unit_common.fit_mixed_model`, the same fallback
  logic and degeneracy check used by every mixed-model analysis in this codebase). The primary
  test statistic is the joint Wald test that both non-reference group coefficients are zero
  (`caban.single_unit_common.joint_wald_test`).
  - **Reported point estimate is EQUAL-MOUSE-WEIGHTED, not cell-weighted**
    (`caban.sp_rates_lmm.compute_group_contrast_point_estimates`): the mean of each mouse's own
    mean log-amplitude, matching what the 17-mouse-level figure shows. Amplitude is only defined
    for event-active cells, cell inclusion is post-treatment and activity-dependent, and cell
    count is itself group-correlated (hM3D contributes ~1.45x as many active cells/mouse as the
    other groups) — a pooled cell-weighted mean would over-weight mice with more active cells.
    The cell-weighted pooled mean is still reported, but only as a supporting "average detected
    active cell" quantity, never the headline number. The mouse-label permutation test (below)
    is run at both weightings for the same reason; only the point estimate/reduce_fn's inputs
    change with weighting — the permutation *inference* is valid either way, since labels always
    permute at the mouse level.
- **Co-primary endpoint**: within-cell delta, `delta_log_amplitude ~ C(group, Treatment('mCherry'))`,
  one row per cell, random intercept on mouse
  (`caban.sp_rates_lmm.compute_epoch_delta_table` + `fit_epoch_delta_model`). For each cell active
  in BOTH the trace epoch and the pre_tone baseline, `delta_log_amplitude = log_amplitude[trace] -
  log_amplitude[pre_tone]`; only cells active in both qualify (report the qualifying count
  alongside the result — this is a real, reportable selection, not a formality).
  - **This replaced a `group * epoch + trial` interaction fit directly on the cell x trial x
    epoch table** (mouse random intercept only). A mouse random intercept alone does not
    represent within-cell repetition: a single cell could contribute up to 20 rows there (4
    epochs x 5 trials), and two observations of the SAME cell are far more correlated than two
    arbitrary cells from the same mouse. That design reported p=1.2e-11 — not defensible, and
    visibly overstated relative to the epoch-profile figure, whose dominant pattern is hM3D
    elevated at every epoch with modest interaction coefficients (+0.08–0.09) against a much
    larger main group effect (+0.43).
  - Within-cell differencing cancels each cell's own baseline level — the dominant contaminating
    source — BY CONSTRUCTION, the same logic already used and validated in the LT1→LT2
    manipulation check below. The resulting one-row-per-cell delta converts what was an epoch x
    group INTERACTION into a plain group MAIN-EFFECT test on a per-cell contrast, i.e. a
    BETWEEN-mouse comparison structurally identical to the primary endpoint — which is also why
    `n_groups - 1` is exactly appropriate here, not merely conservative.
  - **A full nested-random-effects confirmation** — `(1|mouse) + (1|mouse:cell)` via
    statsmodels' `vc_formula`, correctly representing the repeated-measures structure the
    within-cell delta only approximates by collapsing trial and epoch-vs-baseline into one number
    — is available as a one-off, opt-in check
    (`caban.sp_rates_lmm.fit_epoch_interaction_nested_attempt`), NOT part of the routine pipeline:
    at ~8,000+ cell levels over ~100k rows this is expected to be slow and may not converge, and
    is reported only if it converges cleanly.
- **Holm correction is applied across exactly these two omnibus p-values** — the entire
  confirmatory multiplicity burden of this analysis (`caban.sp_rates_lmm.holm_correct_confirmatory`).
  Everything else below is secondary (BH-FDR) or purely descriptive/sensitivity.

### The co-primary null, and the complete epoch profile (descriptive)

**The co-primary within-cell trace-vs-pre_tone delta is NULL.** The hM3D amplitude effect is a
large group MAIN effect that is present at every epoch, not an elevation specific to the trace
interval. (The previously reported p = 1.2e-11 for a trace-specific effect was an artifact of the
pseudoreplicating interaction fit described above and must not be cited.)

Two consequences for how this analysis is written up:

- **No figure, caption or claim may imply a trace-specific effect.** The trace interval's
  privileged position in this design rests on prior anatomy and on the behavioural readout, not
  on this result.
- Because "elevated globally rather than trace-specifically" is a claim about *all* the epochs,
  the same within-cell delta contrast is computed and reported for **every** non-reference epoch
  (`caban.sp_rates_lmm.compute_all_epoch_deltas`, plotted by `plot_epoch_delta_forest`):
  tone−pre_tone and post_shock−pre_tone alongside the confirmatory trace−pre_tone. A reader can
  then see the flat profile directly rather than taking it on trust, and cannot mistake "no
  trace-specific effect" for "no effect".

The non-confirmatory epochs are **descriptive**. They spend no alpha, are not in the confirmatory
Holm family (which is exactly two tests), and are not in the secondary BH-FDR family either —
their role is to characterize a null, not to test a hypothesis. Only the trace row is flagged
`is_confirmatory` in the output table.

### Bursting evidence: run structure and threshold sensitivity

- **Run-structure panels** (`caban.sp_rates_lmm.build_run_structure_table` /
  `summarize_run_structure` / `plot_run_structure`), trace epoch only (matching the primary
  endpoint's own scope): per-RUN width (frames), local maxima per run, and the fraction of runs
  with >=2 local maxima (temporally clustered peaks that MERGED into one run under this analysis'
  contiguous-run event definition — see "Event definition" above). Reported as mouse-level means
  with a mouse-label permutation test per metric, and displayed as mouse-level violins (the full
  per-run distribution behind them is written to `stats/run_structure.txt`). This is the direct
  signature that distinguishes "genuinely burst-like" from
  "a threshold artifact of run-merging": a height-independent, fattened right tail in run width is
  what a bursting interpretation predicts and a pure rate or detection-threshold explanation does
  not.
- **Threshold sensitivity** (`caban.sp_rates_lmm.run_threshold_sensitivity` /
  `plot_threshold_sensitivity`): the primary amplitude contrast re-fit at
  `thres in {1.5, 2.0 (assumed default — verify against a live session), 3.0}`, reported as a
  forest plot of effect estimates + 95% CI. If the hM3D effect stays on the same side of zero
  across all three thresholds, run-merging at any one threshold choice cannot be the sole
  explanation for the effect.

### Prespecified secondary family (BH-FDR)

**The family, and what is deliberately outside it** (`caban.sp_rates_lmm.build_secondary_fdr_table`;
written to `stats/secondary_fdr_family.csv` and `.txt`). Benjamini-Hochberg at `alpha = 0.05`
across the FREQUENTIST secondary tests only:

- Test_B and Test_B_1wk post-tone amplitude omnibus tests,
- the `group x trial` photobleaching interaction,
- the run-structure mouse-label permutation tests (run width, local maxima per run, multi-peak
  fraction; each DREADD group vs control).

Excluded, by declaration rather than omission:

- **The two confirmatory omnibus tests.** They carry their own Holm correction over a family of
  exactly two (`holm_correct_confirmatory`) and control a different error rate. A test placed in
  two families is corrected twice.
- **The Bambi Negative-Binomial rate model.** It reports posterior contrasts, HDIs and an
  ELPD-LOO comparison; there is no p-value to correct. Manufacturing one so a Bayesian result can
  be folded into a frequentist FDR family is a category error, not a conservative choice.
- **Purely descriptive and sensitivity output**: the non-confirmatory within-cell epoch deltas,
  the threshold-sensitivity forest, the LT1→LT2 dropout fractions, and the amplitude permutation
  tests that re-express the primary contrast under a different weighting or a tail statistic.
  Correcting descriptives enlarges the family and costs power on the tests actually making claims.

- **Event rate**, trace epoch: `n_events ~ group * epoch + trial` with a
  `log(exposure_seconds)` offset, at the mouse x trial x epoch level (summed over cells within
  each mouse-trial-epoch window), fit as a genuine Negative-Binomial mixed model via Bambi (PyMC
  backend) — `caban.sp_rates_lmm.fit_rate_group_epoch_model`.
  - **Random effects: `(1|mouse) + (1|mouse_trial)`.** The mouse-trial intercept was added on
    methodological review. A trial's four epochs are not independent replicates of that mouse:
    they are consecutive windows of one behavioural episode sharing its arousal/locomotor state,
    its position in the photobleaching decline, and its imaging conditions. With only `(1|mouse)`
    that shared variation falls into the residual and the epoch / `group x epoch` terms get
    standard errors that are too small. Sampler convergence (max r-hat, min ESS, divergence
    count, for both the full and reduced fits) is reported in `stats/secondary_rate.txt`.
  - **Rate ratios are always reported next to absolute differences** (events/s, equal-mouse-
    weighted, both among active cells and across all cells). A fold-change computed off a small
    base overstates the practical size of a change, which matters most for interpreting the hM4D
    rate result.
  - **Exposure offset**: `exposure_seconds` in the mouse x trial x epoch table
    (`build_mouse_trial_epoch_rate_table`) is `E_mte = sum over cells of T_mcte` — total valid
    cell-seconds, i.e. `n_cells_in_window * window_seconds` (every cell in one window shares the
    same window duration). **This used to be taken with `'first'`** (a single cell's window
    duration) while `n_events` was already summed across all cells, so the offset's numerator was
    population-level but its denominator was one cell's window — missing a factor of `n_cells`.
    Since `log(n_cells * T) = log(n_cells) + log(T)` and `n_cells` is GROUP-CORRELATED (hM3D
    contributes more amplitude-active cells/mouse than the other groups), the missing
    `log(n_cells)` term leaked into the group fixed effect rather than being absorbed by the
    mouse random intercept (a shrunk random effect, not a free per-mouse parameter).
  - **Implementation note**: `statsmodels`' `mixedlm` is Gaussian-only and has no
    negative-binomial/GLMM path, so the rate endpoint cannot use the same fitting machinery as
    the amplitude endpoints above — a genuine Python-tooling limitation (`statsmodels` alone
    cannot fit this model), not a statistical preference. Bambi closes that gap directly:
    dispersion (alpha) is estimated jointly with the fixed and random effects in a single fit,
    the same way R's `glmmTMB`/`brms` would, rather than via a separate pre-estimation step.
  - **Implementation note**: the interaction's contribution is assessed via **LOO
    cross-validation** (`arviz.compare`, expected log predictive density) between this full model
    and a reduced model without the group x epoch interaction, rather than a p-value — the
    natural Bayesian analog of the joint Wald test used for the (Holm-corrected) amplitude
    confirmatory family. This is reported for context; it carries no confirmatory alpha, and rate
    stays secondary regardless of the comparison's outcome.
  - **Implementation note**: `bambi`/`formulae` does **not** respect a pandas `Categorical`
    column's `categories=` order for reference-level selection the way `patsy`'s
    `C(x, Treatment(reference=...))` does purely from column dtype — confirmed empirically before
    relying on it (it silently fell back to something resembling alphabetical order instead). The
    formula therefore names the reference level explicitly via `C(col, Treatment('level'))`
    (positional argument — `formulae`'s own syntax, not `patsy`'s `reference=` keyword).
- **Fraction of cells active** (>=1 event), trace epoch, per mouse.
- **Test_B / Test_B_1wk post-tone amplitude** — the recall complement to the trace endpoint (same
  fitting procedure as the primary, applied to the post-tone window). Motivated by a separately
  observed freezing phenotype (hM3D maintains elevated post-tone freezing at 1 week while hM4D
  and mCherry extinguish), so this asks whether the single-cell amplitude signature persists at
  the timepoint where the behavioural difference is most pronounced.
  - **Recall N caveat**: Test_B lacks one mouse and Test_B_1wk lacks a DIFFERENT one (both
    currently hM4D, so hM4D n=5 in each rather than 6) —
    `caban.sp_rates_lmm.report_recall_cohort_note` re-derives and reports exactly which mice are
    present per session rather than hardcoding this, since it can change as the dataset does. The
    consequence: the 48h -> 1wk comparison is not evaluated on a fixed cohort.
- **group x trial interaction**, trace epoch amplitude
  (`caban.sp_rates_lmm.build_mouse_trial_trace_amplitude` + `fit_group_trial_model`): the
  photobleaching control. Directly addresses the freezing confound described below: since
  freezing only emerges in trials 4-5 of conditioning, this interaction tests whether groups
  diverge as freezing develops, without restricting to any particular behavioural-state subset of
  frames. Fit on a per-MOUSE-per-trial table (cells collapsed within mouse first), not the
  cell x trial table directly with only a mouse random intercept — the latter would repeat the
  co-primary's original pseudoreplication bug exactly (a cell contributing up to 5 trial rows).
  Collapsing to one value per mouse-trial first makes trial a proper within-mouse repeated
  measure with no remaining cell-level nesting.
  `caban.sp_rates_lmm.report_group_trial_slopes` reports a simple descriptive per-group slope
  (log-amplitude/trial) alongside the categorical omnibus, for a directly interpretable number —
  it is NOT itself a formal test.

### Small-n inference: mouse-label permutation

- In addition to the model-based tests above, `caban.sp_rates_lmm.mouse_label_permutation_test`
  provides an assumption-light check: shuffle GROUP LABELS across the 17 mice (holding every
  cell's own data fixed), recompute the group contrast under each shuffle, and build a null
  distribution. This is Monte Carlo (default 20,000 draws), not exact enumeration — there are
  ~5.7 million distinct 5/6/6 relabellings, which is not feasible to enumerate when the statistic
  re-touches cell-level data on every draw.
- Any cell-level statistic can be supplied (difference in pooled mean log-amplitude,
  difference in the 90th percentile, a KS distance between pooled ECDFs); the tail/percentile
  contrast is the more direct test of a bursting hypothesis, since bursting should fatten the
  right tail of the amplitude distribution rather than merely shift its mean.
- Every primary conclusion is triangulated three ways before being reported: the mixed-model
  estimate, the 17-mouse-level plot (`mouse_level_trace_amplitude`, a display/sanity-check
  aggregation, never the model input), and the permutation-test CI. If a conclusion depends on
  the model and is invisible at the level of the 17 animals, it should not be claimed.

### Freezing/locomotion is deliberately NOT covaried in the primary model

- Freezing during TFC acquisition emerges predominantly in the 4th-5th tone-shock pairings, not
  earlier trials. Restricting or covarying by an immobility mask would therefore confound
  treatment group with trial number (early vs. late trials), not isolate a "movement-corrected"
  neural signal. A separate whole-session locomotion comparison (navigation-aware single-cell
  analysis) found no broad between-group difference in total movement during conditioning.
- The group x trial interaction (secondary family, above) is the intended way this analysis
  probes the freezing confound: it asks whether the group difference changes across trials as
  freezing develops, without requiring a movement-state restriction that would itself introduce
  the very trial-number confound being avoided.
- Movement-conditioned/standardized rate analyses may still be run as an explicit sensitivity
  analysis (not the primary/co-primary estimand) — see Sensitivity/exploratory below.

## Detection bias (activity-dependent ROI inclusion)

- Cells that hM4D silences completely are never segmented by CNMF-E, so they are absent from
  every denominator here, not zeros in it — biasing the silenced group's rate estimate upward
  and making a null uninterpretable. Keeping zero-event cells (above) is correct conditional on
  the ROI universe actually detected, but cannot recover cells that never entered it. This is not
  fixed by requiring detection across MORE sessions (cross-registration) — that selects more
  strongly toward persistently active cells, not less.
- **Measured, not just acknowledged.** The conditioning-day protocol records `LT1` (no CNO) before
  `LT2` (CNO) in the same FOV, minutes apart, so `compute_lt1_lt2_dropout` measures the fraction
  of a mouse's LT1-detected cells that fail to register into LT2, by group. Because
  drift/registration failure over that short a gap should be small and roughly group-independent,
  an elevated dropout in hM4D relative to mCherry is evidence of CNO-induced (rather than purely
  technical) cell loss — quantifying the bias rather than only gesturing at it.
- ROI count per FOV/session should be reported as a QC/secondary outcome, not as a rate
  estimator (segmentation yield also varies with optics, focus, motion, and expression).

## Manipulation check (no confirmatory alpha spent)

- `compute_lt1_lt2_amplitude_delta` / `fit_lt1_lt2_manipulation_check`: within-cell delta
  log(mean event amplitude), LT2 minus LT1, on the cross-registered `LT1+LT2` cell set. Same day,
  same task, same FOV, so each cell is its own control. **LT1 always precedes LT2** — session
  order, elapsed time, habituation, and photobleaching differ between them too, not only CNO's
  presence — so the raw LT2-minus-LT1 delta alone does not isolate a CNO effect. The group
  CONTRAST on that delta (a difference-in-differences against mCherry, which experiences the same
  order/time/bleaching confounds) is what does: it removes between-animal baseline variance AND
  the shared session-order confound. This is the best-powered comparison available in this
  dataset, and establishes that the DREADD tool works; it is NOT a test of the memory hypothesis,
  so it needs no multiplicity correction with the confirmatory family above.
- Cells with zero events in EITHER session are excluded from the paired delta (undefined for
  them) — a cell with events in LT1 and none in LT2 is exactly the dropout phenomenon
  `compute_lt1_lt2_dropout` measures, not something the paired amplitude check can express.

## Cell-set selection

- **Primary cell set is `full`** (all cells CNMF-E detected in the session in question, no
  cross-registration) — the least selective option available, since every cross-registered
  subset is strictly more selective (requires detection in additional sessions).
- The five cross-registration subsets historically used by `sp_rates` are **sensitivity
  analyses, not independent replications** — they are nested/overlapping views of the same 17
  animals' cells, not five separate biological samples. Present as a forest plot of effect
  estimates and confidence intervals across cell-set definitions, with no per-row significance
  stars; the question is whether the estimated treatment effect is robust to plausible
  cell-selection choices, not whether every overlapping subset individually crosses p < 0.05.

## Figure convention: unit of inference is the mouse, unit of display may be the cell

**The single invariant: every statistic on every panel is computed from per-mouse values.** The
17 animals (n = 5 hM3D / 6 hM4D / 6 mCherry) are the unit of inference throughout. What varies
between panels is only what gets *drawn*.

- **Mouse-level violin + scatter** (`_mouse_values_per_group` / `_draw_mouse_violin_panel`, over
  `caban.analysis._draw_violin_triplet`): one point per animal. Used where the quantity only
  exists per mouse — *fraction active* and *LT1→LT2 dropout* are proportions computed **over** a
  mouse's cells, and a per-mouse 90th percentile is already a one-number-per-animal summary — and
  for the run-structure small multiples, whose full per-run distribution is written to
  `stats/run_structure.txt` instead.
- **Cell-level SuperPlot** (Lord et al. 2020, *J Cell Biol* 219:e202001064;
  `caban.single_unit_common.draw_superplot_triplet` via `_draw_cell_superplot_panel`): every cell
  plotted semi-transparently and colour-coded by mouse (a within-group lightness ramp), each
  mouse's mean overlaid as a large black-edged marker, cells sharing their mouse's x-offset so a
  mouse's cloud and its mean group together. Used for the genuine per-cell quantities in the
  decomposition figure. **The cell cloud is descriptive only and is never passed to the
  statistics.**
  - An earlier attempt at this was reverted, on the grounds that these long-right-tailed
    quantities crushed all 17 mouse means into the bottom ~6–13% of each panel. That was an
    **axis-scale bug, not an argument against showing cells**: `yscale='auto'`
    (`_resolve_superplot_yscale`) now puts strictly-positive data on a log axis and non-negative
    data containing exact zeros on symlog, with `linthresh` set to the smallest positive value so
    zero-event cells stay visible at the floor rather than being dropped.
  - A column that is **already log-transformed** (`log_amplitude`) is drawn on a *linear* axis:
    `'auto'` would see all-positive values and apply a second log, plotting log10(ln(x)).
- **Sub-quantum vertical jitter** on the rate panels (`y_quantum = 1/exposure_seconds`). A
  per-cell rate is a small integer count over a fixed window, so its values land on a handful of
  hard horizontal lines. Each cell is displaced uniformly within ±0.4 quanta — strictly inside
  its own quantization bin, so no point can be mistaken for a neighbouring count level — and
  exact zeros are pinned at exactly zero, since on a symlog axis zero is the meaningful floor
  occupied by silent cells. Display only: means, axis choice and statistics use untouched values.

### Panel annotation: significance brackets on the figure, estimates in a companion file

**Every panel — linear, log and symlog alike — carries Holm-corrected significance brackets**
(`caban.single_unit_common.annotate_pairwise_brackets`), the established CLAUDE.md
bracket-and-stars style. The p-values are `do_pairwise_holm_plot`'s, computed from the per-mouse
values; only the drawing is new.

This required fixing the reason brackets were previously impossible on a log axis.
`caban.utilities.barplot_annotate_brackets` positions brackets in DATA coordinates and scales its
offsets as fractions of `ylim[1] - ylim[0]`, which is meaningless once the axis is non-linear (on
a 0.01-to-300 log axis that range is ~300, so brackets land off the panel and the stacking
increments collapse). `annotate_pairwise_brackets` instead uses matplotlib's blended
`get_xaxis_transform()` — x in data coordinates, y in **axes fractions** — and reserves headroom
with `reserve_top_fraction`, which does its arithmetic in the axis' own scaled space so
"the data occupies the bottom 80%" means the same thing on a log axis as on a linear one.

An intermediate version put an **effect estimate + 95% interval text block on every panel**
instead. It made the row's annotation uniform, but was worse to read: five blocks of small type
crowding the clouds, and a reader had to parse "[1.14, 2.05]" to notice an effect that one
asterisk conveys instantly. Those estimates are still wanted — they are what makes a null
interpretable — but they belong beside the figure, not on it.

**`decomposition_contrasts.md`**, written next to the figure
(`write_decomposition_contrasts_markdown`), therefore carries, per panel, the equal-mouse-weighted
contrast vs control as a **ratio with a 95% interval alongside the absolute difference** on the
measured scale (`mouse_contrast_ci`; Welch two-sample intervals, Welch–Satterthwaite denominator
df). Two properties it exists to guarantee:

1. **Nulls are self-documenting.** At n = 5/6/6 a non-significant result is only interpretable
   together with what its interval still admits — an interval reaching 1.55 has not excluded a
   +55% effect. This is the analysis' substitute for a formal power simulation, applied to every
   null it reports.
2. **Ratios never appear without absolute differences.** A fold-change off a small base
   overstates the practical size of a change; both numbers are always emitted together.

The file states explicitly that **its intervals are not multiplicity-corrected while the figure's
asterisks are**, so a contrast whose interval excludes 1.0 may still carry no star. Both are
reported deliberately: the interval describes the effect, the star describes the corrected
decision.

The primary/co-primary contrasts additionally get a multiplicative-scale forest plot
(`plot_effect_forest`) giving the fold-change + 95% CI directly (e.g. "1.16x [1.14, 1.19]").

### Pairwise correction family on the panels

Panels in this module Holm-correct across the **two control contrasts only** (hM3D-vs-mCherry,
hM4D-vs-mCherry), reporting hM3D-vs-hM4D uncorrected alongside — `PANEL_HOLM_FAMILY` in
`caban.sp_rates_lmm`, via `do_pairwise_holm_plot(holm_family='vs_control')`. The design question
is whether each DREADD differs from its own control; hM3D-vs-hM4D spends no alpha of its own, and
including it costs power without protecting any error rate at risk. Correcting over all three was
conservative rather than wrong, but at this n conservatism is not free. `do_pairwise_holm_plot`'s
own default remains `'all'`, because it is shared with `freezing_tuned_cells` and the other
single-unit suites whose published figures report p-values corrected over three.

### Other conventions

- The cell-level spread is also shown by the pooled per-cell **ECDF** (`amplitude_ecdf`, with
  per-mouse curves at raised prominence — the 17 mice, not thousands of pooled cells, are the
  unit of inference), which is the panel most sensitive to the fattened right tail a bursting
  interpretation predicts.
- Regression tests (`scratchpad/test_event_amplitude_integration.py`,
  `test_new_items.py`) assert that a SuperPlot's annotated p-values equal a standalone
  mouse-level `do_pairwise_holm_plot` on the same data, that `annotate='ci'` contrasts are built
  from the mouse means, that `_mouse_values_per_group` yields exactly one point per animal, and
  that the sub-quantum jitter leaves annotated p-values bit-identical. These are the ways this
  figure family could silently reintroduce pseudoreplication.
- Display order is `mCherry, hM3D, hM4D` (CLAUDE.md's DREADD convention, control first;
  `caban.single_unit_common.DREADD_DISPLAY_ORDER`), consistently across every panel of every
  figure. This differs from the `hM3D, hM4D, mCherry` order used for model dummy-coding
  throughout this module (`GROUP_ORDER`) — display order only, never affecting which values are
  compared (the same split `caban.place_cell_rates` and `caban.speed_tuning` already use).

## Example-trace panels (raw-data evidence)

- Summary statistics alone do not show a reader what the effect *is*. `plot_example_traces`
  (`caban.sp_rates_lmm`) shows representative cells' deconvolved S and denoised C traces over the
  trace-epoch window (+/- 2 s context), with every supra-threshold run shaded and its per-event
  integral annotated.
- **Selection is reproducible, never hand-picked**: `select_example_cells` chooses, per group, the
  cell closest to each of the 10th/50th/90th percentile of that group's OWN trace-epoch
  log-amplitude distribution (deterministic, no RNG) — a low/median/high triplet so the reader
  sees the shape of the distribution, not one flattering example. The plotted trial for each
  selected cell is the one (within the trace epoch) with the MOST detected runs, ties broken by
  lowest trial index — again deterministic, not hand-chosen.
- Shading is drawn from `find_event_runs_ca()` called directly on the plotted cell's own S row at
  the same threshold used to build the amplitude tables — the shaded span IS what was measured,
  not a separately re-derived approximation (verified: a hand-built run round-trips through
  `find_event_runs_ca` with exact `start`/`width`, and the plotting code's `axvspan` reuses those
  same returned arrays with no independent boundary computation).
- `plot_width_vs_height_matched_examples` makes the run-width story (above) visually concrete: one
  run per group, each matched to the SAME target peak height (pooled 50th percentile by default),
  so that if hM3D's height-matched run is still visibly WIDER, the reader sees directly that the
  effect is not merely "taller events."

## Deferred (not implemented this pass)

The following are explicitly OUT OF SCOPE for the current confirmatory-family build-out and are
listed here so this document does not silently promise analyses that do not exist:

- **LT1->LT2 dropout DECOMPOSITION**: what "dropout" comprises — cells missed by one session's
  independent CNMF-E run, CellReg confidence thresholding, footprint-match failure, and
  QC-exclusion — each as its own measured quantity, reported alongside per-FOV ROI counts. Until
  this is broken down, retention is only ~18-23% (hM3D 81.7% dropout, hM4D 77.1%, mCherry 76.8%),
  which is itself a larger finding than any group difference in it: the hM4D detection-bias
  hypothesis is NOT supported (hM4D dropout is indistinguishable from mCherry, a real negative
  result that makes the hM4D amplitude null interpretable) — but it also means the LT1->LT2
  manipulation check describes a stable, high-SNR minority of cells, not the population, until
  this decomposition explains why ~4 in 5 cells fail to cross-register between two same-day
  sessions minutes apart.
- **Cross-registration-subset sensitivity forest plot**: the five historical `sp_rates`
  cross-registration subsets, as overlapping (not independent) views of the same 17 animals,
  presented as effect estimates + CI without per-row significance stars (see "Cell-set selection"
  above for why).
- **Movement-conditioned/standardized analyses**: an explicit sensitivity analysis restricting or
  standardizing by locomotion state, separate from the primary/co-primary estimand (which
  deliberately does not covary by movement — see above).
- **ROI-per-FOV QC panel**: segmentation yield per session as a QC/secondary outcome, not folded
  into any rate estimator.
- **Reference-seeded extraction**: an alternate CNMF-E extraction pass seeded from a
  cross-session-stable reference, as a robustness check on cell detection itself.
- **Count-model posterior-predictive / zero diagnostics**: posterior-predictive checks and a
  zero-inflation diagnostic for the secondary Negative-Binomial rate model.
- **Peri-shock analysis**: the shock window itself (excluded from the confirmatory family — see
  "Epoch structure" above) via YrA/C rather than the event-based approach used elsewhere in this
  module.
*(The BH-FDR family that used to be listed here as unimplemented is now implemented — see
"Prespecified secondary family (BH-FDR)" above.)*

## Outputs

- `PLOTS_DIR/sp_rates_lmm/TFC_cond/`
  - `primary_trace_amplitude.png/.svg` — 17 mouse-level points, group violins, Holm-corrected
    pairwise brackets (visual triangulation; the confirmatory omnibus estimate lives in the
    companion stats file).
  - `primary_effect_forest.png/.svg`, `coprimary_effect_forest.png/.svg` — fold-change + 95% CI
    on the multiplicative scale for the primary and co-primary group contrasts.
  - `epoch_profile.png/.svg` — descriptive per-mouse pooled mean log-amplitude by epoch and
    group, with faint per-animal trajectories.
  - `epoch_delta_forest.png/.svg` — within-cell amplitude elevation over each cell's own pre_tone
    baseline, as a fold-change + 95% CI, for EVERY epoch. Only the trace row is confirmatory; the
    rest are the descriptive context that keeps its null from being misread as trace-specific.
  - `amplitude_ecdf.png/.svg` — pooled per-cell ECDF of trace-period log-amplitude by group
    (bold), with per-mouse ECDFs (thin, raised prominence) — the panel most directly sensitive to
    a tail/bursting shift rather than a mean shift.
  - `amplitude_p90.png/.svg` — mouse-level 90th-percentile log-amplitude (17 points), the
    tail-shift summary matching the `_p90` mouse-label permutation test.
  - `decomposition.png/.svg` — "components of population calcium activity": fraction active
    (mouse-level only), rate among active cells, overall event rate across ALL cells (the exact
    product `fraction_active x rate_active`), mean per-event amplitude, and total S/s (the exact
    product `overall_rate x amplitude`) — all five panels together make the decomposition
    identity explicit rather than left for the reader to multiply. Fraction active is a
    mouse-level violin; the other four are cell-level SuperPlots. Overall event rate is drawn
    wider than its siblings, being the summary quantity of the first half of the chain. Every
    panel carries Holm-corrected significance brackets (log axes included), and the two rate
    panels carry sub-quantum jitter — see "Figure convention" above.
  - `decomposition_contrasts.md` — the companion file for the panel above: per-panel
    equal-mouse-weighted ratio + 95% interval + absolute difference vs control, with an explicit
    note that these intervals are uncorrected while the figure's asterisks are Holm-corrected.
  - `run_structure.png/.svg` — run width (frames), local maxima per run, and fraction of
    multi-peak runs, trace epoch, per-mouse means.
  - `threshold_sensitivity.png/.svg` — forest plot of the primary contrast re-fit at
    `thres in {1.5, 2.0, 3.0}`.
  - `manipulation_check.png/.svg` — LT1->LT2 within-cell delta log-amplitude and LT1->LT2
    detection dropout fraction, by group (both per-mouse).
  - `example_traces.png/.svg` — representative S/C traces with shaded runs and annotated
    integrals, low/median/high within-group amplitude percentile triplet per group.
  - `width_height_matched_examples.png/.svg` — one run per group matched on peak height, making
    the width-vs-height distinction visually concrete.
- `PLOTS_DIR/sp_rates_lmm/{Test_B,Test_B_1wk}/post_tone_amplitude.png/.svg` — the same mouse-level
  panel reused for the 20 s post-tone recall window (with a correctly-labelled title, not the
  trace-period default).
- `PLOTS_DIR/sp_rates_lmm/<session_type>/stats/` — companion `.txt`/`.csv` files with the full
  mixed-model / Bambi posterior summaries, Holm/BH-corrected p-values, permutation-test results
  (both mouse- and cell-weighted), the run-structure summary, the threshold-sensitivity table, and
  (for the recall sessions) the recall cohort note, for every panel above. Specifically including:
  - `secondary_fdr_family.csv/.txt` — the declared BH-FDR family: every member, its raw p-value,
    its q-value, and the alpha=0.05 decision, plus the written record of what is excluded and why.
  - `epoch_deltas_descriptive.csv` — the per-epoch within-cell delta table behind
    `epoch_delta_forest`, with the `is_confirmatory` flag.
  - `secondary_rate.txt` — the NB rate model, including sampler convergence diagnostics and the
    observed rate contrasts as ratio *and* absolute difference in events/s.
