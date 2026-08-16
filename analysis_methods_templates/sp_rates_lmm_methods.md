# Cell-level pyramidal event amplitude across DREADD groups (trace fear conditioning)

## Overview

- This analysis replaces the ~205 uncorrected three-group ANOVAs of the earlier `sp_rates`
  analysis (session x behaviour period x cross-registration subset x metric) with a small,
  pre-declared confirmatory family built on **per-event deconvolved amplitude at the cell
  level**. No step in this analysis collapses cells to a per-mouse scalar before testing;
  amplitude/rate/fraction-active are computed per cell and modelled with a mouse random
  intercept, so hundreds of cells per animal inform the estimate without inflating the effective
  n of independent DREADD assignments (still 5 hM3D / 6 hM4D / 6 mCherry mice).
- **The scientific claim under test is that hM3D increases pyramidal BURSTING** (larger per-event
  Ca2+ influx), not merely more frequent events. **Event amplitude is therefore the PRIMARY
  endpoint; event rate is SECONDARY.** This is a deliberate departure from the usual convention
  (and from an external methodological review consulted while designing this analysis, which
  recommended rate as primary) — the physiological claim here is specifically about burst
  magnitude, and rate is reported in every panel alongside amplitude with a full confidence
  interval, never hidden or dropped, so "secondary" means it carries no confirmatory alpha, not
  that it is de-emphasized.

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
  not mixed with legacy `find_spikes_ca()` counts.
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

### Confirmatory family (Holm-corrected across exactly these two tests)

- **Primary endpoint**: `log(mean per-event amplitude) ~ C(group, Treatment('mCherry'))`,
  cell-level, trace epoch pooled (summed, not averaged) across the five trials, random intercept
  on mouse. Falls back to mouse-clustered OLS if the mixed model does not converge or converges
  to a degenerate boundary fit (`caban.single_unit_common.fit_mixed_model`, the same fallback
  logic and degeneracy check used by every mixed-model analysis in this codebase). The primary
  test statistic is the joint Wald test that both non-reference group coefficients are zero
  (`caban.single_unit_common.joint_wald_test`).
- **Co-primary endpoint**: `log(mean per-event amplitude) ~ group * epoch + C(trial)`, cell x
  trial x epoch level (shock excluded), random intercept on mouse. Tests whether the effect is
  epoch-specific rather than a global shift. The co-primary test statistic is the joint Wald test
  that every group x epoch interaction coefficient is zero.
- **Holm correction is applied across exactly these two omnibus p-values** — the entire
  confirmatory multiplicity burden of this analysis (`caban.sp_rates_lmm.holm_correct_confirmatory`).
  Everything else below is secondary (BH-FDR) or purely descriptive/sensitivity.

### Prespecified secondary family (BH-FDR)

- **Event rate**, trace epoch: `n_events ~ group * epoch + trial` with a
  `log(exposure_seconds)` offset, at the mouse x trial x epoch level (summed over cells within
  each mouse-trial-epoch window), fit as a genuine Negative-Binomial mixed model (random
  intercept on mouse) via Bambi (PyMC backend) — `caban.sp_rates_lmm.fit_rate_group_epoch_model`.
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
- **group x trial interaction**, trace epoch amplitude. Directly addresses the freezing confound
  described below: since freezing only emerges in trials 4-5 of conditioning, this interaction
  tests whether groups diverge as freezing develops, without restricting to any particular
  behavioural-state subset of frames.

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
  same task, same FOV — CNO the only difference — so each cell is its own control, and the group
  contrast is a difference-in-differences that removes between-animal baseline variance. This is
  the best-powered comparison available in this dataset, and establishes that the DREADD tool
  works; it is NOT a test of the memory hypothesis, so it needs no multiplicity correction with
  the confirmatory family above.
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

## Outputs

- `PLOTS_DIR/sp_rates_lmm/<session_type>/`
  - `primary_trace_amplitude.png/.svg` — 17 mouse-level points, group violins, Holm-corrected
    pairwise brackets (visual triangulation; the confirmatory omnibus estimate lives in the
    companion stats file).
  - `epoch_profile.png/.svg` — descriptive per-mouse pooled mean log-amplitude by epoch and
    group, with faint per-animal trajectories.
  - `amplitude_ecdf.png/.svg` — pooled per-cell ECDF of trace-period log-amplitude by group
    (bold), with per-mouse ECDFs (thin) — the panel most directly sensitive to a tail/bursting
    shift rather than a mean shift.
  - `decomposition.png/.svg` — total S/s, fraction active, rate among active cells, mean
    per-event amplitude, side by side (total visually dominant; the rest are conditional
    components of it, not four independent activity measures).
  - `manipulation_check.png/.svg` — LT1->LT2 within-cell delta log-amplitude and LT1->LT2
    detection dropout fraction, by group.
- `PLOTS_DIR/sp_rates_lmm/<session_type>/stats/` — companion `.txt` files with the full
  mixed-model / Bambi posterior summaries, Holm/BH-corrected p-values, and permutation-test results for each
  panel above.
