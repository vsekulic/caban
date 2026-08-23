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

### The post-shock response window, and what `post_shock` used to mean

`post_shock` is the **20 s window from shock offset** (`caban.epoch_analysis`,
`TRACE_MATCHED_WINDOW_S`). It previously meant the entire inter-trial interval — shock offset to
the next tone onset, 198 s, and on the final trial to the end of the recording, so neither fixed
nor equal across trials. Every result computed under the old definition averaged the post-shock
response over roughly ten times its own duration and must not be compared with results computed
under the new one. The full ITI is still available, under the name it should always have had:
`iti`.

The redefinition is motivated, not cosmetic. Puhger et al. 2024 (*iScience* 27:109035) report in
dorsal CA1 during trace fear conditioning: no bulk calcium response during the trace interval at
all; a large sustained response to the footshock; that optogenetic silencing 0–40 s after the
shock impairs both tone and context memory; and that the *same* silencing delivered 140 s after
the shock impairs neither. There are therefore two distinct hypotheses in this literature about
where in a TFC trial a hippocampal manipulation should act — the trace interval and the
post-shock window — and an analysis that windows only the former answers only one of them. 20 s
is also exactly Puhger et al.'s own post-shock quantification window (42–62 s after CS onset).

Note that `pre_tone` is itself a *late* post-shock window for trials 2–5: at 35 s before a tone
onset it begins 163 s after the preceding shock offset, past the point at which Puhger et al.
find CA1 activity has returned to baseline and silencing is inert. That supports its use as a
reference, and it means trial 1's baseline (genuinely shock-naive) is not the same quantity as
trials 2–5's.

### The early-versus-late post-shock contrast, and why the late window starts at 90 s

Puhger et al.'s result is a **contrast** between early and delayed silencing, so reproducing it
needs a delayed window, not just an early one. `post_shock` versus `pre_tone` is not that
contrast: each epoch is pooled across trials *before* differencing, so the pooled reference is a
mixture of one shock-naive baseline (trial 1's, which no shock precedes) and four windows 163 s
after a shock, at trial indices that do not align — and trial 5's shock has no `pre_tone`
counterpart at all, the session ending first.

`post_shock_late` (`epoch_analysis.POST_SHOCK_LATE_ONSET_S`) is therefore a second
`TRACE_MATCHED_WINDOW_S`-long window in the same trial, beginning 90 s after shock offset. The
within-cell contrast `log_amplitude[post_shock] − log_amplitude[post_shock_late]` is then
within-trial at aligned indices with no shock-naive window in the reference.

**The late window does not exist on every trial, and both sides are trial-matched.** The final
trial's inter-trial interval ends with the *recording*, not a next tone onset, and its real
length is data-dependent and far shorter than the nominal protocol timing implies — observed
sessions stop as little as **20.5 s** after the last shock offset, which is barely the 20 s
`post_shock` window itself. On such a trial no late window fits at *any* onset value.
`get_epoch_frames` therefore returns `None` for `post_shock_late` there (a definitional absence,
as for `pre_tone` on trial 1, which no shock precedes) while continuing to *raise* for
`post_shock`, which is a locked confirmatory window every trial must supply.

Because the late window is missing on some trials, pooling each epoch across all its available
trials before differencing would compare a late window over *fewer* trials against an early
window over more — reintroducing the very trial-index misalignment this epoch exists to remove,
now confounded with trial position (photobleaching, arousal). Both sides are therefore cut to
the trials where **both** windows exist, per mouse
(`caban.sp_rates_lmm.restrict_to_shared_trials`), before the within-cell delta is taken.
Matching is per-mouse rather than global because recordings are ragged and cutting every mouse
to the globally-shared trials would discard good data from mice whose recordings ran long; each
cell's delta is then trial-matched within itself, which is what the within-cell contrast
requires. The surviving per-mouse trial counts are written to
`stats/descriptive_early_vs_late_trial_coverage.csv` and printed at run time — **this coverage
must be reported alongside the estimate**, since it is what the contrast rests on. The same
matching is applied to every row of the descriptive epoch-delta forest, where it is a no-op for
the epochs present on all trials.

**Why 90 s rather than Puhger et al.'s 140 s.** The choice maximizes trial coverage; it does not
rescue the truncated final trial, which no onset value can. What the constant controls is the
*mid*-session trials, where a shorter onset can only ever admit more trials than a longer one
(the 90 s window fits inside every ITI the 140 s window fits in). 90 s still places the window
far outside the 0–20 s sustained response, and 90–110 s remains disjoint from both `post_shock`
(0–20 s) and `pre_tone` (163–198 s) on any full-length ITI. Raising it toward 140 s buys
fidelity to Puhger's protocol at the cost of trials — consult the coverage file before doing
so.

**This contrast is descriptive.** Its role is to characterize the post-shock null, not to test a
hypothesis: it spends no alpha, is in neither the confirmatory Holm family nor the secondary
BH-FDR family, and is reported as an estimate with a 95% interval rather than a significance
verdict. The confirmatory family remains exactly the three tests listed below. Note also that
comparing the `post_shock` and `post_shock_late` decomposition figures by eye is the
difference-of-significance fallacy — the early-versus-late claim is carried by the within-cell
delta, not by two independently fit figures.

### Exposure matching, and why there are two baselines

`pre_tone_matched` is the same baseline as `pre_tone`, ending at the same tone onset, but 20 s
long instead of 35 s. Both exist because window length matters for some endpoints and not
others:

- **Duration-sensitive**: event rate and the fraction of cells with ≥1 event. At a fixed
  underlying rate both grow with the window, so comparing them between a 35 s baseline and a
  20 s response window would confound epoch with exposure. These use `pre_tone_matched`, which
  is exposure-matched to `trace`, `post_shock` and `post_shock_late` (`TFC_MATCHED_EPOCHS`).
- **Duration-insensitive**: mean per-event amplitude, a per-event quantity. The confirmatory
  amplitude contrasts therefore keep the 35 s `pre_tone` reference they were locked with.

Neither baseline is the "right" one in general; using the wrong one for a given endpoint is what
would be wrong.

**The two baselines overlap in time and must never be modelled together.** `pre_tone_matched` is
the last 20 s of the same window as `pre_tone`, so any model carrying both on one epoch factor
counts 20 s of every baseline twice — inflating the baseline's apparent precision, corrupting any
epochs-within-trial variance component, and (for the NB-GLMM) leaving the sampler on a
near-collinear ridge. Anything with an epoch factor therefore draws its epochs from
`TFC_DISJOINT_EPOCHS` (`pre_tone`, `tone`, `trace`, `post_shock`, `post_shock_late`), not from
`TFC_EPOCHS`. Disjointness is necessary but not sufficient: the secondary **rate** model draws
from the narrower `TFC_RATE_MODEL_EPOCHS` (the same four epochs it has always used), holding out
`post_shock_late` because that window is descriptive, plays no part in the rate endpoint, and is
absent on final trials in a model that carries `trial` as a covariate — so its epoch dummy would
be correlated with `trial`. Empirically the fifth level also made NUTS pathological (~27 s to
sample at four epochs versus >280 s without finishing at five, even with the fifth level
artificially balanced), which is the plain cost of two more weakly-identified interaction
parameters and *not* the duplicated-data ridge described above — all five windows are verified
pairwise disjoint, across trial indices too. Holding the rate model at four epochs also means its
reported results are the same model that produced them before `post_shock_late` existed. The
secondary rate table enforces this at its own source rather than trusting callers. The
decomposition figures are the exception that motivates keeping both: they are fit one epoch at a
time, never jointly.

- TFC_cond epochs used by the confirmatory model: `pre_tone` (35 s baseline window, a chosen
  default, not protocol-derived), `pre_tone_matched` (20 s baseline, see below), `tone` (20 s),
  `trace` (`tone_offsets[i]` to `shock_onsets[i]`), `post_shock` (20 s from shock offset),
  `post_shock_late` (20 s, beginning 90 s after shock offset; descriptive only, see above).
  **Shock (2 s) is excluded**: at 0.05-0.2 Hz a 2 s window
  yields ~0-1 events per cell, dominated by quantization, and the window carries motion artifact.
  Handled separately elsewhere via YrA/C, not here.
- **Trace duration is NOT constant across the five trials.** From `TraceFearCondSession`'s own
  timing arrays (`tone_onsets_def = [185, 420, 660, 900, 1140]`, `tone_duration = 20`,
  `shock_onsets_def = [220, 460, 700, 940, 1180]`): trace durations are `[15, 20, 20, 20, 20]`
  seconds. Trial 1 is short because `tone_onsets_def[0] = 185` rather than an intended 180 s
  (flagged as an accident in the source comment — at 180 the first trace would also be 20 s).
  Every window's exposure (`exposure_seconds`) is computed from that trial's ACTUAL onset/offset
  frames, not an assumed constant duration, so the unequal first trial is correctly absorbed
  **wherever an exposure offset is used** — the rate model, and any per-event quantity such as
  amplitude.
  **It is not absorbed where there is no offset to absorb it.** The fraction of cells with >=1
  event is a bounded proportion with no exposure term, and rate-among-active conditions on
  `N > 0`, which is itself duration-dependent. Since epochs are pooled across trials by summing
  exposure, trace pools to ~95 s against `pre_tone_matched`'s ~100 s, and that 5% mismatch lands
  directly on exactly those two components. The cross-epoch decomposition and the joint epoch
  test therefore use `restrict_to_exposure_matched_trials`, which keeps only the
  `(mouse, trial)` pairs where every requested window is present at its full 20 s. The
  restriction is derived from the MEASURED `exposure_seconds`, never from a trial index —
  hard-coding "trials 2-5" would encode nominal protocol timing, which is data-dependent (trial
  counts vary per mouse and recordings can stop as little as 20.5 s after the last shock). Per-mouse
  trial coverage is written to `stats/matched_decomposition_trial_coverage.csv` and is reported
  with the estimates.
- Test_B / Test_B_1wk post-tone window is pinned to **20 s** explicitly (via
  `get_testb_epoch_frames(..., post_tone_duration_s=20.0)`), matching the representative TFC
  trace duration (4 of 5 trials). Neither of the two existing alternatives is the correct recall
  analog of trace: `get_testb_epoch_frames`'s own default is 35 s, and the sp_rates-native
  `TestBSession.post_tone_onsets/offsets` extends all the way to the next tone onset
  (~200+ s inter-trial interval, not a matched window at all).

## Statistics

### The paper-facing analysis: one unified mouse-level model per outcome

**Everything the manuscript reports comes from here. Everything in the rest of this section is
internal and sensitivity output.**

Per-event amplitude and population event rate are presented as parallel outcomes of one two-row
figure, so they receive one identical statistical treatment rather than two different ones:

1. **One mouse-level value per epoch, for each outcome**, over the three exposure-matched 20 s
   windows (`pre_tone_matched`, `trace`, `post_shock`) and the same retained (mouse, trial) pairs
   for both — 17 mice × 3 epochs = 51 rows
   (`caban.sp_rates_lmm.build_mouse_epoch_unified_table`).
   - *Amplitude*: each active cell's mean event-run integral over the retained trials → natural
     log → arithmetic mean of those cell-level logs within the mouse. Cells with no event in that
     epoch are excluded (amplitude is undefined for them). **exp(a group contrast) is therefore a
     ratio of GEOMETRIC means** of the cell-level mean event amplitudes, not of pooled arithmetic
     means.
   - *Population rate*: total events over all detected cells (zero-event cells included) divided
     by total cell-seconds, then logged. A zero rate raises rather than receiving a pseudocount.
2. **The same model for each**: `log(metric) ~ group * epoch + (1|mouse)`, reference levels
   `mCherry` and `pre_tone_matched` set explicitly as ordered categoricals, fit by
   `caban.single_unit_common.fit_mixed_model` (statsmodels MixedLM, REML). If EITHER outcome
   fails to fit as a mixed model, the paper analysis hard-fails
   (`require_common_unified_method`) rather than letting one outcome take the clustered-OLS
   fallback while the other does not — the claim that the two received the same treatment would
   otherwise be false.
3. **One joint Wald test of the four group × epoch coefficients per outcome**
   (`joint_wald_test`, `df2 = n_mice - 1 = 16`). This is the paper's only test of epoch
   dependence.
4. **Six planned treatment-vs-control simple effects per outcome** (3 epochs × {hM3D, hM4D} vs
   mCherry), each a linear contrast of the fitted model (`linear_contrast_test`, same `df = 16`):
   the group coefficient at the reference epoch, the group coefficient plus the corresponding
   interaction coefficient elsewhere. A simple effect is a contrast, not a coefficient, and cannot
   be read off the model summary.
5. **A Holm family of two contrasts WITHIN each epoch** (`p_holm_epoch`): for every
   `(outcome, epoch)`, the hM3D-vs-mCherry and hM4D-vs-mCherry comparisons are corrected
   together. That is six two-member families — 3 epochs × 2 outcomes — and the procedure is
   identical in all of them, so every visually equivalent panel of the paper figure is treated
   equivalently. hM3D-vs-hM4D is in no family.
6. **The six-comparison across-epoch Holm correction is retained as SENSITIVITY** (`p_holm_six`):
   all six simple effects corrected together within each outcome, computed from the same twelve
   raw contrasts. It annotates nothing and supports no Results claim.

`stats/unified_lmm_posthoc_contrasts.csv` is the single authoritative source for every estimate,
interval, P-value and asterisk on a paper-facing figure; `stats/unified_lmm_mouse_epoch_values.csv`
is the inferential dataset and is what the figures' large mouse markers plot (asserted at draw
time). `stats/unified_lmm_interactions.csv` carries the two interaction tests.

Descriptive companions reported beside the model output, never in place of it: the observed
equal-mouse-weighted group means, and for rate their absolute difference in events/s/cell (a
fold-change off a small base overstates the practical size of a change).

**Wording.** A non-significant interaction means there was *no evidence that the treatment effect
differed across* the sampled epochs — not that the effect is identical, global, tonic, or
equivalent across them. A significant treatment-control comparison within one epoch does not imply
that the treatment effect differs from another epoch; epoch dependence is tested by the
interaction. This analysis was finalised after substantial inspection of the dataset and is
described as the **main statistical analysis**, never as prospectively confirmatory.

Diagnostics (`stats/unified_lmm_diagnostics.*`, `unified_lmm_residuals.csv`,
`unified_lmm_influence.csv`): residual-vs-fitted and normal Q-Q plots per outcome, and
leave-one-mouse-out refits reporting each animal's effect on the twelve contrast ESTIMATES. These
are descriptive. No residual-normality test is used as an acceptance criterion (a mixed model's
residuals are not 51 independent observations), and significance-decision flips are deliberately
not computed (at n = 17 a threshold crossing near α is expected and says nothing about
robustness). `stats/unified_lmm_synthetic_verification.txt` records two planted-effect designs
run through the same code: an equal shift in every epoch must be detected while the interaction
stays null, and a trace-only shift must make the interaction significant.

---

### Internal architecture below this line

**The tier system, multiplicity families and models described from here on are SENSITIVITY and
historical infrastructure.** They all still run and still write their files, and they are what a
reviewer asking "does this survive a different modelling choice" should be shown. None of them
supplies a number to the manuscript or to a paper figure. The tier vocabulary below applies
within that internal output only.

### Evidential tiers: what may be called "significant" (internal output only)

Every internal output of this analysis sits in exactly one of three tiers. **The tier, not the
p-value, determines how a result may be written up.**

| tier | members | how to report |
|---|---|---|
| **Confirmatory** | the three Holm-corrected tests below | "significant", quoting `p_holm` |
| **Secondary** | the declared BH-FDR family | "significant" with its `q`, described as secondary |
| **Exploratory** | everything else — the direct DREADD-vs-DREADD contrasts, descriptive epoch deltas, the early-vs-late contrast, threshold sensitivity, cross-registration subsets | **effect estimate + 95% interval only.** Never "significant" |

A tier-3 result is not weak evidence of the same kind as a tier-1 result; it is a different kind
of statement. Four properties recur and none is repaired by computing a better p-value:

- **Post-hoc selection.** A contrast chosen after inspecting a figure cannot re-enter this
  internal confirmatory family. That family was a locked reanalysis, not a prospective
  preregistration, which makes the boundary load-bearing rather than pedantic.
- **No multiplicity protection.** Tier-3 output is deliberately in no family, so nothing controls
  an error rate across it.
- **The contrast may not be the design's question.** A DREADD-vs-DREADD difference contains no
  control group; it licenses "A differs from B", never "A raised X". If neither group differs from
  control, that distinction is the whole result.
- **Sensitivity to analysis choices.** Where an estimate moves with a defensible change of subset
  or window, that variation belongs in the report alongside the estimate.

Prefer folding a tier-3 observation into a claim that a tier-1 or tier-2 result already supports
("consistent with"), which spends no alpha. If it must be a finding in its own right, declare it
prospectively for a future cohort.

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

### Internal confirmatory family (Holm-corrected across exactly these three tests)

*Historical: this was the paper's primary analysis before the unified mouse-level models above.
It is retained as a sensitivity check — it uses a different unit of aggregation (cell rather than
mouse), a different baseline window (35 s `pre_tone` rather than the 20 s matched one), and a
different multiplicity family. It supplies no manuscript number.*

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
- **Co-primary endpoint (post-shock window)**: the identical within-cell delta contrast with
  `post_shock` in place of `trace` — `delta_log_amplitude = log_amplitude[post_shock] -
  log_amplitude[pre_tone]`, cells active in both. Fit, written and plotted through the *same*
  code path as the trace delta (`caban.sp_rates_lmm.fit_and_report_epoch_delta`), so any
  difference between the two is a difference in the data rather than in the analysis.

  **This member is prespecifiable in the honest sense.** It was added *before* any
  properly-windowed post-shock result existed: under the previous epoch definition `post_shock`
  meant the 198 s ITI, so no post-shock response-window contrast had ever been fit or inspected.
  The dated analysis plan must be locked before this pipeline is next run.

- **Holm correction is applied across exactly these three omnibus p-values** — the entire
  confirmatory multiplicity burden of this analysis (`caban.sp_rates_lmm.holm_correct_confirmatory`).
  Everything else below is secondary (BH-FDR) or purely descriptive/sensitivity.

  Why three rather than two: the absolute trace-amplitude test and the two delta tests answer
  genuinely different questions. A manipulation that shifts amplitude uniformly across the whole
  session is biologically real and produces *no* delta at any epoch, so a family containing only
  the deltas would define that result out of existence — which is exactly the result this
  dataset currently shows. The cost of the third member is that Holm's smallest threshold moves
  from α/2 to α/3.

### The co-primary null, and the complete epoch profile (descriptive)

**The co-primary within-cell trace-vs-pre_tone delta is NULL.** (This describes the *trace*
co-primary only. The post-shock co-primary is new and its result is not yet known; nothing in
this section may be read as anticipating it.) The hM3D amplitude effect is a
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
  tone−pre_tone and post_shock_late−pre_tone alongside the confirmatory trace−pre_tone and
  post_shock−pre_tone. A reader can then see the per-epoch profile of estimates directly rather
  than taking it on trust, and cannot mistake "no evidence of epoch dependence" for "no effect".
- The early-versus-late post-shock delta (post_shock−post_shock_late) is reported separately from
  this forest, since its reference is not `pre_tone`. It goes through the same code path as the
  confirmatory deltas (`fit_and_report_epoch_delta`, with `is_confirmatory=False`), and its
  output is named `descriptive_*` rather than `coprimary_*` so its status is legible from the
  filename alone.

The non-confirmatory epochs are **descriptive**. They spend no alpha, are not in the confirmatory
Holm family (which is exactly three tests), and are not in the secondary BH-FDR family either —
their role is to characterize a null, not to test a hypothesis. The trace **and post_shock** rows
are flagged `is_confirmatory` in the output table; `pre_tone_matched` is excluded from the table
entirely, since its delta against `pre_tone` is a window-length artifact rather than an epoch
effect.

### The joint group x epoch specificity test (secondary)

Everything described immediately above compares epochs ONE AT A TIME. None of it tests whether
the group effect CHANGES across epochs, and **reading the confirmatory p-values against each
other — significant in trace, not significant in the epoch deltas — does not test it either.**
That is the difference-of-significance fallacy: a difference between two p-values is not itself
a test. Epoch specificity gets exactly one test here
(`caban.sp_rates_lmm.fit_and_report_epoch_interaction`), reported once.

**One test per decomposition component.** The test below is run separately for fraction active,
rate among active cells, population event rate, and per-event amplitude — the same four
components the decomposition grid plots, from the same shared definitions, so each grid row is
annotated by a test computed from exactly the frame and column that row is drawn from. A single
pooled test across all four would answer "did ANY component's profile change", which is not a
question anyone asks, and would be driven by whichever component has the largest between-mouse
spread.

**The four resulting p-values are strongly DEPENDENT** — the components are one exact identity
(overall_rate = fraction_active x rate_active, plus amplitude) computed over overlapping cells.
Benjamini-Hochberg remains valid under positive dependence, so the q-values stand, but they
describe ONE decomposition and must not be counted as four independent findings.

**Profile.** For each mouse, the mean WITHIN-CELL elevation of the component over the
exposure-matched baseline in each response window: `delta_trace` and `delta_post_shock` against
`pre_tone_matched`, over the exposure-matched trials only (see "Exposure matching" above). This
is a 17 x 2 matrix, one complete profile per animal. Differencing against the reference epoch
removes the epoch MAIN effect by construction, so what remains is only the interaction — a mouse
whose component is uniformly elevated contributes a flat profile of zeros regardless of how
elevated it is.

**Fraction active is per-mouse by nature and has no paired variant.** It is a proportion computed
OVER a mouse's cells, so there is no per-cell value to pair across epochs; its paired and
unpaired tests are the same test, not two. The other three components are genuine per-cell
quantities and carry both.

**The delta is taken on each component's own scale** — a log-ratio for amplitude (whose column is
already logged), a raw difference in events/s for the two rates, a difference in proportion for
fraction active. Those are not commensurable with one another, which does not matter: the
statistic standardizes each epoch's delta by its own pooled between-mouse SD, so it is scale-free
and each component's test is computed in the units its own contrasts are reported in. Note that
only the amplitude DELTA is a log-difference and therefore has a meaningful fold-change on
exponentiation; the other deltas are differences of possibly-negative quantities and are reported
as differences only.

**Statistic.** The standardized sum of squared difference-of-differences,

```
T = sum over non-reference groups g, over response epochs e of
        ( ( mean_g[delta_e] - mean_control[delta_e] ) / sd_pooled(delta_e) ) ** 2
```

Each term is exactly the difference-of-differences reported beside it with a 95% interval, e.g.
`[hM3D - Ctl]_trace - [hM3D - Ctl]_pre_tone_matched`. Standardizing by the per-epoch pooled
between-mouse SD is required, not cosmetic: without it whichever delta has the larger scale
dominates the sum and the omnibus silently becomes a test about that one epoch. The SD is
recomputed under every permutation, so it is a function of the data being permuted.

**Inference.** Mouse-label permutation (see "Small-n inference" below): the 17 group labels are
shuffled while each mouse keeps its ENTIRE epoch profile, preserving both the 5/6/6 group sizes
and the within-mouse covariance between epochs. Because `T` is non-negative, the two-sided
`|null| >= |observed|` rule is an upper-tail omnibus test. The permutation distribution is what
carries validity, so the choice of statistic affects POWER, not correctness; this one is used
because each of its terms is separately reportable.

An epoch with zero between-mouse variance contributes 0 to `T`, never a NaN, and the permutation
test computes its p-value over the finite draws only, raising rather than reporting a number if
the observed statistic is non-finite or if more than half the draws are. This matters because a
NaN in a permutation null is not a neutral value: `abs(nan) >= abs(observed)` is False, so NaN
draws would count as "not extreme" and push the p-value DOWN. On a deliberately degenerate
fixture (a component with no between-mouse variation at all) the unguarded version returned
p <= 0.05 on 87% of true nulls.

**Two variants are run and both reported.** The primary is PAIRED — the delta is computed within
cell, so a cell must be amplitude-active in all three epochs (a stricter intersection than any
pairwise one; `n_cells` is reported). The sensitivity variant is UNPAIRED — each epoch's per-mouse
mean is taken over that epoch's own active cells and the means differenced afterwards.
Permutation validity never depended on the cell pairing, so a disagreement between the two is
evidence that the triple intersection selects a special subpopulation, not that either is invalid.

**Status: secondary BH-FDR family**, not confirmatory. The confirmatory Holm family stays at
exactly three members. It is NOT re-formed around this test, even though doing so would be
conceptually tidier, because its two epoch members are already known to be null and dropping them
now would move the primary `p_holm` from 0.027 back toward 0.009 — an outcome-driven change to a
locked family, which this analysis does not make.

**Interpreting a null.** A null here does not mean the manipulation has no effect; the primary
trace amplitude contrast is separately significant. It means the data do not support the stronger
claim that the effect is specific to any one epoch — i.e. amplitude is elevated broadly across
TFC epochs. Report the difference-of-differences intervals for what magnitude of epoch
specificity remains admissible; at n = 5/6/6 that range is wide.

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
  fraction; each DREADD group vs control),
- the **joint group x epoch specificity permutation tests**, one per decomposition component
  (`epoch_specificity_{fraction_active,rate_active,population_rate,amplitude}`; see its section
  above, including the note that these four are strongly dependent and describe one
  decomposition).

Excluded, by declaration rather than omission:

- **The three confirmatory omnibus tests.** They carry their own Holm correction over a family of
  exactly three (`holm_correct_confirmatory`) and control a different error rate. A test placed in
  two families is corrected twice.
- **The Bambi Negative-Binomial rate model.** It reports posterior contrasts, HDIs and an
  ELPD-LOO comparison; there is no p-value to correct. Manufacturing one so a Bayesian result can
  be folded into a frequentist FDR family is a category error, not a conservative choice.
  - **This is a statement about THIS model, not about the rate endpoint.** The paper-facing rate
    endpoint is the unified mouse-level Gaussian LMM on `log(population_rate)` described at the
    top of this section; it has a frequentist P-value and IS Holm-corrected, in its own 6-member
    planned-contrast family. The NB model is the distribution-aware sensitivity check on it.
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
    negative-binomial/GLMM path, so a COUNT model of rate cannot use the same fitting machinery as
    the amplitude endpoints — a genuine Python-tooling limitation (`statsmodels` alone cannot fit
    this model), not a statistical preference. (The paper-facing rate model sidesteps this
    entirely by modelling the mouse-level population rate, a continuous positive summary over many
    events and cell-seconds, on the log scale — which statsmodels fits with exactly the same
    machinery as amplitude. This NB model exists to check that the simplification does not change
    the conclusion.) Bambi closes that gap directly:
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
- **Two exchangeability assumptions are available, and they answer different questions.** By
  default all 17 mice are exchangeable under the null, which tests the GLOBAL null "no group
  differs from any other". Passing `restrict_to_groups=(a, b)` restricts the shuffle to two
  groups, testing the PAIRWISE null "a does not differ from b" while assuming nothing about the
  third group.

  The distinction matters whenever the EXCLUDED group is the most variable one. Under the global
  null a permuted `hM3D` bucket can contain mCherry mice, so mCherry's between-mouse spread
  enters the null distribution even though mCherry appears nowhere in an hM3D-vs-hM4D statistic.
  In this dataset the control is by far the most variable group (trace fraction-active
  between-mouse SD 0.131 for mCherry against 0.032 for hM3D), and the trace fraction-active
  hM3D-vs-hM4D contrast gives p = 0.099 under the global null against p = 0.027 under the
  pairwise one — the latter matching exhaustive enumeration over all 462 splits of the 11 DREADD
  mice (p = 0.026).

  The reported family members use the GLOBAL null, which is what every previously reported
  permutation p in this analysis was computed under. The run-structure output additionally prints
  the pairwise-null value as a clearly labelled SENSITIVITY figure that enters no multiplicity
  family — the same hypothesis under a different assumption, so entering both would correct one
  question twice.

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

### The decomposition grid: estimates across components x epochs, no stars

`plot_decomposition_grid` is a SEPARATE figure from the per-epoch decomposition panels above, not
a restyling of them. The panels show the distributions and answer "what do these cells look
like"; the grid shows only the contrasts — four components (fraction active, rate among active,
population event rate, per-event amplitude) x the exposure-matched epochs, each as hM3D/Ctl and
hM4D/Ctl with a 95% interval — and answers "which component accounts for the change, and is that
consistent across the session". Both are kept: the per-epoch panels remain the distribution
figures and the sensitivity archive.

**No significance stars appear anywhere on the grid**, deliberately. The four rows are not four
independent phenotypes; they are one exact identity,
`overall_rate = fraction_active x rate_active`, plus the amplitude term. Starring them separately
invites a reader to count significant components as if each were fresh evidence, and starring
them per epoch invites the difference-of-significance fallacy across columns on top of that.
Epoch specificity gets ONE number per ROW — that component's joint group x epoch test, at its
BH-ADJUSTED q — printed on the row label, never one per cell of the grid. (The figure is drawn
after the BH pass for exactly this reason: a member of a multiplicity family is reported at its
corrected value.)

**Three points per panel, not two.** hM3D/Ctl, hM4D/Ctl, and the DIRECT hM3D-vs-hM4D contrast in
grey. The third is there because a reader looking at a row where red sits above the reference
line and blue below it will conclude the two DREADDs differ, and the two plotted vs-control
intervals do not license that: they SHARE mCherry as their reference, so the comparison between
them is not either of the intervals shown. The direct contrast is computed with hM4D as the
reference group and has its own standard error. Drawing it removes the temptation to infer it by
eye. It is deliberately NOT added to the standalone decomposition panels, whose correction family
(`PANEL_HOLM_FAMILY`) is `'vs_control'` because that is where this design's question sits.

**The direct contrast is not systematically wider than the vs-control ones.** Its variance is
`Va/na + Vb/nb`, against `Va/na + Vc/nc` for a vs-control contrast: it drops the control group's
contribution and keeps both DREADD groups'. Whether it is wider or narrower therefore depends on
whether the CONTROL is the noisy group. In this dataset the control is by some margin the most
variable — trace fraction-active between-mouse SD 0.131 for mCherry against 0.032 for hM3D — so
the direct contrast is often the tightest of the three, and can exclude the null in panels where
neither vs-control interval does. This is a genuine property of the comparison, not an
inconsistency. **These intervals are uncorrected and belong to no declared multiplicity family**
(sixteen appear on the grid), so a direct contrast excluding the null is an effect estimate worth
following up, not a discovery to report as one.

Two axis choices carry meaning:

- **Fraction active is a bounded proportion**, so `mouse_contrast_ci` reports it as a DIFFERENCE
  rather than a fold-change, and its row gets a linear axis centred on 0.0 while the other three
  are fold-changes on a log axis centred on 1.0. Forcing one shared axis would plot a difference
  against a ratio reference line.
- **The x-axis is shared WITHIN each row.** The figure exists to be read across epochs, and
  per-panel autoscaling would render a large effect and a small one at the same apparent distance
  from the reference line — which would make the expected "flat across epochs" reading
  unfalsifiable by eye. Fold-change rows use a log axis so a halving and a doubling sit
  symmetrically about the reference.

### Pairwise correction family on the panels

**This describes the INTERNAL panels only.** The paper-facing figures do not compute a statistic
at all: their asterisks are the Holm-adjusted planned contrasts from the unified models
(`_precomputed_stat_fn`, fed from `unified_lmm_posthoc_contrasts.csv`), corrected across six
contrasts per outcome rather than two per panel, and no hM3D-vs-hM4D bracket is drawn.

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
  - `decomposition_grid.png/.svg` — the four decomposition components x the exposure-matched
    epochs, as hM3D/Ctl, hM4D/Ctl **and the direct hM3D-vs-hM4D** effect estimates with 95%
    intervals and **no significance stars**, x-axis shared within each row, with each row
    labelled by that component's BH-adjusted group x epoch q. See "The decomposition grid" above
    for why this is a separate figure from `decomposition*.png` rather than a restyling of it.
  - `decomposition_grid_contrasts.md` — its companion file, keyed per (component, epoch), with
    the direct DREADD-vs-DREADD contrast reported under each panel's vs-control pair.
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
  - `secondary_epoch_interaction.txt` — the joint group x epoch specificity tests, one section
    per decomposition component: the permutation p (paired and unpaired variants where both
    exist), the observed statistic, cell/mouse/trial coverage, and the per-epoch
    difference-of-differences with 95% intervals.
  - `matched_decomposition_trial_coverage.csv` — per-mouse count of trials surviving the
    exposure-matched restriction, against that mouse's total.
  - `descriptive_epoch_delta_post_shock_vs_pre_tone_matched_within_trial.txt` — the descriptive
    post-shock-minus-baseline delta paired WITHIN trial, the alternative to the confirmatory
    member's trial-pooled differencing.
  - `secondary_rate.txt` — the NB rate model, including sampler convergence diagnostics and the
    observed rate contrasts as ratio *and* absolute difference in events/s.

### The paper lane — `PLOTS_DIR/sp_rates_lmm/paper/tfc_amplitude_rate/`

Where the unified paper-facing analysis is fit and where the manuscript figures live. Everything
under `TFC_cond/` above is the internal/sensitivity record.

- `tfc_amplitude_rate_by_epoch.png/.svg` — the two-row (per-event amplitude, population event
  rate) x three-column (pre-tone, trace, post-shock) SuperPlot figure. Per-cell clouds are
  descriptive; the large markers are the exact mouse-level values both models are fit on (asserted
  at draw time against `unified_lmm_mouse_epoch_values.csv`), drawn on the natural scale — the
  amplitude marker is `exp(mouse_mean_log_amplitude)`, i.e. that animal's geometric mean event
  amplitude. Asterisks are the within-epoch Holm-adjusted contrasts (`p_holm_epoch`) read from
  `unified_lmm_posthoc_contrasts.csv` — every panel is annotated by the same procedure, and a
  panel without a bracket is one where neither comparison reached α, not one exempted from
  testing. The panel computes no statistic of its own.
  - `..._contrasts.md` — the same contrasts as text, with their estimates, intervals and adjusted
    P values.
- `tfc_decomposition_forest.png/.svg` — the same twelve contrasts as ratios with 95% CI, one row
  per outcome and one column per epoch, each row labelled with that outcome's joint group x epoch
  Wald test. Same source table as the figure above.
- `stats/`
  - `unified_lmm_mouse_epoch_values.csv` — the inferential dataset: one row per (mouse, epoch),
    both outcomes and their inputs (cell counts, total events, total cell-seconds).
  - `unified_lmm_amplitude_summary.txt`, `unified_lmm_rate_summary.txt` — formula, fitting method,
    n, joint Wald result and the full statsmodels summary per outcome. The embedded coefficient
    table's `P>|z|` column is statsmodels' own **asymptotic z test** and is diagnostic only; the
    paper's P-values are the separately computed `t(df=16)` contrasts and joint Wald F tests in
    the CSVs. Each file states this above and below the table.
  - `unified_lmm_interactions.csv` — the two group x epoch joint Wald tests (F, df1, df2, P).
  - `unified_lmm_posthoc_contrasts.csv` — **the authoritative table.** Twelve rows: outcome,
    epoch, comparison, log estimate, SE, log CI, ratio, ratio CI, raw P, then two adjusted
    P columns — `p_holm_epoch` (paper-facing: Holm across that epoch's two treatment-vs-control
    comparisons; the only one behind a figure asterisk or a Results claim) and `p_holm_six`
    (sensitivity: Holm across all six simple effects within the outcome) — each with its
    rejection flag. Rate rows also carry the descriptive absolute difference in events/s/cell and
    the observed group means.
  - `unified_lmm_diagnostics.png/.txt`, `unified_lmm_residuals.csv`, `unified_lmm_influence.csv` —
    descriptive model diagnostics; see the paper-facing statistics section above.
  - `unified_lmm_synthetic_verification.txt` — the two planted-effect designs and their realised
    values.
  - `paper_results_summary.md` — every number the Results paragraph needs, with Part 1 (the
    paper-facing analysis) and Part 2 (sensitivity and supplementary) separated explicitly.
  - `rate_group_epoch_contrasts.csv` — the NB sensitivity model's posterior rate ratios and HDIs.
