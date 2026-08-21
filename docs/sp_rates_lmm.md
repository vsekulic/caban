# sp_rates_lmm — analysis reference and decision record

Cell-level pyramidal event-**amplitude** analysis of DREADD effects during trace fear conditioning.

**This document is the single source of truth for why this analysis is built the way it is.** It
exists because the design deliberately departs from an external methodological review in several
places, and sessions that read only the review (or only the code) keep re-litigating settled
decisions. If you are picking this up cold, read §2.3 and §8 before changing anything — and §3.4
plus §4.5 before touching an epoch definition or adding one to any model.

| | |
|---|---|
| Module | [caban/sp_rates_lmm.py](../caban/sp_rates_lmm.py) |
| Entry point | `caban.sections.run_sp_rates_lmm(ds, cfg)`, gated on `cfg.plot_sp_rates_lmm` |
| METHODS template | [analysis_methods_templates/sp_rates_lmm_methods.md](../analysis_methods_templates/sp_rates_lmm_methods.md) |
| Notebook | `run_pipeline.ipynb`, cells 24–29 (immediately after `run_sp_rates`) |
| Output | `PLOTS_DIR/sp_rates_lmm/{TFC_cond,Test_B,Test_B_1wk}/` + `stats/` subdirs |
| Design | 17 mice — hM3D n=5, hM4D n=6, mCherry n=6 |

---

## 1. What this replaced, and why

The original `sp_rates` section ([sections.py](../caban/sections.py), `run_sp_rates`) produced
**~205 uncorrected three-group one-way ANOVAs** — every combination of session × behaviour period ×
cross-registration subset × metric, each with ANOVA-gated Tukey. Counting: TFC_cond alone was
3 periods × 5 mappings × 2 metrics × 2 panel variants = 60, plus 10 whole-session; Test_B and
Test_B_1wk 56 each; LT1/LT2 10 each; Test_A/Test_A_1wk 2 each.

Three structural problems, not just multiplicity:

1. **No correction across the ~205 omnibus tests.** Tukey only limits multiplicity *within* one
   three-group ANOVA.
2. **The unit of analysis changed between panels.** The `-avg` variant plotted one point per
   *behaviour period* (n=3–5) — periods were never independently assigned to a DREADD group, so
   that is pseudoreplication, not a treatment test.
3. **Overlapping cross-registration subsets were presented as if they were replications.** They are
   nested views of the same 17 animals.

`sp_rates_lmm` replaces the inferential backbone with **three Holm-corrected confirmatory tests**
and re-files everything else as declared secondary or sensitivity analysis. The old panels are not
deleted — their status changes to a sensitivity/multiverse archive.

> The family was **two** tests until the epoch split (§3.4) gave the post-shock window a real
> 20 s definition and made `post_shock_vs_baseline` a co-equal member. Anything in this document
> quoting a two-test family or a `p_holm` computed against one is stale — the primary trace
> amplitude p_holm moved 0.018 → **0.027** for this reason alone, with an unchanged p_raw.

---

## 2. The external methodological review

A Deep Research report was commissioned mid-design (PDF supplied by the user; not in the repo). It
is a good report and most of it was adopted. It is **not** authoritative where it conflicts with
this document — several of its assumptions about the data turned out to be wrong, and one of its
central recommendations was deliberately inverted.

### 2.1 What it recommended

| Issue | Review's recommendation |
|---|---|
| Experimental unit | Mouse is the independent treatment unit; cells/trials add precision, not replication |
| Primary TFC outcome | **Trace-interval event RATE** as the confirmatory centrepiece |
| Model | Negative-binomial mixed model on counts with `log(exposure)` offset, mouse × trial level |
| Epoch analysis | One integrated `group × epoch` model, not per-epoch ANOVAs |
| Multiplicity | 1–2 primary hypotheses under Holm; remainder exploratory under BH-FDR |
| Locomotion | Primary estimand **unadjusted** for freezing/speed (they are post-treatment); movement-conditioned analyses reported separately |
| Cell subsets | One primary definition; the others are sensitivity analyses, never replications |
| Amplitude | Mean amplitude **conditional on a detected event** as a *secondary* property; summed-amplitude-per-second confounds frequency with amplitude |
| Detection bias | Real; reference-seeded / joint cross-session extraction is preferable to intersecting independently detected ROI lists |
| Nulls | Report rate ratios **and** absolute differences with 95% CIs; a non-significant result at n=5/6/6 is weak evidence of absence |
| Software | `glmmTMB` first choice, `brms` second; explicitly **not** `statsmodels.MixedLM` for counts |

Useful quantitative content worth keeping (all from the review, none of it implemented as code):

- **Power.** At α=0.05 / 80%: detectable Cohen's *d* ≈ **1.91** (5 vs 6), **1.80** (6 vs 6),
  **2.02** (5 vs 5); at α≈0.025, 2.18 and 2.04. Three-group omnibus at N=17: Cohen's *f* ≈ **0.84**.
  These are very large standardized effects.
- **Rate-ratio conversion.** `RR_detectable ≈ exp(d · σ_log r)`. At *d*≈1.9: mouse-level CV 30% →
  RR 1.75; 50% → 2.45; 70% → 3.32.
- **Repeated trials.** For ICC ρ over *m* trials, variance of the mean scales as `ρ + (1−ρ)/m`.
  Five trials give an effective standardized-effect gain of ×1.67 (ρ=0.2), ×1.29 (ρ=0.5),
  ×1.09 (ρ=0.8). **5 trials × 5 epochs ≠ 25 replicates** — epochs are conditions, not repeats.
- **2 s shock window.** At 0.05–0.2 Hz, P(zero events per cell) is 90.5%–67.0% for 2 s, vs
  36.8%–1.8% for 20 s. Poisson counting CV `1/√(λT)` ≈ 158–316% at 2 s. This is why shock is
  excluded from the confirmatory factorial model.
- **Bibliographic correction the review makes:** Yu et al. 2022 is the mixed-effects primer;
  the hierarchical-bootstrap paper is **Saravanan, Berman & Sober 2020**. Aarts et al. is the
  nested-data / pseudoreplication reference.

### 2.2 Where we followed it

- Mouse as the unit of inference, everywhere, without exception.
- Small confirmatory family under Holm (now three tests — §4.1); everything else declared
  secondary/sensitivity, with the BH-FDR family the review asked for now implemented (§4.2).
- Integrated epoch model rather than per-epoch tests.
- Locomotion **not** covaried in the primary estimand.
- One primary cell set (`full`); cross-registration subsets demoted to sensitivity.
- Zero-event cells retained in every rate/fraction-active denominator.
- Shock (2 s) excluded from the confirmatory factorial model.
- Trace interval privileged as the primary epoch, on the review's own *a priori* grounds
  (hippocampal dependence of trace conditioning) rather than because it gave a good result.
- "Locked confirmatory reanalysis", **not** prospective preregistration — the endpoint was chosen
  after prior inspection of this dataset. The METHODS file says so explicitly.
- Detection-bias limitation stated in the review's own terms (activity-dependent ROI inclusion).

### 2.3 Where we deliberately departed — **do not reopen these**

**(a) Amplitude is PRIMARY; rate is SECONDARY. The review recommended the opposite.**

This is the single most-relitigated point. The user's scientific claim is specifically about
**burst magnitude** — that hM3D produces larger per-event Ca²⁺ influx — not about firing more
often. The review recommended rate primary partly on the grounds that deconvolved `S` amplitude is
not calibrated, which is true and is handled by relabelling the measurement (§3.1), not by changing
the endpoint. The user's position, verbatim: *"It should be precisely the other way around
(amplitude is primary; frequency secondary)."*

Rate is reported with a full interval in **every** panel alongside amplitude. "Secondary" means it
carries no confirmatory alpha, not that it is de-emphasized.

There is also a technical reason the review's fix cannot be applied as stated: the rate model is
Bayesian (LOO/HDI), and you cannot Holm-correct across a mixed Bayesian/frequentist family. Holm
stays across a small all-frequentist family — exactly **3** tests (§4.1).

**(b) Within-cell epoch differencing instead of `(1|mouse) + (1|mouse:cell)`.**

The review is right that a mouse random intercept alone is insufficient for a cell × trial × epoch
table. But the nested version is ~8,000 cell levels over ~100k rows in `statsmodels` — slow and
likely singular. Instead, for each cell we compute `Δ = mean log amplitude[epoch X] − mean log
amplitude[pre_tone]` and fit `Δ ~ group + (1|mouse)`. **The cell random intercept cancels by
construction**, which is both honest and tractable, and reuses the paired logic already validated
in `compute_lt1_lt2_amplitude_delta`. Cost: a cell must have ≥1 event in *both* epochs to
contribute; report how many qualify.

**(c) Bambi/PyMC instead of `glmmTMB`/`brms`.**

Python-only constraint. `statsmodels.mixedlm` is Gaussian-only with no NB-GLMM path — the review is
correct to reject it for counts. Bambi (on PyMC) gives a genuine NB mixed model with dispersion
estimated jointly, which is what `glmmTMB` would do. No R runtime exists on this machine.

**(d) Per-event INTEGRAL, not peak amplitude.** See §3.1. The review's critique of
"summed-amplitude-per-second" is accepted and that metric is retired as an *amplitude* measure —
but the replacement is a per-event integral, not the peak.

### 2.4 Where the review was factually wrong about this dataset

- **Trace duration is not ~35 s.** From [sessions.py:1261-1264](../caban/sessions.py#L1261):
  `tone_onsets_def = [185, 420, 660, 900, 1140]`, `tone_duration = 20`,
  `shock_onsets_def = [220, 460, 700, 940, 1180]` → trace = **[15, 20, 20, 20, 20] s**. Trial 1 is
  short because `tone_onsets_def[0] = 185` rather than an intended 180 (flagged as an accident in
  the source comment). Handled automatically: exposure is computed from each trial's actual frames.
- **`S` is not impulse-like.** The review implicitly treats events as point processes. Measured on
  real data: only **11.7%** of supra-threshold runs are single-frame; median 4 frames, mean 5.44,
  max 118. This is why the per-event integral is meaningful at all (§3.1).
- **hM4D detection dropout is not elevated.** The review flagged activity-dependent ROI loss as
  likely to bias hM4D upward. Measured LT1→LT2 dropout: hM3D 81.7%, hM4D 77.1%, mCherry 76.8% —
  hM4D is indistinguishable from control. This is a reportable **negative** result that makes the
  hM4D null interpretable. The larger finding is that dropout is ~77–82% in *all* groups.

---

## 3. Measurement definitions

### 3.1 Event definition: per-event integral of a contiguous run

`find_spikes_ca` / `find_spikes_ca_S` ([utilities.py](../caban/utilities.py)) detect one event per
**local maximum** and report the `S` value at that single peak frame. Summed over a window that is
exactly `rate × mean peak height` — mathematically incapable of separating "bigger events" from
"more events".

This analysis uses `find_event_runs_ca` / `find_event_runs_ca_S` instead:

- **one event = one contiguous supra-threshold run** of `S`
- **amplitude = `sum(S[run])`**, the per-event integral, not the peak
- `n_local_maxima` is returned as a diagnostic (how many events the legacy detector would have
  reported for that run), reusing `find_spikes_ca` itself so the two can never disagree

**Consequences that must be stated, not glossed:**

- A run with two local maxima is **one** event here, not two. Event *counts* therefore differ from
  the legacy convention (~50,318 → 45,857 on one mouse; 4.5% of runs are multi-peak). Rate
  endpoints are recomputed under this same definition — never mix with legacy counts, and never
  compare new rate numbers directly against old `sp_rates` panels.
- Because temporally clustered peaks **merge**, part of the observed "high amplitude + low rate
  among active cells" pattern in hM3D is a mechanical consequence of the event definition. The
  run-structure evidence (§4.4) is what distinguishes genuine bursting from a merging artifact.
- **"Bursting" is the interpretation; "larger deconvolved event run integral" is the
  measurement.** Keep that distinction in all text and figure captions.

### 3.2 Zero-event cells

Kept in every rate / fraction-active denominator (the codebase convention — a manipulation that
silences cells must show up as reduced population rate, which only happens if silenced cells stay in
the denominator). Amplitude is **undefined** for a cell with zero events, so those cells are
excluded from amplitude computations — definitional, not missing data. Say so in captions, or the
amplitude panel looks like it silently dropped cells.

### 3.3 The decomposition identity

At the cell-window level, exactly:

```
summed S per second  =  event rate  ×  mean per-event amplitude
log(total S/sec)     =  log(rate)   +  log(mean amplitude)
```

`report_decomposition_additivity()` performs the arithmetic check. It is **not** a fitted model or a
significance test — a proper SE for the sum needs the joint covariance of two separately-fit models.

**Weighting rule (state it):** aggregated across cells the identity holds with **event-weighted**
mean amplitude, not cell-weighted. The per-cell models use **cell-weighted** means (one row per
cell). Only the additivity check is event-weighted.

### 3.4 Epochs

**`post_shock` used to mean the entire 198 s inter-trial interval.** It now means a 20 s response
window from shock offset. Any result predating that split averaged the post-shock response over
roughly ten times its own duration and **must not be compared with results computed after it**.
The full ITI is still available under the name it should always have had, `iti`.

| epoch | window | role |
|---|---|---|
| `pre_tone` | 35 s ending at tone onset | locked confirmatory reference |
| `pre_tone_matched` | 20 s ending at tone onset | exposure-matched baseline; **overlaps `pre_tone`** |
| `tone` | 20 s | descriptive |
| `trace` | tone offset → shock onset (15 s trial 1, 20 s trials 2–5, per §2.4) | primary |
| `shock` | 2 s | **excluded** from everything confirmatory (review's Poisson numbers, §2.1) |
| `post_shock` | 20 s from shock offset | co-primary |
| `post_shock_late` | 20 s, beginning 90 s after shock offset | descriptive; early-vs-late control |
| `iti` | shock offset → next tone onset (198 s) | defined, not used by the pipeline |

**Three epoch sets exist and the distinctions are load-bearing.** Getting these confused is how
the sampler once ran for 113 minutes instead of one.

| constant | members | why this set |
|---|---|---|
| `TFC_EPOCHS` | everything the event table computes | carries **both** baselines |
| `TFC_DISJOINT_EPOCHS` | `pre_tone`, `tone`, `trace`, `post_shock`, `post_shock_late` | safe to model **jointly**; excludes `pre_tone_matched`, which overlaps `pre_tone` in time |
| `TFC_RATE_MODEL_EPOCHS` | `pre_tone`, `tone`, `trace`, `post_shock` | what the **rate model** may use — see §4.5 |

Putting both time-overlapping baselines on one epoch factor is a duplicated-data bug, not an
inefficiency: 20 s of every baseline enters the likelihood twice, inflating the baseline's
precision, corrupting the epochs-within-trial variance component, and leaving NUTS on a
near-collinear ridge. `build_mouse_trial_epoch_rate_table` filters at its own source rather than
trusting callers. **Preserve that.**

#### The early-vs-late post-shock contrast

Puhger et al. 2024 (*iScience* 27:109035) report no bulk CA1 calcium response during the trace
interval at all, a large sustained response to the footshock, memory impairment when CA1 is
silenced 0–40 s after the shock, and **no** impairment when the same silencing arrives 140 s
after it. That result is a *contrast* between early and delayed, so reproducing it needs a late
window — which is what `post_shock_late` is for.

`post_shock` vs `pre_tone` is **not** that contrast. Each epoch is pooled across trials *before*
differencing, so the reference is a mixture: trial 1's baseline is shock-naive, trials 2–5's sit
163 s after the preceding shock, the indices don't align, and trial 5's shock has no `pre_tone`
counterpart at all. The descriptive contrast `log_amp(post_shock) − log_amp(post_shock_late)` is
within-trial at aligned indices with no naive window in the reference.

Two things about it that are easy to get wrong:

- **It is DESCRIPTIVE.** It is in neither multiplicity family, spends no alpha, and is reported as
  an estimate with a 95% interval, never a significance verdict. Its output is named
  `descriptive_*` rather than `coprimary_*` so the status is legible from the filename. The
  confirmatory family stays at exactly three.
- **The late window does not exist on every trial, and both sides are trial-matched.** The final
  trial's ITI ends with the *recording*, and real sessions stop as little as **20.5 s** after the
  last shock offset — barely `post_shock` itself. On such a trial no late window fits at *any*
  onset value. `get_epoch_frames` returns `None` there (a definitional absence, as for `pre_tone`
  on trial 1) while still **raising** for `post_shock`, which every trial must supply.
  `restrict_to_shared_trials` then cuts both sides to the trials where both exist, per mouse, and
  writes the surviving counts to `stats/descriptive_early_vs_late_trial_coverage.csv`. **Report
  that coverage with the estimate** — observed range is 3–5 of 5 trials per mouse.

**Why 90 s and not Puhger's 140 s.** It maximizes trial coverage; it does not rescue the truncated
final trial, which nothing can. What the constant controls is the *mid*-session trials, where a
shorter onset can only ever admit more trials than a longer one. Raising it toward 140 s buys
protocol fidelity at the cost of trials — check the coverage file first.

Test_B / Test_B_1wk post-tone is pinned to **20 s** explicitly via
`get_testb_epoch_frames(..., post_tone_duration_s=20.0)`. Neither existing default is the recall
analog of trace: the function's own default is 35 s, and `TestBSession.post_tone_onsets/offsets`
runs to the *next tone onset* (~200+ s ITI).

### 3.5 Cohort gaps

`Test_B_exp_frames['G07'] = []` and `dpath_Test_B_1wk_day['G15'] = ''` — both hM4D. So **each
recall session runs at hM4D n=5, N=16, and it is a *different* mouse missing from each**. The
48 h → 1 wk comparison is therefore not on a fixed cohort. This is session *existence*, distinct
from the mapping-availability lists (`mappings_all_Test_B_G15` etc.), which describe which crossreg
subsets exist for a mouse that **is** present.

---

## 4. Statistical architecture

### 4.1 Confirmatory family — Holm across exactly three tests

| key | Endpoint | Model |
|---|---|---|
| `trace_amplitude` | Trace-period per-event amplitude (primary) | `log_amplitude ~ C(group)`, cell-level, trials pooled, `(1\|mouse)` |
| `trace_vs_baseline` | Trace epoch specificity (co-primary) | Within-cell `Δ = trace − pre_tone`, then `Δ ~ C(group) + (1\|mouse)` |
| `post_shock_vs_baseline` | Post-shock epoch specificity (co-primary) | Same, `Δ = post_shock − pre_tone` |

The two co-primary members go through **one** code path,
`fit_and_report_epoch_delta(..., is_confirmatory=True)`, so they cannot diverge in cell selection,
model specification, or what gets reported — a difference between them has to be a difference in
the data. That function also serves the descriptive early-vs-late contrast with
`is_confirmatory=False`, which is not cosmetic: it decides whether the stats file *claims* its p
enters the Holm family, and writing "CONFIRMATORY" over a descriptive estimate would be a false
inferential claim in a file someone will quote.

Its output paths key on **both** epochs whenever the reference is not `pre_tone`, so two contrasts
sharing a response epoch cannot silently overwrite each other. The two confirmatory calls keep
their historical single-epoch filenames byte-identical.

All use `fit_mixed_model` ([single_unit_common.py](../caban/single_unit_common.py)), which falls
back to mouse-clustered OLS on convergence failure or boundary degeneracy — a documented fallback,
not a silent error swallow. Omnibus tests use `joint_wald_test`.

**Adding a fourth member is a design change, not a code change.** The family is locked; new
epochs go in as descriptive (that is what `post_shock_late` did).

### 4.2 Secondary family

Event rate (Bambi NB-GLMM, mouse × trial × epoch, `log(total cell-seconds)` offset, LOO vs a
no-interaction model); fraction active; Test_B / Test_B_1wk post-tone amplitude; `group × trial`
(photobleaching control); run structure; threshold sensitivity.

**BH-FDR is implemented** (`build_secondary_fdr_table`, α=0.05, written to
`stats/secondary_fdr_family.csv`/`.txt`). Members are accumulated into `secondary_pvalues` as each
is fit and corrected in one pass at the end: `group_x_trial_interaction`, `run_structure_*`, and
`{Test_B,Test_B_1wk}_post_tone_amplitude`.

**The family is frequentist secondary tests only.** Explicitly outside it, by design:

- the three confirmatory omnibus tests (own Holm family, different error rate — a test in two
  families is corrected twice);
- the Bambi NB rate model — posterior contrasts, HDIs and ELPD-LOO, no p-value to correct.
  Manufacturing one so a Bayesian result can be folded into a frequentist FDR family is a category
  error, not a conservative choice;
- all descriptive and sensitivity output, **including the early-vs-late post-shock delta**.

### 4.3 Small-n inference — mouse-label permutation

`mouse_label_permutation_test` shuffles **group labels across the 17 mice** (never cells) and
recomputes an arbitrary statistic. ~5.7 M distinct 5/6/6 relabellings exist, so this is Monte Carlo
(default 20,000 draws), not exhaustive enumeration.

`make_contrast_stat(..., weight='cell'|'mouse')`. **The equal-mouse-weighted contrast is the primary
reported point estimate** — cell-weighted estimates let mice with more active cells dominate, and
cell inclusion is itself post-treatment and activity-dependent (hM3D contributes ~561
amplitude-active cells/mouse vs ~379–389 for the others). Inference is valid either way; only the
point estimate's weighting changes.

**Prefer the permutation p-value to the clustered-OLS p-value.** Cluster-robust SEs are
anti-conservative at 17 clusters. Triangulate three ways every time: model estimate, 17-point
mouse-level plot, permutation CI.

### 4.4 Run structure — the bursting evidence

Direct, height-independent burst signature: run **width**, **local maxima per run**, **fraction of
multi-peak runs**, each with mouse-level points and permutation tests, plus threshold sensitivity at
`thres ∈ {1.5, 2.0, 3.0}`. Measured on real data (trace epoch):

| group | impulse_frac | mean width | p99 | max |
|---|---|---|---|---|
| hM3D | 0.112 | 5.74 | 26 | 118 |
| hM4D | 0.138 | 4.85 | 22 | 119 |
| mCherry | 0.131 | 4.90 | 22 | 110 |

hM3D runs are **wider with the same median (4.0)** — a fattened right tail. Per-mouse means separate
almost completely (hM3D min 5.12 > hM4D max 5.09; only mCherry's G16 at 5.25 overlaps).

### 4.5 The rate model takes four epochs, not five — `TFC_RATE_MODEL_EPOCHS`

Being disjoint in time is **necessary but not sufficient** to enter the NB-GLMM.
`post_shock_late` is disjoint (verified pairwise against all other windows, across trial indices
too) and is in `TFC_DISJOINT_EPOCHS`, but `build_mouse_trial_epoch_rate_table` filters to the
narrower `TFC_RATE_MODEL_EPOCHS`, holding it out for three reasons:

1. **It has no role here.** It is a descriptive window carrying a within-cell *amplitude*
   contrast. The rate model is the prespecified secondary *rate* endpoint; adding a descriptive
   epoch enlarges a model that makes claims without contributing to any of them.
2. **It is ragged in a way that matters specifically here.** The window is absent on final trials
   whose recording stops early, so its rows skew toward low trial indices — and `trial` is a
   continuous covariate in this very model, so its epoch dummy correlates with `trial`. The
   amplitude contrasts, differenced within cell and trial-matched, never face this.
3. **Empirically it makes NUTS pathological.** Measured on this model: 4 epochs / 340 rows samples
   in **~27 s**; 5 epochs runs **>280 s without finishing** — and does so even when the fifth
   level is made artificially balanced across all trials, so the cost is the extra level in an
   already weakly-identified `group × epoch` interaction, *not* the raggedness.

**This is not the duplicated-data ridge of §3.4** — that one is about overlapping windows, and
these windows are verifiably disjoint. Different failure, same symptom, and conflating them will
send you looking for an overlap that isn't there. Holding this set at four also means the reported
rate results are the same model that produced them before `post_shock_late` existed, so they need
no reinterpretation.

---

## 5. Results as of the last full run

**Confirmatory (Holm across 3):**

| test | p_raw | p_holm | |
|---|---|---|---|
| `trace_amplitude` | 0.0089 | **0.027** | reject |
| `trace_vs_baseline` | 0.998 | 1.0 | — |
| `post_shock_vs_baseline` | 0.935 | 1.0 | — |

- **Primary — trace amplitude: hM3D +0.435 log units (~1.5× control), p_holm = 0.027.** hM4D null
  (+0.092, p=0.60). The permutation p is also 0.027; quote that one.
- **Both co-primary epoch-specificity tests are NULL.** Neither the trace window nor the
  post-shock window shows a group-specific within-cell elevation over baseline.

> `p_holm` for the primary was **0.018** under the old two-test family. `p_raw` did not move;
> only the family size did (§1). Do not mix the two numbers across document versions.

**The co-primary nulls matter and are easy to get wrong.** An earlier version reported
p = 1.2 × 10⁻¹¹ from a `group × epoch` model fit on 100,264 cell × trial × epoch rows with only a
mouse random intercept — that was pure pseudoreplication. Corrected, there is **no trace-specific
effect**: hM3D elevates amplitude **globally across all epochs**. No figure, caption, or text may
imply a trace-specific effect. The trace interval's privileged status now rests on prior anatomy and
behaviour, not on this result.

The same now holds for the post-shock window: with a properly-windowed 20 s definition (§3.4) it
is *also* null, so the epoch split did not rescue a post-shock-specific effect either. The flat
profile across every epoch is the finding. `plot_epoch_profile` and `epoch_delta_forest` exist to
make that visible rather than asking a reader to take it on trust — a figure showing only the
trace contrast cannot distinguish "no trace-specific effect" from "no effect".

**Dissociation (the main scientific story):** hM3D → amplitude; hM4D → rate. hM4D's
rate-among-active-cells is reduced vs control while its amplitude is flat; the Bambi model agrees
independently (`hM4D:trace` rate coefficient −0.278, HDI [−0.527, −0.047]).

**Manipulation check (LT1 drug-free → LT2 CNO, within-cell):** hM3D +0.474 vs control, p = 5×10⁻⁷.
Control cells *drop* 0.27 log units LT1→LT2 (time/bleaching); hM3D reverses that to +0.20.

**Recall:** Test_B (48 h) hM3D +0.343, p<0.001 — persists. Test_B_1wk +0.220, p=0.26 with a wide CI
— underpowered, **not** demonstrated absence. Note this diverges from the behavioural phenotype
(hM3D keeps elevated post-tone freezing at 1 wk), so it cannot be reported as supporting that link.

**Trial effect:** amplitude declines monotonically across trials (−0.098, −0.187, −0.287, −0.372;
~31% by trial 5), consistent with photobleaching. Does not threaten a between-group contrast unless
the *rate of decline* differs by group — hence the `group × trial` check.

---

## 6. Plotting architecture

Three layers. Per CLAUDE.md's dedup rule, **extend the shared primitive rather than adding a
parallel one.**

**Shared primitives — [caban/single_unit_common.py](../caban/single_unit_common.py):**

- `draw_superplot_triplet(ax, cell_values_per_group, mouse_means_per_group, group_order,
  group_colours, stat_fn, ylabel=, cell_size=1.5, cell_alpha=0.18, mean_size=90,
  jitter_width=0.34, yscale='auto')` — SuperPlot (Lord et al. 2020, *J Cell Biol* 219:e202001064):
  every cell semi-transparent and coloured by mouse (`_mouse_colour_shades`), per-mouse means as
  large black-edged markers. Cells share one per-mouse x-offset so a mouse's cloud and its mean
  group visually.
- `_resolve_superplot_yscale` — `'auto'` → `'log'` (all positive), `'symlog'` (non-negative with
  exact zeros; `linthresh` = smallest positive value), `'linear'` (any negative).
- `ecdf_panel`, `grow_ylim_for_bracket_headroom`, `bracket_ylim`, `save_fig` (PNG+SVG pair).

**[caban/analysis.py](../caban/analysis.py):** `_draw_violin_triplet`, `do_pairwise_holm_plot` /
`do_anova1_plot` (the `stat_fn` interface), and `barplot_annotate_brackets` (in
[utilities.py](../caban/utilities.py)).

**[caban/sp_rates_lmm.py](../caban/sp_rates_lmm.py) panel wrappers:**
`_mouse_values_per_group`, `_draw_mouse_violin_panel`, `_draw_cell_superplot_panel` (takes the
*same* tidy per-cell frame, so the two are interchangeable per panel), `_save_panel`.

Entry points: `plot_primary_trace_amplitude`, `plot_epoch_profile`, `plot_amplitude_ecdf`,
`plot_amplitude_p90`, `plot_decomposition`, `plot_manipulation_check`, `plot_run_structure`,
`plot_threshold_sensitivity`, `plot_effect_forest`, `plot_example_traces`,
`plot_width_vs_height_matched_examples`.

### Four plotting rules that are not negotiable

1. **Unit of inference is the mouse; unit of display may be the cell.** `draw_superplot_triplet`
   passes **only** the per-mouse means to `stat_fn`; the cell cloud never reaches it. A regression
   test asserts a SuperPlot's p-value equals a standalone mouse-level `do_pairwise_holm_plot` on the
   same data — keep it passing. This is the one way these figures could silently reintroduce
   pseudoreplication.
2. **Axis scale is load-bearing.** A linear axis on these heavy-tailed quantities crushes all 17
   mouse means into the bottom 6–13% of the panel — that is why a first SuperPlot attempt was
   reverted. **But pass `yscale='linear'` for an already-log-transformed column** (e.g.
   `log_amplitude`): `'auto'` sees all-positive values and applies a *second* log, plotting
   log10(ln(x)).
3. **Brackets only work on linear axes.** `barplot_annotate_brackets` scales offsets as a fraction
   of `ylim[1]−ylim[0]` in *data* coordinates — meaningless in log space. On non-linear axes,
   `stat_fn` is called with `annotate=False` and the p-values are rendered as a text block. Same
   numbers either way.
4. **Display order is `DREADD_DISPLAY_ORDER = ['mCherry','hM3D','hM4D']`** (Ctl, Exc, Inh) on every
   panel — CLAUDE.md. Do **not** confuse with `GROUP_ORDER = ['hM3D','hM4D','mCherry']`, which is
   model dummy-coding and `stat_fn`'s positional argument order and must not change.

---

## 7. Corrections history

Everything below was found and fixed. Listed so nobody re-fixes, re-introduces, or re-argues them.

| # | Problem | Resolution |
|---|---|---|
| 1 | `joint_wald_test` used observation-level denominator df for cluster-robust fits | `n_groups` is now a **required** argument; `df2 = n_groups − 1`. Primary trace p went 0.0016 → **0.0089**, matching statsmodels' own `Prob(F-statistic)`. Also fixed the same pre-existing bug in `pca_state_metrics._lmm_holm_pairs` — its omnibus p-values changed. |
| 2 | Rate-model exposure offset used `'first'` (one cell's window) while summing counts across cells | `'sum'` → total cell-seconds. Mattered because cell count is group-correlated, so the omitted `log(n_cells)` leaked into the group fixed effect rather than being absorbed by the mouse intercept. |
| 3 | Recall figures carried a hardcoded "Trace-period amplitude" title | `title` parameter threaded through. |
| 4 | Co-primary understated its uncertainty (p = 1.2e-11) | Replaced with within-cell differencing (§2.3b). Now p = 0.998. |
| 5 | Primary point estimate was implicitly cell-weighted | `weight='mouse'|'cell'`; equal-mouse-weighted is now primary. |
| 6 | SuperPlot on a linear axis was unreadable | `yscale='auto'`; smaller/fainter markers. |
| 7 | Amplitude panel was double-logged | Explicit `yscale='linear'` for pre-logged columns. |
| 8 | `compute_lt1_lt2_amplitude_delta` asserted equal `unit_id` across LT1/LT2 | Wrong invariant — see §8. Now asserts equal *length* and pairs by row position. |
| 9 | Section took ~50–60 min | `find_spikes_ca`'s per-frame Python loop vectorised (**22×**, output bit-identical, verified on 3,008 adversarial cases); `make_contrast_stat` precomputes per-mouse arrays outside the permutation closure (**85–205×**). Now well under 10 min. |
| 10 | `post_shock` meant the whole 198 s ITI, diluting a ~20 s response by ~10× | Split into `post_shock` (20 s response) and `iti` (the old meaning). `post_shock_vs_baseline` became the third confirmatory member, moving the primary `p_holm` 0.018 → 0.027. |
| 11 | Fitting both time-overlapping baselines on one epoch factor | Sampler ran **113 min instead of ~1**. `TFC_DISJOINT_EPOCHS` introduced and enforced inside `build_mouse_trial_epoch_rate_table`. |
| 12 | `fit_and_report_epoch_delta` hardcoded "CONFIRMATORY" and keyed output filenames on `response_epoch` alone | Gained `is_confirmatory`; paths key on both epochs when the reference isn't `pre_tone`. Without this the first descriptive reuse would have both overwritten `coprimary_epoch_delta_post_shock.txt` and stamped a false inferential claim on it. |
| 13 | `post_shock_late` in the rate model | 4→5 epoch levels took NUTS from ~27 s to >280 s without finishing, *not* from any overlap. `TFC_RATE_MODEL_EPOCHS` added (§4.5). |
| 14 | The last ITI was assumed ~117 s from the nominal protocol timing | It is data-dependent and can be **20.5 s**. Assumption was written into code comments and METHODS before being measured; corrected, and the early-vs-late contrast is now trial-matched rather than relying on a fixed trial count. |

---

## 8. Gotchas that have already bitten

- **Cross-registration correspondence is ROW ORDER, never `unit_id` equality.** The same physical
  cell has a *different* `unit_id` in each session — each session reads its own column of the
  crossreg table. This is why `get_actual_cells_from_df_session` must never sort (see its
  `# OMG NO!!!` comment). Pair across sessions by zipping the two `S_idx` lists positionally.
- **LT1, LT2 and TFC_cond share one crossreg object** (`TFC_cond_crossreg`), which is what makes
  positional pairing between them valid. **Test_B / Test_B_1wk use a different one**
  (`TFC_B_B_1wk_crossreg`) — never pair those positionally against TFC_cond.
- **`build_epoch_event_table`'s `'cell'` column is session-local.** Never join it across sessions.
- **Import order:** `caban.decoder` must be imported before `caban.analysis` (deferred cross-module
  block at the bottom of `decoder.py`). Any standalone script needs this.
- **Environment:** `/Users/vsekulic/miniforge3/envs/caban/bin/python`. The repo `.venv` has no
  numpy. `bambi`/`pymc`/`arviz` live in the conda env.
- **Never run `caban.sections.run_*` from an agent session** (AGENTS.md) — the user drives it from a
  live notebook kernel. Tell them which cell to run. Synthetic-data renders and read-only snippets
  are fine and are how bug #7 was caught.
- **Synthetic test fixtures must give LT1/LT2 different `unit_id`s per session**, or they encode the
  same false assumption as bug #8 and cannot catch it. That is exactly how it slipped through.
- **Never derive session timing from the nominal protocol constants.** `tone_onsets_def` /
  `shock_onsets_def` / `light_onsets = [0, 1299]` describe the *intended* session. The last ITI in
  particular ends at `post_shock_offsets[-1] = miniscope_exp_fnum[stop_idx]` — the real recording
  stop — and has been observed at **20.5 s** where the nominal figure suggests ~117 s. Trial count
  varies too. Measure per mouse; do not reason from the constants (correction #14).
- **A slow sampler has two distinct causes.** Overlapping windows on one epoch factor (§3.4) and
  simply too many epoch levels (§4.5) present identically. Check disjointness *first* — it is
  cheap and settles which one you have — before assuming an overlap.
- **`get_epoch_frames` returning `None` is not the same as an error.** It means the window does
  not exist for that trial (`pre_tone` on trial 1, `post_shock_late` on a truncated final trial),
  and `_iter_event_windows` skips it. That is definitional absence, not a swallowed failure — but
  it does mean any epoch that can be absent needs trial-matching before it is differenced against
  one that cannot (`restrict_to_shared_trials`).
- **The event table is built in ONE pass.** `build_epoch_and_run_tables` does a single
  `find_event_runs_ca_S` per mouse and slices every epoch out of it, so adding an epoch costs one
  extra window slice, **not** a re-detection. Never build a second table.

---

## 9. Open items

**Done since this list was written** — left visible so they are not re-proposed: `(1|mouse:trial)`
in the Bambi rate model (now `(1|mouse) + (1|mouse_trial)`); the BH-FDR family (§4.2); descriptive
`tone`/`post_shock` within-cell deltas alongside the confirmatory trace one
(`compute_all_epoch_deltas`, plotted by `plot_epoch_delta_forest`).

**Statistical:**

- Confirmatory panels currently Holm-correct all 3 pairwise contrasts; only hM3D-vs-control and
  hM4D-vs-control need alpha (Dunnett-style). Current state is *conservative* — costs power, does
  not inflate error. Add an **option** to `do_pairwise_holm_plot`; do not change its default, it is
  shared with `freezing_tuned_cells` and others.
- Report absolute differences alongside ratios (matters most for the hM4D null).
- Power/sensitivity statement: every null must report its interval and what it excludes
  (hM4D amplitude ~1.10 [0.77, 1.55] ⇒ effects up to +55% not excluded). This now applies to
  **both** co-primary nulls, and to the early-vs-late delta, which is reported as an interval by
  construction.
- Decide whether `POST_SHOCK_LATE_ONSET_S` should move toward Puhger's 140 s now that the
  coverage numbers are known (3–5 of 5 trials at 90 s). Purely a fidelity-vs-coverage trade;
  one-line change plus a re-run.

**Plotting:**

- Rate panels show horizontal **striping** — per-cell rate is `n_events / exposure`, so small
  integer counts give discrete values. Real quantization, not a bug. Options: leave it, small
  vertical jitter, or revert those panels to mouse-level violins.
- **Mixed annotation style** in the decomposition row (brackets on the linear amplitude panel, text
  p-values on the log panels). Forcing text everywhere is probably right and matches the move to
  effect estimates + CIs.
- Overall event rate visually dominant in the decomposition; per-mouse ECDF curves more prominent
  than the pooled curve.

**Deferred (documented as future work, not silently missing):** LT1→LT2 dropout decomposition;
cross-registration sensitivity forest plot; movement-conditioned / speed-binned / movement-
standardized analyses; ROI-per-FOV QC panel; reference-seeded (CaliAli) re-extraction; count-model
posterior-predictive and zero-inflation diagnostics; peri-shock analysis.

---

## 10. File map

| File | Role |
|---|---|
| [caban/sp_rates_lmm.py](../caban/sp_rates_lmm.py) | Everything: tables, models, permutations, all figures, orchestrator |
| [caban/single_unit_common.py](../caban/single_unit_common.py) | `fit_mixed_model`, `joint_wald_test`, `draw_superplot_triplet`, `ecdf_panel`, `fdr_correct` |
| [caban/utilities.py](../caban/utilities.py) | `find_event_runs_ca{,_S}`, `find_spikes_ca{,_S}`, `barplot_annotate_brackets` |
| [caban/epoch_analysis.py](../caban/epoch_analysis.py) | `get_epoch_frames`, `get_testb_epoch_frames`, `TRACE_MATCHED_WINDOW_S`, `POST_SHOCK_LATE_ONSET_S`. `EPOCH_NAMES` here is the **PV/RDM pipeline's** epoch set, not a list of every window `get_epoch_frames` supports — `post_shock_late` is deliberately absent from it, since adding it would put a new row and column on every existing RDM figure |
| [caban/analysis.py](../caban/analysis.py) | `_draw_violin_triplet`, `do_pairwise_holm_plot` |
| [caban/pca_state_metrics.py](../caban/pca_state_metrics.py) | Also consumes `joint_wald_test` — inherits the df fix |
| [caban/sections.py](../caban/sections.py) | `run_sp_rates_lmm(ds, cfg)` |
| [caban/config.py](../caban/config.py) | `plot_sp_rates_lmm`, `sp_rates_lmm_n_perm`, `sp_rates_lmm_seed` |
| [analysis_methods_templates/sp_rates_lmm_methods.md](../analysis_methods_templates/sp_rates_lmm_methods.md) | METHODS text copied into every output dir at runtime |
| [analysis_methods_templates/sp_rates_lmm_figure_guide.md](../analysis_methods_templates/sp_rates_lmm_figure_guide.md) | Per-figure reading guide |
