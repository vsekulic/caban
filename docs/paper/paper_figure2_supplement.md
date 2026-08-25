# Supplementary Figure 2 — Methods and Results

Epoch-resolved decomposition of conditioning-day activity, and the drug-free recall analyses at
48 h and 1 week.

**Scope.** Companion to [`paper_figure2.md`](paper_figure2.md). Panels **a–c** extend Fig. 2e
(the trace-interval decomposition) to the other conditioning epochs; panels **d–e** are the
paper-facing **recall** lane, which is a separate analysis from conditioning and is reported here
because no CNO was present at recall. Methods common to both figures are not repeated — this
document states only what is specific to these panels and cross-references the rest.

**Provenance.** Every number was read from the run under
`PLOTS_DIR = .../plots/CURRENT/` (`sp_rates_lmm/`); §5 maps each to its file. No value comes from
`docs/sp_rates_lmm.md` §M.

**Panel lettering.** The supplied composite labels five panels **A–E**. The accompanying file list
named four items and labelled the recall by-epoch panel "C"; in the composite that content is
panel **D**, and **C** is the four-row decomposition grid. This document follows the composite.
See §6 note 1.

---

## 1. Panel map

| panel | content | source file (relative to `PLOTS_DIR/sp_rates_lmm/`) |
|---|---|---|
| **a** | Decomposition of population activity during the **pre-tone baseline** (20 s, matched) | `TFC_cond/decomposition_pre_tone_matched.png` |
| **b** | Decomposition during the **post-shock** window (20 s) | `TFC_cond/decomposition_post_shock.png` |
| **c** | Effect estimates with 95% CI for all four decomposition components × four epochs, with each component's BH-adjusted group × epoch *q* | `TFC_cond/decomposition_grid.png` |
| **d** | Recall, per-event amplitude (top) and population event rate (bottom), pre-tone vs post-tone. Left, 48 h (Test B); right, 1 week (Test B 1wk) | `paper/recall/Test_B/testb_amplitude_rate_by_epoch.png`; `paper/recall/Test_B_1wk/testb_1wk_amplitude_rate_by_epoch.png` |
| **e** | Per-animal pre-tone → post-tone modulation. Top, 48 h; bottom, 1 week | `paper/recall/Test_B/testb_amplitude_rate_modulation.png`; `paper/recall/Test_B_1wk/testb_1wk_amplitude_rate_modulation.png` |

Panels **a–c** are secondary/descriptive: they are one exact decomposition computed over
overlapping cells, and their intervals are uncorrected. Panels **d–e** are paper-facing, with the
same statistical framework as Fig. 2i.

---

## 2. Results

### The conditioning-day decomposition across epochs (Supplementary Fig. 2a–c)

The trace-interval decomposition of Fig. 2e was computed identically at the pre-tone baseline
(Supplementary Fig. 2a) and the post-shock window (Supplementary Fig. 2b), and all four components
are shown side by side across four epochs — including the late post-shock window, which has
reduced trial coverage — in Supplementary Fig. 2c.

**During the pre-tone baseline**, group differences were small and none of the
treatment-versus-control comparisons survived correction: fraction of cells active differed by
+0.023 (95% CI −0.062 to 0.107) for hM3D and −0.041 (−0.121 to 0.040) for hM4D; population event
rate was 0.95-fold (0.59 to 1.53) and 0.66-fold (0.39 to 1.10) of control; per-event amplitude was
1.41-fold (1.03 to 1.93) and 1.12-fold (0.77 to 1.63). The only annotated comparison in this epoch
was between the two DREADD groups, in the total deconvolved amplitude-rate (hM3D 1.60-fold,
95% CI 0.88 to 2.92 vs control; hM4D 0.81-fold, 0.49 to 1.35).

**During the post-shock window** the same dissociation seen at the trace interval was present:
per-event amplitude was 1.50-fold in hM3D (95% CI 1.06 to 2.13) and 1.22-fold in hM4D (0.86 to
1.74) relative to control, while population event rate was 1.11-fold (0.68 to 1.83) and 0.67-fold
(0.40 to 1.13). The two DREADD groups separated from each other on fraction active, event rate
among active cells, population event rate and total amplitude-rate, while neither differed from
control after correction on those quantities.

**Across all four epochs (Supplementary Fig. 2c)**, the pattern was one of parallel offsets rather
than epoch-specific effects: the hM3D per-event amplitude ratio was 1.38, 1.55, 1.47 and 1.77 at
pre-tone, trace, post-shock and late post-shock, and the hM4D population-rate ratio was 0.63, 0.51,
0.65 and 0.76 across the same windows, with intervals overlapping throughout. Accordingly, the
joint group × epoch permutation test was null for **every** component (BH-adjusted *q* = 0.717 for
fraction active, 0.738 for event rate among active cells, 0.753 for population event rate and
0.717 for per-event amplitude), matching the null interactions of the primary models (Fig. 2i).
These four components are one decomposition computed over overlapping cells, and the number of
panels whose interval excludes the null is not a meaningful count; epoch specificity has exactly
one test per component and it was null in all four. The direct hM3D-versus-hM4D contrasts are
displayed in Supplementary Fig. 2c because they are not derivable from the two
treatment-versus-control rows, which share the control as their reference; they are in no
multiplicity family.

### The conditioning-day phenotype at drug-free recall (Supplementary Fig. 2d,e)

Recall was analysed **in the absence of CNO** and **separately at each timepoint**, because the
animals available were not identical (16 animals at each session; the hM4D animal missing at 48 h
is not the one missing at 1 week). Each session used the same framework as Fig. 2i, applied to two
duration-matched 20 s windows — a pre-tone baseline and a post-tone retrieval window — with the
group × epoch interaction as the sole test of whether any group difference was preferentially
recruited by the tone.

**At 48 h (Test B; 16 animals, 5 hM3D / 5 hM4D / 6 mCherry, 7,689 cells, 3 of 3 tone trials
retained per animal), no within-epoch comparison reached significance after correction**
(Supplementary Fig. 2d, left). Per-event amplitude in the post-tone window was 1.41-fold in hM3D
relative to control (95% CI 0.96 to 2.07; raw *P* = 0.073, Holm-adjusted *P* = 0.145) and
0.98-fold in hM4D (0.67 to 1.43; adjusted *P* = 0.896); the two groups were indistinguishable from
control at pre-tone (0.94-fold, 0.64 to 1.38 and 0.92-fold, 0.63 to 1.35; both adjusted
*P* = 1.0). Population event rate showed no comparison approaching significance in either window
(post-tone hM3D 1.10-fold, 0.74 to 1.64, adjusted *P* = 0.621; hM4D 0.77-fold, 0.52 to 1.15,
adjusted *P* = 0.371). The group × epoch interaction was *F*(2,15) = 3.02, *P* = 0.079 for
per-event amplitude and *F*(2,15) = 0.44, *P* = 0.655 for population event rate.

**The pre-tone → post-tone modulation makes the shape of that amplitude interaction visible**
(Supplementary Fig. 2e, top). Control animals' per-event amplitude declined across the tone
(0.73-fold, 95% CI 0.56 to 0.93), whereas hM3D animals showed no detectable change (1.09-fold,
0.83 to 1.43) and hM4D animals showed no detectable change (0.77-fold, 0.59 to 1.02) — descriptive
within-group estimates, not a treatment comparison. Comparing those changes between groups, the
hM3D-versus-control difference in modulation was 1.50-fold (95% CI 1.03 to 2.18; unadjusted
model-derived *P* = 0.034; Holm-adjusted *P* = 0.103 as a multiplicity reference), hM4D versus
control 1.06-fold (0.73 to 1.54; *P* = 0.728) and hM3D versus hM4D 1.41-fold (0.96 to 2.08;
*P* = 0.078), beneath an omnibus of *F*(2,15) = 3.02, *P* = 0.079. Population event rate showed no
modulation difference on any comparison (hM3D vs control 1.16-fold, 0.79 to 1.72, *P* = 0.420;
hM4D vs control 1.00-fold, 0.68 to 1.48, *P* = 0.996; hM3D vs hM4D 1.17-fold, 0.78 to 1.75,
*P* = 0.437; omnibus *F*(2,15) = 0.44, *P* = 0.655).

**At 1 week (Test B 1wk; 16 animals, 5 hM3D / 5 hM4D / 6 mCherry, 9,076 cells, 3 of 3 trials
retained), no comparison reached significance in either outcome or either window**
(Supplementary Fig. 2d, right): post-tone per-event amplitude 1.11-fold in hM3D (95% CI 0.78 to
1.59; adjusted *P* = 0.543) and 1.21-fold in hM4D (0.85 to 1.73; adjusted *P* = 0.543); post-tone
population event rate 1.37-fold (0.76 to 2.45) and 0.83-fold (0.46 to 1.49), both adjusted
*P* = 0.543. Interactions were *F*(2,15) = 1.48, *P* = 0.260 (amplitude) and *F*(2,15) = 0.30,
*P* = 0.745 (rate). The modulation decomposition (Supplementary Fig. 2e, bottom) showed the same
qualitative arrangement as at 48 h with wider intervals: control animals declined across the tone
(0.80-fold, 0.66 to 0.98) while hM3D animals did not (1.00-fold, 0.80 to 1.25) and hM4D animals did
not (0.82-fold, 0.65 to 1.02), and no between-group comparison of that change was detectable
(hM3D vs control 1.25-fold, 0.92 to 1.68, *P* = 0.136; hM4D vs control 1.01-fold, 0.75 to 1.37,
*P* = 0.921; hM3D vs hM4D 1.23-fold, 0.90 to 1.68, *P* = 0.179; omnibus *F*(2,15) = 1.48,
*P* = 0.260). Population event rate again showed nothing (all *P* ≥ 0.451; omnibus *F*(2,15) = 0.30,
*P* = 0.745).

**These two sessions are not compared with each other**, and none of the above is evidence that an
effect declined, resolved or persisted between 48 h and 1 week: the two cohorts are not the same
animals, and a longitudinal claim requires a fixed-cohort model that was not fitted. Because no
CNO was present at either recall session, any recall difference is a persistent consequence of the
conditioning-day manipulation and not evidence of ongoing receptor activation.

---

## 3. Methods

Event detection, amplitude definition, animal-level summarization, the mixed-model specification,
the within-epoch Holm families and the interpretive constraints are as described for Fig. 2 and
are not repeated. What follows is specific to these panels.

### Epoch-resolved decomposition (Supplementary Fig. 2a–c)

Panels **a** and **b** are the Fig. 2e decomposition — fraction of cells active × event rate among
active cells = population event rate, plus per-event amplitude and their product, the total
deconvolved amplitude-rate — computed with the identical code and identical annotation procedure
at the matched 20 s pre-tone baseline and the 20 s post-shock window respectively. As in Fig. 2e,
fraction active is an animal-level quantity; the other four are per-cell quantities drawn as
SuperPlots with all statistics computed from the per-animal means; brackets are Welch two-sample
tests on those means, Holm-corrected across the two treatment-versus-control comparisons, with the
hM3D-versus-hM4D comparison computed and displayed uncorrected; and the estimates and uncorrected
95% intervals are written to the companion `*_contrasts.md` files.

Panel **c** displays the same estimator across four epochs at once — pre-tone (matched), trace,
post-shock and late post-shock — as effect estimates with 95% confidence intervals, with **no
significance stars by design**. It includes the direct hM3D-versus-hM4D contrast, which is not
derivable from the two treatment-versus-control rows because those share the control as their
reference, and is in no multiplicity family. The late post-shock window (90–110 s after shock
offset) is absent on trials whose recording stopped early and is labelled as having reduced
coverage; it enters no primary model.

Each **row** of panel **c** is labelled with that component's group × epoch specificity test: a
mouse-label permutation test of the standardized sum of squared difference-of-differences across
epochs, with each animal keeping its whole epoch profile (20,000 permutations, seed 0), computed
on within-cell-paired values where the component admits pairing and reported after
Benjamini–Hochberg adjustment within the 13-member secondary family. Because the four components
are one exact decomposition computed over overlapping cells, their *q* values are strongly
dependent and are read as a set describing one decomposition, never as four independent findings;
comparing significance between epochs within a row is the difference-of-significance fallacy and
epoch specificity has exactly one test per component.

### Recall analyses (Supplementary Fig. 2d,e)

Recall was analysed at 48 h (Test B) and 1 week (Test B 1wk) **independently**, because the
animals available at the two sessions were not identical (16 each; the animal absent at 48 h is
not the animal absent at 1 week). No direct 48 h versus 1 week comparison was made, and none is
approximated by placing both sessions on one figure; a longitudinal comparison would require a
separate fixed-cohort model restricted to animals present at both sessions. **No CNO was
administered before either recall session.**

Event detection was identical to conditioning and was not re-tuned. Two duration-matched **20 s**
windows were used around each tone: a **pre-tone** window ending at tone onset, and a **post-tone**
window beginning at tone offset — the retrieval analogue of the conditioning trace interval. The
tone epoch itself was not analysed, and neither the legacy 35 s post-tone window nor the
run-to-the-next-tone interval was used, since the two epochs must be duration-matched for a
duration-sensitive rate outcome. Analyses were restricted, within each animal, to tone trials on
which both windows were completely observed, determined from each window's measured extent rather
than from a trial index; a post-tone window running past the end of the recording or into the next
tone was treated as absent, and a pre-tone window overlapping the previous trial's post-tone
window raised rather than being silently accepted. All 16 animals retained all 3 tone trials in
both sessions. Both outcomes were computed from that same set of animal × trial windows.

For each session, per-event amplitude and population event rate were analysed separately with the
same model as conditioning, `log(metric) ~ group * epoch + (1|animal)`, reference levels mCherry
and pre-tone. Within each epoch, hM3D and hM4D were compared with mCherry by linear contrasts of
the fitted model — the group coefficient alone at pre-tone, and that coefficient plus the group ×
post-tone interaction coefficient, with their covariance, at post-tone — Holm-corrected across the
two treatment-versus-control comparisons in that epoch, with amplitude and rate as separate
families and the hM3D-versus-hM4D comparison in neither. **Retrieval preferentiality was tested by
the group × epoch joint Wald test of the two interaction coefficients and by nothing else**: a
significant post-tone difference under a null interaction does not establish that a difference is
tone-evoked. All inference used that session's own animal-level denominator degrees of freedom,
*n*<sub>animals present</sub> − 1 = 15. The across-epoch six-comparison sensitivity family used in
conditioning was not computed for recall, where no such family was ever declared. The large
markers in Supplementary Fig. 2d are the exact per-animal values the models were fitted on,
verified programmatically against the model input table.

**Pre-tone → post-tone modulation (Supplementary Fig. 2e).** To display the interaction directly,
each animal's change in the log outcome (post-tone − pre-tone) was computed from the same
animal-level table the models were fitted on, and the three pairwise between-group comparisons of
that change were obtained as **linear contrasts of the same fitted models — no new model was
fitted and no statistic was computed from the plotted change scores**. Against mCherry each
comparison is that group's group × epoch coefficient; between hM3D and hM4D it is the difference
of their two interaction coefficients, computed with their covariance rather than by subtracting
two published standard errors. All three comparisons were computed for both outcomes; which are
bracketed on the figure is a display choice over that complete table and changes no computed
value. **The reported *P* values for these comparisons are unadjusted model-derived contrasts**;
they were not prospectively preregistered, and a Holm adjustment across the three comparisons
within each outcome (amplitude and rate as separate three-member families) is tabulated alongside
as a multiplicity reference only. Each group's own model-implied pre-to-post change is reported
**descriptively** to characterize the trajectory and is not a between-group test — that one group's
interval excludes zero and another's does not is not a test that the two differ. The group × epoch
joint Wald test remains the omnibus, is quoted beside every pairwise value, and does not gate the
contrasts.

Three questions are kept apart throughout and can disagree in both directions: whether groups
differ *within* an epoch (the within-epoch contrasts, Supplementary Fig. 2d); whether a group
changes at all across the tone (the within-group estimates, descriptive); and whether the change
*differs between* groups (the pairwise contrasts and their omnibus, Supplementary Fig. 2e). Equal
post-tone levels reached from unequal baselines is a difference in modulation without a difference
in level; parallel decline from unequal baselines is the reverse.

**Not included in this pass.** No cross-session comparison; no cross-registered cell-identity
persistence analysis; no responder classification; no tone-epoch statistics; no negative-binomial
count model as a recall rate sensitivity analysis; and no hierarchical cell-level companion
analysis — that suite is gated off by default and was not run for the output reported here.

---

## 4. Figure legend

**Supplementary Figure 2 | Epoch-resolved decomposition of conditioning-day activity, and CA1
calcium event amplitude and rate during drug-free recall.**

**a,b**, Decomposition of population calcium activity during the matched 20 s pre-tone baseline
(**a**) and the 20 s post-shock window (**b**), in the same five quantities and the same format as
Fig. 2e: fraction of cells active; event rate among active cells; population event rate over all
cells (the product of the first two); per-event amplitude; and total deconvolved amplitude-rate.
Small points, individual cells (description only); large markers, per-animal means, which are the
values tested. *n* = 5 hM3D, 6 hM4D, 6 mCherry animals. **c**, The same four components across
four conditioning epochs, as effect estimates with 95% confidence intervals; rows are labelled with
that component's Benjamini–Hochberg-adjusted group × epoch specificity *q*. The Exc/Inh row is the
direct DREADD-versus-DREADD contrast, which is not derivable from the two rows above it. No
significance stars are drawn on this panel by design; the late post-shock column has reduced trial
coverage. **d**, Recall. Per-event amplitude (top) and population event rate (bottom) in the 20 s
pre-tone and 20 s post-tone windows, at 48 h (left, Test B) and 1 week (right, Test B 1wk); small
points, individual cells (description only); large markers, the per-animal values the models were
fitted on. **e**, Per-animal change across the tone (post-tone − pre-tone) in log per-event
amplitude and log population event rate, at 48 h (top) and 1 week (bottom). One point per animal;
dashed line, no change; horizontal line, median. Brackets are unadjusted model-derived contrast
*P* values from the fitted model of that session — not multiplicity-adjusted, and no statistic is
computed from the plotted values; each panel's group × epoch omnibus is given in its title.
*n* = 16 animals at each recall session (5 hM3D, 5 hM4D, 6 mCherry); the two sessions are not the
same animals and are never compared. **No CNO was present at either recall session.** Brackets in
**a** and **b** are Welch tests on per-animal means, Holm-corrected across the two
treatment-versus-control comparisons, with the hM3D-versus-hM4D comparison shown uncorrected.
**P* < 0.05, ***P* < 0.01.

---

## 5. Number-to-source lookup

Paths are relative to `PLOTS_DIR/sp_rates_lmm/`.

| value in §2 | source |
|---|---|
| all panel **a** estimates and intervals | `TFC_cond/decomposition_pre_tone_matched_contrasts.md` |
| all panel **b** estimates and intervals | `TFC_cond/decomposition_post_shock_contrasts.md` |
| panel **c** estimates, intervals and Exc/Inh contrasts (4 components × 4 epochs) | `TFC_cond/decomposition_grid_contrasts.md` |
| panel **c** row *q* values 0.717 / 0.738 / 0.753 / 0.717 | `TFC_cond/stats/secondary_epoch_interaction.txt`, `TFC_cond/stats/secondary_fdr_family.txt` |
| all recall within-epoch ratios, CIs and Holm-adjusted *P* | `paper/recall/{Test_B,Test_B_1wk}/stats/unified_recall_posthoc_contrasts.csv` |
| recall interactions *F*(2,15) = 3.02 / 0.44 (48 h), 1.48 / 0.30 (1 wk) | `paper/recall/*/stats/unified_recall_interactions.csv` |
| within-group and pairwise modulation estimates, intervals and both *P* columns | `paper/recall/*/stats/unified_recall_modulation_contrasts.md` (sections B and C) and `...contrasts.csv` |
| per-animal change scores behind panel **e** | `paper/recall/*/stats/unified_recall_modulation_by_mouse.csv` |
| 16 animals per session, 3 of 3 trials retained, cohort membership | `paper/recall/*/stats/unified_recall_trial_coverage.csv` |
| recall cell counts 7,689 (48 h) and 9,076 (1 wk); 32-row datasets | `paper/recall/*/stats/unified_recall_mouse_epoch_values.csv` |

---

## 6. Points flagged for the author (not changed here)

1. **Panel lettering.** Your file list gave four items (A, B, C, E) and labelled the recall
   by-epoch figure "C"; the composite shows five panels, with the decomposition grid as **c** and
   the recall by-epoch pair as **d**. This document follows the composite. If the grid is not
   meant to be in this figure, drop §2's third paragraph, the panel **c** Methods block and the
   corresponding legend sentence — nothing else depends on it.
2. **Panel c is not in the file list you sent** but is present in the composite; its source is
   `TFC_cond/decomposition_grid.png` with `decomposition_grid_contrasts.md` beside it.
3. **Panels a and b asterisks have no *P* value on disk**, exactly as for Fig. 2e: the brackets
   are computed inside the plotting call and only estimates and intervals are written out. The
   Results text therefore quotes estimates and intervals for those panels and no *P*. Note that in
   both epochs **every** drawn asterisk is an hM3D-versus-hM4D bracket, which is the *uncorrected*
   comparison — worth an explicit sentence in the legend if reviewers are likely to read those
   stars as treatment-versus-control.
4. **Panel b, per-event amplitude**: hM3D versus control is 1.50-fold with an interval excluding 1
   (1.06 to 2.13) but carries no star after Holm correction. Stated as an estimate with its
   interval above, with no significance claim.
5. **Recall modulation *P* values are unadjusted by design** (`docs/sp_rates_lmm.md` §A.7.1). The
   one comparison below 0.05 — hM3D versus control amplitude modulation at 48 h, *P* = 0.034 —
   sits beneath a non-significant omnibus (*P* = 0.079) and has a Holm reference value of 0.103.
   It is written above as weak evidence with all three numbers attached; do not let it compress to
   "significantly different" in the manuscript.
6. **The 48 h rate trajectory has an outlier-looking within-group test.** The hM4D paired
   *t*-test on the trajectory figure is *t*(4) = −14.06, *P* = 1.5 × 10⁻⁴ — a within-group
   question (B), from a different estimator than the model contrast (which gives 0.83-fold,
   *P* = 0.195 for the same group). That figure is not part of this supplement's panels, but if you
   add it, the two estimators' disagreement needs the sentence the companion markdown already
   supplies.
7. **The hierarchical cell-level companion analysis was not run** in this output (no
   `hierarchical_cells/` directory under either recall session), so no number from it appears here.
   If you want its paired-cell amplitude sensitivity result in the supplement, that lane has to be
   rerun with `run_hierarchical_cell_analysis=True`.
