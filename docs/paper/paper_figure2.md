# Figure 2 — Methods and Results

Chemogenetic modulation of dorsal CA1 SST interneurons dissociates the size and the frequency of
pyramidal-cell calcium events during trace fear conditioning.

**Scope.** This document is the manuscript text for **Figure 2 only** — Results, Methods, and the
figure legend, in *Nature* / *Nature Neuroscience* register (past tense, effect size with its
interval before any *P*, the animal named as the unit of inference). It is written from the
paper-facing conditioning lane of `caban.sp_rates_lmm` (`docs/sp_rates_lmm.md` Part A) plus the
internal `TFC_cond` panels the figure borrows for description. Recall (Test B, Test B 1 wk) is a
separate figure and no recall number appears here — see
[`paper_figure2_supplement.md`](paper_figure2_supplement.md), which also carries the
epoch-resolved decomposition panels.

**Provenance of every number below.** All values were read from the run under
`PLOTS_DIR = .../plots/CURRENT/` (`sp_rates_lmm/`), not from `docs/sp_rates_lmm.md` §M, whose
worked example predates this run in places. §5 maps every number to its file, row and column.
Items still owed by other sections of the paper (surgery, viral constructs, CNO dosing,
miniscope hardware, Minian parameters) are marked ⟨…⟩ rather than invented here.

---

## 1. Panel map

| panel | content | source file (relative to `PLOTS_DIR`) |
|---|---|---|
| **a** | Injection strategy: bilateral dorsal CA1, SST-Cre⁺ᐟ⁺; AAV.Syn.SomaGCaMP6f.f2 + AAV.DIO.hM3D(Gq).mCherry / hM4D(Gi).mCherry / EF1a.mCherry; GRIN lens | schematic ⟨not produced by this code base⟩ |
| **b** | Miniscope field of view (left, mean fluorescence) and Minian spatial footprints (right) | ⟨Minian output; not produced by `sp_rates_lmm`⟩ |
| **c** | Example simultaneous traces: ΔF/F, denoised **C**, deconvolved **S** with detected events | `plot_sample_traces/plot_sample_traces.png` |
| **d** | Manipulation check, LT1 (drug-free) → LT2 (CNO): within-cell Δlog(amplitude) and cross-session detection dropout | `sp_rates_lmm/TFC_cond/manipulation_check.png` |
| **e** | Decomposition of population activity during the trace interval: fraction active × event rate among active cells = population event rate; per-event amplitude; total deconvolved amplitude-rate | `sp_rates_lmm/TFC_cond/decomposition.png` |
| **f** | Peak-height-matched example event runs, one per group | `sp_rates_lmm/TFC_cond/width_height_matched_examples_vertical.png` |
| **g** | Descriptive per-animal amplitude profile across five disjoint conditioning epochs | `sp_rates_lmm/TFC_cond/epoch_profile.png` |
| **h** | Trace-interval per-cell amplitude ECDF, pooled and per animal | `sp_rates_lmm/TFC_cond/amplitude_ecdf.png` |
| **i** | **Primary analysis.** Per-event amplitude (top) and population event rate (bottom) by group, in each of the three duration-matched epochs | `sp_rates_lmm/paper/tfc_amplitude_rate/tfc_amplitude_rate_by_epoch.png` |

Panels **e–h** are descriptive or secondary; panel **i** carries the manuscript's inferential
claims. Panel **d** is a manipulation check and is in no multiplicity family with either.

---

## 2. Results

### Chemogenetic modulation of SST interneurons dissociates the magnitude and the frequency of CA1 pyramidal calcium events

To ask how modulating dorsal CA1 somatostatin-expressing (SST) interneurons reshapes pyramidal
output during trace fear conditioning, we expressed a soma-targeted calcium indicator
(SomaGCaMP6f) together with a Cre-dependent excitatory (hM3D(Gq)), inhibitory (hM4D(Gi)) or
control (mCherry) DREADD in dorsal CA1 of SST-Cre mice, and imaged the same field of view through
a GRIN lens across the conditioning and retrieval sessions (Fig. 2a,b). Regions of interest and
their denoised and deconvolved activity traces were extracted with Minian and registered across
sessions (Fig. 2b,c).

Because the summed deconvolved signal per second is the exact product of event rate and per-event
amplitude, a single "activity" measure cannot distinguish larger events from more frequent ones.
We therefore defined a calcium event as one **contiguous supra-threshold run** of the deconvolved
trace *S* and quantified it by the **integral of *S* over that run**, so that a wider event is
distinguishable from a taller one (Fig. 2c,f), and analysed per-event amplitude and population
event rate as two separate outcomes given one identical statistical treatment. Throughout, the
**animal** is the unit of inference (*n* = 5 hM3D, 6 hM4D, 6 mCherry); per-cell distributions are
shown for description only, and every statistic was computed from per-animal values.

**The manipulation was effective (Fig. 2d).** In cells tracked across two same-day linear-track
sessions recorded before (LT1, drug-free) and after (LT2) CNO administration (1,751 paired cells,
17 animals), hM3D expression increased per-event amplitude relative to control: the
difference-in-differences against mCherry was +0.474 log units (95% CI 0.289 to 0.659), whereas
hM4D did not differ detectably from control (+0.142, 95% CI −0.032 to 0.317; group omnibus
*F*(2,16) = 13.02, *P* = 4.4 × 10⁻⁴). Control cells declined by 0.270 log units across the session
pair (95% CI −0.395 to −0.146), consistent with photobleaching and with the session order that
every group experienced equally; the group contrast against control, not the raw LT2 − LT1
change, is what isolates the CNO-specific effect. Because region-of-interest detection is
activity-dependent, we also measured the fraction of LT1 cells that failed to re-register into
LT2. Dropout was comparable across groups (mCherry 0.768 ± 0.038, hM4D 0.771 ± 0.055, hM3D
0.817 ± 0.025; mean ± s.d. of per-animal fractions), arguing against selective loss of silenced
cells as an explanation for the hM4D results below.

**The two treatments moved different terms of the activity decomposition (Fig. 2e).** During the
trace interval we decomposed population activity into its exact components — fraction of cells
active × event rate among active cells = population event rate — alongside per-event amplitude
and their product, the total deconvolved amplitude-rate. hM3D raised per-event amplitude
(1.53-fold vs control, 95% CI 1.14 to 2.05; +0.425 log units, 95% CI 0.132 to 0.718) while hM4D
did not (1.18-fold, 95% CI 0.85 to 1.65). Conversely, hM4D lowered the population event rate
(0.54-fold, 95% CI 0.36 to 0.82; −0.037 events s⁻¹ per cell, 95% CI −0.069 to −0.005) with hM3D
intermediate and not separated from control (0.79-fold, 95% CI 0.55 to 1.16). Both DREADDs
reduced the event rate among active cells (hM3D 0.73-fold, 95% CI 0.56 to 0.95; hM4D 0.56-fold,
95% CI 0.41 to 0.77), and the fraction of active cells differed between the two DREADD groups
(+0.053 hM3D vs control, −0.038 hM4D vs control) without either separating from control. The
product term inherits the dissociation: total deconvolved amplitude-rate was 1.55-fold in hM3D
(95% CI 0.93 to 2.59) and 0.74-fold in hM4D (95% CI 0.46 to 1.19) relative to control, and the two
DREADD groups differed from each other. These five quantities are **one decomposition computed
over overlapping cells**, not five independent phenotypes, and are read as a set.

**The amplitude effect was not simply taller events (Fig. 2f,h).** Selecting, per group, the
single detected run whose peak height was closest to the pooled median peak height, the hM3D
example run remained visibly wider than its height-matched control and hM4D counterparts
(Fig. 2f) — the qualitative signature of temporally clustered activity merging into one wider,
larger-integral event. Across all trace-interval runs, mean run width was 5.48 ± 0.34 frames in
hM3D versus 4.87 ± 0.82 in control and 4.75 ± 0.29 in hM4D (mean ± s.d. of per-animal means, 20 Hz
sampling), a difference **not resolved** at this sample size (mouse-label permutation
*P* = 0.096, BH-adjusted *q* = 0.62), and the fraction of runs containing more than one local
maximum did not differ between groups (hM3D 0.046 ± 0.010, control 0.037 ± 0.016; *P* = 0.25,
*q* = 0.72). A burst-like origin for the larger events is therefore **suggested but not
established**. Consistent with a shift of the whole distribution rather than of its tail alone,
the per-cell amplitude ECDF during the trace interval (Fig. 2h; 2,804 hM3D, 2,273 hM4D and 2,333
control cells) was displaced rightward in hM3D across its full range, with the per-animal curves
tracking the group curves; the animal-level 90th-percentile amplitude was shifted by the same
order as the mean (+0.404 log units in hM3D vs control, mouse-label permutation *P* = 0.0067,
against +0.425 for the mean, *P* = 0.0083).

**The amplitude difference was present across conditioning epochs, not confined to one (Fig. 2g,i).**
The descriptive per-animal amplitude profile across five disjoint epochs (pre-tone, tone, trace,
post-shock, late post-shock) showed the three groups running approximately in parallel, with hM3D
elevated at every epoch and hM4D intermediate (Fig. 2g). We then fitted the primary models: for
each outcome separately, `log(metric) ~ group × epoch + (1|animal)` over one value per animal per
epoch (51 rows: 17 animals × 3 duration-matched 20 s epochs), with hM3D and hM4D compared with
mCherry within each epoch by linear contrasts of the fitted model, Holm-corrected across those two
treatment-versus-control comparisons in that epoch (Fig. 2i).

hM3D increased per-event amplitude relative to control during the **trace** interval (1.55-fold,
95% CI 1.11 to 2.16; Holm-adjusted *P* = 0.026) and during the **post-shock** window (1.47-fold,
95% CI 1.06 to 2.06; *P* = 0.050); the pre-tone comparison did not reach significance (1.38-fold,
95% CI 0.99 to 1.93; *P* = 0.113). hM4D did not significantly alter per-event amplitude in any
epoch (trace 1.18-fold, 95% CI 0.86 to 1.62, *P* = 0.281; pre-tone 1.14-fold, *P* = 0.404;
post-shock 1.18-fold, *P* = 0.277) (Fig. 2i, top).

Population event rate showed the complementary pattern. hM4D reduced the event rate during the
**trace** interval (rate ratio 0.51, 95% CI 0.33 to 0.77; absolute difference −0.039 events s⁻¹
per cell; Holm-adjusted *P* = 0.0069), whereas the pre-tone (0.63, 95% CI 0.42 to 0.97,
*P* = 0.073) and post-shock (0.65, 95% CI 0.43 to 1.00, *P* = 0.098) comparisons did not survive
correction. hM3D did not significantly alter population event rate in any epoch (trace rate ratio
0.78, 95% CI 0.50 to 1.21, *P* = 0.246; pre-tone 0.89, *P* = 0.586; post-shock 1.10, *P* = 0.646)
(Fig. 2i, bottom).

Despite these epoch-wise simple effects, **there was no evidence that the magnitude of either
treatment effect differed across epochs**, for per-event amplitude (group × epoch joint Wald test
*F*(4,16) = 0.37, *P* = 0.828) or for population event rate (*F*(4,16) = 1.54, *P* = 0.237); an
independent mouse-label permutation test of epoch specificity agreed (amplitude *T* = 1.99,
*P* = 0.27, *q* = 0.72; population rate *T* = 0.76, *P* = 0.64, *q* = 0.75). A significant
comparison in one epoch and not in another is not itself evidence of epoch dependence, which was
tested by the interaction and by nothing else. The data therefore support **differential effects
of SST-interneuron excitation and inhibition on the magnitude versus the frequency of CA1
pyramidal calcium events** — hM3D predominantly on event magnitude, hM4D predominantly on event
frequency — but do not demonstrate that either effect was specific to a particular conditioning
epoch. At *n* = 5/6/6 animals the interaction intervals remain wide, so this is an absence of
evidence for epoch specificity, not evidence of a uniform, tonic effect.

Two further analyses, run independently of the mixed models, agreed. A distribution-aware
negative-binomial mixed-effects count model of the same event counts (log cell-seconds offset,
random intercepts for animal and animal × trial) reproduced the direction and approximate size of
the rate result: trace-interval rate ratio 0.58 (94% HDI 0.35 to 0.83) for hM4D and 0.85 (94% HDI
0.50 to 1.26) for hM3D, with leave-one-out comparison giving no support for the group × epoch
interaction over the additive model (Δelpd 1.50, s.e. of the difference 4.51, favouring the
model without the interaction). And a cell-level model of the trace interval, with animal-clustered
inference, gave the same amplitude ordering (group omnibus *F*(2,16) = 6.44, *P* = 0.0089;
Holm-adjusted *P* = 0.027 across that lane's three-member family), while the two within-cell
epoch-change contrasts — trace minus pre-tone and post-shock minus pre-tone, computed on the
7,067 and 7,609 cells active in both windows — were null (*F*(2,16) = 0.002, *P* = 0.998 and
*F*(2,16) = 0.067, *P* = 0.935; both Holm-adjusted *P* = 1.0), consistent with the parallel profile
in Fig. 2g and with the null interaction above.

**Controls.** The amplitude effect was stable across event-detection thresholds spanning 1.5–3.0
in units of the deconvolved trace (hM3D group coefficient 0.429, 0.435 and 0.408 log units at
thresholds 1.5, 2.0 and 3.0, each with an interval excluding zero), excluding a threshold-induced
run-merging artifact as its source. Per-event amplitude declined monotonically across the five
conditioning trials in every group at closely matched rates (−0.076, −0.092 and −0.087 log units
per trial in hM3D, hM4D and mCherry), consistent with photobleaching; this is a main effect of
trial that cancels in a between-group contrast (group × trial interaction *F*(8,16) = 0.21,
*P* = 0.98). Finally, analyses of this kind are conditional on the neurons that source extraction
detects: the absence of an hM4D amplitude effect cannot exclude changes in neurons that became
undetectable, although the matched cross-session dropout in Fig. 2d makes such loss unlikely to
account for it.

---

## 3. Methods

*(Sections in ⟨…⟩ belong to other parts of the Methods and are not written here.)*

### Animals, surgery and viral constructs

SST-Cre⁺ᐟ⁺ mice received bilateral dorsal CA1 injections of AAV.Syn.SomaGCaMP6f.f2 together with
one of AAV.DIO.hM3D(Gq).mCherry, AAV.DIO.hM4D(Gi).mCherry or AAV.DIO.EF1a.mCherry, followed by
implantation of a GRIN lens over dorsal CA1 (Fig. 2a). ⟨Serotypes, titres, volumes, coordinates,
lens dimensions, recovery interval, expression interval, CNO dose, route and pre-session timing,
and the histological verification of injection and lens placement.⟩ Group sizes were *n* = 5
hM3D, *n* = 6 hM4D and *n* = 6 mCherry; all 17 animals contributed to every conditioning analysis
reported in Fig. 2.

### Miniscope imaging and source extraction

⟨Miniscope model, objective/GRIN configuration, field of view, illumination power, gain and
acquisition software.⟩ Imaging was performed at 20 Hz. Regions of interest, their denoised
calcium traces (*C*) and their deconvolved activity traces (*S*) were extracted with **Minian**, a
CNMF-E implementation (⟨version and parameter set⟩; Fig. 2b). Cells were registered across
sessions using Minian's cross-registration output, and cross-session correspondence was taken from
the registration table directly by row position; no analysis matched cells across sessions by
within-session identifier. An automated quality-control pass, applied at load, excluded ROIs
failing right-skewness, hyperactivity/sparsity, plateau-artifact or minimum-event-count criteria
(thresholds: skewness ≥ 1.5, plateau ≤ 15 s, ≥ 3 detected peaks).

### Calcium event detection and measurement

Calcium events were detected on the deconvolved trace *S* of each pyramidal cell. **One event was
defined as one contiguous run of *S* at or above a fixed threshold of 2 in the arbitrary units of
*S***, and its amplitude as the **integral of *S* over that run**. This departs from the more
common convention — one event per local maximum, quantified by the value at its peak frame — for a
specific reason: summed over a window, the peak-based quantity is exactly the product of event
rate and mean peak height and therefore cannot distinguish larger events from more frequent ones.
Under the run definition, temporally clustered peaks merge into a single wider event of larger
integral (median run length 4 frames; 11.7% of runs confined to a single frame), so event counts
are not interchangeable between the two definitions and were recomputed under this one throughout.
Throughout, the measurement is the integral of a deconvolved event run; "burst" is an
interpretation of that measurement and not a synonym for it.

For each run we additionally recorded its width in frames, its first and argmax frames, its peak
height, and the number of conventional local maxima falling inside it; these supported the
run-structure analyses and the example panels (Fig. 2c,f) and entered no primary model.

Cells with no detected events in a window were **retained in the population event-rate
denominator**, so that a manipulation silencing cells appears as a reduced population rate.
Per-event amplitude is undefined for such cells, which are therefore necessarily absent from
amplitude analyses; this is a definitional exclusion rather than missing data.

### Behavioural epochs

Trial timing was measured per animal from each recording rather than taken from nominal protocol
constants, as recordings were ragged and trial counts varied. The primary analyses used three
duration-matched **20 s** windows of trace fear conditioning: a **pre-tone baseline** ending at
tone onset, the **trace interval** (tone offset to shock onset), and a **post-shock** window
beginning at shock offset. The 2 s shock itself was excluded from all models: at the observed
event rates, expected counts in a 2 s window are dominated by counting noise (P(zero events per
cell) 67–91% at 0.05–0.2 Hz).

Because the trace interval on the first trial was 15 s rather than 20 s, analyses were restricted
for each animal to trials in which all three windows were present at the full 20 s duration,
determined from each window's measured exposure rather than from a trial index (3–4 of 5 trials
per animal). **Both outcomes were computed from that same set of animal × trial windows**, so the
amplitude and rate rows of Fig. 2i provably describe the same animals, epochs and trials.

The descriptive epoch profile (Fig. 2g) instead used the five mutually disjoint windows the
recording supports — pre-tone, tone, trace, post-shock (0–20 s from shock offset) and late
post-shock (90–110 s from shock offset) — and is descriptive only; no *P* value in the Results is
taken from it.

### Statistical analysis — primary conditioning models (Fig. 2i)

**The animal was the unit of inference** (*n* = 5 hM3D, 6 hM4D, 6 mCherry). Cells and trials
contributed precision, not replication, and no analysis treated cells as independent experimental
units. Figures showing per-cell distributions display them for description only; all statistics
were computed from per-animal values (SuperPlot convention; Lord et al., *J. Cell Biol.* **219**,
e202001064, 2020). The inferential dataset contained one value per animal per epoch (51 rows;
9,531 detected cells contributed).

For each animal and epoch, **per-event amplitude** was summarized by first calculating each active
cell's mean event-run integral across the retained trials, log-transforming that value, and then
averaging across cells within the animal. Because the response is a within-animal mean of
cell-level logarithms, an exponentiated group contrast is a **ratio of geometric means** of the
cell-level mean event amplitudes, not a ratio of pooled arithmetic means. **Population event
rate** was calculated for the same trials as the total number of detected events divided by the
corresponding total cell-seconds across all detected pyramidal cells, including cells with zero
events, and was log-transformed before analysis. No animal × epoch had a zero population rate and
no pseudocount was applied.

Per-event amplitude and population event rate were analysed separately with the **same linear
mixed-effects model**: DREADD group (mCherry, hM3D, hM4D), epoch (pre-tone, trace, post-shock) and
their interaction as fixed effects, animal as a random intercept
(`log(metric) ~ group * epoch + (1|animal)`; reference levels mCherry and pre-tone, set
explicitly), fitted by restricted maximum likelihood (statsmodels `MixedLM`, L-BFGS).

Within each epoch, hM3D and hM4D were compared with mCherry by **linear contrasts of the fitted
model** — at the reference epoch the group coefficient alone, and at the trace and post-shock
epochs that coefficient plus the corresponding group × epoch interaction coefficient, with their
covariance — with **Holm correction across the two treatment-versus-control comparisons within
that epoch**. Amplitude and rate formed separate correction families, and the direct
hM3D-versus-hM4D comparison was in neither. Group × epoch dependence was assessed once per outcome
by a **joint Wald test of the four interaction coefficients**. All Wald and contrast inference used
an animal-level denominator degrees of freedom of *n*<sub>animals</sub> − 1 = 16, applied
identically to both outcomes; no Satterthwaite or Kenward–Roger approximation was applied.

Effects are reported as exponentiated model contrasts (fold change or rate ratio) with 95%
confidence intervals; for amplitude this is the ratio of animal-level geometric mean event
amplitudes, for rate a rate ratio. Population-rate effects are additionally reported as observed
absolute differences in events s⁻¹ per cell — equal-animal-weighted descriptive summaries carrying
no separate test, reported because a rate ratio off a small base overstates the practical size of
a change. *P* < 0.05 after Holm correction was considered statistically significant. The large
markers in Fig. 2i are the exact per-animal values the models were fitted on, verified
programmatically against the model input table.

As a conservative **sensitivity analysis**, the six treatment-versus-control simple effects
spanning all three epochs were additionally Holm-corrected together within each outcome. Under
that wider family only the hM4D population-rate reduction during the trace interval remained
significant (*P* = 0.021). This sensitivity correction determined no figure annotation and no
significance statement in the Results.

Locomotion and freezing were **not** covaried: both are post-treatment variables, and conditioning
on them would remove part of the effect being estimated. One cross-registration cell set was
primary; the others were sensitivity analyses, not replications.

### Statistical analysis — manipulation check (Fig. 2d)

Within-cell change in log mean event amplitude, LT2 minus LT1, was computed on the cells
cross-registered between the two same-day linear-track sessions (LT1 drug-free, LT2 on CNO; same
field of view, minutes apart). Cells with no detected event in either session were excluded, the
change being undefined for them. The paired changes were modelled as
`Δlog(amplitude) ~ group + (1|animal)` (1,751 paired cells, 17 animals), with the group contrast
against mCherry — a difference-in-differences, since LT1 always preceded LT2 and session order,
elapsed time, habituation and photobleaching therefore differ between the sessions for every group
alike — as the reported effect, and a joint Wald test of both group coefficients on df = 16 as the
omnibus. This is a manipulation check, not a test of the memory hypothesis, and is in no
multiplicity family with the primary analysis. Cross-session detection dropout was computed per
animal as the fraction of LT1-detected cells not successfully registered into LT2. Panel brackets
in Fig. 2d are Welch two-sample tests on the per-animal means, Holm-corrected across the two
treatment-versus-control comparisons.

### Statistical analysis — descriptive and secondary panels (Fig. 2e–h)

Panel **e** decomposes trace-interval activity into fraction of cells active, event rate among
active cells, population event rate over all cells (their exact product), per-event amplitude, and
total deconvolved amplitude-rate (the product of the last two). Fraction active is a proportion
computed over an animal's cells and is shown as an animal-level distribution; the other four are
per-cell quantities shown as SuperPlots, with every cell drawn for description, the per-animal
means overlaid as large markers, and **all statistics computed from those animal means alone**.
Rate panels carry sub-quantum vertical jitter (1/exposure-seconds) for display only. Brackets are
Welch two-sample tests on the per-animal means, Holm-corrected across the two
treatment-versus-control comparisons; the hM3D-versus-hM4D comparison is computed and displayed
uncorrected, as it spends no α of its own in this design. Estimates and 95% intervals for every
panel are reported in the companion file `decomposition_contrasts.md` and are **not**
multiplicity-corrected, so a contrast whose interval excludes 1 may carry no asterisk; both are
reported deliberately — the interval describes the effect, the asterisk describes the corrected
decision. These five quantities are one exact decomposition computed over overlapping cells and
are interpreted as a set, never as five independent findings.

Panel **f** shows, per group, the single trace-interval event run whose peak height was closest to
the 50th percentile of peak height pooled across all groups, with ties broken deterministically by
(animal, cell, trial) order; shading marks the detected run, re-derived from the raw *S* trace by
the same detector, and the dotted line marks the detection threshold. Panel **g** plots each
animal's pooled mean log per-event amplitude at each of the five disjoint epochs (thin lines) with
the group mean ± s.e.m. (bold); it is descriptive and carries no test. Panel **h** plots the
pooled per-cell ECDF of trace-interval log per-event amplitude by group with the per-animal ECDFs
behind it, drawn at raised prominence because the 17 animals, not the 7,410 cells, are the unit of
inference.

Epoch specificity was additionally tested outside the mixed models by a mouse-label permutation
test of the standardized group × epoch difference-of-differences, computed separately for each
decomposition component, with each animal keeping its whole epoch profile (20,000 permutations,
seed 0). Run width, local maxima per run and multi-peak fraction were compared by the same
mouse-label permutation procedure on the per-animal means. These and the other frequentist
secondary tests entered a 13-member Benjamini–Hochberg family (α = 0.05); adjusted *q* values are
reported alongside raw *P* values. Threshold sensitivity refitted the trace-interval amplitude
contrast at detection thresholds of 1.5, 2.0 and 3.0, and a group × trial model
(`log(amplitude) ~ group * trial + (1|animal)` on animal × trial trace-interval means) tested
whether the across-trial decline differed by group.

### Supplementary methods — sensitivity analyses

A **negative-binomial mixed-effects count model** of the per-cell event counts was fitted as a
distribution-aware sensitivity analysis of the population-rate result:
`n_events ~ group * epoch + trial + offset(log cell-seconds) + (1|animal) + (1|animal × trial)`,
with jointly estimated dispersion (Bambi/PyMC). Its epoch factor is anchored to the full 35 s
pre-tone window rather than the 20 s matched baseline the figures use, so its contrasts are
comparable in direction and approximate magnitude but not numerically identical to the primary
model's. Contrasts are reported as posterior rate ratios with 94% highest-density intervals.
Convergence met the preset targets (max r̂ = 1.000, minimum bulk ESS 1,548, zero divergences), and
the group × epoch interaction was assessed by leave-one-out comparison of the full and
interaction-free models. This model has no *P* value and is in no multiplicity family.

An internal **cell-level amplitude lane** was retained from this module's earlier design and
provides secondary evidence only: `log(amplitude) ~ group` over trace-interval cells with
animal-clustered standard errors, and the two within-cell epoch changes (trace − pre-tone,
post-shock − pre-tone) over cells active in both windows, each tested by a joint Wald test on
df = 16 and Holm-corrected across those three omnibus tests. The reported effect size for this
lane is the equal-animal-weighted contrast, not the cell-weighted one, because cell count is
group-correlated and activity-dependent. **This lane is superseded by the animal-level models of
Fig. 2i and supplies no figure annotation and no headline claim.**

The permutation, threshold-sensitivity and photobleaching controls described under Fig. 2e–h above,
together with cross-registration subset variants, complete the sensitivity set. None of them
contributes to the statistics of Fig. 2i.

### Interpretive constraints

The analysis reported here was finalised after substantial prior inspection of this dataset. It is
the main statistical analysis; it is not prospectively preregistered and is not described as
prospectively confirmatory. That is why the conservative six-comparison correction is retained and
reported as a sensitivity analysis rather than dropped. The post-hoc procedure is identical in all
three displayed epochs; the trace interval's biological importance is argued in the Introduction
and Results and is not encoded in the statistical structure.

With *n* = 5/6/6 animals the design has 80% power only for very large standardized effects
(Cohen's *d* ≈ 1.8–2.0 for a pairwise comparison). Every null is therefore reported with its
interval and with what that interval still admits; none is presented as evidence of absence. A
non-significant group × epoch interaction means there was **no evidence that the treatment effect
differed across the pre-tone, trace and post-shock epochs** — not that the effect is identical,
global, tonic or equivalent across them. Region-of-interest inclusion is activity-dependent, so
cells silent throughout a session may not be detected at all; measured session-to-session
detection dropout was comparable across groups (77–82%), which constrains but does not eliminate
this as a source of bias.

### Software

Analyses used Python 3.11.15 with statsmodels 0.14.6 (`MixedLM`, REML, L-BFGS), scipy 1.17.1,
numpy 2.4.5 and pandas 3.0.2. ⟨Minian version.⟩ Analysis code is available at ⟨repository/DOI⟩.

---

## 4. Figure legend

**Figure 2 | Chemogenetic modulation of dorsal CA1 SST interneurons dissociates the magnitude and
the frequency of pyramidal-cell calcium events during trace fear conditioning.**

**a**, Viral strategy. SST-Cre⁺ᐟ⁺ mice received bilateral dorsal CA1 injections of
AAV.Syn.SomaGCaMP6f.f2 with Cre-dependent hM3D(Gq)–mCherry (Exc), hM4D(Gi)–mCherry (Inh) or
mCherry alone (Ctl), and a GRIN lens over dorsal CA1. **b**, Example miniscope field of view
(left, mean fluorescence) and the corresponding Minian spatial footprints (right). **c**,
Representative simultaneous traces from example cells of one animal per group: ΔF/F (faint),
denoised *C* (grey) and deconvolved *S* (coloured); open circles mark detected events. Colour
code: Ctl black, Inh blue, Exc red, used throughout. Scale bar, ⟨see note in §6⟩; 2.5 s. **d**,
Manipulation check across a same-day drug-free → CNO linear-track session pair. Left, within-cell
change in log per-event amplitude (LT2 − LT1); right, fraction of LT1 cells not re-registered in
LT2. Points, individual animals; violins, the per-animal distribution; horizontal line, median.
**e**, Decomposition of trace-interval population activity: fraction of cells active; event rate
among active cells; population event rate over all cells (the product of the first two);
per-event amplitude; and total deconvolved amplitude-rate (population rate × amplitude). Small
points, individual cells (description only); large markers, per-animal means, which are the values
tested. **f**, One trace-interval event run per group, selected to match the pooled median peak
height; shading marks the detected supra-threshold run, dotted line the detection threshold.
**g**, Descriptive per-animal profile of mean log per-event amplitude across five disjoint
conditioning epochs; thin lines, individual animals; bold lines and shading, group mean ± s.e.m.
**h**, Cumulative distribution of per-cell log per-event amplitude during the trace interval;
bold, pooled per group; thin, individual animals. **i**, Primary analysis. Per-event amplitude
(top) and population event rate (bottom) by group in each of the three duration-matched 20 s
epochs. Small points, individual cells (description only); large markers, the per-animal values
the models were fitted on. *n* = 5 hM3D, 6 hM4D, 6 mCherry animals throughout. Statistics in
**i** are linear contrasts of `log(metric) ~ group × epoch + (1|animal)`, Holm-corrected across
the two treatment-versus-control comparisons within each epoch; in **d** and **e**, Welch tests on
per-animal means with the same two-member Holm family, with the hM3D-versus-hM4D comparison shown
uncorrected. **P* < 0.05, ***P* < 0.01. Panels **g** and **h** are descriptive and carry no test.

---

## 5. Number-to-source lookup

Paths are relative to `PLOTS_DIR/sp_rates_lmm/`. Rows of
`paper/tfc_amplitude_rate/stats/unified_lmm_posthoc_contrasts.csv` are keyed by
`(outcome, epoch, comparison)`.

| value in §2 | source |
|---|---|
| 51-row inferential dataset; 9,531 cells | `paper/tfc_amplitude_rate/stats/unified_lmm_mouse_epoch_values.csv` |
| all Fig. 2i ratios, CIs, adjusted *P* | `paper/.../unified_lmm_posthoc_contrasts.csv` — `ratio`, `ratio_ci_low/high`, `p_holm_epoch` |
| six-comparison sensitivity *P* = 0.021 | same file, `p_holm_six` (`population_rate / trace / hM4D_vs_mCherry`) |
| interaction *F*(4,16) = 0.37, *P* = 0.828; *F*(4,16) = 1.54, *P* = 0.237 | `paper/.../unified_lmm_interactions.csv` |
| manipulation check +0.474 / +0.142 / −0.270, *F*(2,16) = 13.02, *P* = 4.4 × 10⁻⁴, 1,751 cells | `TFC_cond/stats/manipulation_check.txt` |
| dropout 0.768 / 0.771 / 0.817 | same file, dropout block |
| all Fig. 2e ratios, differences and intervals | `TFC_cond/decomposition_contrasts.md` |
| run width 5.48 / 4.87 / 4.75; multi-peak 0.046 / 0.037; permutation *P* | `TFC_cond/stats/run_structure.txt` |
| epoch-specificity *T* and *P* (amplitude, population rate) | `TFC_cond/stats/secondary_epoch_interaction.txt` |
| BH *q* values | `TFC_cond/stats/secondary_fdr_family.txt` |
| mean/P90 permutation contrasts (+0.425, *P* = 0.0083; +0.404, *P* = 0.0067) | `TFC_cond/stats/secondary_permutation_tests.txt` |
| ECDF cell counts 2,804 / 2,273 / 2,333 | `TFC_cond/stats/primary_trace_amplitude.txt` |
| threshold sensitivity 0.429 / 0.435 / 0.408 | `TFC_cond/stats/threshold_sensitivity.csv` |
| trial slopes −0.076 / −0.092 / −0.087; *F*(8,16) = 0.21, *P* = 0.98 | `TFC_cond/stats/group_trial_photobleaching.txt` |
| NB rate ratios 0.58 / 0.85 with 94% HDIs | `paper/tfc_amplitude_rate/stats/rate_group_epoch_contrasts.csv` |
| NB convergence (r̂, ESS, divergences) and LOO Δelpd 1.50 ± 4.51 | `TFC_cond/stats/secondary_rate.txt` |
| cell-level trace omnibus *F*(2,16) = 6.44, *P* = 0.0089; 1.53× [1.14, 2.05] | `TFC_cond/stats/primary_trace_amplitude.txt` |
| within-cell epoch deltas *P* = 0.998 / 0.935; 7,067 and 7,609 cells | `TFC_cond/stats/coprimary_epoch_delta_{trace,post_shock}.txt` |
| Holm-adjusted 0.027 / 1.0 / 1.0 across that three-member family | `TFC_cond/stats/confirmatory_holm_correction.txt` |

---

## 6. Points flagged for the author (not changed here)

1. **Event-detection threshold units.** `docs/sp_rates_lmm.md` §M.1 and
   `sp_rates_lmm_paper_methods.md` describe the threshold as "2 s.d."
   `caban.sessions` sets `self.thres = 2` and `caban.utilities.find_event_runs_ca` documents
   `thres` as "the arbitrary y-value unit from S", comparing `trace >= thres` directly. The
   Methods above therefore say **2 in the arbitrary units of *S***, and the sensitivity thresholds
   1.5/2.0/3.0 likewise. If a per-cell s.d. normalisation is applied upstream of these calls,
   correct this document; otherwise the existing METHODS templates need the same fix.
2. **Panel c scale bar.** `analysis.plot_sample_traces` normalises each trace to its own maximum
   before plotting (`C/(height_scale*max_val)`, and ΔF/F likewise), so the "100% ΔF/F" calibration
   printed on the composite is not a valid scale for these traces. This is the known deferred
   axis-label issue; the legend above leaves the vertical scale as ⟨…⟩ rather than restating it.
3. **Stale numbers in `docs/sp_rates_lmm.md` §M.2.** That draft quotes run widths of 5.74 versus
   4.90 and multi-peak fractions of 0.112 versus 0.131. The current run gives 5.48 versus 4.87 and
   0.046 versus 0.037. This document uses the current run; §M.2 should be refreshed.
4. **Panel e asterisks.** `decomposition_contrasts.md` reports uncorrected intervals while the
   figure's asterisks are Holm-corrected, so the "all neurons event rate" hM4D-versus-control
   contrast has an interval excluding 1 (0.54, 95% CI 0.36 to 0.82) without carrying a star. The
   Results text above states the estimate and interval and does not claim significance for it.
5. **Panel g epoch labels.** The figure axis prints raw epoch keys (`pre_tone`, `tone`, `trace`,
   `post_shock`, `post_shock_late`); these want prose labels for publication.
6. **The asterisks in panels d and e have no *P* value on disk.** Those brackets are computed
   inside the plotting call (Welch on per-animal means, Holm across the two
   treatment-versus-control comparisons) and no file in `CURRENT/` records them —
   `decomposition_contrasts.md` deliberately writes estimates and intervals only. The Results text
   therefore states the estimates and intervals for those panels and does not quote a *P*. If you
   want those numbers in the manuscript, the panels need to write their *P* values out on the next
   run.
7. **Superseded lane.** The cell-level omnibus, the two within-cell epoch deltas and their
   three-member Holm family are `docs/sp_rates_lmm.md` Part B — the module's older internal
   confirmatory family, superseded by the animal-level unified lane. They appear in the Results
   only as agreeing secondary evidence and in Methods as sensitivity; do not let them read as the
   headline.
8. **NB model baseline mismatch.** `rate_group_epoch_contrasts.csv` anchors its epoch factor to
   the full 35 s pre-tone window, not the 20 s matched baseline the figures use — stated in the
   file's own `note` column and repeated in the Supplementary Methods above. Its ratios are
   therefore directionally comparable but not numerically identical to the primary model's.
9. **Fig. 2i post-shock amplitude.** The hM3D post-shock adjusted *P* is 0.0497 — significant, but
   at the boundary. Reported as *P* = 0.050 above; consider printing three decimals in the final
   manuscript to avoid a reader seeing "0.05" and a star together.
