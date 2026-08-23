# Figure guide: what each panel actually tells us

Companion to `sp_rates_lmm_methods.md` (which says how each number is computed and why the
statistics are the way they are). This file says **what each panel is for** — the reading of the
figure, the logical flow for the paper, and drafts of the Results text and legend.

> **⚠️ The numbers in this document are illustrative — they are here to make the explanation
> concrete, NOT as a source of record. This file is static prose copied from
> `analysis_methods_templates/` on every run; it does not read the results of the run that
> produced the figures next to it.** For the real numbers always use the run's own outputs:
> `decomposition_contrasts.md` and `stats/*.txt` / `stats/*.csv` in this same directory. If a
> value here disagrees with those, those are right and this file is out of date. Never copy a
> figure into the manuscript from this document without checking every quoted value against
> `stats/` first.

Numbers below were the TFC_cond outputs at the time this guide was written (n = 5 hM3D / 6 hM4D /
6 mCherry mice; equal-mouse-weighted contrasts vs mCherry with Welch 95% intervals).

---

## Part 1 — Panel-by-panel, for internal use

> **Four decomposition figures, one layout.** `decomposition.png` is the trace epoch;
> `decomposition_pre_tone_matched.png`, `decomposition_post_shock.png` and
> `decomposition_post_shock_late.png` are the same five panels for the 20 s matched baseline, the
> 20 s post-shock response window, and the 20 s late post-shock window beginning 90 s after shock
> offset. All four windows are the same length, which is what makes their fraction-active and
> event-rate panels comparable at all — everything below applies unchanged to each.
>
> **`decomposition_post_shock_late.png` pools FEWER TRIALS than the other three.** The late
> window does not exist on a final trial whose recording stops shortly after the shock, so that
> figure's cells have less total exposure than the same cells in `decomposition_post_shock.png`.
> Fraction-active is not exposure-normalized, so it is biased DOWNWARD there relative to the
> other three figures. The *within-figure* group comparison is unaffected — trial coverage
> depends on recording length, not on DREADD group — but the across-figure reading, already
> disallowed below, is doubly invalid for this panel. Per-mouse coverage is in
> `stats/descriptive_early_vs_late_trial_coverage.csv`.
>
> **They are descriptive, and comparing them by eye is not a test.** Reading "starred in one,
> not starred in another" is the difference-of-significance fallacy. Epoch-specificity is carried
> by the within-cell delta contrasts and `epoch_delta_forest.png`, which compare epochs *within*
> a cell instead of comparing two independently-fit figures. This applies with particular force
> to the early-versus-late post-shock comparison: `decomposition_post_shock.png` next to
> `decomposition_post_shock_late.png` is *not* that contrast — the within-cell
> `descriptive_epoch_delta_post_shock_vs_post_shock_late.txt` / `_effect_forest_` pair is.

### 1.1 The one idea the whole figure is built on

Everything in `decomposition.png` is one exact algebraic identity, split into its factors:

```
fraction active  x  rate among active cells   =   overall event rate (all cells)
overall event rate  x  mean per-event amplitude  =  total S per second
```

Nothing here is five independent "measures of activity". They are **two multiplications**, drawn
so the reader can see where any change enters the chain rather than being asked to trust a summary.
This is the direct answer to both of your questions — see 1.3 and 1.5 below.

The legacy `sp_rates` metric (summed peak height per second) *was* the right-hand end of that
chain, collapsed. It equals `rate x mean peak height` by construction, so it is mathematically
incapable of separating "more events" from "bigger events" — which is precisely the claim under
test. Splitting the product is the reason this analysis exists.

### 1.2 Panel A — Fraction active (mouse-level violin)

**Asks:** did the manipulation change *how many cells participate at all* during the trace window?
`P(N >= 1 event)`, computed over every CNMF-E-detected cell in the session, one value per mouse.

**Reading:** Exc/Ctl +0.053 [−0.085, +0.19]; Inh/Ctl −0.038 [−0.178, +0.102]. **Neither DREADD
changes the participating fraction relative to control.** The bracket on the panel is the
Exc-vs-Inh contrast (0.82 vs 0.71), which spends no alpha of its own.

**Caveat to state in text:** fraction active is strongly window-duration dependent (a cell is far
more likely to fire ≥1 event in 35 s than in 2 s at the same underlying rate). It is only
comparable here because every group is being compared over the *same* trace windows. It is a
mouse-level quantity by construction — a proportion over a mouse's cells — hence a violin, not a
SuperPlot.

### 1.3 Panel B — Rate among active cells vs 1.4 Panel C — Overall rate (all cells)

**This is your "all cells vs active cells only" question, and the two panels answer different
questions on purpose.**

**Panel C (all cells) is the estimand.** It is the population quantity — *what is the mean event
intensity of the recorded CA1 pyramidal population in this behavioural state* — and zero-event
cells stay in the denominator precisely so that a manipulation which silences cells shows up as a
reduced rate. If you report only one rate number, this is the one.

**Panel B (active cells only) is a diagnostic, not an endpoint.** It is `E(rate | N > 0)`, and
conditioning on `N > 0` is **post-treatment, activity-dependent selection**. It can move
perversely: if a manipulation silences the weakest cells outright, the mean among the surviving
active cells can *rise* while total population activity *falls*. It must therefore never carry the
headline claim.

**So why show it?** Because A × B = C exactly, and the pair localizes *where* a change in C came
from: fewer cells taking part (A), or each participating cell firing less often (B). Here:

- Inh/Ctl overall rate **0.54x [0.36, 0.82]**, −0.037 [−0.069, −0.005] events/s — a real reduction.
- Inh fraction active is unchanged, and Inh rate among active cells is 0.56x [0.41, 0.77].
- ⇒ **hM4D's suppression is entirely "each active cell fires less", not "fewer cells participate".**
- Exc/Ctl overall rate 0.79x [0.55, 1.16] — a reduction of the same sign but not resolved at this n;
  the interval still admits everything from −45% to +16%.

That decomposition is the whole payoff of carrying both panels. Reporting only "active cells" would
be a selection-biased endpoint; reporting only "all cells" would leave the mechanism unresolved.

### 1.5 Panel D — log(mean per-event amplitude) — the PRIMARY endpoint

**Asks:** conditional on an event being detected, how large is it? Per-event **integral**
(`sum(S)` over the whole contiguous supra-threshold run), not peak height — so a wider, burst-like
run scores higher even at the same peak.

**This is your "why is amplitude reported separately" question.** Because
`total S/s = rate x amplitude`, amplitude is the *orthogonal factor* of the same product that rate
occupies. Folding them together (the legacy summed-amplitude-per-second measure) destroys exactly
the contrast the hypothesis is about. Separating them is what makes "fewer but larger events"
expressible at all.

**Reading:** Exc/Ctl **1.53x [1.14, 2.05]** (+0.425 [+0.132, +0.718] log units); primary omnibus
`F(2,16) = 6.44, p = 0.0089`, Holm-corrected across the confirmatory family of two: **p = 0.018**.
Mouse-label permutation (20,000 draws) agrees: p = 0.0083. Inh/Ctl 1.18x [0.85, 1.65] — null, but
an interval reaching 1.65 has not excluded a +65% effect, and must be reported that way.

Amplitude is undefined for a zero-event cell; those cells are excluded here by definition (not
imputed to zero), which is why this panel's denominator differs from panels A and C.

### 1.6 Panel E — Total S/s — the closure of the identity

**Asks:** what do the two factors multiply out to — did net calcium output change?
C × D exactly. It is *not* an independent sixth measure; it is here so the reader can see the
arithmetic close.

**Reading:** Exc/Ctl 1.55x [0.93, 2.59]; Inh/Ctl 0.74x [0.46, 1.19]. **Neither is resolved against
control**; the bracket is Exc vs Inh.

### 1.7 The one sentence the whole figure exists to license

The two DREADDs dissociate along the two factors of the same product:

| | fraction active | rate | per-event amplitude | net output |
|---|---|---|---|---|
| **hM3D (Exc)** | unchanged | ↓ (unresolved) | **↑ 1.53x** | unchanged |
| **hM4D (Inh)** | unchanged | **↓ 0.54x** | unchanged | ↓ (unresolved) |

hM3D **redistributes** activity — fewer, larger events, net output preserved.
hM4D **suppresses** it — same-size events, fewer of them.
That is a qualitative dissociation, not two versions of "more/less activity", and it is only
visible because the product was split.

### 1.8 The supporting figures, in the order they earn their place

| Figure | What it adds | Current reading |
|---|---|---|
| `manipulation_check` | **Does the tool work?** Within-cell LT2−LT1 amplitude delta, same day/FOV, difference-in-differences vs control; plus LT1→LT2 detection dropout | hM3D +0.474 vs Ctl, `F(2,16)=13.0, p=4.4e-4` — best-powered comparison in the dataset. Dropout hM3D 0.82 / hM4D 0.77 / Ctl 0.77: **no hM4D-specific cell loss**, which is what makes the hM4D amplitude null interpretable rather than an artifact |
| `primary_trace_amplitude` | Triangulation: the primary effect at the level of the **17 animals**, not the model | Visible without the model — the requirement stated in Methods |
| `primary_effect_forest`, `coprimary_effect_forest` | Fold-change + CI on the multiplicative scale | 1.53x [1.14, 2.05] |
| `epoch_profile`, `epoch_delta_forest` | **Is the effect trace-specific?** Within-cell delta over each cell's own pre-tone baseline, every epoch | Co-primary is **NULL** (`F(2,16)=0.002, p=0.998`; trace 1.02x [0.87, 1.21]) — no evidence that the amplitude effect differed between epochs. That is not evidence that it is identical, global or tonic across them, and no claim of trace specificity may be made either. The paper's version of this question is the unified group x epoch Wald test (Part 4) |
| `decomposition_grid` | **Which component changed, and was it consistent across the session?** All four components x the matched epochs as effect estimates + 95% CI, three contrasts per panel (Exc/Ctl, Inh/Ctl, **Exc/Inh**), each row labelled with that component's BH-adjusted group x epoch q | Read the rows as a decomposition, never as four independent findings — they are one identity and their q-values are strongly dependent. This figure — not a comparison of per-epoch p-values — is what settles epoch specificity. **Use the grey Exc/Inh point** for any DREADD-vs-DREADD claim; the red and blue intervals share a reference and cannot be compared to each other by eye |
| `amplitude_ecdf`, `amplitude_p90` | **Shape**, not just mean: bursting predicts a fattened right tail | hM3D p90 shifted +0.404, permutation p = 0.0067 — the effect is a tail shift, not only a mean shift |
| `run_structure` | **Mechanism check**: is this genuinely burst-like, or an artifact of temporally clustered peaks merging into one run? | Run width hM3D 5.48 vs Ctl 4.87 frames (+0.61), permutation p = 0.096, BH q = 0.43. **Suggestive, not established** — report as such |
| `threshold_sensitivity` | Is the effect a detection-threshold artifact? | Coefficient 0.429 / 0.435 / 0.408 at thres 1.5 / 2.0 / 3.0 — **stable**; run-merging at one threshold cannot be the sole explanation |
| `example_traces`, `width_height_matched_examples` | Raw-data evidence; height-matched runs make "wider, not just taller" concrete | Selection is deterministic (10th/50th/90th percentile per group), never hand-picked |
| `Test_B`, `Test_B_1wk` post-tone amplitude | Does the signature persist where the freezing phenotype is? | Test_B 48 h: Exc/Ctl **1.41x [1.05, 1.91]**, `p = 1.0e-3`, BH q = 0.009. 1 wk: 1.11x [0.71, 1.73], n.s. (and hM4D n=5 in each, different mouse missing) |

### 1.9 Recommended logical flow for the paper

1. **The tool works** — LT1→LT2 within-cell manipulation check.
2. **What we measure and why** — per-event run integral, not peak; state that "bursting" is the
   interpretation, not the measurement.
3. **Primary result** — hM3D enlarges per-event amplitude during conditioning (1.53x).
4. **Where it sits in the activity budget** — the decomposition figure, and the hM3D/hM4D
   dissociation.
5. **Epoch dependence** — the paper's number is the **unified joint group x epoch Wald test**,
   one per outcome, in `paper/tfc_amplitude_rate/stats/unified_lmm_interactions.csv` and on the
   paper forest's row labels. The internal permutation version (`secondary_epoch_interaction.txt`,
   visible on `decomposition_grid`) is the distribution-free sensitivity check. Write a null as
   *"there was no evidence that the treatment effect differed across the pre-tone, trace and
   post-shock epochs"* — never as evidence that the effect is identical, global or tonic. The
   trace interval's privileged status rests on prior anatomy and on behaviour, not on this
   result.

   **Do not argue this from a comparison of p-values.** "Significant in trace, not significant in
   the epoch deltas" is the difference-of-significance fallacy and a reviewer will say so. The
   epoch-specificity claim has exactly one supporting number per component — that component's
   joint interaction test — and the difference-of-differences intervals beside it say what
   magnitude of specificity the data still admit. Note also that "there is no effect during trace"
   is the opposite of what the amplitude endpoint found; the null is about the DIFFERENCE between
   epochs, not about any epoch.

   **Two more comparisons that are not licensed by looking at the grid:**

   - *Across columns within a row.* An estimate that grows left-to-right (say, fraction active
     rising toward post-shock) is not evidence of an epoch-specific effect unless that row's q
     says so — and if the same drift appears in BOTH groups, it is an epoch main effect, i.e. a
     property of the trial structure rather than of the manipulation.
   - *Between the red and blue points.* They share mCherry as their reference. The DREADD-vs-DREADD
     comparison is the grey point, and its interval is usually wider than the visual gap suggests.
6. **Robustness** — threshold, tail statistic, permutation, run width (honestly: suggestive).
7. **Persistence** — present at 48 h recall, absent at 1 week.
8. **Limitation** — activity-dependent ROI inclusion, measured rather than merely acknowledged.

---

## Part 2 — Results and Methods draft (Nature style)

**Moved.** The manuscript draft lives in `docs/sp_rates_lmm.md` §M — Methods (§M.1), Results
(§M.2) and the placeholder lookup table (§M.3). It is written from that document's Part A (the
unified paper-facing models) only.

It used to be duplicated here, and the two copies diverged: this one still quoted a
Holm-corrected *P* of 0.018 from the retired two-test confirmatory family (it is 0.027 across
three), along with several decomposition estimates that predate the current run. That is exactly
the failure mode a second copy produces, so there is now one draft in one place.

What stays here is Part 3 onward: the figure legends, which belong beside the per-figure reading
guide they describe.

---

## Part 3 — Figure legend draft (Nature style)

> For `decomposition.png`. Panel letters assume a → e left-to-right.

**Fig. X | SST-interneuron modulation dissociates the frequency and the size of CA1 pyramidal
calcium events during trace fear conditioning.**
**a–e**, Components of population calcium activity during the trace interval of conditioning,
shown as the exact factorization
`fraction active x rate among active cells = overall event rate` and
`overall event rate x per-event amplitude = total signal per second`.
**a**, Fraction of detected pyramidal cells with at least one event (one point per mouse; violin,
kernel density; horizontal line, group mean). **b**, Event rate among event-active cells.
**c**, Overall event rate across all detected cells, including cells with zero events.
**d**, Mean per-event amplitude, defined as the integral of the deconvolved signal over each
contiguous supra-threshold run (log scale; undefined for, and therefore excluding, cells with no
events). **e**, Total deconvolved signal per second, the product of **c** and **d**.
In **b–e**, small semi-transparent points are individual cells, colour-shaded by mouse, and large
black-edged markers are the 17 per-mouse means; cell-level points are shown for distributional
context only and never enter any statistic. Rate panels carry sub-quantum vertical jitter
(< ±0.4 of one event per window) so that discrete count levels are legible; cells with zero events
are pinned at the axis floor. Ctl, mCherry (n = 6 mice); Exc, hM3D (n = 5); Inh, hM4D (n = 6).
All statistics are computed from per-mouse values. Brackets, Holm-corrected pairwise comparisons
across the two treatment-versus-control contrasts (hM3D vs mCherry, hM4D vs mCherry), with the
hM3D-vs-hM4D contrast shown uncorrected; *P < 0.05, **P < 0.01. Equal-mouse-weighted effect sizes
with 95% confidence intervals for every panel, including non-significant ones, are given in
Supplementary Table S1.


### Legends for the paper-lane figures

> For `paper/tfc_amplitude_rate/`. These are the figures a manuscript would carry; the
> `decomposition.png` legend above describes the internal five-panel version.

**Fig. 1 | hM3D enlarges CA1 pyramidal calcium events throughout trace fear conditioning, whereas
hM4D reduces their frequency.**
**a**, Mean per-event amplitude (log scale) and **b**, population event rate, for each treatment
group across the three duration-matched 20 s windows of a conditioning trial: pre-tone baseline,
trace interval and post-shock. Small semi-transparent points are individual cells, colour-shaded
by mouse; large black-edged markers are the per-mouse means. **Cell-level points are shown for
distributional context only and never enter any statistic** — all comparisons are computed from
the per-mouse means. Amplitude is the integral of the deconvolved signal over each contiguous
supra-threshold run and is undefined for, and therefore excludes, cells with no events; rate
panels retain those cells and carry sub-quantum vertical jitter (< ±0.4 of one event per window)
so that discrete count levels remain legible. Analyses use exposure-matched trials, in which every
window was present at its full 20 s. The y-axis is shared within each row, so the three windows
are directly comparable. Ctl, mCherry (*n* = 6 mice); Exc, hM3D (*n* = 5); Inh, hM4D (*n* = 6).
Brackets, Holm-corrected comparisons across the two treatment-versus-control contrasts;
\*P < 0.05, \*\*P < 0.01. **Stars are corrected within a panel and are not comparable between
panels**; whether the group effect differs between windows is tested once, and is reported in
Fig. 2 and in the text. Effect estimates with 95% confidence intervals for every panel, including
non-significant ones, are in Supplementary Table ⟨n⟩.

**Fig. 2 | The dissociation is stable across the conditioning trial.**
Effect estimates with 95% confidence intervals for each component of population calcium activity —
**a**, fraction of cells active; **b**, population event rate; **c**, mean per-event amplitude —
in each of the three matched windows, expressed as the treatment-versus-control difference
(**a**, a bounded proportion) or fold-change (**b**, **c**, log axis, so that a halving and a
doubling are equidistant from the reference line). Points are equal-mouse-weighted estimates;
error bars are Welch 95% confidence intervals computed from the per-mouse means. **No significance
stars are shown, by design**: the three components are one exact decomposition
(population rate = fraction active × rate among active cells, with amplitude the separate
magnitude term) rather than independent measures, and counting significant panels across the grid
is not a valid reading of it. Each row is instead annotated with a single group × epoch
permutation test asking whether that component's group effect changes across windows, corrected
across the secondary family; all are null (*q* ≥ 0.72). Ctl, mCherry (*n* = 6 mice); Exc, hM3D
(*n* = 5); Inh, hM4D (*n* = 6).

---

## Part 3a — `conditioning_phase_amplitude` (early vs late conditioning)

Two files, both descriptive:

- `conditioning_phase_amplitude.png` — per-event amplitude by group, 2 rows (pre-tone baseline
  35 s; trace interval) x 2 columns (trials 1-2; trials 3-5), y-axis shared within each row.
- `conditioning_phase_amplitude_forest.png` — **the panel that actually answers the question**:
  each DREADD group's amplitude ratio vs control, computed separately in early and late trials.
  Open markers are early, filled are late.

**The question it answers.** Every other epoch contrast in this module compares windows *within*
a trial. This one splits *across* trials, which is orthogonal: given that the amplitude effect is
comparable across the within-trial windows, is it present from trial 1 (a property of
the drug being on board) or does it **develop** as conditioning proceeds? If a group's open and
filled points sit in the same place, the effect does not develop over conditioning. That is not
the same as establishing a tonic drug effect, which would need a conditioning-naive, drug-on
baseline this design does not provide.

**Read the contrast, not the level.** Absolute amplitude falls ~31% from trial 1 to trial 5, in
every group, consistent with photobleaching. That is a main effect of trial — it moves both
columns of the distribution figure down together and cancels in a group contrast. A drop between
the columns is expected and means nothing about the manipulation. This is exactly why the forest
exists alongside the distributions.

**No significance stars, deliberately.** This figure spends no alpha and is in neither
multiplicity family. The formal test of whether the group effect changes across trials already
exists: `group_x_trial_interaction`, a member of the secondary BH-FDR family
(`stats/secondary_fdr_family.csv`, `stats/group_trial_photobleaching.txt`). Comparing stars
between the two columns would be the difference-of-significance fallacy, so there are none to
compare.

**Amplitude only, and that is not an omission.** The rate analog cannot be drawn honestly at this
split: rate and fraction active are duration-sensitive and so need exposure-matched trials, which
drops trial 1 (its trace window is 15 s, not 20 s) — reducing the early column to a single trial.
Amplitude is a per-event quantity, needs no matching, and keeps every trial.

Per-mouse trial counts behind the split are in `stats/conditioning_phase_trial_coverage.csv`;
recordings are ragged, so report them with the estimate.

---

## Part 4 — The paper lane (`sp_rates_lmm/paper/tfc_amplitude_rate/`)

Everything above describes the internal output. **This is where the analysis the manuscript
reports actually lives.** Both figures below are annotated entirely from two tables — the
mouse-level inferential dataset and the unified models' contrasts — and neither computes a
statistic of its own. See §1 of `docs/sp_rates_lmm.md` for the models.

### `tfc_amplitude_rate_by_epoch.png` — the distributions

Two rows (per-event amplitude; population event rate) × three columns (pre-tone 20 s baseline,
trace, post-shock 20 s), on exposure-matched trials — the same trials for both rows.

Small faint points are individual cells, colour-shaded by mouse; large black-edged markers are the
**exact mouse-level values the models are fit on**, drawn on the natural scale. For amplitude that
marker is `exp(mouse_mean_log_amplitude)` — that animal's geometric mean event amplitude, which is
why it need not sit at the visual centre of its cloud. For rate it is total events over total
cell-seconds, a cell-seconds-weighted mean of the cloud. Only these values enter any statistic;
the clouds are description. Amplitude is undefined for a cell with no events, so those cells are
absent from the top row by definition, not by exclusion; they are retained in the rate row's
denominator.

**Every panel is annotated by the same procedure.** Brackets carry the `p_holm_epoch` values from
`stats/unified_lmm_posthoc_contrasts.csv` — within each epoch, hM3D and hM4D are compared with
mCherry and those two treatment-versus-control comparisons are Holm-corrected together. All six
panels run that identical procedure, and a bracket is drawn wherever the adjusted P clears α, so
a panel without one is a panel where neither comparison reached significance, not one that was
exempted from testing. No hM3D-versus-hM4D bracket is drawn — it is in no correction family and is
not the design's question — and no `p_holm_six` sensitivity value annotates this figure.

**Read down a column** to see how the two quantities dissociate within one window. **Read across a
row** to compare the same quantity between windows — the columns share one y-scale precisely so
that comparison is honest.

The presence of a bracket in one column and none in another is **not** evidence that the treatment
effect differs between windows — the two comparisons were never tested against each other. Epoch
dependence has exactly one test, the group × epoch interaction in
`stats/unified_lmm_interactions.csv`, reported once per outcome.

Suggested caption core: *Per-event amplitude and population event rate across exposure-matched
pre-tone, trace and post-shock epochs. Small points indicate individual cells and large outlined
points indicate animal-level values used in the mixed-effects models. Within each epoch, hM3D and
hM4D were compared with mCherry using model-derived post-hoc contrasts with Holm correction across
the two treatment-versus-control comparisons. Asterisks denote Holm-adjusted P < 0.05 (\*),
P < 0.01 (\*\*), P < 0.001 (\*\*\*). Group × epoch interactions were assessed separately by joint
Wald tests.*

### `tfc_decomposition_forest.png` — the same effects as estimates

Two rows (population event rate; per-event amplitude) × the same three windows. Two points per
panel: hM3D-versus-control and hM4D-versus-control as ratios with 95% confidence intervals on a
log axis centred on 1.

**These are the same rows of `stats/unified_lmm_posthoc_contrasts.csv` that produce the asterisks
on the distribution figure** — one model, one table, two views of it. Where the distribution
figure shows the decision, this one shows the effect size and what its interval still admits,
which is the part a null needs.

All three epochs are kept here, including any that carry no bracket on the distribution figure:
an estimate and its interval are scientifically informative whether or not the adjusted P cleared
α. **No significance stars are placed on this figure** — it is an effect-size display. Its
companion markdown may list the within-epoch `p_holm_epoch` values; the six-comparison sensitivity
Holm P is never shown on the forest, and where both appear in a companion table they are labelled
distinctly and never interchanged.

Each row label carries that outcome's **joint group × epoch Wald test** — `F(4, 16)` and its P.
That number, not a comparison of columns, is the answer to whether the treatment effect differs
across windows. A null there is *no evidence that the effect differed across epochs*; it is not
evidence that the effect is identical, global or tonic.

Fraction active is not a row here: it has no unified mouse-level model, and including it would put
a second, differently derived inferential source on a paper figure. It is unchanged on the
internal four-component `decomposition_grid`. The direct hM3D-versus-hM4D contrast is likewise not
drawn — exploratory, uncorrected, in no multiplicity family; the defensible sentence is that the
two DREADDs show divergent recruitment profiles while neither is claimed to differ from the other.

### `stats/paper_results_summary.md`

Every number a Results paragraph needs, split into **Part 1 — paper-facing analysis** (the two
unified mixed models, all twelve treatment-versus-control contrasts by epoch with their
within-epoch Holm-adjusted P values, the two group × epoch interaction tests, and the descriptive
absolute population-rate differences) and **Part 2 — sensitivity and supplementary** (which is
where the conservative six-comparison across-epoch Holm correction lives, and the only place it
appears). Quote only from Part 1. Part 2's numbers are real and
useful as robustness
evidence, but the separation is the part that gets lost in transit: a sensitivity estimate written
up as the headline once it has been detached from the file that said otherwise is this module's
recurring failure mode.
