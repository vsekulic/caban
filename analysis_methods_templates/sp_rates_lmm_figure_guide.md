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
| `epoch_profile`, `epoch_delta_forest` | **Is the effect trace-specific?** Within-cell delta over each cell's own pre-tone baseline, every epoch | Co-primary is **NULL** (`F(2,16)=0.002, p=0.998`; trace 1.02x [0.87, 1.21]). The amplitude effect is a **global main effect present at every epoch**, not a trace-specific one. No claim of trace specificity may be made |
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
5. **It is global, not trace-specific** — the co-primary null; the trace interval's privileged
   status rests on prior anatomy and on behaviour, not on this result.
6. **Robustness** — threshold, tail statistic, permutation, run width (honestly: suggestive).
7. **Persistence** — present at 48 h recall, absent at 1 week.
8. **Limitation** — activity-dependent ROI inclusion, measured rather than merely acknowledged.

---

## Part 2 — Results section draft (Nature style)

> Working draft — the prose is the deliverable here, the numbers are placeholders. Group labels
> as in the figure (Ctl = mCherry, Exc = hM3D, Inh = hM4D). **Every statistic below must be
> re-read off this run's `stats/` and `decomposition_contrasts.md` before it goes into a
> manuscript**; see the warning at the top of this file.

**Chemogenetic modulation of SST interneurons dissociates the size and the frequency of CA1
pyramidal calcium events.**

To ask how SST-interneuron modulation reshapes dorsal CA1 pyramidal output during trace fear
conditioning, we detected calcium events as contiguous supra-threshold runs of the deconvolved
signal and quantified each event by its integral rather than its peak, so that a wider event is
distinguishable from a taller one. Because summed event amplitude per second is the exact product
of event frequency and per-event amplitude, we analysed the two factors separately and report
their product alongside them (Fig. Xa–e). All statistics treat the mouse as the unit of inference
(n = 5 hM3D, 6 hM4D, 6 mCherry), with cell-level observations entering mixed models carrying a
mouse random intercept.

The manipulation was effective: within cells tracked across two same-day linear-track sessions
recorded before and after CNO, hM3D increased per-event amplitude relative to control
(difference-in-differences +0.47 log units, F(2,16) = 13.0, P = 4.4 × 10⁻⁴), while the fraction of
cells failing to re-register between the two sessions did not differ between hM4D and control
(0.77 versus 0.77), arguing against activity-dependent loss of silenced cells as an explanation
for the hM4D results below.

During the trace interval of conditioning, hM3D mice showed larger individual calcium events than
controls (1.53-fold, 95% CI 1.14–2.05; joint Wald F(2,16) = 6.44, P = 0.0089; Holm-corrected across
the two confirmatory tests, P = 0.018; mouse-label permutation P = 0.008), whereas hM4D did not
(1.18-fold, 95% CI 0.85–1.65). The effect was a shift of the whole amplitude distribution rather
than of its mean alone: the 90th percentile of per-cell amplitude was elevated in hM3D
(permutation P = 0.007), and the estimate was stable across event-detection thresholds spanning
1.5–3.0 s.d. (coefficient 0.41–0.43), excluding a threshold artifact as its sole source.

Decomposing population activity into its exact factors localized where each manipulation acted
(Fig. Xa–e). Neither DREADD changed the fraction of cells recruited during the trace interval
(hM3D +0.05, 95% CI −0.08 to +0.19; hM4D −0.04, 95% CI −0.18 to +0.10). hM4D instead reduced the
overall population event rate (0.54-fold, 95% CI 0.36–0.82; −0.037 events s⁻¹, 95% CI −0.069 to
−0.005) through a reduction in the rate of the cells that remained active (0.56-fold, 95% CI
0.41–0.77), with per-event amplitude unchanged. hM3D showed the converse profile: enlarged events
with a rate change in the same direction as hM4D but not resolved at this sample size (0.79-fold,
95% CI 0.55–1.16). Net calcium output, the product of rate and amplitude, was consequently not
distinguishable from control in either group (hM3D 1.55-fold, 95% CI 0.93–2.59; hM4D 0.74-fold,
95% CI 0.46–1.19), although the two DREADDs differed from each other. Excitatory and inhibitory
modulation of SST interneurons therefore acted on orthogonal factors of the same quantity: hM3D
redistributed a comparable amount of activity into fewer, larger events, whereas hM4D reduced the
number of events without altering their size.

This amplitude effect was not specific to the trace interval. Referencing each cell to its own
pre-tone baseline, the group difference in trace-minus-baseline amplitude was null
(F(2,16) = 0.002, P = 0.998; hM3D 1.02-fold, 95% CI 0.87–1.21), and the same was true for tone and
post-shock windows, indicating a tonic elevation of event size across the conditioning session
rather than a state-specific one. Consistent with a persistent circuit change, the hM3D amplitude
signature was still present in the post-tone window at 48-h recall (1.41-fold, 95% CI 1.05–1.91,
P = 0.001, q = 0.009) but was no longer detectable at one week (1.11-fold, 95% CI 0.71–1.73).

A burst-like origin for the larger events is suggested but not established: supra-threshold runs
were wider in hM3D than in control (5.5 versus 4.9 frames), though this difference was not
significant after correction (P = 0.10, q = 0.43), and the fraction of runs containing more than
one local maximum did not differ between groups. Because temporally clustered events merge into a
single run under this event definition, run width is the measurement that would distinguish
genuine bursting from that merging, and at the present sample size it does not do so decisively.
Finally, cellular analyses of this kind are conditional on the neurons that source extraction
detects; the absence of an hM4D amplitude effect therefore cannot exclude changes in neurons that
became undetectable, although the matched cross-session dropout reported above makes such loss
unlikely to be large.

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
