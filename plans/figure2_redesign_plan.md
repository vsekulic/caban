# Plan: Figure 2 redesign — one story, in order

Status: **agreed in outline (VS and PI), 2026-10-21; decisions in §7.** The composite is
scaffolded as `fig2_dissociation` (§8); the new panels are not built. Supersedes the panel list
in [handover_figure2_nature_and_next_steps.md](handover_figure2_nature_and_next_steps.md) §2 once
agreed; the paper-facing results of [../docs/paper/paper_figure2.md](../docs/paper/paper_figure2.md)
stay valid and move panel by panel as below.

## 1. The problem with the current figure

The current figure (a–k) answers its questions out of order and alternates between two
measures: amplitude (d, e, g-top, h, i, j-left) and frequency (e, g-bottom, j-right). The
manipulation check (d) uses a measure (cross-session dropout) that persuades nobody. Recall (j)
belongs to another day. Panel k is a third kind of measure (the YrA mean over all cells), and its
tests are null.

## 2. The story, as one line per step

1. **The DREADDs act on SST cells as intended.** CNO raises SST-cell activity with hM3D and
   lowers it with hM4D (DIO-GCaMP mice).
2. **CNO changes pyramidal events, and does so before conditioning.** Within the same day and the
   same context, before versus after the injection, against mCherry: hM3D raises per-event
   amplitude; hM4D lowers event frequency.
3. **The same dissociation holds during conditioning.** Amplitude for hM3D, frequency for hM4D,
   with no evidence that either depends on the epoch. This is the current primary analysis,
   unchanged.
4. **At the shock.** The shock-evoked response of pyramidal cells, measured with the same
   event-based quantities as steps 2–3.

Each panel carries one measure, and amplitude and frequency each sit in their own row or column
throughout.

## 3. Proposed panels

| new | content | status | from |
|---|---|---|---|
| a–c | strategy, field of view, example traces | keep | a–c |
| **d** | SST cells (DIO mice): event rate and amplitude before vs after CNO, per cell and per animal, hM3D and hM4D | **new; needs Minian on 4 mice (§4.2)** | — |
| **e** | Pyramidal cells, homecage before vs after CNO on the conditioning day: per-animal change in amplitude (left) and frequency (right), Ctl/Exc/Inh | **new; data exist (§4.1)** | replaces d |
| **f** | Conditioning: per-event amplitude by epoch | keep (primary) | g top |
| **g** | Conditioning: population event rate by epoch | keep (primary) | g bottom |
| **h** | Example matched events | keep or drop (VS) | f |
| **i** | Shock-locked event response (amplitude and event probability around the shock) | **redesign (§5)** | k |

**Moves to the supplement:** the decomposition (current e; fraction active, rate among active
cells and amplitude-rate are the explanation of panel g, not a separate claim); the ECDF (h); the
epoch profile (i); the LT1→LT2 manipulation check and dropout (d); recall (j), with the Test B
material already in [../docs/paper/paper_figure2_supplement.md](../docs/paper/paper_figure2_supplement.md);
the YrA per-cell modulation lanes, per-animal traces and motion check (Supplementary Fig. 2f–h, as
drafted 2026-10-18).

## 4. New analyses

### 4.1 Pyramidal cells before vs after CNO (new e) — data ready

Every main-cohort mouse (G05–G21, n = 5/6/6) recorded on the conditioning day, in this order:
homecage HC1, LT1 (G12 onward: HC1, LT1, HC2), injection, **CNO1** (and **CNO2** in 11 of 17), LT2,
usually a homecage session, then TFC_cond. Exceptions: G07 and G08 have no homecage session
between LT2 and conditioning, and G17 recorded LT2 after conditioning. The Razer batch has finished Minian for all of them
(audited 2026-10-20). This gives a within-day **pre/post-injection design with mCherry as the
control for injection, handling and time**, in an identical context.

- **Endpoint:** per-animal mean log per-event amplitude and population event rate, from the same
  event detection as the primary analysis. Pre = the homecage session(s) before injection (HC1,
  and HC2 where it precedes the injection); post = **CNO1**, which every animal has, with CNO2
  where it exists. CNO1 begins right after the injection, so the post window is defined by time
  from injection, not by session name (the injection time is to be read from each day's notes).
- **Model:** `log(metric) ~ group × time + (1|animal)`, with the group × time contrasts
  (Exc − Ctl, Inh − Ctl) as the test, Holm across the two. **This is a new, prospectively stated
  analysis**; the measures and windows are fixed here before it is run.
- **Optional time course:** CNO1 → CNO2 → the later homecage session where it exists, as onset of
  the drug (descriptive).
- **Work:** a homecage session type in the loader (no behaviour-epoch boundaries), reuse
  `sp_rates_lmm`'s event detection, and a section plus METHODS template. Per-animal means need no
  cross-registration. A within-cell version would need the cross-registration runner (D3,
  not yet run) and is a later option.

### 4.2 SST validation (new d) — needs Minian first

G22 and G23 (hM3D) and G26 and G27 (hM4D) have no Minian output at all (audited 2026-10-20). The
existing stub, [sst_dio_dreadd_validation_plan.md](sst_dio_dreadd_validation_plan.md), lists the
open questions; the ones that decide feasibility:

- **CNMF-E parameters for sparse SST cells.** The frozen parameters were tuned on dense pyramidal
  layers. A parameter gate on one session per mouse comes first.
- **n = 2 per DREADD, no mCherry-DIO control.** This is a within-cell, within-animal
  demonstration, shown per animal. It cannot carry a group-level *P*, and time and injection are
  bounded only by the pre-injection homecage drift.
- **Work:** about 10 sessions per mouse of the conditioning day (HC1, LT1, HC2, CNO / CNO1, CNO2,
  LT2, HC3, TFC_cond, HC4) on the Razer, then the same session-level analysis as §4.1 on SST
  cells.

If the gate shows the SST recordings are not usable, d moves out and the figure starts at e.

### 4.3 Panel e of the current figure — answered, not redesigned

- **The low mCherry point in "Active"** (fraction active ≈ 0.5): to be identified and checked
  (cell count, signal-to-noise, session notes). It stays in the data whatever the check finds;
  any exclusion is a decision for VS.
- **hM4D versus mCherry in fraction active** is −0.04 (95% CI −0.18 to 0.10). The frequency effect
  of hM4D is in the event rate of active cells and in the population rate, not in the fraction of
  cells that are active. That is a result, and the redesign reports it rather than looking for a
  version in which it is significant.

## 5. Panel k: the shock response

**What the data show so far.** The group-mean YrA trace rises more and for longer in hM3D. Per
animal, two of the five hM3D animals respond strongly, one goes negative, and controls also
respond. On the YrA index, hM3D versus control does not differ (2 s shock: *P* = 0.93; 3 s after
shock offset: *P* = 0.11). Four animals have motion artifacts during the 2 s shock.

**What can legitimately change.** The YrA mean over all cells is the weakest measure in the paper.
It is not the measure the rest of the figure uses, and it averages responsive and unresponsive
cells together. The redesign measures the shock response with the paper's own quantities:

1. **Per-event amplitude and event probability of pyramidal events around the shock** (S events,
   as in f and g), in 1 s bins from −5 to +10 s, per animal, as the panel.
2. **One pre-specified test**: hM3D versus control in per-event amplitude in the 20 s after shock
   offset. This is already the declared secondary comparison of the primary model (Holm *P* = 0.038,
   already in the paper text). The new panel shows its time
   course; it does not add a test.
3. **Optional, separately stated before it is run:** the fraction of shock-responsive cells per
   animal, from the existing `event_locked_responsiveness` section (its outputs exist and have not
   been read in this redesign; read them before deciding).

**What will not be done.** No search across signals, windows or summary statistics until one of
them crosses 0.05. With 5 versus 6 animals and two hM3D responders, a group-level effect at the
shock is not guaranteed by any measure. Whatever is pre-specified above is reported whether it is
significant or not, and the YrA null stays in the supplement. `statistics-checker` reviews the
pre-specification before the analysis runs, not after.

**Motion.** S events are deconvolved from the same traces, so the four animals' shock-window
motion can produce spurious events. The 20 s window after shock offset (point 2) begins after
motion has returned to baseline (motion check, 2026-10-17). The 2 s shock window is reported with
that caveat, and no frames are removed.

## 6. Order of work

| step | what | where | blocks |
|---|---|---|---|
| 0 | VS and PI agree §2–§3; the shock endpoint in §5 is written down and reviewed | — | everything |
| 1 | Identify and check the mCherry "Active" outlier; read the existing responsiveness outputs | Mac | §4.3, §5.3 |
| 2 | Homecage session type in the loader; pre/post-CNO analysis (§4.1) | Mac, ds cache rebuild | new e |
| 3 | SST parameter gate, then Minian on the 4 DIO mice (§4.2) | Razer | new d |
| 4 | Shock-locked event panel (§5) | Mac | new i |
| 5 | New composite layout, text, legend; move panels to the supplement | Mac | — |
| 6 | `scientific-code-reviewer` and `statistics-checker` on steps 2–4 | — | write-up |

Steps 2 and 3 run in parallel: step 3 is Razer compute, step 2 is Mac code. Step 2 needs the
homecage sessions loaded into `ds`, so the cache is rebuilt once (on the Razer if memory requires).

## 7. Decisions (VS, 2026-10-21)

1. **Story: yes.** Step 3 becomes the paper's key feature: the **double dissociation** —
   hM3D changes amplitude and not frequency, hM4D frequency and not amplitude — gets its own block of
   panels and its own pre-specified test (§9).
2. **Panel d (SST cells) in the main figure**, if the data are usable; the pyramidal pre/post-CNO
   panel too. The SST validation (Minian parameter gate on the DIO mice, then the runs) is **a
   separate project, in its own chat**, starting from
   [sst_dio_dreadd_validation_plan.md](sst_dio_dreadd_validation_plan.md).
3. **Panel e: homecage pre/post-injection.** LT1→LT2 goes to the supplement at most.
4. **Example events: keep, redone (§10).**
5. **Shock panel: the pre-specified test of §5.2 for now**; VS revisits after seeing it, with
   §5.3 open.
6. **Recall: supplement**, probably.

## 8. The scaffold

`run_figure2_dissociation_composite(ds, cfg)` (notebook: Figure 2, dissociation layout) writes
`<PAPER_DIR>/fig2_dissociation/`, beside the current `fig2/`, which stays as it is. Panels that
exist are placed from their files; panels not yet built are drawn as labelled empty boxes, listed
in the module and printed on every build, so the layout can be judged while the panels are made.

## 9. The double dissociation (new block, replacing the single primary panel)

The claim is a crossover: relative to control, hM3D moves amplitude and not frequency, and hM4D
frequency and not amplitude. The primary model already gives the four simple effects; a crossover
needs its own test, or it remains two separate results.

- **Test (to pre-specify with `statistics-checker` before running):** per animal, the
  trace-interval log amplitude and log population rate, each expressed relative to the control
  mean; a model of measure × group (Exc, Inh) with animal as a random effect; the measure × group
  interaction is the dissociation. Holm with nothing else; the four simple effects remain as
  reported.
- **Panels (proposal):** (i) per-animal amplitude change against rate change, one point per animal,
  control at the origin; (ii) the four effect estimates with 95% CI as a 2 × 2 (Exc/Inh ×
  amplitude/rate); (iii) the same 2 × 2 in other sessions — homecage after CNO (§4.1) and,
  descriptively, the linear track and recall — to show it is not specific to conditioning.
- **Already known, to be stated as it is:** hM3D frequency was not changed (trace rate ratio 0.79,
  95% CI 0.52 to 1.22), and hM4D amplitude was not detectably changed; "not changed" is an
  interval, not equivalence, at n = 5/6/6.

## 10. Example events, redone

The current panel picks, for each group, the event whose peak height is closest to the **pooled**
median, so the three peaks are equal by construction and the panel cannot show that hM3D events are
larger. Redone: each group's own median event (by its own amplitude distribution, from the trace
interval), on a common y-scale, with the group's average event waveform (mean ± s.e.m. across
animals) beside it. Same detector, same threshold, no per-group scaling.

## 11. Why memory can differ without a detectable shock response

The behavioural difference does not require a group difference in the shock response as measured
here. The trace-interval amplitude (hM3D) and rate (hM4D) effects are present during learning, and
the shock response is measured in five trials from 5–6 animals per group in the population-average
activity, which is blind to which cells respond. A null at the shock is therefore consistent with
the behaviour; it is not evidence against it, and it does not oblige a significant shock panel.
