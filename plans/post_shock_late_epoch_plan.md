# Plan: add a `post_shock_late` epoch for a within-trial early-vs-late post-shock contrast

Status: **not started.** Written 2026-08-21 for a fresh session to implement.

---

## 1. Why

Two literatures disagree about *where* in a trace-fear-conditioning trial a hippocampal
manipulation should act:

- **Trace interval.** The classical account: CA1 bridges the CS–US gap.
- **Post-shock window.** Puhger et al. 2024 (*iScience* 27:109035) find **no** bulk CA1 calcium
  response during the trace interval at all, a large sustained response to the footshock,
  memory impairment when CA1 is silenced 0–40 s after the shock, and **no** impairment when the
  same silencing is delivered 140 s after the shock.

`caban/sp_rates_lmm.py` now tests both loci (see §2). What it cannot yet do is reproduce
Puhger's *internal* control — the early-versus-delayed post-shock comparison — as a clean
within-trial contrast. This plan adds it.

## 2. State of the code (read this first — it is recent and not obvious)

A previous session split the TFC epoch definitions. Before that split, `post_shock` meant the
**entire 198 s inter-trial interval**; any result predating the split averaged the post-shock
response over ~10x its own duration.

Current epochs in `caban/epoch_analysis.py`:

| epoch | window | notes |
|---|---|---|
| `pre_tone` | 35 s ending at tone onset | locked confirmatory reference |
| `pre_tone_matched` | 20 s ending at tone onset | exposure-matched baseline; **overlaps `pre_tone`** |
| `tone` | 20 s | |
| `trace` | tone offset → shock onset | 15 s trial 1, 20 s trials 2–5 |
| `shock` | 2 s | excluded from everything confirmatory |
| `post_shock` | **20 s from shock offset** | `TRACE_MATCHED_WINDOW_S` |
| `iti` | shock offset → next tone onset (198 s) | the old `post_shock`; defined, not used by the pipeline |

Trial timing (from `TraceFearCondSession.__init__`, `caban/sessions.py:1261-1264`):
tone onsets `[185, 420, 660, 900, 1140]`, tone 20 s, shock onsets `[220, 460, 700, 940, 1180]`,
shock 2 s. One ITI therefore looks like:

```
shock offset                                                    next tone onset
     |                                                                  |
     |<-- post_shock -->|<------ unnamed, ~143 s ------>|<-- pre_tone -->|
     0                20s                             163s            198s
```

Two module-level epoch sets in `caban/sp_rates_lmm.py`, and **the distinction is load-bearing**:

- `TFC_EPOCHS` — everything the event table computes. Carries **both** baselines.
- `TFC_DISJOINT_EPOCHS` — the subset safe to model **jointly**, i.e. on a shared epoch factor.
  Excludes `pre_tone_matched`, because it overlaps `pre_tone` in time.

> **Do not skip this.** Putting both time-overlapping baselines on one epoch factor is a
> duplicated-data bug: 20 s of every baseline enters the likelihood twice, which inflates the
> baseline's precision, corrupts the epochs-within-trial variance component, and leaves the
> Bambi NB-GLMM's sampler on a near-collinear ridge. It was observed as `fit_rate_group_epoch_model`
> running **113 minutes instead of ~1**. `build_mouse_trial_epoch_rate_table` now filters to
> `TFC_DISJOINT_EPOCHS` at its own source. Preserve that invariant.

Confirmatory family is **three** Holm-corrected tests (`holm_correct_confirmatory`):
`trace_amplitude`, `trace_vs_baseline`, `post_shock_vs_baseline`. Last run:

| test | p_raw | p_holm | |
|---|---|---|---|
| `trace_amplitude` | 0.0089 | 0.027 | reject |
| `trace_vs_baseline` | 0.998 | 1.0 | — |
| `post_shock_vs_baseline` | 0.935 | 1.0 | — |

## 3. The problem this plan fixes

`post_shock_vs_baseline` pools each epoch across trials (`aggregate_over_trials`) **before**
differencing, so its reference is a mixture:

| pooled reference (`pre_tone`) | latency from a shock |
|---|---|
| `pre_tone[0]` | none — shock-naive |
| `pre_tone[1..4]` | 163 s after shocks 0–3 |

So it leans early-vs-late but is not that contrast: one naive window is mixed in, trial indices
do not align, and `shock[4]` has no late counterpart at all (the session ends first).

## 4. What to build

A `post_shock_late` epoch: **20 s starting 140 s after shock offset**, per trial.

- 20 s → `TRACE_MATCHED_WINDOW_S`, exposure-matched to `trace`, `post_shock`, `pre_tone_matched`.
- 140 s → Puhger et al.'s delayed-inhibition onset.
- Lands at 140–160 s post-shock-offset, inside the currently-unnamed gap. **Verify it overlaps
  nothing**: `post_shock` ends at 20 s, `pre_tone` begins at 163 s. It is disjoint from both, so
  unlike `pre_tone_matched` it **may** join `TFC_DISJOINT_EPOCHS`.

Then the new descriptive contrast, at aligned trial indices with no naive window:

```
per cell:  log_amp(post_shock) − log_amp(post_shock_late)
```

## 5. Steps

### 5.1 `caban/epoch_analysis.py`

1. Add `'post_shock_late'` to `EPOCH_NAMES` and a colour to `EPOCH_COLOURS`.
2. Add module constant near `TRACE_MATCHED_WINDOW_S`:
   `POST_SHOCK_LATE_ONSET_S = 140.0`, with a comment citing Puhger's delayed-inhibition timing.
3. Add a `post_shock_duration_s`-style parameter `post_shock_late_onset_s=POST_SHOCK_LATE_ONSET_S`
   to `get_epoch_frames`, and a branch:
   onset = `shock_offsets[i] + round(post_shock_late_onset_s * fps)`,
   offset = onset + `round(post_shock_duration_s * fps)`.
4. **Reuse the existing `iti_end` guard** from the `post_shock` branch — raise (do not truncate,
   do not return `None`) if the window does not fit before `post_shock_offsets[i]`. Factor the
   two branches' shared guard rather than copy-pasting it (CLAUDE.md: no duplicated pathways).

### 5.2 `caban/sp_rates_lmm.py`

5. `TFC_POST_SHOCK_LATE_EPOCH = 'post_shock_late'`.
6. Add to `TFC_EPOCHS` **and** `TFC_DISJOINT_EPOCHS` (it is genuinely disjoint — §4).
7. Add to `TFC_MATCHED_EPOCHS` so it gets its own decomposition figure
   (`decomposition_post_shock_late.png`) alongside the existing three.
8. Compute the new contrast in `run_sp_rates_lmm`, reusing the existing helper:
   `fit_and_report_epoch_delta(df_fine, TFC_POST_SHOCK_EPOCH, ..., reference_epoch=TFC_POST_SHOCK_LATE_EPOCH)`.
   Check the helper's stats filename is keyed on **both** epochs before doing this — as written
   it names the file after `response_epoch` alone, so this call would collide with the existing
   `coprimary_epoch_delta_post_shock.txt`. Fix the filename to include the reference.
9. `compute_all_epoch_deltas` defaults to `TFC_DISJOINT_EPOCHS`, so step 6 gives the descriptive
   `post_shock_late − pre_tone` delta and its `epoch_delta_forest` row for free. Confirm it is
   flagged `is_confirmatory = False`.

### 5.3 Inferential status — do not get this wrong

10. **`post_shock_late` is DESCRIPTIVE.** Do **not** add it to `holm_correct_confirmatory`; the
    family stays at exactly three. Do **not** add it to `build_secondary_fdr_table`. Its role is
    to characterize the post-shock null, not to test a hypothesis. Report the estimate and its
    95% interval, not a significance verdict.

### 5.4 METHODS

11. Update `analysis_methods_templates/sp_rates_lmm_methods.md`: the epoch table in the
    "Epoch structure" section, the early-vs-late rationale with the Puhger citation, and an
    explicit statement that the confirmatory family remains three tests.
12. Update `analysis_methods_templates/sp_rates_lmm_figure_guide.md`'s decomposition note —
    it currently says three figures; it becomes four.

## 6. Verification

- Stub-session check of the window arithmetic, as used during the epoch split:
  build an object with the timing arrays from §2 and print
  `get_epoch_frames(stub, e, t)` for every `e` and `t in (0, 1, 4)`. Assert
  `post_shock_late` is 20 s, at 140–160 s after shock offset, and **disjoint from both
  `post_shock` and `pre_tone`** at every trial index.
- **Trial 4 is the risk case**: its `post_shock_offsets` is end-of-recording, not a next tone
  onset. Confirm against a real session that the recording extends ≥160 s past the last shock
  offset. If it does not, the guard will raise — that is correct behaviour, but the plan then
  needs a decision (drop trial 4 from this contrast explicitly, or shorten the offset), not a
  silent skip.
- Confirm `build_mouse_trial_epoch_rate_table` now returns **5** epochs per mouse-trial, and that
  `fit_rate_group_epoch_model` still samples in roughly a minute. A 4→5 level epoch factor adds
  two interaction parameters; that is legitimate, non-degenerate cost, but if sampling time
  blows up again, suspect an overlap you did not intend.
- Re-check that `plot_epoch_profile` (defaults to `TFC_DISJOINT_EPOCHS`) renders sensibly with
  five x-positions.

## 7. Gotchas

- Do **not** run the pipeline in the background (AGENTS.md). Hand the user a reload snippet and
  let them run it (CLAUDE.md).
- The pipeline is expensive; `build_epoch_and_run_tables` does one event-detection pass over all
  epochs. Adding an epoch costs one extra window slice, **not** a re-detection — so adding
  `post_shock_late` is cheap. Do not build a second table.
- Group DISPLAY order is always `mCherry`, `hM3D`, `hM4D` (CLAUDE.md); use
  `caban.single_unit_common.DREADD_DISPLAY_ORDER`.
- No silent skips or fallbacks anywhere; raise with a clear message (CLAUDE.md).
- Comparing two decomposition figures by eye is the difference-of-significance fallacy. The
  early-vs-late claim is carried by the within-cell delta, not by two independently-fit figures.

## 8. Out of scope

- Any change to the three-member confirmatory Holm family.
- Any change to `pre_tone` / `pre_tone_matched`, or to the locked confirmatory contrasts.
- Reprocessing / re-extraction (the activity-dependent ROI-detection question) — separate work.
