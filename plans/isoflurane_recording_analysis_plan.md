# Plan: CA1 activity under isoflurane and on waking — a supplementary analysis (FLAGGED for VS)

Status: **flagged 2026-10-10, not started.** Waits for the session to be run through Minian
([unlabelled_and_salvage_sessions_plan.md](unlabelled_and_salvage_sessions_plan.md), misc). Tracked as D4 in
[figure2_per_cell_completion_plan.md](figure2_per_cell_completion_plan.md) §6.D.

## 1. The recording

Miniscopes were attached under brief isoflurane anaesthesia: 1–2 min down, then rapid waking. On one
day VS recorded during it (VS, 2026-10-10: "just to see what the neurons would look like"):

| | session | start | length |
|---|---|---|---|
| baseline | G05 `2021_09_01-TFC_test_B/15_11_43-HC1` | 15:11 | 6,009 frames, ~5 min, 20 fps setting |
| **isoflurane** | G05 `2021_09_01-TFC_test_B/15_20_05-iso` | 15:20 | 4,054 frames over 181 s |

The same day then had LT1 (15:37), HC2, TFC_test_B and HC3. G05 is hM3D; whether a drug was given
before these sessions should be confirmed (VS).

**Its timing is irregular.** 4,054 frames in 181 s is 22.4 fps on average, with 1,000-frame stretches
near 30 fps and one 214 ms gap, though the setting was 20 fps. The frame-rate check (±0.1 % of 19.76) will
reject it by design. So the analysis must work from its timestamps, not frame counts. Either list it
as an exception and re-time it as for G05 Test_A ([miniscope_frame_rate_plan.md](miniscope_frame_rate_plan.md)
§5), but with an irregular rate the grid has to be built from the timestamps themselves; or analyse it
on its own timestamps outside the loader.

## 2. Questions

1. What does CA1 activity look like under isoflurane compared with awake homecage, the same mouse
   minutes earlier: event rate, amplitude, synchrony (pairwise correlation, population coupling)?
2. How does it change over the 3 min as the mouse wakes: activity against time since recording start,
   in 15–30 s bins?
3. Per cell: are the cells active under isoflurane the same ones active awake? This needs HC1 and iso
   cross-registered for G05 (the crossreg runner, D3).

## 3. Design constraints

- **One mouse, one recording.** Descriptive only: no inference across animals, and no claim beyond
  "in this mouse". Cells are not independent units for any test. A within-mouse test can only use time
  (e.g. a permutation over time bins), and its result describes this recording only.
- **State changes during the recording** (anaesthesia → waking), so "the iso session" is not one
  condition. Report the time course, not a single average.
- **Movement:** under anaesthesia there is none, and on waking it starts. Check the behaviour camera
  for when the mouse first moves, and mark it on the time course.
- **Imaging artefacts:** focus and motion can change as the mouse wakes. Check motion correction and
  the field of view before reading activity changes as neural.
- A METHODS template in `analysis_methods_templates/`, copied into the output folder (CLAUDE.md).

## 4. Steps

1. Minian on the iso session (in the run plan above). HC1 is already processed (batch, `done`).
2. Timing: decide re-timing vs. timestamp-native analysis (§1); add it to the loader's exceptions or
   keep it outside the loader.
3. Population comparison (questions 1–2) on each session's own cells, no cross-registration needed.
4. Per-cell comparison (question 3), after the cross-registration runner exists (D3).
5. One supplementary panel set and a short write-up, after `scientific-code-reviewer` and
   `statistics-checker` (VS's rule).

## 5. Open for VS

- Which comparisons matter most, and whether it goes in the supplement of this paper or elsewhere.
- Whether any other mouse has a recording under isoflurane (only this one was found).
