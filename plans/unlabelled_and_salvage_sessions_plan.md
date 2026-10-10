# Plan: run Minian on the sessions the batch left out — unlabelled, misc and salvaged

Status: **planned 2026-10-10** (VS: "let's run it on Razer"). Items A1, A2, A5, A6 of
[figure2_per_cell_completion_plan.md](figure2_per_cell_completion_plan.md) §6.A. Runs with the existing
runner on the Razer ([razer_runner_plan.md](razer_runner_plan.md), [minian_batch_runner_plan.md](minian_batch_runner_plan.md)),
staged through the copier ([session_staging_copier_plan.md](session_staging_copier_plan.md)), synced to MINISCOPE
afterwards.

## 1. Why these were left out

- **Time-only names were treated as stubs.** The 2026-09-22 inventory
  ([local_minian_pipeline_plan.md](local_minian_pipeline_plan.md) §2) called every session folder named only
  by time an "aborted stub… almost certainly recording mistakes", to be eyeballed before queueing. Their
  lengths were never checked, and the runner encoded the rule (`minian_runner.is_stub`: no type in the
  name → never queued). 28 of them are real sessions.
- **Unfinished `.avi` files:** sessions whose recording stopped partway through a file failed in
  Minian's `load_videos` (`nb_frames` missing).
- **Leftover partial output:** G09 HC3 holds an aborted 2022 Minian run, and the runner refuses to
  overwrite it.

## 2. The sessions (38)

**Unlabelled track-day sessions: 27.** Each day is a triplet in time order: shorter, longest, shorter,
i.e. HC1 → LT1 → HC2 (VS, 2026-10-10: track days are homecage plus linear track, habituation; "these
MUST be processed").

| mouse | days | sessions |
|---|---|---|
| G07 | `2021_10_11-track_day1`, `2021_10_13-track_day2`, `2021_10_15-track_day3` | 9 |
| G08 | `2021_10_26-track_day1`, `2021_10_28-track_day2`, `2021_11_01-track_day3`, `2021_11_03-track_day4`, `2021_11_05-track_day5` | 15 |
| G18 | `2022_02_01-track_day1` | 3 |

**misc: 7.** A new category for checks and one-off recordings, so that no analysis picks them up as
ordinary sessions (VS, 2026-10-10):

| session | what it is |
|---|---|
| G12 `2021_12_13/14_43_31`, G13 `2021_12_13/15_53_03` | **HC0**: ~5 min homecage, before any track day; GCaMP expression check |
| G06 `2021_10_07-check1/15_00_46` | 3 min check recording |
| G05 `2021_09_01-TFC_test_B/15_20_05-iso` | 3 min under isoflurane, then waking; supplementary analysis in [isoflurane_recording_analysis_plan.md](isoflurane_recording_analysis_plan.md) |
| G05 `2021_08_23-track_day1-tests/13_59_41-HC1a`, `14_23_55-LT1a` | short repeats beside the labelled HC1 / LT1 |
| G13 `2022_01_10-TFC_test_B_1wk/16_38_56-HC2b` | 3 min repeat of HC2 |

**On their finalised `.avi` files only: 4.** Listed in `session_queue.UNUSED_VIDEOS`; the loader's
timestamp mapping then pairs frames with timestamps around the gap, as for G21
([yra_recompute_plan.md](yra_recompute_plan.md) §15.1).

| session | used files | left out |
|---|---|---|
| G05 `2021_09_09-TFC_test_A_1wk/16_11_07-LT1` | 0–9, 12 (10,242 frames) | 10, 11 |
| G06 `2021_10_22-TFC_test_A/12_18_46-LT1` | 0–6, 11, 12 (8,315 frames) | 7, 8, 9, 10 |
| G11 `2021_12_02-TFC_test_A_1wk/15_32_02-HC2` | 0–2 (3,000 frames, 2.5 min) | 3, 4, 5 |
| G18 `2022_02_07-TFC_cond/14_03_44-HC1` | 0–3 (4,000 frames, 3.4 min) | 4, 5, 6, 7 |

**Set aside and run: 1.** G09 `2021_11_08-TFC_cond/19_28_33-HC3`: the recording is fine. Its leftover
2022 partial output (`minian/` with only an all-NaN `motion.zarr`, `minian_intermediate/`) is set aside
first.

**Not run.** G07 `2021_10_20-TFC_test_B/16_25_43-HC3`: no finalised file at all (disk full that day).
G07 `15_49_59-TFC_test_B`, G14 `14_52_35-…-borked` and G18 `14_57_23-CNO2-borked` are ignored by VS's
decision (disk full; re-recorded).

## 3. Code changes

1. **Labels in one table, not in folder names.** `session_queue.SESSION_LABELS`: session tail → type
   (`HC1`/`LT1`/`HC2` for the 27; `misc` with a note for the 7). Discovery, the runner and the status
   report read the type from it when the folder name has none. Nothing on MINISCOPE is renamed.
2. **Stubs.** `minian_runner.is_stub` becomes "no type and no label", and `STUB_SESSION_TYPES` empties:
   every remaining session is now labelled.
3. **Skip unfinished files at staging (A1; absorbs A5).** `session_staging.input_files` leaves out the
   session's `session_queue.unused_videos`, so Minian only ever sees finalised files. One list then
   drives staging, the YrA replay and the loader's timestamp mapping.
4. **G09 HC3:** set aside its old output with the existing `session_queue.set_aside_minian_output`,
   recorded in `minian_set_aside.json` as for every other set-aside.

## 4. Checks before the batch

- **Timestamp rows of the 4 gapped sessions:** each left-out file must account for exactly 1,000
  timestamp rows (as G21's did), or the frame ↔ timestamp mapping would be off. Check before listing
  them.
- **One short session first:** G18 HC1 (4 files). Check that `C` has 4,000 frames, the full-frame check
  passes, and the loader's checks pass on it.
- **Dry run:** `scripts/run_minian_batch.py --dry-run` lists exactly the 38.

## 5. Run, sync, record

One stream on the Razer, as for the batch: about 0.3–0.6 h per session, so roughly **15–20 h**. Then the
sync to MINISCOPE (manifest on the Razer, copy from the Mac), the status report, and the Figure 2 plan
§6.A items closed.

## 6. Status

- 2026-10-10: planned.
