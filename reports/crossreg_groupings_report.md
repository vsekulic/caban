# Cross-registration groupings: what was decided, what was checked, and why

**As of 2026-10-11.** This report collects everything decided and checked about cross-registration of
cells across sessions in the SSTCa2 cohort (G05–G21): how cross-registration works, which sessions form
each grouping and why, what the checks found in the published mappings, the recording irregularities
that shape the groupings (with VS's own notes from the data drive), and every session left out, with its
reason. Nothing new has been cross-registered yet; this is the state before the runner is built.

Where things live:

| what | file |
|---|---|
| **the definition** (short, human-readable) | [caban/crossreg_groupings_spec.yaml](../caban/crossreg_groupings_spec.yaml) |
| the full per-session list, compiled from it (for code) | [caban/crossreg_groupings.json](../caban/crossreg_groupings.json) |
| the compiler and all checks | [scripts/compile_crossreg_groupings.py](../scripts/compile_crossreg_groupings.py) |
| session labels, sessions never run, unfinished videos | `caban/session_queue.py`: `SESSION_LABELS`, `SESSIONS_NOT_RUN`, `UNUSED_VIDEOS` |
| the plan (runner, registry, renaming) | [plans/crossreg_batch_runner_plan.md](../plans/crossreg_batch_runner_plan.md) |
| the overall to-do list (items D1–D4) | [plans/figure2_per_cell_completion_plan.md](../plans/figure2_per_cell_completion_plan.md) §6.D |

## 1. How cross-registration works

Each session is processed by Minian independently, so a cell has a different identity (unit id) in every
session. Cross-registration matches cells across the sessions of one mouse. It uses the forked Minian
notebook `cross-registration-WORKING.ipynb` (`~/code/minian_vsekulic`), run once per mouse and grouping:

1. It aligns the sessions' fields of view (`shiftds`).
2. It matches footprints by centroid distance (`param_dist = 5` px).
3. It writes, per mouse and grouping *N*, `mappings_crossreg_<N>.{csv,pkl}`, `cents_crossreg_<N>.pkl` and
   `shiftds_crossreg_<N>.nc`. The mapping table has one row per matched cell group and one column per
   session, holding that session's unit id (empty where the cell was not found in that session).

**Cross-registration never changes a session's `A`/`C`/`S`** ([plans/local_minian_pipeline_plan.md](../plans/local_minian_pipeline_plan.md)
§3.4); it only adds mapping files.

**Membership until now was set by folder names.** A session joined grouping *N* by having `_crossregN`
appended, by hand, to its Minian output folder (e.g. `minian_crossreg1_crossreg2_crossreg4_crossreg6_crossreg7`),
and the notebook opened every folder whose name contained `crossreg<N>`. From now on membership is defined
in the spec and compiled into the JSON; folder names are no longer the source of truth. The plan is
eventually to rename every output folder to plain `minian` ([crossreg plan](../plans/crossreg_batch_runner_plan.md) §4).

**The key trade-off: drift (VS).** The more sessions a grouping holds, and the longer the time it spans,
the fewer cells are matched across all of them, because the field of view and cell footprints drift. A
large grouping may have very few, or no, cells common to every session. Hence the groupings below are
targeted, and some overlap on purpose (8, 9, 10a, 10b).

**Principle for extra sessions (VS, 2026-10-11).** A session that exists in only some mice is left out of
shared groupings: including it would cut those mice's matched-cell counts relative to the others. Missing
sessions, by contrast, cannot be helped; where they make groupings uneven across mice, this is stated.

## 2. The groupings

1–7 are the **published** groupings behind the existing analyses (the loader uses 1, 4, 6 and 7). They stay
exactly as they are. 8–18 are new, defined 2026-10-10/11, to be **computed ahead of time, once**, so later
work needs only the small mapping files (cached on FUTROLA), not MINISCOPE. **No analysis uses 8–18
automatically**; each is used deliberately, when a question needs it (VS).

| # | name | sessions per mouse | mice / sessions | question |
|---|---|---|---|---|
| 1 | conditioning_day | LT1, LT2, TFC_cond | 17 / 51 | Fig. 2 |
| 2 | cond_to_testB | TFC_cond, Test_B | 16 / 32 | |
| 3 | testB_day | LT1, Test_B (Test_B day) | 16 / 32 | |
| 4 | cond_testB_testB1wk | TFC_cond, Test_B, Test_B_1wk | 17 / 49 | used by the loader |
| 5 | testB1wk_day | LT1, Test_B_1wk (Test_B_1wk day) | 16 / 32 | |
| 6 | cond_all_tests | TFC_cond + all four tests | 17 / 82 | used by the loader |
| 7 | cond_testA_testA1wk | TFC_cond, Test_A, Test_A_1wk | 17 / 50 | used by the loader |
| 8 | conditioning_day_all | every role of the conditioning day (§3) | 17 / 137 | per-cell trajectory across drug onset and conditioning |
| 9 | conditioning_day_no_HC | LT_pre, CNO_early, CNO_late, LT_post, TFC_cond | 17 / 79 | as 8, without homecage (HC activity varies greatly) |
| 10a | CNO_early_late | CNO_early + CNO_late | 11 / 22 | right after vs ~20 min after injection (**key**) |
| 10b | preinjection_CNO_late | HC_preinjection + CNO_late | 9 / 18 | homecage before injection vs ~20 min after |
| 11 | all_LT | every LT1 of track days 1–3; every LT1/LT2 of every other day | 17 / 151 | place-cell stability over days and weeks |
| 12–14 | track_day1–3 | HC1, LT1, HC2 of that track day | 17 / 51 each | homecage vs track; reactivation in HC after LT |
| 15–16 | track_day4–5 | HC1, LT1, HC2 | 2 (G08, G09) / 6 each | as 12 |
| 17 | all_HC | every HC of track days 1–3, test days, and conditioning-day HC roles | 17 / 351 | homecage baseline over the experiment |
| 18 | iso_pair | G05 `2021_09_01-TFC_test_B`: HC1 + iso | 1 / 2 | awake vs isoflurane ([plans/isoflurane_recording_analysis_plan.md](../plans/isoflurane_recording_analysis_plan.md)) |

Dropped: **within each test day** (VS: "HC is not a behavior").

**Unevenness across mice**, after the extra-session rule:

- **8**: 7–9 sessions per mouse because sessions are *missing*, not extra (§3). G13, G18, G19 have all 9
  roles; G05–G11 have no pre-injection HC; G07 and G08 have no HC before TFC; G12, G14–G16, G20, G21 have
  one CNO recording. Matched-cell counts in 8 are therefore not comparable across mice; 9, 10a and 10b are
  the uniform versions.
- **17**: 16–25 HC sessions per mouse, from protocol differences between cohorts and days. It is also the
  largest grouping and the one most likely to match very few cells.
- **11**: 7–9 LT sessions per mouse (G15 has fewer test days). Track days 4–5 (G08, G09 only) are left out
  of 11 and 17 as extra, and appear only in 15 and 16.

## 3. The conditioning day: two cohorts, defined by role

The two cohorts recorded the conditioning day differently, so folder names do not line up across mice.
Sessions are therefore assigned to **roles**, one row per mouse in the spec's `conditioning_day` table.

| role | G05–G11 | G12–G21 |
|---|---|---|
| HC_start | HC1 | HC1 |
| LT_pre | LT1 (G07: **LT1b**, as published) | LT1 |
| HC_preinjection | — | HC2 (recorded right after LT1, before the injection) |
| CNO_early (right after injection) | CNO1 | CNO1 where there are two; G12's single CNO1 |
| CNO_late (~20 min after, just before LT2) | CNO2 | CNO2 where there are two; the single CNO1 of G14, G15, G16, G20, G21 |
| LT_post | LT2 | LT2 (G17: recorded after TFC, see below) |
| HC_preTFC | HC2 (none in G07, G08) | HC3 |
| TFC_cond | TFC_cond | TFC_cond |
| HC_postTFC | HC3 | HC4 (G17: HC3) |

**How CNO1 / CNO2 were identified.** VS recalled that CNO1 was right after the injection and CNO2 ~20 min
later, and that in mice with a single CNO recording it was the ~20-minute one. Session timing (start, length,
gap) and the notes on MINISCOPE confirm this:

- G05–G11: LT1, then a 13–35 min gap (the injection), CNO1 (5 min), 10–20 min, CNO2 (5 min), ~10 min, LT2.
- G13, G17, G18: HC2 right after LT1, then 2 min, CNO1, 13–16 min, CNO2, LT2.
- G14, G15, G16, G20, G21: HC2 right after LT1, then **22–24 min**, a single "CNO1" that runs straight into
  LT2. That is the ~20-minute recording, so it is **CNO_late**.
- G12: a single CNO1 4 min after HC2, then 15 min with nothing recorded, then LT2. So it is **CNO_early**;
  its notes say CNO2 was forgotten.
- G19: CNO1 starts 12 min after HC2 (9 min long), CNO2 follows immediately, then LT2. Its CNO1 note reads
  "basically entire wait period", so CNO1 = early, CNO2 = late.

**Caveats on the conditioning day:**

- **G17's LT2 was recorded after conditioning.** Its sequence is HC1, LT1, HC2, CNO1, CNO2, TFC_cond, HC3,
  LT2, HC4. The note says LT2 was forgotten and recorded after TFC and HC, "should still have been under the
  effects of DREADDs". **G17's LT2 is in published grouping 1 and Fig. 2d** (the LT1 → LT2 manipulation
  check), where it stands for LT2 under CNO *before* conditioning. That is worth checking for Fig. 2d (open).
  G17's HC4, recorded after that late LT2, has no counterpart in other mice and is not used.
- **G07 has LT1 and LT1b**; LT1b is the one in published grouping 1, so it is LT_pre, and LT1 is not used.
- **G18 has `CNO2-borked`** (disk full; re-recorded straight after as `15_03_49-CNO2`). Only the re-recording
  is used.
- **G12–G21's HC2 is a pre-injection homecage baseline that G05–G11 do not have.** It gets its own role
  (HC_preinjection) rather than being dropped, and is the "before" in 10b.

## 4. The published mappings: what the checks found (2026-10-10)

The compiler takes membership of 1–7 from each **mapping file's own session columns**, the ground truth of
what was registered, not from folder names. It checks every member. All checks pass:

- **Every mapped unit id exists in its member's output**, for all 7 groupings, 16–17 mice each.
- **The loader's cached copies equal the files on disk, by value.** The loader builds every `CrossRegMapping`
  with its cache under `<cache>/TFC_cond/crossreg/` (FUTROLA). A 2023 cache stores object columns where
  today's pandas reads strings, so the comparison is by value, not type.

Findings, each recorded in the JSON or the plan:

- **Grouping 1's files: a historical naming difference.** They are in each mouse's *conditioning-day* folder,
  not the mouse folder (grouping 1 is within that day), named `mappings_crossreg1.*` (no underscore) in 16
  mice and `mappings_crossreg_1.*` in G15. G06 and G07 have only the `.csv`. Content and cache agree.
- **G05 holds stray copies**: `mappings_crossreg1 - Copy.csv`, a set in the mouse folder (identical), and
  `old_mappings/`, which holds earlier, *different* versions of groupings 1–4 and 6. The published files are
  the ones in the mouse folder (2–7) and the conditioning-day folder (1).
- **18 members had no crossreg-tagged folder**: grouping 5's Test_B_1wk LT1 in all 16 mice, and grouping 3's
  Test_B LT1 in G10 and G11. These sessions were re-run by the 2026 Minian batch, and **their new output
  carries exactly the mapping's unit ids** (as does the set-aside original, `minian-ORIG`, where one exists).
  So the re-runs reproduced their cell ids and the published mappings stay valid. This is also a second,
  larger confirmation that re-running Minian reproduces production.
- **One stale published mapping: G15, grouping 2.** G15's conditioning session was re-processed in 2026 (417
  units; the earlier run, 451 units, is kept in `prev/minian`), and its groupings 1, 4, 6 and 7 were recomputed
  then. Grouping 2 dates from 2022 and refers to the earlier run: all 451 of its ids are in `prev/minian`. The
  JSON records the member as `output: prev/minian` with a `stale` note. Grouping 2 is not used by the loader.
- **One stale orphan cache file**: G15 grouping 4 under `<cache>/Test_B/crossreg/` (2023), referring to that
  earlier run. **Not read by the loader**, which reads the current copy. The 16 other files in that folder
  equal their mappings; they are leftovers of an older code version.
- **G12: two sessions started at 14:03:29** (the Test_B-day LT1 and TFC_test_A_1wk). Mappings name sessions
  by start time only; the compiler resolves each by the grouping's folder tag.
- **G16 had two output folders in its Test_B-day LT1.** `minian_crossreg2_crossreg3` was a byte-identical copy
  of G16's TFC_test_B output, put in the wrong folder. `minian_crossreg3` (13,923 frames = the LT1's own videos,
  and what every other mouse uses) is the real one, and G16's published grouping 3 used it (all 787 LT1 ids).
  **The copy was set aside on 2026-10-10** on MINISCOPE and MINIRAZER as `duplicate_of_TFC_test_B_output`
  (no "crossreg" in the name, so no grouping or scan can match it), recorded in `crossreg_duplicate_set_aside.json`.
- **G17's conditioning folder name.** On the old server it was misnamed `918_26_30-TFC_cond`; on MINISCOPE it is
  `18_26_30-TFC_cond`, and the published mappings use `18_26_30`. The notebook carries a hard-coded fix that
  *sets* `918_26_30`, which on MINISCOPE would mismatch. The runner's check must reproduce G17 exactly.

**Re-running all groupings (VS, 2026-10-10).** The runner will re-derive 1–7 for every mouse as well as
computing 8–18. That makes it a full reproducibility test of every published mapping. The published files
are set aside (`-ORIG`), never overwritten, and analyses keep using them unless VS decides otherwise. G15's
grouping 2 is the one expected difference.

## 5. Session labels and the notes on MINISCOPE

**28 sessions had no type in their folder names** (named by time only) and were labelled in
`session_queue.SESSION_LABELS`, without renaming anything on MINISCOPE:

- The track days of G07 (days 1–3), G08 (1–5) and G18 (day 1): each day a triplet in time order, labelled
  HC1 → LT1 → HC2 (the middle session is always the longest). **24 of these 27 carry a marker file in their
  folder** (`HC1.txt`, `LT1.txt`, `HC2.txt`), **and all 24 agree with the labels**. G18's three have none.
- G19 `2022_02_05-track_day3/16_04_02-HC3` is that day's post-track homecage (the day is HC1, LT1, HC3, with no
  HC2), so it is labelled **HC2**.

These had been wrongly treated as stubs and left out of the 2026 Minian batch; they are being run now
([plans/unlabelled_and_salvage_sessions_plan.md](../plans/unlabelled_and_salvage_sessions_plan.md)).

**VS's notes in the session folders, quoted:**

| where | note | bearing |
|---|---|---|
| G12 `2021_12_27-track_day1` | "put in room / 15 min later, hc1 / 5 min later, lt1 / immediately after, hc2 / immediately after (so 5 min after lt1 end), hc3"; days 2 and 3: "same as day1", "same as day2" | G12's track days have a deliberate fourth session, HC3; left out as extra |
| G13 `2021_12_27-track_day1` | "Same as G12 for today, but put 5 min break between HC2 and HC3 due to high gain of led..." | as G12 |
| G13 `2021_12_29-track_day2` | "led was now at 25 so didn't wait so long as yesterday. hc1, 5 min wait. lt1, hc2, hc3 then all back-to-back." | as G12 |
| G12 `2022_01_03-TFC_cond` | "Followed 'new' day 1 protocol in evernote. But forgot to record CNO2...." | G12's single CNO1 is the early one |
| G13 `2022_01_03-TFC_cond` | "Followed 'new' protocol in Evernote, went smoothly, didn't forget CNO2 unlike with G12..." | the G12–G21 protocol |
| G17 `2022_01_24-TFC_cond` | "Recorded first 10 min or so of CNO just to see if there would be changes, and now know that 10 led etc won't lead to bleaching. But forgot LT2!! so after TFC and HC, put back on LT to record LT2 and HC afterwards. Didn't do too many laps though, but should still have been under the effects of DREADDs...." | G17's LT2 is after TFC (§3) |
| G19 `2022_02_07-TFC_cond/16_57_49-CNO1` | "basically entire wait period" | CNO1 = early |
| G21 `2022_03_09-monitor2/15_24_21-HC1` | "right after plugged in, recovering from iso" | a second isoflurane recording (recovery) |
| G21 `2022_03_09-monitor2/16_22_52-HC2` | "after 30m rest" | its awake follow-up |
| G26 `2023_05_22-TFC_cond` (later cohort) | "cno1 – right after injection / +10 min / cno2 – right before LT2 / – so total period after injection before recording = 20 min" | the CNO1 / CNO2 protocol |

## 6. Sessions in no grouping, and why

The compiler requires every G05–G21 session with video (657) to be in a grouping or on this list, and fails
otherwise. Currently 625 are in groupings and 32 are listed here (as `not_used` in the spec).

| sessions | why |
|---|---|
| G05 `2021_08_20-track_day0` (HC1, LT1, HC2) | track day 0: left out for now (VS) |
| G07 `2021_10_18-TFC_cond/13_13_24-LT1` | LT1b is the conditioning-day LT1 used, as published |
| G17 `2022_01_24-TFC_cond/19_13_33-HC4` | homecage after G17's late LT2; no counterpart |
| G05 `2021_09_07-TFC_test_B_1wk-redux` | test re-recorded when the bedding was changed; not for analysis by default (VS, 2026-09-28) |
| G12 track days 1–3 HC3, G13 track days 1–3 HC3, G14 track day 2 HC3 (7) | extra: second post-track homecage right after HC2 |
| G10 `2021_11_20-track_day3/14_05_55-HC1_high_led` | extra: 5 min test at higher LED power, right after HC1 |
| G05 HC1a, LT1a (`track_day1-tests`); G13 HC2b (`TFC_test_B_1wk`) | misc: short repeats |
| G06 `2021_10_07-check1/15_00_46` | misc: check recording |
| G12 and G13 `2021_12_13` | misc: **HC0**, GCaMP expression check before any track day |
| G20 and G21 `monitor1`, `monitor2` (5 sessions) | **NOTE (VS, 2026-10-11): kept out for now**, like the HC0s; G21 monitor2 is an iso-recovery recording |
| G07 `2021_10_20-TFC_test_B/15_49_59-TFC_test_B`, `16_25_43-HC3` | never run: the disk filled during recording; the HC3 has no finalised video |
| G14 `2022_01_18-TFC_test_B_1wk/14_52_35-…-borked` | never run: re-recorded straight after as `15_10_50-TFC_test_B_1wk` |
| G18 `2022_02_07-TFC_cond/14_57_23-CNO2-borked` | never run: re-recorded straight after as `15_03_49-CNO2` |

G05's `iso` session is in grouping 18 only.

## 7. Recordings salvaged or with gaps

These sessions are in groupings, but are being run through Minian on their **finalised video files only**
(the recording stopped partway through a file; `session_queue.UNUSED_VIDEOS`). The loader then pairs frames
with timestamps around the gap, as already done for G21 Test_B:

| session | used | left out |
|---|---|---|
| G05 `2021_09_09-TFC_test_A_1wk/16_11_07-LT1` | files 0–9, 12 (10,242 frames) | 10, 11 (mid-recording gap) |
| G06 `2021_10_22-TFC_test_A/12_18_46-LT1` | files 0–6, 11, 12 (8,315 frames) | 7–10 (mid-recording gap) |
| G11 `2021_12_02-TFC_test_A_1wk/15_32_02-HC2` | files 0–2 (2.5 min) | 3–5 (stopped early) |
| G18 `2022_02_07-TFC_cond/14_03_44-HC1` | files 0–3 (3.4 min) | 4–7 (stopped early) |

G09 `2021_11_08-TFC_cond/19_28_33-HC3` held a leftover partial Minian run from 2022. The recording is fine;
the old output is set aside and the session re-run.

## 8. Decisions, in order

| date | decision (VS) |
|---|---|
| 2026-09-26 | automate cross-registration like the Minian runner; membership computed, not encoded in folder names |
| 2026-10-10 | compute all groupings ahead of time; no automatic analyses; caches on FUTROLA |
| 2026-10-10 | groupings 8–18 as in §2; drop within-test-day groupings |
| 2026-10-10 | re-run all groupings, 1–7 included, as a reproducibility check; published files kept |
| 2026-10-10 | G16's duplicate output set aside |
| 2026-10-10 | the conditioning day defined by role; CNO_early / CNO_late as in §3; 10a and 10b both computed |
| 2026-10-10 | G07: LT1b; G05 track day 0 left out for now; keep a list of every unused session |
| 2026-10-11 | extra sessions left out of every grouping (track-day HC3s, G10 high-LED test, track days 4–5 from 11 and 17) |
| 2026-10-11 | G19's track-day-3 HC3 is its HC2; monitor days kept out for now (NOTE) |
| 2026-10-11 | the definition is a short YAML spec; the full list is compiled JSON |

## 9. Open

- **The runner** (D3), on the Razer once the current Minian batch finishes: re-derive 1–7, compute 8–18, with
  the reproducibility check (including G17's naming) and `A`/`C`/`S` unchanged.
- **G17's LT2 in Fig. 2d**: recorded after conditioning; decide whether it stays in the LT1 → LT2 check.
- **Monitor days and HC0s**: kept out for now; G21 monitor2 may join the isoflurane analysis.
- **G05 track day 0**: left out for now.
- **The orphan cache folder** `<cache>/Test_B/crossreg/` (17 files, one stale): not read; may be deleted.
- **After the runner**: the loader reads the registry, then every output folder is renamed to `minian`.
