# Handover (2026-10-11): after the Minian batch — the left-out sessions run, cross-registration, Figure 2

For the next chat. **Read this whole file first**, then the documents in §2 as needed. It supersedes
[handover_razer_batch_and_review.md](handover_razer_batch_and_review.md) for everything after 2026-10-06.

## 0. Rules VS set — follow them exactly (also in memory)

- **One chat works in this repo at a time.**
- **Every command on the Razer** (`ssh minastirith`, WSL Ubuntu) goes through `~/bin/logrun "<title>" <command…>`,
  including read-only checks; every file copied from the Mac gets a logged line (`logrun "copied X from the Mac" md5sum X`).
  **Read** `~/minian.log` with plain `ssh minastirith 'grep …'` (through logrun the output is appended to the log).
- **Heavy compute and memory jobs run on the Razer**, never the Mac (the Mac crashed 2026-09-29). The Mac does analyses
  from the cache. **Minian runs on the Mac are parked.**
- **Storage:** MINISCOPE (always carried) is a **mostly read-only data snapshot**; writes reach it only through the verified
  sync. **Analysis caches live on FUTROLA** (`/Volumes/FUTROLA/caban_cache/npy_files`, = `utilities.NPY_SAVE_PATH`; loading
  fails loudly if FUTROLA is not mounted). **Plots stay on the Mac's disk** (`~/data/vsekulic/OF_test/plots`).
- **Before any delete or unregister, confirm on screen where the live copy is.**
- **Never run `caffeinate`**; tell VS when the Mac must stay awake.
- **Queue follow-ups on the Razer** (screen + a script), not as Mac-side waits. VS runs nothing for the batch.
- **Don't change logic outside the task's scope** (VS, 2026-10-09). Ask first.
- **No silent fallbacks; hard-fail** (CLAUDE.md). Imports at the top. Descriptive names, no `phase1`-style labels.
- **Run `scientific-code-reviewer` and `statistics-checker` before any result is written up.**
- Group display order mCherry, hM3D, hM4D; `run_pipeline.ipynb` stays in the repo root (VS declined the move).
- **Lesson from 2026-10-10:** a background run failed 15 min after launch and sat unnoticed for ~25 h. **After launching
  anything, check it within ~20 min, and again before telling VS it is running.**

## 1. In progress on the Razer — check this first

**The left-out sessions run, attempt 2, launched 2026-10-11 10:51** (`screen left_out`, `~/bin/run_left_out_sessions.sh`,
logged in `~/minian.log` under "left-out sessions run, attempt 2"). Plan:
[unlabelled_and_salvage_sessions_plan.md](unlabelled_and_salvage_sessions_plan.md). It runs:

1. **G18 `2022_02_07-TFC_cond/14_03_44-HC1` alone as a check**, on its finalised files 0–3; the script stops unless the
   run is `done` and `C` has exactly 4,000 frames.
2. **The 38 pending sessions by day** (`run_minian_batch.py --by-day`): 27 unlabelled track-day sessions (G07 days 1–3,
   G08 days 1–5, G18 day 1), 7 misc (HC0 ×2, check1, iso, HC1a, LT1a, HC2b), and 4 on their finalised files (G05 and
   G06 LT1 with a mid-recording gap; G11 HC2 and G18 HC1 that stopped early).
3. **G09 `2021_11_08-TFC_cond/19_28_33-HC3`** by path (its old partial output is set aside by the run).

About 15–20 h in all. **Attempt 1 failed** (2026-10-10 10:14) at the check: the YrA replay demanded the unused videos be
present in the *staged* copy, where staging had left them out on purpose. Fixed in `2d2b72d`
(`yra_recompute.replay_frame_set` verifies them in the original session folder instead); tested on a mock staged folder.

**Monitor:** `ssh minastirith tail -f minian.log`, or only progress lines:
`ssh minastirith 'tail -f minian.log | grep --line-buffered -E "^=====|^\[[0-9]+/[0-9]+\]|: done|FAILED|check|status [0-9]"'`.
The 20-min `status` lines read "N of 522 batch sessions done, F failed, P pending" (522 = 484 + 38).

**When it finishes:**

1. Check every session is `done` (status report `~/minian_batch_status.md`; failures have their error in the note column).
   A failed session: read its error in the log, fix, `mr.clear_failed_run(item)` (keeps the failed run as
   `minian_run-failed-<time>`), rerun by `--path`.
2. **Sync to MINISCOPE** (needs MINISCOPE on the Mac; ~1 h per 30 GB; the Razer's disk is the bottleneck, ~13–18 MB/s):
   on the Razer `python scripts/sync_manifest.py manifest ~/sync_manifest.json` (in `screen`, logrun); then on the Mac a
   chained script like `~/minian_sync/run_final_sync.sh` (waits for the manifest, `scp` + md5 check,
   `scripts/sync_to_miniscope.py`, then `sync_manifest.py mark` on the Razer). Activate conda **before** `set -euo pipefail`.
   `~/minian_sync/run_sync_parts.sh <part>` runs a named manifest `~/minian_sync/sync_<part>.json`.
3. Snapshot the status report: `scp minastirith:minian_batch_status.md reports/`; commit.
4. Close A1, A2, A5, A6 in [figure2_per_cell_completion_plan.md](figure2_per_cell_completion_plan.md) §6.A and the
   status of [unlabelled_and_salvage_sessions_plan.md](unlabelled_and_salvage_sessions_plan.md).
5. Recompile the cross-registration list (`python scripts/compile_crossreg_groupings.py`): members now `pending` get
   their `minian` output.

## 2. Documents

| document | what it holds |
|---|---|
| [figure2_per_cell_completion_plan.md](figure2_per_cell_completion_plan.md) | **the main plan and the one to-do list (§6, items A–E)**; §2.8 the frame-rate before/after and reviews |
| [miniscope_frame_rate_plan.md](miniscope_frame_rate_plan.md) | B1, done: 19.76 fps, the per-session check, G05 Test_A re-timing (§5) |
| [unlabelled_and_salvage_sessions_plan.md](unlabelled_and_salvage_sessions_plan.md) | the 38-session Razer run in progress |
| [crossreg_batch_runner_plan.md](crossreg_batch_runner_plan.md) | cross-registration: registry, runner, rename; §3.1 the checks; §6 the groupings |
| [../reports/crossreg_groupings_report.md](../reports/crossreg_groupings_report.md) | **one-stop report on cross-registration**: logic, groupings, roles, checks, VS's notes, unused sessions, decisions |
| [../caban/crossreg_groupings_spec.yaml](../caban/crossreg_groupings_spec.yaml) | the short definition of the 18 groupings (compiled to `crossreg_groupings.json` by `scripts/compile_crossreg_groupings.py`) |
| [isoflurane_recording_analysis_plan.md](isoflurane_recording_analysis_plan.md) | D4, flagged: G05's iso recording (and G21 monitor2, a recovery recording) |
| [fps_inventory_scratch.md](fps_inventory_scratch.md) | every frame-rate use in the code |
| [../docs/epoch_modulation.md](../docs/epoch_modulation.md) | the per-cell epoch-modulation analysis (Fig. 2 panels K–P); Part Y current results; §A.4 YrA primary |
| [../docs/paper/paper_figure2.md](../docs/paper/paper_figure2.md) | the Figure 2 text (Results/Methods/notes); post-shock now a secondary comparison |
| [yra_recompute_plan.md](yra_recompute_plan.md), [yra_unit_alignment_plan.md](yra_unit_alignment_plan.md) | why YrA was recomputed (rows sheared against S/C; recall had none) |
| Done (banners "ALL DONE"): [local_minian_pipeline_plan.md](local_minian_pipeline_plan.md), [minian_batch_runner_plan.md](minian_batch_runner_plan.md), [razer_runner_plan.md](razer_runner_plan.md), [session_staging_copier_plan.md](session_staging_copier_plan.md) | |

## 3. Where things stand (2026-10-11)

- **Minian batch:** 484 of 492 done (2026-10-06), all synced to MINISCOPE (2026-10-07). The left-out 38 (§1) are running.
- **Frame rate (B1): done.** `MINISCOPE_FPS = 19.76`; per-session check (±0.1 %) at load; G05 Test_A (24.64 fps) re-timed at
  load (`MINISCOPE_FPS_EXCEPTIONS`). Before/after in the Figure 2 plan §2.8: rates −1.2 % for all groups (the correct
  direction); **Fig. 2i post-shock amplitude went Holm 0.0497 → 0.0557**; the sequence test is unstable. Reviews done.
- **Fig. 2 text:** trace is the single primary comparison; post-shock a hypothesis-driven secondary (Puhger et al. 2024)
  reported with CIs (C7, decided). The Results paragraphs, interaction numbers, Methods sentence and note 9 are updated.
  **Other 20-fps numbers in that file are not yet refreshed** (imaging-rate line ~199, line ~104, note 11, the
  six-comparison sensitivity sentence, §5's number table, the legend).
- **`ds_cache.pkl`** rebuilt 2026-10-09 on the 19.76 code (11.5 GB, now on FUTROLA). It holds 197-frame binned rates; the
  code now rounds to 198 (binned rates only, not Figure 2) until the next rebuild. The loader releases what sessions rebuild
  on demand after each mouse (`07bb506`), so a rebuild peaks ~30 GB instead of ~119 GB.
- **Cross-registration:** groupings 1–18 defined and compiled (all 657 sessions accounted for: 625 in groupings, 32
  `not_used` with reasons); the published mappings checked. **Nothing new computed yet.** Runner not built.
- **The trial-average windows** (`population.PERIOD_FRAMES`, 20 frames ≈ 1 s; meant as 20 s/2 s) are item **B4**, kept out of
  the frame-rate work; the fix must use the per-trial trace (15 s on TFC_cond trial 1).

## 4. What to do next — in this order (VS agreed 2026-10-11)

1. **Build the cross-registration runner (D3)** on the Mac as code, ready to test on the Razer when the batch frees it
   ([crossreg_batch_runner_plan.md](crossreg_batch_runner_plan.md) §5, §7):
   - a protected, read-only copy of `cross-registration-WORKING.ipynb` (four versions exist in the fork: diff their code to
     find the canonical one, as for the pipeline notebook);
   - membership from `caban/crossreg_groupings.json`, via a scratch view of the mouse folder with links named
     `minian_crossreg<N>` to the real outputs (confirm on a real run that `open_minian_mf` follows them);
   - never overwrite mapping files (`session_queue.set_aside_crossreg_output`); check `A`/`C`/`S` unchanged; every registry
     session appears in the mapping;
   - **re-derive 1–7 for every mouse** (VS, 2026-10-10) and compare with the published mappings: a full reproducibility
     check. Expected difference: G15 grouping 2 (stale, refers to `prev/minian`). **G17:** the notebook's hard-coded fix sets
     `918_26_30` but MINISCOPE and the published mappings use `18_26_30` — the check must reproduce G17 exactly.
   - then compute 8–18 on the Razer; results synced to MINISCOPE; mappings cached on FUTROLA.
2. **C4 + C5 (VS's choice): the per-neuron part of Figure 2** ([../docs/epoch_modulation.md](../docs/epoch_modulation.md)):
   C4 = sort panel K on held-out trials (O.8); C5 = split-half stability (O.2) for the supplement.
3. **G17's LT2 in Fig. 2d:** it was recorded *after* conditioning (notes: "forgot LT2!! so after TFC and HC, put back on LT").
   Check whether Fig. 2d (LT1 → LT2 manipulation check, grouping 1) changes without G17; add it as an item in §6.
4. The rest, as convenient (the earlier suggestion, verbatim in essence):
   - **C1** event-proximal omnibus at ≥ 20,000 draws before its P is quoted; **C2** G09 trial-dropped sensitivity;
   - **C8** sensitivity table for Fig. 2i (window 15/20/25 s, plateau limit, minimum events per cell);
   - **C9** sequence test: many random balanced trial splits instead of one odd/even split; a positive control;
   - **B2** loader guard: assert `YrA_idx == S_idx` before aligning (~10 lines);
   - **the paper text refresh** of the remaining 20-fps numbers (§3);
   - **E4** the Minian Methods text (frame rate, re-timing, salvaged files now belong in it).
   Building the runner (1) is on the critical path and needs no input from VS; C4/C5 is the substance of the per-neuron
   panels in VS's draft.

## 5. Other open items (all in the Figure 2 plan §6)

B3 (the parked collaborator notice — recheck its numbers after B1), B4 (trial-average windows), C3 (Fig. 2c re-run — the
sample traces were regenerated 2026-10-09; check and close), C6 (delete `ds_cache-20260512-exported-YrA.pkl` after the
write-up), D4 (isoflurane; VS to decide on G21 monitor2), E1 (a second copy of MINISCOPE), E2 (FUTROLA gate-run links),
E3 (videos drop 1 frame in 5 — document), E5 (`recompute_yra.ipynb` outputs), E6 (bring `minian_open_items.md` up to date),
E8 (optional review videos, parked). Also: the orphan cache folder `<cache>/Test_B/crossreg/` (17 files, one stale, not read
by the loader — VS may approve deleting it); G05 track day 0 and the monitor days kept out for now.

## 6. Things learned the hard way (this chat)

- **Check background runs early.** See §0.
- **zsh does not word-split** `$(…)` lists in `for`; use Python for anything non-trivial over file lists.
- **PyYAML reads `16_29_01` as the integer 162901** (YAML 1.1): quote such values.
- **pandas now reads text columns as StringDtype**; a 2023 pickle has object columns, so `DataFrame.equals` says "differs"
  for identical data: compare by value.
- **Mapping files name sessions by start time only**; G12 has two sessions at 14:03:29 (resolve by the folder tag).
- **Session types for time-only folders** live in `session_queue.SESSION_LABELS`, never in folder renames.
- **The float `MINISCOPE_FPS`** breaks code that used it as an integer (slices, `np.arange` shifts): use `int(round(...))`.
- **Time Machine local snapshots** keep deleted files' space until they expire (`tmutil deletelocalsnapshots <date>` is VS's
  call); large swap also eats disk.

## 7. Repo state

Branch `feat/yra-unit-alignment`, pushed through `2d2b72d`. Uncommitted: `docs/0_INDEX_docs.md`, `plans/0_INDEX_plans.md`
(an index refresh not made by this chat — ask VS before committing).
