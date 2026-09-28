# The copier: every Minian run works on a local copy of its session

Status: **implemented 2026-09-28** (`caban/session_staging.py`; wiring in `caban/minian_runner.py`:
`stage_inputs`, `run_session(stage_root=…)`, `stage_out_session`, `finish_stage_out`,
`run_all_staged`; `scripts/run_minian_batch.py --stage-root`). Unit-tested on the Mac; the
end-to-end test on the Razer is recorded in §10.
Depends on: [razer_runner_plan.md](razer_runner_plan.md) (the Razer, its drives, Phase 3.4 and
3.4b measurements, the G14 bad read), [minian_batch_runner_plan.md](minian_batch_runner_plan.md)
(the runner, `run.json`, the set-aside), [yra_recompute_plan.md](yra_recompute_plan.md) (the
replay behind the full-frame check).

## 1. Why

On the Razer the session data live on **MINIRAZER, a WD My Passport hard disk on USB**, which
WSL reaches through `/mnt/e` (Windows' 9P file-sharing layer). Measured 2026-09-28
([razer_runner_plan.md](razer_runner_plan.md) §5, Phase 3.4b and "G14's bad read"):

- **It is slow, and slower when shared**: 80 MB/s for one sequential reader; three readers
  get 19 MB/s each, 58 MB/s in total — the heads seek between files.
- **Every run touched it many times**: the notebook reads the videos (glow removal and
  preprocessing, and again for `minian.mp4`'s raw panel), writes thousands of small zarr chunk
  files into `minian/` and two 0.5–1.3 GB videos, the re-encode reads and rewrites those videos,
  and the YrA replay reads the videos twice more. A run could not avoid competing with itself,
  and every other run, for one spinning disk.
- **Its read path has faults**: G14's YrA step once got 3 frames too many from `17.avi`
  (13:32, with three streams running); Windows logged a controller error on the same drive at
  14:33:00. The cause of the first is not established; both happened under load.

The Razer's own NVMe (`D:`, Samsung 990 PRO, 1.5 TB free), which holds WSL's disk and the
scratch folder, has none of these problems.

**What the copier does not fix.** The Razer's CPU is power-limited at about six busy cores
(Phase 3.4b): one stream with six dask workers already uses it, so the copier is not a way to
run more streams. Its gains are that no run waits on the HDD, the HDD only ever sees one
sequential copy at a time, and every byte a run reads has been verified.

## 2. What happens to one session

```
            MINIRAZER (/mnt/e, USB HDD)                    local NVMe (~/minian_stage, ~/minian_scratch)
            ─────────────────────────────                  ──────────────────────────────────────────────
stage-in    <session>/Miniscope/*.avi  ── copy, verify ──▶ <work folder>/*.avi
run         set-aside, minian_run/run.json (the record)     notebook, report, YrA, re-encode in the work
                                                            folder; intermediates in the scratch folder
stage-out   <session>/Miniscope/minian, *.mp4,  ◀── copy, verify ── <work folder>/minian, *.mp4,
            minian_run/<files>                                    minian_run/<files>
cleanup                                                     work folder and scratch folder deleted
```

1. **Stage-in** (copier thread; `mr.stage_inputs` → `ss.stage_in`). The session's videos —
   exactly the files the notebook's `load_videos` pattern `[0-9]+\.avi$` matches, nothing else —
   are copied into `<work folder>.stage-partial`, each verified (§5), and the folder is renamed to
   the work folder only when all have verified. Nothing in the session folder is touched; the
   session stays `pending`.
2. **Prepare** (main thread; `mr.run_session`). As before: the pre-flight import check, the
   set-aside of any old output in the session folder (`minian` → `minian-ORIG`, …), the scratch
   folder, `minian_run/run.json` written in the **session folder** with status `running`.
3. **Run** (main thread). The notebook's `dpath` is the **work folder**; `intpath` the scratch
   folder. The notebook, the report, the YrA recompute with its full-frame check (§5.3), the video
   re-encode and the README all read and write the work folder. The scratch folder is deleted.
   The record says `computed`: the results are complete and exist only in the work folder.
4. **Stage-out** (copier thread; `mr.stage_out_session` → `ss.stage_out`). `minian/`,
   `minian.mp4` and `minian_mc.mp4` are copied into the session folder under
   `<name>.stage-partial`, every file verified by reading the copy back, then renamed to their real
   names; the work folder's `minian_run/` files (executed notebook, html, logs, figures,
   parameters, README) are copied into the session's `minian_run/` beside `run.json`. The record
   says `done`. Then the work folder is deleted.

### 2.1 The work folder

`<stage root>/<mouse folder>/<day>/<session>/Miniscope`, e.g.
`~/minian_stage/G14-ST719-hM4D/2022_01_15-TFC_test-A/17_14_14-LT1/Miniscope` — the last four
components of the session path, unchanged. This is required, not cosmetic: the notebook's
`param_save_minian["meta_dict"] = dict(session=-2, animal=-4)` names the saved arrays' `session`
and `animal` coordinates from those path components, so a run in the work folder writes the same
arrays a run in place would. The YrA recompute's per-session tables (`UNUSED_VIDEOS`,
`NOTEBOOK_DEL_FRAMES`) are keyed on the same tail, so they apply unchanged.

Stage root: `~/minian_stage` on Linux (`mr.DEFAULT_STAGE_ROOT`, `$CABAN_STAGE_ROOT` overrides).
On the Mac there is none and runs work in place, as before (the Mac does not process sessions
while MINIRAZER is the working copy).

## 3. The copier: one thread, one job at a time

`mr.run_all_staged` (what `scripts/run_minian_batch.py` uses when there is a stage root) runs the
sessions one after another on the main thread, and gives all bulk I/O on the data drive to a
single background thread — a `ThreadPoolExecutor(max_workers=1)`, so its jobs run strictly in
the order submitted:

```
main thread:   [run 1 ─────────────────]  [run 2 ─────────────────]  [run 3 ───────
copier:        [in 1] [in 2]              [out 1] [in 3]             [out 2] [in 4]
```

- **Prefetch depth one**: while session *k* runs, the copier stages in *k+1*; when *k* finishes
  its stage-out is queued, and runs while *k+1* computes. A run starts only once its own stage-in
  has verified.
- **One job at a time** is the point: the HDD streams one file at full speed instead of seeking
  between readers (80 vs 58 MB/s in total, §1).
- **One copier per process**. Two batch processes would have two copiers competing for the drive.
  One stream is the plan anyway (§1); if two are ever run, expect them to contend.
- Stage-outs are reported as they finish (the copier prints its own line), and failures are
  collected into the batch's final list, as for unattended `run_all`.
- **On an interrupt** (Ctrl-C in the `screen`, or the process killed): the copy in progress
  finishes or is cut short — either way under its `.stage-partial` name — and queued copier jobs
  are dropped. §6 says how to pick up.

Transfer time per session: a 7-video session is ~2.6 GB (~35 s to read at 80 MB/s, twice for the
verification's second read, plus the local read-back); a result is ~1–3 GB. All of it overlaps a
run of 20 min to 2 h+.

## 4. States in `run.json`

The record stays in the **session folder** (`minian_run/run.json`) from start to end, so the queue
table always tells the truth about where a session is.

| status | meaning | results are | next |
|---|---|---|---|
| (no `run.json`) `pending` | not run; may have a prefetched work folder | — | a run |
| `running` | the notebook or a later step is in progress | work folder (partial) | wait |
| `computed` | every step finished; not yet copied back | **only in the work folder** | stage-out (automatic) or `finish_stage_out` |
| `computed` + `stage_out_error` | the copy back failed | **only in the work folder** | `finish_stage_out` |
| `done` | copied back, verified | session folder | — |
| `failed` / `interrupted` | a step raised / was interrupted | partial, discardable | `resume_after_notebook` (if the notebook completed) or `clear_failed_run` |

Fields added to `run.json` by staging: `work_dir` (absent in older records = the session
folder), `stage_in` (per file: bytes and md5; totals; seconds), `stage_out` (per output: files,
bytes; seconds), `timings.stage_in_s`, `timings.stage_out_s`, `staged_out` (time),
`stage_out_error` (while a stage-out has failed). `finished` is when the run's steps finished;
`staged_out` when its results reached the session folder.

## 5. What the verification proves — and what it does not

### 5.1 Stage-in: three md5s per video

`ss.copy_verified(…, reread_source=True)`: the md5 of the bytes **as read while copying**, of the
**copy read back** from the NVMe, and of the **source read a second time**, must all be equal (and
the sizes). The first two prove the copy holds what was read. The third is what catches a read
that went wrong — a copy of wrong bytes is otherwise perfectly self-consistent.

**Limit**: the second source read goes through the same Windows file cache. If the first read
brought wrong bytes into the cache (a fault below it: the USB bridge, the disk), the second read
may be served from the cache and agree. A fault in WSL's 9P layer above the cache would be caught.
This is why §5.3 exists as well.

### 5.2 Stage-out: read back

Every output file is copied under a `.stage-partial` name and its md5 compared with the copy read
back through `/mnt/e`. Real names appear only after every file of every output has verified, so
the session folder never shows a partial result under a real name. **Limit**: the read-back may
come from Windows' write cache rather than the platter, so it proves the transfer into Windows,
not the disk surface. A later, independent check is the sync to MINISCOPE (its own comparison).

### 5.3 The full-frame check (every run, staged or not)

The notebook writes the motion-corrected movie `Y_fm_chk` to scratch from its own read of the
videos. The YrA step replays the same movie from a second, independent read of the videos
(`caban.yra_recompute`) and now compares the two **in every frame** (`compare_replayed_movie(…,
n_frames=None)`), refusing to go on unless they are exactly equal (`full_movie_check=True`). On
the Razer they always have been, in every frame compared: 50 frames each for G05
`18_22_57-TFC_cond`, G05 `15_44_41-HC2` and the two G11 LT1 sessions, and all 17,683 of G14.
It costs one extra streamed pass, ~2 min for a 17k-frame session. In a staged run both reads come
from the work folder, so this check proves the local copy was read consistently; §5.1 proves the
local copy equals the source.

## 6. When something goes wrong

All in the `caban` env on the Razer, in Python, with `mr = caban.minian_runner` and `item` from
`mr.select_sessions(mr.discover_sessions(), path=["<mouse>/<day>/<session>"])[0]`.

| situation | what you see | do |
|---|---|---|
| a stage-in failed (bad read, disk full) | batch line `FAILED (stage-in)`; session `pending`; maybe `<work>.stage-partial` | nothing: the next attempt replaces the partial |
| interrupted before its run started | a work folder, session `pending` | nothing: `stage_inputs` replaces an unused work folder of a pending session |
| a step failed after the notebook completed | `failed`, `error` in the record | `mr.resume_after_notebook(item)` — reruns report/YrA/videos in the work folder, then stages out |
| the notebook failed, or the process died during it | `failed` / `interrupted` / stale `running` | `mr.clear_failed_run(item[, even_if_running=True])` — keeps `minian_run-failed-<time>/` (with the work folder's notebook and logs as `work_dir_run/`), deletes the work folder, back to `pending` |
| the stage-out failed or was interrupted | `computed` (+ `stage_out_error`); maybe `*.stage-partial` in the session folder | `mr.finish_stage_out(item)` — replaces the partials, copies again |

`clear_failed_run` refuses a `computed` session: its results are complete and exist only in the
work folder. Nothing deletes a work folder that holds results except a successful stage-out.

## 7. Space

`ss.stage_in` refuses to start below **150 GB free** on the stage root's disk (`MIN_FREE_GB`):
the videos (≤ ~7 GB for the pending sessions), the notebook's intermediates in scratch on the
same disk (tens of GB for the longest), the outputs, and the next session's prefetched videos.
The Razer's WSL disk has ~945 GB free.

## 8. Running and watching

```
ssh minastirith
screen -S minian_batch
~/bin/logrun "batch: <selection>" bash -c "source ~/miniforge3/etc/profile.d/conda.sh && conda activate caban && cd ~/code/caban && python -u scripts/run_minian_batch.py <selection> --n-workers 6 2>&1 | grep --line-buffered -v param.Dimension"
```

`ssh minastirith tail -f minian.log` shows each session's header, a progress line every 2 min
(cells done, memory), `staged in …` / `staged out …` lines from the copier, and each session's
report. `ls ~/minian_stage/*/*/*` shows the work folders present: at most the running session's
and the next one's, plus any `computed` session waiting for its copy back.

## 9. Decisions and rejected alternatives

- **Results go back one session at a time, not at the end of a batch** (VS's alternative): at the
  end, MINIRAZER would be days out of date and a problem with the NVMe would take the whole batch's
  results with it.
- **Not several copiers or parallel copies**: concurrent readers make the HDD seek (§1).
- **Not robocopy or a Windows-side copy**: it would need Windows paths into WSL's disk and a
  second tool; one sequential copy through `/mnt/e` runs at the disk's speed anyway.
- **Only the videos are staged in**: nothing else in the session folder is read by a run.
- **The record is never staged**: it stays in the session folder, so the queue, the set-aside
  refusals and `clear_failed_run` see the truth at every moment.

## 10. Record

- 2026-09-28: built. Unit test on the Mac (temporary folders): stage-in/stage-out round trip
  byte-identical; refuses an existing work folder and an existing output; replaces leftover
  `.stage-partial` copies; detects a second source read that disagrees with the copy (the G14
  case, simulated); the free-space refusal.
