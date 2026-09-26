# Plan: a top-level runner that executes the Minian pipeline notebook over many sessions

Status: **planning**, written 2026-09-26. Nothing built.
Depends on: `plans/local_minian_pipeline_plan.md` — the `minian-native` env (§4.2), the
`-ORIG` set-aside and `--scratch` link (§5.2), the gate (§6), the queue in
`caban/session_queue.py` (§9). This plan is §9's "shape of the code", revised by VS's
requirement that the notebook itself runs.

## 1. What it is for

VS's workflow was: open a per-type WORKING notebook, uncomment the next `dpath`, run all,
mark it `DONE`, re-comment it, repeat; several notebooks open at once, purely to run
sessions in parallel by hand. The per-type split was a convenience for a human, not a
difference in method.

The runner replaces the human loop, not the notebook: pick sessions, then — attended or
unattended — each one is prepared, the **real pipeline notebook** is executed on it, and the
results (figures, videos, numbers) land in the session's folder and, in attended mode, inline.
Manual use of the notebooks stays exactly as it is.

## 2. The central decision: execute the notebook, do not re-implement it

The notebook's effective parameters are **not** all in its parameter cell: later cells
override them — e.g. `param_first_spatial['sparse_penal'] = 0.0001` (≈ cell 73), the
`param_first_temporal` block (≈ cell 84) — and plan §3.2's frozen set is the post-override
result. Anything that re-implements the pipeline has to reproduce those overrides by hand and
is one missed line from silently different output. Executing the notebook top to bottom gets
them for free. So: **papermill** (the standard notebook-as-batch-job executor) runs a copy of
the template in the `minian-native` kernel and saves the executed copy, figures included.

## 3. The canonical template: `pipeline-WORKING-TFC_cond-4.ipynb`

VS, 2026-09-26: one canonical notebook; the per-type ones existed for hand-parallelism.
Checked by diffing code cells (comments and the parameter cell excluded):

- `TFC_cond-4` ≡ `TFC_cond-3`, line for line.
- vs `prev/…-HC`, `-LT1`, `-Test_B` (the per-type ones): only `threads_per_worker = 1` (theirs
  2 — dask scheduling, not numerics) and two blocks keyed on G25's TFC_cond `dpath` (a
  hand-picked `del_frames` list and a hand-drawn `subset_mc` crop), inert for every other
  session. `TFC_cond-4` is the superset.
- `TFC_cond-1` additionally carries a selenium PNG-export cell; `pipeline.ipynb` is the
  upstream-style notebook with 2 GB workers. Neither is used.
- `prev/…-TFC_cond-RED` has **different** parameters (`wnd 10`, `size_thres (10, None)`) for
  red-channel recordings and is out of scope here.

The template is read from `~/code/minian_vsekulic` (working copy of the fork, §4.2 of the
parent plan) and its md5 recorded per run, so a later edit to the notebook is visible.

## 4. What the runner changes in the copy it executes

Only what VS changed by hand, all in the parameter cell, all recorded in the run record:

| line | set to | why |
|---|---|---|
| `dpath = …` | the session's `Miniscope/` folder; every other `dpath` line commented | the manual uncomment/re-comment step |
| `interactive` | `False` | gates every viewer, preview **and the parameter grid searches** (cells ≈71, 81 re-run `update_spatial` / `update_temporal` over `itt.product(...)`) — searched values that are never selected |
| `interactive_CNMF` | `False` | gates `CNMFViewer` and the `unit_labels` assignment (§4.1) |
| `want_mc_video`, `want_final_video` | `True` | write `minian_mc.mp4` and `minian.mp4`; no search involved |

`interactive_noparam` needs no setting: it only acts inside `interactive` blocks.

### 4.1 Consequence: no `unit_labels` coordinate

Cell ≈128, under `interactive_CNMF`, does `A.assign_coords(unit_labels=cnmfviewer.unit_labels)`
(and C, S, b0, c0) before saving. With the flag off the saved arrays lack that coordinate.
Harmless: `caban` never reads `unit_labels`, and without manual curation the viewer's value is a
copy of `unit_id` (`CNMFViewer.__init__`'s fallback). Recorded per run.

## 5. One session, step by step

1. **Prepare**: `set_aside_minian_output(<Miniscope>, reason, scratch_root=FUTROLA)` —
   renames any existing output to `-ORIG` (now also `minian_run/`) and links
   `minian_intermediate` onto FUTROLA.
2. **Execute**: papermill runs the edited copy in the `minian-native` kernel with the fork as
   working directory (the first cell imports `minian` from there). The executed notebook is
   written *as it runs* into `minian_run/`, so progress and a failure's traceback are on disk.
3. **Measure**: sample memory (kernel + dask workers) and wall time throughout.
4. **Report** (in the `caban` env, from the saved `minian/`): cell count, footprint-size
   distribution, `corr(C, YrA)`, summary figures (§6).
5. **Videos**: small re-encodes (§7).
6. **Clean up**: delete the scratch folder and link on success — except gate runs.
7. **Record**: `minian_run/run.json` — status, timings, peak memory, template + md5, fork
   commit, env, the §4 edits, the report numbers.

## 6. What lands in the session folder

```
<session>/Miniscope/
├── minian/                  notebook output (A, C, S, YrA, motion, A/C/S.npy, …)
├── minian.mp4               notebook's 4-panel video, full quality — kept as the sanity check
└── minian_run/              everything the runner adds
    ├── pipeline.ipynb       the executed notebook: every figure, exactly as a manual run
    ├── pipeline.html        the same, viewable in any browser
    ├── summary_*.png        max projection + footprints, sample traces, report plots
    ├── *_small.mp4          small re-encodes of the videos
    └── run.json             the record (§5 step 7)
```

`minian_run/` joins `NOTEBOOK_OUTPUT_DIRS` so a re-run sets it aside with the rest.

## 7. Videos

`minian.mp4` is a 2×2 grid (`generate_videos`): **raw** (`varr`) | **CNMF input** (`Y_fm_chk`:
denoised, background-removed, motion-corrected, ×1.5 gain) / **residual** `Y − A·C` |
**reconstruction** `A·C`. No YrA panel.

The notebook encodes at `crf 18, preset ultrafast` — fast, large. The runner re-encodes at the
**same pixel dimensions**, higher CRF, slow preset: much smaller files, small visual cost.

- `minian.mp4`: full-quality original **kept** (VS's sanity check) + `minian_small.mp4`.
- `minian_mc.mp4`: replaced by `minian_mc_small.mp4` once the small one is verified readable.
- **Open (VS):** an extra runner-made video, leaving the notebook's untouched — candidates
  `A·(C+YrA)` (per-cell reconstruction before denoising) and/or a preprocessing strip
  (raw → denoised → background-removed → motion-corrected).

## 8. The top-level notebook: `notebooks/run_minian_pipeline.ipynb`

Same four-cell shape as `recompute_yra.ipynb`, code in `caban/minian_runner.py`:

1. **Setup** — data root, scratch root, template, mode.
2. **Choose** — from `scan_sessions`: all never-processed, by mouse, by session type, an
   explicit list, or the gate set.
3. **Preview** — queue table: session, type, n_avi, status (pending / done / failed), last
   run's time and peak memory. Replaces the commented `dpath` lists and `DONE` marks; the
   template itself is never edited.
4. **Run** — serial, resumable (sidecar = `run.json`). **Attended**: after each session show
   its summary figures and small video inline. **Unattended**: progress table + log only;
   a failure is recorded and the queue moves on.

## 9. Resources

- **One session at a time** (VS), measuring peak memory; revisit after a few sessions.
- The template's cluster is `n_workers = int(os.getenv("MINIAN_NWORKERS", 6))` with
  `memory_limit="32GB"` per worker on a 32 GB machine. `MINIAN_NWORKERS` is the knob that
  needs no notebook edit; start at the notebook's own default and lower it if memory says so.
- Video reads come off the USB disk (~60 MB/s); intermediates go to FUTROLA.

## 10. Out of scope for the first version

- Red-channel sessions (`TFC_cond-RED` parameters).
- G25 (manual `del_frames` / crop, keyed on an old `/Users/vsekulic/data/...` path that would
  not match on MINISCOPE anyway), and FRAMCa2 generally.
- The ~36 aborted stubs (parent plan §2).
- Cross-registration of the new output (parent plan §7).

## 11. Setup and order of work

1. `papermill` into the `caban` env; register `minian-native` as a Jupyter kernel.
2. Build `caban/minian_runner.py` + the notebook; add `minian_run` to the set-aside list.
3. Dry run: execute the template on the **shortest never-processed HC session**; check the
   executed notebook, html, videos, memory.
4. **Gate**: G06 `2021_10_18-TFC_cond/09_52_24-HC1` — set aside, run, compare stage by stage
   against `minian_intermediate-ORIG/` (parent plan §6.1). Nothing is batched before it passes.
5. Batch in the parent plan's §8 order.
