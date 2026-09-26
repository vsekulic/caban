# Plan: a top-level runner that executes the Minian pipeline notebook over many sessions

Status: **§11 steps 1–4 done** (2026-09-26): the gate **passed** against production G10
`TFC_cond` — the re-run reproduces its 799 units to solver precision; see §13. Next: step 5,
the batch.
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

**The runner never reads `TFC_cond-4` itself** — VS may open and edit it. It reads
**`notebooks/minian_pipeline_BASELINE.ipynb`** (created 2026-09-26): an output-free copy of
`TFC_cond-4` — all 305 cells, code sources identical, execution counts and embedded figures
stripped (46.4 MB → 163 KB), kernel set to `minian-native`. Protected three ways: tracked in
git (any edit shows in `git diff`), read-only on disk, and the runner checks the md5 of its
concatenated cell sources (`5eb502c4a0252de26fc6170626d8ad00`) before every run and refuses on
a mismatch. It executes with the fork's working copy (`~/code/minian_vsekulic`) as working
directory, since the first cell imports `minian` from there.

## 4. What the runner changes in the copy it executes

Only what VS changed by hand, all in the parameter cell, all recorded in the run record:

| line | set to | why |
|---|---|---|
| `dpath = …` | the session's `Miniscope/` folder; every other `dpath` line commented | the manual uncomment/re-comment step |
| `interactive` | `False` | gates every viewer, preview **and the parameter grid searches** (cells ≈71, 81 re-run `update_spatial` / `update_temporal` over `itt.product(...)`) — searched values that are never selected |
| `interactive_CNMF` | `False` | gates `CNMFViewer` and the `unit_labels` assignment (§4.1) |
| `want_mc_video`, `want_final_video` | `True` | write `minian_mc.mp4` and `minian.mp4`; no search involved |

`interactive_noparam` needs no setting: it only acts inside `interactive` blocks.

**One addition: a final, read-only cell** appended to the executed copy, after everything else
has run. It writes every `param_*` dict, the §4 flags, `FRAMERATE`, `subset`, `n_workers` and
the package versions, *as the kernel holds them at the end*, to `minian_run/parameters.json`.
Reading them from the live kernel rather than from the notebook text is the only reliable way,
since later cells override the parameter cell (§2). It changes no computed value.

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
5. **Videos**: re-encode the notebook's two in place; make the two extra ones (§7).
6. **Clean up**: delete the scratch folder and link on success — except gate runs.
7. **Record**: `minian_run/run.json` — status, timings, peak memory, template + md5, fork
   commit, env, the §4 edits, the report numbers.

## 6. What lands in the session folder

```
<session>/Miniscope/
├── minian/                  notebook output (A, C, S, YrA, motion, max_proj, …; no .npy —
│                            the template's export cell 301 is commented out)
├── minian.mp4, minian_mc.mp4  the notebook's videos, re-encoded smaller in place (§7)
└── minian_run/              everything the runner adds
    ├── pipeline.ipynb       the executed notebook: every figure, exactly as a manual run
    ├── pipeline.html        the same, viewable in any browser
    ├── summary_*.png        max projection + footprints, sample traces, report plots
    ├── minian_raw_traces.mp4, minian_preprocessing.mp4   the two extra videos (§7)
    ├── parameters.json      every effective parameter, from the kernel (§4)
    ├── README.md            what each video shows, panel by panel; the parameter table; the run
    ├── minian_methods.md    copy of the paper Methods text (§7a)
    └── run.json             the record (§5 step 7)
```

`minian_run/` joins `NOTEBOOK_OUTPUT_DIRS` so a re-run sets it aside with the rest.

## 7. Videos

What the notebook's two videos contain (baseline cells 38 and 125):

- `minian.mp4` (`generate_videos`), 2×2: **top-left** raw movie (`varr`) · **top-right** CNMF
  input (`Y_fm_chk`: denoised, background-removed, motion-corrected; ×255/max × 1.5 gain) ·
  **bottom-left** residual `Y − A·C` · **bottom-right** reconstruction `A·C` (footprints ×
  denoised traces), scaled to `Y` by a least-squares factor fitted on 200 random frames. No
  YrA panel.
- `minian_mc.mp4`, 1×2: **left** `varr_ref` (denoised + background-removed, *before* motion
  correction) · **right** `Y_fm_chk` (the same, *after* motion correction).

Each session's `minian_run/README.md` spells this out, for every video in the folder, so it
never has to be looked up again.

The notebook encodes at `crf 18, preset ultrafast` — fast, large. The runner re-encodes at the
**same pixel dimensions**, higher CRF, slow preset: much smaller files, small visual cost.

Measured 2026-09-26 on G06 `09_52_24-HC1`'s 2021 `minian.mp4` (1216×1216, 20 fps, 238 s,
527 MB), first 60 s re-encoded with libx264 `preset slow`:

| crf | per 60 s | whole file | shrink | SSIM vs original |
|---|---|---|---|---|
| 18 ultrafast (notebook) | 133 MB | 527 MB | — | 1 |
| 20 | 35 MB | ~138 MB | 3.8× | 0.982 |
| **23** | 14 MB | **~55 MB** | 9.5× | 0.977 |
| 26 | 5.5 MB | ~22 MB | 24× | 0.975 |

Side by side (zoomed, ×4 brightness) the cells are indistinguishable at every setting; what
goes is background grain — kept at 20, slightly smoothed at 23, visibly softened at 26. The
notebook's own file is already lossy (crf 18). Over ~657 sessions: ~330 GB at the notebook's
setting vs ~35 GB at crf 23.

Decided (VS, 2026-09-26): same pixel dimensions, compressed only.
- `minian.mp4` — **re-encoded in place** at **crf 23** (VS, 2026-09-26), after the
  re-encode is verified readable with the right frame count. Still the full 2×2 sanity check.
- `minian_mc.mp4` — likewise.
- **Two extra runner-made videos**, small, in `minian_run/`, the notebook's video untouched:
  `minian_raw_traces.mp4` — `A·(C+YrA)`, each cell's activity before temporal denoising,
  beside `A·C`; and `minian_preprocessing.mp4` — raw → denoised → background-removed →
  motion-corrected, one panel per stage (the steps `minian_mc.mp4` does not show).

### 7a. Paper Methods

`analysis_methods_templates/minian_preprocessing_paper_methods.md`, a Nature-style Methods
section for the whole Minian procedure (acquisition format → preprocessing → motion correction
→ seed initialisation and refinement → CNMF-E spatial/temporal updates and merges → outputs),
with citations (Minian: Dong et al. 2022, *eLife*; CNMF-E: Zhou et al. 2018, *eLife*). Its
numbers are taken from the first run's `parameters.json`, not from reading the notebook — the
top cell alone gives wrong values (§2). Copied into every `minian_run/` at runtime
(`_copy_analysis_methods_template`, as `CLAUDE.md` requires), so each session carries the text
describing how it was made.

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

1. ✅ `papermill` into the `caban` env; register `minian-native` as a Jupyter kernel.
2. ✅ Build `caban/minian_runner.py` + the notebook; add `minian_run` to the set-aside list.
3. ✅ Dry run: execute the template on the **shortest never-processed HC session**; check the
   executed notebook, html, videos, memory.
4. ✅ **Gate**: G06 `2021_10_18-TFC_cond/09_52_24-HC1` — set aside, run, compare stage by stage
   against `minian_intermediate-ORIG/` (parent plan §6.1). Nothing is batched before it passes.
   *Done differently*: G06 HC1's reference proved to be a junk 2022 run (VS); the gate was
   passed against production G10 `TFC_cond` instead (§13).
5. Batch in the parent plan's §8 order.

## 12. Working notes for whoever implements this

Practical gotchas found while setting this up (2026-09-23 → 26), not recorded elsewhere:

- **Activating envs:** `source ~/bin/conda-init.sh && conda activate <env>` (it evals
  `conda shell.zsh hook`). `python`/`conda` are not on `PATH` otherwise. Each tool-driven shell
  command starts fresh, so prefix every command with it: `caban` for the runner, the queue and
  the set-aside scripts; `minian-native` for anything that imports `minian`.
- **Shell:** zsh, with `grep` aliased to `ugrep` — `grep -e "-x"` style patterns and some flags
  behave differently; use `/usr/bin/grep` in scripts. A space-separated list held in one zsh
  variable does **not** word-split in `for x in $LIST` — use arrays or Python.
- **Dask from a piped script gets no workers.** `LocalCluster` launched from Python read on
  stdin (`python - <<EOF`) starts with `{}` workers on macOS (spawn cannot re-import `<stdin>`).
  Test cluster code as a real notebook (`jupyter nbconvert --execute`) or a `.py` file with an
  `if __name__ == "__main__":` guard.
- **Notebooks that import `minian` must sit in `~/code/minian_vsekulic`** (or run with it as
  working directory): the first cell imports `minian` before `sys.path.append(minian_path)`.
  For papermill, pass the fork as `cwd`; the output notebook can be written anywhere.
- **`minian-native` bokeh:** any `conda install` into the env reinstates bokeh 2.4.3 under the
  pip-installed 1.4.0 (datashader's dependency). Re-run `envs/minian-native-postinstall.sh`
  afterwards; it re-applies the server's jinja2 patch and checks it by md5.
- **`param` warnings flood stderr** (`WARNING:param.Dimension: Use method 'get_param_values'…`)
  from param 1.13 + holoviews 1.12.7 — the server's own combination. Filter them from logs;
  they are not errors.
- **Registering the kernel** for papermill:
  `conda activate minian-native && python -m ipykernel install --user --name minian-native`.
- **Drives:** MINISCOPE and FUTROLA each on their own port, never a shared hub (MINISCOPE's
  README, rule 2). Spotlight is off for both. The notebook writes into the session folder on
  MINISCOPE; with `--scratch`, intermediates go to `/Volumes/FUTROLA/minian_scratch/…`.
- **Test data locations:** the gate session is
  `/Volumes/MINISCOPE/SSTCa2/G06-ST688_hM4D/2021_10_18-TFC_cond/09_52_24-HC1/Miniscope`
  (all 27 intermediate arrays in `minian_intermediate/`). Never point a notebook at a session
  with existing output without the set-aside first.

## 13. Implementation record (2026-09-26)

**Step 1.** papermill 2.7.0 from conda-forge into `caban` (additions only, nothing
updated). `minian-native` registered as a user kernelspec; this writes to
`~/Library/Jupyter/kernels/`, not into the env, so the bokeh caveat of §12 does not apply.
Smoke test: papermill in `caban` → `minian-native` kernel, cwd the fork, `minian` imported
from `~/code/minian_vsekulic/minian/`.

**Step 2.** Built, and tested piece by piece without touching a session:

- Papermill runs as a child process that reads the edited notebook on stdin (`papermill -`),
  so papermill, the kernel and the dask workers form one process tree. Memory is the RSS summed
  over that tree. An interrupt terminates the whole tree.
- The edits are checked, not assumed: each flag line exists exactly once in the parameter cell,
  no other cell assigns a flag, and no `dpath` line is active in the template (none is; the
  runner inserts one). After the run, `parameters.json` must show the flags and `dpath` as set,
  or the run fails.
- Queue: 657 sessions in G05–G21; 476 never processed, of which 34 are stubs (listed, left out);
  **442 queued**, 327 of them HC. One session with partial output
  (`G09/2021_11_08-TFC_cond/19_28_33-HC3`, `minian` + `minian_intermediate`) is excluded as
  not-never-processed. Shortest HC: `G13/2022_01_10-TFC_test_B_1wk/16_35_20-HC2` (4 `.avi`).
- A failed session stays `failed` until `clear_failed_run`, which removes what that run wrote
  and keeps its `minian_run/` as `minian_run-failed-<time>/`.

Findings while building:

- **`YrA.zarr` unit order.** Cell 284 reorders `A` to `C` but `YrA` is saved as
  `compute_trace` left it, so a fresh run may save `YrA` in a different unit order from `C`/`S`.
  In existing output, 31 of 53 `minian*` folders that hold A/C/S/YrA/max_proj have the same set
  in a different order and 1 has a different set. The report aligns `YrA` by `unit_id` and
  records `yra_unit_order_matches_C` per run; the saved arrays are not touched.
- Cell numbers in the baseline: `unit_labels` is cell 293 (not ≈128); the videos are cells 99
  and 287 (not 38/125).
- Free memory during the smoke test was 5.8 GB of 32 GB, with other work open. At 6 workers,
  watch `min_available_gb` in the first run.

Not built yet:

- The two runner-made videos (`minian_raw_traces.mp4`, `minian_preprocessing.mp4`, §7). They
  need the intermediates before scratch cleanup, and the "denoised" stage is not saved, so it has
  to be recomputed in `minian-native`.
- The paper Methods (§7a) and its copy into `minian_run/`. By design its numbers come from the
  first run's `parameters.json`.
- Attended mode shows the report and summary figures inline, but no video; the notebook's
  videos are 50+ MB each.

**Step 3, dry run** (2026-09-26): `G13/2022_01_10-TFC_test_B_1wk/16_35_20-HC2`, 4 `.avi`,
3,591 frames (3.0 min). Status `done`, **12 min wall** (notebook 532 s, re-encode 168 s,
report 7 s). Peak RSS 6.2 GB over 10 processes at 6 workers; system available memory never
fell below 5.6 GB. All 306 cells executed, no error outputs, 16 figure outputs; html 33 MB.
88 units, median footprint 368 px, median corr(C, YrA) 0.51, `YrA` unit order matched `C`.
Scratch deleted. `parameters.json` confirms the §2 override: `sparse_penal` is 0.0001 in
both spatial updates, not the parameter cell's 0.01. Re-encode at crf 23: `minian.mp4`
540 → 174 MB (3.1×), `minian_mc.mp4` 317 → 108 MB. That is less shrink than the 9.5×
measured on G06 in §7, presumably content-dependent.

- **Minian's videos drop one frame in five.** `minian.visualization.write_video` pipes raw
  frames without an input rate, so ffmpeg assumes 25 fps and resamples to `r=framerate` (20):
  `drop=` in the log, and both videos hold 2,874 = 0.8 × 3,591 frames. They play in real time
  but miss 20 % of frames. This is upstream behaviour and applies to every 2021 video too; the
  arrays are unaffected. The runner executes the notebook as-is, so this is recorded, not
  patched. The runner-made videos (§7) must declare the input rate.
- The notebook's ffmpeg writes its progress into `papermill.log`; the progress reader now
  scans the whole log for papermill's last cell count.
- Empty `<mouse>/<day>/<session>/` folders stay under `minian_scratch/` after cleanup.

**Step 4, gate run** (2026-09-26): G06 `2021_10_18-TFC_cond/09_52_24-HC1`, 5,972 frames,
`keep_scratch=True`; 15 min wall, peak 5.1 GB. Compared by `caban/minian_gate.py`; report at
`<session>/Miniscope/minian_run/gate_report.json`, figures `gate_units_*.png` beside it.

| stage | result |
|---|---|
| `varr`, `varr_ref` | identical |
| `motion` (re-estimated) | identical, all 5,972 frames |
| `Y_fm_chk` (re-run) | identical |
| `Y_fm_chk` (2021 motion applied in `caban`, cv2 5.0) | ±1 in 132 frames — diagnostic only |
| `sn_spatial` / `max_res` | rel. diff 4e-15 / 0 |
| `A_init`/`C_init` | 450 = 450, all matched, corr 1.000 |
| `A_mrg` | 325 → 334; 275 matched with centroid shift ~1e-13 px (identical footprints), corr median 0.947 |
| final `A`/`C` | **105 → 334**; 89 of 105 matched (85 %), matched corr median 0.973, p5 0.82 |

So the pipeline reproduces everything deterministic, exactly, and the two runs share most
footprints through the first CNMF update. They differ in which units merge, and then
decisively in the second spatial/temporal update: the reference run went 325 → 105, the re-run
334 → 334. Many of the 245 re-run-only footprints are large or ragged; the reference's are compact.

**What the reference is.** The "2021" intermediates are dated **2022-02-22**, four months after
the recording. This session has no `minian_crossreg*`, and HC sessions were never
cross-registered (`local_minian_pipeline_plan.md` §2), so this output never fed a published
number. No surviving notebook matches it: the two that name `09_52_24` (`prev/pipeline-WORKING_clean`,
`prev/pipeline-WORKING_prev`) use `pnr_refine noise_freq 0.02`, which would have given
different seeds, whereas the seeds here are identical. The fork's CNMF code is unchanged since
2021-09-10. So the reference ran the template's parameters up to seeding, then something else
in the CNMF updates — parameters or solver versions — that no record shows.

Also: `local_minian_pipeline_plan.md` §3.2 lists `seeds_init max_wnd 7`. The template has 15,
and identical seeds show the 2022 run used 15 as well.

VS, 2026-09-26: that HC run is junk; compare against a production TFC_cond (G05 or G10).

**Gate against production** (2026-09-26): G10 `2021_11_23-TFC_cond/16_32_14-TFC_cond`, 26
`.avi`, 25,995 frames, against `minian_crossreg1_crossreg2_crossreg4_crossreg6_crossreg7`
(production A/C/S; parent plan §3.4). Chosen over G05 because the YrA plan already knows this
session in detail. No intermediates survive for it, so the gate compares `motion`, `max_proj`
and the final units (`caban/minian_gate.py`, report `minian_run/gate_vs_production_report.json`).

| check | result |
|---|---|
| `motion` | identical |
| `max_proj` (max over the motion-corrected movie) | identical |
| units | 799 = 799, same `unit_id`s in the same order, all matched |
| `A` / `C` / `S` max relative difference | 4e-6 / 2e-6 / 2e-5 |
| footprint centroid shift | median 3e-9 px |
| per-unit corr(C) | min 0.999999999995 |

**Passed.** The runner reproduces the production output to solver precision: the ~1e-5 run-to-run
noise of `yra_recompute_plan.md` §2.2, and no unit-set difference at all. Wall 1.05 h
(notebook 52 min, re-encode 10 min); peak RSS 12.7 GB at 6 workers, system available
never below 5.6 GB. `prev/minian` in this session is a byte-level copy of production
(799/799, corr 1.000) — not an independent run.

**The `YrA` defect comes from the notebook.** The re-run's saved `YrA.zarr` holds units 343/346
in place of `C`'s 344/347 — the exact discrepancy `yra_recompute_plan.md` §12.2 found in the
production output. A single run produces it. So it comes from the notebook itself, most likely
`YrA.sel(unit_id=mask)` in cell 297, with `YrA` sorted and `mask` in `C`'s order. It is not a
second run's doing. The report records it (`yra_missing_C_units`, `yra_extra_units`) and no
longer fails on it. A run that failed only after the notebook finishes with
`resume_after_notebook`, without recomputing.
