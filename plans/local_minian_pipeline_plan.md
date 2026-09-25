# Plan: run Minian locally to generate C/S/YrA for every session

Status: **planning**, written 2026-09-22. Nothing executed.
Depends on: `plans/yra_recompute_plan.md` (§11 implemented; the work queue in §13 is the
piece this reuses) and `plans/cbp_server_migration_plan.md` §4, which this **un-defers** —
a full CNMF-E run needs a real Minian environment, which the YrA recompute did not.
Out of scope: re-processing the 130 sessions that already have output (§3.1); any change
to existing crossreg mappings or cell identity.

## 1. Why this exists

`caban` analyses 130 sessions. The cohort recorded **657**. The missing 527 are not
missing data — the `.avi` are all there — they were simply never put through CNMF-E.
That excludes, among other things, every homecage baseline and every CNO-onset session,
so questions like "what does a cell do before vs after CNO?" cannot currently be asked
at all.

Two things make this tractable now that were not obvious before:

- the pipeline's parameters turn out to be **one frozen set**, identical across every
  production notebook (§3.2), and
- **no human curation was ever applied** (§3.3),

so "automate the WORKING notebook" is a real proposition rather than a euphemism for
"redo a year of manual work".

## 2. Inventory

G05–G21, folders that contain numbered `.avi`. Measured 2026-09-22 over both backup
drives (1,106 `Miniscope` folders scanned; G01–G04, `testMouse`, `baseplating`,
`CA1-Cre1`, G22–G27 excluded).

| session type | has crossreg output | plain `minian/` only | nothing | `.avi` files, if nothing |
|---|---|---|---|---|
| HC | 0 | 31 | **328** | 2,485 |
| LT1 | 30 | 12 | **89** | 1,256 |
| CNO | 0 | 6 | **22** | 163 |
| LT2 | 17 | 0 | 0 | |
| TFC_cond | 17 | 0 | 0 | |
| TFC_test_A | 17 | 0 | 0 | |
| TFC_test_A_1wk | 16 | 0 | 0 | |
| TFC_test_B | 16 | 0 | 1 (`-borked`) | 18 |
| TFC_test_B_1wk | 16 | 0 | 1 (`-borked`) | 18 |
| aborted stubs (timestamp-only names, `iso`, `HC1a`, `HC2b`, `LT1a`) | 1 | 1 | ~36 | ~330 |
| **total** | **130** | **50** | **477** | **4,288** |

Three populations, and they are different jobs:

1. **130 with crossreg output** — done. `yra_recompute` covers their `YrA`. **Not touched
   here** (§3.1).
2. **50 with plain `minian/`** — CNMF-E already run, never cross-registered. The cheap
   win: no CNMF-E needed, only `YrA` recompute plus cross-registration.
3. **477 with nothing** — the actual CNMF-E job.

The ~36 aborted stubs (5–17 `.avi`, names that are bare timestamps) are almost certainly
recording mistakes. They should be listed and eyeballed before being queued, not
processed on the strength of having `.avi` in them.

## 3. Why automating the WORKING notebook is safe

### 3.1 Re-running CNMF-E on the existing 130 is still forbidden

`plans/yra_recompute_plan.md` §3 stands: re-deriving footprints for a processed session
gives new unit ids, hence new crossreg mappings, hence invalidates every longitudinal
result. This plan **only adds** sessions. The single exception is the gate (§6), which
re-derives a handful of processed sessions into a **separate output directory** purely
for comparison, and never promotes them.

### 3.2 The parameters are one frozen set

Every `param_*` assignment and later override was extracted from all 60 notebooks in the
Minian mirror and diffed. Across the thirteen *production* notebooks — `HC`, `HCb`,
`LT1`, `LT2`, `TFC_cond-1..4`, `TFC_cond-RED`, `Test_A`, `Test_B`, `Test_B-LT1`,
`Test_B_1wk`, `Test_B_1wk-LT1` — **every value is identical**:

```
denoise            median, ksize 5
remove_background  tophat, wnd 15
seeds_init         wnd_size 1000, method rolling, stp_size 500, max_wnd 7, diff_thres 3
pnr_refine         noise_freq 0.06, thres 1
ks_refine          sig 0.05
seeds_merge        thres_dist 10, thres_corr 0.8, noise_freq 0.06
initialize         thres_corr 0.8, wnd 10, noise_freq 0.06
init_merge         thres_corr 0.8
get_noise          noise_range (0.06, 0.5)
first/second spatial   sparse_penal 1e-4, dl_wnd 5
first/second temporal  p 1, sparse_penal 1, add_lag 20, noise_freq 0.06
first/second merge     thres_corr 0.8
```

The only divergences anywhere are in `TFC_cond-EXPERIMENTAL` and `TFC_cond-FRESH`
(`ksize 3`, `wnd 7`, `sparse_penal 1e-5`/`0.5`, `noise_freq 0.05`) — notebooks whose
names say they are experiments. This set is what produced every published number, and it
is what the automation freezes.

### 3.3 No human curation was ever applied

The notebook opens `CNMFViewer` and does
`A.assign_coords(unit_labels=("unit_id", cnmfviewer.unit_labels))`, which looks like a
manual accept/reject step. It is not: across all 130 sessions `unit_labels` is either
`0..n-1` or **equal to `unit_id`**, with no negatives and no duplicates. No unit was ever
relabelled. The `### CHECKPOINT` cells were visual *parameter* checks, and §3.2 is their
outcome.

### 3.4 Cross-registration does not modify `A`/`C`/`S`

`cross-registration.ipynb` writes exactly three things: `mappings.pkl`, `cents.pkl`,
`shiftds.nc`. It never re-saves per-session arrays. So adding sessions to a *new* crossreg
grouping cannot disturb existing per-session output or existing mappings. This is the
fact that makes the whole plan additive rather than destructive, and it is worth
re-verifying before §7 runs.

## 4. What CNMF-E needs that the YrA recompute did not

### 4.1 `Y` must be persisted — §11.4 does not carry over

The YrA recompute streams because `compute_trace` is a single pass. CNMF-E is not:
`seeds_init`, `pnr_refine`, `ks_refine`, `get_noise_fft` and `update_spatial` all make
repeated random access over the whole movie, which is why the stock notebook saves both
`Y_fm_chk` (frame-chunked) and `Y_hw_chk` (height/width-chunked) to `intpath`.

Per session, float32: ~10 GB per chunking for an 8-file HC session, ~19 GB for a 13-file
LT, ~38 GB for a 26-file TFC_cond — so **20–76 GB of scratch**, twice over, deleted
between sessions. This is the binding operational constraint and it dictates §5.

### 4.2 Dependencies: this needs a real Minian install

The YrA recompute vendored five functions. A full run pulls in `cnmf.py`,
`initialization.py` and `motion_correction.py` more or less entire. Missing from the
`caban` env: **`cvxpy`, `pyfftw`, `pymetis`, `rechunker`** (plus `holoviews`,
`datashader`, `panel` for the viewers, which the automation does not need, and `medpy`,
which only serves the anisotropic-denoise branch this pipeline never takes).

`pymetis` is the one to expect trouble from on arm64 — it is a C extension wrapping METIS.
**Corrected 2026-09-25, from reading the fork:** its only use is `graph_optimize_corr`
(`cnmf.py:1757`), which serves `seeds_merge`, `initA` and `unit_merge`. There METIS partitions
the correlation graph purely to *schedule* the out-of-core computation — edges inside a
partition are computed per partition, edges across partitions in a second pass
(`cnmf.py:1790-1808`) — so every requested correlation is computed whatever the partition.
It is a **performance** dependency, not a correctness one: if it will not build, a stand-in
partition gives the same correlations at higher memory/recompute cost, and the §6 gate
confirms it. (First written as "a decision point … seed merging changes which cells exist",
which was wrong.)

Vendoring is no longer the cheaper option at this surface area. **Decision to make:** a
separate conda env (`minian-local`) pinned to whatever numpy the fork tolerates, driven
from `caban` as a subprocess, versus forcing the fork onto numpy 2.4 in place. The first
is more likely to work and keeps `caban` untouched; the second avoids a process boundary.
Recommend the first, and record the env spec next to the fork.

**Built 2026-09-25: `minian-local`, spec in `envs/minian-local.yml`.** Better than either option
above: every pin from the production server env (`minian_vsekulic_cbp-db.yaml`) — Python 3.8.15,
numpy 1.20.2, dask/distributed 2021.2.0, xarray 0.16.2, numba 0.52.0, opencv 4.2.0, cvxpy 1.2.1,
pyfftw 0.13.0, pymetis 2020.1, scipy 1.9.1, scikit-image 0.18.1, zarr 2.17.1, … — solved exactly
as **osx-64 packages run under Rosetta** (`CONDA_SUBDIR=osx-64`; the env's own `.condarc` pins
`subdir: osx-64` so later installs stay Intel). No porting, no version drift: this is the 2021
software stack, differing from the server only in CPU (x86-64 under emulation vs native) and
OS. Minian is not installed; the fork is put on `PYTHONPATH`. Smoke test: all five pipeline
modules import; `pymetis.part_graph`, a `cvxpy` solve (ECOS/OSQP/SCS present) and `pyfftw`
all run correctly. GUI packages (bokeh/holoviews/panel/datashader) left out — no pipeline
module imports them. The §6 gate is still what establishes equivalence; Rosetta speed is
unmeasured until the first real session.

**Superseded the same night by a native build: `minian-native`, spec in
`envs/minian-native.yml` — this is the working env.** Of the ~30 production pins, all but five
exist as native osx-arm64 builds at the 2021 version — including every numerically heavy one
(numpy 1.20.2, dask, xarray, numba 0.52, cvxpy, pymetis, scipy, scikit-image). The five:

| package | 2021 | native | where it acts | gate stage (§6.1) |
|---|---|---|---|---|
| opencv | 4.2.0 | 4.5.0 | median/tophat, decode, warps | 1 |
| simpleitk | 2.0.2 | 2.1.1 (PyPI wheel) | motion estimation | 2 |
| pyfftw | 0.13.0 | 0.12.0 (same FFTW 3.3.10) | noise estimation | 3 |
| scikit-learn | 0.22.1 | 0.23.2 | GMM in seed refine, neighbours | 4 |
| medpy | 0.4.0 | 0.4.0 (PyPI) | import only | — |

Synthetic benchmark, same script in both envs: native is **1.1–1.5x faster** on every
operation (median+tophat, pyfftw, cvxpy, numba, matmul, pymetis), and the median+tophat
preprocessing output is **bit-identical** between opencv 4.2.0/Rosetta and 4.5.0/native — a
first, synthetic look at gate stage 1. `minian-local` (Rosetta, exact pins) is kept as the
fidelity reference: if a gate stage diverges under `minian-native`, re-running that stage under
`minian-local` says whether a version bump is to blame. Rosetta's general availability ends
after macOS 27, which is a second reason not to depend on it.

**Requirement (VS, 2026-09-26): the pipeline notebooks run as-is.** Not a headless
re-implementation: the same `pipeline*.ipynb` — the `WORKING-TFC_cond-1..4` and every
per-session-type notebook in `prev/` — opened in Jupyter and run. What that took:

- *Which server env actually ran them.* The server had several: `minian_cbp-db` (the export
  first built from; its unpatched bokeh cannot import under its own jinja2 3.1.4, so it cannot
  have run the GUI), `minian_vsekulic`/`_py311` (a 2026 modernisation, bokeh 3), and **`minian`**
  — the one `bin/jupyter_minian.sh` activates. `minian-native` now tracks `minian`, GUI and
  Jupyter stack included (bokeh 1.4.0, holoviews 1.12.7, panel 0.8.0, datashader 0.12.1,
  notebook 6.5.7, xarray 0.17.0, sk-video 1.1.10, …), with PyPI for what conda lacks on arm64.
- *The server's own patches.* `minian` had a one-line fix in `bokeh/core/templates.py` and
  `panel/io/resources.py` (`Markup` from `markupsafe`, not `jinja2`). Reproduced byte-for-byte by
  `envs/minian-native-postinstall.sh`, which checks the md5 of the server's copies.
- *Which Minian code ran.* The server env also carried conda-forge's stock Minian 1.0.0rc0 in
  `site-packages`, which differs from the fork in every module — including `20da1b5c fix: use
  fft filter for pnr computation`, a seed-refinement change. It is not what produced the
  results: the notebooks import `minian` from their own folder, which Jupyter puts ahead of
  `site-packages`, and the G06 outputs (Feb 2022) postdate the fork's Minian 1.1.0 (Sep 2021)
  plus VS's Dec 2021 commit. `minian-native` installs no Minian at all, so the fork's code is the
  only one importable.
- *Where it runs.* A working copy of the fork at `~/code/minian_vsekulic` (clone of
  `vsekulic_v4`; the `WORKING` notebooks and `prev/` are untracked in git and were copied from the
  `~/cbp-db` mirror, which stays untouched). Launch: `cd ~/code/minian_vsekulic && conda
  activate minian-native && jupyter notebook` — classic Notebook 6 in a browser, as the server
  did; the 2021 bokeh/panel viewers are not expected to work in VS Code's renderer.
- *Verified:* the notebooks' own import cells, `hv.notebook_extension("bokeh")` and the
  `LocalCluster` + `TaskAnnotation` setup execute under `jupyter nbconvert` from the working copy.
  One trap found doing so: the first cell imports `minian` before `sys.path.append(minian_path)`,
  so the notebook must live in the fork folder.

## 5. Storage

Measured across both backup drives, 2026-09-22:

| | size | files |
|---|---|---|
| `.avi` (base) | 3,262 GB | 14,847 |
| other base (BehavCam, timestamps, metadata, notes) | 15 GB | 23,833 |
| **base total** | **3,277 GB** | **38,680** |
| existing Minian output (`minian/`, `minian_crossreg*/`, `prev/`, `TFC_all/`) | 283 GB | 768,529 |
| `minian_intermediate/` (regenerable scratch) | 47 GB | 21,029 |
| `*.mp4` previews | 0.3 GB | 23 |
| **grand total** | **3,606 GB** | **828,261** |

### 5.1 The new volume

**Format: APFS (case-INsensitive), not encrypted, GUID Partition Map. Name: `MINISCOPE`.**

Not exFAT: the Minian output is ~769k small files and the new output adds more; exFAT's
large cluster sizes, absent journaling and poor directory performance at that file count
are a bad match, and exFAT only buys Windows interoperability that nothing here needs.

**Case-insensitive, revising an earlier recommendation of case-sensitive.** The evidence
says case-sensitivity buys nothing here and carries a small tail risk:

- a scan of all **847,435** entries across both drives found **zero** names differing only
  by case, so there is nothing for case-sensitivity to disambiguate;
- every path string in `loader.py` — **17/17 mouse directories and 117/117
  day/session pairs** — matches the on-disk name with exact case, so case-insensitivity
  cannot mask a bug that case-sensitivity would have caught;
- case-insensitive is the macOS default, so it stays compatible with software that
  refuses case-sensitive volumes (Dropbox among them — and a `.dropbox.device` file on
  drive 1a shows Dropbox has touched this data before);
- the material to be merged in from the `2-MINISCOPE` drive is exFAT, hence already
  case-insensitive and incapable of containing case-only duplicates.

**The primary and its backup must match.** Restoring a case-sensitive volume onto a
case-insensitive one is where case-only collisions would actually bite, so
`MINISCOPE-BAK` must be formatted the same way.

**Naming.** `MINISCOPE`, not `3-MINISCOPE`: a number disambiguates members of a set, and
this volume supersedes the set rather than joining it. Backup: `MINISCOPE-BAK`.

> **Footgun:** an APFS volume name travels with a block-level clone. Cloning `MINISCOPE`
> onto the backup drive produces a second volume also called `MINISCOPE`, and macOS then
> mounts one of them at `/Volumes/MINISCOPE 1` — silently, and not necessarily the one a
> hardcoded path meant. Back up with `rsync`, not a block clone; if a clone is ever used,
> rename the copy before both are mounted.

On a 5 TB volume that leaves ~1,390 GB for the new output, which needs ~53 GB.

### 5.1b Directory structure: flatten, and split data from working files

`data/vsekulic/OF_test` is a relic of the McHugh-lab era — "OF test" means *open field
test*, which this fear-conditioning project has never been. VS's call, 2026-09-22: drop
it, namespace by project, and change `caban` to match. Both SSTCa2 and the future FRAM
cohort number their mice `G*`, so a project level is required regardless.

```
/Volumes/MINISCOPE/
├── README.md                what this drive is, where it came from, how to restore
├── SSTCa2/                  this project's lineage, pilots included
│   └── G05-ST637_hM3D/<day>/<session>/
│       ├── Miniscope/
│       │   ├── 0.avi … N.avi         raw, immutable
│       │   ├── timeStamps.csv, metaData.json
│       │   ├── minian_crossreg*/     2021-22 -- IMMUTABLE, every result rests on it
│       │   ├── minian/               2021-22 -- IMMUTABLE
│       │   ├── minian_intermediate/  the 9 survivors (§6.1) -- IMMUTABLE, gate material
│       │   ├── minian_local/         NEW: this plan's CNMF-E output
│       │   └── YrA_recomputed.zarr   NEW: yra_recompute output
│       ├── BehavCam/
│       └── experiment/
├── FRAMCa2/                 G24/G25 (SGFR1), SGFR3_bad: engram label (red) + activity (green)
├── OLT/                     G30/G31 (2025): object location task, SSTCa2-related batch
├── _misc/                   prism tests (G28, G29, 2024_07_04), unlabelled/2024_10_05,
│                            Yijun/, Yinghao/; to sort: plugins, prev-test, testMouse
├── _drive_roots/{1a,1b}/    recycle bins, volume metadata, stray dotfiles
└── _provenance/             consolidate.log, checksum manifests, drive history
```

Everything below `<mouse>/` is unchanged: `<day>/<session>/{Miniscope,BehavCam,experiment}`
is the Miniscope V4 DAQ's own output shape, not a relic, and `discover_sessions` and every
`dpath` in the archived notebooks already speak it.

**Copy verbatim first, sort afterwards.** The consolidation writes each top-level mouse
directory straight into `SSTCa2/`, which keeps `verify` a clean 1:1 checksum. Moving a
directory to `FRAM/` or `_misc/` afterwards is instantaneous on the same APFS volume, so
nothing is locked in. The one open question — whether G22/G23 (and G24–G27, on a third
drive) belong under `SSTCa2` or a cohort of their own — therefore does not block the copy.

### 5.1c `caban` change: one path module, two roots

Today `POSIX_DATA_ROOT` conflates two things that want opposite storage:

| under `~/data/vsekulic/OF_test/` | size | what it really is |
|---|---|---|
| `npy_files/` (incl. `ds_cache.pkl`, 8.7 GB) | 96 GB | derived cache — wants fast local SSD |
| `plots/` | 2.4 GB | output — wants local |
| `G05-ST637_hM3D/` | 13 GB | partial raw-data copy — belongs on the drive |

Split them:

- **`DATA_ROOT`** — raw video and Minian output. `/Volumes/MINISCOPE/SSTCa2`.
- **`WORK_ROOT`** — `npy_files`, `plots`, `ds_cache.pkl`. Stays local; `~/caban-work`.
  A `mv` of the existing 98 GB, instant on the same volume.

The whole change surface is **six definition sites in four files**:

| file | lines | what |
|---|---|---|
| `caban/utilities.py` | 36, 40–44 | `POSIX_DATA_ROOT`, `MAIN_DRIVE`, `NPY_SAVE_PATH` |
| `caban/loader.py` | 162–165 | plots base |
| `caban/loader.py` | 562–576 | `DRIVE_1a/1b`, `mouse_path_prefix` |
| `caban/main.py` | 70–74 | `PLOTS_DIR` |
| `caban/main.py` | 165–180 | `DRIVE_1a/1b`, `path_prefix` — **duplicates loader.py's** |
| `caban/debug_check_S_A.py` | 2 | one-off debug script |

The hundreds of other `PLOTS_DIR=` occurrences are keyword-argument passing and need no
change. Note the duplication at `main.py:165-180`: the drive mapping is defined twice,
independently, which is the copy-paste `CLAUDE.md` forbids. Collapsing it is part of this
work, not a side quest.

Proposed `caban/paths.py`, the single source of truth, resolving in order and **hard-failing
with the candidates listed** if nothing is found rather than yielding a broken path:

```
CABAN_DATA_ROOT env var  ->  /Volumes/MINISCOPE/SSTCa2  ->  legacy ~/data/vsekulic/OF_test
CABAN_WORK_ROOT env var  ->  ~/caban-work               ->  legacy ~/data/vsekulic/OF_test
```

The legacy fallbacks mean nothing breaks before the drive exists, and they can be deleted
once it does. `mouse_data_dir(mouse)` replaces both copies of `mouse_path_prefix`, and the
Windows drive-letter branches go away entirely — no Windows host is in play any more.

Sequencing: this lands **after** the copy and **before** the pipeline work, and it is
testable the moment the drive is mounted. It is also what finally closes
`yra_recompute_plan.md` §8's "loader.py is Windows-only" item.

### 5.1d Requirement: analyses must run with `MINISCOPE` unplugged

VS, 2026-09-22: once `YrA`/`C`/`S` are regenerated, every `.npy` needed for analysis must
live under `~` so the existing workflow runs with the external drive detached. This is a
hard requirement on the `DATA_ROOT` / `WORK_ROOT` split, not a nice-to-have, and it is
satisfiable — the offline layer is far smaller than `npy_files`' 96 GB suggests.

Measured composition of `npy_files/`, 2026-09-22:

| layer | current size | regenerable? |
|---|---|---|
| `CS_matrices/` (C and S per session) | 13.0 GB | no — **the core** |
| `YrA/` | 2.3 GB | no — **the core** |
| `A_matrix/` (stored **sparse**, `*_A_sparse.pkl`) | 0.09 GB | no — **the core** |
| `stash/` | 18 GB | yes |
| `*-S_shuffled-*.npz` (up to 1.1 GB each) | ~25 GB | yes — shuffled nulls |
| `*-PlaceFields.npz`, `Population_PCA/`, `PV_dist/`, … | ~20 GB | yes — derived |
| `ds_cache.pkl` | 8.7 GB | yes — rebuilt by `load_all_mice` |

**The irreducible core is ~15.4 GB for the 117 sessions currently analysed**, because `A`
is stored sparse and the bulk of `npy_files` is shuffles and derived products.

Scaling the core to all 657 sessions — ~6.7 M frames against the current ~2.0 M, plus a
`YrA` for every session where today only `TFC_cond` has one:

- **~70–100 GB in float64, ~35–50 GB in float32**, plus a proportionally larger
  `ds_cache.pkl` (~30 GB).

Against ~90 GB free locally today, that fits **only** if the regenerable layers are
treated as regenerable — pruned when space is needed and rebuilt on demand — or if the
core is written float32. Both are reasonable; neither is automatic. Concretely:

1. `WORK_ROOT` must distinguish **core** (`CS_matrices`, `YrA`, `A_matrix`) from
   **derived** (`stash`, shuffles, place fields, PCA), so the derived layer can be cleared
   without losing the ability to work offline.
2. **The core is written float32. Settled 2026-09-22, measured not assumed.**

   Round-tripping real cached arrays through float32 (three mice x C, S and YrA):

   | array | zeros | max relative error | smallest non-zero |
   |---|---|---|---|
   | `C` | 47–54 % | **5.96e-08** | 3.2e-07 |
   | `S` | 97–98 % | **5.94e-08** | 6.2e-07 |
   | `YrA` | ~0 % | **5.96e-08** | 3.5e-03 |

   The error is 5.96e-08 across every array — exactly half a float32 ulp, i.e. the
   theoretical floor, with no pathology anywhere. Nothing approaches underflow: the
   smallest non-zero magnitude is 3.2e-07 against float32's ~1.2e-38 smallest normal, and
   `S`'s 98 % zeros are preserved exactly.

   For scale, two runs of the *same* Minian pipeline already differ by ~1e-5 relative
   (`yra_recompute_plan.md` §12.4). **float32 storage error is ~170x smaller than the
   nondeterminism already in the data**, and far below anything an analysis resolves.
   No code in `caban` branches on the dtype of `C`/`S`/`YrA` — the `float64` occurrences
   are group-fraction concatenations and `astype(float).astype(int)` on mapping ids.

   Caveat carried into the implementation: **store float32, compute float64.** The cache
   dtype must not dictate the arithmetic dtype; reductions over ~26k frames should upcast
   on load, which costs nothing and removes the only way this could bite.

   Saving: the core drops from ~15.4 GB to ~7.7 GB today, and from ~70–100 GB to
   ~35–50 GB at full scale.
3. Caching should be **opt-in per session**, not automatic for all 657: the HC and CNO
   sessions are wanted for specific comparisons, not for every analysis.

Incidental confirmation while measuring: the `YrA/` directories for `Test_A`, `Test_B`,
`Test_A_1wk` and `Test_B_1wk` are **0 bytes** — §1 of `yra_recompute_plan.md` said recall
sessions have essentially no YrA, and the cache layer shows it directly.

### 5.2 Scratch does not go on this drive

**Superseded 2026-09-26 by VS's decision: the notebooks write where they always did — into
the session's own `Miniscope/` folder on MINISCOPE** (`intpath = dpath/minian_intermediate`,
outputs beside the videos). A session that already has `minian/` or `minian_intermediate/` gets
them renamed `minian-ORIG/` / `minian_intermediate-ORIG/` first, by
`python -m scripts.set_aside_minian_output <Miniscope dir> --reason ...`
(`caban.session_queue.set_aside_minian_output`), which refuses if anything is already set
aside — a second rename would push the first re-run into `-ORIG` and lose the original — and
records itself in `minian_set_aside.json`. The scanner's `plain_minian_dir`,
`intermediate_dir` and `saved_movie_path` always mean the *original* output and follow it
into `-ORIG`, so neither the YrA recompute's movie check nor anything else can take a re-run
for 2021 output. Cost: CNMF-E's random access now hits the USB disk, not an SSD — slower, not
less correct. The analysis below is kept for the record.

`intpath` must be on fast local storage, not the USB volume: CNMF-E hammers it with
random reads, and 79 GB free on the internal SSD is enough for one session at a time if
intermediates are float32 and deleted between sessions. A TFC_cond session at 76 GB for
both chunkings is right at the edge — the queue should run smallest-first within a
priority band, and the largest sessions may need a dedicated scratch SSD.

### 5.3 Consolidation — the new drive becomes the ONLY copy

VS intends to wipe and repurpose both source drives afterwards (2026-09-22). That
changes what this copy is: not a consolidation with a backup retained, but a migration
whose destination is the sole surviving copy of data that cannot be re-collected.

Consequences, all reflected in `scripts/consolidate_miniscope.sh`:

- **Copy everything, exclude nothing** but `.DS_Store` (Finder window state, created by
  browsing the drives on this Mac). `minian_intermediate/` now comes across too — see
  §6.1, it turns out to be the most valuable thing on the drives after the raw video.
- **The Windows recycle bins come across.** 1a's holds **36 real deleted files, 451 MB**,
  including a 370 MB `.avi` and four smaller ones. Whether they matter is unknown; they
  are unrecoverable once the drive is wiped. They land under
  `_original_drive_roots/1a/` rather than at the destination root, since both drives
  carry the same names.
- **A free-space precheck** refuses to start below 3,700 GB free.
- **`verify` (`rsync -n -c`, every byte on both sides) is SKIPPED by decision**
  (VS, 2026-09-22): the copy is trusted and the hours are not worth spending. The stage
  remains in the script. Before any drive is *wiped* this is the check that should run —
  and short of it, re-running the `data` stage is a cheap size+mtime pass that catches
  missing or truncated files without reading every byte.

**Done 2026-09-23.** Stage 1 (`data`) ran 2026-09-22 14:05 → 09-23 05:36 (15.5 h): **828,261
files, 3,606.4 GB**, exactly the inventory below, no rsync errors. Stage 2 (`extras`) run by hand
with the script's own rsync line, because the 3,700 GB free-space precheck cannot pass once
stage 1 has landed: 194 files, 479 MB, into `_drive_roots/{1a,1b}/` (the layout's
`_drive_roots/`, not `_original_drive_roots/`). A size+mtime dry-run pass over all 29 mouse
directories and both drive roots then listed **zero differences**.

**Hub incident, 2026-09-23, and the verify it forced.** Stage 1 ran with MINISCOPE (a
bus-powered WD My Passport) on a USB hub shared with three other bus-powered drives, while
Spotlight indexed the new volume throughout. On starting the `drive4` stage the drive clicked,
a 36 KB crossreg file copied the day before failed to read (`Illegal byte sequence`), and
freshly appended log lines read back as nulls/foreign blocks — writes landing wrong while
size and mtime stayed right, which a size+mtime pass cannot see. All jobs were stopped. On a
dedicated port and its own cable the drive went silent and the same file read back
byte-identical to 1b: **cause taken to be insufficient power, not the disk**. Spotlight is now
off for MINISCOPE (`mdutil -i off` plus System Settings -> Spotlight -> Privacy, which is what
persists across remounts). Rules carried forward: one bus-powered drive per port, never a
shared hub; disable Spotlight on any data volume before copying onto it.

Checksum verification (`rsync -n -c`), 2026-09-24, each source alone on its own port — **zero
differences anywhere**:

| source | full, every byte | sampled |
|---|---|---|
| 1a | G01–G05 (~600 GB), `plugins`, `prev-test`, `testMouse`, `baseplating`, drive root | G06–G11: every non-`.avi` file (~468k, 150 GB) + 2 % of `.avi` (86) |
| 1b | `baseplating`, drive root | CA1-Cre1, G12–G23: every non-`.avi` file (~310k, 123 GB) + 2 % of `.avi` |

The sample (seed `20260924`) was a time decision: the full pass ran at ~50 MB/s, bound by
macOS's user-space NTFS driver, and would have taken ~20 h for both drives. Logs and the exact
file lists are in `MINISCOPE/_provenance/`. Before either source is wiped, the unsampled
`.avi` (98 % of G06–G23's video) remain the one unverified layer.

The `drive4` copy (4-MINISCOPE; G24–G31) was stopped by the incident, then run on dedicated
ports 2026-09-24/25 (paused once mid-G27 and resumed): done 2026-09-25 22:19, no errors, sorted
by project on the way in (layout §5.1b). G22/G23 were skipped as identical to 1b's copies. G26/G27
are now on two drives. Verified 2026-09-25/26, **zero differences**: G26/G27 by full checksum,
likewise every small directory and the drive root; G24/G25/G30/G31 by every non-`.avi` file
plus 2 % of `.avi` (seed `20260924`). Log: `MINISCOPE/_provenance/verify_4.txt`.

| | size | files |
|---|---|---|
| `data/` (both drives, everything) | 3,606 GB | 828,261 |
| `$RECYCLE.BIN` 1a / 1b | 0.451 GB / ~0 | 163 / 22 |
| `System Volume Information` (both) | 0.028 GB | 8 |
| **total to copy** | **~3,607 GB** | **~828,450** |

On a 5 TB volume that leaves ~1,390 GB for the new output, which needs ~53 GB. Expect
**6–12 h** for the copy and a comparable time for `verify` — two USB drives on one bus,
220 MB average file size, bandwidth-bound.

`baseplating` is the only directory name on both drives (17 vs 21 files); it is copied to
`baseplating-from-1a/` and `baseplating-from-1b/` rather than merged, so a same-named
file cannot silently win.

### 5.4 The single-copy window

Between "sources wiped" and "a second copy exists", every session ever recorded for this
project lives on one USB drive. A drive failure there ends the project. This is a real
exposure and it should be closed deliberately, not drifted through:

- the two 1.8 TB source drives **cannot** together back up 3.6 TB with any margin, so
  repurposing one as the backup does not work;
- the honest options are to keep the sources intact until a second ≥5 TB drive exists, or
  to accept a known window;
- **at minimum, run `verify` before wiping** — it is skipped for now by decision, but a
  wipe is the one operation that makes a silent copy error permanent — and wipe one drive
  at a time so the second is still a partial fallback.

Recommend: keep the sources until a second large drive is available. Nothing in this plan
needs them wiped — the scratch SSD (§5.2) is what unblocks the pipeline, and VS already
has that.

## 6. The gate: re-derive sessions that are already done

There is no ground truth for the 477 — nothing to compare a new `C`/`S` against. The
processed sessions are the calibration set, and the gate is the direct analogue of the
G10 gate in the YrA plan: **run the automated pipeline on already-processed sessions into
a separate directory, and compare.**

### 6.1 Four sessions make this a stage-by-stage gate, not an end-to-end one

Scanning the backup drives turned up **9 surviving `minian_intermediate/` directories**,
and four of them — G06 `2021_10_18-TFC_cond` `09_52_24-HC1`, `10_34_09-CNO1`,
`10_49_52-CNO2`, `11_26_52-HC2` — hold **27 arrays each**: the complete internal state of
a production CNMF-E run.

```
varr, varr_ref            preprocessing
Y_fm_chk, Y_hw_chk        after motion correction
max_res, sn_spatial       noise estimation
A_init, C_init            initialization
A_mrg, C_mrg, C_mrg_chk, sig_mrg   merge
A_new, C_new, S_new, b0_new, c0_new, g   spatial/temporal updates
A, C, S, YrA, b, b0, c0, f               final
```

This is worth more than four extra sessions. It converts the gate from "is the final
answer distributionally similar?" to **"at which stage does the automated pipeline first
diverge from the 2021 one?"** — a far sharper instrument, and one that localises a fault
instead of merely detecting it. Check them in pipeline order and stop at the first
mismatch:

1. `varr_ref` — preprocessing (denoise, tophat). Should be **exact**; it is integer
   arithmetic on uint8 through cv2.
2. `Y_fm_chk` — motion correction. **This is the one that needs care**: these sessions
   have `motion.zarr`, so it can be checked both ways — motion *applied* (should be
   exact, and is the same code path the YrA recompute already validated) and motion
   *re-estimated* (what the 477 will actually need, and never yet exercised).
3. `sn_spatial`, `max_res` — noise estimation. Sensitive to FFT backend, so a plausible
   place for `pyfftw`-vs-numpy drift to show.
4. `A_init` / `C_init` — seed detection and initialization. First stage where cell
   identity is created, and where `pymetis` enters via seed merging.
5. `A_mrg` / `C_mrg` → `A_new` / `C_new` / `S_new` → final `A`/`C`/`S`.

Stages 1–3 should be reproducible to floating-point tolerance. From stage 4 on,
divergence is expected and the question becomes how much.

### 6.2 Acceptance must be distributional from stage 4 on

CNMF-E is not deterministic and we already know by how much: the same session re-run gave
`C` differing by ~1e-5 relative, and unit sets differing by 1–4 marginal cells in 8 of 17
mice (`yra_recompute_plan.md` §2.2, §12.4). Identity is the wrong bar and would fail a
correct implementation.

Proposed criteria for the final comparison, to be calibrated on the four G06 sessions
before being fixed:
- **cell count** within a few percent of the original;
- **footprint centroids**: ≥95 % of original units have a new unit whose centroid is
  within ~2 px, by mutual nearest neighbour;
- **matched-cell traces**: median `corr(C_old, C_new) > 0.9` over matched pairs;
- **unmatched cells** in either direction reported with their footprints, not summarised
  away.

Beyond the four, run the end-to-end comparison on at least one session per type and per
drive — a TFC_cond, an LT1, a Test_B — before any of the 477 is queued.

## 7. Cross-registration

Out of scope for the first pass and deliberately so: generate `A`/`C`/`S`/`YrA` for the
new sessions first, then decide how they join the existing mappings. §3.4 says a new
grouping cannot disturb the old ones, but that must be re-verified on real output before
anything is registered, and the question of *which* sessions belong in a grouping is
scientific, not mechanical.

The 50 `plain minian/` sessions are the natural pilot: they need no CNMF-E, so they
isolate the cross-registration question from everything else.

## 8. Execution order

Priority set by VS, 2026-09-22 — scientific value first, cheapest-to-verify within band:

1. **Gate** (§6) — re-derive a few already-processed sessions. Nothing proceeds until it
   passes.
2. **The 2 borked TFC sessions** (`TFC_test_B-borked`, `TFC_test_B_1wk-borked`) — TFC is
   the top priority and these are the only TFC sessions missing.
3. **LT1 / LT2** — 89 sessions, 1,256 `.avi`. Highest value per hour: place-cell work, and
   LT1 already has 30 processed siblings to sanity-check against.
4. **CNO** — 22 sessions, 163 `.avi`. The per-cell before/after-CNO question.
5. **HC** — 328 sessions, 2,485 `.avi`. Largest by count, cheapest individually, last.
6. **The 50 `plain minian/` sessions** — `YrA` recompute plus cross-registration, no
   CNMF-E. Can run in parallel with any of the above since it uses a different pipeline.
7. The ~36 aborted stubs — only after being eyeballed (§2).

At the recorded 0.3–2.1 h per session (median ~1.3 h on the CBP server), the 477 are
roughly **1–2 weeks of continuous compute**, and steps 2–4 are ~2–3 days of it.

## 9. Shape of the code

**Done 2026-09-22: `caban/session_queue.py` exists.** The queue was lifted out of
`yra_recompute.py` and split mechanism-from-policy:

- **mechanism** (`session_queue`): `scan_sessions` reports facts about every session it
  finds and *selects nothing*; `resolve_minian_dir` hard-fails on ambiguity;
  `next_pending` / `queue_frame` / `run_queue` handle the cursor, the view and the serial
  sweep; sidecar read/write is parameterised by filename.
- **policy** (each pipeline): which candidates are its work, and what its queue view
  reports. `yra_recompute.discover_sessions` now filters `scan_sessions` to "has `.avi`
  **and** a complete `minian_crossreg*`"; the CNMF-E pipeline will filter to the
  complement, from the same scan.

`scan_sessions` already surfaces everything the CNMF-E side needs — `plain_minian_dir`,
`intermediate_dir`, `saved_movie_path` — so that pipeline needs no new discovery code.

Verified behaviour-preserving two ways: a synthetic tree exercising the edge cases (the
two-crossreg ambiguity, `YrA` at either level, sidecar completeness, cursor advance), and
a real scan returning **the same 130 items and the same 2 done, with identical numbers**.
The YrA notebook's API is unchanged, which matters because the §12 gate result stands
on it.

Then `notebooks/run_minian_pipeline.ipynb`, same four-cell shape (setup / discover /
preview / run), with the pipeline itself in `caban/minian_pipeline.py`:

- the frozen §3.2 parameters as a module constant, with provenance;
- the same sidecar idempotency (§11.2 of the YrA plan), so 477 sessions resume after a
  crash;
- scratch created and **deleted** per session, with a hard failure if free space is
  insufficient rather than a half-written run;
- a per-session report: cell count, `corr(C, YrA)`, footprint area distribution, fraction
  of seeds surviving each refine stage — the numbers that say "this session processed
  sensibly" without ground truth.

**Not** an extension of `recompute_yra.ipynb`: that notebook's safety argument is "we
touch nothing but YrA", and a generative pipeline in the same driver would wreck it.

## 10. Risks

1. **`pymetis` on arm64** (§4.2). Performance only: it schedules the correlation
   computation and does not change results, so a stand-in partition is an acceptable
   fallback. Still the first thing to test after the env is built.
2. **Scratch space** (§4.1, §5.2). 79 GB free locally against 76 GB for a TFC_cond
   session. Must hard-fail on insufficient space, never half-write.
3. **The gate fails** (§6). Means the automated pipeline is not the 2021 pipeline. Most
   likely causes, in order: a dependency version that changes a numerical path, a
   parameter not captured by §3.2, or `estimate_motion` behaving differently — note the
   new sessions need motion *estimated*, not applied, which the YrA recompute never
   exercised.
4. **Nondeterminism between runs** is expected and quantified (§6), but it means the new
   sessions' cell identities are a fresh draw. Nothing longitudinal can assume they
   correspond to anything until §7 is done.
5. **The single-copy window** (§5.4). Ranked here rather than first only because it is
   avoidable by not wiping. If the sources are wiped before a second copy exists, this is
   the largest risk in either plan by a wide margin, and unlike the others it is not
   recoverable by re-running anything.
6. **1–2 weeks of compute on a laptop.** Thermal throttling, sleep, and USB dropouts are
   real. The sidecar idempotency is what makes this survivable; test resumption
   deliberately rather than discovering it during run 300.
