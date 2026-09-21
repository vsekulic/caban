# Plan: migrate off the RIKEN CBP server before access ends

Status: **scoped 2026-09-21, urgent — access ends within days.** Written for VS on `osgiliath`
(MacBook Pro, **arm64**, macOS 26.6).
Share: `//vsekulic@cbp-db.bnf.brain.riken.jp/vsekulic` → `/Volumes/vsekulic`.
Goal: be able to run the Minian framework as it ran on the server, and keep anything else on that
share that is not reproducible from elsewhere.

## 0. Do this first — there is unbacked-up raw data

`/Volumes/vsekulic/data/vsekulic/OF_test` holds **G05–G27**. The backup drives hold:

| | mice |
|---|---|
| `1a-MINISCOPE-BAK` | G01–G11 |
| `1b-MINISCOPE-BAK` | G12–G23 |
| **server only** | **G24-ST861-hM3D-SGFR1, G25-ST862-hM3D-SGFR1, G26-ST894-hM4D-DIOGC, G27-ST895-hM4D-DIOGC** |

**G24–G27 exist on neither backup drive.** They are outside the current paper's cohort (G05–G21),
but they are raw acquisition data that disappears with the mount. Decide explicitly whether to keep
them; if yes, copy them first, because they are the only irreplaceable thing on the share. Everything
else below is code and can be re-derived or re-downloaded.

Note the backup drives are mounted **read-only** (`ntfs, read-only`), so they cannot receive the copy
— it has to go to local disk or another external volume. Local free space is ~103 GB.

## 1. Tier 1 — must copy, small

| path | size | why |
|---|---|---|
| `minian_vsekulic/minian/` | 736 K | the actual Minian source, VS's fork |
| `minian_vsekulic/*.ipynb` | 295 M | `pipeline-WORKING-TFC_cond-{1..4}`, `cross-registration-WORKING*` — the run record |
| `minian_vsekulic/prev/*.ipynb` | (of 1.1 G) | **`pipeline-WORKING-{Test_A,Test_B,Test_B_1wk,LT1,LT2,HC}.ipynb`** — the per-session-type notebooks, i.e. the only record of which sessions were processed with what parameters, including per-session `del_frames` |
| `minian_vsekulic/{environment.yml,requirements/,pyproject.toml,setup.py}` | 40 K | upstream dependency spec, portable |
| `environment_minian_vsekulic.yml`, `minian_spec.txt`, `minian_vsekulic_export.yaml`, `conda_envs/` | ~80 K | the *exact* server env, for reference |
| `bin/` | 60 K | `activate_minian.sh`, `jupyter_minian.sh`, `minian2mat.sh`, … |
| `.bashrc`, `.bash_profile`, `.condarc`, `.gitconfig`, `.bash_history` | ~30 K | `.bash_history` is a record of how things were actually invoked |
| `.jupyter/`, `.ssh/` | small | config; review `.ssh` before copying |
| top-level `mappings_crossreg_crossreg{1,2}.csv`, `params-export.json` | 100 K | crossreg outputs / parameter export |

Notebook sizes are almost entirely stored outputs (`pipeline_clean.ipynb` alone is 36 M). Copy them
**with** outputs — the outputs are the record. Strip only for a git-tracked copy.

## 2. Tier 2 — probably worth it

| path | size | note |
|---|---|---|
| `abnormal_cell_filter_1/` (`roi`, `traces`) | 95 M | QC diagnostic plots; regenerable but cheap to keep |
| `code/SSTCa2` | 16 K | trivial |
| `code/minian2mat` | 24 K | trivial |
| `claude/`, `.claude/`, `.claude.json` | small | prior Claude Code config/history on the server |
| `minian_vsekulic/demo_data` | 10 M | lets the Minian demo run end-to-end as a smoke test |

## 3. Skip

| path | size | why |
|---|---|---|
| `miniforge3/` | large | a **linux-64** conda install; useless on arm64 — rebuild instead (§4) |
| `minian_git/` | 1.7 G | upstream clone, re-clonable from GitHub (check `git log` for local commits first) |
| `minian_vsekulic/demo_movies` | 689 M | upstream demo videos, re-downloadable |
| `dask-worker-space/` | 32 K | scratch |
| `data/vsekulic/OF_test` G05–G23 | huge | already on the two BAK drives — **verified per-mouse, but spot-check before relying on it** |
| `code/caban` | 144 M | server copy is at `cd1aa5a`; the local repo is ahead. Nothing to rescue — confirm with `git log` before deleting |
| `code/bbnp`, `bbnp/`, `MATLAB/` | 2.2 G / 13 M / ? | separate projects; decide independently of this migration |

## 4. The environment — the genuinely hard part

`environment_minian_vsekulic.yml` is **linux-64 with full build strings and Python 3.8**
(`py38h01eb140_4`, `_libgcc_mutex`, …). It **cannot** be recreated on osx-arm64. Upstream
`environment.yml` is more portable but pins 2021-era versions — `python=3.8`, `numpy=1.20.2`,
`dask=2021.2.0`, `xarray=0.16.2`, `numba=0.52.0`, `opencv=4.2.0`, `holoviews=1.12.7`, `bokeh=1.4.0`,
`panel=0.8.0` — most of which have **no osx-arm64 builds**.

Three routes, in order of how faithfully they reproduce the server:

**A. Docker, `linux/amd64`** — the only true "AS-IS". Recreate `environment_minian_vsekulic.yml`
verbatim inside a linux-64 container, mount the data, run Jupyter on a published port. Emulated on
arm64, so slow, but nothing about the numerics changes. Best if the interactive pipeline (CNMF-E,
seed exploration, crossreg GUI) must be re-run.

**B. `CONDA_SUBDIR=osx-64` + Rosetta 2** — conda-forge does have osx-64 py38 builds for most of
these pins, so upstream `environment.yml` has a real chance of solving. Native-ish speed, no
container. Try this first; fall back to A.

**C. Native osx-arm64 with modern versions** — requires porting Minian across ~4 years of
xarray/dask API drift. Not "as-is"; do not attempt for this migration.

**Important scoping relief:** the YrA recompute
(`plans/yra_recompute_plan.md`) does **not** need the interactive stack. Its call path —
`utilities.{load_videos,open_minian,save_minian}`, `preprocessing.{denoise,remove_background}`,
`motion_correction.apply_transform`, `cnmf.compute_trace` — imports none of
holoviews / bokeh / panel / datashader (those live in `visualization.py`, which `cnmf.py` does not
import). It needs cv2, dask, distributed, xarray, zarr, numpy, pandas, scipy, scikit-image,
SimpleITK, sparse, numba, cvxpy, networkx, pyfftw, pymetis, medpy, rechunker, tifffile,
ffmpeg-python, natsort, scikit-learn, statsmodels. That is a far easier solve than the full
environment, and it is the only part needed for the recompute. **Build the slim env first (route B);
treat the full interactive env (route A) as a separate, lower-urgency task.**

## 5. Suggested order of work

1. **Decide on G24–G27** and copy them if keeping — they are the only irreplaceable data (§0).
2. Copy Tier 1 (§1). Small and fast; do it before anything can go wrong with the mount.
3. Copy Tier 2 (§2).
4. `git log` the server's `minian_git` and `code/caban` to confirm no local commits are stranded;
   then skip both.
5. Build the **slim** recompute env (§4 route B) and verify `import minian.cnmf` succeeds.
6. Only then tackle the full interactive env (§4 route A) if you still need it.

Use `rsync -avP --no-perms --no-owner --no-group` for the copies (SMB + POSIX permissions do not mix
well), and re-run it to resume — it is restartable, which matters on a flaky share.

## 6. Verification before the mount goes away

- `rsync -n -avc` a second pass over Tier 1/2 and confirm zero differences.
- Confirm `minian/` imports on the new env.
- Confirm every `pipeline-WORKING-*.ipynb` opens and its parameter cell is readable — these are the
  only record of `del_frames` and per-session parameters that the recompute plan depends on.
- Spot-check a few G05–G23 sessions against the BAK drives so the §3 "already backed up" claim is
  not taken on trust.
