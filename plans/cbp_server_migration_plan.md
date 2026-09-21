# Plan: migrate off the RIKEN CBP server before access ends

Status: **mirror complete and verified 2026-09-22; two items earmarked (§5b).** Access ends within days. Written for VS on
`osgiliath` (MacBook Pro, **arm64**, macOS 26.6).
Mirror destination: **`~/cbp-db/vsekulic/`** — host/share structure preserved, so `~/cbp-ndb/` can
hold that server's shares later without collision. Driven by `~/cbp-db/mirror_cbp_db.sh`
(restartable; log at `~/cbp-db/mirror.log`).
Share: `//vsekulic@cbp-db.bnf.brain.riken.jp/vsekulic` → `/Volumes/vsekulic`.
Goal: be able to run the Minian framework as it ran on the server, and keep anything else on that
share that is not reproducible from elsewhere.

## 0. Raw data — resolved

`/Volumes/vsekulic/data/vsekulic/OF_test` holds G05–G27, while the backup drives cover G01–G11
(`1a`) and G12–G23 (`1b`). **G24-ST861, G25-ST862, G26-ST894 and G27-ST895 are on neither** — but VS
confirms they live on a third, currently unmounted drive. **No action needed; no raw data is at
risk.** Recorded because the gap is invisible from the two mounted drives alone.

Note the BAK drives are mounted **read-only** (`ntfs, read-only`), so they cannot receive any copy.

## 0b. The real at-risk item: the Minian fork's uncommitted work

`minian_vsekulic` **is** a git repo, remote `git@github.com:vsekulic/minian.git`. But:

- **1 commit is unpushed** — `88e76c47 Added framerate arguments to some visualization functions.`
- **98 paths are dirty** — 71 modified, 27 untracked, including `minian/install.py`, the test suite,
  the docs extensions, and the `pipeline-WORKING-*` notebooks that carry the per-session parameters.

**So GitHub is not a backup of this.** The working tree *and* `.git` must both be copied; pushing to
GitHub afterwards is worth doing but does not substitute for the copy, because the untracked files
and notebook outputs would not go with it.

By contrast `minian_git` is **not a git repo at all** — just a pristine upstream extract (it carries
a `qless` directory that the fork lacks). Low value, copied anyway since it is cheap.

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
| ~~`minian_git/`~~ | 1.7 G | **now copied** — it is not a git repo, so "re-clonable" was wrong; cheap enough to keep |
| `minian_vsekulic/demo_movies` | 689 M | upstream demo videos, re-downloadable |
| `dask-worker-space/` | 32 K | scratch |
| `data/vsekulic/OF_test` G05–G23 | huge | already on the two BAK drives — **verified per-mouse, but spot-check before relying on it** |
| `code/caban` | 144 M | server copy is at `cd1aa5a`; the local repo is well ahead. **Confirmed nothing to rescue** |
| `code/bbnp`, `bbnp/`, `MATLAB/` | 2.2 G / 13 M / ? | separate projects; decide independently of this migration |

## 4. The environment — DEFERRED

**VS 2026-09-21: hold off on getting Minian running on macOS for now.** Copy first; solve the
environment later. The analysis below stands for when it is picked up.

### The genuinely hard part

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

## 5. Order of work

1. ~~Decide on G24–G27~~ — resolved, they are on a third drive (§0).
2. **Run `~/cbp-db/mirror_cbp_db.sh`** (in progress). Three stages, smallest and most critical first:
   Stage 1 config/scripts/env-specs/notebooks, Stage 2 `minian_vsekulic` including `.git`,
   Stage 3 `minian_git` + `abnormal_cell_filter_1`. Restartable — just re-run it.
3. Push `88e76c47` and commit the dirty tree to the GitHub fork, as a second copy (§0b).
4. Environment work — **deferred** (§4).

`.ssh` is **deliberately excluded** from the script: it holds private keys, and duplicating those
should be a conscious act rather than a side effect of a mirror. Copy it by hand if wanted.

## 5b. Done, and earmarked for later

**Done 2026-09-21/22:**
- Mirror complete at `~/cbp-db/vsekulic/` — **6.5 G**, all stages, no rsync errors. Verified by
  checksum dry-run (only `.git/index` differed, and only because running `git status` on the copy
  refreshed it) and by `git fsck --connectivity-only`, which came back clean.
- `.local/share/fonts` (4.7 M) copied — Liberation Sans/Mono/Serif and TeX Gyre Heros, i.e. two of
  the four families in `caban/utilities.py`'s `FONT_SANS_SERIF` chain. Without them figure text
  falls through to DejaVu Sans and panel metrics change.
- `.local/share/jupyter` copied — carries `kernels/caban/kernel.json`, the registered kernelspec.
- Fork pushed: `a3216ae2` on `origin/vsekulic_v4` (pipeline.ipynb with the YrA export,
  cross-registration.ipynb, minian_vsekulic_cbp-db.yaml). `core.fileMode false` first, since 71 of
  the 98 "dirty" paths were SMB-induced 100644->100755 mode flips with no content change.

**Correction to §0b as first written:** there was **no** unpushed commit. `88e76c47` was already on
`origin/vsekulic_v4`; the earlier claim came from comparing against `origin/master`, which is a
different branch. The genuinely-untracked content was the `*-WORKING*.ipynb` notebooks and `prev/`.

**Earmarked for later (VS, 2026-09-22):**
- **`MATLAB/` (18 G)** — not copied. Worth a look before the mount goes, in case anything in it is
  hand-written rather than just the installation. The only top-level item not confidently disposable.
- **Spot-check the BAK drives** against the server for a few G05-G23 sessions. §3 treats those as
  already backed up on the strength of directory listings alone; that has not been verified by
  content.
- Not copied, judged disposable: `.local/tmp` (519 M, sysadmin-designated scratch per its own
  `README_DENIS.txt`), top-level `demo_movies` (1.1 G, upstream copy already mirrored inside
  `minian_vsekulic`), `code/bbnp` (2.2 G, separate project), `.local/share/{Trash,mc,pki}`.

## 6. Verification before the mount goes away

- `rsync -n -avc` a second pass over Tier 1/2 and confirm zero differences.
- Confirm `minian/` imports on the new env.
- Confirm every `pipeline-WORKING-*.ipynb` opens and its parameter cell is readable — these are the
  only record of `del_frames` and per-session parameters that the recompute plan depends on.
- Spot-check a few G05–G23 sessions against the BAK drives so the §3 "already backed up" claim is
  not taken on trust.
