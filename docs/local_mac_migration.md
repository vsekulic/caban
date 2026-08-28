# Migrating the pipeline from cbp-db to the local Mac

Runbook for moving `run_pipeline.ipynb` work off the Riken server
(`cbp-db.bnf.brain.riken.jp`) and onto the local Mac (`osgiliath`), keeping the
notebook workflow, the GitHub remote, and the Claude Code session history.

Written 2026-08-27, against branch `feat/sp-rates-axis-labels`.

---

## What has to move, and what does not

| Thing | Where it lives on cbp-db | Size | Action |
|---|---|---|---|
| Repo | `/Users/vsekulic/code/caban` | 124 MB | via GitHub |
| Analysis caches | `/Users/vsekulic/data/vsekulic/OF_test/npy_files` | 97 GB | rsync |
| Raw Minian data | `/Users/vsekulic/data/vsekulic/OF_test/G*` | 664 GB | **already on the Mac** |
| conda env `caban` | `~/miniforge3/envs/caban` | — | rebuild from export |
| Claude sessions | `~/.claude/projects/-Users-vsekulic-code-caban` | 154 MB | rsync |

Three properties of the code make this straightforward:

- `load_all_mice(cfg)` loads `ds` from `<NPY_SAVE_PATH>/ds_cache.pkl` and skips
  the raw build entirely (`caban/loader.py`, `load_all_mice`). The big arrays
  pruned from that pickle (`BehaviourSession._cache_pruned_fields` in
  `caban/sessions.py`) are lazily restored from the `Saver` pickles under
  `npy_files/<session>/` — **not** from the Minian zarrs. So the raw data is
  not on the critical path for a cache-driven run.
- Every `Saver` pickled inside `ds_cache.pkl` carries an absolute
  `parent_path`/`save_path`. Because `/Users/vsekulic/data/vsekulic/OF_test`
  is identical on both machines, no path rewriting is needed anywhere.
- Claude Code keys its transcripts by the slugified workspace path
  (`-Users-vsekulic-code-caban`). Same repo path ⇒ same slug ⇒ history copies
  verbatim.

---

## Memory budget (osgiliath has 32 GB)

Measured on cbp-db, 2026-08-27:

- Unpickling `ds_cache.pkl` (8.7 GB on disk): **9.40 GB peak RSS**, 7 s wall.
- The lazily-restored caches add up to **~16 GB** on disk (`CS_matrices` +
  `YrA` + `A_matrix` + `loc_data` + `timestamps` over the five session dirs;
  `TFC_cond` alone is 9.4 GB).

Those restores are `setattr`'d onto the session objects on first access, so
they **stay resident for the life of the kernel** — RSS grows monotonically as
sections run, and the derived `S_mov`/`S_imm`/`C_mov`/`C_imm` variants are full
extra copies of their base arrays. On 250 GB this is invisible; on 32 GB it is
the binding constraint.

Practical guidance:

1. Run sections in groups and **restart the kernel between the heavy ones** —
   `run_epoch_modulation` with `epoch_modulation_hierarchical_cells=True`, the
   decoders, and the PyMC/bambi fits.
2. Between sections, drop restores you no longer need (`del sess.C_full`,
   `del sess.YrA_full`, …). They come back from disk cheaply on next access.
3. Keep SSD free space for swap; Apple Silicon memory compression makes a
   moderate overshoot slow rather than fatal.

If one long-lived kernel turns out to be unworkable, the fallback is a headless
script per section — decide that after measuring a real local run, not before.

---

## 1. Push the current code

On cbp-db, on `feat/sp-rates-axis-labels`:

```bash
git push origin feat/sp-rates-axis-labels
```

Git-ignored files the clone will **not** carry (`.gitignore` ignores
`freeze_data`):

- `freeze_data/` — 76 KB on cbp-db. `caban/sections.py` reads
  `<repo_root>/freeze_data/TFC_miniscope.json` in the freeze/mobility
  verification section. The pre-existing local `caban` directory has a 395 MB
  `freeze_data/`; check whether it already contains `TFC_miniscope.json`
  before copying.
- `.claude/settings.local.json` — local tool permissions, optional.
- `plots/` — regenerated locally.

## 2. Reconcile the old local repo

`/Users/vsekulic/code/caban` on the Mac holds the pre-rename `SSTCa2_*.py`
generation. Diagnose first:

```bash
git -C /Users/vsekulic/code/caban remote -v
git -C /Users/vsekulic/code/caban log --oneline -3
```

**If it is a clone of `git@github.com:vsekulic/caban.git`** (likely):

```bash
cd /Users/vsekulic/code/caban
git stash        # or commit, if anything local is worth keeping
git fetch origin
git checkout feat/sp-rates-axis-labels
git status       # leftover untracked SSTCa2_*.py etc. can be deleted
```

The `SSTCa2_*.py` files vanish as tracked renames; anything remaining is
untracked cruft.

**If it is not a git repo, or points elsewhere:**

```bash
cd /Users/vsekulic/code
mv caban caban-legacy-$(date +%F)
git clone git@github.com:vsekulic/caban.git caban
cd caban && git checkout feat/sp-rates-axis-labels
```

The path must stay exactly `/Users/vsekulic/code/caban` — the Claude history
slug (step 6) depends on it.

Make sure the Mac's SSH key is registered with GitHub so push/pull works both
from the shell and from VS Code's SCM panel. Keep `caban-legacy-*` until a
local run reproduces server output.

## 3. Rebuild the environment

There is no `environment.yml` in the repo, so the env is exported on the server
and rebuilt on the Mac. Every command below is labelled with where it runs.

### 3a. Export — **on cbp-db**

```bash
conda env export -n caban --from-history > ~/caban-env.yml   # portable, use this one
conda env export -n caban            > ~/caban-env-full.yml  # linux-64 pins, reference only
```

`--from-history` lists only the packages that were explicitly requested, so it
solves on any platform. The full export pins linux-64 builds and **will not
solve on the Mac** — keep it only as a record of what versions the caches were
written with.

Two gaps in `--from-history` you must close by hand (verified 2026-08-27):

- It carries **no version pins** except `python=3.11`, so the Mac would
  otherwise take whatever is newest today. Add the pins from the table below
  before creating the env.
- It omits **pip-installed** packages. In this env those are
  `rastermap==1.0`, `qtpy==2.4.3`, `superqt==0.8.1` — installed separately in
  step 3c.

For reference, `~/caban-env.yml` currently contains: `python=3.11`,
`jupyterlab`, `ipykernel`, `ipywidgets`, `numpy`, `pandas`, `matplotlib`,
`scipy`, `scikit-learn`, `statsmodels`, `xarray`, `scikit-image`, `seaborn`,
`opencv`, `umap-learn`, `networkx`, `tqdm`, `pyyaml`, `optuna`, `zarr`,
`natsort`, `pingouin`, `nbstripout`, `nbconvert`, `r-base`, `r-emmeans`,
`r-lme4`, `r-lmertest`, `numba`, `pyqt6`, `pyqtgraph`, `bambi` (which pulls in
`pymc` and `arviz`), on channel `conda-forge`. All are available for
`osx-arm64`.

### 3b. Copy the export to the Mac — **on the Mac**

```bash
scp vsekulic@cbp-db.bnf.brain.riken.jp:caban-env.yml      ~/caban-env.yml
scp vsekulic@cbp-db.bnf.brain.riken.jp:caban-env-full.yml ~/caban-env-full.yml
```

Then edit `~/caban-env.yml` and pin the versions that govern **pickle
compatibility** — the caches were written by:

| package | version on cbp-db |
|---|---|
| python | 3.11.15 |
| numpy | 2.4.5 |
| pandas | 3.0.2 |
| scipy | 1.17.1 |
| statsmodels | 0.14.6 |
| scikit-learn | 1.8.0 |
| xarray | 2026.4.0 |
| zarr | 3.1.6 |
| matplotlib | 3.10.9 |

i.e. the dependency lines become `python=3.11`, `numpy=2.4.5`, `pandas=3.0.2`,
`scipy=1.17.1`, `statsmodels=0.14.6`, `scikit-learn=1.8.0`, `xarray=2026.4.0`,
`zarr=3.1.6`, `matplotlib=3.10.9`, with the rest left unpinned.

`matplotlib` is pinned for a hard reason, not tidiness: matplotlib 3.11 removed
`matplotlib.style.core`, which `arviz_plots` (pulled in by `bambi`/`pymc` via
`arviz`) still uses — see the troubleshooting note at the end of this step.

`Saver.load` routes DataFrames through `pd.read_pickle`, so pandas on the Mac
must be ≥ 3.0 or the cached DataFrames will not unpickle. If conda-forge lacks
an exact `osx-arm64` build for something, drop that pin and match the numpy and
pandas **majors** first — those two decide whether the caches load at all.
`~/caban-env-full.yml` is the reference for any other version you need to
check.

### 3c. Create the env — **on the Mac**

```bash
conda env create -f ~/caban-env.yml          # name: caban comes from the file
conda activate caban
pip install rastermap==1.0 qtpy==2.4.3 superqt==0.8.1   # the pip-only packages
```

Use `mamba env create -f ...` instead if you have mamba — same file, much
faster solve. If a `caban` env already exists on the Mac, either
`conda env remove -n caban` first, or create under a different name with
`conda env create -f ~/caban-env.yml -n caban-local` (and use that name
everywhere below).

Sanity-check the versions that matter:

```bash
python -c "import numpy, pandas, scipy, statsmodels, sklearn, xarray, zarr; \
print(numpy.__version__, pandas.__version__, scipy.__version__, \
statsmodels.__version__, sklearn.__version__, xarray.__version__, zarr.__version__)"
python -c "import pymc, bambi, arviz, rastermap, pingouin, umap, cv2, optuna; print('imports OK')"
```

**If the second command fails with**
`AttributeError: module 'matplotlib.style' has no attribute 'core'`
(raised from `arviz_plots/__init__.py` while importing `pymc` → `arviz`):
the env solved to matplotlib 3.11, which removed `matplotlib.style.core`;
`arviz-plots` 0.8.0 still calls it. Fix by aligning to the server versions —
this also repairs any other pins that were skipped when the env was created:

```bash
conda install -n caban -c conda-forge \
    matplotlib=3.10.9 statsmodels=0.14.6 scikit-learn=1.8.0 xarray=2026.4.0
```

If the solver refuses that combination, `matplotlib=3.10.9` alone clears the
import error — the other three are for numerical parity with cbp-db, and drift
there is a reason to take the parity check (verification step 5) seriously
rather than a blocker. Re-run both check commands afterwards.

### 3d. Register the Jupyter kernel — **on the Mac**

```bash
conda activate caban
python -m ipykernel install --user --name caban --display-name "caban"
```

This writes `~/Library/Jupyter/kernels/caban/` on the Mac. It is a per-machine
registration, so the one done on cbp-db does not carry over — run it locally
even though the same command appears in [JUPYTER_SETUP.md](JUPYTER_SETUP.md).
Afterwards `jupyter kernelspec list` should show `caban`, and VS Code / Jupyter
Lab will offer it as a kernel choice for `run_pipeline.ipynb`.

Fonts: `FONT_SANS_SERIF` in `caban/utilities.py` puts Helvetica first and macOS
ships it, so nothing to install. (TeX Gyre Heros, the free Helvetica clone
installed on cbp-db, is only the fallback for machines without it.)

## 4. rsync the analysis caches

From the Mac:

```bash
rsync -aHh --partial --inplace --info=progress2 \
  vsekulic@cbp-db.bnf.brain.riken.jp:/Users/vsekulic/data/vsekulic/OF_test/npy_files/ \
  /Users/vsekulic/data/vsekulic/OF_test/npy_files/
```

Resumable — re-run the identical command after any drop. Composition, for
pacing:

| Path | Size | Notes |
|---|---|---|
| `ds_cache.pkl` | 8.7 GB | the one must-have |
| `TFC_cond/` | 41 GB | `CS_matrices` 7.1 G + `YrA` 2.4 G are the parts the pickle restores from; the rest is `*-S_shuffled-*.npz` and `stash/` |
| `Test_B/` | 16 GB | |
| `Test_B_1wk/` | 18 GB | |
| `Population_PCA/` | 6.7 GB | |
| `Test_A/`, `Test_A_1wk/` | ~4 GB | |

Adding `--exclude 'stash/' --exclude 'bak/'` drops ~13.6 GB of pure backup
copies with no effect on any analysis. The `*-S_shuffled-*.npz` files
(~50 GB total) are place-cell shuffle tensors; normal runs read the small
`*-sig_responses-*.npz` instead, so they can be dropped too if bandwidth is
tight — at the cost of an expensive recompute if any code path re-derives
place-cell significance.

The raw `OF_test/G*` trees are deliberately **not** in this transfer.

Verify afterwards:

```bash
shasum -a 256 /Users/vsekulic/data/vsekulic/OF_test/npy_files/ds_cache.pkl   # Mac
ssh vsekulic@cbp-db.bnf.brain.riken.jp \
  sha256sum /Users/vsekulic/data/vsekulic/OF_test/npy_files/ds_cache.pkl
```

or a final `rsync -n -c` over the whole tree.

## 5. Host detection in the code

Already handled on `feat/sp-rates-axis-labels`: `caban/utilities.py` now
resolves `NPY_SAVE_PATH` from a **path test** (`/Users/vsekulic/data/vsekulic/OF_test`
exists) rather than the `cbp-db` hostname, with the hostname check kept as a
fallback and the Windows `D:` branch untouched. That covers osgiliath
automatically because the data root is mirrored at the same absolute path.

Everything else follows: the session save paths derive from `NPY_SAVE_PATH`
(`caban/loader.py`), `cfg.PLOTS_DIR` defaults to a repo-relative
`plots/<timestamp>`, and the legacy `caban/main.py` branch keys off
`MAIN_DRIVE == ''`.

Not changed, and not needed for this migration: the Windows-style raw-data
prefixes in `caban/loader.py` (`mouse_path_prefix`). They are only used by a
fresh `_build_dataset` (`use_cache=False`). Rebuilding `ds` from the local raw
data would need the same POSIX-root treatment applied there.

## 6. Claude Code session history

The 57 sessions and the per-project memory directory (`memory/MEMORY.md` plus
its fact files) live under the slugified workspace path. Copy to the identical
path on the Mac:

```bash
rsync -aHh --info=progress2 \
  vsekulic@cbp-db.bnf.brain.riken.jp:.claude/projects/-Users-vsekulic-code-caban/ \
  ~/.claude/projects/-Users-vsekulic-code-caban/
```

Opening `/Users/vsekulic/code/caban` in VS Code then lists those sessions, and
`/resume` finds them, because the slug matches the workspace path.

Also worth bringing (all small):

- `~/.claude/CLAUDE.md`, `~/.claude/rules/router.md`,
  `~/.claude/agents/haiku-worker.md`, `~/.claude/agents/sonnet-worker.md`,
  `~/.claude/model-postures.md`, `~/.claude/hooks/` — the global operating
  rules and worker agents those sessions were run under.
- `~/.claude/plans/` (428 KB) — plan files the transcripts reference.
- `~/.claude/file-history/` (59 MB) — only if `/rewind` on old sessions
  matters.

Do **not** overwrite `~/.claude.json` or `~/.claude/settings.json` wholesale:
`~/.claude.json` holds machine-specific `machineID`/`userID` and the Mac's own
state. To carry over the project's command history and allowed-tools list,
merge only the `projects["/Users/vsekulic/code/caban"]` key from the server's
copy; hand-merge `settings.json`.

## 7. Run

```bash
conda activate caban
cd /Users/vsekulic/code/caban
jupyter lab
```

The tmux / SSH-tunnel / Remote-SSH half of [JUPYTER_SETUP.md](JUPYTER_SETUP.md)
is moot locally — only its env and kernel registration parts (A1–A2) still
apply. VS Code opens the folder directly.

---

## Verification

1. **Notebook cell 5** (the environment check): `sys.executable` inside the Mac
   `caban` env, hostname `osgiliath`, numpy/pandas matching the table in step 3.
2. **Path resolution**:
   ```bash
   python -c "from caban.utilities import NPY_SAVE_PATH; print(NPY_SAVE_PATH)"
   ```
   must print `/Users/vsekulic/data/vsekulic/OF_test/npy_files`, not `D:\...`.
3. **Cache load** — cells 7 → 9 (`PipelineConfig(...)`, then
   `load_all_mice(cfg)`). Expect `loading ds from pickle (skipping fresh
   build)` followed by the memory breakdown. Watch RSS in Activity Monitor and
   compare against the 9.4 GB measured on cbp-db.
4. **Lazy restore** — run a section that touches `S_full` / `A` (e.g.
   `run_ROIs`, or the sample-traces section). A `FileNotFoundError` here means
   a `Saver` subdirectory did not transfer.
5. **Numerical parity** — run one cheap statistical section end-to-end (e.g.
   `run_sp_rates`) and diff its CSV against the server's copy of the same file.
   Numbers should match exactly; small figure differences from font
   rasterisation are expected.
6. **Tooling** — in VS Code on the local folder, confirm past Claude sessions
   are listed and that `git fetch` / `git push` reach `origin`.

## Known risks

- **32 GB RAM** — the main constraint (see the memory budget above). Plan on
  kernel restarts between heavy sections rather than one continuous kernel.
- **Version drift** — match numpy and pandas majors before anything else.
- **Old local repo** — keep `caban-legacy-*` until a local run reproduces
  server output; it also holds the 395 MB `freeze_data/` copy.
- **Wall clock** — the slow sections (hierarchical epoch modulation ~25 min on
  cbp-db, the decoders, the PyMC fits) will be slower under memory pressure,
  which dominates any core-count difference between the machines.
