# Plan: cross-registration by registry, and every output folder named `minian`

Status: **planning**, written 2026-09-26. Nothing built. **Blocked on VS for §6 (which groupings).**
Depends on: `plans/minian_batch_runner_plan.md` (the runner pattern: protected template, papermill,
parameters from the kernel, gate against production), `plans/local_minian_pipeline_plan.md` §3.4
(cross-registration never modifies `A`/`C`/`S`) and §7 (cross-registration is out of the CNMF-E pass).

## 1. What it is for

The 442 sessions the Minian runner is about to process (`minian_batch_runner_plan.md`) have `A`/`C`/`S`
but no cross-registration, so no longitudinal analysis can use them. Two decisions by VS, 2026-09-26:

1. **Automate cross-registration the way the Minian runner automates CNMF-E**: run the real
   cross-registration notebook per mouse and grouping, unattended.
2. **Membership is computed, not encoded in folder names.** Today a session joins grouping *N* by
   having `_crossregN` appended to its output folder name by hand (a "lazy grep option", VS). From
   now on a registry says which sessions form each grouping, and **every output folder is named
   `minian`**, as Minian intended (its own `open_minian_mf` default pattern is `r"minian$"`).

## 2. How cross-registration works today

`cross-registration-WORKING.ipynb` (fork, `~/code/minian_vsekulic`), 59 cells:

- Parameters (cell 4): `dpath` = the **mouse** folder, `f_pattern_prefix = "crossreg"`,
  `f_pattern = "<N>"`, `id_dims = ["session"]`; `param_dist = 5` px (cell 7).
- Cell 15: `open_minian_mf(dpath, id_dims, pattern="crossreg<N>")` walks the mouse folder and opens
  every output folder whose **name** contains `crossreg<N>`. That name match *is* the grouping.
- Cell 16 renames sessions to the time part of the folder (`16_32_14`), with one hard-coded fix:
  `if 'G17' in dpath and TFC_cond: new_session[2] = '918_26_30'` (G17's misnamed conditioning
  folder, `yra_recompute_plan.md` §6.5).
- Cell 55 writes, into the mouse folder: `mappings_crossreg_<N>.pkl` / `.csv`,
  `cents_crossreg_<N>.pkl`, `shiftds_crossreg_<N>.nc`. Nothing else (parent plan §3.4).

The seven groupings in production, measured 2026-09-26 from folder names on MINISCOPE (typical
mouse; a few mice differ by a session):

| grouping | sessions | mice |
|---|---|---|
| crossreg1 | 3 sessions within the TFC_cond day | 17 |
| crossreg2 | TFC_cond → Test_B | 16 |
| crossreg3 | 2 sessions within the Test_B day | 16 |
| crossreg4 | TFC_cond → Test_B → Test_B_1wk | 17 |
| crossreg5 | Test_B_1wk | 16 |
| crossreg6 | TFC_cond + all four tests | 17 |
| crossreg7 | TFC_cond → Test_A → Test_A_1wk | 17 |

Mapping files present: `mappings_crossreg_2..7.pkl` in 16–17 mice each; grouping 1 exists as
`mappings_crossreg1.pkl` (no underscore) in **one** mouse only — to be explained before the
registry claims grouping 1 has mappings everywhere.

## 3. The registry

One file, tracked in git so every change to a grouping is reviewable: `caban/crossreg_groupings.json`
(or `.yaml`; VS to choose). Per grouping: a number, a short name, a description of the scientific
question, and per mouse the list of session labels `<mouse>/<day>/<session>`.

- **Seeded from today's folder names**, mechanically: for every `minian_crossreg*` folder, each
  `crossreg<N>` in its name puts that session in grouping *N*. A script writes it and a check
  proves the registry reproduces the folder-name grouping exactly, for all 17 mice and 7 groupings.
- New groupings (§6) are added to the registry by hand, reviewed, committed.
- `caban` code that needs membership reads the registry — never a folder name.

## 4. Renaming every output folder to `minian`

**Not before §3 is in place and read by the code.** Today these depend on the `minian_crossreg*`
name:

| what | where | reads the name for |
|---|---|---|
| the analysis loader | `sessions.py:521` globs `minian_crossreg*` | finding a session's `A`/`C`/`S`/`YrA` |
| shared discovery | `session_queue._complete_minian_dirs`, `resolve_minian_dir`, `scan_sessions` summary | "is this session processed and cross-registered?" |
| YrA recompute | `yra_recompute.discover_sessions` | which sessions are its work |
| Minian runner | `minian_runner.is_never_processed` | telling production from new |
| cross-registration notebook | cell 15 `pattern` | grouping membership |

And one conflict: **G16 `2022_01_26-TFC_test_B/14_23_45-LT1` has two output folders**
(`minian_crossreg2_crossreg3`, `minian_crossreg3`); only the second fits the session's own videos
(`yra_recompute_plan.md` §13.3). Both cannot become `minian`; VS decides what the other becomes
(set aside as `minian-ORIG`-style, or archived).

Order: registry (§3) → loader and discovery read it, with a test that every session resolves to
the same folder as before → rename, recorded per session like `set_aside_minian_output` records
its renames, so it can be reversed → re-run the equivalence test on the new names.

Recomputed `YrA` lives inside the output folder (`YrA_recomputed.zarr`), so it moves with the
rename; `yra_recompute.completed_run` validates by input hashes, not path, so nothing recomputes.

## 5. The runner

Same shape as `caban/minian_runner.py`:

- **Protected template**: an output-free, read-only copy of `cross-registration-WORKING.ipynb`
  in `notebooks/`, md5 of its code checked before every run. Which of the four crossreg notebooks
  is canonical is checked by diffing their code first, as for the pipeline notebook.
- **Membership without renaming anything**: for a (mouse, grouping), build a scratch view of the
  mouse folder — real `<day>/<session>/Miniscope/` directories, and in each chosen one a symlink
  named `minian_crossreg<N>` pointing at the real `minian`. `open_minian_mf` uses `os.walk`
  without following links, but matches on the names in each directory listing, symlinks included,
  and opens the match through the link — so the unchanged notebook sees exactly the registry's
  sessions. To confirm on a real run before relying on it.
- **Edits**: `dpath` (the scratch view), `f_pattern` (the grouping number), `TFC_cond` as the
  grouping requires; `param_dist = 5` unchanged. Effective parameters written from the kernel.
- **Outputs** are copied from the scratch view into the real mouse folder. Existing mapping files
  are never overwritten: `session_queue.set_aside_crossreg_output` already renames a grouping's four
  files to `-ORIG` and refuses if that was done before.
- **Checks**: `A`/`C`/`S` hashes unchanged before/after (parent §3.4, "worth re-verifying");
  every registry session appears in the mapping; the mapping's session names are the ones expected.

## 6. Which groupings (VS)

Scientific, not mechanical — the registry records the answer. Candidates, as raised in the parent
plan:

- **Whole TFC_cond day** — HC1, LT1, CNO1, CNO2, LT2, HC2, TFC_cond, HC3 — for the per-cell
  before/after-CNO question.
- **LT across track days** — LT1 of track_day1–3 (and beyond) — place-cell stability.
- **HC sessions** with the LT/TFC session of the same day, if HC activity is to be compared per cell.
- Whether any existing grouping should be **extended** with new sessions, or left exactly as
  published (recommended: leave 1–7 untouched; new questions get new numbers).

## 7. Order of work

1. Seed the registry from folder names; prove it reproduces them (§3).
2. Build the runner (§5); **gate**: re-derive one existing grouping for one mouse (e.g. crossreg4,
   G10) from the registry, compare with its production `mappings_crossreg_4`.
3. VS chooses the new groupings (§6); run them after the Minian batch has produced the sessions.
4. Loader and discovery read the registry; then the rename to `minian` (§4).

Steps 1–2 can proceed while the Minian batch runs; the batch itself needs nothing from this plan —
new sessions are written as `minian/` already, and the YrA recompute writes inside whatever folder
holds `A`/`C`/`S`.
