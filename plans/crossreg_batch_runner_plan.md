# Plan: cross-registration by registry, and every output folder named `minian`

Status: written 2026-09-26; groupings decided 2026-10-10 (§6); **registry for 1–7 seeded and checked 2026-10-10 (§3.1)**. Next: groupings 8–18 into the registry (after VS's answers in §6), then the runner and its gate (§5, §7), on the Razer.

**One-stop summary of everything decided and checked: [reports/crossreg_groupings_report.md](../reports/crossreg_groupings_report.md).**
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
- **Format: YAML** (VS, 2026-10-10): a master list, readable and editable by hand.
- New groupings (§6) are added to the registry by hand, reviewed, committed.
- **Grouping 1's mapping files (resolved 2026-10-10).** They exist for all 17 mice, in each mouse's
  **conditioning-day** folder, not the mouse folder (grouping 1 is within that day), so the 09-26 survey
  of mouse folders missed them. Names: `mappings_crossreg1.{csv,pkl}` (no underscore) in 16 mice,
  `mappings_crossreg_1.*` in G15; G06 and G07 have the `.csv` only. G05 also holds stray copies
  (`mappings_crossreg1 - Copy.csv`, `old_mappings/`, and a set in the mouse folder). The analysis
  loader reads a cached copy (`CrossRegMapping`'s saver on FUTROLA), made once from a file it calls
  `mappings_crossreg_1.csv`. The registry records the real file per mouse, and its check proves the
  cached copy equals that file, so it is known which file the published results came from.
- **G17 (VS: "don't bork this up").** Its conditioning folder is misnamed, and the notebook carries a
  hard-coded fix (cell 16; [yra_recompute_plan.md](yra_recompute_plan.md) §6.5). The registry records
  it explicitly, and the runner's gate must reproduce G17's published mappings with it.
- `caban` code that needs membership reads the registry — never a folder name.

### 3.1 Seeded 2026-10-10 — what the check found

`scripts/seed_crossreg_registry.py` → `caban/crossreg_groupings.yaml` (groupings 1–7). Membership comes
from each **mapping file's own session columns**, not from folder names, and each member is matched to
the output folder whose cells the mapping refers to. All checks pass:

- **Every mapped unit id exists in its member's output**, for all 7 groupings, 16–17 mice each.
- **The loader's cached mappings** (`<cache>/TFC_cond/crossreg/`, the only copies the loader reads; it
  builds every `CrossRegMapping` with that savepath) **equal the files on disk by value**. The 2023
  cache stores object columns where today's pandas reads strings, so the comparison is by value.
- **18 members have no crossreg-tagged folder**: grouping 5's Test_B_1wk LT1 in all 16 mice, and
  grouping 3's Test_B LT1 in G10/G11. Their current `minian` (the batch's re-run) carries **exactly the
  mapping's unit ids**, as does the set-aside `minian-ORIG` where one exists. So the re-runs reproduced
  those sessions' cell ids, and the published mappings stay valid with the current output.
- **One stale published mapping: G15 grouping 2.** Its TFC_cond column refers to an earlier run, kept in
  `prev/minian` (451 units; all mapping ids there). The current output (417 units) has its groupings 1, 4,
  6 and 7 recomputed in 2026, but grouping 2 is from 2022. The registry records the member as
  `output: prev/minian` with a `stale:` note. Grouping 2 is not used by the loader.
- **One stale orphan cache copy: G15 grouping 4** under `<cache>/Test_B/crossreg/` (2023-11-07), referring
  to that earlier run. It is not read by the loader (which reads the up-to-date `TFC_cond/crossreg/` copy).
  The other 16 orphans in `Test_B/crossreg/` equal their files.
- **G12: two sessions started at 14:03:29** (the Test_B-day LT1 and TFC_test_A_1wk). Mappings name sessions
  by start time only; the seed resolves each by the grouping's folder tag.
- **G05's `old_mappings/`** holds earlier, *different* versions of groupings 1–4 and 6. The published files
  are the ones in the mouse folder (2–7) and the conditioning-day folder (1).
- **G17:** the published mappings name its conditioning session `18_26_30` (MINISCOPE's folder name). The
  notebook's hard-coded fix *sets* it to `918_26_30`, the server's misnamed folder. Applied to MINISCOPE it
  would mismatch the published name, so the runner's gate must reproduce G17 exactly (§5).

**Re-run all groupings (VS, 2026-10-10).** The runner re-derives 1–7 for every mouse as well as computing
8–18, which makes its gate a full reproducibility test of every published mapping. The published files are
set aside (`-ORIG`), never overwritten, and analyses keep using them unless VS decides otherwise. G15's
stale grouping 2 is the one expected difference.

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
(`yra_recompute_plan.md` §13.3). Both cannot become `minian`. **Resolved 2026-10-10:** `minian_crossreg2_crossreg3` is a
byte-identical copy of G16's TFC_test_B output (`14_47_09-TFC_test_B/…/minian_crossreg2_crossreg3_crossreg4_crossreg6`;
only `.DS_Store` and the YrA sidecar differ), and every other mouse's Test_B-day LT1 uses
`minian_crossreg3`. G16's published grouping 3 used the right folder: all 787 LT1 unit ids in
`mappings_crossreg_3` are in `minian_crossreg3`. **Decision (VS): set the copy aside** (recorded, not
deleted), before the rename and before seeding the registry, which would otherwise put it in groupings
2 and 3.

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

## 6. Which groupings (VS) — decided 2026-10-10

**Purpose: compute every grouping ahead of time, once**, so later work needs only the small mapping
files and the local caches on FUTROLA, not MINISCOPE. **No analysis is run with them automatically**;
each is used deliberately, when a question needs it (VS).

**Keep in mind (VS):** the more sessions a grouping holds, and the longer it spans, the fewer cells
are matched across all of them (drift). A large grouping may have very few, or no, cells common to
every session. That is why the groupings are targeted, and why some overlap.

Groupings 1–7 stay exactly as published. New ones (numbers to confirm when the registry is written):

| grouping | sessions (per mouse) | question | from VS's list |
|---|---|---|---|
| 8 | conditioning day, all but HC4: HC1, LT1, CNO1, CNO2, LT2, HC2, TFC_cond, HC3 | per-cell trajectory across the drug onset, the whole day | A (G18's HC4 left out, as for the other mice) |
| 9 | conditioning day without homecage: LT1, CNO1, CNO2, LT2, TFC_cond | as 8, without HC (HC activity varies greatly) | H |
| 10 | CNO1, CNO2 | cells before vs after CNO; both are homecage sessions with the injection in between | G — **key** |
| 11 | every LT session: LT1 of each track day, and LT1/LT2 of every other day | place-cell stability over days and weeks | B |
| 12 – 16 | within one track day: HC1, LT1, HC2 (one grouping per track day 1–5; days 4–5 for G08 only) | homecage vs track per cell; reactivation in HC after LT | C |
| 17 | every HC session, **not** the HC0s | homecage baseline over the experiment | E |
| 18 | G05 only: `2021_09_01-TFC_test_B` HC1 + iso | the per-cell part of the isoflurane analysis ([isoflurane_recording_analysis_plan.md](isoflurane_recording_analysis_plan.md)) | F |

Dropped: **D** (within each test day; VS: "HC is not a behavior"). The HC0s and the other misc
sessions are in no grouping.
## 7. Order of work

1. Seed the registry from folder names; prove it reproduces them (§3).
2. Build the runner (§5); **gate**: re-derive one existing grouping for one mouse (e.g. crossreg4,
   G10) from the registry, compare with its production `mappings_crossreg_4`.
3. VS chooses the new groupings (§6); run them after the Minian batch has produced the sessions.
4. Loader and discovery read the registry; then the rename to `minian` (§4).

Steps 1–2 can proceed while the Minian batch runs; the batch itself needs nothing from this plan —
new sessions are written as `minian/` already, and the YrA recompute writes inside whatever folder
holds `A`/`C`/`S`.
