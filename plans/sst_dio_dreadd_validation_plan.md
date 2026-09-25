# Plan: validate the DREADDs on SST cells themselves (DIO-GCaMP6f mice)

Status: **stub**, written 2026-09-23. Nothing designed or executed.
Depends on: `plans/local_minian_pipeline_plan.md` (none of these mice has usable Minian output,
so every session needs CNMF-E first).
Out of scope: the main cohort (G05–G21), whose imaged cells are CA1 pyramidal cells, not the
DREADD-expressing population.

## 1. Why this exists

The main cohort images CA1 pyramidal cells while hM3D / hM4D act on SST interneurons, so every
result so far is an inference about what the DREADD did to SST cells. The DIO mice close that
gap: GCaMP6f is Cre-dependent in SST-Cre mice, so **the imaged cells are the DREADD-expressing
SST cells themselves**. This is the paper's direct control for "do the DREADDs do what we think?"
— hM3D should raise SST activity after CNO, hM4D should lower it.

It gets its own notebook and analysis rather than riding along with the main cohort: the cell
type, the expected effect and the comparisons all differ.

## 2. Cohort — audited 2026-09-23

| mouse | DREADD | recorded | where | sessions | Minian output |
|---|---|---|---|---|---|
| G03-ST639-hM3D_inSST-DIO_GCaMP | hM3D | 2021-07/08, 4 days | MINISCOPE (from 1a) | CNO day 2021_07_29 (§2.1) | partial: `minian/` in 1 session, `minian_intermediate/` in 2 |
| G04-ST624-hM3D_inSST-DIO_GCaMP | hM3D | 2021-07/08, 3 days | MINISCOPE (from 1a) | CNO day 2021_07_27 (§2.1) | partial: `minian/` + `minian_intermediate/` in 2 sessions |
| **G22-ST875-hM3D-DIOGC** | hM3D | 2023-03, 10 days | MINISCOPE **and** 4-MINISCOPE (identical) | full protocol, one `CNO` session | none |
| **G23-ST856-hM3D-DIOGC** | hM3D | 2023-03, 10 days | MINISCOPE **and** 4-MINISCOPE (identical) | full protocol, one `CNO` session | none |
| **G26-ST894-hM4D-DIOGC** | hM4D | 2023-05/06, 11 days | 4-MINISCOPE and MINISCOPE (copied 2026-09-25; server: 2 days) | full protocol, `CNO1` + `CNO2` | none |
| **G27-ST895-hM4D-DIOGC** | hM4D | 2023-05/06, 11 days | 4-MINISCOPE and MINISCOPE (copied 2026-09-25; server: 1 day) | full protocol, `CNO1` + `CNO2` | none |

"Full protocol": HC1–HC3 baseline days, three LT days (HC1/LT1/HC2 each), a TFC_cond day
(HC1, LT1, HC2, CNO, LT2, HC3, TFC_cond, HC4), then TestB, TestA, TestB_1wk, TestA_1wk (each
HC1/LT1/HC2/Test/HC3) — the same shape as the main cohort.

From `G26/2023_05_22-TFC_cond/notes.txt`: `CNO1` starts right after injection, `CNO2` ~10 min
later, right before LT2 — ~20 min from injection to LT2. `14_13_17-TFC_cond-borked` was
restarted because the shocker was off; `14_22_37-TFC_cond` is the real one.

Other oddities seen, not yet understood:
- G22/G23 `HC1` day has `-R` and `-G` recordings plus `AlignG`/`AlignR` images — possibly a
  red (DREADD-mCherry) vs green channel check.
- G22 `HC2` day is an electrowetting-lens focus sweep (`ewl_n90` … `ewl_p90`), and
  `track_day3` has an `HC2_ewl_borked`. Not experimental sessions.
- G22 `TestA_1wk` has `HC1a`/`HC1b` ("HC1a , b because line scans in a"); G27 `HC3` day the same.
- Every day also has ~15–18 bare-timestamp folders before the named sessions — presumably
  setup/focus recordings, to be checked before anything is queued.

### 2.1 G03/G04 session labels, recovered from the files

The 2021 pilots' folders are bare timestamps, but each session carries a label file
(`CNO1.txt` etc.) whose content is the duration, e.g. `315s`. Read 2026-09-23:

| mouse | day | sessions, in order |
|---|---|---|
| G03 | 2021_07_16 | unlabelled ×2 (11.2, 5.4 min) |
| G03 | 2021_07_26 | unlabelled ×3 (3.7, 4.5, 4.5 min) |
| G03 | **2021_07_29** | HC1 15:05, LT1 15:20, **CNO1 15:47, CNO2 16:07**, LT2 16:23, HC2 16:51 |
| G03 | 2021_08_04 | HC1a, HC1b, LT1, HC2 — no CNO label |
| G04 | 2021_07_26 | unlabelled ×1 (11.0 min; `notes.txt`: "657s g04"; ~30 fps, the others ~20) |
| G04 | **2021_07_27** | HC1 11:47, LT1 12:13, **CNO1 12:36, CNO2 12:55** ("mouse essentially immobile in cage"), LT2 13:10, HC2 13:30 |
| G04 | 2021_08_06 | HC1a, HC1b, LT1, HC2 — no CNO label |

Same HC → LT → CNO1 → CNO2 → LT2 → HC shape as the 2023 mice. Not recoverable from the files,
to ask of the lab notebooks: CNO dose and exact injection time; whether the 08_04 / 08_06 days
were drug-free re-tests (washout) or something else; what the unlabelled 07_16 / 07_26 sessions
were.

Not DIO, out of scope here: G24/G25 (`hM3D-SGFR1`), G28/G29 (NPY prism), G30/G31 (2025 HB/OLT).

## 3. Open questions

1. **CNMF-E parameters.** The frozen set (`local_minian_pipeline_plan.md` §3.2) was tuned on
   dense CA1 pyramidal layers. SST interneurons are sparse, larger and fire differently; the
   seed, merge and spatial-penalty values may not transfer. This is a real departure from the
   "one frozen set" argument and needs its own gate.
2. **The primary comparison.** Within-mouse, within-cell: pre-CNO (HC2 / LT1) vs post-CNO
   (CNO1 → CNO2 → LT2 → HC3) on the TFC_cond day. hM4D has two post-injection timepoints;
   hM3D has one.
3. **No DIO-mCherry control.** Time-in-session and handling/injection effects are not
   separated from CNO by a control group. Whether the within-session HC1 → HC2 baseline drift
   is enough to bound them needs deciding.
4. **n = 2 per DREADD.** Inference is at the cell level with mouse as a grouping factor, and
   per-mouse effects should be shown, not only pooled. `statistics-checker` before any figure.
5. **G03/G04.** Were they given CNO? Are the 2021 pilots usable at all, or context only?
6. **Cross-registration.** Needed only if single SST cells are followed across days;
   the CNO question is within one day and may not need it.

## 4. Deliverables (to be designed)

- a METHODS template in `analysis_methods_templates/`;
- `caban/` analysis module and a dedicated notebook;
- per-panel figures in the DREADD box-and-strip style, display order `hM3D`/`hM4D` only
  (there is no mCherry group here).
