# Plan: validate the DREADDs on SST cells themselves (DIO-GCaMP6f mice)

Status: **planning**, 2026-10-21 (stub written 2026-09-23). Nothing run. Worked on in its own chat,
worktree and branch `feat/sst-dio-validation`, in parallel with the Figure 2 v2 chat — rules in
[handover_sst_dio_validation.md](handover_sst_dio_validation.md) §0. Becomes Figure 2 v2 panel **d**
([figure2_redesign_plan.md](figure2_redesign_plan.md) §3, §4.2, §7). The plan proper is §5–§10;
§2.2 is the 2026-10-21 audit it rests on.
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
  setup/focus recordings, to be checked before anything is queued. **Checked for the conditioning
  day, §2.2: an electrowetting-lens focal sweep.**

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

### 2.2 The conditioning day, audited 2026-10-21 (Razer, `/mnt/e/SSTCa2`)

Same layout as the main cohort (`<session>/Miniscope/*.avi`, 1000 frames per file, `metaData.json`
frameRate 20; measured 19.75 fps on G22 TFC1, the main cohort's 19.76). Within each day the
electrowetting lens (ewl) and LED are constant across the named sessions, so every recording of a
day images the same plane: G22 ewl −35 / LED 18; G23 ewl 40 / LED 20; G26 ewl 5 / LED 13; G27
ewl −20 / LED 13.

**The bare-timestamp folders are a focal sweep.** G22 has 16 (13_34_00 … 13_46_05), G23 18
(15_55_49 … 16_03_13): one `.avi` each, ~400 frames (20 s), ewl stepped by 10 from −90 to +80, LED
20, and one `AlignR/` screenshot per folder within seconds of its start (with an `AlignG.jpg` beside
it). They are the plane search before the day, matched to a reference image — not sessions.
**Confirmed by VS (2026-10-21): VS's own focal sweep to find the plane.** Kept on disk as data
(they map the focal axis around the imaged plane, possibly useful later); listed in
`session_queue.SESSIONS_NOT_RUN` with that reason so the decision is visible. G26 and G27 have none
on this day.

Minutes from the start of the first CNO recording (folder times; durations from `timeStamps.csv`):

| mouse | HC1 | LT1 | HC2 (ends) | CNO / CNO1 | CNO2 | LT2 | HC3 | conditioning | HC4 |
|---|---|---|---|---|---|---|---|---|---|
| G22 hM3D | −42.4 | −36.4 | −23.5 (−18.5) | 0 (7.6 min) | — | +8.1 | +18.9 | TFC1 +26.3 (8.6 min), TFC2 +37.3 (11.2 min) | +51.0 |
| G23 hM3D | −32.3 | −26.9 | −14.9 (−9.6) | 0 (5.1 min) | — | +5.2 | +17.4 | +25.9 | +51.6 |
| G26 hM4D | −34.5 | −28.1 | −9.5 (−4.4) | 0 | +15.3 | +21.3 | +32.7 | borked +40.6; real +49.9 | +74.6 |
| G27 hM4D | −24.9 | −19.6 | −7.0 (−1.8) | 0 | +15.9 | +21.8 | +34.2 | +41.9 | +66.5 |

- **Injection time.** G26's `notes.txt` puts CNO1 right after the injection (G27 has no note; its
  2-min HC2→CNO1 gap fits the same). For G22/G23 the single `CNO` recording starts 18.5 and 9.6
  min after HC2 ended, so the injection could be anywhere in that gap — and in the main cohort a
  lone "CNO1" meant ~30 min after injection ([figure2_redesign_plan.md](figure2_redesign_plan.md)
  §4.1.1). This decides which hM3D recordings count as "post"; it needs the lab notebook (§11).
  **Inferred meanwhile (2026-10-21):** the main cohort's 30-min rule is impossible here — it puts
  G22's injection inside LT1 (on the track) and G23's inside HC1's recording. So the injection lies
  between HC2's end and the CNO recording's start: G22 0–18.5 min, G23 0–9.6 min before `CNO`.
  Most likely right before it, as in G26/G27 (same lab, 2023). Under either end of that bracket,
  HC3 is 17–37 min after the injection, past CNO onset, so post = HC3 holds without knowing the
  time (and is +32.7/+34.2 min in G26/G27).
- **G22's conditioning is split** into TFC1 and TFC2, 514.9 s and 671.7 s with a ~2.4-min gap
  (≈ 20 min recorded vs ~22 min for the others). No note in the folders. It matters for any
  conditioning-epoch use of G22, not for the pre/post-CNO homecage comparison.
- **No `notes.csv` has content** on this day for any of the four mice.
- G26's `TFC_cond-borked` (shocker off, restarted) is a real 10559-frame recording under CNO.
  Keep-the-data default: queue it like the others; it is simply not a conditioning session.

## 3. Open questions — status 2026-10-21

1. **CNMF-E parameters** — the parameter check (§6) decides; frozen set first.
2. **The primary comparison** — proposed in §8, to settle with VS, then `statistics-checker`.
3. **No DIO-mCherry control** — the HC1 → HC2 drift is the only bound on time/handling; §8 uses
   it explicitly as the reference change, and the write-up says so.
4. **n = 2 per DREADD** — per-animal and per-cell display; no group-level *P* (§8).
5. **G03/G04** — context only unless VS says otherwise: 2021 pilots, partial old Minian output,
   no injection time; not in the queue (§7).
6. **Cross-registration** — reframed in §8.2: for hM4D, cells silenced by CNO may not be detected
   in a post recording at all, which neither per-recording units nor cross-registration can see.
7. **New:** injection time for G22/G23; why G22's conditioning is split; whether the focal-sweep
   folders are left unrun (§2.2).

## 4. Deliverables

- `caban/sst_dio_validation.py` (parameter-check diagnostics, recording table, events, analysis, panel) and
  `notebooks/sst_dio_validation.ipynb`; a `scripts/run_sst_dio_validation.py` for the Razer, in the
  style of `scripts/run_homecage_cno.py`.
- `analysis_methods_templates/sst_dio_validation_methods.md`, copied into the plot folder.
- Plots under `plots/CURRENT/sst_dio_validation/` (rsynced from the Razer); a parameter-check report in
  `reports/sst_dio_parameter_check.md`.
- Panel d at its box size (123 × 36.8 mm), path handed to VS for the main chat (§10).

## 5. Code change first: scope and labels (small, additive, on this branch)

`discover_sessions` keeps only `MOUSE_RANGE = ("G05", "G21")`, so even a `--path` run on
G22 fails with "not found among … scanned sessions". Change, in `caban/minian_runner.py`:

```python
MOUSE_RANGE = ("G05", "G21")
SST_DIO_MICE = ("G22", "G23", "G26", "G27")   # DIO-GCaMP6f, plans/sst_dio_dreadd_validation_plan.md
# discover_sessions: keep c.mouse in range OR in SST_DIO_MICE
```

Side effect to check before committing: the main chat's `--by-day` batch (no selection) would then
see DIO sessions as pending. It has finished (520/520), and it runs from `~/code/caban` on
`feat/yra-unit-alignment`, which does not get this change until VS merges it — but on merge the
DIO mice must already be done or the next `--by-day` would pick them up. Say so in the commit
message. In `caban/session_queue.py`: the focal-sweep folders into `SESSIONS_NOT_RUN` (after VS
decides). Both changes verified by `--dry-run` on the Razer clone.

## 6. Stage: the parameter check (Razer, own clone `~/code/caban_sst`)

**Run.** HC2 of each mouse's conditioning day (pre-injection, ~5 min, ~6000 frames; four runs),
frozen parameters, `--keep-scratch` so seeds and every intermediate stay:

```
~/bin/logrun "sst: baseline (frozen parameters, HC2 x4)" python scripts/run_minian_batch.py --keep-scratch --path \
  G22-ST875-hM3D-DIOGC/2023_03_20-TFC_cond/14_15_20-HC2 \
  G23-ST856-hM3D-DIOGC/2023_03_20-TFC_cond/16_26_10-HC2 \
  G26-ST894-hM4D-DIOGC/2023_05_22-TFC_cond/13_23_16-HC2 \
  G27-ST895-hM4D-DIOGC/2023_05_22-TFC_cond/15_21_41-HC2
```

in `screen -S sst_gate` (named before VS's no-"gate" rule), after `--dry-run`. Output lands in the session folders like any runner
run; not synced to MINISCOPE until the parameters are decided.

**Inspect** (new diagnostics in `caban/sst_dio_validation.py`, beside the runner's own
`summary_*.png`; there is no reference run, so this is an inspection, not
`caban.minian_gate`'s reproduction check):
- `max_proj` with seeds (initial, after PNR/KS refine, after merge) and final footprints outlined,
  numbered;
- unit count; footprint area and equivalent diameter (px) against what an SST soma should span;
- final units overlaid on a local-correlation image: SST somata visible but missed, and units that
  are fragments of one cell (pairs with centroid < ~soma radius and high C correlation);
- every unit's C, YrA and S (sparse, so all of them, not a sample); PNR per unit;
- residual (`Y − AC − b·f`) max projection: unexplained cells show up there.

**How VS and Claude review it (VS, 2026-10-21).** Claude computes parameter grids in advance on
the Razer, stage by stage from the baseline run's kept intermediates (no stage recomputed
upstream of the one varied), starting with the cell-size-linked values (`wnd` 15, `max_wnd` 7,
`dl_wnd` 5, `thres_dist` 10) set against measured SST soma sizes, then `pnr_refine` thres, the
spatial/temporal `sparse_penal` and the merge `thres_corr`. VS judges them in an interactive review
notebook on the Razer's Jupyter (its own `screen sst_jupyter`, own port, SSH tunnel): per stage, a
widget flips between grid values on the same field (seeds or footprints over `max_proj`, traces),
with the baseline beside it. Every checkpoint also writes a PNG to the parameter-check folder so Claude sees
what VS sees. Decisions go into a parameter log (stage, values tried, value chosen, why). One mouse in
full, then the chosen set headless on the other three and a shorter review.

**Raw projections (2026-10-21, `plots/CURRENT/sst_dio_validation/parameter_check/raw_projections_HC2.png`):**
G27 has the most cells (~30) and the cleanest field; G26 ~15, a few elongated; G22 and G23 under
~10 each. G22's max − mean shows **horizontal line artefacts**: whole bright rows in some frames,
which would seed false units unless handled; G22 and G23 also have bright out-of-focus discs
(debris on the optics). To check: how many G22 frames carry lines, and on which other days.

**Decision with VS.** Accept the frozen set, or a documented per-cohort set. If parameters
change, the runner needs an override mechanism (additive: a named set applied as assignments after
the template's last assignment of each key, recorded in `run.json`, the README and
`parameters.json`, which is read back from the kernel). Not built unless the check needs it.
Baseline runs with the frozen set that are superseded are set aside by the runner (`-ORIG`), not
deleted.

## 7. Stage: the queue

Conditioning-day sessions, all of them (36): G22 9 (incl. TFC1, TFC2), G23 8, G26 10 (incl.
`TFC_cond-borked`), G27 9. Roughly 1.5 days of Razer time at the batch's pace. Selection:
`--labels "G2[2367]-.*-TFC_cond/"`, dry-run shown to VS, then `screen -S sst_cond_day`, checked
within ~20 min. Synced to MINISCOPE through the verified sync. The other days (baseline HC, track,
tests) only if a later question needs them.

## 8. Stage: analysis design (pre-specified with VS, `statistics-checker` before it runs)

### 8.1 What is compared

**Pencilled in by VS (2026-10-21), until the lab notebooks are back:** pre = **HC1**, post = the
**CNO** recording — VS recorded it as a short (5–8 min) post-injection snapshot before LT2.
For G26/G27 (two CNO recordings), post = **CNO2** (+15 min; VS). The
proposals below are superseded where they differ.

- **Recordings:** the homecage recordings of the conditioning day. Pre = HC2 (the last before the
  injection); drift reference = HC1 → HC2; post = the post-injection homecage recordings (CNO /
  CNO1, CNO2, HC3). LT1/LT2 shown alongside as context, not pooled with homecage.
- **Primary post window**, in line with [figure2_redesign_plan.md](figure2_redesign_plan.md)
  §4.1.1: the homecage recording starting 15–35 min after the injection. On §2.2's table, hM4D
  has two (CNO2 +15.3/+15.9 and HC3 +32.7/+34.2) — a tie-break is needed (proposed: CNO2, the
  first in window); hM3D depends on the injection time (§2.2). The full per-animal time course is
  the figure either way.
- **Measures**, as the primary analysis: `find_event_runs_ca_S(S, 2)` per unit; event rate
  (events/min) and mean log per-event amplitude. The threshold is in raw S units, set on pyramidal
  cells: thresholds 1.5 and 3 as a sensitivity, as in `caban.homecage_cno`.
- **Detection diagnostics** per recording: unit count, fraction with ≥ 1 event, S noise level.

### 8.2 Units: why "within cell" is the hard part

With one CNMF-E run per recording, a cell is found only where it is active enough to seed. hM4D
should silence SST cells, so the cells it silences best can vanish from the post recording:
per-recording population means then measure the survivors, and cross-registration can only pair
cells found in both. Both bias the hM4D effect toward zero (and dropout is the measure the
redesign called unpersuasive). Three options:

- **A — session level, no matching** (fallback in the handover): per-animal population rate and
  mean amplitude per recording, plus the diagnostics. Needs nothing beyond §7. Biased as above.
- **B — per-recording runs + within-day cross-registration** (`caban.crossreg_runner`, a new
  grouping using the existing conditioning-day roles). Pairs cells found in both; same bias for
  silenced cells, and each recording's S has its own scale per unit.
- **C — one CNMF-E run over the day's homecage recordings concatenated** (HC1, HC2, CNO/CNO1,
  CNO2, HC3; ~30k frames, about one conditioning session's size). Footprints come from all
  frames, so a cell active before the injection keeps its footprint after it and its silence is
  measured; every unit has one S scale; matching is by construction. Costs a runner extension
  (stage the selected recordings' videos into one work folder, renumbered, with a frame→recording
  table) and one CNMF-E fit spanning ~1 h of a single mount.

**Recommendation:** A first (it comes free with §7 and gives the time course), C for the
within-cell claim; B only if C fails. Decide after the parameter check.

### 8.3 Inference

n = 2 animals per DREADD and no control: no group-level test. VS (2026-10-21): an hM3D vs hM4D
pilot comparison (2 vs 2) is wanted beside the per-animal display — descriptive, effect sizes
with every animal shown, stated as a pilot. Per animal, cell-level: change
post vs pre against the same cells' HC1 → HC2 change (a Poisson/NB model of event counts with
recording duration as offset and cell as random effect, per animal, is the candidate — to be
specified with `statistics-checker`). Shown per cell and per animal; the expected sign (hM3D up,
hM4D down) stated beforehand.

### 8.4 Later: SST cells and memory (VS, 2026-10-21 — a new question, not panel d)

All four mice have the full recall series, each day HC1, LT1, HC2, test, HC3 (G22 TestA_1wk:
HC1a/HC1b): TestB +48 h (2023_03_22 / 05_24), TestA (03_24 / 05_26), TestB_1wk (03_27 / 05_30),
TestA_1wk (03_29–30 / 06_01). With conditioning (under CNO) this allows SST-cell tone/context
coding across conditioning and recall. Needs the same Minian runs on those days (the queue's
later stage) and, for single cells across days, cross-registration. Scoped after panel d.

## 9. Stage: panel d

Nature style, at `figure2_composite.panel_figsize('d', LAYOUT_V2_MM)` (123 × 36.8 mm): per-animal
time course of event rate and amplitude relative to pre (four animals, hM3D then hM4D), and paired
per-cell pre/post strips. Drafted after §8 is fixed; the layout is a proposal for VS then.

## 10. Needs from the main chat (shared files; not edited here)

- `caban/figure2_composite.py`: move `d` from `PENDING_V2` to `PANEL_FILES_V2` with the path given.
- `caban/sections.py`: a `run_sst_dio_validation` entry in `FIGURE2_V2_PANEL_SECTIONS`, if the
  panel is rebuilt from the notebook.
- None of `loader.py` / `sessions.py`: the module reads Minian output directly (as
  `caban.homecage_cno` reads its export) with its own DIO mouse → DREADD map.

## 11. Order of work and what VS decides now

1. VS: the focal-sweep folders — leave unrun, listed in `SESSIONS_NOT_RUN` (recommended), or run.
2. Code (§5) on this branch; Razer clone `~/code/caban_sst`; dry-run.
3. Baseline runs (§6, ~4 h), parameter grids, review with VS → VS decides parameters.
4. Queue (§7) → analysis A, while C is built if VS agrees (§8.2).
5. VS / lab notebook: injection times for G22/G23; G22's TFC1/TFC2 split.
6. §8 pre-specified, `statistics-checker`, run, `scientific-code-reviewer`, panel d (§9).
