# Plan: validate the DREADDs on SST cells themselves (DIO-GCaMP6f mice)

Status: **planning**, 2026-10-21 (stub written 2026-09-23). Nothing run. Worked on in its own chat,
worktree and branch `feat/sst-dio-validation`, in parallel with the Figure 2 v2 chat — rules in
[handover_sst_dio_validation.md](handover_sst_dio_validation.md) §0. Becomes Figure 2 v2 panel **d**
([figure2_redesign_plan.md](figure2_redesign_plan.md) §3, §4.2, §7). The plan proper is §5–§10;
§2.2 is the 2026-10-21 audit it rests on.

## Checkpoint 2026-10-24 — Razer shut down for VS's move; resume here

**Where it stands.** The parameter walk is finished and decided (§6.1, parameter log): the SST set
is the pyramidal frozen set throughout, plus two SST-specific steps that change no parameter —
line-artefact frames dropped, lens-rim seed mask. Check runs on G22/G23/G26 HC2 succeeded (G23 no
longer crashes). Nothing has been queued for production; nothing was running at shutdown.

**Open decisions for VS (asked 2026-10-22, not yet answered):**
1. Approve the proposed ellipses for G22, G23, G26 (`sst_dio_common.SEED_MASKS_PROPOSED`; figures
   `parameter_check/seed_mask/<mouse>_HC2_try3.png`) — then they go, with G27's, into
   `session_queue.SEED_MASKS` for every session of each mouse's conditioning day.
2. G22 TFC1: drop its ~24 isolated line frames and the corrupted end block (frame 9918 to the end,
   ~250 frames)? Plus G22 LT1 frame 3905. Then into `yra_recompute.NOTEBOOK_DEL_FRAMES`.

**Then, in order:** fill `SEED_MASKS` and `NOTEBOOK_DEL_FRAMES`; decide how the runner treats the HC2
sessions that already hold a frozen-set run without the mask (G22, G26, G27 `done`; G23 `failed`:
set aside as `-ORIG`, never deleted); `--dry-run` of the 36 conditioning-day sessions; queue them
(`screen sst_cond_day`); then the analysis design (§8, `statistics-checker` first).

**Where things are.**
- Code: branch `feat/sst-dio-validation` (worktree `.claude/worktrees/sst-dio-validation` on the
  Mac, `~/code/caban_sst` on the Razer), all pushed. Work on the worktree by path from the caban
  folder; never `EnterWorktree`, never `cd` into it (memory: feedback_never_enter_worktree).
- Minian fork fix (`adj_corr`, empty merge graph): commit `494e38e5` on the Razer's
  `~/code/minian_vsekulic` (`vsekulic_v4`), **not pushed**; a copy as a patch in
  [minian_fork_adj_corr_empty_merge_graph.patch](minian_fork_adj_corr_empty_merge_graph.patch).
- Results: `~/sst_dio_out/` on the Razer (271 GB with intermediates); everything but the Minian
  arrays copied to `/Volumes/FUTROLA/caban_cache/sst_dio_out_checkpoint/` (1.9 GB: parameter log,
  stage tables, `variant.json` records, executed notebooks, figures, VS's notebook runs). Figures
  also in `plots/CURRENT/sst_dio_validation/`.
- Baseline (frozen-set) runs of the four HC2 sessions: in their session folders on `/mnt/e`
  (G23's failed run's work folder in `~/minian_stage`).

**Restarting on the Razer.** The review server:
`screen -dmS sst_jupyter bash -lc "source ~/miniforge3/etc/profile.d/conda.sh && conda activate
minian-native && cd ~/code/caban_sst && ~/bin/logrun 'sst: jupyter' jupyter notebook --no-browser
--ip 127.0.0.1 --port 8899 --notebook-dir ~/code/caban_sst"`, then `jupyter notebook list` for the
new token; VS tunnels with `ssh -N -L 8899:localhost:8899 minastirith`. The main chat's
`status_report` screen (batch status every 20 min) ends with the shutdown and does not restart by
itself.
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

**Baseline with the frozen set (2026-10-21, ~16–18 min per HC2 on the Razer; figures in
`plots/CURRENT/sst_dio_validation/parameter_check/baseline_frozen/`):**

| mouse | units | what the footprints show |
|---|---|---|
| G22 | 84 | a few somata; most units sit on the bright line artefacts (one spans the full width) |
| G23 | — | **failed** at `unit_merge` after initialisation: "need at least one array to concatenate" (no pair to merge; the same failure as G13's HC0) |
| G26 | 23 | ~17 somata, plus units on the lens edge (top right) |
| G27 | 151 | ~50 somata found well in the centre; ~40+ units along the lens rim and image edges |

So the first problems are not cell size: (1) line artefacts (G22), (2) units on the lens rim
(G26, G27), (3) the empty-merge failure on a sparse field (G23).

**Line artefacts = Miniscope connection drop-outs (VS).** VS's hand-run notebooks dropped such
frames per session (the template's commented "CUSTOM SUBSETS FOR FRAMES" cell: a `del_frames`
list, `varr_ref.where(~frame.isin(del_frames), drop=True)` before motion correction, frame labels
kept; e.g. G25 TFC_cond, 27 frames). Automated here by score
(`sst_dio_validation.row_artefact_scores`: each row's contrast against its neighbours, robust z
against the same row over the recording, so the steady lens edge does not score): HC2 G22 **8 of
5978 frames** (206, 1249, 2814, 3705, 3740, 4375, 4396, 4702; z 17–58), G23/G26/G27 none; every
unflagged frame z ≤ ~5. All eight visibly carry a line. Each flagged frame goes to VS on a
contact sheet before it is listed. To apply them the runner needs a per-session dropped-frames
registry and a notebook edit after the `varr_ref` cell (small, additive).

**Done for the parameter check (2026-10-21, VS agreed):**
- `caban.minian_runner` (additive; unset, the run notebook is byte-identical to before, tested):
  `del_frames` from `yra_recompute.NOTEBOOK_DEL_FRAMES` dropped from `varr` after `load_videos`
  (as the template's G25 block; the YrA replay reads the same table); an optional lens-interior
  **seed mask** (`session_queue.SEED_MASKS`, a cell after `seeds_init` keeping seeds inside an
  ellipse; the movie is not altered); optional **parameter overrides**, appended after every cell
  assigning that parameter and checked against the kernel's `parameters.json`.
- G27's ellipse proposed (`sst_dio_validation.SEED_MASKS_PROPOSED`; centre 335, 300; radii 265,
  285): 60 baseline units inside, every rim/edge unit outside
  (`parameter_check/seed_mask/G27_HC2_try2.png`). Into `SEED_MASKS` once VS approves.
- **Cell size is not the problem:** with the frozen set, G27's footprints (median 330 px, A > 0)
  are *smaller* than pyramidal G12 homecage's (390–420 px). Several G27 cells carry 2–3 units
  (fragments), which points at the merge and seed-merge settings more than at sizes.
- Effective `max_wnd` is **15** (kernel's `parameters.json`; template cell 14), not the 7 in
  [local_minian_pipeline_plan.md](local_minian_pipeline_plan.md) §3.2's table — for VS / the main
  chat, whose plan that is.
- Grid on G27 HC2, one value at a time (`PARAMETER_GRID_ONE_AT_A_TIME`, 14 variants: tophat `wnd`
  10/20, `max_wnd` 10/20, PNR `thres` 1.5/2, seed-merge `thres_dist` 5/15, all merge `thres_corr`
  0.7/0.9, spatial `sparse_penal` 1e-3/1e-5, `dl_wnd` 8), full runs outside the session folder in
  `~/sst_dio_out/parameter_grid/G27_HC2/` on the Razer, intermediates kept (~7 GB each), four at a
  time. Kernel starts are spaced 30 s (four started together: one kernel died before replying).
- Review: `notebooks/sst_dio_parameter_review.ipynb` (helpers `caban/sst_dio_parameter_review.py`),
  served by Jupyter Notebook from the `minian-native` env (nothing installed) in
  `screen sst_jupyter`, port 8899 on the Razer's loopback; VS tunnels with
  `ssh -N -L 8899:localhost:8899 minastirith`.

**G27 grid, first pass (2026-10-21; all with the seed mask; ~28 min per variant, 4 at a time):**

| variant | units | close pairs (< 8 px) | median footprint px | note |
|---|---|---|---|---|
| frozen | 61 | 11 | 451 | left cluster split into several units |
| background `wnd` 10 / 20 | 48 / 64 | 9 / 7 | 356 / 516 | 10 loses faint peripheral cells |
| `max_wnd` 20 | 60 | 10 | 455 | ≈ frozen |
| PNR `thres` 1.5 / 2 | 25 / 10 | 2 / 1 | 604 / 619 | clean but misses visibly active cells |
| seed-merge `thres_dist` 5 / 15 | 62 / 61 | 11 / 11 | 451 | no effect |
| merge `thres_corr` 0.9 | 75 | 26 | 384 | more splitting: the useful direction is lower |
| merge `thres_corr` 0.7 | 52 | 2 | 499 | crashed before the minian fix; after it (494e38e5): the left cluster mostly one outline per cell, peripheral cells kept — best so far (Claude's read; VS judges) |
| spatial `sparse_penal` 1e-3 / 1e-5 | 61 / 61 | 12 / 10 | 376 / 458 | small effect |
| `max_wnd` 10 | 54 | 8 | 479 | re-run after a kernel-start race |
| spatial `dl_wnd` 8 | 61 | 6 | 688 | fewest splits, larger footprints; re-run after one .avi read returned 1001 frames under four readers |

**The empty-merge failure, diagnosed:** `unit_merge` → `adj_corr` → `graph_optimize_corr` builds the
graph of spatially overlapping unit pairs; when **no two units overlap** the graph has no edges,
no correlations are computed and `np.concatenate([])` raises. It is a valid state Minian does not
handle (G23 HC2 with the frozen set; G27 at merge 0.7). Proposed fix (VS's decision; the fork
`~/code/minian_vsekulic` is shared with the main chat): when there are no overlapping pairs,
return the units unchanged. It changes behaviour only where Minian now crashes, so no finished
run would differ. **Done (VS approved, 2026-10-21):** minian fork `vsekulic_v4` commit `494e38e5`
on the Razer (`adj_corr` returns the all-zero matrix when the overlap graph has no edges; covers
`unit_merge` and `seeds_merge`). Not pushed to GitHub. **For the main chat:** runs from now on
record this fork commit in `run.json`; G13's HC0 (`SESSIONS_NOT_RUN`, the same failure) could now
run, if VS wants it.

### 6.1 The staged walk (VS, 2026-10-22)

VS: the one-at-a-time grid conflates stages (a step-1 change is only seen after every later step
has reacted to it), and deciding each value against the frozen set ignores that the steps build on
each other. So the pipeline is walked one stage at a time, as in the Minian notebook: each stage's
grid stops right after the stage (`STAGE_CUT_AFTER`), is judged at the stage's own output, and VS's
decision (`DECIDED_OVERRIDES`, the parameter log) is carried into the next stage's grid. Claude
gives its own recommendation per stage, **in bold**. The one-at-a-time grid stays as a
sensitivity map. Runs: `scripts/sst_dio_parameter_stage.py`, output
`~/sst_dio_out/parameter_stages/G27_HC2/<stage>/` (Razer); figures in
`plots/CURRENT/sst_dio_validation/parameter_check/stage_<stage>_G27/`.

**Stage 1: preprocessing** (denoise median `ksize` 3/5/7 × tophat `wnd` 10/15/20/25/30, runs to
`max_proj` after motion correction). Measured on the same 52 reference cells (the merge 0.7
run's units) for every variant: contrast in `max_proj` (cell disc − ring, in background s.d.),
transient SNR (disc − ring trace of `Y_fm_chk`, 99.9th-percentile excursion over the frame-
difference noise), residual background (s.d. of the smoothed `max_proj` away from cells, relative
to cell contrast), core width (radial profile, 1-px bins: coarse). Cell size for reference: cores
~10 px at half peak, ~16 px (largest ~21) at 20 % of peak.

| | wnd 10 | 15 | 20 | 25 | 30 |
|---|---|---|---|---|---|
| ksize 3: contrast / SNR / residual | 3.07 / 5.23 / 0.190 | 3.36 / 6.10 / 0.185 | 3.36 / 6.10 / 0.194 | 3.24 / 6.34 / 0.210 | 3.17 / 6.32 / 0.223 |
| ksize 5 | 4.72 / 5.67 / 0.100 | 4.99 / 6.38 / 0.109 | **4.89 / 6.74 / 0.124** | 4.61 / 6.59 / 0.143 | 4.53 / 6.71 / 0.157 |
| ksize 7 | 5.12 / 5.98 / 0.083 | 4.93 / 6.65 / 0.111 | 5.23 / 6.82 / 0.121 | 5.05 / 6.91 / 0.136 | 4.75 / 6.81 / 0.154 |

Core width 10 px at wnd 10 (ksize 5, 7), 12 px elsewhere. By eye (`max_proj_grid.png`): ksize 3 is
grainy; wnd 25–30 leave a diffuse band (neuropil/processes, top right) that invites false seeds,
faint at 20, mostly gone at 15; wnd 10 shrinks the cells; ksize 7 makes cells blobbier and brings
out processes.

**Claude's recommendation (revised 2026-10-22): ksize 5, wnd 15 — the frozen values, as for the
pyramidal cohort.** First recommended wnd 20, on the argument that SST cells are large; measured
against the pyramidal output, they are not: at the same settings, footprint diameters are the same
(half peak: G27 SST median 9.9–10.4 px, pyramidal G12 10.4, G16 9.0; 20 % of peak: SST 15.8–16.3,
max 21; pyramidal 14.6–16.2, max 24–34). Without a cell-size reason, wnd 20 buys +6 % transient SNR
(6.38 → 6.74) for +14 % residual background and the faint diffuse band, and makes SST preprocessing
differ from the pyramidal cells it is compared with. `ksize`: 3 is clearly worse (contrast −33 %,
residual ×1.7); 7 gains only 1–4 % SNR over 5 at the cost of spatial blur. Alternative if VS sees
the largest cells trimmed in the viewer: ksize 5, wnd 20.

| stage 1 | pyramidal (Direct.GCaMP, G05–G21, all finished runs) | SST DIO (recommended) |
|---|---|---|
| video load | all frames (`[0-9]+\.avi`) | same; plus `del_frames` where line artefacts (G22 HC2: 8 frames), VS's G25 practice |
| glow removal | `varr − min over frames` | same |
| denoise | median, `ksize` 5 | **median, `ksize` 5** |
| background removal | tophat, `wnd` 15 | **tophat, `wnd` 15** |
| motion correction | `estimate_motion(dim="frame")`, no subset | same |
| footprint diameter (half / 20 % of peak, median) | 9.0–10.4 / 14.6–16.2 px (G12, G16 HC2) | 9.9–10.4 / 15.8–16.3 px (G27 HC2) |
| G27 HC2 at these settings: contrast / transient SNR / residual | — | 4.99 / 6.38 / 0.109 |

**Stage 1 decided (VS, 2026-10-22): ksize 5, wnd 15** (logged in
`~/sst_dio_out/parameter_stages/G27_HC2/parameter_log.csv`; `DECIDED_OVERRIDES`).

**Stage 2: seeds** (seed `max_wnd` 10/15/20 × PNR threshold 0.75/1/1.25/1.5; KS `sig` 0.05 and the
seed merge (`thres_dist` 10, `thres_corr` 0.8) as in the frozen set). Run in Minian's env from
stage 1's decided movies, calling Minian's seed functions as the notebook does
(`caban/sst_dio_seed_grid.py`, ~3 min for all 12): `seeds_init` once per `max_wnd`; PNR computed
once and each threshold applied to it (Minian's `pnr_refine` sets `mask_pnr = pnr > thres`); seed
merge per combination. The lens mask removes ~73 % of the initial seeds (1264–1532 → 323–389),
nearly all on the rim. KS removes almost none.

| max_wnd \ PNR | 0.75 | 1.0 | 1.25 | 1.5 |
|---|---|---|---|---|
| 10: final seeds / recall / cells ≥ 2 seeds / unmatched | 67 / 0.88 / 8 / 13 | 58 / 0.87 / 7 / 6 | 44 / 0.67 / 5 / 4 | 25 / 0.44 / 2 / 0 |
| 15 | 78 / 1.00 / 7 / 16 | **66 / 1.00 / 5 / 7** | 45 / 0.71 / 4 / 4 | 25 / 0.44 / 2 / 0 |
| 20 | 81 / 1.00 / 9 / 19 | 66 / 1.00 / 7 / 7 | 44 / 0.69 / 4 / 4 | 25 / 0.44 / 2 / 0 |

Recall: reference cells with a final seed within 6 px. The unmatched seeds were checked one by one
(`unmatched_<variant>.png`: crop and trace with PNR): at max_wnd 15, PNR 0.75 adds 9 seeds with
PNR 0.89–0.99, all noise (flat traces, off-cell or on a neighbour's halo); the 7 with PNR > 1 sit on
or beside reference cells (second seeds on a cell, for the later merges), not missed cells.

**Claude's recommendation: max_wnd 15, PNR threshold 1.0 — the frozen values.** Full recall needs
max_wnd ≥ 15 and PNR ≤ 1.0; PNR 1.25 already loses 29 % of the cells, PNR 0.75 adds only noise;
max_wnd 20 gives the same seeds as 15 with more cells holding two or more (7 vs 5). The split cells
are left to the merge stages, where the one-at-a-time grid showed the effect (merge 0.7). Caveat:
the reference cells came from a PNR 1.0 run, so a cell only a lower threshold could find would not
be in it; the unmatched-seed check is what covers that.

**Stage 2 decided (VS, 2026-10-22): max_wnd 15, PNR 1.0, lens mask on** — on Claude's
recommendation without a notebook review (values unchanged from the pyramidal set).

**Stage 3: initialisation and its merge** (initialise `thres_corr` 0.7/0.8/0.9 × `wnd` 10/15 ×
init-merge `thres_corr` 0.6/0.7/0.8/0.9; `caban/sst_dio_init_grid.py`, Minian's `initA` → `initC`
→ `unit_merge` as the notebook calls them, ~2 min for all 24). Measured on the reference cells:
recall, cells with ≥ 2 units (split), units covering ≥ 2 reference centroids at 0.3 of their peak
(fused), unmatched units.

| initialise corr (wnd 10) \ init-merge | 0.6 | 0.7 | 0.8 | 0.9 |
|---|---|---|---|---|
| 0.7: units / recall / split / fused | 55 / 0.98 / 2 / 1 | 58 / 0.98 / 3 / 1 | 61 / 0.98 / 5 / 1 | 64 / 1.00 / 6 / 1 |
| 0.8 | 59 / 0.98 / 3 / 1 | 61 / 1.00 / 4 / 1 | **63 / 1.00 / 5 / 1** | 65 / 1.00 / 6 / 1 |
| 0.9 | 63 / 1.00 / 5 / 0 | 63 / 1.00 / 5 / 0 | 65 / 1.00 / 6 / 0 | 66 / 1.00 / 7 / 0 |

`wnd` 15 gives the same as 10. The initialise correlation sets the size of the *initial* footprints
(median ~110 px at 0.7, 60 at 0.8, 20 at 0.9), which the CNMF spatial updates then reshape; at 0.7
one reference cell loses its unit. The init merge changes little: 0.7 vs 0.8 is nearly identical
on the overlay (two units fewer, in the left cluster); 0.6 loses a cell.

**Claude's recommendation: initialise corr 0.8, wnd 10, init-merge 0.8 — the frozen values.**
Merging here works on the small initial footprints and their rough traces; the split cells are
better judged at stage 4, after the spatial update, by the merge after it (`param_first_merge`),
on full footprints and refined traces — where 0.6 here already showed the cost of merging early.

**Stage 3 decided (VS, 2026-10-22): initialise corr 0.8, wnd 10, init-merge 0.8** — on Claude's
recommendation. Stages 1-3 are therefore the pyramidal values throughout, plus the SST-specific
handling (line-artefact frames dropped, lens-rim seed mask).

**Stage 4: CNMF** — full notebook runs (the runner's notebook, all decisions as overrides, the lens
mask): the merge after the first update (`param_first_merge` `thres_corr` 0.6/0.7/0.8) × the
spatial updates' `dl_wnd` (5/8, both updates); output `~/sst_dio_out/parameter_stages/G27_HC2/cnmf/`.

| first merge \ dl_wnd | 5 | 8 |
|---|---|---|
| 0.6: units / split / fused / unmatched / footprint px | 54 / 1 / 1 / 2 / 505 | 55 / 0 / 2 / 4 / 726 |
| 0.7 | 56 / 1 / 1 / 4 / 493 | 56 / 1 / 2 / 4 / 723 |
| 0.8 | **61 / 3 / 1 / 6 / 451** | 61 / 4 / 2 / 5 / 688 |

Recall reads 1.00 at 0.8 and 0.98 at 0.6/0.7, but no reference cell is left without a unit: the
reference set holds pairs 7–8 px apart (cells 4/5, 19/21, 25/26, 39/40), and when one unit covers
a pair it is credited to only one of them. `dl_wnd` 8 enlarges footprints by ~50 % and doubles the
fused units with no gain on splits.

What merge 0.7 joins that 0.8 does not (unit ids in the 0.8 run; distances between centroids,
footprint overlap as a share of the smaller, correlation of the final traces): A 6+7 — 5.6 px,
53 %, r 0.59; B 12+13 — 9.1 px, 43 %, r 0.55; C 35+36 — 7.7 px, 41 %, r 0.32; D 49+52+53 —
7.1–10.6 px, 23–56 %, r 0.47–0.69. Cores are ~10 px across at half peak, so units 9–11 px apart
can be neighbours, and C's traces differ enough that merging it would likely fuse two SST cells.

**Claude's recommendation: first merge 0.8, dl_wnd 5 — the frozen values.** 0.7 removes split
cells only partly by fusing real neighbours (C at least); the split cells left at 0.8 are few (3
of 52 reference cells) and, being duplicates of a cell, inflate the unit count rather than mix two
cells' activity; dl_wnd 8 only enlarges footprints and fuses more. VS checks groups A–D in
`CNMFViewer` (the notebook's stage 4 viewer opens on the 0.8 run).

**Stage 4 decided (VS, 2026-10-22): first merge 0.8, dl_wnd 5.** So **the SST parameter set is
the pyramidal frozen set throughout** (VS: "good to know that they're the same parameters in the
end"), with two
SST-specific additions that alter no parameter: line-artefact frames dropped (VS's practice) and
the lens-rim seed mask. Still to do before the queue: lens ellipses for G22, G23, G26 (each shown to
VS); line-artefact detection on every conditioning-day session (not only HC2); a check run on
those three mice's HC2 (G23 should no longer hit the empty-merge failure; watch `pnr_refine`'s
< 128-seed case).

**Under way (2026-10-22, VS agreed all three):**
1. Lens ellipses proposed for G22, G23, G26 (`sst_dio_common.SEED_MASKS_PROPOSED`), drawn over the
   frozen-set `max_proj` and the raw mean, where the lens edge shows
   (`parameter_check/seed_mask/<mouse>_HC2_try3.png`): G22 (340, 315; 265, 270) inside the
   left/bottom-left rim; G23 (250, 250; 300, 225), centred high and narrow — no clear rim, but a
   bright out-of-focus disc on the right is left out and the active cell top right kept; G26 (335,
   300; 275, 295), G27's enlarged for a cell at the left edge. Into `session_queue.SEED_MASKS` once
   VS approves; the HC2 ellipse is drawn on every session of the day to check it fits them all.
2. Every conditioning-day session (36): raw max − mean per session with the ellipse, for VS to see
   lines by eye, and the frame-by-frame detection beside it (`scripts/sst_dio_day_overview.py`,
   `~/sst_dio_out/conditioning_day_overview/`).
3. The decided set on G22, G23, G26 HC2 (`scripts/sst_dio_final_set_check.py`,
   `~/sst_dio_out/decided_set_check/`).

**Results (2026-10-22):**
- Check runs, decided set + proposed ellipse (+ G22's 8 line frames dropped): G22 8 units, **G23
  11 units (no empty-merge failure: the Minian fix holds)**, G26 19 units — every unit on a visible
  cell, none on a rim or the G23 disc; G22's `max_proj` has no lines left. A few faint cells have no
  unit (PNR 1.0, as in the pyramidal runs).
- Each mouse's HC2 ellipse fits every session of its day (`<mouse>_conditioning_day.png`).
- Line artefacts over the whole day: G23, G26, G27 none in any session. G22: HC2 8 frames, LT1 1
  (3905), **TFC1 266**: ~24 isolated line frames (2426–9617) and, from frame 9918 to the end
  (~250 frames, the last ~13 s), every frame corrupted (rows sheared/interleaved) — the connection
  failing outright. **This is the likely reason G22's conditioning is split**: TFC1 died at ~502 s
  and TFC2 is the restart ~2.4 min later (to confirm from the lab notebook).
- G23 `TFC_cond` decodes to 26 118 frames against 26 119 timestamp rows: one frame missing at
  decode — for the loader's timing checks, not for Minian.

**Missed cells (2026-10-27, VS: "G22 is missing some cells"):** cell-like spots found without
judgement in each check run's max projection (`seed_grid.candidate_spots`: local maxima of the
smoothed image inside the ellipse, > 3 s.d.), each uncovered one followed back through the seed
steps (`scripts/sst_dio_missed_cells.py`). G26: 14/16 spots covered (the 2 faintest missed); G23:
10/16 (misses on the image's top edge or faint); **G22: 7/14 — its three clearest missed cells,
with real transients, never got a seed.** Cause: `seeds_init`'s `diff_thres` (3 in the pyramidal
set; Minian's default 2) — a local maximum must exceed its surroundings by that many intensity
values, and G22 (dim, LED 18) has transients of only 2–4 counts. Seeds at diff_thres 3/2/1
(`scripts/sst_dio_diff_thres_check.py`; spots seeded / merged seeds on no spot):

| mouse | 3 | 2 | 1 |
|---|---|---|---|
| G22 | 7/14 / 0 | 12/14 / 4 | 13 / 48 |
| G26 | 13/16 / 4 | 15 / 25 | 15 / 243 |
| G23 | 10/16 / 1 | 11 / 19 | 12 / 744 |
| G27 | 33/34 / 25 | 32 / 258 | 33 / 1975 |

Full G22 run at diff_thres 2: 14 units (8 at 3), spots covered 12/14 (7), **no unit on a non-spot**
(the extra seeds die in CNMF); new units: the three missed cells, plus one ragged unit inside the
out-of-focus debris disc (top left).

**Claude's recommendation: diff_thres 2 for G22 only** — a documented per-mouse exception for its
dim imaging (transients 2–4 counts); G23, G26, G27 keep the pyramidal 3, where 2 gains at most 2
spots for 5–10× the noise seeds. The debris unit: either an exclusion region over the disc's centre
in G22's seed mask (keeping the real cell at its edge, ~(230, 124)), or leave it and list it — VS's
call (removing a unit after the run would be dropping data).

Open for VS before the queue: the three ellipses; G22 TFC1's frames (drop the ~24 isolated ones and
the corrupted ~250-frame end, VS's practice — the end block carries no image — or decide otherwise).

Note (Minian): `pnr_refine` chunks seeds by `len(seeds) // 128`, which is 0 with fewer than 128
seeds — a ZeroDivisionError waiting for a sparse field (G23?). Not hit yet.

Run note: one variant (`ksize_7_wnd_10`) failed under load when ffmpeg returned 1001 frames for one
`.avi` (`dup=2`); the retried task carried on with a half-written `varr` (4857 of 6157 frames
wrong, in 50-frame chunks). Its record said `failed`, but the first stage script measured it anyway;
fixed (only finished variants are measured, failed ones listed by name), the attempt kept in
`failed_attempts/`, re-run alone: raw movie identical to a clean single-process decode, as are all
14 others. Unloaded, ffmpeg decodes that file to the same 1000 frames every time (identical to
`-fps_mode passthrough`). Production is guarded: the runner's YrA replay compares every frame with
the notebook's movie and hard-fails on any difference (all 520 batch sessions passed). Possible
hardening for VS: `-fps_mode passthrough` in Minian's `load_avi_ffmpeg` (the fork).

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
