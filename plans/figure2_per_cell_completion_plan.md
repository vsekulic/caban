# Plan: finish Figure 2's per-cell block on the corrected YrA — and what the batch adds after

Status: **planning**, written 2026-09-29 while the Minian batch runs on the Razer (~1 week left;
[reports/minian_batch_status.md](../reports/minian_batch_status.md)). Nothing executed.
Depends on: [docs/epoch_modulation.md](../docs/epoch_modulation.md) (the per-cell analysis and its
last results, §R), [docs/paper/paper_figure2.md](../docs/paper/paper_figure2.md) (the figure text),
[yra_recompute_plan.md](yra_recompute_plan.md) (§7.3, §9, §11.3, §14–15),
[handover_razer_batch_and_review.md](handover_razer_batch_and_review.md) §6 (the parked review
branch), [minian_open_items.md](minian_open_items.md) (items 1–7, 13),
[crossreg_batch_runner_plan.md](crossreg_batch_runner_plan.md).

## 1. The key point: Figure 2's per-cell block does not wait for the batch

Figure 2's per-cell block — `caban.epoch_modulation`: panels K (tone-aligned heatmaps), L (modulation
index by epoch × group), the hierarchical, event-proximal and cross-validated-selectivity companions —
is computed on the **17 TFC_cond sessions** only. Those are production sessions: they already have
`A`/`C`/`S`, and all 17 already have a recomputed `YrA` (`YrA_recomputed.zarr` in their
`minian_crossreg*` folder; 128 of the 130 production sessions, the 2 missing are G05 TFC_test_B
recall sessions).

What the block **does** depend on is **YrA, its primary signal** (§A.4 there), and every published
epoch-modulation number (§R, run 2026-08-26/27) still comes from the **old** `YrA` export, which is
known to be wrong in two ways ([yra_recompute_plan.md](yra_recompute_plan.md) §1, §12.4): row-sheared
against `S`/`C` (repaired in memory since 2026-09-21), and computed from a different run's `C`
(before the second temporal update, in a run whose unit set differs by 1–4 cells in 8 mice). The
recompute is the correct residual of *these* footprints against *these* traces. **Switching Figure 2
to it is the reason the YrA work was done, and it can be done now**, on the Mac, while the Razer runs
the batch.

What the batch adds is different: the HC, CNO and LT sessions of every day, which make **new**
per-cell questions answerable (§4) — none of which is in Figure 2 today.

## 2. Now, while the batch runs (Mac; MINISCOPE needed for step 2.3 only)

### 2.1 Review the parked branch `review/timing-and-yra-fixes` (handover §6)

Before anything is rebuilt, because two of its commits change what the rebuild loads.

| commit | what | for Figure 2 |
|---|---|---|
| `96137b9` | `max_proj` replay check for every YrA recompute; `verify_all` | **needed**: a second, independent proof of the replay for the 17 TFC_cond recomputes. So far each rests on agreement with the old export at the units whose footprints overlap nothing — a handful per session (13 in G10, 5 in G05 LT1). Run it on the 17 before using them. |
| `5c3a1bd` | loader maps timestamp rows to `C` frames through the imaged rows; G21 Test_B tone 3 excluded; `write_behaviour_params` from timestamps; old pickles refuse to load | **needed**: TFC_cond epochs are the declared times snapped to timestamps (§A.3 there), so row *i* must be frame *i*. Of the 17 TFC_cond sessions only **G09** changes (timestamps after imaging stopped, clamped). Forces the `ds_cache.pkl` rebuild. |
| `d2b74ce` | re-estimate motion for the G05 NaN-motion sessions | recall only (G05 TFC_test_B); not Figure 2 |
| `7fcac0f` | "Run All `recompute_yra.ipynb` on the Mac" | **reject** (would write MINISCOPE outside the sync) |
| `9015c23` | collaborator notice on behaviour timing + G21 comparison | a draft for other people: verify every number; confirm nothing was sent |

Each by `/code-review` on its diff; `5c3a1bd` and `96137b9` also by `scientific-code-reviewer` and
`statistics-checker` (VS's rule before any result is written up). Cherry-pick what passes onto
`feat/yra-unit-alignment`.

**Add to the timing review — found 2026-09-29:** `caban/utilities.py:16` `MINISCOPE_FPS = 20`
converts frames to seconds in the event-rate denominators (`utilities.py:348, 364, 399, 460, 494`),
while the camera runs at **19.76 fps** (open item 13). Every absolute rate (events s⁻¹) is then off
by the same ~1.2 % in every group: ratios, contrasts and *P* values are unaffected, the absolute
differences quoted in Fig. 2 (e.g. −0.039 events s⁻¹ per cell) are not. Decide: fix (seconds from
timestamps, as `5c3a1bd` does for `write_behaviour_params`) and restate those numbers, or document.
Also G05 `2021_09_03-TFC_test_A` at 24.6 fps (recall; not Figure 2).

### 2.2 Promote the recomputed YrA in the loader (open item 7; yra plan §11.3)

Design, to be agreed before coding:
- `sessions.py:739` reads `YrA_recomputed.zarr` from the session's output folder where it exists
  (all 17 TFC_cond), and **hard-fails** for a session whose output folder has a recompute sidecar
  that is incomplete — no silent fallback to the old export.
- For sessions with no recompute (the 2 G05 NaN-motion recall sessions, until item 2): an explicit,
  listed decision — old export, or no YrA — not a fallback.
- `_align_YrA_to_S_units` stays as a guard; on recomputed YrA it must report **0 rows moved, 0
  NaN rows, 0 YrA-only** for every session (yra plan §7.2) — a hard check in the run.
- The per-session YrA caches (`npy_files/<type>/YrA/*_YrA_{full,idx}.pkl`) are invalidated (moved
  aside, not deleted), so the rebuild writes new ones.
- `zarr.load` → `zarr.open_group` (yra plan §8; zarr 3 cannot `load` a group).

### 2.3 Rebuild the caches, offline-capable

A fresh `load_all_mice(use_cache=False)` (yra plan §9; VS prefers a clean full reload), reading the
recomputed YrA **from MINISCOPE** (the Mac's copy; the 17 TFC_cond recomputes are there since
2026-09-27). The old `ds_cache.pkl` (9.3 GB) is **moved aside, not deleted** — it is the reference
for §2.5 (a Time Machine snapshot of 2026-09-27 also exists). Afterwards the analyses run with
MINISCOPE unplugged, as required (local plan §5.1d). Expect every S/C-based number to move at
≤ 1.4 × 10⁻⁴ relative (raw vs cached S/C, epoch_modulation §O.5) — to be measured, not assumed.

### 2.4 Re-run the per-cell block

`run_epoch_modulation` with every lane on (mouse-level, hierarchical, event-proximal, cell
selectivity — the last **never had its results reported**, epoch_modulation header), plus
`run_epoch_sequence`; both signals. Expected, and checked in the run:
- the **14 cells** previously without a YrA trace now carry one — analysed cells 9,515 → up to 9,531
  (`dropped['YrA']['missing'] == []`);
- the C lane moves only by the cache rebuild (§2.3).

### 2.5 Quantify what moved (yra plan §7.3)

Against the 2026-08-27 reference (`PLOTS_DIR/CURRENT/epoch_modulation/`, kept): per-mouse
modulation index per epoch, every contrast, both omnibus tests, the signal comparison. Report the
differences, not only the new values. The prior results were a null with one named, not-claimed
contrast (Inh vs Ctl at shock); whether that changes is the question to answer first. Also re-run
`sp_rates_lmm` (Figure 2 main panels, on `S`) to confirm its numbers are unchanged beyond the cache
rebuild, G09's clamp and the frame-rate decision of §2.1.

### 2.6 Write it up

`scientific-code-reviewer` and `statistics-checker` on the re-run before anything is written
(VS's rule). Then: [docs/epoch_modulation.md](../docs/epoch_modulation.md) §R rewritten for the new
run (the old §R kept as the superseded run); the per-cell panels and text in
[docs/paper/paper_figure2.md](../docs/paper/paper_figure2.md) (or its supplement); the METHODS
templates (`analysis_methods_templates/epoch_modulation*_methods.md`) stating the recomputed YrA.

### 2.7 Decisions for VS in this phase

1. **Which per-cell panels go into Figure 2** — K, L, the selectivity panels (O/P), the hierarchical
   panel; and whether "per-cell responses" also means `event_locked_responsiveness`.
2. **O.8**: sort panel K on held-out trials (honest by construction) — a change to a paper-facing
   panel.
3. **O.2**: report the split-half stability (Q2), or not.
4. **Frame rate** (§2.1): fix the 20 fps constant or document it.

## 3. When the batch finishes (~1 week)

1. Final sync to MINISCOPE; the status report shows every batch session `done` or explained.
2. Decide the failed and excluded sessions (open items 10): the two unreadable recordings (G07
   TFC_test_B, G14 TFC_test_B_1wk-borked — salvage the readable files as for G21, or leave out), the
   34 stubs, G09 HC3's partial output.
3. The two G05 NaN-motion recall sessions (open item 2, `d2b74ce`), if recall YrA is wanted.

## 4. What the new sessions make possible — and what it takes

The batch gives `A`/`C`/`S`/`YrA` for every HC, CNO and LT session of every day, but **each session's
cell identities are its own**: nothing longitudinal can use them until they are cross-registered
(local plan §7, risk 4). So every new per-cell question below runs through the cross-registration
work:

1. **Which groupings (VS; open item 4, crossreg plan §6).** Candidates: the whole TFC_cond day (HC1,
   LT1, CNO1, CNO2, LT2, HC2, TFC_cond, HC3) for **per-cell before/after CNO** — a within-cell
   manipulation check richer than Fig. 2d's LT1 → LT2 pair; LT1 across track days (place-cell
   stability); HC with the same day's session. Recommended: leave groupings 1–7 exactly as published.
2. **The groupings registry** (item 5) — can be built **now**, from folder names.
3. **The cross-registration runner** (crossreg plan §5) and its gate (re-derive one existing grouping,
   compare with production) — can be built and gated **now**, on production sessions.
4. **The loader reads the registry** (crossreg plan §4), since new output is `minian/`, not
   `minian_crossreg*`.
5. Then the new analyses, each with its METHODS template, and a decision on which figure they belong
   in.

## 5. Suggested order

While the batch runs: **2.1 → 2.2 → 2.3 → 2.4 → 2.5 → 2.6** (Figure 2's per-cell block, ~2–4 days
of work with reviews), and in the gaps 4.2–4.3 (registry, crossreg runner). After the batch: §3,
then 4.1 (VS) → 4.4 → 4.5.
