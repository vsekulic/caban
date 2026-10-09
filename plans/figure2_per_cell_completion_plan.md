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

**Built 2026-09-29** (`BehaviourSession._load_recomputed_YrA`, `session_queue.locate_recomputed_yra`,
`NO_RECOMPUTED_YRA`), reviewed (`/code-review`, four fixes) and tested read-only on MINIRAZER (G10
TFC_cond found; G05 Test_B none; G16 Test_B LT1's two outputs refused). Not yet used for a load: the
rebuild (§2.3) waits for the `max_proj` re-check of every recompute (running on the Razer since
21:15, TFC_cond first) and for the updated sidecars to be copied to MINISCOPE. As built:

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

### 2.2b The max_proj re-check of every recompute (the loader's gate) — status 2026-09-29, 23:58

- 17 TFC_cond: checked on the Razer, **all exact** (0 px differ); records copied to MINISCOPE (old
  versions in `MINISCOPE/_provenance/yra_sidecars_before_max_proj_check_20260929/`).
- 39 others: checked on the Mac against MINISCOPE, all passing (±1 at a few px, the Mac's cv2); copied
  to MINIRAZER (backups in `/mnt/e/_provenance/…`). **The Mac run was stopped by a memory crash of
  osgiliath** (Jetsam 23:46; the re-check peaked at 5.7 GB beside Safari's 12.7 GB) — no more heavy runs
  on the Mac while VS works on it.
- The remaining 72 (+ the 2 NaN-motion G05 sessions, failing as expected, listed in `NO_RECOMPUTED_YRA`):
  running on the Razer overnight (`screen verify_yra`, nice 19). **Next**: copy their records to
  MINISCOPE with `scripts/copy_recheck_sidecars.py`, then VS runs the rebuild and the per-cell cells of
  `run_pipeline.ipynb` (reference kept as `plots/CURRENT-20260929-exported-YrA`; old cache as
  `npy_files/ds_cache-20260512-exported-YrA.pkl`).

**Done 2026-09-30 05:08** (Razer): the remaining 72 re-checked, **111 of 113 non-TFC_cond pass** (the 2
failures are the NaN-motion G05 sessions). Records copied to MINISCOPE (72 updated). **All 128 recompute
sidecars on MINISCOPE carry a passing `max_proj` check**; both drives agree. Ready for §2.3 (VS).

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

**Done 2026-10-02** (VS ran the cells on 2026-10-01; reference `plots/CURRENT-20260929-exported-YrA`, its
`epoch_modulation` from 2026-09-18, `sp_rates_lmm` from 2026-08-24). Before → after:
- **Epoch modulation, YrA (primary)**: group × epoch F(6,16) 0.459, P 0.828 → **0.421, P 0.854**; every contrast
  moves by ≤ 0.02, all Holm P ≥ 0.36; the named Inh-vs-Ctl shock contrast **weakens** −0.097 (P 0.12, Holm 0.24) →
  −0.079 (P 0.18, Holm 0.36). Per-mouse index |change| median 0.005, max 0.029 (G18 shock). Cells 9,515 → 9,529
  (the 14 cells without a YrA trace now carry one; 2 constant-trace cells still excluded).
- **C (confirmatory)**: unchanged (per-mouse |change| ≤ 0.0009, from the 14 cells); shock Inh −0.095, Holm P 0.098.
- **Hierarchical lane (YrA)**: F 0.900, P 0.568 → 0.916, P 0.561; shock Inh −0.097 (Holm 0.46) → −0.079 (Holm 0.63).
- **Event-proximal lane (YrA)**: omnibus F 1.870, P 0.151 → **2.064, P 0.095** — the one lane that moved
  noticeably; still not significant.
- **Cross-validated selectivity (YrA)**: null in both (all exact P ≥ 0.24); estimates shift < 0.01.
- **Sequence test**: unchanged, null (mean ρ +0.040, P 0.27).
- **Figure 2 main (`sp_rates_lmm`)**: amplitude results identical to the last digit (Fig. 2i trace 1.552×, Holm P
  0.0256; post-shock 1.475×, 0.0497; interaction F 0.368, P 0.828). One input changed: G09's post-shock exposure
  −295.7 cell-s (the 13 never-imaged frames × 455 cells / 20 fps), events identical → Inh post-shock rate 0.655 →
  0.653 (P 0.098 → 0.097), rate interaction F 1.544, P 0.237 → 1.533, P 0.240.
- Fig. 2c (sample traces, plots YrA) not yet re-run.

### 2.6 Write it up

`scientific-code-reviewer` and `statistics-checker` on the re-run before anything is written
(VS's rule). Then: [docs/epoch_modulation.md](../docs/epoch_modulation.md) §R rewritten for the new
run (the old §R kept as the superseded run); the per-cell panels and text in
[docs/paper/paper_figure2.md](../docs/paper/paper_figure2.md) (or its supplement); the METHODS
templates (`analysis_methods_templates/epoch_modulation*_methods.md`) stating the recomputed YrA.

**Done 2026-10-02 (reviews and write-up).**
- **Reviews:** both ran on the re-run, read-only.
  - `scientific-code-reviewer` found no defect in the changed code paths.
    - It confirmed the 9,529 cells, both constant-cell exclusions, the C lane and G09's 387-frame
      handling (exposure counted as 387/20 s).
    - The event-proximal move comes from the recomputed YrA itself. Windows, trials and G09 are
      unchanged; the cell set changed by exactly the 14 cells, and mice that gained no cells moved
      too.
  - `statistics-checker`: "the nulls hold" is right as *no detectable effect*, not equivalence.
    The lanes after the mouse-level one are post-hoc companions, and the family size is stated
    (6 omnibus tests, smallest P 0.095). The §R.11.1 wording is corrected. The post-shock
    Exc-vs-Inh event-proximal comparison (exact P 0.019, in no family) is disclosed as exploratory.
- **§2.5 missed changes in `sp_rates_lmm` outside Fig. 2i**, found by diffing every stats file.
  - **The cell-level trace-amplitude lane changed model.** On 2026-08-24 its lbfgs mixed-model
    fit was degenerate (non-finite standard errors), so it fell back to mouse-clustered OLS:
    P 0.0089, Holm 0.027. On 2026-10-01 the same fit of the same 7,410 cells converged and was
    used: **P 0.037, Holm 0.111, not significant**.
  - The same flip moved the within-cell deltas and the threshold sensitivity (0.429/0.435 →
    0.416/0.420). It also moved recall Test B's cell-level post-tone amplitude: BH q 0.013 → 0.445.
  - **The NB model comparison** now nominally favours the interaction model (Δelpd 1.16 ± 4.69;
    before, 1.50 ± 4.51 the other way), within 1 s.e. in both runs.
  - **Recall Test B moved** through the timestamp fix: G21 keeps 2 of 3 tone trials. The 48 h
    amplitude interaction went F 3.02, P 0.079 → 3.19, P 0.070. hM4D's within-group modulation
    0.76 (0.58–1.00), P 0.048, now excludes 1.
- **Written:**
  - [../docs/epoch_modulation.md](../docs/epoch_modulation.md) **Part Y**: new results; the old
    Part R is marked superseded, with a correction note at §R.11.1.
  - Every changed number and sentence in [../docs/paper/paper_figure2.md](../docs/paper/paper_figure2.md)
    (new §6 notes 10–11) and [../docs/paper/paper_figure2_supplement.md](../docs/paper/paper_figure2_supplement.md).
  - METHODS: `epoch_modulation_methods.md` (recomputed YrA) and `sp_rates_lmm_paper_methods.md`
    (G09 window).
- **Open on 2026-10-02:**
  - (a) Fit the cell-level `sp_rates_lmm` lane with `method=['bfgs','cg','powell']`, as the
    hierarchical lanes do, so the branch cannot flip on numerical noise. **Refit check done
    2026-10-02 (VS):** lbfgs/bfgs/cg/powell all give F 4.08, P 0.037, identical coefficients, so
    the new value stands and switching optimizers moves no reported number. **Switched 2026-10-02:**
    `fit_primary_trace_amplitude` and `fit_epoch_delta_model` use `HIERARCHICAL_CELL_LMM_OPTIMIZER`.
    To confirm on VS's next `sp_rates_lmm` run: no "Mixed model unusable" in any stats file, and
    trace omnibus F 4.08, P 0.037. The manipulation check, also cell-level, was left on lbfgs: it
    never fell back. **Confirmed
    2026-10-02 16:44 (VS's re-run):** no fallback in any `sp_rates_lmm` stats file; trace omnibus
    F 4.08, P 0.0369, Holm 0.111; deltas P 0.988 / 0.947; thresholds 0.416 / 0.420 / 0.408; Test B
    post-tone q 0.445; Fig. 2i and the NB model identical. Every number in the paper text stands.
  - (b)–(f): **Moved 2026-10-09 to §6 (C1, C2, B2, C3, B1; B1, the frame rate, has its own plan, [miniscope_frame_rate_plan.md](miniscope_frame_rate_plan.md)).**

### 2.7 Decisions for VS in this phase

1. ~~Which per-cell panels go into Figure 2~~ — **decided (VS, 2026-09-29): compute every per-cell
   analysis and plot it as before; which panels enter the figure is decided at the end, from the
   results.**
2.–4. O.8, O.2 and the frame rate: **Moved 2026-10-09 to §6 (C4, C5, B1).**

### 2.8 The frame-rate correction (20 → 19.76 fps): before/after and reviews (2026-10-09)

Item B1 (§6.B); mechanics in [miniscope_frame_rate_plan.md](miniscope_frame_rate_plan.md). Caches rebuilt
2026-10-09 (`ds_cache.pkl`, 11.5 GB, after the loader memory fix `07bb506`); VS re-ran Sample traces,
`sp_rates_lmm`, Event-locked responsiveness, Epoch modulation, the sequence test and Per-cell activity
distributions. Reference: `plots/CURRENT-20261001-20fps` (the 10-01 run, recomputed YrA at 20 fps).

**Before → after:**

| | before (20 fps) | after (19.76 fps) |
|---|---|---|
| **Fig. 2i post-shock amplitude, hM3D vs mCherry** | **1.475× [1.057, 2.057], raw P 0.0248, Holm 0.0497** | **1.458× [1.048, 2.030], raw P 0.0279, Holm 0.0557** |
| Fig. 2i trace amplitude, hM3D vs mCherry | 1.552×, Holm 0.0256 | 1.552×, Holm 0.0247 |
| amplitude group × epoch interaction | F 0.368, P 0.828 | F 0.392, P 0.811 |
| hM4D post-shock event rate | 0.653×, Holm 0.0968 | 0.652×, Holm 0.0968 |
| rate interaction | F 1.533, P 0.240 | F 1.536, P 0.239 |
| cell-level trace omnibus | F 4.08, P 0.0369, Holm 0.111 | identical |
| Epoch modulation (YrA), group × epoch | F 0.421, P 0.854 | F 0.422, P 0.854 |
| its Inh-vs-Ctl shock contrast | −0.079, Holm 0.362 | −0.080, Holm 0.359 |
| C lane, shock Inh | −0.095, Holm 0.098 | −0.096, Holm 0.096 |
| hierarchical lane (YrA) | F 0.916, P 0.561 | F 0.921, P 0.562 |
| event-proximal, hierarchical companion (YrA) | F 2.064, P 0.095 | F 2.039, P 0.101 |
| event-proximal, mouse-level (YrA) | F 1.801, P 0.162 | F 1.768, P 0.170 |
| sequence test | mean ρ +0.040, P 0.27 | **mean ρ +0.020, P 0.56** |

- **Rates went down by about 1.2 %, which is the correct direction.** A rate is events ÷ (frames ÷ fps): the
  same frames span more seconds at 19.76, so the old rates were 1.2 % too high. (The plan first expected
  "up"; that was wrong.) For example, pre-tone mCherry 0.0822 → 0.0809 events/s/cell. All groups shift alike.
- **Cell counts** in `sp_rates_lmm` are down 0.2–0.6 % (e.g. post-shock 7,609 → 7,585), from the
  abnormal-cell filter's plateau limit (300 → 296 frames) and windows 400 → 395 frames. Mouse counts
  are unchanged.
- **Everything else** moved by 0.1–3 %, with P shifts under 0.02. Event-locked responsiveness G05
  (TFC_cond) post-shock responsive cells 3 → 7 (window change).
- 61 of 210 statistics files are byte-identical (e.g. every amplitude stat in
  `cell_activity_distributions`); only the copied METHODS templates were not rewritten.

**Code review (`scientific-code-reviewer`, 2026-10-09):**
- One bug, fixed: `config.bin_width_frames` truncated (10 s → 197 frames); now rounds (198). The
  current `ds_cache.pkl` still has 197-frame binned rates (`process_binned_sp_rates_mapping`; not
  Figure 2) until the next rebuild.
- Checked and fine: the G05 Test_A re-timing (totals exact, grid inside the recorded span, every
  frame-axis array on the new grid, lazy reloads re-timed, old pickles refused); the frame-rate
  check; the memory release (every dropped field is rebuilt on demand, nothing reads it silently
  as `None`); the isomap factor (exactly 2). No stale `S_spikes`/`S_mov` files exist on disk that
  could bypass the re-timing.
- Noted: re-timed `S` has 1 or 2 recorded frames per new frame (about 25 % get 2), so per-frame
  amplitudes in G05 Test_A alternate; events and peak values there are affected, totals and rates
  are not. Accepted with the re-timed event counts (VS, 2026-10-09; frame-rate plan §5.4).

**Statistics check (`statistics-checker`, 2026-10-09):**
- **Fig. 2i post-shock:** the model and unit are right (mouse-level values, mouse random intercept).
  But significance depends on the Holm family: two comparisons (Exc, Inh) per epoch. Across all six
  epoch × group comparisons, both post-shock and trace fail. The effect is not post-shock-specific:
  hM3D/mCherry is 1.38× [0.99, 1.92] pre-tone, 1.55× trace and 1.46× post-shock, with an interaction
  P of 0.81. Amplitude is averaged over active cells only, which conditions on activity (a design
  caveat). **Recommendation:** report the effect with its CI, not significant/non-significant:
  "1.46× [1.05, 2.03], Holm P = 0.056 (raw 0.028), similar to the trace and pre-tone epochs;
  interaction P = 0.81". Don't call it post-shock-specific. Preferably add a pooled hM3D-vs-mCherry
  contrast across epochs, and a sensitivity table (window 15/20/25 s, plateau limit, minimum events
  per cell).
- **Sequence test:** valid in construction (disjoint halves, mouse as unit), but underpowered and
  unstable. Each half's peak rests on 2–3 trials; argmax latencies flip with small window shifts;
  cells within a mouse are not independent. Mean ρ 95 % CI about [−0.05, +0.09]; effects below about
  0.1 are undetectable. **Recommendation:** report "+0.02 [−0.05, +0.09]; no evidence of a
  reproducible sequence; effects above ~0.1 not excluded", with its sensitivity to the window change.
  Average over many random balanced trial splits (`epoch_sequence.split_trials` uses one fixed
  odd/even split); add a positive control; try a responsive-cell subset.
- **Paper text not yet changed:** [../docs/paper/paper_figure2.md](../docs/paper/paper_figure2.md)
  ~line 126 and note 9 (~line 519) still report post-shock as significant at P = 0.050 (§6.C, C7).

## 3. When the batch finishes (~1 week)

Done or moved: the final sync and the batch finished 2026-10-06/07 (484 of 492; every done session on MINISCOPE); the G05 NaN-motion sessions were recovered on 2026-09-30 (`76013ae`). The failed and excluded sessions: **Moved 2026-10-09 to §6 (A1–A6).**

## 4. What the new sessions make possible — and what it takes

The batch gives `A`/`C`/`S`/`YrA` for every HC, CNO and LT session of every day, but **each session's
cell identities are its own**: nothing longitudinal can use them until they are cross-registered
(local plan §7, risk 4). So every new per-cell question below runs through the cross-registration
work:

**Moved 2026-10-09 to §6 (D1–D3).**

## 5. Suggested order

Superseded: the order is now in §6.F.

## 6. Outstanding items after the batch (2026-10-07) — the to-do list

The one list of what is still open, written after the batch finished (484 of 492 done, 8 failed) and
every done session was synced to MINISCOPE ([../reports/minian_batch_status.md](../reports/minian_batch_status.md)).
It holds this plan's own open items (§2.6 (b)–(f), §2.7, §3, §4, moved here) and what is still open
from [minian_open_items.md](minian_open_items.md) ("item N" below) and
[handover_razer_batch_and_review.md](handover_razer_batch_and_review.md) (its 2026-10-02 update).
Those files keep the detail. Close items here as they are done, with the commit; results that move
Figure 2's numbers are written up in §2.5–2.6 style here too. (Kept as `outstanding_items_plan.md`
from 2026-10-07 to 10-09, then folded in here, VS.)

**Already done, though [minian_open_items.md](minian_open_items.md) still lists them** (E6 updates it):
the Razer track steps 1–5 (the batch, 09-28 → 10-06); item 1, the `max_proj` replay check (`e293ed5`;
128 of 128 pass on both drives, `05062da`); item 2, the G05 NaN-motion recovery (`76013ae`); item 3, the
G21 timing fix (`044b4f1`; caches rebuilt, Figure 2 re-run 10-01/02); item 7, the loader reading the
recomputed YrA (`24fe3a9`); item 12's index part (`0e6657b`); item 14, Tailscale.

### 6.A Failed and excluded batch sessions

| # | item | where | recommendation |
|---|---|---|---|
| A1 | **The two LT1s with a gap mid-recording**: G06 `2021_10_22-TFC_test_A/12_18_46-LT1`, G05 `2021_09_09-TFC_test_A_1wk/16_11_07-LT1` | item 15 | **Checked 2026-10-07: not analysis sessions** — no existing result depends on them. They sit in the loader's `dpath_Test_A_LT1` / `dpath_Test_A_1wk_LT1` tables, but the loader builds no session from those tables, nothing else reads them, and neither ever had Minian output. **Recommendation: park them until D1.** If a grouping of test-day LT1s is chosen, run them on their finalised files (broken files in `session_queue.UNUSED_VIDEOS`; first check that the batch runner honours that table) and pair timestamps around the gap through the §15.1 mapping ([yra_recompute_plan.md](yra_recompute_plan.md) §15.1). |
| A2 | **Three more sessions with unfinished video files**: G07 `2021_10_20-TFC_test_B/16_25_43-HC3`, G11 `2021_12_02-TFC_test_A_1wk/15_32_02-HC2`, G18 `2022_02_07-TFC_cond/14_03_44-HC1` | item 15 | Leave out G07 HC3 (at most ~100 frames). Run G11 HC2 (3,000 frames) and G18 HC1 (4,000 frames) on their finalised files only if an HC analysis needs them; otherwise leave them out and note it. |
| A3 | **Two unfinished recordings**: G07 `2021_10_20-TFC_test_B/15_49_59-TFC_test_B`, G14 `2022_01_18-TFC_test_B_1wk/14_52_35-TFC_test_B_1wk-borked` | item 10; §3 | Check that neither is a session the loader reads. It already reads other sessions for those test days, so these are most likely spares. If so, leave them out. |
| A4 | **G18 `2022_02_07-TFC_cond/14_57_23-CNO2-borked`**: the YrA step fails in `compute_trace` (`ValueError: All arrays must be instances of SparseArray`, after a divide-by-zero warning) | handover, 10-02 update | Debug on the Razer (probably an all-zero footprint), then `mr.resume_after_notebook`, with no notebook re-run. Small. It matters only if per-cell CNO analyses happen (D1). |
| A5 | **The runner's pre-check** catches only sessions where every `.avi` is unfinished | item 15 | Make it flag *any* unfinished file. Cheap, and it stops a repeat on future data. |
| A6 | **The 34 stub sessions and G09 `2021_11_08-TFC_cond/19_28_33-HC3`'s partial output** | item 10 | Leave them out. Look at 3–4 stubs to confirm they are bare timestamps, then close the item. |

### 6.B Correctness of the existing analysis

| # | item | where | recommendation |
|---|---|---|---|
| B1 | **Frame rate: 19.76 fps measured, 20 assumed; G05 `2021_09_03-TFC_test_A/16_34_42-TFC_test_A` recorded at 25** | item 13; §2.6 (f), §2.7 item 4 | **Fix it rather than just document it**: use each session's rate from its timestamps. G05 Test_A's rates are about 19 % low, a real error in one hM3D mouse. The 1.2 % offset elsewhere is harmless for group comparisons, but windows built as seconds × 20 run 20.24 s while the timestamp-derived epochs run 20.0 s. Decide before more write-up, then do a before/after comparison as in the §2.5. **Done 2026-10-09** (code, rebuild, re-run, reviews): §2.8. |
| | **B1 decided (VS, 2026-10-07): one constant, `MINISCOPE_FPS = 19.76`, with a loud ±0.1 % check per session and G05 Test_A as the one listed exception — plan: [miniscope_frame_rate_plan.md](miniscope_frame_rate_plan.md).** (Earlier the same day: fix everything that assumes 20 fps.) Evidence, checked the same day over every `Miniscope/timeStamps.csv` on MINISCOPE (G05–G21, sessions ≥ 2,000 frames): **655 sessions set to 20 fps run at 19.760 fps** (median; 1–99 % 19.754–19.763). **The rate is constant along each recording**: the rate in every 1,000-frame block lies within 19.720–19.767 (1–99 %; the median session's SD over blocks is 0.001 fps), and first vs second half differ by −0.013 to +0.006 fps. Single frame intervals jitter 48–53 ms (ms-quantised PC stamps) around 50.61 ms. Dropped frames are rare: 539 of 655 sessions have none; most of the rest have 1–2 (a 100–150 ms interval), and G09 `TFC_cond` has 19. **Not the PC clock**: the behaviour camera, stamped by the same PC, reads 15.00–15.03 fps (nominal 15) over 140 TFC_cond-day sessions, so the 1.2 % shortfall belongs to the Miniscope sensor. Two sessions are set to 25 fps (24.64 measured): G05 Test_A (known) and G15 `2022_01_11-TFC_cond/18_39_20-HC4` (new). Odd: G18 `14_57_23-CNO2-borked` (304 gaps, rate 27; ties in with A4), G11 `2021_11_27-TFC_test_A/17_11_20-HC2` (17 gaps). Scope: `MINISCOPE_FPS = 20` (`caban/utilities.py:16`) is used 145 times in ~20 modules, plus any literal 20 or frame windows sized at 20 fps (e.g. G09's 400-frame windows). |
| B2 | **Loader guard**: assert `YrA_idx == S_idx` (0 rows moved) before aligning | §2.6 (d) | Do it: about 10 lines, and it turns a quiet misalignment into a loud failure. |
| B3 | **The parked collaborator notice** (`9015c23` on `review/timing-and-yra-fixes`) | handover, 10-02 update | Don't send it yet. Its numbers predate the YrA promotion and the `sp_rates_lmm` model change, so check them and regenerate `behaviour_params` first, after B1. |
| B4 | **The trial-averaged windows are 20 frames (~1 s), meant as 20 s / 2 s** (`population.PERIOD_FRAMES` = 20/20/20/2/20; used by the trial-averaged PCA, `PCA_avg` / `PCA_crossreg_avg`, and isomap's trial-averaged trajectory) | VS, 2026-10-09; [miniscope_frame_rate_plan.md](miniscope_frame_rate_plan.md) §4 | Found during B1. Kept out of B1 (VS: change nothing outside the frame-rate scope); a fix made and reverted the same day (`6451414`, `988416b`). **The fix must use the per-trial trace, not a fixed width.** Post-tone = tone offset → that trial's shock onset, as the epoch analyses already do (`epoch_analysis.py:262-264`). TFC_cond trial 1's tone starts at 185 s and its shock at 220 s, so its trace is 15 s, not 20 s; a fixed 20 s window would take in its shock and the 3 s after. Trials are averaged frame by frame, so trial 1's shorter trace also needs a rule for the last 5 s (e.g. average over the trials that have the frame). Also fix the Methods texts: `population_pca_methods.txt` says post-tone starts at tone offset, but the code starts it 20 frames after onset; `isomap_methods.txt` says no downsampling, but the code halves the rate. Then a before/after check of the trial-averaged PCA and isomap trajectories. |

### 6.C Figure 2's own loose ends (from §2.6–2.7)

| # | item | recommendation |
|---|---|---|
| C1 | (b) The event-proximal omnibus test at ≥ 20,000 draws | Do it before quoting its P (0.095). Run it on the Razer. |
| C2 | (c) G09 sensitivity check with that trial dropped | Do it; it's a cheap robustness check for the text. |
| C3 | (e) Re-run Fig. 2c, the sample traces (it plots YrA) | VS's notebook run. Quick. |
| C4 | O.8: sort panel K on held-out trials | Yes. It's honest by construction, and panel O already does it for its own analysis ([../docs/epoch_modulation.md](../docs/epoch_modulation.md) §O.8). |
| C5 | O.2: split-half stability (Q2) | Report it as supplementary. It's the only remaining answer to the per-neuron question. |
| C6 | Delete the old cache `ds_cache-20260512-exported-YrA.pkl` | After the Figure 2 write-up is final. |
| C7 | **How to report Fig. 2i post-shock** | **Decided (VS, 2026-10-09):** trace stays the single primary comparison. Post-shock is a **hypothesis-driven secondary**, motivated by Puhger et al. 2024 (CA1 activity after the footshock is needed for trace fear memory), reported as effect sizes with CIs, not as significant: hM3D amplitude 1.46× [1.05, 2.03], Holm 0.056 (raw 0.028); hM4D event rate 0.65× [0.43, 1.00], Holm 0.097 (raw 0.048). The amplitude effect is similar across epochs (interaction P 0.81), so it is not called shock-specific. Considered and rejected: post-shock as co-primary (a 4-test Holm family gives trace 0.049, post-shock 0.084); an hM3D-only family, which would pass but was not declared before the data were seen. Next: revise [../docs/paper/paper_figure2.md](../docs/paper/paper_figure2.md) (~lines 125–149 and note 9) with every number from the 19.76 fps run; draft to VS first. |
| C8 | **Sensitivity table for the Fig. 2i amplitude results** (window 15/20/25 s, plateau limit, minimum events per cell) | Recommended by the statistics check (§2.8): the result sits on the threshold, and a 400 → 395-frame change moved it across. |
| C9 | **Sequence test: report as underpowered; optionally make it more stable** | Report "+0.02 [−0.05, +0.09]; effects above ~0.1 not excluded". Optional: many random trial splits instead of one odd/even split; a positive control; a responsive-cell subset (§2.8). |

### 6.D New sessions and cross-registration ([crossreg_batch_runner_plan.md](crossreg_batch_runner_plan.md); §4)

| # | item | recommendation |
|---|---|---|
| D1 | Which new groupings (item 4; crossreg plan §6) | Keep groupings 1–7 exactly as published. Add the whole TFC_cond day (HC1, LT1, CNO1, CNO2, LT2, HC2, TFC_cond, HC3) if per-cell before/after CNO is wanted, which is the most valuable new question. |
| D2 | Groupings registry (item 5; crossreg plan §3) | Build it now; it needs only folder names. Explain first why grouping 1's `mappings_crossreg1.pkl` exists in one mouse only. |
| D3 | Cross-registration runner and its check (crossreg plan §5), then the loader reading the registry (§4) | After D2. Check it by re-deriving one published grouping and comparing it with production. |

### 6.E Infrastructure and housekeeping

| # | item | recommendation |
|---|---|---|
| E1 | Item 11: a second copy of MINISCOPE | Wanted. MINIRAZER now holds the raw data and all results, but both drives sit in one flat, and MINISCOPE is 90 % full. Get one off-site or at the lab. |
| E2 | Item 6: gate-run links and scratch on FUTROLA (G06 `09_52_24-HC1`, 6.6 GB; G10 `16_32_14-TFC_cond`, 25 GB) | Remove them; both checks are recorded. Confirm on screen where the live copies are first. |
| E3 | Item 8: the notebook's videos drop one frame in five | Leave the protected notebook alone. The videos are only a visual check, so document it in the Methods text (E4). |
| E4 | Item 9: Methods text for the Minian procedure (`analysis_methods_templates/`) | Write it now: about 1 h, unblocked. |
| E5 | Item 12: saved output in `notebooks/recompute_yra.ipynb` | Clear it; every report is also in each session's `YrA_recompute.json`. |
| E6 | Bring [minian_open_items.md](minian_open_items.md) up to date | Mark the done items with their commits, point to this plan, and add A4 and A5. |
| E7 | **Move the local analysis cache off the Mac, to FUTROLA** (VS, 2026-10-09: MINISCOPE stays a mostly read-only snapshot of the data; caches go on FUTROLA). `npy_files` is 105 GB (incl. `ds_cache.pkl` 11.5 GB); the Mac has 89 GB free. `NPY_SAVE_PATH` derives from `utilities.POSIX_DATA_ROOT` (`caban/utilities.py:41-46`). | Copy to FUTROLA, verify file for file, point `NPY_SAVE_PATH` there, load `ds` once from it; delete the Mac copy only after VS confirms. Supersedes [local_minian_pipeline_plan.md](local_minian_pipeline_plan.md) §5.1c–5.1d. If FUTROLA is unplugged, loading fails loudly. **Copied and verified 2026-10-10** (rsync, 10.6 min; checksum pass: no differences; 4,083 files) to `/Volumes/FUTROLA/caban_cache/npy_files`; `utilities.NPY_SAVE_PATH` points there, with `require_cache_volume` raising if FUTROLA is not mounted (in `Saver` and `load_all_mice`). **Done 2026-10-10:** VS loaded `ds` from FUTROLA; the Mac copy was deleted after a final no-change check. Plots stay on the Mac's disk (VS). |
| E8 | The two optional runner-made review videos (`minian_raw_traces.mp4`, `minian_preprocessing.mp4`; [minian_batch_runner_plan.md](minian_batch_runner_plan.md) §7) | Not built. Optional: the notebook's own videos exist. Park unless wanted. |

### 6.F Suggested order

B1 → C1–C3 and B2 (the Figure 2 loose ends) → D2–D3, with E4 and E6 in the gaps; A1 waits for D1.
(A1 was first until 2026-10-07, when it turned out to touch no existing analysis.) The decisions on B1
and D1 are VS's; once they are made, the code and the runs on the Razer are Claude's.
