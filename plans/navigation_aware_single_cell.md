# Navigation-aware single cell analyses

## Context

Group differences (hM3D / mCherry / hM4D) are visible in the whole-session spike rate and
activity panels produced by `run_sp_rates`. Those numbers average over **every** cell in a
mapping and over **every** frame of the recording, so a group that simply navigates more will
show a higher population rate for purely behavioural reasons — more running means more
place-cell recruitment and fewer immobile frames diluting the average. The current panels
cannot distinguish that from a cell-intrinsic DREADD effect.

The goal is to decompose the existing whole-session numbers along two axes — **cell class**
(place cell vs non-place cell) and **frame class** (movement vs immobility) — and to test the
locomotion covariate directly, so the big-picture whole-session panels can be presented
alongside a navigation-aware drill-down that settles the question.

### What already exists (verified — do not rebuild)

- **Place fields are already fit for all five chamber sessions.** `run_pf_and_loc`
  ([caban/sections.py:1362-1401](caban/sections.py#L1362-L1401)) runs `plot_fluorescence_map`
  on `TFC_cond`, `Test_B`, `Test_B_1wk`, `Test_A`, `Test_A_1wk` with identical parameters
  (`bin_width=34, max_fields=45, only_fm_pcells=True, merge_distance=4`). Cached fits exist on
  disk for every mouse in all five, e.g.
  `$NPY_SAVE_PATH/Test_A/Test_A-G05-318cells-PlaceFields.npz`,
  `$NPY_SAVE_PATH/Test_A_1wk/Test_A_1wk-G05-467cells-PlaceFields.npz`.
  **No extension of the place-field fitting is required.**
- LT place fields are cached as whole-`FluorescenceMap` pickles in
  `$NPY_SAVE_PATH/fm_TFC_cond_LT1/` and `fm_TFC_cond_LT2/` (17 mice each), reloadable via
  `load_fm` ([caban/analysis.py:4744](caban/analysis.py#L4744)).
- `run_occupancy_analysis` ([caban/sections.py:1460](caban/sections.py#L1460)) already computes
  per-mouse coverage / entropy / immobility for all seven sessions, but **not** distance or
  speed, and never regresses them against the rates.

### Constraints from the user

- Everything **additive**. Do not modify or remove any existing plot, and do not touch the
  existing place-field code (`plot_pf_analyses`, `plot_fluorescence_map`, `caban/spatial.py`)
  beyond importing from it.
- The pooled-cell / KS-test pseudoreplication in `plot_pf_analyses` is **intended**. Leave it
  alone; add a clearly-separate per-mouse companion that demonstrates what the same metrics
  look like with n = mice.
- Every new output goes under one new parent folder,
  `PLOTS_DIR/navigation_aware_single_cell/`, split into clearly-named sub-analyses so
  provenance is obvious.

---

## Scope

| Axis | Values |
|---|---|
| Sessions | `TFC_cond`, `Test_A`, `Test_A_1wk`, `Test_B`, `Test_B_1wk`, `LT1`, `LT2` |
| Window | `whole_session` (all 7); `pretone_180s` (5 chamber sessions only — LT has no tones) |
| Frame class | `all_frames`, `movement_only`, `immobility_only` |
| Cell class | `place`, `non_place`, `all` |
| Metric | spike rate (`want_peakval=False`), activity (`want_peakval=True`) |
| Mapping | `'full'` by default (parameterised — see "Panel-count control") |

---

## Implementation

### 1. Mask-based rate helpers — `caban/utilities.py` (additive)

The existing `get_avg_sp_rate_in_period` / `get_avg_activity_in_period`
([caban/utilities.py:235-271](caban/utilities.py#L235-L271)) take a **contiguous** `[beg, end]`
window and divide by `(end-beg)/MINISCOPE_FPS`. Movement/immobility frames are not contiguous,
so add two parallel functions immediately below them:

```python
def get_avg_sp_rate_in_frame_mask(f_spikes, frame_mask): ...
def get_avg_activity_in_frame_mask(S_spikes, S_peakval, frame_mask): ...
```

- `frame_mask` is a boolean array over frames of the trimmed `sess.S`.
- Duration = `frame_mask.sum() / MINISCOPE_FPS`.
- Average over **all** cells in `f_spikes`, including silent ones — matching the deliberate
  convention documented at [caban/sessions.py:1038-1042](caban/sessions.py#L1038-L1042).
- Hard-fail (raise) if `frame_mask.sum() < MINISCOPE_FPS` (under 1 s of eligible time) or if
  `f_spikes` is empty. No silent skips, per CLAUDE.md.

Leave the two existing functions untouched.

### 2. New module — `caban/place_cell_rates.py`

**Place-field source — standalone, no dependency on `run_pf_and_loc` having run.**
`PlaceFields(to_pickle=True, sess=sess, mouse=mouse)`
([caban/spatial.py:150-193](caban/spatial.py#L150-L193)) reconstitutes `pf_size`,
`compactness_pf`, `spatial_selectivity`, `merged_means` and the rest **directly from the
`*-PlaceFields.npz` on disk**, deriving the path from `sess.savepath`, `sess.session_type` and
`sess.S.shape[0]`. Those files already exist for all seven sessions and every mouse
(`Test_A/Test_A-G05-318cells-PlaceFields.npz`, `TFC_cond/LT1-G05-292cells-PlaceFields.npz`, …).

```python
def load_place_fields(sess, mouse):
    """Prefer sess.fm.pf if already in memory; else load from the cached npz."""
```
Use `sess.fm.pf` when present, otherwise construct `PlaceFields(to_pickle=True, …)` and check
its `.loaded` flag; raise naming `run_pf_and_loc` / `run_LT_pfs` if `.loaded` is False. This is
the existing, sanctioned load path — no new file-format code — and it means the new section can
sit anywhere in the notebook regardless of whether `run_pf_and_loc` has been run in this
kernel. `load_fm` / `save_fm` are therefore **not** needed.

**Cell partition.** PF dicts (`fm.pf.pf_size`, `compactness_pf`, `spatial_selectivity`) are
keyed by **position index into `S`**, and `get_S_mapping` returns `S_spikes` keyed by the same
row indices — the convention `plot_pf_analyses` relies on at
[caban/analysis.py:4806](caban/analysis.py#L4806). Bridge through the sanctioned helpers rather
than reimplementing: `spatial._get_pf_cells(fm)` (returns unit IDs,
[caban/spatial.py:3669](caban/spatial.py#L3669)) then `spatial._pf_pos(fm, uid)`
([caban/spatial.py:3649](caban/spatial.py#L3649)) back to row indices.

```python
def partition_place_cells(sess, mapping):
    """-> {'place': [row idx], 'non_place': [row idx], 'n_place': int, 'n_total': int}"""
```
Raise a clear error if `sess.fm` or `sess.fm.pf` is missing, naming the section to run first.

**Frame masks.**
- `whole_session` → all-True over `sess.S.shape[1]`.
- `pretone_180s` → reuse `spatial._build_pretone_mask(sess, first_n_sec=180.0)`
  ([caban/spatial.py:768](caban/spatial.py#L768)), which already prefers `tone_onsets[0]` and
  falls back to 180 s. Reusing it makes these panels line up exactly with the occupancy and
  2D spatial-information suites.
- `movement_only` / `immobility_only` → `sess.velocities_miniscope_smooth[:T] >= 2.0` (and
  `< 2.0`), the identical expression used by `_compute_velocity_masked_activity`
  ([caban/sessions.py:426-431](caban/sessions.py#L426-L431)); threshold from
  `utilities.VELOCITY_THRESHOLD`. Combine with the window mask by logical AND.

**Rate computation.**
```python
def rates_by_cell_class(sess, mapping, *, want_peakval=False,
                        window='whole_session', frame_class='all_frames'):
    """-> {'place': float, 'non_place': float, 'all': float, 'n_place': int, 'n_total': int}"""
```
Calls `sess.get_S_mapping(mapping, want_peakval=want_peakval)` — reuse, do not reimplement —
then subsets `S_spikes` / `S_peakval` to each cell class and calls the new mask helpers.
`'all'` reproduces the existing whole-session number when
`window='whole_session', frame_class='all_frames'`, which is the sanity check in §5.

**Per-mouse PF properties.**
```python
def pf_properties_per_mouse(sess, mapping):
    """-> {'pct_place_cells', 'mean_n_pfs', 'mean_pf_size', 'mean_compactness',
           'mean_spatial_selectivity'}  — per-cell values averaged within the mouse."""
```
Reads the same `fm.pf` dicts `plot_pf_analyses` reads, but collapses to one value per mouse.
No `G07`/`G15` skip — include every mouse the session dict actually contains.

**Plotting** (all reuse `_draw_violin_triplet` [caban/analysis.py:852](caban/analysis.py#L852)
and `do_anova1_plot` [caban/analysis.py:574](caban/analysis.py#L574), n = mice, ANOVA-gated
Tukey — the same machinery as `plot_whole_session_sp_rates`
[caban/analysis.py:194](caban/analysis.py#L194)):

- `plot_place_cell_rate_split(...)` — 3 panels (place / non-place / all) sharing a row.
- `plot_place_cell_proportion(...)` — %-place-cells per mouse.
- `plot_pf_properties_per_mouse(...)` — 4 panels (n PFs, PF size, compactness, spatial
  selectivity), the explicit "same metrics without pseudoreplication" companion to
  `plot_pf_analyses`.

### 3. New module — `caban/locomotion.py`

```python
def compute_locomotion_metrics(sess, *, window='whole_session', speed_thresh=VELOCITY_THRESHOLD):
    """-> {'distance_cm', 'mean_speed_cms', 'mean_speed_moving_cms',
           'pct_time_moving', 'duration_s'}"""
```
Reads `velocities_miniscope_smooth` directly — the same source `OccupancyAnalysis` uses
([caban/analysis.py:12546-12548](caban/analysis.py#L12546-L12548)) — so `OccupancyAnalysis`
itself is left untouched. `distance_cm = sum(vel) * dt`; `dt = MINISCOPE_FRAME_MS / 1000`.

- `plot_locomotion_group_comparison(...)` — the "show there is no group difference" figure:
  4-metric violin triplet across groups per session, n = mice.
- `plot_rate_vs_locomotion(...)` — per-mouse scatter of whole-session rate against distance and
  mean speed, coloured by group with per-group regression lines, plus **ANCOVA**
  (`rate ~ locomotion + C(group)`, statsmodels OLS) reporting the group effect *after*
  adjusting for locomotion, and the group-partialled correlation. Stats printed into the
  figure and written to a `.txt` beside it.

### 4. Wiring — `caban/sections.py`, `caban/config.py`, `run_pipeline.ipynb`

**Four sections, not one monolith** — following the existing one-`run_*`-per-`cfg.plot_*`
convention, so each can be re-run independently during iteration:

| Section function | `cfg` flag | Writes to |
|---|---|---|
| `run_locomotion_comparison(ds, cfg)` | `plot_locomotion_comparison` | `navigation_aware_single_cell/locomotion_metrics/` |
| `run_place_cell_properties(ds, cfg)` | `plot_place_cell_properties` | `place_cell_proportions/`, `place_field_properties_per_mouse/` |
| `run_place_cell_rates(ds, cfg, mappings_per_session=None)` | `plot_place_cell_rates` | `navigation_aware_single_cell/place_cell_rates/` |
| `run_rate_vs_locomotion(ds, cfg)` | `plot_rate_vs_locomotion` | `navigation_aware_single_cell/rate_vs_locomotion/` |

Flags added to `PipelineConfig` near
[caban/config.py:79-86](caban/config.py#L79-L86), all defaulting to `True`. Each section follows
the house conventions: `if not cfg.<flag>: return`, a `ds`-attribute unpacking block, and
`msg_start` / `msg_end`.

**No ordering dependency.** Because place fields are read from the cached
`*-PlaceFields.npz` (§2), these sections do **not** require `run_pf_and_loc` or `run_LT_pfs` to
have run earlier in the kernel — they only need `ds`. They can therefore sit before
`## Single-unit responses`, where those two sections currently live much further down (notebook
cells 48 and 53).

**Notebook placement** — one new H1 section inserted between the current cell 27 (end of
`# Analysis sections`) and cell 28 (`## Single-unit responses`), with an individual code cell
per run:

```
# Navigation-aware single cell analyses               (markdown, H1)

### Locomotion group comparison                       (markdown, H3)
run_locomotion_comparison(ds, cfg)

### Place-cell proportions and PF properties (per mouse)
run_place_cell_properties(ds, cfg)

### Place-cell / movement-restricted spike rates
run_place_cell_rates(ds, cfg)

### Spike rate vs locomotion (ANCOVA)
run_rate_vs_locomotion(ds, cfg)
```

This reads naturally after the `run_sp_rates` / `run_binned_sp_rates` cells it is meant to
qualify. Note the existing `## Single-unit responses` heading has drifted — it currently spans
cells 28-104, including `run_pf_and_loc`, `run_occupancy_analysis`, `run_LT_pfs` and the whole
decoder suite. The new section is inserted **before** it and does not disturb it; splitting that
heading up is explicitly out of scope for now.

**Mouse exclusions.** Build per-group arrays from the mice actually present in each session
dict, exactly as `plot_whole_session_sp_rates` does
([caban/analysis.py:211-215](caban/analysis.py#L211-L215)) — `Test_B` lacks G07, `Test_A_1wk`
and `Test_B_1wk` lack G15.

### 5. Output layout — everything under one new parent directory

All new output goes under a **single** new top-level directory,
`PLOTS_DIR/navigation_aware_single_cell/`. No existing directory is written to.

```
PLOTS_DIR/
├── sp_rates/                 (existing — untouched)
├── place_fields_<session>/   (existing — untouched)
├── fluorescence_maps_*/      (existing — untouched)
├── occupancy_*/ , fm_pf/     (existing — untouched)
└── navigation_aware_single_cell/               ← ALL new output
    ├── place_cell_rates/
    │   ├── place_cell_restricted_rates_methods.txt
    │   ├── place_cell_rates_summary.csv                    ← tidy per-mouse table, every cell of the grid
    │   ├── TFC_cond/                                        (spike rate)
    │   │   ├── whole_session/{all_frames,movement_only,immobility_only}/
    │   │   │        TFC_cond_place_cell_rate_split-full.png|.svg
    │   │   └── pretone_180s/{all_frames,movement_only,immobility_only}/  …
    │   ├── TFC_cond-activity/                               (peak-S activity, same subtree)
    │   ├── Test_A/ Test_A-activity/ Test_A_1wk/ Test_A_1wk-activity/
    │   ├── Test_B/ Test_B-activity/ Test_B_1wk/ Test_B_1wk-activity/
    │   └── LT1/ LT1-activity/ LT2/ LT2-activity/            (whole_session only — no tones)
    │
    ├── place_cell_proportions/
    │   ├── place_cell_proportion_methods.txt
    │   ├── place_cell_proportion-<session>-full.png|.svg            (×7)
    │   └── place_cell_proportion-all_sessions-full.png|.svg         (across-session summary)
    │
    ├── place_field_properties_per_mouse/
    │   ├── place_field_properties_per_mouse_methods.txt
    │   └── <session>/  num_pfs-<session>-full.png|.svg
    │                   pf_size-<session>-full.png|.svg
    │                   pf_compactness-<session>-full.png|.svg
    │                   spatial_selectivity-<session>-full.png|.svg
    │                   pf_properties_4panel-<session>-full.png|.svg
    │
    ├── locomotion_metrics/
    │   ├── locomotion_group_comparison_methods.txt
    │   ├── whole_session/  locomotion_group_comparison-<session>.png|.svg   (×7)
    │   │                   locomotion_summary.csv, locomotion_group_stats.txt
    │   └── pretone_180s/   same, ×5 chamber sessions
    │
    └── rate_vs_locomotion/
        ├── rate_locomotion_ancova_methods.txt
        └── <session>[-activity]/<window>/  rate_vs_distance-<session>-full.png|.svg
                                            rate_vs_mean_speed-<session>-full.png|.svg
                                            ancova-<session>-full.txt
```

Filename stems under `place_field_properties_per_mouse/` deliberately mirror those
`plot_pf_analyses` writes into `place_fields_<session>/` (`num_pfs`, `pf_size`,
`pf_compactness`, `spatial_selectivity`) so the pooled-cell and per-mouse versions can be viewed
side by side — but they live in a separate tree and the originals are never touched.

**Panel count.** The full grid is produced: 5 chambers × 2 windows × 3 frame classes × 2 metrics
= 60, plus LT1/LT2 × 1 window × 3 frame classes × 2 metrics = 12, for **72** three-panel figures
(PNG + SVG each). `mappings_per_session` defaults to `['full']` for every session; pass the
`ds.mappings_all_*` lists to widen it. Every value behind every panel is also written once to
`place_cell_rates_summary.csv` so nothing has to be recomputed for downstream stats.

### 6. METHODS templates

Per CLAUDE.md, add to `analysis_methods_templates/` and copy at runtime with
`_copy_analysis_methods_template` (imported from `caban.decoder`,
[caban/decoder.py:685](caban/decoder.py#L685) — the new modules are leaves, so no import cycle):

- `place_cell_restricted_rates_methods.txt`
- `place_cell_proportion_methods.txt`
- `place_field_properties_per_mouse_methods.txt` — must state explicitly that this is the
  per-mouse (n = mice) companion to the pooled-cell KS analysis in `place_fields_*`, and that
  the pooled version is retained deliberately.
- `locomotion_group_comparison_methods.txt`
- `rate_locomotion_ancova_methods.txt`

`place_cell_restricted_rates_methods.txt` must note that LT place fields are fit at
`bin_width=4.5` px versus `34` px for the chambers
([caban/sections.py:1610](caban/sections.py#L1610) vs
[1365](caban/sections.py#L1365)), so **absolute** %-place-cells is not comparable between LT and
chamber sessions — only the within-session group comparison is.

---

## Files touched

| File | Change |
|---|---|
| `caban/utilities.py` | +2 mask-based rate functions (additive) |
| `caban/place_cell_rates.py` | **new** |
| `caban/locomotion.py` | **new** |
| `caban/sections.py` | +4 `run_*` sections, + imports |
| `caban/config.py` | +4 `plot_*` flags |
| `run_pipeline.ipynb` | +1 H1 markdown header, +4 H3 headers, +4 code cells, inserted between cells 27 and 28 |
| `analysis_methods_templates/` | +5 templates |

Not modified: `caban/spatial.py`, `caban/sessions.py`, `plot_pf_analyses`,
`plot_fluorescence_map`, `plot_whole_session_sp_rates`, `OccupancyAnalysis`, `run_pf_and_loc`,
`run_sp_rates`, `run_occupancy_analysis`.

---

## Verification

1. **Reload + run in the live session** (the `ds` cache plus the cached `*-PlaceFields.npz` hold
   everything needed; no loader rebuild, no re-fitting, no raw-file access):
   ```python
   import importlib
   import caban.utilities, caban.place_cell_rates, caban.locomotion, caban.sections
   importlib.reload(caban.utilities)
   importlib.reload(caban.place_cell_rates)
   importlib.reload(caban.locomotion)
   importlib.reload(caban.sections)
   from caban.sections import (run_locomotion_comparison, run_place_cell_properties,
                               run_place_cell_rates, run_rate_vs_locomotion)
   run_locomotion_comparison(ds, cfg)
   run_place_cell_properties(ds, cfg)
   run_place_cell_rates(ds, cfg)
   run_rate_vs_locomotion(ds, cfg)
   ```

2. **Standalone-load check.** In a fresh kernel where `run_pf_and_loc` has *not* been run,
   confirm `run_place_cell_properties(ds, cfg)` still succeeds — i.e. `PlaceFields.loaded` is
   `True` for all 7 sessions × all mice straight off the npz caches. This is what makes the
   notebook placement before `## Single-unit responses` valid.

3. **Decomposition sanity check (the key correctness test).** For each session and mapping,
   `rates_by_cell_class(..., window='whole_session', frame_class='all_frames')['all']` must
   equal the value `sess.process_whole_session_sp_rates_mapping(mapping)[0]` returns
   ([caban/sessions.py:1032](caban/sessions.py#L1032)) to within floating-point tolerance. Assert
   this in the section so a mismatch fails loudly. Also assert
   `n_place + n_non_place == n_total`.

4. **Frame-class partition check.** For a fixed cell class,
   `rate(all_frames) * n_frames_all == rate(movement) * n_frames_mov + rate(immobility) * n_frames_imm`
   (event counts are additive across a disjoint frame partition). Assert per mouse.

5. **Place-cell counts against the caches.** Spot-check that `n_place` for e.g. `Test_A`/`G05`
   matches the non-empty `pf_size` entries in
   `$NPY_SAVE_PATH/Test_A/Test_A-G05-318cells-PlaceFields.npz`.

6. **Exclusion coverage.** Confirm `Test_B` panels have no G07, `Test_A_1wk` and `Test_B_1wk`
   have no G15, and every other session has all 17 mice — visible in the printed per-group n.

7. **Visual review.** Confirm `PLOTS_DIR/navigation_aware_single_cell/` is populated, that no file
   appeared under `PLOTS_DIR/sp_rates/`, `place_fields_*`, `fluorescence_maps_*`, `fm_pf/` or the
   occupancy directories, and that `git status` shows the existing plotting functions unmodified.

8. Provide the copy-pasteable `importlib.reload()` snippet at the end of the work, per CLAUDE.md.

---

## Implementation notes (post-run)

Implemented and run end-to-end against the cached `ds` on 2026-07-29. Two things differed from
the plan as written.

### 1. Import ordering, not a new cycle

`caban.analysis` and `caban.decoder` are mutually dependent. Importing `caban.analysis` first
leaves `caban.decoder`'s deferred import of it half-initialised, so both new modules import
`caban.decoder` **before** `caban.analysis` — the same bootstrap ordering `caban/sections.py`
already uses. Resolved by ordering top-level imports, not by any in-function import.

### 2. One mouse has no immobility on LT1

`G06` never drops below the 2 cm/s threshold on LT1: minimum smoothed speed 2.057 cm/s across
612 s, median 22.8 cm/s, velocity trace complete and the same length as `S`. It therefore has no
immobility period and no defined immobility rate. Every other session/mouse has at least 380
immobility frames.

`rates_by_cell_class` returns `defined=False` with NaN values when a frame class holds under
1 s of eligible time. The section excludes those mice from that one panel, **prints the
exclusion**, and keeps the row in `place_cell_rates_summary.csv` with `defined=False` rather
than dropping it. If exclusions were to leave a group with fewer than two mice the panel raises.
This is documented in `place_cell_restricted_rates_methods.txt`.

The raw helpers in `caban/utilities.py` still hard-fail on an under-1 s mask — the softening is
only at the analysis layer, where the empty condition is a known behavioural property.

### Verification performed

- Synthetic checks: frame-mask helpers reproduce `get_avg_sp_rate_in_period` /
  `get_avg_activity_in_period` on contiguous windows, counts are additive across a disjoint
  frame partition, silent cells stay in the denominator, and all three failure modes raise.
- `assert_decomposition_consistent` and `assert_frame_partition_additive` pass for every mouse
  in all 7 sessions (17/17/16/16/16/17/17).
- Place-field caches load standalone for all 7 sessions without `run_pf_and_loc` having run.
- Output: 174 PNG + 174 SVG + 3 CSV + 55 TXT, including all 72 rate-split panels and 3564 CSV
  rows. Nothing written outside `PLOTS_DIR/navigation_aware_single_cell/`.
- `n_place + n_non_place == n_total` holds everywhere; place-cell rate exceeds non-place-cell
  rate in 111/116 session-mouse pairs.
