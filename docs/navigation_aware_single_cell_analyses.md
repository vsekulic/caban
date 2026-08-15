# Navigation-aware single-cell analyses

A map of what the `navigation_aware_single_cell` suite does, and how it relates to the spike-rate /
activity panels and the original place-field plotting.

## What the suite is

`navigation_aware_single_cell` is not a config flag — it is an output-directory constant,
`NAV_AWARE_DIR` in [place_cell_rates.py:49](../caban/place_cell_rates.py#L49). Three modules write
under it:

| Module | Role |
|---|---|
| [caban/place_cell_rates.py](../caban/place_cell_rates.py) | frame masks, place/non-place partition, restricted rates, PF properties per mouse |
| [caban/locomotion.py](../caban/locomotion.py) | locomotion metrics + ANCOVA of rate on locomotion |
| [caban/speed_tuning.py](../caban/speed_tuning.py) | speed-binned occupancy, tuning curves, Poisson GLM, standardized rate |

Five entry points in [caban/sections.py:1050-1482](../caban/sections.py#L1050), each gated on a
`cfg.plot_*` flag ([config.py:91-102](../caban/config.py#L91)):
`run_locomotion_comparison`, `run_place_cell_properties`, `run_place_cell_rates`,
`run_rate_vs_locomotion`, `run_speed_tuning`.

### The core idea

The suite exists to **decompose the whole-session spike-rate/activity number** produced by the
`sp_rates` section. It takes the same per-mouse scalar and factors it along three axes:

- **cell class** — place / non-place / all, where "place cell" = has ≥1 detected place field
  (`partition_place_cells`, [place_cell_rates.py:198](../caban/place_cell_rates.py#L198))
- **frame class** — `all_frames` / `movement_only` (v ≥ 2 cm/s) / `immobility_only`
  (`build_frame_mask`, [place_cell_rates.py:237](../caban/place_cell_rates.py#L237))
- **window** — `whole_session` / `pretone_180s` (reuses `spatial._build_pretone_mask`)

…and then asks whether a group difference in rate survives conditioning on locomotion (ANCOVA) and on
graded speed (Poisson GLM + direct standardization).

### Behavioural gating

All gating is **speed- and window-based only**. There is no lap, direction, or trial-epoch gating
anywhere in this suite. Threshold is the shared `VELOCITY_THRESHOLD = 2.0 cm/s`
([utilities.py:18](../caban/utilities.py#L18)) applied to `sess.velocities_miniscope_smooth` — the
same trace the occupancy analysis consumes. Speed bin edges start at that same threshold so bin 0 is
byte-identical to `immobility_only` (asserted by `assert_bin_zero_is_immobility`).

## Similarities to the spike-rate / activity panels

Strong deliberate reuse:

- **Same two metrics** — events/s and summed `S_peakval`/s, selected by `want_peakval`; y-labels come
  from `_WHOLE_SESSION_YLABEL` imported straight out of `analysis.py`.
- **Same denominator convention** — mean across *all* cells in the mapping, silent cells included.
- **Same plot form and stats** — `_draw_violin_triplet` + `do_anova1_plot` (ANOVA-gated Tukey HSD),
  n = mice, groups `hM3D / mCherry / hM4D` labelled Exc/Ctl/Inh.
- **Same cell sourcing** — `sess.get_S_mapping(mapping, want_peakval=...)`.
- **Explicitly verified to agree** — `assert_decomposition_consistent`
  ([place_cell_rates.py:339](../caban/place_cell_rates.py#L339)) asserts the whole-session /
  all-frames / all-cells cell equals `process_whole_session_sp_rates_mapping`, and
  `assert_frame_partition_additive` asserts movement + immobility partition the window exactly.

### Differences

| | `sp_rates` / binned | nav-aware |
|---|---|---|
| Time selection | contiguous periods (tone, shock, post-shock) or fixed-width bins | arbitrary boolean frame mask |
| Helper | `get_avg_*_in_period` | `get_avg_*_in_frame_mask` ([utilities.py:358-420](../caban/utilities.py#L358)) |
| Question | "does rate differ by group in epoch X?" | "is that difference explained by where/how fast the mouse was?" |
| Undefined data | n/a | `defined=False` + NaN when <1 s of eligible frames (e.g. G06 on LT1) — dropped from panels, kept in CSV |
| Verification | none | assertion suite + a synthetic Poisson recovery test for the GLM |

Compared with `proportional_activities` / `cell_activity_distributions`, the nav-aware suite differs on
two conventions that are worth remembering: those two average over **active cells only** and use
`S.shape[1]/FPS` for duration, whereas the nav-aware and `sp_rates` families include silent cells and
use the `n_frames-1` convention.

Compared with `event_locked_responsiveness` / `cell_activity_distributions`: those work at the
per-cell level with shuffle nulls, BH-FDR, mixed models and ECDFs. The nav-aware suite is mostly
n = mice, except `speed_tuning`, which does drop to per-cell (`fit_per_cell_speed_glm`, slope ECDFs,
cell-nested-in-mouse LMM) and reuses `single_unit_common.ecdf_panel` / `fit_group_mixed_model`.

## Relationship to the original PF plotting

The nav-aware suite **consumes** place fields, it does not compute them. Detection stays in
[caban/spatial.py](../caban/spatial.py): `FluorescenceMap.generate_occupancy_map` (:295),
circular-shift shuffles (:441), 99th-percentile significant responses (:389), BGMM field fitting in
`find_place_fields` (:494). `load_place_fields`
([place_cell_rates.py:158](../caban/place_cell_rates.py#L158)) reads `sess.fm.pf` if live, else
reconstitutes from the cached `*-PlaceFields.npz`, so it has no ordering dependency on
`run_pf_and_loc`.

The real divergence is the **unit of analysis**:

| | original `plot_pf_analyses` ([analysis.py:4789](../caban/analysis.py#L4789)) | `plot_pf_properties_per_mouse` ([place_cell_rates.py:572](../caban/place_cell_rates.py#L572)) |
|---|---|---|
| Unit | pooled cells/fields | mice |
| Plot | cumulative step histograms + notched boxplots | violin triplets |
| Test | `scipy.stats.kstest` on pooled cells | ANOVA-gated Tukey |
| Mice | hard-skips `G07`, `G15` | no blanket skip |
| Filenames | `num_pfs`, `pf_size`, `pf_compactness`, `spatial_selectivity` | **same four stems, deliberately**, so the two can be read side by side |

Both share `pf_bin_area_cm2` / `assert_pf_bin_width_matches` from `decoder.py` for the bin→cm²
conversion. The pooled version is retained on purpose — this is a companion, not a replacement.

Unrelated to either: the per-cell rate-map galleries (`plot_fluorescence_map_plotter`, `hot` colormap,
no normalization) and the LT tuning/PV-correlation stack (`plot_lt_spatial_responses`, viridis,
per-cell normalization, PF-centre sorting). The nav-aware suite emits no rate-map images at all —
only violin panels, curves, scatter+fit, ECDFs, CSVs and stats text.

## Outputs and METHODS

Everything lands under `<PLOTS_DIR>/navigation_aware_single_cell/` in subdirs
`place_cell_rates/`, `place_cell_proportions/`, `place_field_properties_per_mouse/`,
`locomotion_metrics/`, `rate_vs_locomotion/`, `speed_tuning/`. `_save`
([place_cell_rates.py:515](../caban/place_cell_rates.py#L515)) writes png (300 dpi) **and** svg — the
legacy PF code mostly writes png only.

Six METHODS templates are copied at runtime: `place_cell_restricted_rates_methods.txt`,
`place_cell_proportion_methods.txt`, `place_field_properties_per_mouse_methods.txt`,
`locomotion_group_comparison_methods.txt`, `rate_locomotion_ancova_methods.txt`,
`speed_tuning_methods.txt`. Notably, the original `plot_pf_analyses` pooled-cell KS analysis has
**no** METHODS template.

## Two gotchas

1. **Group order collision** — `single_unit_common.GROUP_ORDER` is `['hM3D','hM4D','mCherry']`; the
   nav-aware order is `['hM3D','mCherry','hM4D']`. The suite threads its order through
   `ecdf_panel(group_order=...)` rather than mutating the shared default.
2. **Import cycle** — `caban.decoder` must be imported before `caban.analysis` in all three nav-aware
   modules.
