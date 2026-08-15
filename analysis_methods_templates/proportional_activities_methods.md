# Proportional cell activity across sessions and cross-registration mappings

## Overview

- This analysis summarizes, at the mouse level, how calcium activity is distributed across a
  set of sessions and/or cross-registration mappings (e.g. cells active only in LT1, only in
  LT2, in both, etc.), and compares the three viral groups (hM3D / hM4D / mCherry).
- Three interchangeable value modes share the same collection, plotting, and statistics code:
  - `fraction_active`: number of active cells per mapping, row-normalised per mouse so the
    values across the requested mapping set sum to 1.
  - `event_rate`: mean per-cell event rate (events/s) among active cells in a mapping.
  - `amplitudes`: mean per-cell event amplitude (mean `S` peak value) among active cells in a
    mapping.
- "Active" = at least one deconvolved event in `S_spikes` for that cell within the mapping.

## Caveat on cross-day registration dropout

- The single-session ("only") regions of a cross-day partition absorb registration failures. A
  cell that is genuinely active at both `TFC_cond` and `Test_B`, but which the cross-registration
  fails to match across days, is assigned the `TFC_cond` group and counted as if it had not been
  reactivated.
- The bias therefore runs consistently toward the single-session regions and away from the
  intersections, for all three value modes. The `TFC_cond` panel of a cross-day comparison should
  be read as an **upper bound** on the non-reactivated fraction, not a point estimate.
- This is inherent to the cross-registration design rather than to any parameter choice here:
  FOV drift, non-rigid tissue deformation and differing cell-detection yield across days all
  reduce match rates. It is the main reason cross-day reactivation fractions from calcium imaging
  are generally not comparable in absolute terms across studies with different registration
  pipelines, even though within-study group comparisons (as reported here) remain valid, since
  all three viral groups are subject to the same registration procedure.

## Caveat on the "active" criterion

- There is no field-wide standard definition of an "active" cell in calcium imaging, in the way
  that shuffle-tested spatial information is a recognisable framework for place-cell status.
  Published hippocampal criteria range from ≥1 detected transient, through ≥5 or ≥10 events
  per session, to minimum event-rate thresholds such as >0.01 Hz (Zhou et al., *Nat Commun*
  2024).
- The permissive ≥1-event criterion used here has explicit hippocampal precedent: Wirtshafter &
  Disterhoft (*J Neurosci* 2022) deliberately included cells firing ≥1 transient, and Geva et
  al. (*Neuron* 2023) required ≥1 Ca²⁺ event in both compared sessions.
- **Important**: "≥1 validated calcium transient" and "≥1 nonzero deconvolved estimate" are not
  the same criterion. `S_spikes` here is a CNMF-E/OASIS deconvolution output whose event count
  depends on the noise model, sparsity constraint, and component-initialisation thresholds, so
  this criterion is sensitive to processing parameters in a way that a MAD-thresholded
  transient detector (e.g. Rubin et al. 2015 at 4–5 MAD, Kirschen et al. 2017 at >8 MAD) is
  not.
- A minimum-rate gate was evaluated on this dataset and is **not** reported, because it does
  not discriminate here: per-cell median event rates are ~0.03 Hz on TFC_cond (~21 min
  sessions) and ~0.04–0.05 Hz on LT1 (~10 min), so a 0.01 Hz threshold sits at only the
  ~3rd–20th percentile and 80–97% of cells pass. Two reasons this is expected rather than
  informative:
  1. CNMF-E component initialisation is itself activity-dependent (PNR / local-correlation), so
     a genuinely silent cell tends not to be extracted as a component at all and never enters
     the denominator — making "fraction of extracted components that are active" partly
     circular.
  2. A fixed absolute threshold lands at very different percentiles of the rate distributions
     of different session types, so such a fraction is not comparable across LT and chamber
     sessions even though the within-session group comparison would be.
- The informative version of this question is the continuous per-cell event-rate distribution,
  which is reported separately by the cell activity distributions analysis
  (`cell_activity_distributions_methods.txt`) rather than by a thresholded fraction.

## Inputs

- Session dictionaries from the loaded dataset namespace, e.g. `TFC_cond`, `TFC_cond_LT1`,
  `TFC_cond_LT2` for the within-TFC-day comparison, or `TFC_cond`/`Test_B`/`Test_B_1wk` (and
  the `Test_A` counterpart) for cross-day comparisons.
- Cross-registration mapping strings (e.g. `'LT1+LT2'`, `'TFC_cond+Test_B+Test_B_1wk'`)
  resolved via `session_obj.get_S_mapping()`, which returns `S`, `S_spikes`, `S_peakval`,
  `S_idx` for the cell subset active in that specific mapping.
- Group labels come from `mice_per_group` (hM3D, hM4D, mCherry).

## Computation

1. For cross-day comparisons, mice are first filtered to those with a valid required mapping
   (e.g. `TFC_cond+Test_B+Test_B_1wk`); mice missing it are excluded per-group and logged.
2. Each panel is resolved against an explicit cross-registration. Within-day (conditioning-day)
   panels use each session object's own default `.crossreg`, which for `TFC_cond`, `TFC_cond_LT1`
   and `TFC_cond_LT2` is the LT1/LT2/TFC_cond registration. Cross-day panels instead pass the
   cross-day registration (`TFC_B_B_1wk` or `TFC_A_A_1wk`) explicitly via `crossreg_to_use`, so
   that every panel in a cross-day comparison is drawn from one cell universe.

   This is load-bearing for the `TFC_cond` panel specifically: that session object's *default*
   registration is the conditioning-day one, so without the explicit registration the `TFC_cond`
   panel of a cross-day comparison would report "TFC cells not active on the linear track" while
   its neighbouring panels reported cross-day mappings — two different cell populations side by
   side in the same row. With the registration passed explicitly it instead reports
   "TFC-recruited cells not reactivated at either test", which is a member of the same partition
   as the other panels (see below).
3. For each mouse and each requested mapping, `get_S_mapping()` is queried to determine the
   active cell subset and, depending on `value_mode`:
   - `fraction_active`: count of active cells.
   - `event_rate`: per-cell event count divided by session duration (frames /
     `MINISCOPE_FPS`), averaged across active cells.
   - `amplitudes`: per-cell mean `S` peak value, averaged across active cells.

   Mappings that raise (i.e. cross-registration does not resolve for that mouse) contribute 0.
4. For `fraction_active` only, per-mouse values are row-normalised (divided by their row sum)
   so panels represent a proportion of total activity rather than raw counts.

### Mapping sets are partitions, not overlapping selections

- Cells are bucketed by the exact `group` tuple assigned by the minian cross-registration output,
  and every registered cell carries exactly one such tuple. The mapping strings are therefore
  **mutually exclusive**: a cell counted under `TFC_cond+Test_B` is not also counted under
  `TFC_cond`.
- A full mapping set over an N-session registration enumerates all 2^N − 1 non-empty regions of
  the N-way Venn diagram. The seven-panel cross-day sets used by the paper figures
  (`TFC_cond`, `<test>`, `<test>_1wk`, the three pairwise intersections, and the triple) are the
  complete 3-way partition, so the panels account for every registered cell exactly once.
- Consequently the single-session panels are not redundant with the intersection panels that
  share a session name. `TFC_cond` is the TFC-recruited population that is *not* reactivated at
  either test, and the TFC-active population as a whole is the union of `TFC_cond`,
  `TFC_cond+<test>`, `TFC_cond+<test>_1wk` and the triple mapping.
- This partition property is what makes the `fraction_active` row-normalisation meaningful: the
  row sum is the mouse's total registered active-cell count, so each panel is a genuine
  proportion rather than a ratio against an arbitrary subset. Reporting a strict subset of the
  regions (as the three-panel `-sessions-` variants do) normalises against only those regions and
  should not be read as a partition.

## Group comparison statistics

- For each session/mapping panel, a one-way ANOVA (`scipy.stats.f_oneway`) is run across the
  three groups (hM3D, hM4D, mCherry).
- If the ANOVA p-value is below a PVALS threshold (0.05 / 0.01 / 0.001), a Tukey HSD post-hoc
  test (`statsmodels MultiComparison.tukeyhsd`) is run and significant pairwise comparisons are
  annotated as brackets on the panel.

## Visualization

- Two interchangeable panel styles:
  - `boxplot`: bar of the group mean with error bars (± SD).
  - `violin`: violin plot with per-mouse jittered strip points and a black median line; used
    for all paper-figure variants.
- Standard group coding: hM3D = red ("Exc"), hM4D = blue ("Inh"), mCherry = black/grey ("Ctl").
- Paper-figure variants additionally build: a 4-panel main figure (selected panels from the
  within-day and cross-day comparisons), a 7+7-panel two-section supplementary figure for a
  single cross-day test family, and a 7+7+7-panel three-section supplementary figure combining
  both cross-day test families (Test_A and Test_B). The three-section supplementary figure is
  generated for all three value modes.
- Panel labels mark single-session regions with an explicit `only` suffix (`LT1 only`, `TFC only`,
  `48hr only`, …) while intersection panels are named by the sessions they combine (`LT1+LT2`,
  `TFC+48hr+1wk`, …). A bare session name would read as "all cells active in that session", which
  is not what any panel reports.
- The suffix is relative to the sessions of that panel's own registration, which the section
  heading names. `TFC only` therefore denotes different cell populations in different sections:
  under *TFC Conditioning Day* it is TFC cells not active during either linear-track session,
  whereas under *Cross-day Test B* / *Cross-day Test A* it is TFC-recruited cells not reactivated
  at either test of that family. The two are not comparable as a column across sections.
- Within each section the y-axis is shared across panels and fixed before the significance
  brackets are drawn, with headroom above the section maximum, so that bracket geometry (which is
  computed relative to the axis range) is correct and unclipped at any metric scale.

### Main-figure candidate layouts

Three compact alternatives to the full 7+7+7-panel supplementary figure are generated side by
side in `main_figure_candidates/`, for eventual selection of one as the manuscript main-text
panel. All three draw from the same three-section partition (conditioning day, cross-day Test B,
cross-day Test A) across all three value modes, so they are exact re-renderings of a subset or
summary of the supplementary data, not an independently computed result.

- **`hero-region-strips`**: a hand-selected subset of regions judged visually significant in the
  supplementary figures, one single-axis grouped-violin strip per value mode, each panel a
  violin triplet (hM3D / hM4D / mCherry) with per-mouse jittered points and Tukey post-hoc
  brackets — the same visual grammar as the supplementary figure's own single-panel style. The
  region selection is a fixed list in code (`_MAIN_FIGURE_HERO_PANELS` in `caban/analysis.py`)
  and is a judgment call, not a statistical selection procedure — it is not corrected for
  selecting the largest-looking effects out of the full 21-region set, so the ANOVA/Tukey
  p-values on the displayed panels should be read as would any other point estimate, without
  additional adjustment.
- **`group-profile`**: 3-value-mode × 3-section grid; each cell plots group mean ± SEM across all
  7 regions of that section's partition (no subsetting), showing whether a group's activity
  profile differs across the partition in shape rather than in one isolated region.
- **`difference-from-control`**: same 3×3 grid, plotting (hM3D − mCherry) and (hM4D − mCherry)
  group-mean differences with a percentile bootstrap 95% CI (2000 resamples per region, fixed
  RNG seed) instead of the raw group means, against a zero reference line. With per-group mouse
  counts in the range of ~5-8, treat the interval as indicative rather than as a precise
  confidence bound; it is a different statistical procedure from the ANOVA-gated Tukey test used
  elsewhere in this analysis; and it is not corrected across regions.

## Output

- Figures are saved under `<PLOTS_DIR>/proportional_activities/` as matched `.png` (300 dpi)
  and `.svg` pairs. Filenames encode the `value_mode` metric prefix (`frac-active` /
  `event-rate` / `amplitudes`), the panel set (sessions vs. mappings), and, for cross-day
  panels, the test family suffix (`TFC_B_B_1wk` / `TFC_A_A_1wk`).
- The paper figures use the `-paper-main` and `-paper-supplementary` basenames with the same
  metric prefixes, giving `frac-active-paper-supplementary`, `event-rate-paper-supplementary` and
  `amplitudes-paper-supplementary` for the three-section combined supplementary figure, and
  `frac-active-paper-main` (plus a `-TestA` suffixed variant) for the main figure.
- When `paper_fig2_dir` is supplied, selected panels are additionally exported as standalone
  SVGs into that directory for direct manuscript figure assembly.
- The main-figure candidate layouts are saved separately under
  `<PLOTS_DIR>/proportional_activities/main_figure_candidates/`, as
  `hero-region-strips`, `group-profile` and `difference-from-control` `.png`/`.svg` pairs. They
  are not exported to `paper_fig2_dir`, since only one is expected to be selected for the
  manuscript.

## Error handling

- Cross-day panels hard-fail if `crossreg_to_use` is not supplied, and if no mice remain in any
  group after filtering for the required mapping.
- Unknown `value_mode` or `plot_type` values raise `ValueError` rather than silently falling
  back.
