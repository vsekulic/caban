# Cell-averaged spike rates and activity across behaviour periods and whole sessions

## Overview

- This analysis summarizes calcium activity, averaged across cells, for a cross-registration
  cell mapping, either within discrete behaviour periods (tones, shock, post-shock, post-tone)
  or across the whole session as a single period, and compares the three viral groups
  (hM3D / hM4D / mCherry).
- Two interchangeable metrics share the same collection code:
  - **spike rate**: number of deconvolved events per second.
  - **activity**: summed `S` peak value per second (`want_peakval=True`), which weights each
    event by its deconvolved amplitude rather than counting it as 1.
- **CRITICAL**: the average is taken over EVERY cell in the mapping, INCLUDING cells with zero
  detected events in the period. Silent cells contribute 0 to the numerator but still count in
  the denominator. This is deliberate. A manipulation that pushes cells to silence (e.g. hM4D)
  must show up as a reduced average rate, and it only does so if those cells remain in the
  denominator.
- **NAMING**: the figures label this "Avg. spike rate (events/s)" / "Avg. activity (peak S/s)",
  not a "population rate". The quantity is the MEAN ACROSS CELLS of each cell's own rate, not a
  sum over cells; the two differ by a factor of `n_cells`, which varies between animals, so the
  "population" wording invited a reading of the axis that is wrong by a per-mouse factor.
- This is the complement of the `proportional_activities` analysis, which restricts its
  `event_rate` and `amplitudes` metrics to ACTIVE cells only (≥1 event). Because that analysis
  drops silenced cells from the denominator entirely, its per-cell metrics understate a silencing
  manipulation: the surviving active cells look near-normal. The two analyses answer different
  questions and are expected to disagree in magnitude:

  | Analysis | Question it answers |
  |---|---|
  | `sp_rates` | "how much is the average cell firing overall?" |
  | `proportional_activities` | "among cells that still fire, how hard do they fire?" |

## Inputs

- Session dictionaries from the loaded dataset namespace: `TFC_cond`, `Test_A`, `Test_A_1wk`,
  `Test_B`, `Test_B_1wk`, `LT1`, `LT2` (each a dict of mouse → session object).
- Cross-registration mapping strings (e.g. `'full'`, `'LT1+LT2+TFC_cond'`,
  `'TFC_cond+Test_B+Test_B_1wk'`) resolved via `session_obj.get_S_mapping()`, which returns
  `S`, `S_spikes`, `S_peakval`, `S_idx` for the cell subset in that mapping. The `'full'` mapping
  means all cells in the session, with no cross-registration constraint.
- Group labels come from `mouse_groups` / `mice_per_group` (hM3D, hM4D, mCherry).

## Computation

1. **Period-wise metrics.** Each session class defines its behaviour periods as (onset, offset)
   frame pairs: `TFC_cond` has 5 tone, 5 shock and 5 post-shock periods; `Test_B` has 3 tone,
   3 post-tone and 3 tone+post-tone periods. For each period, every cell's events falling inside
   the window are counted, divided by the period duration, and averaged over all cells in the
   mapping (`get_avg_sp_rate_in_period` / `get_avg_activity_in_period`).
2. **Whole-session metric.** The entire recording is treated as a single period spanning
   `[0, n_frames - 1]` of the trimmed activity matrix `S`. `Test_A` / `Test_A_1wk` (context-only
   exposure) and `LT1` / `LT2` (linear track) have no discrete behaviour periods, so this is
   their only spike-rate metric; for `TFC_cond` and `Test_B` it complements the period-wise
   panels.
3. **Frame convention.** All period bounds are expressed RELATIVE to the trimmed `S`, which
   begins at the experiment start. `S_spikes` is derived from that same trimmed matrix, so the
   two are in the same frame of reference. (`miniscope_exp_fnum` holds indices into the
   *untrimmed* recording and is deliberately not used as a period bound; mixing the two silently
   drops the leading frames' events while leaving the duration denominator correct, biasing
   rates low.)
4. **Duration convention.** Period length is computed as `(offset - onset) / MINISCOPE_FPS`, so
   a whole-session window of `[0, n_frames - 1]` is one frame short of the true recording
   duration (~0.02% at n = 6000 frames). This matches the convention used for every other period
   in the codebase and is not corrected.
5. **Binned metrics.** The same across-cell average is computed over consecutive fixed-width bins
   (`BIN_WIDTH` frames) to give a time course across the session. Average tone and shock onsets
   across mice are overlaid as dashed vertical lines, except for `Test_A` / `Test_A_1wk`, which
   are context-only and present no tones or shocks.
6. **Binned line plots** show the across-mouse mean per bin with a shaded SEM ribbon, computed as
   the standard deviation across mice divided by `sqrt(number of mice contributing data to that
   session)`. Mice absent from a session (e.g. G07 for `Test_B`, G15 for `Test_A_1wk`) are
   excluded from both the mean and the SEM rather than entering as zeros.

## Statistics

- **Whole-session panels**: the unit of analysis is the MOUSE. Each animal contributes one value
  per group, shown as a violin with the individual animals overlaid as points. Groups are
  compared by one-way ANOVA (`scipy.stats.f_oneway`); only if the omnibus p < 0.05 is a Tukey HSD
  post-hoc test (`statsmodels` `MultiComparison`) run and significant pairs annotated.
  n = number of mice per group. A group with fewer than 2 mice raises rather than silently
  producing an untestable panel.
- **Period-averaged panels**: the same period-averaged quantity is emitted twice, as two files
  differing only in the unit of analysis. Both use the identical violin + scatter presentation
  and the same ANOVA-gated Tukey annotation, so they are directly comparable by eye:

  | File suffix | One point is | n | Notes |
  |---|---|---|---|
  | `-avg` | a behaviour period (group mean) | 3–5 | Collapses the period axis before testing. Retained for continuity with previously published figures; with n = 3 for `Test_B` the violin is a thin density estimate. |
  | `-avg-mice` | a mouse | number of mice | Each animal is averaged over its own periods first. Same unit as the whole-session panel, and the correct across-animal comparison. |

  The `-avg-mice` panel is the one to prefer when the two disagree: the `-avg` panel's points
  are group means over periods, which are not independent animals, so its n overstates the
  evidence available. A group contributing fewer than 2 observations raises rather than
  silently producing an untestable panel.
- Group order in all group-comparison panels is hM3D, mCherry, hM4D (displayed as Exc, Ctl,
  Inh). The period-averaged panels previously used hM3D, hM4D, mCherry; they were reordered to
  match the whole-session panels when they were converted from bars to violins.

## Figure axes

- Every panel produced by this analysis — the period-wise bars, the two period-averaged group
  panels, the binned time course and the whole-session violins — is plotted in the SAME units:
  events/s for the spike-rate variant, summed `S` peak value per second for the `-activity`
  variant. Both come from `get_avg_sp_rate_in_period` / `get_avg_activity_in_period`, which
  divide by the period duration.
- All group-comparison panels are violin + jittered scatter with a black median line and
  ANOVA-gated Tukey brackets, drawn by the shared `_draw_violin_triplet` helper in
  `caban/analysis.py`. The period-averaged panels were previously bar charts with SD error
  bars; showing the individual observations makes the small n visible rather than hiding it
  behind a summary bar. Panel size is fixed at `_SP_RATES_VIOLIN_FIGSIZE`, matching the
  whole-session panels so the two can be arranged interchangeably in a figure.
- **NO percentage normalization** is performed anywhere in this analysis. An earlier version of
  the period-wise and binned panels drew y-ticks labelled 0/25/50/75/100 on the unnormalized
  bars, and placed them at bare fractions of the axis range (`range*0.25` rather than
  `min + 0.25*range`), so the ticks were also misplaced whenever the axis did not start at zero.
  Those ticks have been removed in favour of the default autoscaled axis; the plotted values were
  never percentages and were never changed.
- **NO ΔF/F is computed on this path.** These panels read the deconvolved `S` matrix (via
  `S_spikes` / `S_peakval`) and nothing else. The earlier `Normalized ΔF/F (%)` y-label was
  simply wrong; the axis label now comes from a single shared helper (`_sp_rates_ylabel` in
  `caban/analysis.py`) so the period-wise, binned and whole-session panels cannot drift apart
  again. The only genuine ΔF/F machinery in the codebase is `run_avg_population_activity`
  (`caban/sections.py`), which does not feed this analysis.
- The binned panels' x-axis differs by variant: the line plot (`plot_bars=False`) is in elapsed
  minutes, while the bar plot (`plot_bars=True`) is in bin index. The label follows the variant.
- **Significance bracket geometry.** `barplot_annotate_brackets` expresses bracket offsets as
  fractions of the axis range, and `do_anova1_plot` stacks up to three brackets at
  `0.01`, `0.01 + incr`, `0.01 + 2*incr`. Two consequences are handled explicitly rather than
  by eye:
  - The y-range is derived from that increment
    (`_violin_ylim_with_bracket_headroom`), not hardcoded, so the top bracket can never be
    drawn past the top of the axes. A fixed 1.35x factor previously left the post-shock
    activity panels (which passed `tot_dh_incr=0.25`, needing 2.38x) with brackets off-canvas
    and orphaned asterisks floating over the title.
  - All brackets anchor at the SAME baseline, the tallest group in the panel. Anchoring each
    bracket at its own pair's maximum let a bracket spanning two low groups start far below the
    others and collide with them despite a larger increment.

  These are presentation-only; no test, p-value or plotted value depends on them.

## Outputs

- `PLOTS_DIR/sp_rates/<session_type>_sp_rates/`
  - `<session_type>_sp_rates-<period label> <mapping>.png` — period-wise bars (time course)
  - `<session_type>_sp_rates-<period label> <mapping>-avg.png` — violin, n = periods
  - `<session_type>_sp_rates-<period label> <mapping>-avg-mice.png` — violin, n = mice
  - `<session_type>_sp_rates-Whole-session <mapping>.png` / `.svg` — per-mouse violin + stats
- `PLOTS_DIR/sp_rates/<session_type>_binned_sp_rates_mapping/`
  - `<session_type>_binned_sp_rates_mapping-<mapping>.png` — binned time course
- `<session_type>` is suffixed with `-activity` for the amplitude-weighted variant, which lands
  in its own sibling directory.
