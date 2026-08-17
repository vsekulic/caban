"""
Cell-level pyramidal event-AMPLITUDE analysis for DREADD effects on trace fear conditioning.

Replaces sp_rates' ~205 uncorrected three-group ANOVAs (session x epoch x cross-registration
subset x metric) with a small, pre-declared confirmatory family built on PER-EVENT DECONVOLVED
AMPLITUDE at the cell level, never collapsing cells to a per-mouse scalar before testing. See
analysis_methods_templates/sp_rates_lmm_methods.md for the full statistical rationale.

Amplitude, not rate, is primary here: the scientific claim under test is that hM3D increases
pyramidal BURSTING (larger per-event Ca2+ influx), not merely more frequent events. Event rate,
fraction of cells active, and total S/s are always reported alongside amplitude in the same
panel -- "secondary" means they carry no confirmatory alpha, not that they are hidden.

This is a LOCKED CONFIRMATORY REANALYSIS, not a prospective preregistration: the amplitude-primary
decision followed prior inspection of this dataset (see METHODS). A post-hoc methodological
review of an earlier version of this module found three correctness bugs (denominator df, a
missing exposure factor, a mislabelled recall title) and two overstated designs (the co-primary
epoch model's pseudoreplication, cell- vs mouse-weighting of the primary contrast); all are fixed
here -- see the module CHANGELOG below and analysis_methods_templates/sp_rates_lmm_methods.md.

Module layout
-------------
  Event/run table construction -- _iter_event_windows() (shared traversal), build_epoch_event_table(),
                                  build_run_structure_table(), aggregate_over_trials(),
                                  filter_amplitude_rows(), build_mouse_trial_epoch_rate_table(),
                                  build_mouse_trial_trace_amplitude(), compute_epoch_delta_table()
  Confirmatory models        -- fit_primary_trace_amplitude(), fit_epoch_delta_model(),
                                fit_epoch_interaction_nested_attempt() (one-off, not in the main
                                pipeline -- see its docstring), fit_rate_group_epoch_model(),
                                holm_correct_confirmatory()
  Secondary/sensitivity models -- fit_group_trial_model(), report_group_trial_slopes(),
                                run_threshold_sensitivity(), compute_group_contrast_point_estimates()
  Small-n inference           -- mouse_label_permutation_test(), make_contrast_stat(),
                                make_amplitude_contrast_stat()
  Detection-bias / manip check -- compute_lt1_lt2_dropout(), compute_lt1_lt2_amplitude_delta()
  Descriptive/triangulation  -- mouse_level_trace_amplitude(), mouse_level_p90_amplitude(),
                                fraction_active_table(), summarize_run_structure(),
                                report_recall_cohort_note()
  Figures                     -- plot_primary_trace_amplitude(), plot_epoch_profile(),
                                plot_amplitude_ecdf(), plot_amplitude_p90(), plot_decomposition(),
                                plot_manipulation_check(), plot_run_structure(),
                                plot_threshold_sensitivity(), plot_effect_forest(),
                                plot_example_traces(), plot_width_vs_height_matched_examples()

CHANGELOG (post-review corrections, see analysis_methods_templates/sp_rates_lmm_methods.md and
the plan this module was built from for the full rationale):
  - joint_wald_test's denominator df is now n_groups-1 (caban.single_unit_common), not
    nobs-n_fixed -- fixes a massively anti-conservative omnibus p-value on every cluster-robust
    or MixedLM joint test in this module (and in caban.pca_state_metrics, which shared the bug).
  - build_mouse_trial_epoch_rate_table's exposure offset now SUMS exposure_seconds across cells
    (was 'first'), fixing a missing factor of n_cells that leaked into the rate model's group
    effect (cell count is group-correlated).
  - plot_primary_trace_amplitude/plot_amplitude_ecdf take an explicit `title`, fixing a
    hardcoded trace-period title that was silently wrong when reused for the Test_B/Test_B_1wk
    post-tone window.
  - The co-primary is now compute_epoch_delta_table + fit_epoch_delta_model (within-cell
    trace-minus-pre_tone delta, one row per cell), replacing a group x epoch interaction fit
    directly on the cell x trial x epoch table with only a mouse random intercept -- a cell could
    contribute up to 20 correlated rows there, understating the true uncertainty by many orders
    of magnitude. The new design cancels each cell's own baseline by construction and converts
    the test into a between-mouse contrast, structurally identical to the primary endpoint.
  - The primary contrast's reported point estimate is now the EQUAL-MOUSE-WEIGHTED contrast
    (mean of each mouse's own mean; compute_group_contrast_point_estimates), not the pooled
    cell-weighted mean, since cell count is itself group-correlated (hM3D contributes ~1.45x as
    many active cells/mouse as the other groups) and post-treatment/activity-dependent.
  - New run-structure evidence (build_run_structure_table/summarize_run_structure/
    plot_run_structure) and a threshold-sensitivity check (run_threshold_sensitivity) directly
    address whether the amplitude effect is genuinely burst-like or a run-merging artifact of the
    contiguous-run event definition.
  - New group x trial check (build_mouse_trial_trace_amplitude/fit_group_trial_model) is the
    photobleaching control the module previously left outstanding.
  - Every group-comparison panel stays a MOUSE-LEVEL violin+scatter (one point per animal), in
    CLAUDE.md's hM3D/mCherry/hM4D display order, built through the shared
    _mouse_values_per_group/_draw_mouse_violin_panel pair. A cell-level SuperPlot variant was
    implemented and then REVERTED after inspection on the real dataset -- see
    _mouse_values_per_group's docstring for exactly why (long right tails put all 17 mouse means
    in the bottom ~6-13% of every panel). draw_superplot_triplet remains available in
    caban.single_unit_common but is no longer called here.
  - New example-trace panels (plot_example_traces, plot_width_vs_height_matched_examples) show
    representative raw S/C traces with the measured runs shaded, reproducibly selected by
    within-group amplitude percentile.

DEFERRED (not implemented this pass -- see analysis_methods_templates/sp_rates_lmm_methods.md's
Deferred section): the LT1->LT2 dropout DECOMPOSITION (independent-CNMF-E-miss vs CellReg
threshold vs footprint-match failure vs QC exclusion), a cross-registration-subset sensitivity
forest plot, movement-conditioned/standardized analyses, ROI-per-FOV QC panels, reference-seeded
extraction, count-model posterior-predictive/zero diagnostics, peri-shock analysis, and a
declared BH-FDR family for the frequentist secondaries.
"""
import os
import functools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import bambi as bmb
import arviz as az

from caban.utilities import find_event_runs_ca, find_event_runs_ca_S, MINISCOPE_FPS
from caban.single_unit_common import (
    GROUP_ORDER, GROUP_LABELS, GROUP_COLOURS, DREADD_DISPLAY_ORDER,
    ensure_dirs, write_text, save_fig, ecdf_panel, fdr_correct,
    fit_mixed_model, joint_wald_test, draw_superplot_triplet, no_stat_annotation,
    mouse_contrast_ci, annotate_contrast_ci, format_contrast_ci_lines,
    annotate_pairwise_brackets,
)
from caban.decoder import _copy_analysis_methods_template
from caban.analysis import _draw_violin_triplet, do_pairwise_holm_plot
from caban.epoch_analysis import get_epoch_frames, get_testb_epoch_frames

METHODS_FILENAME = 'sp_rates_lmm_methods.md'

# TFC_cond epochs used by the primary/co-primary confirmatory model. Shock is deliberately
# excluded from this set: at 0.05-0.2 Hz a 2 s window yields ~0-1 events/cell, so per-cell
# amplitude there is dominated by quantization and the window carries motion artifact. Handled
# separately later via YrA/C, not here.
TFC_EPOCHS = ('pre_tone', 'tone', 'trace', 'post_shock')
TFC_TRACE_EPOCH = 'trace'
TFC_REFERENCE_EPOCH = 'pre_tone'

# Test_B/Test_B_1wk post-tone window pinned to 20 s -- NOT either existing default -- to match
# the representative TFC trace duration (tone_offsets[i] to shock_onsets[i] is 20 s for trials
# 2-5; trial 1 is 15 s because tone_onsets_def[0]=185 rather than an intended 180, see
# TraceFearCondSession.__init__). Neither get_testb_epoch_frames' own default (35 s) nor
# TestBSession.post_tone_onsets/offsets (extends to the next tone onset, ~200+ s) is the recall
# analog of the true trace duration -- only this explicit 20 s call is.
TESTB_POST_TONE_DURATION_S = 20.0
TESTB_EPOCHS = ('pre_tone', 'tone', 'post_tone')

# Sensitivity thresholds for the run-merging check (plan section 4): if the primary amplitude
# contrast survives across all three, run-merging (temporally adjacent peaks collapsing into one
# wider run at a looser threshold) cannot be the sole explanation for the effect. 2.0 is assumed
# to match each session's own default `.thres` -- verify against a live session before reporting
# (see the reload snippet / verification notes this module ships with).
THRESHOLD_SENSITIVITY_VALUES = (1.5, 2.0, 3.0)

# Reproducible, non-hand-picked example-cell selection (plan section 5c): within-group amplitude
# percentiles shown as a low/median/high triplet, and the context window padding around each
# plotted trace epoch.
EXAMPLE_TRACE_PERCENTILES = (10, 50, 90)
EXAMPLE_TRACE_PAD_S = 2.0

# Which pairwise contrasts this module's panels Holm-correct ACROSS. 'vs_control' corrects over
# the two contrasts that actually carry an inferential claim here -- hM3D-vs-mCherry and
# hM4D-vs-mCherry -- and reports hM3D-vs-hM4D uncorrected alongside.
#
# The design question is "does each DREADD differ from its own control", so hM3D-vs-hM4D spends
# no alpha of its own; including it in the family costs power without protecting any error rate
# that is at risk. Correcting over three was CONSERVATIVE rather than wrong (it inflates nothing),
# but at n=5/6/6 animals power is the binding constraint and conservatism is not free.
#
# Scoped to THIS module deliberately. do_pairwise_holm_plot's own default stays 'all', because it
# is shared with freezing_tuned_cells and the other single-unit suites whose published figures
# report p-values corrected over three. Flip this one constant to 'all' to restore the previous
# behaviour for every panel here at once.
PANEL_HOLM_FAMILY = 'vs_control'


# ─────────────────────────────────────────────────────────────────────────────
# Event/run table construction
# ─────────────────────────────────────────────────────────────────────────────

def _iter_event_windows(mice_per_group, sessions, epoch_names, get_frames_fn, mapping='full',
                        thres=None, crossreg_to_use=None):
    """
    Shared traversal behind build_epoch_event_table() (aggregate n_events/sum_amplitude per
    cell-trial-epoch window) and build_run_structure_table() (one row per individual detected
    run, for the run-width/local-maxima bursting evidence in plan section 4) -- the
    group/mouse/session/trial/epoch/cell nesting and the find_event_runs_ca_S() call are
    identical for both; only what each caller does with a window's runs differs. Per CLAUDE.md's
    dedup rule, this traversal exists exactly once.

    Yields (group, mouse, trial_idx, epoch, cell_id, exposure_s, frameidx, amplitude,
    n_local_maxima, width, peak_height) once per (mouse, trial, epoch) window get_frames_fn
    resolves, for EVERY cell in that session's mapping. The five run-level arrays are each
    restricted to runs whose ARGMAX FRAME falls inside [onset, offset) -- possibly length-0 for a
    cell with no qualifying events in the window. peak_height is S[cell_row, frameidx] at each
    run's own argmax frame (find_event_runs_ca returns the per-event INTEGRAL as amplitude, not
    the peak value -- callers needing peak height, e.g. plot_width_vs_height_matched_examples's
    matched selection, get it computed once here rather than re-indexing S a second time).

    CAVEAT (inherited from this module's original event-table design, not new here): membership
    is decided by the run's PEAK frame, not its full extent, so a run whose peak falls just inside
    a window but whose start (or end) crosses the window boundary contributes its WHOLE
    amplitude/width to that one epoch/trial -- none of it is truncated at the boundary or split
    across the two neighbouring windows. This is a real, if likely small, source of epoch-boundary
    noise for boundary-adjacent bursts (most relevant to compute_epoch_delta_table's trace-vs-
    pre_tone contrast, right at the tone/trace transition); flagged here rather than silently
    assumed away, but not changed in this pass -- fixing it would mean deciding how to split a
    single run's amplitude across two epochs, a design question of its own.

    'cell_id' is the mapping's own unit_id (get_S_mapping's S_idx) -- SESSION-LOCAL, never join
    it across sessions (the same physical cell carries a DIFFERENT unit_id in each session under
    cross-registration; cross-session correspondence is ROW ORDER, not unit_id equality -- see
    compute_lt1_lt2_amplitude_delta()).
    """
    for group in GROUP_ORDER:
        for mouse in mice_per_group.get(group, []):
            if mouse not in sessions:
                continue
            session = sessions[mouse]
            cell_thres = session.thres if thres is None else thres
            [S, _S_spikes, _S_peakval, S_idx] = session.get_S_mapping(mapping, with_crossreg=crossreg_to_use)
            if S.shape[0] == 0:
                raise RuntimeError(
                    f'_iter_event_windows: mouse {mouse} has 0 cells for mapping={mapping!r}.')
            events = find_event_runs_ca_S(S, cell_thres)

            for trial_idx in session.periods:
                for epoch in epoch_names:
                    frames = get_frames_fn(session, epoch, trial_idx)
                    if frames is None:
                        continue
                    onset, offset = frames
                    exposure_s = (offset - onset) / MINISCOPE_FPS
                    for cell_row, cell_id in enumerate(S_idx):
                        frameidx, amplitude, n_local_maxima, width, _start = events[cell_row]
                        in_window = (frameidx >= onset) & (frameidx < offset)
                        peak_height = S[cell_row, frameidx[in_window]] if np.any(in_window) else np.array([])
                        yield (group, mouse, int(trial_idx), epoch, cell_id, exposure_s,
                              frameidx[in_window], amplitude[in_window],
                              n_local_maxima[in_window], width[in_window], peak_height)


def build_epoch_event_table(mice_per_group, sessions, epoch_names, get_frames_fn, mapping='full',
                            thres=None, crossreg_to_use=None):
    """
    Build a tidy (mouse, group, trial, epoch, cell) event table for one session family.

    One row per (mouse, trial, epoch, cell) combination with a valid window (get_frames_fn
    returns non-None). n_events and sum_amplitude are 0/0.0 for a cell with no qualifying events
    in that window -- they are NOT dropped, so the rate and fraction-active endpoints (which need
    every cell, per the codebase's existing zero-event-cell convention) can be computed from the
    same table the amplitude endpoint uses. Amplitude-specific consumers must filter to
    n_events > 0 themselves (see filter_amplitude_rows()) -- mean amplitude is undefined for a
    cell with zero events, which is definitional, not a missing-data problem.

    mice_per_group : dict of group -> list of mouse IDs. Drives iteration order and the 'group'
                     column; only mice present in `sessions` are used.
    sessions       : dict of mouse -> session object (e.g. ds.TFC_cond or ds.Test_B).
    epoch_names    : iterable of epoch name strings to compute (e.g. TFC_EPOCHS or TESTB_EPOCHS).
    get_frames_fn  : callable(session, epoch_name, trial_idx) -> (onset, offset) or None. Pass
                     functools.partial(get_epoch_frames, ...) for TFC_cond, or
                     functools.partial(get_testb_epoch_frames,
                                       post_tone_duration_s=TESTB_POST_TONE_DURATION_S)
                     for Test_B/Test_B_1wk.
    mapping        : cross-registration mapping string, or 'full'.
    thres          : per-cell deconvolution threshold override; defaults to session.thres.
    crossreg_to_use: passed through to get_S_mapping's with_crossreg.

    Returns a tidy pandas.DataFrame:
      mouse, group, trial, epoch, cell, n_events, sum_amplitude, exposure_seconds

    ** 'cell' is SESSION-LOCAL -- never join it across sessions. ** See _iter_event_windows() and
    compute_lt1_lt2_amplitude_delta() for the cross-session row-order-not-id-equality rationale.
    """
    rows = []
    for (group, mouse, trial_idx, epoch, cell_id, exposure_s, frameidx, amplitude,
        _n_local_maxima, _width, _peak_height) in _iter_event_windows(
            mice_per_group, sessions, epoch_names, get_frames_fn, mapping=mapping, thres=thres,
            crossreg_to_use=crossreg_to_use):
        n_events = int(len(frameidx))
        sum_amp = float(np.sum(amplitude)) if n_events else 0.0
        rows.append({
            'mouse': mouse, 'group': group, 'trial': trial_idx, 'epoch': epoch,
            'cell': cell_id, 'n_events': n_events, 'sum_amplitude': sum_amp,
            'exposure_seconds': exposure_s,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError('build_epoch_event_table: produced an empty table -- check '
                           'mapping/epoch_names/get_frames_fn inputs.')
    return df


def _run_rows_from_window(rows, group, mouse, trial_idx, epoch, cell_id, frameidx, amplitude,
                          n_local_maxima, width, peak_height):
    """Append build_run_structure_table()'s per-run row schema for one window's detected runs.
    Factored out so build_run_structure_table() and build_epoch_and_run_tables() build IDENTICAL
    rows from the same _iter_event_windows() fields, whether iterated separately or together."""
    for fi, amp, nlm, w, ph in zip(frameidx, amplitude, n_local_maxima, width, peak_height):
        rows.append({'mouse': mouse, 'group': group, 'trial': trial_idx, 'epoch': epoch,
                    'cell': cell_id, 'width_frames': int(w), 'n_local_maxima': int(nlm),
                    'amplitude': float(amp), 'peak_height': float(ph), 'peak_frame': int(fi)})


def build_run_structure_table(mice_per_group, sessions, epoch_names, get_frames_fn, mapping='full',
                              thres=None, crossreg_to_use=None):
    """
    Plan section 4: one row per individual detected event RUN (not per cell-trial-epoch
    aggregate, unlike build_epoch_event_table) -- the run-width/local-maxima bursting evidence
    needs each run's own width and local-maxima count, which build_epoch_event_table's per-window
    sum discards.

    STANDALONE, this runs its OWN _iter_event_windows() pass (hence its own
    find_event_runs_ca_S() detection per mouse) -- if a caller also needs
    build_epoch_event_table()'s table for the SAME mice/epoch_names/mapping/thres, use
    build_epoch_and_run_tables() instead to get both from a single shared traversal (this is
    exactly what run_sp_rates_lmm does for TFC_cond's main pass).

    Returns a tidy DataFrame: mouse, group, trial, epoch, cell, width_frames, n_local_maxima,
    amplitude, peak_height, peak_frame -- one row per run (a cell-trial-epoch window with 3
    detected runs contributes 3 rows). No zero-event bookkeeping: a window with no runs
    contributes no rows here, unlike build_epoch_event_table's explicit zero-event rows (needed
    there for the rate/fraction-active denominators, which this table is not used for).
    """
    rows = []
    for (group, mouse, trial_idx, epoch, cell_id, _exposure_s, frameidx, amplitude,
        n_local_maxima, width, peak_height) in _iter_event_windows(
            mice_per_group, sessions, epoch_names, get_frames_fn, mapping=mapping, thres=thres,
            crossreg_to_use=crossreg_to_use):
        _run_rows_from_window(rows, group, mouse, trial_idx, epoch, cell_id, frameidx, amplitude,
                              n_local_maxima, width, peak_height)
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError('build_run_structure_table: produced an empty table -- check '
                           'mapping/epoch_names/get_frames_fn inputs.')
    return df


def build_epoch_and_run_tables(mice_per_group, sessions, epoch_names, get_frames_fn, mapping='full',
                               thres=None, crossreg_to_use=None):
    """
    Single-pass builder for BOTH build_epoch_event_table()'s aggregate table and
    build_run_structure_table()'s per-run table, sharing ONE _iter_event_windows() traversal (and
    therefore one find_event_runs_ca_S() detection pass per mouse) instead of two. Calling the
    two standalone builders back to back over the SAME mice/epoch_names/mapping/thres -- as an
    earlier version of run_sp_rates_lmm did for TFC_cond's main event table and its run-structure
    table -- silently re-runs the identical (and dominant-cost) per-cell event detection twice.
    Prefer this whenever both tables are needed for the same epoch set; fall back to the
    standalone builders only when just one table is needed (e.g. Test_B/Test_B_1wk, which never
    need the run-structure table) or when a DIFFERENT threshold is required per table (e.g.
    run_threshold_sensitivity, which cannot share a detection pass with the thres=None default
    pass regardless).

    Returns (df_fine, df_runs) -- exactly build_epoch_event_table()'s and
    build_run_structure_table()'s own return values.
    """
    epoch_rows, run_rows = [], []
    for (group, mouse, trial_idx, epoch, cell_id, exposure_s, frameidx, amplitude,
        n_local_maxima, width, peak_height) in _iter_event_windows(
            mice_per_group, sessions, epoch_names, get_frames_fn, mapping=mapping, thres=thres,
            crossreg_to_use=crossreg_to_use):
        n_events = int(len(frameidx))
        sum_amp = float(np.sum(amplitude)) if n_events else 0.0
        epoch_rows.append({
            'mouse': mouse, 'group': group, 'trial': trial_idx, 'epoch': epoch,
            'cell': cell_id, 'n_events': n_events, 'sum_amplitude': sum_amp,
            'exposure_seconds': exposure_s,
        })
        _run_rows_from_window(run_rows, group, mouse, trial_idx, epoch, cell_id, frameidx,
                              amplitude, n_local_maxima, width, peak_height)
    df_fine = pd.DataFrame(epoch_rows)
    if df_fine.empty:
        raise RuntimeError('build_epoch_and_run_tables: produced an empty epoch table -- check '
                           'mapping/epoch_names/get_frames_fn inputs.')
    df_runs = pd.DataFrame(run_rows)
    if df_runs.empty:
        raise RuntimeError('build_epoch_and_run_tables: produced an empty run table -- check '
                           'mapping/epoch_names/get_frames_fn inputs.')
    return df_fine, df_runs


def aggregate_over_trials(df, epoch):
    """
    Pool a fine (mouse, group, trial, epoch, cell) event table down to one row per
    (mouse, group, cell) for a single epoch, SUMMING n_events/sum_amplitude/exposure_seconds
    across trials.

    Summing (not averaging per-trial means) so the pooled mean_amplitude = sum_amplitude /
    n_events over the epoch is exactly the trial-pooled per-event mean -- an average of per-trial
    averages would instead give every trial equal weight regardless of how many events it
    contributed, which is not what "pooled across the five conditioning trials" (plan's primary
    endpoint definition) means.

    Used to build the PRIMARY endpoint's cell-level trace table, and (per non-reference epoch,
    including TFC_REFERENCE_EPOCH itself) the CO-PRIMARY within-cell delta table -- see
    compute_epoch_delta_table().
    """
    sub = df[df['epoch'] == epoch]
    if sub.empty:
        raise RuntimeError(f'aggregate_over_trials: no rows for epoch={epoch!r}.')
    agg = sub.groupby(['mouse', 'group', 'cell'], as_index=False).agg(
        n_events=('n_events', 'sum'),
        sum_amplitude=('sum_amplitude', 'sum'),
        exposure_seconds=('exposure_seconds', 'sum'),
    )
    return agg


def filter_amplitude_rows(df):
    """
    Restrict an event table (fine or trial-pooled) to rows with >=1 event and attach
    'mean_amplitude' and 'log_amplitude' columns.

    Amplitude is conditional on an event by construction, so a row with zero events has no
    defined mean amplitude and is EXCLUDED here, not imputed. This is definitional, not a
    missing-data problem -- the rate/fraction-active endpoints (built from the table BEFORE this
    filter) are what carries the zero-event cells forward.
    """
    out = df[df['n_events'] > 0].copy()
    if out.empty:
        raise RuntimeError('filter_amplitude_rows: no rows with >=1 event -- check thres/window.')
    out['mean_amplitude'] = out['sum_amplitude'] / out['n_events']
    out['log_amplitude'] = np.log(out['mean_amplitude'])
    return out


def build_mouse_trial_epoch_rate_table(df_fine):
    """
    Aggregate a fine (mouse, group, trial, epoch, cell) event table to (mouse, group, trial,
    epoch) by summing n_events across cells (Y_mte in the plan's notation) and SUMMING
    exposure_seconds across cells too (E_mte = sum over cells of T_mcte -- 'total valid
    cell-seconds', matching sp_rates_lmm_methods.md). This is the population-level table the
    SECONDARY rate endpoint is fit on -- see fit_rate_group_epoch_model() -- not the cell-level
    table the amplitude endpoints use.

    PLAN SECTION 1.2 FIX: exposure_seconds used to be taken with 'first' (a single cell's window
    duration), while n_events was already summed across ALL cells -- so the offset's numerator
    was population-level but its denominator was one cell's window, missing a factor of n_cells.
    Since log(n_cells * T) = log(n_cells) + log(T) and n_cells is GROUP-CORRELATED (hM3D
    contributes more amplitude-active cells/mouse than the other groups), the missing
    log(n_cells) term leaked into the group fixed effect rather than being absorbed by the mouse
    random intercept (a shrunk random effect, not a free per-mouse parameter). Since every cell
    in one mouse-trial-epoch window shares the SAME window duration T, summing exposure_seconds
    across the n_cells rows in that window is exactly n_cells * T -- see the E_mte ==
    n_cells_in_window * window_seconds assertion this module's verification runs.
    """
    agg = df_fine.groupby(['mouse', 'group', 'trial', 'epoch'], as_index=False).agg(
        n_events=('n_events', 'sum'),
        exposure_seconds=('exposure_seconds', 'sum'),
    )
    return agg


def build_mouse_trial_trace_amplitude(df_fine):
    """
    Plan section 5 (group x trial / photobleaching check): per-mouse, per-trial mean log
    per-event amplitude in the TRACE epoch, collapsing cells within the mouse.

    This -- not a cell x trial table with only a mouse random intercept -- is the correct unit
    for a group x TRIAL interaction. Trial (unlike epoch, which section 2 handles by within-cell
    differencing) is tested here at the level where its within-mouse repeated-measures structure
    is honoured without reintroducing cell-level pseudoreplication: a cell contributing up to 5
    trial rows to a model with only a mouse random intercept would repeat the co-primary's
    original bug (see compute_epoch_delta_table()'s docstring) exactly. Collapsing to one
    log-amplitude value per (mouse, trial) first removes that risk by construction; the trial
    factor is then a proper within-mouse repeated measure with no remaining cell-level nesting.

    Returns one row per (mouse, group, trial): mouse, group, trial, log_amplitude (the mean, over
    that mouse-trial's own event-active cells, of log per-event amplitude).
    """
    sub = df_fine[df_fine['epoch'] == TFC_TRACE_EPOCH]
    amp = filter_amplitude_rows(sub)
    return amp.groupby(['mouse', 'group', 'trial'], as_index=False)['log_amplitude'].mean()


def compute_epoch_delta_table(df_fine, epoch, reference_epoch=TFC_REFERENCE_EPOCH):
    """
    CO-PRIMARY endpoint's cell selection + delta computation (plan section 2). For each cell,
    pool n_events/sum_amplitude/exposure_seconds across all trials separately for `epoch` and
    `reference_epoch` (aggregate_over_trials, matching the primary endpoint's own trial-pooling),
    keep only cells with >=1 event in BOTH pooled epochs (filter_amplitude_rows on each side,
    then inner-join on (mouse, group, cell)), and compute
    delta_log_amplitude = log_amplitude[epoch] - log_amplitude[reference_epoch].

    This REPLACES a group x epoch interaction fit directly on the cell x trial x epoch table
    (only a mouse random intercept, no cell nesting -- a cell could contribute up to 20 rows
    there, up to 5 trials x 4 epochs, which is exactly the pseudoreplication the reviewer flagged:
    "a mouse random intercept alone does not represent within-cell repetition"). Differencing
    within-cell cancels each cell's own baseline level -- the dominant contaminating source -- by
    construction, the same logic already used and validated in compute_lt1_lt2_amplitude_delta().
    The resulting one-row-per-cell delta table converts what was an epoch x group INTERACTION
    into a plain group MAIN-EFFECT test on a per-cell contrast, i.e. a BETWEEN-mouse comparison,
    structurally identical to the primary endpoint (see fit_epoch_delta_model) -- which is also
    why joint_wald_test's n_groups-1 denominator df is exactly appropriate here, not merely a
    conservative stand-in (contrast plan section 1.1's note on the group x trial check in
    fit_group_trial_model, where trial genuinely IS a within-mouse factor and n_groups-1 remains
    only a documented conservative approximation).

    Cell selection this induces: only cells active in BOTH `epoch` and `reference_epoch` qualify
    -- callers should report len(returned df) (see run_sp_rates_lmm's coprimary stats file).

    Returns a tidy DataFrame: mouse, group, cell, log_amplitude_epoch, log_amplitude_reference,
    delta_log_amplitude.
    """
    epoch_amp = filter_amplitude_rows(aggregate_over_trials(df_fine, epoch))[
        ['mouse', 'group', 'cell', 'log_amplitude']]
    ref_amp = filter_amplitude_rows(aggregate_over_trials(df_fine, reference_epoch))[
        ['mouse', 'group', 'cell', 'log_amplitude']]
    merged = epoch_amp.merge(ref_amp, on=['mouse', 'group', 'cell'],
                             suffixes=('_epoch', '_reference'))
    if merged.empty:
        raise RuntimeError(f'compute_epoch_delta_table: no cells with >=1 event in BOTH '
                           f'{epoch!r} and {reference_epoch!r}.')
    merged['delta_log_amplitude'] = merged['log_amplitude_epoch'] - merged['log_amplitude_reference']
    return merged


def fraction_active_table(df_epoch_slice):
    """
    Prespecified secondary endpoint: fraction of cells with >=1 event, per mouse, for one epoch
    slice (output of aggregate_over_trials() or a df_fine subset already restricted to one
    epoch). One row per mouse -- fraction active is inherently a per-mouse population statistic,
    unlike the per-cell amplitude/rate tables it is computed from (a single cell cannot itself be
    "20% active"), so it arrives at plot_decomposition already at the per-mouse granularity every
    panel there plots.
    """
    out = df_epoch_slice.copy()
    out['active'] = out['n_events'] > 0
    return (out.groupby(['mouse', 'group'], as_index=False)['active'].mean()
              .rename(columns={'active': 'fraction_active'}))


def mouse_level_trace_amplitude(df_amp_pooled):
    """
    Triangulation check (plan section 3): collapse a cell-level amplitude table (already passed
    through filter_amplitude_rows()) to ONE value per mouse -- the mean of that mouse's own
    cells' log_amplitude. This is NOT the model input; it exists purely so the 17-mouse-level
    plot can be checked against the cell-level model estimate, AND is now the PRIMARY reported
    point estimate's own numbers (compute_group_contrast_point_estimates -- equal-mouse-weighted,
    since cell count is group-correlated and post-treatment). If a conclusion depends on the
    model and is invisible at this level, it should not be claimed.
    """
    return df_amp_pooled.groupby(['mouse', 'group'], as_index=False)['log_amplitude'].mean()


def mouse_level_p90_amplitude(df_amp_pooled):
    """
    Plan section 6: mouse-level 90th-percentile log-amplitude, complementing
    mouse_level_trace_amplitude's mean -- the tail-shift/bursting-relevant summary, matching what
    the '..._p90' mouse-label permutation test already targets (make_contrast_stat with a
    percentile reduce_fn). One row per mouse.
    """
    out = df_amp_pooled.groupby(['mouse', 'group'], as_index=False)['log_amplitude'].quantile(0.9)
    return out.rename(columns={'log_amplitude': 'p90_log_amplitude'})


def summarize_run_structure(df_runs):
    """
    Plan section 4: per-mouse summary of a run table (build_run_structure_table's rows, normally
    already restricted to the trace epoch by the caller) -- mean run width (frames), mean
    n_local_maxima per run, and the fraction of runs with >=2 local maxima (temporally clustered
    peaks that MERGED into one run under this analysis' contiguous-run event definition -- the
    direct run-structure signature that distinguishes "genuinely burst-like" from "a threshold
    artifact"). One row per mouse.
    """
    out = df_runs.copy()
    out['multi_peak'] = out['n_local_maxima'] >= 2
    return out.groupby(['mouse', 'group'], as_index=False).agg(
        mean_width_frames=('width_frames', 'mean'),
        mean_n_local_maxima=('n_local_maxima', 'mean'),
        fraction_multi_peak=('multi_peak', 'mean'),
        n_runs=('width_frames', 'size'),
    )


def report_recall_cohort_note(mice_per_group, sessions, label):
    """
    Plan section 6 (recall N caveat): which mice are present/missing from a recall session,
    computed from the actual data rather than hardcoded -- Test_B and Test_B_1wk are each known
    to be missing a DIFFERENT mouse (both currently hM4D as of this module's baseline dataset),
    which means the 48h -> 1wk comparison is not evaluated on a fixed cohort. Re-derived here so
    this note tracks the live dataset rather than going stale.
    """
    lines = [f'{label} cohort:']
    for group in GROUP_ORDER:
        mice = mice_per_group.get(group, [])
        missing = [m for m in mice if m not in sessions]
        present = [m for m in mice if m in sessions]
        lines.append(f'  {group}: n={len(present)} present' +
                     (f', missing={missing}' if missing else ''))
    return '\n'.join(lines) + '\n'


# ─────────────────────────────────────────────────────────────────────────────
# Confirmatory models
# ─────────────────────────────────────────────────────────────────────────────

def _fe_names_and_params(result):
    """(fe_names, params) for a fit_mixed_model() result -- MixedLM exposes these via
    .fe_params (a Series), OLS via .params (also a Series, variance-component-free). Every call
    site in this module that needs a fixed-effect coefficient BY NAME goes through this rather
    than repeating the hasattr(result, 'fe_params') branch inline."""
    if hasattr(result, 'fe_params'):
        return list(result.fe_params.index), result.fe_params
    return list(result.params.index), result.params


def _nonref_group_coef_names(fe_names, reference):
    """Locate the fixed-effect dummy name for each non-reference group. Shared helper for
    fit_primary_trace_amplitude() and anywhere else that needs the plain group main-effect
    coefficients (not an interaction) by name rather than by guessing patsy's naming."""
    names = []
    for g in [gg for gg in GROUP_ORDER if gg != reference]:
        match = [n for n in fe_names if n.endswith(f'[T.{g}]') and ':' not in n]
        if len(match) != 1:
            raise RuntimeError(f'Could not locate FE dummy for group {g!r} in {fe_names}')
        names.append(match[0])
    return names


def _nonref_group_coefs(result, reference):
    """dict group -> (coef_name, coef_value) for every non-reference group's plain main-effect
    dummy on a fitted result -- reuses _nonref_group_coef_names for the lookup (same
    'exactly one match' guard) rather than re-deriving the name pattern inline. Shared by
    run_threshold_sensitivity() and plot_effect_forest(), which both need the coefficient VALUE
    (not just its name, which is all fit_*'s own omnibus callers need)."""
    fe_names, params = _fe_names_and_params(result)
    names = _nonref_group_coef_names(fe_names, reference)
    groups = [g for g in GROUP_ORDER if g != reference]
    return {g: (name, float(params[name])) for g, name in zip(groups, names)}


def fit_primary_trace_amplitude(df_trace_pooled, reference='mCherry'):
    """
    PRIMARY endpoint: log(mean per-event amplitude) ~ group, cell-level, trace epoch pooled
    across trials, random intercept on mouse (falls back to mouse-clustered OLS on degeneracy --
    see caban.single_unit_common.fit_mixed_model).

    df_trace_pooled : output of filter_amplitude_rows(aggregate_over_trials(df_fine, 'trace')).

    Returns dict(result, method, summary_text, omnibus, formula). omnibus is the joint Wald test
    (caban.single_unit_common.joint_wald_test, df2 = n_mice - 1) that both non-reference group
    coefficients are zero -- the single primary confirmatory p-value. This is a BETWEEN-mouse
    contrast (group does not vary within a mouse), so n_mice - 1 is the correct cluster-robust
    denominator df here, not merely a conservative stand-in.
    """
    df = df_trace_pooled.copy()
    df['group'] = pd.Categorical(df['group'], categories=[reference] + [g for g in GROUP_ORDER if g != reference])
    formula = f"log_amplitude ~ C(group, Treatment(reference='{reference}'))"
    result, method, text = fit_mixed_model(df, formula, group_col='mouse')
    fe_names, _params = _fe_names_and_params(result)
    n_groups = int(df['mouse'].nunique())
    omnibus = joint_wald_test(result, _nonref_group_coef_names(fe_names, reference), n_groups)
    return {'result': result, 'method': method, 'summary_text': text, 'omnibus': omnibus,
            'formula': formula, 'n_groups': n_groups}


def fit_epoch_delta_model(delta_df, reference='mCherry'):
    """
    CO-PRIMARY endpoint (plan section 2, replacing a group x epoch interaction fit that
    understated its own uncertainty -- see compute_epoch_delta_table()'s docstring): fit
    delta_log_amplitude ~ group, one row per cell (compute_epoch_delta_table's within-cell
    trace-minus-pre_tone delta), random intercept on mouse. Tests whether the WITHIN-CELL
    elevation of trace-epoch amplitude over each cell's own pre_tone baseline differs by group --
    a difference-in-differences design that cancels cell-level baseline variance by construction,
    structurally identical to fit_primary_trace_amplitude (a between-mouse contrast).
    """
    df = delta_df.copy()
    df['group'] = pd.Categorical(df['group'], categories=[reference] + [g for g in GROUP_ORDER if g != reference])
    formula = f"delta_log_amplitude ~ C(group, Treatment(reference='{reference}'))"
    result, method, text = fit_mixed_model(df, formula, group_col='mouse')
    fe_names, _params = _fe_names_and_params(result)
    n_groups = int(df['mouse'].nunique())
    omnibus = joint_wald_test(result, _nonref_group_coef_names(fe_names, reference), n_groups)
    return {'result': result, 'method': method, 'summary_text': text, 'omnibus': omnibus,
            'formula': formula, 'n_cells': len(df), 'n_groups': n_groups}


def compute_all_epoch_deltas(df_fine, epochs=TFC_EPOCHS, reference_epoch=TFC_REFERENCE_EPOCH,
                             confirmatory_epoch=TFC_TRACE_EPOCH, reference_group='mCherry'):
    """
    Run the co-primary within-cell delta contrast (compute_epoch_delta_table +
    fit_epoch_delta_model) for EVERY non-reference epoch, not just the confirmatory trace one.

    Why this exists: the confirmatory trace-vs-pre_tone within-cell delta came back NULL, while
    the group main effect on amplitude is large. The interpretation that follows -- hM3D elevates
    per-event amplitude GLOBALLY rather than specifically during the trace interval -- is a claim
    about ALL the epochs, so it cannot be supported by looking at only one of them. Reporting
    tone-vs-pre_tone and post_shock-vs-pre_tone alongside makes the epoch profile complete and
    lets a reader see directly that no epoch stands out, rather than taking that on trust.

    These extra epochs are DESCRIPTIVE. They spend no alpha, they are not in the confirmatory
    Holm family (which is exactly two tests: the primary and the trace delta -- see
    holm_correct_confirmatory), and they are not in the secondary BH-FDR family either
    (build_secondary_fdr_table), because their role is to characterize a null rather than to
    test a hypothesis. `confirmatory_epoch` is recorded per row purely so a reader of the output
    table can tell which single row carries inferential weight.

    Returns (summary_df, fits) where summary_df has one row per (epoch, non-reference group) with
    the effect estimate and 95% interval, and fits is dict epoch -> fit_epoch_delta_model output.
    """
    rows, fits = [], {}
    for epoch in [e for e in epochs if e != reference_epoch]:
        delta_df = compute_epoch_delta_table(df_fine, epoch, reference_epoch)
        fit = fit_epoch_delta_model(delta_df, reference=reference_group)
        fits[epoch] = fit
        mouse_delta = (delta_df.groupby(['mouse', 'group'], as_index=False)['delta_log_amplitude']
                      .mean())
        per_group = {g: mouse_delta.loc[mouse_delta['group'] == g, 'delta_log_amplitude'].to_numpy()
                     for g in GROUP_ORDER}
        contrasts = mouse_contrast_ci(per_group, reference=reference_group, scale='log')
        for group, c in contrasts.items():
            rows.append({
                'epoch': epoch, 'group': group, 'reference_epoch': reference_epoch,
                'is_confirmatory': epoch == confirmatory_epoch,
                'n_cells': fit['n_cells'], 'n_mice': c['n'],
                'delta_log': c['diff'], 'delta_log_lo': c['diff_lo'], 'delta_log_hi': c['diff_hi'],
                'ratio': c['ratio'], 'ratio_lo': c['ratio_lo'], 'ratio_hi': c['ratio_hi'],
                'omnibus_p': fit['omnibus']['p'],
            })
    return pd.DataFrame(rows), fits


def fit_epoch_interaction_nested_attempt(df_fine_amp, reference_group='mCherry',
                                         reference_epoch=TFC_REFERENCE_EPOCH, maxiter=100):
    """
    ONE-OFF confirmation fit (plan section 2's "attempt the full nested fit once" rejected
    alternative) -- NOT part of run_sp_rates_lmm's main pipeline; call this separately from a
    notebook cell and be ready to interrupt the kernel if it hangs.

    Fits the FULL group x epoch interaction on the cell x trial x epoch table with a properly
    NESTED random effect, (1|mouse) + (1|mouse:cell) via statsmodels' vc_formula, which correctly
    represents the repeated-measures structure that fit_epoch_delta_model()'s within-cell
    differencing only approximates by collapsing trial and epoch-vs-reference into one number per
    cell. At ~8,000+ cell levels over ~100k rows this is expected to be slow and may not converge
    -- statsmodels' variance-components implementation is not built for this scale. A
    clustered-OLS fallback (fit_mixed_model's usual degeneracy handling) cannot represent a nested
    random effect at all, so there is nothing sensible to fall back to here; this function reports
    failure explicitly instead.

    Returns dict(result, converged, reason). result is None and reason explains why if the fit
    failed or did not converge; report this model only when converged is True.
    """
    df = df_fine_amp.copy()
    df['group'] = pd.Categorical(df['group'], categories=[reference_group] + [g for g in GROUP_ORDER if g != reference_group])
    other_epochs = [e for e in df['epoch'].unique() if e != reference_epoch]
    df['epoch'] = pd.Categorical(df['epoch'], categories=[reference_epoch] + other_epochs)
    df['trial'] = df['trial'].astype(int)
    df['cell_uid'] = df['mouse'].astype(str) + '_' + df['cell'].astype(str)

    formula = (f"log_amplitude ~ C(group, Treatment(reference='{reference_group}')) * "
              f"C(epoch, Treatment(reference='{reference_epoch}')) + C(trial)")
    vc = {'cell': '0 + C(cell_uid)'}
    try:
        model = smf.mixedlm(formula, data=df, groups=df['mouse'], vc_formula=vc)
        result = model.fit(reml=True, method='lbfgs', maxiter=maxiter)
    except Exception as exc:
        return {'result': None, 'converged': False, 'reason': f'fit failed: {exc}'}
    if not getattr(result, 'converged', True):
        return {'result': result, 'converged': False, 'reason': 'optimizer did not converge'}
    return {'result': result, 'converged': True, 'reason': None}


def fit_group_trial_model(df_mouse_trial, reference='mCherry'):
    """
    Plan section 5 (photobleaching control): log_amplitude ~ group * C(trial) + (1|mouse) on the
    per-mouse-per-trial trace amplitude table (build_mouse_trial_trace_amplitude). The omnibus
    (joint Wald test on every group x trial interaction coefficient) tests whether the trial-wise
    decline differs by group; report_group_trial_slopes() gives per-group slopes for a directly
    interpretable number alongside the categorical omnibus.

    df2 = n_mice - 1 is used for the omnibus (caban.single_unit_common.joint_wald_test) as a
    DOCUMENTED CONSERVATIVE approximation, not the exact df: trial is a within-mouse factor here
    (unlike the primary/co-primary contrasts, which are between-mouse), so a Satterthwaite/
    Kenward-Roger denominator informed by the mouse x trial residual would in principle give more
    power. This codebase implements no such correction; every joint Wald test in this module uses
    the same conservative n_groups-1 rather than a one-off ad hoc formula for this test alone.
    This costs power, not Type I error control.
    """
    df = df_mouse_trial.copy()
    df['group'] = pd.Categorical(df['group'], categories=[reference] + [g for g in GROUP_ORDER if g != reference])
    df['trial'] = df['trial'].astype(int)
    formula = f"log_amplitude ~ C(group, Treatment(reference='{reference}')) * C(trial)"
    result, method, text = fit_mixed_model(df, formula, group_col='mouse')
    fe_names, _params = _fe_names_and_params(result)
    interaction_names = [n for n in fe_names if ':' in n]
    if len(interaction_names) == 0:
        raise RuntimeError(f'fit_group_trial_model: no interaction terms found in {fe_names}')
    n_groups = int(df['mouse'].nunique())
    omnibus = joint_wald_test(result, interaction_names, n_groups)
    return {'result': result, 'method': method, 'summary_text': text, 'omnibus': omnibus,
            'formula': formula, 'interaction_names': interaction_names, 'n_groups': n_groups}


def report_group_trial_slopes(df_mouse_trial):
    """
    Descriptive companion to fit_group_trial_model's categorical interaction: a simple per-group
    OLS slope (log_amplitude per trial) fit on each group's own mouse-averaged trial means, for a
    directly interpretable number -- the categorical group x trial coefficients test whether ANY
    trial differs from trial 0, not a single summary slope. NOT a formal test (no SE propagated
    through the two-stage mean-then-slope procedure); report alongside the omnibus, never instead
    of it.

    Returns dict group -> slope (log_amplitude units per trial).
    """
    out = {}
    for group in GROUP_ORDER:
        sub = df_mouse_trial[df_mouse_trial['group'] == group]
        trial_means = sub.groupby('trial')['log_amplitude'].mean().sort_index()
        slope, _intercept = np.polyfit(trial_means.index.to_numpy(dtype=float),
                                       trial_means.to_numpy(dtype=float), 1)
        out[group] = float(slope)
    return out


def run_threshold_sensitivity(mice_per_group, TFC_cond, tfc_frames_fn, thresholds=THRESHOLD_SENSITIVITY_VALUES,
                              mapping='full', reference='mCherry'):
    """
    Plan section 4's threshold-sensitivity check: re-fit the PRIMARY trace-amplitude contrast at
    each of `thresholds` (only the 'trace' epoch is built per threshold, not the full 4-epoch
    TFC_EPOCHS set, since that is all this check needs). If hM3D's effect survives across all
    three thresholds, run-merging (temporally adjacent peaks collapsing into one wider run at a
    looser threshold) cannot be the sole explanation for the amplitude effect.

    Returns a tidy DataFrame: thres, group, coef, ci_lo, ci_hi, n_cells -- one row per
    (threshold, non-reference group), ready for plot_threshold_sensitivity().
    """
    rows = []
    for thres in thresholds:
        df_fine_t = build_epoch_event_table(mice_per_group, TFC_cond, (TFC_TRACE_EPOCH,),
                                            tfc_frames_fn, mapping=mapping, thres=thres)
        df_trace_amp_t = filter_amplitude_rows(aggregate_over_trials(df_fine_t, TFC_TRACE_EPOCH))
        fit_t = fit_primary_trace_amplitude(df_trace_amp_t, reference=reference)
        result = fit_t['result']
        ci = result.conf_int()
        for group, (name, coef) in _nonref_group_coefs(result, reference).items():
            ci_row = ci.loc[name]
            rows.append({'thres': thres, 'group': group, 'coef': coef,
                        'ci_lo': float(ci_row.iloc[0]), 'ci_hi': float(ci_row.iloc[1]),
                        'n_cells': len(df_trace_amp_t)})
    return pd.DataFrame(rows)


def compute_group_contrast_point_estimates(df_trace_amp, reference='mCherry'):
    """
    Plan section 3: report BOTH weightings of the primary group contrast explicitly. Amplitude is
    only defined for event-active cells, and cell inclusion is post-treatment and
    activity-dependent, with cell count itself group-correlated (hM3D contributes ~1.45x as many
    active cells/mouse as the other groups) -- a pooled cell-weighted mean therefore over-weights
    mice with more active cells. The EQUAL-MOUSE-WEIGHTED contrast (mean of each mouse's own
    mean -- exactly mouse_level_trace_amplitude's own numbers, matching what the mouse-level
    figure shows) is the PRIMARY reported estimand; the cell-weighted pooled mean is reported only
    as a supporting "average detected active cell" quantity, never the headline number. Neither
    number here is itself a formal test -- fit_primary_trace_amplitude's omnibus remains that.

    Returns dict: for each non-reference group, {mouse_weighted_diff, mouse_weighted_se,
    cell_weighted_diff, n_mice_group, n_cells_group}.
    """
    mouse_df = mouse_level_trace_amplitude(df_trace_amp)
    ref_mouse_vals = mouse_df.loc[mouse_df['group'] == reference, 'log_amplitude'].to_numpy()
    ref_cell_vals = df_trace_amp.loc[df_trace_amp['group'] == reference, 'log_amplitude'].to_numpy()
    if len(ref_mouse_vals) < 2:
        raise RuntimeError(f'compute_group_contrast_point_estimates: reference group {reference} '
                           f'has {len(ref_mouse_vals)} mouse(s); need >=2 for a variance estimate.')
    out = {}
    for group in [g for g in GROUP_ORDER if g != reference]:
        g_mouse_vals = mouse_df.loc[mouse_df['group'] == group, 'log_amplitude'].to_numpy()
        g_cell_vals = df_trace_amp.loc[df_trace_amp['group'] == group, 'log_amplitude'].to_numpy()
        if len(g_mouse_vals) < 2:
            raise RuntimeError(f'compute_group_contrast_point_estimates: group {group} has '
                               f'{len(g_mouse_vals)} mouse(s); need >=2 for a variance estimate.')
        mouse_diff = float(np.mean(g_mouse_vals) - np.mean(ref_mouse_vals))
        pooled_se = float(np.sqrt(np.var(g_mouse_vals, ddof=1) / len(g_mouse_vals) +
                                  np.var(ref_mouse_vals, ddof=1) / len(ref_mouse_vals)))
        cell_diff = float(np.mean(g_cell_vals) - np.mean(ref_cell_vals))
        out[group] = {'mouse_weighted_diff': mouse_diff, 'mouse_weighted_se': pooled_se,
                     'cell_weighted_diff': cell_diff, 'n_mice_group': int(len(g_mouse_vals)),
                     'n_cells_group': int(len(g_cell_vals))}
    return out


def holm_correct_confirmatory(primary_p, coprimary_p):
    """
    Holm correction across the ENTIRE confirmatory family: the primary trace-period group
    omnibus p-value and the co-primary within-cell epoch-delta omnibus p-value
    (fit_epoch_delta_model). This is the complete confirmatory multiplicity burden -- everything
    else in this module is secondary (BH-FDR, see fdr_correct in caban.single_unit_common) or
    purely descriptive/sensitivity.

    Returns dict keyed 'trace' and 'interaction', each {p_raw, p_holm, reject} at alpha=0.05.
    """
    reject, p_holm, _, _ = multipletests([primary_p, coprimary_p], alpha=0.05, method='holm')
    return {
        'trace': {'p_raw': float(primary_p), 'p_holm': float(p_holm[0]), 'reject': bool(reject[0])},
        'interaction': {'p_raw': float(coprimary_p), 'p_holm': float(p_holm[1]), 'reject': bool(reject[1])},
    }


def build_secondary_fdr_table(secondary_pvalues, alpha=0.05):
    """
    Benjamini-Hochberg FDR across the prespecified SECONDARY family. sp_rates_lmm_methods.md has
    declared this family since the module was written; until now it was prose only and nothing
    applied a correction, so the secondary p-values were being reported uncorrected while the
    METHODS file said otherwise. This closes that gap.

    ** The family is the FREQUENTIST secondary tests ONLY. ** Membership:
      - Test_B and Test_B_1wk post-tone amplitude omnibus tests (recall complement),
      - the group x trial photobleaching interaction,
      - the run-structure mouse-label permutation tests (width, local maxima, multi-peak
        fraction, each group vs control).

    Deliberately EXCLUDED, and these exclusions are the substantive part of the declaration:

      - The two CONFIRMATORY omnibus tests. They carry their own Holm correction
        (holm_correct_confirmatory) and belong to a different family with a different error rate
        being controlled. Putting a test in two families corrects it twice.
      - The Bambi negative-binomial rate model. It reports posterior contrasts, HDIs and an
        ELPD-LOO comparison -- there is no p-value to correct, and manufacturing one to fold a
        Bayesian result into a frequentist FDR family is a category error, not a conservative
        choice. It is reported on its own terms with no BH language applied.
      - Purely descriptive quantities: the non-confirmatory within-cell epoch deltas
        (compute_all_epoch_deltas), the threshold-sensitivity forest, the LT1->LT2 dropout
        fractions, and the amplitude permutation tests that only re-express the primary contrast
        under a different weighting or a tail statistic. Correcting descriptives inflates the
        family and costs power on the tests that are actually making claims.

    secondary_pvalues : dict name -> p-value, in the order they should be reported.

    Returns a DataFrame with columns name, p_raw, q_value, reject, sorted by p_raw.
    """
    names = list(secondary_pvalues)
    if not names:
        raise ValueError('build_secondary_fdr_table: empty secondary family -- the BH declaration '
                         'in sp_rates_lmm_methods.md requires at least one member.')
    p_raw = np.array([float(secondary_pvalues[n]) for n in names])
    reject, qvalues = fdr_correct(p_raw, alpha=alpha)
    return (pd.DataFrame({'name': names, 'p_raw': p_raw, 'q_value': qvalues, 'reject': reject})
            .sort_values('p_raw', ignore_index=True))


def fit_rate_group_epoch_model(df_mte, reference_group='mCherry', reference_epoch=TFC_REFERENCE_EPOCH,
                               epoch_categories=None, draws=1000, tune=1000, chains=4, seed=0):
    """
    SECONDARY endpoint: event count ~ group * epoch + trial, with a log(exposure_seconds) offset,
    at the mouse x trial x epoch level (df_mte from build_mouse_trial_epoch_rate_table() -- whose
    exposure_seconds is now correctly SUMMED across cells, plan section 1.2).

    Negative-binomial mixed model fit via Bambi (PyMC backend) -- a genuinely joint fit, with
    dispersion (alpha) estimated together with the fixed and random effects.

    RANDOM EFFECTS: (1|mouse) + (1|mouse_trial). The mouse-trial term was added on
    methodological review. The four epochs of a single trial are not independent replicates of
    that mouse: they are four consecutive windows of one behavioural episode, sharing that
    trial's arousal/locomotor state, its position in the photobleaching decline, and whatever
    that trial's imaging conditions were. With only (1|mouse), all of that lands in the residual
    and the epoch and group x epoch terms get standard errors that are too small. `mouse_trial`
    is an explicit pasted mouse-x-trial label rather than formulae's `(1|mouse:trial)`: `trial`
    is simultaneously a NUMERIC fixed effect here (the monotonic photobleaching slope), and
    reusing the same column as a grouping factor invites it being read on the wrong scale. The
    pasted label is unambiguously categorical and means exactly the same thing. statsmodels has no NB-GLMM path (mixedlm is Gaussian-only, which the amplitude
    models above rely on since log-amplitude is continuous; GEE -- an earlier version of this
    function -- only fixes the dispersion parameter via a separate pre-estimation step rather
    than estimating it jointly), which is why this endpoint uses a different package from the
    amplitude models.

    The interaction's contribution is assessed via LOO cross-validation (expected log predictive
    density, `arviz.compare`) between this full model and a reduced model without the group x
    epoch interaction -- the natural Bayesian analog of the frequentist joint Wald test used for
    the (Holm-corrected) amplitude confirmatory family. This is NOT a p-value, and rate remains
    SECONDARY regardless of the result: no confirmatory alpha is spent here.

    IMPLEMENTATION NOTE -- reference levels: bambi/formulae does not respect a pandas
    Categorical's `categories=` order for reference-level selection the way patsy's
    C(x, Treatment(reference=...)) does when driven purely by column dtype (confirmed
    empirically: it silently fell back to something resembling alphabetical order instead). The
    formula therefore names the reference level explicitly via C(col, Treatment('level')) --
    positional, no `reference=` keyword, which is formulae's own (not patsy's) syntax -- rather
    than relying on the input DataFrame's categorical dtype.

    Returns dict(idata_full, idata_reduced, comparison, summary_full, summary_text, formula_full,
    formula_reduced). comparison is an arviz ELPD-LOO comparison table between the full and
    reduced models (see az.compare); a positive ELPD difference favouring 'full' (with an SE
    that excludes zero) is evidence the interaction improves predictive fit.
    """
    df = df_mte.copy()
    if epoch_categories is None:
        epoch_categories = [reference_epoch] + [e for e in df['epoch'].unique() if e != reference_epoch]
    df['trial'] = df['trial'].astype(int)
    df['log_exposure'] = np.log(df['exposure_seconds'].to_numpy())
    # Grouping factor for the epochs-within-a-trial random intercept (see docstring).
    df['mouse_trial'] = df['mouse'].astype(str) + '_t' + df['trial'].astype(str)

    group_term = f"C(group, Treatment('{reference_group}'))"
    epoch_term = f"C(epoch, Treatment('{reference_epoch}'))"
    random_terms = '(1|mouse) + (1|mouse_trial)'
    formula_full = (f"n_events ~ {group_term} * {epoch_term} + trial + offset(log_exposure) "
                    f"+ {random_terms}")
    formula_reduced = (f"n_events ~ {group_term} + {epoch_term} + trial + offset(log_exposure) "
                       f"+ {random_terms}")

    fit_kwargs = dict(draws=draws, tune=tune, chains=chains, random_seed=seed, progressbar=False,
                      idata_kwargs={'log_likelihood': True})

    model_full = bmb.Model(formula_full, data=df, family='negativebinomial')
    idata_full = model_full.fit(**fit_kwargs)

    model_reduced = bmb.Model(formula_reduced, data=df, family='negativebinomial')
    idata_reduced = model_reduced.fit(**fit_kwargs)

    comparison = az.compare({'group_x_epoch': idata_full, 'no_interaction': idata_reduced})

    interaction_name = f'{group_term}:{epoch_term}'
    summary_full = az.summary(idata_full, var_names=[interaction_name], filter_vars='like')

    # Convergence must be reported, not assumed. Adding the mouse_trial random intercept adds one
    # level per mouse x trial (~85 here) to a model already carrying a mouse intercept and an NB
    # dispersion, which is exactly the kind of change that can start producing divergences or
    # stuck chains -- so max r-hat, min ESS and the divergence count travel with every fit rather
    # than being something a reader has to go and check separately.
    diagnostics = {}
    for name, idata in (('full', idata_full), ('reduced', idata_reduced)):
        summ = az.summary(idata)
        diagnostics[name] = {
            'max_rhat': float(summ['r_hat'].max()),
            'min_ess_bulk': float(summ['ess_bulk'].min()),
            'n_divergent': int(idata.sample_stats['diverging'].values.sum()),
        }
    diag_text = '\n'.join(
        f"  {name}: max r_hat={d['max_rhat']:.4f}, min ess_bulk={d['min_ess_bulk']:.0f}, "
        f"divergences={d['n_divergent']}"
        f"{'   <-- CHECK: r_hat > 1.01' if d['max_rhat'] > 1.01 else ''}"
        for name, d in diagnostics.items())

    text = (f'Formula (full): {formula_full}\n'
           f'Formula (reduced, no group x epoch): {formula_reduced}\n\n'
           f'Sampler convergence (target: r_hat <= 1.01, ess_bulk >= ~400, 0 divergences):\n'
           f'{diag_text}\n\n'
           f'Interaction coefficients (posterior mean, sd, 94% HDI):\n{summary_full.to_string()}\n\n'
           f'LOO model comparison (full vs reduced; positive elpd_diff favours "full"):\n'
           f'{comparison.to_string()}\n')
    return {'idata_full': idata_full, 'idata_reduced': idata_reduced, 'comparison': comparison,
            'summary_full': summary_full, 'summary_text': text, 'diagnostics': diagnostics,
            'formula_full': formula_full, 'formula_reduced': formula_reduced}


def report_decomposition_additivity(rate_group_coef, amplitude_group_coef):
    """
    Consistency check: log(total S/sec) group coefficient should equal
    log(rate) group coefficient + log(mean amplitude) group coefficient, since
    summed S/sec = rate x mean amplitude exactly at the cell-window level (event-weighted when
    aggregated -- see sp_rates_lmm_methods.md's weighting-rule note).

    This does NOT fit a third model. It is an arithmetic check on coefficients already fit
    separately by fit_rate_group_epoch_model() and fit_primary_trace_amplitude() /
    fit_epoch_delta_model(). No significance test is attached to the sum: a proper SE for
    it needs the joint covariance of the two separately-fit models, which is not available here.
    Report the point estimates and let agreement (or disagreement) speak for itself.

    Returns dict(rate=..., amplitude=..., total_implied=..., ...).
    """
    total_implied = rate_group_coef + amplitude_group_coef
    return {'rate_coef': rate_group_coef, 'amplitude_coef': amplitude_group_coef,
            'total_implied_coef': total_implied}


# ─────────────────────────────────────────────────────────────────────────────
# Small-n inference: mouse-label permutation
# ─────────────────────────────────────────────────────────────────────────────

def mouse_label_permutation_test(stat_fn, mice_per_group, n_perm=20000, seed=0):
    """
    Monte Carlo permutation test: shuffle GROUP LABELS across mice (holding each mouse's own
    cell/event data fixed) and recompute stat_fn under each shuffle, building a null distribution
    for the observed group contrast.

    This is the small-n-appropriate inference layer for the sensitivity/tail statistics in this
    module: it makes no distributional assumption on the underlying cell-level statistic,
    respects the TRUE unit of randomization (mouse, not cell or event), and never collapses a
    cell/event-level statistic to a coarser summary before testing it -- only the group LABELS
    are permuted; every cell and event stays exactly where it was observed.

    stat_fn        : callable(mouse_to_group: dict[str, str]) -> float. Must recompute the
                     statistic entirely from mouse_to_group (e.g. re-pool per-cell rows into the
                     new group buckets and compute the contrast) -- see make_contrast_stat() /
                     make_amplitude_contrast_stat() for the standard factories.
    mice_per_group : dict of group -> list of mouse IDs (the TRUE/observed assignment). All mice
                     across all groups are pooled and re-labelled under each permutation,
                     preserving the original per-group counts (e.g. 5/6/6) -- a relabelling, not
                     an independent draw -- so every permutation is valid under exchangeability
                     of treatment assignment.
    n_perm         : number of Monte Carlo shuffles. There are ~5.7M distinct 5/6/6 relabellings
                     in a 17-mouse, three-group dataset, so this is Monte Carlo, not exact
                     enumeration -- exhaustive enumeration is not feasible when stat_fn re-touches
                     cell/event-level data on every call.
    seed           : RNG seed for reproducibility.

    Returns dict(observed, p_two_sided, n_perm, null=ndarray of length n_perm).
    """
    true_assignment = {}
    for group, mice in mice_per_group.items():
        for m in mice:
            true_assignment[m] = group
    mouse_list = list(true_assignment.keys())
    group_labels = np.array([true_assignment[m] for m in mouse_list])

    observed = stat_fn(true_assignment)

    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        shuffled = rng.permutation(group_labels)
        perm_assignment = dict(zip(mouse_list, shuffled))
        null[i] = stat_fn(perm_assignment)

    # +1/+1 (conventional Monte Carlo correction) so a finite number of draws never reports p=0.
    n_as_extreme = int(np.sum(np.abs(null) >= np.abs(observed)))
    p_two_sided = (n_as_extreme + 1) / (n_perm + 1)
    return {'observed': observed, 'p_two_sided': float(p_two_sided), 'n_perm': n_perm, 'null': null}


def make_contrast_stat(df, value_col, group_a, group_b, reduce_fn=None, weight='cell'):
    """
    General stat_fn factory for mouse_label_permutation_test(): contrasts pooled `value_col`
    between group_a and group_b under a given (possibly permuted) mouse-to-group mapping. Backs
    make_amplitude_contrast_stat() (value_col='log_amplitude') and the run-structure permutation
    tests (value_col='width_frames' / 'n_local_maxima' / 'multi_peak', plan section 4) -- same
    permutation-test logic, different column, per CLAUDE.md's dedup rule.

    df        : a table with 'mouse' and `value_col` columns (e.g. output of
                filter_amplitude_rows() or build_run_structure_table()).
    reduce_fn : callable(vals_a, vals_b) -> float. Defaults to difference in means. Pass e.g.
                ``lambda a, b: np.percentile(a, 90) - np.percentile(b, 90)`` for the tail contrast
                that most directly targets the bursting hypothesis (fattened right tail, not just
                a mean shift).
    weight    : 'cell' (default) -- every row counts once, matching the row-level unit of
                observation the confirmatory models use (a cell for amplitude, a run for
                run-structure). 'mouse' -- each mouse's own mean is computed FIRST and only those
                per-mouse means are passed to reduce_fn: the equal-mouse-weighted estimand (plan
                section 3), now the PRIMARY reported point estimate for the amplitude contrast.
                Permutation INFERENCE is valid either way (labels always permute at mouse level --
                see mouse_label_permutation_test); only what reduce_fn is computed over changes.
    """
    if reduce_fn is None:
        reduce_fn = lambda a, b: float(np.mean(a) - np.mean(b))
    if weight not in ('cell', 'mouse'):
        raise ValueError(f"make_contrast_stat: weight must be 'cell' or 'mouse', got {weight!r}")

    # Split the values by mouse ONCE, here, outside the returned closure. A permutation only ever
    # changes which GROUP a mouse is labelled with -- never which rows belong to a mouse, nor
    # their values -- so every pandas operation (.map, boolean masking, groupby) can be hoisted
    # out of the loop. Doing it per draw instead cost ~2.3 ms x 20,000 draws x several contrasts,
    # which was minutes per figure; this reduces each draw to a few array concatenations.
    mouse_col = df['mouse'].to_numpy()
    values = df[value_col].to_numpy(dtype=float)
    mice = list(dict.fromkeys(mouse_col.tolist()))  # stable first-seen order
    vals_by_mouse = {m: values[mouse_col == m] for m in mice}
    means_by_mouse = {m: float(np.mean(v)) if v.size else np.nan for m, v in vals_by_mouse.items()}

    def stat_fn(mouse_to_group):
        # .get, not [], so a mouse absent from the mapping falls into neither group -- matching
        # the previous .map()-based behaviour, where an unmapped mouse became NaN and was
        # excluded from both sides rather than raising.
        a_mice = [m for m in mice if mouse_to_group.get(m) == group_a]
        b_mice = [m for m in mice if mouse_to_group.get(m) == group_b]
        if not a_mice or not b_mice:
            return np.nan
        if weight == 'cell':
            vals_a = np.concatenate([vals_by_mouse[m] for m in a_mice])
            vals_b = np.concatenate([vals_by_mouse[m] for m in b_mice])
        else:
            vals_a = np.array([means_by_mouse[m] for m in a_mice], dtype=float)
            vals_b = np.array([means_by_mouse[m] for m in b_mice], dtype=float)
        if vals_a.size == 0 or vals_b.size == 0:
            return np.nan
        return float(reduce_fn(vals_a, vals_b))
    return stat_fn


def make_amplitude_contrast_stat(df_amp, group_a, group_b, reduce_fn=None, weight='cell'):
    """Thin wrapper over make_contrast_stat() fixing value_col='log_amplitude'. See its
    docstring for the weight= option (plan section 3's equal-mouse-weighted estimand).

    df_amp : an amplitude table already restricted to n_events > 0 and carrying 'log_amplitude'
             (output of filter_amplitude_rows())."""
    return make_contrast_stat(df_amp, 'log_amplitude', group_a, group_b, reduce_fn=reduce_fn, weight=weight)


# ─────────────────────────────────────────────────────────────────────────────
# Detection-bias measurement and LT1->LT2 manipulation check (plan section 5 / section 2)
# ─────────────────────────────────────────────────────────────────────────────

def compute_lt1_lt2_dropout(mice_per_group, lt1_sessions, mapping='LT1+LT2', crossreg_to_use=None):
    """
    Tier-1 detection-bias measurement: fraction of a mouse's LT1-detected cells that were NOT
    successfully cross-registered into LT2, by group. LT1 is drug-free and LT2 is on CNO,
    recorded the same day in the same FOV minutes apart, so drift/registration failure should be
    small and roughly group-independent -- an elevated dropout in hM4D relative to mCherry is
    evidence of CNO-induced (rather than purely technical) cell loss, quantifying rather than
    merely gesturing at the activity-dependent detection bias.

    lt1_sessions : dict of mouse -> LT1 session object (drug-free).

    Returns a tidy DataFrame: mouse, group, n_cells_lt1_full, n_cells_survived_to_lt2,
    dropout_fraction.
    """
    rows = []
    for group in GROUP_ORDER:
        for mouse in mice_per_group.get(group, []):
            if mouse not in lt1_sessions:
                continue
            session = lt1_sessions[mouse]
            [_S_full, _, _, S_idx_full] = session.get_S_mapping('full')
            [_S_cr, _, _, S_idx_survived] = session.get_S_mapping(mapping, with_crossreg=crossreg_to_use)
            n_full = len(S_idx_full)
            n_survived = len(S_idx_survived)
            if n_full == 0:
                raise RuntimeError(f'compute_lt1_lt2_dropout: mouse {mouse} has 0 LT1 cells.')
            rows.append({
                'mouse': mouse, 'group': group,
                'n_cells_lt1_full': n_full, 'n_cells_survived_to_lt2': n_survived,
                'dropout_fraction': 1.0 - n_survived / n_full,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError('compute_lt1_lt2_dropout: produced an empty table.')
    return df


def compute_lt1_lt2_amplitude_delta(mice_per_group, lt1_sessions, lt2_sessions, mapping='LT1+LT2',
                                    thres=None, crossreg_to_use=None):
    """
    Manipulation check: within-cell delta log(mean event amplitude), LT2 minus LT1, on the
    cross-registered 'LT1+LT2' cell set. Same day, same task, same FOV. LT1 always precedes LT2,
    so session order, elapsed time, habituation, and photobleaching differ between them too, not
    only CNO -- this within-cell delta removes each cell's own baseline, and the group CONTRAST on
    it (a difference-in-differences against mCherry, which experiences the same order/time/
    bleaching confounds) is what isolates the CNO-specific effect, not the raw LT2-minus-LT1
    delta alone. This establishes the tool works; it is NOT a test of the memory hypothesis, so it
    needs no multiplicity correction with the confirmatory family.

    Cells with zero events in EITHER session are excluded -- the delta is undefined for them; a
    cell with events in LT1 and none in LT2 is exactly the dropout phenomenon
    compute_lt1_lt2_dropout() measures, not something this paired amplitude check can express.

    Returns a tidy DataFrame: mouse, group, cell_lt1, cell_lt2, log_amplitude_lt1,
    log_amplitude_lt2, delta_log_amplitude. Both unit_ids are kept because a registered cell has
    a DIFFERENT unit_id in each session -- carrying both makes each paired row traceable back to
    the right cell in either session.
    """
    rows = []
    for group in GROUP_ORDER:
        for mouse in mice_per_group.get(group, []):
            if mouse not in lt1_sessions or mouse not in lt2_sessions:
                continue
            lt1 = lt1_sessions[mouse]
            lt2 = lt2_sessions[mouse]
            cell_thres_lt1 = lt1.thres if thres is None else thres
            cell_thres_lt2 = lt2.thres if thres is None else thres

            [S1, _, _, idx1] = lt1.get_S_mapping(mapping, with_crossreg=crossreg_to_use)
            [S2, _, _, idx2] = lt2.get_S_mapping(mapping, with_crossreg=crossreg_to_use)
            # The cross-session correspondence is ROW ORDER, not unit_id equality: the same
            # physical cell carries a DIFFERENT unit_id in LT1 than in LT2, and get_S_mapping
            # returns each session's own ids (its own column of the crossreg table). Position k
            # of idx1 and position k of idx2 are the same registered cell -- which is exactly
            # why get_actual_cells_from_df_session() must never sort ("OMG NO!!! Never sort
            # this!!! Otherwise lose cross-reg assignments!!!", caban/utilities.py). So pair by
            # position and assert only that both sessions resolved the same number of registered
            # cells; unequal lengths would mean the two sessions resolved different crossreg
            # objects or mappings, which WOULD silently misalign every pairing below.
            if len(idx1) != len(idx2):
                raise RuntimeError(
                    f'compute_lt1_lt2_amplitude_delta: {mouse} resolved {len(idx1)} LT1 cells but '
                    f'{len(idx2)} LT2 cells for mapping={mapping!r}. A cross-registration mapping '
                    f'must yield one row per registered cell in both sessions; unequal counts mean '
                    f'the two sessions used different crossreg objects or mappings.')

            events1 = find_event_runs_ca_S(S1, cell_thres_lt1)
            events2 = find_event_runs_ca_S(S2, cell_thres_lt2)

            for cell_row, (cell_id_lt1, cell_id_lt2) in enumerate(zip(idx1, idx2)):
                _f1, amp1, _n1, _w1, _s1 = events1[cell_row]
                _f2, amp2, _n2, _w2, _s2 = events2[cell_row]
                if len(amp1) == 0 or len(amp2) == 0:
                    continue
                log_amp1 = float(np.log(np.mean(amp1)))
                log_amp2 = float(np.log(np.mean(amp2)))
                rows.append({
                    'mouse': mouse, 'group': group,
                    'cell_lt1': cell_id_lt1, 'cell_lt2': cell_id_lt2,
                    'log_amplitude_lt1': log_amp1, 'log_amplitude_lt2': log_amp2,
                    'delta_log_amplitude': log_amp2 - log_amp1,
                })
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError('compute_lt1_lt2_amplitude_delta: produced an empty table.')
    return df


def fit_lt1_lt2_manipulation_check(df_delta, reference='mCherry'):
    """
    Fit delta_log_amplitude ~ group on the LT1->LT2 within-cell paired table, random intercept on
    mouse. No confirmatory multiplicity applies -- this is a manipulation check, not a test of
    the memory hypothesis (see compute_lt1_lt2_amplitude_delta()).
    """
    df = df_delta.copy()
    df['group'] = pd.Categorical(df['group'], categories=[reference] + [g for g in GROUP_ORDER if g != reference])
    formula = f"delta_log_amplitude ~ C(group, Treatment(reference='{reference}'))"
    result, method, text = fit_mixed_model(df, formula, group_col='mouse')
    fe_names, _params = _fe_names_and_params(result)
    n_groups = int(df['mouse'].nunique())
    omnibus = joint_wald_test(result, _nonref_group_coef_names(fe_names, reference), n_groups)
    return {'result': result, 'method': method, 'summary_text': text, 'omnibus': omnibus,
            'formula': formula, 'n_groups': n_groups}


# ─────────────────────────────────────────────────────────────────────────────
# Figures (one file per panel; assembled by the caller)
# ─────────────────────────────────────────────────────────────────────────────

_PANEL_FIGSIZE = (2.4, 3.2)
_YLABEL_LOG_AMPLITUDE = 'log(mean per-event amplitude)'


def _save_panel(fig, save_dir, filename_root):
    ensure_dirs(save_dir)
    _copy_analysis_methods_template(METHODS_FILENAME, save_dir)
    save_fig(fig, os.path.join(save_dir, filename_root + '.png'))
    plt.close(fig)


def _panel_stat_fn():
    """do_pairwise_holm_plot bound to this module's PANEL_HOLM_FAMILY, as a stat_fn for
    _draw_violin_triplet / draw_superplot_triplet. A single place where the family choice is
    applied, so no panel can quietly use a different one from its neighbours."""
    return functools.partial(do_pairwise_holm_plot, holm_family=PANEL_HOLM_FAMILY)


def _mouse_values_per_group(df, value_col, group_order=DREADD_DISPLAY_ORDER, panel_name='panel'):
    """Violin-ready {group: (n_mice, 1) array of that group's per-mouse means} from a cell- or
    run-level table carrying 'mouse'/'group'/`value_col`. Every group-comparison panel in this
    module builds its input through here, so the cell -> mouse collapse happens identically
    everywhere and each panel gets the same >=2-mice guard.

    ** The per-mouse mean IS the unit of INFERENCE everywhere in this module. ** The unit of
    DISPLAY varies by panel: this helper feeds the mouse-level violin panels, and
    _draw_cell_superplot_panel is its cell-level counterpart, consuming the same tidy frame.

    A panel stays mouse-level here for one of two reasons. Either the quantity only exists per
    mouse (fraction active and LT1->LT2 dropout are proportions computed OVER a mouse's cells, so
    there is no per-cell value to draw; likewise a per-mouse 90th percentile), or the panel is a
    small multiple where a cell cloud would not survive the panel size (the run-structure row,
    whose full per-run distribution is written to stats/run_structure.txt instead).

    The one thing that is NOT a reason: an unreadable axis. The first attempt at the cell-level
    variant used a LINEAR autoscaled axis, and on these long-right-tailed quantities (per-event
    amplitude, event rate, total S/s) a handful of extreme cells set the y-range and compressed
    all 17 mouse means into the bottom ~6-13% of the panel. That was an axis bug, not an argument
    against showing cells -- see draw_superplot_triplet and _resolve_superplot_yscale in
    caban.single_unit_common for the log/symlog rule that fixed it.
    """
    out = {}
    for group in group_order:
        vals = df[df['group'] == group].groupby('mouse')[value_col].mean().to_numpy()
        if len(vals) < 2:
            raise RuntimeError(f'{panel_name}: group {group} has {len(vals)} mouse(s) for '
                               f'{value_col!r}; need >=2 for a group comparison.')
        out[group] = vals.reshape(-1, 1)
    return out


def _panel_contrasts(df_or_mouse_df, value_col, ci_scale, ci_unit):
    """{'scale', 'unit', 'contrasts'} for one panel's equal-mouse-weighted contrasts vs control.

    Kept separate from the drawing so a panel's estimates can be written to a companion file
    without being rendered onto the panel itself -- five interval blocks crowded into one figure
    row was strictly harder to read than five asterisks (see plot_decomposition's docstring)."""
    per_group = _mouse_values_per_group(df_or_mouse_df, value_col, group_order=GROUP_ORDER,
                                        panel_name='panel_contrasts')
    return {'scale': ci_scale, 'unit': ci_unit,
            'contrasts': mouse_contrast_ci({g: v.ravel() for g, v in per_group.items()},
                                           scale=ci_scale)}


def write_decomposition_contrasts_markdown(contrasts_by_panel, save_dir, filename):
    """Write the decomposition figure's per-panel effect estimates + 95% intervals to a markdown
    file next to the figure.

    These numbers matter -- a null is only interpretable together with what its interval still
    admits, and a rate ratio needs its absolute difference beside it -- but they are reference
    material, not something to read off a panel. On the figure they competed with the data;
    here they can be read properly and quoted directly.
    """
    lines = [
        '# Decomposition panel contrasts',
        '',
        'Equal-mouse-weighted contrasts against the mCherry control (n = 5 hM3D / 6 hM4D / '
        '6 mCherry), Welch two-sample 95% intervals computed from the per-mouse means. The '
        'cell-level clouds on the figure are descriptive and never enter these numbers.',
        '',
        'Ratios and absolute differences are reported together: a fold-change computed off a '
        'small base overstates the practical size of a change.',
        '',
        '**These intervals are not multiplicity-corrected.** The asterisks on the figure are '
        f'Holm-corrected across the family set by `PANEL_HOLM_FAMILY` (currently '
        f'`{PANEL_HOLM_FAMILY!r}`), so a contrast whose interval excludes 1.0 here may still '
        'carry no star. Both are reported deliberately: the interval describes the effect, the '
        'star describes the corrected decision.',
        '',
    ]
    for panel_label, payload in contrasts_by_panel.items():
        lines.append(f'## {panel_label}')
        lines.append('')
        if payload['scale'] == 'log':
            lines.append('*Values are already log-transformed; the ratio is exp(difference).*')
            lines.append('')
        for line in format_contrast_ci_lines(payload['contrasts'], 'mCherry', unit=payload['unit']):
            lines.append(f'- {line}' if not line.startswith('  ') else f'  - {line.strip()}')
        lines.append('')
    ensure_dirs(save_dir)
    write_text(os.path.join(save_dir, filename), '\n'.join(lines))


def _draw_mouse_violin_panel(ax, values_per_group, ylabel, title=None,
                             group_order=DREADD_DISPLAY_ORDER, label_size='small',
                             annotate='stats', ci_scale='linear', ci_unit='',
                             bracket_mode='data'):
    """Draw one mouse-level violin+scatter panel -- the established DREADD comparison style
    (caban.analysis._draw_violin_triplet), in CLAUDE.md's mCherry/hM3D/hM4D display order. Shared
    by every mouse-level group-comparison panel below so the spine/tick/label/title treatment is
    identical across panels and figures.

    annotate : 'stats' -- Holm-corrected pairwise brackets + significance stars, the established
               style. 'ci' -- effect estimates + 95% intervals drawn onto the panel; prefer
               write_decomposition_contrasts_markdown() to a companion file instead, which is
               what the decomposition figure does.
    bracket_mode : 'data' -- brackets positioned in data coordinates by _draw_violin_triplet's
               own stat_fn path (the established look for the standalone panels).
               'axes' -- brackets positioned in axes fractions (annotate_pairwise_brackets), so
               this panel's brackets sit at the SAME height as its neighbours' in a multi-panel
               row whose other panels are on log axes. Used by the decomposition figure.
    ci_scale, ci_unit : passed to annotate_contrast_ci when annotate='ci'. Note ci_scale='log'
               means "these values are ALREADY log-transformed", not "put the axis on a log
               scale" -- same distinction as _draw_cell_superplot_panel's yscale.
    """
    ax.spines[['right', 'top']].set_visible(False)
    draw_brackets_here = annotate == 'stats' and bracket_mode == 'data'
    _draw_violin_triplet(ax, values_per_group, 0, group_order, GROUP_COLOURS,
                         stat_fn=(_panel_stat_fn() if draw_brackets_here else no_stat_annotation),
                         ylabel=ylabel)
    if annotate == 'stats' and bracket_mode == 'axes':
        annotate_pairwise_brackets(ax, {g: v.ravel() for g, v in values_per_group.items()},
                                   group_order, _panel_stat_fn())
    elif annotate == 'ci':
        annotate_contrast_ci(ax, {g: v.ravel() for g, v in values_per_group.items()},
                             scale=ci_scale, unit=ci_unit)
    ax.set_xticks(range(len(group_order)))
    ax.set_xticklabels([GROUP_LABELS[g] for g in group_order], size=label_size)
    if title is not None:
        ax.set_title(title, size='small')


def _draw_cell_superplot_panel(ax, df, value_col, ylabel, panel_name, title=None,
                               group_order=DREADD_DISPLAY_ORDER, label_size='small',
                               yscale='auto', y_quantum=None, annotate='stats',
                               ci_scale=None, ci_unit='', bracket_mode='axes'):
    """Draw one CELL-level SuperPlot panel: every cell shown, coloured by mouse, per-mouse means
    overlaid as large markers, statistics computed from the mouse means only.

    The cell-level counterpart of _draw_mouse_violin_panel, taking the same tidy per-cell frame
    (columns: group, mouse, ``value_col``) that _mouse_values_per_group consumes, so the two are
    interchangeable per panel. Axis scale is delegated to draw_superplot_triplet's ``yscale='auto'``
    -- essential here, since a linear axis on these heavy-tailed quantities is exactly what made
    the first attempt at this figure unreadable (see plot_decomposition's docstring).

    ** Pass yscale='linear' for a value that is ALREADY log-transformed. ** 'auto' inspects the
    values, and a column of log-amplitudes is all-positive, so 'auto' would apply a SECOND log and
    plot log10(ln(amplitude)) -- a meaningless double transform whose axis reads in powers of ten
    of a natural log. A pre-logged column on a linear axis already IS a log display, which is why
    such a panel renders correctly where the raw-scale ones do not. ``ci_scale='log'`` is the
    same statement made to the annotation ("already logged"), and the two travel together.

    y_quantum : sub-quantum vertical spread for a DISCRETE per-cell quantity -- pass
                1/exposure_seconds for an event-rate column, whose per-cell values are a small
                integer count over a fixed window and therefore land on a handful of hard
                horizontal lines. Display only; see draw_superplot_triplet's y_quantum.
    annotate  : 'stats' (Holm-corrected significance brackets) or 'ci' (effect estimates + 95%
                intervals drawn onto the panel). 'stats' is the default and what the
                decomposition figure uses; its estimates/intervals go to a companion markdown
                file instead (write_decomposition_contrasts_markdown).
    bracket_mode : always 'axes' in practice -- annotate_pairwise_brackets positions brackets in
                axes fractions, which is what makes them drawable on this panel's log/symlog
                axis at all. Exposed only for symmetry with _draw_mouse_violin_panel.
    """
    if bracket_mode != 'axes':
        raise ValueError("_draw_cell_superplot_panel: bracket_mode must be 'axes' -- "
                         "data-coordinate brackets cannot be positioned on a log/symlog axis.")
    ax.spines[['right', 'top']].set_visible(False)
    cell_values, mouse_means = {}, {}
    for group in group_order:
        sub = df[df['group'] == group]
        mice = sorted(sub['mouse'].unique())
        if len(mice) < 2:
            raise RuntimeError(f'{panel_name}: group {group} has {len(mice)} mouse/mice for '
                               f'{value_col!r}; need >=2 for a group comparison.')
        cell_values[group] = {m: sub.loc[sub['mouse'] == m, value_col].to_numpy() for m in mice}
        mouse_means[group] = {m: float(np.mean(cell_values[group][m])) for m in mice}
    draw_superplot_triplet(ax, cell_values, mouse_means, group_order, GROUP_COLOURS,
                           stat_fn=_panel_stat_fn(), ylabel=ylabel, yscale=yscale,
                           y_quantum=y_quantum, annotate=annotate, ci_scale=ci_scale,
                           ci_unit=ci_unit)
    ax.set_xticks(range(len(group_order)))
    ax.set_xticklabels([GROUP_LABELS[g] for g in group_order], size=label_size)
    ax.set_title(ylabel if title is None else title, size='small')


def plot_primary_trace_amplitude(df_trace_pooled, save_dir, filename_root='primary_trace_amplitude',
                                 title='Trace-period amplitude'):
    """
    Panel 1: mouse-level points + violin group summaries + Holm-corrected pairwise brackets, one
    point per animal (mouse_level_trace_amplitude). The confirmatory omnibus estimate/CI lives in
    the companion stats .txt (fit_primary_trace_amplitude()'s summary_text) written alongside this
    figure by the caller, and the multiplicative-scale effect + 95% CI in the companion
    plot_effect_forest() panel -- this panel is the visual triangulation check, not the sole home
    of the inferential claim.

    title : PLAN SECTION 1.3 FIX -- this function is reused for the Test_B/Test_B_1wk 20 s
           post-tone window, which is NOT the trace period; the caller must pass the correct
           window label rather than relying on a hardcoded default describing only the TFC_cond
           call site (the bug this default's presence guards against reintroducing).
    """
    mouse_df = mouse_level_trace_amplitude(df_trace_pooled)
    values_per_group = _mouse_values_per_group(mouse_df, 'log_amplitude',
                                               panel_name='plot_primary_trace_amplitude')
    fig, ax = plt.subplots(figsize=_PANEL_FIGSIZE)
    _draw_mouse_violin_panel(ax, values_per_group, _YLABEL_LOG_AMPLITUDE, label_size='medium')
    ax.set_title(f'{title}\n(mouse-level, n={len(mouse_df)})', size='small')
    plt.tight_layout(pad=0.5)
    _save_panel(fig, save_dir, filename_root)


def plot_epoch_profile(df_fine_amp, save_dir, filename_root='epoch_profile',
                       epoch_order=TFC_EPOCHS):
    """
    Panel 2: descriptive (not model-marginal) per-mouse-per-epoch pooled mean log-amplitude,
    plotted by group across epoch order with faint per-animal trajectories behind the group
    mean +/- SEM line. Model-estimated marginal effects are reported in the companion co-primary
    stats .txt (fit_epoch_delta_model's summary_text), not extracted into this panel, to avoid
    the added complexity of a margeff pipeline for a first pass -- see sp_rates_lmm_methods.md.
    """
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.spines[['right', 'top']].set_visible(False)
    x = np.arange(len(epoch_order))

    for group in GROUP_ORDER:
        per_mouse_means = []
        for mouse in sorted(df_fine_amp.loc[df_fine_amp['group'] == group, 'mouse'].unique()):
            mouse_epoch_means = []
            for epoch in epoch_order:
                sub = df_fine_amp[(df_fine_amp['mouse'] == mouse) & (df_fine_amp['epoch'] == epoch)]
                mouse_epoch_means.append(sub['log_amplitude'].mean() if len(sub) else np.nan)
            per_mouse_means.append(mouse_epoch_means)
            ax.plot(x, mouse_epoch_means, color=GROUP_COLOURS[group], alpha=0.25, linewidth=0.8)
        arr = np.asarray(per_mouse_means, dtype=float)
        group_mean = np.nanmean(arr, axis=0)
        n_mice = np.sum(~np.isnan(arr), axis=0)
        group_sem = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(np.maximum(n_mice, 1))
        ax.plot(x, group_mean, color=GROUP_COLOURS[group], linewidth=2.5,
               label=GROUP_LABELS[group])
        ax.fill_between(x, group_mean - group_sem, group_mean + group_sem,
                        color=GROUP_COLOURS[group], alpha=0.2)

    ax.set_xticks(x)
    ax.set_xticklabels(epoch_order, rotation=30, ha='right')
    ax.set_ylabel(_YLABEL_LOG_AMPLITUDE)
    ax.set_title('Epoch profile (descriptive; per-mouse pooled means)', size='small')
    ax.legend(fontsize=7, frameon=False)
    plt.tight_layout(pad=0.5)
    _save_panel(fig, save_dir, filename_root)


def plot_epoch_delta_forest(delta_summary, save_dir, filename_root='epoch_delta_forest',
                            reference_epoch=TFC_REFERENCE_EPOCH):
    """
    Companion to plot_epoch_profile: the WITHIN-CELL amplitude elevation over each cell's own
    pre_tone baseline, as a fold-change with a 95% interval, for every epoch
    (compute_all_epoch_deltas). The confirmatory trace row is marked; the others are descriptive.

    ** This figure exists to prevent a specific misreading. ** The confirmatory trace-vs-pre_tone
    delta is NULL, and the honest conclusion is that hM3D raises per-event amplitude across all
    epochs rather than specifically during the trace interval. A figure showing only the trace
    contrast cannot distinguish "no trace-specific effect" from "no effect", and a reader who saw
    only the large group main effect plus a trace-labelled panel would naturally infer a
    trace-specific one. Showing every epoch's interval side by side makes the flat profile --
    intervals overlapping each other and straddling 1.0 -- the thing the reader actually sees.
    """
    epochs = [e for e in delta_summary['epoch'].unique()]
    groups = [g for g in GROUP_ORDER if g in set(delta_summary['group'])]
    fig, ax = plt.subplots(figsize=(5.2, 0.9 + 0.42 * len(epochs) * len(groups)))
    yticks, yticklabels = [], []
    y = 0
    for epoch in epochs:
        for group in groups:
            row = delta_summary[(delta_summary['epoch'] == epoch) &
                                (delta_summary['group'] == group)].iloc[0]
            ax.errorbar(row['ratio'], y,
                       xerr=[[row['ratio'] - row['ratio_lo']], [row['ratio_hi'] - row['ratio']]],
                       fmt='o', color=GROUP_COLOURS[group], capsize=3,
                       markersize=8 if row['is_confirmatory'] else 5)
            marker = ' (confirmatory)' if row['is_confirmatory'] else ''
            yticks.append(y)
            yticklabels.append(f'{epoch} - {reference_epoch}, {GROUP_LABELS[group]}{marker}')
            y -= 1
    ax.axvline(1.0, color='k', linewidth=0.8, linestyle='--')
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, size='small')
    ax.set_xlabel('Fold-change vs Ctl in within-cell amplitude elevation')
    ax.set_title('Within-cell epoch deltas (only the trace row is confirmatory;\n'
                'the others are descriptive context for its null)', size='small')
    ax.spines[['right', 'top']].set_visible(False)
    plt.tight_layout(pad=0.5)
    _save_panel(fig, save_dir, filename_root)


def plot_amplitude_ecdf(df_trace_pooled, save_dir, filename_root='amplitude_ecdf',
                        title='Trace-period amplitude distribution'):
    """
    Panel 3: pooled per-cell ECDF of log-amplitude by group (bold), with per-mouse ECDFs
    (thin, RAISED prominence -- plan section 6: these 17 mice, not 7,410 independent cells, are
    the unit of inference, so the per-mouse curves are drawn more visible than the pooled default
    caban.single_unit_common.ecdf_panel uses elsewhere) behind -- the bursting hypothesis predicts
    a fattened right tail, which a mean contrast alone can miss.

    title : PLAN SECTION 1.3 FIX -- same hardcoded-title bug class as plot_primary_trace_amplitude
           (this panel is not currently reused for the recall window, but the parameter removes
           the latent bug rather than leaving it dormant).
    """
    per_cell_by_group = {g: df_trace_pooled.loc[df_trace_pooled['group'] == g, 'log_amplitude'].to_numpy()
                         for g in GROUP_ORDER}
    per_mouse_by_group = {
        g: {m: sub['log_amplitude'].to_numpy()
            for m, sub in df_trace_pooled[df_trace_pooled['group'] == g].groupby('mouse')}
        for g in GROUP_ORDER
    }
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ecdf_panel(ax, per_cell_by_group, per_mouse_by_group=per_mouse_by_group,
              xlabel=_YLABEL_LOG_AMPLITUDE, title=title,
              mouse_alpha=0.5, mouse_linewidth=1.1)
    plt.tight_layout(pad=0.5)
    _save_panel(fig, save_dir, filename_root)


def plot_amplitude_p90(df_trace_pooled, save_dir, filename_root='amplitude_p90'):
    """
    Panel (plan section 6): mouse-level 90th-percentile trace-period log-amplitude by group --
    complements plot_amplitude_ecdf's tail visualization with the same per-mouse-scalar summary
    the '..._p90' mouse-label permutation test already targets. A per-mouse P90 is already a
    one-number-per-mouse summary, so this panel was mouse-level even while the others were
    briefly SuperPlots.
    """
    mouse_df = mouse_level_p90_amplitude(df_trace_pooled)
    values_per_group = _mouse_values_per_group(mouse_df, 'p90_log_amplitude',
                                               panel_name='plot_amplitude_p90')
    fig, ax = plt.subplots(figsize=_PANEL_FIGSIZE)
    _draw_mouse_violin_panel(ax, values_per_group, '90th percentile log(amplitude)',
                             label_size='medium')
    ax.set_title(f'Trace-period amplitude, 90th percentile\n(mouse-level, n={len(mouse_df)})', size='small')
    plt.tight_layout(pad=0.5)
    _save_panel(fig, save_dir, filename_root)


def plot_decomposition(df_trace_pooled_raw, save_dir, filename_root='decomposition'):
    """
    Panel 4 ("components of population calcium activity" -- plan section 6 renamed this from
    "activity decomposition" and added the missing overall-rate panel so the identity
    fraction_active x rate_active = overall_rate is explicit rather than left for the reader to
    multiply panels themselves): fraction active (mouse-level only, a proportion computed OVER
    cells), rate among active cells, overall event rate across ALL cells (their exact product),
    mean per-event amplitude, and total S/s (= overall_rate x amplitude), left to right following
    the decomposition chain.

    **Fraction active is a mouse-level violin; the other four are cell-level SuperPlots.** That
    split is not arbitrary: fraction active is a proportion computed OVER a mouse's cells, so it
    has no per-cell value to plot and only exists once per animal. The other four are genuine
    per-cell quantities, so every cell is shown (see draw_superplot_triplet), with the per-mouse
    means overlaid as the large markers and the statistics still computed from those means alone.

    An earlier attempt at this was reverted because the SuperPlots used a LINEAR axis and these
    quantities' long right tails crushed all 17 mouse means into the bottom ~6-13% of each panel.
    The fix was the axis, not the figure type: `yscale='auto'` puts them on log (rate among active
    cells, which is strictly positive) or symlog (overall rate, total S/s, and amplitude, which
    contain exact zeros from zero-event cells that must not be dropped).

    Three further presentation decisions, all visible only on this figure:

    * **Every panel is annotated with Holm-corrected significance brackets**, on log and symlog
      axes as well as the linear one, via annotate_pairwise_brackets' axes-fraction positioning.
      Two earlier attempts are worth not repeating. Data-coordinate brackets
      (barplot_annotate_brackets) cannot be drawn on a log axis at all, which forced a split
      style: stars on the one linear panel, a corner p-value text block on the other four. The
      obvious fix -- put an estimate + interval text block on ALL five panels -- made the style
      uniform but was WORSE to read: five blocks of small type crowding the clouds, and a reader
      had to parse "[1.14, 2.05]" to notice an effect that a single asterisk conveys instantly.
      The estimates and intervals are genuinely wanted, but they belong in a companion file, not
      on the panel: write_decomposition_contrasts_markdown() emits them alongside the figure.
    * **The two rate panels get sub-quantum vertical jitter** (y_quantum = 1/exposure_seconds).
      A per-cell rate is a small integer count over a fixed window, so without it the cloud is a
      few hard horizontal stripes -- real quantization, but it hides the density that is the
      whole point of showing cells. The jitter stays strictly within each cell's own quantization
      bin and never touches the means or the statistics.
    * **Overall event rate is drawn wider than its siblings.** It is the summary quantity of the
      first half of the chain (fraction_active x rate_active), and the hM4D effect lives in it, so
      it should not read as one of five equal-weight panels.

    df_trace_pooled_raw : output of aggregate_over_trials(df_fine, 'trace') BEFORE
                          filter_amplitude_rows() -- i.e. still carrying zero-event cells, since
                          fraction-active/overall-rate/total-S/s need them.
    """
    df = df_trace_pooled_raw.copy()
    df['total_per_s'] = df['sum_amplitude'] / df['exposure_seconds']
    df['overall_rate'] = df['n_events'] / df['exposure_seconds']
    df['active'] = df['n_events'] > 0

    mouse_frac_active = fraction_active_table(df)
    # rate_active is overall_rate restricted to active cells -- same n_events/exposure_seconds
    # ratio already computed above, just renamed for this panel's label; no need to recompute it.
    active_only = df[df['active']].copy().rename(columns={'overall_rate': 'rate_active'})
    amp_df = filter_amplitude_rows(df)

    # One count = one quantum of rate. exposure_seconds is the same fixed trace window for every
    # cell within a trial, so the median is that window (pooled over trials) up to mice with a
    # dropped trial; using the median rather than a per-row value keeps the spread uniform across
    # the panel instead of varying with a mouse's trial count.
    rate_quantum = 1.0 / float(np.median(df['exposure_seconds']))

    # cell_level=False only for fraction active -- it is a proportion over a mouse's cells and so
    # has no per-cell value to draw; every other panel here is a genuine per-cell quantity.
    # ci_scale='log' means "this column is ALREADY log-transformed", matching yscale='linear';
    # 'difference_only' is for the bounded proportion, where a fold-change is not the natural
    # summary. width is the gridspec width ratio.
    panels = [
        (mouse_frac_active, 'fraction_active', 'Fraction active', False, 'auto',
         None, 'difference_only', '', 1.0),
        (active_only, 'rate_active', 'Rate among active cells (/s)', True, 'auto',
         rate_quantum, 'linear', '/s', 1.0),
        (df, 'overall_rate', 'Overall event rate (/s, all cells)', True, 'auto',
         rate_quantum, 'linear', '/s', 1.9),
        (amp_df, 'log_amplitude', _YLABEL_LOG_AMPLITUDE, True, 'linear',
         None, 'log', '', 1.0),
        (df, 'total_per_s', 'Total S/s', True, 'auto', None, 'linear', 'S/s', 1.0),
    ]

    fig, axs = plt.subplots(1, len(panels), figsize=(15.0, 3.2),
                            gridspec_kw={'width_ratios': [p[-1] for p in panels]})
    contrasts_by_panel = {}
    for ax, (sub, col, ylabel, cell_level, yscale, quantum, ci_scale, ci_unit, _w) in zip(axs, panels):
        if cell_level:
            _draw_cell_superplot_panel(ax, sub, col, ylabel, panel_name='plot_decomposition',
                                       yscale=yscale, y_quantum=quantum, annotate='stats',
                                       bracket_mode='axes')
        else:
            values_per_group = _mouse_values_per_group(sub, col, panel_name='plot_decomposition')
            _draw_mouse_violin_panel(ax, values_per_group, ylabel, title=ylabel,
                                     annotate='stats', bracket_mode='axes')
        # The estimates/intervals go to the companion markdown rather than onto the panel.
        contrasts_by_panel[ylabel] = _panel_contrasts(sub, col, ci_scale, ci_unit)

    fig.suptitle('Components of population calcium activity: fraction_active x rate_active = '
                'overall_rate; overall_rate x amplitude = total_per_s', size='small')
    fig.subplots_adjust(left=0.05, bottom=0.14, right=0.99, top=0.82, wspace=0.5)
    _save_panel(fig, save_dir, filename_root)
    write_decomposition_contrasts_markdown(contrasts_by_panel, save_dir,
                                           filename_root + '_contrasts.md')


def plot_manipulation_check(df_delta, dropout_df, save_dir, filename_root='manipulation_check'):
    """
    Panel 5: LT1->LT2 within-cell delta log-amplitude by group (left) and LT1->LT2 detection
    dropout fraction by group (right) -- the manipulation check plus the Tier-1 detection-bias
    measurement, side by side since both come from the same drug-free-to-drug session pair. Both
    mouse-level violins; the left panel's per-cell deltas are collapsed to per-mouse means by
    _mouse_values_per_group (see its docstring on the reverted SuperPlot variant).
    """
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 3.5))
    for ax, (sub, col, ylabel) in zip(axs, [
        (df_delta, 'delta_log_amplitude', 'LT2 - LT1 Delta log(amplitude)'),
        (dropout_df, 'dropout_fraction', 'LT1->LT2 dropout fraction'),
    ]):
        values_per_group = _mouse_values_per_group(sub, col, panel_name='plot_manipulation_check')
        _draw_mouse_violin_panel(ax, values_per_group, ylabel, title=ylabel)

    fig.suptitle('Manipulation check (LT1 drug-free -> LT2 CNO)', size='medium')
    fig.subplots_adjust(left=0.1, bottom=0.14, right=0.98, top=0.84, wspace=0.5)
    _save_panel(fig, save_dir, filename_root)


def plot_run_structure(df_runs_trace, save_dir, filename_root='run_structure'):
    """
    Panel (plan section 4): per-mouse mean run width (frames), mean local maxima per run, and
    fraction of multi-peak runs (n_local_maxima>=2), trace epoch only -- the direct run-structure
    signature of bursting (a fattened right tail in run width, height-independent), promoted from
    the notebook's own ad hoc diagnostic into the analysis proper. Mouse-level violins, matching
    every other group-comparison panel in this module; the per-run values are collapsed to
    per-mouse means by _mouse_values_per_group (see its docstring on the reverted SuperPlot
    variant). The full per-run distribution behind these means is in stats/run_structure.txt.
    """
    df = df_runs_trace.copy()
    df['multi_peak'] = (df['n_local_maxima'] >= 2).astype(float)

    fig, axs = plt.subplots(1, 3, figsize=(9, 3.2))
    for ax, value_col, ylabel in zip(
        axs, ['width_frames', 'n_local_maxima', 'multi_peak'],
        [f'Run width (frames, {MINISCOPE_FPS} fps)', 'Local maxima per run', 'Fraction multi-peak runs']):
        values_per_group = _mouse_values_per_group(df, value_col, panel_name='plot_run_structure')
        _draw_mouse_violin_panel(ax, values_per_group, ylabel, title=ylabel)
    fig.suptitle('Run-structure evidence (trace epoch)', size='medium')
    fig.subplots_adjust(left=0.08, bottom=0.14, right=0.98, top=0.86, wspace=0.4)
    _save_panel(fig, save_dir, filename_root)


def plot_threshold_sensitivity(df_sens, save_dir, filename_root='threshold_sensitivity'):
    """
    Panel (plan section 4): forest plot of the primary group coefficient (log-amplitude scale) at
    each sensitivity threshold in df_sens (run_threshold_sensitivity's output). A coefficient/CI
    that stays on the same side of zero across all three thresholds means the primary conclusion
    is not an artifact of run-merging (temporally adjacent peaks collapsing into one wider run) at
    any one threshold choice.
    """
    groups = [g for g in GROUP_ORDER if g in df_sens['group'].unique()]
    thresholds = sorted(df_sens['thres'].unique())
    fig, ax = plt.subplots(figsize=(4.8, 0.9 + 0.5 * len(groups) * len(thresholds)))
    yticks, yticklabels = [], []
    y = 0
    for group in groups:
        for thres in thresholds:
            row = df_sens[(df_sens['group'] == group) & (df_sens['thres'] == thres)].iloc[0]
            ax.errorbar(row['coef'], y,
                       xerr=[[row['coef'] - row['ci_lo']], [row['ci_hi'] - row['coef']]],
                       fmt='o', color=GROUP_COLOURS[group], capsize=3)
            yticks.append(y)
            yticklabels.append(f"{GROUP_LABELS[group]}, thres={thres:g}")
            y -= 1
    ax.axvline(0, color='k', linewidth=0.8, linestyle='--')
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, size='small')
    ax.set_xlabel('Group coefficient (log-amplitude scale, vs mCherry)')
    ax.set_title('Threshold sensitivity: primary amplitude contrast', size='small')
    ax.spines[['right', 'top']].set_visible(False)
    plt.tight_layout(pad=0.5)
    _save_panel(fig, save_dir, filename_root)


def plot_effect_forest(fit, save_dir, filename_root, reference='mCherry',
                       xlabel='Fold-change in mean per-event amplitude', title=''):
    """
    Plan section 6: effect estimates + 95% CI on the MULTIPLICATIVE (exp) scale for a group
    contrast already fit by fit_primary_trace_amplitude() or fit_epoch_delta_model() --
    exp(coef) is a directly interpretable fold-change relative to `reference` (e.g. "1.5x"),
    unlike the log-scale coefficient in the stats .txt. This is the quantitative complement to
    the mouse-level panel's significance-star brackets (plot_primary_trace_amplitude), not a
    replacement for it -- both are saved.

    fit : a dict with a 'result' key (fit_primary_trace_amplitude / fit_epoch_delta_model's
          return value) fit against `reference`.
    """
    result = fit['result']
    fe_names = list(result.fe_params.index) if hasattr(result, 'fe_params') else list(result.params.index)
    params = result.fe_params if hasattr(result, 'fe_params') else result.params
    ci = result.conf_int()
    groups = [g for g in GROUP_ORDER if g != reference]

    fig, ax = plt.subplots(figsize=(3.8, 0.7 + 0.6 * len(groups)))
    for i, group in enumerate(groups):
        name = [n for n in fe_names if n.endswith(f'[T.{group}]') and ':' not in n][0]
        coef = float(params[name])
        ci_row = ci.loc[name]
        lo, hi = float(ci_row.iloc[0]), float(ci_row.iloc[1])
        ratio, ratio_lo, ratio_hi = float(np.exp(coef)), float(np.exp(lo)), float(np.exp(hi))
        ax.errorbar(ratio, i, xerr=[[ratio - ratio_lo], [ratio_hi - ratio]], fmt='o',
                   color=GROUP_COLOURS[group], capsize=3, markersize=7)
        ax.text(ratio_hi + 0.03 * ratio_hi, i, f'{ratio:.2f}x [{ratio_lo:.2f}, {ratio_hi:.2f}]',
               va='center', fontsize=7)
    ax.axvline(1.0, color='k', linewidth=0.8, linestyle='--')
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([f'{GROUP_LABELS[g]}/{GROUP_LABELS[reference]}' for g in groups], size='small')
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title, size='small')
    ax.spines[['right', 'top']].set_visible(False)
    plt.tight_layout(pad=0.5)
    _save_panel(fig, save_dir, filename_root)


# ─────────────────────────────────────────────────────────────────────────────
# Example-trace panels (plan section 5c)
# ─────────────────────────────────────────────────────────────────────────────

def select_example_cells(df_trace_amp, percentiles=EXAMPLE_TRACE_PERCENTILES):
    """
    Plan section 5c: reproducible, non-hand-picked cell selection for the example-trace panels --
    per group, the cell whose trace-epoch log_amplitude is closest to each of `percentiles` of
    that group's OWN distribution (a low/median/high triplet by default), so the reader sees the
    shape of the distribution, not one flattering example. Deterministic: no RNG, ties broken by
    (mouse, cell) sort order.

    Returns dict group -> dict percentile -> {'mouse':..., 'cell':..., 'log_amplitude':...}.
    """
    out = {}
    for group in GROUP_ORDER:
        sub = df_trace_amp[df_trace_amp['group'] == group].sort_values(['mouse', 'cell']).reset_index(drop=True)
        if sub.empty:
            raise RuntimeError(f'select_example_cells: no trace-active cells for group {group!r}.')
        out[group] = {}
        for p in percentiles:
            target = np.percentile(sub['log_amplitude'], p)
            idx = (sub['log_amplitude'] - target).abs().idxmin()
            row = sub.loc[idx]
            out[group][p] = {'mouse': row['mouse'], 'cell': row['cell'],
                            'log_amplitude': float(row['log_amplitude'])}
    return out


def _pick_representative_trial(S_row, cell_thres, tfc_frames_fn, session, epoch=TFC_TRACE_EPOCH):
    """
    Deterministic trial choice for one cell's example-trace panel: the trial (within `epoch`)
    with the MOST detected runs for this cell, ties broken by lowest trial index -- guarantees
    the plotted window contains something to show, without hand-picking which trial.
    """
    frameidx, _amp, _nlm, _w, _start = find_event_runs_ca(S_row, cell_thres)
    best_trial, best_n = None, -1
    for trial_idx in session.periods:
        frames = tfc_frames_fn(session, epoch, trial_idx)
        if frames is None:
            continue
        onset, offset = frames
        n_here = int(np.sum((frameidx >= onset) & (frameidx < offset)))
        if n_here > best_n:
            best_trial, best_n, best_window = trial_idx, n_here, (onset, offset)
    if best_trial is None:
        raise RuntimeError('_pick_representative_trial: no valid trace window found for this session.')
    return best_trial, best_window


def plot_example_traces(sessions, df_trace_amp, save_dir, mapping='full', thres=None,
                        tfc_frames_fn=None, percentiles=EXAMPLE_TRACE_PERCENTILES,
                        pad_seconds=EXAMPLE_TRACE_PAD_S, filename_root='example_traces'):
    """
    Plan section 5c: representative-cell raw-trace figure -- deconvolved S and denoised C over
    the trace-epoch window (+/- pad_seconds context), with each supra-threshold run shaded and
    its integral annotated, for a reproducible low/median/high (default 10/50/90th percentile)
    within-group amplitude triplet (select_example_cells) -- never hand-picked.

    Shading is drawn from find_event_runs_ca() called directly on the plotted cell's own S row at
    the SAME threshold used to build the amplitude tables, so the shaded boundaries are exactly
    what was measured, not a separately re-derived approximation -- see the alignment regression
    test in scratchpad/test_event_amplitude_integration.py.
    """
    if tfc_frames_fn is None:
        raise ValueError('plot_example_traces: tfc_frames_fn is required.')
    selections = select_example_cells(df_trace_amp, percentiles=percentiles)
    n_groups, n_perc = len(GROUP_ORDER), len(percentiles)
    fig, axs = plt.subplots(n_perc, n_groups, figsize=(4.2 * n_groups, 2.1 * n_perc), squeeze=False)

    for col, group in enumerate(GROUP_ORDER):
        for row, p in enumerate(percentiles):
            ax = axs[row][col]
            sel = selections[group][p]
            mouse, cell_id = sel['mouse'], sel['cell']
            session = sessions[mouse]
            cell_thres = session.thres if thres is None else thres
            _S, _Ss, _Sp, S_idx = session.get_S_mapping(mapping)
            cell_row = list(S_idx).index(cell_id)
            S_row, C_row = session.S[cell_row, :], session.C[cell_row, :]

            trial_idx, (onset, offset) = _pick_representative_trial(S_row, cell_thres, tfc_frames_fn, session)
            pad = int(round(pad_seconds * MINISCOPE_FPS))
            win_lo, win_hi = max(0, onset - pad), min(S_row.shape[0], offset + pad)
            t = (np.arange(win_lo, win_hi) - onset) / MINISCOPE_FPS

            frameidx, amplitude, _nlm, width, start = find_event_runs_ca(S_row, cell_thres)
            in_plot = (frameidx >= win_lo) & (frameidx < win_hi)
            for fi, amp, w, st in zip(frameidx[in_plot], amplitude[in_plot], width[in_plot], start[in_plot]):
                run_lo_s, run_hi_s = (st - onset) / MINISCOPE_FPS, (st + w - onset) / MINISCOPE_FPS
                ax.axvspan(run_lo_s, run_hi_s, color=GROUP_COLOURS[group], alpha=0.18, zorder=0)
                ax.text((run_lo_s + run_hi_s) / 2, 1.02, f'{amp:.1f}', ha='center', va='bottom',
                       fontsize=5.5, color=GROUP_COLOURS[group], transform=ax.get_xaxis_transform())

            c_win = C_row[win_lo:win_hi]
            s_win = S_row[win_lo:win_hi]
            ax.plot(t, c_win / max(np.max(c_win), 1e-12), color='0.5', linewidth=0.8, label='C (norm.)')
            ax.plot(t, s_win / max(np.max(s_win), 1e-12), color=GROUP_COLOURS[group], linewidth=1.0,
                   label='S (norm.)')
            ax.axvline(0, color='k', linewidth=0.6, linestyle=':')
            ax.axvline((offset - onset) / MINISCOPE_FPS, color='k', linewidth=0.6, linestyle=':')
            ax.set_ylim(-0.05, 1.15)
            ax.spines[['right', 'top']].set_visible(False)
            if row == 0:
                ax.set_title(f'{GROUP_LABELS[group]}', size='small')
            if col == 0:
                ax.set_ylabel(f'p{p}\n({mouse}, cell {cell_id})', size=6)
            if row == n_perc - 1:
                ax.set_xlabel('Time from trace onset (s)', size='small')
            if row == 0 and col == 0:
                ax.legend(fontsize=5.5, frameon=False, loc='upper right')

    fig.suptitle('Example traces: trace-epoch window, shaded runs = detected events '
                '(label = per-event integral)', size='small')
    fig.subplots_adjust(left=0.08, bottom=0.1, right=0.98, top=0.88, hspace=0.5, wspace=0.3)
    _save_panel(fig, save_dir, filename_root)


def plot_width_vs_height_matched_examples(sessions, df_runs_trace, save_dir, mapping='full',
                                          thres=None, target_height_percentile=50,
                                          filename_root='width_height_matched_examples'):
    """
    Plan section 5c's last bullet: one run per group, each matched to the SAME target peak
    HEIGHT (the `target_height_percentile`-th percentile of run peak height POOLED across all
    groups), so that if hM3D's height-matched run is still visibly WIDER, the reader sees
    directly that the effect is not merely "taller events" -- concretely separating the
    run-width story (plan section 4) from a peak-amplitude-only account. Selection is
    deterministic: per group, the run whose own peak_height is closest to the pooled target,
    ties broken by (mouse, cell, trial) sort order.

    Re-derives each selected run's exact shading from find_event_runs_ca() on the raw S row
    (matched by peak_frame), the same alignment guarantee plot_example_traces() gives.
    """
    target = np.percentile(df_runs_trace['peak_height'], target_height_percentile)
    fig, axs = plt.subplots(1, len(GROUP_ORDER), figsize=(4.2 * len(GROUP_ORDER), 2.3), squeeze=False)
    axs = axs[0]

    for ax, group in zip(axs, GROUP_ORDER):
        gsub = df_runs_trace[df_runs_trace['group'] == group].sort_values(['mouse', 'cell', 'trial'])
        if gsub.empty:
            raise RuntimeError(f'plot_width_vs_height_matched_examples: no runs for group {group!r}.')
        idx = (gsub['peak_height'] - target).abs().idxmin()
        sel = gsub.loc[idx]
        mouse, cell_id = sel['mouse'], sel['cell']
        session = sessions[mouse]
        cell_thres = session.thres if thres is None else thres
        _S, _Ss, _Sp, S_idx = session.get_S_mapping(mapping)
        cell_row = list(S_idx).index(cell_id)
        S_row = session.S[cell_row, :]

        frameidx, amplitude, _nlm, width, start = find_event_runs_ca(S_row, cell_thres)
        match = np.where(frameidx == int(sel['peak_frame']))[0]
        if len(match) != 1:
            raise RuntimeError(f'plot_width_vs_height_matched_examples: could not relocate the '
                               f'selected run (mouse={mouse}, cell={cell_id}, peak_frame='
                               f'{sel["peak_frame"]}) via find_event_runs_ca -- selection table '
                               f'and a fresh detection pass disagree.')
        i = match[0]
        st, w, amp = int(start[i]), int(width[i]), float(amplitude[i])
        pad = int(round(EXAMPLE_TRACE_PAD_S * MINISCOPE_FPS))
        win_lo, win_hi = max(0, st - pad), min(S_row.shape[0], st + w + pad)
        t = (np.arange(win_lo, win_hi) - st) / MINISCOPE_FPS
        s_win = S_row[win_lo:win_hi]

        ax.plot(t, s_win, color=GROUP_COLOURS[group], linewidth=1.1)
        ax.axvspan(0, w / MINISCOPE_FPS, color=GROUP_COLOURS[group], alpha=0.18)
        ax.axhline(cell_thres, color='k', linewidth=0.5, linestyle=':')
        ax.spines[['right', 'top']].set_visible(False)
        ax.set_title(f'{GROUP_LABELS[group]}: peak={sel["peak_height"]:.2f}, '
                    f'width={w} frames, integral={amp:.1f}', size=6.5)
        ax.set_xlabel('Time from run start (s)', size='small')
        if ax is axs[0]:
            ax.set_ylabel('S (raw)', size='small')

    fig.suptitle(f'Height-matched examples (target peak height = pooled p{target_height_percentile} '
                f'= {target:.2f})', size='small')
    plt.tight_layout(pad=0.6, rect=[0, 0, 1, 0.9])
    _save_panel(fig, save_dir, filename_root)


# ─────────────────────────────────────────────────────────────────────────────
# Top-level orchestrator (mirrors caban.freezing_tuned_cells.run_freezing_tuned_cells /
# caban.population_coupling.run_population_coupling: one entry point, called from
# caban/sections.py, that builds every table, fits every model, writes every stats file, and
# saves every figure panel this analysis produces)
# ─────────────────────────────────────────────────────────────────────────────

def run_sp_rates_lmm(PLOTS_DIR, mice_per_group, TFC_cond, TFC_cond_LT1, TFC_cond_LT2,
                        Test_B, Test_B_1wk, mapping='full', thres=None, n_perm=20000, seed=0,
                        rate_draws=1000, rate_tune=1000, rate_chains=4, auto_close=True):
    """
    Full cell-level event-amplitude analysis: primary trace-period amplitude, co-primary
    within-cell epoch delta, run-structure/threshold-sensitivity bursting evidence, group x trial
    photobleaching check, secondary rate/fraction-active/Test_B endpoints, LT1->LT2 manipulation
    check + detection-dropout measurement, permutation-test sensitivity (both weightings), example
    -trace panels, and every figure panel this analysis produces. See
    analysis_methods_templates/sp_rates_lmm_methods.md for the full statistical rationale behind
    every choice made here, and this module's own CHANGELOG docstring for what changed and why.

    PLOTS_DIR              : ds.PLOTS_DIR-equivalent root; this analysis writes under
                             PLOTS_DIR/sp_rates_lmm/.
    mice_per_group          : ds.mice_per_group.
    TFC_cond, TFC_cond_LT1,
    TFC_cond_LT2, Test_B,
    Test_B_1wk              : dict of mouse -> session object, as on ds.
    mapping                 : primary cross-registration mapping ('full' by default -- the least
                              selective cell set; see sp_rates_lmm_methods.md).
    thres                   : per-cell deconvolution threshold override; defaults to each
                              session's own .thres attribute.
    n_perm, seed             : passed to mouse_label_permutation_test().
    rate_draws, rate_tune,
    rate_chains              : passed to fit_rate_group_epoch_model()'s Bambi/PyMC MCMC fit
                              (two fits -- full and reduced -- each at these settings). Lower
                              these for faster iteration during development; the defaults
                              (1000/1000/4) are what should be used for the reported result.
    """
    out_dir = os.path.join(PLOTS_DIR, 'sp_rates_lmm', 'TFC_cond')
    stats_dir = os.path.join(out_dir, 'stats')
    ensure_dirs(out_dir, stats_dir)
    _copy_analysis_methods_template(METHODS_FILENAME, out_dir)

    print('[sp_rates_lmm] Building TFC_cond event table + run-structure table (single pass)...')
    tfc_frames_fn = functools.partial(get_epoch_frames, pre_tone_duration_s=35.0)
    # build_epoch_and_run_tables, not the two standalone builders back to back: both need the
    # same full-session per-cell event detection (find_event_runs_ca_S) over the SAME mice at the
    # SAME threshold, so building them separately would silently re-run that dominant-cost step
    # twice. df_runs (all TFC_EPOCHS) is filtered to the trace epoch below where it is used.
    df_fine, df_runs = build_epoch_and_run_tables(mice_per_group, TFC_cond, TFC_EPOCHS,
                                                  tfc_frames_fn, mapping=mapping, thres=thres)

    # ---- Primary: trace-period amplitude, pooled across trials -------------------------------
    df_trace_raw = aggregate_over_trials(df_fine, TFC_TRACE_EPOCH)
    df_trace_amp = filter_amplitude_rows(df_trace_raw)

    # BH-FDR family for the frequentist secondaries, accumulated as each member is fit and
    # corrected in one pass at the end (see build_secondary_fdr_table for what is in and out).
    secondary_pvalues = {}

    print('[sp_rates_lmm] Fitting primary trace-amplitude model...')
    primary = fit_primary_trace_amplitude(df_trace_amp)
    point_estimates = compute_group_contrast_point_estimates(df_trace_amp)
    primary_contrasts = mouse_contrast_ci(
        _mouse_values_per_group(mouse_level_trace_amplitude(df_trace_amp), 'log_amplitude',
                                panel_name='primary_contrast_ci', group_order=GROUP_ORDER),
        scale='log')
    write_text(os.path.join(stats_dir, 'primary_trace_amplitude.txt'),
              f"PRIMARY: log(mean per-event amplitude) ~ group, trace epoch, pooled trials\n"
              f"Omnibus (joint Wald, both non-reference groups, df2=n_mice-1): {primary['omnibus']}\n\n"
              f"PRIMARY REPORTED ESTIMAND (plan section 3): equal-mouse-weighted contrast (mean "
              f"of each mouse's own mean) -- NOT the pooled cell-weighted mean, since cell count "
              f"is group-correlated and post-treatment/activity-dependent.\n"
              + '\n'.join(format_contrast_ci_lines(primary_contrasts, 'mCherry')) + '\n\n'
              f"Every contrast above reports its interval whether or not it is significant: at "
              f"n=5/6/6 animals a null is only interpretable together with what its interval "
              f"still admits (e.g. an interval reaching 1.55 has not excluded a +55% effect).\n\n"
              f"{point_estimates}\n\n"
              f"{primary['summary_text']}")
    plot_primary_trace_amplitude(df_trace_amp, out_dir, title='Trace-period amplitude')
    plot_effect_forest(primary, out_dir, 'primary_effect_forest',
                       title='Primary contrast, multiplicative scale')

    # ---- Co-primary: within-cell trace-vs-pre_tone delta (plan section 2) --------------------
    print('[sp_rates_lmm] Fitting co-primary within-cell epoch-delta model...')
    delta_epoch_df = compute_epoch_delta_table(df_fine, TFC_TRACE_EPOCH, TFC_REFERENCE_EPOCH)
    coprimary = fit_epoch_delta_model(delta_epoch_df)
    write_text(os.path.join(stats_dir, 'coprimary_epoch_delta.txt'),
              f"CO-PRIMARY: delta_log_amplitude ~ group, within-cell (trace - pre_tone), one row "
              f"per cell active in BOTH epochs\n"
              f"Cell selection: {coprimary['n_cells']} of {len(df_trace_amp)} trace-active cells "
              f"also had >=1 event in pre_tone and so qualify for this contrast.\n"
              f"Omnibus (joint Wald, both non-reference groups, df2=n_mice-1): {coprimary['omnibus']}\n\n"
              f"{coprimary['summary_text']}")
    plot_epoch_profile(filter_amplitude_rows(df_fine), out_dir)
    plot_effect_forest(coprimary, out_dir, 'coprimary_effect_forest',
                       xlabel='Fold-change in within-cell trace-vs-pre_tone amplitude elevation',
                       title='Co-primary contrast, multiplicative scale')

    # ---- Descriptive: the SAME within-cell delta for every other epoch ------------------------
    # Context for the co-primary null, not a test -- see compute_all_epoch_deltas' docstring.
    print('[sp_rates_lmm] Within-cell epoch deltas for the remaining epochs (descriptive)...')
    delta_summary, _delta_fits = compute_all_epoch_deltas(df_fine)
    write_text(os.path.join(stats_dir, 'epoch_deltas_descriptive.csv'),
              delta_summary.to_csv(index=False))
    plot_epoch_delta_forest(delta_summary, out_dir)

    # ---- Holm correction across the confirmatory family (the ONLY multiplicity burden here) ---
    holm = holm_correct_confirmatory(primary['omnibus']['p'], coprimary['omnibus']['p'])
    write_text(os.path.join(stats_dir, 'confirmatory_holm_correction.txt'),
              f"Holm correction across the two confirmatory omnibus tests:\n{holm}\n")
    print(f"[sp_rates_lmm] Confirmatory (Holm-corrected): {holm}")

    # ---- Amplitude distribution (tail) ----------------------------------------------------------
    plot_amplitude_ecdf(df_trace_amp, out_dir)
    plot_amplitude_p90(df_trace_amp, out_dir)

    # ---- Decomposition (fraction active, rate|active, overall rate, amplitude, total S/s) ------
    plot_decomposition(df_trace_raw, out_dir)

    # ---- group x trial (photobleaching control, plan section 5) --------------------------------
    print('[sp_rates_lmm] Fitting group x trial (photobleaching) model...')
    df_mouse_trial = build_mouse_trial_trace_amplitude(df_fine)
    trial_fit = fit_group_trial_model(df_mouse_trial)
    trial_slopes = report_group_trial_slopes(df_mouse_trial)
    secondary_pvalues['group_x_trial_interaction'] = trial_fit['omnibus']['p']
    write_text(os.path.join(stats_dir, 'group_trial_photobleaching.txt'),
              f"log_amplitude ~ group * C(trial) + (1|mouse), mouse-trial trace-epoch means\n"
              f"Omnibus (joint Wald, all interaction terms, df2=n_mice-1, CONSERVATIVE -- trial "
              f"is within-mouse; see fit_group_trial_model docstring): {trial_fit['omnibus']}\n\n"
              f"Per-group descriptive trial slopes (log_amplitude/trial, NOT a formal test):\n"
              f"{trial_slopes}\n\n{trial_fit['summary_text']}")

    # ---- Run-structure evidence (plan section 4) ------------------------------------------------
    print('[sp_rates_lmm] Run-structure evidence (trace epoch, from the table built above)...')
    df_runs_trace = df_runs[df_runs['epoch'] == TFC_TRACE_EPOCH].copy()
    if df_runs_trace.empty:
        raise RuntimeError('run_sp_rates_lmm: no trace-epoch runs in df_runs -- check '
                           'mapping/thres inputs.')
    run_summary = summarize_run_structure(df_runs_trace)
    plot_run_structure(df_runs_trace, out_dir)
    df_runs_trace_mp = df_runs_trace.copy()
    df_runs_trace_mp['multi_peak'] = (df_runs_trace_mp['n_local_maxima'] >= 2).astype(float)
    run_perm_results = {}
    for other_group in ('hM3D', 'hM4D'):
        for label, col in (('width', 'width_frames'), ('n_local_maxima', 'n_local_maxima'),
                           ('multi_peak_fraction', 'multi_peak')):
            stat_fn = make_contrast_stat(df_runs_trace_mp, col, other_group, 'mCherry', weight='mouse')
            key = f'{other_group}_vs_mCherry_{label}'
            run_perm_results[key] = mouse_label_permutation_test(
                stat_fn, mice_per_group, n_perm=n_perm, seed=seed)
            secondary_pvalues[f'run_structure_{key}'] = run_perm_results[key]['p_two_sided']
    run_perm_text = '\n'.join(
        f"{k}: observed={v['observed']:.4g}, p_two_sided={v['p_two_sided']:.4g}, n_perm={v['n_perm']}"
        for k, v in run_perm_results.items())
    write_text(os.path.join(stats_dir, 'run_structure.txt'),
              f"Per-mouse run-structure summary (trace epoch):\n"
              f"{run_summary.groupby('group')[['mean_width_frames', 'mean_n_local_maxima', 'fraction_multi_peak']].agg(['mean', 'std'])}\n\n"
              f"Mouse-label permutation tests (equal-mouse-weighted mean contrast):\n{run_perm_text}\n")

    # ---- Threshold sensitivity (plan section 4) --------------------------------------------------
    print('[sp_rates_lmm] Running threshold sensitivity (re-fits the primary contrast at '
         f'{THRESHOLD_SENSITIVITY_VALUES})...')
    df_sens = run_threshold_sensitivity(mice_per_group, TFC_cond, tfc_frames_fn)
    plot_threshold_sensitivity(df_sens, out_dir)
    write_text(os.path.join(stats_dir, 'threshold_sensitivity.csv'), df_sens.to_csv(index=False))

    # ---- Secondary: rate (Bambi NB-GLMM; see fit_rate_group_epoch_model docstring) ------------------
    print('[sp_rates_lmm] Fitting secondary rate model (Bambi NB-GLMM; this samples via MCMC, '
         'expect roughly a minute for two model fits)...')
    df_mte = build_mouse_trial_epoch_rate_table(df_fine)
    rate_fit = fit_rate_group_epoch_model(df_mte, draws=rate_draws, tune=rate_tune, chains=rate_chains, seed=seed)
    # The NB model speaks in rate RATIOS. A ratio off a small base is easy to over-read, so the
    # observed equal-mouse-weighted rate difference in events/s is reported next to it -- both
    # among active cells and across all cells.
    df_rate_cells = df_trace_raw.copy()
    df_rate_cells['overall_rate'] = df_rate_cells['n_events'] / df_rate_cells['exposure_seconds']
    rate_abs_lines = []
    for scope, sub in (('all cells', df_rate_cells),
                       ('active cells only', df_rate_cells[df_rate_cells['n_events'] > 0])):
        contrasts = mouse_contrast_ci(
            _mouse_values_per_group(sub, 'overall_rate', panel_name='secondary_rate_abs',
                                    group_order=GROUP_ORDER), scale='linear')
        rate_abs_lines.append(f'  {scope}:')
        rate_abs_lines += ['    ' + line
                           for line in format_contrast_ci_lines(contrasts, 'mCherry', unit='events/s')]
    write_text(os.path.join(stats_dir, 'secondary_rate.txt'),
              f"SECONDARY: event rate ~ group * epoch + trial, log(exposure) offset (Bambi NB-GLMM)\n"
              f"Reported as posterior contrasts + HDIs + LOO. NOT in the BH-FDR secondary family: "
              f"there is no p-value here to correct (see build_secondary_fdr_table).\n\n"
              f"Observed trace-epoch rate, equal-mouse-weighted, RATIO AND ABSOLUTE DIFFERENCE "
              f"together (a fold-change off a small base overstates the practical size of the "
              f"change):\n" + '\n'.join(rate_abs_lines) + '\n\n'
              f"{rate_fit['summary_text']}")

    # ---- Secondary: fraction active --------------------------------------------------------------
    frac_active = fraction_active_table(df_trace_raw)
    write_text(os.path.join(stats_dir, 'secondary_fraction_active.csv'),
              frac_active.to_csv(index=False))

    # ---- Secondary: permutation tests (mean + tail contrasts, both weightings, plan section 3) ---
    print('[sp_rates_lmm] Running mouse-label permutation tests...')
    perm_results = {}
    for other_group in ('hM3D', 'hM4D'):
        for label, reduce_fn in (
            ('mean', None),
            ('p90', lambda a, b: np.percentile(a, 90) - np.percentile(b, 90)),
        ):
            for weight in ('mouse', 'cell'):
                stat_fn = make_amplitude_contrast_stat(df_trace_amp, other_group, 'mCherry',
                                                       reduce_fn=reduce_fn, weight=weight)
                perm_results[f'{other_group}_vs_mCherry_{label}_{weight}weighted'] = (
                    mouse_label_permutation_test(stat_fn, mice_per_group, n_perm=n_perm, seed=seed))
    perm_text = '\n'.join(
        f"{k}: observed={v['observed']:.4g}, p_two_sided={v['p_two_sided']:.4g}, n_perm={v['n_perm']}"
        for k, v in perm_results.items())
    write_text(os.path.join(stats_dir, 'secondary_permutation_tests.txt'),
              "'mouseweighted' is the PRIMARY reported estimand (plan section 3); "
              "'cellweighted' is supporting.\n\n" + perm_text + '\n')

    # ---- Manipulation check + detection-bias measurement (LT1 drug-free -> LT2 CNO) ---------------
    print('[sp_rates_lmm] LT1->LT2 manipulation check + dropout measurement...')
    dropout_df = compute_lt1_lt2_dropout(mice_per_group, TFC_cond_LT1)
    delta_df = compute_lt1_lt2_amplitude_delta(mice_per_group, TFC_cond_LT1, TFC_cond_LT2)
    manip_fit = fit_lt1_lt2_manipulation_check(delta_df)
    write_text(os.path.join(stats_dir, 'manipulation_check.txt'),
              f"LT1->LT2 within-cell delta log(amplitude) ~ group\n"
              f"Omnibus: {manip_fit['omnibus']}\n\n{manip_fit['summary_text']}\n\n"
              f"Detection dropout by group (mean fraction):\n"
              f"{dropout_df.groupby('group')['dropout_fraction'].agg(['mean', 'std', 'count'])}\n")
    plot_manipulation_check(delta_df, dropout_df, out_dir)

    # ---- Example-trace panels (plan section 5c) ---------------------------------------------------
    print('[sp_rates_lmm] Building example-trace panels...')
    plot_example_traces(TFC_cond, df_trace_amp, out_dir, mapping=mapping, thres=thres,
                        tfc_frames_fn=tfc_frames_fn)
    plot_width_vs_height_matched_examples(TFC_cond, df_runs_trace, out_dir, mapping=mapping, thres=thres)

    # ---- Secondary: Test_B / Test_B_1wk post-tone amplitude (recall complement) -------------------
    for label, sessions in (('Test_B', Test_B), ('Test_B_1wk', Test_B_1wk)):
        print(f'[sp_rates_lmm] {label} post-tone amplitude...')
        recall_out_dir = os.path.join(PLOTS_DIR, 'sp_rates_lmm', label)
        recall_stats_dir = os.path.join(recall_out_dir, 'stats')
        ensure_dirs(recall_out_dir, recall_stats_dir)
        _copy_analysis_methods_template(METHODS_FILENAME, recall_out_dir)

        testb_frames_fn = functools.partial(get_testb_epoch_frames,
                                            post_tone_duration_s=TESTB_POST_TONE_DURATION_S)
        df_recall_fine = build_epoch_event_table(mice_per_group, sessions, TESTB_EPOCHS,
                                                 testb_frames_fn, mapping=mapping, thres=thres)
        df_recall_raw = aggregate_over_trials(df_recall_fine, 'post_tone')
        df_recall_amp = filter_amplitude_rows(df_recall_raw)
        recall_fit = fit_primary_trace_amplitude(df_recall_amp)
        cohort_note = report_recall_cohort_note(mice_per_group, sessions, label)
        secondary_pvalues[f'{label}_post_tone_amplitude'] = recall_fit['omnibus']['p']
        recall_contrasts = mouse_contrast_ci(
            _mouse_values_per_group(mouse_level_trace_amplitude(df_recall_amp), 'log_amplitude',
                                    panel_name=f'{label}_contrast_ci', group_order=GROUP_ORDER),
            scale='log')
        write_text(os.path.join(recall_stats_dir, 'post_tone_amplitude.txt'),
                  f"SECONDARY: log(mean per-event amplitude) ~ group, post-tone (20 s) window, "
                  f"pooled trials, {label}\nOmnibus: {recall_fit['omnibus']}\n"
                  f"(BH-FDR corrected across the secondary family -- see "
                  f"TFC_cond/stats/secondary_fdr_family.csv for the q-value.)\n\n"
                  f"Equal-mouse-weighted contrasts vs Ctl, with intervals reported for nulls too:\n"
                  + '\n'.join(format_contrast_ci_lines(recall_contrasts, 'mCherry')) + '\n\n'
                  f"{cohort_note}\n{recall_fit['summary_text']}")
        plot_primary_trace_amplitude(df_recall_amp, recall_out_dir,
                                     filename_root='post_tone_amplitude',
                                     title=f'{label}: post-tone amplitude (20 s window)')

    # ---- BH-FDR across the frequentist secondary family ----------------------------------------
    # Written last because its members are fit throughout the run above; the correction itself is
    # a single pass over the accumulated p-values.
    fdr_table = build_secondary_fdr_table(secondary_pvalues)
    write_text(os.path.join(stats_dir, 'secondary_fdr_family.csv'), fdr_table.to_csv(index=False))
    write_text(os.path.join(stats_dir, 'secondary_fdr_family.txt'),
              f"Benjamini-Hochberg FDR (alpha=0.05) across the {len(fdr_table)} prespecified "
              f"FREQUENTIST secondary tests.\n"
              f"EXCLUDED by design: the two confirmatory omnibus tests (Holm-corrected in their "
              f"own family), the Bambi NB rate model (posterior/LOO, no p-value to correct), and "
              f"all descriptive/sensitivity output. See build_secondary_fdr_table's docstring.\n\n"
              f"{fdr_table.to_string(index=False)}\n")
    print(f'[sp_rates_lmm] Secondary BH-FDR family: {len(fdr_table)} tests, '
         f'{int(fdr_table["reject"].sum())} significant at q<0.05.')

    print('[sp_rates_lmm] Done.')
    if auto_close:
        plt.close('all')
