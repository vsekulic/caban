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
import shutil
import functools
import collections

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
    annotate_pairwise_brackets, reserve_top_fraction,
)
from caban.decoder import _ANALYSIS_METHODS_TEMPLATES_DIR, _copy_analysis_methods_template
from caban.analysis import _draw_violin_triplet, do_pairwise_holm_plot
from caban.epoch_analysis import (get_epoch_frames, get_testb_epoch_frames,
                                  TRACE_MATCHED_WINDOW_S, POST_SHOCK_LATE_ONSET_S)

METHODS_FILENAME = 'sp_rates_lmm_methods.md'
# Interpretive companion to METHODS_FILENAME: the panel-by-panel reading of the decomposition
# figure, the paper's logical flow, and the Results/legend drafts. TFC_cond only -- it is the
# guide to the conditioning figure, and the recall sessions reuse only a single panel of it.
FIGURE_GUIDE_FILENAME = 'sp_rates_lmm_figure_guide.md'


def write_figure_guide(out_dir):
    """Write the figure guide into *out_dir*, OVERWRITING any copy already there.

    Deliberately not _copy_analysis_methods_template: that helper returns early when the
    destination file exists, which is right for a locked analysis plan but wrong here. The
    guide quotes the current result numbers, so a plots directory that already holds an older
    version of it must be refreshed on every run, not left alone.
    """
    src_path = os.path.join(_ANALYSIS_METHODS_TEMPLATES_DIR, FIGURE_GUIDE_FILENAME)
    if not os.path.isfile(src_path):
        raise FileNotFoundError(f'Figure guide template missing: {src_path}')
    ensure_dirs(out_dir)
    dest_path = os.path.join(out_dir, FIGURE_GUIDE_FILENAME)
    shutil.copy2(src_path, dest_path)
    print(f'[METHODS] Wrote {FIGURE_GUIDE_FILENAME} → {dest_path}')

# TFC_cond epochs used by the primary/co-primary confirmatory model. Shock is deliberately
# excluded from this set: at 0.05-0.2 Hz a 2 s window yields ~0-1 events/cell, so per-cell
# amplitude there is dominated by quantization and the window carries motion artifact. Handled
# separately later via YrA/C, not here.
TFC_EPOCHS = ('pre_tone', 'pre_tone_matched', 'tone', 'trace', 'post_shock', 'post_shock_late')
TFC_TRACE_EPOCH = 'trace'
TFC_REFERENCE_EPOCH = 'pre_tone'

# The second confirmatory response window. CA1 pyramidal activity after an aversive US is
# elevated for only tens of seconds, and that window -- not the trace interval -- is the one
# whose disruption impairs trace fear memory: Puhger et al. 2024 (iScience 27:109035) find no
# CA1 response during the trace interval at all, a large sustained post-shock response,
# optogenetic silencing 0-40 s after the shock impairs both tone and context memory, and the
# SAME silencing delivered 140 s after the shock does nothing. There are therefore two live
# hypotheses in this literature about WHERE in a TFC trial a hippocampal manipulation should
# act, and testing only the trace interval answers just one of them.
TFC_POST_SHOCK_EPOCH = 'post_shock'

# The DELAYED arm of that same window, and the internal control for its time-limited-ness: the
# Puhger result above is a CONTRAST between silencing early and silencing late, and reproducing
# it needs a late window, not just an early one. post_shock-vs-pre_tone is not that contrast --
# aggregate_over_trials pools each epoch across trials BEFORE differencing, so its reference is a
# mixture of one shock-naive baseline (trial 1's, which no shock precedes) and four windows
# 163 s after a shock, at trial indices that do not align. post_shock-vs-post_shock_late is
# within-trial at aligned indices with no naive window mixed in.
#
# DESCRIPTIVE, NOT CONFIRMATORY. It is deliberately absent from
# TFC_CONFIRMATORY_RESPONSE_EPOCHS and from both multiplicity families (Holm and BH-FDR): its
# job is to characterize the post-shock null, not to test a hypothesis. Report its estimate and
# 95% interval, never a significance verdict. See epoch_analysis.POST_SHOCK_LATE_ONSET_S for
# why the window begins 90 s and not Puhger's 140 s after shock offset.
TFC_POST_SHOCK_LATE_EPOCH = 'post_shock_late'

# Exposure-matched baseline, used ONLY for the duration-sensitive endpoints (event rate,
# fraction of cells active) that are compared across epochs -- see epoch_analysis'
# TRACE_MATCHED_WINDOW_S. The confirmatory amplitude contrasts keep the 35 s TFC_REFERENCE_EPOCH
# they were locked with; mean per-event amplitude is a per-event quantity and so is not biased
# by an unequal window, which is why both baselines can coexist without either being wrong.
TFC_MATCHED_REFERENCE_EPOCH = 'pre_tone_matched'

# The epochs whose decomposition figures are directly comparable to one another: all
# TRACE_MATCHED_WINDOW_S long, so fraction-active and event rate mean the same thing in each.
TFC_MATCHED_EPOCHS = (TFC_MATCHED_REFERENCE_EPOCH, TFC_TRACE_EPOCH, TFC_POST_SHOCK_EPOCH,
                      TFC_POST_SHOCK_LATE_EPOCH)

# ** The epochs the JOINT group x epoch specificity test puts on its profile. ** A strict subset
# of TFC_MATCHED_EPOCHS: every member is TRACE_MATCHED_WINDOW_S long AND present on every
# non-truncated trial.
#
# TFC_POST_SHOCK_LATE_EPOCH is held out for the same raggedness reason it is held out of
# TFC_RATE_MODEL_EPOCHS: it does not exist on a final trial whose recording stops shortly after
# the shock (observed as little as 20.5 s -- see epoch_analysis.POST_SHOCK_LATE_ONSET_S), so
# requiring it would cut the exposure-matched trial set for every mouse whose recording ran
# short, and the joint test needs each mouse to contribute a COMPLETE profile. It stays a
# descriptive column on the decomposition grid, where a per-column coverage note is enough.
TFC_MATCHED_PROFILE_EPOCHS = (TFC_MATCHED_REFERENCE_EPOCH, TFC_TRACE_EPOCH, TFC_POST_SHOCK_EPOCH)

# ** The four components of population calcium activity, defined ONCE. ** Consumed by both
# plot_decomposition_grid (one row each) and build_mouse_epoch_profile / the joint epoch tests
# (one test each), so a component cannot be defined one way on the figure and another way in the
# test that annotates it. Ordered along the decomposition chain:
#     overall_rate = fraction_active x rate_active
# with per-event amplitude the separate magnitude term.
#
# key       : stable identifier; used in stats filenames and BH-FDR family member names.
# frame     : which of _decomposition_grid_frames' outputs to read.
# pairable  : whether a WITHIN-CELL profile is possible. Fraction active is a proportion computed
#             OVER a mouse's cells, so it has no per-cell value to pair -- it is inherently a
#             per-mouse quantity and its profile is unpaired by nature, not by choice.
# ci_scale  : passed to mouse_contrast_ci. 'log' means the column is ALREADY log-transformed.
_DecompComponent = collections.namedtuple(
    '_DecompComponent', 'key label frame col ci_scale ci_unit pairable')

_DECOMPOSITION_COMPONENTS = (
    _DecompComponent('fraction_active', 'Fraction active', 'fraction_active',
                     'fraction_active', 'difference_only', '', False),
    _DecompComponent('rate_active', 'Rate | active', 'active_only',
                     'rate_active', 'linear', '/s', True),
    _DecompComponent('population_rate', 'Population event rate', 'all',
                     'overall_rate', 'linear', '/s', True),
    _DecompComponent('amplitude', 'Per-event amplitude', 'amplitude',
                     'log_amplitude', 'log', '', True),
)
_DECOMPOSITION_COMPONENTS_BY_KEY = {c.key: c for c in _DECOMPOSITION_COMPONENTS}

# The component whose joint epoch test is the one to lead with, matching the PRIMARY endpoint.
# The other three are the decomposition of the activity budget around it.
PRIMARY_PROFILE_COMPONENT = 'amplitude'

# ** The components the PAPER-FACING figures show. ** A strict subset of the four above, resolved
# through _DECOMPOSITION_COMPONENTS_BY_KEY so a component cannot be defined one way on the paper
# figure and another way on the internal grid or in the test that annotates it.
#
# 'rate_active' and the total amplitude-rate panel are deliberately NOT here. They are diagnostics
# of the identity overall_rate = fraction_active x rate_active -- mathematically useful, and kept
# in full in the TFC_cond output as supplement material -- but they are not separate biological
# claims, and putting five quantities in front of a reader to make a three-quantity point is how
# a figure stops being read. See docs/sp_rates_lmm.md for what else is supplement.
PAPER_COMPONENT_KEYS = ('fraction_active', 'population_rate', 'amplitude')

# The two response windows whose within-cell elevation over TFC_REFERENCE_EPOCH is confirmatory
# (holm_correct_confirmatory's second and third members). Every other epoch's delta is
# descriptive.
TFC_CONFIRMATORY_RESPONSE_EPOCHS = (TFC_TRACE_EPOCH, TFC_POST_SHOCK_EPOCH)

# ** Early vs late conditioning. ** Splits the five tone-shock trials into the first two and the
# last three, to ask whether a group difference is TONIC (present from the first trial, i.e. a
# property of the drug) or DEVELOPS as conditioning proceeds.
#
# `trial_numbers` are 1-BASED trial numbers, which is how the protocol is described and discussed.
# The event table's `trial` column is the session's own 0-based `trial_idx` (see
# _iter_event_windows), so split_by_conditioning_phase converts exactly once, at its one point of
# use, rather than leaving two conventions loose in the module.
_ConditioningPhase = collections.namedtuple('_ConditioningPhase', 'key label trial_numbers')

CONDITIONING_PHASES = (
    _ConditioningPhase('early', 'Early conditioning\n(trials 1-2)', (1, 2)),
    _ConditioningPhase('late', 'Late conditioning\n(trials 3-5)', (3, 4, 5)),
)

# ** The epochs that may be modelled JOINTLY -- i.e. treated as a set of disjoint observations
# of one trial. ** TFC_EPOCHS is deliberately NOT that set: it carries two baselines,
# TFC_REFERENCE_EPOCH (35 s) and TFC_MATCHED_REFERENCE_EPOCH (its last 20 s), which OVERLAP IN
# TIME. Both are wanted -- see their constants above -- but only ever one at a time.
#
# Feeding both to a model with an epoch factor is not a mild inefficiency, it is a
# duplicated-data bug: 20 s of every baseline window enters the likelihood twice, at a rate that
# is near-identical once a log-exposure offset normalizes it. The duplicate inflates the
# baseline's apparent precision, corrupts any epochs-within-trial variance component, and (for
# the Bambi NB-GLMM in fit_rate_group_epoch_model) leaves NUTS on a near-collinear ridge where
# the step size collapses -- observed as a sampler that runs for hours instead of about a
# minute. Anything with an epoch factor must take its epochs from HERE, not from TFC_EPOCHS.
#
# TFC_POST_SHOCK_LATE_EPOCH DOES belong here, unlike TFC_MATCHED_REFERENCE_EPOCH, and the
# difference is purely one of time: measured from shock offset, post_shock occupies 0-20 s,
# post_shock_late 90-110 s, and pre_tone the last 35 s of the 198 s ITI (163-198 s). The late
# window shares no frame with any other member, so it adds a genuine fifth level to the epoch
# factor rather than a duplicate of an existing one.
TFC_DISJOINT_EPOCHS = (TFC_REFERENCE_EPOCH, 'tone', TFC_TRACE_EPOCH, TFC_POST_SHOCK_EPOCH,
                       TFC_POST_SHOCK_LATE_EPOCH)

# ** The epochs the SECONDARY RATE MODEL puts on its epoch factor. ** A strict subset of
# TFC_DISJOINT_EPOCHS: disjointness is necessary to enter a joint model, but not sufficient.
#
# TFC_POST_SHOCK_LATE_EPOCH is excluded for three reasons, only the last of which is about speed:
#
#   1. It has no role here. It is a DESCRIPTIVE window that exists to carry the within-cell
#      early-vs-late AMPLITUDE contrast. The rate model is the prespecified secondary event-rate
#      endpoint, and adding a descriptive epoch to it enlarges a model that makes claims without
#      contributing to any of them.
#   2. It is ragged in a way that matters HERE specifically. The window is absent on final trials
#      whose recording stops early, so its rows skew toward low trial indices -- and `trial` is a
#      continuous covariate in this very model. Its epoch dummy is therefore correlated with
#      `trial`, which is a confound the amplitude contrasts (differenced within cell, within
#      trial) never face.
#   3. Empirically it makes NUTS pathological. Measured on this model: 4 epochs / 340 rows
#      samples in ~27 s, while 5 epochs samples for >280 s WITHOUT FINISHING -- and it does so
#      even when the fifth epoch is made artificially balanced across all trials, so the cost is
#      the extra level in an already weakly-identified group x epoch interaction, not the
#      raggedness. This is NOT the duplicated-data ridge described above (all five windows are
#      verified pairwise disjoint, across trial indices too); it is the plain cost of two more
#      interaction parameters the data cannot pin down.
#
# Keeping this set at four also means the reported secondary rate results are the SAME model that
# produced them before post_shock_late existed, so they need no reinterpretation.
TFC_RATE_MODEL_EPOCHS = (TFC_REFERENCE_EPOCH, 'tone', TFC_TRACE_EPOCH, TFC_POST_SHOCK_EPOCH)

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


def restrict_to_shared_trials(df_fine, epoch_a, epoch_b):
    """
    Restrict a fine event table to the (mouse, trial) cells where BOTH epochs are present.

    Needed because an epoch can be genuinely ABSENT on some trials: 'post_shock_late' does not
    exist on a final trial whose recording stops shortly after the shock (see
    epoch_analysis.get_epoch_frames, which returns None there rather than inventing a window).

    Without this, a within-cell early-vs-late delta would pool a different set of trials on each
    side -- e.g. post_shock over trials 1-4 against post_shock_late over trials 1-3 -- which
    reintroduces exactly the trial-index misalignment the late window exists to remove. The
    contrast is only interpretable at matched indices, so both sides are cut to the same trials
    here, PER MOUSE: recordings are ragged, and cutting every mouse to the globally-shared trials
    would discard good data from the mice whose recordings ran long. Each cell's delta is then
    trial-matched within itself, which is what the within-cell contrast actually requires.

    Returns (restricted_df, coverage), where coverage is a per-mouse DataFrame of how many trials
    survived -- callers are expected to REPORT it rather than let the restriction happen silently.
    """
    pair = df_fine[df_fine['epoch'].isin([epoch_a, epoch_b])]
    if pair.empty:
        raise RuntimeError(f'restrict_to_shared_trials: no rows for {epoch_a!r} or {epoch_b!r}.')
    n_epochs = pair.groupby(['mouse', 'trial'])['epoch'].nunique()
    shared = set(n_epochs[n_epochs == 2].index)
    if not shared:
        raise RuntimeError(
            f'restrict_to_shared_trials: no (mouse, trial) has BOTH {epoch_a!r} and {epoch_b!r}, '
            f'so the contrast has no trial-matched data at all. If {epoch_b!r} is a late '
            f'post-shock window, every recording is too short for it -- lower '
            f'epoch_analysis.POST_SHOCK_LATE_ONSET_S or drop the contrast.')
    keep = pair[[(m, t) in shared for m, t in zip(pair['mouse'], pair['trial'])]]
    total = df_fine.groupby('mouse')['trial'].nunique().rename('n_trials_total')
    kept = keep.groupby('mouse')['trial'].nunique().rename('n_trials_matched')
    coverage = pd.concat([total, kept], axis=1).fillna(0).astype(int).reset_index()
    return keep, coverage


def split_by_conditioning_phase(df_fine, phases=CONDITIONING_PHASES):
    """Split a fine event table into early / late conditioning sub-frames (see
    CONDITIONING_PHASES), plus the per-mouse trial coverage behind the split.

    Returns ({phase key: sub-frame}, coverage DataFrame). The coverage table is not optional
    bookkeeping: trial counts are ragged across mice (recordings stop early), so how many trials
    a mouse actually contributed to each phase is part of what the estimate rests on and is
    written out beside it.

    Raises if any mouse contributes zero trials to either phase -- that mouse could not appear in
    both columns of the figure, and quietly dropping it from one would make the two columns rest
    on different animals while looking like a within-cohort comparison.
    """
    trial_numbers = df_fine['trial'].to_numpy() + 1   # 0-based trial_idx -> 1-based trial number
    frames = {phase.key: df_fine[np.isin(trial_numbers, phase.trial_numbers)] for phase in phases}

    rows = []
    for mouse, sub in df_fine.groupby('mouse'):
        sub_numbers = sub['trial'] + 1
        row = {'mouse': mouse, 'group': sub['group'].iloc[0],
               'n_trials_total': int(sub['trial'].nunique())}
        for phase in phases:
            n = int(sub.loc[sub_numbers.isin(phase.trial_numbers), 'trial'].nunique())
            if n == 0:
                raise RuntimeError(
                    f'split_by_conditioning_phase: mouse {mouse} has no trials in phase '
                    f'{phase.key!r} (wanted 1-based trials {phase.trial_numbers}, has '
                    f'{sorted(sub_numbers.unique())}). Both phases must be populated for every '
                    f'mouse or the two columns rest on different cohorts.')
            row[f'n_trials_{phase.key}'] = n
        rows.append(row)
    return frames, pd.DataFrame(rows)


def restrict_to_exposure_matched_trials(df_fine, epochs, window_seconds=TRACE_MATCHED_WINDOW_S,
                                        tol_seconds=0.5):
    """
    Restrict a fine event table to the (mouse, trial) cells where EVERY epoch in `epochs` is
    present AND its window is `window_seconds` long.

    TFC_MATCHED_EPOCHS are called "exposure-matched" because each is defined as
    TRACE_MATCHED_WINDOW_S long -- but the TRACE window on TRIAL 1 is only 15 s, not 20 s
    (tone_onsets_def[0]=185 rather than an intended 180, see TraceFearCondSession.__init__ and
    the trace = [15, 20, 20, 20, 20] s note in docs/sp_rates_lmm.md section 2.4). Since
    aggregate_over_trials SUMS exposure_seconds across trials, the trial-pooled comparison is
    therefore ~95 s of trace against ~100 s of pre_tone_matched.

    That 5% mismatch does not touch mean per-event AMPLITUDE, which is a per-event quantity --
    which is why the confirmatory contrasts can and do keep the unmatched 35 s
    TFC_REFERENCE_EPOCH. It matters for the DURATION-SENSITIVE decomposition components:
    P(active) = 1 - exp(-lambda*T) rises with T at a fixed underlying rate, and conditioning on
    N>0 makes rate-among-active duration-dependent too. Comparing those across epochs at unequal
    exposure shows a difference that is partly pure window length.

    ** The restriction is derived from the MEASURED exposure_seconds, never from a trial index. **
    Hard-coding "trials 2-5" would encode the nominal protocol timing, which docs section 8
    forbids for exactly the reason correction #14 records: session timing is data-dependent and
    trial count varies per mouse. Testing the measured window instead drops trial 1 BECAUSE its
    trace window is short, and drops a truncated final trial BECAUSE its window is missing, under
    one rule that stays correct if either assumption changes.

    Returns (restricted_df, coverage) in the same shape as restrict_to_shared_trials -- the
    restricted frame carries only `epochs`' rows, and coverage is a per-mouse DataFrame the
    caller is expected to REPORT rather than let the restriction happen silently.
    """
    epochs = tuple(epochs)
    sub = df_fine[df_fine['epoch'].isin(epochs)]
    if sub.empty:
        raise RuntimeError(f'restrict_to_exposure_matched_trials: no rows for any of {epochs}.')

    # A (mouse, trial) qualifies when it carries all len(epochs) windows and every one of them is
    # within tol of the nominal length. Checked on the per-(mouse, trial, epoch) window duration,
    # which every cell in that window shares, so max() is that window's own length.
    per_window = sub.groupby(['mouse', 'trial', 'epoch'], as_index=False)['exposure_seconds'].max()
    per_window['matched'] = (per_window['exposure_seconds'] - window_seconds).abs() <= tol_seconds
    ok = per_window.groupby(['mouse', 'trial'])['matched'].agg(['sum', 'size'])
    shared = set(ok[(ok['sum'] == len(epochs)) & (ok['size'] == len(epochs))].index)
    if not shared:
        raise RuntimeError(
            f'restrict_to_exposure_matched_trials: no (mouse, trial) has all of {epochs} at '
            f'{window_seconds} s (tol {tol_seconds} s), so there is no exposure-matched data at '
            f'all. Check TRACE_MATCHED_WINDOW_S against the durations actually passed to '
            f'get_epoch_frames.')

    keep = sub[[(m, t) in shared for m, t in zip(sub['mouse'], sub['trial'])]]
    total = df_fine.groupby('mouse')['trial'].nunique().rename('n_trials_total')
    kept = keep.groupby('mouse')['trial'].nunique().rename('n_trials_matched')
    coverage = pd.concat([total, kept], axis=1).fillna(0).astype(int).reset_index()
    return keep, coverage


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

    ** Restricted to TFC_RATE_MODEL_EPOCHS. ** The rate model puts every epoch on a single epoch
    factor, so it must not see the two time-overlapping baselines at once -- see
    TFC_DISJOINT_EPOCHS for exactly what goes wrong. It is filtered to the narrower
    TFC_RATE_MODEL_EPOCHS rather than to TFC_DISJOINT_EPOCHS because being disjoint in time is
    necessary but not sufficient to belong in THIS model; see that constant for why the late
    post-shock window is held out. Filtered HERE rather than at the call site so that a future
    caller cannot reintroduce either problem by passing df_fine straight through.
    """
    sub = df_fine[df_fine['epoch'].isin(TFC_RATE_MODEL_EPOCHS)]
    if sub.empty:
        raise RuntimeError('build_mouse_trial_epoch_rate_table: no rows for any of '
                           f'{TFC_RATE_MODEL_EPOCHS} -- check the epoch names in df_fine.')
    agg = sub.groupby(['mouse', 'group', 'trial', 'epoch'], as_index=False).agg(
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


def compute_epoch_delta_table(df_fine, epoch, reference_epoch=TFC_REFERENCE_EPOCH,
                              pair_within_trial=False):
    """
    CO-PRIMARY endpoint's cell selection + delta computation (plan section 2). For each cell,
    pool n_events/sum_amplitude/exposure_seconds across all trials separately for `epoch` and
    `reference_epoch` (aggregate_over_trials, matching the primary endpoint's own trial-pooling),
    keep only cells with >=1 event in BOTH pooled epochs (filter_amplitude_rows on each side,
    then inner-join on (mouse, group, cell)), and compute
    delta_log_amplitude = log_amplitude[epoch] - log_amplitude[reference_epoch].

    This REPLACES a group x epoch interaction fit directly on the cell x trial x epoch table
    (only a mouse random intercept, no cell nesting -- a cell could contribute up to 20 rows
    there, up to 5 trials x 5 epochs, which is exactly the pseudoreplication the reviewer flagged:
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

    pair_within_trial : False (default, and what every CONFIRMATORY caller uses) -- pool each
        epoch across trials first, as described above, then difference. True -- difference each
        cell against ITS OWN TRIAL's reference window, then average those per-trial deltas within
        the cell.

        Why the option exists. Pooling before differencing makes the reference a MIXTURE across
        trial indices, which is harmless when both windows are present on the same trials and
        misleading when they are not: post_shock exists on every trial while pre_tone does not
        exist on trial 1, so post_shock-vs-pre_tone differences a five-trial response against a
        four-trial baseline whose remaining members all sit 163 s after a preceding shock. The
        within-trial pairing removes that misalignment directly, at the cost of requiring a cell
        to be active in both windows of the SAME trial (a stricter selection -- report n_cells).
        It is a second, independent route to the same problem TFC_POST_SHOCK_LATE_EPOCH exists to
        solve; agreement between the two is the point of having both.

        ** The per-trial deltas are averaged back down to ONE ROW PER CELL. ** That is not a
        presentational choice. fit_epoch_delta_model carries only a mouse random intercept, so
        handing it cell x trial rows would let a single cell contribute up to five of them and
        would reproduce exactly the pseudoreplication that correction #4 removed from the
        co-primary endpoint. The averaging keeps the returned table structurally identical to the
        pooled one, so both feed the same model with the same unit of observation.

    Returns a tidy DataFrame: mouse, group, cell, log_amplitude_epoch, log_amplitude_reference,
    delta_log_amplitude. (Under pair_within_trial the two log_amplitude columns are the cell's
    means over the trials that survived the pairing, so their difference remains exactly
    delta_log_amplitude.)
    """
    keys = ['mouse', 'group', 'cell'] + (['trial'] if pair_within_trial else [])
    if pair_within_trial:
        # No aggregate_over_trials: keep the fine table's own (mouse, trial, epoch, cell) rows so
        # the join can match on trial. filter_amplitude_rows still applies per side, so the
        # "active in both" rule now means active in both windows OF THE SAME TRIAL.
        epoch_amp = filter_amplitude_rows(df_fine[df_fine['epoch'] == epoch])[keys + ['log_amplitude']]
        ref_amp = filter_amplitude_rows(df_fine[df_fine['epoch'] == reference_epoch])[
            keys + ['log_amplitude']]
    else:
        epoch_amp = filter_amplitude_rows(aggregate_over_trials(df_fine, epoch))[
            keys + ['log_amplitude']]
        ref_amp = filter_amplitude_rows(aggregate_over_trials(df_fine, reference_epoch))[
            keys + ['log_amplitude']]
    merged = epoch_amp.merge(ref_amp, on=keys, suffixes=('_epoch', '_reference'))
    if merged.empty:
        raise RuntimeError(f'compute_epoch_delta_table: no cells with >=1 event in BOTH '
                           f'{epoch!r} and {reference_epoch!r}'
                           f'{" ON THE SAME TRIAL" if pair_within_trial else ""}.')
    merged['delta_log_amplitude'] = merged['log_amplitude_epoch'] - merged['log_amplitude_reference']
    if pair_within_trial:
        # Collapse the per-trial deltas to one row per cell -- see the docstring; a cell
        # contributing several rows to a mouse-intercept-only model is the bug this avoids.
        merged = merged.groupby(['mouse', 'group', 'cell'], as_index=False)[
            ['log_amplitude_epoch', 'log_amplitude_reference', 'delta_log_amplitude']].mean()
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


def compute_all_epoch_deltas(df_fine, epochs=TFC_DISJOINT_EPOCHS, reference_epoch=TFC_REFERENCE_EPOCH,
                             confirmatory_epochs=TFC_CONFIRMATORY_RESPONSE_EPOCHS,
                             reference_group='mCherry'):
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
    Holm family (see holm_correct_confirmatory for its three members), and they are not in the
    secondary BH-FDR family either (build_secondary_fdr_table), because their role is to
    characterize a null rather than to test a hypothesis. `confirmatory_epochs` is recorded per
    row purely so a reader of the output table can tell which rows carry inferential weight --
    since the epoch split it is TWO of them, trace and post_shock.

    TFC_MATCHED_REFERENCE_EPOCH is skipped: it is the same baseline as `reference_epoch`
    measured over a shorter window, so its delta against `reference_epoch` is a window-length
    artifact rather than an epoch effect.

    Returns (summary_df, fits) where summary_df has one row per (epoch, non-reference group) with
    the effect estimate and 95% interval, and fits is dict epoch -> fit_epoch_delta_model output.
    """
    rows, fits = [], {}
    for epoch in [e for e in epochs
                  if e not in (reference_epoch, TFC_MATCHED_REFERENCE_EPOCH)]:
        # Trial-match before differencing. A no-op for epochs present on every trial (tone,
        # trace, post_shock), but load-bearing for post_shock_late, which is absent on final
        # trials whose recording stops early -- without it that row would pool a late window over
        # a SHORTER set of trials than its own pre_tone reference, confounding the epoch
        # comparison with trial position (photobleaching, arousal) rather than isolating it.
        matched, _coverage = restrict_to_shared_trials(df_fine, epoch, reference_epoch)
        delta_df = compute_epoch_delta_table(matched, epoch, reference_epoch)
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
                'is_confirmatory': epoch in confirmatory_epochs,
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


def holm_correct_confirmatory(family_pvalues):
    """
    Holm correction across the ENTIRE confirmatory family. This is the complete confirmatory
    multiplicity burden -- everything else in this module is secondary (BH-FDR, see fdr_correct
    in caban.single_unit_common) or purely descriptive/sensitivity.

    family_pvalues : ordered dict of hypothesis name -> raw omnibus p-value. The family is
                     THREE tests (see run_sp_rates_lmm, which is the only caller):

      'trace_amplitude'      absolute trace-period amplitude group omnibus. The GLOBAL
                             hypothesis: does the manipulation change per-event amplitude during
                             the trace interval at all? Kept as a member in its own right
                             because a uniform shift across the whole session is biologically
                             real and produces NO epoch delta -- making only the deltas
                             confirmatory would define that result out of existence.
      'trace_vs_baseline'    within-cell trace-minus-pre_tone delta group omnibus. Is the effect
                             SPECIFIC to the trace interval?
      'post_shock_vs_baseline'  the same within-cell delta for the post-shock response window.
                             Is the effect specific to the post-shock window? Added because the
                             TFC literature supports two distinct loci (TFC_POST_SHOCK_EPOCH's
                             comment) and testing only the trace one silently picks a side.

    Three tests rather than two costs almost nothing -- Holm's smallest threshold moves from
    alpha/2 to alpha/3 -- and buys a confirmatory family that matches the actual hypotheses.

    ** The post-shock member is prespecifiable in the honest sense. ** It is being added before
    any properly-windowed post-shock result exists: until the epoch split in
    caban.epoch_analysis, 'post_shock' meant the whole 198 s inter-trial interval, so no
    post-shock response window has been fit or inspected. Lock this file's plan before running.

    Returns dict keyed by the same names, each {p_raw, p_holm, reject} at alpha=0.05.
    """
    names = list(family_pvalues)
    reject, p_holm, _, _ = multipletests([family_pvalues[n] for n in names],
                                         alpha=0.05, method='holm')
    return {n: {'p_raw': float(family_pvalues[n]), 'p_holm': float(ph), 'reject': bool(rj)}
            for n, ph, rj in zip(names, p_holm, reject)}


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
        fraction, each group vs control),
      - the joint group x epoch specificity permutation tests, ONE PER DECOMPOSITION COMPONENT
        (fit_and_report_epoch_interaction). These four are strongly DEPENDENT -- the components
        are one exact identity computed over overlapping cells -- which BH tolerates (it is valid
        under positive dependence) but which means their q-values describe one decomposition and
        must not be counted as four independent findings.

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
    methodological review. The epochs of a single trial are not independent replicates of
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


RATE_CONTRAST_HDI_PROB = 0.94


def summarize_rate_group_epoch_contrasts(rate_fit, epochs=(TFC_TRACE_EPOCH, TFC_POST_SHOCK_EPOCH),
                                         reference_group='mCherry',
                                         reference_epoch=TFC_REFERENCE_EPOCH,
                                         hdi_prob=RATE_CONTRAST_HDI_PROB):
    """
    Per-epoch rate RATIOS vs control, read out of the already-fitted NB mixed model.

    ** This fits nothing and tests nothing. ** fit_rate_group_epoch_model's own summary_text
    reports the raw interaction coefficients, which are differences-of-differences against the
    reference epoch -- not the quantity anyone writes in a Results section. What a reader wants is
    "hM4D's event rate during trace was X times control's", and that is
    exp(group main effect + that group x epoch interaction), formed draw by draw so the interval
    is a genuine posterior interval on the ratio rather than a delta-method approximation of one.

    The rate endpoint stays SECONDARY and stays out of both multiplicity families: there is no
    p-value here to correct (see build_secondary_fdr_table), and reformatting a posterior does not
    change that.

    ** The baseline differs from the figures', deliberately. ** This model's epoch factor is
    TFC_RATE_MODEL_EPOCHS, whose reference is the 35 s `pre_tone`; the paper figures use the 20 s
    `pre_tone_matched`. `trace` and `post_shock` are the SAME windows in both -- only the epoch the
    contrast is anchored to differs, and a group ratio computed at a given epoch is anchored to
    that epoch, not to the baseline. The caveat is carried in the returned frame's `note` so it
    cannot be dropped in transit.

    Returns a DataFrame: group, epoch, rate_ratio (posterior median), hdi_lo, hdi_hi, hdi_prob.
    """
    posterior = rate_fit['idata_full'].posterior
    group_term = f"C(group, Treatment('{reference_group}'))"
    epoch_term = f"C(epoch, Treatment('{reference_epoch}'))"
    interaction_term = f'{group_term}:{epoch_term}'
    for name in (group_term, interaction_term):
        if name not in posterior:
            raise KeyError(
                f"summarize_rate_group_epoch_contrasts: {name!r} is not in the posterior. The "
                f"reference_group/reference_epoch passed here must match the ones "
                f"fit_rate_group_epoch_model was fit with (its formula was: "
                f"{rate_fit['formula_full']!r}).")

    rows = []
    for group in [g for g in DREADD_DISPLAY_ORDER if g != reference_group]:
        main = posterior[group_term].sel({f'{group_term}_dim': group})
        for epoch in epochs:
            if epoch == reference_epoch:
                log_ratio = main
            else:
                # bambi labels an interaction level 'group, epoch' (verified against the fitted
                # model, not assumed) -- one coordinate per cell of the group x epoch table.
                log_ratio = main + posterior[interaction_term].sel(
                    {f'{interaction_term}_dim': f'{group}, {epoch}'})
            draws = np.exp(log_ratio.values.ravel())
            lo, hi = az.hdi(draws, hdi_prob=hdi_prob)
            rows.append({'group': group, 'epoch': epoch,
                         'rate_ratio': float(np.median(draws)),
                         'hdi_lo': float(lo), 'hdi_hi': float(hi), 'hdi_prob': hdi_prob})
    out = pd.DataFrame(rows)
    out['note'] = (f'NB mixed model, epoch factor anchored to {reference_epoch!r} '
                   f'(35 s), not to the 20 s matched baseline the figures use.')
    return out


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

def mouse_label_permutation_test(stat_fn, mice_per_group, n_perm=20000, seed=0,
                                 restrict_to_groups=None):
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

    restrict_to_groups : which groups' mice are EXCHANGEABLE under the null. None (default)
                     means all of them, which tests the GLOBAL null "no group differs from any
                     other". A sequence of group names restricts the shuffle to those groups'
                     mice, testing the PAIRWISE null "these groups do not differ from each
                     other", assuming nothing about the groups left out.

                     ** These are different hypotheses and can give materially different
                     p-values. ** For a contrast between two DREADD groups, the default lets a
                     permuted 'hM3D' bucket contain mCherry mice -- so mCherry's between-mouse
                     spread enters the null distribution even though mCherry appears nowhere in
                     the statistic. When the control is the most variable group, that widens the
                     null and makes the observed contrast look less extreme. Measured on the
                     trace-epoch fraction-active hM3D-vs-hM4D contrast, where mCherry's
                     between-mouse SD is 0.131 against hM3D's 0.032: the global null gives
                     p = 0.099, the pairwise null over the 11 DREADD mice gives p = 0.026.

                     Neither is wrong; they answer different questions. The default stays the
                     global null because every existing call site was computed under it and its
                     numbers are reported. Pass this explicitly when the claim is specifically
                     "group A differs from group B".

                     (Filtering `mice_per_group` before the call has the same effect -- the
                     statistic factories use `.get`, so an unmapped mouse falls into neither
                     side. This parameter exists so the choice is visible and documented at the
                     call site rather than being an easily-missed dict comprehension.)

    Returns dict(observed, p_two_sided, n_perm, n_finite, null=ndarray of length n_perm).
    """
    if restrict_to_groups is not None:
        restrict_to_groups = tuple(restrict_to_groups)
        missing = [g for g in restrict_to_groups if g not in mice_per_group]
        if missing:
            raise KeyError(f'mouse_label_permutation_test: restrict_to_groups names {missing}, '
                           f'absent from mice_per_group ({sorted(mice_per_group)}).')
        if len(restrict_to_groups) < 2:
            raise ValueError('mouse_label_permutation_test: restrict_to_groups needs >=2 groups '
                             'to have anything to exchange.')
        mice_per_group = {g: mice_per_group[g] for g in restrict_to_groups}
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

    # ** Non-finite draws must be removed, not compared. ** A stat_fn returns NaN for a
    # degenerate split (see make_contrast_stat / make_group_epoch_interaction_stat), and
    # `abs(nan) >= abs(observed)` is False -- so leaving NaNs in the null would silently count
    # them as "not extreme" and DEFLATE the p-value toward 1/(n_perm+1). That is an
    # anti-conservative failure that looks exactly like a real result: on a degenerate fixture it
    # produced p <= 0.05 on 87% of true nulls. The p is therefore computed over the finite draws
    # only, and a null that is mostly non-finite is an error rather than a small p-value.
    if not np.isfinite(observed):
        raise RuntimeError(
            'mouse_label_permutation_test: the OBSERVED statistic is not finite, so no p-value '
            'is defined. This usually means the statistic is degenerate on these data (e.g. a '
            'quantity with no between-mouse variation at all), not that the effect is large.')
    finite = np.isfinite(null)
    n_finite = int(finite.sum())
    if n_finite < 0.5 * n_perm:
        raise RuntimeError(
            f'mouse_label_permutation_test: only {n_finite} of {n_perm} permutations produced a '
            f'finite statistic. The null distribution is too degenerate to test against; fix the '
            f'statistic or drop the test rather than reporting a p-value from it.')
    # +1/+1 (conventional Monte Carlo correction) so a finite number of draws never reports p=0.
    n_as_extreme = int(np.sum(np.abs(null[finite]) >= np.abs(observed)))
    p_two_sided = (n_as_extreme + 1) / (n_finite + 1)
    return {'observed': observed, 'p_two_sided': float(p_two_sided), 'n_perm': n_perm,
            'n_finite': n_finite, 'null': null}


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
# Joint group x epoch specificity test
# ─────────────────────────────────────────────────────────────────────────────
#
# ** WHY THIS EXISTS. ** The confirmatory family answers "is there a group effect WITHIN the
# trace window" (trace_amplitude, rejected) and "is each response window's within-cell elevation
# over baseline group-dependent" (trace_vs_baseline p=0.998, post_shock_vs_baseline p=0.935,
# both null). Reading those three against each other -- significant here, not significant there
# -- and concluding the effect is or is not epoch-specific is the DIFFERENCE-OF-SIGNIFICANCE
# FALLACY: a difference between two p-values is not itself a test of anything. The question
# "does the group effect CHANGE across epochs" needs its own single test, which is what this is.
#
# It does NOT join or replace the confirmatory family. That family is locked at three members
# (docs/sp_rates_lmm.md section 4.1), and shrinking or re-forming it now that its two epoch
# members are known to be null would move the primary p_holm from 0.027 back toward 0.009 and
# would rightly read as outcome-driven. This is a SECONDARY test and enters the secondary BH-FDR
# family (build_secondary_fdr_table) alongside the run-structure permutation p-values.

def build_mouse_epoch_profile(df_fine, epochs=TFC_MATCHED_PROFILE_EPOCHS,
                              reference_epoch=TFC_MATCHED_REFERENCE_EPOCH, paired=True,
                              component=PRIMARY_PROFILE_COMPONENT):
    """
    One profile row per mouse: its mean elevation of `component` over `reference_epoch` in each
    non-reference epoch. This is the 17 x (len(epochs)-1) matrix the joint group x epoch
    permutation test permutes group labels over.

    component : a key of _DECOMPOSITION_COMPONENTS -- 'amplitude' (the default and the primary),
        'fraction_active', 'rate_active' or 'population_rate'. The component definitions are
        shared with plot_decomposition_grid, so the test that annotates a grid row is computed
        from exactly the same frame and column that row is drawn from.

        ** The delta is taken on the column's OWN scale ** -- a log-ratio for amplitude (whose
        column is already logged), a raw difference in events/s for the two rates, a raw
        difference in proportion for fraction active. Those are not commensurable with each
        other, which does not matter: make_group_epoch_interaction_stat standardizes each epoch's
        delta by its own pooled between-mouse SD, so the statistic is scale-free and each
        component's test is computed in the units its own contrasts are reported in.

    Differencing against the reference epoch, rather than modelling the epoch levels directly,
    removes the epoch MAIN effect by construction and leaves only the interaction -- the thing
    actually under test. A mouse whose amplitude is uniformly high contributes a flat profile of
    zeros regardless of how high; only a mouse whose trace/post-shock elevation differs from
    other mice's contributes signal.

    paired : True (default) -- the delta is computed WITHIN CELL, so a cell must appear in the
        component's frame in EVERY epoch in `epochs` to contribute. For amplitude that is
        compute_epoch_delta_table's "active in both epochs" rule extended to a triple
        intersection, and it cancels each cell's own baseline level the same way the co-primary
        contrast does. The intersection is stricter than any pairwise one, so n_cells is reported
        and is expected to be smaller.
        False -- each epoch's per-mouse mean is taken over THAT epoch's own cells and the means
        are differenced afterwards. This is the sensitivity variant: permutation validity never
        depended on the cell pairing (only the 17 group labels move), so a disagreement between
        the two is evidence that the triple intersection selects a special subpopulation, not
        that one of them is invalid.

        ** Ignored, with paired=False forced, for a component whose `pairable` is False. **
        Fraction active is a proportion computed OVER a mouse's cells; there is no per-cell value
        to pair, so its profile is per-mouse by construction and its "unpaired" variant is the
        only one that exists. That is a property of the quantity, not a limitation here, and it
        means its paired and unpaired tests are the same test rather than two.

    Expects `df_fine` ALREADY restricted to exposure-matched trials -- see
    restrict_to_exposure_matched_trials, and note this function does not apply that restriction
    itself, because the caller must report its coverage.

    Returns (profile_df, n_cells). profile_df has one row per mouse: mouse, group, and one
    'delta_{epoch}' column per non-reference epoch.
    """
    epochs = tuple(epochs)
    if reference_epoch not in epochs:
        raise ValueError(f'build_mouse_epoch_profile: reference_epoch {reference_epoch!r} must be '
                         f'one of epochs={epochs}.')
    response_epochs = [e for e in epochs if e != reference_epoch]
    if not response_epochs:
        raise ValueError('build_mouse_epoch_profile: need at least one non-reference epoch.')
    if component not in _DECOMPOSITION_COMPONENTS_BY_KEY:
        raise KeyError(f'build_mouse_epoch_profile: unknown component {component!r}; expected one '
                       f'of {sorted(_DECOMPOSITION_COMPONENTS_BY_KEY)}.')
    spec = _DECOMPOSITION_COMPONENTS_BY_KEY[component]
    # Not a silent fallback: a non-pairable component has no per-cell value in the first place,
    # so there is no pairing to decline. See the docstring.
    paired = paired and spec.pairable

    # Read every epoch's frame through the SAME helper the grid draws from, so the test and the
    # row it annotates can never diverge in how the component is defined.
    keys = ['mouse', 'group'] + (['cell'] if spec.pairable else [])
    per_epoch = {e: _decomposition_grid_frames(aggregate_over_trials(df_fine, e))[spec.frame]
                 [keys + [spec.col]].rename(columns={spec.col: 'value'})
                 for e in epochs}

    if paired:
        # Triple (or n-fold) intersection: successive inner joins on (mouse, group, cell), so a
        # cell survives only if it appears in the component's frame in EVERY epoch.
        merged = per_epoch[reference_epoch].rename(columns={'value': 'ref'})
        for e in response_epochs:
            merged = merged.merge(per_epoch[e].rename(columns={'value': f'val_{e}'}),
                                  on=['mouse', 'group', 'cell'])
        if merged.empty:
            raise RuntimeError(f'build_mouse_epoch_profile: no cell appears in the {component!r} '
                               f'frame for ALL of {epochs}, so the paired profile is empty. '
                               f'Try paired=False.')
        for e in response_epochs:
            merged[f'delta_{e}'] = merged[f'val_{e}'] - merged['ref']
        n_cells = int(len(merged))
        # Equal-mouse weighting: collapse each mouse's cells to that mouse's mean delta. Cell
        # count is group-correlated and itself post-treatment (docs section 4.3), so a
        # cell-weighted profile would let hM3D's larger active-cell count set the statistic.
        profile = merged.groupby(['mouse', 'group'], as_index=False)[
            [f'delta_{e}' for e in response_epochs]].mean()
    else:
        # Unpaired: per-mouse mean over each epoch's OWN cells, differenced afterwards. For a
        # non-pairable component the frame is already one row per mouse, so the groupby is an
        # identity and this is simply that component's per-mouse profile.
        means = {e: df.groupby(['mouse', 'group'], as_index=False)['value'].mean()
                 for e, df in per_epoch.items()}
        profile = means[reference_epoch].rename(columns={'value': 'ref'})
        for e in response_epochs:
            profile = profile.merge(means[e].rename(columns={'value': f'val_{e}'}),
                                    on=['mouse', 'group'])
        for e in response_epochs:
            profile[f'delta_{e}'] = profile[f'val_{e}'] - profile['ref']
        n_cells = int(sum(len(df) for df in per_epoch.values()))
        profile = profile[['mouse', 'group'] + [f'delta_{e}' for e in response_epochs]]

    if profile.empty:
        raise RuntimeError('build_mouse_epoch_profile: no mouse has a complete epoch profile.')
    return profile, n_cells


def make_group_epoch_interaction_stat(profile_df, delta_cols, reference='mCherry',
                                      groups=None):
    """
    stat_fn factory for mouse_label_permutation_test(): a SCALAR summary of how much the group
    effect varies across epochs, computed from build_mouse_epoch_profile()'s per-mouse profile.

    The statistic is the standardized sum of squared difference-of-differences:

        T = sum over non-reference groups g, over epochs e of
                ( ( mean_g[delta_e] - mean_ref[delta_e] ) / sd_pooled(delta_e) ) ** 2

    where each term is exactly the difference-of-differences the accompanying stats file reports
    with a confidence interval -- e.g. [hM3D - Ctl]_trace - [hM3D - Ctl]_pre_tone_matched.

    ** Why standardized. ** Without dividing by that epoch's pooled between-mouse SD, whichever
    delta happens to have the larger scale dominates the sum, and the test silently becomes a
    test about that one epoch. The SD is recomputed under each permutation from the permuted
    labels, so it is a function of the data being permuted, not a fixed nuisance constant.

    ** Why a scalar at all. ** mouse_label_permutation_test accumulates into
    np.empty(n_perm, dtype=float); a vector-valued statistic raises on assignment. Squaring and
    summing is also what makes the test two-sided in every epoch at once and directionless
    overall, which is right for an omnibus interaction: the alternative is "the profile shape
    differs", not "it differs upward".

    The permutation distribution is what carries validity here, so the choice of statistic
    affects POWER, not correctness -- any scalar function of the labelled profile gives a valid
    test. This one is chosen because each of its terms is separately reportable.

    Returns callable(mouse_to_group: dict[str, str]) -> float, np.nan on a degenerate split.
    """
    delta_cols = list(delta_cols)
    missing = [c for c in delta_cols if c not in profile_df.columns]
    if missing:
        raise KeyError(f'make_group_epoch_interaction_stat: profile_df lacks {missing}.')
    if groups is None:
        groups = [g for g in GROUP_ORDER if g in set(profile_df['group'])]
    if reference not in set(profile_df['group']):
        raise KeyError(f'make_group_epoch_interaction_stat: reference group {reference!r} not in '
                       f'profile_df.')
    non_ref = [g for g in groups if g != reference]
    if not non_ref:
        raise ValueError('make_group_epoch_interaction_stat: no non-reference group.')

    # Hoist every pandas operation out of the closure -- the same optimization make_contrast_stat
    # documents. A permutation changes only which group a mouse is labelled with, never which row
    # belongs to a mouse, so the per-mouse delta matrix is fixed across all n_perm draws and the
    # closure reduces to boolean-mask arithmetic on a 17 x len(delta_cols) array.
    mice = profile_df['mouse'].tolist()
    deltas = profile_df[delta_cols].to_numpy(dtype=float)  # (n_mice, n_epochs)
    mouse_index = {m: i for i, m in enumerate(mice)}

    def stat_fn(mouse_to_group):
        idx_by_group = {}
        for g in [reference] + non_ref:
            rows = [mouse_index[m] for m in mice if mouse_to_group.get(m) == g]
            if len(rows) < 2:
                # <2 mice gives no within-group variance, so the pooled SD is undefined.
                return np.nan
            idx_by_group[g] = rows
        ref_rows = deltas[idx_by_group[reference]]
        ref_mean = ref_rows.mean(axis=0)

        # Pooled between-mouse SD per epoch, over ALL groups under the current labelling.
        ss, dof = np.zeros(deltas.shape[1]), 0
        for g, rows in idx_by_group.items():
            block = deltas[rows]
            ss += ((block - block.mean(axis=0)) ** 2).sum(axis=0)
            dof += len(rows) - 1
        if dof <= 0:
            return np.nan
        sd = np.sqrt(ss / dof)
        usable = sd > 0
        if not np.any(usable):
            # Every epoch is perfectly flat: no between-mouse variation anywhere, so there is
            # nothing to standardize against and the statistic is genuinely undefined.
            return np.nan

        # An epoch with sd == 0 contributes 0, not NaN. sd is pooled over ALL groups, so sd == 0
        # means every mouse has the identical delta there -- which forces that epoch's
        # difference-of-differences to 0 too. Contributing 0 is therefore the correct value, and
        # NaN-ing the whole statistic over one degenerate epoch would discard the others.
        #
        # ** This is not cosmetic. ** A NaN returned here propagates into
        # mouse_label_permutation_test's null array, where `abs(null) >= abs(observed)` is False
        # for NaN -- so NaN draws would count as "not extreme" and deflate the p-value. Measured
        # on a deliberately degenerate fixture, that produced p <= 0.05 on 87% of true nulls.
        total = 0.0
        for g in non_ref:
            did = deltas[idx_by_group[g]].mean(axis=0) - ref_mean
            total += float(np.sum((did[usable] / sd[usable]) ** 2))
        return total
    return stat_fn


def epoch_interaction_contrasts(profile_df, delta_cols, reference='mCherry', scale='log'):
    """{delta_col: mouse_contrast_ci(...)} -- the difference-of-differences behind each term of
    make_group_epoch_interaction_stat's statistic, as estimates with 95% intervals.

    The joint test returns ONE p-value per component, which is the point: it answers
    epoch-specificity once rather than once per cell of the grid. But a single omnibus p says
    nothing about magnitude or direction, and at n=5/6/6 the result is expected to be null -- so
    the per-term estimates have to be reported beside it, or the null cannot be distinguished
    from "no effect of any size" (docs section 9).

    scale : the component's own ci_scale. 'log' for amplitude, whose delta is already a
        log-difference, so mouse_contrast_ci reports exp(diff) as the ratio without taking a
        second log (the double-log trap of correction #7, in interval form). The rate components'
        deltas are raw differences in events/s and the fraction-active delta is a difference in
        proportion -- both are differences of quantities that can be negative, so they are
        reported as 'difference_only'; a "fold-change in a difference" is not a meaningful
        summary and would be undefined wherever the reference delta crosses zero.
    """
    out = {}
    for col in delta_cols:
        per_group = {g: sub[col].to_numpy(dtype=float)
                     for g, sub in profile_df.groupby('group') if len(sub) >= 2}
        if reference not in per_group:
            raise KeyError(f'epoch_interaction_contrasts: reference {reference!r} has <2 mice.')
        out[col] = mouse_contrast_ci(per_group, reference=reference, scale=scale)
    return out


# A component's DELTA is a difference of that component between two epochs. Only the amplitude
# delta is a log-difference (hence a ratio on exponentiation); the others are differences of
# possibly-negative quantities, for which a ratio is not defined. Not the same thing as the
# component's own ci_scale, which describes its LEVEL.
_PROFILE_DELTA_CI_SCALE = {'amplitude': 'log'}


def fit_and_report_epoch_interaction(df_fine, stats_dir, mice_per_group, coverage,
                                     epochs=TFC_MATCHED_PROFILE_EPOCHS,
                                     reference_epoch=TFC_MATCHED_REFERENCE_EPOCH,
                                     components=None, n_perm=20000, seed=0,
                                     filename='secondary_epoch_interaction.txt'):
    """
    Run and report the joint group x epoch specificity test for EVERY decomposition component --
    the single answer, per component, to "is this effect epoch-specific", replacing the invalid
    practice of reading separate per-epoch p-values against each other.

    One test per component rather than one overall. The four components are one exact identity
    (overall_rate = fraction_active x rate_active, plus amplitude), so a single pooled test across
    all of them would answer a question nobody asks -- "did ANY component's profile change" -- and
    would be driven by whichever component has the largest between-mouse spread. Per component,
    each test is interpretable on its own row of the grid, and each is a genuine secondary
    hypothesis, so each is a BH-FDR family member.

    ** The four p-values are strongly DEPENDENT ** -- they are computed from overlapping cells and
    are linked by the decomposition identity. BH is valid under positive dependence, so the
    q-values stand, but they must be read as a set describing one decomposition, never as four
    independent findings. This is stated in the output file too.

    For each component the permutation runs twice: on the within-cell PAIRED profile (primary)
    and on the UNPAIRED one (sensitivity). For a non-pairable component (fraction active) the two
    coincide by construction and only one is reported. See build_mouse_epoch_profile.

    Returns {component key: dict(p_two_sided, observed, n_cells, n_mice, profile, contrasts,
    unpaired)}. The caller registers each p_two_sided in the secondary BH-FDR family.
    """
    components = tuple(components or [c.key for c in _DECOMPOSITION_COMPONENTS])
    response_epochs = [e for e in epochs if e != reference_epoch]
    results, lines = {}, [
        'SECONDARY: joint group x epoch specificity tests (mouse-label permutation).',
        '',
        f'Epochs: {reference_epoch} (reference) vs {", ".join(response_epochs)}. Every window is '
        f'{TRACE_MATCHED_WINDOW_S:g} s and every contributing trial carries all of them at that '
        f'length (restrict_to_exposure_matched_trials).',
        f'Trial coverage per mouse: {coverage["n_trials_matched"].min()}-'
        f'{coverage["n_trials_matched"].max()} of {coverage["n_trials_total"].max()}.',
        '',
        'H0 (per component): the group effect has the SAME profile across epochs, i.e. no '
        'group x epoch interaction.',
        'Statistic: standardized sum of squared difference-of-differences; group labels permuted '
        'across the 17 mice, each mouse keeping its whole epoch profile. Since the statistic is '
        'non-negative, the two-sided |null| >= |observed| rule is an upper-tail omnibus test. '
        f'{n_perm} permutations, seed {seed}.',
        '',
        'ONE TEST PER COMPONENT. Each p enters the SECONDARY BH-FDR family '
        '(secondary_fdr_family.csv) as its own member; report the FDR-adjusted value. None of '
        'them enters the three-member confirmatory Holm family, which is locked and unchanged.',
        '',
        '** These four p-values are strongly DEPENDENT. ** The components are one exact '
        'decomposition (overall_rate = fraction_active x rate_active, plus the amplitude term) '
        'computed over overlapping cells. BH remains valid under positive dependence, so the '
        'q-values stand -- but read them as a set describing one decomposition, never as four '
        'independent findings, and do not count how many cross 0.05.',
        '',
    ]

    for key in components:
        spec = _DECOMPOSITION_COMPONENTS_BY_KEY[key]
        delta_scale = _PROFILE_DELTA_CI_SCALE.get(key, 'difference_only')
        profile, n_cells = build_mouse_epoch_profile(df_fine, epochs, reference_epoch,
                                                     paired=True, component=key)
        delta_cols = [c for c in profile.columns if c.startswith('delta_')]
        perm = mouse_label_permutation_test(
            make_group_epoch_interaction_stat(profile, delta_cols),
            mice_per_group, n_perm=n_perm, seed=seed)
        perm_unpaired = None
        if spec.pairable:
            unpaired_profile, _ = build_mouse_epoch_profile(df_fine, epochs, reference_epoch,
                                                            paired=False, component=key)
            perm_unpaired = mouse_label_permutation_test(
                make_group_epoch_interaction_stat(unpaired_profile, delta_cols),
                mice_per_group, n_perm=n_perm, seed=seed)
        contrasts = epoch_interaction_contrasts(profile, delta_cols, scale=delta_scale)

        unit_note = ('log units' if delta_scale == 'log' else f'{spec.col} units')
        lines += [
            f'## {spec.label}  [{key}]',
            '',
            (f'  Cells: {n_cells} present in the {key!r} frame for ALL {len(epochs)} epochs '
             f'(within-cell paired). Mice: {len(profile)}.' if spec.pairable else
             f'  Per-mouse quantity -- a proportion computed OVER cells, so there is no '
             f'within-cell pairing to do and the paired/unpaired distinction does not apply. '
             f'Mice: {len(profile)}.'),
            '',
            f'  PAIRED (primary):      T = {perm["observed"]:.4f}, p = {perm["p_two_sided"]:.4f}'
            if spec.pairable else
            f'  PROFILE:               T = {perm["observed"]:.4f}, p = {perm["p_two_sided"]:.4f}',
        ]
        if perm_unpaired is not None:
            lines.append(f'  UNPAIRED (sensitivity): T = {perm_unpaired["observed"]:.4f}, '
                         f'p = {perm_unpaired["p_two_sided"]:.4f}')
        lines += [
            '',
            f'  Difference-of-differences, equal-mouse-weighted, Welch 95% intervals, in '
            f'{unit_note} -- "trace" means [group - Ctl]_trace - [group - Ctl]_{reference_epoch}:',
            '',
        ]
        for col in delta_cols:
            lines.append(f'    {col[len("delta_"):]}:')
            for line in format_contrast_ci_lines(contrasts[col], 'mCherry'):
                lines.append(f'      {line}')
        lines.append('')
        results[key] = {'p_two_sided': perm['p_two_sided'], 'observed': perm['observed'],
                        'n_cells': n_cells, 'n_mice': int(len(profile)), 'profile': profile,
                        'contrasts': contrasts, 'unpaired': perm_unpaired}

    lines += [
        'INTERPRETATION. A null here does NOT mean the manipulation has no effect -- the primary '
        'trace amplitude contrast is separately significant. It means the data do not support '
        'the stronger claim that the effect is SPECIFIC to any one epoch, i.e. the component is '
        'shifted broadly across TFC epochs. Read the difference-of-differences intervals for what '
        'magnitude of epoch-specificity remains admissible; at n=5/6/6 that range is wide.',
        '',
    ]
    write_text(os.path.join(stats_dir, filename), '\n'.join(lines) + '\n')
    return results


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


def _panel_contrasts(df_or_mouse_df, value_col, ci_scale, ci_unit, include_exc_vs_inh=False):
    """{'scale', 'unit', 'contrasts', 'exc_vs_inh'} for one panel's equal-mouse-weighted
    contrasts vs control.

    Kept separate from the drawing so a panel's estimates can be written to a companion file
    without being rendered onto the panel itself -- five interval blocks crowded into one figure
    row was strictly harder to read than five asterisks (see plot_decomposition's docstring).

    include_exc_vs_inh : also compute the DIRECT hM3D-vs-hM4D contrast, by re-running
        mouse_contrast_ci with hM4D as the reference and keeping the hM3D entry.

        ** This is not derivable from the two vs-control contrasts. ** Reading "Exc sits above
        zero, Inh sits below it, therefore Exc exceeds Inh" compares two contrasts that share a
        reference group, by eye, with no interval for the comparison being made -- the same
        difference-of-significance error the grid exists to prevent, rotated ninety degrees.

        The direct contrast's variance is Va/na + Vb/nb, where the vs-control ones are
        Va/na + Vc/nc and Vb/nb + Vc/nc. It is therefore NOT systematically wider or narrower --
        it drops the control's contribution and keeps both DREADD groups'. Which way it goes
        depends entirely on whether the CONTROL is the noisy group. In this dataset it is: trace
        fraction-active between-mouse SD is 0.131 for mCherry against 0.032 for hM3D, so the
        direct contrast is frequently the TIGHTEST of the three and can exclude the null where
        both vs-control intervals do not. That is a real feature of the comparison, not a
        paradox -- but see the multiplicity note below before reading much into it.

        ** These intervals carry no multiplicity correction and are in no declared family. **
        Sixteen of them appear on the grid. Treat one that excludes the null as an effect
        estimate worth following up, not as a discovery.

        Off by default: PANEL_HOLM_FAMILY is 'vs_control' precisely because that is where this
        design's question sits, and the standalone panels should not start carrying a contrast
        that spends no alpha there. The grid turns it on because the grid is read across
        conditions, which is exactly when someone will try to make this comparison by eye.
    """
    per_group = _mouse_values_per_group(df_or_mouse_df, value_col, group_order=GROUP_ORDER,
                                        panel_name='panel_contrasts')
    vals = {g: v.ravel() for g, v in per_group.items()}
    out = {'scale': ci_scale, 'unit': ci_unit,
           'contrasts': mouse_contrast_ci(vals, scale=ci_scale), 'exc_vs_inh': None}
    if include_exc_vs_inh:
        # Keyed on 'hM3D' with reference 'hM4D', so format_contrast_ci_lines' GROUP_ORDER
        # iteration renders exactly this one row and labels it 'Exc/Inh' -- no second formatter.
        out['exc_vs_inh'] = {'hM3D': mouse_contrast_ci(vals, reference='hM4D',
                                                       scale=ci_scale)['hM3D']}
    return out


def write_decomposition_contrasts_markdown(contrasts_by_panel, save_dir, filename,
                                           figure_has_stars=True, no_star_note=None,
                                           title='Decomposition panel contrasts'):
    """Write the decomposition figure's per-panel effect estimates + 95% intervals to a markdown
    file next to the figure.

    These numbers matter -- a null is only interpretable together with what its interval still
    admits, and a rate ratio needs its absolute difference beside it -- but they are reference
    material, not something to read off a panel. On the figure they competed with the data;
    here they can be read properly and quoted directly.

    figure_has_stars : whether the accompanying figure carries Holm-corrected significance
        brackets. True for plot_decomposition's per-epoch panels; False for
        plot_decomposition_grid, which is deliberately star-free. This is not a formatting
        detail -- the note it selects tells the reader how an interval here relates to what the
        figure shows, and the star note would be simply FALSE next to a figure with no stars.
    no_star_note : replaces the default `figure_has_stars=False` paragraph. That default is
        written about the four-component DECOMPOSITION and says so explicitly ("one exact
        decomposition, not independent phenotypes"), which is the right warning there and a
        false description of a star-free figure whose panels are one component across epochs or
        trial phases. A caller whose panels are not the decomposition supplies its own note
        rather than inheriting a wrong one.
    """
    lines = [
        f'# {title}',
        '',
        'Equal-mouse-weighted contrasts against the mCherry control (n = 5 hM3D / 6 hM4D / '
        '6 mCherry), Welch two-sample 95% intervals computed from the per-mouse means. Only '
        'per-mouse values enter these numbers; any cell-level display is descriptive.',
        '',
        'Ratios and absolute differences are reported together: a fold-change computed off a '
        'small base overstates the practical size of a change.',
        '',
    ]
    if figure_has_stars:
        lines += [
            '**These intervals are not multiplicity-corrected.** The asterisks on the figure are '
            f'Holm-corrected across the family set by `PANEL_HOLM_FAMILY` (currently '
            f'`{PANEL_HOLM_FAMILY!r}`), so a contrast whose interval excludes 1.0 here may still '
            'carry no star. Both are reported deliberately: the interval describes the effect, '
            'the star describes the corrected decision.',
            '',
        ]
    elif no_star_note is not None:
        lines += [no_star_note, '']
    else:
        lines += [
            '**These intervals are not multiplicity-corrected, and the figure carries no '
            'significance stars by design.** The components below are one exact decomposition '
            '(`overall_rate = fraction_active x rate_active`, plus the amplitude term), not '
            'independent phenotypes, so counting significant cells across this table is not a '
            'valid reading of it -- and comparing significance BETWEEN epochs is the '
            'difference-of-significance fallacy. Epoch specificity has exactly one test '
            '(`secondary_epoch_interaction.txt`); read these as effect sizes.',
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
        if payload.get('exc_vs_inh'):
            lines.append('')
            lines.append('Direct DREADD-vs-DREADD contrast (not derivable from the two rows '
                         'above, which share the control as their reference):')
            for line in format_contrast_ci_lines(payload['exc_vs_inh'], 'hM4D',
                                                 unit=payload['unit']):
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
                               ci_scale=None, ci_unit='', bracket_mode='axes',
                               jitter_width=0.34):
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
    jitter_width : passed straight through to draw_superplot_triplet -- the per-mouse x-offset
                spread within a group's column. Narrower than the 0.34 default tightens the
                cloud into a slimmer strip, which reads better in a multi-panel row where each
                panel's own footprint has also been narrowed (see plot_decomposition).
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
                           ci_unit=ci_unit, jitter_width=jitter_width)
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
                       epoch_order=TFC_DISJOINT_EPOCHS):
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
    (compute_all_epoch_deltas). The confirmatory rows (TFC_CONFIRMATORY_RESPONSE_EPOCHS) are
    marked; the others are descriptive.

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
    ax.set_title('Within-cell epoch deltas (confirmatory rows marked;\n'
                'the others are descriptive context for their null)', size='small')
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


def fit_and_report_epoch_delta(df_fine, response_epoch, stats_dir, save_dir,
                               n_trace_active_cells, reference_epoch=TFC_REFERENCE_EPOCH,
                               is_confirmatory=True, pair_within_trial=False):
    """
    Fit, write, and plot ONE within-cell epoch-delta contrast (response_epoch minus
    reference_epoch, per cell, pooled over trials).

    Factored out of run_sp_rates_lmm when the post-shock response window joined the trace
    interval as a co-equal confirmatory member. Both go through this single function so the two
    can never diverge in cell selection, model specification, or what gets reported -- a
    difference between them has to be a difference in the DATA, not in the code path.

    n_trace_active_cells : denominator for the cell-selection line only (len(df_trace_amp)), so
                           a reader can see what fraction of the primary endpoint's cells
                           survive the "active in BOTH epochs" requirement.

    is_confirmatory : whether this contrast is a member of the confirmatory Holm family. This is
                   NOT cosmetic -- it decides what the stats file CLAIMS about the p it reports,
                   and the same machinery serves descriptive contrasts (the early-vs-late
                   post-shock one) whose p spends no alpha and enters no multiplicity family.
                   Writing "CONFIRMATORY" over a descriptive estimate would be a false
                   inferential claim in a file someone will quote, so it is a required
                   distinction rather than a label. Descriptive output is named descriptive_*
                   so the status is legible from the filename alone.

    pair_within_trial : threaded to compute_epoch_delta_table -- see its docstring. Confirmatory
                   callers leave it False; the locked family's cell selection and pooling must
                   not change. The filename gains a _within_trial suffix when True so a paired
                   contrast can never overwrite its pooled counterpart.

    Output paths key on BOTH epochs whenever reference_epoch is not the default, so that two
    contrasts sharing a response epoch cannot silently overwrite each other's files. The two
    confirmatory calls keep their historical single-epoch names byte-identical.

    Returns the fit_epoch_delta_model() output.
    """
    if pair_within_trial and is_confirmatory:
        raise ValueError(
            'fit_and_report_epoch_delta: pair_within_trial=True is not available to a '
            'confirmatory contrast. The three-member Holm family is locked on the trial-POOLED '
            'differencing it was computed with (docs/sp_rates_lmm.md section 4.1); changing a '
            'locked member\'s cell selection after its result is known is a design change, not '
            'a code change. Report the paired version as descriptive alongside it.')
    slug = (response_epoch if reference_epoch == TFC_REFERENCE_EPOCH
            else f'{response_epoch}_vs_{reference_epoch}')
    if pair_within_trial:
        slug += '_within_trial'
    prefix = 'coprimary' if is_confirmatory else 'descriptive'
    if is_confirmatory:
        status_line = 'CONFIRMATORY: '
        multiplicity_note = ("This omnibus p enters the three-member confirmatory Holm family; "
                             "see confirmatory_holm_correction.txt for the adjusted value, and "
                             "report THAT.")
        title = f'Confirmatory {response_epoch} contrast, multiplicative scale'
    else:
        status_line = 'DESCRIPTIVE: '
        multiplicity_note = ("DESCRIPTIVE -- this omnibus p spends no alpha and enters NEITHER "
                             "multiplicity family (not the confirmatory Holm family, not the "
                             "secondary BH-FDR family). Report the estimate and its 95% "
                             "interval, not a significance verdict.")
        title = (f'Descriptive {response_epoch} vs {reference_epoch} contrast, '
                 f'multiplicative scale')
    delta_df = compute_epoch_delta_table(df_fine, response_epoch, reference_epoch,
                                         pair_within_trial=pair_within_trial)
    fit = fit_epoch_delta_model(delta_df)
    pairing_note = (' PAIRED WITHIN TRIAL (each cell differenced against its own trial\'s '
                    'reference window, then averaged over trials)' if pair_within_trial
                    else ', pooled over trials')
    write_text(os.path.join(stats_dir, f'{prefix}_epoch_delta_{slug}.txt'),
              f"{status_line}delta_log_amplitude ~ group, within-cell "
              f"({response_epoch} - {reference_epoch}){pairing_note}, one row per cell active in "
              f"BOTH epochs\n"
              f"Cell selection: {fit['n_cells']} of {n_trace_active_cells} trace-active cells "
              f"also had >=1 event in both {response_epoch} and {reference_epoch}"
              f"{' ON THE SAME TRIAL' if pair_within_trial else ''} and so "
              f"qualify for this contrast.\n"
              f"Omnibus (joint Wald, both non-reference groups, df2=n_mice-1): "
              f"{fit['omnibus']}\n\n"
              f"{multiplicity_note}\n\n"
              f"{fit['summary_text']}")
    plot_effect_forest(fit, save_dir, f'{prefix}_effect_forest_{slug}',
                       xlabel=f'Fold-change in within-cell {response_epoch}-vs-{reference_epoch} '
                              f'amplitude elevation',
                       title=title)
    return fit


def plot_decomposition(df_trace_pooled_raw, save_dir, filename_root='decomposition'):
    """
    Panel 4 ("components of population calcium activity" -- plan section 6 renamed this from
    "activity decomposition" and added the missing overall-rate panel so the identity
    fraction_active x rate_active = overall_rate is explicit rather than left for the reader to
    multiply panels themselves): fraction active (mouse-level only, a proportion computed OVER
    cells), rate among active cells, overall event rate across ALL cells (their exact product),
    mean per-event amplitude, and total S/s (= overall_rate x amplitude), left to right following
    the decomposition chain.

    No in-figure title is drawn -- the panel y-labels already name each quantity and this figure
    is always presented with its own caption stating the decomposition identity, so a redundant
    suptitle was dropped as visual clutter (this also narrows the effective margin available for
    the "narrow panel" look below).

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
        (active_only, 'rate_active', 'Event rate (event/s)', True, 'auto',
         rate_quantum, 'linear', '/s', 1.0),
        (df, 'overall_rate', 'Event rate (events/s)', True, 'auto',
         rate_quantum, 'linear', '/s', 1.9),
        (amp_df, 'log_amplitude', 'Log of mean per-event amplitude', True, 'linear',
         None, 'log', '', 1.0),
        (df, 'total_per_s', 'Deconvolved amplitude rate (a.u./s)', True, 'auto', None, 'linear',
         'S/s', 1.0),
    ]

    # Narrow per-panel footprint (figure width per unit width_ratio) + a tighter cell-cloud
    # jitter -- a "sexy nature paper" column reads as a slim strip of points, not a wide scatter
    # block, so both the panel's horizontal real estate and the cloud's own spread are pulled in
    # together. No figure-level title (see the "Components..." caption removal below): each panel
    # already carries its own title/y-label stating what it is, and this figure is always
    # presented with its own caption, so a redundant in-figure title was purely visual clutter.
    total_width_units = sum(p[-1] for p in panels)
    fig, axs = plt.subplots(1, len(panels), figsize=(1.9 * total_width_units, 3.0),
                            gridspec_kw={'width_ratios': [p[-1] for p in panels]})
    # rate_active's y-axis label is precise (units, "active cells" qualifier) but too long for a
    # panel title at this narrow width; the title uses the shorter, title-cased phrasing instead.
    panel_titles = {
        'rate_active': 'Active cell event rate',
        'overall_rate': 'All neurons event rate',
        'log_amplitude': 'Per-event amplitude',
        'total_per_s': 'Total event amplitude-rate',
    }
    contrasts_by_panel = {}
    for ax, (sub, col, ylabel, cell_level, yscale, quantum, ci_scale, ci_unit, _w) in zip(axs, panels):
        if cell_level:
            _draw_cell_superplot_panel(ax, sub, col, ylabel, panel_name='plot_decomposition',
                                       title=panel_titles.get(col), yscale=yscale,
                                       y_quantum=quantum, annotate='stats',
                                       bracket_mode='axes', jitter_width=0.22)
        else:
            values_per_group = _mouse_values_per_group(sub, col, panel_name='plot_decomposition')
            _draw_mouse_violin_panel(ax, values_per_group, ylabel, title=ylabel,
                                     annotate='stats', bracket_mode='axes')
        # The estimates/intervals go to the companion markdown rather than onto the panel.
        contrasts_by_panel[ylabel] = _panel_contrasts(sub, col, ci_scale, ci_unit)

    fig.subplots_adjust(left=0.05, bottom=0.16, right=0.99, top=0.90, wspace=0.55)
    _save_panel(fig, save_dir, filename_root)
    write_decomposition_contrasts_markdown(contrasts_by_panel, save_dir,
                                           filename_root + '_contrasts.md')




# Candidate tick positions for a fold-change axis, in ratio units. Matplotlib's default
# LogFormatter labels a narrow log axis (these intervals live inside 0.8-1.25) with crowded
# minor-tick scientific notation -- '9 x 10^-1' overlapping '1.05 x 10^0' -- which is both
# unreadable and the wrong register for a fold-change. Ticks are chosen from this list instead
# and labelled as plain multipliers.
_RATIO_TICKS = (0.25, 0.5, 0.67, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 4.0)


def _set_ratio_xaxis(ax, lo, hi, max_ticks=5):
    """Label a log fold-change axis with plain multipliers ('0.8', '1', '1.25').

    Thinned to at most `max_ticks`. _RATIO_TICKS is dense near 1.0 so that a narrow row (all
    intervals inside 0.9-1.1) still gets several labels, but a WIDE row would then render every
    one of them and the labels collide into an unreadable smear -- which is what the first
    version of this figure did. Thinning keeps 1.0 (the reference line must always be labelled)
    and drops alternate neighbours outward from it.
    """
    ticks = [t for t in _RATIO_TICKS if lo <= t <= hi]
    if len(ticks) < 2:
        ticks = [t for t in _RATIO_TICKS if lo / 1.05 <= t <= hi * 1.05] or [1.0]
    while len(ticks) > max_ticks:
        # Keep the endpoints and 1.0; thin what is between them.
        keep = [t for i, t in enumerate(ticks)
                if t == 1.0 or i in (0, len(ticks) - 1) or i % 2 == 0]
        if len(keep) == len(ticks):
            keep = [t for i, t in enumerate(ticks) if t == 1.0 or i % 2 == 0]
        if len(keep) == len(ticks):
            break
        ticks = keep
    ax.set_xticks(ticks)
    ax.set_xticklabels([('1' if t == 1.0 else f'{t:g}') for t in ticks])
    ax.xaxis.set_minor_locator(plt.NullLocator())


def _decomposition_grid_frames(df_epoch_pooled_raw):
    """The four row-frames plot_decomposition_grid needs for one epoch, derived from that epoch's
    trial-pooled table BEFORE filter_amplitude_rows (zero-event cells are needed by three of the
    four). Same derivations plot_decomposition performs, kept in one place so a component cannot
    be defined differently on the grid than on the per-epoch panels."""
    df = df_epoch_pooled_raw.copy()
    df['overall_rate'] = df['n_events'] / df['exposure_seconds']
    df['active'] = df['n_events'] > 0
    return {
        'fraction_active': fraction_active_table(df),
        'active_only': df[df['active']].rename(columns={'overall_rate': 'rate_active'}),
        'all': df,
        'amplitude': filter_amplitude_rows(df),
    }


def plot_decomposition_grid(df_fine, save_dir, epochs=TFC_MATCHED_EPOCHS,
                            filename_root='decomposition_grid', interaction_q=None,
                            reduced_coverage_epochs=(TFC_POST_SHOCK_LATE_EPOCH,),
                            components=_DECOMPOSITION_COMPONENTS, include_exc_vs_inh=True,
                            epoch_labels=None):
    """
    The decomposition as EFFECT ESTIMATES, one row per component and one column per
    exposure-matched epoch: hM3D/Ctl and hM4D/Ctl with 95% intervals, and no significance stars
    anywhere on the figure.

    ** This answers a different question from plot_decomposition, which is why it is a separate
    figure rather than a restyling of it. ** Those per-epoch panels show the DISTRIBUTIONS and
    are the right figure for "what do these cells look like". This one shows only the contrasts,
    side by side across epochs, and answers "which component accounts for the change, and is that
    consistent across the session".

    ** Why no stars. ** The four rows are not four independent phenotypes -- they are one exact
    identity, overall_rate = fraction_active x rate_active, plus the amplitude term. Starring
    them separately invites a reader to count significant components as if each were fresh
    evidence, and starring them PER EPOCH invites the difference-of-significance fallacy across
    columns on top of that. Epoch specificity gets ONE number per row -- that component's joint
    group x epoch test -- annotated on the row, never one per cell of the grid.

    ** Three points per panel, not two. ** hM3D/Ctl, hM4D/Ctl, and the DIRECT hM3D-vs-hM4D
    contrast in grey. The third exists because a reader looking at a row where red sits above the
    reference line and blue below it will conclude the two DREADDs differ, and the two plotted
    vs-control intervals do not license that: they share mCherry as a reference, and the contrast
    between them has its own (generally wider) interval. Drawing it removes the temptation to
    infer it. See _panel_contrasts' include_exc_vs_inh.

    Two axis details that are not cosmetic:

    * Fraction active is a bounded proportion, so mouse_contrast_ci reports it as a DIFFERENCE,
      not a fold-change (scale='difference_only'). Its row therefore gets its own linear axis
      centred on 0.0 while the other three are fold-changes on a log axis centred on 1.0.
      Forcing one shared x-axis would either plot a difference against a ratio reference line or
      silently drop the row.
    * The fold-change rows use a LOG x-axis so that a halving and a doubling are the same visual
      distance from the reference line. On a linear axis a ratio interval is asymmetric by
      construction and hM4D's reductions would read as smaller than hM3D's increases.

    interaction_q : {component key: q-value} from the joint group x epoch tests
                    (fit_and_report_epoch_interaction), BH-adjusted within the secondary family.
                    Annotated on each row as that component's single answer to temporal
                    specificity. The BH-ADJUSTED value is what is drawn, matching the convention
                    that a member of a multiplicity family is reported at its corrected value --
                    which is also why this figure is drawn AFTER build_secondary_fdr_table rather
                    than beside the tests themselves. A missing key omits that row's annotation
                    rather than inventing one.
    reduced_coverage_epochs : columns whose window is not present on every trial (post_shock_late
                    is absent on truncated final trials), flagged in the column label so a reader
                    does not compare their n against the others' unknowingly.
    components    : which _DECOMPOSITION_COMPONENTS specs to draw, one row each. Defaults to all
                    four -- the internal grid. The paper lane passes the PAPER_COMPONENT_KEYS
                    subset; see that constant for why three rather than four.
    include_exc_vs_inh : draw the third, grey, direct hM3D-vs-hM4D point per panel. On by default
                    because the internal grid IS read across conditions, which is exactly when a
                    reader tries to infer that contrast by eye from the two vs-control ones (see
                    _panel_contrasts). Off for the paper figures: those 16 intervals are tier-3
                    exploratory output (docs/sp_rates_lmm.md section 5.2) in no multiplicity
                    family, and they are not this design's question.
    epoch_labels  : {epoch key: column title}. Defaults to the raw epoch keys, which is right for
                    the internal figure -- they are what every stats filename and every other
                    reference to a window is keyed on. The paper lane passes readable window
                    names, since "pre_tone_matched" tells a reader nothing and the 20 s matching
                    is the reason the columns are comparable at all.
    """
    epochs = tuple(epochs)
    epoch_labels = epoch_labels or {}
    interaction_q = interaction_q or {}
    rows = tuple(components)
    contrasts = {}   # (component key, epoch) -> _panel_contrasts payload
    for epoch in epochs:
        frames = _decomposition_grid_frames(aggregate_over_trials(df_fine, epoch))
        for spec in rows:
            contrasts[(spec.key, epoch)] = _panel_contrasts(
                frames[spec.frame], spec.col, spec.ci_scale, spec.ci_unit,
                include_exc_vs_inh=include_exc_vs_inh)

    groups = [g for g in DREADD_DISPLAY_ORDER if g != 'mCherry']
    # The direct DREADD-vs-DREADD contrast sits below the two vs-control ones, in grey: it is a
    # different KIND of comparison (no control involved), so it must not read as a third group.
    _EXC_VS_INH_COLOUR = '0.45'
    # sharex='row' is load-bearing, not tidiness: the figure exists to be read ACROSS epochs, and
    # per-panel autoscaled x-limits would render a large effect and a small one at the same
    # apparent distance from the reference line. One scale per row makes the columns comparable,
    # which is the only way the "flat across epochs" reading is honest.
    fig, axs = plt.subplots(len(rows), len(epochs), sharey=True, sharex='row',
                            figsize=(2.3 * len(epochs), 1.5 * len(rows) + 1.0))
    axs = np.atleast_2d(axs)

    for r, spec in enumerate(rows):
        is_ratio = spec.ci_scale != 'difference_only'
        row_lo, row_hi = np.inf, -np.inf
        # Row entries: the two vs-control contrasts, then the direct DREADD-vs-DREADD one.
        entries = [(GROUP_LABELS[g], GROUP_COLOURS[g], g, 'contrasts') for g in groups]
        if include_exc_vs_inh:
            entries.append((f'{GROUP_LABELS["hM3D"]}/{GROUP_LABELS["hM4D"]}',
                            _EXC_VS_INH_COLOUR, 'hM3D', 'exc_vs_inh'))
        for c, epoch in enumerate(epochs):
            ax = axs[r, c]
            payload = contrasts[(spec.key, epoch)]
            for i, (_lbl, colour, group, which) in enumerate(entries):
                cd = payload[which][group]
                # A ratio row falls back to the difference when mouse_contrast_ci could not form
                # a ratio (a group mean at or below zero) -- see its scale= docs.
                if is_ratio and cd['ratio'] is not None:
                    est, lo, hi = cd['ratio'], cd['ratio_lo'], cd['ratio_hi']
                else:
                    est, lo, hi = cd['diff'], cd['diff_lo'], cd['diff_hi']
                row_lo, row_hi = min(row_lo, lo), max(row_hi, hi)
                ax.errorbar(est, -i, xerr=[[est - lo], [hi - est]], fmt='o',
                            color=colour, capsize=2.5, markersize=5,
                            markeredgecolor='k', markeredgewidth=0.3, linewidth=1.2)
            ax.axvline(1.0 if is_ratio else 0.0, color='k', linewidth=0.8, linestyle='--')
            if is_ratio:
                ax.set_xscale('log')
            ax.set_yticks([-i for i in range(len(entries))])
            ax.set_yticklabels([e[0] for e in entries], size='x-small')
            ax.set_ylim(-len(entries) + 0.5, 0.5)
            ax.tick_params(axis='x', labelsize='xx-small')
            ax.spines[['right', 'top']].set_visible(False)
            if c == 0:
                # The row's ONE epoch-specificity number lives on the row's own label, so it
                # cannot be mistaken for a per-column (per-epoch) claim.
                q = interaction_q.get(spec.key)
                q_txt = '' if q is None else f'\ngroup x epoch q = {q:.3f}'
                ax.set_ylabel(f'{spec.label}{q_txt}', size='x-small')
            if r == 0:
                flag = '\n(reduced coverage)' if epoch in reduced_coverage_epochs else ''
                ax.set_title(f'{epoch_labels.get(epoch, epoch)}{flag}', size='x-small')
            ax.set_xlabel('ratio' if is_ratio else 'difference', size='xx-small')

        # One padded scale for the whole row, applied once the row's full extent is known.
        pad = 1.08 if is_ratio else 0.15 * max(row_hi - row_lo, 1e-9)
        if is_ratio:
            axs[r, 0].set_xlim(row_lo / pad, row_hi * pad)
            for ax in axs[r]:
                _set_ratio_xaxis(ax, row_lo / pad, row_hi * pad)
        else:
            axs[r, 0].set_xlim(row_lo - pad, row_hi + pad)

    title = ('Components of population calcium activity: effect estimates with 95% CI\n'
             'Per-row group x epoch q: BH-adjusted joint permutation test for that component')
    fig.suptitle(title, size='small')
    fig.subplots_adjust(left=0.17, right=0.98, top=0.82, bottom=0.08, hspace=0.80, wspace=0.18)
    _save_panel(fig, save_dir, filename_root)

    # Keyed on (component, epoch) explicitly rather than on a y-label string: plot_decomposition's
    # own dict is ylabel-keyed and two of its panels differ only by 'event/s' vs 'events/s', which
    # is a collision waiting to happen. This grid has four rows x four columns of the same four
    # labels, so the tuple key is required, not defensive.
    write_decomposition_contrasts_markdown(
        {f'{spec.label} — {epoch}': contrasts[(spec.key, epoch)]
         for spec in rows for epoch in epochs},
        save_dir, filename_root + '_contrasts.md', figure_has_stars=False)


# Display names for the paper figures' epoch columns. The internal output uses the raw epoch keys
# throughout (they are what the stats filenames and the code are keyed on); a figure going into a
# manuscript needs the window stated in the label, since "pre_tone_matched" means nothing to a
# reader and the 20 s matching is the reason the three columns are comparable at all.
_PAPER_EPOCH_LABELS = {
    TFC_MATCHED_REFERENCE_EPOCH: 'Pre-tone\n(20 s baseline)',
    TFC_TRACE_EPOCH: 'Trace',
    TFC_POST_SHOCK_EPOCH: 'Post-shock\n(20 s)',
}

# Per-row DISPLAY attributes for plot_paper_epoch_distributions -- deliberately NOT on
# _DecompComponent, which defines what a component IS (frame, column, contrast scale) and is
# shared with the joint epoch tests. Axis scale and quantization jitter are properties of how one
# figure draws a component, not of the component.
_PAPER_ROW_DISPLAY = {
    # yscale='linear' because log_amplitude is ALREADY logged -- 'auto' would see all-positive
    # values and apply a SECOND log (plotting rule 2 / correction #7).
    'amplitude': {'yscale': 'linear', 'quantum': False,
                  'ylabel': 'Log of mean per-event amplitude'},
    # A per-cell rate is a small integer count over a fixed window, so without sub-quantum jitter
    # the cloud collapses onto a few hard horizontal stripes -- real quantization, but it hides
    # the density that is the reason for drawing cells at all.
    'population_rate': {'yscale': 'auto', 'quantum': True,
                        'ylabel': 'Event rate (events/s)'},
}


# One cell of a SuperPlot grid. `key` identifies it in error messages, `label` keys its entry in
# the companion contrasts file. The remaining fields are exactly what _draw_cell_superplot_panel
# and _panel_contrasts each need, so a panel carries its own display AND contrast configuration
# and the grid driver below needs to know nothing about what is being plotted.
_GridPanel = collections.namedtuple('_GridPanel', 'key label df col yscale quantum ci_scale ci_unit')


def _draw_superplot_panel_grid(panels, row_ylabels, col_titles, save_dir, filename_root,
                               panel_name, annotate='stats', figure_has_stars=True,
                               no_star_note=None, contrasts_title='Decomposition panel contrasts',
                               figsize_per_panel=(2.0, 3.0)):
    """Draw a rows x columns grid of cell-level SuperPlot panels sharing one y-scale per ROW,
    save it, and write the companion per-panel contrasts file.

    Factored out of plot_paper_epoch_distributions when a second figure wanted the same grid over
    a different pair of factors (epoch x conditioning phase rather than component x epoch). Per
    CLAUDE.md's dedup rule this drawing logic exists once; the callers differ only in which
    _GridPanel they put in each cell.

    ** sharey='row' is load-bearing, not tidiness. ** Every figure built on this helper exists to
    be read ACROSS its columns. Independently autoscaled panels would let a reader take a
    difference straight off the axis limits, which is precisely the comparison these figures are
    meant to make honestly.

    ** The per-row headroom reservation must happen once, after the whole row is drawn. ** Each
    panel's annotate_pairwise_brackets call reserves the top 20% of its axes for brackets, and
    under sharey those reservations COMPOUND: three panels leave 0.8^3 ~ 51% of the row to the
    data and the clouds end up squashed into the bottom half against a band of empty axis.
    Resetting each row to its own pooled data range and reserving once fixes it. Brackets are
    positioned in axes fractions, so they follow the new limits rather than being orphaned.

    panels    : {(row, col): _GridPanel}. Every cell of the grid must be present.
    annotate  : 'stats' for Holm-corrected vs-control brackets, 'none' for a bare distribution
                figure whose contrasts are read off a companion forest instead.
    figure_has_stars : passed to write_decomposition_contrasts_markdown, which selects a
                DIFFERENT explanatory note depending on it -- the star note would be simply false
                next to a figure drawn with annotate='none'.

    Returns {panel label: _panel_contrasts payload} so a caller can reuse the exact payloads the
    panels were drawn from rather than recomputing them and risking a disagreement.
    """
    n_rows, n_cols = len(row_ylabels), len(col_titles)
    missing = [(r, c) for r in range(n_rows) for c in range(n_cols) if (r, c) not in panels]
    if missing:
        raise ValueError(f'{panel_name}: no _GridPanel for grid cell(s) {missing}.')

    fig, axs = plt.subplots(n_rows, n_cols, sharey='row',
                            figsize=(figsize_per_panel[0] * n_cols,
                                     figsize_per_panel[1] * n_rows))
    # np.atleast_2d turns a single-COLUMN grid into a (1, n_rows) row vector, which would silently
    # transpose the figure; reshape to the intended shape explicitly.
    axs = np.atleast_2d(axs).reshape(n_rows, n_cols)

    contrasts_by_panel = {}
    row_values = [[] for _ in range(n_rows)]
    for (r, c), panel in sorted(panels.items()):
        row_values[r].append(panel.df[panel.col].to_numpy(dtype=float))
        _draw_cell_superplot_panel(
            axs[r, c], panel.df, panel.col,
            ylabel=(row_ylabels[r] if c == 0 else ''),
            panel_name=f'{panel_name}[{panel.key}]',
            # Column titles go on the top row only, and are re-set below with a pad: the bracket
            # stack is drawn above the axes in axes-fraction coordinates and the title sits there
            # too, so a zero pad puts the topmost asterisks through the column label.
            title='', yscale=panel.yscale, y_quantum=panel.quantum,
            annotate=annotate, bracket_mode='axes', jitter_width=0.22)
        if r == 0:
            axs[r, c].set_title(col_titles[c], size='small', pad=26)
        contrasts_by_panel[panel.label] = _panel_contrasts(
            panel.df, panel.col, panel.ci_scale, panel.ci_unit)

    for r in range(n_rows):
        ax0 = axs[r, 0]
        vals = np.concatenate(row_values[r])
        vals = vals[np.isfinite(vals)]
        if ax0.get_yscale() == 'linear':
            lo, hi = float(vals.min()), float(vals.max())
            pad = 0.05 * max(hi - lo, 1e-9)
            ax0.set_ylim(lo - pad, hi + pad)
        else:
            # A log axis cannot show the exact zeros that zero-event cells contribute; a symlog
            # one can, and its bottom must stay at zero so those cells are not silently dropped.
            positive = vals[vals > 0]
            bottom = 0.0 if ax0.get_yscale() == 'symlog' else float(positive.min()) / 1.3
            ax0.set_ylim(bottom, float(positive.max()) * 1.3)
        # Only reserve bracket headroom when brackets were actually drawn -- annotate='none'
        # would otherwise leave a fifth of every panel empty for annotations that do not exist.
        if annotate == 'stats':
            reserve_top_fraction(ax0)

    # No suptitle: each row carries its own y-label and each column its own title, and these
    # figures are always presented with a caption. Same choice plot_decomposition made.
    fig.subplots_adjust(left=0.16, bottom=0.09, right=0.98, top=0.90, wspace=0.18, hspace=0.30)
    _save_panel(fig, save_dir, filename_root)
    write_decomposition_contrasts_markdown(contrasts_by_panel, save_dir,
                                           filename_root + '_contrasts.md',
                                           figure_has_stars=figure_has_stars,
                                           no_star_note=no_star_note, title=contrasts_title)
    return contrasts_by_panel


def plot_paper_epoch_distributions(df_matched, save_dir, epochs=TFC_MATCHED_PROFILE_EPOCHS,
                                   rows=('amplitude', 'population_rate'),
                                   filename_root='tfc_amplitude_rate_by_epoch'):
    """
    The paper-facing distribution figure: per-event amplitude (top) and population event rate
    (bottom) for each DREADD group, across the three exposure-matched TFC windows.

    ** This is a re-cut of output that already exists, not a new analysis. ** Every panel draws
    the same quantities plot_decomposition draws and computes its statistics through the same
    _draw_cell_superplot_panel path; what changes is the selection (two components, three epochs,
    one figure) and the fact that the columns are directly comparable. No test is run here that is
    not already run in the TFC_cond lane.

    ** sharey='row' is the point of the figure, not tidiness. ** The scientific claim these panels
    support is that the hM3D amplitude elevation is a GLOBAL shift rather than a trace-specific
    one (the joint group x epoch test is null for every component -- see
    fit_and_report_epoch_interaction). Three independently autoscaled columns would let a reader
    read an epoch difference straight off the axis limits, which is the opposite of what the data
    say. One y-scale per row makes "flat across epochs" an honest reading.

    ** The input must be exposure-matched. ** Pass the frame restricted by
    restrict_to_exposure_matched_trials -- not the raw df_fine. The rate row is duration-sensitive
    (P(active) = 1 - e^(-lambda*T) rises with T at a fixed underlying rate) and trial 1's trace
    window is 15 s rather than 20 s, so on unmatched trials part of the trace-vs-baseline
    difference in that row would be pure exposure. Amplitude is a per-event quantity and does not
    care, but the two rows must be drawn over the same cells to be read together.

    Brackets are Holm-corrected vs-control pairwise tests computed from the PER-MOUSE means only
    (draw_superplot_triplet never passes the cell cloud to stat_fn). Comparing stars BETWEEN
    columns is the difference-of-significance fallacy -- epoch specificity has exactly one test
    per component, reported on the companion forest figure and in paper_results_summary.md.
    """
    epochs = tuple(epochs)
    specs = [_DECOMPOSITION_COMPONENTS_BY_KEY[k] for k in rows]

    panels = {}
    for c, epoch in enumerate(epochs):
        frames = _decomposition_grid_frames(aggregate_over_trials(df_matched, epoch))
        for r, spec in enumerate(specs):
            display = _PAPER_ROW_DISPLAY[spec.key]
            sub = frames[spec.frame]
            # One count = one quantum of rate. exposure_seconds is the same matched window for
            # every cell within a trial, so the median is that window pooled over trials.
            quantum = (1.0 / float(np.median(sub['exposure_seconds']))
                       if display['quantum'] else None)
            panels[(r, c)] = _GridPanel(
                key=f'{spec.key}/{epoch}', label=f'{spec.label} — {epoch}', df=sub,
                col=spec.col, yscale=display['yscale'], quantum=quantum,
                ci_scale=spec.ci_scale, ci_unit=spec.ci_unit)

    # Returned so the Results summary can quote the SAME payloads the panels were drawn from,
    # rather than recomputing the contrasts and risking a figure and its own summary disagreeing.
    return _draw_superplot_panel_grid(
        panels,
        row_ylabels=[_PAPER_ROW_DISPLAY[spec.key]['ylabel'] for spec in specs],
        col_titles=[_PAPER_EPOCH_LABELS.get(e, e) for e in epochs],
        save_dir=save_dir, filename_root=filename_root,
        panel_name='plot_paper_epoch_distributions', annotate='stats', figure_has_stars=True)


# Column titles for the conditioning-phase figure's epoch rows. `pre_tone` is deliberately the
# 35 s baseline rather than the 20 s matched one: amplitude is a PER-EVENT quantity and so is
# duration-insensitive (see the exposure-matching note in sp_rates_lmm_methods.md), the longer
# window yields more events per cell and therefore a better-estimated per-cell mean, and it is the
# reference the locked confirmatory amplitude contrasts already use -- so these numbers sit on the
# same scale as the ones in docs/sp_rates_lmm.md section 5.
_PHASE_FIGURE_EPOCH_LABELS = {
    TFC_REFERENCE_EPOCH: 'Pre-tone baseline (35 s)\nlog(mean per-event amplitude)',
    TFC_TRACE_EPOCH: 'Trace interval\nlog(mean per-event amplitude)',
}

_PHASE_CONTRASTS_NOTE = (
    '**These intervals are not multiplicity-corrected, and the figure carries no significance '
    'stars by design.** This figure is DESCRIPTIVE: it spends no alpha and is in neither the '
    'confirmatory Holm family nor the secondary BH-FDR family. The formal test of whether the '
    'group effect changes across trials already exists and is `group_x_trial_interaction` in the '
    'secondary family — see `stats/secondary_fdr_family.csv` for its BH-adjusted q and '
    '`stats/group_trial_photobleaching.txt` for the model. Read the four intervals below, not a '
    'comparison of stars between columns.\n\n'
    '**Absolute amplitude falls from early to late trials in every group** (~31% by trial 5, '
    'consistent with photobleaching). That is an epoch-independent main effect of trial and it '
    'cancels in a group contrast, which is why the contrast — not the level — is what this '
    'figure is for. A drop in the raw values between the two columns is expected and means '
    'nothing about the manipulation.'
)


def plot_conditioning_phase_amplitude(df_fine, save_dir, epochs=(TFC_REFERENCE_EPOCH,
                                                                 TFC_TRACE_EPOCH),
                                      phases=CONDITIONING_PHASES,
                                      filename_root='conditioning_phase_amplitude'):
    """
    Per-event amplitude by group in EARLY (trials 1-2) versus LATE (trials 3-5) conditioning, for
    the pre-tone baseline and the trace interval.

    ** The question. ** The amplitude effect is known to be a global shift rather than a
    trace-specific one (the joint group x epoch tests are null for every component). That leaves
    a different question open, which no epoch contrast can answer: is the elevation TONIC --
    present from the first trial, a property of the drug being on board -- or does it DEVELOP as
    conditioning proceeds? Epochs are windows within a trial; this splits ACROSS trials instead,
    which is an orthogonal axis.

    ** DESCRIPTIVE. Spends no alpha, in neither multiplicity family. ** The formal version of this
    question is already asked and corrected: `group_x_trial_interaction` (fit_group_trial_model)
    is a member of the secondary BH-FDR family and tests whether the group effect changes across
    trials. This figure exists so that result can be SEEN rather than taken on trust; it is not a
    second test of it, and no significance stars are drawn (annotate='none') precisely because
    comparing stars between the two columns would be the difference-of-significance fallacy.

    ** Amplitude only, and that is not an oversight. ** The rate analog of this figure cannot be
    drawn honestly at this split. Event rate and fraction active are duration-sensitive, so they
    need exposure-matched trials -- and restrict_to_exposure_matched_trials drops trial 1, whose
    trace window is 15 s rather than 20 s. That would reduce the 'early' column to trial 2 alone,
    gutting exactly the half of the figure the question rests on. Amplitude is a per-event
    quantity, needs no exposure matching, and therefore keeps every trial.

    ** Read the CONTRAST, not the level. ** Absolute amplitude declines across trials in every
    group (~31% by trial 5, consistent with photobleaching). That is a main effect of trial: it
    moves both columns down together and cancels in a group contrast. The companion forest is
    where the answer is legible -- four intervals, and the question is simply whether a group's
    early and late intervals sit in the same place.

    Writes: the distribution grid, the contrast forest, the per-panel contrasts markdown, and the
    per-mouse trial coverage behind the split.
    """
    epochs, phases = tuple(epochs), tuple(phases)
    spec = _DECOMPOSITION_COMPONENTS_BY_KEY['amplitude']
    frames_by_phase, coverage = split_by_conditioning_phase(df_fine, phases)

    panels, label_by_cell = {}, {}
    for r, epoch in enumerate(epochs):
        for c, phase in enumerate(phases):
            sub = filter_amplitude_rows(
                aggregate_over_trials(frames_by_phase[phase.key], epoch))
            label = f'{epoch} — {phase.key}'
            label_by_cell[(epoch, phase.key)] = label
            panels[(r, c)] = _GridPanel(
                key=f'{epoch}/{phase.key}', label=label, df=sub, col=spec.col,
                # yscale='linear' because log_amplitude is ALREADY logged -- 'auto' would see
                # all-positive values and apply a second log (plotting rule 2 / correction #7).
                yscale='linear', quantum=None,
                ci_scale=spec.ci_scale, ci_unit=spec.ci_unit)

    contrasts = _draw_superplot_panel_grid(
        panels,
        row_ylabels=[_PHASE_FIGURE_EPOCH_LABELS.get(e, e) for e in epochs],
        col_titles=[phase.label for phase in phases],
        save_dir=save_dir, filename_root=filename_root,
        panel_name='plot_conditioning_phase_amplitude',
        annotate='none', figure_has_stars=False, no_star_note=_PHASE_CONTRASTS_NOTE,
        contrasts_title='Per-event amplitude by conditioning phase — panel contrasts')

    _draw_phase_contrast_forest({cell: contrasts[label] for cell, label in label_by_cell.items()},
                                epochs, phases, save_dir, filename_root + '_forest')
    write_text(os.path.join(save_dir, 'stats', 'conditioning_phase_trial_coverage.csv'),
               coverage.to_csv(index=False))
    return contrasts, coverage


def _draw_phase_contrast_forest(contrasts_by_cell, epochs, phases, save_dir, filename_root):
    """The companion to plot_conditioning_phase_amplitude: each DREADD group's amplitude ratio
    against control, computed separately in early and late conditioning, one panel per epoch.

    ** This is where the figure is actually read. ** Four clouds of cells cannot be compared by
    eye across two columns; four intervals can. If a group's early and late points sit at the
    same place, the effect is tonic -- and because the photobleaching decline is a main effect of
    trial, it has already cancelled here, which it has not in the distribution panels.

    Open markers are early trials, filled are late, so the pairing is legible without reading the
    labels. sharex=True: the two epochs are meant to be compared, and per-panel autoscaling would
    render the same ratio at two different distances from the reference line.
    """
    groups = [g for g in DREADD_DISPLAY_ORDER if g != 'mCherry']
    entries = [(g, phase) for g in groups for phase in phases]

    fig, axs = plt.subplots(1, len(epochs), sharex=True,
                            figsize=(3.1 * len(epochs), 0.42 * len(entries) + 1.6))
    axs = np.atleast_1d(axs)
    lo_all, hi_all = np.inf, -np.inf
    for c, epoch in enumerate(epochs):
        ax = axs[c]
        for i, (group, phase) in enumerate(entries):
            cd = contrasts_by_cell[(epoch, phase.key)]['contrasts'][group]
            # A ratio row falls back to the difference where mouse_contrast_ci could not form a
            # ratio (a group mean at or below zero) -- see its scale= docs.
            if cd['ratio'] is None:
                raise RuntimeError(
                    f'_draw_phase_contrast_forest: no ratio for {group}/{epoch}/{phase.key}; '
                    f'log-amplitude means should be strictly positive.')
            est, lo, hi = cd['ratio'], cd['ratio_lo'], cd['ratio_hi']
            lo_all, hi_all = min(lo_all, lo), max(hi_all, hi)
            ax.errorbar(est, -i, xerr=[[est - lo], [hi - est]], fmt='o',
                        color=GROUP_COLOURS[group], capsize=2.5, markersize=6,
                        markerfacecolor=('white' if phase.key == phases[0].key
                                         else GROUP_COLOURS[group]),
                        markeredgecolor=GROUP_COLOURS[group], markeredgewidth=1.2, linewidth=1.2)
        ax.axvline(1.0, color='k', linewidth=0.8, linestyle='--')
        ax.set_xscale('log')
        ax.set_yticks([-i for i in range(len(entries))])
        ax.set_yticklabels([f'{GROUP_LABELS[g]} {p.key}' for g, p in entries], size='x-small')
        ax.set_ylim(-len(entries) + 0.5, 0.5)
        ax.set_xlabel('amplitude ratio vs Ctl', size='xx-small')
        ax.tick_params(axis='x', labelsize='xx-small')
        ax.spines[['right', 'top']].set_visible(False)
        ax.set_title(_PHASE_FIGURE_EPOCH_LABELS.get(epoch, epoch).split('\n')[0], size='x-small')

    pad = 1.08
    axs[0].set_xlim(lo_all / pad, hi_all * pad)
    for ax in axs:
        _set_ratio_xaxis(ax, lo_all / pad, hi_all * pad)
    fig.suptitle('Per-event amplitude vs control: tonic or conditioning-induced?\n'
                 'Open, trials 1-2; filled, trials 3-5. Descriptive — no alpha spent.',
                 size='small')
    fig.subplots_adjust(left=0.17, right=0.98, top=0.72, bottom=0.20, wspace=0.25)
    _save_panel(fig, save_dir, filename_root)


PAPER_METHODS_FILENAME = 'sp_rates_lmm_paper_methods.md'

# Output that stays in the TFC_cond lane. Named explicitly in the summary file so the reduction to
# a paper-sized figure set reads as a decision on record rather than as things having gone
# missing -- every one of these is computed, kept, and available if a reviewer asks.
_PAPER_SUPPLEMENT_ITEMS = (
    'Event rate among ACTIVE cells, and total deconvolved amplitude-rate (a.u./s) -- the other '
    'two terms of the decomposition identity (`decomposition*.png`).',
    'The direct hM3D-vs-hM4D contrasts -- exploratory, uncorrected, in no multiplicity family '
    '(docs/sp_rates_lmm.md section 5.2). Describe the two groups as showing divergent profiles; '
    'do not report either as differing from control on this basis.',
    'Amplitude ECDF and per-mouse 90th percentile -- where in the distribution the effect sits '
    '(`amplitude_ecdf.png`, `amplitude_p90.png`).',
    'Event-detection threshold sensitivity at thres in {1.5, 2.0, 3.0} (`threshold_sensitivity.png`).',
    'Run-structure evidence: run width, local maxima per run, multi-peak fraction (`run_structure.png`).',
    'group x trial photobleaching control (`stats/group_trial_photobleaching.txt`).',
    'Early-vs-late post-shock within-cell contrast (Puhger et al. 2024 internal control) '
    '-- descriptive, spends no alpha.',
    'Cross-registration subset sensitivity, and the LT1->LT2 detection-dropout measurement.',
    'Recall sessions: Test_B (48 h) and Test_B_1wk post-tone amplitude.',
)


def write_paper_results_summary(save_dir, primary_contrasts, holm, perm_results,
                                interactions, q_by_name, panel_contrasts, rate_contrasts,
                                manip_contrasts, filename='paper_results_summary.md'):
    """
    Every number the Results paragraph needs, in one file.

    ** Nothing here is computed for the first time. ** Each block reformats a result the TFC_cond
    lane already produced; this file exists so that writing the manuscript does not mean
    reassembling six numbers out of twenty stats files in three evidential tiers, and so that the
    tier of each number travels WITH it. That last part is the point: the recurring failure mode
    this module has documented (docs/sp_rates_lmm.md section 5.2) is a tier-3 estimate being
    written up as "significant" once it has been separated from the file that said otherwise.

    Every null is reported with its interval and what that interval still admits. At n = 5/6/6 a
    non-significant result is weak evidence of absence, and an interval reaching 1.55 has not
    excluded a +55% effect.
    """
    lines = [
        '# TFC cellular results — paper summary',
        '',
        'Per-event amplitude and event rate by DREADD group across the three exposure-matched '
        '20 s TFC windows (pre-tone baseline, trace, post-shock). n = 5 hM3D / 6 hM4D / '
        '6 mCherry animals; the mouse is the unit of inference throughout.',
        '',
        '**This is a re-cut of the `TFC_cond` output, not a separate analysis.** The full '
        'rationale, every sensitivity analysis, and the decision record live in '
        '`docs/sp_rates_lmm.md` and in the METHODS template beside this file.',
        '',
        '## Primary — trace-period per-event amplitude',
        '',
        f"Omnibus (mixed model, `log_amplitude ~ group + (1|mouse)`, cell-level, trials pooled): "
        f"p = {holm['trace_amplitude']['p_raw']:.4g}, "
        f"**Holm-corrected p = {holm['trace_amplitude']['p_holm']:.4g}** across the "
        f"{len(holm)}-member confirmatory family"
        f"{' (reject at alpha = 0.05)' if holm['trace_amplitude']['reject'] else ''}.",
        '',
        'Equal-mouse-weighted contrasts vs mCherry:',
        '',
    ]
    lines += [f'- {ln}' if not ln.startswith('  ') else f'  - {ln.strip()}'
              for ln in format_contrast_ci_lines(primary_contrasts, 'mCherry')]
    # The permutation p is the one to quote: cluster-robust SEs are anti-conservative at 17
    # clusters (docs/sp_rates_lmm.md section 4.3).
    perm_lines = [f"- {k.replace('_', ' ')}: p = {v['p_two_sided']:.4g} ({v['n_perm']} draws)"
                  for k, v in perm_results.items() if k.endswith('_mean_mouseweighted')]
    if perm_lines:
        lines += ['', 'Mouse-label permutation test on the same contrast (prefer this p to the '
                      'model p — cluster-robust standard errors are anti-conservative at 17 '
                      'clusters):', ''] + perm_lines

    lines += [
        '',
        '## Epoch analysis — does the group effect differ across windows?',
        '',
        'One joint `group x epoch` test per component (mouse-label permutation over the 17 '
        'animals, each keeping its whole profile across the three matched windows), BH-corrected '
        'within the secondary family. This is the single answer to temporal specificity: '
        'comparing per-epoch p-values against each other is the difference-of-significance '
        'fallacy and is not a test of anything.',
        '',
    ]
    for key in PAPER_COMPONENT_KEYS:
        if key not in interactions:
            continue
        res = interactions[key]
        q = q_by_name.get(f'epoch_specificity_{key}')
        q_txt = '' if q is None else f', q = {q:.3f}'
        lines.append(f"- **{_DECOMPOSITION_COMPONENTS_BY_KEY[key].label}**: "
                     f"p = {res['p_two_sided']:.3f}{q_txt} "
                     f"({res['n_cells']} cells, {res['n_mice']} mice)")
    lines += [
        '',
        'Reading: no component shows a group effect that changes across pre-tone, trace and '
        'post-shock. The hM3D amplitude elevation is a **global shift across the session**, not '
        'a trace-specific one. No figure, caption or sentence may imply a trace-specific effect; '
        "the trace interval's privileged status rests on prior anatomy and behaviour.",
        '',
        '## Per-epoch effect estimates (the numbers the figures draw)',
        '',
        'Equal-mouse-weighted contrasts vs mCherry with Welch 95% intervals, exposure-matched '
        'trials. Ratios and absolute differences together — a fold-change off a small base '
        'overstates the practical size of a change.',
        '',
    ]
    for panel_label, payload in panel_contrasts.items():
        lines.append(f'**{panel_label}**')
        lines.append('')
        lines += [f'- {ln}' if not ln.startswith('  ') else f'  - {ln.strip()}'
                  for ln in format_contrast_ci_lines(payload['contrasts'], 'mCherry',
                                                     unit=payload['unit'])]
        lines.append('')

    lines += [
        '## Secondary — event rate (negative-binomial mixed model)',
        '',
        'Counts at the mouse x trial x epoch level with a `log(total cell-seconds)` exposure '
        'offset, `(1|mouse) + (1|mouse:trial)`, dispersion estimated jointly. Posterior rate '
        'ratios vs mCherry at each window, with highest-density intervals. **Secondary and '
        'Bayesian**: there is no p-value here and none should be manufactured; report the '
        'estimate and its interval.',
        '',
        '| group | epoch | rate ratio vs Ctl | HDI |',
        '|---|---|---|---|',
    ]
    for _, row in rate_contrasts.iterrows():
        lines.append(f"| {GROUP_LABELS.get(row['group'], row['group'])} | {row['epoch']} | "
                     f"{row['rate_ratio']:.3f} | "
                     f"[{row['hdi_lo']:.3f}, {row['hdi_hi']:.3f}] ({row['hdi_prob']:.0%}) |")
    lines += ['', f"*{rate_contrasts['note'].iloc[0]}*", '']

    lines += [
        '## Manipulation check — LT1 (drug-free) to LT2 (CNO), within cell',
        '',
        'Independent confirmation that the DREADD does what it should, in a session pair that '
        'carries no memory hypothesis. Differenced within cell, then contrasted against mCherry '
        'so the shared order/time/photobleaching decline cancels.',
        '',
    ]
    lines += [f'- {ln}' if not ln.startswith('  ') else f'  - {ln.strip()}'
              for ln in format_contrast_ci_lines(manip_contrasts, 'mCherry')]

    lines += [
        '',
        '## Deliberately supplementary',
        '',
        'Computed, kept, and available — but out of the main figures by decision, not by '
        'omission. All of it is in the `TFC_cond/` output alongside its own stats files.',
        '',
    ]
    lines += [f'- {item}' for item in _PAPER_SUPPLEMENT_ITEMS]
    lines.append('')

    ensure_dirs(save_dir)
    write_text(os.path.join(save_dir, filename), '\n'.join(lines))


def render_paper_tfc_amplitude_rate(PLOTS_DIR, df_matched, primary_contrasts, holm,
                                    perm_results, interactions, q_by_name, rate_fit, delta_df,
                                    epochs=TFC_MATCHED_PROFILE_EPOCHS):
    """
    The paper lane: two figures and one Results-ready numbers file, under
    PLOTS_DIR/sp_rates_lmm/paper/tfc_amplitude_rate/.

    ** Re-cut, not re-analysis. ** Every input here is an object the TFC_cond lane already
    produced. Nothing is re-fit, no event detection is re-run, no test is added, and neither
    multiplicity family changes size. The TFC_cond output is untouched and remains the internal
    record; this folder is what a manuscript figure set looks like.

    ** Must be called after build_secondary_fdr_table. ** The forest annotates each row with that
    component's BH-ADJUSTED q, which does not exist until the whole secondary family has been
    fit — the same ordering constraint that already puts the internal decomposition grid last in
    run_sp_rates_lmm.

    df_matched : df_fine restricted to exposure-matched (mouse, trial) pairs. The rate row is
                 duration-sensitive and trial 1's trace window is 15 s rather than 20 s, so the
                 columns are only comparable on matched trials.
    """
    paper_dir = os.path.join(PLOTS_DIR, 'sp_rates_lmm', 'paper', 'tfc_amplitude_rate')
    paper_stats_dir = os.path.join(paper_dir, 'stats')
    ensure_dirs(paper_dir, paper_stats_dir)
    # Both templates land here: the short paper-facing one, and (via _save_panel) the full
    # internal METHODS. The paper text is what goes in the manuscript; the internal one is what
    # answers a reviewer who asks why a window is 20 s.
    _copy_analysis_methods_template(PAPER_METHODS_FILENAME, paper_dir)

    print('[sp_rates_lmm] Paper figures: amplitude + rate across the matched TFC windows...')
    panel_contrasts = plot_paper_epoch_distributions(df_matched, paper_dir, epochs=epochs)

    # The same grid the internal lane draws, restricted to the three paper components and with
    # the exploratory DREADD-vs-DREADD point off. Not a second implementation -- see
    # plot_decomposition_grid's components/include_exc_vs_inh arguments.
    plot_decomposition_grid(
        df_matched, paper_dir, epochs=epochs,
        components=[_DECOMPOSITION_COMPONENTS_BY_KEY[k] for k in PAPER_COMPONENT_KEYS],
        include_exc_vs_inh=False, filename_root='tfc_decomposition_forest',
        interaction_q={k: q_by_name[f'epoch_specificity_{k}'] for k in PAPER_COMPONENT_KEYS
                       if f'epoch_specificity_{k}' in q_by_name},
        # Every column here is one of the three COMPLETE 20 s windows; post_shock_late, the one
        # window with reduced trial coverage, is not among them.
        reduced_coverage_epochs=(), epoch_labels=_PAPER_EPOCH_LABELS)

    rate_contrasts = summarize_rate_group_epoch_contrasts(rate_fit)
    write_text(os.path.join(paper_stats_dir, 'rate_group_epoch_contrasts.csv'),
               rate_contrasts.to_csv(index=False))

    manip_contrasts = mouse_contrast_ci(
        _mouse_values_per_group(delta_df, 'delta_log_amplitude',
                                panel_name='paper_manipulation_check', group_order=GROUP_ORDER),
        scale='log')

    write_paper_results_summary(paper_stats_dir, primary_contrasts, holm, perm_results,
                                interactions, q_by_name, panel_contrasts, rate_contrasts,
                                manip_contrasts)
    print(f'[sp_rates_lmm] Paper figures written to {paper_dir}')


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
    write_figure_guide(out_dir)

    print('[sp_rates_lmm] Building TFC_cond event table + run-structure table (single pass)...')
    # Both matched windows passed explicitly rather than left to the defaults: the 20 s
    # post_shock response window and the 20 s matched baseline are analysis decisions of this
    # module (see TFC_POST_SHOCK_EPOCH / TFC_MATCHED_REFERENCE_EPOCH), so they are stated here
    # where the locked plan can be read off the call rather than inherited silently.
    tfc_frames_fn = functools.partial(get_epoch_frames, pre_tone_duration_s=35.0,
                                      pre_tone_matched_duration_s=TRACE_MATCHED_WINDOW_S,
                                      post_shock_duration_s=TRACE_MATCHED_WINDOW_S,
                                      post_shock_late_onset_s=POST_SHOCK_LATE_ONSET_S)
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

    # ---- Co-primary: within-cell response-vs-pre_tone deltas (plan section 2) ----------------
    # Two response windows, ONE code path (fit_and_report_epoch_delta): the trace interval and
    # the post-shock window are co-equal confirmatory members, so they must not be able to drift
    # apart in cell selection, model, or reporting.
    coprimary_fits = {}
    for response_epoch in TFC_CONFIRMATORY_RESPONSE_EPOCHS:
        print(f'[sp_rates_lmm] Fitting within-cell {response_epoch} epoch-delta model...')
        coprimary_fits[response_epoch] = fit_and_report_epoch_delta(
            df_fine, response_epoch, stats_dir, out_dir, n_trace_active_cells=len(df_trace_amp))

    # ---- Descriptive: early-vs-late post-shock, WITHIN trial ---------------------------------
    # Puhger et al.'s internal control, through the same code path as the confirmatory deltas but
    # explicitly is_confirmatory=False: the confirmatory family stays at exactly three tests and
    # this one spends no alpha. Its reference is the late post-shock window rather than pre_tone,
    # so unlike post_shock_vs_baseline it contrasts two windows at ALIGNED trial indices with no
    # shock-naive window mixed into the reference (see TFC_POST_SHOCK_LATE_EPOCH).
    # post_shock_late does not exist on a final trial whose recording stops shortly after the
    # shock, so both sides are cut to the trials where both windows exist before differencing.
    print('[sp_rates_lmm] Fitting descriptive within-cell early-vs-late post-shock delta...')
    df_early_late, late_coverage = restrict_to_shared_trials(
        df_fine, TFC_POST_SHOCK_EPOCH, TFC_POST_SHOCK_LATE_EPOCH)
    print(f'[sp_rates_lmm]   trial-matched coverage (of {len(late_coverage)} mice): '
          f'{late_coverage["n_trials_matched"].min()}-{late_coverage["n_trials_matched"].max()} '
          f'of {late_coverage["n_trials_total"].max()} trials per mouse')
    write_text(os.path.join(stats_dir, 'descriptive_early_vs_late_trial_coverage.csv'),
               late_coverage.to_csv(index=False))
    fit_and_report_epoch_delta(
        df_early_late, TFC_POST_SHOCK_EPOCH, stats_dir, out_dir,
        n_trace_active_cells=len(df_trace_amp),
        reference_epoch=TFC_POST_SHOCK_LATE_EPOCH, is_confirmatory=False)

    plot_epoch_profile(filter_amplitude_rows(df_fine), out_dir)

    # ---- Descriptive: the SAME within-cell delta for every other epoch ------------------------
    # Context for the co-primary null, not a test -- see compute_all_epoch_deltas' docstring.
    print('[sp_rates_lmm] Within-cell epoch deltas for the remaining epochs (descriptive)...')
    delta_summary, _delta_fits = compute_all_epoch_deltas(df_fine)
    write_text(os.path.join(stats_dir, 'epoch_deltas_descriptive.csv'),
              delta_summary.to_csv(index=False))
    plot_epoch_delta_forest(delta_summary, out_dir)

    # ---- Holm correction across the confirmatory family (the ONLY multiplicity burden here) ---
    holm = holm_correct_confirmatory({
        'trace_amplitude': primary['omnibus']['p'],
        'trace_vs_baseline': coprimary_fits[TFC_TRACE_EPOCH]['omnibus']['p'],
        'post_shock_vs_baseline': coprimary_fits[TFC_POST_SHOCK_EPOCH]['omnibus']['p'],
    })
    write_text(os.path.join(stats_dir, 'confirmatory_holm_correction.txt'),
              f"Holm correction across the three confirmatory omnibus tests:\n{holm}\n")
    print(f"[sp_rates_lmm] Confirmatory (Holm-corrected): {holm}")

    # ---- Amplitude distribution (tail) ----------------------------------------------------------
    plot_amplitude_ecdf(df_trace_amp, out_dir)
    plot_amplitude_p90(df_trace_amp, out_dir)

    # ---- Decomposition (fraction active, rate|active, overall rate, amplitude, total S/s) ------
    # One figure per exposure-matched epoch. The three windows are all TRACE_MATCHED_WINDOW_S
    # long, which is what makes their panels comparable at all: fraction-active and event rate
    # both scale with window length at a fixed underlying rate, so a 35 s baseline next to a
    # 20 s response window would have shown an epoch difference that was pure exposure. The
    # unsuffixed 'decomposition' filename stays on the trace epoch so existing references to it
    # keep resolving to the same figure.
    #
    # ** These are descriptive, not a test of epoch-specificity. ** Reading "starred here, not
    # starred there" across two of these figures is the difference-of-significance fallacy; the
    # epoch-specificity claim is carried by the within-cell deltas above and by
    # plot_epoch_delta_forest, which compare epochs WITHIN a cell rather than comparing two
    # independently-fit figures by eye.
    for epoch in TFC_MATCHED_EPOCHS:
        filename_root = ('decomposition' if epoch == TFC_TRACE_EPOCH
                         else f'decomposition_{epoch}')
        plot_decomposition(aggregate_over_trials(df_fine, epoch), out_dir,
                           filename_root=filename_root)

    # ---- Joint group x epoch specificity test + the decomposition GRID ------------------------
    # The single answer to "is the amplitude effect epoch-specific". Everything above compares
    # epochs one at a time; nothing above tests whether the group effect CHANGES across them, and
    # reading the confirmatory trace and post-shock p-values against each other does not either
    # (that is the difference-of-significance fallacy). See make_group_epoch_interaction_stat.
    #
    # ADDITIVE: the confirmatory Holm family above is untouched and still has exactly three
    # members. This p enters the SECONDARY BH-FDR family.
    #
    # Exposure-matched trials only. Trial 1's trace window is 15 s rather than 20 s, so pooled
    # across trials the matched comparison is ~95 s of trace against ~100 s of pre_tone_matched,
    # and both duration-sensitive components (fraction active, rate|active) are biased by that.
    print('[sp_rates_lmm] Joint group x epoch specificity test (mouse-label permutation)...')
    df_matched, matched_coverage = restrict_to_exposure_matched_trials(
        df_fine, TFC_MATCHED_PROFILE_EPOCHS)
    print(f'[sp_rates_lmm]   exposure-matched coverage (of {len(matched_coverage)} mice): '
          f'{matched_coverage["n_trials_matched"].min()}-'
          f'{matched_coverage["n_trials_matched"].max()} of '
          f'{matched_coverage["n_trials_total"].max()} trials per mouse')
    write_text(os.path.join(stats_dir, 'matched_decomposition_trial_coverage.csv'),
               matched_coverage.to_csv(index=False))
    interactions = fit_and_report_epoch_interaction(
        df_matched, stats_dir, mice_per_group, matched_coverage,
        epochs=TFC_MATCHED_PROFILE_EPOCHS, n_perm=n_perm, seed=seed)
    for _key, _res in interactions.items():
        secondary_pvalues[f'epoch_specificity_{_key}'] = _res['p_two_sided']
        print(f"[sp_rates_lmm]   group x epoch ({_key}): p = {_res['p_two_sided']:.4f} "
              f"({_res['n_cells']} cells, {_res['n_mice']} mice)")

    # Descriptive: post_shock - pre_tone_matched paired WITHIN TRIAL. The confirmatory
    # post_shock_vs_baseline pools each epoch across trials before differencing, so its reference
    # mixes trial 1's shock-naive baseline with four post-shock ones at unaligned indices. This
    # is a second, independent route to the problem TFC_POST_SHOCK_LATE_EPOCH also addresses.
    fit_and_report_epoch_delta(
        df_matched, TFC_POST_SHOCK_EPOCH, stats_dir, out_dir,
        n_trace_active_cells=len(df_trace_amp),
        reference_epoch=TFC_MATCHED_REFERENCE_EPOCH, is_confirmatory=False,
        pair_within_trial=True)

    # The grid reads across epochs, so it uses the exposure-matched trials too -- but over all of
    # TFC_MATCHED_EPOCHS, restricted only on the three complete windows, so post_shock_late keeps
    # whatever trials it has and is flagged as reduced-coverage rather than dropped.
    #
    # Restricted on (mouse, trial) PAIRS, not on the trial index: recordings are ragged, so which
    # trials qualify differs per mouse and a global index filter would cut good trials from the
    # mice whose recordings ran long (the same reason restrict_to_shared_trials matches per pair).
    #
    # ** The grid itself is drawn near the END of this function, not here. ** It annotates each
    # row with that component's BH-ADJUSTED q, and the BH pass cannot run until every secondary
    # member has been fit. Drawing it here would mean either annotating raw p-values on a figure
    # whose own stats file says to report the corrected ones, or correcting twice.
    matched_pairs = set(zip(df_matched['mouse'], df_matched['trial']))
    df_grid = df_fine[[(m, t) in matched_pairs
                       for m, t in zip(df_fine['mouse'], df_fine['trial'])]]

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

    # Early (trials 1-2) vs late (trials 3-5) conditioning -- the DESCRIPTIVE companion to the
    # group x trial model just fit. It asks whether the amplitude effect is tonic or develops
    # with conditioning, an axis orthogonal to every epoch contrast above (epochs are windows
    # WITHIN a trial). No test is added: the formal version is trial_fit's interaction, already a
    # member of the secondary BH-FDR family. Drawn here rather than at the end of the run so it
    # sits beside the model it visualizes; it annotates no p-value of its own, so unlike the
    # decomposition grid it has no dependency on the BH pass.
    print('[sp_rates_lmm] Early vs late conditioning amplitude (descriptive)...')
    _phase_contrasts, phase_coverage = plot_conditioning_phase_amplitude(df_fine, out_dir)
    print(f'[sp_rates_lmm]   trials per mouse -- early: '
          f'{phase_coverage["n_trials_early"].min()}-{phase_coverage["n_trials_early"].max()}, '
          f'late: {phase_coverage["n_trials_late"].min()}-'
          f'{phase_coverage["n_trials_late"].max()}')

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
    run_perm_results, run_perm_pairwise = {}, {}
    for other_group in ('hM3D', 'hM4D'):
        for label, col in (('width', 'width_frames'), ('n_local_maxima', 'n_local_maxima'),
                           ('multi_peak_fraction', 'multi_peak')):
            stat_fn = make_contrast_stat(df_runs_trace_mp, col, other_group, 'mCherry', weight='mouse')
            key = f'{other_group}_vs_mCherry_{label}'
            # GLOBAL null (all 17 mice exchangeable) -- the family member, and the null every
            # previously reported run-structure p was computed under. Unchanged deliberately.
            run_perm_results[key] = mouse_label_permutation_test(
                stat_fn, mice_per_group, n_perm=n_perm, seed=seed)
            secondary_pvalues[f'run_structure_{key}'] = run_perm_results[key]['p_two_sided']
            # PAIRWISE null (only the two groups being contrasted are exchangeable). Reported as
            # a sensitivity line, NOT added to the BH family: it is the same hypothesis tested
            # under a different exchangeability assumption, and entering both would correct one
            # question twice while halving the family's power. See mouse_label_permutation_test's
            # restrict_to_groups for why the two can differ materially.
            run_perm_pairwise[key] = mouse_label_permutation_test(
                stat_fn, mice_per_group, n_perm=n_perm, seed=seed,
                restrict_to_groups=(other_group, 'mCherry'))
    run_perm_text = '\n'.join(
        f"{k}: observed={v['observed']:.4g}, p_two_sided={v['p_two_sided']:.4g}, "
        f"n_perm={v['n_perm']}  |  pairwise-null p={run_perm_pairwise[k]['p_two_sided']:.4g}"
        for k, v in run_perm_results.items())
    write_text(os.path.join(stats_dir, 'run_structure.txt'),
              f"Per-mouse run-structure summary (trace epoch):\n"
              f"{run_summary.groupby('group')[['mean_width_frames', 'mean_n_local_maxima', 'fraction_multi_peak']].agg(['mean', 'std'])}\n\n"
              f"Mouse-label permutation tests (equal-mouse-weighted mean contrast).\n"
              f"p_two_sided is under the GLOBAL null (all 17 mice exchangeable) and is what "
              f"enters the BH-FDR family.\n"
              f"'pairwise-null p' restricts exchangeability to the two groups being contrasted, "
              f"assuming nothing about the third. It is a SENSITIVITY value, in no multiplicity "
              f"family -- report the family member unless you are specifically claiming a "
              f"two-group difference. The two can differ materially when the excluded group is "
              f"the most variable one.\n{run_perm_text}\n")

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
              f"EXCLUDED by design: the three confirmatory omnibus tests (Holm-corrected in their "
              f"own family), the Bambi NB rate model (posterior/LOO, no p-value to correct), and "
              f"all descriptive/sensitivity output -- including the early-vs-late post-shock "
              f"delta, which spends no alpha in either family. See build_secondary_fdr_table's "
              f"docstring.\n\n"
              f"{fdr_table.to_string(index=False)}\n")
    print(f'[sp_rates_lmm] Secondary BH-FDR family: {len(fdr_table)} tests, '
         f'{int(fdr_table["reject"].sum())} significant at q<0.05.')

    # ---- The decomposition grid, drawn LAST so it can carry BH-adjusted q-values ---------------
    # Each row is annotated with its own component's group x epoch q, which is why this waits for
    # the FDR pass above rather than being drawn beside the tests that produced it.
    _q_by_name = dict(zip(fdr_table['name'], fdr_table['q_value']))
    plot_decomposition_grid(
        df_grid, out_dir,
        interaction_q={k: _q_by_name[f'epoch_specificity_{k}'] for k in interactions
                       if f'epoch_specificity_{k}' in _q_by_name})

    # ---- The paper lane -------------------------------------------------------------------------
    # A manuscript-sized re-cut of everything above: two figures and one Results-ready numbers
    # file under sp_rates_lmm/paper/. Nothing is re-fit and no test is added -- it reuses the
    # objects already in scope. It goes last for the same reason the grid does: the forest carries
    # BH-adjusted q-values, which do not exist until the secondary family is complete.
    render_paper_tfc_amplitude_rate(PLOTS_DIR, df_grid, primary_contrasts, holm,
                                    perm_results, interactions, _q_by_name, rate_fit, delta_df)

    print('[sp_rates_lmm] Done.')
    if auto_close:
        plt.close('all')
