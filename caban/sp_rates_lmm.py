"""
Pyramidal event-AMPLITUDE and event-RATE analysis of DREADD effects on trace fear conditioning.

** The analysis the manuscript reports is the UNIFIED PAPER-FACING one. ** Per-event amplitude
and population event rate are presented as parallel outcomes of one two-row figure, so they get
one statistical framework rather than two: for each outcome, one mouse-level value per epoch, the
same `log(metric) ~ group * epoch + (1|mouse)` model, the same joint Wald interaction test, the
same model-derived treatment-vs-control contrasts, and the same multiplicity structure: WITHIN
each epoch the two treatment-vs-control contrasts are Holm-corrected together (p_holm_epoch), the
same way in every epoch and both outcomes, and the conservative six-comparison across-epoch Holm
correction is retained as sensitivity output. Every
marker, interval, p-value and asterisk on a paper-facing figure is read out of that one contrast
table. Start at UNIFIED_OUTCOMES and
build_mouse_epoch_unified_table(); the entry point is render_paper_tfc_amplitude_rate().

This is the module's MAIN statistical analysis, finalised after substantial inspection of this
dataset -- it is not a prospective preregistration and is not described as confirmatory.

** Everything else here is sensitivity/internal output. ** The cell-level amplitude model and its
three-member Holm family, the Bayesian negative-binomial rate model, the mouse-label permutation
tests, the BH-FDR secondary family, threshold sensitivity, run structure, the group x trial
photobleaching control, and the direct hM3D-vs-hM4D contrasts all still run, still write every
file they always did, and are all valuable as robustness evidence. None of them supplies a number
to a manuscript sentence or a paper figure. docs/sp_rates_lmm.md is organised into exactly those
two parts, and so is paper_results_summary.md.

The whole thing replaces sp_rates' ~205 uncorrected three-group ANOVAs (session x epoch x
cross-registration subset x metric). Amplitude remains the primary BIOLOGICAL endpoint and rate
the secondary one -- the claim under test is that hM3D increases pyramidal BURSTING (larger
per-event Ca2+ influx), not merely more frequent events -- but both now receive identical
statistical treatment. See analysis_methods_templates/sp_rates_lmm_methods.md for the full
rationale behind every measurement choice.

A post-hoc methodological review of an earlier version of this module found three correctness
bugs (denominator df, a missing exposure factor, a mislabelled recall title) and two overstated
designs (the co-primary epoch model's pseudoreplication, cell- vs mouse-weighting of the primary
contrast); all are fixed here -- see the module CHANGELOG below.

Module layout
-------------
  Unified paper analysis      -- build_mouse_epoch_unified_table(), fit_unified_group_epoch_model(),
                                require_common_unified_method(), unified_posthoc_contrasts(),
                                unified_interactions_table(), unified_model_diagnostics(),
                                verify_unified_synthetic(), render_paper_tfc_amplitude_rate(),
                                render_paper_recall_amplitude_rate()
  Hierarchical cell companion -- build_recall_cell_amplitude_modulation(),
   (both recall sessions;       fit_hierarchical_cell_delta_model(),
    additive)                   hierarchical_cell_amplitude_permutation(),
                                fit_hierarchical_cell_unpaired_sensitivity(),
                                build_recall_cell_epoch_count_table(),
                                build_cell_epoch_rate_model(), fit_cell_epoch_rate_model(),
                                hierarchical_cell_rate_prior_predictive(),
                                hierarchical_cell_rate_posterior_predictive(),
                                summarize_cell_rate_contrasts(),
                                plot_hierarchical_cell_amplitude_modulation(),
                                plot_hierarchical_cell_rate_modulation(),
                                write_hierarchical_cell_vs_mouse_summary(),
                                run_hierarchical_cell_suite()
                                -- development tools, NOT part of a real-data run:
                                verify_hierarchical_cell_synthetic(),
                                verify_hierarchical_cell_rate_synthetic()
  Recall pre->post modulation -- build_recall_modulation_by_mouse(),
                                recall_modulation_contrasts(), recall_modulation_lookup(),
                                recall_within_group_lookup(), write_recall_modulation_summary(),
                                plot_recall_modulation(), plot_recall_prepost_trajectories(),
                                recall_trajectory_paired_tests(), recall_trajectory_lookup(),
                                verify_recall_modulation_synthetic()
  Event/run table construction -- _iter_event_windows() (shared traversal), build_epoch_event_table(),
                                  build_run_structure_table(), aggregate_over_trials(),
                                  filter_amplitude_rows(), build_mouse_trial_epoch_rate_table(),
                                  build_mouse_trial_trace_amplitude(), compute_epoch_delta_table()
  Sensitivity models (was     -- fit_primary_trace_amplitude(), fit_epoch_delta_model(),
  the confirmatory family)      fit_epoch_interaction_nested_attempt() (one-off, not in the main
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
  - THE TRAJECTORY FIGURE NOW CARRIES A WITHIN-GROUP PAIRED t-TEST (recall lane, both sessions;
    recall_trajectory_paired_tests, recall_trajectory_lookup, plot_recall_prepost_trajectories).
    ** This is the one place in this lane where a statistic IS computed on plotted values, and it
    was added deliberately. ** The per-animal trajectory panel was previously annotation-free on
    the grounds that a per-group paired comparison invites being read as the treatment effect.
    That risk is real and is now managed explicitly rather than by omission: each facet gets a
    two-tailed paired t of that group's own animals' pre/post values ON THE LOG SCALE the models
    were fit on (df = n_g - 1), the figure footer states in as many words that a bracket says
    whether THAT group changed and is NOT a comparison between groups, and the companion markdown
    repeats it. Stars below 0.05, the p-value itself printed for 0.05 <= P < 0.10 so a near-miss
    is stated rather than inferred from an absent bracket, nothing above that. Brackets read the
    RAW p; a three-group Holm family per outcome travels beside it as a multiplicity reference,
    the same convention the modulation figure already uses.
    ** It is a SECOND estimator of a quantity within_group_df already reports, and the two do not
    agree. ** The model contrast pools residual variance across the three groups on the session's
    animal-level df; the paired t uses one group's animals and its own df. Both are written --
    the new `<prefix>_trajectory_paired_tests.csv`, and adjacent sections B and B-panel of
    `<prefix>_modulation_contrasts.md` -- precisely so the divergence is on the record instead of
    surfacing as a figure that appears to contradict a table. ** NOTHING ELSE CHANGED: ** the
    between-group modulation contrasts, their omnibus, the modulation figure's brackets, the
    within-epoch simple effects, every Holm family and every other output are untouched, and no
    paper-facing number moved. The model contrast remains this lane's estimate of the within-group
    change; the paired t is the panel's own annotation and nothing else reads it.
  - THE TWO SESSION-SCOPED RECALL ADDITIONS NOW RUN ON BOTH RECALL SESSIONS. The pre->post
    modulation decomposition and the hierarchical cell-level companion were each introduced
    scoped to Test_B for the pass that specified them, leaving Test_B_1wk behind a lane that was
    otherwise already identical for both sessions -- including the recall METHODS file copied
    into Test_B_1wk's own output directory, which pointed readers at a modulation markdown that
    was never written there. RECALL_MODULATION_SESSIONS and RECALL_HIERARCHICAL_CELL_SESSIONS now
    both name ('Test_B', 'Test_B_1wk'). ** No model, estimator, contrast, interval, omnibus,
    epoch definition, matched-trial construction, figure grammar or output filename changed, and
    no Test_B number changed. ** The machinery was already session-agnostic: each session's
    cohort, group Ns, denominator df, randomization space, file stems and titles are derived from
    that session's own data or from RECALL_SESSION_LABELS/_RECALL_FILE_PREFIX, so extending the
    scope was opening the two gates and nothing else. ** The two recall sessions remain analysed
    SEPARATELY and are still never compared with each other ** -- a decomposition now existing at
    both timepoints licenses no 48 h versus 1 week sentence, for the cohort reason
    render_paper_recall_amplitude_rate states. RUNTIME: a routine pass gains only the (fast)
    modulation outputs for Test_B_1wk; run_hierarchical_cell_analysis=True now costs ~80 min
    rather than ~40, since the companion suite runs once per recall session.
  - THE HIERARCHICAL COMPANION IS NOW EXPLICITLY INVOKED, AND THE Test_B MODULATION FIGURE
    REPORTS UNADJUSTED MODEL-DERIVED CONTRASTS. Two changes, neither of them statistical:
    (1) RUNTIME. run_hierarchical_cell_analysis now defaults to FALSE, in run_sp_rates_lmm and in
    render_paper_recall_amplitude_rate. The companion suite cost ~40 min of a routine pass --
    exact mouse-label MixedLM enumerations over the paired-cell table, the hierarchical NB count
    model, and the prior/posterior-predictive machinery -- against a few minutes for everything
    else, and its evaluation is finished. Nothing was deleted: the suite, all eleven components
    and every output file it has already written are intact, and passing True reruns it exactly
    as before. Its paired-cell amplitude result is retained as SUPPLEMENTARY SENSITIVITY evidence
    (it reproduced the mouse-level effect magnitude); its NB rate model is not usable as
    sensitivity evidence at all, because it failed its own posterior-predictive adequacy check,
    and it is deliberately NOT redesigned or replaced here.
    (2) REPORTING. The Test_B modulation figure and the drafted wording in its companion markdown
    now read `p_raw` -- the unadjusted contrast p-values of the SAME mouse-level
    `log(metric) ~ group * epoch + (1|mouse)` fits, read out of the same contrast table, with no
    value hard-coded anywhere. hM3D-vs-mCherry carries a star, hM3D-vs-hM4D is bracketed with its
    p-value (annotate_pairwise_brackets gained an opt-in ns_label_pairs= whose default reproduces
    its previous behaviour exactly), and hM4D-vs-mCherry is computed and tabulated as always but
    not bracketed. The three-comparison Holm family is STILL COMPUTED and still written to the
    CSV and the markdown as a multiplicity reference -- the record is not erased and nothing raw
    is relabelled as adjusted; it simply no longer governs this figure. ** No model, estimator,
    contrast, interval, omnibus, event or epoch definition, matched-trial construction, TFC
    output or Test_B_1wk output changed. ** The group x epoch omnibus is still reported in each
    panel title and still does not gate the contrasts.
  - THE HIERARCHICAL CELL-LEVEL COMPANION ANALYSIS (recall lane, Test_B when introduced; both
    recall sessions as of the entry above;
    RECALL_HIERARCHICAL_CELL_SESSIONS, run_hierarchical_cell_suite). ** STRICTLY ADDITIVE:
    NOTHING PRE-EXISTING CHANGED. ** No existing statistical path, estimator, model, contrast,
    Holm family, omnibus test, output file, table or figure was replaced or altered -- in this
    lane or in the TFC one. What changed is orchestration only, and as of the entry above this
    suite is NO LONGER part of a routine pass: run_hierarchical_cell_analysis defaults to False
    and the suite is an explicitly invoked companion/sensitivity analysis.
    The paper-facing recall analysis collapses each animal to one scalar per epoch before fitting.
    That stays primary. This asks the complementary question -- does the pre->post modulation
    occur coherently across the cellular population WITHIN animals when the cell hierarchy is
    modelled rather than collapsed? -- and writes into its own subdirectory
    (paper/recall/<session>/hierarchical_cells/) so its numbers cannot be confused with the primary
    lane's. Amplitude: one PAIRED within-cell delta per eligible cell (log post - log pre, no
    imputation, no pseudocount), estimated by `delta ~ group + (1|mouse)` and tested by EXACT
    mouse-label randomization OF THAT MODEL'S COEFFICIENT -- the full three-group model refit
    under every one of the 462/462/252 restricted relabelings, Holm across those three exact
    p-values. The omnibus permutes a 2-df model-based statistic (Monte Carlo; the global space is
    2,018,016). A design-based permutation of the 16 mouse-mean deltas is retained as a labelled
    sensitivity in no family. Rate: a hierarchical NB count model over ALL cells including
    zero-event cells, with frozen priors, a prior-predictive check BEFORE fitting, and
    posterior-predictive zero/dispersion checks by group x epoch. ** Implementation validation
    against planted synthetic truth is NOT part of a real-data run: ** the two verify_*_synthetic
    functions are development tools, called by hand after changing this module, because an
    execution that produces the reported numbers should compute those numbers and nothing else.
    Both long-format models carry a
    mouse-level epoch random slope, without which between-animal variation in the pre->post change
    would land at the cell level and narrow the group x epoch terms.
    ** No asymptotic cell-level p-value is paper-facing anywhere in this suite ** -- 16 animals
    were randomized, not N cells, and every model summary says so in its own header. One gate
    (run_hierarchical_cell_analysis), all-or-nothing, no result-dependent branching, and no
    statistical fallback: a specified model that fails its predefined adequacy gate RAISES rather
    than being replaced (fit_mixed_model's clustered-OLS fallback is explicitly barred here).
    Output is staged and promoted only after every component succeeds, so a partial run cannot be
    mistaken for a complete one. mouse_label_permutation_test gained an opt-in `exact=` parameter
    whose default reproduces its previous behaviour exactly, single_unit_common.fit_mixed_model
    gained an opt-in `method=` optimizer parameter defaulting to its previous 'lbfgs', and the NB
    convergence-diagnostics block was extracted to _nb_convergence_diagnostics with a default that
    emits byte-identical text for the TFC lane. See docs/sp_rates_lmm.md section A.7.2.
  - THE PRE->POST MODULATION DECOMPOSITION (recall lane, Test_B when introduced; both recall
    sessions as of the entry above; RECALL_MODULATION_SESSIONS). The
    recall models already tested whether the pre-tone -> post-tone change differs among the groups
    -- that IS the group x epoch joint Wald test -- but nothing showed the change itself, and the
    individual pairwise comparisons the fitted model contains were never read out. Added:
    build_recall_modulation_by_mouse (one change score per animal per outcome, pivoted from the
    inferential table itself), recall_modulation_contrasts (each group's model-implied pre->post
    change, DESCRIPTIVE; and the three pairwise comparisons of that change, with a within-outcome
    Holm adjustment alongside the raw contrast p-values -- see the entry above for which of the
    two the figure reports), a two-panel per-animal modulation figure whose brackets are looked up
    from those contrasts, a pre/post trajectory figure (descriptive when introduced; it gained a
    within-group paired t-test as of the entry above), a companion markdown, and
    verify_recall_modulation_synthetic. ** No model is fit, and as introduced here no statistic was
    computed on the plotted change scores: ** every number is a linear contrast of the two fits the
    lane already made, on the same animal-level df, and each panel and table carries the omnibus it
    decomposes. That remains true of every BETWEEN-GROUP number in this block.
    All three groups and both outcomes are treated identically -- no group is a headline. Nothing
    pre-existing changed: the within-epoch simple effects, their Holm families, the omnibus
    definition and every existing figure and table are untouched.
  - RECALL REPORTING BOILERPLATE (display only, no statistic changed). The statsmodels-summary
    header/footer were module constants hard-coding the CONDITIONING lane's df = 16, its
    `unified_lmm_*` filenames, its p_holm_six column and its trace/post-shock epochs -- all wrong
    when emitted over a recall fit. They are now _statsmodels_pvalue_note/_footer, parameterized
    per lane and fed the df read off the fit; the conditioning text is byte-identical to before.
    format_unified_interaction also rendered a tiny p as the literally false `P = 0.000`; it now
    reports `P < 0.001` via format_p_display.
  - THE PAPER-FACING RECALL LANE (render_paper_recall_amplitude_rate). The unified analysis above
    is now also run on Test_B (48 h) and Test_B_1wk, each SEPARATELY, over two duration-matched
    20 s windows -- a pre-tone baseline and the post-tone retrieval window, the recall analogue of
    the conditioning trace interval. Same event definition, same animal-level summarization, same
    model, same within-epoch Holm family, same figure grammar; what differs is the epoch set and
    the cohort, and both are read off the data (each session's denominator df is its own
    n_animals_present - 1, not the conditioning model's 16). The group x epoch interaction is the
    2-df joint test of whether the treatment effect changes from pre-tone to post-tone, and is the
    ONLY test of retrieval preferentiality. The two sessions are never compared with each other:
    the animal missing at 48 h is not the one missing at 1 week. To support this the unified
    helpers took a `reference_epoch` parameter and derive their interaction count from the fitted
    model; the TFC lane's behaviour is unchanged. See docs/sp_rates_lmm.md section A.7. The older
    single-epoch Test_B/Test_B_1wk post-tone amplitude output is untouched and remains a Part B
    secondary.
  - THE UNIFIED PAPER-FACING ANALYSIS. The paper figure's two rows used to be supported by three
    unrelated frameworks: a cell-level frequentist LMM for amplitude, a Bayesian negative-binomial
    model with HDIs for rate, per-panel Welch/Holm tests for the asterisks, and equal-mouse-
    weighted Welch intervals for the effect sizes -- with epoch specificity coming from a fourth,
    a mouse-label permutation statistic. No number on the figure traced to a model, and the
    statistical treatment of the two rows could not be described in one sentence. They are now
    one procedure (build_mouse_epoch_unified_table -> fit_unified_group_epoch_model ->
    unified_posthoc_contrasts), and the figures read their markers, intervals, p-values and stars
    out of it via _precomputed_stat_fn / _unified_contrast_payloads. All previous machinery is
    retained as sensitivity output and supplies no paper number.
  - WITHIN-EPOCH POST-HOC FAMILIES (`p_holm_epoch`). The unified models, matched trials,
    estimates, raw p-values and interaction tests are unchanged; only the multiplicity/display
    structure moved. Within EACH epoch the two treatment-vs-control comparisons (hM3D vs mCherry,
    hM4D vs mCherry) are Holm-corrected together -- six identical two-member families, one per
    (outcome, epoch). The procedure is the same in all six panels of the paper figure; no epoch
    receives a different kind of inferential treatment, and epoch dependence is tested only by
    the group x epoch interaction. The conservative across-epoch six-comparison correction is
    kept in full as `p_holm_six` sensitivity output. (Earlier passes corrected all six simple
    effects together, and then treated the trace epoch's two contrasts as a privileged primary
    family; both are superseded -- see docs/sp_rates_lmm.md section 7.)
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
import time
import math
import shutil
import datetime
import functools
import itertools
import traceback
import collections

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import bambi as bmb
import arviz as az
import pymc as pm

from caban.utilities import find_event_runs_ca, find_event_runs_ca_S, MINISCOPE_FPS
from caban.single_unit_common import (
    GROUP_ORDER, GROUP_LABELS, GROUP_COLOURS, DREADD_DISPLAY_ORDER,
    ensure_dirs, write_text, save_fig, ecdf_panel, fdr_correct, holm_correct,
    fit_mixed_model, joint_wald_test, linear_contrast_test,
    draw_superplot_triplet, no_stat_annotation,
    mouse_contrast_ci, annotate_contrast_ci, format_contrast_ci_lines,
    annotate_pairwise_brackets, reserve_top_fraction,
)
from caban.decoder import _ANALYSIS_METHODS_TEMPLATES_DIR, _copy_analysis_methods_template
from caban.analysis import _draw_violin_triplet, do_pairwise_holm_plot
from caban.epoch_analysis import (get_epoch_frames, get_testb_epoch_frames,
                                  TRACE_MATCHED_WINDOW_S, POST_SHOCK_LATE_ONSET_S,
                                  _stars_from_p)

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
#
# ** 'fraction_active' was removed when the unified paper-facing models were introduced. ** Every
# number on a paper figure now comes from one of the two unified LMMs (see UNIFIED_OUTCOMES
# below), and fraction active has no such model -- drawing its row would put a second, differently
# derived inferential source on the same figure, which is exactly what the unification exists to
# remove. It is unchanged on the internal four-component decomposition_grid and is described in
# the paper summary as supplementary.
PAPER_COMPONENT_KEYS = ('population_rate', 'amplitude')


# ─────────────────────────────────────────────────────────────────────────────
# The unified paper-facing analysis: one mouse-level model per outcome
# ─────────────────────────────────────────────────────────────────────────────
#
# ** This is the analysis the manuscript reports. ** Per-event amplitude and population event rate
# are presented as parallel outcomes on one two-row figure, so they get ONE statistical framework:
# the same mouse-level response construction, the same `group * epoch + (1|mouse)` model, the same
# joint Wald interaction test, the same model-derived contrasts, and the same multiplicity
# structure (see unified_posthoc_contrasts). Every marker, interval, p-value and asterisk on a paper-facing panel is read out of
# unified_lmm_mouse_epoch_values.csv / unified_lmm_posthoc_contrasts.csv and nowhere else.
#
# Everything else this module computes -- the cell-level amplitude model, the Bayesian NB rate
# model, the mouse-label permutation tests, the BH-FDR family, threshold sensitivity, run
# structure -- is retained in full as SENSITIVITY/internal output and supplies no paper number.
# See docs/sp_rates_lmm.md, which is organised into exactly those two parts.
UNIFIED_REFERENCE_GROUP = 'mCherry'
UNIFIED_REFERENCE_EPOCH = TFC_MATCHED_REFERENCE_EPOCH
UNIFIED_TREATMENT_GROUPS = ('hM3D', 'hM4D')

# ** The multiplicity family is WITHIN an epoch, and is the same in every epoch. **
# For each (outcome, epoch) the two treatment-vs-control simple effects -- hM3D vs mCherry and
# hM4D vs mCherry -- are Holm-corrected together (p_holm_epoch). That gives six two-member
# families: 3 epochs x 2 outcomes. The procedure is identical in all six panels of the paper
# figure, so visually equivalent panels are treated equivalently and no epoch receives a
# different kind of inferential treatment. Whether the treatment effect DIFFERS across epochs is
# a separate question, tested directly and only by the group x epoch interaction.
#
# The conservative alternative -- Holm across all six treatment-vs-control simple effects
# spanning the three epochs within an outcome -- is retained in full as p_holm_six and reported
# as a sensitivity analysis. It generates no figure asterisk and no manuscript claim.

# outcome key -> (model response column, natural-scale display column on the mouse-level table).
# The response is what the LMM is fit on; the display column is what the figure's large mouse
# markers show, and the two are related by exp() in both cases (see _UNIFIED_MARKER_TRANSFORM).
_UnifiedOutcome = collections.namedtuple(
    '_UnifiedOutcome', 'key label response_col marker_col component_key ratio_label')

UNIFIED_OUTCOMES = (
    _UnifiedOutcome('amplitude', 'Per-event amplitude', 'mouse_mean_log_amplitude',
                    'geometric_mean_amplitude', 'amplitude', 'fold change'),
    _UnifiedOutcome('population_rate', 'Population event rate', 'log_population_rate',
                    'population_rate', 'population_rate', 'rate ratio'),
)
UNIFIED_OUTCOMES_BY_KEY = {o.key: o for o in UNIFIED_OUTCOMES}

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

# ── The paper-facing RECALL windows ──────────────────────────────────────────
#
# The recall question is whether the conditioning-day phenotype is still present during drug-free
# retrieval, and whether any group difference is EVOKED by the tone/retrieval period rather than
# already present at baseline. That needs two duration-matched windows around each tone and
# nothing else:
#
#   pre_tone  : the 20 s ending at tone onset -- the within-session baseline the interaction is
#               tested against.
#   post_tone : the 20 s beginning at tone offset -- the retrieval analogue of the conditioning
#               TRACE interval (the tone has ended, and CA1 activity is measured over the
#               following 20 s with no shock).
#
# The TONE epoch itself is deliberately excluded: it is not the trace analogue, and adding it
# would change the interaction from a 2-df test of "does the treatment effect change from
# pre-tone to post-tone" into something else.
#
# Both are pinned to 20 s for the same reason TESTB_POST_TONE_DURATION_S is (see its comment):
# neither get_testb_epoch_frames' own 35 s default nor TestBSession.post_tone_offsets (which runs
# to the NEXT tone onset, ~200+ s) is the recall analogue of the true trace duration, and the two
# epochs must be duration-matched because population event rate is duration-sensitive.
TESTB_PRE_TONE_DURATION_S = 20.0
RECALL_WINDOW_S = 20.0
RECALL_EPOCHS = ('pre_tone', 'post_tone')
RECALL_REFERENCE_EPOCH = 'pre_tone'
RECALL_RESPONSE_EPOCH = 'post_tone'

# Human-readable session names. Test_B and Test_B_1wk are analysed SEPARATELY and are never
# compared with each other in this lane -- see render_paper_recall_amplitude_rate's docstring for
# why (the two recall cohorts are not the same animals).
RECALL_SESSION_LABELS = {'Test_B': '48-h recall (Test B)',
                         'Test_B_1wk': '1-week recall (Test B 1wk)'}


def get_recall_epoch_frames(session, epoch_name, trial_idx,
                            pre_tone_duration_s=TESTB_PRE_TONE_DURATION_S,
                            post_tone_duration_s=TESTB_POST_TONE_DURATION_S):
    """
    (onset, offset) for one of the two paper-facing RECALL windows, or None when the window is
    not FULLY present on that trial.

    Thin guard over get_testb_epoch_frames -- the window arithmetic lives there and is not
    duplicated here. What this adds is the completeness check get_testb_epoch_frames does not
    make: it happily returns a post-tone window that runs past the end of the recording (or into
    the next tone), because its 35 s default is used elsewhere for descriptive purposes where a
    short window is tolerable. It is NOT tolerable here: the recall analysis compares a
    duration-matched pre/post pair, and a truncated post-tone window would make the rate outcome
    partly a measure of window length.

    ** A missing window returns None; an AMBIGUOUS one raises. ** The two are different problems.
    A recording that stops 12 s after the last tone offset genuinely has no 20 s post-tone window
    on that trial -- a definitional absence, exactly like `pre_tone` on TFC trial 0 or
    `post_shock_late` on a truncated final trial, handled downstream by restricting each animal
    to the trials where BOTH windows exist (restrict_to_exposure_matched_trials) and reporting the
    resulting coverage. But a tone structure in which the 20 s pre-tone window would OVERLAP the
    previous trial's 20 s post-tone window means the two epochs are not disjoint and the
    pre-vs-post contrast is not the contrast this analysis claims to compute; that is a session
    the analyst has to look at, so it raises rather than silently returning a shorter or
    overlapping window.

    Scoped to RECALL_EPOCHS: 'tone' is not part of this pass and asking for it here raises, so a
    caller cannot quietly widen the model to three epochs through this function.
    """
    if epoch_name not in RECALL_EPOCHS:
        raise ValueError(
            f'get_recall_epoch_frames: {epoch_name!r} is not one of the paper-facing recall '
            f'windows {RECALL_EPOCHS}. The tone epoch itself is deliberately excluded from this '
            f'analysis; use get_testb_epoch_frames directly for descriptive tone-window work.')

    frames = get_testb_epoch_frames(session, epoch_name, trial_idx,
                                    pre_tone_duration_s=pre_tone_duration_s,
                                    post_tone_duration_s=post_tone_duration_s)
    if frames is None:
        return None
    onset, offset = frames

    if epoch_name == 'pre_tone':
        if trial_idx > 0:
            prev_post_tone_end = (session.tone_offsets[trial_idx - 1]
                                  + int(round(post_tone_duration_s * MINISCOPE_FPS)))
            if onset < prev_post_tone_end:
                raise RuntimeError(
                    f'get_recall_epoch_frames: the {pre_tone_duration_s} s pre-tone window for '
                    f'trial {trial_idx} of {session.mouse} starts at frame {onset}, before the '
                    f'end of trial {trial_idx - 1}\'s {post_tone_duration_s} s post-tone window '
                    f'(frame {prev_post_tone_end}). The two recall epochs must be disjoint for '
                    f'the pre-vs-post contrast to mean what this analysis says it means. Check '
                    f'the measured tone timing for this session rather than shortening a window.')
        return onset, offset

    # post_tone: the window must fit before whatever ends this trial -- the next tone onset, or
    # (on the last trial) the end of the recording. TestBSession.post_tone_offsets is exactly
    # that boundary, so the check is the same one on both.
    trial_end = session.post_tone_offsets[trial_idx]
    if offset > trial_end:
        return None
    return onset, offset

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


def _nb_convergence_diagnostics(idata, var_names=None):
    """Max r_hat / min ESS / divergence count for one fitted Bayesian model.

    Factored out of fit_rate_group_epoch_model so the hierarchical cell-level rate model
    (fit_cell_epoch_rate_model) reports convergence through the SAME code rather than a parallel
    copy, per CLAUDE.md's dedup rule.

    ``var_names=None`` (the default) summarizes EVERY parameter, which is exactly what
    fit_rate_group_epoch_model did inline and what it continues to do -- that lane's emitted text
    is unchanged. A caller passes ``var_names`` when the hard convergence GATE must be scoped to
    the parameters the scientific claims rest on: a model with thousands of nuisance per-cell
    random intercepts would otherwise have its pass/fail decided by whichever poorly-identified
    singleton cell happened to mix worst, which is not a statement about the fixed effects,
    the dispersion, or the contrasts being reported. Scoping the gate is not the same as hiding
    the rest -- the caller is expected to summarize the unscoped block too and report it.

    ``ess_tail`` is included alongside ``ess_bulk``: bulk ESS speaks to the posterior mean, tail
    ESS to the interval endpoints, and this suite reports intervals.
    """
    summ = az.summary(idata) if var_names is None else az.summary(idata, var_names=var_names)
    if len(summ) == 0:
        raise RuntimeError(
            f'_nb_convergence_diagnostics: var_names={var_names!r} matched NO parameters in the '
            f'posterior, so this would be a convergence gate over an empty set -- which always '
            f'passes and means nothing. Check the parameter names against the fitted model.')
    return {
        'max_rhat': float(summ['r_hat'].max()),
        'min_ess_bulk': float(summ['ess_bulk'].min()),
        'min_ess_tail': float(summ['ess_tail'].min()),
        'n_divergent': int(idata.sample_stats['diverging'].values.sum()),
        'n_params': int(len(summ)),
        'worst_rhat_param': str(summ['r_hat'].idxmax()),
        'worst_ess_bulk_param': str(summ['ess_bulk'].idxmin()),
    }


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
    diagnostics = {name: _nb_convergence_diagnostics(idata)
                   for name, idata in (('full', idata_full), ('reduced', idata_reduced))}
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

def n_distinct_relabelings(group_labels):
    """Number of DISTINCT group relabelings of `group_labels` that preserve the observed group
    sizes -- the multinomial coefficient n! / prod_g n_g!.

    Exposed (rather than inlined into the enumerator) so a call site can report and ASSERT the
    size of the randomization space it is testing over: with a 6/5/5 cohort a pairwise-restricted
    space is C(11,6)=462 or C(10,5)=252 and the global one is 2,018,016, and those numbers belong
    in the output next to the p-value. Computed from the observed sizes, never hard-coded.
    """
    counts = collections.Counter(list(group_labels))
    total = sum(counts.values())
    out = math.factorial(total)
    for k in counts.values():
        out //= math.factorial(k)
    return int(out)


def _iter_group_relabelings(group_labels):
    """Yield every distinct size-preserving relabeling of `group_labels`, as a list of labels
    positionally aligned with the input.

    Enumeration (not sampling): the groups are taken in a fixed sorted order and each in turn is
    assigned to every combination of the still-unassigned positions, so each distinct multiset
    permutation is produced exactly once. The observed labeling is among them, which is what makes
    the resulting p-value a genuine exact randomization p-value rather than a conditional one.
    """
    labels = list(group_labels)
    counts = collections.Counter(labels)
    groups = sorted(counts)
    n = len(labels)

    def _rec(positions, gi, acc):
        if gi == len(groups) - 1:
            arr = list(acc)
            for p in positions:
                arr[p] = groups[gi]
            yield arr
            return
        g = groups[gi]
        for chosen in itertools.combinations(positions, counts[g]):
            arr = list(acc)
            chosen_set = set(chosen)
            for p in chosen:
                arr[p] = g
            yield from _rec([p for p in positions if p not in chosen_set], gi + 1, arr)

    yield from _rec(list(range(n)), 0, [None] * n)


def _permutation_heartbeat(label, i, total, t0, every):
    """Progress line for a long permutation loop: how far along, and how much longer.

    Display only -- it computes nothing and changes no result. It exists because the exact
    pairwise enumerations and the Monte Carlo omnibus each refit a mixed model hundreds to
    thousands of times, which is minutes of total silence otherwise.
    """
    if not every or i == 0 or i % every or i >= total:
        return
    elapsed = time.perf_counter() - t0
    rate = elapsed / i
    print(f'[hier   perm]     {label}: {i}/{total} ({100.0 * i / total:.0f}%), '
          f'{elapsed / 60:.1f} min elapsed, ~{rate * (total - i) / 60:.1f} min left', flush=True)


def mouse_label_permutation_test(stat_fn, mice_per_group, n_perm=20000, seed=0,
                                 restrict_to_groups=None, exact=False,
                                 exact_max_relabelings=3_000_000,
                                 progress_label=None, progress_every=None):
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

    exact          : False (default) keeps the Monte Carlo path above EXACTLY as it was -- every
                     pre-existing call site in this module is unchanged and no reported number
                     moves. True instead ENUMERATES every distinct size-preserving relabeling
                     (_iter_group_relabelings) and computes

                         p = #{|stat| >= |observed|} / n_relabelings

                     with NO +1/+1 correction: the observed labeling is itself one of the
                     enumerated draws, so the ratio is already the exact randomization p-value and
                     the Monte Carlo correction would only bias it upward.

                     ** Exactness is bought by the RESTRICTION, not by patience. ** The global
                     6/5/5 space holds 2,018,016 relabelings, enumerable only for a statistic
                     costing microseconds; a pairwise-restricted space holds 462 or 252, which is
                     enumerable even when stat_fn refits a mixed model on every draw. Raises if
                     the space exceeds `exact_max_relabelings` rather than silently starting a
                     computation that will not finish.

    exact_max_relabelings : guard on the enumerated space size (default 3,000,000).

    progress_label, progress_every : DISPLAY ONLY. When both are given, a progress line is printed
                     every `progress_every` draws. Default None/None prints nothing and leaves
                     every existing call site's behaviour byte-identical. Nothing about the
                     statistic, the null distribution or the p-value depends on them.

    Returns dict(observed, p_two_sided, n_perm, n_finite, null=ndarray, exact, n_relabelings).
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

    if exact:
        n_relabelings = n_distinct_relabelings(group_labels)
        if n_relabelings > exact_max_relabelings:
            raise ValueError(
                f'mouse_label_permutation_test(exact=True): the randomization space holds '
                f'{n_relabelings} distinct relabelings, above exact_max_relabelings='
                f'{exact_max_relabelings}. Restrict exchangeability (restrict_to_groups), raise '
                f'the cap deliberately if the statistic is cheap enough, or use the Monte Carlo '
                f'path -- do not start an enumeration that will not finish.')
        null = np.empty(n_relabelings, dtype=float)
        n_seen = 0
        _t0 = time.perf_counter()
        for i, shuffled in enumerate(_iter_group_relabelings(group_labels)):
            null[i] = stat_fn(dict(zip(mouse_list, shuffled)))
            n_seen += 1
            _permutation_heartbeat(progress_label, i, n_relabelings, _t0, progress_every)
        if n_seen != n_relabelings:
            raise RuntimeError(
                f'mouse_label_permutation_test(exact=True): the enumerator produced {n_seen} '
                f'relabelings but the multinomial coefficient for these group sizes is '
                f'{n_relabelings}. One of the two is wrong; refusing to report a p-value over an '
                f'enumeration that is not the space it claims to be.')
        n_perm = n_relabelings
    else:
        n_relabelings = None
        rng = np.random.default_rng(seed)
        null = np.empty(n_perm, dtype=float)
        _t0 = time.perf_counter()
        for i in range(n_perm):
            shuffled = rng.permutation(group_labels)
            perm_assignment = dict(zip(mouse_list, shuffled))
            null[i] = stat_fn(perm_assignment)
            _permutation_heartbeat(progress_label, i, n_perm, _t0, progress_every)

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
    n_as_extreme = int(np.sum(np.abs(null[finite]) >= np.abs(observed)))
    if exact:
        # NO +1/+1 here: the observed labeling is one of the enumerated draws, so the ratio is
        # already the exact randomization p-value. Adding the Monte Carlo correction on top would
        # double-count the observed draw and bias every exact p upward.
        p_two_sided = n_as_extreme / n_finite
    else:
        # +1/+1 (conventional Monte Carlo correction) so a finite number of draws never reports p=0.
        p_two_sided = (n_as_extreme + 1) / (n_finite + 1)
    return {'observed': observed, 'p_two_sided': float(p_two_sided), 'n_perm': n_perm,
            'n_finite': n_finite, 'null': null, 'exact': bool(exact),
            'n_relabelings': n_relabelings}


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
# THE UNIFIED PAPER-FACING MODELS (see UNIFIED_OUTCOMES for what this is and why)
# ─────────────────────────────────────────────────────────────────────────────


def build_mouse_epoch_unified_table(df_matched, epochs=TFC_MATCHED_PROFILE_EPOCHS,
                                    reference_epoch=UNIFIED_REFERENCE_EPOCH):
    """
    The inferential dataset for the paper: ONE row per (mouse, epoch), carrying both paper-facing
    outcomes. With a complete cohort this is 17 mice x 3 epochs = 51 rows for the TFC lane, and
    n_animals_present x 2 rows for a recall session (see render_paper_recall_amplitude_rate).

    ** Both outcomes come off ONE frame, so they cannot silently diverge. ** Amplitude and rate
    are presented as parallel rows of one figure, so they must be computed from the same animals,
    the same epochs and the same retained trials. Passing `df_matched` (the output of
    restrict_to_exposure_matched_trials over `epochs`) once and deriving both here makes that true
    by construction rather than by a downstream assertion -- there is no code path in which the
    rate row is built from a different trial set than the amplitude row above it.

    ** Per-event amplitude. ** Per epoch, aggregate_over_trials pools each cell's events across the
    retained trials, filter_amplitude_rows drops the cells with no event in that epoch (amplitude
    is undefined for them, definitionally, not missing) and takes log of each cell's mean event-run
    integral, and those cell-level LOGS are averaged within the mouse. That weighting is
    deliberate and is the one this module has always used: each active cell contributes equally
    within its animal, and each animal contributes exactly one value to the model. Cells are NOT
    pooled across animals before the group mean.

    ** exp(a group contrast on this response) is a ratio of GEOMETRIC means ** of the cell-level
    mean event-run integrals -- not a ratio of pooled arithmetic mean event amplitudes. The mean
    of logs is the log of the geometric mean; describing it as a ratio of arithmetic means would
    be wrong by Jensen's inequality, and the amount wrong depends on each group's cell-level
    spread. Every place this ratio is reported says so.

    ** Population event rate. ** Total events over ALL detected pyramidal cells (zero-event cells
    included, which is the whole point of a POPULATION rate) divided by the corresponding total
    cell-seconds. Because thousands of cells contribute, a zero is not expected -- but it is
    checked rather than assumed, and a zero RAISES instead of receiving a pseudocount. A
    pseudocount would put an arbitrary constant inside a log on the response scale, where its size
    determines the answer.

    Categorical dtypes are set here with the REFERENCE LEVEL FIRST (mCherry, and `reference_epoch`
    -- pre_tone_matched for TFC, pre_tone for recall) so the model's reference never depends on
    alphabetical category order; fit_mixed_model deliberately does not touch dtypes. The same
    `reference_epoch` must be passed to fit_unified_group_epoch_model and
    unified_posthoc_contrasts, or the contrasts would be built around a different baseline from
    the one the model was fit with.

    Returns a DataFrame with columns
        mouse, group, epoch, n_cells, n_active_cells,
        mouse_mean_log_amplitude, geometric_mean_amplitude,
        total_events, total_cell_seconds, population_rate, log_population_rate
    """
    epochs = tuple(epochs)
    if reference_epoch not in epochs:
        raise ValueError(f'build_mouse_epoch_unified_table: reference_epoch '
                         f'{reference_epoch!r} is not among the epochs {epochs}.')
    rows = []
    for epoch in epochs:
        pooled = aggregate_over_trials(df_matched, epoch)
        amp = filter_amplitude_rows(pooled)
        amp_by_mouse = amp.groupby(['mouse', 'group'], observed=True)['log_amplitude'].agg(
            ['mean', 'size'])
        for (mouse, group), grp in pooled.groupby(['mouse', 'group'], observed=True):
            if (mouse, group) not in amp_by_mouse.index:
                raise RuntimeError(
                    f'build_mouse_epoch_unified_table: mouse {mouse!r} ({group}) has no cell with '
                    f'>=1 event in epoch {epoch!r}, so its per-event amplitude is undefined. The '
                    f'unified model needs a complete mouse x epoch grid; investigate the '
                    f'recording rather than dropping the row.')
            mean_log_amp = float(amp_by_mouse.loc[(mouse, group), 'mean'])
            total_events = float(grp['n_events'].sum())
            total_cell_seconds = float(grp['exposure_seconds'].sum())
            if total_cell_seconds <= 0:
                raise RuntimeError(
                    f'build_mouse_epoch_unified_table: mouse {mouse!r} epoch {epoch!r} has '
                    f'{total_cell_seconds} cell-seconds of exposure.')
            population_rate = total_events / total_cell_seconds
            if population_rate <= 0:
                raise RuntimeError(
                    f'build_mouse_epoch_unified_table: mouse {mouse!r} ({group}) has a population '
                    f'event rate of {population_rate} in epoch {epoch!r} over '
                    f'{len(grp)} cells. log() is undefined there and NO pseudocount is applied -- '
                    f'a zero population rate across every cell of an animal is a data problem, '
                    f'not a value to smooth over.')
            if not np.isfinite(mean_log_amp):
                raise RuntimeError(
                    f'build_mouse_epoch_unified_table: mouse {mouse!r} epoch {epoch!r} has a '
                    f'non-finite mean log amplitude ({mean_log_amp}).')
            rows.append({
                'mouse': mouse, 'group': group, 'epoch': epoch,
                'n_cells': int(len(grp)),
                'n_active_cells': int(amp_by_mouse.loc[(mouse, group), 'size']),
                'mouse_mean_log_amplitude': mean_log_amp,
                'geometric_mean_amplitude': float(np.exp(mean_log_amp)),
                'total_events': total_events,
                'total_cell_seconds': total_cell_seconds,
                'population_rate': population_rate,
                'log_population_rate': float(np.log(population_rate)),
            })

    out = pd.DataFrame(rows)
    # Completeness: every mouse must carry every epoch, or the model silently becomes unbalanced
    # and the interaction test is answering a different question from the one documented.
    mice = sorted(out['mouse'].unique())
    missing = [(m, e) for m in mice for e in epochs
               if not ((out['mouse'] == m) & (out['epoch'] == e)).any()]
    if missing:
        raise RuntimeError(f'build_mouse_epoch_unified_table: missing (mouse, epoch) rows '
                           f'{missing}; the unified model requires a complete grid.')
    if len(out) != len(mice) * len(epochs):
        raise RuntimeError(f'build_mouse_epoch_unified_table: got {len(out)} rows for '
                           f'{len(mice)} mice x {len(epochs)} epochs -- a mouse appears under '
                           f'more than one group.')

    out['group'] = pd.Categorical(
        out['group'],
        categories=[UNIFIED_REFERENCE_GROUP] + [g for g in GROUP_ORDER
                                                if g != UNIFIED_REFERENCE_GROUP])
    out['epoch'] = pd.Categorical(
        out['epoch'], categories=[reference_epoch] + [e for e in epochs
                                                      if e != reference_epoch])
    return out.sort_values(['group', 'mouse', 'epoch']).reset_index(drop=True)


def _unified_formula(response_col, reference_epoch=UNIFIED_REFERENCE_EPOCH):
    return (f'{response_col} ~ C(group, Treatment(reference="{UNIFIED_REFERENCE_GROUP}"))'
            f' * C(epoch, Treatment(reference="{reference_epoch}"))')


def fit_unified_group_epoch_model(df_me, response_col,
                                  reference_epoch=UNIFIED_REFERENCE_EPOCH):
    """
    Fit `response_col ~ group * epoch + (1|mouse)` on the mouse x epoch table -- IDENTICALLY for
    both paper-facing outcomes, which is the entire point of this function existing rather than
    two bespoke ones.

    Goes through fit_mixed_model so the module's one documented convergence/degeneracy criterion
    applies; the caller is expected to run require_common_unified_method() over both fits before
    using either, because a fallback taken by ONE outcome would break the claim that amplitude and
    rate received the same treatment.

    The omnibus is the joint Wald test that EVERY group x epoch coefficient is zero, with
    df2 = n_mice - 1 (joint_wald_test's animal-level convention -- the same denominator df the
    post-hoc contrasts use, so the interaction test and its simple effects are one procedure).
    Its df1 is (n_groups - 1) x (n_epochs - 1): 4 for the three-epoch TFC model, 2 for the
    two-epoch recall model, where it is exactly the test of whether the treatment-vs-control
    difference changes from pre-tone to post-tone.

    ** n_mice is read off `df_me`, never assumed. ** The recall sessions are each missing a
    DIFFERENT animal, so the denominator df of a recall model is that session's own
    n_animals_present - 1 and is not interchangeable with the conditioning model's 16.

    Returns dict(result, method, summary_text, formula, n_mice, fe_names, interaction_names,
    omnibus, response_col).
    """
    formula = _unified_formula(response_col, reference_epoch=reference_epoch)
    n_mice = int(df_me['mouse'].nunique())
    n_groups = int(df_me['group'].nunique())
    n_epochs = int(df_me['epoch'].nunique())
    result, method, text = fit_mixed_model(df_me, formula, group_col='mouse')
    fe_names, _params = _fe_names_and_params(result)
    # Read the interaction coefficients off the FITTED model rather than reconstructing
    # statsmodels' dummy-name format, which depends on the formula.
    interaction_names = [n for n in fe_names if ':' in n]
    expected_interactions = (n_groups - 1) * (n_epochs - 1)
    if len(interaction_names) != expected_interactions:
        raise RuntimeError(f'fit_unified_group_epoch_model: expected {expected_interactions} '
                           f'group x epoch coefficients ({n_groups} groups x {n_epochs} epochs), '
                           f'found {len(interaction_names)}: {interaction_names}. '
                           f'Available: {fe_names}')
    omnibus = joint_wald_test(result, interaction_names, n_mice)
    return {'result': result, 'method': method, 'summary_text': text, 'formula': formula,
            'n_mice': n_mice, 'fe_names': fe_names, 'interaction_names': interaction_names,
            'omnibus': omnibus, 'response_col': response_col}


def require_common_unified_method(fits):
    """Raise unless EVERY unified fit used the same intended random-intercept mixed model.

    ** The paper claims amplitude and rate received the same statistical treatment. ** That claim
    is false the moment one of them quietly takes fit_mixed_model's documented clustered-OLS
    fallback while the other stays on MixedLM -- the two would then differ in how between-animal
    variance is handled, which is precisely the heterogeneity this rewrite removed. So the
    paper-facing analysis HARD-FAILS and names the offending outcome instead of proceeding.

    Adopting clustered OLS as a COMMON fallback (both outcomes refit that way) is a deliberate
    decision to be made after inspecting the failure, not something to apply automatically here.
    The internal analyses keep their per-model fallback and are unaffected.
    """
    bad = {key: fit['method'] for key, fit in fits.items() if fit['method'] != 'mixedlm'}
    if bad:
        details = '\n\n'.join(f'--- {key} ({fits[key]["method"]}) ---\n'
                              f'{fits[key]["summary_text"].splitlines()[0]}' for key in bad)
        raise RuntimeError(
            f'require_common_unified_method: outcome(s) {sorted(bad)} did not fit as a mixed '
            f'model (methods: {bad}). The unified paper analysis requires BOTH outcomes on the '
            f'same estimator; it will not report one MixedLM result beside one clustered-OLS '
            f'result as though they were the same procedure. Inspect the fit and decide '
            f'explicitly whether to refit BOTH with the common fallback.\n\n{details}')


def _unified_group_coef(fe_names, group):
    """The main-effect coefficient name for `group` (no ':' -- an interaction term also contains
    the group name and would otherwise match)."""
    matches = [n for n in fe_names if f'[T.{group}]' in n and ':' not in n]
    if len(matches) != 1:
        raise RuntimeError(f'_unified_group_coef: expected exactly one main-effect coefficient '
                           f'for group {group!r}, found {matches} in {fe_names}.')
    return matches[0]


def _unified_epoch_coef(fe_names, epoch):
    """The main-effect coefficient name for `epoch` -- the reference group's pre->post change.

    The epoch counterpart of _unified_group_coef, and matched the same way: an interaction term
    also contains the epoch name, so ':' terms are excluded. Used by the modulation contrasts,
    where the control group's own pre->post change IS this coefficient and each treatment group's
    is this coefficient plus that group's interaction term.
    """
    matches = [n for n in fe_names if f'[T.{epoch}]' in n and ':' not in n]
    if len(matches) != 1:
        raise RuntimeError(f'_unified_epoch_coef: expected exactly one main-effect coefficient '
                           f'for epoch {epoch!r}, found {matches} in {fe_names}.')
    return matches[0]


def _unified_epoch_interaction_coef(fe_names, group, epoch):
    """The `group x epoch` coefficient name for this (group, epoch) pair."""
    matches = [n for n in fe_names if ':' in n and f'[T.{group}]' in n and f'[T.{epoch}]' in n]
    if len(matches) != 1:
        raise RuntimeError(f'_unified_epoch_interaction_coef: expected exactly one interaction '
                           f'coefficient for ({group!r}, {epoch!r}), found {matches}.')
    return matches[0]


def unified_posthoc_contrasts(fits, df_me, epochs=TFC_MATCHED_PROFILE_EPOCHS,
                              reference_epoch=UNIFIED_REFERENCE_EPOCH,
                              across_epoch_family=True, alpha=0.05):
    """
    The planned treatment-vs-control simple effects, from the fitted models -- the single
    authoritative source for every effect estimate, interval, p-value and asterisk on the
    paper-facing figures.

    ** Twelve contrasts; the family is the two comparisons WITHIN an epoch. ** Three epochs x
    {hM3D vs mCherry, hM4D vs mCherry} are estimated for each outcome. For each (outcome, epoch)
    those two treatment-vs-control simple effects are Holm-corrected together -> `p_holm_epoch`,
    finite on every row. That is six two-member families (3 epochs x 2 outcomes), and the
    procedure is IDENTICAL in all six: every visually equivalent panel of the paper figure is
    treated equivalently, and no epoch is a special case. Amplitude and rate remain separate
    outcomes. hM3D-vs-hM4D is in no family and is not computed here -- it is not this design's
    question and appears only as exploratory output.

    ** `p_holm_epoch` is the only adjusted P behind a paper asterisk or Results claim. ** Whether
    the treatment effect DIFFERS across epochs is a different question and is tested by the
    group x epoch interaction and by nothing else (unified_interactions_table). A significant
    comparison in one epoch and not another is not itself evidence that the epochs differ.

    ** The conservative six-comparison family is retained, not deleted. ** `p_holm_six` Holm-
    corrects all six treatment-vs-control simple effects ACROSS the three epochs within each
    outcome -- the more expansive definition of the family -- and is reported as a SENSITIVITY
    analysis. It supplies no figure asterisk and no Part 1 number in paper_results_summary.md.
    Both families are computed from the same twelve raw contrasts; only the grouping differs.

    ** `across_epoch_family` is the TFC lane's sensitivity correction and is scoped to it. ** The
    six-comparison family exists because the conditioning analysis reports three epochs and the
    wider definition of the family had to stay auditable after the multiplicity structure was
    settled post-inspection. The RECALL lane has two epochs and two comparisons per epoch, its
    epochs are a baseline and its single response window rather than three co-equal displayed
    windows, and no such wider family was ever declared for it -- so it passes False and the
    `p_holm_six` columns are simply absent there rather than present under a name that would be
    numerically wrong (there are four, not six). Consumers read `p_holm_six` with `.get`.

    ** A simple effect is a CONTRAST, not a coefficient. ** At the reference epoch the
    treatment-vs-control difference is the group main coefficient alone; at any other epoch it is
    that coefficient PLUS the corresponding group x epoch coefficient, with a variance that
    involves their covariance. Reading the group coefficient off the model summary and calling it
    "the effect at trace" would be wrong at every non-reference epoch.

    Estimates are on the model's log scale; `ratio` and its interval are exp() of the estimate and
    of the interval bounds (never of the standard error). For amplitude that ratio is a ratio of
    geometric means -- see build_mouse_epoch_unified_table.

    For the rate outcome the observed equal-mouse-weighted group means and their absolute
    difference in events/s/cell are attached as DESCRIPTIVE columns. They carry no test: a rate
    ratio computed off a small base can overstate the practical size of a change, and the absolute
    difference is what tells a reader how large the change actually is. They are NOT the
    inferential estimate; the inferential estimate is always the model contrast.

    Returns a tidy DataFrame, one row per (outcome, epoch, comparison).
    """
    epochs = tuple(epochs)
    rows = []
    for outcome in UNIFIED_OUTCOMES:
        fit = fits[outcome.key]
        fe_names = fit['fe_names']
        for epoch in epochs:
            for group in UNIFIED_TREATMENT_GROUPS:
                weights = {_unified_group_coef(fe_names, group): 1.0}
                if epoch != reference_epoch:
                    weights[_unified_epoch_interaction_coef(fe_names, group, epoch)] = 1.0
                res = linear_contrast_test(fit['result'], weights, fit['n_mice'], alpha=alpha)
                row = {
                    'outcome': outcome.key, 'epoch': epoch,
                    'comparison': f'{group}_vs_{UNIFIED_REFERENCE_GROUP}',
                    'group': group,
                    'estimate_log': res['estimate'], 'se': res['se'],
                    'ci_low_log': res['ci_low'], 'ci_high_log': res['ci_high'],
                    'ratio': float(np.exp(res['estimate'])),
                    'ratio_ci_low': float(np.exp(res['ci_low'])),
                    'ratio_ci_high': float(np.exp(res['ci_high'])),
                    'df': res['df'], 't': res['t'], 'p_raw': res['p'],
                }
                if outcome.key == 'population_rate':
                    # DESCRIPTIVE ONLY -- equal-mouse-weighted observed means, not model output.
                    sub = df_me[df_me['epoch'] == epoch]
                    mean_t = float(sub.loc[sub['group'] == group, 'population_rate'].mean())
                    mean_c = float(sub.loc[sub['group'] == UNIFIED_REFERENCE_GROUP,
                                           'population_rate'].mean())
                    row['mean_population_rate_treatment'] = mean_t
                    row['mean_population_rate_control'] = mean_c
                    row['absolute_difference_events_per_s_per_cell'] = mean_t - mean_c
                rows.append(row)

    out = pd.DataFrame(rows)

    # PAPER-FACING family: Holm within (outcome, epoch) across that epoch's two
    # treatment-vs-control contrasts. Finite on every row -- the six families are the same shape
    # and the same procedure, so no row is a special case.
    out['p_holm_epoch'] = np.nan
    out['holm_epoch_reject'] = False
    if across_epoch_family:
        # SENSITIVITY family: Holm within outcome across all six, populated on every row.
        out['p_holm_six'] = np.nan
        out['holm_six_reject'] = False

    for (outcome_key, epoch), idx in out.groupby(['outcome', 'epoch'],
                                                 observed=True).groups.items():
        idx = list(idx)
        if len(idx) != len(UNIFIED_TREATMENT_GROUPS):
            raise RuntimeError(
                f'unified_posthoc_contrasts: outcome {outcome_key!r} at epoch {epoch!r} produced '
                f'{len(idx)} treatment-vs-control contrasts, expected exactly '
                f'{len(UNIFIED_TREATMENT_GROUPS)}. Each epoch\'s Holm family is exactly its two '
                f'treatment-vs-control comparisons.')
        reject_epoch, padj_epoch = holm_correct(out.loc[idx, 'p_raw'].to_numpy(), alpha=alpha)
        out.loc[idx, 'p_holm_epoch'] = padj_epoch
        out.loc[idx, 'holm_epoch_reject'] = reject_epoch

    for outcome_key, idx in (out.groupby('outcome', observed=True).groups.items()
                             if across_epoch_family else ()):
        idx = list(idx)
        if len(idx) != len(epochs) * len(UNIFIED_TREATMENT_GROUPS):
            raise RuntimeError(
                f'unified_posthoc_contrasts: outcome {outcome_key!r} produced {len(idx)} '
                f'contrasts, expected {len(epochs) * len(UNIFIED_TREATMENT_GROUPS)} '
                f'({len(epochs)} epochs x {len(UNIFIED_TREATMENT_GROUPS)} treatment groups). '
                f'The multiplicity families are defined by that shape.')
        reject_six, padj_six = holm_correct(out.loc[idx, 'p_raw'].to_numpy(), alpha=alpha)
        out.loc[idx, 'p_holm_six'] = padj_six
        out.loc[idx, 'holm_six_reject'] = reject_six
    return out


def unified_contrast_lookup(contrasts, outcome, epoch, group):
    """The single row of `contrasts` for one (outcome, epoch, treatment group), as a Series.

    Every paper-facing consumer -- the figure's brackets, its companion markdown, the forest, the
    Results summary -- goes through here, so a panel cannot read a different row from the one its
    caption describes. Raises rather than returning an empty match.
    """
    sel = contrasts[(contrasts['outcome'] == outcome) & (contrasts['epoch'] == epoch)
                    & (contrasts['group'] == group)]
    if len(sel) != 1:
        raise RuntimeError(f'unified_contrast_lookup: expected exactly one row for '
                           f'({outcome!r}, {epoch!r}, {group!r}), found {len(sel)}.')
    return sel.iloc[0]


def unified_interactions_table(fits):
    """One row per outcome: the joint group x epoch Wald test. This is THE epoch-specificity
    result for the paper -- the permutation interaction statistic remains internal."""
    return pd.DataFrame([
        {'outcome': outcome.key, 'response_col': fits[outcome.key]['response_col'],
         'F': fits[outcome.key]['omnibus']['F'], 'df1': fits[outcome.key]['omnibus']['df1'],
         'df2': fits[outcome.key]['omnibus']['df2'], 'p': fits[outcome.key]['omnibus']['p'],
         'n_mice': fits[outcome.key]['n_mice'], 'method': fits[outcome.key]['method'],
         'formula': fits[outcome.key]['formula']}
        for outcome in UNIFIED_OUTCOMES])


# Below this, three-decimal fixed-point formatting renders a real p-value as the literally false
# 'P = 0.000'. Reported as an inequality instead -- the convention every journal uses and the one
# thing a fixed-point format cannot express.
_P_DISPLAY_FLOOR = 0.001


def format_p_display(p, decimals=3):
    """`0.0789` -> `P = 0.079`; anything below _P_DISPLAY_FLOOR -> `P < 0.001`.

    A tiny p-value printed at three decimals reads `P = 0.000`, which claims a p-value of exactly
    zero. One formatter so no caller re-invents the threshold.
    """
    p = float(p)
    if not np.isfinite(p):
        raise ValueError(f'format_p_display: p must be finite, got {p!r}.')
    if p < _P_DISPLAY_FLOOR:
        return f'P < {_P_DISPLAY_FLOOR:g}'
    return f'P = {p:.{decimals}f}'


def format_unified_interaction(row):
    """`group x epoch F(4, 16) = 1.23, P = 0.34` -- one wording, used by every figure annotation
    and every text file, so the interaction can never be quoted two different ways."""
    return (f'group x epoch F({int(row["df1"])}, {int(row["df2"])}) = {row["F"]:.2f}, '
            f'{format_p_display(row["p"])}')


# ── Diagnostics (descriptive; they gate nothing and add no inference) ─────────

# Minimum absolute shift, in log units (~10%), before a leave-one-mouse-out estimate change is
# worth mentioning at all. See unified_model_diagnostics for why a purely relative rule is useless
# on a null contrast.
_INFLUENCE_FLAG_ABS_LOG = 0.10

def unified_model_diagnostics(fits, df_me, contrasts, save_dir,
                              epochs=TFC_MATCHED_PROFILE_EPOCHS,
                              reference_epoch=UNIFIED_REFERENCE_EPOCH,
                              across_epoch_family=True,
                              filename_root='unified_lmm_diagnostics',
                              residuals_csv='unified_lmm_residuals.csv',
                              influence_csv='unified_lmm_influence.csv'):
    """
    Residual and leave-one-mouse-out diagnostics for both unified models.

    ** Descriptive only. ** The main rate endpoint is a log-transformed mouse-level population
    rate analysed with a Gaussian LMM, which is a modelling simplification (the distribution-aware
    check is the negative-binomial count model, retained as a sensitivity analysis), so its
    residual behaviour should be LOOKED AT. Nothing here gates the analysis or produces a
    p-value that enters any family.

    ** No normality test is used as an acceptance criterion. ** A mixed model's residuals are not
    51 independent draws, and a residual-normality test crossing 0.05 is not a principled
    pass/fail rule at this n. The plots are the diagnostic.

    ** Leave-one-mouse-out reports ESTIMATES, not decisions. ** Whether an adjusted p-value
    crosses 0.05 when one of 17 animals is dropped is expected behaviour near alpha and says
    nothing about robustness; what matters is whether the effect ESTIMATE changes qualitatively.
    Only that is flagged, and only as a note.

    Writes <filename_root>.png, <filename_root>.txt, <residuals_csv> and <influence_csv>. The
    filenames are parameters because the recall lane writes its own copies of these into its own
    stats directory under `unified_recall_*` names, matching the rest of that lane's output.
    """
    ensure_dirs(save_dir)
    resid_rows = []
    fig, axs = plt.subplots(len(UNIFIED_OUTCOMES), 2,
                            figsize=(7.0, 3.0 * len(UNIFIED_OUTCOMES)))
    axs = np.atleast_2d(axs)
    for r, outcome in enumerate(UNIFIED_OUTCOMES):
        fit = fits[outcome.key]
        result = fit['result']
        observed = df_me[outcome.response_col].to_numpy(dtype=float)
        fitted = np.asarray(result.fittedvalues, dtype=float).reshape(-1)
        resid = observed - fitted
        for i, (_, meta) in enumerate(df_me.iterrows()):
            resid_rows.append({'outcome': outcome.key, 'mouse': meta['mouse'],
                               'group': meta['group'], 'epoch': meta['epoch'],
                               'observed': observed[i], 'fitted': fitted[i],
                               'resid': resid[i]})
        axs[r, 0].scatter(fitted, resid, s=14, c='0.3', edgecolor='k', linewidth=0.3)
        axs[r, 0].axhline(0.0, color='k', linewidth=0.8, linestyle='--')
        axs[r, 0].set_xlabel('fitted', size='x-small')
        axs[r, 0].set_ylabel(f'{outcome.label}\nresidual', size='x-small')
        axs[r, 0].set_title('Residual vs fitted', size='small')
        # STANDARDIZED before the Q-Q, so the reference line is the identity and a departure from
        # it is readable as such. probplot(fit=False) returns the ordered sample VALUES, which on
        # a residual in native units would be plotted against unit-normal quantiles and make even
        # a perfectly normal residual look flat.
        resid_sd = float(np.std(resid, ddof=1))
        osm, osr = scipy_stats.probplot(resid / resid_sd if resid_sd > 0 else resid,
                                        dist='norm', fit=False)
        axs[r, 1].scatter(osm, osr, s=14, c='0.3', edgecolor='k', linewidth=0.3)
        lim = [min(osm.min(), osr.min()), max(osm.max(), osr.max())]
        axs[r, 1].plot(lim, lim, color='k', linewidth=0.8, linestyle='--')
        axs[r, 1].set_xlabel('theoretical quantiles', size='x-small')
        axs[r, 1].set_ylabel('standardized residual', size='x-small')
        axs[r, 1].set_title(f'Normal Q-Q (residual SD {resid_sd:.3g})', size='small')
        for ax in axs[r]:
            ax.spines[['right', 'top']].set_visible(False)
            ax.tick_params(labelsize='xx-small')
    fig.suptitle('Unified LMM diagnostics — descriptive; no acceptance criterion', size='small')
    fig.subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.10, hspace=0.55, wspace=0.30)
    save_fig(fig, os.path.join(save_dir, filename_root + '.png'))
    plt.close(fig)
    write_text(os.path.join(save_dir, residuals_csv),
               pd.DataFrame(resid_rows).to_csv(index=False))

    influence, notes = [], []
    for mouse in sorted(df_me['mouse'].unique()):
        sub = df_me[df_me['mouse'] != mouse]
        lomo_fits = {o.key: fit_unified_group_epoch_model(sub, o.response_col,
                                                          reference_epoch=reference_epoch)
                     for o in UNIFIED_OUTCOMES}
        lomo = unified_posthoc_contrasts(lomo_fits, sub, epochs=epochs,
                                         reference_epoch=reference_epoch,
                                         across_epoch_family=across_epoch_family)
        for _, row in lomo.iterrows():
            full = unified_contrast_lookup(contrasts, row['outcome'], row['epoch'], row['group'])
            delta = row['estimate_log'] - full['estimate_log']
            influence.append({
                'omitted_mouse': mouse, 'omitted_method': lomo_fits[row['outcome']]['method'],
                'outcome': row['outcome'], 'epoch': row['epoch'],
                'comparison': row['comparison'], 'estimate_log': row['estimate_log'],
                'ratio': row['ratio'], 'ratio_ci_low': row['ratio_ci_low'],
                'ratio_ci_high': row['ratio_ci_high'],
                'full_data_estimate_log': full['estimate_log'],
                'delta_from_full_estimate': delta})
            # A qualitative change: the estimate moves by more than half its full-data magnitude
            # AND by more than _INFLUENCE_FLAG_ABS_LOG in absolute terms. Deliberately NOT "an
            # adjusted p crossed 0.05". The absolute floor is not decoration: on a contrast that
            # is essentially null, a relative rule alone fires constantly (a ratio of 0.99 moving
            # to 1.05 is a >50% change in a log effect of -0.01) and buries the cases that matter.
            if abs(delta) > max(_INFLUENCE_FLAG_ABS_LOG, 0.5 * abs(full['estimate_log'])):
                notes.append(f"  {row['outcome']} / {row['epoch']} / {row['comparison']}: "
                             f"omitting {mouse} moves the ratio "
                             f"{full['ratio']:.3f} -> {row['ratio']:.3f}")
    influence_df = pd.DataFrame(influence)
    write_text(os.path.join(save_dir, influence_csv),
               influence_df.to_csv(index=False))
    n_contrasts = len(contrasts)
    n_mice_diag = int(df_me['mouse'].nunique())

    lines = [
        'Unified LMM diagnostics (descriptive)',
        '=' * 60,
        '',
        'These outputs describe the two paper-facing models. None of them is a test, none enters '
        'a multiplicity family, and none determines whether the models are accepted.',
        '',
        f'Residual plots: {filename_root}.png (residual-vs-fitted and normal Q-Q per outcome).',
        f'Per-row residuals: {residuals_csv}.',
        '',
        'No residual-normality test is reported as an acceptance criterion: a mixed model\'s '
        f'residuals are not {len(df_me)} independent observations, so a normality test crossing '
        '0.05 would not be a principled pass/fail rule here. Read the plots.',
        '',
        f'Leave-one-mouse-out ({influence_csv})',
        '-' * 60,
        f'Each of the {n_mice_diag} animals is dropped in turn and BOTH models are '
        f'refit, giving that animal\'s effect on each of the {n_contrasts} planned contrast '
        f'ESTIMATES. Significance-decision flips are deliberately not computed: at n = '
        f'{n_mice_diag} an adjusted p-value crossing 0.05 when one animal is removed is expected '
        'and is not evidence that one animal drives a result.',
        '',
    ]
    lines += ([f'Contrasts whose estimate moved by more than half the full-data effect and by '
               f'more than {_INFLUENCE_FLAG_ABS_LOG} log units ({len(notes)}):'] + notes if notes
              else [f'No omitted animal moved any contrast estimate by more than half the '
                    f'full-data effect and more than {_INFLUENCE_FLAG_ABS_LOG} log units.'])
    lines.append('')
    write_text(os.path.join(save_dir, filename_root + '.txt'), '\n'.join(lines))
    return influence_df


# ── Synthetic verification of the whole unified path ─────────────────────────

# Planted effects for verify_unified_synthetic, chosen so both expected conclusions sit far
# from alpha. A regression test whose correctness depends on a realization landing at P = 0.049
# rather than 0.051 tests the random seed, not the code.
#
# The treatment-vs-control simple effect is a BETWEEN-animal contrast, so its precision is set by
# the between-animal SD and by n = 5/6 -- which is why the planted shift is large relative to
# _SYNTHETIC_MOUSE_SD. The interaction, by contrast, is a WITHIN-animal comparison and is
# precise; the trace-only arm is therefore detected easily.
#
# The global arm's requirement (a NULL interaction) is the one quantity here that cannot be made
# arbitrarily safe by choosing effect sizes: under a true null the p-value is uniform, so the seed
# is fixed and the realised value is recorded in the output file.
_SYNTHETIC_SHIFT = 1.00           # log-scale treatment effect (~2.7x)
_SYNTHETIC_MOUSE_SD = 0.15        # between-animal SD of the random intercept
_SYNTHETIC_RESID_SD = 0.10        # within-animal residual SD
_SYNTHETIC_GROUP_SIZES = (('mCherry', 6), ('hM3D', 5), ('hM4D', 6))


def _synthetic_mouse_epoch_table(shift_epochs, epochs, rng,
                                 reference_epoch=UNIFIED_REFERENCE_EPOCH,
                                 group_sizes=_SYNTHETIC_GROUP_SIZES,
                                 shift_fn=None):
    """A synthetic mouse x epoch table in exactly build_mouse_epoch_unified_table's output shape,
    with `_SYNTHETIC_SHIFT` planted on both treatment groups in `shift_epochs` only.

    `group_sizes` is a parameter so a recall session can plant ITS OWN cohort (one animal fewer,
    and a different one in each session) rather than the conditioning cohort's 6/5/6.

    `shift_fn(group, epoch) -> log-scale offset` overrides the planted effect entirely, for
    designs that need a DIFFERENT shift per group -- which the modulation checks do, since a
    verification that plants the same effect on hM3D and hM4D cannot distinguish a correct
    hM3D-vs-hM4D contrast from one wired to the wrong coefficient. The default reproduces the
    original rule exactly, so verify_unified_synthetic is unaffected."""
    rows = []
    for group, n in group_sizes:
        for m in range(n):
            mouse = f'{group}_{m}'
            intercept = rng.normal(0.0, _SYNTHETIC_MOUSE_SD)
            for epoch in epochs:
                effect = (shift_fn(group, epoch) if shift_fn is not None
                          else (_SYNTHETIC_SHIFT if (group != UNIFIED_REFERENCE_GROUP
                                                     and epoch in shift_epochs) else 0.0))
                value = intercept + effect + rng.normal(0.0, _SYNTHETIC_RESID_SD)
                rows.append({'mouse': mouse, 'group': group, 'epoch': epoch,
                             'n_cells': 100, 'n_active_cells': 80,
                             'mouse_mean_log_amplitude': value,
                             'geometric_mean_amplitude': float(np.exp(value)),
                             'total_events': 1000.0, 'total_cell_seconds': 2000.0,
                             'population_rate': float(np.exp(value)),
                             'log_population_rate': value})
    out = pd.DataFrame(rows)
    out['group'] = pd.Categorical(
        out['group'], categories=[UNIFIED_REFERENCE_GROUP]
        + [g for g in GROUP_ORDER if g != UNIFIED_REFERENCE_GROUP])
    out['epoch'] = pd.Categorical(
        out['epoch'], categories=[reference_epoch]
        + [e for e in epochs if e != reference_epoch])
    return out


def verify_unified_synthetic(save_dir, epochs=TFC_MATCHED_PROFILE_EPOCHS,
                             reference_epoch=UNIFIED_REFERENCE_EPOCH,
                             response_epoch=TFC_TRACE_EPOCH,
                             across_epoch_family=True,
                             group_sizes=_SYNTHETIC_GROUP_SIZES,
                             model_label='TFC', seed=0,
                             filename='unified_lmm_synthetic_verification.txt'):
    """
    Run the ACTUAL unified fitting/contrast code against two planted datasets whose correct
    answers are known, and hard-fail if it does not recover them.

    Design 1 -- an EQUAL treatment shift in EVERY epoch. The planned simple effects must detect
    it, and the group x epoch interaction must stay null: a real effect that does not vary across
    epochs must not manufacture epoch specificity.

    Design 2 -- a shift in `response_epoch` ONLY. The interaction must become significant: an
    effect that does vary across epochs must be detected as varying.

    Together these are the two ways the unified analysis could be wrong in the direction that
    matters for the manuscript's claims. The margins required are deliberately wide (see
    _SYNTHETIC_SHIFT and friends) so this is a test of the code and not of one lucky realization.

    ** The same two designs are the recall lane's verification. ** The recall models reuse this
    machinery with two epochs and a session-specific cohort, where the second design is exactly
    the claim that matters there: a post-tone-only group shift must show up as a significant
    group x epoch interaction and a uniform shift must not. `response_epoch`, `epochs`,
    `reference_epoch` and `group_sizes` are therefore parameters rather than TFC constants.

    Runs on tables of a few dozen rows, so both designs cost milliseconds.
    """
    epochs = tuple(epochs)
    lines = [f'Synthetic verification of the unified {model_label} models',
             '=' * 60, '',
             f'Planted log-scale shift {_SYNTHETIC_SHIFT} on both treatment groups; '
             f'between-mouse SD {_SYNTHETIC_MOUSE_SD}, residual SD {_SYNTHETIC_RESID_SD}; '
             f'cohort {dict(group_sizes)}; epochs {epochs} (reference {reference_epoch!r}); '
             f'seed {seed}.', '']
    for offset, (label, shift_epochs) in enumerate([('global (all epochs)', epochs),
                                                    (f'{response_epoch}-only', (response_epoch,))]):
        # A fresh generator per design, so one arm's draws do not depend on the other's and either
        # can be re-run in isolation and reproduce the number in this file.
        df_syn = _synthetic_mouse_epoch_table(shift_epochs, epochs,
                                              np.random.default_rng(seed + offset),
                                              reference_epoch=reference_epoch,
                                              group_sizes=group_sizes)
        fits = {o.key: fit_unified_group_epoch_model(df_syn, o.response_col,
                                                     reference_epoch=reference_epoch)
                for o in UNIFIED_OUTCOMES}
        require_common_unified_method(fits)
        contrasts = unified_posthoc_contrasts(fits, df_syn, epochs=epochs,
                                              reference_epoch=reference_epoch,
                                              across_epoch_family=across_epoch_family)
        p_inter = fits['amplitude']['omnibus']['p']
        row = unified_contrast_lookup(contrasts, 'amplitude', response_epoch, 'hM3D')
        p_response = float(row['p_holm_epoch'])
        ratio = float(row['ratio'])
        lines += [f'{label}:',
                  f'  {response_epoch} hM3D-vs-mCherry: ratio {ratio:.3f}, '
                  f'within-epoch Holm-adjusted P at {response_epoch} = {p_response:.3g}',
                  f'  group x epoch: {format_unified_interaction(unified_interactions_table(fits).iloc[0])}',
                  '']
        assert p_response < 1e-3, (
            f'verify_unified_synthetic [{model_label}/{label}]: the planted {response_epoch} '
            f'effect (ratio {np.exp(_SYNTHETIC_SHIFT):.2f}x) should be detected with a '
            f'within-epoch Holm-adjusted P < 0.001 at {response_epoch}, got {p_response:.3g}.')
        if shift_epochs == epochs:
            assert p_inter > 0.2, (
                f'verify_unified_synthetic [{model_label}/{label}]: an equal shift in every epoch '
                f'must not produce epoch specificity; the group x epoch interaction should be '
                f'clearly null (P > 0.2) but was P = {p_inter:.3g}.')
        else:
            assert p_inter < 0.01, (
                f'verify_unified_synthetic [{model_label}/{label}]: a {response_epoch}-only shift '
                f'must be detected as epoch-dependent; the group x epoch interaction should be '
                f'clearly significant (P < 0.01) but was P = {p_inter:.3g}.')
    lines.append('All assertions passed.')
    ensure_dirs(save_dir)
    write_text(os.path.join(save_dir, filename), '\n'.join(lines))


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


def _precomputed_stat_fn(p_hm3d_vs_ctl, p_hm4d_vs_ctl, p_hm3d_vs_hm4d=np.nan):
    """A stat_fn that returns ALREADY-COMPUTED p-values and looks at no data at all.

    ** This is what makes a paper panel incapable of inventing its own statistic. ** The
    draw_superplot_triplet / annotate_pairwise_brackets interface calls stat_fn with the per-mouse
    arrays and annotates whatever comes back; the default (_panel_stat_fn) runs an independent
    per-panel Welch test there. On the paper figures the numbers must instead be the unified
    model's Holm-adjusted simple effects, so this closure ignores its arguments entirely and hands
    back the two values looked up from unified_lmm_posthoc_contrasts.csv. There is no code path
    by which a paper panel's asterisk disagrees with that table.

    Returns the pair order annotate_pairwise_brackets expects -- [(Exc,Inh), (Exc,Ctl), (Inh,Ctl)]
    -- with the first slot defaulting to NaN: on the ABSOLUTE-value paper panels hM3D-vs-hM4D is in
    neither Holm family and is not a paper comparison, and a NaN is skipped by the bracket drawer
    rather than drawn as non-significant.

    ** `p_hm3d_vs_hm4d` may be supplied only where that comparison is a declared member of the
    panel's OWN correction family. ** It is on the recall modulation panels, whose question is
    symmetric across the three groups and whose Holm family is exactly the three pairwise
    comparisons of the pre->post change (recall_modulation_contrasts). It is not on any
    absolute-value panel, which is why the default keeps those call sites unchanged.

    The NaN behaviour is also how a whole panel is left un-annotated: the paper distribution
    figure passes (NaN, NaN) for every non-primary epoch, so pre-tone and post-shock render no
    bracket at all while keeping the identical axis treatment as the trace column.
    """
    def _stat_fn(group_hM3D, group_hM4D, group_mCherry, ax, heights, annotate=True,
                 tot_dh_incr=0.12, barh=0, group_order=None, test=None):
        return np.array([float(p_hm3d_vs_hm4d), float(p_hm3d_vs_ctl), float(p_hm4d_vs_ctl)],
                        dtype=float)
    return _stat_fn


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
                                           title='Decomposition panel contrasts', preamble=None):
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
    no_star_note : replaces whichever default paragraph `figure_has_stars` would select. That
        `figure_has_stars=False` default is
        written about the four-component DECOMPOSITION and says so explicitly ("one exact
        decomposition, not independent phenotypes"), which is the right warning there and a
        false description of a star-free figure whose panels are one component across epochs or
        trial phases. A caller whose panels are not the decomposition supplies its own note
        rather than inheriting a wrong one.
    preamble : replaces the default provenance paragraphs, which describe equal-mouse-weighted
        WELCH intervals and are simply false beside model-derived ones. The paper lane, whose
        payloads come from the unified LMM contrasts, supplies its own.
    """
    default_preamble = [
        'Equal-mouse-weighted contrasts against the mCherry control (n = 5 hM3D / 6 hM4D / '
        '6 mCherry), Welch two-sample 95% intervals computed from the per-mouse means. Only '
        'per-mouse values enter these numbers; any cell-level display is descriptive.',
        '',
        'Ratios and absolute differences are reported together: a fold-change computed off a '
        'small base overstates the practical size of a change.',
        '',
    ]
    lines = [f'# {title}', ''] + (default_preamble if preamble is None else [preamble, ''])
    if preamble is not None:
        # A caller supplying its own provenance paragraph has already said where its numbers come
        # from; both defaults below describe how a WELCH interval relates to a separately computed
        # Holm-corrected star, which is not the relationship on a figure whose intervals and stars
        # come from one model. An explicit no_star_note still overrides.
        if no_star_note is not None:
            lines += [no_star_note, '']
    elif figure_has_stars:
        lines += [
            '**These intervals are not multiplicity-corrected.** The asterisks on the figure are '
            f'Holm-corrected across the family set by `PANEL_HOLM_FAMILY` (currently '
            f'`{PANEL_HOLM_FAMILY!r}`), so a contrast whose interval excludes 1.0 here may still '
            'carry no star. Both are reported deliberately: the interval describes the effect, '
            'the star describes the corrected decision.',
            '',
        ]
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
                             bracket_mode='data', stat_fn=None, ns_label_pairs=()):
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
    stat_fn  : override for the panel's statistic, exactly as on _draw_cell_superplot_panel.
               Defaults to _panel_stat_fn() (an independent per-panel Welch/Holm test on the
               per-mouse values), which is right for the internal figures. The recall modulation
               panels pass _precomputed_stat_fn so their brackets come from the fitted models
               instead of from a second test on the same numbers.
    ns_label_pairs : passed through to annotate_pairwise_brackets (bracket_mode='axes' only);
               pairs named here are bracketed with their p-value even when it is >= 0.05. Empty
               by default, so every other panel is unchanged. See that function's docstring.
    """
    ax.spines[['right', 'top']].set_visible(False)
    panel_stat_fn = _panel_stat_fn() if stat_fn is None else stat_fn
    draw_brackets_here = annotate == 'stats' and bracket_mode == 'data'
    _draw_violin_triplet(ax, values_per_group, 0, group_order, GROUP_COLOURS,
                         stat_fn=(panel_stat_fn if draw_brackets_here else no_stat_annotation),
                         ylabel=ylabel)
    if annotate == 'stats' and bracket_mode == 'axes':
        annotate_pairwise_brackets(ax, {g: v.ravel() for g, v in values_per_group.items()},
                                   group_order, panel_stat_fn, ns_label_pairs=ns_label_pairs)
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
                               jitter_width=0.34, stat_fn=None, mouse_means_override=None):
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
    stat_fn   : override for the panel's statistic. Defaults to _panel_stat_fn() (an independent
                per-panel Welch/Holm test on the per-mouse means), which is right for the internal
                figures. The paper lane passes _precomputed_stat_fn so its asterisks come from the
                unified models instead.
    mouse_means_override : {group: {mouse: value}} to draw as the large markers INSTEAD of the
                unweighted mean of that mouse's cell cloud.

                ** The marker must be the value the model was fit on. ** For the paper's rate row
                those differ: the population rate is total events over total cell-seconds, i.e. a
                cell-seconds-WEIGHTED mean of the per-cell rates, and the model is fit on its log.
                For the amplitude row the model response is the mean of cell-level LOG amplitudes,
                whose exp() is the mouse's geometric mean amplitude -- again not the unweighted
                mean of the raw cloud. Drawing the cloud mean beside a model fit on something else
                would put two different quantities under one marker.

                The override's mouse keys must match the cloud's exactly (asserted), and it must
                be on the same scale as `value_col` -- see the paper lane's
                _assert_markers_match_model, which checks that end to end against
                unified_lmm_mouse_epoch_values.csv.
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
        if mouse_means_override is None:
            mouse_means[group] = {m: float(np.mean(cell_values[group][m])) for m in mice}
        else:
            override = mouse_means_override[group]
            if set(override) != set(mice):
                raise RuntimeError(
                    f'{panel_name}: mouse_means_override for group {group} covers '
                    f'{sorted(override)} but the cell cloud covers {mice}. The marker and the '
                    f'cloud must describe the same animals.')
            mouse_means[group] = {m: float(override[m]) for m in mice}
    draw_superplot_triplet(ax, cell_values, mouse_means, group_order, GROUP_COLOURS,
                           stat_fn=(_panel_stat_fn() if stat_fn is None else stat_fn),
                           ylabel=ylabel, yscale=yscale,
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
                            epoch_labels=None, contrast_payloads=None, interaction_note=None,
                            subtitle=None, contrasts_note=None, contrasts_preamble=None,
                            row_height=1.5):
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
                    _panel_contrasts). Off for the paper figures: hM3D-vs-hM4D is in neither
                    unified multiplicity family (see unified_posthoc_contrasts) and is exploratory
                    internal output only -- docs/sp_rates_lmm.md section 5.2.
    epoch_labels  : {epoch key: column title}. Defaults to the raw epoch keys, which is right for
                    the internal figure -- they are what every stats filename and every other
                    reference to a window is keyed on. The paper lane passes readable window
                    names, since "pre_tone_matched" tells a reader nothing and the 20 s matching
                    is the reason the columns are comparable at all.
    contrast_payloads : {(component key, epoch): payload} to draw INSTEAD of the equal-mouse-
                    weighted Welch contrasts this function computes itself. The paper lane passes
                    the UNIFIED LMM contrasts (see _unified_contrast_payloads), so the forest's
                    points and intervals are the same model-derived numbers as the distribution
                    figure's stars rather than a second estimate of the same quantity computed a
                    different way. `df_fine` is then unused for the estimates and only the payload
                    shape is required; `include_exc_vs_inh` must be False, since a model-derived
                    payload carries no DREADD-vs-DREADD entry.
    interaction_note : {component key: annotation string} replacing `interaction_q`'s float, for a
                    caller whose epoch-specificity result is not a BH-adjusted permutation q. The
                    paper lane passes the unified joint Wald test (format_unified_interaction);
                    the internal grid keeps interaction_q. Passing both raises.
    subtitle, contrasts_note : override the figure's second title line and the companion markdown's
                    explanatory paragraph, which describe the internal figure's provenance and
                    would be false next to model-derived intervals.
    contrasts_preamble : override the companion markdown's DEFAULT provenance paragraphs, which
                    describe equal-mouse-weighted WELCH intervals over the conditioning cohort and
                    are simply false beside model-derived ones from a different cohort. The recall
                    lane supplies its own, naming that session's own animals and stats files.
    row_height    : inches per row. The row's epoch-specificity annotation lives on its y-LABEL,
                    so it has only the row's own axes height to occupy; a short figure with few
                    rows clips it. The four-row internal grid has height to spare at the default;
                    the two-row paper forest asks for more.
    """
    epochs = tuple(epochs)
    epoch_labels = epoch_labels or {}
    if interaction_q and interaction_note:
        raise ValueError('plot_decomposition_grid: pass interaction_q OR interaction_note, not '
                         'both -- a row must carry exactly one epoch-specificity annotation.')
    interaction_q = interaction_q or {}
    interaction_note = interaction_note or {}
    rows = tuple(components)
    if contrast_payloads is not None:
        if include_exc_vs_inh:
            raise ValueError('plot_decomposition_grid: include_exc_vs_inh is not available with '
                             'supplied contrast_payloads -- the unified model-derived payloads '
                             'carry no hM3D-vs-hM4D entry by design.')
        missing = [(spec.key, e) for spec in rows for e in epochs
                   if (spec.key, e) not in contrast_payloads]
        if missing:
            raise ValueError(f'plot_decomposition_grid: contrast_payloads is missing {missing}.')
        contrasts = dict(contrast_payloads)
    else:
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
                            figsize=(2.3 * len(epochs), row_height * len(rows) + 1.0))
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
                note = interaction_note.get(spec.key)
                if note is not None:
                    ax.set_ylabel(f'{spec.label}\n{note}', size='x-small')
                else:
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

    default_subtitle = ('Per-row group x epoch q: BH-adjusted joint permutation test for that '
                        'component')
    title = ('Components of population calcium activity: effect estimates with 95% CI\n'
             + (subtitle if subtitle is not None else default_subtitle))
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
        save_dir, filename_root + '_contrasts.md', figure_has_stars=False,
        no_star_note=contrasts_note, preamble=contrasts_preamble)


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
#
# ** Cells and mouse markers must be on the SAME scale, and the marker must be the model's own
# value. ** The unified models are fit on mouse_mean_log_amplitude and log_population_rate; both
# rows are DISPLAYED on the natural scale, with the marker the exp() of the fitted response:
#
#   amplitude       cells = mean_amplitude (each cell's mean event-run integral)
#                   marker = exp(mouse_mean_log_amplitude), that mouse's GEOMETRIC mean amplitude
#   population rate cells = overall_rate (events/s for that cell)
#                   marker = population_rate = total events / total cell-seconds
#
# The amplitude row previously plotted log_amplitude cells on a linear axis. Plotting raw
# amplitudes with a log-scale marker on one axis -- or the reverse -- would put two different
# quantities under one y-axis; _assert_markers_match_model checks the pairing that is actually
# drawn against unified_lmm_mouse_epoch_values.csv.
_PAPER_ROW_DISPLAY = {
    # yscale='auto' -> log, which is what a heavy-right-tailed per-cell amplitude needs; the
    # column is raw, not pre-logged, so this is a single log, not the double transform plotting
    # rule 2 warns about.
    'amplitude': {'yscale': 'auto', 'quantum': False, 'col': 'mean_amplitude',
                  'marker_col': 'geometric_mean_amplitude',
                  'ylabel': 'Per-event amplitude (a.u.)'},
    # A per-cell rate is a small integer count over a fixed window, so without sub-quantum jitter
    # the cloud collapses onto a few hard horizontal stripes -- real quantization, but it hides
    # the density that is the reason for drawing cells at all.
    'population_rate': {'yscale': 'auto', 'quantum': True, 'col': 'overall_rate',
                        'marker_col': 'population_rate',
                        'ylabel': 'Event rate (events/s/cell)'},
}


# One cell of a SuperPlot grid. `key` identifies it in error messages, `label` keys its entry in
# the companion contrasts file. The remaining fields are exactly what _draw_cell_superplot_panel
# and _panel_contrasts each need, so a panel carries its own display AND contrast configuration
# and the grid driver below needs to know nothing about what is being plotted.
# stat_fn / mouse_values / contrast_payload default to None, which keeps the internal behaviour
# (a per-panel Welch/Holm test, markers = the cloud mean, Welch contrasts in the companion file).
# The paper lane sets all three from the unified model tables so that no number on a paper panel
# is computed by the panel itself.
_GridPanel = collections.namedtuple(
    '_GridPanel', 'key label df col yscale quantum ci_scale ci_unit '
                  'stat_fn mouse_values contrast_payload',
    defaults=(None, None, None))


def _draw_superplot_panel_grid(panels, row_ylabels, col_titles, save_dir, filename_root,
                               panel_name, annotate='stats', figure_has_stars=True,
                               no_star_note=None, contrasts_title='Decomposition panel contrasts',
                               figsize_per_panel=(2.0, 3.0), contrasts_preamble=None):
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
            annotate=annotate, bracket_mode='axes', jitter_width=0.22,
            stat_fn=panel.stat_fn, mouse_means_override=panel.mouse_values)
        if r == 0:
            axs[r, c].set_title(col_titles[c], size='small', pad=26)
        contrasts_by_panel[panel.label] = (
            _panel_contrasts(panel.df, panel.col, panel.ci_scale, panel.ci_unit)
            if panel.contrast_payload is None else panel.contrast_payload)

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
                                           no_star_note=no_star_note, title=contrasts_title,
                                           preamble=contrasts_preamble)
    return contrasts_by_panel


_PAPER_CONTRASTS_PREAMBLE = (
    'Model-derived treatment-vs-control contrasts from the unified mouse-level mixed models '
    '(`log(metric) ~ group * epoch + (1|mouse)`, n = 5 hM3D / 6 hM4D / 6 mCherry animals, one '
    'value per animal per epoch). Every estimate, interval, P-value and asterisk below and on the '
    'accompanying figure is one row of `stats/unified_lmm_posthoc_contrasts.csv`; nothing here is '
    'computed by the panel.\n\n'
    'Amplitude ratios are ratios of GEOMETRIC means of the cell-level mean event-run integrals '
    '(the model response is the within-animal mean of cell-level log amplitudes). Rate ratios are '
    'accompanied by the observed absolute difference in events/s/cell, which is descriptive: a '
    'fold-change computed off a small base overstates the practical size of a change.\n\n'
    'Within each epoch, hM3D and hM4D are compared with mCherry and those two '
    'treatment-vs-control comparisons are Holm-corrected together (`p_holm_epoch`). The '
    'procedure is the same in every epoch and every panel. A conservative six-comparison Holm '
    'correction across all three epochs is retained as a SENSITIVITY analysis in `p_holm_six` of '
    '`stats/unified_lmm_posthoc_contrasts.csv`; it determines no annotation and no manuscript '
    'significance statement.\n\n'
    'A significant comparison within one epoch does not imply that the treatment effect differs '
    'from another epoch; epoch dependence was tested directly by the group x epoch interaction '
    '(`stats/unified_lmm_interactions.csv`).'
)

_PAPER_DISTRIBUTION_STAR_NOTE = (
    '**Every panel is annotated by the same procedure.** Within each epoch, hM3D and hM4D are '
    'compared with mCherry using model-derived post-hoc contrasts, Holm-corrected across those '
    'two treatment-versus-control comparisons (`p_holm_epoch`). Asterisks denote Holm-adjusted '
    'P < 0.05 (*), P < 0.01 (**), P < 0.001 (***); a panel with no bracket is one where neither '
    'adjusted P reached 0.05, not one that was exempted from testing. The presence of a star in '
    'one column and not another is not itself a test of any difference between columns — epoch '
    'dependence was assessed separately by the group x epoch joint Wald test.'
)


def _unified_contrast_payloads(contrasts, epochs=TFC_MATCHED_PROFILE_EPOCHS):
    """{(outcome key, epoch): payload} in the shape the panel/forest/markdown code already
    consumes, built from the unified model contrasts.

    ** This adapter is why there is no second forest and no second markdown writer. ** The
    existing drawing code speaks the dict shape mouse_contrast_ci returns; rather than duplicate
    that code to accept a tidy frame, the tidy frame is translated once, here. The fields are the
    model's: `diff` is the log-scale contrast, `ratio` its exponential, and the intervals are the
    t-based model intervals -- not Welch intervals on the per-mouse means.
    """
    epochs = tuple(epochs)
    payloads = {}
    for outcome in UNIFIED_OUTCOMES:
        for epoch in epochs:
            entry = {}
            for group in UNIFIED_TREATMENT_GROUPS:
                row = unified_contrast_lookup(contrasts, outcome.key, epoch, group)
                # 'p_holm' is what format_contrast_ci_lines renders as "Holm-adjusted P". It
                # carries the paper-facing within-epoch value on EVERY row, because every row is a
                # member of its own epoch's two-comparison family and they are all corrected the
                # same way. The six-way sensitivity value travels separately as 'p_holm_six' and
                # is never rendered as the decision.
                entry[group] = {
                    'diff': row['estimate_log'], 'diff_lo': row['ci_low_log'],
                    'diff_hi': row['ci_high_log'], 'ratio': row['ratio'],
                    'ratio_lo': row['ratio_ci_low'], 'ratio_hi': row['ratio_ci_high'],
                    'p_raw': row['p_raw'],
                    'p_holm': float(row['p_holm_epoch']),
                    'p_holm_epoch': float(row['p_holm_epoch']),
                    # Absent on the recall lane, which declares no across-epoch sensitivity
                    # family (see unified_posthoc_contrasts' across_epoch_family).
                    'p_holm_six': row.get('p_holm_six', np.nan),
                    # The contrast is on a log response, so the 'diff' above is in log units --
                    # the same statement mouse_contrast_ci(scale='log') makes about its output.
                    'unit_is_log': True, 'n': None, 'n_ref': None,
                }
            payloads[(outcome.key, epoch)] = {
                # 'log' tells format_contrast_ci_lines the difference is already on a log scale,
                # so it reports exp(difference) as the ratio -- which is exactly what these are.
                'scale': 'log', 'unit': '', 'contrasts': entry, 'exc_vs_inh': None}
    return payloads


def _assert_markers_match_model(mouse_epoch, marker_col, drawn, panel_key, epoch):
    """Hard-fail unless the values about to be drawn as a panel's large markers ARE that panel's
    rows of the unified mouse x epoch table.

    Validation that the figure and the model describe the same quantity, made mechanical rather
    than visual: the figure's whole claim is that its markers are the inferential dataset, and a
    weighting or scale slip between the cloud collapse and the model response is exactly the kind
    of error nobody spots by eye.
    """
    expected = mouse_epoch[mouse_epoch['epoch'] == epoch].set_index('mouse')[marker_col]
    for group, per_mouse in drawn.items():
        for mouse, value in per_mouse.items():
            if mouse not in expected.index:
                raise RuntimeError(f'_assert_markers_match_model [{panel_key}]: mouse {mouse!r} '
                                   f'is drawn but absent from the unified table.')
            if not np.isclose(value, float(expected.loc[mouse]), rtol=1e-9, atol=0.0):
                raise RuntimeError(
                    f'_assert_markers_match_model [{panel_key}]: mouse {mouse!r} ({group}) would '
                    f'be drawn at {value!r} but the model was fit on {marker_col} = '
                    f'{float(expected.loc[mouse])!r}. The marker must be the model\'s own value.')


def _paper_panel_star_pvalues(contrasts, outcome_key, epoch):
    """(hM3D-vs-Ctl, hM4D-vs-Ctl) p-values for one paper panel's asterisks -- that panel's own two
    within-epoch Holm-adjusted values (`p_holm_epoch`).

    ** This is the single place a paper asterisk is authorized, and it has no per-epoch branch. **
    Every panel of the 2 x 3 figure goes through the same lookup and the same correction, so
    visually equivalent panels are annotated by an identical procedure; whether a bracket is
    actually drawn is decided downstream by the adjusted P against alpha, not by which column the
    panel sits in. The six-way sensitivity correction (`p_holm_six`) can never reach a panel
    through here.
    """
    out = []
    for group in UNIFIED_TREATMENT_GROUPS:
        row = unified_contrast_lookup(contrasts, outcome_key, epoch, group)
        if not np.isfinite(row['p_holm_epoch']):
            raise RuntimeError(
                f'_paper_panel_star_pvalues: {outcome_key}/{epoch}/{group} has a non-finite '
                f'p_holm_epoch ({row["p_holm_epoch"]!r}). Every paper panel must carry both of '
                f'its epoch\'s treatment-vs-control Holm-adjusted P-values.')
        out.append(float(row['p_holm_epoch']))
    return tuple(out)


def plot_paper_epoch_distributions(df_matched, mouse_epoch, contrasts, save_dir,
                                   epochs=TFC_MATCHED_PROFILE_EPOCHS,
                                   rows=('amplitude', 'population_rate'),
                                   filename_root='tfc_amplitude_rate_by_epoch',
                                   epoch_labels=None, contrasts_preamble=None,
                                   contrasts_title='Paper figure contrasts — unified mixed models',
                                   no_star_note=None):
    """
    The paper-facing distribution figure: per-event amplitude (top) and population event rate
    (bottom) for each DREADD group, across the three exposure-matched TFC windows.

    ** Every number on this figure comes from the unified models. ** The large markers are the
    rows of `mouse_epoch` (build_mouse_epoch_unified_table) -- the exact values both LMMs are fit
    on, asserted panel by panel -- and the asterisks are the within-epoch Holm-adjusted contrasts
    from `contrasts` (unified_posthoc_contrasts), handed to the panels through
    _precomputed_stat_fn so no panel can compute a statistic of its own. The per-cell clouds are
    display: they show the within-animal distribution behind each marker and enter no calculation.

    ** All six panels are annotated identically. ** Each takes its own epoch's two
    treatment-vs-control contrasts, Holm-corrected within that epoch (_paper_panel_star_pvalues),
    and draws a bracket wherever the adjusted P clears alpha. There is no per-epoch branch: a
    panel without a bracket is one where neither comparison reached significance, not one that
    was exempted from testing. No `p_holm_six` value reaches this figure.

    ** Cells and markers share a scale. ** Both rows are drawn on the natural scale with a log
    axis; the amplitude marker is exp(mouse_mean_log_amplitude), i.e. that animal's geometric mean
    event amplitude, and the rate marker is total events over total cell-seconds. See
    _PAPER_ROW_DISPLAY.

    ** sharey='row' is the point of the figure, not tidiness. ** The figure exists to be read
    across its three columns. Independently autoscaled columns would let a reader take an epoch
    difference straight off the axis limits; one y-scale per row makes the comparison honest.
    Whether the treatment effect actually differs across epochs is not read off these columns at
    all -- it is the group x epoch interaction, reported once per outcome in
    stats/unified_lmm_interactions.csv. A significant comparison in one column and not another
    does not by itself establish an epoch difference.

    ** The input must be exposure-matched. ** Pass the frame restricted by
    restrict_to_exposure_matched_trials -- not the raw df_fine -- and the SAME frame `mouse_epoch`
    was built from. The rate row is duration-sensitive (P(active) = 1 - e^(-lambda*T) rises with T
    at a fixed underlying rate) and trial 1's trace window is 15 s rather than 20 s, so on
    unmatched trials part of the trace-vs-baseline difference in that row would be pure exposure.
    """
    epochs = tuple(epochs)
    # Defaults resolved here rather than in the signature so the recall lane can supply its own
    # column labels and its own provenance paragraph without either figure inheriting text that
    # describes the other one's epochs, cohort or stats filenames.
    epoch_labels = _PAPER_EPOCH_LABELS if epoch_labels is None else epoch_labels
    contrasts_preamble = (_PAPER_CONTRASTS_PREAMBLE if contrasts_preamble is None
                          else contrasts_preamble)
    no_star_note = _PAPER_DISTRIBUTION_STAR_NOTE if no_star_note is None else no_star_note
    specs = [_DECOMPOSITION_COMPONENTS_BY_KEY[k] for k in rows]
    payloads = _unified_contrast_payloads(contrasts, epochs=epochs)

    panels = {}
    for c, epoch in enumerate(epochs):
        frames = _decomposition_grid_frames(aggregate_over_trials(df_matched, epoch))
        for r, spec in enumerate(specs):
            display = _PAPER_ROW_DISPLAY[spec.key]
            sub = frames[spec.frame]
            col = display['col']
            # One count = one quantum of rate. exposure_seconds is the same matched window for
            # every cell within a trial, so the median is that window pooled over trials.
            quantum = (1.0 / float(np.median(sub['exposure_seconds']))
                       if display['quantum'] else None)
            marker_col = display['marker_col']
            epoch_rows = mouse_epoch[mouse_epoch['epoch'] == epoch]
            mouse_values = {
                g: dict(zip(epoch_rows.loc[epoch_rows['group'] == g, 'mouse'],
                            epoch_rows.loc[epoch_rows['group'] == g, marker_col]))
                for g in DREADD_DISPLAY_ORDER}
            _assert_markers_match_model(mouse_epoch, marker_col, mouse_values,
                                        f'{spec.key}/{epoch}', epoch)
            stars = _paper_panel_star_pvalues(contrasts, spec.key, epoch)
            panels[(r, c)] = _GridPanel(
                key=f'{spec.key}/{epoch}', label=f'{spec.label} — {epoch}', df=sub,
                col=col, yscale=display['yscale'], quantum=quantum,
                ci_scale=spec.ci_scale, ci_unit=spec.ci_unit,
                stat_fn=_precomputed_stat_fn(*stars),
                mouse_values=mouse_values,
                contrast_payload=payloads[(spec.key, epoch)])

    # Returned so the Results summary can quote the SAME payloads the panels were drawn from,
    # rather than recomputing the contrasts and risking a figure and its own summary disagreeing.
    return _draw_superplot_panel_grid(
        panels,
        row_ylabels=[_PAPER_ROW_DISPLAY[spec.key]['ylabel'] for spec in specs],
        col_titles=[epoch_labels.get(e, e) for e in epochs],
        save_dir=save_dir, filename_root=filename_root,
        panel_name='plot_paper_epoch_distributions', annotate='stats', figure_has_stars=True,
        contrasts_title=contrasts_title,
        contrasts_preamble=contrasts_preamble,
        no_star_note=no_star_note)


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
    'Fraction of cells active, event rate among ACTIVE cells, and total deconvolved '
    'amplitude-rate (a.u./s) -- the other terms of the decomposition identity '
    '(`decomposition*.png`, `decomposition_grid.png`). Fraction active has no unified '
    'mouse-level model and therefore no paper-facing panel: every number on a paper figure comes '
    'from one of the two unified models.',
    'The direct hM3D-vs-hM4D contrasts -- exploratory, uncorrected, in no unified multiplicity '
    'family (docs/sp_rates_lmm.md section 5.2). Describe the two groups as showing divergent '
    'profiles; do not report either as differing from control on this basis.',
    'Amplitude ECDF and per-mouse 90th percentile -- where in the distribution the effect sits '
    '(`amplitude_ecdf.png`, `amplitude_p90.png`).',
    'Event-detection threshold sensitivity at thres in {1.5, 2.0, 3.0} (`threshold_sensitivity.png`).',
    'Run-structure evidence: run width, local maxima per run, multi-peak fraction (`run_structure.png`).',
    'group x trial photobleaching control (`stats/group_trial_photobleaching.txt`).',
    'Early-vs-late post-shock within-cell contrast (Puhger et al. 2024 internal control) '
    '-- descriptive, spends no alpha.',
    'Cross-registration subset sensitivity, and the LT1->LT2 detection-dropout measurement.',
    'The older single-epoch Test_B / Test_B_1wk post-tone CELL-LEVEL amplitude omnibus '
    '(`Test_B*/post_tone_amplitude.*`, BH-FDR secondary family). This is NOT the paper-facing '
    'recall analysis -- that is the unified pre-tone/post-tone lane under '
    '`paper/recall/{Test_B,Test_B_1wk}/`, which has its own summary files.',
)


# ** The embedded statsmodels table is NOT the paper's inference. ** MixedLM's summary reports an
# asymptotic z test per fixed-effect coefficient (`P>|z|`), which uses neither the animal-level
# denominator df this analysis fixes at n_animals - 1 nor the contrast/joint-test structure the
# manuscript actually reports. A coefficient is also not a simple effect at a non-reference epoch
# (see unified_posthoc_contrasts). Both distinctions are stated at the one place a reader is most
# likely to conflate them: bracketing the raw summary, so neither half can be read out of context.
#
# ** Parameterized because the two lanes are not interchangeable. ** The conditioning lane's
# df = 16, its CSVs are `unified_lmm_*` and it carries a p_holm_six sensitivity column; the recall
# lane's df is that session's own n_animals_present - 1, its CSVs are `unified_recall_*` and it
# declares no across-epoch family. Emitting the conditioning text over a recall fit -- which is
# what a shared constant did -- pointed a reader at the wrong df and at files that do not exist in
# that directory.

# The non-reference-epoch sentence's tail, PRE-WRAPPED per lane so each file's line breaks are its
# own. The conditioning wording is byte-identical to what the shared constant emitted.
_TFC_NON_REFERENCE_EPOCH_PHRASE = 'at trace\nand post-shock'
_RECALL_NON_REFERENCE_EPOCH_PHRASE = 'at\npost-tone'


def _statsmodels_pvalue_note(df2, contrasts_csv, interactions_csv, holm_cols,
                             non_reference_epoch_phrase):
    """The header block bracketing a raw statsmodels summary in a paper-facing text file.

    df2                          - the animal-level denominator df this lane's inference uses,
                                   passed in from the fit rather than written as a literal.
    contrasts_csv,
    interactions_csv             - where THIS lane's manuscript P-values actually live.
    holm_cols                    - the adjusted-P column names present in that contrasts CSV.
    non_reference_epoch_phrase   - pre-wrapped naming of this lane's non-reference epoch(s).
    """
    return (
        '--- READ BEFORE THE TABLE BELOW ---------------------------------------------------------\n'
        "The coefficient table below is statsmodels' own output: its `P>|z|` column is an ASYMPTOTIC\n"
        'Z TEST of each fixed-effect coefficient. It is diagnostic only.\n'
        '\n'
        'The paper-facing inferential P-values are computed separately from this same fit:\n'
        '  - treatment-vs-control simple effects -> linear contrasts (linear_contrast_test),\n'
        '  - group x epoch epoch-dependence     -> joint Wald F tests (joint_wald_test),\n'
        f'both on the common animal-level denominator convention t / F with df = n_animals - 1 = {int(df2)}.\n'
        f'Read them from {contrasts_csv} ({", ".join(holm_cols)}) and\n'
        f'{interactions_csv}.\n'
        '\n'
        'DO NOT QUOTE A `P>|z|` VALUE FROM THE TABLE BELOW AS A MANUSCRIPT P-VALUE. Note also that a\n'
        f'group coefficient is the treatment-vs-control effect only AT THE REFERENCE EPOCH; '
        f'{non_reference_epoch_phrase} the simple effect is that coefficient plus the corresponding interaction\n'
        'coefficient, with their covariance.\n'
        '-----------------------------------------------------------------------------------------'
    )


def _statsmodels_pvalue_footer(df2):
    """The closing block of the same bracket. Same df convention, stated again at the point a
    reader leaving the table is most likely to carry a `P>|z|` value away with them."""
    return (
        '--- END OF STATSMODELS OUTPUT -----------------------------------------------------------\n'
        'The `P>|z|` column above is an asymptotic z test and is diagnostic only; the manuscript\n'
        f'P-values are the t(df={int(df2)}) contrasts and joint Wald F tests in the CSVs named above.\n'
        '-----------------------------------------------------------------------------------------'
    )


def write_paper_results_summary(save_dir, mouse_epoch, fits, contrasts, interactions_table,
                                primary_contrasts, holm, perm_results, interactions, q_by_name,
                                rate_contrasts, manip_contrasts,
                                filename='paper_results_summary.md'):
    """
    Every number the manuscript's Results paragraph needs, in one file, in two clearly separated
    parts.

    ** Part one is the analysis the paper reports. ** The two unified mixed models, all twelve
    treatment-vs-control contrasts with their within-epoch Holm-adjusted P-values
    (`p_holm_epoch`), the two group x epoch interaction tests, and the descriptive absolute
    population-rate differences. This is the only part any manuscript sentence should quote, and
    it is the only part any figure annotation comes from.

    ** No six-comparison Holm P appears anywhere in Part 1. ** The conservative across-epoch
    family lives under Part 2 with its own heading, so the paper-facing and the sensitivity
    decision rules are typographically separate and a six-way adjusted P cannot be lifted out of
    a Part 1 table and quoted as the reported result.

    ** Part two is everything else this module computes. ** The cell-level amplitude model and its
    three-member Holm family, the mouse-label permutation tests, the Bayesian negative-binomial
    rate model, the BH-FDR family, the manipulation check, the supplement list. All of it is real,
    kept, and useful as sensitivity evidence -- and none of it supplies a paper number. Keeping
    the separation typographic rather than implicit is the point: the recurring failure mode this
    module has documented is a secondary estimate being written up as the headline once it has
    been separated from the file that said otherwise.

    Every null is reported with its interval and what that interval still admits. At n = 5/6/6 a
    non-significant result is weak evidence of absence, and an interval reaching 1.55 has not
    excluded a +55% effect.
    """
    payloads = _unified_contrast_payloads(contrasts,
                                          epochs=tuple(mouse_epoch['epoch'].cat.categories))
    lines = [
        '# TFC cellular results — paper summary',
        '',
        'Per-event amplitude and population event rate by DREADD group across the three '
        'exposure-matched 20 s TFC windows (pre-tone baseline, trace, post-shock). '
        f'n = 5 hM3D / 6 hM4D / 6 mCherry animals; the inferential dataset is '
        f'{len(mouse_epoch)} rows — one value per animal per epoch.',
        '',
        '**Part 1 is the analysis the paper reports. Part 2 is sensitivity and supplementary '
        'output and supplies no number in any manuscript sentence or on any paper figure.** The '
        'full rationale and decision record live in `docs/sp_rates_lmm.md`.',
        '',
        '# Part 1 — paper-facing analysis',
        '',
        'The same group x epoch mixed-effects model is fit for each outcome. Within each epoch, '
        'hM3D and hM4D are compared with mCherry using model-derived post-hoc contrasts, with '
        'Holm correction across the two treatment-versus-control comparisons in that epoch. The '
        'group x epoch interaction tests whether the magnitude of the treatment effect differs '
        'across epochs.',
        '',
        '## Unified mixed models',
        '',
        'Both outcomes are analysed with the SAME mouse-level linear mixed-effects model: DREADD '
        'group, epoch and their interaction as fixed effects, animal as a random intercept, '
        'reference levels mCherry and pre-tone. Amplitude is summarised per animal as the mean '
        'over its active cells of log(cell mean event-run integral), so an exponentiated contrast '
        'is a ratio of GEOMETRIC means. Population rate is total events over total cell-seconds '
        'across all detected cells (zero-event cells included), log-transformed.',
        '',
    ]
    for outcome in UNIFIED_OUTCOMES:
        fit = fits[outcome.key]
        lines.append(f"- **{outcome.label}**: `{fit['formula']} + (1|mouse)` "
                     f"({fit['method']}, {fit['n_mice']} animals)")
    lines += [
        '',
        '## Treatment-vs-control contrasts by epoch',
        '',
        'All twelve treatment-vs-control simple effects — 3 epochs x {hM3D, hM4D} vs mCherry, per '
        'outcome. Each is a linear contrast of the same full group x epoch model (the group '
        'coefficient alone at the reference epoch; that coefficient plus the corresponding '
        'group x epoch coefficient elsewhere, with their covariance), on an animal-level '
        f'denominator df = n_animals - 1 = {int(interactions_table["df2"].iloc[0])} used '
        'consistently for the joint Wald and contrast inference.',
        '',
        'Within each epoch, the two treatment-vs-control comparisons are Holm-corrected together '
        '(`p_holm_epoch`) — six two-member families, the same procedure in every epoch and both '
        'outcomes. hM3D-vs-hM4D is in no family and is not computed here.',
        '',
        '**These are the only P-values that generate a figure asterisk or a manuscript '
        'significance statement.**',
        '',
        '| outcome | epoch | comparison | ratio | 95% CI | raw P | Holm-adjusted P (within epoch) |',
        '|---|---|---|---|---|---|---|',
    ]
    for _, row in contrasts.iterrows():
        flag = ' (significant)' if row['holm_epoch_reject'] else ''
        lines.append(
            f"| {UNIFIED_OUTCOMES_BY_KEY[row['outcome']].label} | {row['epoch']} | "
            f"{GROUP_LABELS.get(row['group'], row['group'])} vs Ctl | {row['ratio']:.3f} | "
            f"[{row['ratio_ci_low']:.3f}, {row['ratio_ci_high']:.3f}] | {row['p_raw']:.4g} | "
            f"**{row['p_holm_epoch']:.4g}**{flag} |")
    lines += [
        '',
        '## Group x epoch interaction — does the treatment effect differ across epochs?',
        '',
        'Joint Wald test that all four group x epoch coefficients are zero, one per outcome. '
        'This is the paper\'s single test of epoch dependence. A significant treatment-control '
        'comparison within one epoch does not imply that the treatment effect differs from '
        'another epoch.',
        '',
        '| outcome | F | df1 | df2 | P |',
        '|---|---|---|---|---|',
    ]
    for _, row in interactions_table.iterrows():
        lines.append(f"| {UNIFIED_OUTCOMES_BY_KEY[row['outcome']].label} | {row['F']:.3f} | "
                     f"{int(row['df1'])} | {int(row['df2'])} | {row['p']:.4g} |")
    lines += [
        '',
        'A non-significant interaction is **no evidence that the treatment effect differed across '
        'the pre-tone, trace and post-shock epochs**. It is not evidence that the effect is '
        'identical, global, tonic, or equivalent across them.',
        '',
        '## Absolute population-rate differences',
        '',
    ]
    rate_rows = contrasts[contrasts['outcome'] == 'population_rate']
    lines += [
        'The population-rate effects above, as observed absolute differences in events/s per '
        'cell (DESCRIPTIVE — equal-mouse-weighted observed means, no test; a rate ratio off a '
        'small base overstates the practical size of a change):',
        '',
    ]
    for _, row in rate_rows.iterrows():
        lines.append(
            f"- {row['epoch']}, {GROUP_LABELS.get(row['group'], row['group'])} vs Ctl: "
            f"{row['absolute_difference_events_per_s_per_cell']:+.4f} events/s/cell "
            f"({row['mean_population_rate_treatment']:.4f} vs "
            f"{row['mean_population_rate_control']:.4f})")
    lines += [
        '',
        '## The same contrasts as the figures render them',
        '',
        'The Holm-adjusted P on every line is that row\'s within-epoch value (`p_holm_epoch`) — '
        'the same number the corresponding figure panel is annotated from.',
        '',
    ]
    for outcome in UNIFIED_OUTCOMES:
        for epoch in mouse_epoch['epoch'].cat.categories:
            lines.append(f'**{outcome.label} — {epoch}**')
            lines.append('')
            lines += [f'- {ln}' if not ln.startswith('  ') else f'  - {ln.strip()}'
                      for ln in format_contrast_ci_lines(
                          payloads[(outcome.key, epoch)]['contrasts'], 'mCherry')]
            lines.append('')

    lines += [
        '# Part 2 — sensitivity and supplementary analyses',
        '',
        '**Nothing below is a paper-facing result.** These analyses are retained in full because '
        'they test whether the conclusions above survive different modelling choices, weightings '
        'and distributional assumptions. None of them supplies an asterisk, an interval or a '
        'P-value to any manuscript sentence or paper figure, and none should be reported '
        'alongside the Part 1 numbers as though it were an alternative primary result.',
        '',
        '## Sensitivity — conservative six-comparison across-epoch Holm correction',
        '',
        'The same twelve model contrasts and the same raw P-values as Part 1, corrected more '
        'conservatively: all six treatment-vs-control simple effects spanning the three epochs '
        'are Holm-corrected together within each outcome (`p_holm_six` in '
        '`unified_lmm_posthoc_contrasts.csv`), rather than two at a time within each epoch. It '
        'is a wider definition of the inferential family, retained so the choice stays auditable. '
        '**These adjusted P-values determine no figure annotation and no manuscript significance '
        'statement.**',
        '',
        '| outcome | epoch | comparison | raw P | six-comparison Holm P |',
        '|---|---|---|---|---|',
    ]
    for _, row in contrasts.iterrows():
        flag = ' (significant)' if row['holm_six_reject'] else ''
        lines.append(
            f"| {UNIFIED_OUTCOMES_BY_KEY[row['outcome']].label} | {row['epoch']} | "
            f"{GROUP_LABELS.get(row['group'], row['group'])} vs Ctl | {row['p_raw']:.4g} | "
            f"{row['p_holm_six']:.4g}{flag} |")
    lines += [
        '',
        '## Sensitivity — cell-level trace-period amplitude model (historical primary)',
        '',
        f"`log_amplitude ~ group + (1|mouse)` at the CELL level, trials pooled, 35 s pre-tone "
        f"reference: p = {holm['trace_amplitude']['p_raw']:.4g}, Holm-corrected p = "
        f"{holm['trace_amplitude']['p_holm']:.4g} across its own {len(holm)}-member family. This "
        f"was the primary analysis before the unified models; it is kept as a sensitivity check "
        f"on the mouse-level amplitude result and uses a different epoch definition and a "
        f"different unit of aggregation.",
        '',
        'Equal-mouse-weighted Welch contrasts vs mCherry on the same endpoint:',
        '',
    ]
    lines += [f'- {ln}' if not ln.startswith('  ') else f'  - {ln.strip()}'
              for ln in format_contrast_ci_lines(primary_contrasts, 'mCherry')]
    perm_lines = [f"- {k.replace('_', ' ')}: p = {v['p_two_sided']:.4g} ({v['n_perm']} draws)"
                  for k, v in perm_results.items() if k.endswith('_mean_mouseweighted')]
    if perm_lines:
        lines += ['', 'Mouse-label permutation test on that same cell-level contrast '
                      '(distribution-free sensitivity check):', ''] + perm_lines

    lines += [
        '',
        '## Sensitivity — permutation group x epoch tests',
        '',
        'A mouse-label permutation interaction statistic per component, BH-corrected within the '
        'secondary family. It answers the same question as the unified joint Wald tests in Part 1 '
        'without a distributional assumption. The Part 1 tests are the reported ones.',
        '',
    ]
    for key, res in interactions.items():
        q = q_by_name.get(f'epoch_specificity_{key}')
        q_txt = '' if q is None else f', q = {q:.3f}'
        label = _DECOMPOSITION_COMPONENTS_BY_KEY[key].label
        lines.append(f"- {label}: p = {res['p_two_sided']:.3f}{q_txt} "
                     f"({res['n_cells']} cells, {res['n_mice']} mice)")

    lines += [
        '',
        '## Sensitivity — negative-binomial count model of event rate',
        '',
        'Counts at the mouse x trial x epoch level with a `log(total cell-seconds)` exposure '
        'offset, `(1|mouse) + (1|mouse:trial)`, dispersion estimated jointly, fit by MCMC. This '
        'is the distribution-aware check on the Part 1 rate model, which treats log(population '
        'rate) as Gaussian. It reports posterior intervals, not p-values; **no highest-density '
        'interval here generates a figure asterisk or a Results claim.**',
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
    The paper lane: the two unified models, two figures, and one Results-ready numbers file, under
    PLOTS_DIR/sp_rates_lmm/paper/tfc_amplitude_rate/.

    ** This is where the paper-facing analysis is FIT. ** Everything the manuscript reports comes
    from the two mouse-level models built here: the mouse x epoch inferential dataset
    (build_mouse_epoch_unified_table), one `group * epoch + (1|mouse)` fit per outcome, one joint
    Wald interaction test per outcome, and twelve treatment-vs-control contrasts Holm-corrected
    two at a time WITHIN each epoch (`p_holm_epoch`; the six-comparison across-epoch correction is
    retained beside them as `p_holm_six` sensitivity output). Both figures then read their
    markers, intervals and asterisks straight out of those two tables and compute nothing
    themselves; every panel is annotated by the same procedure.

    The TFC_cond lane's own analyses are untouched, still run, and still write every file they
    always did -- they are the sensitivity/internal record. Their objects are passed in here only
    so the summary file can report them under its clearly-separated Part 2.

    ** Must be called after build_secondary_fdr_table. ** Part 2 of the summary quotes the
    BH-adjusted q-values of the permutation epoch tests, which do not exist until the whole
    secondary family has been fit.

    df_matched : df_fine restricted to exposure-matched (mouse, trial) pairs. Both outcomes are
                 derived from this one frame, so the two figure rows provably describe the same
                 animals, epochs and trials. Trial 1's trace window is 15 s rather than 20 s, and
                 the rate row is duration-sensitive, so the columns are only comparable on
                 matched trials.
    """
    epochs = tuple(epochs)
    paper_dir = os.path.join(PLOTS_DIR, 'sp_rates_lmm', 'paper', 'tfc_amplitude_rate')
    paper_stats_dir = os.path.join(paper_dir, 'stats')
    ensure_dirs(paper_dir, paper_stats_dir)
    # Both templates land here: the short paper-facing one, and (via _save_panel) the full
    # internal METHODS. The paper text is what goes in the manuscript; the internal one is what
    # answers a reviewer who asks why a window is 20 s.
    _copy_analysis_methods_template(PAPER_METHODS_FILENAME, paper_dir)

    # ---- The unified models -----------------------------------------------------------------
    print('[sp_rates_lmm] Unified paper models: mouse-level amplitude + population rate...')
    verify_unified_synthetic(paper_stats_dir, epochs=epochs)
    df_paper = df_matched[df_matched['epoch'].isin(epochs)]
    mouse_epoch = build_mouse_epoch_unified_table(df_paper, epochs=epochs)
    write_text(os.path.join(paper_stats_dir, 'unified_lmm_mouse_epoch_values.csv'),
               mouse_epoch.to_csv(index=False))

    fits = {o.key: fit_unified_group_epoch_model(mouse_epoch, o.response_col)
            for o in UNIFIED_OUTCOMES}
    require_common_unified_method(fits)
    contrasts = unified_posthoc_contrasts(fits, mouse_epoch, epochs=epochs)
    interactions_table = unified_interactions_table(fits)

    # One df for both outcomes -- require_common_unified_method above has already established that
    # the two fits are the same procedure on the same animals, so a per-outcome df would be a
    # distinction without a difference. Read off the fit, never written as a literal.
    tfc_df2 = fits[UNIFIED_OUTCOMES[0].key]['n_mice'] - 1
    tfc_pvalue_note = _statsmodels_pvalue_note(
        tfc_df2, 'unified_lmm_posthoc_contrasts.csv', 'unified_lmm_interactions.csv',
        ('p_raw', 'p_holm_epoch', 'p_holm_six'), _TFC_NON_REFERENCE_EPOCH_PHRASE)
    for outcome in UNIFIED_OUTCOMES:
        write_text(os.path.join(paper_stats_dir, f'unified_lmm_{outcome.key}_summary.txt'),
                   f"PAPER-FACING model: {outcome.label}\n"
                   f"Response: {outcome.response_col} (one value per animal per epoch, "
                   f"{len(mouse_epoch)} rows, {fits[outcome.key]['n_mice']} animals)\n"
                   f"Group x epoch joint Wald: {fits[outcome.key]['omnibus']}\n"
                   f"Contrasts, within-epoch Holm-adjusted p-values (p_holm_epoch) and the "
                   f"six-comparison across-epoch sensitivity correction (p_holm_six): "
                   f"unified_lmm_posthoc_contrasts.csv\n\n"
                   f"{tfc_pvalue_note}\n\n"
                   f"{fits[outcome.key]['summary_text']}\n\n"
                   f"{_statsmodels_pvalue_footer(tfc_df2)}\n")
    write_text(os.path.join(paper_stats_dir, 'unified_lmm_interactions.csv'),
               interactions_table.to_csv(index=False))
    write_text(os.path.join(paper_stats_dir, 'unified_lmm_posthoc_contrasts.csv'),
               contrasts.to_csv(index=False))
    for _, row in interactions_table.iterrows():
        print(f"[sp_rates_lmm]   {row['outcome']}: {format_unified_interaction(row)}")

    unified_model_diagnostics(fits, mouse_epoch, contrasts, paper_stats_dir, epochs=epochs)

    # ---- Figures, annotated entirely from the tables above ------------------------------------
    print('[sp_rates_lmm] Paper figures: amplitude + rate across the matched TFC windows...')
    plot_paper_epoch_distributions(df_paper, mouse_epoch, contrasts, paper_dir, epochs=epochs)

    # The same forest drawing code the internal lane uses, given the UNIFIED contrasts instead of
    # its own Welch ones -- so the forest's points and intervals are the same rows of
    # unified_lmm_posthoc_contrasts.csv that the distribution figure's asterisks come from.
    interaction_notes = {row['outcome']: format_unified_interaction(row)
                         for _, row in interactions_table.iterrows()}
    plot_decomposition_grid(
        df_paper, paper_dir, epochs=epochs,
        components=[_DECOMPOSITION_COMPONENTS_BY_KEY[k] for k in PAPER_COMPONENT_KEYS],
        include_exc_vs_inh=False, filename_root='tfc_decomposition_forest',
        contrast_payloads=_unified_contrast_payloads(contrasts, epochs=epochs),
        interaction_note={k: interaction_notes[k] for k in PAPER_COMPONENT_KEYS},
        subtitle='Mixed-model contrasts with 95% CI; per-row group x epoch joint Wald test',
        contrasts_note=_PAPER_CONTRASTS_PREAMBLE, row_height=2.4,
        # Every column here is one of the three COMPLETE 20 s windows; post_shock_late, the one
        # window with reduced trial coverage, is not among them.
        reduced_coverage_epochs=(), epoch_labels=_PAPER_EPOCH_LABELS)

    # ---- Sensitivity output reported under Part 2 of the summary -------------------------------
    rate_contrasts = summarize_rate_group_epoch_contrasts(rate_fit)
    write_text(os.path.join(paper_stats_dir, 'rate_group_epoch_contrasts.csv'),
               rate_contrasts.to_csv(index=False))

    manip_contrasts = mouse_contrast_ci(
        _mouse_values_per_group(delta_df, 'delta_log_amplitude',
                                panel_name='paper_manipulation_check', group_order=GROUP_ORDER),
        scale='log')

    write_paper_results_summary(paper_stats_dir, mouse_epoch, fits, contrasts, interactions_table,
                                primary_contrasts, holm, perm_results, interactions, q_by_name,
                                rate_contrasts, manip_contrasts)
    print(f'[sp_rates_lmm] Paper figures written to {paper_dir}')
    return {'mouse_epoch': mouse_epoch, 'fits': fits, 'contrasts': contrasts,
            'interactions': interactions_table}


# ─────────────────────────────────────────────────────────────────────────────
# The paper-facing RECALL lane: Test_B and Test_B_1wk, each analysed on its own
# ─────────────────────────────────────────────────────────────────────────────
#
# ** Same statistical logic as the conditioning analysis, applied to retrieval. ** One mouse-level
# value per animal per epoch, the same `log(metric) ~ group * epoch + (1|mouse)` for both
# outcomes, the same within-epoch Holm family, the same model-derived contrasts, the same joint
# Wald interaction test -- so the manuscript can describe conditioning and recall in one sentence
# about method. What differs is only the epoch set (a 20 s pre-tone baseline and the 20 s
# post-tone retrieval window) and the cohort (each recall session is missing one animal).
#
# ** The two recall sessions are NEVER compared with each other here. ** See
# render_paper_recall_amplitude_rate.

RECALL_PAPER_METHODS_FILENAME = 'sp_rates_lmm_recall_paper_methods.md'

_RECALL_EPOCH_LABELS = {
    RECALL_REFERENCE_EPOCH: 'Pre-tone\n(20 s baseline)',
    RECALL_RESPONSE_EPOCH: 'Post-tone\n(20 s retrieval)',
}

# Output filename stem per recall session, so both sessions' files are self-identifying if they
# are ever copied out of their directories.
_RECALL_FILE_PREFIX = {'Test_B': 'testb', 'Test_B_1wk': 'testb_1wk'}

# Which recall sessions get the pre->post MODULATION decomposition (the per-mouse change scores,
# their three pairwise between-group contrasts and the two new figures). BOTH recall sessions, each
# decomposed entirely within itself: every number comes from that session's own two fits, its own
# cohort and its own denominator df, and nothing here compares the two timepoints -- see
# render_paper_recall_amplitude_rate on why that comparison is not available in this lane. It is a
# constant rather than an argument because a per-call flag invites two sessions being rendered
# under different rules by accident.
RECALL_MODULATION_SESSIONS = ('Test_B', 'Test_B_1wk')

# The interpretation rules from the recall plan, stated literally. The point of routing this
# through a lookup rather than prose written once per run is that the reading of a result is FIXED
# in advance and cannot drift toward whichever sentence the observed numbers would flatter.
#
# ** The simple effects and the interaction are read SEPARATELY, and only the simple effects are
# per-group. ** The group x epoch test is a JOINT 2-df Wald test over both treatment groups'
# interaction coefficients; it says whether the treatment-vs-control differences changed from
# pre-tone to post-tone, and it does NOT localize that change to one group. Attaching its verdict
# to an individual group's line would be a claim the test does not make -- so the per-group
# sentences below describe only that group's two within-epoch comparisons, and the interaction
# gets its own sentence per outcome.
#
# Key is (pre-tone significant, post-tone significant).
_RECALL_SIMPLE_EFFECT_READING = {
    (False, True):
        'differed from control during the post-tone retrieval window but not during the pre-tone '
        'baseline',
    (True, True):
        'differed from control during both the pre-tone baseline and the post-tone retrieval '
        'window',
    (True, False):
        'differed from control during the pre-tone baseline but not during the post-tone '
        'retrieval window (a difference in significance is not a significant difference)',
    (False, False):
        'was not detectably different from control in either window — at this n a null is weak '
        'evidence of absence, so read the intervals above',
}

# Read once per outcome, from the joint test alone.
_RECALL_INTERACTION_READING = {
    False:
        'The group x epoch interaction was not significant: **there is no evidence that the '
        'treatment-vs-control differences changed from pre-tone to post-tone.** Any difference '
        'seen in the post-tone window is therefore not established as retrieval-evoked; it may '
        'reflect a group difference already present around recall. This is not evidence that the '
        'effect is identical, global or tonic across the two windows.',
    True:
        'The group x epoch interaction was significant: the treatment-vs-control differences '
        'changed between the pre-tone and post-tone windows. Because the test is a joint 2-df '
        'test over both treatment groups, it establishes that at least one treatment group\'s '
        'effect differed between windows; it does not by itself attribute that change to a '
        'particular group.',
}


def _recall_contrasts_preamble(session_label, n_by_group, stats_prefix):
    """The provenance paragraph for a recall figure's companion contrasts file.

    Written per session because it states that session's OWN cohort: the conditioning lane's
    'n = 5 hM3D / 6 hM4D / 6 mCherry' is wrong for both recall sessions, and each recall session
    is missing a different animal, so neither can inherit the other's either.
    """
    cohort = ' / '.join(f'{n} {GROUP_LABELS.get(g, g)}' for g, n in n_by_group.items())
    return (
        f'{session_label}. Model-derived treatment-vs-control contrasts from the unified '
        f'mouse-level mixed models (`log(metric) ~ group * epoch + (1|mouse)`, n = {cohort} '
        f'animals present in this session, one value per animal per epoch). Every estimate, '
        f'interval, P-value and asterisk below and on the accompanying figure is one row of '
        f'`stats/{stats_prefix}_posthoc_contrasts.csv`; nothing here is computed by the panel.'
        '\n\n'
        'Epochs are two duration-matched 20 s windows around each tone: the baseline ending at '
        'tone onset, and the retrieval window beginning at tone offset. Analyses are restricted '
        'within each animal to tone trials on which BOTH windows are completely observed '
        f'(`stats/{stats_prefix}_trial_coverage.csv`); amplitude and rate use that identical '
        'trial set.\n\n'
        'Amplitude ratios are ratios of GEOMETRIC means of the cell-level mean event-run '
        'integrals (the model response is the within-animal mean of cell-level log amplitudes). '
        'Rate ratios are accompanied by the observed absolute difference in events/s/cell, which '
        'is descriptive: a fold-change computed off a small base overstates the practical size of '
        'a change.\n\n'
        'Within each epoch, hM3D and hM4D are compared with mCherry and those two '
        'treatment-vs-control comparisons are Holm-corrected together (`p_holm_epoch`). The '
        'procedure is the same in both epochs and every panel.\n\n'
        'A significant comparison in the post-tone window does not by itself establish that the '
        'effect is retrieval-evoked; that is the group x epoch interaction '
        f'(`stats/{stats_prefix}_interactions.csv`) and nothing else.\n\n'
        '**No CNO was present at recall.** A group difference here is a persistent consequence of '
        'the conditioning-day manipulation, not evidence of ongoing receptor activation.')


def build_recall_trial_coverage(session_key, df_recall_fine, df_matched):
    """Per-animal tone-trial coverage for one recall session: how many tone trials carried BOTH
    complete 20 s windows, and which ones.

    Written to a CSV rather than left implicit because the restriction is real and animal-specific
    -- a recording that stops shortly after the last tone offset loses that trial's post-tone
    window entirely -- and a reader has to be able to see how much of each animal's session
    survived it. `restrict_to_exposure_matched_trials` already reports the counts; this attaches
    the group and the retained trial indices so the file stands on its own.
    """
    total = (df_recall_fine.groupby(['mouse', 'group'], observed=True)['trial'].nunique()
             .rename('n_tone_trials_present'))
    kept = (df_matched.groupby(['mouse', 'group'], observed=True)['trial']
            .agg(n_trials_retained='nunique',
                 trials_retained=lambda s: ','.join(str(t) for t in sorted(set(s)))))
    out = pd.concat([total, kept], axis=1).reset_index()
    out.insert(0, 'session', session_key)
    if out['n_trials_retained'].isna().any():
        missing = out.loc[out['n_trials_retained'].isna(), 'mouse'].tolist()
        raise RuntimeError(
            f'build_recall_trial_coverage: animal(s) {missing} in {session_key} have no tone '
            f'trial with BOTH complete 20 s windows, so they cannot contribute a pre/post pair. '
            f'The recall model needs a complete animal x epoch grid; investigate the recording '
            f'rather than dropping the animal silently.')
    return out


# ── Pre->post MODULATION: a representation and pairwise decomposition of the interaction ─────
#
# ** This adds no model. ** Every number below is a linear contrast of the two models already fit
# by fit_unified_group_epoch_model, read through the same linear_contrast_test on the same
# animal-level df. The group x epoch joint Wald test remains the omnibus question -- does the
# pre->post change differ among the three groups -- and these contrasts are its pairwise
# decomposition, reported beside it and never in place of it.
#
# ** Symmetric across groups and across outcomes. ** All three pairwise comparisons of the change
# are computed for both outcomes, Holm-corrected within an outcome, and drawn by one procedure. No
# group is a headline and no raw P is privileged; a decomposition that singled one group out would
# be a different (and unstated) inferential structure.
#
# ** Three questions that this lane must keep apart. ** (A) Is there a group difference WITHIN an
# epoch -- the existing unified_posthoc_contrasts. (B) Does a group change from pre to post at all
# -- the within-group estimates below, DESCRIPTIVE, and not a treatment comparison. (C) Does the
# pre->post change DIFFER between two groups -- the pairwise contrasts below, which is the
# interaction. (A) and (C) can disagree in both directions (equal post-tone values over unequal
# baselines is a large modulation difference and no absolute difference), which is why they are
# tabulated separately and why the synthetic verification plants exactly that case.

# The pairwise comparisons of the pre->post change, as (group_a, group_b) with the estimate signed
# a - b. All three, in one place, so the contrast table, the Holm family and the figure's brackets
# cannot disagree about what the set of comparisons is. ALL THREE are always computed and
# tabulated, for both outcomes -- what the figure draws is a separate, narrower question below.
RECALL_MODULATION_PAIRS = (('hM3D', 'mCherry'), ('hM4D', 'mCherry'), ('hM3D', 'hM4D'))

# Which of those comparisons the modulation FIGURE brackets, per outcome, and which of them are
# bracketed with their p-value even when it is >= 0.05 (annotate_pairwise_brackets' ns_label_pairs).
# The brackets show the UNADJUSTED model-derived contrast p-values; hM4D-vs-mCherry is left off the
# figure entirely, and every comparison stays in the contrast table regardless. Drawing is a
# display choice; nothing here changes a computed value or which comparisons exist.
RECALL_MODULATION_FIGURE_PAIRS = (('hM3D', 'mCherry'), ('hM3D', 'hM4D'))
RECALL_MODULATION_NS_LABEL_PAIRS = {'amplitude': (('hM3D', 'hM4D'),), 'population_rate': ()}


def build_recall_modulation_by_mouse(mouse_epoch, reference_epoch=RECALL_REFERENCE_EPOCH,
                                     response_epoch=RECALL_RESPONSE_EPOCH):
    """
    One row per (animal, outcome): that animal's pre->post CHANGE on the model's own log scale.

    ** Derived from the inferential table itself, not recomputed from the event data. ** The
    values pivoted here are literally the rows the models were fit on
    (build_mouse_epoch_unified_table / `<prefix>_mouse_epoch_values.csv`), so the change scores a
    reader sees plotted are differences of the numbers the model saw, and there is no second
    aggregation path that could weight a cell or a trial differently.

    ** Both outcomes come off the one frame. ** Same reasoning as build_mouse_epoch_unified_table:
    the amplitude and rate panels of the modulation figure are read against each other, so they
    must describe the same animals and the same retained trials by construction.

    `delta_log` is post - pre in log units; `fold_change_post_vs_pre` is its exponential -- 1.0 is
    no change, >1 an increase, <1 a decrease. For amplitude that fold change is a ratio of the
    animal's GEOMETRIC mean event amplitudes (the response is a mean of cell-level logs); for rate
    it is a ratio of population event rates.

    Returns a DataFrame with columns
        mouse, group, outcome, response_col, pre_tone_value_log, post_tone_value_log,
        delta_log, fold_change_post_vs_pre
    """
    rows = []
    for outcome in UNIFIED_OUTCOMES:
        for (mouse, group), sub in mouse_epoch.groupby(['mouse', 'group'], observed=True):
            pre = sub[sub['epoch'] == reference_epoch]
            post = sub[sub['epoch'] == response_epoch]
            if len(pre) != 1 or len(post) != 1:
                raise RuntimeError(
                    f'build_recall_modulation_by_mouse: animal {mouse!r} ({group}) has '
                    f'{len(pre)} {reference_epoch!r} and {len(post)} {response_epoch!r} rows; a '
                    f'change score needs exactly one of each. The inferential table is supposed '
                    f'to be a complete animal x epoch grid.')
            pre_value = float(pre.iloc[0][outcome.response_col])
            post_value = float(post.iloc[0][outcome.response_col])
            delta = post_value - pre_value
            if not np.isfinite(delta):
                raise RuntimeError(
                    f'build_recall_modulation_by_mouse: animal {mouse!r} ({group}) has a '
                    f'non-finite {outcome.key} change ({post_value} - {pre_value}).')
            rows.append({
                'mouse': mouse, 'group': group, 'outcome': outcome.key,
                'response_col': outcome.response_col,
                'pre_tone_value_log': pre_value, 'post_tone_value_log': post_value,
                'delta_log': delta,
                'fold_change_post_vs_pre': float(np.exp(delta)),
            })

    out = pd.DataFrame(rows)
    # The two arithmetic identities the whole table rests on, checked rather than trusted: a sign
    # slip in either would flip the direction of every panel and every contrast reading.
    if not np.allclose(out['delta_log'],
                       out['post_tone_value_log'] - out['pre_tone_value_log'],
                       rtol=0.0, atol=1e-12):
        raise RuntimeError('build_recall_modulation_by_mouse: delta_log is not post - pre.')
    if not np.allclose(out['fold_change_post_vs_pre'], np.exp(out['delta_log']),
                       rtol=1e-12, atol=0.0):
        raise RuntimeError('build_recall_modulation_by_mouse: fold_change_post_vs_pre is not '
                           'exp(delta_log).')
    n_mice = int(mouse_epoch['mouse'].nunique())
    if len(out) != n_mice * len(UNIFIED_OUTCOMES):
        raise RuntimeError(f'build_recall_modulation_by_mouse: got {len(out)} rows for {n_mice} '
                           f'animals x {len(UNIFIED_OUTCOMES)} outcomes.')
    out['group'] = pd.Categorical(out['group'], categories=list(DREADD_DISPLAY_ORDER))
    return out.sort_values(['outcome', 'group', 'mouse']).reset_index(drop=True)


def recall_modulation_contrasts(fits, reference_epoch=RECALL_REFERENCE_EPOCH,
                                response_epoch=RECALL_RESPONSE_EPOCH, alpha=0.05):
    """
    The pre->post change per group, and the three pairwise comparisons of that change, for BOTH
    outcomes -- all as linear contrasts of the already-fitted models.

    ** Within-group changes are DESCRIPTIVE. ** In a treatment-reference parameterization the
    control group's pre->post change IS the epoch main coefficient, and a treatment group's is
    that coefficient plus its own group x epoch coefficient (a contrast, with covariance -- not
    something readable off the summary table). They characterize the trajectory and explain the
    shape of the interaction. They are not the treatment comparison: "hM3D changed and mCherry
    did not" is not a test that the two changed differently.

    ** The pairwise comparisons of the change ARE the interaction. ** Against the reference group
    each is a single group x epoch coefficient; between the two treatment groups it is the
    DIFFERENCE of their two interaction coefficients, which needs their covariance and is
    therefore a contrast rather than a subtraction of two published standard errors. All three are
    asserted below to equal exactly those linear combinations.

    ** `p_raw` -- the unadjusted model-derived contrast p-value -- is what the modulation figure's
    brackets and the drafted wording read. ** A three-comparison Holm adjustment WITHIN one
    outcome is also computed and reported beside it (`p_holm_modulation`,
    `holm_modulation_reject`; two independent three-member families, one per outcome, the same
    shape as unified_posthoc_contrasts' rule that amplitude and rate never share a correction).
    It is retained as a MULTIPLICITY REFERENCE for auditability -- it is not the governing
    decision rule for this lane and drives no annotation. Both columns keep their names and
    values; every text that quotes one says which it is.

    Returns (within_group_df, pairwise_df).
    """
    within_rows, pairwise_rows = [], []
    for outcome in UNIFIED_OUTCOMES:
        fit = fits[outcome.key]
        fe_names, result, n_mice = fit['fe_names'], fit['result'], fit['n_mice']
        epoch_coef = _unified_epoch_coef(fe_names, response_epoch)
        inter = {g: _unified_epoch_interaction_coef(fe_names, g, response_epoch)
                 for g in UNIFIED_TREATMENT_GROUPS}

        for group in DREADD_DISPLAY_ORDER:
            weights = {epoch_coef: 1.0}
            if group != UNIFIED_REFERENCE_GROUP:
                weights[inter[group]] = 1.0
            res = linear_contrast_test(result, weights, n_mice, alpha=alpha)
            within_rows.append({
                'block': 'within_group', 'outcome': outcome.key, 'group': group,
                'delta_log': res['estimate'], 'se': res['se'],
                'ci_low_log': res['ci_low'], 'ci_high_log': res['ci_high'],
                'post_pre_ratio': float(np.exp(res['estimate'])),
                'ratio_ci_low': float(np.exp(res['ci_low'])),
                'ratio_ci_high': float(np.exp(res['ci_high'])),
                't': res['t'], 'df': res['df'], 'p_raw': res['p'],
            })

        for group_a, group_b in RECALL_MODULATION_PAIRS:
            # Signed a - b. A group's own "interaction coefficient" is 0 for the reference group,
            # which is exactly why the vs-control comparisons come out as a single coefficient.
            weights = {}
            for group, sign in ((group_a, 1.0), (group_b, -1.0)):
                if group != UNIFIED_REFERENCE_GROUP:
                    weights[inter[group]] = weights.get(inter[group], 0.0) + sign
            res = linear_contrast_test(result, weights, n_mice, alpha=alpha)
            pairwise_rows.append({
                'block': 'pairwise', 'outcome': outcome.key,
                'comparison': f'{group_a}_vs_{group_b}',
                'group_a': group_a, 'group_b': group_b,
                'estimate_log': res['estimate'], 'se': res['se'],
                'ci_low_log': res['ci_low'], 'ci_high_log': res['ci_high'],
                'relative_modulation_ratio': float(np.exp(res['estimate'])),
                'ratio_ci_low': float(np.exp(res['ci_low'])),
                'ratio_ci_high': float(np.exp(res['ci_high'])),
                't': res['t'], 'df': res['df'], 'p_raw': res['p'],
            })

        # The contrast estimates must BE the coefficients they are documented to be. Cheap, and it
        # is the one failure mode -- a coefficient-name mismatch picking up the wrong term -- that
        # would produce a plausible-looking wrong answer rather than an exception.
        # Names and values off the SAME call, so the two cannot fall out of order.
        _names, _params = _fe_names_and_params(result)
        params = dict(zip(_names, np.asarray(_params, dtype=float)))
        expected = {
            f'hM3D_vs_{UNIFIED_REFERENCE_GROUP}': params[inter['hM3D']],
            f'hM4D_vs_{UNIFIED_REFERENCE_GROUP}': params[inter['hM4D']],
            'hM3D_vs_hM4D': params[inter['hM3D']] - params[inter['hM4D']],
        }
        for row in pairwise_rows[-len(RECALL_MODULATION_PAIRS):]:
            want = expected[row['comparison']]
            if not np.isclose(row['estimate_log'], want, rtol=1e-10, atol=1e-12):
                raise RuntimeError(
                    f'recall_modulation_contrasts: {outcome.key} {row["comparison"]} estimated '
                    f'{row["estimate_log"]!r} but the fitted coefficients imply {want!r}. The '
                    f'modulation contrast is not the group x epoch term it is documented to be.')

    within_group_df = pd.DataFrame(within_rows)
    pairwise_df = pd.DataFrame(pairwise_rows)

    pairwise_df['p_holm_modulation'] = np.nan
    pairwise_df['holm_modulation_reject'] = False
    for outcome_key, idx in pairwise_df.groupby('outcome', observed=True).groups.items():
        idx = list(idx)
        if len(idx) != len(RECALL_MODULATION_PAIRS):
            raise RuntimeError(
                f'recall_modulation_contrasts: outcome {outcome_key!r} produced {len(idx)} '
                f'pairwise modulation contrasts, expected exactly '
                f'{len(RECALL_MODULATION_PAIRS)}. The Holm family is defined by that shape.')
        reject, padj = holm_correct(pairwise_df.loc[idx, 'p_raw'].to_numpy(), alpha=alpha)
        pairwise_df.loc[idx, 'p_holm_modulation'] = padj
        pairwise_df.loc[idx, 'holm_modulation_reject'] = reject
    return within_group_df, pairwise_df


def recall_modulation_lookup(pairwise_df, outcome, group_a, group_b):
    """The single pairwise-modulation row for one (outcome, group_a, group_b), as a Series.

    The modulation counterpart of unified_contrast_lookup, and there for the same reason: the
    panel's brackets, the markdown tables and the drafted paragraph all go through one accessor,
    so a figure cannot annotate itself from a different row than its companion text describes.
    Raises rather than returning an empty match.
    """
    sel = pairwise_df[(pairwise_df['outcome'] == outcome)
                      & (pairwise_df['group_a'] == group_a)
                      & (pairwise_df['group_b'] == group_b)]
    if len(sel) != 1:
        raise RuntimeError(f'recall_modulation_lookup: expected exactly one row for '
                           f'({outcome!r}, {group_a!r}, {group_b!r}), found {len(sel)}.')
    return sel.iloc[0]


def recall_within_group_lookup(within_group_df, outcome, group):
    """The single within-group pre->post row for one (outcome, group), as a Series."""
    sel = within_group_df[(within_group_df['outcome'] == outcome)
                          & (within_group_df['group'] == group)]
    if len(sel) != 1:
        raise RuntimeError(f'recall_within_group_lookup: expected exactly one row for '
                           f'({outcome!r}, {group!r}), found {len(sel)}.')
    return sel.iloc[0]


# How a within-group pre->post estimate is READ, fixed in advance and keyed only on whether its
# interval excludes zero and in which direction. The reason this is a lookup rather than prose
# written per run is the one this module applies everywhere: an estimate whose interval spans zero
# licenses "no detectable change" and nothing stronger, and that must not become "increased"
# because the point estimate happened to land above 1.0.
_MODULATION_WITHIN_GROUP_READING = {
    +1: 'increased from pre-tone to post-tone',
    0: 'showed no detectable pre-tone to post-tone change',
    -1: 'decreased from pre-tone to post-tone',
}

_MODULATION_PAIRWISE_READING = {
    True: 'differed',
    False: 'did not detectably differ',
}


def _modulation_direction(row):
    """+1 / 0 / -1 for a within-group change whose 95% interval excludes zero above, spans it, or
    excludes it below. The key into _MODULATION_WITHIN_GROUP_READING."""
    if row['ci_low_log'] > 0.0:
        return +1
    if row['ci_high_log'] < 0.0:
        return -1
    return 0


def write_recall_modulation_summary(save_dir, session_key, session_label, interactions_table,
                                    within_group_df, pairwise_df, stats_prefix,
                                    trajectory_df, filename=None):
    """
    The paper-facing account of the pre->post modulation decomposition, for ONE recall session.

    ** The omnibus is printed before any pairwise row, in every outcome's section. ** The pairwise
    contrasts are a decomposition of the group x epoch joint Wald test, and a decomposition read
    without the test it decomposes is how a 3-df family of comparisons turns into a headline. The
    ordering here is the mechanism that prevents it.

    ** The three questions are separated typographically, not just described. ** Absolute
    within-epoch group differences live in `<prefix>_posthoc_contrasts.csv` and are not restated
    here; the within-group changes are labelled DESCRIPTIVE; the pairwise comparisons of those
    changes are labelled as the inferential question and carry the Holm-adjusted P.

    ** Question B is answered TWICE, adjacently, and the file says why. ** Section B is the
    model-implied within-group change (pooled variance, session df); section B-panel is the paired
    t-test the trajectory figure actually draws (that group's animals only, df = n_g - 1). They
    are different estimators of one estimand and will not agree exactly. Printing them next to
    each other is deliberate -- it is what stops a reader meeting the figure's bracket and the
    model's interval as if one contradicted the other. Neither is a between-group comparison.
    """
    filename = filename or f'{stats_prefix}_modulation_contrasts.md'
    inter = {row['outcome']: row for _, row in interactions_table.iterrows()}
    df2 = int(interactions_table['df2'].iloc[0])

    lines = [
        f'# {session_label} — pre-tone → post-tone modulation',
        '',
        'Each animal\'s **change** in activity across the tone, and the between-group comparisons '
        'of that change. Every number here is a linear contrast of the two mixed models already '
        f'fit for this session (`log(metric) ~ group * epoch + (1|animal)`, '
        f'`stats/{stats_prefix}_{{amplitude,population_rate}}_summary.txt`). **No new model is fit '
        'and no test is computed on the plotted change scores.** This is a representation and a '
        'pairwise decomposition of the group × epoch interaction, not a replacement for it.',
        '',
        '## Three questions this file keeps apart',
        '',
        '| question | answered by |',
        '|---|---|',
        '| **A.** Do the groups differ *within* an epoch (absolute pre-tone or post-tone level)? '
        f'| the existing within-epoch contrasts, `stats/{stats_prefix}_posthoc_contrasts.csv` |',
        '| **B.** Does a group change at all from pre-tone to post-tone? '
        '| the within-group estimates below — **descriptive** |',
        '| **C.** Does the pre→post change *differ between* groups? '
        '| the pairwise modulation contrasts below, and their omnibus |',
        '',
        'A and C are different questions and can disagree in both directions. Two groups with '
        'identical post-tone levels reached from different baselines differ in modulation and not '
        'in absolute level; two groups that decline in parallel from different baselines differ '
        'in absolute level and not in modulation. Neither pattern is evidence about the other.',
        '',
        f'All contrasts use this session\'s own animal-level denominator df = n_animals − 1 = '
        f'{df2}, the same convention as the joint Wald tests, and are t-based. statsmodels\' '
        'asymptotic `P>|z|` values are not used anywhere.',
        '',
    ]

    for outcome in UNIFIED_OUTCOMES:
        row_int = inter[outcome.key]
        lines += [
            f'## {outcome.label}',
            '',
            '### Omnibus — does the pre→post change differ among the three groups?',
            '',
            f'**{format_unified_interaction(row_int)}** '
            f'(joint Wald test of the {int(row_int["df1"])} group × post-tone coefficients).',
            '',
            'This is the general test. A non-significant omnibus is **not** evidence that the '
            'groups modulate identically, and a significant pairwise comparison beneath a '
            'non-significant omnibus is weak evidence that should be described as such.',
            '',
            '### B. Within-group pre→post change — DESCRIPTIVE',
            '',
            'Model-implied change for each group; these characterize the trajectory and explain '
            'the shape of the interaction. **They are not the treatment comparison** — that one '
            'group\'s interval excludes zero and another\'s does not is not a test that the two '
            'differ.',
            '',
            f'| group | Δ log | post/pre {outcome.ratio_label} | 95% CI | t | P (raw) |',
            '|---|---|---|---|---|---|',
        ]
        for group in DREADD_DISPLAY_ORDER:
            row = recall_within_group_lookup(within_group_df, outcome.key, group)
            lines.append(
                f"| {GROUP_LABELS.get(group, group)} | {row['delta_log']:+.3f} | "
                f"{row['post_pre_ratio']:.3f} | "
                f"[{row['ratio_ci_low']:.3f}, {row['ratio_ci_high']:.3f}] | "
                f"{row['t']:.2f} | {row['p_raw']:.4g} |")

        lines += [
            '',
            '### B-panel. The same question, as the trajectory figure draws it — paired *t*-test',
            '',
            'The brackets on the **pre→post trajectories figure** '
            '(`*_amplitude_rate_prepost_trajectories.png`) are **not** the model contrasts in '
            'section B. They are a paired *t*-test of each group\'s own animals — the pairing '
            'that figure draws, one line per animal — on the log scale, with '
            '**df = n − 1 for that group alone**, from '
            f'`stats/{stats_prefix}_trajectory_paired_tests.csv`.',
            '',
            'Section B pools residual variance across the three groups on the session\'s '
            'animal-level df; this uses one group\'s animals and its own df. **They are two '
            'estimators of the same quantity and will not agree exactly.** That is expected, it '
            'is why both are printed here, and neither supersedes the other: section B remains '
            'this lane\'s estimate of the within-group change, and this is the panel\'s own '
            'annotation.',
            '',
            '**Still question B.** A star in one facet and none in another is *not* evidence '
            'that the two groups modulate differently — that is section C, and nothing in this '
            'table may be used to infer it.',
            '',
            'Brackets read the raw *P*; stars below 0.05, the *P* itself printed up to '
            f'{RECALL_TRAJECTORY_NS_LABEL_MAX:g}, no bracket above that. The Holm adjustment '
            'across the three groups within this outcome is a multiplicity reference, as '
            'everywhere else in this lane.',
            '',
            f'| group | n | mean Δ log | post/pre {outcome.ratio_label} | t | df | '
            'P (raw, drawn) | Holm-adjusted P (multiplicity reference) |',
            '|---|---|---|---|---|---|---|---|',
        ]
        for group in DREADD_DISPLAY_ORDER:
            row = recall_trajectory_lookup(trajectory_df, outcome.key, group)
            flag = ' (rejects at Holm 0.05)' if row['holm_trajectory_reject'] else ''
            drawn = _recall_trajectory_bracket_label(row['p_raw'])
            lines.append(
                f"| {GROUP_LABELS.get(group, group)} | {int(row['n_animals'])} | "
                f"{row['mean_delta_log']:+.3f} | {row['post_pre_ratio']:.3f} | "
                f"{row['t']:.2f} | {int(row['df'])} | "
                f"**{row['p_raw']:.4g}**{'' if drawn is None else f' (drawn: {drawn})'} | "
                f"{row['p_holm_trajectory']:.4g}{flag} |")

        lines += [
            '',
            '### C. Pairwise comparisons of the change — the inferential question',
            '',
            'Difference of the two groups\' pre→post changes, signed *a* − *b*. Against mCherry '
            'this is that group\'s group × post-tone coefficient; between hM3D and hM4D it is the '
            'difference of their two interaction coefficients **including their covariance**. '
            'The `ratio` column is exp(Δ log) — a ratio OF the two groups\' post/pre '
            f'{outcome.ratio_label}s: a value of 1.50 means group *a*\'s post/pre ratio is 1.50 '
            'times group *b*\'s.',
            '',
            '**The reported values are the unadjusted model-derived contrasts — the `P (raw)` '
            'column.** That is what the modulation figure\'s brackets and the drafted paragraph '
            'below read. A Holm adjustment across the three pairwise comparisons within this '
            'outcome (amplitude and rate as separate families, as everywhere else in this '
            'analysis) is also tabulated as a **multiplicity reference**, for transparency; it is '
            'not the governing decision rule for this lane. Do not describe a value taken from '
            'the raw column as Holm-adjusted, or either column as prospectively preregistered.',
            '',
            '| comparison | Δ log | 95% CI (log) | ratio | 95% ratio CI | t | df | '
            'P (raw, reported) | Holm-adjusted P (multiplicity reference) |',
            '|---|---|---|---|---|---|---|---|---|',
        ]
        for group_a, group_b in RECALL_MODULATION_PAIRS:
            row = recall_modulation_lookup(pairwise_df, outcome.key, group_a, group_b)
            flag = ' (rejects at Holm 0.05)' if row['holm_modulation_reject'] else ''
            lines.append(
                f"| {GROUP_LABELS.get(group_a, group_a)} vs {GROUP_LABELS.get(group_b, group_b)} "
                f"| {row['estimate_log']:+.3f} "
                f"| [{row['ci_low_log']:+.3f}, {row['ci_high_log']:+.3f}] "
                f"| {row['relative_modulation_ratio']:.3f} "
                f"| [{row['ratio_ci_low']:.3f}, {row['ratio_ci_high']:.3f}] "
                f"| {row['t']:.2f} | {int(row['df'])} | **{row['p_raw']:.4g}** "
                f"| {row['p_holm_modulation']:.4g}{flag} |")
        lines.append('')

    # ---- The drafted paragraph, with this run's own numbers already substituted ---------------
    lines += [
        f'# Drafted Results paragraph — {session_label} modulation',
        '',
        '*Generated from this run\'s fitted values. The reading of each estimate is fixed in '
        'advance by its interval (see `_MODULATION_WITHIN_GROUP_READING`), so the wording does '
        'not depend on which sentence the numbers would flatter.*',
        '',
    ]
    for outcome in UNIFIED_OUTCOMES:
        row_int = inter[outcome.key]
        trajectory = ', '.join(
            f"{GROUP_LABELS.get(g, g)} mice "
            f"{_MODULATION_WITHIN_GROUP_READING[_modulation_direction(w)]} "
            f"({w['post_pre_ratio']:.2f}-fold, 95% CI {w['ratio_ci_low']:.2f}–"
            f"{w['ratio_ci_high']:.2f})"
            for g, w in ((g, recall_within_group_lookup(within_group_df, outcome.key, g))
                         for g in DREADD_DISPLAY_ORDER))
        pairwise_text = '; '.join(
            f"{GROUP_LABELS.get(a, a)} versus {GROUP_LABELS.get(b, b)} "
            f"{_MODULATION_PAIRWISE_READING[bool(r['p_raw'] < 0.05)]} "
            f"({r['relative_modulation_ratio']:.2f}-fold, 95% CI {r['ratio_ci_low']:.2f}–"
            f"{r['ratio_ci_high']:.2f}; unadjusted model-derived P = {r['p_raw']:.3g}; "
            f"Holm-adjusted P = {r['p_holm_modulation']:.3g})"
            for a, b, r in ((a, b, recall_modulation_lookup(pairwise_df, outcome.key, a, b))
                            for a, b in RECALL_MODULATION_PAIRS))
        lines += [
            f'**{outcome.label}.** Across the tone, {trajectory}. Comparing those changes between '
            f'groups: {pairwise_text}. The omnibus test of whether pre-to-post modulation differed '
            f'among the three groups was {format_unified_interaction(row_int)}.',
            '',
        ]

    lines += [
        '**Interpretive constraints.**',
        '',
        '- **No CNO was present at recall.** A modulation difference is a persistent consequence '
        'of the conditioning-day manipulation, not evidence of ongoing receptor activation. Do '
        'not write that receptor activation occurred during recall.',
        '- A group whose within-group interval spans zero "showed no detectable change"; it did '
        'not "increase" or "decrease". Where one group declines and another does not, the '
        'supportable wording is that the decline was absent or attenuated in the second group, '
        'not that the second group increased.',
        '- The pairwise contrasts decompose the omnibus; they are not three independent tests, '
        'which is why the omnibus is quoted beside them. Their reported P-values are the '
        '**unadjusted** model-derived contrasts; the Holm-adjusted values across the three '
        'comparisons within each outcome are tabulated above as a multiplicity reference. Report '
        'each for what it is — do not call a raw value adjusted, and do not present either as a '
        'prospectively preregistered decision. The omnibus does not gate the contrasts.',
        '- These contrasts do not replace the absolute within-epoch comparisons in '
        f'`stats/{stats_prefix}_posthoc_contrasts.csv`, which answer a different question.',
        f'- This file describes {session_key} only. The two recall sessions do not contain the '
        'same animals and are never compared.',
        '',
    ]

    ensure_dirs(save_dir)
    write_text(os.path.join(save_dir, filename), '\n'.join(lines))


# Planted magnitudes for the modulation verification. Wide relative to _SYNTHETIC_RESID_SD so the
# checks test the CODE and not one lucky realization; the null arms' margins are correspondingly
# generous, since a p-value under a true null is uniform and no seed makes it safe.
_MODULATION_SYNTHETIC_DECLINE = -0.60     # every group's planted pre->post change, log units
_MODULATION_SYNTHETIC_RESCUE = 0.60       # added back for a group whose decline is abolished
_MODULATION_SYNTHETIC_SPREAD = 0.80       # hM3D/hM4D separation with mCherry planted midway
_MODULATION_SYNTHETIC_BASELINE = 0.50     # pre-tone baseline offset, design 4


def verify_recall_modulation_synthetic(save_dir, epochs=RECALL_EPOCHS,
                                       reference_epoch=RECALL_REFERENCE_EPOCH,
                                       response_epoch=RECALL_RESPONSE_EPOCH,
                                       group_sizes=_SYNTHETIC_GROUP_SIZES,
                                       model_label='Test_B recall', seed=0,
                                       filename='unified_recall_modulation_synthetic_verification.txt'):
    """
    Run the ACTUAL modulation code -- the same fits, the same recall_modulation_contrasts, the
    same Holm family -- against four designs (five planted datasets) whose correct answers are
    known, and hard-fail if it does not recover them. Both outcomes, every design.

    Design 1 -- an IDENTICAL pre->post decline in all three groups. All three pairwise modulation
    contrasts must be ~0 and the omnibus must stay null: a real change that every group shares is
    not a group difference in modulation.

    Design 2 -- ONE group's decline abolished, the other two declining together. That group's two
    contrasts must be clearly positive and the omnibus significant, while the contrast between the
    two unaffected groups stays ~0. Run twice, once with hM3D affected and once with hM4D, so the
    check itself is symmetric: a version that only ever plants the effect on hM3D would pass even
    if the two interaction coefficients were swapped.

    Design 3 -- hM3D and hM4D separated with mCherry planted midway. The hM3D-vs-hM4D contrast
    must recover the full planted separation, which is what exercises the two-coefficient contrast
    and its covariance term; each group-vs-control contrast must recover half of it.

    Design 4 -- EQUAL absolute post-tone values reached from UNEQUAL pre-tone baselines. The
    post-tone within-epoch simple effect must be ~0 while the modulation contrast is clearly
    non-zero. This is the case that makes the point of the whole file: question A (absolute
    difference within an epoch) and question C (difference in the pre->post change) are not the
    same question, and a reader who conflates them reads this design backwards.

    Runs on tables of a few dozen rows, so all four designs cost milliseconds.
    """
    epochs = tuple(epochs)
    decline, rescue = _MODULATION_SYNTHETIC_DECLINE, _MODULATION_SYNTHETIC_RESCUE
    spread, baseline = _MODULATION_SYNTHETIC_SPREAD, _MODULATION_SYNTHETIC_BASELINE

    def _post_only(per_group):
        """shift_fn planting `per_group[group]` at the response epoch and 0 elsewhere."""
        return lambda group, epoch: (per_group[group] if epoch == response_epoch else 0.0)

    def _baselines(per_group_pre, per_group_post):
        return lambda group, epoch: (per_group_pre[group] if epoch == reference_epoch
                                     else per_group_post[group])

    # (label, shift_fn, expected pairwise Δ per comparison, expect a significant omnibus)
    designs = [
        ('shared decline in all three groups',
         _post_only({'mCherry': decline, 'hM3D': decline, 'hM4D': decline}),
         {('hM3D', 'mCherry'): 0.0, ('hM4D', 'mCherry'): 0.0, ('hM3D', 'hM4D'): 0.0},
         False),
        ('decline abolished in hM3D only',
         _post_only({'mCherry': decline, 'hM3D': decline + rescue, 'hM4D': decline}),
         {('hM3D', 'mCherry'): rescue, ('hM4D', 'mCherry'): 0.0, ('hM3D', 'hM4D'): rescue},
         True),
        ('decline abolished in hM4D only',
         _post_only({'mCherry': decline, 'hM3D': decline, 'hM4D': decline + rescue}),
         {('hM3D', 'mCherry'): 0.0, ('hM4D', 'mCherry'): rescue, ('hM3D', 'hM4D'): -rescue},
         True),
        ('hM3D and hM4D separated, mCherry midway',
         _post_only({'mCherry': decline, 'hM3D': decline + spread / 2.0,
                     'hM4D': decline - spread / 2.0}),
         {('hM3D', 'mCherry'): spread / 2.0, ('hM4D', 'mCherry'): -spread / 2.0,
          ('hM3D', 'hM4D'): spread},
         True),
        ('equal post-tone levels from unequal pre-tone baselines',
         _baselines({'mCherry': 0.0, 'hM3D': baseline, 'hM4D': baseline},
                    {'mCherry': 0.0, 'hM3D': 0.0, 'hM4D': 0.0}),
         {('hM3D', 'mCherry'): -baseline, ('hM4D', 'mCherry'): -baseline,
          ('hM3D', 'hM4D'): 0.0},
         True),
    ]

    lines = [f'Synthetic verification of the {model_label} pre->post MODULATION decomposition',
             '=' * 78, '',
             'The same fits, the same recall_modulation_contrasts and the same three-comparison '
             'within-outcome',
             f'Holm family the run itself uses. Between-animal SD {_SYNTHETIC_MOUSE_SD}, residual '
             f'SD {_SYNTHETIC_RESID_SD};',
             f'cohort {dict(group_sizes)}; epochs {epochs} (reference {reference_epoch!r}); '
             f'seed {seed}.', '']

    # Wide enough that a correct implementation clears it at this noise level, tight enough that a
    # coefficient wired to the wrong term does not.
    estimate_tol = 0.20

    for offset, (label, shift_fn, expected, expect_significant_omnibus) in enumerate(designs):
        df_syn = _synthetic_mouse_epoch_table(
            (response_epoch,), epochs, np.random.default_rng(seed + offset),
            reference_epoch=reference_epoch, group_sizes=group_sizes, shift_fn=shift_fn)
        fits = {o.key: fit_unified_group_epoch_model(df_syn, o.response_col,
                                                     reference_epoch=reference_epoch)
                for o in UNIFIED_OUTCOMES}
        require_common_unified_method(fits)
        _within, pairwise = recall_modulation_contrasts(
            fits, reference_epoch=reference_epoch, response_epoch=response_epoch)
        posthoc = unified_posthoc_contrasts(fits, df_syn, epochs=epochs,
                                            reference_epoch=reference_epoch,
                                            across_epoch_family=False)
        interactions = unified_interactions_table(fits)

        lines.append(f'{label}:')
        for outcome in UNIFIED_OUTCOMES:
            p_inter = float(fits[outcome.key]['omnibus']['p'])
            lines.append(f'  {outcome.label}: '
                         f'{format_unified_interaction(interactions[interactions["outcome"] == outcome.key].iloc[0])}')
            for (group_a, group_b), want in expected.items():
                row = recall_modulation_lookup(pairwise, outcome.key, group_a, group_b)
                lines.append(f'    {group_a} vs {group_b}: planted {want:+.3f}, estimated '
                             f'{row["estimate_log"]:+.3f} (ratio '
                             f'{row["relative_modulation_ratio"]:.3f}, Holm-adjusted P '
                             f'{row["p_holm_modulation"]:.3g})')
                assert abs(row['estimate_log'] - want) < estimate_tol, (
                    f'verify_recall_modulation_synthetic [{model_label}/{label}/{outcome.key}]: '
                    f'the {group_a}-vs-{group_b} modulation contrast should recover the planted '
                    f'{want:+.3f} log units but estimated {row["estimate_log"]:+.3f}.')
            if expect_significant_omnibus:
                assert p_inter < 0.01, (
                    f'verify_recall_modulation_synthetic [{model_label}/{label}/{outcome.key}]: '
                    f'a group-dependent pre->post change must be detected by the group x epoch '
                    f'omnibus (P < 0.01), got P = {p_inter:.3g}.')
            else:
                assert p_inter > 0.2, (
                    f'verify_recall_modulation_synthetic [{model_label}/{label}/{outcome.key}]: '
                    f'a decline shared by all three groups must not manufacture a group '
                    f'difference in modulation; the omnibus should be clearly null (P > 0.2) but '
                    f'was P = {p_inter:.3g}.')

            # Design 4 is the one that separates question A from question C, so it carries the
            # extra assertion: identical post-tone levels means NO absolute post-tone difference,
            # while the modulation contrast is large.
            if label.startswith('equal post-tone levels'):
                for group in UNIFIED_TREATMENT_GROUPS:
                    simple = unified_contrast_lookup(posthoc, outcome.key, response_epoch, group)
                    lines.append(f'    [question A] {group} vs mCherry at {response_epoch}: '
                                 f'{simple["estimate_log"]:+.3f} log units (P '
                                 f'{simple["p_raw"]:.3g}) -- absolute levels were planted equal')
                    assert abs(simple['estimate_log']) < estimate_tol, (
                        f'verify_recall_modulation_synthetic [{model_label}/{label}/'
                        f'{outcome.key}]: post-tone levels were planted EQUAL, so the absolute '
                        f'{group}-vs-mCherry simple effect should be ~0, got '
                        f'{simple["estimate_log"]:+.3f}.')
        lines.append('')

    lines += [
        'All assertions passed.',
        '',
        'The last design is the point of this file. Equal post-tone levels reached from unequal',
        'pre-tone baselines give NO absolute post-tone group difference and a LARGE difference in',
        'pre->post modulation. The within-epoch contrasts and the modulation contrasts answer',
        'different questions, and neither is evidence about the other.',
    ]
    ensure_dirs(save_dir)
    write_text(os.path.join(save_dir, filename), '\n'.join(lines))


def _modulation_values_per_group(modulation_df, outcome_key,
                                 group_order=DREADD_DISPLAY_ORDER):
    """{group: (n_mice, 1) array of per-animal delta_log} for one outcome -- the shape
    _draw_mouse_violin_panel consumes. Mirrors _mouse_values_per_group, but the collapse to one
    value per animal has already happened (build_recall_modulation_by_mouse), so this only
    reshapes; it does not average anything."""
    sub = modulation_df[modulation_df['outcome'] == outcome_key]
    values = {}
    for group in group_order:
        vals = sub.loc[sub['group'] == group, 'delta_log'].to_numpy(dtype=float)
        if len(vals) < 2:
            raise RuntimeError(f'_modulation_values_per_group: group {group} has {len(vals)} '
                               f'animal(s) for {outcome_key!r}; need >=2 for a group comparison.')
        values[group] = vals.reshape(-1, 1)
    return values


def _assert_modulation_markers_match_table(modulation_df, outcome_key, values_per_group):
    """Hard-fail unless the values about to be plotted ARE that outcome's rows of the modulation
    table.

    The modulation counterpart of _assert_markers_match_model, and there for the same reason: the
    panel's whole claim is that each point is one animal's model-scale change score, and a
    reshaping or outcome-keying slip between the table and the panel is exactly the kind of error
    nobody catches by eye.
    """
    sub = modulation_df[modulation_df['outcome'] == outcome_key]
    for group, drawn in values_per_group.items():
        expected = np.sort(sub.loc[sub['group'] == group, 'delta_log'].to_numpy(dtype=float))
        got = np.sort(np.asarray(drawn, dtype=float).reshape(-1))
        if got.shape != expected.shape or not np.allclose(got, expected, rtol=1e-12, atol=0.0):
            raise RuntimeError(
                f'_assert_modulation_markers_match_table [{outcome_key}, {group}]: the panel '
                f'would draw {got!r} but the modulation table holds {expected!r}.')


def plot_recall_modulation(modulation_df, pairwise_df, interactions_table, save_dir,
                           session_label, stats_prefix, filename_root):
    """
    The per-animal pre->post CHANGE, one panel per outcome, with the model's own pairwise
    comparisons of that change as the brackets.

    ** One point per animal, and no cell cloud. ** This panel is explicitly a change-score view:
    the quantity plotted is a difference of two of that animal's model rows, which has no
    cell-level counterpart to draw beneath it. The absolute per-cell distributions are the
    business of the `_amplitude_rate_by_epoch` figure.

    ** The brackets are looked up, never computed. ** `_precomputed_stat_fn` is handed this
    outcome's modulation P-values and ignores the plotted arrays entirely, so there is no code
    path by which a bracket disagrees with `stats/<prefix>_modulation_contrasts.csv`. A Welch test
    on these five-to-six change scores would be a different (and unstated) statistic that happens
    to sit on the same axes.

    ** The displayed P-values are the UNADJUSTED model-derived contrasts (`p_raw`). ** They are
    linear contrasts of the same fitted `log(metric) ~ group * epoch + (1|mouse)` model the rest
    of this lane reports, read out of the contrast table. The three-comparison Holm-adjusted
    values (`p_holm_modulation`) are still computed and still written to the CSV and the companion
    markdown as a MULTIPLICITY REFERENCE, but they are not what this figure or the drafted wording
    reads and they are not a decision rule here. Nothing displayed is described as adjusted.

    ** Which comparisons are drawn is a display choice, set by RECALL_MODULATION_FIGURE_PAIRS. **
    hM4D-vs-mCherry is computed and tabulated for both outcomes like the other two but carries no
    bracket here (NaN into _precomputed_stat_fn, which the bracket drawer skips). A comparison in
    RECALL_MODULATION_NS_LABEL_PAIRS is bracketed with its P-value even when P >= 0.05, so a
    reported null is shown as a number rather than left to be inferred from a missing bracket.

    ** Every panel carries its own omnibus in its title. ** The pairwise contrasts decompose the
    joint Wald test; showing a star without it is how a decomposition becomes a headline.

    ** Log-difference, not a post/pre ratio axis. ** The fitted interaction is additive on the log
    scale, so the difference the brackets describe is the quantity actually plotted. The
    multiplicative reading is one exp() away and is tabulated in the companion markdown.
    """
    inter = {row['outcome']: row for _, row in interactions_table.iterrows()}
    fig, axs = plt.subplots(1, len(UNIFIED_OUTCOMES),
                            figsize=(3.1 * len(UNIFIED_OUTCOMES), 3.6))
    axs = np.atleast_1d(axs)
    for ax, outcome in zip(axs, UNIFIED_OUTCOMES):
        values_per_group = _modulation_values_per_group(modulation_df, outcome.key)
        _assert_modulation_markers_match_table(modulation_df, outcome.key, values_per_group)
        pairwise = {(a, b): recall_modulation_lookup(pairwise_df, outcome.key, a, b)
                    for a, b in RECALL_MODULATION_PAIRS}
        _draw_mouse_violin_panel(
            ax, values_per_group,
            ylabel=f'Δ log {outcome.label.lower()}\n(post-tone − pre-tone)',
            title=f'{outcome.label}\n{format_unified_interaction(inter[outcome.key])}',
            annotate='stats', bracket_mode='axes',
            ns_label_pairs=RECALL_MODULATION_NS_LABEL_PAIRS[outcome.key],
            stat_fn=_precomputed_stat_fn(*(
                float(pairwise[pair]['p_raw']) if pair in RECALL_MODULATION_FIGURE_PAIRS
                else np.nan
                for pair in (('hM3D', 'mCherry'), ('hM4D', 'mCherry'), ('hM3D', 'hM4D')))))
        # No change is zero on this scale, and it is the only reference the eye needs; without it
        # a panel of small negative deltas reads as "no effect" purely from the axis limits.
        ax.axhline(0.0, color='k', linewidth=0.8, linestyle='--', zorder=1)

    fig.suptitle(f'{session_label} — per-animal pre-tone → post-tone modulation', size='medium')
    fig.text(0.5, 0.015,
             # Three short lines: this figure is ~6.2 in wide, and a caption line much past ~110
             # characters is clipped at both edges rather than wrapped.
             'One point per animal. Brackets are UNADJUSTED model-derived contrast P-values from '
             'the fitted model —\nnot multiplicity-adjusted, not prospectively preregistered; no '
             'statistic is computed from the plotted\nvalues. Holm-adjusted values are retained '
             f'in `stats/{stats_prefix}_modulation_contrasts.csv` as a reference.',
             ha='center', size='xx-small')
    fig.subplots_adjust(left=0.13, bottom=0.20, right=0.98, top=0.80, wspace=0.42)
    ensure_dirs(save_dir)
    save_fig(fig, os.path.join(save_dir, filename_root + '.png'))
    plt.close(fig)


# ── The trajectory panel's own WITHIN-GROUP paired test ──────────────────────────────────────
#
# ** This answers question B and ONLY question B: did THIS group change from pre-tone to
# post-tone? ** It is not a treatment comparison and may never be read as one. "hM3D changed and
# mCherry did not" is not a test that the two changed differently -- that is question C, it stays
# with the model-derived pairwise modulation contrasts in recall_modulation_contrasts, and nothing
# computed here substitutes for them. The between-group brackets on the modulation figure continue
# to come from the fitted models alone.
#
# ** It is deliberately a SECOND estimate of a quantity the LMM already reports, and the two
# DISAGREE by construction. ** within_group_df carries the same pre->post change as a linear
# contrast of the fitted mixed model, pooling residual variance across the three groups on that
# session's animal-level df. The paired t here uses only that group's own animals and its own
# df = n_g - 1, because that is precisely what this panel draws -- one line per animal, within one
# group, on the log scale the model was fit on. Neither is wrong; they are different estimators of
# the same estimand. The disagreement is therefore put ON THE RECORD rather than hidden: both
# tables are written, the companion markdown prints them adjacent under question B, and each
# states which it is. The model contrast remains the lane's estimate of the within-group change;
# this is the trajectory panel's own annotation and nothing else reads it.

# Sets `holm_trajectory_reject` only. The STAR LADDER drawn on the panel is the conventional
# ***/**/* of _stars_from_p (< 0.001 / < 0.01 / < 0.05), which is imported rather than restated so
# this module cannot drift from the one every other caban figure uses.
RECALL_TRAJECTORY_ALPHA = 0.05
# 0.05 <= P < this gets a bracket labelled with its own P instead of stars, so a near-miss is
# STATED rather than left to be inferred from an absent bracket. Above it, no bracket is drawn.
RECALL_TRAJECTORY_NS_LABEL_MAX = 0.10


def recall_trajectory_paired_tests(modulation_df, alpha=RECALL_TRAJECTORY_ALPHA,
                                   group_order=DREADD_DISPLAY_ORDER):
    """
    A paired t-test of each group's pre-tone -> post-tone change, computed on the LOG-scale
    per-animal values that plot_recall_prepost_trajectories draws. One row per (outcome, group).

    ** The test is on the pairing the panel shows. ** Each animal contributes one pre-tone and one
    post-tone value -- the very rows the models were fit on, pivoted by
    build_recall_modulation_by_mouse -- and the test is scipy's ttest_rel over those pairs, which
    is identically a one-sample t on the animal's `delta_log`. It is computed on the LOG scale, not
    the natural scale the panel's axis is drawn on, because that is the scale the change is
    additive on and the scale every other number in this lane lives on; exp(mean delta) is reported
    beside it as the fold change.

    ** df = n_g - 1, this group's own animals only. ** That is the deliberate difference from the
    model contrast in `within_group_df`, which pools variance across the three groups on the
    session's animal-level df -- see the block comment above. Both are written out.

    ** Raw P is what the panel draws; Holm is computed beside it. ** Three groups within one
    outcome form one Holm family, amplitude and rate as separate families -- the same shape as
    recall_modulation_contrasts' pairwise family and unified_posthoc_contrasts' within-epoch one.
    `p_holm_trajectory`/`holm_trajectory_reject` are a MULTIPLICITY REFERENCE, exactly as
    `p_holm_modulation` is for the modulation figure; the bracket reads `p_raw` and the companion
    text says so.

    Hard-fails on a group with fewer than two animals, a non-finite value, or zero within-group
    variance in the change -- each is a data pathology that would otherwise emit a meaningless or
    infinite t.

    Returns a DataFrame with columns
        block, outcome, group, n_animals, mean_delta_log, sd_delta_log, se_delta_log,
        post_pre_ratio, t, df, p_raw, p_holm_trajectory, holm_trajectory_reject
    """
    rows = []
    for outcome in UNIFIED_OUTCOMES:
        sub_outcome = modulation_df[modulation_df['outcome'] == outcome.key]
        for group in group_order:
            # Sorted so the paired arrays are in a deterministic animal order; the t is invariant
            # to it, but a reproducible row order matters for the written table.
            sub = sub_outcome[sub_outcome['group'] == group].sort_values('mouse')
            pre = sub['pre_tone_value_log'].to_numpy(dtype=float)
            post = sub['post_tone_value_log'].to_numpy(dtype=float)
            n = len(sub)
            if n < 2:
                raise RuntimeError(
                    f'recall_trajectory_paired_tests: group {group!r} has {n} animal(s) with a '
                    f'{outcome.key} change score; a paired t-test needs at least two. The '
                    f'inferential table is supposed to be a complete animal x epoch grid.')
            if not (np.all(np.isfinite(pre)) and np.all(np.isfinite(post))):
                raise RuntimeError(
                    f'recall_trajectory_paired_tests: group {group!r} has a non-finite '
                    f'{outcome.key} pre- or post-tone value; investigate the animal rather than '
                    f'dropping it.')
            delta = post - pre
            sd = float(np.std(delta, ddof=1))
            if not np.isfinite(sd) or sd == 0.0:
                raise RuntimeError(
                    f'recall_trajectory_paired_tests: group {group!r} has zero (or non-finite) '
                    f'between-animal variance in its {outcome.key} pre->post change (sd={sd!r}), '
                    f'so a paired t is undefined. That is a data pathology, not a result.')
            res = scipy_stats.ttest_rel(post, pre)
            mean_delta = float(np.mean(delta))
            se = sd / np.sqrt(n)
            # The one failure mode that would produce a plausible-looking wrong answer is
            # ttest_rel being handed the arrays the other way round (a sign flip) or an unpaired
            # variant sneaking in. Both are caught by reconstructing the statistic.
            t_manual = mean_delta / se
            if not np.isclose(float(res.statistic), t_manual, rtol=1e-8, atol=1e-10):
                raise RuntimeError(
                    f'recall_trajectory_paired_tests: scipy returned t={float(res.statistic)!r} '
                    f'for {outcome.key}/{group} but the paired construction implies {t_manual!r}. '
                    f'The reported statistic is not the paired test it is documented to be.')
            rows.append({
                'block': 'trajectory_paired_t', 'outcome': outcome.key, 'group': group,
                'n_animals': int(n), 'mean_delta_log': mean_delta, 'sd_delta_log': sd,
                'se_delta_log': float(se), 'post_pre_ratio': float(np.exp(mean_delta)),
                't': float(res.statistic), 'df': int(n - 1), 'p_raw': float(res.pvalue),
            })

    trajectory_df = pd.DataFrame(rows)
    trajectory_df['p_holm_trajectory'] = np.nan
    trajectory_df['holm_trajectory_reject'] = False
    for outcome_key, idx in trajectory_df.groupby('outcome', observed=True).groups.items():
        idx = list(idx)
        if len(idx) != len(group_order):
            raise RuntimeError(
                f'recall_trajectory_paired_tests: outcome {outcome_key!r} produced {len(idx)} '
                f'within-group tests, expected exactly {len(group_order)}. The Holm family is '
                f'defined by that shape.')
        reject, padj = holm_correct(trajectory_df.loc[idx, 'p_raw'].to_numpy(), alpha=alpha)
        trajectory_df.loc[idx, 'p_holm_trajectory'] = padj
        trajectory_df.loc[idx, 'holm_trajectory_reject'] = reject
    return trajectory_df


def recall_trajectory_lookup(trajectory_df, outcome, group):
    """The single within-group paired-test row for one (outcome, group), as a Series.

    Same role as recall_modulation_lookup and recall_within_group_lookup: the panel's brackets and
    the companion markdown go through ONE accessor, so the figure cannot annotate itself from a
    different row than the text describes. Raises rather than returning an empty match.
    """
    match = trajectory_df[(trajectory_df['outcome'] == outcome)
                          & (trajectory_df['group'] == group)]
    if len(match) != 1:
        raise RuntimeError(
            f'recall_trajectory_lookup: expected exactly one row for outcome {outcome!r}, group '
            f'{group!r}, found {len(match)}.')
    return match.iloc[0]


def _recall_trajectory_bracket_label(p_raw):
    """The panel's label for one within-group paired test, or None for no bracket at all.

    Stars by the conventional ladder (_stars_from_p), then a bare `P = 0.0xx` up to
    RECALL_TRAJECTORY_NS_LABEL_MAX so a near-miss is stated rather than inferred from an absent
    bracket, then nothing. One function, so the panel and its companion text cannot end up
    applying two different rules to the same p-value.
    """
    p = float(p_raw)
    if not np.isfinite(p):
        raise ValueError(f'_recall_trajectory_bracket_label: p must be finite, got {p!r}.')
    stars = _stars_from_p(p)
    if stars is not None:
        return stars
    if p < RECALL_TRAJECTORY_NS_LABEL_MAX:
        return format_p_display(p)
    return None


def plot_recall_prepost_trajectories(modulation_df, trajectory_df, save_dir, session_label,
                                     stats_prefix, filename_root,
                                     group_order=DREADD_DISPLAY_ORDER):
    """
    The mechanism behind the change scores: each animal's pre-tone and post-tone value joined by a
    line, one facet per group, one row per outcome.

    ** The bracket in each facet is that group's own WITHIN-GROUP paired t-test ** -- question B,
    "did this group change across the tone?" -- read from `trajectory_df`
    (recall_trajectory_paired_tests), which is written to
    `stats/<prefix>_trajectory_paired_tests.csv` so every drawn p-value is auditable. It is the
    test of the pairing this panel actually shows: one line per animal, within one group.

    ** A bracket here is NOT a treatment comparison and must never be read as one. ** That hM3D
    carries a star and mCherry does not is not evidence that the two groups modulate differently;
    that is question C, and it stays with the model-derived pairwise modulation contrasts and
    their omnibus in `stats/<prefix>_modulation_contrasts.csv`, drawn on the modulation figure.
    The footer says so on the figure itself, because the panel is the thing a reader meets first.

    ** The brackets read `p_raw`, unadjusted ** -- the same convention the modulation figure uses.
    The Holm adjustment across the three groups within an outcome travels in the CSV as a
    multiplicity reference. Labelling: stars below 0.05, the p-value itself up to
    RECALL_TRAJECTORY_NS_LABEL_MAX, no bracket above that (_recall_trajectory_bracket_label).

    Values are on the NATURAL scale -- exp() of the model rows -- because the point of this panel
    is to show where each animal started and finished, and a reader knows amplitudes and rates in
    their own units. For amplitude that is the animal's geometric mean event amplitude, the
    exponential of the mean of its cell-level log amplitudes. ** The TEST is on the log scale **
    (recall_trajectory_paired_tests), which is the scale the change is additive on and the scale
    the models were fit on; only the drawing is natural-scale.
    """
    fig, axs = plt.subplots(len(UNIFIED_OUTCOMES), len(group_order), sharey='row',
                            figsize=(2.1 * len(group_order), 2.9 * len(UNIFIED_OUTCOMES)))
    axs = np.atleast_2d(axs).reshape(len(UNIFIED_OUTCOMES), len(group_order))
    for r, outcome in enumerate(UNIFIED_OUTCOMES):
        sub_outcome = modulation_df[modulation_df['outcome'] == outcome.key]
        labels = {}
        for c, group in enumerate(group_order):
            ax = axs[r, c]
            ax.spines[['right', 'top']].set_visible(False)
            sub = sub_outcome[sub_outcome['group'] == group]
            for _, row in sub.iterrows():
                ax.plot([0, 1],
                        [np.exp(row['pre_tone_value_log']), np.exp(row['post_tone_value_log'])],
                        color=GROUP_COLOURS[group], marker='o', markersize=4,
                        markeredgecolor='k', markeredgewidth=0.3, linewidth=1.0, alpha=0.85)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['pre', 'post'], size='small')
            ax.set_xlim(-0.35, 1.35)
            if r == 0:
                ax.set_title(f'{GROUP_LABELS.get(group, group)} (n = {len(sub)})', size='small')
            if c == 0:
                ax.set_ylabel(outcome.label, size='small')
            ax.tick_params(labelsize='x-small')
            labels[c] = _recall_trajectory_bracket_label(
                recall_trajectory_lookup(trajectory_df, outcome.key, group)['p_raw'])

        # Headroom is reserved ONCE per row and only when that row carries a bracket: the facets
        # share a y-axis (sharey='row'), so growing one grows all three, and reserving space no
        # bracket will occupy just adds whitespace. Bracket y-positions are AXES FRACTIONS via
        # get_xaxis_transform, so they land identically in all three facets whatever the shared
        # data range is -- the same reasoning as annotate_pairwise_brackets.
        if any(lab is not None for lab in labels.values()):
            reserve_top_fraction(axs[r, 0], occupancy=0.84)
            for c, label in labels.items():
                if label is None:
                    continue
                ax = axs[r, c]
                trans = ax.get_xaxis_transform()
                bar, tick = 0.90, 0.025
                ax.plot([0, 0, 1, 1], [bar - tick, bar, bar, bar - tick], transform=trans,
                        color='k', linewidth=0.9, clip_on=False, solid_capstyle='butt')
                ax.text(0.5, bar + 0.015, label, transform=trans, ha='center', va='bottom',
                        color='k', size='small' if label.startswith('*') else 'x-small')

    fig.suptitle(f'{session_label} — per-animal pre-tone → post-tone trajectories',
                 size='medium')
    fig.text(0.5, 0.015,
             'One line per animal, natural scale. Brackets are each group\'s OWN within-group '
             'paired t-test of its pre→post change\n(log scale, df = n−1, unadjusted; '
             f'`stats/{stats_prefix}_trajectory_paired_tests.csv`) — they say whether THAT group '
             'changed,\nand are NOT a comparison between groups. Whether the groups modulate '
             'differently is the model-derived pairwise\ncontrasts and their group × epoch '
             f'omnibus in `stats/{stats_prefix}_modulation_contrasts.csv`.',
             ha='center', size='xx-small')
    fig.subplots_adjust(left=0.14, bottom=0.17, right=0.98, top=0.88, wspace=0.20, hspace=0.35)
    ensure_dirs(save_dir)
    save_fig(fig, os.path.join(save_dir, filename_root + '.png'))
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# HIERARCHICAL CELL-LEVEL COMPANION / SENSITIVITY ANALYSIS  (recall, Test_B)
# ─────────────────────────────────────────────────────────────────────────────
#
# ** WHY THIS EXISTS. ** The paper-facing recall lane collapses each animal to ONE scalar per
# epoch before fitting (build_mouse_epoch_unified_table). That is a valid hierarchical analysis
# and it stays the primary one. It is not the ONLY valid one, and it discards two things:
#
#   * the WITHIN-CELL PAIRING. An animal's pre-tone value is a mean over the cells active in
#     pre-tone and its post-tone value is a mean over the cells active in post-tone -- different
#     cell sets -- so cell-identity variance never cancels out of the change score.
#   * the WITHIN-MOUSE CELLULAR HETEROGENEITY that says whether an animal's shift is coherent
#     across its population or carried by a handful of cells.
#
# This suite is the companion analysis that keeps both while leaving treatment assignment where
# it actually is: at the MOUSE. It replaces nothing. Every existing recall and TFC model,
# contrast, Holm family, omnibus, table and figure is untouched.
#
# ** WHAT MAKES IT A HIERARCHICAL TEST RATHER THAN A BETTER-COMPUTED MOUSE MEAN. ** The estimator
# is a paired-cell mixed model (delta ~ group + (1|mouse)), and the INFERENCE is the exact
# mouse-label randomization OF THAT MODEL'S OWN COEFFICIENT: the model is refit under every
# relabeling of the mouse group labels, so the hierarchy sits inside the statistic being tested
# instead of being replaced by an average taken before the model. A design-based permutation of
# the 16 mouse-mean deltas is retained beside it as a labelled SENSITIVITY -- model-free and a
# useful check, but it collapses each animal to one number and so carries no cellular hierarchy.
#
# ** THE HAZARD THIS IS BUILT AGAINST. ** Thousands of cells are not thousands of randomized
# units. NO asymptotic cell-level p-value from any model here is paper-facing; every reported p
# comes from permuting the 16 MOUSE labels, with every cell staying attached to its own animal.
# n is 16 mice. Cell counts are descriptive and are always reported as such.
#
# ** IMPLEMENTATION VALIDATION IS NOT PART OF A REAL-DATA RUN. ** verify_hierarchical_cell_
# synthetic() and verify_hierarchical_cell_rate_synthetic() plant a known truth in SIMULATED data
# and check that this machinery recovers it. They validate the CODE, not the experiment, and they
# are DEVELOPMENT TOOLS: run them by hand after changing this module. They are deliberately not
# called by run_hierarchical_cell_suite(), because a real-data execution should compute the real
# result and nothing else -- self-testing on synthetic data does not belong in the run that
# produces the numbers, and it was dominating the runtime of one.
#
# ** ONE PREDEFINED FLOW, NO FALLBACKS, NO RESULT-DEPENDENT BRANCHING. ** run_hierarchical_cell_
# suite() is all-or-nothing over the ANALYSIS components: every component in
# _HIERARCHICAL_CELL_COMPONENTS runs on every pass. If a specified model fails
# to fit or fails its predefined adequacy gate, the suite RAISES and reports the diagnostics. It
# never substitutes a simpler model, a pooled-cell test or a Gaussian approximation -- including
# fit_mixed_model's own documented clustered-OLS fallback, which is correct for the other lanes
# and is explicitly barred here (clustered OLS over cell rows with 16 clusters is precisely the
# anti-conservative cell-level inference this analysis exists to avoid).

# Both recall sessions, each analysed entirely within itself. Still gated behind
# run_hierarchical_cell_analysis (False by default), so a routine pass reaches neither.
RECALL_HIERARCHICAL_CELL_SESSIONS = ('Test_B', 'Test_B_1wk')
HIERARCHICAL_CELL_METHODS_FILENAME = 'sp_rates_lmm_recall_hierarchical_cells_methods.md'
HIERARCHICAL_CELL_DIRNAME = 'hierarchical_cells'
HIERARCHICAL_CELL_STATS_PREFIX = 'hierarchical_cell'
HIERARCHICAL_CELL_REFERENCE_GROUP = 'mCherry'
HIERARCHICAL_CELL_DELTA_COL = 'delta_log_amplitude_cell'

# The three pairwise comparisons, in (a, b) order, which are ALSO exactly the Holm family. The
# question here is symmetric across the three groups -- unlike the absolute-level paper panels,
# where hM3D-vs-hM4D is in no family -- so all three enter one correction and all three get a
# bracket. Same structure as RECALL_MODULATION_PAIRS, and deliberately the same shape.
HIERARCHICAL_CELL_PAIRS = (('hM3D', 'mCherry'), ('hM4D', 'mCherry'), ('hM3D', 'hM4D'))

# Monte Carlo settings for the OMNIBUS only (the pairwise tests are exact -- see
# hierarchical_cell_amplitude_permutation). Frozen for production; a development pass may lower
# n_perm, and the value actually used is written into the output so it can never be mistaken for
# a reported run.
HIERARCHICAL_CELL_N_PERM_OMNIBUS = 2000
HIERARCHICAL_CELL_SEED = 0

# ** NUMERICAL optimizer for this suite's mixed models -- frozen, and not a statistical choice. **
# fit_mixed_model's default 'lbfgs' silently returns a BOUNDARY solution on frames of this shape
# (a few hundred cell rows in each of ~16 mice): the mouse variance is pinned at 0 and the
# fixed-effect standard errors come back non-finite, which _mixed_model_degeneracy correctly
# rejects -- and which would then take the clustered-OLS fallback this suite bars, so the run
# would simply stop. bfgs/cg/powell all reach the same interior optimum on the same likelihood.
# The list is statsmodels' own deterministic escalation over OPTIMIZERS; the model, the
# likelihood and the estimand are identical in every case, and a genuinely degenerate or
# non-converged fit still hard-fails. This is frozen here rather than chosen per fit so no run can
# pick an optimizer after seeing a result. The other lanes keep 'lbfgs' and their numbers are
# untouched.
HIERARCHICAL_CELL_LMM_OPTIMIZER = ['bfgs', 'cg', 'powell']


def build_recall_cell_amplitude_modulation(df_matched, reference_epoch=RECALL_REFERENCE_EPOCH,
                                           response_epoch=RECALL_RESPONSE_EPOCH):
    """
    One PAIRED pre->post modulation value per eligible cell, plus the per-animal eligibility
    table that says which cells those were.

    ** Everything upstream is inherited, nothing is retuned. ** `df_matched` is the recall lane's
    OWN retained-trial frame (restrict_to_exposure_matched_trials over the two 20 s windows), so
    the event definition, threshold, windows, trials, cohort and ROI set are identical to the
    paper-facing analysis by construction rather than by a downstream check.

    Per epoch: aggregate_over_trials pools each cell's events across the retained trials and
    filter_amplitude_rows drops the cells with no event in that epoch and takes log of each
    cell's mean event-run integral -- the module's existing convention, applied unchanged. A cell
    is ELIGIBLE when amplitude is defined in BOTH epochs, and

        delta_log_amplitude_cell = log(mean_amp_post) - log(mean_amp_pre).

    ** NO imputation and NO pseudocount. ** Amplitude is conditional on an event by construction,
    so a cell with no events in an epoch has no defined amplitude there; that is definitional, not
    missing data, and a zero or a pseudocount would put an arbitrary constant inside a log where
    its size determines the answer.

    ** THE ESTIMAND IS CONDITIONAL, AND THAT IS REPORTED, NOT BURIED. ** What this quantity
    estimates is the pre->post amplitude modulation OF NEURONS WITH MEASURABLE EVENT AMPLITUDE IN
    BOTH EPOCHS. It is not an estimate over all detected cells and is never generalized to them.
    The restriction is unavoidable for a PAIRED quantity, but it is a real one: the eligible set is
    selected on activity in both windows and that selection could itself differ by group, which is
    why the eligibility table is a reported result rather than bookkeeping and why
    `fraction_eligible_both` travels with the estimate everywhere it is quoted. The analysis that
    includes zero-event cells is the RATE path (build_recall_cell_epoch_count_table), and that is
    one of the reasons it exists.

    Returns (delta_df, eligibility_df):
      delta_df       -- mouse, group, cell_id, n_events_pre/post, mean_amp_pre/post,
                        log_amp_pre/post, delta_log_amplitude_cell
      eligibility_df -- mouse, group, n_cells_total, n_cells_amp_defined_pre,
                        n_cells_amp_defined_post, n_cells_amp_defined_both, fraction_eligible_both
    """
    pooled, amp = {}, {}
    for epoch in (reference_epoch, response_epoch):
        pooled[epoch] = aggregate_over_trials(df_matched, epoch)
        amp[epoch] = filter_amplitude_rows(pooled[epoch])

    # Every detected ROI appears in every window of a retained trial (build_epoch_event_table
    # emits explicit zero-event rows), so the two epochs must cover the same cell set. Checked,
    # not assumed: if they ever diverge, `n_cells_total` would silently mean different things in
    # the two columns of the eligibility table.
    cells_pre = set(map(tuple, pooled[reference_epoch][['mouse', 'cell']].to_numpy()))
    cells_post = set(map(tuple, pooled[response_epoch][['mouse', 'cell']].to_numpy()))
    if cells_pre != cells_post:
        raise RuntimeError(
            f'build_recall_cell_amplitude_modulation: the {reference_epoch!r} and '
            f'{response_epoch!r} epoch-pooled tables cover different (mouse, cell) sets '
            f'({len(cells_pre)} vs {len(cells_post)}; symmetric difference '
            f'{len(cells_pre ^ cells_post)}). Both windows are present on every retained trial by '
            f'construction, so this is a data or aggregation problem, not something to intersect '
            f'away.')

    pre = amp[reference_epoch][['mouse', 'group', 'cell', 'n_events', 'mean_amplitude',
                                'log_amplitude']]
    post = amp[response_epoch][['mouse', 'group', 'cell', 'n_events', 'mean_amplitude',
                                'log_amplitude']]
    delta_df = pre.merge(post, on=['mouse', 'group', 'cell'], how='inner',
                         suffixes=('_pre', '_post'))
    if delta_df.empty:
        raise RuntimeError(
            'build_recall_cell_amplitude_modulation: no cell has amplitude defined in BOTH '
            'epochs, so no paired modulation value exists. Check the threshold and the windows '
            'rather than relaxing the pairing.')
    delta_df = delta_df.rename(columns={
        'cell': 'cell_id',
        'n_events_pre': 'n_events_pre', 'n_events_post': 'n_events_post',
        'mean_amplitude_pre': 'mean_amp_pre', 'mean_amplitude_post': 'mean_amp_post',
        'log_amplitude_pre': 'log_amp_pre', 'log_amplitude_post': 'log_amp_post'})
    delta_df[HIERARCHICAL_CELL_DELTA_COL] = delta_df['log_amp_post'] - delta_df['log_amp_pre']

    # The identities the whole analysis rests on, asserted rather than trusted. A sign slip or a
    # merge that duplicated rows would flip or inflate every downstream estimate silently.
    if not np.allclose(delta_df[HIERARCHICAL_CELL_DELTA_COL],
                       np.log(delta_df['mean_amp_post']) - np.log(delta_df['mean_amp_pre']),
                       rtol=0.0, atol=1e-12):
        raise RuntimeError('build_recall_cell_amplitude_modulation: delta is not '
                           'log(post) - log(pre).')
    if not (delta_df['n_events_pre'] > 0).all() or not (delta_df['n_events_post'] > 0).all():
        raise RuntimeError('build_recall_cell_amplitude_modulation: a paired cell has zero events '
                           'in an epoch; amplitude is undefined there and it must not be paired.')
    if not np.isfinite(delta_df[HIERARCHICAL_CELL_DELTA_COL]).all():
        raise RuntimeError('build_recall_cell_amplitude_modulation: non-finite paired delta.')
    per_cell = delta_df.groupby(['mouse', 'cell_id'], observed=True).size()
    if (per_cell != 1).any():
        raise RuntimeError('build_recall_cell_amplitude_modulation: a (mouse, cell) appears more '
                           'than once in the paired table.')
    mouse_per_cell = delta_df.groupby('cell_id', observed=True)['mouse'].nunique()
    if (mouse_per_cell > 1).any():
        # Expected and harmless -- `cell` is a SESSION-LOCAL unit_id, so the same integer occurs
        # in several animals. Recorded here so nobody later treats cell_id as a global key: every
        # grouping in this suite is keyed on (mouse, cell_id), never cell_id alone.
        pass

    rows = []
    for (mouse, group), sub in pooled[reference_epoch].groupby(['mouse', 'group'], observed=True):
        n_pre = int((amp[reference_epoch]['mouse'] == mouse).sum())
        n_post = int((amp[response_epoch]['mouse'] == mouse).sum())
        n_both = int((delta_df['mouse'] == mouse).sum())
        n_total = int(len(sub))
        if n_both == 0:
            raise RuntimeError(
                f'build_recall_cell_amplitude_modulation: animal {mouse!r} ({group}) has no cell '
                f'with amplitude defined in both epochs ({n_pre} pre, {n_post} post of {n_total} '
                f'cells). It cannot contribute to the paired analysis; investigate the recording '
                f'rather than letting the cohort shrink silently.')
        rows.append({'mouse': mouse, 'group': group, 'n_cells_total': n_total,
                     'n_cells_amp_defined_pre': n_pre, 'n_cells_amp_defined_post': n_post,
                     'n_cells_amp_defined_both': n_both,
                     'fraction_eligible_both': n_both / n_total})
    eligibility_df = pd.DataFrame(rows)
    eligibility_df['group'] = pd.Categorical(eligibility_df['group'],
                                             categories=list(DREADD_DISPLAY_ORDER))
    eligibility_df = eligibility_df.sort_values(['group', 'mouse']).reset_index(drop=True)

    delta_df['group'] = pd.Categorical(delta_df['group'], categories=list(DREADD_DISPLAY_ORDER))
    delta_df = delta_df.sort_values(['group', 'mouse', 'cell_id']).reset_index(drop=True)
    return delta_df, eligibility_df


def _hierarchical_delta_formula():
    return (f'{HIERARCHICAL_CELL_DELTA_COL} ~ '
            f'C(group, Treatment("{HIERARCHICAL_CELL_REFERENCE_GROUP}"))')


def fit_hierarchical_cell_delta_model(delta_df):
    """
    THE PRIMARY AMPLITUDE ESTIMATOR: `delta_log_amplitude_cell ~ group + (1|mouse)` over the
    paired cells, fit through the module's own fit_mixed_model.

    ** This is not a diagnostic. ** Its group coefficients are the reported effect estimates, and
    the SAME coefficients are the statistics the randomization test is built on
    (hierarchical_cell_amplitude_permutation) -- which is what keeps estimate, interval and
    p-value describing one model specification instead of three.

    ** The clustered-OLS fallback is barred here. ** fit_mixed_model falls back to mouse-clustered
    OLS on non-convergence or a degenerate fit -- correct for the lanes that documented it, and
    exactly wrong here: clustered OLS over thousands of cell rows with 16 clusters is the
    anti-conservative cell-level inference this whole suite exists to avoid. A fallback therefore
    RAISES rather than being reported under this function's name.

    ** The summary's own P>|z| column is asymptotic over CELLS and is not paper-facing. ** 16
    animals were randomized, not N cells. The file this fit is written into says so in its header,
    and the inference lives in the permutation CSV.

    Returns dict(result, method, summary_text, formula, n_mice, n_cells, fe_names, coef_names,
    omnibus).
    """
    formula = _hierarchical_delta_formula()
    result, method, text = fit_mixed_model(delta_df, formula, group_col='mouse',
                                           method=HIERARCHICAL_CELL_LMM_OPTIMIZER)
    if method != 'mixedlm':
        raise RuntimeError(
            f'fit_hierarchical_cell_delta_model: fit_mixed_model fell back to {method!r} instead '
            f'of the specified random-intercept mixed model. This suite permits NO statistical '
            f'fallback -- clustered OLS over {len(delta_df)} cell rows with '
            f'{delta_df["mouse"].nunique()} clusters is precisely the cell-level inference this '
            f'analysis is built against. Stopping so the failure is visible; the fit text was:\n'
            f'{text}')
    fe_names, _params = _fe_names_and_params(result)
    coef_names = dict(zip([g for g in GROUP_ORDER if g != HIERARCHICAL_CELL_REFERENCE_GROUP],
                          _nonref_group_coef_names(fe_names, HIERARCHICAL_CELL_REFERENCE_GROUP)))
    n_mice = int(delta_df['mouse'].nunique())
    omnibus = joint_wald_test(result, list(coef_names.values()), n_mice)
    return {'result': result, 'method': method, 'summary_text': text, 'formula': formula,
            'n_mice': n_mice, 'n_cells': int(len(delta_df)), 'fe_names': fe_names,
            'coef_names': coef_names, 'omnibus': omnibus}


def _hierarchical_pair_weights(coef_names, group_a, group_b):
    """{coefficient name: weight} for one pairwise comparison of the paired-cell model.

    Against the reference group a comparison is a single coefficient; between the two treatment
    groups it is their DIFFERENCE, which needs the coefficients' covariance and is therefore a
    linear contrast, not a subtraction of two published SEs (see linear_contrast_test).
    """
    ref = HIERARCHICAL_CELL_REFERENCE_GROUP
    if group_b == ref:
        return {coef_names[group_a]: 1.0}
    if group_a == ref:
        return {coef_names[group_b]: -1.0}
    return {coef_names[group_a]: 1.0, coef_names[group_b]: -1.0}


def _make_hierarchical_model_stat(delta_df, true_assignment, weights, kind='contrast',
                                  coef_names=None):
    """
    stat_fn factory for mouse_label_permutation_test whose statistic is the FITTED PAIRED-CELL
    MODEL, refit on every relabeling.

    ** The full three-group model is refit every time -- never a two-group subset. ** The reported
    estimate and interval come from the three-group fit, so the statistic being permuted has to
    come from the same specification or the interval would describe a different model from the one
    that was tested. When mouse_label_permutation_test is called with restrict_to_groups=(a, b) it
    hands this closure a mapping covering ONLY those two groups' mice; the third group's mice keep
    their true labels (taken from `true_assignment`) and stay in the fit. The restriction is on the
    RANDOMIZATION, not on the data.

    kind='contrast' returns the linear combination of group coefficients given by `weights`;
    kind='wald' returns the 2-df joint Wald F over both group coefficients (the omnibus statistic).

    Cost is one MixedLM fit per draw, which is affordable only over a restricted exact space
    (462/462/252 here) or a modest Monte Carlo sample -- see hierarchical_cell_amplitude_
    permutation for which test uses which.
    """
    base = delta_df[['mouse', HIERARCHICAL_CELL_DELTA_COL]].copy()
    mice = list(dict.fromkeys(base['mouse'].tolist()))
    formula = _hierarchical_delta_formula()
    n_mice = len(mice)
    state = {'n_fallback': 0, 'fallback_examples': []}

    def stat_fn(mouse_to_group):
        assignment = dict(true_assignment)
        assignment.update(mouse_to_group)
        df = base.copy()
        df['group'] = df['mouse'].map(assignment)
        if df['group'].isna().any():
            missing = sorted(df.loc[df['group'].isna(), 'mouse'].unique())
            raise RuntimeError(f'_make_hierarchical_model_stat: mice {missing} have no group '
                               f'under this relabeling.')
        result, method, text = fit_mixed_model(df, formula, group_col='mouse',
                                               method=HIERARCHICAL_CELL_LMM_OPTIMIZER)
        if method != 'mixedlm':
            # Counted AND raised: letting a clustered-OLS draw into the null distribution would
            # quietly mix two estimators inside one randomization test.
            state['n_fallback'] += 1
            if len(state['fallback_examples']) < 3:
                state['fallback_examples'].append(dict(assignment))
            raise RuntimeError(
                f'_make_hierarchical_model_stat: a permuted refit fell back to {method!r} '
                f'(occurrence {state["n_fallback"]}). No fallback estimator may enter this null '
                f'distribution. Offending relabeling: {dict(assignment)}\n{text}')
        names, params = _fe_names_and_params(result)
        if kind == 'wald':
            return float(joint_wald_test(result, list(coef_names.values()), n_mice)['F'])
        missing = [n for n in weights if n not in names]
        if missing:
            raise RuntimeError(f'_make_hierarchical_model_stat: coefficient(s) {missing} absent '
                               f'from the permuted fit. Available: {names}')
        return float(sum(w * float(params[n]) for n, w in weights.items()))

    stat_fn.state = state
    return stat_fn


def hierarchical_cell_amplitude_permutation(delta_df, fit, mice_per_group,
                                            n_perm_omnibus=HIERARCHICAL_CELL_N_PERM_OMNIBUS,
                                            seed=HIERARCHICAL_CELL_SEED, alpha=0.05,
                                            include_global_null_sensitivity=True):
    """
    Mouse-label randomization inference for the paired-cell amplitude analysis.

    ** PAIRWISE (primary): EXACT randomization of the model coefficient. ** For each of the three
    comparisons the statistic is the fitted three-group `delta ~ group + (1|mouse)` contrast --
    beta_Exc, beta_Inh, or beta_Exc - beta_Inh -- and exchangeability is restricted to the two
    groups being compared, with the third group's mice keeping their true labels and staying in
    every fit. With a 6/5/5 cohort those restricted spaces hold C(11,6)=462, C(11,6)=462 and
    C(10,5)=252 relabelings, so the model is refit for EVERY one of them and the p-values are
    exact -- no Monte Carlo error. The three exact p-values are the Holm family
    (`p_holm_pairwise`) and are what the figure's brackets read.

    ** OMNIBUS: Monte Carlo randomization of a 2-df MODEL-BASED statistic. ** The joint Wald test
    of both group coefficients of the same fit, permuted under the GLOBAL null. Two million mixed-
    model refits is not attempted; this is Monte Carlo at a frozen seed and `n_perm_omnibus`, both
    reported. H0 is that those coefficients are jointly zero -- i.e. that mean pre->post cellular
    modulation does not differ by assigned group. It tests THOSE COEFFICIENTS, not the shape,
    spread or tails of the cellular distribution.

    ** DESIGN-BASED SENSITIVITY, demoted from primary but always run. ** The exact permutation of
    the 16 mouse-mean deltas (make_contrast_stat weight='mouse') over the same restricted spaces,
    plus its cell-weighted variant and its global-null counterpart. Model-free, and therefore a
    genuine check on the model-based result -- but it collapses each animal to one number before
    testing, so it carries no cellular hierarchy and is not the primary. It is in NO Holm family.
    No equivalence between it and the model-based statistic is claimed: the model's GLS weighting
    and equal-mouse weighting coincide only at equal cluster sizes, and cells per mouse vary here.

    ** Interval provenance is explicit. ** `estimate_lmm`/`ci_lmm_*` is the model contrast with its
    own Wald interval; `estimate_mouse_weighted`/`ci_mouse_weighted_*` is the design-based estimate
    with an interval computed over the same 16 mouse means (mouse_contrast_ci). Neither interval is
    ever attached to the other estimate. The randomization p-values carry no interval of their own.

    include_global_null_sensitivity : True for every REPORTED run. Set False only by the synthetic
    verification suite, which validates the primary path and the restricted-null sensitivity and
    has no use for the global-null column. That column costs an exact enumeration of all 2,018,016
    global relabelings per comparison -- seconds for one reported run, but tens of minutes across a
    synthetic suite that calls this function seven times. This is a COMPUTATIONAL scoping of a
    verification run, the same allowance the reduced sampler settings get in
    verify_hierarchical_cell_rate_synthetic. It never applies to a reported analysis and never
    touches a primary statistic.

    Returns a DataFrame with one `omnibus` row and, per comparison, one `pairwise_primary` row and
    one `pairwise_sensitivity` row.
    """
    result, coef_names, n_mice = fit['result'], fit['coef_names'], fit['n_mice']
    present = {g: [m for m in mice_per_group.get(g, []) if m in set(delta_df['mouse'])]
               for g in GROUP_ORDER}
    present = {g: ms for g, ms in present.items() if ms}
    true_assignment = {m: g for g, ms in present.items() for m in ms}
    if set(true_assignment) != set(delta_df['mouse']):
        raise RuntimeError('hierarchical_cell_amplitude_permutation: the paired-cell table and '
                           'mice_per_group cover different animals.')

    mouse_delta = {g: delta_df[delta_df['group'] == g].groupby('mouse', observed=True)
                   [HIERARCHICAL_CELL_DELTA_COL].mean().to_numpy(dtype=float)
                   for g in DREADD_DISPLAY_ORDER if g in present}

    rows = []

    # ---- omnibus: 2-df model-based statistic, global null, Monte Carlo -----------------------
    omnibus_stat = _make_hierarchical_model_stat(delta_df, true_assignment, weights=None,
                                                 kind='wald', coef_names=coef_names)
    print(f'[hier   perm] omnibus: {n_perm_omnibus} Monte Carlo mixed-model refits '
          f'(global null, seed {seed})...', flush=True)
    _t = time.perf_counter()
    omnibus_perm = mouse_label_permutation_test(omnibus_stat, present, n_perm=n_perm_omnibus,
                                                seed=seed, progress_label='omnibus',
                                                progress_every=max(1, n_perm_omnibus // 10))
    print(f'[hier   perm] omnibus done in {(time.perf_counter() - _t) / 60:.1f} min: '
          f'P = {omnibus_perm["p_two_sided"]:.4g}', flush=True)
    rows.append({
        'block': 'omnibus', 'group_a': 'all', 'group_b': 'all',
        'statistic_kind': 'joint Wald F on both group coefficients of delta ~ group + (1|mouse)',
        'estimate_lmm': float(fit['omnibus']['F']), 'ci_lmm_low': np.nan, 'ci_lmm_high': np.nan,
        'p_raw': float(omnibus_perm['p_two_sided']), 'p_holm_pairwise': np.nan,
        'estimate_mouse_weighted': np.nan, 'ci_mouse_weighted_low': np.nan,
        'ci_mouse_weighted_high': np.nan, 'p_raw_design_based': np.nan,
        'p_raw_global_null': float(omnibus_perm['p_two_sided']),
        'estimate_cell_weighted': np.nan, 'null_type': 'global',
        'n_relabelings': np.nan, 'exact': False, 'n_perm': int(omnibus_perm['n_perm']),
        'seed': seed, 'n_mice_a': n_mice, 'n_mice_b': n_mice,
        'n_cells_a': int(len(delta_df)), 'n_cells_b': int(len(delta_df)),
        'asymptotic_p_not_for_inference': float(fit['omnibus']['p']),
    })

    # ---- pairwise ----------------------------------------------------------------------------
    for group_a, group_b in HIERARCHICAL_CELL_PAIRS:
        weights = _hierarchical_pair_weights(coef_names, group_a, group_b)
        contrast = linear_contrast_test(result, weights, n_groups=n_mice, alpha=alpha)

        model_stat = _make_hierarchical_model_stat(delta_df, true_assignment, weights)
        n_space = n_distinct_relabelings([group_a] * len(present[group_a])
                                         + [group_b] * len(present[group_b]))
        print(f'[hier   perm] {group_a} vs {group_b}: EXACT enumeration, {n_space} '
              f'three-group model refits...', flush=True)
        _t = time.perf_counter()
        model_perm = mouse_label_permutation_test(model_stat, present, seed=seed,
                                                  restrict_to_groups=(group_a, group_b),
                                                  exact=True,
                                                  progress_label=f'{group_a} vs {group_b}',
                                                  progress_every=max(1, n_space // 5))
        print(f'[hier   perm] {group_a} vs {group_b} done in '
              f'{(time.perf_counter() - _t) / 60:.1f} min: exact P = '
              f'{model_perm["p_two_sided"]:.4g}', flush=True)
        expected = n_distinct_relabelings([group_a] * len(present[group_a])
                                          + [group_b] * len(present[group_b]))
        if int(model_perm['n_relabelings']) != expected:
            raise RuntimeError(
                f'hierarchical_cell_amplitude_permutation: {group_a} vs {group_b} enumerated '
                f'{model_perm["n_relabelings"]} relabelings but the multinomial coefficient for '
                f'{len(present[group_a])}/{len(present[group_b])} mice is {expected}.')

        print(f'[hier   perm] {group_a} vs {group_b}: design-based sensitivity '
              f'(model-free, exact)...', flush=True)
        design_stat = make_contrast_stat(delta_df, HIERARCHICAL_CELL_DELTA_COL,
                                         group_a, group_b, weight='mouse')
        design_perm = mouse_label_permutation_test(design_stat, present, seed=seed,
                                                   restrict_to_groups=(group_a, group_b),
                                                   exact=True)
        design_global = (mouse_label_permutation_test(design_stat, present, seed=seed, exact=True)
                         if include_global_null_sensitivity else None)
        cell_stat = make_contrast_stat(delta_df, HIERARCHICAL_CELL_DELTA_COL, group_a, group_b,
                                       weight='cell')
        cell_perm = mouse_label_permutation_test(cell_stat, present, seed=seed,
                                                 restrict_to_groups=(group_a, group_b), exact=True)
        ci = mouse_contrast_ci({group_a: mouse_delta[group_a], group_b: mouse_delta[group_b]},
                               reference=group_b, scale='log', alpha=alpha)[group_a]

        rows.append({
            'block': 'pairwise_primary', 'group_a': group_a, 'group_b': group_b,
            'statistic_kind': 'three-group MixedLM contrast ' + ' + '.join(
                f'{w:+g}*{n}' for n, w in weights.items()),
            'estimate_lmm': contrast['estimate'], 'ci_lmm_low': contrast['ci_low'],
            'ci_lmm_high': contrast['ci_high'],
            'p_raw': float(model_perm['p_two_sided']), 'p_holm_pairwise': np.nan,
            'estimate_mouse_weighted': np.nan, 'ci_mouse_weighted_low': np.nan,
            'ci_mouse_weighted_high': np.nan, 'p_raw_design_based': np.nan,
            'p_raw_global_null': np.nan, 'estimate_cell_weighted': np.nan,
            'null_type': f'pairwise ({group_a}, {group_b})',
            'n_relabelings': int(model_perm['n_relabelings']), 'exact': True,
            'n_perm': int(model_perm['n_perm']), 'seed': seed,
            'n_mice_a': len(present[group_a]), 'n_mice_b': len(present[group_b]),
            'n_cells_a': int((delta_df['group'] == group_a).sum()),
            'n_cells_b': int((delta_df['group'] == group_b).sum()),
            'asymptotic_p_not_for_inference': contrast['p'],
        })
        rows.append({
            'block': 'pairwise_sensitivity', 'group_a': group_a, 'group_b': group_b,
            'statistic_kind': 'design-based: difference of group means of the per-mouse mean '
                              'paired-cell delta (model-free)',
            'estimate_lmm': np.nan, 'ci_lmm_low': np.nan, 'ci_lmm_high': np.nan,
            'p_raw': np.nan, 'p_holm_pairwise': np.nan,
            'estimate_mouse_weighted': float(design_perm['observed']),
            'ci_mouse_weighted_low': ci['diff_lo'], 'ci_mouse_weighted_high': ci['diff_hi'],
            'p_raw_design_based': float(design_perm['p_two_sided']),
            'p_raw_global_null': (float(design_global['p_two_sided'])
                                  if design_global is not None else np.nan),
            'estimate_cell_weighted': float(cell_perm['observed']),
            'p_raw_cell_weighted': float(cell_perm['p_two_sided']),
            'null_type': f'pairwise ({group_a}, {group_b}); global-null column alongside',
            'n_relabelings': int(design_perm['n_relabelings']), 'exact': True,
            'n_perm': int(design_perm['n_perm']), 'seed': seed,
            'n_mice_a': len(present[group_a]), 'n_mice_b': len(present[group_b]),
            'n_cells_a': int((delta_df['group'] == group_a).sum()),
            'n_cells_b': int((delta_df['group'] == group_b).sum()),
            'asymptotic_p_not_for_inference': np.nan,
        })

    out = pd.DataFrame(rows)
    # Holm over EXACTLY the three primary (model-based, exact) p-values. The design-based and
    # global-null columns are labelled sensitivities and enter no family: correcting both would
    # correct one question twice and halve the family's power (docs/sp_rates_lmm.md section 4.3).
    primary = out['block'] == 'pairwise_primary'
    _reject, p_adj = holm_correct(out.loc[primary, 'p_raw'].to_numpy(dtype=float), alpha=alpha)
    out.loc[primary, 'p_holm_pairwise'] = p_adj
    return out


def _post_indicator(epoch_series, response_epoch=RECALL_RESPONSE_EPOCH):
    """0/1 column for the post-tone epoch. Explicit rather than relying on a dummy-coded factor,
    because it is used as a random-SLOPE variable, where a categorical would be read on the wrong
    scale -- the same reasoning behind fit_rate_group_epoch_model's pasted `mouse_trial` label."""
    return (epoch_series.astype(str) == response_epoch).astype(float)


def fit_hierarchical_cell_unpaired_sensitivity(df_matched, save_dir,
                                               reference_epoch=RECALL_REFERENCE_EPOCH,
                                               response_epoch=RECALL_RESPONSE_EPOCH,
                                               filename=None):
    """
    SENSITIVITY: the long-format amplitude model over ALL epoch-defined cells, not just the paired
    intersection --

        log_mean_amplitude_cell_epoch ~ group * epoch
            + (1|mouse) + (0 + post_indicator|mouse) + (1|mouse:cell)

    where a cell contributes only the epochs in which its amplitude is defined.

    ** What it is for. ** The primary analysis conditions on cells active in BOTH windows (see
    build_recall_cell_amplitude_modulation). This asks whether that restriction is what produces
    the answer. Predefined and fit on every pass OF THE SUITE -- it is not switched on by what the
    primary produced. (The suite itself is explicitly invoked; see run_hierarchical_cell_suite.)

    ** Why the mouse epoch random slope is required HERE and not in the primary. ** This is a
    `group * epoch` model, so between-mouse variation in the pre->post CHANGE -- the very effect
    being compared across groups -- has nowhere to go without it, and the group x epoch terms
    would get intervals that are too narrow, with cells acting as replicates for the epoch
    contrast. The primary model is fit on the within-cell delta directly, where (1|mouse) already
    IS the mouse-specific modulation effect and no slope term is meaningful.

    Random effects are independent variance components: a mouse intercept, an uncorrelated
    mouse-level post-tone slope, and a cell intercept nested within mouse. No cell-level epoch
    slope -- that would be a further large expansion this analysis does not claim to resolve.

    ** Its asymptotic p-values are cell-level and enter NO family. ** Stated in the file header,
    not left to the reader.

    ** No silent simplification of the random-effect structure. ** Both requested variance
    components must appear in the fitted model or this raises -- a (1|mouse)-only fit must never
    be reported under this function's label. A variance estimate at or near the ZERO BOUNDARY is
    NOT a failure and does not raise: it is a legitimate statement about the data (little variance
    at that level), qualitatively different from the term having been dropped, and is reported
    explicitly as a boundary estimate.
    """
    epochs = (reference_epoch, response_epoch)
    frames = []
    for epoch in epochs:
        amp = filter_amplitude_rows(aggregate_over_trials(df_matched, epoch))
        frames.append(amp.assign(epoch=epoch))
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={'log_amplitude': 'log_mean_amplitude_cell_epoch'})
    df['post_indicator'] = _post_indicator(df['epoch'], response_epoch)
    df['group'] = pd.Categorical(
        df['group'], categories=[HIERARCHICAL_CELL_REFERENCE_GROUP]
        + [g for g in GROUP_ORDER if g != HIERARCHICAL_CELL_REFERENCE_GROUP])
    df['epoch'] = pd.Categorical(df['epoch'], categories=list(epochs))
    df['cell_key'] = df['cell'].astype(str)

    formula = (f'log_mean_amplitude_cell_epoch ~ '
               f'C(group, Treatment("{HIERARCHICAL_CELL_REFERENCE_GROUP}"))'
               f' * C(epoch, Treatment("{reference_epoch}"))')
    vc_formula = {'mouse_post_slope': '0 + post_indicator', 'mouse_cell': '0 + C(cell_key)'}
    model = smf.mixedlm(formula, data=df, groups=df['mouse'], re_formula='1',
                        vc_formula=vc_formula)
    result = model.fit(reml=True, method=HIERARCHICAL_CELL_LMM_OPTIMIZER)

    # The requested structure must be PRESENT. This checks the model that was built, not the size
    # of what it estimated.
    param_index = [str(n) for n in result.params.index]
    missing = [name for name in vc_formula if not any(name in p for p in param_index)]
    if missing:
        raise RuntimeError(
            f'fit_hierarchical_cell_unpaired_sensitivity: variance component(s) {missing} are '
            f'absent from the fitted model ({param_index}). A reduced random-effect structure '
            f'must never be reported under a (1|mouse) + (0+post|mouse) + (1|mouse:cell) label.')
    if not result.converged:
        raise RuntimeError(
            'fit_hierarchical_cell_unpaired_sensitivity: the mixed model did not converge. No '
            'simplified structure is substituted; stopping so the failure is visible.')

    boundary_notes = []
    for name in list(vc_formula) + ['Group Var']:
        matches = [p for p in param_index if name in p]
        for p in matches:
            val = float(result.params[p])
            if val <= 1e-8:
                boundary_notes.append(
                    f'  {p} = {val:.3g} -- AT/NEAR THE ZERO BOUNDARY. This is a statement about '
                    f'the data (these data support little variance at that level), NOT evidence '
                    f'that the term was dropped: it is present in the fitted model, which is '
                    f'checked separately. Read the corresponding fixed effects with that in mind.')
    header = (
        'SENSITIVITY -- hierarchical cell-level companion analysis, recall Test_B.\n'
        'Long-format amplitude model over ALL epoch-defined cells (a cell contributes only the\n'
        'epochs in which its amplitude is defined), the counterpart to the PAIRED primary\n'
        'analysis, which conditions on cells active in both epochs. Its purpose is to check\n'
        'whether that conditioning drives the result.\n\n'
        f'Formula: {formula}\n'
        f'Random effects: (1|mouse) + (0 + post_indicator|mouse) + (1|mouse:cell)\n'
        f'  vc_formula = {vc_formula}\n'
        f'Optimizer: {HIERARCHICAL_CELL_LMM_OPTIMIZER} (numerical only; see '
        f'HIERARCHICAL_CELL_LMM_OPTIMIZER)\n'
        f'N rows = {len(df)} cell x epoch observations, N cells = '
        f'{df.groupby(["mouse", "cell"], observed=True).ngroups}, N mice = '
        f'{df["mouse"].nunique()}\n\n'
        '** THE P-VALUES BELOW ARE ASYMPTOTIC OVER CELLS AND ARE NOT A PAPER-FACING INFERENCE. **\n'
        '16 animals were randomized, not N cells. This model supplies an ESTIMATE and a\n'
        'structural check only; it is a member of NO multiplicity family, and no figure reads a\n'
        'star from it. The inference for this suite is the mouse-label randomization in\n'
        f'{HIERARCHICAL_CELL_STATS_PREFIX}_amplitude_permutation.csv.\n')
    if boundary_notes:
        header += '\nBOUNDARY VARIANCE ESTIMATES (reported, not treated as failure):\n' \
                  + '\n'.join(boundary_notes) + '\n'
    text = f'{header}\n{result.summary()}\n'
    if filename is None:
        filename = f'{HIERARCHICAL_CELL_STATS_PREFIX}_amplitude_unpaired_sensitivity.txt'
    write_text(os.path.join(save_dir, filename), text)
    return {'result': result, 'summary_text': text, 'formula': formula, 'n_rows': int(len(df)),
            'boundary_notes': boundary_notes}


def build_recall_cell_epoch_count_table(df_matched, epochs=RECALL_EPOCHS):
    """
    The cell x epoch COUNT table the hierarchical rate model is fit on: one row per
    (mouse, group, cell_id, epoch) with `n_events` and `exposure_seconds`, over the same retained
    trials as everything else in this lane.

    ** Every detected cell is here, zero-event cells included. ** That is the whole point of a
    population rate, and it is what distinguishes this path from the paired-amplitude one, whose
    estimand is conditional on cells being active in both windows. Built from
    aggregate_over_trials BEFORE filter_amplitude_rows, so no activity filter is applied at all.

    ** No log of a zero rate and no pseudocount. ** The counts stay counts and the model carries
    the log link and the exposure offset itself, so a zero is an observation rather than something
    that has to be smoothed before it can be modelled.
    """
    frames = []
    for epoch in epochs:
        pooled = aggregate_over_trials(df_matched, epoch)
        frames.append(pooled.assign(epoch=epoch))
    out = pd.concat(frames, ignore_index=True).rename(columns={'cell': 'cell_id'})
    out = out[['mouse', 'group', 'cell_id', 'epoch', 'n_events', 'exposure_seconds']]
    if (out['exposure_seconds'] <= 0).any():
        raise RuntimeError('build_recall_cell_epoch_count_table: non-positive exposure.')
    if not (out['n_events'] >= 0).all():
        raise RuntimeError('build_recall_cell_epoch_count_table: negative event count.')
    n_expected = out.groupby('epoch', observed=True).size().unique()
    if len(n_expected) != 1:
        raise RuntimeError(
            f'build_recall_cell_epoch_count_table: the epochs cover different numbers of cells '
            f'{n_expected}. Both windows exist on every retained trial by construction.')
    out['group'] = pd.Categorical(out['group'], categories=list(DREADD_DISPLAY_ORDER))
    return out.sort_values(['group', 'mouse', 'cell_id', 'epoch']).reset_index(drop=True)


# ── The hierarchical cell-level RATE model ───────────────────────────────────────────────────
#
# The rate counterpart of the amplitude hierarchy: same group x epoch modulation question, asked
# on the cell-level COUNT structure with every zero-event cell retained, instead of on one
# collapsed population rate per animal. It is a REQUIRED component of this suite, not a
# contingency selected if the mouse-level rate analysis looks insufficient -- that analysis
# remains authoritative and untouched.

HIERARCHICAL_CELL_RATE_HDI_PROB = 0.95   # this suite's own choice; RATE_CONTRAST_HDI_PROB (0.94)
                                         # belongs to the TFC lane and is left alone.

# ** PRIORS ARE PART OF THIS MODEL AND ARE FROZEN BEFORE THE REAL DATA IS FITTED. ** Bambi assigns
# priors automatically from the data when none are given, which would make them a silent,
# data-dependent input to a model whose posterior intervals are the reported result. They are
# therefore stated here explicitly and written into the model's own output file, so a reported
# number is never separated from the prior that produced it.
#
# Scale: the log-exposure offset makes the intercept a log(events/second), so a unit change is a
# factor of e.
#   Intercept  Normal(-3, 2)   -- centred on ~0.05 events/s, the order of magnitude of sparse CA1
#                                 calcium-event rates on this event definition, with a 95% prior
#                                 interval of roughly 0.001-2.7 events/s. This encodes what a
#                                 calcium event IS, not what these data show.
#   common     Normal(0, 1)    -- group, epoch and group x epoch effects up to roughly e^+-2
#                                 (0.14x-7.4x), wide relative to any effect this field reports
#                                 while excluding physically impossible ones.
#   RE sigmas  HalfNormal(0.5) on the two MOUSE-level terms (between-animal spread in baseline
#                                 rate and in the pre->post change is the smaller of the two
#                                 scales, and the epoch slope over 16 animals is the least
#                                 well-identified parameter in the model, so it is the one most in
#                                 need of regularization);
#              HalfNormal(1)   on the CELL intercept, because between-cell rate heterogeneity
#                                 within an animal genuinely is large.
#   alpha      Gamma(2, 0.5)   -- NB dispersion, kept away from both the zero boundary and the
#                                 Poisson limit. Its meaning depends on the parameterization,
#                                 which is CONFIRMED programmatically rather than assumed --
#                                 see _confirm_nb_alpha_parameterization.
#
# ** THESE WERE REVISED ONCE, AT SPECIFICATION TIME, ON PRIOR SIMULATIONS ONLY. ** The first
# frozen set -- Intercept Normal(0, 5), common Normal(0, 2.5), all three RE sigmas HalfNormal(1) --
# FAILED its own prior-predictive check before any model was fit to the real data: it implied a
# median per-cell rate of 1.23 events/s and a 99th percentile of 4.5e6 events/s, which is not a
# wide prior but a physically impossible one (a cell cannot emit millions of calcium events per
# second). It was revised here, re-frozen, and re-checked. That is exactly the sequence
# hierarchical_cell_rate_prior_predictive exists to force, and it happened on prior draws alone --
# no posterior, and no real data, was consulted.
HIERARCHICAL_CELL_RATE_PRIORS = {
    'Intercept': bmb.Prior('Normal', mu=-3.0, sigma=2.0),
    'common': bmb.Prior('Normal', mu=0.0, sigma=1.0),
    '1|mouse': bmb.Prior('Normal', mu=0.0, sigma=bmb.Prior('HalfNormal', sigma=0.5)),
    'post_indicator|mouse': bmb.Prior('Normal', mu=0.0,
                                      sigma=bmb.Prior('HalfNormal', sigma=0.5)),
    '1|mouse_cell': bmb.Prior('Normal', mu=0.0, sigma=bmb.Prior('HalfNormal', sigma=1.0)),
    'alpha': bmb.Prior('Gamma', alpha=2.0, beta=0.5),
}

# Plausibility band the prior-predictive check is read against, in events/second per cell, for
# this event definition (one contiguous supra-threshold run of S) over 20 s windows. Stated as a
# constant so the check has a written criterion rather than an impression.
HIERARCHICAL_CELL_RATE_PLAUSIBLE_HZ = (1e-4, 5.0)

# ** FROZEN PRODUCTION SAMPLER CONFIGURATION. ** Module constants, not call-site defaults, so the
# reported fit cannot drift.
#
# target_accept: the project has no existing standard -- fit_rate_group_epoch_model passes none
# and inherits PyMC's 0.8 -- so rather than inherit a default silently this suite fixes 0.95,
# appropriate for a hierarchy carrying a per-cell random intercept and a mouse-level slope, where
# 0.8 is the classic source of divergences. The TFC lane is NOT changed and its numbers do not
# move.
#
# mp_ctx='spawn': NOT a statistical choice -- a process-management one, and it is here because its
# absence cost a three-hour hang. PyMC's default `fork` start method forks the calling process;
# run from a notebook kernel holding a loaded `ds` (~18.6 GB across ~30 threads) it deadlocked
# outright -- parent blocked in poll(), four chain workers created and then never scheduled, ZERO
# CPU across all five processes for 2h52m, nothing sampled. Forking a large multi-threaded process
# is a classic hang, and it had nothing to do with this model: the design matrices are sparse
# (0.9 MB total), the model builds in 0.7 s, its gradient compiles in 8.7 s, and one gradient
# evaluation costs 3.73 ms. `spawn` starts each worker fresh instead of inheriting that address
# space. Same model, same priors, same random_seed, same per-chain draws -- only the way the four
# chains are launched changes.
#
# ** No idata_kwargs={'log_likelihood': True} here, deliberately. ** The TFC lane's
# fit_rate_group_epoch_model needs the pointwise log-likelihood because it runs az.compare/LOO
# between a full and a reduced model. THIS suite does no model comparison and no LOO: its
# diagnostics are r_hat/ESS/divergences, its contrasts are posterior transforms, and its adequacy
# check is posterior-predictive. Requesting it would compute and store a 4000 x n_obs array
# (~492 MB here) that nothing ever reads -- the idiom was copied across from the TFC lane by
# mistake. Dropping it changes no reported quantity.
# progressbar=True is DISPLAY ONLY and affects no result. It is on because this is the single
# longest step in the suite and PyMC's bar is the only thing that reports draws/s and an ETA
# while it runs; with it off, a stalled sampler is indistinguishable from a slow one.
HIERARCHICAL_CELL_RATE_SAMPLER = dict(
    draws=1000, tune=1000, chains=4, random_seed=HIERARCHICAL_CELL_SEED,
    target_accept=0.95, progressbar=True, mp_ctx='spawn',
)

# ** CONVERGENCE GATE -- SCOPE FIXED BEFORE FITTING. ** Hard-failed on the parameters the
# scientific claims rest on; the thousands of nuisance per-cell intercepts are summarized in full
# and REPORTED but do not define failure, because a few poorly-identified singleton cells would
# otherwise condemn a model that is fine everywhere that matters. Divergences are gated GLOBALLY:
# a divergence is a property of the sampler's trajectory, not of any one parameter, so it is never
# scoped away.
HIERARCHICAL_CELL_RATE_GATE = dict(max_rhat=1.01, min_ess_bulk=400.0, min_ess_tail=400.0,
                                   max_divergences=0)


def _confirm_nb_alpha_parameterization(seed=HIERARCHICAL_CELL_SEED, n=400000):
    """Determine the installed backend's negative-binomial `alpha` convention empirically.

    ** Not a formality. ** A Gamma(2, 0.5) prior that is weakly informative under
    Var = mu + mu^2/alpha is strongly informative under its reciprocal Var = mu + alpha*mu^2, so
    the frozen dispersion prior cannot be interpreted -- or defended -- without knowing which one
    is in force. Read off the sampler rather than from memory or a docstring, and written into the
    model's summary file.

    Returns (convention_string, measured_variance, expected_under_that_convention).
    """
    mu, alpha = 5.0, 2.0
    draws = pm.draw(pm.NegativeBinomial.dist(mu=mu, alpha=alpha), draws=n,
                    random_seed=seed)
    measured = float(np.var(draws))
    quadratic = mu + mu ** 2 / alpha     # Var = mu + mu^2/alpha  (pymc/bambi convention)
    reciprocal = mu + alpha * mu ** 2    # Var = mu + alpha*mu^2
    if abs(measured - quadratic) < abs(measured - reciprocal):
        return ('Var = mu + mu^2/alpha  (larger alpha -> closer to Poisson)', measured, quadratic)
    return ('Var = mu + alpha*mu^2  (larger alpha -> more overdispersed)', measured, reciprocal)


def _cell_rate_model_frame(counts_df, reference_epoch=RECALL_REFERENCE_EPOCH,
                           response_epoch=RECALL_RESPONSE_EPOCH):
    """Model frame for the cell-level NB rate model: adds the log-exposure offset, the 0/1
    post-tone indicator used as the mouse-level random SLOPE variable, and the pasted
    `mouse_cell` grouping label.

    `mouse_cell` is `mouse + '_c' + cell_id`, not `cell_id`: cell ids are SESSION-LOCAL unit_ids
    and the same integer occurs in every animal, so grouping on the bare id would pool unrelated
    neurons across mice into one random effect. Same reasoning as fit_rate_group_epoch_model's
    pasted `mouse_trial`.
    """
    df = counts_df.copy()
    df['log_exposure'] = np.log(df['exposure_seconds'].to_numpy(dtype=float))
    df['post_indicator'] = _post_indicator(df['epoch'], response_epoch)
    df['mouse_cell'] = df['mouse'].astype(str) + '_c' + df['cell_id'].astype(str)
    df['group'] = df['group'].astype(str)
    df['epoch'] = df['epoch'].astype(str)
    df['n_events'] = df['n_events'].astype(int)
    if df.groupby(['mouse_cell', 'epoch'], observed=True).size().max() != 1:
        raise RuntimeError('_cell_rate_model_frame: a (mouse_cell, epoch) appears more than once.')
    return df


def _cell_rate_formula(reference_group=HIERARCHICAL_CELL_REFERENCE_GROUP,
                       reference_epoch=RECALL_REFERENCE_EPOCH):
    group_term = f"C(group, Treatment('{reference_group}'))"
    epoch_term = f"C(epoch, Treatment('{reference_epoch}'))"
    return (f"n_events ~ {group_term} * {epoch_term} + offset(log_exposure) "
            f"+ (1|mouse) + (0 + post_indicator|mouse) + (1|mouse_cell)"), group_term, epoch_term


def build_cell_epoch_rate_model(counts_df, reference_group=HIERARCHICAL_CELL_REFERENCE_GROUP,
                                reference_epoch=RECALL_REFERENCE_EPOCH,
                                response_epoch=RECALL_RESPONSE_EPOCH):
    """
    Build (do not fit) the hierarchical Negative-Binomial cell-count model, with the frozen priors
    attached and verified.

        n_events ~ group * epoch + offset(log(exposure))
                 + (1|mouse) + (0 + post_indicator|mouse) + (1|mouse_cell)

    ** Why the mouse-level epoch random slope is part of the frozen structure. ** (1|mouse) +
    (1|mouse_cell) model mouse and cell heterogeneity in BASELINE rate only. The quantity being
    compared across treatment groups is the pre->post CHANGE, and mice differ in that change too.
    Without a mouse-level epoch random effect, that between-mouse variation lands in the residual
    /cell level and the group x epoch fixed effects -- the entire scientific target -- get
    posterior intervals that are too narrow, with cells effectively serving as replicates for the
    epoch contrast. That is the count-model form of the same pseudoreplication the amplitude path
    guards against, so it is specified in advance rather than added after seeing a result.

    The slope is INDEPENDENT (uncorrelated) of the mouse intercept: with 16 mice there is little
    information about an intercept-slope correlation, and an unstructured 2x2 mouse covariance is
    the part of this model most likely to sample badly. There is deliberately NO cell-level epoch
    slope -- a further large expansion this analysis does not claim to resolve.

    Reference levels are named explicitly via C(col, Treatment('level')): formulae does not honour
    a pandas Categorical's category order the way patsy does (confirmed empirically in
    fit_rate_group_epoch_model, and unchanged here).

    Raises if any frozen prior failed to attach -- a silent fall-back to a Bambi default would
    make the priors data-dependent again, which is the whole thing this guards against.

    Returns dict(model, df, formula, group_term, epoch_term, priors_applied).
    """
    df = _cell_rate_model_frame(counts_df, reference_epoch, response_epoch)
    formula, group_term, epoch_term = _cell_rate_formula(reference_group, reference_epoch)
    model = bmb.Model(formula, data=df, family='negativebinomial',
                      priors=HIERARCHICAL_CELL_RATE_PRIORS)
    model.build()

    # ** Verify every frozen prior actually attached. ** Read off the BUILT model, never assumed:
    # a prior that silently fails to attach leaves a Bambi data-dependent default in its place,
    # which is precisely what specifying them was for.
    mu_component = model.components['mu']
    applied = {name: str(term.prior) for name, term in mu_component.terms.items()}
    applied['alpha'] = str(model.constant_components['alpha'].prior)

    expected = {'Intercept': HIERARCHICAL_CELL_RATE_PRIORS['Intercept'],
                '1|mouse': HIERARCHICAL_CELL_RATE_PRIORS['1|mouse'],
                'post_indicator|mouse': HIERARCHICAL_CELL_RATE_PRIORS['post_indicator|mouse'],
                '1|mouse_cell': HIERARCHICAL_CELL_RATE_PRIORS['1|mouse_cell'],
                'alpha': HIERARCHICAL_CELL_RATE_PRIORS['alpha']}
    for name, prior in expected.items():
        if name not in applied:
            raise RuntimeError(
                f'build_cell_epoch_rate_model: the frozen prior {name!r} matches no term in the '
                f'built model (terms: {sorted(applied)}).')
        if applied[name] != str(prior):
            raise RuntimeError(
                f'build_cell_epoch_rate_model: term {name!r} carries {applied[name]} but the '
                f'frozen prior is {prior}. The model must not fall back to a Bambi default.')
    # The 'common' catch-all must have reached every fixed effect except the intercept and offset.
    common_expected = str(HIERARCHICAL_CELL_RATE_PRIORS['common'])
    for name in (group_term, epoch_term, f'{group_term}:{epoch_term}'):
        if applied.get(name) != common_expected:
            raise RuntimeError(
                f'build_cell_epoch_rate_model: fixed effect {name!r} carries '
                f'{applied.get(name)} rather than the frozen common prior {common_expected}.')
    return {'model': model, 'df': df, 'formula': formula, 'group_term': group_term,
            'epoch_term': epoch_term, 'priors_applied': applied}


def hierarchical_cell_rate_prior_predictive(built, save_dir, draws=500,
                                            seed=HIERARCHICAL_CELL_SEED, filename=None):
    """
    PRIOR-PREDICTIVE simulation, run BEFORE the inferential fit.

    ** Mandatory, and it is about the priors, not the data. ** Priors specified on a log-rate scale
    are easy to state and hard to feel: the question this answers is whether they imply event
    rates a CA1 pyramidal cell could plausibly show over a 20 s window, or whether they put
    substantial mass on physically absurd ones. If the implied rates are not plausible the priors
    are revised and RE-FROZEN before the real model is fit -- a revision made at specification
    time on prior simulations only, never after seeing a posterior.

    Reports quantiles of the implied per-cell event RATE (counts / exposure) and of the raw counts.
    Written out rather than inspected interactively, so what the priors implied is on the record
    beside what the posterior said.
    """
    model, df = built['model'], built['df']
    idata = model.prior_predictive(draws=draws, random_seed=seed)
    var = list(idata.prior_predictive.data_vars)[0]
    counts = np.asarray(idata.prior_predictive[var].values, dtype=float).reshape(-1, len(df))
    exposure = df['exposure_seconds'].to_numpy(dtype=float)
    rates = counts / exposure[None, :]
    qs = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    rate_q = np.quantile(rates, qs)
    count_q = np.quantile(counts, qs)
    lo_hz, hi_hz = HIERARCHICAL_CELL_RATE_PLAUSIBLE_HZ
    median_rate = float(np.median(rates))
    p95_rate = float(np.quantile(rates, 0.95))
    # A written criterion, not an impression: the prior's CENTRAL mass must sit in a plausible
    # band for this event definition, and its upper bulk must not be physically impossible. The
    # tails of a deliberately wide prior are allowed to be wide; the middle is not allowed to be
    # absurd.
    plausible = (lo_hz <= median_rate <= hi_hz) and (p95_rate <= hi_hz * 100)
    verdict = (
        f'** PRIOR-PREDICTIVE VERDICT: PLAUSIBLE. ** Median implied rate {median_rate:.4g} '
        f'events/s sits\ninside the stated band {lo_hz:g}-{hi_hz:g} events/s and the upper bulk is '
        f'not physically absurd.\n'
        if plausible else
        f'** PRIOR-PREDICTIVE VERDICT: IMPLAUSIBLE. ** Median implied rate {median_rate:.4g} '
        f'events/s\n(95th percentile {p95_rate:.4g}) against a stated plausible band of '
        f'{lo_hz:g}-{hi_hz:g} events/s.\nThe priors must be REVISED AND RE-FROZEN before the '
        f'inferential fit -- at specification\ntime, on prior draws only, never after seeing a '
        f'posterior.\n')
    text = (
        'PRIOR-PREDICTIVE CHECK -- hierarchical cell-level rate model (recall Test_B).\n'
        'Run BEFORE the inferential fit. This is a check on the FROZEN PRIORS, not on the data:\n'
        'do they imply biologically plausible per-cell event rates over the 20 s windows?\n\n'
        f'Formula: {built["formula"]}\n'
        f'Prior draws: {draws} (seed {seed}) over {len(df)} cell x epoch rows.\n\n'
        'Frozen priors:\n'
        + '\n'.join(f'  {k}: {v}' for k, v in HIERARCHICAL_CELL_RATE_PRIORS.items()) + '\n\n'
        f'Stated plausibility band for this event definition: '
        f'{lo_hz:g}-{hi_hz:g} events/s per cell.\n\n'
        f'{verdict}\n'
        f'Implied per-cell event RATE (events/s), quantiles {qs}:\n'
        f'  {np.array2string(rate_q, precision=4, suppress_small=True)}\n'
        f'Implied per-cell COUNT per window, quantiles {qs}:\n'
        f'  {np.array2string(count_q, precision=2, suppress_small=True)}\n'
        f'  median rate {np.median(rates):.4g} events/s; '
        f'fraction of prior-predictive counts that are zero: {float((counts == 0).mean()):.3f}\n\n'
        'Reading it: the prior is deliberately wide, so the tails are wide too. What matters is\n'
        'that the central mass sits in a plausible range for sparse CA1 calcium events and that\n'
        'the prior does not concentrate on absurd rates. If it does, the priors are revised and\n'
        're-frozen HERE, before the inferential fit -- never after seeing the posterior.\n\n'
        'HISTORY: the first frozen prior set (Intercept Normal(0, 5), fixed effects Normal(0, 2.5),\n'
        'all three random-effect SDs HalfNormal(1)) FAILED this check -- median implied rate 1.23\n'
        'events/s, 99th percentile 4.5e6 events/s -- and was revised to the set above before any\n'
        'model was fit to the real data. See HIERARCHICAL_CELL_RATE_PRIORS.\n')
    if filename is None:
        filename = f'{HIERARCHICAL_CELL_STATS_PREFIX}_rate_prior_predictive.txt'
    write_text(os.path.join(save_dir, filename), text)
    if not plausible:
        raise RuntimeError(
            f'hierarchical_cell_rate_prior_predictive: the frozen priors imply a median per-cell '
            f'event rate of {median_rate:.4g} events/s (95th pct {p95_rate:.4g}) against a stated '
            f'plausible band of {lo_hz:g}-{hi_hz:g}. Revise and re-freeze '
            f'HIERARCHICAL_CELL_RATE_PRIORS before fitting -- this is a specification-time '
            f'decision made on prior draws, and it must not be made after seeing a posterior. '
            f'Full report: {filename}')
    return {'idata': idata, 'rate_quantiles': rate_q, 'count_quantiles': count_q, 'text': text,
            'plausible': plausible, 'median_rate_hz': median_rate}


def _cell_rate_gated_var_names(built):
    """The parameters the hard convergence gate is evaluated over -- fixed effects, the NB
    dispersion, every random-effect SCALE hyperparameter, the mouse-level intercepts and slopes.
    Deliberately excludes the thousands of individual `mouse_cell` intercepts, which are
    summarized and reported separately but do not define model failure (see
    HIERARCHICAL_CELL_RATE_GATE)."""
    return ['Intercept', built['group_term'], built['epoch_term'],
            f'{built["group_term"]}:{built["epoch_term"]}',
            'alpha',
            '1|mouse', '1|mouse_sigma',
            'post_indicator|mouse', 'post_indicator|mouse_sigma',
            '1|mouse_cell_sigma']


def fit_cell_epoch_rate_model(built, sampler=None):
    """
    Fit the hierarchical NB cell-count model and apply the predefined convergence gate.

    ** Required component, fixed configuration, no fallback. ** The sampler settings are the
    frozen HIERARCHICAL_CELL_RATE_SAMPLER; a caller may override them only for the synthetic
    verification suite, which is not a reported inferential fit. If sampling fails, or the gate
    fails, this RAISES with the full diagnostics. No substitute model is fit -- not a Poisson, not
    a quasi-Poisson, not a Gaussian approximation, not a reduced random-effect structure, not a
    pooled-cell test. A replacement rate path would be a separate decision, frozen from the
    beginning, not something improvised at the point of failure.

    ** The gate's SCOPE is fixed in advance ** (_cell_rate_gated_var_names): fixed effects, the NB
    dispersion, every random-effect scale hyperparameter, and the mouse-level intercepts and
    slopes. The thousands of individual `mouse_cell` intercepts are summarized in full and
    reported -- including how many exceed the thresholds and which -- but do not define failure,
    because a handful of poorly-identified singleton cells would otherwise condemn a model that is
    fine everywhere the claims live. Divergences are gated GLOBALLY at zero.

    Returns dict(idata, diagnostics_gated, diagnostics_cell_block, gate_pass, gate_text).
    """
    model = built['model']
    fit_kwargs = dict(HIERARCHICAL_CELL_RATE_SAMPLER if sampler is None else sampler)
    idata = model.fit(**fit_kwargs)

    gated_names = _cell_rate_gated_var_names(built)
    present = [n for n in gated_names if n in idata.posterior]
    missing = [n for n in gated_names if n not in idata.posterior]
    if missing:
        raise RuntimeError(
            f'fit_cell_epoch_rate_model: gated parameter(s) {missing} are absent from the '
            f'posterior ({sorted(idata.posterior.data_vars)}). A convergence gate over a subset '
            f'that does not exist is a gate in name only.')
    gated = _nb_convergence_diagnostics(idata, var_names=present)
    cell_block = _nb_convergence_diagnostics(idata, var_names=['1|mouse_cell'])
    cell_summary = az.summary(idata, var_names=['1|mouse_cell'])
    n_cell_over_rhat = int((cell_summary['r_hat'] > HIERARCHICAL_CELL_RATE_GATE['max_rhat']).sum())
    n_cell_under_ess = int(
        (cell_summary['ess_bulk'] < HIERARCHICAL_CELL_RATE_GATE['min_ess_bulk']).sum())

    failures = []
    if gated['max_rhat'] > HIERARCHICAL_CELL_RATE_GATE['max_rhat']:
        failures.append(f"max r_hat {gated['max_rhat']:.4f} > "
                        f"{HIERARCHICAL_CELL_RATE_GATE['max_rhat']} "
                        f"(worst: {gated['worst_rhat_param']})")
    if gated['min_ess_bulk'] < HIERARCHICAL_CELL_RATE_GATE['min_ess_bulk']:
        failures.append(f"min ess_bulk {gated['min_ess_bulk']:.0f} < "
                        f"{HIERARCHICAL_CELL_RATE_GATE['min_ess_bulk']:.0f} "
                        f"(worst: {gated['worst_ess_bulk_param']})")
    if gated['min_ess_tail'] < HIERARCHICAL_CELL_RATE_GATE['min_ess_tail']:
        failures.append(f"min ess_tail {gated['min_ess_tail']:.0f} < "
                        f"{HIERARCHICAL_CELL_RATE_GATE['min_ess_tail']:.0f}")
    # Divergences are NOT scoped -- a divergence belongs to the sampler's trajectory, not to a
    # parameter, so it cannot be attributed to the nuisance block and set aside.
    n_div = int(idata.sample_stats['diverging'].values.sum())
    if n_div > HIERARCHICAL_CELL_RATE_GATE['max_divergences']:
        failures.append(f"{n_div} divergences > "
                        f"{HIERARCHICAL_CELL_RATE_GATE['max_divergences']} (gated globally)")

    gate_text = (
        'CONVERGENCE GATE (scope fixed before fitting; see HIERARCHICAL_CELL_RATE_GATE)\n'
        f'  Gated over {gated["n_params"]} parameters: fixed effects, NB alpha, random-effect\n'
        f'  scale hyperparameters, mouse intercepts and mouse epoch slopes.\n'
        f'    max r_hat      = {gated["max_rhat"]:.4f}  (worst: {gated["worst_rhat_param"]})\n'
        f'    min ess_bulk   = {gated["min_ess_bulk"]:.0f}  '
        f'(worst: {gated["worst_ess_bulk_param"]})\n'
        f'    min ess_tail   = {gated["min_ess_tail"]:.0f}\n'
        f'    divergences    = {n_div}   (gated GLOBALLY, never scoped away)\n'
        f'  REPORTED BUT NOT GATED -- the {cell_block["n_params"]} individual mouse_cell '
        f'intercepts:\n'
        f'    max r_hat      = {cell_block["max_rhat"]:.4f}  '
        f'(worst: {cell_block["worst_rhat_param"]})\n'
        f'    min ess_bulk   = {cell_block["min_ess_bulk"]:.0f}\n'
        f'    {n_cell_over_rhat} cell intercepts exceed r_hat '
        f'{HIERARCHICAL_CELL_RATE_GATE["max_rhat"]}; {n_cell_under_ess} fall below ess_bulk '
        f'{HIERARCHICAL_CELL_RATE_GATE["min_ess_bulk"]:.0f}.\n'
        f'    These are reported so a SYSTEMATICALLY badly-mixed random-effect block stays '
        f'visible;\n    they are excluded from the pass/fail decision, not from the record.\n')
    if failures:
        raise RuntimeError(
            'fit_cell_epoch_rate_model: the hierarchical NB rate model FAILED its predefined '
            'convergence gate:\n  - ' + '\n  - '.join(failures) + '\n\n' + gate_text +
            '\nNo substitute model is fit and no weaker test is run in its place. Stopping so '
            'the failure is visible and can be diagnosed.')
    return {'idata': idata, 'diagnostics_gated': gated, 'diagnostics_cell_block': cell_block,
            'n_divergent': n_div, 'n_cell_over_rhat': n_cell_over_rhat,
            'n_cell_under_ess': n_cell_under_ess, 'gate_text': gate_text,
            'sampler': fit_kwargs}


def summarize_cell_rate_contrasts(idata, built, hdi_prob=HIERARCHICAL_CELL_RATE_HDI_PROB):
    """
    Posterior contrasts of the hierarchical NB rate model, formed DRAW BY DRAW so the posterior
    covariance between coefficients is preserved (a delta-method or SE-subtraction version of the
    hM3D-vs-hM4D comparison would be neither).

    ** Two DIFFERENT quantities, kept apart because conflating them is easy and wrong: **

      block='modulation'  -- the pre->post MODULATION ratio-of-ratios, which is the group x epoch
                             question this suite asks and is the INTERACTION ALONE:
                                 exp(beta_int[Exc]), exp(beta_int[Inh]),
                                 exp(beta_int[Exc] - beta_int[Inh]).
                             "How much more did this group's event rate change from pre to post
                             than the comparison group's did."

      block='post_tone_simple_effect' -- the ABSOLUTE post-tone group-vs-control rate ratio,
                             exp(beta_group + beta_int). "How much higher was this group's rate
                             than control's during post-tone." This is what
                             summarize_rate_group_epoch_contrasts computes for the TFC lane; that
                             function's sampling and HDI mechanics are the model for this one, but
                             its CONTRAST DEFINITION is not the modulation quantity and must not
                             be read as it.

    ** No frequentist p-value is manufactured from any of this. ** A posterior is not a star: this
    supplies no asterisk to any figure and changes no existing paper-facing number.
    """
    posterior = idata.posterior
    group_term, epoch_term = built['group_term'], built['epoch_term']
    interaction_term = f'{group_term}:{epoch_term}'
    for name in (group_term, interaction_term):
        if name not in posterior:
            raise KeyError(f'summarize_cell_rate_contrasts: {name!r} is not in the posterior '
                           f'({sorted(posterior.data_vars)}).')

    def _draws(term, level):
        return np.asarray(posterior[term].sel({f'{term}_dim': level}).values).ravel()

    treatments = [g for g in DREADD_DISPLAY_ORDER if g != HIERARCHICAL_CELL_REFERENCE_GROUP]
    inter_level = {g: f'{g}, {RECALL_RESPONSE_EPOCH}' for g in treatments}
    beta_int = {g: _draws(interaction_term, inter_level[g]) for g in treatments}
    beta_group = {g: _draws(group_term, g) for g in treatments}

    rows = []
    for group_a, group_b in HIERARCHICAL_CELL_PAIRS:
        if group_b == HIERARCHICAL_CELL_REFERENCE_GROUP:
            log_ratio = beta_int[group_a]
            definition = f'exp(interaction[{group_a}])'
        else:
            log_ratio = beta_int[group_a] - beta_int[group_b]
            definition = f'exp(interaction[{group_a}] - interaction[{group_b}])'
        draws = np.exp(log_ratio)
        lo, hi = az.hdi(draws, hdi_prob=hdi_prob)
        rows.append({'block': 'modulation', 'group_a': group_a, 'group_b': group_b,
                     'quantity': 'pre->post modulation ratio-of-ratios', 'definition': definition,
                     'posterior_median': float(np.median(draws)), 'hdi_low': float(lo),
                     'hdi_high': float(hi), 'hdi_prob': hdi_prob,
                     'p_direction': float(max((log_ratio > 0).mean(), (log_ratio < 0).mean()))})

    for group in treatments:
        draws = np.exp(beta_group[group] + beta_int[group])
        lo, hi = az.hdi(draws, hdi_prob=hdi_prob)
        rows.append({'block': 'post_tone_simple_effect', 'group_a': group,
                     'group_b': HIERARCHICAL_CELL_REFERENCE_GROUP,
                     'quantity': 'ABSOLUTE post-tone rate ratio vs control',
                     'definition': f'exp(group[{group}] + interaction[{group}])',
                     'posterior_median': float(np.median(draws)), 'hdi_low': float(lo),
                     'hdi_high': float(hi), 'hdi_prob': hdi_prob,
                     'p_direction': np.nan})

    out = pd.DataFrame(rows)
    out['note'] = ('modulation = interaction only (the group x epoch question); '
                   'post_tone_simple_effect = absolute post-tone level. Different quantities.')
    return out


def hierarchical_cell_rate_posterior_predictive(idata, built, save_dir, save_figure=True,
                                                filename=None, figure_root=None):
    """
    POSTERIOR-PREDICTIVE adequacy checks, by group x epoch: the count distribution, the ZERO
    FRACTION, and overdispersion (variance-to-mean).

    ** Part of adequacy, not an optional extra. ** Convergence diagnostics say the sampler
    explored the posterior of THIS model; they say nothing about whether this model can produce
    data that look like the observed counts. A negative-binomial that cannot reproduce the
    observed fraction of silent cells is not describing the population it is being used to make
    statements about, and that has to be visible next to the estimates rather than discoverable.

    A model that fails these checks is REPORTED AS INADEQUATE in plain words beside its estimates.
    The estimates are not quietly presented as though the model fit.

    Returns dict(table, text, inadequate) -- `table` carries, per group x epoch, the observed
    statistic, its posterior-predictive interval and a Bayesian p-value (the posterior-predictive
    probability of a replicate at least as extreme; values near 0 or 1 indicate misfit).
    """
    model, df = built['model'], built['df']
    if 'posterior_predictive' not in idata:
        model.predict(idata, kind='response')
    var = list(idata.posterior_predictive.data_vars)[0]
    rep = np.asarray(idata.posterior_predictive[var].values, dtype=float).reshape(-1, len(df))
    obs = df['n_events'].to_numpy(dtype=float)

    def _stats(x):
        mean = x.mean(axis=-1)
        var_ = x.var(axis=-1)
        return {'zero_fraction': (x == 0).mean(axis=-1), 'mean_count': mean,
                'variance_to_mean': np.divide(var_, mean, out=np.full_like(mean, np.nan),
                                              where=mean > 0),
                'max_count': x.max(axis=-1)}

    rows = []
    for (group, epoch), sub in df.groupby(['group', 'epoch'], observed=True):
        idx = sub.index.to_numpy()
        obs_stats = _stats(obs[idx][None, :])
        rep_stats = _stats(rep[:, idx])
        for stat in ('zero_fraction', 'mean_count', 'variance_to_mean', 'max_count'):
            o = float(obs_stats[stat][0])
            r = np.asarray(rep_stats[stat], dtype=float)
            r = r[np.isfinite(r)]
            bayes_p = float((r >= o).mean()) if r.size else np.nan
            lo, hi = (float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975))) \
                if r.size else (np.nan, np.nan)
            rows.append({'group': group, 'epoch': epoch, 'statistic': stat, 'n_cells': len(idx),
                         'observed': o, 'ppc_median': float(np.median(r)) if r.size else np.nan,
                         'ppc_lo_2.5': lo, 'ppc_hi_97.5': hi, 'bayes_p': bayes_p,
                         'observed_in_interval': bool(lo <= o <= hi) if np.isfinite(lo) else False})
    table = pd.DataFrame(rows)

    key = table[table['statistic'].isin(['zero_fraction', 'variance_to_mean'])]
    failed = key[~key['observed_in_interval']]
    inadequate = len(failed) > 0
    verdict = (
        '** MODEL ADEQUACY: FAILED. ** The observed value falls OUTSIDE the 95% '
        'posterior-predictive\ninterval for the following group x epoch cells:\n'
        + failed[['group', 'epoch', 'statistic', 'observed', 'ppc_lo_2.5', 'ppc_hi_97.5']]
        .to_string(index=False)
        + '\n\nThis model does not reproduce the observed count structure it is being used to\n'
          'make statements about. Its estimates below are reported WITH this failure, not as\n'
          'though the model fit. Convergence diagnostics do not substitute for this check.\n'
        if inadequate else
        '** MODEL ADEQUACY: PASSED. ** Every observed zero fraction and variance-to-mean ratio\n'
        'falls inside its 95% posterior-predictive interval, per group x epoch.\n')
    text = (
        'POSTERIOR-PREDICTIVE CHECKS -- hierarchical cell-level NB rate model (recall Test_B).\n'
        'Observed vs replicated counts BY GROUP x EPOCH: zero fraction, mean count,\n'
        'variance-to-mean (overdispersion) and maximum count.\n\n'
        f'Formula: {built["formula"]}\n'
        f'{len(df)} cell x epoch rows, {rep.shape[0]} posterior-predictive replicates.\n\n'
        f'{verdict}\n'
        'bayes_p is the posterior-predictive probability of a replicate at least as large as the\n'
        'observation; values near 0 or 1 indicate misfit, values near 0.5 indicate agreement.\n\n'
        f'{table.to_string(index=False)}\n')
    if filename is None:
        filename = f'{HIERARCHICAL_CELL_STATS_PREFIX}_rate_posterior_predictive.txt'
    write_text(os.path.join(save_dir, filename), text)

    if save_figure:
        zero = table[table['statistic'] == 'zero_fraction']
        fig, ax = plt.subplots(figsize=(4.6, 3.2))
        ax.spines[['right', 'top']].set_visible(False)
        xs = np.arange(len(zero))
        ax.vlines(xs, zero['ppc_lo_2.5'], zero['ppc_hi_97.5'], color='0.6', linewidth=3,
                  label='95% posterior-predictive')
        ax.plot(xs, zero['observed'], 'o', color='k', markersize=5, label='observed')
        ax.set_xticks(xs)
        ax.set_xticklabels([f'{g}\n{e}' for g, e in zip(zero['group'], zero['epoch'])],
                           size='xx-small')
        ax.set_ylabel('fraction of cells with zero events', size='small')
        ax.set_title('Posterior-predictive zero fraction\nhierarchical cell-level NB rate model',
                     size='small')
        ax.legend(fontsize='xx-small', frameon=False)
        fig.tight_layout()
        if figure_root is None:
            figure_root = f'{HIERARCHICAL_CELL_STATS_PREFIX}_rate_posterior_predictive'
        save_fig(fig, os.path.join(save_dir, figure_root + '.png'))
        plt.close(fig)
    return {'table': table, 'text': text, 'inadequate': inadequate, 'verdict': verdict}


def _hierarchical_pairwise_lookup(perm_df, block, group_a, group_b, column):
    """One value out of the permutation table, by (block, comparison, column). Raises on a miss
    rather than returning NaN: a figure whose bracket silently became NaN because a lookup key
    drifted would render as 'not significant', which is a wrong claim, not a missing one."""
    sub = perm_df[(perm_df['block'] == block) & (perm_df['group_a'] == group_a)
                  & (perm_df['group_b'] == group_b)]
    if len(sub) != 1:
        raise RuntimeError(f'_hierarchical_pairwise_lookup: {len(sub)} rows for block={block!r} '
                           f'{group_a} vs {group_b}; expected exactly 1.')
    return float(sub.iloc[0][column])


def plot_hierarchical_cell_amplitude_modulation(delta_df, perm_df, eligibility_df, save_dir,
                                                session_label, filename_root):
    """
    Cell-level SuperPlot of the PAIRED pre->post amplitude modulation.

    ** The cloud is descriptive; the brackets are looked up. ** Every eligible cell is drawn, faint
    and coloured by mouse, with each animal's mean cell delta as a large black-edged marker. Those
    markers are a DESCRIPTIVE summary: the primary statistic is the paired-cell MixedLM
    coefficient, which is not the unweighted mean of the markers, and the footer says so. The
    brackets come from _precomputed_stat_fn fed the three Holm-adjusted EXACT randomization
    p-values of that model coefficient, so no statistic is ever computed from the plotted arrays.

    ** All three brackets are populated ** because all three comparisons are declared members of
    this panel's own Holm family -- exactly the condition _precomputed_stat_fn's docstring sets for
    supplying the hM3D-vs-hM4D slot. The question here is symmetric across the three groups,
    unlike the absolute-level paper panels where that comparison is in no family.

    ** yscale='linear' is mandatory. ** The plotted column is already a log-scale DIFFERENCE and
    takes both signs; 'auto' would apply a second log (see _draw_cell_superplot_panel's warning).

    The footer states n = 16 mice, the eligible-cell count and the eligibility fraction, and that
    the estimand is conditional on cells with amplitude defined in BOTH epochs -- never
    'n = thousands of cells'.
    """
    n_mice = int(delta_df['mouse'].nunique())
    n_cells = int(len(delta_df))
    frac = eligibility_df['fraction_eligible_both']
    omnibus_p = _hierarchical_pairwise_lookup(perm_df, 'omnibus', 'all', 'all', 'p_raw')
    stat_fn = _precomputed_stat_fn(
        _hierarchical_pairwise_lookup(perm_df, 'pairwise_primary', 'hM3D', 'mCherry',
                                      'p_holm_pairwise'),
        _hierarchical_pairwise_lookup(perm_df, 'pairwise_primary', 'hM4D', 'mCherry',
                                      'p_holm_pairwise'),
        _hierarchical_pairwise_lookup(perm_df, 'pairwise_primary', 'hM3D', 'hM4D',
                                      'p_holm_pairwise'))

    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    _draw_cell_superplot_panel(
        ax, delta_df, HIERARCHICAL_CELL_DELTA_COL,
        ylabel='Δ log per-event amplitude\n(post-tone − pre-tone), per cell',
        panel_name='hierarchical_cell_amplitude_modulation',
        title='Paired cell-level modulation\nomnibus permutation '
              f'{format_p_display(omnibus_p)}',
        yscale='linear', annotate='stats', ci_scale='log', bracket_mode='axes',
        stat_fn=stat_fn)
    # No change is zero on this scale, and it is the only reference the eye needs.
    ax.axhline(0.0, color='k', linewidth=0.8, linestyle='--', zorder=1)

    fig.suptitle(f'{session_label} — hierarchical cell-level companion analysis', size='small')
    fig.text(0.5, 0.015,
             f'n = {n_mice} mice — the unit of treatment assignment and of inference. '
             f'{n_cells} eligible cells drawn descriptively\n'
             f'(per-animal eligible fraction {frac.min():.2f}–{frac.max():.2f}); the estimand is '
             f'CONDITIONAL on cells with amplitude defined in BOTH epochs.\n'
             f'Cell cloud and per-mouse markers are descriptive. Brackets are Holm-adjusted exact '
             f'mouse-label randomization P-values\nof the paired-cell mixed-model coefficient '
             f'(`stats/{HIERARCHICAL_CELL_STATS_PREFIX}_amplitude_permutation.csv`); no statistic '
             f'is computed from the plotted values.',
             ha='center', size='xx-small')
    fig.subplots_adjust(left=0.17, bottom=0.30, right=0.97, top=0.83)
    ensure_dirs(save_dir)
    save_fig(fig, os.path.join(save_dir, filename_root + '.png'))
    plt.close(fig)


def plot_hierarchical_cell_rate_modulation(counts_df, contrasts_df, save_dir, session_label,
                                           filename_root,
                                           reference_epoch=RECALL_REFERENCE_EPOCH,
                                           response_epoch=RECALL_RESPONSE_EPOCH):
    """
    Cell-level pre->post EVENT-RATE change, zeros retained.

    ** Zeros are the point. ** The paired-amplitude panel above necessarily shows only cells active
    in both windows; this one shows every detected cell, including those that were silent in one
    or both epochs, which is the structure the count model is fit on.

    ** No stars. ** _precomputed_stat_fn is handed NaNs, so no bracket is drawn: the inferential
    statement here is a posterior, and a posterior is not an asterisk. The NB model's modulation
    rate ratios and their 95% HDIs go in the footer and the companion markdown instead.

    `y_quantum` is set from the window exposure -- a per-cell count over a fixed window lands on a
    handful of hard horizontal lines otherwise (see draw_superplot_triplet's y_quantum).
    """
    wide = counts_df.pivot_table(index=['mouse', 'group', 'cell_id'], columns='epoch',
                                 values=['n_events', 'exposure_seconds'], observed=True)
    rate_pre = wide[('n_events', reference_epoch)] / wide[('exposure_seconds', reference_epoch)]
    rate_post = wide[('n_events', response_epoch)] / wide[('exposure_seconds', response_epoch)]
    cells = pd.DataFrame({'delta_rate_hz': (rate_post - rate_pre).to_numpy()},
                         index=wide.index).reset_index()
    cells['group'] = pd.Categorical(cells['group'], categories=list(DREADD_DISPLAY_ORDER))
    exposure = float(counts_df['exposure_seconds'].median())

    modulation = contrasts_df[contrasts_df['block'] == 'modulation']
    ratio_note = '; '.join(
        f"{row['group_a']} vs {row['group_b']}: {row['posterior_median']:.3f} "
        f"[{row['hdi_low']:.3f}, {row['hdi_high']:.3f}]"
        for _, row in modulation.iterrows())

    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    _draw_cell_superplot_panel(
        ax, cells, 'delta_rate_hz',
        ylabel='Δ event rate (events/s)\n(post-tone − pre-tone), per cell',
        panel_name='hierarchical_cell_rate_modulation',
        title='Cell-level rate modulation\n(all detected cells; zeros retained)',
        yscale='linear', y_quantum=1.0 / exposure, annotate='stats', bracket_mode='axes',
        stat_fn=_precomputed_stat_fn(np.nan, np.nan, np.nan))
    ax.axhline(0.0, color='k', linewidth=0.8, linestyle='--', zorder=1)

    fig.suptitle(f'{session_label} — hierarchical cell-level companion analysis', size='small')
    fig.text(0.5, 0.015,
             f'n = {cells["mouse"].nunique()} mice; {len(cells)} cells drawn descriptively, '
             f'including zero-event cells.\nDeliberately carries NO significance brackets: the '
             f'inferential statement is the hierarchical NB count model\'s\nposterior, and a '
             f'posterior is not a star. Posterior pre→post modulation rate ratio [95% HDI] —\n'
             f'{ratio_note}.\n'
             f'See `stats/{HIERARCHICAL_CELL_STATS_PREFIX}_rate_contrasts.csv` and '
             f'`..._rate_nb_summary.txt`.',
             ha='center', size='xx-small')
    fig.subplots_adjust(left=0.17, bottom=0.32, right=0.97, top=0.83)
    ensure_dirs(save_dir)
    save_fig(fig, os.path.join(save_dir, filename_root + '.png'))
    plt.close(fig)
    return cells


def write_hierarchical_cell_vs_mouse_summary(save_dir, session_key, session_label, fit, perm_df,
                                             eligibility_df, interactions_table, pairwise_df,
                                             rate_contrasts=None, rate_fit=None, ppc=None,
                                             filename='hierarchical_cell_vs_mouse_level_summary.md'):
    """
    The two analyses side by side, per comparison, with every existing number READ from the frames
    the recall lane already produced -- never hard-coded.

    ** The purpose is a question, not a verdict: ** does using cell-level hierarchical information
    materially sharpen, weaken, or leave unchanged the biological conclusion? The smaller p-value
    is NOT automatically preferred, and the file says so. The mouse-level analysis remains the
    primary paper-facing one; this is companion evidence.
    """
    def _existing(outcome, group_a, group_b):
        return recall_modulation_lookup(pairwise_df, outcome, group_a, group_b)

    amp_omnibus = interactions_table[interactions_table['outcome'] == 'amplitude'].iloc[0]
    rate_omnibus = interactions_table[interactions_table['outcome'] == 'population_rate'].iloc[0]
    new_omnibus = _hierarchical_pairwise_lookup(perm_df, 'omnibus', 'all', 'all', 'p_raw')
    n_mice = fit['n_mice']

    lines = [
        f'# Hierarchical cell-level companion analysis vs the mouse-summary analysis '
        f'— {session_label}',
        '',
        '**This is a companion / sensitivity analysis. It replaces nothing.** The mouse-level '
        'models, their contrasts, their Holm families, their omnibus tests and every existing '
        'figure and table are unchanged and remain the paper-facing analysis. Everything in the '
        '"hierarchical-cell" column below is additive evidence.',
        '',
        f'**n = {n_mice} mice** in both analyses — that is the number of independently assigned '
        f'experimental units, and it does not change because cells were modelled. Cell counts '
        f'below are descriptive.',
        '',
        '## Amplitude',
        '',
        '### Omnibus',
        '',
        '| analysis | test | result |',
        '|---|---|---|',
        f'| mouse-summary (existing, authoritative) | group × epoch joint Wald | '
        f'{format_unified_interaction(amp_omnibus)} |',
        f'| hierarchical-cell (companion) | mouse-label permutation of the 2-df joint Wald '
        f'statistic of `delta ~ group + (1\\|mouse)` | {format_p_display(new_omnibus)} |',
        '',
        '### Pairwise',
        '',
        '| comparison | mouse-summary ratio [95% CI] | mouse-summary raw P | mouse-summary Holm P '
        '| hierarchical Δlog estimate [95% CI] | exact randomization raw P | Holm P | '
        'design-based sensitivity P |',
        '|---|---|---|---|---|---|---|---|',
    ]
    for group_a, group_b in HIERARCHICAL_CELL_PAIRS:
        old = _existing('amplitude', group_a, group_b)
        est = _hierarchical_pairwise_lookup(perm_df, 'pairwise_primary', group_a, group_b,
                                            'estimate_lmm')
        lo = _hierarchical_pairwise_lookup(perm_df, 'pairwise_primary', group_a, group_b,
                                           'ci_lmm_low')
        hi = _hierarchical_pairwise_lookup(perm_df, 'pairwise_primary', group_a, group_b,
                                           'ci_lmm_high')
        p_raw = _hierarchical_pairwise_lookup(perm_df, 'pairwise_primary', group_a, group_b,
                                              'p_raw')
        p_holm = _hierarchical_pairwise_lookup(perm_df, 'pairwise_primary', group_a, group_b,
                                               'p_holm_pairwise')
        p_design = _hierarchical_pairwise_lookup(perm_df, 'pairwise_sensitivity', group_a, group_b,
                                                 'p_raw_design_based')
        lines.append(
            f'| {group_a} vs {group_b} | {old["relative_modulation_ratio"]:.3f} '
            f'[{old["ratio_ci_low"]:.3f}, {old["ratio_ci_high"]:.3f}] | '
            f'{format_p_display(float(old["p_raw"]))} | '
            f'{format_p_display(float(old["p_holm_modulation"]))} | '
            f'{est:.3f} [{lo:.3f}, {hi:.3f}] (ratio {np.exp(est):.3f}) | '
            f'{format_p_display(p_raw)} | {format_p_display(p_holm)} | '
            f'{format_p_display(p_design)} |')

    lines += [
        '',
        '**Interval provenance.** The hierarchical Δlog estimate and its CI are the paired-cell '
        'MixedLM contrast and its own Wald interval — the same specification the randomization '
        'test permutes. The design-based sensitivity column is a *different estimator* (the '
        'difference of group means of the 16 per-mouse mean cell deltas) with its own '
        'equal-mouse-weighted interval, tabulated in the permutation CSV; the two are never '
        'cross-paired. The randomization P-values carry no interval of their own.',
        '',
        '**Estimand.** The mouse-summary amplitude value at each epoch averages over the cells '
        'active *in that epoch*. The hierarchical-cell estimate is **conditional on cells with '
        'measurable amplitude in BOTH epochs** — the price of within-cell pairing, and a real '
        'restriction, since eligibility is selected on activity in both windows and could itself '
        'differ by group. Per-animal eligible fractions:',
        '',
        '| group | mice | eligible fraction (min–median–max) | eligible cells |',
        '|---|---|---|---|',
    ]
    for group in DREADD_DISPLAY_ORDER:
        sub = eligibility_df[eligibility_df['group'] == group]
        if sub.empty:
            continue
        f = sub['fraction_eligible_both']
        lines.append(f'| {group} | {len(sub)} | {f.min():.3f} – {f.median():.3f} – {f.max():.3f} '
                     f'| {int(sub["n_cells_amp_defined_both"].sum())} |')

    lines += [
        '',
        '## Population rate',
        '',
        '| analysis | test | result |',
        '|---|---|---|',
        f'| mouse-summary (existing, authoritative) | group × epoch joint Wald on one collapsed '
        f'population rate per animal per epoch | {format_unified_interaction(rate_omnibus)} |',
    ]
    if rate_contrasts is not None:
        lines.append(
            '| hierarchical-cell (companion) | Bayesian NB count model over all cells including '
            'zero-event cells, `n_events ~ group * epoch + offset(log exposure) + (1\\|mouse) + '
            '(0 + post\\|mouse) + (1\\|mouse:cell)` | posterior modulation rate ratios below |')
        lines += [
            '',
            '**No frequentist P is manufactured from the posterior**, and this model supplies no '
            'star to any figure and changes no existing paper-facing number.',
            '',
            '| comparison | pre→post modulation rate ratio [95% HDI] | definition |',
            '|---|---|---|',
        ]
        for _, row in rate_contrasts[rate_contrasts['block'] == 'modulation'].iterrows():
            lines.append(f'| {row["group_a"]} vs {row["group_b"]} | '
                         f'{row["posterior_median"]:.3f} [{row["hdi_low"]:.3f}, '
                         f'{row["hdi_high"]:.3f}] | `{row["definition"]}` |')
        lines += [
            '',
            '| group | ABSOLUTE post-tone rate ratio vs control [95% HDI] |',
            '|---|---|',
        ]
        for _, row in rate_contrasts[
                rate_contrasts['block'] == 'post_tone_simple_effect'].iterrows():
            lines.append(f'| {row["group_a"]} | {row["posterior_median"]:.3f} '
                         f'[{row["hdi_low"]:.3f}, {row["hdi_high"]:.3f}] |')
        lines += [
            '',
            'The two blocks are **different quantities**: the modulation ratio is the interaction '
            'alone (how much more this group\'s rate changed pre→post than the comparison '
            'group\'s), the post-tone ratio is the absolute level during post-tone. They can '
            'point in different directions and neither is a restatement of the other.',
        ]
        if rate_fit is not None:
            d = rate_fit['diagnostics_gated']
            lines += [
                '',
                f'**Convergence (gated scope):** max r_hat {d["max_rhat"]:.4f}, min ess_bulk '
                f'{d["min_ess_bulk"]:.0f}, min ess_tail {d["min_ess_tail"]:.0f}, '
                f'{rate_fit["n_divergent"]} divergences. The individual `mouse_cell` intercepts '
                f'are reported in the model summary but do not define the gate.',
            ]
        if ppc is not None:
            adequacy = ('FAILED — see the posterior-predictive file; the estimates above are '
                        'reported WITH that failure, not as though the model fit.'
                        if ppc['inadequate'] else
                        'passed (observed zero fraction and variance-to-mean inside their 95% '
                        'posterior-predictive intervals, per group × epoch).')
            lines += ['', f'**Posterior-predictive adequacy:** {adequacy}']
    else:
        lines.append('| hierarchical-cell (companion) | hierarchical NB count model | '
                     'not available in this run |')

    lines += [
        '',
        '## Reading this comparison',
        '',
        'The question is whether retaining cell-level hierarchical information **sharpens, '
        'weakens, or leaves unchanged** the biological conclusion — not which column produced the '
        'smaller number. A smaller P here is not automatically the better answer, and it is not '
        'grounds for replacing the mouse-level result: the two analyses answer slightly different '
        'questions on slightly different cell sets, and the hierarchical one buys its precision '
        'from within-cell pairing, not from additional animals.',
        '',
        '**What legitimises the cell information.** Cells contribute information about each '
        'animal\'s cellular response distribution, while treatment assignment stays at the mouse: '
        'the model is refit under relabelings of the 16 **mouse** labels, with every cell fixed to '
        'its own animal. What would be pseudoreplication — treating thousands of cells as '
        'independently randomized treatment replicates — is exactly what the randomization scheme '
        'prevents, and is why no asymptotic cell-level z-value from any model here is reported as '
        'an inferential result.',
        '',
        f'**No CNO was present at recall.** All {session_key} measurements are drug-free.',
        '',
    ]
    text = '\n'.join(lines) + '\n'
    write_text(os.path.join(save_dir, filename), text)
    return text


_HIER_SYNTH_MOUSE_SD = 0.20      # between-animal SD of the per-mouse mean cell delta
_HIER_SYNTH_CELL_SD = 0.50       # within-animal, between-cell SD of the paired delta
_HIER_SYNTH_SHIFT = 0.60         # planted group shift, log units (~1.8x)
_HIER_SYNTH_ONE_MOUSE_SHIFT = 6.0   # deliberately absurd, and confined to ONE animal
_HIER_SYNTH_CELLS = 150
_HIER_SYNTH_ALPHA = 0.05


def _synthetic_cell_delta_table(group_sizes, shift_by_group, rng, n_cells=_HIER_SYNTH_CELLS,
                                mouse_sd=_HIER_SYNTH_MOUSE_SD, cell_sd=_HIER_SYNTH_CELL_SD,
                                one_mouse_shift=None):
    """Synthetic paired-cell delta table with a KNOWN planted structure: a per-group shift, a
    between-animal random intercept, and between-cell noise within each animal.

    `one_mouse_shift` adds a shift to exactly ONE animal (the first hM3D mouse) instead of to a
    group -- the design that separates a real treatment effect from a single-animal outlier.
    """
    rows, mice_per_group = [], {}
    for group, n in group_sizes:
        mice_per_group[group] = []
        for i in range(n):
            mouse = f'{group}_{i}'
            mice_per_group[group].append(mouse)
            mouse_effect = rng.normal(0.0, mouse_sd)
            extra = (one_mouse_shift
                     if (one_mouse_shift is not None and group == 'hM3D' and i == 0) else 0.0)
            deltas = (shift_by_group.get(group, 0.0) + mouse_effect + extra
                      + rng.normal(0.0, cell_sd, size=n_cells))
            for cell_id, d in enumerate(deltas):
                rows.append({'mouse': mouse, 'group': group, 'cell_id': cell_id,
                             HIERARCHICAL_CELL_DELTA_COL: float(d)})
    df = pd.DataFrame(rows)
    df['group'] = pd.Categorical(df['group'], categories=list(DREADD_DISPLAY_ORDER))
    return df, mice_per_group


def verify_hierarchical_cell_synthetic(save_dir, group_sizes, seed=HIERARCHICAL_CELL_SEED,
                                       n_perm_omnibus=500, n_cells=_HIER_SYNTH_CELLS,
                                       filename=None):
    """
    Synthetic verification of the PAIRED-AMPLITUDE path, run on the statistic this suite actually
    reports -- the paired-cell MixedLM coefficient with exact mouse-label randomization -- with
    the design-based sensitivity reported beside it.

    ** DEVELOPMENT TOOL -- NOT CALLED BY run_hierarchical_cell_suite(). ** This validates the
    CODE against a known planted truth in SIMULATED data; it says nothing about the experiment and
    produces no real-data result. Run it by hand after changing the paired-amplitude path:

        verify_hierarchical_cell_synthetic(some_dir, (('mCherry', 6), ('hM3D', 5), ('hM4D', 5)))

    It is deliberately absent from the real-data run: an execution that produces the reported
    numbers should compute those numbers and nothing else. (It also costs ~19 min, which is most
    of a real Test_B pass.) Nothing about it is or ever was contingent on what a real fit produced.

    Four designs, each hard-failing if the machinery does not behave:

      1. NULL -- no group difference in cell delta. Omnibus and all three pairwise randomization
         p-values must be non-significant.
      2. COHERENT GROUP SHIFT across cells in several mice -- must be detected.
      3. HUGE DELTA CONFINED TO ONE MOUSE -- must NOT reach significance. A cell-level test would
         call this overwhelming evidence; the mouse-label randomization cannot, because that
         animal's LABEL is the thing that moves. This is the design that distinguishes a treatment
         effect from one unusual animal.
      4. CELLS-PER-MOUSE SCALING at a FIXED number of mice -- the same null regenerated at
         increasing cells per animal. The randomization p must NOT drift toward zero (calibration
         is a function of 16 mice, not of cell count) while the model's within-mouse precision
         does improve. This is the pseudoreplication guard and the most important of the four: the
         MixedLM's own asymptotic P>|z| WOULD shrink with cell count here, and the file prints
         both side by side so the difference is visible rather than asserted.
    """
    rng = np.random.default_rng(seed)
    lines = ['SYNTHETIC VERIFICATION -- hierarchical cell-level PAIRED AMPLITUDE path',
             '=' * 78,
             'Implementation validation on simulated data with a KNOWN planted structure. This is',
             'a DEVELOPMENT TOOL and is NOT part of a real-data run: it validates the code, not',
             'the experiment, and produces no real-data result.',
             '',
             'Primary statistic: the three-group MixedLM contrast of '
             f'`{_hierarchical_delta_formula()} + (1|mouse)`,',
             'with EXACT mouse-label randomization restricted to the two groups compared.',
             'Design-based sensitivity (per-mouse mean delta) reported beside it.',
             f'Cohort: {dict(group_sizes)}; {n_cells} cells/mouse; mouse SD '
             f'{_HIER_SYNTH_MOUSE_SD}, cell SD {_HIER_SYNTH_CELL_SD}; seed {seed}.',
             f'Omnibus: Monte Carlo, n_perm={n_perm_omnibus}.', '']
    failures = []

    def _run(shift_by_group, one_mouse_shift=None, cells=n_cells):
        df, mpg = _synthetic_cell_delta_table(group_sizes, shift_by_group, rng, n_cells=cells,
                                              one_mouse_shift=one_mouse_shift)
        fit = fit_hierarchical_cell_delta_model(df)
        # include_global_null_sensitivity=False: this suite validates the PRIMARY path and the
        # restricted-null sensitivity. The global-null column would add an exact enumeration of
        # 2,018,016 relabelings per comparison per design, for a labelled sensitivity number this
        # verification makes no assertion about. Computational scoping of a verification run only
        # -- reported runs always compute it.
        perm = hierarchical_cell_amplitude_permutation(
            df, fit, mpg, n_perm_omnibus=n_perm_omnibus, seed=seed,
            include_global_null_sensitivity=False)
        return df, fit, perm

    def _single_exact_pairwise(df, fit, mpg, group_a, group_b):
        """Just ONE exact model-based pairwise randomization test, for a design that reports only
        that. Identical machinery to the primary path -- the full three-group model refit under
        every restricted relabeling -- without computing the two comparisons and the omnibus the
        design says nothing about."""
        present = {g: ms for g, ms in mpg.items() if ms}
        true_assignment = {m: g for g, ms in present.items() for m in ms}
        weights = _hierarchical_pair_weights(fit['coef_names'], group_a, group_b)
        stat = _make_hierarchical_model_stat(df, true_assignment, weights)
        return mouse_label_permutation_test(stat, present, seed=seed,
                                            restrict_to_groups=(group_a, group_b), exact=True)

    def _report(label, perm, fit):
        lines.append(f'--- {label}')
        omn = _hierarchical_pairwise_lookup(perm, 'omnibus', 'all', 'all', 'p_raw')
        lines.append(f'    omnibus permutation P = {omn:.4f}   '
                     f'(model asymptotic F P = {fit["omnibus"]["p"]:.3g}, NOT used)')
        for a, b in HIERARCHICAL_CELL_PAIRS:
            lines.append(
                f'    {a:>7} vs {b:<8} est {_hierarchical_pairwise_lookup(perm, "pairwise_primary", a, b, "estimate_lmm"):+.3f}'
                f'  exact P {_hierarchical_pairwise_lookup(perm, "pairwise_primary", a, b, "p_raw"):.4f}'
                f'  Holm {_hierarchical_pairwise_lookup(perm, "pairwise_primary", a, b, "p_holm_pairwise"):.4f}'
                f'  | design-based P '
                f'{_hierarchical_pairwise_lookup(perm, "pairwise_sensitivity", a, b, "p_raw_design_based"):.4f}')
        return omn

    # --- 1. null ------------------------------------------------------------------------------
    _df, fit1, perm1 = _run({})
    omn1 = _report('DESIGN 1: NULL (no group difference planted)', perm1, fit1)
    if omn1 <= _HIER_SYNTH_ALPHA:
        failures.append(f'design 1 (null): omnibus P={omn1:.4f} is significant under a true null.')
    for a, b in HIERARCHICAL_CELL_PAIRS:
        p = _hierarchical_pairwise_lookup(perm1, 'pairwise_primary', a, b, 'p_holm_pairwise')
        if p <= _HIER_SYNTH_ALPHA:
            failures.append(f'design 1 (null): {a} vs {b} Holm P={p:.4f} under a true null.')

    # --- 2. coherent group shift --------------------------------------------------------------
    _df, fit2, perm2 = _run({'hM3D': _HIER_SYNTH_SHIFT})
    _report(f'DESIGN 2: COHERENT GROUP SHIFT (+{_HIER_SYNTH_SHIFT} in every hM3D mouse\'s cells)',
            perm2, fit2)
    p2 = _hierarchical_pairwise_lookup(perm2, 'pairwise_primary', 'hM3D', 'mCherry', 'p_raw')
    if p2 > _HIER_SYNTH_ALPHA:
        failures.append(f'design 2 (coherent shift): hM3D vs mCherry raw exact P={p2:.4f} -- a '
                        f'shift planted in every mouse of a group was NOT detected.')

    # --- 3. one mouse only --------------------------------------------------------------------
    _df3, fit3, perm3 = _run({}, one_mouse_shift=_HIER_SYNTH_ONE_MOUSE_SHIFT)
    _report(f'DESIGN 3: HUGE SHIFT IN ONE MOUSE ONLY (+{_HIER_SYNTH_ONE_MOUSE_SHIFT} in one hM3D '
            f'animal)', perm3, fit3)
    p3 = _hierarchical_pairwise_lookup(perm3, 'pairwise_primary', 'hM3D', 'mCherry', 'p_raw')
    lines.append(f'    ** THE POINT OF THIS DESIGN: ** one animal carries an enormous, entirely '
                 f'real cell-level\n    effect. A test treating cells as replicates would call '
                 f'this overwhelming. The exact\n    mouse-label randomization gives P = {p3:.4f} '
                 f'-- it cannot be extreme, because the smallest\n    attainable P when the '
                 f'signal rides on ONE label is bounded by that label\'s own\n    exchangeability.')
    if p3 <= _HIER_SYNTH_ALPHA:
        failures.append(
            f'design 3 (one mouse): raw exact P={p3:.4f} is significant. A shift confined to a '
            f'single animal is masquerading as treatment evidence, which is the exact failure '
            f'this suite exists to prevent.')

    # --- 4. cells-per-mouse scaling at fixed n mice --------------------------------------------
    lines.append('')
    lines.append('--- DESIGN 4: CELLS PER MOUSE SCALING AT FIXED n MICE (true null throughout)')
    lines.append('    More cells must improve WITHIN-MOUSE estimation and buy NO treatment-level')
    lines.append('    evidence. Two things are asserted, and both are printed rather than claimed:')
    lines.append('      (a) the mean WITHIN-MOUSE standard error MUST shrink -- more cells really')
    lines.append('          do estimate each animal\'s own modulation better;')
    lines.append('      (b) the GROUP-CONTRAST standard error must NOT shrink with it, because it')
    lines.append('          is bounded below by BETWEEN-MOUSE variance, which no number of cells')
    lines.append('          reduces; and the randomization P must not drift toward 0 under a true')
    lines.append('          null. If (a) held while (b) failed, cells would be acting as')
    lines.append('          independent treatment replicates -- pseudoreplication.')
    lines.append('')
    lines.append('      cells/mouse   randomization P   contrast P (mouse df)      contrast SE')
    scaling = []
    for cells in (50, 500, 5000):
        df4, mpg4 = _synthetic_cell_delta_table(group_sizes, {}, rng, n_cells=cells)
        fit4 = fit_hierarchical_cell_delta_model(df4)
        # Only the hM3D-vs-mCherry exact test is reported by this design, so only it is computed.
        # Same machinery as the primary path; at 5000 cells/mouse the frame is ~80k rows and
        # computing the comparisons this design says nothing about would dominate the runtime.
        p_perm = float(_single_exact_pairwise(df4, fit4, mpg4, 'hM3D', 'mCherry')['p_two_sided'])
        contrast = linear_contrast_test(
            fit4['result'], _hierarchical_pair_weights(fit4['coef_names'], 'hM3D', 'mCherry'),
            n_groups=fit4['n_mice'])
        within = df4.groupby('mouse', observed=True)[HIERARCHICAL_CELL_DELTA_COL].sem().mean()
        scaling.append((cells, p_perm, contrast['p'], contrast['se'], float(within)))
        lines.append(f'      {cells:>11}   {p_perm:>15.4f}   {contrast["p"]:>18.3g}   '
                     f'{contrast["se"]:>14.4f}   (mean within-mouse SEM {within:.4f})')
    lines.append('')
    lines.append(f'    within-mouse SEM fell {scaling[0][4]:.4f} -> {scaling[-1][4]:.4f} '
                 f'(factor {scaling[0][4] / scaling[-1][4]:.1f}) as cells rose 100x,')
    lines.append(f'    while the group-contrast SE went {scaling[0][3]:.4f} -> '
                 f'{scaling[-1][3]:.4f} -- it is a BETWEEN-mouse quantity and stays put.')
    lines.append('    That gap IS the guard: cells sharpen the per-animal estimate and buy no')
    lines.append('    treatment-level precision. Note the asymptotic P column is also mouse-level')
    lines.append('    here (linear_contrast_test uses df = n_mice - 1, this codebase\'s')
    lines.append('    convention), so it does not shrink either; the naive cell-level P that')
    lines.append('    WOULD shrink is the MixedLM summary\'s own P>|z|, which this suite never')
    lines.append('    reports for exactly that reason.')
    if scaling[-1][4] >= scaling[0][4]:
        failures.append('design 4: the mean within-mouse standard error did not shrink as cells '
                        'per mouse increased -- more cells are not improving within-mouse '
                        'estimation, so the model is not using them as claimed.')
    # (b) The group-contrast SE is a BETWEEN-mouse quantity. If it tracked the within-mouse SEM
    # downward as cells were added, the contrast would be drawing precision from cell count at a
    # fixed number of animals, which is the definition of the failure this suite guards against.
    se_first, se_last = scaling[0][3], scaling[-1][3]
    sem_ratio = scaling[-1][4] / scaling[0][4]
    if se_last < 0.5 * se_first:
        failures.append(
            f'design 4: the GROUP-CONTRAST standard error fell from {se_first:.4f} to '
            f'{se_last:.4f} ({100 * (1 - se_last / se_first):.0f}%) as cells per mouse rose 100x '
            f'under a TRUE NULL, while the within-mouse SEM fell by a factor of '
            f'{1 / sem_ratio:.1f}. A between-mouse contrast SE must not track cell count like '
            f'that -- cells are buying treatment-level precision they cannot legitimately buy.')
    if scaling[-1][1] <= _HIER_SYNTH_ALPHA:
        failures.append(
            f'design 4: at {scaling[-1][0]} cells/mouse the randomization P fell to '
            f'{scaling[-1][1]:.4f} under a TRUE NULL. Adding cells at a fixed number of mice is '
            f'behaving like adding independent treatment replicates -- pseudoreplication.')

    lines += ['', '=' * 78]
    if failures:
        lines.append('FAILED:')
        lines += [f'  - {f}' for f in failures]
    else:
        lines.append('ALL FOUR DESIGNS PASSED.')
    text = '\n'.join(lines) + '\n'
    if filename is None:
        filename = f'{HIERARCHICAL_CELL_STATS_PREFIX}_synthetic_verification.txt'
    write_text(os.path.join(save_dir, filename), text)
    if failures:
        raise RuntimeError('verify_hierarchical_cell_synthetic: the paired-amplitude machinery '
                           'failed its synthetic checks:\n  - ' + '\n  - '.join(failures)
                           + f'\n\nFull report: {filename}')
    return text


# Dev-only. mp_ctx='spawn' for the same reason as the production config: this is often run from
# a notebook, where forking a large kernel deadlocks.
_NB_SYNTH_SAMPLER = dict(draws=400, tune=400, chains=2, random_seed=HIERARCHICAL_CELL_SEED,
                         target_accept=0.9, progressbar=False, mp_ctx='spawn')


def _synthetic_cell_count_table(group_sizes, interaction_by_group, rng, n_cells=60,
                                exposure_s=100.0, baseline_rate=0.05, mouse_sd=0.25,
                                cell_sd=0.8, slope_sd=0.0, alpha=2.0):
    """Simulate a cell x epoch count table from the count model's OWN generative structure, with
    every planted parameter known.

    `slope_sd` is the between-mouse SD of the pre->post log-rate change -- the quantity the
    mouse-level epoch random slope exists to absorb. `cell_sd` is tuned so the table carries a
    realistic fraction of zero-event cells rather than a convenient one.
    """
    rows = []
    for group, n in group_sizes:
        for i in range(n):
            mouse = f'{group}_{i}'
            mouse_effect = rng.normal(0.0, mouse_sd)
            mouse_slope = rng.normal(0.0, slope_sd) if slope_sd > 0 else 0.0
            cell_effects = rng.normal(0.0, cell_sd, size=n_cells)
            for cell_id in range(n_cells):
                for epoch in RECALL_EPOCHS:
                    post = 1.0 if epoch == RECALL_RESPONSE_EPOCH else 0.0
                    log_mu = (np.log(baseline_rate * exposure_s) + mouse_effect
                              + cell_effects[cell_id]
                              + post * (mouse_slope + interaction_by_group.get(group, 0.0)))
                    mu = float(np.exp(log_mu))
                    # NB with Var = mu + mu^2/alpha (the confirmed convention), sampled as a
                    # gamma-Poisson mixture so the simulation does not depend on a backend RNG.
                    lam = rng.gamma(shape=alpha, scale=mu / alpha)
                    rows.append({'mouse': mouse, 'group': group, 'cell_id': cell_id,
                                 'epoch': epoch, 'n_events': int(rng.poisson(lam)),
                                 'exposure_seconds': exposure_s})
    df = pd.DataFrame(rows)
    df['group'] = pd.Categorical(df['group'], categories=list(DREADD_DISPLAY_ORDER))
    return df


def verify_hierarchical_cell_rate_synthetic(save_dir, group_sizes, seed=HIERARCHICAL_CELL_SEED,
                                            n_cells=40, sampler=None, filename=None):
    """
    Synthetic verification of the hierarchical NB COUNT path.

    ** DEVELOPMENT TOOL -- NOT CALLED BY run_hierarchical_cell_suite(). ** This validates the
    CODE against a known planted truth in SIMULATED data; it says nothing about the experiment and
    produces no real-data result. Run it by hand after changing the count path:

        verify_hierarchical_cell_rate_synthetic(some_dir,
                                                (('mCherry', 6), ('hM3D', 5), ('hM4D', 5)))

    It is deliberately absent from the real-data run: an execution that produces the reported
    numbers should compute those numbers and nothing else. Nothing about it is or ever was
    contingent on what a real fit produced.

    Four designs, simulated from the model's own generative structure at realistic zero fractions:

      1. NULL INTERACTION -- no group difference in the pre->post change. Every modulation
         ratio's HDI must cover 1.
      2. PLANTED GROUP-SPECIFIC MODULATION -- a known interaction on hM3D. Its HDI must cover the
         planted value AND exclude 1 (the effect is planted large enough that it should).
      3. MOUSE-TO-MOUSE SLOPE HETEROGENEITY -- mice given genuinely different pre->post slopes
         within a group. The `post_indicator|mouse` SD must recover near its planted value, and
         the group x epoch interval must be WIDER than in a matched simulation with no slope
         heterogeneity. That is the direct check that the mouse epoch random slope does the job it
         was added for: without it, that between-mouse variation would be absorbed by the cell
         level and the interaction interval would be too narrow.
      4. REALISTIC ZEROS -- the posterior-predictive zero fraction must match the simulated one,
         validating the adequacy check against a known truth before it is read on real data.

    Reduced sampler settings are permitted HERE ONLY -- none of these is a reported inferential
    fit -- and the settings used are printed in the file.
    """
    rng = np.random.default_rng(seed)
    fit_kwargs = dict(_NB_SYNTH_SAMPLER if sampler is None else sampler)
    convention, measured, expected = _confirm_nb_alpha_parameterization(seed=seed)
    lines = ['SYNTHETIC VERIFICATION -- hierarchical cell-level NB RATE path',
             '=' * 78,
             'Implementation validation on simulated data from the model\'s own generative',
             'structure, with every planted parameter known. This is a DEVELOPMENT TOOL and is',
             'NOT part of a real-data run; it validates the code, not the experiment.',
             '',
             f'NB parameterization confirmed empirically: {convention}',
             f'  (measured Var {measured:.3f} vs expected {expected:.3f} at mu=5, alpha=2)',
             f'Cohort {dict(group_sizes)}, {n_cells} cells/mouse x 2 epochs; seed {seed}.',
             f'Sampler (REDUCED -- verification only, not a reported fit): {fit_kwargs}', '']
    failures = []

    def _fit(df, label):
        built = build_cell_epoch_rate_model(df)
        model = built['model']
        idata = model.fit(**fit_kwargs)
        contrasts = summarize_cell_rate_contrasts(idata, built)
        zero_obs = float((df['n_events'] == 0).mean())
        lines.append(f'--- {label}')
        lines.append(f'    observed zero fraction in the simulated table: {zero_obs:.3f}')
        for _, row in contrasts[contrasts['block'] == 'modulation'].iterrows():
            lines.append(f'    modulation {row["group_a"]:>7} vs {row["group_b"]:<8} '
                         f'ratio {row["posterior_median"]:.3f} '
                         f'[{row["hdi_low"]:.3f}, {row["hdi_high"]:.3f}]')
        return built, idata, contrasts, zero_obs

    def _mod(contrasts, group_a, group_b='mCherry'):
        sub = contrasts[(contrasts['block'] == 'modulation') & (contrasts['group_a'] == group_a)
                        & (contrasts['group_b'] == group_b)]
        return float(sub.iloc[0]['posterior_median']), float(sub.iloc[0]['hdi_low']), \
            float(sub.iloc[0]['hdi_high'])

    # --- 1. null interaction ------------------------------------------------------------------
    df1 = _synthetic_cell_count_table(group_sizes, {}, rng, n_cells=n_cells)
    _b1, _i1, c1, _z1 = _fit(df1, 'DESIGN 1: NULL INTERACTION')
    for group_a, group_b in HIERARCHICAL_CELL_PAIRS:
        sub = c1[(c1['block'] == 'modulation') & (c1['group_a'] == group_a)
                 & (c1['group_b'] == group_b)].iloc[0]
        if not (sub['hdi_low'] <= 1.0 <= sub['hdi_high']):
            failures.append(f'design 1 (null interaction): {group_a} vs {group_b} HDI '
                            f'[{sub["hdi_low"]:.3f}, {sub["hdi_high"]:.3f}] excludes 1.')

    # --- 2. planted group-specific modulation --------------------------------------------------
    planted = 0.8
    df2 = _synthetic_cell_count_table(group_sizes, {'hM3D': planted}, rng, n_cells=n_cells)
    _b2, _i2, c2, _z2 = _fit(df2, f'DESIGN 2: PLANTED hM3D MODULATION '
                                  f'(log {planted}, ratio {np.exp(planted):.3f})')
    med, lo, hi = _mod(c2, 'hM3D')
    lines.append(f'    planted ratio {np.exp(planted):.3f}; recovered {med:.3f} [{lo:.3f}, '
                 f'{hi:.3f}]')
    if not (lo <= np.exp(planted) <= hi):
        failures.append(f'design 2: the HDI [{lo:.3f}, {hi:.3f}] does not cover the planted ratio '
                        f'{np.exp(planted):.3f} -- the contrast is not recovering truth.')
    if lo <= 1.0 <= hi:
        failures.append(f'design 2: the HDI [{lo:.3f}, {hi:.3f}] still covers 1 at a planted '
                        f'ratio of {np.exp(planted):.3f}.')

    # --- 3. mouse-to-mouse slope heterogeneity -------------------------------------------------
    slope_sd = 0.6
    df3 = _synthetic_cell_count_table(group_sizes, {}, rng, n_cells=n_cells, slope_sd=slope_sd)
    b3, i3, c3, _z3 = _fit(df3, f'DESIGN 3: MOUSE EPOCH-SLOPE HETEROGENEITY (planted SD '
                                f'{slope_sd})')
    recovered_sd = float(az.summary(i3, var_names=['post_indicator|mouse_sigma'])['mean'].iloc[0])
    width3 = np.mean([hi - lo for _, lo, hi in
                      [_mod(c3, g) for g in ('hM3D', 'hM4D')]])
    width1 = np.mean([hi - lo for _, lo, hi in
                      [_mod(c1, g) for g in ('hM3D', 'hM4D')]])
    lines.append(f'    planted post_indicator|mouse SD {slope_sd}; recovered posterior mean '
                 f'{recovered_sd:.3f}')
    lines.append(f'    mean modulation-HDI width WITH slope heterogeneity {width3:.3f} vs '
                 f'{width1:.3f} without')
    lines.append('    (the interval MUST widen: that between-mouse variation in the pre->post')
    lines.append('     change is exactly what the mouse epoch slope exists to absorb, and')
    lines.append('     without it the group x epoch interval would be too narrow.)')
    if not (0.4 * slope_sd <= recovered_sd <= 2.5 * slope_sd):
        failures.append(f'design 3: planted mouse epoch-slope SD {slope_sd} recovered as '
                        f'{recovered_sd:.3f} -- outside the tolerance band, so the slope term is '
                        f'not estimating what it claims to.')
    if width3 <= width1:
        failures.append(f'design 3: the modulation HDI did NOT widen under genuine mouse-to-mouse '
                        f'slope heterogeneity ({width3:.3f} vs {width1:.3f}). The mouse epoch '
                        f'slope is not absorbing that variation, which is the whole reason it is '
                        f'in the frozen structure.')

    # --- 4. realistic zeros --------------------------------------------------------------------
    built4 = build_cell_epoch_rate_model(df1)
    idata4 = built4['model'].fit(**fit_kwargs)
    ppc4 = hierarchical_cell_rate_posterior_predictive(
        idata4, built4, save_dir, save_figure=False,
        filename=f'{HIERARCHICAL_CELL_STATS_PREFIX}_rate_synthetic_posterior_predictive.txt')
    zero_rows = ppc4['table'][ppc4['table']['statistic'] == 'zero_fraction']
    lines.append('')
    lines.append('--- DESIGN 4: REALISTIC ZEROS (posterior-predictive zero fraction vs truth)')
    lines.append(f'    simulated overall zero fraction {float((df1["n_events"] == 0).mean()):.3f}')
    for _, row in zero_rows.iterrows():
        lines.append(f'    {row["group"]:>7} {row["epoch"]:<10} observed {row["observed"]:.3f}  '
                     f'ppc [{row["ppc_lo_2.5"]:.3f}, {row["ppc_hi_97.5"]:.3f}]  '
                     f'{"OK" if row["observed_in_interval"] else "OUTSIDE"}')
    outside = zero_rows[~zero_rows['observed_in_interval']]
    if len(outside):
        failures.append(f'design 4: the posterior-predictive zero fraction missed the simulated '
                        f'truth in {len(outside)} of {len(zero_rows)} group x epoch cells, so the '
                        f'zero check cannot be trusted on real data.')

    lines += ['', '=' * 78]
    if failures:
        lines.append('FAILED:')
        lines += [f'  - {f}' for f in failures]
    else:
        lines.append('ALL FOUR DESIGNS PASSED.')
    text = '\n'.join(lines) + '\n'
    if filename is None:
        filename = f'{HIERARCHICAL_CELL_STATS_PREFIX}_rate_synthetic_verification.txt'
    write_text(os.path.join(save_dir, filename), text)
    if failures:
        raise RuntimeError('verify_hierarchical_cell_rate_synthetic: the NB count machinery '
                           'failed its synthetic checks:\n  - ' + '\n  - '.join(failures)
                           + f'\n\nFull report: {filename}')
    return text


_HIERARCHICAL_CELL_COMPONENTS = (
    'paired cell amplitude table + eligibility',
    'paired-cell MixedLM (primary estimator)',
    'mouse-label randomization inference (exact pairwise + MC omnibus + design-based sensitivity)',
    'unpaired amplitude sensitivity model',
    'cell x epoch count table',
    'NB prior-predictive check',
    'hierarchical NB rate model + convergence gate',
    'NB posterior-predictive adequacy checks',
    'amplitude modulation figure',
    'rate modulation figure',
    'comparison against the mouse-summary analysis',
)


def run_hierarchical_cell_suite(recall_dir, session_key, session_label, df_matched, mice_per_group,
                                interactions_table, pairwise_df,
                                n_perm_omnibus=HIERARCHICAL_CELL_N_PERM_OMNIBUS,
                                seed=HIERARCHICAL_CELL_SEED,
                                rate_sampler=None):
    """
    Run the WHOLE hierarchical cell-level companion suite for one recall session, atomically.

    ** EXPLICITLY INVOKED, not part of a routine pass. ** This suite costs ~40 min against a few
    minutes for the rest of the analysis, so run_hierarchical_cell_analysis defaults to False and
    a normal run_sp_rates_lmm() never reaches here. Pass run_hierarchical_cell_analysis=True (to
    run_sp_rates_lmm or render_paper_recall_amplitude_rate), or call this directly, to rerun the
    sensitivity analysis. It is a companion to the mouse-level recall lane, which remains the
    paper-facing analysis and is complete without it.

    ** One predefined analysis, all-or-nothing. ** Every component in
    _HIERARCHICAL_CELL_COMPONENTS runs on every pass. There is no per-component switch and no
    result-dependent branch: nothing here is activated because another component was or was not
    significant, and no model is substituted if a specified one fails.

    ** Analyses and their diagnostics only. ** The components are the paired-cell construction,
    the hierarchical amplitude fit and its mouse-label randomization inference, the declared
    amplitude sensitivity model, the hierarchical rate fit with its prior/posterior-predictive and
    convergence diagnostics, and the figures and reports. Implementation validation against
    planted synthetic truth (verify_hierarchical_cell_synthetic,
    verify_hierarchical_cell_rate_synthetic) is NOT here -- those are development tools, called by
    hand. A run that produces the reported numbers computes those numbers and nothing else.

    ** Atomic output. ** Everything is written to a sibling `<dir>__staging/` directory and the
    previous complete output is replaced wholesale only after EVERY component has succeeded. So:
    a run that raises leaves the previous complete `hierarchical_cells/` untouched and the partial
    staging directory on disk for inspection; there is no window in which the promoted directory
    holds a mixture of two passes; and a stale rate figure from an earlier run cannot survive
    beside fresh amplitude output, because promotion replaces the directory rather than
    overwriting file by file.

    On failure a FAILED.txt naming the component, the exception and the traceback is written into
    the staging directory and the exception is then RE-RAISED. The artifact is a record, not a
    handler -- nothing is swallowed.

    Returns dict of the frames and fits produced.
    """
    final_dir = os.path.join(recall_dir, HIERARCHICAL_CELL_DIRNAME)
    staging_dir = os.path.join(recall_dir, HIERARCHICAL_CELL_DIRNAME + '__staging')
    if os.path.isdir(staging_dir):
        shutil.rmtree(staging_dir)
    stats_dir = os.path.join(staging_dir, 'stats')
    ensure_dirs(staging_dir, stats_dir)
    prefix = _RECALL_FILE_PREFIX.get(session_key, session_key.lower())
    sp = HIERARCHICAL_CELL_STATS_PREFIX
    # Plain forward iterator over the component list: _advance() records the component just
    # finished and names the next one. Hand-written indices drifted the moment a component was
    # removed, and a stale index misattributes the failing step in FAILED.txt -- the one place it
    # has to be right.
    _remaining = iter(_HIERARCHICAL_CELL_COMPONENTS)
    completed, current = [], next(_remaining)
    _n_components = len(_HIERARCHICAL_CELL_COMPONENTS)
    _t_step = [time.perf_counter()]
    _t_suite = time.perf_counter()

    def _step_start():
        # DISPLAY ONLY. This suite has two multi-minute steps (the randomization block and the NB
        # fit) that otherwise run in total silence, which is how a deadlock went unnoticed for
        # nearly three hours. Every component announces itself with a wall-clock stamp so a
        # pasted log says exactly where a run is and how long each step took.
        print(f'[hier {len(completed) + 1:>2}/{_n_components} '
              f'{datetime.datetime.now():%H:%M:%S}] START {current}', flush=True)
        _t_step[0] = time.perf_counter()

    def _advance():
        nonlocal current
        print(f'[hier {len(completed) + 1:>2}/{_n_components} '
              f'{datetime.datetime.now():%H:%M:%S}] DONE  {current} '
              f'({time.perf_counter() - _t_step[0]:.1f} s)', flush=True)
        completed.append(current)
        current = next(_remaining, 'done')
        if current != 'done':
            _step_start()

    print(f'[sp_rates_lmm]   {session_key} hierarchical cell-level companion suite '
          f'({_n_components} components, all required)...', flush=True)
    _step_start()
    try:
        _copy_analysis_methods_template(HIERARCHICAL_CELL_METHODS_FILENAME, staging_dir)

        # ---- amplitude: paired cell table -----------------------------------------------------
        delta_df, eligibility_df = build_recall_cell_amplitude_modulation(df_matched)
        delta_df.insert(0, 'session', session_key)
        write_text(os.path.join(stats_dir, f'{sp}_amplitude_modulation.csv'),
                   delta_df.to_csv(index=False))
        write_text(os.path.join(stats_dir, f'{sp}_eligibility.csv'),
                   eligibility_df.to_csv(index=False))
        print(f'[sp_rates_lmm]     {len(delta_df)} paired cells over '
              f'{delta_df["mouse"].nunique()} animals; eligible fraction '
              f'{eligibility_df["fraction_eligible_both"].min():.2f}'
              f'–{eligibility_df["fraction_eligible_both"].max():.2f}')
        _advance()

        # ---- amplitude: the primary estimator -------------------------------------------------
        fit = fit_hierarchical_cell_delta_model(delta_df.drop(columns=['session']))
        write_text(os.path.join(stats_dir, f'{sp}_amplitude_lmm_summary.txt'),
                   'PRIMARY ESTIMATOR -- hierarchical cell-level companion analysis, recall '
                   f'{session_key}.\n'
                   f'Paired within-cell pre->post modulation: {fit["formula"]} + (1|mouse)\n'
                   f'{fit["n_cells"]} paired cells in {fit["n_mice"]} animals.\n'
                   f'Optimizer: {HIERARCHICAL_CELL_LMM_OPTIMIZER} (numerical only).\n\n'
                   '** THE P>|z| COLUMN BELOW IS ASYMPTOTIC OVER CELLS AND IS NOT A PAPER-FACING\n'
                   'INFERENCE. ** 16 animals were randomized, not N cells. The inference for these\n'
                   'same coefficients is the EXACT mouse-label randomization in\n'
                   f'{sp}_amplitude_permutation.csv, which refits this model under every\n'
                   'size-preserving relabeling of the mouse group labels.\n\n'
                   'ESTIMAND: the pre->post amplitude modulation of neurons with measurable event\n'
                   f'amplitude in BOTH epochs (see {sp}_eligibility.csv). Not an estimate '
                   f'over all\n'
                   'detected cells, and not generalized to them. The analysis that includes\n'
                   f'zero-event cells is the rate path ({sp}_rate_*).\n\n'
                   f'Group x epoch omnibus (asymptotic, NOT used for inference): '
                   f'{fit["omnibus"]}\n\n{fit["summary_text"]}\n')
        _advance()

        # ---- amplitude: randomization inference -----------------------------------------------
        perm_df = hierarchical_cell_amplitude_permutation(
            delta_df.drop(columns=['session']), fit, mice_per_group,
            n_perm_omnibus=n_perm_omnibus, seed=seed)
        perm_df.insert(0, 'session', session_key)
        write_text(os.path.join(stats_dir, f'{sp}_amplitude_permutation.csv'),
                   perm_df.to_csv(index=False))
        for _, row in perm_df[perm_df['block'] == 'pairwise_primary'].iterrows():
            print(f'[sp_rates_lmm]     {row["group_a"]}-vs-{row["group_b"]}: '
                  f'Δlog {row["estimate_lmm"]:+.3f} '
                  f'[{row["ci_lmm_low"]:.3f}, {row["ci_lmm_high"]:.3f}], exact P '
                  f'{row["p_raw"]:.4g} over {int(row["n_relabelings"])} relabelings, Holm P '
                  f'{row["p_holm_pairwise"]:.4g}')
        _advance()

        # ---- amplitude: unpaired sensitivity --------------------------------------------------
        unpaired = fit_hierarchical_cell_unpaired_sensitivity(df_matched, stats_dir)
        _advance()

        # ---- rate: count table ----------------------------------------------------------------
        counts_df = build_recall_cell_epoch_count_table(df_matched)
        counts_out = counts_df.copy()
        counts_out.insert(0, 'session', session_key)
        write_text(os.path.join(stats_dir, f'{sp}_rate_counts.csv'),
                   counts_out.to_csv(index=False))
        print(f'[sp_rates_lmm]     rate table: {len(counts_df)} cell x epoch rows, '
              f'{float((counts_df["n_events"] == 0).mean()):.3f} of them zero-event')
        _advance()

        # ---- rate: prior predictive (BEFORE the inferential fit) ------------------------------
        built = build_cell_epoch_rate_model(counts_df)
        hierarchical_cell_rate_prior_predictive(built, stats_dir, seed=seed)
        _advance()

        # ---- rate: the hierarchical NB model + gate -------------------------------------------
        convention, measured, expected_var = _confirm_nb_alpha_parameterization(seed=seed)
        rate_fit = fit_cell_epoch_rate_model(built, sampler=rate_sampler)
        rate_contrasts = summarize_cell_rate_contrasts(rate_fit['idata'], built)
        rate_contrasts.insert(0, 'session', session_key)
        write_text(os.path.join(stats_dir, f'{sp}_rate_contrasts.csv'),
                   rate_contrasts.to_csv(index=False))
        coef_summary = az.summary(
            rate_fit['idata'],
            var_names=[f'{built["group_term"]}:{built["epoch_term"]}', built['group_term'],
                       built['epoch_term'], 'alpha', '1|mouse_sigma',
                       'post_indicator|mouse_sigma', '1|mouse_cell_sigma'])
        write_text(os.path.join(stats_dir, f'{sp}_rate_nb_summary.txt'),
                   'HIERARCHICAL CELL-LEVEL RATE MODEL -- recall '
                   f'{session_key} (companion analysis).\n\n'
                   f'Formula: {built["formula"]}\n'
                   f'{len(counts_df)} cell x epoch rows over '
                   f'{counts_df["mouse"].nunique()} animals; ALL detected cells, zero-event cells '
                   f'included.\n\n'
                   'FROZEN PRIORS (specified before fitting; verified attached on the built '
                   'model):\n'
                   + '\n'.join(f'  {k}: {v}' for k, v in built['priors_applied'].items()) + '\n\n'
                   f'NB dispersion parameterization, CONFIRMED empirically against the installed '
                   f'backend:\n  {convention}\n  (measured Var {measured:.3f} vs expected '
                   f'{expected_var:.3f} at mu=5, alpha=2)\n\n'
                   f'FROZEN SAMPLER: {rate_fit["sampler"]}\n\n'
                   f'{rate_fit["gate_text"]}\n'
                   f'CONTRASTS: see {sp}_rate_contrasts.csv. `modulation` is the group x '
                   f'epoch\n'
                   'interaction alone (the pre->post ratio-of-ratios); `post_tone_simple_effect`\n'
                   'is the ABSOLUTE post-tone group-vs-control ratio. Different quantities.\n'
                   'No frequentist p-value is manufactured from this posterior, it supplies no\n'
                   'star to any figure, and it alters no existing paper-facing number.\n\n'
                   f'{coef_summary.to_string()}\n')
        _advance()

        # ---- rate: posterior-predictive adequacy ----------------------------------------------
        ppc = hierarchical_cell_rate_posterior_predictive(rate_fit['idata'], built, stats_dir)
        if ppc['inadequate']:
            print('[sp_rates_lmm]     ** NB rate model FAILED its posterior-predictive adequacy '
                  'checks; estimates are reported WITH that failure. **')
        _advance()

        # ---- figures --------------------------------------------------------------------------
        plot_hierarchical_cell_amplitude_modulation(
            delta_df.drop(columns=['session']), perm_df, eligibility_df, staging_dir,
            session_label, filename_root=f'{prefix}_hierarchical_cell_amplitude_modulation')
        _advance()

        plot_hierarchical_cell_rate_modulation(
            counts_df, rate_contrasts, staging_dir, session_label,
            filename_root=f'{prefix}_hierarchical_cell_rate_modulation')
        _advance()

        # ---- comparison against the existing mouse-summary analysis ---------------------------
        write_hierarchical_cell_vs_mouse_summary(
            stats_dir, session_key, session_label, fit, perm_df, eligibility_df,
            interactions_table, pairwise_df, rate_contrasts=rate_contrasts, rate_fit=rate_fit,
            ppc=ppc)
        _advance()

        write_text(os.path.join(staging_dir, 'RUN_STATUS.txt'),
                   f'COMPLETE -- hierarchical cell-level companion suite, {session_key}\n'
                   f'{datetime.datetime.now().isoformat(timespec="seconds")}\n\n'
                   'This suite is all-or-nothing: the directory is promoted into place only after\n'
                   'every required component below has succeeded, so its presence means the whole\n'
                   'suite ran. A failed run leaves the previous complete directory untouched and\n'
                   'its partial output in <dir>__staging/ with a FAILED.txt.\n\n'
                   'Components (all required, none optional, none result-dependent):\n'
                   + '\n'.join(f'  [done] {c}' for c in _HIERARCHICAL_CELL_COMPONENTS) + '\n')
    except BaseException as exc:
        write_text(os.path.join(staging_dir, 'FAILED.txt'),
                   f'FAILED -- hierarchical cell-level companion suite, {session_key}\n'
                   f'{datetime.datetime.now().isoformat(timespec="seconds")}\n\n'
                   f'Failing component: {current}\n\n'
                   f'Completed before the failure:\n'
                   + ('\n'.join(f'  [done] {c}' for c in completed) or '  (none)') + '\n\n'
                   f'NOT run:\n'
                   + '\n'.join(f'  [skipped] {c}' for c in _HIERARCHICAL_CELL_COMPONENTS
                               if c not in completed and c != current) + '\n\n'
                   'No substitute model was fit and no weaker test was run in place of the failing\n'
                   'component. The previous complete output (if any) is untouched at\n'
                   f'{final_dir}\n\n{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}\n')
        print(f'[sp_rates_lmm]   {session_key} hierarchical cell suite FAILED at: {current} '
              f'after {(time.perf_counter() - _t_suite) / 60:.1f} min. Partial output and '
              f'FAILED.txt in {staging_dir}; the previous complete output is untouched.',
              flush=True)
        raise

    # ---- promote atomically: replace the directory wholesale, never file by file ---------------
    if os.path.isdir(final_dir):
        shutil.rmtree(final_dir)
    os.replace(staging_dir, final_dir)
    print(f'[sp_rates_lmm]   {session_key} hierarchical cell-level companion suite complete '
          f'in {(time.perf_counter() - _t_suite) / 60:.1f} min -> {final_dir}', flush=True)
    return {'hierarchical_cell_delta': delta_df, 'hierarchical_cell_eligibility': eligibility_df,
            'hierarchical_cell_fit': fit, 'hierarchical_cell_permutation': perm_df,
            'hierarchical_cell_unpaired': unpaired, 'hierarchical_cell_counts': counts_df,
            'hierarchical_cell_rate_fit': rate_fit,
            'hierarchical_cell_rate_contrasts': rate_contrasts,
            'hierarchical_cell_rate_ppc': ppc}


def write_recall_results_summary(save_dir, session_key, session_label, mouse_epoch, fits,
                                 contrasts, interactions_table, coverage, stats_prefix,
                                 filename='paper_results_summary.md'):
    """
    Every number a recall Results paragraph needs, for ONE recall session, plus a drafted
    paragraph in the manuscript's register with this run's own values already in it.

    ** One session per file, and no cross-session sentence anywhere in it. ** The two recall
    sessions do not contain the same animals, so "significant at 48 h but not at 1 week" is not
    evidence that anything declined; that comparison would need a fixed-cohort timepoint model
    which this lane does not fit. The drafted paragraph therefore describes this session only.

    ** The interpretation of each (simple effect, interaction) pair is fixed in advance ** by
    _RECALL_INTERPRETATION, so what the numbers are read to mean does not depend on what they
    turn out to be.
    """
    n_by_group = (mouse_epoch.drop_duplicates('mouse').groupby('group', observed=True)
                  .size().reindex(DREADD_DISPLAY_ORDER).dropna().astype(int))
    cohort = ', '.join(f'n = {n} {GROUP_LABELS.get(g, g)}' for g, n in n_by_group.items())
    df2 = int(interactions_table['df2'].iloc[0])
    payloads = _unified_contrast_payloads(contrasts, epochs=RECALL_EPOCHS)

    lines = [
        f'# {session_label} — cellular amplitude and event rate: paper summary',
        '',
        f'Per-event amplitude and population event rate by DREADD group across the two '
        f'duration-matched 20 s recall windows (pre-tone baseline, post-tone retrieval). '
        f'{cohort} animals present in this session; the inferential dataset is '
        f'{len(mouse_epoch)} rows — one value per animal per epoch.',
        '',
        f'**This file describes {session_key} only.** Test_B and Test_B_1wk are analysed '
        'independently because the animals available at the two recall sessions are not the same, '
        'so no sentence here — and no sentence in the manuscript drawn from here — may compare '
        'the two timepoints. A formal 48 h versus 1 week comparison would require a separate '
        'fixed-cohort model restricted to animals present at both sessions, which this analysis '
        'does not fit.',
        '',
        '**No CNO was present during recall.** Any group difference reported below is a '
        'persistent consequence of the conditioning-day manipulation, not evidence of ongoing '
        'chemogenetic receptor activation.',
        '',
        '## Trial coverage',
        '',
        'Analyses are restricted within each animal to tone trials on which BOTH the 20 s '
        'pre-tone and the 20 s post-tone window are completely observed; amplitude and rate use '
        'that identical trial set.',
        '',
        '| animal | group | tone trials present | trials retained | retained |',
        '|---|---|---|---|---|',
    ]
    for _, row in coverage.iterrows():
        lines.append(f"| {row['mouse']} | {GROUP_LABELS.get(row['group'], row['group'])} | "
                     f"{int(row['n_tone_trials_present'])} | {int(row['n_trials_retained'])} | "
                     f"{row['trials_retained']} |")

    lines += [
        '',
        '## Unified mixed models',
        '',
        'Both outcomes are analysed with the SAME mouse-level linear mixed-effects model: DREADD '
        'group, epoch and their interaction as fixed effects, animal as a random intercept, '
        'reference levels mCherry and pre-tone. Event definition and animal-level summarization '
        'are identical to the conditioning analysis. All Wald and contrast inference uses this '
        f'session\'s own animal-level denominator df = n_animals − 1 = {df2}.',
        '',
    ]
    for outcome in UNIFIED_OUTCOMES:
        fit = fits[outcome.key]
        lines.append(f"- **{outcome.label}**: `{fit['formula']} + (1|mouse)` "
                     f"({fit['method']}, {fit['n_mice']} animals)")

    lines += [
        '',
        '## Treatment-vs-control contrasts by epoch',
        '',
        'Each is a linear contrast of the same full group x epoch model — the group coefficient '
        'alone at the pre-tone reference epoch, that coefficient plus the corresponding '
        'group x epoch coefficient at post-tone, with their covariance. Within each epoch the two '
        'treatment-vs-control comparisons are Holm-corrected together (`p_holm_epoch`).',
        '',
        '**These are the only P-values that generate a figure asterisk or a manuscript '
        'significance statement.**',
        '',
        '| outcome | epoch | comparison | ratio | 95% CI | raw P | Holm-adjusted P (within epoch) |',
        '|---|---|---|---|---|---|---|',
    ]
    for _, row in contrasts.iterrows():
        flag = ' (significant)' if row['holm_epoch_reject'] else ''
        lines.append(
            f"| {UNIFIED_OUTCOMES_BY_KEY[row['outcome']].label} | {row['epoch']} | "
            f"{GROUP_LABELS.get(row['group'], row['group'])} vs Ctl | {row['ratio']:.3f} | "
            f"[{row['ratio_ci_low']:.3f}, {row['ratio_ci_high']:.3f}] | {row['p_raw']:.4g} | "
            f"**{row['p_holm_epoch']:.4g}**{flag} |")

    lines += [
        '',
        '## Group x epoch interaction — is the effect specific to the retrieval window?',
        '',
        'Joint Wald test of the two group x post-tone interaction coefficients, one per outcome. '
        '**This is the formal test of retrieval preferentiality**, and the only one. A '
        'significant post-tone comparison with a null interaction does not establish that the '
        'difference is tone-evoked; it may reflect a group difference already present around '
        'recall.',
        '',
        '| outcome | F | df1 | df2 | P |',
        '|---|---|---|---|---|',
    ]
    for _, row in interactions_table.iterrows():
        lines.append(f"| {UNIFIED_OUTCOMES_BY_KEY[row['outcome']].label} | {row['F']:.3f} | "
                     f"{int(row['df1'])} | {int(row['df2'])} | {row['p']:.4g} |")

    lines += [
        '',
        'A non-significant interaction is **no evidence that the treatment effect differed '
        'between the pre-tone and post-tone windows**. It is not evidence that the effect is '
        'identical, global or tonic across them.',
        '',
        '## Reading of each result, by the rule fixed in advance',
        '',
    ]
    interaction_p = {row['outcome']: float(row['p'])
                     for _, row in interactions_table.iterrows()}
    for outcome in UNIFIED_OUTCOMES:
        lines.append(f'**{outcome.label}**')
        lines.append('')
        for group in UNIFIED_TREATMENT_GROUPS:
            pre = unified_contrast_lookup(contrasts, outcome.key,
                                          RECALL_REFERENCE_EPOCH, group)
            post = unified_contrast_lookup(contrasts, outcome.key,
                                           RECALL_RESPONSE_EPOCH, group)
            key = (bool(pre['holm_epoch_reject']), bool(post['holm_epoch_reject']))
            lines.append(f"- {group} {_RECALL_SIMPLE_EFFECT_READING[key]}.")
        lines.append(f"- {_RECALL_INTERACTION_READING[interaction_p[outcome.key] < 0.05]}")
        lines.append('')

    rate_rows = contrasts[contrasts['outcome'] == 'population_rate']
    lines += [
        '## Absolute population-rate differences',
        '',
        'DESCRIPTIVE — equal-mouse-weighted observed means, no test.',
        '',
    ]
    for _, row in rate_rows.iterrows():
        lines.append(
            f"- {row['epoch']}, {GROUP_LABELS.get(row['group'], row['group'])} vs Ctl: "
            f"{row['absolute_difference_events_per_s_per_cell']:+.4f} events/s/cell "
            f"({row['mean_population_rate_treatment']:.4f} vs "
            f"{row['mean_population_rate_control']:.4f})")

    lines += ['', '## The same contrasts as the figures render them', '']
    for outcome in UNIFIED_OUTCOMES:
        for epoch in RECALL_EPOCHS:
            lines.append(f'**{outcome.label} — {epoch}**')
            lines.append('')
            lines += [f'- {ln}' if not ln.startswith('  ') else f'  - {ln.strip()}'
                      for ln in format_contrast_ci_lines(
                          payloads[(outcome.key, epoch)]['contrasts'], 'mCherry')]
            lines.append('')

    # ---- The drafted Results paragraph, with this run's own numbers already substituted --------
    # The DREADD receptor names are used verbatim here rather than GROUP_LABELS' internal 'Exc' /
    # 'Inh' shorthand: this block is manuscript prose, and the receptor is what a reader outside
    # this codebase knows the group by.
    def _clause(outcome_key, group, trailing):
        row = unified_contrast_lookup(contrasts, outcome_key, RECALL_RESPONSE_EPOCH, group)
        stats = (f"ratio {row['ratio']:.2f}, 95% CI {row['ratio_ci_low']:.2f}–"
                 f"{row['ratio_ci_high']:.2f}; Holm-adjusted P = {row['p_holm_epoch']:.3g}")
        if row['holm_epoch_reject']:
            verb = 'it differed from controls' if trailing else 'differed from mCherry controls'
        else:
            verb = ('it did not detectably differ from controls' if trailing
                    else 'did not detectably differ from mCherry controls')
        return f"in {group} mice {verb} ({stats})"

    inter = {row['outcome']: row for _, row in interactions_table.iterrows()}
    lines += [
        f'# Drafted Results paragraph — {session_label}',
        '',
        '*Register is the manuscript\'s: effect size with its interval before any P, the animal '
        'as the unit of inference, and no comparison with the other recall session.*',
        '',
        f'At {session_label.split("(")[0].strip().lower()}, in the absence of CNO, post-tone '
        f'per-event amplitude {_clause("amplitude", "hM3D", False)}, whereas '
        f'{_clause("amplitude", "hM4D", True)}. Post-tone population event rate '
        f'{_clause("population_rate", "hM3D", False)}, and '
        f'{_clause("population_rate", "hM4D", True)}. The group x epoch interaction was '
        f'F({int(inter["amplitude"]["df1"])}, {int(inter["amplitude"]["df2"])}) = '
        f'{inter["amplitude"]["F"]:.2f}, P = {inter["amplitude"]["p"]:.3g} for per-event '
        f'amplitude and F({int(inter["population_rate"]["df1"])}, '
        f'{int(inter["population_rate"]["df2"])}) = {inter["population_rate"]["F"]:.2f}, '
        f'P = {inter["population_rate"]["p"]:.3g} for population event rate.',
        '',
        '**Interpretive constraints for whoever writes this up.**',
        '',
        '- Do not write "the effect disappeared by 1 week", "the effect persisted significantly '
        'longer", or "48 h differed from 1 week". None of those is tested by this analysis, and '
        'the two recall cohorts are not the same animals.',
        '- Do not write "hM3D activation increased activity at recall". No DREADD ligand was '
        'present at recall. The supportable form is: "transient SST-interneuron manipulation '
        'during conditioning was associated with a persistent alteration in later CA1 activity '
        'during drug-free recall."',
        '- A recall difference may reflect altered memory formation, subsequent network '
        'plasticity, a different behavioural state or freezing level, or another downstream '
        'consequence of the conditioning-day manipulation. Do not infer ongoing receptor '
        'activation.',
        '- Retrieval preferentiality is the interaction, not the presence or absence of a star in '
        'the post-tone column.',
        f'- Every null above is reported with its interval. At {cohort} a non-significant result '
        'is weak evidence of absence.',
        '',
    ]

    ensure_dirs(save_dir)
    write_text(os.path.join(save_dir, filename), '\n'.join(lines))


def render_paper_recall_amplitude_rate(PLOTS_DIR, mice_per_group, sessions, session_key,
                                       mapping='full', thres=None,
                                       run_hierarchical_cell_analysis=False,
                                       n_perm_omnibus=HIERARCHICAL_CELL_N_PERM_OMNIBUS,
                                       hierarchical_seed=HIERARCHICAL_CELL_SEED):
    """
    The paper-facing RECALL analysis for ONE session (`Test_B` or `Test_B_1wk`), under
    PLOTS_DIR/sp_rates_lmm/paper/recall/<session_key>/.

    ** Deliberately the same statistical logic as the conditioning analysis. ** Same event
    definition (one contiguous supra-threshold run of `S`, amplitude = integral over the run --
    unchanged, and reached through the same find_event_runs_ca_S traversal), same animal-level
    summarization, the same `log(metric) ~ group * epoch + (1|mouse)` for both outcomes, the same
    within-epoch Holm family, the same joint Wald interaction test, the same figure grammar. The
    manuscript can therefore describe conditioning and recall with one description of method. What
    is different is the epoch set and the cohort, and both are read off the data.

    ** The two recall sessions are analysed SEPARATELY and are never compared here. ** The hM4D
    animal missing at Test_B is not the one missing at Test_B_1wk, so "significant at 48 h and not
    at 1 week" is NOT evidence that the effect declined with time -- it is two independent
    analyses of two different animal sets. A formal 48 h -> 1 week comparison needs a fixed-cohort
    timepoint model restricted to animals present at both sessions, which is out of scope for this
    lane and deliberately not approximated by putting both sessions on one figure.

    ** Group Ns and the denominator df are derived from the session, never hard-coded. ** Each
    model's df = n_animals_present - 1 for that session, which is the same animal-level
    convention the conditioning models use and a different number from their 16.

    ** Amplitude and rate use one identical retained trial set. ** Both outcomes come off the
    single frame returned by restrict_to_exposure_matched_trials over the two 20 s windows, so
    there is no code path in which the rate row describes different trials from the amplitude row
    above it. A tone trial enters only if BOTH windows are completely observed
    (get_recall_epoch_frames); the per-animal coverage is written out rather than left implicit.

    Returns dict(mouse_epoch, fits, contrasts, interactions, coverage).
    """
    session_label = RECALL_SESSION_LABELS.get(session_key, session_key)
    prefix = _RECALL_FILE_PREFIX.get(session_key, session_key.lower())
    stats_prefix = 'unified_recall'
    recall_dir = os.path.join(PLOTS_DIR, 'sp_rates_lmm', 'paper', 'recall', session_key)
    recall_stats_dir = os.path.join(recall_dir, 'stats')
    ensure_dirs(recall_dir, recall_stats_dir)
    _copy_analysis_methods_template(RECALL_PAPER_METHODS_FILENAME, recall_dir)

    print(f'[sp_rates_lmm] Paper recall lane: {session_label}...')

    # ---- Event table over the two duration-matched 20 s windows ------------------------------
    recall_frames_fn = functools.partial(get_recall_epoch_frames,
                                         pre_tone_duration_s=TESTB_PRE_TONE_DURATION_S,
                                         post_tone_duration_s=TESTB_POST_TONE_DURATION_S)
    df_fine = build_epoch_event_table(mice_per_group, sessions, RECALL_EPOCHS,
                                      recall_frames_fn, mapping=mapping, thres=thres)
    df_matched, _counts = restrict_to_exposure_matched_trials(
        df_fine, RECALL_EPOCHS, window_seconds=RECALL_WINDOW_S)
    coverage = build_recall_trial_coverage(session_key, df_fine, df_matched)
    write_text(os.path.join(recall_stats_dir, f'{stats_prefix}_trial_coverage.csv'),
               coverage.to_csv(index=False))
    print(f'[sp_rates_lmm]   {session_key}: retained '
          f'{coverage["n_trials_retained"].sum()} of {coverage["n_tone_trials_present"].sum()} '
          f'animal x tone-trial windows across {len(coverage)} animals.')

    # ---- The inferential dataset: one row per animal x epoch ----------------------------------
    mouse_epoch = build_mouse_epoch_unified_table(df_matched, epochs=RECALL_EPOCHS,
                                                  reference_epoch=RECALL_REFERENCE_EPOCH)
    present = {m for g in GROUP_ORDER for m in mice_per_group.get(g, []) if m in sessions}
    dropped = sorted(present - set(mouse_epoch['mouse']))
    if dropped:
        raise RuntimeError(
            f'render_paper_recall_amplitude_rate [{session_key}]: animal(s) {dropped} have a '
            f'{session_key} recording but contribute no row to the inferential table. That can '
            f'only happen if no tone trial carried both complete 20 s windows; investigate the '
            f'recording rather than letting the cohort shrink silently.')
    n_by_group = (mouse_epoch.drop_duplicates('mouse').groupby('group', observed=True)
                  .size().reindex(DREADD_DISPLAY_ORDER).dropna().astype(int))
    print(f'[sp_rates_lmm]   {session_key} cohort: '
          + ', '.join(f'{g} n={n}' for g, n in n_by_group.items()))
    write_text(os.path.join(recall_stats_dir, f'{stats_prefix}_mouse_epoch_values.csv'),
               mouse_epoch.to_csv(index=False))

    # The synthetic check runs on THIS session's cohort: a uniform treatment shift must give a
    # simple effect with a null interaction, and a post-tone-only shift must make the interaction
    # significant. That second design is precisely the claim the recall figure would make.
    verify_unified_synthetic(
        recall_stats_dir, epochs=RECALL_EPOCHS, reference_epoch=RECALL_REFERENCE_EPOCH,
        response_epoch=RECALL_RESPONSE_EPOCH, across_epoch_family=False,
        group_sizes=tuple((g, int(n_by_group[g])) for g in GROUP_ORDER if g in n_by_group),
        model_label=f'{session_key} recall',
        filename=f'{stats_prefix}_synthetic_verification.txt')

    # ---- The two models ----------------------------------------------------------------------
    fits = {o.key: fit_unified_group_epoch_model(mouse_epoch, o.response_col,
                                                 reference_epoch=RECALL_REFERENCE_EPOCH)
            for o in UNIFIED_OUTCOMES}
    require_common_unified_method(fits)
    contrasts = unified_posthoc_contrasts(fits, mouse_epoch, epochs=RECALL_EPOCHS,
                                          reference_epoch=RECALL_REFERENCE_EPOCH,
                                          across_epoch_family=False)
    contrasts.insert(0, 'session', session_key)
    interactions_table = unified_interactions_table(fits)
    interactions_table.insert(0, 'session', session_key)

    # This session's OWN denominator df and its OWN filenames -- both differ from the conditioning
    # lane's, and the boilerplate used to state the conditioning values here.
    recall_df2 = fits[UNIFIED_OUTCOMES[0].key]['n_mice'] - 1
    recall_pvalue_note = _statsmodels_pvalue_note(
        recall_df2, f'{stats_prefix}_posthoc_contrasts.csv',
        f'{stats_prefix}_interactions.csv', ('p_raw', 'p_holm_epoch'),
        _RECALL_NON_REFERENCE_EPOCH_PHRASE)
    for outcome in UNIFIED_OUTCOMES:
        write_text(os.path.join(recall_stats_dir, f'{stats_prefix}_{outcome.key}_summary.txt'),
                   f"PAPER-FACING recall model ({session_label}): {outcome.label}\n"
                   f"Response: {outcome.response_col} (one value per animal per epoch, "
                   f"{len(mouse_epoch)} rows, {fits[outcome.key]['n_mice']} animals)\n"
                   f"Group x epoch joint Wald: {fits[outcome.key]['omnibus']}\n"
                   f"Contrasts and within-epoch Holm-adjusted p-values (p_holm_epoch): "
                   f"{stats_prefix}_posthoc_contrasts.csv\n"
                   f"Analysed independently of the other recall session -- the cohorts differ.\n\n"
                   f"{recall_pvalue_note}\n\n"
                   f"{fits[outcome.key]['summary_text']}\n\n"
                   f"{_statsmodels_pvalue_footer(recall_df2)}\n")
    write_text(os.path.join(recall_stats_dir, f'{stats_prefix}_interactions.csv'),
               interactions_table.to_csv(index=False))
    write_text(os.path.join(recall_stats_dir, f'{stats_prefix}_posthoc_contrasts.csv'),
               contrasts.to_csv(index=False))
    for _, row in interactions_table.iterrows():
        print(f"[sp_rates_lmm]   {session_key} {row['outcome']}: "
              f"{format_unified_interaction(row)}")

    unified_model_diagnostics(fits, mouse_epoch, contrasts, recall_stats_dir,
                              epochs=RECALL_EPOCHS, reference_epoch=RECALL_REFERENCE_EPOCH,
                              across_epoch_family=False,
                              filename_root=f'{stats_prefix}_diagnostics',
                              residuals_csv=f'{stats_prefix}_residuals.csv',
                              influence_csv=f'{stats_prefix}_influence.csv')

    # ---- The pre->post MODULATION decomposition -----------------------------------------------
    # A representation and pairwise decomposition of the group x epoch interaction already tested
    # above, from the SAME two fits -- no new model, and the omnibus travels with it everywhere.
    # Runs for every session in RECALL_MODULATION_SESSIONS (both recall sessions), each entirely
    # within itself: this session's fits, this session's cohort, this session's df.
    modulation = {'modulation_by_mouse': None, 'modulation_within_group': None,
                  'modulation_pairwise': None, 'trajectory_paired_tests': None}
    if session_key in RECALL_MODULATION_SESSIONS:
        verify_recall_modulation_synthetic(
            recall_stats_dir, epochs=RECALL_EPOCHS, reference_epoch=RECALL_REFERENCE_EPOCH,
            response_epoch=RECALL_RESPONSE_EPOCH,
            group_sizes=tuple((g, int(n_by_group[g])) for g in GROUP_ORDER if g in n_by_group),
            model_label=f'{session_key} recall',
            filename=f'{stats_prefix}_modulation_synthetic_verification.txt')

        modulation_by_mouse = build_recall_modulation_by_mouse(
            mouse_epoch, reference_epoch=RECALL_REFERENCE_EPOCH,
            response_epoch=RECALL_RESPONSE_EPOCH)
        modulation_by_mouse.insert(0, 'session', session_key)
        within_group_df, pairwise_df = recall_modulation_contrasts(
            fits, reference_epoch=RECALL_REFERENCE_EPOCH,
            response_epoch=RECALL_RESPONSE_EPOCH)
        for frame in (within_group_df, pairwise_df):
            frame.insert(0, 'session', session_key)

        # The trajectory panel's own within-group paired tests. Its own file, because it is a
        # SECOND estimate of the within-group change the model contrasts above already report on
        # a different df -- keeping it in its own table (and printing both adjacent in the
        # markdown) is what stops the two disagreeing silently. It decides nothing between groups.
        trajectory_tests = recall_trajectory_paired_tests(modulation_by_mouse)
        trajectory_tests.insert(0, 'session', session_key)

        write_text(os.path.join(recall_stats_dir, f'{stats_prefix}_modulation_by_mouse.csv'),
                   modulation_by_mouse.to_csv(index=False))
        write_text(os.path.join(recall_stats_dir, f'{stats_prefix}_trajectory_paired_tests.csv'),
                   trajectory_tests.to_csv(index=False))
        # Both blocks in one file, distinguished by `block`: the within-group trajectory estimates
        # are only ever read next to the pairwise comparisons they explain, and splitting them
        # into two files invites the descriptive half being quoted on its own.
        write_text(os.path.join(recall_stats_dir, f'{stats_prefix}_modulation_contrasts.csv'),
                   pd.concat([within_group_df, pairwise_df], ignore_index=True)
                   .to_csv(index=False))
        write_recall_modulation_summary(recall_stats_dir, session_key, session_label,
                                        interactions_table, within_group_df, pairwise_df,
                                        stats_prefix, trajectory_df=trajectory_tests)
        plot_recall_modulation(modulation_by_mouse, pairwise_df, interactions_table, recall_dir,
                               session_label, stats_prefix,
                               filename_root=f'{prefix}_amplitude_rate_modulation')
        plot_recall_prepost_trajectories(
            modulation_by_mouse, trajectory_tests, recall_dir, session_label, stats_prefix,
            filename_root=f'{prefix}_amplitude_rate_prepost_trajectories')
        for _, row in pairwise_df.iterrows():
            print(f"[sp_rates_lmm]   {session_key} {row['outcome']} modulation "
                  f"{row['group_a']}-vs-{row['group_b']}: "
                  f"ratio {row['relative_modulation_ratio']:.3f} "
                  f"[{row['ratio_ci_low']:.3f}, {row['ratio_ci_high']:.3f}], "
                  f"P_raw {row['p_raw']:.3g}, P_holm {row['p_holm_modulation']:.3g}")
        for _, row in trajectory_tests.iterrows():
            print(f"[sp_rates_lmm]   {session_key} {row['outcome']} within-group pre->post "
                  f"{row['group']}: {row['post_pre_ratio']:.3f}-fold, "
                  f"t({int(row['df'])}) = {row['t']:.2f}, P_raw {row['p_raw']:.3g}, "
                  f"P_holm {row['p_holm_trajectory']:.3g}")
        modulation = {'modulation_by_mouse': modulation_by_mouse,
                      'modulation_within_group': within_group_df,
                      'modulation_pairwise': pairwise_df,
                      'trajectory_paired_tests': trajectory_tests}

    # ---- The HIERARCHICAL CELL-LEVEL companion / sensitivity suite -----------------------------
    # Strictly ADDITIVE. Everything above this block -- the models, the contrasts, the Holm
    # families, the omnibus tests, the modulation decomposition and every figure and table already
    # written -- is untouched and remains the paper-facing analysis. This suite asks the
    # complementary question (does the modulation pattern occur coherently across the cellular
    # population WITHIN mice, with the cell hierarchy modelled instead of collapsed?) and writes
    # into its own subdirectory so its numbers can never be confused with the primary lane's.
    # RECALL_HIERARCHICAL_CELL_SESSIONS now names both recall sessions, so the real gate is
    # run_hierarchical_cell_analysis, which is False by default -- this is an explicitly invoked
    # companion, not part of a routine pass (it costs ~40 min per session against the few minutes
    # the rest of the analysis takes, so an explicit pass now spends ~80 min here). When it does
    # run, every one of its components runs.
    hierarchical = {}
    if run_hierarchical_cell_analysis and session_key in RECALL_HIERARCHICAL_CELL_SESSIONS:
        if session_key not in RECALL_MODULATION_SESSIONS:
            raise RuntimeError(
                f'render_paper_recall_amplitude_rate [{session_key}]: the hierarchical cell suite '
                f'needs the mouse-level modulation contrasts to compare itself against, but this '
                f'session is not in RECALL_MODULATION_SESSIONS.')
        hierarchical = run_hierarchical_cell_suite(
            recall_dir, session_key, session_label, df_matched, mice_per_group,
            interactions_table, modulation['modulation_pairwise'],
            n_perm_omnibus=n_perm_omnibus, seed=hierarchical_seed)

    # ---- Figures, annotated entirely from the tables above ------------------------------------
    preamble = _recall_contrasts_preamble(session_label, n_by_group, stats_prefix)
    plot_paper_epoch_distributions(
        df_matched, mouse_epoch, contrasts, recall_dir, epochs=RECALL_EPOCHS,
        filename_root=f'{prefix}_amplitude_rate_by_epoch',
        epoch_labels=_RECALL_EPOCH_LABELS, contrasts_preamble=preamble,
        contrasts_title=f'{session_label} — figure contrasts, unified mixed models')

    interaction_notes = {row['outcome']: format_unified_interaction(row)
                         for _, row in interactions_table.iterrows()}
    # Rows in the SAME order as the distribution figure above (amplitude on top, rate below), so
    # the two figures of one session can be read against each other row by row. The TFC forest's
    # PAPER_COMPONENT_KEYS order is the reverse and is left alone.
    forest_row_keys = ('amplitude', 'population_rate')
    plot_decomposition_grid(
        df_matched, recall_dir, epochs=RECALL_EPOCHS,
        components=[_DECOMPOSITION_COMPONENTS_BY_KEY[k] for k in forest_row_keys],
        include_exc_vs_inh=False, filename_root=f'{prefix}_amplitude_rate_forest',
        contrast_payloads=_unified_contrast_payloads(contrasts, epochs=RECALL_EPOCHS),
        interaction_note={k: interaction_notes[k] for k in forest_row_keys},
        # Kept shorter than the figure's own title line, which sets the width: a two-column
        # forest is ~4.6 in wide and a longer subtitle is clipped at both edges.
        subtitle=f'{session_label}; per-row group x epoch joint Wald test',
        contrasts_preamble=preamble,
        contrasts_note='**This forest carries no significance stars by design.** It is a second '
                       'view of the same contrast table the distribution figure is annotated '
                       'from, drawn as effect sizes; the Holm-adjusted decisions are in '
                       f'`stats/{stats_prefix}_posthoc_contrasts.csv`.',
        row_height=1.9,
        # Both columns are complete 20 s windows on the retained trials by construction.
        reduced_coverage_epochs=(), epoch_labels=_RECALL_EPOCH_LABELS)

    write_recall_results_summary(recall_stats_dir, session_key, session_label, mouse_epoch, fits,
                                 contrasts, interactions_table, coverage, stats_prefix)
    print(f'[sp_rates_lmm] {session_key} recall figures written to {recall_dir}')
    return {'mouse_epoch': mouse_epoch, 'fits': fits, 'contrasts': contrasts,
            'interactions': interactions_table, 'coverage': coverage, **modulation,
            **hierarchical}


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
                                          orientation='horizontal',
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

    orientation : 'horizontal' (groups side by side) or 'vertical' (groups stacked in rows,
                  for narrow figure slots).
    """
    if orientation not in ('horizontal', 'vertical'):
        raise ValueError(f"plot_width_vs_height_matched_examples: orientation must be "
                         f"'horizontal' or 'vertical', got {orientation!r}.")
    target = np.percentile(df_runs_trace['peak_height'], target_height_percentile)
    n = len(DREADD_DISPLAY_ORDER)
    if orientation == 'horizontal':
        fig, axs = plt.subplots(1, n, figsize=(2.0 * n, 2.3), squeeze=False)
        axs = axs[0]
    else:
        fig, axs = plt.subplots(n, 1, figsize=(2.3, 1.6 * n), squeeze=False)
        axs = axs[:, 0]

    for ax, group in zip(axs, DREADD_DISPLAY_ORDER):
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
        st, w = int(start[i]), int(width[i])
        pad = int(round(EXAMPLE_TRACE_PAD_S * MINISCOPE_FPS))
        win_lo, win_hi = max(0, st - pad), min(S_row.shape[0], st + w + pad)
        t = (np.arange(win_lo, win_hi) - st) / MINISCOPE_FPS
        s_win = S_row[win_lo:win_hi]

        ax.plot(t, s_win, color=GROUP_COLOURS[group], linewidth=1.1)
        ax.axvspan(0, w / MINISCOPE_FPS, color=GROUP_COLOURS[group], alpha=0.18)
        ax.axhline(cell_thres, color='k', linewidth=0.5, linestyle=':')
        ax.spines[['right', 'top']].set_visible(False)
        ax.set_title(GROUP_LABELS[group], size='small')
        ax.set_xlim(-1, 1)
        if orientation == 'horizontal':
            ax.set_xlabel('Time from run start (s)', size='small')
            if ax is axs[0]:
                ax.set_ylabel('S (raw)', size='small')
        else:
            ax.set_ylabel('S (raw)', size='small')
            if ax is axs[-1]:
                ax.set_xlabel('Time from run start (s)', size='small')

    plt.tight_layout(pad=0.6)
    _save_panel(fig, save_dir, filename_root)


# ─────────────────────────────────────────────────────────────────────────────
# Top-level orchestrator (mirrors caban.freezing_tuned_cells.run_freezing_tuned_cells /
# caban.population_coupling.run_population_coupling: one entry point, called from
# caban/sections.py, that builds every table, fits every model, writes every stats file, and
# saves every figure panel this analysis produces)
# ─────────────────────────────────────────────────────────────────────────────

def run_sp_rates_lmm(PLOTS_DIR, mice_per_group, TFC_cond, TFC_cond_LT1, TFC_cond_LT2,
                        Test_B, Test_B_1wk, mapping='full', thres=None, n_perm=20000, seed=0,
                        rate_draws=1000, rate_tune=1000, rate_chains=4, auto_close=True,
                        run_hierarchical_cell_analysis=False,
                        n_perm_omnibus=HIERARCHICAL_CELL_N_PERM_OMNIBUS):
    """
    Full cell-level event-amplitude analysis: primary trace-period amplitude, co-primary
    within-cell epoch delta, run-structure/threshold-sensitivity bursting evidence, group x trial
    photobleaching check, secondary rate/fraction-active/Test_B endpoints, LT1->LT2 manipulation
    check + detection-dropout measurement, permutation-test sensitivity (both weightings), example
    -trace panels, and every figure panel this analysis produces. See
    analysis_methods_templates/sp_rates_lmm_methods.md for the full statistical rationale behind
    every choice made here, and this module's own CHANGELOG docstring for what changed and why.

    Ends with the two PAPER-FACING lanes: the unified conditioning models
    (render_paper_tfc_amplitude_rate) and then the unified recall models, one INDEPENDENT analysis
    per recall session (render_paper_recall_amplitude_rate for Test_B and Test_B_1wk). Everything
    before them is the internal/sensitivity record.

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
    run_hierarchical_cell_analysis : the ONE gate on the additive hierarchical cell-level
                              companion suite for the recall lane (both recall sessions; see
                              RECALL_HIERARCHICAL_CELL_SESSIONS). ** False by default: the
                              companion is EXPLICITLY INVOKED, not part of a routine pass. ** It
                              costs ~40 min PER SESSION, so ~80 min over the two recall sessions
                              (exact mouse-label MixedLM enumerations over the
                              paired-cell table, plus the hierarchical NB count model and its
                              prior/posterior-predictive checks), which is an order of magnitude
                              more than the rest of this analysis, and it is a sensitivity
                              question rather than a paper-facing one. Pass True to rerun it;
                              that runs EVERY component. There is no per-component switch -- it
                              is one predefined analysis, and nothing in it is activated or
                              omitted because of what another result showed.
    n_perm_omnibus           : Monte Carlo draws for the hierarchical suite's OMNIBUS
                              randomization test only -- its pairwise tests are exact
                              enumerations and take no such parameter. Production is fixed at
                              HIERARCHICAL_CELL_N_PERM_OMNIBUS with seed
                              HIERARCHICAL_CELL_SEED; a development pass may lower it, and the
                              value actually used is written into the output.
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
    plot_width_vs_height_matched_examples(TFC_cond, df_runs_trace, out_dir, mapping=mapping, thres=thres,
                                         orientation='vertical',
                                         filename_root='width_height_matched_examples_vertical')

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
    # ** This is where the PAPER-FACING analysis is fit. ** Everything above is the internal
    # record: the cell-level amplitude model and its Holm family, the permutation tests, the
    # Bayesian NB rate model, the BH-FDR family, the sensitivity analyses. The manuscript reports
    # none of those directly -- it reports the two unified mouse-level models built here, which
    # give per-event amplitude and population event rate one identical statistical treatment. The
    # objects above are passed in so the summary can report them, clearly separated, as
    # sensitivity evidence.
    #
    # It goes last because that summary quotes the BH-adjusted permutation q-values, which do not
    # exist until the secondary family is complete.
    render_paper_tfc_amplitude_rate(PLOTS_DIR, df_grid, primary_contrasts, holm,
                                    perm_results, interactions, _q_by_name, rate_fit, delta_df)

    # ---- The paper-facing RECALL lane, one INDEPENDENT analysis per recall session -------------
    # Same statistical logic as the conditioning lane above, applied to the two duration-matched
    # 20 s windows around each recall tone. Test_B and Test_B_1wk are fit separately and never
    # compared with each other: the animal missing at 48 h is not the animal missing at 1 week,
    # so a difference in which session reaches significance is not a difference between the
    # timepoints. See render_paper_recall_amplitude_rate.
    for recall_key, recall_sessions in (('Test_B', Test_B), ('Test_B_1wk', Test_B_1wk)):
        render_paper_recall_amplitude_rate(
            PLOTS_DIR, mice_per_group, recall_sessions, recall_key, mapping=mapping, thres=thres,
            run_hierarchical_cell_analysis=run_hierarchical_cell_analysis,
            n_perm_omnibus=n_perm_omnibus)

    print('[sp_rates_lmm] Done.')
    if auto_close:
        plt.close('all')
