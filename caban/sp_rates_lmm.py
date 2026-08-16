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

Module layout
-------------
  Event table construction  -- build_epoch_event_table(), aggregate_over_trials(),
                                filter_amplitude_rows(), build_mouse_trial_epoch_rate_table()
  Confirmatory models        -- fit_primary_trace_amplitude(), fit_epoch_group_interaction(),
                                fit_rate_group_epoch_model(), holm_correct_confirmatory()
  Small-n inference           -- mouse_label_permutation_test(), make_amplitude_contrast_stat()
  Detection-bias / manip check -- compute_lt1_lt2_dropout(), compute_lt1_lt2_amplitude_delta()
  Descriptive/triangulation  -- mouse_level_trace_amplitude(), fraction_active_table()
  Figures                     -- plot_primary_trace_amplitude(), plot_epoch_profile(),
                                plot_amplitude_ecdf(), plot_decomposition(),
                                plot_manipulation_check()
"""
import os
import functools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
import bambi as bmb
import arviz as az

from caban.utilities import find_event_runs_ca_S, MINISCOPE_FPS
from caban.single_unit_common import (
    GROUP_ORDER, GROUP_LABELS, GROUP_COLOURS,
    ensure_dirs, write_text, save_fig, ecdf_panel,
    fit_mixed_model, joint_wald_test,
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


# ─────────────────────────────────────────────────────────────────────────────
# Event table construction
# ─────────────────────────────────────────────────────────────────────────────

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

    'cell' is the mapping's own unit_id (get_S_mapping's S_idx), which by construction indexes
    the SAME row position in the returned (already-subsetted) S matrix -- see get_S_mapping's
    docstring and get_S_indeces()'s "sorted according to the unit_id list" guarantee. This
    function calls find_event_runs_ca_S() directly on that subsetted S, so it never needs to
    touch S_spikes' differently-keyed (full-session-position) dict.

    ** 'cell' is SESSION-LOCAL -- never join it across sessions. ** Under a cross-registration
    mapping, the same physical cell carries a DIFFERENT unit_id in each session (each session
    contributes its own column of the crossreg table), so equal 'cell' values in two sessions'
    tables are unrelated cells and unequal ones may well be the same cell. The cross-session
    correspondence is ROW ORDER within a mapping, which is why
    get_actual_cells_from_df_session() must never sort. To pair cells across sessions, zip the
    two S_idx lists positionally -- see compute_lt1_lt2_amplitude_delta().
    """
    rows = []
    for group in GROUP_ORDER:
        for mouse in mice_per_group.get(group, []):
            if mouse not in sessions:
                continue
            session = sessions[mouse]
            cell_thres = session.thres if thres is None else thres
            [S, _S_spikes, _S_peakval, S_idx] = session.get_S_mapping(mapping, with_crossreg=crossreg_to_use)
            if S.shape[0] == 0:
                raise RuntimeError(
                    f'build_epoch_event_table: mouse {mouse} has 0 cells for mapping={mapping!r}.')
            events = find_event_runs_ca_S(S, cell_thres)

            for trial_idx in session.periods:
                for epoch in epoch_names:
                    frames = get_frames_fn(session, epoch, trial_idx)
                    if frames is None:
                        continue
                    onset, offset = frames
                    exposure_s = (offset - onset) / MINISCOPE_FPS
                    for cell_row, cell_id in enumerate(S_idx):
                        frameidx, amplitude, _n_local_max = events[cell_row]
                        in_window = (frameidx >= onset) & (frameidx < offset)
                        n_events = int(np.sum(in_window))
                        sum_amp = float(np.sum(amplitude[in_window])) if n_events else 0.0
                        rows.append({
                            'mouse': mouse, 'group': group, 'trial': int(trial_idx), 'epoch': epoch,
                            'cell': cell_id, 'n_events': n_events, 'sum_amplitude': sum_amp,
                            'exposure_seconds': exposure_s,
                        })
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError('build_epoch_event_table: produced an empty table -- check '
                           'mapping/epoch_names/get_frames_fn inputs.')
    return df


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

    Used to build the PRIMARY endpoint's cell-level trace table; the CO-PRIMARY (group x epoch
    interaction) model instead uses the fine per-trial table directly, with trial as its own
    fixed effect.
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
    epoch) by summing n_events across cells (Y_mte in the plan's notation) and taking
    exposure_seconds (constant across cells within one mouse-trial-epoch window; E_mte). This is
    the population-level table the SECONDARY rate endpoint is fit on -- see
    fit_rate_group_epoch_model() -- not the cell-level table the amplitude endpoints use.
    """
    agg = df_fine.groupby(['mouse', 'group', 'trial', 'epoch'], as_index=False).agg(
        n_events=('n_events', 'sum'),
        exposure_seconds=('exposure_seconds', 'first'),
    )
    return agg


def fraction_active_table(df_epoch_slice):
    """
    Prespecified secondary endpoint: fraction of cells with >=1 event, per mouse, for one epoch
    slice (output of aggregate_over_trials() or a df_fine subset already restricted to one
    epoch). One row per mouse -- fraction active is inherently a per-mouse population statistic,
    unlike the per-cell amplitude/rate tables it is computed from.
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
    plot can be checked against the cell-level model estimate. If a conclusion depends on the
    model and is invisible at this level, it should not be claimed.
    """
    return df_amp_pooled.groupby(['mouse', 'group'], as_index=False)['log_amplitude'].mean()


# ─────────────────────────────────────────────────────────────────────────────
# Confirmatory models
# ─────────────────────────────────────────────────────────────────────────────

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


def fit_primary_trace_amplitude(df_trace_pooled, reference='mCherry'):
    """
    PRIMARY endpoint: log(mean per-event amplitude) ~ group, cell-level, trace epoch pooled
    across trials, random intercept on mouse (falls back to mouse-clustered OLS on degeneracy --
    see caban.single_unit_common.fit_mixed_model).

    df_trace_pooled : output of filter_amplitude_rows(aggregate_over_trials(df_fine, 'trace')).

    Returns dict(result, method, summary_text, omnibus, formula). omnibus is the joint Wald test
    (caban.single_unit_common.joint_wald_test) that both non-reference group coefficients are
    zero -- the single primary confirmatory p-value.
    """
    df = df_trace_pooled.copy()
    df['group'] = pd.Categorical(df['group'], categories=[reference] + [g for g in GROUP_ORDER if g != reference])
    formula = f"log_amplitude ~ C(group, Treatment(reference='{reference}'))"
    result, method, text = fit_mixed_model(df, formula, group_col='mouse')
    fe_names = list(result.fe_params.index) if hasattr(result, 'fe_params') else list(result.params.index)
    omnibus = joint_wald_test(result, _nonref_group_coef_names(fe_names, reference))
    return {'result': result, 'method': method, 'summary_text': text, 'omnibus': omnibus, 'formula': formula}


def fit_epoch_group_interaction(df_fine_amp, reference_group='mCherry', reference_epoch=TFC_REFERENCE_EPOCH):
    """
    CO-PRIMARY endpoint: log(mean per-event amplitude) ~ group * epoch + trial, cell x trial x
    epoch level, random intercept on mouse. Tests whether the treatment effect is epoch-specific
    rather than a global shift.

    df_fine_amp : output of filter_amplitude_rows(df_fine), where df_fine's 'epoch' column is
                  already restricted to the epochs to test (TFC_EPOCHS -- shock excluded upstream
                  by the build_epoch_event_table() call, not here).

    Returns dict(result, method, summary_text, omnibus, formula, interaction_names). omnibus is
    the joint Wald test that every group x epoch interaction coefficient is zero -- the single
    co-primary confirmatory p-value.
    """
    df = df_fine_amp.copy()
    df['group'] = pd.Categorical(df['group'], categories=[reference_group] + [g for g in GROUP_ORDER if g != reference_group])
    other_epochs = [e for e in df['epoch'].unique() if e != reference_epoch]
    df['epoch'] = pd.Categorical(df['epoch'], categories=[reference_epoch] + other_epochs)
    df['trial'] = df['trial'].astype(int)

    formula = (f"log_amplitude ~ C(group, Treatment(reference='{reference_group}')) * "
              f"C(epoch, Treatment(reference='{reference_epoch}')) + C(trial)")
    result, method, text = fit_mixed_model(df, formula, group_col='mouse')

    fe_names = list(result.fe_params.index) if hasattr(result, 'fe_params') else list(result.params.index)
    interaction_names = [n for n in fe_names if ':' in n]
    if len(interaction_names) == 0:
        raise RuntimeError(f'fit_epoch_group_interaction: no interaction terms found in {fe_names}')
    omnibus = joint_wald_test(result, interaction_names)
    return {'result': result, 'method': method, 'summary_text': text, 'omnibus': omnibus,
            'formula': formula, 'interaction_names': interaction_names}


def holm_correct_confirmatory(primary_p, coprimary_p):
    """
    Holm correction across the ENTIRE confirmatory family: the primary trace-period group
    omnibus p-value and the co-primary group x epoch interaction omnibus p-value. This is the
    complete confirmatory multiplicity burden -- everything else in this module is secondary
    (BH-FDR, see fdr_correct in caban.single_unit_common) or purely descriptive/sensitivity.

    Returns dict keyed 'trace' and 'interaction', each {p_raw, p_holm, reject} at alpha=0.05.
    """
    reject, p_holm, _, _ = multipletests([primary_p, coprimary_p], alpha=0.05, method='holm')
    return {
        'trace': {'p_raw': float(primary_p), 'p_holm': float(p_holm[0]), 'reject': bool(reject[0])},
        'interaction': {'p_raw': float(coprimary_p), 'p_holm': float(p_holm[1]), 'reject': bool(reject[1])},
    }


def fit_rate_group_epoch_model(df_mte, reference_group='mCherry', reference_epoch=TFC_REFERENCE_EPOCH,
                               epoch_categories=None, draws=1000, tune=1000, chains=4, seed=0):
    """
    SECONDARY endpoint: event count ~ group * epoch + trial, with a log(exposure_seconds) offset,
    at the mouse x trial x epoch level (df_mte from build_mouse_trial_epoch_rate_table()).

    Negative-binomial mixed model (random intercept on mouse) fit via Bambi (PyMC backend) -- a
    genuinely joint fit, with dispersion (alpha) estimated together with the fixed and random
    effects. statsmodels has no NB-GLMM path (mixedlm is Gaussian-only, which the amplitude
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

    group_term = f"C(group, Treatment('{reference_group}'))"
    epoch_term = f"C(epoch, Treatment('{reference_epoch}'))"
    formula_full = f"n_events ~ {group_term} * {epoch_term} + trial + offset(log_exposure) + (1|mouse)"
    formula_reduced = f"n_events ~ {group_term} + {epoch_term} + trial + offset(log_exposure) + (1|mouse)"

    fit_kwargs = dict(draws=draws, tune=tune, chains=chains, random_seed=seed, progressbar=False,
                      idata_kwargs={'log_likelihood': True})

    model_full = bmb.Model(formula_full, data=df, family='negativebinomial')
    idata_full = model_full.fit(**fit_kwargs)

    model_reduced = bmb.Model(formula_reduced, data=df, family='negativebinomial')
    idata_reduced = model_reduced.fit(**fit_kwargs)

    comparison = az.compare({'group_x_epoch': idata_full, 'no_interaction': idata_reduced})

    interaction_name = f'{group_term}:{epoch_term}'
    summary_full = az.summary(idata_full, var_names=[interaction_name], filter_vars='like')

    text = (f'Formula (full): {formula_full}\n'
           f'Formula (reduced, no group x epoch): {formula_reduced}\n\n'
           f'Interaction coefficients (posterior mean, sd, 94% HDI):\n{summary_full.to_string()}\n\n'
           f'LOO model comparison (full vs reduced; positive elpd_diff favours "full"):\n'
           f'{comparison.to_string()}\n')
    return {'idata_full': idata_full, 'idata_reduced': idata_reduced, 'comparison': comparison,
            'summary_full': summary_full, 'summary_text': text,
            'formula_full': formula_full, 'formula_reduced': formula_reduced}


def report_decomposition_additivity(rate_group_coef, amplitude_group_coef):
    """
    Consistency check: log(total S/sec) group coefficient should equal
    log(rate) group coefficient + log(mean amplitude) group coefficient, since
    summed S/sec = rate x mean amplitude exactly at the cell-window level (event-weighted when
    aggregated -- see sp_rates_lmm_methods.md's weighting-rule note).

    This does NOT fit a third model. It is an arithmetic check on coefficients already fit
    separately by fit_rate_group_epoch_model() and fit_primary_trace_amplitude() /
    fit_epoch_group_interaction(). No significance test is attached to the sum: a proper SE for
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
                     new group buckets and compute the contrast) -- see
                     make_amplitude_contrast_stat() for the standard factory.
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


def make_amplitude_contrast_stat(df_amp, group_a, group_b, reduce_fn=None):
    """
    Build a stat_fn for mouse_label_permutation_test() that contrasts pooled cell-level
    log_amplitude between group_a and group_b under a given (possibly permuted) mouse-to-group
    mapping.

    df_amp    : an amplitude table already restricted to n_events > 0 and carrying 'log_amplitude'
                (output of filter_amplitude_rows()).
    reduce_fn : callable(vals_a, vals_b) -> float. Defaults to difference in means. Pass e.g.
                ``lambda a, b: np.percentile(a, 90) - np.percentile(b, 90)`` for the tail contrast
                that most directly targets the bursting hypothesis (fattened right tail, not just
                a mean shift).

    Cell-weighted, not mouse-weighted: every cell counts once regardless of which mouse or how
    many cells that mouse contributed, consistent with the cell-level unit of observation this
    analysis otherwise commits to. Only the GROUP LABEL differs across permutations -- the
    weighting rule does not.
    """
    if reduce_fn is None:
        reduce_fn = lambda a, b: float(np.mean(a) - np.mean(b))

    def stat_fn(mouse_to_group):
        assigned = df_amp['mouse'].map(mouse_to_group)
        vals_a = df_amp.loc[assigned == group_a, 'log_amplitude'].to_numpy()
        vals_b = df_amp.loc[assigned == group_b, 'log_amplitude'].to_numpy()
        if len(vals_a) == 0 or len(vals_b) == 0:
            return np.nan
        return float(reduce_fn(vals_a, vals_b))
    return stat_fn


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
    cross-registered 'LT1+LT2' cell set. Same day, same task, same FOV -- CNO the only difference
    -- so each cell is its own control and the group contrast is a difference-in-differences that
    removes between-animal baseline variance. This establishes the tool works; it is NOT a test
    of the memory hypothesis, so it needs no multiplicity correction with the confirmatory family.

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
                _f1, amp1, _n1 = events1[cell_row]
                _f2, amp2, _n2 = events2[cell_row]
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
    fe_names = list(result.fe_params.index) if hasattr(result, 'fe_params') else list(result.params.index)
    omnibus = joint_wald_test(result, _nonref_group_coef_names(fe_names, reference))
    return {'result': result, 'method': method, 'summary_text': text, 'omnibus': omnibus, 'formula': formula}


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


def plot_primary_trace_amplitude(df_trace_pooled, save_dir, filename_root='primary_trace_amplitude'):
    """
    Panel 1: 17 mouse-level points (mouse_level_trace_amplitude()) + violin group summaries +
    Holm-corrected pairwise brackets. The confirmatory omnibus estimate/CI lives in the
    companion stats .txt (fit_primary_trace_amplitude()'s summary_text) written alongside this
    figure by the caller -- this panel is the visual triangulation check, not the sole home of
    the inferential claim.
    """
    mouse_df = mouse_level_trace_amplitude(df_trace_pooled)
    values_per_group = {}
    for group in GROUP_ORDER:
        vals = mouse_df.loc[mouse_df['group'] == group, 'log_amplitude'].to_numpy()
        if len(vals) < 2:
            raise RuntimeError(f'plot_primary_trace_amplitude: group {group} has {len(vals)} '
                               f'mouse(s); need >=2 for a group comparison.')
        values_per_group[group] = vals.reshape(-1, 1)

    fig, ax = plt.subplots(figsize=_PANEL_FIGSIZE)
    ax.spines[['right', 'top']].set_visible(False)
    _draw_violin_triplet(ax, values_per_group, 0, GROUP_ORDER, GROUP_COLOURS,
                         stat_fn=do_pairwise_holm_plot, ylabel=_YLABEL_LOG_AMPLITUDE)
    ax.set_xticks(range(len(GROUP_ORDER)))
    ax.set_xticklabels([GROUP_LABELS[g] for g in GROUP_ORDER], size='medium')
    ax.set_title('Trace-period amplitude\n(mouse-level, n={})'.format(len(mouse_df)), size='small')
    plt.tight_layout(pad=0.5)
    _save_panel(fig, save_dir, filename_root)


def plot_epoch_profile(df_fine_amp, save_dir, filename_root='epoch_profile',
                       epoch_order=TFC_EPOCHS):
    """
    Panel 2: descriptive (not model-marginal) per-mouse-per-epoch pooled mean log-amplitude,
    plotted by group across epoch order with faint per-animal trajectories behind the group
    mean +/- SEM line. Model-estimated marginal effects (from fit_epoch_group_interaction) are
    reported in the companion stats .txt, not extracted into this panel, to avoid the added
    complexity of a margeff pipeline for a first pass -- see sp_rates_lmm_methods.md.
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


def plot_amplitude_ecdf(df_trace_pooled, save_dir, filename_root='amplitude_ecdf'):
    """
    Panel 3: pooled per-cell ECDF of trace-period log-amplitude by group (bold), with per-mouse
    ECDFs (thin) behind -- the bursting hypothesis predicts a fattened right tail, which a mean
    contrast alone can miss. Reuses caban.single_unit_common.ecdf_panel directly.
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
              xlabel=_YLABEL_LOG_AMPLITUDE, title='Trace-period amplitude distribution')
    plt.tight_layout(pad=0.5)
    _save_panel(fig, save_dir, filename_root)


def plot_decomposition(df_trace_pooled_raw, save_dir, filename_root='decomposition'):
    """
    Panel 4: total S/s, fraction active, rate among active cells, and mean per-event amplitude,
    each as a per-mouse violin+scatter panel -- total visually dominant since the other three are
    conditional decompositions of it (summed S/s = rate x mean amplitude), not four independent
    activity measures.

    df_trace_pooled_raw : output of aggregate_over_trials(df_fine, 'trace') BEFORE
                          filter_amplitude_rows() -- i.e. still carrying zero-event cells, since
                          fraction-active and total-S/s need them.
    """
    df = df_trace_pooled_raw.copy()
    df['total_per_s'] = df['sum_amplitude'] / df['exposure_seconds']
    df['active'] = df['n_events'] > 0

    mouse_total = df.groupby(['mouse', 'group'], as_index=False)['total_per_s'].mean()
    mouse_frac_active = fraction_active_table(df)
    active_only = df[df['active']].copy()
    active_only['rate_active'] = active_only['n_events'] / active_only['exposure_seconds']
    mouse_rate_active = active_only.groupby(['mouse', 'group'], as_index=False)['rate_active'].mean()
    amp_df = filter_amplitude_rows(df)
    mouse_amp = mouse_level_trace_amplitude(amp_df)

    panels = [
        (mouse_total, 'total_per_s', 'Total S/s'),
        (mouse_frac_active, 'fraction_active', 'Fraction active'),
        (mouse_rate_active, 'rate_active', 'Rate among active cells (/s)'),
        (mouse_amp, 'log_amplitude', _YLABEL_LOG_AMPLITUDE),
    ]
    fig, axs = plt.subplots(1, 4, figsize=(11, 3.2))
    for ax, (mouse_df, col, title) in zip(axs, panels):
        ax.spines[['right', 'top']].set_visible(False)
        values_per_group = {}
        for group in GROUP_ORDER:
            vals = mouse_df.loc[mouse_df['group'] == group, col].to_numpy()
            if len(vals) < 2:
                raise RuntimeError(f'plot_decomposition: group {group} has {len(vals)} mouse(s) '
                                   f'for {col!r}; need >=2.')
            values_per_group[group] = vals.reshape(-1, 1)
        _draw_violin_triplet(ax, values_per_group, 0, GROUP_ORDER, GROUP_COLOURS,
                             stat_fn=do_pairwise_holm_plot, ylabel=title)
        ax.set_xticks(range(len(GROUP_ORDER)))
        ax.set_xticklabels([GROUP_LABELS[g] for g in GROUP_ORDER], size='small')
        ax.set_title(title, size='small')
    fig.suptitle('Trace-period activity decomposition', size='medium')
    fig.subplots_adjust(left=0.06, bottom=0.14, right=0.98, top=0.86, wspace=0.45)
    _save_panel(fig, save_dir, filename_root)


def plot_manipulation_check(df_delta, dropout_df, save_dir, filename_root='manipulation_check'):
    """
    Panel 5: LT1->LT2 within-cell delta log-amplitude by group (left) and LT1->LT2 detection
    dropout fraction by group (right) -- the manipulation check plus the Tier-1 detection-bias
    measurement, side by side since both come from the same drug-free-to-drug session pair.
    """
    mouse_delta = df_delta.groupby(['mouse', 'group'], as_index=False)['delta_log_amplitude'].mean()

    fig, axs = plt.subplots(1, 2, figsize=(6.5, 3.5))
    for ax, (mouse_df, col, title) in zip(
        axs, [(mouse_delta, 'delta_log_amplitude', 'LT2 - LT1 Delta log(amplitude)'),
             (dropout_df, 'dropout_fraction', 'LT1->LT2 dropout fraction')]):
        ax.spines[['right', 'top']].set_visible(False)
        values_per_group = {}
        for group in GROUP_ORDER:
            vals = mouse_df.loc[mouse_df['group'] == group, col].to_numpy()
            if len(vals) < 2:
                raise RuntimeError(f'plot_manipulation_check: group {group} has {len(vals)} '
                                   f'mouse(s) for {col!r}; need >=2.')
            values_per_group[group] = vals.reshape(-1, 1)
        _draw_violin_triplet(ax, values_per_group, 0, GROUP_ORDER, GROUP_COLOURS,
                             stat_fn=do_pairwise_holm_plot, ylabel=title)
        ax.set_xticks(range(len(GROUP_ORDER)))
        ax.set_xticklabels([GROUP_LABELS[g] for g in GROUP_ORDER], size='small')
        ax.set_title(title, size='small')
    fig.suptitle('Manipulation check (LT1 drug-free -> LT2 CNO)', size='medium')
    fig.subplots_adjust(left=0.1, bottom=0.14, right=0.98, top=0.84, wspace=0.5)
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
    group x epoch interaction, secondary rate/fraction-active/Test_B endpoints, LT1->LT2
    manipulation check + detection-dropout measurement, permutation-test sensitivity, and all
    five figure panels. See analysis_methods_templates/sp_rates_lmm_methods.md for the full
    statistical rationale behind every choice made here.

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

    print('[sp_rates_lmm] Building TFC_cond event table...')
    tfc_frames_fn = functools.partial(get_epoch_frames, pre_tone_duration_s=35.0)
    df_fine = build_epoch_event_table(mice_per_group, TFC_cond, TFC_EPOCHS, tfc_frames_fn,
                                      mapping=mapping, thres=thres)

    # ---- Primary: trace-period amplitude, pooled across trials -------------------------------
    df_trace_raw = aggregate_over_trials(df_fine, TFC_TRACE_EPOCH)
    df_trace_amp = filter_amplitude_rows(df_trace_raw)

    print('[sp_rates_lmm] Fitting primary trace-amplitude model...')
    primary = fit_primary_trace_amplitude(df_trace_amp)
    write_text(os.path.join(stats_dir, 'primary_trace_amplitude.txt'),
              f"PRIMARY: log(mean per-event amplitude) ~ group, trace epoch, pooled trials\n"
              f"Omnibus (joint Wald, both non-reference groups): {primary['omnibus']}\n\n"
              f"{primary['summary_text']}")
    plot_primary_trace_amplitude(df_trace_amp, out_dir)

    # ---- Co-primary: group x epoch interaction -------------------------------------------------
    df_fine_amp = filter_amplitude_rows(df_fine)
    print('[sp_rates_lmm] Fitting co-primary group x epoch interaction model...')
    coprimary = fit_epoch_group_interaction(df_fine_amp)
    write_text(os.path.join(stats_dir, 'coprimary_epoch_interaction.txt'),
              f"CO-PRIMARY: log(mean per-event amplitude) ~ group * epoch + trial\n"
              f"Omnibus (joint Wald, all interaction terms): {coprimary['omnibus']}\n\n"
              f"{coprimary['summary_text']}")
    plot_epoch_profile(df_fine_amp, out_dir)

    # ---- Holm correction across the confirmatory family (the ONLY multiplicity burden here) ---
    holm = holm_correct_confirmatory(primary['omnibus']['p'], coprimary['omnibus']['p'])
    write_text(os.path.join(stats_dir, 'confirmatory_holm_correction.txt'),
              f"Holm correction across the two confirmatory omnibus tests:\n{holm}\n")
    print(f"[sp_rates_lmm] Confirmatory (Holm-corrected): {holm}")

    # ---- Amplitude distribution (tail) ----------------------------------------------------------
    plot_amplitude_ecdf(df_trace_amp, out_dir)

    # ---- Decomposition (total S/s, fraction active, rate|active, amplitude) ----------------------
    plot_decomposition(df_trace_raw, out_dir)

    # ---- Secondary: rate (Bambi NB-GLMM; see fit_rate_group_epoch_model docstring) ------------------
    print('[sp_rates_lmm] Fitting secondary rate model (Bambi NB-GLMM; this samples via MCMC, '
         'expect roughly a minute for two model fits)...')
    df_mte = build_mouse_trial_epoch_rate_table(df_fine)
    rate_fit = fit_rate_group_epoch_model(df_mte, draws=rate_draws, tune=rate_tune, chains=rate_chains, seed=seed)
    write_text(os.path.join(stats_dir, 'secondary_rate.txt'),
              f"SECONDARY: event rate ~ group * epoch + trial, log(exposure) offset (Bambi NB-GLMM)\n"
              f"{rate_fit['summary_text']}")

    # ---- Secondary: fraction active --------------------------------------------------------------
    frac_active = fraction_active_table(df_trace_raw)
    write_text(os.path.join(stats_dir, 'secondary_fraction_active.csv'),
              frac_active.to_csv(index=False))

    # ---- Secondary: permutation tests (mean + tail contrasts, hM3D/hM4D vs mCherry) ---------------
    print('[sp_rates_lmm] Running mouse-label permutation tests...')
    perm_results = {}
    for other_group in ('hM3D', 'hM4D'):
        for label, reduce_fn in (
            ('mean', None),
            ('p90', lambda a, b: np.percentile(a, 90) - np.percentile(b, 90)),
        ):
            stat_fn = make_amplitude_contrast_stat(df_trace_amp, other_group, 'mCherry', reduce_fn=reduce_fn)
            perm_results[f'{other_group}_vs_mCherry_{label}'] = mouse_label_permutation_test(
                stat_fn, mice_per_group, n_perm=n_perm, seed=seed)
    perm_text = '\n'.join(
        f"{k}: observed={v['observed']:.4g}, p_two_sided={v['p_two_sided']:.4g}, n_perm={v['n_perm']}"
        for k, v in perm_results.items())
    write_text(os.path.join(stats_dir, 'secondary_permutation_tests.txt'), perm_text + '\n')

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
        write_text(os.path.join(recall_stats_dir, 'post_tone_amplitude.txt'),
                  f"SECONDARY: log(mean per-event amplitude) ~ group, post-tone (20 s) window, "
                  f"pooled trials, {label}\nOmnibus: {recall_fit['omnibus']}\n\n"
                  f"{recall_fit['summary_text']}")
        plot_primary_trace_amplitude(df_recall_amp, recall_out_dir,
                                     filename_root='post_tone_amplitude')

    print('[sp_rates_lmm] Done.')
    if auto_close:
        plt.close('all')
