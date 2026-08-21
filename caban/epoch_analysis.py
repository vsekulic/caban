"""
caban.epoch_analysis.py

Population vector (PV) similarity and representational dissimilarity matrix (RDM)
analysis for TFC epoch responses.

Epochs analysed per trial
--------------------------
  pre_tone   - baseline window immediately before tone onset (duration matched to
               trace period, i.e. 35 s by default)
  tone       - tone CS period (20 s)
  trace      - trace period between tone offset and shock onset (~35 s)
  peri_shock - symmetric window centred on shock onset (default ±10 s)
  shock      - shock US period (2 s)
  post_shock - inter-trial interval from shock offset to next tone onset

Data modes
----------
  S_mean      - mean of S (deconvolved) matrix over epoch frames per neuron
  S_integral  - sum of S over epoch frames per neuron (raw accumulation, not
                divided by duration - differs from mean when epochs have unequal
                durations)
  C_mean      - mean of C (raw fluorescence) matrix over epoch frames per neuron
  C_integral  - sum of C over epoch frames per neuron
  event_rate  - count pre-extracted spikes (peaks of S) per cell in each epoch,
                divide by epoch duration in seconds.  Uses session.S_spikes which
                are already computed elsewhere via find_spikes_ca_S().

Similarity metrics
------------------
  pearson     - Pearson r between two PVs
  cosine      - cosine similarity
  mahalanobis - Mahalanobis distance (uses LedoitWolf shrinkage estimator to
                regularise the covariance when n_cells >> n_observations)
"""

import os
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
from itertools import combinations
from scipy.ndimage import gaussian_filter1d
from sklearn.covariance import LedoitWolf
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity
from scipy.spatial.distance import mahalanobis as scipy_mahalanobis
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
import scipy.stats
import statsmodels.formula.api as smf

from caban.utilities import MINISCOPE_FPS, get_spikes_in_period

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Duration of the TFC_cond trace interval, and therefore of every window that must be
# exposure-matched to it. tone_onsets_def=[185,420,660,900,1140] with tone_duration=20 and
# shock_onsets_def=[220,460,700,940,1180] (TraceFearCondSession.__init__) put the trace at
# tone offset -> shock onset = 20 s on trials 2-5 (15 s on trial 1, whose tone onset was set to
# 185 s rather than 180 s by accident and kept for consistency across mice).
#
# Matching matters for duration-SENSITIVE quantities only: event rate and the fraction of cells
# with >=1 event both depend on window length at a fixed underlying rate, so comparing them
# between a 35 s baseline and a 20 s response window would confound epoch with exposure. Mean
# per-event amplitude does not, which is why the locked confirmatory contrasts can keep using
# the 35 s 'pre_tone' reference.
#
# 20 s is independently the window Puhger et al. 2024 (iScience 27:109035) use to quantify the
# post-shock CA1 response (42-62 s after CS onset = 0-20 s after shock offset), and it already
# matches isomap.POST_SHOCK_SEC and population.PERIOD_FRAMES['post_shock'].
TRACE_MATCHED_WINDOW_S = 20.0

# Onset of the 'post_shock_late' window, measured from shock OFFSET. This is the DELAYED arm of
# the early-vs-late post-shock contrast: Puhger et al. 2024 (iScience 27:109035) find that
# silencing CA1 0-40 s after the footshock impairs memory while the same silencing delivered
# 140 s after it does not, so a late window that behaves like baseline is the internal control
# showing the post-shock response is time-limited rather than a session-wide shift.
#
# Why 90 s and not Puhger's 140 s: the choice MAXIMIZES TRIAL COVERAGE on ragged recordings. It
# does not, as an earlier version of this comment claimed, keep all five trials -- nothing does.
#
# The final trial's inter-trial interval ends with the RECORDING, not a next tone onset, and its
# real length is data-dependent and much shorter than the nominal timing suggests: observed
# sessions stop as little as ~20.5 s after the last shock offset, which is barely 'post_shock'
# (20 s) itself. On such a trial NO late window fits at ANY onset value, so the last trial of a
# truncated session is lost regardless of what this constant says.
#
# What the constant does control is the MID-session trials, where a shorter onset can only ever
# admit more trials than a longer one (the 90 s window is a subset of every ITI the 140 s window
# fits in). 90 s is therefore the permissive choice while still placing the window far outside
# the 0-20 s sustained response, and 90-110 s stays disjoint from 'pre_tone' (which begins 163 s
# before the next tone onset) on any full-length 198 s ITI. Raising this toward Puhger's 140 s
# buys fidelity to their protocol at the cost of trials; check the coverage line that
# sp_rates_lmm prints before doing so.
POST_SHOCK_LATE_ONSET_S = 90.0

# The epoch set of the population-vector / RDM epoch-analysis pipeline (extract_epoch_pvs,
# compute_rdm and the cross-session RDMs all default to it, and it fixes their axis ordering).
# This is NOT an enumeration of every window get_epoch_frames supports: 'post_shock_late' is
# deliberately absent, because adding it here would put a new row and column on every existing
# RDM figure. Consumers that want a different set pass their own (see sp_rates_lmm.TFC_EPOCHS).
EPOCH_NAMES = ['pre_tone', 'pre_tone_matched', 'tone', 'trace', 'peri_shock', 'shock',
               'post_shock', 'iti']


EPOCH_COLOURS = {
    'pre_tone':   '#888888',
    'tone':       '#1f77b4',
    'trace':      '#ff7f0e',
    'peri_shock': '#9467bd',
    'shock':      '#d62728',
    'post_shock': '#2ca02c',
    'post_shock_late': '#98df8a',   # light green, keyed to 'post_shock' as its delayed arm
    'pre_tone_matched': '#bbbbbb',
    'iti':        '#8c564b',
}

GROUP_COLOURS = {
    'hM3D':    'r',
    'hM4D':    'b',
    'mCherry': 'k',
}

TFC_CONDITIONING_PANEL_STYLE_SPACIOUS = {
    'group_order': ['hM3D', 'mCherry', 'hM4D'],
    'group_palette_box': {
        'hM3D': '#f4b8b8',
        'mCherry': '#c8c8c8',
        'hM4D': '#b8d4f0',
    },
    'group_palette_dot': {
        'hM3D': '#cc4444',
        'mCherry': '#666666',
        'hM4D': '#3a7ec0',
    },
    'combined_figsize': (14.5, 8.5),
    'single_panel_figsize': (3.5, 4.0),
    'trace_panel_figsize': (4.0, 3.5),
    'grid_wspace': 0.45,
    'grid_hspace': 0.45,
    'title_fontsize': 11,
    'axis_label_fontsize': 10,
    'tick_fontsize': 9,
    'legend_fontsize': 8,
    'box_width': 0.6,
    'box_linewidth': 0.6,
    'median_linewidth': 0.8,
    'point_size_distribution': 3,
    'point_alpha_distribution': 0.85,
    'point_linewidth': 0.3,
    'trace_linewidth': 1.35,
    'trace_line_alpha': 0.85,
    'trace_alpha': 0.18,
    'trace_smoothing_sigma_frames': 3.0,
    'scatter_size_trial': 18,
    'scatter_size_mouse': 24,
    'scatter_alpha': 0.85,
    'scatter_edgecolors': 'none',
    'fit_linewidth_trial': 1.5,
    'fit_linewidth_mouse': 1.8,
}

TFC_CONDITIONING_PANEL_STYLE_COMPACT = {
    **TFC_CONDITIONING_PANEL_STYLE_SPACIOUS,
    'combined_figsize': (11.5, 6.8),
    'single_panel_figsize': (2.9, 3.2),
    'trace_panel_figsize': (3.2, 2.9),
    'grid_wspace': 0.32,
    'grid_hspace': 0.32,
    'title_fontsize': 9,
    'axis_label_fontsize': 8,
    'tick_fontsize': 7,
    'legend_fontsize': 7,
    'point_size_distribution': 2.4,
    'trace_linewidth': 1.1,
    'trace_line_alpha': 0.8,
    'trace_smoothing_sigma_frames': 2.0,
    'scatter_size_trial': 14,
    'scatter_size_mouse': 18,
}

# Backward-compatibility alias: existing code that imports this name will keep
# using the spacious/default preset.
TFC_CONDITIONING_PANEL_STYLE = TFC_CONDITIONING_PANEL_STYLE_SPACIOUS

PVALS = [0.05, 0.01, 0.001]

# ─────────────────────────────────────────────────────────────────────────────
# Epoch frame extraction
# ─────────────────────────────────────────────────────────────────────────────

def get_epoch_frames(session, epoch_name, trial_idx,
                     peri_shock_pre_s=10.0, peri_shock_post_s=10.0,
                     pre_tone_duration_s=35.0,
                     pre_tone_matched_duration_s=TRACE_MATCHED_WINDOW_S,
                     post_shock_duration_s=TRACE_MATCHED_WINDOW_S,
                     post_shock_late_onset_s=POST_SHOCK_LATE_ONSET_S):
    """
    Return (onset_frame, offset_frame) for a given epoch and trial.

    The offset is exclusive (Python-slice style), so valid frames are
    [onset, offset).  Returns None if the epoch cannot be computed
    (e.g. onset would be before the start of the recording).

    Parameters
    ----------
    session          : TraceFearCondSession
    epoch_name       : str  - name of the window to compute. EPOCH_NAMES lists the PV/RDM
                   pipeline's set; 'post_shock_late' is also supported here but is deliberately
                   not in EPOCH_NAMES (see the comment there).
    trial_idx        : int  - 0-based trial index
    peri_shock_pre_s : float - seconds before shock onset for peri_shock epoch
    peri_shock_post_s: float - seconds after shock onset for peri_shock epoch
    pre_tone_duration_s : float - duration of the 'pre_tone' baseline window in seconds
    pre_tone_matched_duration_s : float - duration of the 'pre_tone_matched' baseline window
                   in seconds. Defaults to TRACE_MATCHED_WINDOW_S so that it is exposure-matched
                   to 'trace' and 'post_shock'; see those constants' rationale above.
    post_shock_duration_s : float - duration of the 'post_shock' AND 'post_shock_late' windows in
                   seconds. Defaults to TRACE_MATCHED_WINDOW_S, which is what makes the two
                   directly comparable to each other and to 'trace'.
    post_shock_late_onset_s : float - how long after shock OFFSET the 'post_shock_late' window
                   begins. Defaults to POST_SHOCK_LATE_ONSET_S; see its rationale above.
    """
    fps = MINISCOPE_FPS

    if epoch_name == 'tone':
        onset  = session.tone_onsets[trial_idx]
        offset = session.tone_offsets[trial_idx]

    elif epoch_name == 'trace':
        onset  = session.post_tone_onsets[trial_idx]   # = tone_offsets[i]
        offset = session.post_tone_offsets[trial_idx]  # = shock_onsets[i]

    elif epoch_name == 'shock':
        onset  = session.shock_onsets[trial_idx]
        offset = session.shock_offsets[trial_idx]

    elif epoch_name == 'peri_shock':
        centre = session.shock_onsets[trial_idx]
        onset  = centre - int(round(peri_shock_pre_s  * fps))
        offset = centre + int(round(peri_shock_post_s * fps))
        onset  = max(onset, 0)

    elif epoch_name in ('pre_tone', 'pre_tone_matched'):
        # Two baseline windows ending at the same tone onset, differing only in length.
        # 'pre_tone' is the historical 35 s window that the locked confirmatory contrasts use as
        # their reference; 'pre_tone_matched' is the exposure-matched TRACE_MATCHED_WINDOW_S one
        # needed whenever a baseline is compared against 'trace'/'post_shock' on a
        # duration-sensitive quantity (event rate, fraction of cells active). Sharing one branch
        # so the two can never drift apart in anything but their duration.
        duration_s = (pre_tone_duration_s if epoch_name == 'pre_tone'
                      else pre_tone_matched_duration_s)
        offset = session.tone_onsets[trial_idx]
        onset  = offset - int(round(duration_s * fps))
        if onset < 0:
            return None   # not enough baseline before first tone

    elif epoch_name in ('post_shock', 'post_shock_late'):
        # Two windows of the SAME length (post_shock_duration_s), differing only in how long
        # after the shock offset they begin -- the early and delayed arms of the within-trial
        # post-shock contrast. Sharing one branch so they can never drift apart in duration or
        # lose the fit guard below.
        #
        # 'post_shock' is the standardized post-shock RESPONSE window, starting at shock offset.
        # This is NOT the inter-trial interval -- see 'iti' below, which is what this epoch name
        # used to mean. The distinction is the whole point of the split: CA1 activity after an
        # aversive US is elevated for only tens of seconds (Puhger et al. 2024 iScience quantify
        # 0-20 s after shock offset, and show that silencing CA1 140 s after the shock has no
        # behavioural effect at all), so averaging that response over the full 198 s ITI dilutes
        # it by roughly an order of magnitude.
        #
        # 'post_shock_late' begins post_shock_late_onset_s after shock offset, landing in the
        # otherwise-unnamed gap between the two. It is the internal control for exactly that
        # time-limited claim: an early-vs-late difference is a within-trial contrast at aligned
        # trial indices, unlike post_shock-vs-pre_tone whose pooled reference mixes one
        # shock-naive window (trial 1's baseline) with four post-shock ones.
        #
        # THE TWO DIFFER IN WHAT A NON-FITTING WINDOW MEANS, which is why the guard below is not
        # symmetric:
        #
        #   'post_shock' RAISES. It is a locked confirmatory window that every trial must supply;
        #   a trial whose recording stops inside it is a data problem the analyst has to see.
        #
        #   'post_shock_late' returns None -- the window genuinely DOES NOT EXIST on that trial.
        #   Real recordings stop as little as ~20 s after the last shock offset, which is barely
        #   'post_shock' itself, so on a truncated final trial NO late window fits at ANY onset
        #   value. This is a definitional absence, not a swallowed error, and it is the same
        #   thing 'pre_tone' does on trial 0 (no baseline exists before the first tone).
        #   Callers must NOT paper over it: the early-vs-late contrast is only meaningful at
        #   matched trial indices, so sp_rates_lmm restricts BOTH epochs to the trials where both
        #   exist (restrict_to_shared_trials) and reports the resulting coverage.
        is_late = epoch_name == 'post_shock_late'
        start_offset_s = post_shock_late_onset_s if is_late else 0.0
        onset  = session.shock_offsets[trial_idx] + int(round(start_offset_s * fps))
        offset = onset + int(round(post_shock_duration_s * fps))
        iti_end = session.post_shock_offsets[trial_idx]
        if offset > iti_end:
            if is_late:
                return None
            raise ValueError(
                f'get_epoch_frames: a {post_shock_duration_s} s {epoch_name} window starting '
                f'{start_offset_s} s after the shock offset does not fit before the end of trial '
                f'{trial_idx} (needs {offset - session.shock_offsets[trial_idx]} frames from the '
                f'shock offset, only {iti_end - session.shock_offsets[trial_idx]} available). '
                f'The last trial ends with the recording, not a next tone onset, so it is the '
                f'usual offender. Shorten post_shock_duration_s, or check the session '
                f'boundaries.')

    elif epoch_name == 'iti':
        # The FULL inter-trial interval: shock offset to the next tone onset (or, on the last
        # trial, to the end of the recording -- so its duration is neither fixed nor equal
        # across trials, which is exactly why it must not be used as a response window).
        onset  = session.post_shock_onsets[trial_idx]
        offset = session.post_shock_offsets[trial_idx]

    else:
        raise ValueError(f'Unknown epoch_name: {epoch_name!r}')

    # Guard against zero-length or inverted windows
    if offset <= onset:
        return None

    return onset, offset


# ─────────────────────────────────────────────────────────────────────────────
# Population vector extraction
# ─────────────────────────────────────────────────────────────────────────────

def _aggregate_epoch(signal, onset, offset, data_mode):
    """
    Aggregate signal[:, onset:offset] into a 1-D population vector
    of length n_cells according to data_mode.
    """
    chunk = signal[:, onset:offset]
    if chunk.shape[1] == 0:
        return None
    if data_mode in ('S_mean', 'C_mean'):
        return np.mean(chunk, axis=1)
    else:  # integral
        return np.sum(chunk, axis=1)


def _event_rate_pv(s_spikes_ordered, onset, offset):
    """
    Build an event-rate population vector from pre-extracted spike times.

    Parameters
    ----------
    s_spikes_ordered : list of np.ndarray
        One array of spike frame-indices per cell, in the same order as the
        rows of the S matrix that was used.
    onset, offset : int
        Frame window [onset, offset).  Duration in seconds is
        (offset - onset) / MINISCOPE_FPS.

    Returns
    -------
    pv : np.ndarray of shape (n_cells,)
        Spike rate (spikes / second) for each cell in the window.
    """
    duration_s = (offset - onset) / MINISCOPE_FPS
    if duration_s <= 0:
        return None
    n_cells = len(s_spikes_ordered)
    pv = np.zeros(n_cells)
    for i, spk_frames in enumerate(s_spikes_ordered):
        if spk_frames is not None and len(spk_frames) > 0:
            count = np.sum((spk_frames >= onset) & (spk_frames < offset))
            pv[i] = count / duration_s
    return pv


def extract_epoch_pvs(session, mapping, data_mode='S_mean',
                      epochs=None,
                      peri_shock_pre_s=10.0, peri_shock_post_s=10.0,
                      pre_tone_duration_s=35.0,
                      mobility_filter=None,
                      with_crossreg=None):
    """
    Extract population vectors for each epoch across all trials.

    Parameters
    ----------
    session      : TraceFearCondSession
    mapping      : str  - crossreg mapping string (e.g. 'full', 'LT1+LT2+TFC_cond')
    data_mode    : str  - one of 'S_mean', 'S_integral', 'C_mean', 'C_integral',
                   'event_rate'
    epochs       : list of str or None - subset of EPOCH_NAMES to compute;
                   if None, all EPOCH_NAMES are computed.
    mobility_filter : None | 'mobile' | 'immobile'
        None  = use full (unfiltered) matrices,
        'mobile'    = use S_mov / C_mov / S_spikes_mov  (speed >= 2 cm/s),
        'immobile'  = use S_imm / C_imm / S_spikes_imm  (speed < 2 cm/s).
    with_crossreg : CrossRegMapping or None
        If not None, passed to session.get_S_mapping() to override the
        session's default crossreg object.  Required when the mapping string
        belongs to a different crossreg type than the session carries.

    Returns
    -------
    pvs  : dict  epoch_name -> np.ndarray of shape (n_trials, n_cells)
           Rows that could not be computed (e.g. pre-tone before start) are NaN.
    n_cells : int
    """
    if epochs is None:
        epochs = EPOCH_NAMES

    use_event_rate = (data_mode == 'event_rate')

    # --- Select the correct matrix / spike dict based on mobility_filter ---
    _suffix = {'mobile': '_mov', 'immobile': '_imm'}.get(mobility_filter, '')

    if mapping == 'full':
        S = getattr(session, f'S{_suffix}')
        C = getattr(session, f'C{_suffix}')
        if use_event_rate:
            spk_dict = getattr(session, f'S_spikes{_suffix}')
            s_spikes_ordered = [spk_dict.get(i, np.array([]))
                                for i in range(S.shape[0])]
    else:
        # get_S_mapping always returns from session.S / session.S_spikes.
        # We re-index into the mobility-filtered versions ourselves.
        _, _, _, cell_ids = session.get_S_mapping(mapping,
                                                  with_crossreg=with_crossreg)
        row_indices = session.get_S_indeces(cell_ids)
        S_full = getattr(session, f'S{_suffix}')
        C_full = getattr(session, f'C{_suffix}')
        S = S_full[row_indices]
        C = C_full[row_indices]
        if use_event_rate:
            spk_dict = getattr(session, f'S_spikes{_suffix}')
            s_spikes_ordered = [spk_dict.get(idx, np.array([]))
                                for idx in row_indices]

    if use_event_rate:
        n_cells = len(s_spikes_ordered)
    else:
        signal = S if data_mode.startswith('S') else C
        n_cells = signal.shape[0]

    n_trials = len(session.tone_onsets)

    pvs = {}
    for epoch in epochs:
        pv_trials = np.full((n_trials, n_cells), np.nan)
        for t in range(n_trials):
            result = get_epoch_frames(
                session, epoch, t,
                peri_shock_pre_s=peri_shock_pre_s,
                peri_shock_post_s=peri_shock_post_s,
                pre_tone_duration_s=pre_tone_duration_s,
            )
            if result is None:
                continue
            onset, offset = result
            if use_event_rate:
                pv = _event_rate_pv(s_spikes_ordered, onset, offset)
            else:
                # Clamp to valid frame range
                offset = min(offset, signal.shape[1])
                pv = _aggregate_epoch(signal, onset, offset, data_mode)
            if pv is not None:
                pv_trials[t] = pv
        pvs[epoch] = pv_trials

    return pvs, n_cells


# ─────────────────────────────────────────────────────────────────────────────
# Similarity between population vectors
# ─────────────────────────────────────────────────────────────────────────────

def _pv_similarity(pv_a, pv_b, metric):
    """
    Compute a scalar (dis)similarity between two 1-D population vectors.

    For 'mahalanobis' both vectors must be supplied as a 2-D array later;
    this helper is for the simple case where we have a single pair.
    """
    if np.any(np.isnan(pv_a)) or np.any(np.isnan(pv_b)):
        return np.nan

    if metric == 'pearson':
        if np.std(pv_a) == 0 or np.std(pv_b) == 0:
            return np.nan
        r, _ = stats.pearsonr(pv_a, pv_b)
        return float(r)

    elif metric == 'cosine':
        sim = sk_cosine_similarity(pv_a.reshape(1, -1), pv_b.reshape(1, -1))
        return float(sim[0, 0])

    elif metric == 'euclidean':
        return float(np.linalg.norm(pv_a - pv_b))

    else:
        raise ValueError(f'Unknown metric: {metric!r}')


def _mahalanobis_distance_matrix(pv_matrix):
    """
    Compute pairwise Mahalanobis distances for a set of population vectors.

    Parameters
    ----------
    pv_matrix : np.ndarray  shape (n_obs, n_cells)

    Returns
    -------
    dist_mat : np.ndarray  shape (n_obs, n_obs)  symmetric distance matrix
    """
    valid = ~np.any(np.isnan(pv_matrix), axis=1)
    n = pv_matrix.shape[0]
    dist_mat = np.full((n, n), np.nan)

    pv_valid = pv_matrix[valid]
    if pv_valid.shape[0] < 2:
        return dist_mat

    # LedoitWolf shrinkage for regularised covariance estimation
    try:
        lw = LedoitWolf().fit(pv_valid)
        VI = lw.precision_
    except np.linalg.LinAlgError:
        return dist_mat

    valid_idx = np.where(valid)[0]
    for i_loc, i_glob in enumerate(valid_idx):
        for j_loc, j_glob in enumerate(valid_idx):
            if i_loc == j_loc:
                dist_mat[i_glob, j_glob] = 0.0
            else:
                d = scipy_mahalanobis(pv_valid[i_loc], pv_valid[j_loc], VI)
                dist_mat[i_glob, j_glob] = d

    return dist_mat


# ─────────────────────────────────────────────────────────────────────────────
# Cross-epoch and cross-trial similarity
# ─────────────────────────────────────────────────────────────────────────────

def compute_within_epoch_stability(pvs, epoch, metric='pearson'):
    """
    Compute pairwise trial-to-trial similarity within one epoch.

    Returns
    -------
    sim_vals : 1-D np.ndarray of unique off-diagonal similarity values
               (length = n_valid_trials*(n_valid_trials-1)//2)
    trial_pairs : list of (trial_i, trial_j) tuples
    """
    pv_mat = pvs[epoch]   # (n_trials, n_cells)
    n_trials = pv_mat.shape[0]
    sim_vals, trial_pairs = [], []

    if metric == 'mahalanobis':
        dist_mat = _mahalanobis_distance_matrix(pv_mat)
        for i, j in combinations(range(n_trials), 2):
            if not np.isnan(dist_mat[i, j]):
                sim_vals.append(dist_mat[i, j])
                trial_pairs.append((i, j))
    else:
        for i, j in combinations(range(n_trials), 2):
            s = _pv_similarity(pv_mat[i], pv_mat[j], metric)
            if not np.isnan(s):
                sim_vals.append(s)
                trial_pairs.append((i, j))

    return np.array(sim_vals), trial_pairs


def compute_cross_epoch_similarity(pvs, epoch_a, epoch_b, metric='pearson',
                                   match_trials=True):
    """
    Compute similarity between epoch_a and epoch_b PVs.

    Parameters
    ----------
    match_trials : bool
        If True, compute trial-matched similarity (trial i of A vs trial i of B).
        If False, compute all pairs.

    Returns
    -------
    sim_vals   : 1-D np.ndarray
    trial_info : list of (trial_a_idx, trial_b_idx) tuples
    """
    pv_a = pvs[epoch_a]   # (n_trials, n_cells)
    pv_b = pvs[epoch_b]
    n_trials = pv_a.shape[0]

    sim_vals, trial_info = [], []

    if metric == 'mahalanobis':
        # Stack both matrices and compute full pairwise, then select cross-pairs
        stacked = np.vstack([pv_a, pv_b])
        dist_mat = _mahalanobis_distance_matrix(stacked)
        if match_trials:
            for t in range(n_trials):
                d = dist_mat[t, n_trials + t]
                if not np.isnan(d):
                    sim_vals.append(d)
                    trial_info.append((t, t))
        else:
            for i in range(n_trials):
                for j in range(n_trials):
                    d = dist_mat[i, n_trials + j]
                    if not np.isnan(d):
                        sim_vals.append(d)
                        trial_info.append((i, j))
    else:
        if match_trials:
            pairs = [(t, t) for t in range(n_trials)]
        else:
            pairs = [(i, j) for i in range(n_trials) for j in range(n_trials)]

        for i, j in pairs:
            s = _pv_similarity(pv_a[i], pv_b[j], metric)
            if not np.isnan(s):
                sim_vals.append(s)
                trial_info.append((i, j))

    return np.array(sim_vals), trial_info


# ─────────────────────────────────────────────────────────────────────────────
# Representational dissimilarity matrix (RDM)
# ─────────────────────────────────────────────────────────────────────────────

def compute_rdm(pvs, metric='pearson', epochs=None):
    """
    Build a representational similarity matrix across all epoch×trial conditions.

    Similarity = Pearson r       (for 'pearson')
               = cosine sim      (for 'cosine')
    Distance   = Mahalanobis     (for 'mahalanobis')  — kept as distance
               = Euclidean       (for 'euclidean')    — kept as distance

    Parameters
    ----------
    pvs    : dict  epoch_name -> (n_trials, n_cells)
    metric : str
    epochs : list of str or None  - epoch ordering in the RDM

    Returns
    -------
    rdm        : np.ndarray  (n_conditions, n_conditions) - similarity (or distance) matrix
    labels     : list of str  condition labels 'epoch_t{i}'
    block_info : list of (epoch_name, start_idx, end_idx) for drawing separators
    """
    if epochs is None:
        epochs = [e for e in EPOCH_NAMES if e in pvs]

    # Build flat list of (epoch, trial_idx) conditions, skipping all-NaN rows
    conditions, labels = [], []
    for epoch in epochs:
        pv_mat = pvs[epoch]
        for t in range(pv_mat.shape[0]):
            if not np.all(np.isnan(pv_mat[t])):
                conditions.append((epoch, t))
                labels.append(f'{epoch}_t{t+1}')

    n = len(conditions)
    rdm = np.full((n, n), np.nan)

    if metric == 'mahalanobis':
        # Build stacked matrix for all conditions
        n_cells = next(iter(pvs.values())).shape[1]
        all_pvs = np.array([pvs[e][t] for e, t in conditions])
        dist_mat = _mahalanobis_distance_matrix(all_pvs)
        rdm = dist_mat
    else:
        for i, (ea, ta) in enumerate(conditions):
            for j, (eb, tb) in enumerate(conditions):
                pv_a = pvs[ea][ta]
                pv_b = pvs[eb][tb]
                if metric in ('pearson', 'cosine', 'euclidean'):
                    raw = _pv_similarity(pv_a, pv_b, metric)
                    if np.isnan(raw):
                        continue
                    rdm[i, j] = raw  # similarity for pearson/cosine, distance for euclidean

    # Build block_info
    block_info = []
    cur_epoch, start = conditions[0][0], 0
    for k, (epoch, _) in enumerate(conditions):
        if epoch != cur_epoch:
            block_info.append((cur_epoch, start, k))
            cur_epoch, start = epoch, k
    block_info.append((cur_epoch, start, n))

    return rdm, labels, block_info


# ─────────────────────────────────────────────────────────────────────────────
# Per-mouse analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch_analysis_mouse(mouse, session, mapping,
                             data_mode='S_mean',
                             metrics=('pearson', 'cosine'),
                             epochs=None,
                             peri_shock_pre_s=10.0, peri_shock_post_s=10.0,
                             pre_tone_duration_s=35.0,
                             mobility_filter=None):
    """
    Full epoch analysis for a single mouse.

    Parameters
    ----------
    mobility_filter : None | 'mobile' | 'immobile'

    Returns
    -------
    result : dict with keys:
        'pvs'             : epoch -> (n_trials, n_cells)
        'within_stability': metric -> epoch -> {'sim': array, 'pairs': list}
        'cross_similarity': metric -> (epoch_a, epoch_b) -> {'sim': array, 'pairs': list}
        'rdm'             : metric -> {'rdm': array, 'labels': list, 'block_info': list}
        'n_cells'         : int
        'n_trials'        : int
    """
    if epochs is None:
        epochs = EPOCH_NAMES

    pvs, n_cells = extract_epoch_pvs(
        session, mapping, data_mode=data_mode, epochs=epochs,
        peri_shock_pre_s=peri_shock_pre_s,
        peri_shock_post_s=peri_shock_post_s,
        pre_tone_duration_s=pre_tone_duration_s,
        mobility_filter=mobility_filter,
    )
    n_trials = len(session.tone_onsets)

    result = {
        'pvs': pvs,
        'n_cells': n_cells,
        'n_trials': n_trials,
        'within_stability': {},
        'cross_similarity': {},
        'rdm': {},
    }

    for metric in metrics:
        result['within_stability'][metric] = {}
        result['cross_similarity'][metric] = {}

        # Within-epoch stability (trial-to-trial)
        for epoch in epochs:
            sim_vals, pairs = compute_within_epoch_stability(pvs, epoch, metric)
            result['within_stability'][metric][epoch] = {
                'sim': sim_vals, 'pairs': pairs
            }

        # Cross-epoch similarity (matched trials)
        for ea, eb in combinations(epochs, 2):
            sim_vals, pairs = compute_cross_epoch_similarity(
                pvs, ea, eb, metric=metric, match_trials=True
            )
            result['cross_similarity'][metric][(ea, eb)] = {
                'sim': sim_vals, 'pairs': pairs
            }

        # RDM
        rdm, labels, block_info = compute_rdm(pvs, metric=metric, epochs=epochs)
        result['rdm'][metric] = {
            'rdm': rdm, 'labels': labels, 'block_info': block_info
        }

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Group-level aggregation
# ─────────────────────────────────────────────────────────────────────────────

def _mean_sim_per_mouse(mouse_result, metric, analysis_type, key):
    """
    Collapse per-trial similarity values to a single scalar per mouse.

    analysis_type : 'within_stability' or 'cross_similarity'
    key           : epoch name (str) or (epoch_a, epoch_b) tuple
    """
    vals = mouse_result[analysis_type][metric][key]['sim']
    if len(vals) == 0:
        return np.nan
    return float(np.nanmean(vals))


def aggregate_group_results(all_mouse_results, mice_per_group, metric,
                             analysis_type, key):
    """
    Collect mean-per-mouse similarity values by group.

    Returns
    -------
    group_data : dict  group -> list of per-mouse mean similarities
    """
    group_data = {g: [] for g in mice_per_group}
    for group, mice in mice_per_group.items():
        for mouse in mice:
            if mouse not in all_mouse_results:
                continue
            val = _mean_sim_per_mouse(
                all_mouse_results[mouse], metric, analysis_type, key
            )
            group_data[group].append(val)
    # Remove NaN entries
    group_data = {
        g: [v for v in vals if not np.isnan(v)]
        for g, vals in group_data.items()
    }
    return group_data


def aggregate_group_rdm(all_mouse_results, mice_per_group, metric):
    """
    Average RDM across mice within each group.

    Returns
    -------
    group_mean_rdm : dict  group -> mean_rdm np.ndarray
    group_labels   : dict  group -> labels list  (from the first valid mouse)
    group_block_info : dict  group -> block_info
    """
    group_mean_rdm, group_labels, group_block_info = {}, {}, {}
    for group, mice in mice_per_group.items():
        rdms = []
        labels_ref = block_ref = None
        for mouse in mice:
            if mouse not in all_mouse_results:
                continue
            rdm_data = all_mouse_results[mouse]['rdm'].get(metric)
            if rdm_data is None:
                continue
            rdms.append(rdm_data['rdm'])
            if labels_ref is None:
                labels_ref = rdm_data['labels']
                block_ref  = rdm_data['block_info']
        if rdms:
            # Mice may have different RDM sizes (e.g. one mouse missing a trial).
            # Pad smaller RDMs to the largest size with NaN before averaging.
            max_n = max(r.shape[0] for r in rdms)
            def _pad_rdm(r, n):
                if r.shape[0] == n:
                    return r
                out = np.full((n, n), np.nan)
                s = r.shape[0]
                out[:s, :s] = r
                return out
            padded  = [_pad_rdm(r, max_n) for r in rdms]
            stacked = np.stack(padded, axis=0)
            group_mean_rdm[group]   = np.nanmean(stacked, axis=0)
            group_labels[group]     = labels_ref
            group_block_info[group] = block_ref
    return group_mean_rdm, group_labels, group_block_info


# ─────────────────────────────────────────────────────────────────────────────
# Statistics helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_pval_str(pval):
    if pval < 0.001:
        return '***'
    elif pval < 0.01:
        return '**'
    elif pval < 0.05:
        return '*'
    return 'ns'


def _do_group_stats(group_data):
    """
    One-way ANOVA + Tukey HSD on group_data dict.

    Returns
    -------
    anova_p : float
    tukey   : statsmodels TukeyHSD result (or None if ANOVA not significant)
    """
    groups = [g for g, vals in group_data.items() if len(vals) >= 2]
    if len(groups) < 2:
        return np.nan, None

    arrays = [np.array(group_data[g]) for g in groups]
    if any(len(a) < 2 for a in arrays):
        return np.nan, None

    f, p = scipy.stats.f_oneway(*arrays)
    if p < 0.05:
        all_vals   = np.concatenate(arrays)
        all_labels = np.concatenate([[g]*len(group_data[g]) for g in groups])
        tukey = pairwise_tukeyhsd(all_vals, all_labels)
        return p, tukey
    return p, None


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _barplot_with_scatter(ax, group_data, title='', ylabel='', ylim=None,
                          annotate_stats=True):
    """
    Bar plot with individual mouse scatter overlay for three groups.
    """
    groups = list(group_data.keys())
    x = np.arange(len(groups))
    means = [np.mean(group_data[g]) if group_data[g] else np.nan for g in groups]
    sems  = [
        np.std(group_data[g]) / np.sqrt(len(group_data[g]))
        if len(group_data[g]) > 1 else 0.0
        for g in groups
    ]

    colours = [GROUP_COLOURS.get(g, 'grey') for g in groups]
    ax.bar(x, means, yerr=sems, color=colours, alpha=0.7, capsize=4, width=0.5,
           error_kw={'linewidth': 1.5})

    for i, g in enumerate(groups):
        y_vals = group_data[g]
        jitter = np.random.uniform(-0.12, 0.12, size=len(y_vals))
        ax.scatter(x[i] + jitter, y_vals, color='k', s=20, zorder=5, linewidths=0)

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=9)
    if ylim is not None:
        ax.set_ylim(ylim)

    if annotate_stats and all(len(group_data[g]) >= 2 for g in groups):
        anova_p, tukey = _do_group_stats(group_data)
        if not np.isnan(anova_p):
            ax.set_xlabel(f'F-test p={anova_p:.3f}', fontsize=7, color='grey')
        if tukey is not None:
            # Mark individual pairs that are significant
            max_y = max(means[i] + sems[i] for i in range(len(groups))
                        if not np.isnan(means[i])) if means else 0
            dh = (ylim[1] - ylim[0]) * 0.05 if ylim else max_y * 0.1
            y_top = max_y + dh
            pair_map = {'hM3D': 0, 'hM4D': 1, 'mCherry': 2}
            for row in tukey.summary().data[1:]:
                g1, g2, *_, reject = row
                if reject:
                    i1 = groups.index(g1) if g1 in groups else None
                    i2 = groups.index(g2) if g2 in groups else None
                    if i1 is not None and i2 is not None:
                        ax.annotate(
                            '', xy=(x[i2], y_top + dh*0.5),
                            xytext=(x[i1], y_top + dh*0.5),
                            arrowprops=dict(arrowstyle='-', color='k', lw=1)
                        )
                        ax.text(
                            (x[i1]+x[i2])/2, y_top + dh*0.6, '*',
                            ha='center', va='bottom', fontsize=10
                        )
                        y_top += dh


def _data_mode_tag(data_mode, mobility_filter):
    """Build a directory / label tag that includes the mobility filter."""
    if mobility_filter is None:
        return data_mode
    return f'{data_mode}_{mobility_filter}'


def plot_epoch_similarity_group(PLOTS_DIR, all_mouse_results, mice_per_group,
                                 metric='pearson', analysis_type='within_stability',
                                 keys=None, mapping='full',
                                 data_mode='S_mean', mobility_filter=None,
                                 auto_close=True):
    """
    Bar + scatter plots of per-group similarity for a set of analysis keys.

    Parameters
    ----------
    keys : list of str (epoch names, for within_stability) or list of
           (epoch_a, epoch_b) tuples (for cross_similarity), or None to use
           all available keys from the first valid mouse.
    """
    # Collect available keys from first valid mouse
    first_mouse_result = next(
        (v for v in all_mouse_results.values()
         if analysis_type in v and metric in v[analysis_type]),
        None
    )
    if first_mouse_result is None:
        raise ValueError(f'No valid results for analysis_type={analysis_type}, metric={metric}')

    if keys is None:
        keys = sorted(first_mouse_result[analysis_type][metric].keys(), key=str)
    n_keys = len(keys)
    if n_keys == 0:
        raise ValueError('No keys found to plot.')
    ncols = min(n_keys, 4)
    nrows = (n_keys + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.5 * nrows),
                             squeeze=False)

    metric_label = {
        'pearson':     'Pearson r',
        'cosine':      'Cosine similarity',
        'mahalanobis': 'Mahalanobis distance',
        'euclidean':   'Euclidean distance',
    }.get(metric, metric)

    for idx, key in enumerate(keys):
        row_i, col_i = divmod(idx, ncols)
        ax = axs[row_i][col_i]

        group_data = aggregate_group_results(
            all_mouse_results, mice_per_group, metric, analysis_type, key
        )
        title = (
            f'{key} stability'
            if isinstance(key, str)
            else f'{key[0]} vs {key[1]}'
        )
        plot_ylabel = metric_label
        _barplot_with_scatter(ax, group_data, title=title, ylabel=plot_ylabel)

    # Hide unused axes
    for idx in range(n_keys, nrows * ncols):
        row_i, col_i = divmod(idx, ncols)
        axs[row_i][col_i].axis('off')

    dm_tag = _data_mode_tag(data_mode, mobility_filter)
    analysis_label = 'within_stability' if analysis_type == 'within_stability' \
        else 'cross_epoch'
    fig.suptitle(
        f'Epoch {analysis_label} — {metric_label}\n'
        f'mapping={mapping}, data={dm_tag}',
        fontsize=10, y=1.01
    )
    plt.tight_layout()

    save_dir = os.path.join(PLOTS_DIR, 'epoch_analysis', mapping, dm_tag)
    os.makedirs(save_dir, exist_ok=True)
    fname = f'{analysis_type}_{metric}.png'
    plt.savefig(os.path.join(save_dir, fname), dpi=200, bbox_inches='tight')
    if auto_close:
        plt.close(fig)
    return fig


def plot_epoch_rdm_group(PLOTS_DIR, all_mouse_results, mice_per_group,
                          metric='pearson', mapping='full',
                          data_mode='S_mean', mobility_filter=None,
                          auto_close=True):
    """
    Heatmap of the mean RDM per group.
    """
    group_mean_rdm, group_labels, group_block_info = aggregate_group_rdm(
        all_mouse_results, mice_per_group, metric
    )
    if not group_mean_rdm:
        print('[epoch_analysis] No RDM data found.')
        return

    groups = list(group_mean_rdm.keys())
    n_groups = len(groups)
    fig, axs = plt.subplots(1, n_groups, figsize=(5 * n_groups, 4.5),
                             squeeze=False)

    is_distance = metric in ('mahalanobis', 'euclidean')
    cbar_label = {
        'pearson':     'Pearson r',
        'cosine':      'cosine',
        'mahalanobis': 'Mahalanobis distance',
        'euclidean':   'Euclidean distance',
    }.get(metric, 'similarity')

    for gi, group in enumerate(groups):
        ax = axs[0][gi]
        rdm = group_mean_rdm[group]
        labels = group_labels.get(group, [])
        block_info = group_block_info.get(group, [])

        if is_distance:
            vmax = np.nanpercentile(rdm, 95)
            im = ax.imshow(rdm, aspect='auto', cmap='viridis',
                           vmin=0, vmax=vmax, interpolation='none')
        elif metric == 'cosine':
            vmin = np.nanpercentile(rdm, 5)
            im = ax.imshow(rdm, aspect='auto', cmap='viridis',
                           vmin=vmin, vmax=1.0, interpolation='none')
        else:
            vmin = np.nanpercentile(rdm, 5)
            im = ax.imshow(rdm, aspect='auto', cmap='RdYlBu_r',
                           vmin=vmin, vmax=1.0, interpolation='none')
        plt.colorbar(im, ax=ax, label=cbar_label, fraction=0.046, pad=0.04)

        # Epoch block separators
        for (_, start, end) in block_info:
            ax.axhline(end - 0.5, color='k', linewidth=0.8, linestyle='--')
            ax.axvline(end - 0.5, color='k', linewidth=0.8, linestyle='--')

        # Epoch block labels on axes
        epoch_ticks, epoch_tick_labels = [], []
        for (epoch_name, start, end) in block_info:
            mid = (start + end - 1) / 2
            epoch_ticks.append(mid)
            epoch_tick_labels.append(epoch_name.replace('_', '\n'))

        ax.set_xticks(epoch_ticks)
        ax.set_xticklabels(epoch_tick_labels, fontsize=7, rotation=0)
        ax.set_yticks(epoch_ticks)
        ax.set_yticklabels(epoch_tick_labels, fontsize=7)
        ax.set_title(group, fontsize=11)

    metric_label = {
        'pearson': 'Pearson r', 'cosine': 'cosine',
        'mahalanobis': 'Mahalanobis distance', 'euclidean': 'Euclidean distance',
    }.get(metric, metric)

    dm_tag = _data_mode_tag(data_mode, mobility_filter)
    title_kind = 'RDM (distance)' if is_distance else 'RSM (similarity)'
    fig.suptitle(
        f'Mean {title_kind} per group — {metric_label}\n'
        f'mapping={mapping}, data={dm_tag}',
        fontsize=10
    )
    plt.tight_layout()

    save_dir = os.path.join(PLOTS_DIR, 'epoch_analysis', mapping, dm_tag)
    os.makedirs(save_dir, exist_ok=True)
    fname = f'rdm_{metric}.png'
    plt.savefig(os.path.join(save_dir, fname), dpi=200, bbox_inches='tight')
    if auto_close:
        plt.close(fig)
    return fig


def plot_trial_trajectory(PLOTS_DIR, all_mouse_results, mice_per_group,
                           metric='pearson', epoch='tone',
                           mapping='full', data_mode='S_mean',
                           mobility_filter=None, auto_close=True):
    """
    Line plot showing how within-epoch stability changes across trial pairs
    (i.e. as a function of learning across the 5 tone-shock trials).

    For each pair of successive trials (t, t+1), the similarity is plotted.
    """
    groups = list(mice_per_group.keys())
    fig, ax = plt.subplots(figsize=(5, 3.5))

    for group in groups:
        colour = GROUP_COLOURS.get(group, 'grey')
        mice   = mice_per_group[group]

        # Collect per-mouse similarity for each consecutive trial pair
        n_pairs = None
        mouse_traces = []
        for mouse in mice:
            if mouse not in all_mouse_results:
                continue
            within = all_mouse_results[mouse]['within_stability'].get(metric, {})
            epoch_data = within.get(epoch, {})
            sim_vals = epoch_data.get('sim', np.array([]))
            pairs    = epoch_data.get('pairs', [])
            if len(sim_vals) == 0:
                continue
            # Only consecutive pairs (t, t+1)
            vals = [s for (t0, t1), s in zip(pairs, sim_vals) if t1 == t0 + 1]
            if not vals:
                continue
            mouse_traces.append(np.array(vals))

        if not mouse_traces:
            continue

        max_len = max(len(t) for t in mouse_traces)
        padded_traces = [
            np.pad(t.astype(float), (0, max_len - len(t)), constant_values=np.nan)
            for t in mouse_traces
        ]
        mat   = np.array(padded_traces)      # (n_mice, n_consec_pairs)
        mean  = np.nanmean(mat, axis=0)
        sem   = np.nanstd(mat, axis=0) / np.sqrt(np.sum(~np.isnan(mat), axis=0))

        x = np.arange(1, len(mean) + 1)
        ax.plot(x, mean, '-o', color=colour, label=group, linewidth=1.8,
                markersize=5)
        ax.fill_between(x, mean - sem, mean + sem, color=colour, alpha=0.2)

    metric_label = {
        'pearson': 'Pearson r', 'cosine': 'Cosine similarity',
        'mahalanobis': 'Mahalanobis distance', 'euclidean': 'Euclidean distance',
    }.get(metric, metric)

    dm_tag = _data_mode_tag(data_mode, mobility_filter)
    ax.set_xlabel('Consecutive trial pair', fontsize=10)
    ax.set_ylabel(metric_label, fontsize=10)
    ax.set_title(
        f'Trial-to-trial {epoch} stability ({metric_label})\n'
        f'mapping={mapping}, data={dm_tag}',
        fontsize=9
    )
    ax.legend(fontsize=9)
    plt.tight_layout()

    save_dir = os.path.join(PLOTS_DIR, 'epoch_analysis', mapping, dm_tag)
    os.makedirs(save_dir, exist_ok=True)
    fname = f'trial_trajectory_{epoch}_{metric}.png'
    plt.savefig(os.path.join(save_dir, fname), dpi=200, bbox_inches='tight')
    if auto_close:
        plt.close(fig)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch_analysis_all_mice(PLOTS_DIR, mice_per_group, TFC_cond,
                                 TFC_cond_crossreg,
                                 mapping='full',
                                 data_mode='S_mean',
                                 metrics=('pearson', 'cosine'),
                                 epochs=None,
                                 peri_shock_pre_s=10.0,
                                 peri_shock_post_s=10.0,
                                 pre_tone_duration_s=35.0,
                                 mobility_filter=None,
                                 auto_close=True):
    """
    Run the complete epoch PV analysis across all mice and produce all plots.

    Parameters
    ----------
    PLOTS_DIR        : str  - root output directory
    mice_per_group   : dict  group -> list of mouse IDs
    TFC_cond         : dict  mouse -> TraceFearCondSession
    TFC_cond_crossreg: dict  mouse -> CrossRegMapping
    mapping          : str  - crossreg mapping (default 'full')
    data_mode        : str  - 'S_mean', 'S_integral', 'C_mean', 'C_integral',
                       or 'event_rate'
    metrics          : tuple of str  - similarity metrics to compute
    epochs           : list of str or None  - epochs to analyse; None = all
    peri_shock_pre_s : float  - seconds before shock for peri_shock epoch
    peri_shock_post_s: float  - seconds after shock for peri_shock epoch
    pre_tone_duration_s : float  - seconds for pre-tone baseline window
    mobility_filter  : None | 'mobile' | 'immobile'
    auto_close       : bool  - close figures after saving

    Returns
    -------
    all_mouse_results : dict  mouse -> run_epoch_analysis_mouse result dict
    """
    if epochs is None:
        epochs = EPOCH_NAMES

    all_mouse_results = {}
    mouse_list = [m for mice in mice_per_group.values() for m in mice]

    for mouse in mouse_list:
        if mouse not in TFC_cond:
            print(f'[epoch_analysis] Mouse {mouse} not in TFC_cond — skipping.')
            continue
        session = TFC_cond[mouse]

        # If mapping is not 'full', ensure the cross-registration is loaded.
        if mapping != 'full' and mouse in TFC_cond_crossreg:
            session.crossreg = TFC_cond_crossreg[mouse]

        try:
            result = run_epoch_analysis_mouse(
                mouse=mouse,
                session=session,
                mapping=mapping,
                data_mode=data_mode,
                metrics=list(metrics),
                epochs=epochs,
                peri_shock_pre_s=peri_shock_pre_s,
                peri_shock_post_s=peri_shock_post_s,
                pre_tone_duration_s=pre_tone_duration_s,
                mobility_filter=mobility_filter,
            )
            all_mouse_results[mouse] = result
            print(f'[epoch_analysis] {mouse} done — '
                  f'{result["n_cells"]} cells, {result["n_trials"]} trials')
        except Exception as exc:
            print(f'[epoch_analysis] {mouse} FAILED: {exc}')

    if not all_mouse_results:
        print('[epoch_analysis] No results computed.')
        return all_mouse_results

    # ── Plots ──────────────────────────────────────────────────────────────

    for metric in metrics:
        # Within-epoch stability (one panel per epoch)
        plot_epoch_similarity_group(
            PLOTS_DIR, all_mouse_results, mice_per_group,
            metric=metric, analysis_type='within_stability',
            mapping=mapping, data_mode=data_mode,
            mobility_filter=mobility_filter, auto_close=auto_close,
        )

        # Cross-epoch similarity (one panel per epoch pair)
        plot_epoch_similarity_group(
            PLOTS_DIR, all_mouse_results, mice_per_group,
            metric=metric, analysis_type='cross_similarity',
            mapping=mapping, data_mode=data_mode,
            mobility_filter=mobility_filter, auto_close=auto_close,
        )

        # RDM heatmaps per group
        plot_epoch_rdm_group(
            PLOTS_DIR, all_mouse_results, mice_per_group,
            metric=metric, mapping=mapping, data_mode=data_mode,
            mobility_filter=mobility_filter,
            auto_close=auto_close,
        )

        # Trial-to-trial trajectory for key epochs
        for epoch in ['tone', 'trace', 'shock']:
            if epoch in epochs:
                plot_trial_trajectory(
                    PLOTS_DIR, all_mouse_results, mice_per_group,
                    metric=metric, epoch=epoch,
                    mapping=mapping, data_mode=data_mode,
                    mobility_filter=mobility_filter,
                    auto_close=auto_close,
                )

    return all_mouse_results


# ═══════════════════════════════════════════════════════════════════════════════
# Cross-session (recall) epoch analysis
# ═══════════════════════════════════════════════════════════════════════════════
#
# Compares epoch-level population vectors extracted from TFC_cond against those
# from recall sessions (Test_B / Test_B_1wk / Test_A / Test_A_1wk).
#
# Test_B has 3 tones (no shocks) → epochs: pre_tone, tone, post_tone.
# Test_A has no tones → pseudo-epochs at matching timepoints (180/420/660 s).
# ═══════════════════════════════════════════════════════════════════════════════

TESTB_EPOCH_NAMES = ['pre_tone', 'tone', 'post_tone']
TESTA_EPOCH_NAMES = ['pre_pseudo_tone', 'pseudo_tone', 'post_pseudo_tone']
TESTA_PSEUDO_TONE_ONSETS_S = [180.0, 420.0, 660.0]

# Matched TFC ↔ recall epochs for bar-plot comparisons
MATCHED_EPOCHS_TESTB = [
    ('pre_tone', 'pre_tone'),
    ('tone',     'tone'),
    ('trace',    'post_tone'),
]
MATCHED_EPOCHS_TESTA = [
    ('pre_tone', 'pre_pseudo_tone'),
    ('tone',     'pseudo_tone'),
    ('trace',    'post_pseudo_tone'),
]


# ─────────────────────────────────────────────────────────────────────────────
# Recall epoch frame helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_testb_epoch_frames(session, epoch_name, trial_idx,
                           pre_tone_duration_s=35.0,
                           post_tone_duration_s=35.0):
    """Return (onset_frame, offset_frame) for a Test_B epoch and trial."""
    fps = MINISCOPE_FPS

    if epoch_name == 'pre_tone':
        offset = session.tone_onsets[trial_idx]
        onset  = offset - int(round(pre_tone_duration_s * fps))
        if onset < 0:
            return None

    elif epoch_name == 'tone':
        onset  = session.tone_onsets[trial_idx]
        offset = session.tone_offsets[trial_idx]

    elif epoch_name == 'post_tone':
        onset  = session.tone_offsets[trial_idx]
        offset = onset + int(round(post_tone_duration_s * fps))

    else:
        raise ValueError(f'Unknown Test_B epoch: {epoch_name!r}')

    if offset <= onset:
        return None
    return onset, offset


def get_testa_epoch_frames(session, epoch_name, trial_idx,
                           pseudo_tone_onsets_s=None,
                           tone_duration_s=20.0,
                           pre_tone_duration_s=35.0,
                           post_tone_duration_s=35.0):
    """Return (onset_frame, offset_frame) for a Test_A pseudo-epoch."""
    fps = MINISCOPE_FPS
    if pseudo_tone_onsets_s is None:
        pseudo_tone_onsets_s = TESTA_PSEUDO_TONE_ONSETS_S

    pt_onset  = int(round(pseudo_tone_onsets_s[trial_idx] * fps))
    pt_offset = pt_onset + int(round(tone_duration_s * fps))

    if epoch_name == 'pre_pseudo_tone':
        offset = pt_onset
        onset  = offset - int(round(pre_tone_duration_s * fps))
        if onset < 0:
            return None

    elif epoch_name == 'pseudo_tone':
        onset  = pt_onset
        offset = pt_offset

    elif epoch_name == 'post_pseudo_tone':
        onset  = pt_offset
        offset = onset + int(round(post_tone_duration_s * fps))

    else:
        raise ValueError(f'Unknown Test_A epoch: {epoch_name!r}')

    if offset <= onset:
        return None
    return onset, offset


# ─────────────────────────────────────────────────────────────────────────────
# Recall PV extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_recall_epoch_pvs(session, mapping, data_mode='S_mean',
                              recall_type='testb',
                              epochs=None,
                              pre_tone_duration_s=35.0,
                              post_tone_duration_s=35.0,
                              mobility_filter=None,
                              with_crossreg=None):
    """
    Extract population vectors for recall session epochs.

    Parameters
    ----------
    session       : TestBSession or TestASession
    mapping       : str
    data_mode     : str
    recall_type   : 'testb' or 'testa'
    epochs        : list of str or None
    pre_tone_duration_s  : float
    post_tone_duration_s : float  – post-tone (Test_B) or post-pseudo-tone (Test_A)
    mobility_filter      : None | 'mobile' | 'immobile'
    with_crossreg        : CrossRegMapping or None

    Returns
    -------
    pvs     : dict  epoch -> (n_trials, n_cells)
    n_cells : int
    """
    if epochs is None:
        epochs = TESTB_EPOCH_NAMES if recall_type == 'testb' else TESTA_EPOCH_NAMES

    use_event_rate = (data_mode == 'event_rate')
    _suffix = {'mobile': '_mov', 'immobile': '_imm'}.get(mobility_filter, '')

    if mapping == 'full':
        S = getattr(session, f'S{_suffix}')
        C = getattr(session, f'C{_suffix}')
        if use_event_rate:
            spk_dict = getattr(session, f'S_spikes{_suffix}')
            s_spikes_ordered = [spk_dict.get(i, np.array([]))
                                for i in range(S.shape[0])]
    else:
        _, _, _, cell_ids = session.get_S_mapping(mapping,
                                                  with_crossreg=with_crossreg)
        row_indices = session.get_S_indeces(cell_ids)
        S_full = getattr(session, f'S{_suffix}')
        C_full = getattr(session, f'C{_suffix}')
        S = S_full[row_indices]
        C = C_full[row_indices]
        if use_event_rate:
            spk_dict = getattr(session, f'S_spikes{_suffix}')
            s_spikes_ordered = [spk_dict.get(idx, np.array([]))
                                for idx in row_indices]

    if use_event_rate:
        n_cells = len(s_spikes_ordered)
    else:
        signal = S if data_mode.startswith('S') else C
        n_cells = signal.shape[0]

    if recall_type == 'testb':
        n_trials = len(session.tone_onsets)
    else:
        n_trials = len(TESTA_PSEUDO_TONE_ONSETS_S)

    pvs = {}
    for epoch in epochs:
        pv_trials = np.full((n_trials, n_cells), np.nan)
        for t in range(n_trials):
            if recall_type == 'testb':
                result = get_testb_epoch_frames(
                    session, epoch, t,
                    pre_tone_duration_s=pre_tone_duration_s,
                    post_tone_duration_s=post_tone_duration_s,
                )
            else:
                result = get_testa_epoch_frames(
                    session, epoch, t,
                    pre_tone_duration_s=pre_tone_duration_s,
                    post_tone_duration_s=post_tone_duration_s,
                )
            if result is None:
                continue
            onset, offset = result
            if use_event_rate:
                pv = _event_rate_pv(s_spikes_ordered, onset, offset)
            else:
                offset = min(offset, signal.shape[1])
                pv = _aggregate_epoch(signal, onset, offset, data_mode)
            if pv is not None:
                pv_trials[t] = pv
        pvs[epoch] = pv_trials

    return pvs, n_cells


# ─────────────────────────────────────────────────────────────────────────────
# Cross-session RDM / RSM
# ─────────────────────────────────────────────────────────────────────────────

def compute_cross_session_rdm(pvs_tfc, pvs_recall, metric='pearson',
                               tfc_epochs=None, recall_epochs=None,
                               tfc_label='TFC', recall_label='Recall'):
    """
    Build a *rectangular* cross-session similarity matrix.

    Rows  = TFC epoch×trial conditions.
    Columns = Recall epoch×trial conditions.

    Returns
    -------
    rdm            : np.ndarray  (n_tfc, n_recall)
    row_labels     : list of str  — TFC condition labels
    col_labels     : list of str  — Recall condition labels
    row_block_info : list of (epoch_label, start, end)
    col_block_info : list of (epoch_label, start, end)
    """
    if tfc_epochs is None:
        tfc_epochs = [e for e in EPOCH_NAMES if e in pvs_tfc]
    if recall_epochs is None:
        recall_epochs = list(pvs_recall.keys())

    # Build row PVs (TFC)
    row_pvs, row_labels, row_block_info = [], [], []
    for epoch in tfc_epochs:
        pv_mat = pvs_tfc[epoch]
        start = len(row_pvs)
        for t in range(pv_mat.shape[0]):
            if not np.all(np.isnan(pv_mat[t])):
                row_pvs.append(pv_mat[t])
                row_labels.append(f'{tfc_label}_{epoch}_t{t+1}')
        if len(row_pvs) > start:
            row_block_info.append((epoch, start, len(row_pvs)))

    # Build col PVs (Recall)
    col_pvs, col_labels, col_block_info = [], [], []
    for epoch in recall_epochs:
        pv_mat = pvs_recall[epoch]
        start = len(col_pvs)
        for t in range(pv_mat.shape[0]):
            if not np.all(np.isnan(pv_mat[t])):
                col_pvs.append(pv_mat[t])
                col_labels.append(f'{recall_label}_{epoch}_t{t+1}')
        if len(col_pvs) > start:
            col_block_info.append((epoch, start, len(col_pvs)))

    nr, nc = len(row_pvs), len(col_pvs)
    rdm = np.full((nr, nc), np.nan)
    if nr == 0 or nc == 0:
        return rdm, row_labels, col_labels, row_block_info, col_block_info

    if metric == 'mahalanobis':
        # Stack and compute full pairwise, then extract rectangular block
        stacked = np.vstack([np.array(row_pvs), np.array(col_pvs)])
        full_dist = _mahalanobis_distance_matrix(stacked)
        rdm = full_dist[:nr, nr:]
    else:
        for i in range(nr):
            for j in range(nc):
                rdm[i, j] = _pv_similarity(row_pvs[i], col_pvs[j], metric)

    return rdm, row_labels, col_labels, row_block_info, col_block_info


def compute_cross_session_matched_similarity(pvs_tfc, pvs_recall, metric,
                                              matched_epochs):
    """
    Mean similarity between matched TFC ↔ recall epoch pairs (all trials
    of one epoch vs all trials of the other).

    Returns
    -------
    result : dict  (tfc_epoch, recall_epoch) -> float
    """
    result = {}
    for tfc_ep, recall_ep in matched_epochs:
        if tfc_ep not in pvs_tfc or recall_ep not in pvs_recall:
            result[(tfc_ep, recall_ep)] = np.nan
            continue

        pv_tfc    = pvs_tfc[tfc_ep]      # (n_tfc_trials, n_cells)
        pv_recall = pvs_recall[recall_ep] # (n_recall_trials, n_cells)

        if metric == 'mahalanobis':
            stacked = np.vstack([pv_tfc, pv_recall])
            dist_mat = _mahalanobis_distance_matrix(stacked)
            nt = pv_tfc.shape[0]
            vals = [dist_mat[i, nt + j]
                    for i in range(nt)
                    for j in range(pv_recall.shape[0])
                    if not np.isnan(dist_mat[i, nt + j])]
        else:
            vals = [_pv_similarity(pv_tfc[i], pv_recall[j], metric)
                    for i in range(pv_tfc.shape[0])
                    for j in range(pv_recall.shape[0])]
            vals = [v for v in vals if not np.isnan(v)]

        result[(tfc_ep, recall_ep)] = float(np.mean(vals)) if vals else np.nan
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Per-mouse cross-session analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_cross_session_analysis_mouse(mouse, tfc_session, recall_session,
                                     mapping, data_mode,
                                     recall_type='testb',
                                     recall_label='Test_B',
                                     metrics=('pearson', 'cosine'),
                                     tfc_epochs=None, recall_epochs=None,
                                     peri_shock_pre_s=10.0,
                                     peri_shock_post_s=10.0,
                                     pre_tone_duration_s=35.0,
                                     post_tone_duration_s=35.0,
                                     mobility_filter=None,
                                     crossreg_override=None):
    """Cross-session epoch PV analysis for a single mouse."""
    if tfc_epochs is None:
        tfc_epochs = EPOCH_NAMES
    if recall_epochs is None:
        recall_epochs = (TESTB_EPOCH_NAMES if recall_type == 'testb'
                         else TESTA_EPOCH_NAMES)

    matched = (MATCHED_EPOCHS_TESTB if recall_type == 'testb'
               else MATCHED_EPOCHS_TESTA)

    pvs_tfc, n_cells_tfc = extract_epoch_pvs(
        tfc_session, mapping, data_mode=data_mode,
        epochs=tfc_epochs,
        peri_shock_pre_s=peri_shock_pre_s,
        peri_shock_post_s=peri_shock_post_s,
        pre_tone_duration_s=pre_tone_duration_s,
        mobility_filter=mobility_filter,
        with_crossreg=crossreg_override,
    )

    pvs_recall, n_cells_recall = extract_recall_epoch_pvs(
        recall_session, mapping, data_mode=data_mode,
        recall_type=recall_type,
        epochs=recall_epochs,
        pre_tone_duration_s=pre_tone_duration_s,
        post_tone_duration_s=post_tone_duration_s,
        mobility_filter=mobility_filter,
        with_crossreg=crossreg_override,
    )

    n_recall_trials = (len(recall_session.tone_onsets)
                       if recall_type == 'testb'
                       else len(TESTA_PSEUDO_TONE_ONSETS_S))

    result = {
        'pvs_tfc': pvs_tfc,
        'pvs_recall': pvs_recall,
        'n_cells_tfc': n_cells_tfc,
        'n_cells_recall': n_cells_recall,
        'n_tfc_trials': len(tfc_session.tone_onsets),
        'n_recall_trials': n_recall_trials,
        'cross_rdm': {},
        'matched_similarity': {},
    }

    for metric in metrics:
        rdm, row_labels, col_labels, row_block_info, col_block_info = \
            compute_cross_session_rdm(
                pvs_tfc, pvs_recall, metric=metric,
                tfc_epochs=tfc_epochs, recall_epochs=recall_epochs,
                recall_label=recall_label,
            )
        result['cross_rdm'][metric] = {
            'rdm': rdm,
            'row_labels': row_labels, 'col_labels': col_labels,
            'row_block_info': row_block_info,
            'col_block_info': col_block_info,
        }

        matched_sim = compute_cross_session_matched_similarity(
            pvs_tfc, pvs_recall, metric, matched,
        )
        result['matched_similarity'][metric] = matched_sim

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Group-level cross-session aggregation
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_cross_session_rdm(all_mouse_results, mice_per_group, metric):
    """Average rectangular cross-session RDM across mice within each group."""
    group_mean_rdm, group_row_labels, group_col_labels = {}, {}, {}
    group_row_block, group_col_block = {}, {}
    for group, mice in mice_per_group.items():
        rdms = []
        rl_ref = cl_ref = rb_ref = cb_ref = None
        for mouse in mice:
            if mouse not in all_mouse_results:
                continue
            rdm_data = all_mouse_results[mouse]['cross_rdm'].get(metric)
            if rdm_data is None:
                continue
            rdms.append(rdm_data['rdm'])
            if rl_ref is None:
                rl_ref = rdm_data['row_labels']
                cl_ref = rdm_data['col_labels']
                rb_ref = rdm_data['row_block_info']
                cb_ref = rdm_data['col_block_info']
        if rdms:
            max_r = max(r.shape[0] for r in rdms)
            max_c = max(r.shape[1] for r in rdms)
            def _pad(r, nr, nc):
                if r.shape == (nr, nc):
                    return r
                out = np.full((nr, nc), np.nan)
                out[:r.shape[0], :r.shape[1]] = r
                return out
            stacked = np.stack([_pad(r, max_r, max_c) for r in rdms], axis=0)
            group_mean_rdm[group]   = np.nanmean(stacked, axis=0)
            group_row_labels[group] = rl_ref
            group_col_labels[group] = cl_ref
            group_row_block[group]  = rb_ref
            group_col_block[group]  = cb_ref
    return (group_mean_rdm, group_row_labels, group_col_labels,
            group_row_block, group_col_block)


# ─────────────────────────────────────────────────────────────────────────────
# Cross-session plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_cross_session_rdm_group(PLOTS_DIR, all_mouse_results, mice_per_group,
                                  metric='pearson', mapping='full',
                                  data_mode='S_mean', recall_label='Test_B',
                                  mobility_filter=None, auto_close=True):
    """Heatmap of the mean rectangular cross-session RSM per group.

    Rows  = TFC_cond epochs (y-axis).
    Columns = Recall epochs (x-axis).
    """
    (group_mean_rdm, group_row_labels, group_col_labels,
     group_row_block, group_col_block) = \
        aggregate_cross_session_rdm(all_mouse_results, mice_per_group, metric)
    if not group_mean_rdm:
        return

    groups = list(group_mean_rdm.keys())
    n_groups = len(groups)
    fig, axs = plt.subplots(1, n_groups, figsize=(5 * n_groups, 5),
                             squeeze=False)

    is_distance = metric in ('mahalanobis', 'euclidean')
    cbar_label = {
        'pearson': 'Pearson r', 'cosine': 'cosine',
        'mahalanobis': 'Mahalanobis distance',
        'euclidean': 'Euclidean distance',
    }.get(metric, 'similarity')

    for gi, group in enumerate(groups):
        ax = axs[0][gi]
        rdm = group_mean_rdm[group]
        row_block = group_row_block.get(group, [])
        col_block = group_col_block.get(group, [])

        if is_distance:
            vmax = np.nanpercentile(rdm, 95)
            im = ax.imshow(rdm, aspect='auto', cmap='viridis',
                           vmin=0, vmax=vmax, interpolation='none')
        elif metric == 'cosine':
            vmin = np.nanpercentile(rdm, 5)
            im = ax.imshow(rdm, aspect='auto', cmap='viridis',
                           vmin=vmin, vmax=1.0, interpolation='none')
        else:
            vmin = np.nanpercentile(rdm, 5)
            im = ax.imshow(rdm, aspect='auto', cmap='RdYlBu_r',
                           vmin=vmin, vmax=1.0, interpolation='none')
        plt.colorbar(im, ax=ax, label=cbar_label, fraction=0.046, pad=0.04)

        # Row (TFC) block separators
        for (_, _, end) in row_block:
            ax.axhline(end - 0.5, color='k', linewidth=0.5, linestyle='--')
        # Col (Recall) block separators
        for (_, _, end) in col_block:
            ax.axvline(end - 0.5, color='k', linewidth=0.5, linestyle='--')

        # Row (TFC) labels — y-axis
        yticks, ylabels = [], []
        for (epoch, start, end) in row_block:
            yticks.append((start + end - 1) / 2)
            ylabels.append(epoch.replace('_', '\n'))
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels, fontsize=7)

        # Col (Recall) labels — x-axis
        xticks, xlabels = [], []
        for (epoch, start, end) in col_block:
            xticks.append((start + end - 1) / 2)
            xlabels.append(epoch.replace('_', '\n'))
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, fontsize=7)

        ax.set_ylabel('TFC_cond epoch', fontsize=8)
        ax.set_xlabel(f'{recall_label} epoch', fontsize=8)
        ax.set_title(group, fontsize=11)

    dm_tag = _data_mode_tag(data_mode, mobility_filter)
    title_kind = 'RDM (distance)' if is_distance else 'RSM (similarity)'
    metric_label = {
        'pearson': 'Pearson r', 'cosine': 'cosine',
        'mahalanobis': 'Mahalanobis', 'euclidean': 'Euclidean',
    }.get(metric, metric)
    fig.suptitle(
        f'Cross-session {title_kind} — TFC_cond vs {recall_label}\n'
        f'{metric_label}, mapping={mapping}, data={dm_tag}',
        fontsize=10,
    )
    plt.tight_layout()

    save_dir = os.path.join(
        PLOTS_DIR, 'cross_session_epoch_analysis', recall_label, mapping, dm_tag)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'cross_session_rdm_{metric}.png'),
                dpi=200, bbox_inches='tight')
    if auto_close:
        plt.close(fig)
    return fig


def plot_cross_session_bars(PLOTS_DIR, all_mouse_results, mice_per_group,
                             metric='pearson', recall_type='testb',
                             recall_label='Test_B', mapping='full',
                             data_mode='S_mean', mobility_filter=None,
                             auto_close=True):
    """Bar + scatter of matched-epoch cross-session similarity per group."""
    matched = (MATCHED_EPOCHS_TESTB if recall_type == 'testb'
               else MATCHED_EPOCHS_TESTA)

    n_pairs = len(matched)
    fig, axs = plt.subplots(1, n_pairs, figsize=(3.5 * n_pairs, 3.5),
                             squeeze=False)

    metric_label = {
        'pearson': 'Pearson r', 'cosine': 'Cosine similarity',
        'mahalanobis': 'Mahalanobis distance',
        'euclidean': 'Euclidean distance',
    }.get(metric, metric)

    for idx, (tfc_ep, recall_ep) in enumerate(matched):
        ax = axs[0][idx]
        group_data = {g: [] for g in mice_per_group}

        for group, mice in mice_per_group.items():
            for mouse in mice:
                if mouse not in all_mouse_results:
                    continue
                ms = all_mouse_results[mouse]['matched_similarity'].get(
                    metric, {})
                val = ms.get((tfc_ep, recall_ep), np.nan)
                if not np.isnan(val):
                    group_data[group].append(val)

        title = f'TFC {tfc_ep}\nvs {recall_label} {recall_ep}'
        _barplot_with_scatter(ax, group_data, title=title, ylabel=metric_label)

    dm_tag = _data_mode_tag(data_mode, mobility_filter)
    fig.suptitle(
        f'Cross-session matched-epoch similarity\n'
        f'TFC_cond vs {recall_label} — {metric_label}\n'
        f'mapping={mapping}, data={dm_tag}',
        fontsize=9, y=1.02,
    )
    plt.tight_layout()

    save_dir = os.path.join(
        PLOTS_DIR, 'cross_session_epoch_analysis', recall_label, mapping, dm_tag)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'cross_session_matched_bars_{metric}.png'),
                dpi=200, bbox_inches='tight')
    if auto_close:
        plt.close(fig)
    return fig


def plot_cross_session_block_mean_rdm(
        PLOTS_DIR, all_mouse_results, mice_per_group,
        metric='pearson', mapping='full', data_mode='S_mean',
        recall_label='Test_B', recall_type='testb',
        mobility_filter=None, auto_close=True):
    """Compact epoch-pair heatmap: each cell is the group-mean of per-mouse
    block-mean similarities (TFC epoch rows × recall epoch columns).

    The per-mouse values are identical to those used in the bar plots
    (matched + non-matched), so this provides a full cross-session matrix
    at epoch resolution rather than trial resolution.
    """
    tfc_epochs = EPOCH_NAMES
    recall_epochs = (TESTB_EPOCH_NAMES if recall_type == 'testb'
                     else TESTA_EPOCH_NAMES)

    groups = list(mice_per_group.keys())
    n_groups = len(groups)
    n_tfc = len(tfc_epochs)
    n_rec = len(recall_epochs)

    # Collect per-mouse block-mean matrices, then average per group
    group_matrices = {}
    for group in groups:
        matrices = []
        for mouse in mice_per_group[group]:
            if mouse not in all_mouse_results:
                continue
            res = all_mouse_results[mouse]
            pvs_tfc = res['pvs_tfc']
            pvs_recall = res['pvs_recall']
            mat = np.full((n_tfc, n_rec), np.nan)
            for ri, tfc_ep in enumerate(tfc_epochs):
                if tfc_ep not in pvs_tfc:
                    continue
                for ci, rec_ep in enumerate(recall_epochs):
                    if rec_ep not in pvs_recall:
                        continue
                    pv_t = pvs_tfc[tfc_ep]
                    pv_r = pvs_recall[rec_ep]
                    vals = [_pv_similarity(pv_t[i], pv_r[j], metric)
                            for i in range(pv_t.shape[0])
                            for j in range(pv_r.shape[0])]
                    vals = [v for v in vals if not np.isnan(v)]
                    if vals:
                        mat[ri, ci] = float(np.mean(vals))
            matrices.append(mat)
        if matrices:
            group_matrices[group] = np.nanmean(np.stack(matrices, axis=0),
                                               axis=0)

    if not group_matrices:
        return

    is_distance = metric in ('mahalanobis', 'euclidean')
    cbar_label = {
        'pearson': 'Pearson r', 'cosine': 'cosine',
        'mahalanobis': 'Mahalanobis distance',
        'euclidean': 'Euclidean distance',
    }.get(metric, 'similarity')

    fig, axs = plt.subplots(1, n_groups, figsize=(3.2 * n_groups + 1, 4),
                             squeeze=False)

    for gi, group in enumerate(groups):
        ax = axs[0][gi]
        mat = group_matrices[group]

        if is_distance:
            vmax = np.nanpercentile(mat, 95)
            im = ax.imshow(mat, aspect='auto', cmap='viridis',
                           vmin=0, vmax=vmax, interpolation='none')
        elif metric == 'cosine':
            vmin = np.nanpercentile(mat, 5)
            im = ax.imshow(mat, aspect='auto', cmap='viridis',
                           vmin=vmin, vmax=1.0, interpolation='none')
        else:
            vmin = np.nanpercentile(mat, 5)
            im = ax.imshow(mat, aspect='auto', cmap='RdYlBu_r',
                           vmin=vmin, vmax=1.0, interpolation='none')
        plt.colorbar(im, ax=ax, label=cbar_label, fraction=0.046, pad=0.04)

        # Numeric annotations
        for ri in range(n_tfc):
            for ci in range(n_rec):
                v = mat[ri, ci]
                if not np.isnan(v):
                    txt_color = 'white' if v < (im.norm.vmin + im.norm.vmax) / 2 else 'black'
                    ax.text(ci, ri, f'{v:.2f}', ha='center', va='center',
                            fontsize=7, color=txt_color)

        ax.set_yticks(range(n_tfc))
        ax.set_yticklabels([e.replace('_', '\n') for e in tfc_epochs],
                           fontsize=7)
        ax.set_xticks(range(n_rec))
        ax.set_xticklabels([e.replace('_', '\n') for e in recall_epochs],
                           fontsize=7)
        ax.set_ylabel('TFC_cond epoch', fontsize=8)
        ax.set_xlabel(f'{recall_label} epoch', fontsize=8)
        ax.set_title(group, fontsize=11)

    dm_tag = _data_mode_tag(data_mode, mobility_filter)
    title_kind = 'RDM (distance)' if is_distance else 'RSM (similarity)'
    metric_label = {
        'pearson': 'Pearson r', 'cosine': 'cosine',
        'mahalanobis': 'Mahalanobis', 'euclidean': 'Euclidean',
    }.get(metric, metric)
    fig.suptitle(
        f'Cross-session block-mean {title_kind} — TFC_cond vs {recall_label}\n'
        f'{metric_label}, mapping={mapping}, data={dm_tag}',
        fontsize=10,
    )
    plt.tight_layout()

    save_dir = os.path.join(
        PLOTS_DIR, 'cross_session_epoch_analysis', recall_label, mapping, dm_tag)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir,
                             f'cross_session_block_mean_rdm_{metric}.png'),
                dpi=200, bbox_inches='tight')
    if auto_close:
        plt.close(fig)
    return fig


def plot_cross_session_mean_pv_rdm(
        PLOTS_DIR, all_mouse_results, mice_per_group,
        metric='pearson', mapping='full', data_mode='S_mean',
        recall_label='Test_B', recall_type='testb',
        mobility_filter=None, auto_close=True):
    """6×3 heatmap where trial PVs are averaged *before* computing similarity.

    For each mouse, the 5 TFC trial PVs per epoch are averaged into one mean
    PV, and the 3 recall trial PVs per epoch into one mean PV.  Similarity is
    then computed between these mean PVs.  The per-mouse 6×3 matrices are
    averaged across mice within each group.
    """
    tfc_epochs = EPOCH_NAMES
    recall_epochs = (TESTB_EPOCH_NAMES if recall_type == 'testb'
                     else TESTA_EPOCH_NAMES)

    groups = list(mice_per_group.keys())
    n_groups = len(groups)
    n_tfc = len(tfc_epochs)
    n_rec = len(recall_epochs)

    group_matrices = {}
    for group in groups:
        matrices = []
        for mouse in mice_per_group[group]:
            if mouse not in all_mouse_results:
                continue
            res = all_mouse_results[mouse]
            pvs_tfc = res['pvs_tfc']
            pvs_recall = res['pvs_recall']
            mat = np.full((n_tfc, n_rec), np.nan)
            for ri, tfc_ep in enumerate(tfc_epochs):
                if tfc_ep not in pvs_tfc:
                    continue
                pv_t = pvs_tfc[tfc_ep]          # (n_trials, n_cells)
                # Mean PV across trials (ignoring NaN rows)
                valid_t = ~np.all(np.isnan(pv_t), axis=1)
                if not np.any(valid_t):
                    continue
                mean_tfc = np.nanmean(pv_t[valid_t], axis=0)
                for ci, rec_ep in enumerate(recall_epochs):
                    if rec_ep not in pvs_recall:
                        continue
                    pv_r = pvs_recall[rec_ep]    # (n_trials, n_cells)
                    valid_r = ~np.all(np.isnan(pv_r), axis=1)
                    if not np.any(valid_r):
                        continue
                    mean_rec = np.nanmean(pv_r[valid_r], axis=0)
                    mat[ri, ci] = _pv_similarity(mean_tfc, mean_rec, metric)
            matrices.append(mat)
        if matrices:
            group_matrices[group] = np.nanmean(np.stack(matrices, axis=0),
                                               axis=0)

    if not group_matrices:
        return

    is_distance = metric in ('mahalanobis', 'euclidean')
    cbar_label = {
        'pearson': 'Pearson r', 'cosine': 'cosine',
        'mahalanobis': 'Mahalanobis distance',
        'euclidean': 'Euclidean distance',
    }.get(metric, 'similarity')

    fig, axs = plt.subplots(1, n_groups, figsize=(3.2 * n_groups + 1, 4),
                             squeeze=False)

    for gi, group in enumerate(groups):
        ax = axs[0][gi]
        mat = group_matrices[group]

        if is_distance:
            vmax = np.nanpercentile(mat, 95)
            im = ax.imshow(mat, aspect='auto', cmap='viridis',
                           vmin=0, vmax=vmax, interpolation='none')
        elif metric == 'cosine':
            vmin = np.nanpercentile(mat, 5)
            im = ax.imshow(mat, aspect='auto', cmap='viridis',
                           vmin=vmin, vmax=1.0, interpolation='none')
        else:
            vmin = np.nanpercentile(mat, 5)
            im = ax.imshow(mat, aspect='auto', cmap='RdYlBu_r',
                           vmin=vmin, vmax=1.0, interpolation='none')
        plt.colorbar(im, ax=ax, label=cbar_label, fraction=0.046, pad=0.04)

        # Numeric annotations
        for ri in range(n_tfc):
            for ci in range(n_rec):
                v = mat[ri, ci]
                if not np.isnan(v):
                    txt_color = ('white'
                                 if v < (im.norm.vmin + im.norm.vmax) / 2
                                 else 'black')
                    ax.text(ci, ri, f'{v:.2f}', ha='center', va='center',
                            fontsize=7, color=txt_color)

        ax.set_yticks(range(n_tfc))
        ax.set_yticklabels([e.replace('_', '\n') for e in tfc_epochs],
                           fontsize=7)
        ax.set_xticks(range(n_rec))
        ax.set_xticklabels([e.replace('_', '\n') for e in recall_epochs],
                           fontsize=7)
        ax.set_ylabel('TFC_cond epoch', fontsize=8)
        ax.set_xlabel(f'{recall_label} epoch', fontsize=8)
        ax.set_title(group, fontsize=11)

    dm_tag = _data_mode_tag(data_mode, mobility_filter)
    title_kind = 'RDM (distance)' if is_distance else 'RSM (similarity)'
    metric_label = {
        'pearson': 'Pearson r', 'cosine': 'cosine',
        'mahalanobis': 'Mahalanobis', 'euclidean': 'Euclidean',
    }.get(metric, metric)
    fig.suptitle(
        f'Cross-session mean-PV {title_kind} — TFC_cond vs {recall_label}\n'
        f'{metric_label}, mapping={mapping}, data={dm_tag}',
        fontsize=10,
    )
    plt.tight_layout()

    save_dir = os.path.join(
        PLOTS_DIR, 'cross_session_epoch_analysis', recall_label, mapping, dm_tag)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir,
                             f'cross_session_mean_pv_rdm_{metric}.png'),
                dpi=200, bbox_inches='tight')
    if auto_close:
        plt.close(fig)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Cross-session orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_cross_session_epoch_analysis_all_mice(
        PLOTS_DIR, mice_per_group, TFC_cond,
        recall_sessions, recall_label, recall_type,
        crossreg_dict, mapping,
        data_mode='S_mean',
        metrics=('pearson', 'cosine'),
        tfc_epochs=None, recall_epochs=None,
        peri_shock_pre_s=10.0, peri_shock_post_s=10.0,
        pre_tone_duration_s=35.0, post_tone_duration_s=35.0,
        mobility_filter=None,
        auto_close=True):
    """
    Cross-session epoch PV analysis across all mice.

    Parameters
    ----------
    PLOTS_DIR       : str
    mice_per_group  : dict  group -> [mice]
    TFC_cond        : dict  mouse -> TraceFearCondSession
    recall_sessions : dict  mouse -> TestBSession or TestASession
    recall_label    : str   e.g. 'Test_B', 'Test_B_1wk', 'Test_A', 'Test_A_1wk'
    recall_type     : str   'testb' or 'testa'
    crossreg_dict   : dict  mouse -> CrossRegMapping  (matching the *mapping*)
    mapping         : str   crossreg mapping string
    data_mode       : str
    metrics         : tuple of str
    tfc_epochs      : list or None
    recall_epochs   : list or None
    peri_shock_pre_s  : float
    peri_shock_post_s : float
    pre_tone_duration_s   : float
    post_tone_duration_s  : float  – for recall post-tone / post-pseudo-tone
    mobility_filter : None | 'mobile' | 'immobile'
    auto_close      : bool

    Returns
    -------
    all_mouse_results : dict  mouse -> run_cross_session_analysis_mouse result
    """
    if tfc_epochs is None:
        tfc_epochs = EPOCH_NAMES
    if recall_epochs is None:
        recall_epochs = (TESTB_EPOCH_NAMES if recall_type == 'testb'
                         else TESTA_EPOCH_NAMES)

    all_mouse_results = {}
    mouse_list = [m for mice in mice_per_group.values() for m in mice]

    for mouse in mouse_list:
        if mouse not in TFC_cond:
            continue
        if mouse not in recall_sessions:
            continue
        if mapping != 'full' and mouse not in crossreg_dict:
            continue

        tfc_session    = TFC_cond[mouse]
        recall_session = recall_sessions[mouse]
        xreg = crossreg_dict.get(mouse) if mapping != 'full' else None

        try:
            result = run_cross_session_analysis_mouse(
                mouse=mouse,
                tfc_session=tfc_session,
                recall_session=recall_session,
                mapping=mapping,
                data_mode=data_mode,
                recall_type=recall_type,
                recall_label=recall_label,
                metrics=list(metrics),
                tfc_epochs=tfc_epochs,
                recall_epochs=recall_epochs,
                peri_shock_pre_s=peri_shock_pre_s,
                peri_shock_post_s=peri_shock_post_s,
                pre_tone_duration_s=pre_tone_duration_s,
                post_tone_duration_s=post_tone_duration_s,
                mobility_filter=mobility_filter,
                crossreg_override=xreg,
            )
            all_mouse_results[mouse] = result
            print(f'[cross-session] {mouse} done — '
                  f'{result["n_cells_tfc"]} TFC cells, '
                  f'{result["n_cells_recall"]} {recall_label} cells')
        except Exception as exc:
            print(f'[cross-session] {mouse} FAILED: {exc}')

    if not all_mouse_results:
        print('[cross-session] No results computed.')
        return all_mouse_results

    # ── Plots ──────────────────────────────────────────────────────────────
    for metric in metrics:
        plot_cross_session_rdm_group(
            PLOTS_DIR, all_mouse_results, mice_per_group,
            metric=metric, mapping=mapping, data_mode=data_mode,
            recall_label=recall_label,
            mobility_filter=mobility_filter, auto_close=auto_close,
        )

        plot_cross_session_bars(
            PLOTS_DIR, all_mouse_results, mice_per_group,
            metric=metric, recall_type=recall_type,
            recall_label=recall_label, mapping=mapping,
            data_mode=data_mode, mobility_filter=mobility_filter,
            auto_close=auto_close,
        )

        plot_cross_session_block_mean_rdm(
            PLOTS_DIR, all_mouse_results, mice_per_group,
            metric=metric, mapping=mapping, data_mode=data_mode,
            recall_label=recall_label, recall_type=recall_type,
            mobility_filter=mobility_filter, auto_close=auto_close,
        )

        plot_cross_session_mean_pv_rdm(
            PLOTS_DIR, all_mouse_results, mice_per_group,
            metric=metric, mapping=mapping, data_mode=data_mode,
            recall_label=recall_label, recall_type=recall_type,
            mobility_filter=mobility_filter, auto_close=auto_close,
        )

    return all_mouse_results


# ─────────────────────────────────────────────────────────────────────────────
# TFC conditioning summary panels (B, C, D, E1, E2, F1, F2)
# ─────────────────────────────────────────────────────────────────────────────

def _stars_from_p(p):
    if p is None or not np.isfinite(p):
        return None
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return None


def _safe_corr(x, y, method='pearson'):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    if x.size < 3:
        return np.nan, np.nan, int(x.size)
    if method == 'spearman':
        r, p = stats.spearmanr(x, y)
    else:
        r, p = stats.pearsonr(x, y)
    return float(r), float(p), int(x.size)


def _welch_holm_pairs(values_by_group):
    groups = [g for g in ['hM3D', 'mCherry', 'hM4D'] if g in values_by_group]
    pairs = []
    raw_ps = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            a = groups[i]
            b = groups[j]
            va = np.asarray(values_by_group[a], dtype=float)
            vb = np.asarray(values_by_group[b], dtype=float)
            va = va[np.isfinite(va)]
            vb = vb[np.isfinite(vb)]
            if va.size < 2 or vb.size < 2:
                p_raw = np.nan
                t_val = np.nan
            else:
                t_val, p_raw = stats.ttest_ind(va, vb, equal_var=False)
            pairs.append({
                'a': a,
                'b': b,
                't': float(t_val) if np.isfinite(t_val) else np.nan,
                'p_raw': float(p_raw) if np.isfinite(p_raw) else np.nan,
                'n_a': int(va.size),
                'n_b': int(vb.size),
                'mean_diff': float(np.nanmean(va) - np.nanmean(vb)),
            })
            raw_ps.append(p_raw)

    finite_ps = [p for p in raw_ps if p is not None and np.isfinite(p)]
    if finite_ps:
        _, p_holm, _, _ = multipletests(finite_ps, method='holm')
        k = 0
        for row in pairs:
            if np.isfinite(row['p_raw']):
                row['p_holm'] = float(p_holm[k])
                k += 1
            else:
                row['p_holm'] = np.nan
    else:
        for row in pairs:
            row['p_holm'] = np.nan
    return pairs


def _draw_pairwise_brackets(ax, values_by_group, pair_rows, groups_order):
    y_vals = []
    for g in groups_order:
        arr = np.asarray(values_by_group.get(g, []), dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            y_vals.append(np.nanmax(arr))
    if not y_vals:
        return
    y_max = float(np.nanmax(y_vals))
    y_min = float(np.nanmin(y_vals)) if y_vals else 0.0
    y_rng = max(y_max - y_min, 1e-6)
    base = y_max + 0.10 * y_rng
    step = 0.11 * y_rng
    h = 0.04 * y_rng

    x_map = {g: i for i, g in enumerate(groups_order)}
    level = 0
    for row in pair_rows:
        star = _stars_from_p(row.get('p_holm', np.nan))
        if star is None:
            continue
        a = row['a']
        b = row['b']
        if a not in x_map or b not in x_map:
            continue
        x1 = x_map[a]
        x2 = x_map[b]
        if x1 > x2:
            x1, x2 = x2, x1
        y = base + level * step
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color='#222222', lw=1.0)
        ax.text((x1 + x2) / 2.0, y + h + 0.01 * y_rng, star,
                ha='center', va='bottom', fontsize=10)
        level += 1


def _extract_population_activity_trace(session, mapping='full', crossreg_override=None):
    if mapping == 'full':
        S = session.S
        S_spikes = session.S_spikes
        S_peakval = session.S_peakval
    else:
        S, S_spikes, S_peakval, _ = session.get_S_mapping(
            mapping, want_peakval=True, with_crossreg=crossreg_override
        )
    n_cells, n_frames = S.shape
    if n_cells <= 0 or n_frames <= 0:
        raise RuntimeError(f'{session.mouse}: invalid S shape {S.shape} for mapping={mapping}.')

    pop_trace = np.zeros(n_frames, dtype=float)
    for cell in S_spikes.keys():
        spk = np.asarray(S_spikes[cell], dtype=int)
        if spk.size == 0:
            continue
        vals = np.asarray(S_peakval[cell], dtype=float)
        keep = (spk >= 0) & (spk < n_frames)
        if not np.any(keep):
            continue
        np.add.at(pop_trace, spk[keep], vals[keep])

    # Convert to per-second transient amplitude rate and average across cells.
    pop_trace = (pop_trace * MINISCOPE_FPS) / float(n_cells)
    return pop_trace


def _extract_peri_shock_population_trace(pop_trace, onset, offset, pre_frames,
                                          smooth_sigma_frames=0.0):
    """Peri-shock window of the population transient amplitude rate trace,
    baseline-subtracted by its pre-shock segment and optionally smoothed.

    Uses the SAME signal as the binned activity plot
    (`_extract_population_activity_trace`): sum of transient amplitudes across
    all mapped cells, normalized by total cell count, per second.
    """
    n_frames = len(pop_trace)
    onset = int(max(0, onset))
    offset = int(min(n_frames, offset))
    if offset <= onset:
        return None
    if pre_frames <= 0 or (offset - onset) <= pre_frames:
        return None

    tr = np.asarray(pop_trace[onset:offset], dtype=float)
    baseline = float(np.nanmean(tr[:pre_frames]))
    tr = tr - baseline
    if smooth_sigma_frames and smooth_sigma_frames > 0:
        tr = gaussian_filter1d(tr, sigma=float(smooth_sigma_frames), mode='nearest')
    return tr


def _extract_active_neuron_peri_trace_DEPRECATED(session, onset, offset, pre_frames, mapping='full', crossreg_override=None,
                                                  smooth_sigma_frames=0.0, dff_eps=1e-6):
    if mapping == 'full':
        C = np.asarray(session.C, dtype=float)
        S = session.S
        S_spikes = session.S_spikes
    else:
        S, S_spikes, _, cell_ids = session.get_S_mapping(
            mapping, want_peakval=True, with_crossreg=crossreg_override
        )
        C_full = np.asarray(session.C, dtype=float)
        C = C_full[np.asarray(cell_ids, dtype=int), :]

    n_cells, n_frames = S.shape
    if n_cells <= 0 or n_frames <= 0:
        raise RuntimeError(f'{session.mouse}: invalid S shape {S.shape} for mapping={mapping}.')
    if C.shape != S.shape:
        raise RuntimeError(
            f'{session.mouse}: C shape {C.shape} does not match S shape {S.shape} for mapping={mapping}.'
        )

    onset = int(max(0, onset))
    offset = int(min(n_frames, offset))
    if offset <= onset:
        return None
    if pre_frames <= 0 or (offset - onset) <= pre_frames:
        return None

    active_cell_traces = []
    for cell in S_spikes.keys():
        spk = np.asarray(S_spikes[cell], dtype=int)
        if spk.size == 0:
            continue
        if not np.any((spk >= onset) & (spk < offset)):
            continue
        c_trace = np.asarray(C[cell, onset:offset], dtype=float)
        baseline = float(np.nanmean(c_trace[:pre_frames]))
        denom = max(baseline, dff_eps)
        dff_trace = 100.0 * (c_trace - baseline) / denom
        active_cell_traces.append(dff_trace)

    if not active_cell_traces:
        return None

    trace = np.nanmean(np.vstack(active_cell_traces), axis=0)
    if smooth_sigma_frames and smooth_sigma_frames > 0:
        trace = gaussian_filter1d(trace, sigma=float(smooth_sigma_frames), mode='nearest')
    return trace


def _mean_in_window(vec, beg, end):
    beg = int(max(0, beg))
    end = int(min(len(vec), end))
    if end <= beg:
        return np.nan
    chunk = np.asarray(vec[beg:end], dtype=float)
    if chunk.size == 0:
        return np.nan
    return float(np.nanmean(chunk))


def _freezing_fraction_in_window(session, beg, end, freeze_thresh_cm_s=2.0):
    vel = getattr(session, 'velocities_miniscope_smooth', None)
    if vel is None:
        vel = getattr(session, 'velocities_miniscope', None)
    if vel is None:
        return np.nan

    beg = int(max(0, beg))
    end = int(min(len(vel), end))
    if end <= beg:
        return np.nan
    chunk = np.asarray(vel[beg:end], dtype=float)
    chunk = chunk[np.isfinite(chunk)]
    if chunk.size == 0:
        return np.nan
    return float(np.mean(chunk < float(freeze_thresh_cm_s)))


def _fit_mixed_model_or_fallback(df, formula, group_col='mouse'):
    if df.empty:
        return {'kind': 'empty', 'summary': 'No rows for model.', 'model': None}
    try:
        md = smf.mixedlm(formula, df, groups=df[group_col])
        fit = md.fit(reml=True)
        return {
            'kind': 'mixedlm',
            'summary': str(fit.summary()),
            'model': fit,
        }
    except Exception as exc:
        ols = smf.ols(formula, df).fit(
            cov_type='cluster',
            cov_kwds={'groups': df[group_col]}
        )
        return {
            'kind': 'ols_cluster',
            'summary': f'MixedLM failed ({exc}); fallback OLS cluster:\n\n{ols.summary()}',
            'model': ols,
        }


def _build_tfc_conditioning_tables(
    TFC_cond,
    mouse_groups,
    *,
    mapping='full',
    peri_pre_s=20.0,
    peri_post_s=40.0,
    baseline_pre_s=30.0,
    post_shock_s=20.0,
    response_pre_s=20.0,
    pre_cs_s=35.0,
    final_follow_s=90.0,
    freeze_thresh_cm_s=2.0,
    c_trace_smoothing_sigma_frames=0.0,
):
    trial_rows = []
    baseline_rows = []
    peri_traces = []

    peri_pre_f = int(round(peri_pre_s * MINISCOPE_FPS))
    peri_post_f = int(round(peri_post_s * MINISCOPE_FPS))
    baseline_pre_f = int(round(baseline_pre_s * MINISCOPE_FPS))
    post_shock_f = int(round(post_shock_s * MINISCOPE_FPS))
    response_pre_f = int(round(response_pre_s * MINISCOPE_FPS))
    pre_cs_f = int(round(pre_cs_s * MINISCOPE_FPS))
    final_follow_f = int(round(final_follow_s * MINISCOPE_FPS))

    for mouse in sorted(TFC_cond.keys()):
        if mouse not in mouse_groups:
            continue
        sess = TFC_cond[mouse]
        group = mouse_groups[mouse]

        tone_on = list(getattr(sess, 'tone_onsets', []) or [])
        shock_on = list(getattr(sess, 'shock_onsets', []) or [])
        shock_off = list(getattr(sess, 'shock_offsets', []) or [])
        if len(shock_on) == 0 or len(shock_off) == 0:
            continue

        n_trials = min(len(shock_on), len(shock_off))
        if len(tone_on) > 0:
            n_trials = min(n_trials, max(len(tone_on), n_trials))

        pop_trace = _extract_population_activity_trace(sess, mapping=mapping)
        S = sess.S if mapping == 'full' else sess.get_S_mapping(mapping, want_peakval=True)[0]
        n_frames = S.shape[1]

        if len(tone_on) > 0:
            pre_cs_end = int(tone_on[0])
            pre_cs_start = pre_cs_end - pre_cs_f
            pre_cs_val = _mean_in_window(pop_trace, pre_cs_start, pre_cs_end)
            baseline_rows.append({
                'mouse': mouse,
                'group': group,
                'pre_cs_value': pre_cs_val,
            })

        for ti in range(n_trials):
            shock_beg = int(shock_on[ti])
            shock_end = int(shock_off[ti])
            shock_number = int(ti + 1)

            baseline_beg = shock_beg - response_pre_f
            baseline_end = shock_beg
            post_beg = shock_beg
            post_end = shock_beg + post_shock_f

            baseline_val = _mean_in_window(pop_trace, baseline_beg, baseline_end)
            post_val = _mean_in_window(pop_trace, post_beg, post_end)
            delta_val = post_val - baseline_val if np.isfinite(post_val) and np.isfinite(baseline_val) else np.nan

            if ti < len(tone_on) - 1:
                freeze_beg = shock_end
                freeze_end = int(tone_on[ti + 1])
                freeze_window = 'following_iti'
            else:
                freeze_beg = shock_end
                freeze_end = min(n_frames, shock_end + final_follow_f)
                freeze_window = 'post_final_shock'

            freeze_frac = _freezing_fraction_in_window(
                sess,
                freeze_beg,
                freeze_end,
                freeze_thresh_cm_s=freeze_thresh_cm_s,
            )

            trial_rows.append({
                'mouse': mouse,
                'group': group,
                'shock_number': shock_number,
                'shock_window': 'early' if shock_number <= 3 else 'late',
                'baseline_value': baseline_val,
                'post_shock_value': post_val,
                'post_shock_delta': delta_val,
                'freeze_fraction': freeze_frac,
                'freeze_percent': float(100.0 * freeze_frac) if np.isfinite(freeze_frac) else np.nan,
                'freeze_window': freeze_window,
            })

            peri_beg = shock_beg - peri_pre_f
            peri_end = shock_beg + peri_post_f
            tr = _extract_peri_shock_population_trace(
                pop_trace,
                peri_beg,
                peri_end,
                peri_pre_f,
                smooth_sigma_frames=c_trace_smoothing_sigma_frames,
            )
            if tr is None or tr.size != (peri_pre_f + peri_post_f):
                continue
            peri_traces.append({
                'mouse': mouse,
                'group': group,
                'shock_number': shock_number,
                'trace_bs': tr,
            })

    trial_df = pd.DataFrame(trial_rows)
    baseline_df = pd.DataFrame(baseline_rows)
    return trial_df, baseline_df, peri_traces, peri_pre_f, peri_post_f


def plot_tfc_conditioning_summary_panels(
    PLOTS_DIR,
    TFC_cond,
    mouse_groups,
    *,
    mapping='full',
    peri_pre_s=20.0,
    peri_post_s=40.0,
    baseline_pre_s=30.0,
    post_shock_s=20.0,
    pre_cs_s=35.0,
    final_follow_s=90.0,
    freeze_thresh_cm_s=2.0,
    paper_dir=None,
    style=None,
    auto_close=True,
):
    """Build and save the TFC conditioning summary figure (panels B-F2).
    
    If paper_dir is provided, copies of the PNG/PDF are saved there as well.
    """
    if style is None:
        style = dict(TFC_CONDITIONING_PANEL_STYLE_SPACIOUS)
    elif isinstance(style, str):
        _k = style.strip().lower()
        if _k in ('spacious', 'default', 'normal'):
            style = dict(TFC_CONDITIONING_PANEL_STYLE_SPACIOUS)
        elif _k in ('compact', 'paper'):
            style = dict(TFC_CONDITIONING_PANEL_STYLE_COMPACT)
        else:
            raise ValueError(f'Unknown style alias for TFC conditioning summary: {style!r}')
    else:
        style = dict(style)
    group_order_style = list(style['group_order'])
    group_palette_box = style['group_palette_box']
    group_palette_dot = style['group_palette_dot']

    def _style_distribution_axis(ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', labelsize=style['tick_fontsize'])

    def _style_trace_axis(ax):
        _style_distribution_axis(ax)
        ax.title.set_fontsize(style['title_fontsize'])
        ax.xaxis.label.set_size(style['axis_label_fontsize'])
        ax.yaxis.label.set_size(style['axis_label_fontsize'])

    def _style_legend(legend):
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(style['legend_fontsize'])

    def _apply_compact_style_to_existing_figure(fig_obj, compact_style):
        fig_obj.set_size_inches(*compact_style['combined_figsize'])
        for ax in fig_obj.axes:
            ax.title.set_fontsize(compact_style['title_fontsize'])
            ax.xaxis.label.set_size(compact_style['axis_label_fontsize'])
            ax.yaxis.label.set_size(compact_style['axis_label_fontsize'])
            ax.tick_params(axis='both', labelsize=compact_style['tick_fontsize'])
            leg = ax.get_legend()
            if leg is not None:
                for txt in leg.get_texts():
                    txt.set_fontsize(compact_style['legend_fontsize'])

    def _draw_group_box_strip(ax, data_df, y_col, order, title, ylabel, pairs=None):
        sns.violinplot(
            data=data_df,
            x='group',
            y=y_col,
            order=order,
            palette=[group_palette_box[g] for g in order],
            inner=None,
            linewidth=style['box_linewidth'],
            ax=ax,
        )
        sns.stripplot(
            data=data_df,
            x='group',
            y=y_col,
            order=order,
            hue='group',
            hue_order=order,
            palette=group_palette_dot,
            dodge=False,
            size=style['point_size_distribution'],
            alpha=style['point_alpha_distribution'],
            edgecolor='k',
            linewidth=style['point_linewidth'],
            jitter=0.15,
            ax=ax,
        )
        # Add median markers
        for i, group in enumerate(order):
            group_data = data_df.loc[data_df['group'] == group, y_col].dropna()
            if len(group_data) > 0:
                median_val = group_data.median()
                ax.scatter(i, median_val, color='black', s=100, marker='_', linewidths=style['median_linewidth']*2, zorder=10)
        
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        ax.set_title(title)
        ax.set_xlabel('')
        ax.set_ylabel(ylabel)
        _style_distribution_axis(ax)
        if pairs is not None:
            value_map = {
                g: data_df.loc[data_df['group'] == g, y_col].astype(float).dropna().values
                for g in order
            }
            _draw_pairwise_brackets(ax, value_map, pairs, order)

    save_dir = os.path.join(PLOTS_DIR, 'tfc_conditioning_summary', mapping)
    os.makedirs(save_dir, exist_ok=True)
    stats_dir = os.path.join(save_dir, 'stats')
    table_dir = os.path.join(save_dir, 'tables')
    os.makedirs(stats_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)

    trial_df, baseline_df, peri_traces, peri_pre_f, peri_post_f = _build_tfc_conditioning_tables(
        TFC_cond,
        mouse_groups,
        mapping=mapping,
        peri_pre_s=peri_pre_s,
        peri_post_s=peri_post_s,
        baseline_pre_s=baseline_pre_s,
        post_shock_s=post_shock_s,
        pre_cs_s=pre_cs_s,
        final_follow_s=final_follow_s,
        freeze_thresh_cm_s=freeze_thresh_cm_s,
        c_trace_smoothing_sigma_frames=style['trace_smoothing_sigma_frames'],
    )

    if trial_df.empty:
        raise RuntimeError('No TFC conditioning rows available for summary plotting.')

    # Convert 'group' to categorical with mCherry as reference level for all stats
    group_cat_order = ['mCherry', 'hM3D', 'hM4D']  # mCherry first = reference
    trial_df['group'] = pd.Categorical(trial_df['group'], categories=group_cat_order, ordered=False)
    baseline_df['group'] = pd.Categorical(baseline_df['group'], categories=group_cat_order, ordered=False)

    baseline_df.to_csv(os.path.join(table_dir, 'panel_B_pre_cs_baseline.csv'), index=False)
    trial_df.to_csv(os.path.join(table_dir, 'trial_level_calcium_freezing.csv'), index=False)

    grp_order = [g for g in group_order_style if g in set(trial_df['group'])]
    fig = plt.figure(figsize=style['combined_figsize'])
    gs = fig.add_gridspec(2, 4, wspace=style['grid_wspace'], hspace=style['grid_hspace'])

    # Panel B
    ax_B = fig.add_subplot(gs[0, 0])
    b_plot_df = baseline_df[['group', 'pre_cs_value']].dropna().copy()
    b_vals = {
        g: b_plot_df.loc[b_plot_df['group'] == g, 'pre_cs_value'].astype(float).dropna().values
        for g in grp_order
    }
    _draw_group_box_strip(ax_B, b_plot_df, 'pre_cs_value', grp_order, 'B: Pre-CS baseline', 'Transient amplitude rate')
    b_pairs = _welch_holm_pairs(b_vals)
    _draw_pairwise_brackets(ax_B, b_vals, b_pairs, grp_order)

    # Panel C (early and late shock aligned)
    ax_Ce = fig.add_subplot(gs[0, 1])
    ax_Cl = fig.add_subplot(gs[0, 2])
    tvec = (np.arange(peri_pre_f + peri_post_f) - peri_pre_f) / float(MINISCOPE_FPS)

    for g in grp_order:
        col = GROUP_COLOURS.get(g, 'gray')
        mouse_early = []
        mouse_late = []
        for mouse in sorted(TFC_cond.keys()):
            if mouse_groups.get(mouse) != g:
                continue
            traces_mouse = [row['trace_bs'] for row in peri_traces if row['mouse'] == mouse]
            shocks_mouse = [row['shock_number'] for row in peri_traces if row['mouse'] == mouse]
            if not traces_mouse:
                continue
            early = [tr for tr, sn in zip(traces_mouse, shocks_mouse) if sn in (1, 2)]
            late = [tr for tr, sn in zip(traces_mouse, shocks_mouse) if sn in (4, 5)]
            if early:
                mouse_early.append(np.nanmean(np.vstack(early), axis=0))
            if late:
                mouse_late.append(np.nanmean(np.vstack(late), axis=0))

        if mouse_early:
            mat = np.vstack(mouse_early)
            mean = np.nanmean(mat, axis=0)
            sem = np.nanstd(mat, axis=0) / np.sqrt(max(mat.shape[0], 1))
            ax_Ce.plot(tvec, mean, color=col, lw=style['trace_linewidth'], alpha=style['trace_line_alpha'])
            ax_Ce.fill_between(tvec, mean - sem, mean + sem, color=col, alpha=style['trace_alpha'])
        if mouse_late:
            mat = np.vstack(mouse_late)
            mean = np.nanmean(mat, axis=0)
            sem = np.nanstd(mat, axis=0) / np.sqrt(max(mat.shape[0], 1))
            ax_Cl.plot(tvec, mean, color=col, lw=style['trace_linewidth'], alpha=style['trace_line_alpha'])
            ax_Cl.fill_between(tvec, mean - sem, mean + sem, color=col, alpha=style['trace_alpha'])

    for axc, title in [(ax_Ce, 'C: Early shocks (1-2)'), (ax_Cl, 'C: Late shocks (4-5)')]:
        axc.axvline(0.0, color='#666666', ls='--', lw=1.0)
        axc.set_xlim(-peri_pre_s, peri_post_s)
        axc.set_xlabel('Time from shock (s)')
        axc.set_ylabel('Δ activity (a.u./s)')
        axc.set_title(title)
        axc.spines['top'].set_visible(False)
        axc.spines['right'].set_visible(False)

    # Panel D: post-shock index across shocks
    ax_D = fig.add_subplot(gs[0, 3])
    d_df = trial_df[['mouse', 'group', 'shock_number', 'post_shock_delta']].dropna().copy()
    d_mouse = (
        d_df.groupby(['mouse', 'group', 'shock_number'], as_index=False)['post_shock_delta']
        .mean()
    )
    for g in grp_order:
        sub = d_mouse.loc[d_mouse['group'] == g]
        if sub.empty:
            continue
        x = []
        y = []
        e = []
        for sn in sorted(sub['shock_number'].unique()):
            vals = sub.loc[sub['shock_number'] == sn, 'post_shock_delta'].astype(float).dropna().values
            if vals.size == 0:
                continue
            x.append(sn)
            y.append(float(np.nanmean(vals)))
            e.append(float(np.nanstd(vals) / np.sqrt(max(vals.size, 1))))
        if x:
            ax_D.errorbar(x, y, yerr=e, color=GROUP_COLOURS.get(g, 'gray'), marker='o', lw=style['fit_linewidth_mouse'], label=g)
    ax_D.set_title('D: Post-shock response index')
    ax_D.set_xlabel('Shock number')
    ax_D.set_ylabel('Delta amplitude rate')
    ax_D.set_xticks([1, 2, 3, 4, 5])
    _style_trace_axis(ax_D)
    _style_legend(ax_D.legend(frameon=False, fontsize=style['legend_fontsize']))

    # Panel E1/E2
    ax_E1 = fig.add_subplot(gs[1, 0])
    ax_E2 = fig.add_subplot(gs[1, 1])
    e1_df = (
        trial_df.loc[trial_df['shock_number'].isin([1, 2, 3]), ['mouse', 'group', 'post_shock_delta']]
        .dropna()
        .groupby(['mouse', 'group'], as_index=False)['post_shock_delta']
        .mean()
    )
    e2_df = (
        trial_df.loc[trial_df['shock_number'].isin([4, 5]), ['mouse', 'group', 'post_shock_delta']]
        .dropna()
        .groupby(['mouse', 'group'], as_index=False)['post_shock_delta']
        .mean()
    )
    e1_vals = {g: e1_df.loc[e1_df['group'] == g, 'post_shock_delta'].astype(float).dropna().values for g in grp_order}
    e2_vals = {g: e2_df.loc[e2_df['group'] == g, 'post_shock_delta'].astype(float).dropna().values for g in grp_order}

    for ax_e, e_df, e_vals, title in [
        (ax_E1, e1_df, e1_vals, 'E1: Early post-shock (1-3)'),
        (ax_E2, e2_df, e2_vals, 'E2: Late post-shock (4-5)'),
    ]:
        pairs = _welch_holm_pairs(e_vals)
        _draw_group_box_strip(ax_e, e_df, 'post_shock_delta', grp_order, title, 'Delta amplitude rate', pairs=pairs)

    # Panel F1: trial-level scatter
    ax_F1 = fig.add_subplot(gs[1, 2])
    f1_df = trial_df[['mouse', 'group', 'shock_number', 'post_shock_delta', 'freeze_percent']].dropna().copy()
    for g in grp_order:
        sub = f1_df.loc[f1_df['group'] == g]
        if sub.empty:
            continue
        col = GROUP_COLOURS.get(g, 'gray')
        ax_F1.scatter(sub['post_shock_delta'], sub['freeze_percent'], color=col, s=style['scatter_size_trial'], alpha=style['scatter_alpha'], edgecolors=style['scatter_edgecolors'])
        if sub.shape[0] >= 2:
            x = sub['post_shock_delta'].to_numpy(dtype=float)
            y = sub['freeze_percent'].to_numpy(dtype=float)
            coeff = np.polyfit(x, y, 1)
            xx = np.linspace(np.nanmin(x), np.nanmax(x), 100)
            yy = coeff[0] * xx + coeff[1]
            ax_F1.plot(xx, yy, color=col, lw=style['fit_linewidth_trial'])
    ax_F1.set_title('F1: Trial-level calcium vs freezing')
    ax_F1.set_xlabel('Post-shock delta')
    ax_F1.set_ylabel('Subsequent freezing (%)')
    _style_trace_axis(ax_F1)

    # Panel F2: mouse-level late summary scatter
    ax_F2 = fig.add_subplot(gs[1, 3])
    f2_df = (
        trial_df.loc[trial_df['shock_number'].isin([4, 5]), ['mouse', 'group', 'post_shock_delta', 'freeze_percent']]
        .dropna()
        .groupby(['mouse', 'group'], as_index=False)
        .mean(numeric_only=True)
    )
    for g in grp_order:
        sub = f2_df.loc[f2_df['group'] == g]
        if sub.empty:
            continue
        col = GROUP_COLOURS.get(g, 'gray')
        ax_F2.scatter(sub['post_shock_delta'], sub['freeze_percent'], color=col, s=style['scatter_size_mouse'], alpha=style['scatter_alpha'], edgecolors=style['scatter_edgecolors'])
        if sub.shape[0] >= 2:
            x = sub['post_shock_delta'].to_numpy(dtype=float)
            y = sub['freeze_percent'].to_numpy(dtype=float)
            coeff = np.polyfit(x, y, 1)
            xx = np.linspace(np.nanmin(x), np.nanmax(x), 100)
            yy = coeff[0] * xx + coeff[1]
            ax_F2.plot(xx, yy, color=col, lw=style['fit_linewidth_mouse'])
    ax_F2.set_title('F2: Late-summary calcium vs freezing')
    ax_F2.set_xlabel('Late post-shock delta (4-5)')
    ax_F2.set_ylabel('Late freezing (%)')
    _style_trace_axis(ax_F2)

    # Stats exports
    with open(os.path.join(stats_dir, 'panel_B_pairwise.txt'), 'w', encoding='utf-8') as f:
        for row in b_pairs:
            f.write(
                f"{row['a']} vs {row['b']}\t"
                f"mean_diff={row['mean_diff']:.6g}\t"
                f"p_raw={row['p_raw']:.6g}\t"
                f"p_holm={row['p_holm']:.6g}\t"
                f"n_a={row['n_a']}\tn_b={row['n_b']}\n"
            )

    e1_pairs = _welch_holm_pairs(e1_vals)
    e2_pairs = _welch_holm_pairs(e2_vals)
    with open(os.path.join(stats_dir, 'panel_E_pairwise.txt'), 'w', encoding='utf-8') as f:
        f.write('[E1] Early shocks 1-3\n')
        for row in e1_pairs:
            f.write(
                f"{row['a']} vs {row['b']}\t"
                f"mean_diff={row['mean_diff']:.6g}\t"
                f"p_raw={row['p_raw']:.6g}\t"
                f"p_holm={row['p_holm']:.6g}\t"
                f"n_a={row['n_a']}\tn_b={row['n_b']}\n"
            )
        f.write('\n[E2] Late shocks 4-5\n')
        for row in e2_pairs:
            f.write(
                f"{row['a']} vs {row['b']}\t"
                f"mean_diff={row['mean_diff']:.6g}\t"
                f"p_raw={row['p_raw']:.6g}\t"
                f"p_holm={row['p_holm']:.6g}\t"
                f"n_a={row['n_a']}\tn_b={row['n_b']}\n"
            )

    d_model = _fit_mixed_model_or_fallback(
        d_mouse[['mouse', 'group', 'shock_number', 'post_shock_delta']].dropna(),
        'post_shock_delta ~ C(group) * C(shock_number)',
        group_col='mouse',
    )
    with open(os.path.join(stats_dir, 'panel_D_mixed_model.txt'), 'w', encoding='utf-8') as f:
        f.write(d_model['summary'] + '\n')

    f1_model = _fit_mixed_model_or_fallback(
        f1_df[['mouse', 'group', 'shock_number', 'post_shock_delta', 'freeze_percent']].dropna(),
        'freeze_percent ~ C(group) * post_shock_delta + C(shock_number)',
        group_col='mouse',
    )
    with open(os.path.join(stats_dir, 'panel_F1_mixed_model.txt'), 'w', encoding='utf-8') as f:
        f.write(f1_model['summary'] + '\n')

    r_p, p_p, n_p = _safe_corr(f2_df['post_shock_delta'], f2_df['freeze_percent'], method='pearson')
    r_s, p_s, n_s = _safe_corr(f2_df['post_shock_delta'], f2_df['freeze_percent'], method='spearman')
    with open(os.path.join(stats_dir, 'panel_F2_mouse_level_correlation.txt'), 'w', encoding='utf-8') as f:
        f.write(f'Pearson: r={r_p:.6g}, p={p_p:.6g}, n={n_p}\n')
        f.write(f'Spearman: r={r_s:.6g}, p={p_s:.6g}, n={n_s}\n')
        for g in grp_order:
            sub = f2_df.loc[f2_df['group'] == g]
            rg, pg, ng = _safe_corr(sub['post_shock_delta'], sub['freeze_percent'], method='pearson')
            f.write(f'{g} Pearson: r={rg:.6g}, p={pg:.6g}, n={ng}\n')

    plt.tight_layout()
    png_path = os.path.join(save_dir, 'tfc_conditioning_summary_panels_B_to_F2.png')
    pdf_path = os.path.join(save_dir, 'tfc_conditioning_summary_panels_B_to_F2.pdf')
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path, dpi=300)
    
    if paper_dir is not None:
        os.makedirs(paper_dir, exist_ok=True)
        compact_style = dict(TFC_CONDITIONING_PANEL_STYLE_COMPACT)
        _apply_compact_style_to_existing_figure(fig, compact_style)
        fig.tight_layout()
        paper_png = os.path.join(paper_dir, 'tfc_conditioning_summary_panels_B_to_F2.png')
        paper_pdf = os.path.join(paper_dir, 'tfc_conditioning_summary_panels_B_to_F2.pdf')
        fig.savefig(paper_png, dpi=300)
        fig.savefig(paper_pdf, dpi=300)
    
    if auto_close:
        plt.close(fig)

    # Save individual panel figures
    # Panel B
    fig_b = plt.figure(figsize=style['single_panel_figsize'])
    ax_b = fig_b.add_subplot(111)
    b_plot_df = baseline_df[['group', 'pre_cs_value']].dropna().copy()
    b_vals_i = {
        g: baseline_df.loc[baseline_df['group'] == g, 'pre_cs_value'].astype(float).dropna().values
        for g in grp_order
    }
    b_pairs_i = _welch_holm_pairs(b_vals_i)
    _draw_group_box_strip(ax_b, b_plot_df, 'pre_cs_value', grp_order, 'B: Pre-CS baseline', 'Transient amplitude rate', pairs=b_pairs_i)
    fig_b.tight_layout()
    fig_b.savefig(os.path.join(save_dir, 'panel_B_pre_cs_baseline.png'), dpi=300, bbox_inches='tight')
    fig_b.savefig(os.path.join(save_dir, 'panel_B_pre_cs_baseline.pdf'), dpi=300, bbox_inches='tight')
    plt.close(fig_b)

    # Panel C - Early
    fig_ce = plt.figure(figsize=style['trace_panel_figsize'])
    ax_ce = fig_ce.add_subplot(111)
    for g in grp_order:
        col = GROUP_COLOURS.get(g, 'gray')
        mouse_early = []
        for mouse in sorted(TFC_cond.keys()):
            if mouse_groups.get(mouse) != g:
                continue
            traces_mouse = [row['trace_bs'] for row in peri_traces if row['mouse'] == mouse]
            shocks_mouse = [row['shock_number'] for row in peri_traces if row['mouse'] == mouse]
            if not traces_mouse:
                continue
            early = [tr for tr, sn in zip(traces_mouse, shocks_mouse) if sn in (1, 2)]
            if early:
                mouse_early.append(np.nanmean(np.vstack(early), axis=0))
        if mouse_early:
            mat = np.vstack(mouse_early)
            mean = np.nanmean(mat, axis=0)
            sem = np.nanstd(mat, axis=0) / np.sqrt(max(mat.shape[0], 1))
            ax_ce.plot(tvec, mean, color=col, lw=style['trace_linewidth'], alpha=style['trace_line_alpha'], label=g)
            ax_ce.fill_between(tvec, mean - sem, mean + sem, color=col, alpha=style['trace_alpha'])
    ax_ce.axvline(0.0, color='#666666', ls='--', lw=1.0)
    ax_ce.set_xlim(-peri_pre_s, peri_post_s)
    ax_ce.set_xlabel('Time from shock (s)')
    ax_ce.set_ylabel('Δ activity (a.u./s)')
    ax_ce.set_title('C: Early shocks (1-2)')
    _style_trace_axis(ax_ce)
    _style_legend(ax_ce.legend(frameon=False))
    fig_ce.tight_layout()
    fig_ce.savefig(os.path.join(save_dir, 'panel_C_early_shocks_1-2.png'), dpi=300, bbox_inches='tight')
    fig_ce.savefig(os.path.join(save_dir, 'panel_C_early_shocks_1-2.pdf'), dpi=300, bbox_inches='tight')
    plt.close(fig_ce)

    # Panel C - Late
    fig_cl = plt.figure(figsize=style['trace_panel_figsize'])
    ax_cl = fig_cl.add_subplot(111)
    for g in grp_order:
        col = GROUP_COLOURS.get(g, 'gray')
        mouse_late = []
        for mouse in sorted(TFC_cond.keys()):
            if mouse_groups.get(mouse) != g:
                continue
            traces_mouse = [row['trace_bs'] for row in peri_traces if row['mouse'] == mouse]
            shocks_mouse = [row['shock_number'] for row in peri_traces if row['mouse'] == mouse]
            if not traces_mouse:
                continue
            late = [tr for tr, sn in zip(traces_mouse, shocks_mouse) if sn in (4, 5)]
            if late:
                mouse_late.append(np.nanmean(np.vstack(late), axis=0))
        if mouse_late:
            mat = np.vstack(mouse_late)
            mean = np.nanmean(mat, axis=0)
            sem = np.nanstd(mat, axis=0) / np.sqrt(max(mat.shape[0], 1))
            ax_cl.plot(tvec, mean, color=col, lw=style['trace_linewidth'], alpha=style['trace_line_alpha'], label=g)
            ax_cl.fill_between(tvec, mean - sem, mean + sem, color=col, alpha=style['trace_alpha'])
    ax_cl.axvline(0.0, color='#666666', ls='--', lw=1.0)
    ax_cl.set_xlim(-peri_pre_s, peri_post_s)
    ax_cl.set_xlabel('Time from shock (s)')
    ax_cl.set_ylabel('Δ activity (a.u./s)')
    ax_cl.set_title('C: Late shocks (4-5)')
    _style_trace_axis(ax_cl)
    _style_legend(ax_cl.legend(frameon=False))
    fig_cl.tight_layout()
    fig_cl.savefig(os.path.join(save_dir, 'panel_C_late_shocks_4-5.png'), dpi=300, bbox_inches='tight')
    fig_cl.savefig(os.path.join(save_dir, 'panel_C_late_shocks_4-5.pdf'), dpi=300, bbox_inches='tight')
    plt.close(fig_cl)

    # Panel D
    fig_d = plt.figure(figsize=style['trace_panel_figsize'])
    ax_d = fig_d.add_subplot(111)
    d_mouse = (
        trial_df[['mouse', 'group', 'shock_number', 'post_shock_delta']].dropna()
        .groupby(['mouse', 'group', 'shock_number'], as_index=False)['post_shock_delta']
        .mean()
    )
    for g in grp_order:
        sub = d_mouse.loc[d_mouse['group'] == g]
        if sub.empty:
            continue
        x = []
        y = []
        e = []
        for sn in sorted(sub['shock_number'].unique()):
            vals = sub.loc[sub['shock_number'] == sn, 'post_shock_delta'].astype(float).dropna().values
            if vals.size == 0:
                continue
            x.append(sn)
            y.append(float(np.nanmean(vals)))
            e.append(float(np.nanstd(vals) / np.sqrt(max(vals.size, 1))))
        if x:
            ax_d.errorbar(x, y, yerr=e, color=GROUP_COLOURS.get(g, 'gray'), marker='o', lw=style['fit_linewidth_mouse'], label=g)
    ax_d.set_title('D: Post-shock response index')
    ax_d.set_xlabel('Shock number')
    ax_d.set_ylabel('Delta amplitude rate')
    ax_d.set_xticks([1, 2, 3, 4, 5])
    _style_trace_axis(ax_d)
    _style_legend(ax_d.legend(frameon=False, fontsize=style['legend_fontsize']))
    fig_d.tight_layout()
    fig_d.savefig(os.path.join(save_dir, 'panel_D_post_shock_index.png'), dpi=300, bbox_inches='tight')
    fig_d.savefig(os.path.join(save_dir, 'panel_D_post_shock_index.pdf'), dpi=300, bbox_inches='tight')
    plt.close(fig_d)

    # Panel E1
    fig_e1 = plt.figure(figsize=style['single_panel_figsize'])
    ax_e1 = fig_e1.add_subplot(111)
    e1_df_i = (
        trial_df.loc[trial_df['shock_number'].isin([1, 2, 3]), ['mouse', 'group', 'post_shock_delta']]
        .dropna()
        .groupby(['mouse', 'group'], as_index=False)['post_shock_delta']
        .mean()
    )
    e1_vals_i = {g: e1_df_i.loc[e1_df_i['group'] == g, 'post_shock_delta'].astype(float).dropna().values for g in grp_order}
    e1_pairs_i = _welch_holm_pairs(e1_vals_i)
    _draw_group_box_strip(ax_e1, e1_df_i, 'post_shock_delta', grp_order, 'E1: Early post-shock (1-3)', 'Delta amplitude rate', pairs=e1_pairs_i)
    fig_e1.tight_layout()
    fig_e1.savefig(os.path.join(save_dir, 'panel_E1_early_post_shock_1-3.png'), dpi=300, bbox_inches='tight')
    fig_e1.savefig(os.path.join(save_dir, 'panel_E1_early_post_shock_1-3.pdf'), dpi=300, bbox_inches='tight')
    plt.close(fig_e1)

    # Panel E2
    fig_e2 = plt.figure(figsize=style['single_panel_figsize'])
    ax_e2 = fig_e2.add_subplot(111)
    e2_df_i = (
        trial_df.loc[trial_df['shock_number'].isin([4, 5]), ['mouse', 'group', 'post_shock_delta']]
        .dropna()
        .groupby(['mouse', 'group'], as_index=False)['post_shock_delta']
        .mean()
    )
    e2_vals_i = {g: e2_df_i.loc[e2_df_i['group'] == g, 'post_shock_delta'].astype(float).dropna().values for g in grp_order}
    e2_pairs_i = _welch_holm_pairs(e2_vals_i)
    _draw_group_box_strip(ax_e2, e2_df_i, 'post_shock_delta', grp_order, 'E2: Late post-shock (4-5)', 'Delta amplitude rate', pairs=e2_pairs_i)
    fig_e2.tight_layout()
    fig_e2.savefig(os.path.join(save_dir, 'panel_E2_late_post_shock_4-5.png'), dpi=300, bbox_inches='tight')
    fig_e2.savefig(os.path.join(save_dir, 'panel_E2_late_post_shock_4-5.pdf'), dpi=300, bbox_inches='tight')
    plt.close(fig_e2)

    # Panel F1
    fig_f1 = plt.figure(figsize=style['trace_panel_figsize'])
    ax_f1 = fig_f1.add_subplot(111)
    f1_df_i = trial_df[['mouse', 'group', 'shock_number', 'post_shock_delta', 'freeze_percent']].dropna().copy()
    for g in grp_order:
        sub = f1_df_i.loc[f1_df_i['group'] == g]
        if sub.empty:
            continue
        col = GROUP_COLOURS.get(g, 'gray')
        ax_f1.scatter(sub['post_shock_delta'], sub['freeze_percent'], color=col, s=style['scatter_size_trial'], alpha=style['scatter_alpha'], edgecolors=style['scatter_edgecolors'], label=g)
        if sub.shape[0] >= 2:
            x = sub['post_shock_delta'].to_numpy(dtype=float)
            y = sub['freeze_percent'].to_numpy(dtype=float)
            coeff = np.polyfit(x, y, 1)
            xx = np.linspace(np.nanmin(x), np.nanmax(x), 100)
            yy = coeff[0] * xx + coeff[1]
            ax_f1.plot(xx, yy, color=col, lw=style['fit_linewidth_trial'])
    ax_f1.set_title('F1: Trial-level calcium vs freezing')
    ax_f1.set_xlabel('Post-shock delta')
    ax_f1.set_ylabel('Subsequent freezing (%)')
    _style_trace_axis(ax_f1)
    _style_legend(ax_f1.legend(frameon=False, fontsize=style['legend_fontsize']))
    fig_f1.tight_layout()
    fig_f1.savefig(os.path.join(save_dir, 'panel_F1_trial_level_calcium_vs_freezing.png'), dpi=300, bbox_inches='tight')
    fig_f1.savefig(os.path.join(save_dir, 'panel_F1_trial_level_calcium_vs_freezing.pdf'), dpi=300, bbox_inches='tight')
    plt.close(fig_f1)

    # Panel F2
    fig_f2 = plt.figure(figsize=style['trace_panel_figsize'])
    ax_f2 = fig_f2.add_subplot(111)
    f2_df_i = (
        trial_df.loc[trial_df['shock_number'].isin([4, 5]), ['mouse', 'group', 'post_shock_delta', 'freeze_percent']]
        .dropna()
        .groupby(['mouse', 'group'], as_index=False)
        .mean(numeric_only=True)
    )
    for g in grp_order:
        sub = f2_df_i.loc[f2_df_i['group'] == g]
        if sub.empty:
            continue
        col = GROUP_COLOURS.get(g, 'gray')
        ax_f2.scatter(sub['post_shock_delta'], sub['freeze_percent'], color=col, s=style['scatter_size_mouse'], alpha=style['scatter_alpha'], edgecolors=style['scatter_edgecolors'], label=g)
        if sub.shape[0] >= 2:
            x = sub['post_shock_delta'].to_numpy(dtype=float)
            y = sub['freeze_percent'].to_numpy(dtype=float)
            coeff = np.polyfit(x, y, 1)
            xx = np.linspace(np.nanmin(x), np.nanmax(x), 100)
            yy = coeff[0] * xx + coeff[1]
            ax_f2.plot(xx, yy, color=col, lw=style['fit_linewidth_mouse'])
    ax_f2.set_title('F2: Late-summary calcium vs freezing')
    ax_f2.set_xlabel('Late post-shock delta (4-5)')
    ax_f2.set_ylabel('Late freezing (%)')
    _style_trace_axis(ax_f2)
    _style_legend(ax_f2.legend(frameon=False, fontsize=style['legend_fontsize']))
    fig_f2.tight_layout()
    fig_f2.savefig(os.path.join(save_dir, 'panel_F2_late_summary_calcium_vs_freezing.png'), dpi=300, bbox_inches='tight')
    fig_f2.savefig(os.path.join(save_dir, 'panel_F2_late_summary_calcium_vs_freezing.pdf'), dpi=300, bbox_inches='tight')
    plt.close(fig_f2)

    return {
        'trial_df': trial_df,
        'baseline_df': baseline_df,
        'mouse_late_df': f2_df,
        'save_dir': save_dir,
        'stats_dir': stats_dir,
    }
