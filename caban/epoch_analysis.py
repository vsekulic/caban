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
from sklearn.covariance import LedoitWolf
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity
from scipy.spatial.distance import mahalanobis as scipy_mahalanobis
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scipy.stats

from caban.utilities import MINISCOPE_FPS, get_spikes_in_period

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

EPOCH_NAMES = ['pre_tone', 'tone', 'trace', 'peri_shock', 'shock', 'post_shock']

EPOCH_COLOURS = {
    'pre_tone':   '#888888',
    'tone':       '#1f77b4',
    'trace':      '#ff7f0e',
    'peri_shock': '#9467bd',
    'shock':      '#d62728',
    'post_shock': '#2ca02c',
}

GROUP_COLOURS = {
    'hM3D':    'r',
    'hM4D':    'b',
    'mCherry': 'k',
}

PVALS = [0.05, 0.01, 0.001]

# ─────────────────────────────────────────────────────────────────────────────
# Epoch frame extraction
# ─────────────────────────────────────────────────────────────────────────────

def get_epoch_frames(session, epoch_name, trial_idx,
                     peri_shock_pre_s=10.0, peri_shock_post_s=10.0,
                     pre_tone_duration_s=35.0):
    """
    Return (onset_frame, offset_frame) for a given epoch and trial.

    The offset is exclusive (Python-slice style), so valid frames are
    [onset, offset).  Returns None if the epoch cannot be computed
    (e.g. onset would be before the start of the recording).

    Parameters
    ----------
    session          : TraceFearCondSession
    epoch_name       : str  - one of EPOCH_NAMES
    trial_idx        : int  - 0-based trial index
    peri_shock_pre_s : float - seconds before shock onset for peri_shock epoch
    peri_shock_post_s: float - seconds after shock onset for peri_shock epoch
    pre_tone_duration_s : float - duration of pre-tone baseline window in seconds
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

    elif epoch_name == 'pre_tone':
        offset = session.tone_onsets[trial_idx]
        onset  = offset - int(round(pre_tone_duration_s * fps))
        if onset < 0:
            return None   # not enough baseline before first tone

    elif epoch_name == 'post_shock':
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
        VI = np.linalg.inv(lw.covariance_)
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
        print(f'[epoch_analysis] No results found for metric={metric}, '
              f'analysis_type={analysis_type}')
        return

    if keys is None:
        keys = list(first_mouse_result[analysis_type][metric].keys())

    n_keys = len(keys)
    if n_keys == 0:
        return

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


print("loaded")