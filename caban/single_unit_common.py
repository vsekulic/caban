"""Shared plotting/statistics helpers for the single-unit response analyses.

The four single-unit analyses — event-locked responsiveness, per-cell activity
distributions, population coupling, and freezing-tuned cells — all compare the
three DREADD groups (hM3D/Exc, hM4D/Inh, mCherry/Ctl) at the single-cell level
instead of collapsing each recording to one per-mouse scalar (the design of the
proportional-activity metrics). This module centralises the pieces they share so
none of them re-implements group styling, ECDF panels, the cell-nested-in-mouse
mixed model, or false-discovery correction.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from scipy.stats import norm
from scipy.ndimage import gaussian_filter1d

from caban.utilities import MINISCOPE_FPS

# Canonical group coding, matching caban.analysis (red/blue/grey, Exc/Inh/Ctl).
GROUP_ORDER = ['hM3D', 'hM4D', 'mCherry']
GROUP_LABELS = {'hM3D': 'Exc', 'hM4D': 'Inh', 'mCherry': 'Ctl'}
GROUP_COLOURS = {'hM3D': 'r', 'hM4D': 'b', 'mCherry': '0.25'}


def bracket_ylim(values_per_group, n_cols):
    """(bottom, top) y-limits with headroom above the data for up to ~3 stacked
    significance brackets. Handles negative values (e.g. correlations). Computed
    across all panels so a shared-y row stays consistent."""
    parts = [np.asarray(values_per_group[g])[:, :n_cols].ravel() for g in values_per_group
             if np.asarray(values_per_group[g]).size]
    vals = np.concatenate(parts) if parts else np.array([])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (0.0, 1.0)
    vmax = float(np.max(vals))
    vmin = float(np.min(vals))
    span = (vmax - vmin) if vmax > vmin else (abs(vmax) or 1.0)
    top = vmax + 0.8 * span + 0.02
    bottom = min(0.0, vmin - 0.05 * span)
    return (bottom, top)


def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def write_text(path, text):
    with open(path, 'w') as fh:
        fh.write(text)


def fdr_correct(pvalues, alpha=0.05):
    """Benjamini-Hochberg FDR. Returns (reject_bool_array, qvalues). NaN p-values
    are left as NaN and never marked significant."""
    pvalues = np.asarray(pvalues, dtype=float)
    reject = np.zeros(pvalues.shape, dtype=bool)
    qvals = np.full(pvalues.shape, np.nan)
    finite = np.isfinite(pvalues)
    if finite.sum() == 0:
        return reject, qvals
    rej, q, _, _ = multipletests(pvalues[finite], alpha=alpha, method='fdr_bh')
    reject[finite] = rej
    qvals[finite] = q
    return reject, qvals


def get_mapping_signal(session, mapping, signal_attr='C', with_crossreg=None):
    """Activity matrix (cells x frames) for a mapping's cell subset.

    Defaults to the denoised calcium trace C, whose transients are far denser than
    the rare, hard-thresholded deconvolved S spikes (event-locked / freezing
    modulation is detectable on C but essentially absent on S). C and S share
    cell-row order, so get_S_indeces indices apply to either."""
    signal = getattr(session, signal_attr)
    if mapping == 'full':
        return signal
    _, _, _, cell_ids = session.get_S_mapping(mapping, with_crossreg=with_crossreg)
    row_indices = session.get_S_indeces(cell_ids)
    return signal[row_indices]


def shuffle_mask_contrast(S, target_mask, reference_mask, n_shuffles=1000, seed=0,
                          min_shift_frames=None, block=200):
    """Per-cell frame-mask contrast with a circular-shift shuffle null.

    S is the activity matrix (cells x frames), the denoised calcium trace C by
    default. For each cell the statistic is (mean activity over target frames)
    minus (mean over reference frames). The null circularly shifts the whole trace
    by random offsets (kept at least ``min_shift_frames`` from zero) and recomputes
    the statistic; because shifting the trace and holding the mask fixed equals
    holding the trace and rolling the weight vector, the whole null is a single
    matmul S @ W (blocked to cap memory).

    Significance is a z-scored (normal-approximation) p-value: z = (observed −
    null_mean) / null_std, p = 2·Φ(−|z|). This is used instead of the raw empirical
    shuffle p-value because the empirical p is floored at 1/(n_shuffles+1), which is
    far too coarse for Benjamini-Hochberg FDR across hundreds of cells (rejecting the
    top cell needs p ≲ alpha/n_cells) — so an affordable number of shuffles would
    otherwise reject nothing. The null statistic is a weighted sum over many frames
    and is approximately normal by the CLT. The empirical two-sided p is also
    returned as 'pval_empirical' for reference. p-values are FDR-corrected.

    S              : (n_cells, n_frames) deconvolved activity.
    target_mask,
    reference_mask : (n_frames,) 0/1 frame masks.

    Returns dict: 'observed', 'z', 'pval', 'pval_empirical', 'direction',
    'responsive' (arrays over cells).
    """
    n_cells, n_frames = S.shape
    n_t = float(np.sum(target_mask))
    n_r = float(np.sum(reference_mask))
    if n_t == 0 or n_r == 0:
        raise RuntimeError('shuffle_mask_contrast: target or reference mask is empty.')
    w = np.asarray(target_mask, dtype=float) / n_t - np.asarray(reference_mask, dtype=float) / n_r

    observed = S @ w
    abs_obs = np.abs(observed)[:, None]

    rng = np.random.default_rng(seed)
    if min_shift_frames is None:
        min_shift_frames = int(round(5 * MINISCOPE_FPS))
    shifts = rng.integers(min_shift_frames, n_frames - min_shift_frames, size=n_shuffles)

    null_sum = np.zeros(n_cells, dtype=float)
    null_sumsq = np.zeros(n_cells, dtype=float)
    null_ge = np.zeros(n_cells, dtype=float)
    for start in range(0, n_shuffles, block):
        blk = shifts[start:start + block]
        W = np.stack([np.roll(w, int(s)) for s in blk], axis=1)   # (n_frames, b)
        null_block = S @ W                                         # (n_cells, b)
        null_sum += null_block.sum(axis=1)
        null_sumsq += (null_block ** 2).sum(axis=1)
        null_ge += (np.abs(null_block) >= abs_obs).sum(axis=1)

    return _pval_from_null(observed, null_sum, null_sumsq, null_ge, n_shuffles)


def _pval_from_null(observed, null_sum, null_sumsq, null_ge, n_shuffles):
    """z-scored (normal-approximation) two-sided p from streamed null moments, with
    the empirical p as reference. Shared by the mask- and peak-based tests — see the
    note in shuffle_mask_contrast on why the z-score (not the empirical fraction) is
    used for FDR."""
    null_mean = null_sum / n_shuffles
    null_var = np.maximum(null_sumsq / n_shuffles - null_mean ** 2, 0.0)
    null_std = np.sqrt(null_var)
    z = np.zeros_like(observed)
    nz = null_std > 0
    z[nz] = (observed[nz] - null_mean[nz]) / null_std[nz]
    pval = 2.0 * norm.sf(np.abs(z))
    pval_empirical = (1.0 + null_ge) / (1.0 + n_shuffles)
    responsive, _ = fdr_correct(pval)
    return {'observed': observed, 'z': z, 'pval': pval, 'pval_empirical': pval_empirical,
            'direction': np.sign(observed), 'responsive': responsive}


def shuffle_peak_contrast(sig, epoch_windows, baseline_mask, smooth_sigma_frames=20.0,
                          n_shuffles=1000, seed=0, min_shift_frames=None):
    """Per-cell PEAK response within an epoch vs baseline, with a circular-shift null.

    For each cell the statistic is (mean over trials of the peak of the smoothed trace
    within that trial's epoch window) minus (mean over the baseline frames). Unlike the
    mean-over-epoch contrast, this is sensitive to a phasic response occurring anywhere
    in the window (e.g. a tone-onset transient that a 20 s mean would dilute). The null
    circularly shifts the whole trace by one global offset per shuffle and recomputes
    the same statistic; p-values are z-scored and FDR-corrected (see shuffle_mask_contrast).

    sig            : (n_cells, n_frames) activity (denoised C by default).
    epoch_windows  : list of (onset, offset) frame pairs, one per trial.
    baseline_mask  : (n_frames,) 0/1 baseline frame mask.
    smooth_sigma_frames : gaussian smoothing (frames) applied before peak detection.
    """
    n_cells, n_frames = sig.shape
    n_b = float(np.sum(baseline_mask))
    if n_b == 0 or not epoch_windows:
        raise RuntimeError('shuffle_peak_contrast: empty baseline or no epoch windows.')

    sig_s = gaussian_filter1d(sig, sigma=smooth_sigma_frames, axis=1) if smooth_sigma_frames > 0 else sig
    base_frames = np.flatnonzero(baseline_mask)

    def _statistic(shift):
        # Peak term: mean over trials of max over each (shifted) epoch window.
        peaks = np.zeros(n_cells)
        for onset, offset in epoch_windows:
            idx = (np.arange(onset, offset) - shift) % n_frames
            peaks += sig_s.take(idx, axis=1).max(axis=1)
        peaks /= len(epoch_windows)
        base = sig_s.take((base_frames - shift) % n_frames, axis=1).mean(axis=1)
        return peaks - base

    observed = _statistic(0)
    abs_obs = np.abs(observed)

    rng = np.random.default_rng(seed)
    if min_shift_frames is None:
        min_shift_frames = int(round(5 * MINISCOPE_FPS))
    shifts = rng.integers(min_shift_frames, n_frames - min_shift_frames, size=n_shuffles)

    null_sum = np.zeros(n_cells)
    null_sumsq = np.zeros(n_cells)
    null_ge = np.zeros(n_cells)
    for s in shifts:
        stat = _statistic(int(s))
        null_sum += stat
        null_sumsq += stat ** 2
        null_ge += (np.abs(stat) >= abs_obs)

    return _pval_from_null(observed, null_sum, null_sumsq, null_ge, n_shuffles)


def _ecdf(values):
    values = np.sort(np.asarray(values, dtype=float))
    y = np.arange(1, len(values) + 1) / len(values)
    return values, y


def ecdf_panel(ax, per_cell_by_group, per_mouse_by_group=None, xlabel='', title=''):
    """Draw per-group pooled per-cell ECDFs (bold) with optional per-mouse ECDFs
    (thin) on one axis.

    per_cell_by_group  : dict group -> 1-D array of per-cell values (pooled).
    per_mouse_by_group : dict group -> {mouse: 1-D array} for the thin lines.
    """
    for group in GROUP_ORDER:
        colour = GROUP_COLOURS[group]
        if per_mouse_by_group is not None:
            for mouse_vals in per_mouse_by_group.get(group, {}).values():
                mouse_vals = np.asarray(mouse_vals, dtype=float)
                mouse_vals = mouse_vals[np.isfinite(mouse_vals)]
                if mouse_vals.size == 0:
                    continue
                mx, my = _ecdf(mouse_vals)
                ax.step(mx, my, where='post', color=colour, alpha=0.25, linewidth=0.7)
        vals = np.asarray(per_cell_by_group.get(group, []), dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        x, y = _ecdf(vals)
        ax.step(x, y, where='post', color=colour, linewidth=2.0,
                label=f'{GROUP_LABELS[group]} (n={vals.size})')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Cumulative fraction of cells')
    ax.set_ylim(0, 1)
    if title:
        ax.set_title(title, size='medium')
    ax.legend(fontsize=7, loc='lower right', frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def build_cell_records(per_cell_by_group_mouse, value_name='value'):
    """Flatten {group: {mouse: 1-D array}} into a tidy DataFrame with columns
    [value_name, 'group', 'mouse'] for mixed-model fitting."""
    rows = []
    for group, per_mouse in per_cell_by_group_mouse.items():
        for mouse, vals in per_mouse.items():
            for v in np.asarray(vals, dtype=float):
                rows.append({value_name: v, 'group': group, 'mouse': mouse})
    return pd.DataFrame(rows)


def fit_group_mixed_model(df, value_col='value', reference='mCherry'):
    """Fit ``value ~ C(group)`` with a per-mouse random intercept (cell nested in
    mouse), so single-cell n is honoured without pseudoreplication.

    Returns (summary_text, method_used). Falls back, with an explicit note in the
    returned text, to mouse-clustered OLS if the mixed model does not converge —
    a documented statistical fallback, not a silent error swallow.
    """
    df = df[np.isfinite(df[value_col])].copy()
    if df['group'].nunique() < 2 or len(df) < 6:
        return (f'Insufficient data for group model: '
                f'{len(df)} cells across {df["group"].nunique()} groups.\n'), 'none'

    df['group'] = pd.Categorical(df['group'],
                                 categories=[reference] + [g for g in GROUP_ORDER if g != reference])
    formula = f'{value_col} ~ C(group, Treatment(reference="{reference}"))'
    try:
        model = smf.mixedlm(formula, data=df, groups=df['mouse'])
        result = model.fit(reml=True, method='lbfgs')
        text = (f'Linear mixed model (cell nested in mouse), reference group = {reference}\n'
                f'Formula: {formula} + (1 | mouse)\n'
                f'N cells = {len(df)}, N mice = {df["mouse"].nunique()}\n\n'
                f'{result.summary()}\n')
        return text, 'mixedlm'
    except Exception as exc:  # documented fallback: cluster-robust OLS
        ols = smf.ols(formula, data=df).fit(
            cov_type='cluster', cov_kwds={'groups': df['mouse']})
        text = (f'Mixed model failed to converge ({exc}); using mouse-clustered OLS.\n'
                f'Formula: {formula}, cluster = mouse\n'
                f'N cells = {len(df)}, N mice = {df["mouse"].nunique()}\n\n'
                f'{ols.summary()}\n')
        return text, 'clustered_ols'


def save_fig(fig, path_png, dpi=300):
    """Save a matched PNG + SVG pair (paper convention)."""
    fig.savefig(path_png, format='png', dpi=dpi)
    fig.savefig(os.path.splitext(path_png)[0] + '.svg', format='svg')
