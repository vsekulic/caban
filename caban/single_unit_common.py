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
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from scipy.stats import norm, t as t_dist, f as f_dist
from scipy.ndimage import gaussian_filter1d

from caban.utilities import MINISCOPE_FPS, get_pval_str

# Canonical group coding, matching caban.analysis (red/blue/grey, Exc/Inh/Ctl).
GROUP_ORDER = ['hM3D', 'hM4D', 'mCherry']
GROUP_LABELS = {'hM3D': 'Exc', 'hM4D': 'Inh', 'mCherry': 'Ctl'}
GROUP_COLOURS = {'hM3D': 'r', 'hM4D': 'b', 'mCherry': '0.25'}

# Paper DREADD DISPLAY order: CONTROL FIRST, then excitatory, then inhibitory. This is the
# convention the paper-facing analyses already use -- decoder._PAPER_GROUP_ORDER,
# spatial._PV_GROUP_ORDER, analysis.py's several _paper_group_order/_GROUP_ORDER locals, and
# epoch_analysis.py:2705 ("mCherry first = reference") are all this order. Deliberately DIFFERENT
# from GROUP_ORDER above, which drives model dummy-coding/iteration order for every analysis that
# imports it. Affects x-axis/draw order only, never which values are compared -- the plotting
# helpers index their inputs by group NAME, so reordering the display cannot reassign a group's
# data. Centralised here so modules stop reinventing it under a fourth name.
DREADD_DISPLAY_ORDER = ['mCherry', 'hM3D', 'hM4D']


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


def holm_correct(pvalues, alpha=0.05):
    """Holm-Bonferroni step-down. Returns (reject_bool_array, adjusted_pvalues), in the same
    shape and with the same NaN handling as :func:`fdr_correct` -- a NaN p is left NaN, is never
    marked significant, and does not count towards the family size.

    Holm rather than BH where the family is a small set of PLANNED comparisons whose members are
    each meant to support a claim on their own (strong FWER control), against BH's role for a
    larger exploratory family where a controlled proportion of false positives is acceptable.
    Both live here so no caller re-implements either correction; see
    caban.sp_rates_lmm.unified_posthoc_contrasts for the planned-contrast family this exists for
    and caban.sp_rates_lmm.build_secondary_fdr_table for the BH one.
    """
    pvalues = np.asarray(pvalues, dtype=float)
    reject = np.zeros(pvalues.shape, dtype=bool)
    padj = np.full(pvalues.shape, np.nan)
    finite = np.isfinite(pvalues)
    if finite.sum() == 0:
        return reject, padj
    rej, p_corrected, _, _ = multipletests(pvalues[finite], alpha=alpha, method='holm')
    reject[finite] = rej
    padj[finite] = p_corrected
    return reject, padj


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


def ecdf_panel(ax, per_cell_by_group, per_mouse_by_group=None, xlabel='', title='',
               group_order=None, mouse_alpha=0.25, mouse_linewidth=0.7):
    """Draw per-group pooled per-cell ECDFs (bold) with optional per-mouse ECDFs
    (thin) on one axis.

    per_cell_by_group  : dict group -> 1-D array of per-cell values (pooled).
    per_mouse_by_group : dict group -> {mouse: 1-D array} for the thin lines.
    group_order        : draw/legend order; defaults to this module's GROUP_ORDER.
                         caban.place_cell_rates and the navigation-aware suite use the
                         CLAUDE.md order (hM3D, mCherry, hM4D), which differs from the one
                         here; they pass their own rather than this module changing, since
                         four existing analyses' published figures depend on the default.
                         Affects draw order only, never values.
    mouse_alpha,
    mouse_linewidth     : per-mouse (thin) curve styling. Defaults preserve every existing
                         caller's current look; caban.sp_rates_lmm's plot_amplitude_ecdf passes
                         higher values (plan section 6) since its 17 mice, not thousands of
                         pooled cells, are this analysis' actual unit of inference.
    """
    for group in (GROUP_ORDER if group_order is None else group_order):
        colour = GROUP_COLOURS[group]
        if per_mouse_by_group is not None:
            for mouse_vals in per_mouse_by_group.get(group, {}).values():
                mouse_vals = np.asarray(mouse_vals, dtype=float)
                mouse_vals = mouse_vals[np.isfinite(mouse_vals)]
                if mouse_vals.size == 0:
                    continue
                mx, my = _ecdf(mouse_vals)
                ax.step(mx, my, where='post', color=colour, alpha=mouse_alpha, linewidth=mouse_linewidth)
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


def _mouse_colour_shades(base_colour, n_mice, lo=0.35, hi=0.9):
    """n_mice colours blending base_colour toward white, for per-mouse-within-group colour
    coding (draw_superplot_triplet's cell cloud). Deterministic and evenly spaced -- callers must
    pass mice in a fixed (e.g. sorted) order for a stable colour-to-mouse mapping across panels.
    lo/hi are blend fractions toward the saturated base_colour (higher = more saturated/darker)."""
    base_rgb = np.asarray(to_rgb(base_colour))
    white = np.ones(3)
    fracs = np.array([0.5 * (lo + hi)]) if n_mice == 1 else np.linspace(lo, hi, n_mice)
    return [tuple(white * (1.0 - f) + base_rgb * f) for f in fracs]


def _resolve_superplot_yscale(all_cell_vals, yscale):
    """(scale_name, kwargs) for ax.set_yscale, for a SuperPlot's cell cloud.

    ``yscale='auto'`` picks the scale from the data: 'linear' if anything is negative (log is
    undefined), plain 'log' if every value is strictly positive, and 'symlog' when the data are
    non-negative but contain exact zeros -- which is the common case here, since a cell with no
    detected events has an event rate and a total-S/s of exactly 0 and cannot be dropped (those
    zero-event cells are load-bearing for the rate/fraction-active estimands; see the module
    docstring in caban.sp_rates_lmm). symlog keeps a small linear band around zero so those cells
    stay visible at the axis floor instead of vanishing.

    ``linthresh`` is set to the smallest strictly-positive value in the data, so the linear band
    is exactly wide enough to hold the zeros and nothing else -- picking it larger would flatten
    the bottom of the real distribution into the linear region."""
    vals = np.concatenate([np.asarray(v, dtype=float).ravel() for v in all_cell_vals])
    vals = vals[np.isfinite(vals)]
    if yscale != 'auto':
        return yscale, {}
    if vals.size == 0 or np.any(vals < 0):
        return 'linear', {}
    positive = vals[vals > 0]
    if positive.size == 0:
        return 'linear', {}
    if np.any(vals == 0):
        return 'symlog', {'linthresh': float(positive.min())}
    return 'log', {}


def no_stat_annotation(*args, **kwargs):
    """A do-nothing stat_fn, for panels annotated by annotate_contrast_ci() instead of by
    significance brackets. Needed as an explicit object rather than None because
    caban.analysis._draw_violin_triplet treats stat_fn=None as "use the default" and falls back
    to do_anova1_plot, which would draw ANOVA-gated Tukey brackets underneath the interval block.
    """
    return None


def mouse_contrast_ci(mouse_means_per_group, reference='mCherry', scale='linear', alpha=0.05):
    """Welch two-sample effect estimate + (1-alpha) confidence interval for each non-`reference`
    group against `reference`, computed from PER-MOUSE values only.

    This is the estimate-and-interval counterpart to do_pairwise_holm_plot's significance stars.
    Both consume the same per-mouse arrays and neither ever sees a cell-level value, so a panel
    annotated with these intervals carries exactly the same unit of inference (the mouse) as one
    annotated with brackets -- only the reporting convention differs.

    Two things it deliberately reports together:

    * a **ratio** (multiplicative fold-change), which is what a reader wants for a positive
      quantity like an event rate or a calcium event integral, and
    * the **absolute difference** on the measured scale. A ratio computed off a small base is
      easy to over-read -- "40% fewer events" sounds large until the absolute change is 0.004
      events/s -- so the two are always emitted as a pair rather than the ratio alone.

    It also makes every NULL self-documenting: a non-significant group still reports an interval,
    so the panel states what the data DO exclude rather than only that nothing was detected.

    mouse_means_per_group : dict group -> 1-D array of per-mouse values (one scalar per animal).
                            Arrays may differ in length across groups (the cohorts here are
                            n=5/6/6, and recall sessions drop a different animal each).
    scale : 'linear'  -- values are on their natural measured scale. The difference CI is a Welch
                         CI on the values themselves; the ratio CI is a Welch CI computed on
                         log(values) and exponentiated, which is the interval that matches a
                         log/symlog display axis. Requires every value > 0 for the ratio; if any
                         group mean is <= 0 the ratio fields come back None and only the
                         difference is reported.
            'log'     -- values are ALREADY log-transformed (e.g. a log_amplitude column). The
                         ratio is then exp(difference of means) directly and NO second log is
                         taken -- the same double-log trap documented on
                         _draw_cell_superplot_panel in caban.sp_rates_lmm, in CI form. The
                         reported 'diff' stays in log units.
            'difference_only' -- report the difference only, no ratio. For bounded proportions
                         (fraction active, dropout fraction) where a fold-change is not the
                         natural summary.

    Returns dict group -> {n, n_ref, diff, diff_lo, diff_hi, ratio, ratio_lo, ratio_hi, unit_is_log}
    with the ratio fields None when unavailable.
    """
    if scale not in ('linear', 'log', 'difference_only'):
        raise ValueError(f"mouse_contrast_ci: scale must be 'linear', 'log' or "
                         f"'difference_only', got {scale!r}")
    if reference not in mouse_means_per_group:
        raise KeyError(f'mouse_contrast_ci: reference group {reference!r} not in '
                       f'{sorted(mouse_means_per_group)}')

    def _welch(a, b):
        """(diff, lo, hi) for mean(a) - mean(b) via a Welch t interval."""
        na, nb = len(a), len(b)
        va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
        se = np.sqrt(va / na + vb / nb)
        diff = float(np.mean(a) - np.mean(b))
        if se == 0:
            return diff, diff, diff
        # Welch-Satterthwaite denominator df -- the same approximation scipy's ttest_ind
        # (equal_var=False) uses, so these intervals and do_pairwise_holm_plot's p-values agree
        # about which contrasts cross zero.
        df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
        crit = float(t_dist.ppf(1 - alpha / 2, df))
        return diff, diff - crit * se, diff + crit * se

    ref = np.asarray(mouse_means_per_group[reference], dtype=float).ravel()
    ref = ref[np.isfinite(ref)]
    if len(ref) < 2:
        raise RuntimeError(f'mouse_contrast_ci: reference group {reference!r} has {len(ref)} '
                           f'mouse/mice; need >=2 for a variance estimate.')

    out = {}
    for group, vals in mouse_means_per_group.items():
        if group == reference:
            continue
        vals = np.asarray(vals, dtype=float).ravel()
        vals = vals[np.isfinite(vals)]
        if len(vals) < 2:
            raise RuntimeError(f'mouse_contrast_ci: group {group!r} has {len(vals)} mouse/mice; '
                               f'need >=2 for a variance estimate.')
        diff, diff_lo, diff_hi = _welch(vals, ref)
        ratio = ratio_lo = ratio_hi = None
        if scale == 'log':
            ratio, ratio_lo, ratio_hi = np.exp([diff, diff_lo, diff_hi])
        elif scale == 'linear' and np.all(vals > 0) and np.all(ref > 0):
            log_diff, log_lo, log_hi = _welch(np.log(vals), np.log(ref))
            ratio, ratio_lo, ratio_hi = np.exp([log_diff, log_lo, log_hi])
        out[group] = {'n': int(len(vals)), 'n_ref': int(len(ref)),
                      'diff': diff, 'diff_lo': diff_lo, 'diff_hi': diff_hi,
                      'ratio': None if ratio is None else float(ratio),
                      'ratio_lo': None if ratio_lo is None else float(ratio_lo),
                      'ratio_hi': None if ratio_hi is None else float(ratio_hi),
                      'unit_is_log': scale == 'log'}
    return out


def format_contrast_ci_lines(contrasts, reference, unit='', group_labels=None, diff_fmt='{:+.3g}'):
    """Lines describing each contrast from mouse_contrast_ci()'s output -- a ratio line plus an
    absolute-difference line ('Exc/Ctl 1.55x [1.21, 1.98]' / '  diff +0.031 [0.008, 0.055] /s'),
    or a single labelled difference line when no ratio is available. Ordered by GROUP_ORDER so
    the same group always occupies the same line across panels.

    Split out from annotate_contrast_ci so the identical wording can be written into the stats
    .txt files without a second formatter drifting away from what the figures say.

    A contrast dict carrying a 'p_holm' key (as the model-derived ones in
    caban.sp_rates_lmm._unified_contrast_payloads do) additionally gets its adjusted p-value on
    the ratio line, so a model-derived contrast is never quoted without the decision that goes
    with it. Contrasts without one -- the equal-mouse-weighted Welch intervals, which are in no
    multiplicity family -- are unchanged."""
    labels = GROUP_LABELS if group_labels is None else group_labels
    ref_label = labels.get(reference, reference)
    unit_suffix = f' {unit}' if unit else ''
    lines = []
    for group in [g for g in GROUP_ORDER if g in contrasts]:
        c = contrasts[group]
        label = labels.get(group, group)
        diff_unit = ' log units' if c['unit_is_log'] else unit_suffix
        diff_text = (f"{diff_fmt.format(c['diff'])} "
                     f"[{diff_fmt.format(c['diff_lo'])}, {diff_fmt.format(c['diff_hi'])}]{diff_unit}")
        p_text = ('' if c.get('p_holm') is None
                  else f", Holm-adjusted P = {c['p_holm']:.4g}")
        if c['ratio'] is None:
            # No ratio (a bounded proportion, or a group mean at/below zero): the difference line
            # carries the group label itself, so a group is never rendered as a bare unlabelled row.
            lines.append(f'{label}/{ref_label} diff {diff_text}{p_text}')
        else:
            lines.append(f"{label}/{ref_label} {c['ratio']:.2f}x "
                         f"[{c['ratio_lo']:.2f}, {c['ratio_hi']:.2f}]{p_text}")
            lines.append(f'  diff {diff_text}')
    return lines


def annotate_contrast_ci(ax, mouse_means_per_group, reference='mCherry', scale='linear', unit='',
                         alpha=0.05, fontsize=5.0, loc=(0.02, 0.98)):
    """Draw mouse_contrast_ci()'s estimates + intervals as a compact corner block on `ax`, and
    return the contrasts dict.

    Replaces significance stars as this module's panel annotation. Stars answer only "did it cross
    0.05", which on n=5/6/6 animals is the least informative thing the data have to say, and they
    also cannot be drawn at all on a log or symlog axis (barplot_annotate_brackets positions
    brackets as fractions of the DATA range, which is meaningless once the axis is non-linear).
    An interval block renders identically on every axis type, which is what lets a mixed
    linear/log row of panels carry ONE annotation style instead of brackets on some panels and a
    p-value text block on the others.

    Placed in axes coordinates at `loc`, so it stays put regardless of the axis scale.
    """
    contrasts = mouse_contrast_ci(mouse_means_per_group, reference=reference, scale=scale,
                                  alpha=alpha)
    lines = format_contrast_ci_lines(contrasts, reference, unit=unit)
    if lines:
        ax.text(loc[0], loc[1], '\n'.join(lines), transform=ax.transAxes, va='top', ha='left',
                fontsize=fontsize, linespacing=1.35,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, pad=1.5))
    return contrasts


def reserve_top_fraction(ax, occupancy=0.80):
    """Grow ax's upper y-limit so the existing data occupies the bottom `occupancy` of the panel,
    leaving the top clear for significance brackets. Works on ANY axis scale.

    The trick is to do the arithmetic in the axis' own SCALED space -- `ax.yaxis.get_transform()`
    is the log / symlog / identity mapping matplotlib itself uses to place ticks -- so "make the
    data occupy the bottom 80%" means the same visual thing on a log axis as on a linear one.
    Doing it in data coordinates (top += 0.2 * (hi - lo)) is what breaks on log axes: on a
    0.01-to-300 axis that range is ~300, so the reserved band is absurd.
    """
    trans = ax.yaxis.get_transform()
    lo, hi = trans.transform(np.asarray(ax.get_ylim(), dtype=float))
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return
    new_hi = lo + (hi - lo) / occupancy
    ax.set_ylim(top=float(trans.inverted().transform(np.array([new_hi]))[0]))


def annotate_pairwise_brackets(ax, mouse_means_per_group, group_order, stat_fn,
                               base_frac=0.82, step_frac=0.075, barh_frac=0.018, fs=9,
                               occupancy=0.80, ns_label_pairs=()):
    """Holm-corrected pairwise significance brackets positioned in AXES-FRACTION coordinates, so
    they render identically on linear, log and symlog axes.

    ** This is what lets a row of panels with different axis scales carry ONE annotation style. **
    caban.utilities.barplot_annotate_brackets positions everything in DATA coordinates and scales
    its offsets as fractions of (ylim[1] - ylim[0]), which is meaningless once the axis is
    non-linear -- brackets land off the top of the panel and the stacking increments collapse.
    That limitation is the entire reason the decomposition figure previously annotated its one
    linear panel with stars and its four log panels with a corner p-value text block.

    Here the x positions stay in data coordinates (they are category indices) while the y
    positions are axes fractions, via ax.get_xaxis_transform() -- matplotlib's blended transform
    for exactly this. Headroom is reserved by reserve_top_fraction() above.

    The STATISTICS are not reimplemented: stat_fn is called with annotate=False and only its
    returned p-values are used, so these brackets show precisely the numbers a bracket-annotated
    mouse-level panel would, and the cell cloud never reaches it. Only significant contrasts get
    a bracket, matching do_pairwise_holm_plot's own convention -- unless a pair is named in
    ``ns_label_pairs``, which is opt-in and empty by default.

    mouse_means_per_group : dict group -> 1-D array of per-mouse values. Keys must include
                            'hM3D', 'hM4D' and 'mCherry' (stat_fn's positional convention);
                            group_order sets only the x positions.
    ns_label_pairs        : OPT-IN. Iterable of (group_a, group_b) pairs that get a bracket even
                            when their p-value is >= 0.05, labelled with the p-value itself
                            ('P = 0.078') instead of stars. Empty by default, which is exactly
                            the behaviour every pre-existing call site has: only p < 0.05 is
                            bracketed and only stars are drawn. Use it where a panel reports a
                            named non-significant comparison rather than leaving the reader to
                            infer it from an absent bracket; the p-values still come from
                            stat_fn, so this changes what is DRAWN and never what is computed.
                            A pair listed here whose p-value is NaN is still skipped -- NaN
                            means "this comparison is not on this panel".

    Returns stat_fn's corrected p-values, in pair order [(Exc,Inh),(Exc,Ctl),(Inh,Ctl)].
    """
    corrected = stat_fn(mouse_means_per_group['hM3D'], mouse_means_per_group['hM4D'],
                        mouse_means_per_group['mCherry'], ax, np.zeros(len(group_order)),
                        group_order=group_order, annotate=False)
    if corrected is None:
        return None

    pairs = [('hM3D', 'hM4D'), ('hM3D', 'mCherry'), ('hM4D', 'mCherry')]
    pvals = np.asarray(corrected, dtype=float)
    ns_labelled = {frozenset(pair) for pair in ns_label_pairs}
    drawn = [(pair, p) for pair, p in zip(pairs, pvals)
             if np.isfinite(p) and (p < 0.05 or frozenset(pair) in ns_labelled)]
    if not drawn:
        # Reserve headroom only when something will occupy it, so a panel with no significant
        # contrast is not silently rescaled to leave an empty band its neighbours use.
        return corrected

    reserve_top_fraction(ax, occupancy=occupancy)
    blended = ax.get_xaxis_transform()   # x in data coords, y in axes fraction
    pos = {g: i for i, g in enumerate(group_order)}
    level = 0
    for (g1, g2), p in drawn:
        stars = get_pval_str(p) or f'P = {p:.3f}'
        y = base_frac + level * step_frac
        lx, rx = pos[g1], pos[g2]
        ax.plot([lx, rx], [y + barh_frac, y + barh_frac], c='black', lw=1,
                transform=blended, clip_on=False)
        ax.plot([lx, lx], [y, y + barh_frac], c='black', lw=1, transform=blended, clip_on=False)
        ax.plot([rx, rx], [y, y + barh_frac], c='black', lw=1, transform=blended, clip_on=False)
        ax.text((lx + rx) / 2, y + barh_frac, stars, ha='center', va='bottom', fontsize=fs,
                transform=blended, clip_on=False)
        level += 1
    return corrected


def draw_superplot_triplet(ax, cell_values_per_group, mouse_means_per_group, group_order,
                           group_colours, stat_fn=None, ylabel=None, cell_size=1.5, cell_alpha=0.18,
                           mean_size=90, jitter_width=0.34, yscale='auto', y_quantum=None,
                           annotate='stats', reference='mCherry', ci_scale=None, ci_unit=''):
    """Draw a SuperPlot panel (Lord et al. 2020, J Cell Biol 219:e202001064): every cell plotted
    semi-transparent and colour-coded by mouse (a within-group lightness ramp of the group
    colour, see _mouse_colour_shades), with each mouse's own mean overlaid as a large opaque
    marker with a black edge. Cells belonging to the same mouse share one small x-jitter offset
    (drawn once per mouse) so a mouse's cloud and its mean marker visually group together and
    inconsistency across animals is immediately visible -- an effect carried by one mouse looks
    completely different from one present in every mouse.

    ** AXIS SCALE IS THE WHOLE BALLGAME FOR THIS FIGURE TYPE -- read before changing yscale. **
    An earlier version of this function forced a LINEAR autoscaled axis, and caban.sp_rates_lmm
    reverted to plain mouse-level violins after seeing the result on real data: for a
    heavy-right-tailed quantity (per-event calcium amplitude, event rate, total S/s -- i.e. most
    things measured per cell) a handful of extreme cells set the y-range and the per-mouse means,
    which carry ALL the inferential content, were compressed into roughly the bottom 6-13% of the
    panel. The diagnostic detail that identified the cause: in that figure the ONE panel that read
    correctly was the one already plotting a log-transformed quantity.

    Hence ``yscale='auto'`` (the default), which puts heavy-tailed positive data on a log or
    symlog axis so the cell cloud spans the panel and the mouse means sit in the middle of it --
    see _resolve_superplot_yscale for the zero-handling rule. ``cell_size``/``cell_alpha`` also
    default small and faint (1.5 / 0.18): at the ~2500 cells-per-group scale this module works at,
    larger markers merge into an opaque blob that hides both the distribution and the means.
    Pass ``yscale='linear'`` only for a quantity that is genuinely not heavy-tailed (e.g. a
    bounded fraction), and re-check the result on real data before keeping it.

    Unit of DISPLAY is the cell; unit of INFERENCE stays the mouse. Statistics/brackets are
    computed from ``mouse_means_per_group`` ONLY, via the same stat_fn interface as
    caban.analysis._draw_violin_triplet (pass caban.analysis.do_pairwise_holm_plot; there is no
    default, to avoid single_unit_common importing from analysis, which already imports this
    module) -- the cell cloud is NEVER passed to stat_fn, so a SuperPlot's annotated p-value is
    exactly what a mouse-level-only figure would report. The same is true of ``annotate='ci'``
    below, which routes the same per-mouse arrays into mouse_contrast_ci(). See the regression
    test in scratchpad/test_event_amplitude_integration.py that checks this equality directly;
    that test is the one way this figure type could silently reintroduce pseudoreplication.

    cell_values_per_group  : dict group -> {mouse: 1-D array of per-cell values}. Descriptive
                             cloud only.
    mouse_means_per_group  : dict group -> {mouse: scalar mean}. Must have the same mice (keys)
                             as cell_values_per_group[group] for every group.
    group_order            : display/x-axis order (e.g. CLAUDE.md's mCherry, hM3D, hM4D). May
                             differ from the fixed hM3D/hM4D/mCherry identity order stat_fn itself
                             requires positionally -- stat_fn is always called as
                             stat_fn(g_hM3D, g_hM4D, g_mCherry, ax, heights, group_order=...),
                             matching _draw_violin_triplet's convention, so display order never
                             changes which values are compared.
    ylabel                 : if given, set as the axis y-label.
    y_quantum              : if given, the spacing between adjacent achievable values of a
                             DISCRETE-valued quantity, used to spread the cell cloud vertically.

                             Some per-cell quantities are ratios of a small integer to a fixed
                             denominator -- an event rate is n_events / exposure_seconds -- so
                             with counts in the low single digits the cloud collapses onto a few
                             hard horizontal lines, one per achievable count, and the density
                             within each line is unreadable. Passing y_quantum = 1/exposure_seconds
                             displaces each cell uniformly within +/- 0.4 quanta, i.e. strictly
                             inside its own quantization bin, so no point can be confused with a
                             neighbouring count level and the bands read as densities.

                             Applied to the CELL CLOUD ONLY: mouse mean markers, the y-scale
                             choice, and every statistic are computed from the untouched values.
                             Exact zeros are left pinned at exactly zero, because on the symlog
                             axis these panels use, zero is the meaningful floor occupied by
                             cells with no detected events -- jittering them off it would invent
                             a nonzero rate for a silent cell. Jittered values are likewise
                             clipped to stay strictly positive.
    annotate               : 'stats' -- call stat_fn and draw Holm-corrected significance
                             brackets via annotate_pairwise_brackets(), which positions them in
                             axes-fraction coordinates and so works on log and symlog axes as
                             well as linear ones. This is the default and the recommended style:
                             it is legible at a glance, which a text block is not.
                             'ci'    -- draw mouse_contrast_ci()'s effect estimates + 95%
                             intervals via annotate_contrast_ci() instead. stat_fn is not called.
                             Prefer writing those estimates to a companion file over crowding
                             them into the panel.
                             'none'  -- no annotation.
    reference, ci_scale,
    ci_unit                : passed to annotate_contrast_ci() when annotate='ci'. ci_scale
                             defaults to 'log' if the axis came out linear (which for these
                             panels means the column was already log-transformed) and 'linear'
                             otherwise -- but pass it explicitly rather than relying on that.

    Returns (panel_max_y, annotation) -- `annotation` is stat_fn's own return value under
    annotate='stats' (do_pairwise_holm_plot's Holm-corrected
    [(hM3D,hM4D),(hM3D,mCherry),(hM4D,mCherry)] p-values), exposed so callers/tests can diff it
    directly against a standalone do_pairwise_holm_plot(...) call on the same mouse means; under
    annotate='ci' it is mouse_contrast_ci()'s contrasts dict; None under annotate='none'.
    """
    if annotate not in ('stats', 'ci', 'none'):
        raise ValueError(f"draw_superplot_triplet: annotate must be 'stats', 'ci' or 'none', "
                         f"got {annotate!r}")
    if annotate == 'stats' and stat_fn is None:
        raise ValueError("draw_superplot_triplet: annotate='stats' requires a stat_fn "
                         "(pass caban.analysis.do_pairwise_holm_plot).")
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    rng = np.random.default_rng(0)
    x_pos = {group: i for i, group in enumerate(group_order)}

    scale_name, scale_kwargs = _resolve_superplot_yscale(
        [v for g in group_order for v in cell_values_per_group[g].values()], yscale)

    mouse_mean_arrays = {}  # group -> 1-D array of per-mouse means, fixed hM3D/hM4D/mCherry identity
    all_vals = []
    for group in group_order:
        mice_sorted = sorted(cell_values_per_group[group].keys())
        if set(mice_sorted) != set(mouse_means_per_group[group].keys()):
            raise ValueError(f'draw_superplot_triplet: mouse set mismatch for group {group!r} '
                             f'between cell_values_per_group and mouse_means_per_group.')
        if len(mice_sorted) < 2:
            raise RuntimeError(f'draw_superplot_triplet: group {group!r} has {len(mice_sorted)} '
                               f'mouse(s); need >=2 for a group comparison.')
        shades = _mouse_colour_shades(group_colours[group], len(mice_sorted))
        mouse_offsets = rng.uniform(-jitter_width / 2, jitter_width / 2, size=len(mice_sorted))
        mouse_mean_arrays[group] = np.array([mouse_means_per_group[group][m] for m in mice_sorted])

        for mouse, shade, offset in zip(mice_sorted, shades, mouse_offsets):
            cell_vals = np.asarray(cell_values_per_group[group][mouse], dtype=float)
            fine_jitter = rng.uniform(-jitter_width / 6, jitter_width / 6, size=len(cell_vals))
            # Sub-quantum vertical spread for discrete-valued quantities; display only, and never
            # off an exact zero (see y_quantum in the docstring).
            drawn_vals = cell_vals
            if y_quantum is not None:
                shift = rng.uniform(-0.4 * y_quantum, 0.4 * y_quantum, size=len(cell_vals))
                drawn_vals = np.where(cell_vals == 0, 0.0,
                                      np.maximum(cell_vals + shift, 0.05 * y_quantum))
            ax.scatter(x_pos[group] + offset + fine_jitter, drawn_vals, s=cell_size, color=shade,
                      edgecolors='none', alpha=cell_alpha, zorder=2)
            ax.scatter([x_pos[group] + offset], [mouse_means_per_group[group][mouse]],
                      s=mean_size, color=shade, edgecolors='black', linewidths=1.0, alpha=1.0,
                      zorder=5)
            all_vals.append(cell_vals)
            all_vals.append(np.asarray([mouse_means_per_group[group][mouse]], dtype=float))

    panel_top = float(np.max(np.concatenate(all_vals))) + 0.01
    heights = np.full(len(group_order), panel_top + 0.02)

    if scale_name != 'linear':
        ax.set_yscale(scale_name, **scale_kwargs)

    if annotate == 'none':
        return panel_top, None

    if annotate == 'ci':
        scale = ci_scale if ci_scale is not None else ('log' if scale_name == 'linear' else 'linear')
        contrasts = annotate_contrast_ci(ax, mouse_mean_arrays, reference=reference, scale=scale,
                                         unit=ci_unit)
        return panel_top, contrasts

    # Significance brackets, drawn in AXES-FRACTION coordinates (annotate_pairwise_brackets) so
    # they work on this panel's log/symlog axis as well as on a linear one. The data-coordinate
    # annotator (barplot_annotate_brackets, via stat_fn's own annotate=True path) cannot do that;
    # see annotate_pairwise_brackets' docstring. stat_fn still computes every p-value, from the
    # mouse means alone.
    corrected_pvals = annotate_pairwise_brackets(ax, mouse_mean_arrays, group_order, stat_fn)
    return panel_top, corrected_pvals


def grow_ylim_for_bracket_headroom(ax):
    """Grow ax's y-limits to cover the topmost significance-bracket line, if any bracket was
    drawn above the current axis range. matplotlib autoscales to the bracket LINES a stat_fn
    annotator draws but not to the asterisk TEXT above them (va='bottom', not clipped to the
    axes), so without this the topmost stars spill over the frame/into the title. Only the
    bracket lines live in ax.lines here -- violin bodies are PolyCollections, the median bar a
    LineCollection, so np.max over ax.lines' ydata picks up bracket geometry only. Shared by
    draw_superplot_triplet() (above) and caban.analysis._draw_violin_triplet, which used to each
    carry an independent copy of this exact block."""
    y0, y1 = ax.get_ylim()
    bracket_tops = [np.max(line.get_ydata()) for line in ax.lines if len(line.get_ydata())]
    if bracket_tops:
        needed = max(bracket_tops) + 0.08 * (y1 - y0)
        if needed > y1:
            ax.set_ylim(y0, needed)


def _mixed_model_degeneracy(result, n_fixed):
    """Reason a converged MixedLM fit is unusable, or None if it is fine.

    statsmodels can *converge* onto the boundary of the parameter space with the random-effect
    variance collapsed to zero. It then reports a singular covariance, standard errors inflated
    by many orders of magnitude (or missing entirely), and p-values pinned at 1.000 — output that
    is indistinguishable from a clean null unless it is checked for. That has to be caught and
    reported, not passed through as a successful fit.
    """
    bse = np.asarray(result.bse[:n_fixed], dtype=float)
    if not np.all(np.isfinite(bse)):
        return 'the fixed-effect standard errors are not all finite'

    params = np.asarray(result.params[:n_fixed], dtype=float)
    scale = np.max(np.abs(params)) if np.max(np.abs(params)) > 0 else 1.0
    if np.max(bse) > 1e6 * scale:
        return ('the fixed-effect standard errors are inflated by more than 1e6 relative to the '
                'coefficients ({:.3g} vs {:.3g})'.format(np.max(bse), scale))

    group_var = float(np.asarray(result.cov_re)[0, 0])
    if not np.isfinite(group_var) or group_var <= 0.0:
        return 'the random-effect (mouse) variance collapsed to {:.3g}'.format(group_var)
    return None


def fit_mixed_model(df, formula, group_col='mouse', extra_header='', method='lbfgs'):
    """Fit an arbitrary ``formula`` as a linear mixed model with a random intercept on
    ``group_col``, falling back to ``group_col``-clustered OLS if the mixed model does not
    converge OR converges to a degenerate solution (see :func:`_mixed_model_degeneracy`) — a
    documented statistical fallback, not a silent error swallow.

    This is the general fitting/fallback primitive behind :func:`fit_group_mixed_model` (which
    fixes ``formula`` to a plain ``value ~ C(group)`` contrast). Pass an explicit ``formula`` for
    anything beyond that — e.g. a ``group * epoch`` interaction — so every mixed-model fit in the
    codebase goes through the same convergence/degeneracy handling rather than a parallel copy.

    ``df`` must already have any categorical columns coded (e.g. via ``pd.Categorical`` with an
    explicit reference level) — this function does not touch column dtypes.

    ``method`` is the NUMERICAL optimizer handed to ``MixedLM.fit`` — a str, or a list of them
    tried in order (statsmodels' own escalation). It defaults to ``'lbfgs'``, which is what every
    pre-existing call site here used and continues to use unchanged. It is exposed because lbfgs
    silently returns a BOUNDARY solution (random-effect variance pinned at 0, non-finite
    fixed-effect standard errors) on some frame shapes — notably a few hundred rows per group
    across ~16 groups, which is what a cell-level table looks like — where bfgs/cg/powell all
    reach the same interior optimum. Changing the optimizer changes neither the model, the
    likelihood, nor the estimand: it is a numerical choice, not a statistical fallback, and a
    degenerate or non-converged result still goes through :func:`_mixed_model_degeneracy`
    exactly as before.

    Returns (result, method_used, summary_text). ``result`` is the fitted statsmodels object
    (``MixedLMResults`` or a cluster-robust ``RegressionResults``) — pass it to
    :func:`joint_wald_test` for omnibus/interaction tests. ``method_used`` is 'mixedlm' or
    'clustered_ols'.
    """
    header = f'Formula: {formula}\nN rows = {len(df)}, N {group_col} = {df[group_col].nunique()}\n\n{extra_header}'

    def _clustered_ols(reason):
        ols = smf.ols(formula, data=df).fit(
            cov_type='cluster', cov_kwds={'groups': df[group_col]})
        text = (f'Mixed model unusable ({reason}); using {group_col}-clustered OLS instead.\n'
                f'Cluster = {group_col}.\n{header}{ols.summary()}\n')
        return ols, 'clustered_ols', text

    try:
        model = smf.mixedlm(formula, data=df, groups=df[group_col])
        with warnings.catch_warnings():
            # statsmodels signals boundary/singular fits through warnings; they are inspected
            # explicitly below rather than printed, so a degenerate fit is reported in the
            # returned text where the reader will actually see it.
            warnings.simplefilter('ignore')
            result = model.fit(reml=True, method=method)
        degenerate = _mixed_model_degeneracy(result, n_fixed=len(model.exog_names))
        if degenerate is not None:
            return _clustered_ols(degenerate)
        text = (f'Linear mixed model, random intercept on {group_col}\n'
                f'Formula: {formula} + (1 | {group_col})\n{header}{result.summary()}\n')
        return result, 'mixedlm', text
    except Exception as exc:  # documented fallback: cluster-robust OLS
        return _clustered_ols(f'it failed to converge: {exc}')


def fit_group_mixed_model(df, value_col='value', reference='mCherry'):
    """Fit ``value ~ C(group)`` with a per-mouse random intercept (cell nested in
    mouse), so single-cell n is honoured without pseudoreplication.

    Returns (summary_text, method_used) — a thin wrapper over :func:`fit_mixed_model` that
    discards the raw fitted result, preserving this function's original signature for existing
    callers. Use :func:`fit_mixed_model` directly when the raw result is needed (e.g. for
    :func:`joint_wald_test`).
    """
    df = df[np.isfinite(df[value_col])].copy()
    if df['group'].nunique() < 2 or len(df) < 6:
        return (f'Insufficient data for group model: '
                f'{len(df)} cells across {df["group"].nunique()} groups.\n'), 'none'

    df['group'] = pd.Categorical(df['group'],
                                 categories=[reference] + [g for g in GROUP_ORDER if g != reference])
    formula = f'{value_col} ~ C(group, Treatment(reference="{reference}"))'
    _result, method, text = fit_mixed_model(df, formula, group_col='mouse')
    return text, method


def _fe_params_and_cov(result, n_fixed=None):
    """(fixed-effect names, coefficient vector, fixed-effect covariance block, n_fixed) for a
    fitted MixedLM result OR a cluster-robust OLS result -- the two things
    :func:`fit_mixed_model` can return.

    The two differ in exactly two places: MixedLM exposes its fixed effects on ``.fe_params``
    while OLS puts everything on ``.params``, and MixedLM's ``cov_params()`` appends the
    variance-component parameters after the fixed effects. Both are handled here once so that
    :func:`joint_wald_test` and :func:`linear_contrast_test` cannot drift apart in how they read
    a model -- an omnibus test and its own post-hoc contrasts disagreeing about which
    coefficients they are reading would be a silent, near-undebuggable error.
    """
    names = (list(result.fe_params.index) if hasattr(result, 'fe_params')
             else list(result.params.index))
    if n_fixed is None:
        n_fixed = len(names)
    beta = (np.asarray(result.fe_params) if hasattr(result, 'fe_params')
            else np.asarray(result.params)).reshape(-1)[:n_fixed]
    cov_full = np.asarray(result.cov_params())
    cov_fe = cov_full[:n_fixed, :n_fixed] if cov_full.shape[0] != n_fixed else cov_full
    return names, beta, cov_fe, n_fixed


def linear_contrast_test(result, weights, n_groups, n_fixed=None, alpha=0.05):
    """Two-sided t-test of one linear combination of fixed effects, ``sum_k w_k * beta_k = 0``.

    The single-contrast counterpart of :func:`joint_wald_test`, sharing its parameter extraction
    (:func:`_fe_params_and_cov`) and -- deliberately -- its denominator degrees of freedom. This
    is what turns a fitted ``group * epoch`` model into an interpretable SIMPLE EFFECT: the
    treatment-vs-control difference AT one epoch is the group main coefficient plus that epoch's
    interaction coefficient, which is a contrast, not a coefficient, and therefore cannot be read
    off the model summary.

    weights  - {fixed-effect coefficient name: weight}. Names are read off the fitted result
               (``result.fe_params.index`` / ``result.params.index``), never guessed: statsmodels'
               dummy-name format depends on the formula. An unknown name raises.
    n_groups - number of clusters (mice) behind the fit. ``df = n_groups - 1``, the SAME
               animal-level convention :func:`joint_wald_test` uses and for the same reason (see
               its docstring): every row of these models is nested in one of a handful of animals,
               so an observation-level df would overstate precision. Using one convention for the
               omnibus test and its post-hoc contrasts is what lets both be reported as one
               procedure.
    alpha    - the confidence interval is (1 - alpha); the p-value is unaffected.

    Returns dict(estimate, se, t, df, p, ci_low, ci_high) on the scale the model was fit on. For
    a log-scale response, exponentiate the estimate and both interval bounds to obtain the ratio
    and its interval -- do NOT exponentiate the standard error.
    """
    names, beta, cov_fe, n_fixed = _fe_params_and_cov(result, n_fixed)
    name_to_idx = {n: i for i, n in enumerate(names)}
    missing = [n for n in weights if n not in name_to_idx]
    if missing:
        raise ValueError(f'linear_contrast_test: parameter(s) not found in fitted model: '
                         f'{missing}. Available: {names}')
    if n_groups < 2:
        raise ValueError(f'linear_contrast_test: n_groups={n_groups} must be >=2 for a defined df.')

    c = np.zeros(n_fixed, dtype=float)
    for name, w in weights.items():
        c[name_to_idx[name]] = float(w)

    estimate = float(c @ beta)
    var = float(c @ cov_fe @ c)
    if not np.isfinite(var) or var <= 0.0:
        raise RuntimeError(f'linear_contrast_test: contrast variance is {var:.3g} for weights '
                           f'{weights} -- the fitted covariance is degenerate, so no interval or '
                           f'p-value can be formed from it.')
    se = float(np.sqrt(var))
    df = n_groups - 1
    tstat = estimate / se
    p = float(2.0 * t_dist.sf(abs(tstat), df))
    crit = float(t_dist.ppf(1.0 - alpha / 2.0, df))
    return {'estimate': estimate, 'se': se, 't': tstat, 'df': df, 'p': p,
            'ci_low': estimate - crit * se, 'ci_high': estimate + crit * se}


def joint_wald_test(result, param_names, n_groups, n_fixed=None):
    """Joint Wald test that every fixed-effect coefficient named in ``param_names`` equals zero.

    Works uniformly for a fitted MixedLM result and a cluster-robust OLS result (the two possible
    outcomes of :func:`fit_mixed_model`) via an explicit restriction matrix, since
    ``MixedLMResults.f_test()`` has a param-vector-shape quirk involving the random-effect
    variance component. This is the shared math behind the omnibus test in
    ``caban.pca_state_metrics._lmm_holm_pairs``; every joint (multi-coefficient) mixed-model test
    in the codebase should go through this function rather than a parallel R-matrix construction.

    param_names - exact fixed-effect coefficient names to restrict to zero jointly, e.g. a subset
                  of ``result.fe_params.index`` (MixedLM) or ``result.params.index`` (OLS). Read
                  these names off the fitted result rather than guessing statsmodels' dummy-name
                  format, since it depends on the formula (``C(group, Treatment(...))[T.hM3D]``,
                  ``C(group)[T.hM3D]:C(epoch)[T.trace]``, ...).
    n_groups    - number of clusters (e.g. mice) underlying the fit's covariance structure. The
                  F-test denominator df is ``n_groups - 1``, REQUIRED (no default) so a call site
                  can never silently fall back to the wrong quantity. This replaced an earlier
                  ``nobs - n_fixed`` denominator that was an observation-level df dressed up as a
                  cluster-level one: correct for i.i.d. rows, but every row here is a cell (or a
                  cell x trial x epoch combination) nested inside one of only a handful of mice,
                  so treating each row as an independent unit of information overstated precision
                  by orders of magnitude (verified: it made a cluster-robust primary-model p-value
                  of 0.0089 report as 0.0016, and a MixedLM co-primary fit report p=1.2e-11 instead
                  of something reasonable for 17 mice). ``n_groups - 1`` is the standard small-G
                  cluster-robust choice (Cameron & Miller 2015) and is exactly right for a
                  BETWEEN-mouse contrast (group does not vary within a mouse, so the cluster-robust
                  covariance is only as informative as the number of mice). For a MixedLM fit whose
                  restricted coefficients involve a WITHIN-mouse factor (e.g. a group x trial
                  interaction on repeated per-mouse measurements), ``n_groups - 1`` is a
                  conservative approximation, not a Satterthwaite/Kenward-Roger denominator df --
                  this codebase implements no such correction, so every call site here uses the
                  same conservative df rather than a per-case ad hoc formula. It costs power, not
                  validity (Type I error control is unaffected).
    n_fixed     - number of fixed-effect parameters at the head of the parameter vector (MixedLM
                  appends variance-component parameters after them). Defaults to the number of
                  named fixed effects on the result.

    Returns dict(F=..., df1=..., df2=..., p=...). F/p are nan if param_names is empty.
    """
    if len(param_names) == 0:
        return {'F': float('nan'), 'df1': 0, 'df2': 0, 'p': float('nan')}

    all_names, beta, cov_fe, n_fixed = _fe_params_and_cov(result, n_fixed)
    name_to_idx = {n: i for i, n in enumerate(all_names)}
    missing = [n for n in param_names if n not in name_to_idx]
    if missing:
        raise ValueError(f'joint_wald_test: parameter(s) not found in fitted model: {missing}. '
                         f'Available: {all_names}')

    Rmat = np.zeros((len(param_names), n_fixed), dtype=float)
    for k, name in enumerate(param_names):
        Rmat[k, name_to_idx[name]] = 1.0

    Rb = Rmat @ beta
    RVR = Rmat @ cov_fe @ Rmat.T
    chi2_stat = float(Rb @ np.linalg.solve(RVR, Rb))
    q = Rmat.shape[0]
    if n_groups < 2:
        raise ValueError(f'joint_wald_test: n_groups={n_groups} must be >=2 for a defined df2.')
    df2 = n_groups - 1
    F = chi2_stat / q
    p = float(f_dist.sf(F, q, df2))
    return {'F': F, 'df1': q, 'df2': df2, 'p': p}


def save_fig(fig, path_png, dpi=300):
    """Save a matched PNG + SVG pair (paper convention)."""
    fig.savefig(path_png, format='png', dpi=dpi)
    fig.savefig(os.path.splitext(path_png)[0] + '.svg', format='svg')
