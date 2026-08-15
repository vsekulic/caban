"""Locomotion covariates for the navigation-aware single-cell analyses.

Two jobs:

1. Test whether the DREADD groups differ in how much they actually navigate. If they do not,
   the "they just moved more" explanation for the cell-averaged rate differences is dead on
   arrival; if they do, the movement-restricted panels in :mod:`caban.place_cell_rates` and the
   ANCOVA here are what adjust for it.
2. Regress the whole-session cell-averaged rate on locomotion, so the group effect can be reported
   *after* adjusting for distance travelled and mean speed.

Metrics are derived from ``velocities_miniscope_smooth`` -- the same trace ``OccupancyAnalysis``
consumes -- so that class is left untouched and the numbers stay comparable with the existing
occupancy panels.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

from caban.utilities import MINISCOPE_FRAME_MS, VELOCITY_THRESHOLD
# caban.decoder before caban.analysis -- see the note on the same imports in
# caban/place_cell_rates.py for why the cycle must be entered from decoder.
from caban.decoder import _copy_analysis_methods_template
from caban.analysis import group_colours
from caban.place_cell_rates import (
    FRAME_CLASS_ALL,
    GROUP_LABELS,
    GROUP_ORDER,
    NAV_AWARE_DIR,
    WINDOW_WHOLE_SESSION,
    _METRIC_YLABEL,
    _panel_row,
    _save,
    _values_per_group,
    build_frame_mask,
    set_session_title,
)

LOCOMOTION_COLUMNS = ['distance_cm', 'mean_speed_cms', 'mean_speed_moving_cms', 'pct_time_moving']
LOCOMOTION_TITLES = ['Distance travelled (cm)', 'Mean speed (cm/s)',
                     'Mean speed while moving (cm/s)', 'Time moving (%)']

_DT_SEC = MINISCOPE_FRAME_MS / 1000.0


def compute_locomotion_metrics(sess, *, window=WINDOW_WHOLE_SESSION,
                               speed_thresh=VELOCITY_THRESHOLD, first_n_sec=180.0):
    """Per-mouse locomotion summary over one analysis window.

    Reuses :func:`caban.place_cell_rates.build_frame_mask` for the window so these numbers cover
    exactly the same frames as the rate panels they are meant to explain.
    """
    mask = build_frame_mask(sess, window, FRAME_CLASS_ALL,
                            first_n_sec=first_n_sec, speed_thresh=speed_thresh)
    velocities = np.asarray(sess.velocities_miniscope_smooth[:sess.S.shape[1]], dtype=float)[mask]

    if velocities.size == 0:
        raise RuntimeError(
            'Window {!r} selected 0 frames for {} {}.'.format(window, sess.mouse, sess.session_type))
    if not np.all(np.isfinite(velocities)):
        raise RuntimeError(
            'Velocity trace for {} {} contains non-finite samples within the {} window; '
            'cannot summarise locomotion.'.format(sess.mouse, sess.session_type, window))

    moving = velocities >= speed_thresh
    return {
        'distance_cm': float(np.sum(velocities) * _DT_SEC),
        'mean_speed_cms': float(np.mean(velocities)),
        # Mean speed *given* the animal is moving separates "runs fast when it runs" from
        # "spends more time running" -- the two combine into mean_speed_cms.
        'mean_speed_moving_cms': float(np.mean(velocities[moving])) if moving.any() else 0.0,
        'pct_time_moving': float(100.0 * np.count_nonzero(moving) / velocities.size),
        'duration_s': float(velocities.size) * _DT_SEC,
        'n_frames': int(velocities.size),
    }


def plot_locomotion_group_comparison(PLOTS_DIR, mouse_groups, per_mouse, session_type, window,
                                     figsize=(7.0, 3.2), auto_close=True):
    """Four-metric violin triplet across DREADD groups, n = mice.

    This is the figure that establishes whether there is a navigation difference to adjust for.
    """
    context = 'plot_locomotion_group_comparison({}, {})'.format(session_type, window)
    values, names = _values_per_group(per_mouse, mouse_groups, LOCOMOTION_COLUMNS, context)

    fig = _panel_row(values, names, LOCOMOTION_COLUMNS, LOCOMOTION_TITLES,
                     'Per-mouse value', figsize, show_mouse_names=False,
                     session_title=session_type)

    save_dir = os.path.join(PLOTS_DIR, NAV_AWARE_DIR, 'locomotion_metrics', window)
    _save(fig, save_dir, 'locomotion_group_comparison-{}'.format(session_type), auto_close)
    return save_dir


def write_locomotion_tables(PLOTS_DIR, rows, window):
    """Write the tidy per-mouse table and the per-metric group-comparison statistics."""
    save_dir = os.path.join(PLOTS_DIR, NAV_AWARE_DIR, 'locomotion_metrics', window)
    os.makedirs(save_dir, exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(save_dir, 'locomotion_summary.csv'), index=False)

    lines = ['Locomotion group comparison — window: {}'.format(window),
             'One-way ANOVA across hM3D / mCherry / hM4D, n = mice.', '']
    for session_type, sub in df.groupby('session', sort=False):
        lines.append('=== {} ==='.format(session_type))
        for column, title in zip(LOCOMOTION_COLUMNS, LOCOMOTION_TITLES):
            groups = [sub.loc[sub['group'] == g, column].to_numpy(dtype=float) for g in GROUP_ORDER]
            f_stat, p_value = stats.f_oneway(*groups)
            means = ', '.join('{}={:.3f}'.format(GROUP_LABELS[g], np.mean(v))
                              for g, v in zip(GROUP_ORDER, groups))
            lines.append('  {:<32s} F={:8.4f}  p={:.4g}   [{}]'.format(title, f_stat, p_value, means))
        lines.append('')

    path = os.path.join(save_dir, 'locomotion_group_stats.txt')
    with open(path, 'w') as fh:
        fh.write('\n'.join(lines))
    return path


# ---------------------------------------------------------------------------
# Rate vs locomotion: ANCOVA
# ---------------------------------------------------------------------------

def _ancova(df, rate_col, locomotion_col):
    """OLS ``rate ~ locomotion + C(group)`` plus the locomotion-only and group-only baselines.

    Returns (fitted model, text report). The question the report answers is whether the group
    term still carries signal once locomotion is in the model.
    """
    full = smf.ols('{} ~ {} + C(group)'.format(rate_col, locomotion_col), data=df).fit()
    reduced = smf.ols('{} ~ {}'.format(rate_col, locomotion_col), data=df).fit()
    group_only = smf.ols('{} ~ C(group)'.format(rate_col), data=df).fit()

    # F-test of the group term over and above locomotion.
    comparison = sm.stats.anova_lm(reduced, full)
    f_group = float(comparison['F'].iloc[-1])
    p_group = float(comparison['Pr(>F)'].iloc[-1])

    # Pearson correlation of rate with locomotion, pooled and within each group.
    r_pooled, p_pooled = stats.pearsonr(df[locomotion_col], df[rate_col])

    lines = [
        'ANCOVA: {} ~ {} + group'.format(rate_col, locomotion_col),
        'n = {} mice'.format(len(df)),
        '',
        'Group effect AFTER adjusting for {}:  F = {:.4f}, p = {:.4g}'.format(
            locomotion_col, f_group, p_group),
        'Group effect WITHOUT adjustment:      F = {:.4f}, p = {:.4g}'.format(
            float(group_only.fvalue), float(group_only.f_pvalue)),
        '',
        'Locomotion slope: {:.6g} (p = {:.4g})'.format(
            float(full.params[locomotion_col]), float(full.pvalues[locomotion_col])),
        'Pooled Pearson r({}, {}) = {:.4f}, p = {:.4g}'.format(
            locomotion_col, rate_col, r_pooled, p_pooled),
        '',
        'Within-group Pearson r:',
    ]
    for group in GROUP_ORDER:
        sub = df[df['group'] == group]
        if len(sub) < 3:
            lines.append('  {:<8s} n = {} (too few for a correlation)'.format(GROUP_LABELS[group], len(sub)))
            continue
        r_g, p_g = stats.pearsonr(sub[locomotion_col], sub[rate_col])
        lines.append('  {:<8s} r = {:+.4f}, p = {:.4g}  (n = {})'.format(
            GROUP_LABELS[group], r_g, p_g, len(sub)))
    lines += ['', '--- full model summary ---', str(full.summary())]

    return full, f_group, p_group, '\n'.join(lines)


def plot_rate_vs_locomotion(PLOTS_DIR, df, session_type, window, locomotion_col,
                            want_peakval=False, mapping='full',
                            figsize=(3.6, 3.2), auto_close=True):
    """Per-mouse scatter of the cell-averaged rate against a locomotion metric, with ANCOVA.

    *df* must carry columns ``mouse``, ``group``, ``rate``, and *locomotion_col*, one row per
    mouse. Per-group regression lines are drawn where a group has >= 3 mice.
    """
    model, f_group, p_group, report = _ancova(df, 'rate', locomotion_col)

    fig, ax = plt.subplots(figsize=figsize)
    ax.spines[['right', 'top']].set_visible(False)
    for group in GROUP_ORDER:
        sub = df[df['group'] == group]
        ax.scatter(sub[locomotion_col], sub['rate'], s=22, color=group_colours[group],
                   edgecolors='none', alpha=0.9, label=GROUP_LABELS[group], zorder=3)
        if len(sub) >= 3:
            slope, intercept = np.polyfit(sub[locomotion_col], sub['rate'], 1)
            xs = np.linspace(sub[locomotion_col].min(), sub[locomotion_col].max(), 50)
            ax.plot(xs, slope * xs + intercept, color=group_colours[group], lw=1.0, alpha=0.7)

    ax.set_xlabel(dict(zip(LOCOMOTION_COLUMNS, LOCOMOTION_TITLES))[locomotion_col])
    ax.set_ylabel(_METRIC_YLABEL[want_peakval])
    # The ANCOVA readout (group F, p) is written in full to the ancova-*.txt beside this figure;
    # the title carries only the session name, per the paper-facing title convention.
    set_session_title(ax, session_type)
    ax.legend(frameon=False, fontsize='xx-small')
    plt.tight_layout(pad=0.5)

    save_dir = os.path.join(PLOTS_DIR, NAV_AWARE_DIR, 'rate_vs_locomotion', session_type, window)
    stem = 'rate_vs_{}-{}-{}'.format(locomotion_col, session_type, mapping)
    _save(fig, save_dir, stem, auto_close)

    with open(os.path.join(save_dir, 'ancova-{}-{}-{}.txt'.format(
            locomotion_col, session_type, mapping)), 'w') as fh:
        fh.write(report)

    return save_dir, f_group, p_group


def copy_methods_templates(PLOTS_DIR):
    """Drop the METHODS templates into the directories this module writes."""
    root = os.path.join(PLOTS_DIR, NAV_AWARE_DIR)
    for template, subdir in (
        ('locomotion_group_comparison_methods.txt', 'locomotion_metrics'),
        ('rate_locomotion_ancova_methods.txt', 'rate_vs_locomotion'),
    ):
        dest = os.path.join(root, subdir)
        os.makedirs(dest, exist_ok=True)
        _copy_analysis_methods_template(template, dest)
