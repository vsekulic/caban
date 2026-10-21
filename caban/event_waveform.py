"""Average single-cell calcium event per group and epoch (Figure 2 v2, panel h), on S, C and YrA.

Events are the primary analysis's: contiguous supra-threshold runs of S, each placed at its S peak
frame, in each of the primary model's three epochs -- pre-tone (20 s), trace, post-shock (20 s)
(sp_rates_lmm.build_run_structure_table with its epoch windows; VS, 2026-10-21).
Around every event, three traces of the same cell are taken:

- S   (deconvolved; -1 to +1 s): the measure the amplitude statistics are computed on;
- C   (Minian's denoised calcium fit; -1 to +3 s): the event-triggered calcium transient;
- YrA (the residual, i.e. the cell's signal plus noise before the fit; -1 to +3 s).

C and YrA segments are baseline-subtracted per event (mean of -1 to -0.5 s); S is not (it is zero
between events). Cells are matched to the C / YrA rows BY UNIT ID (epoch_modulation._unit_id_rows);
an event of a cell with no YrA trace is counted and left out of the YrA average only. An event whose
window runs past the recording (a post-shock event near the end of the last trial) keeps the
samples that exist: the rest are NaN and each time point averages the events that have it; such
events are counted (n_events_truncated).

Per animal, the average over all its events; per group, mean +/- s.e.m. ACROSS ANIMALS (the unit of
inference: the statistics above the panel are the primary model's, n = 5/6/6 animals). The paper
panel is drawn from PANEL_SIGNAL; every signal's group plot, per-animal plot and per-animal table are
written beside it. METHODS: analysis_methods_templates/event_waveform_methods.md.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from caban.utilities import MINISCOPE_FPS
from caban.single_unit_common import (GROUP_COLOURS, GROUP_LABELS, DREADD_DISPLAY_ORDER, ensure_dirs,
                                      save_fig)
from caban.decoder import _copy_analysis_methods_template
from caban.nature_comm_style import nature_style, save_paper_panel, box_strip, LINE_PT, FONT_PT
from caban.single_unit_common import annotate_pairwise_brackets
from caban.figure2_composite import panel_figsize, LAYOUT_V2_MM
from caban.sp_rates_lmm import (build_run_structure_table, tfc_epoch_frames_fn, paper_contrast,
                                _precomputed_stat_fn, TFC_MATCHED_PROFILE_EPOCHS)
from caban.figure2_effect_profile import trace_table
from caban.epoch_modulation import _unit_id_rows

OUT_SUBDIR = os.path.join('sp_rates_lmm', 'TFC_cond')
METHODS_FILENAME = 'event_waveform_methods.md'
PANEL_ROOT = 'group_mean_event_waveform'          # the Figure 2 v2 h file
PANEL_SIGNAL = 'C'                                # VS, 2026-10-21; 'YrA' is the less-modelled option
WINDOWS_S = {'S': (1.0, 1.0), 'C': (1.0, 3.0), 'YrA': (1.0, 3.0)}   # (before, after) the S peak
BASELINE_S = (-1.0, -0.5)
PANEL_EPOCHS = TFC_MATCHED_PROFILE_EPOCHS        # pre_tone_matched, trace, post_shock: f's windows
EPOCH_TITLES = {'pre_tone_matched': 'Pre-tone', 'trace': 'Trace', 'post_shock': 'Post-shock'}
YLABEL = {'S': 'Deconvolved activity S (a.u.)', 'C': 'Calcium C (a.u., baseline-subtracted)',
          'YrA': 'Residual YrA (a.u., baseline-subtracted)'}


def _trace_matrix(session, signal):
    return np.asarray(getattr(session, signal), dtype=float)


def per_animal_waveforms(sessions, df_runs_trace, signal, mapping='full'):
    """{group: (n_animals, n_samples)}, the time axis, and a per-animal table, for the events in
    ``df_runs_trace`` (one epoch's runs)."""
    before, after = WINDOWS_S[signal]
    pre, post = int(round(before * MINISCOPE_FPS)), int(round(after * MINISCOPE_FPS))
    t = np.arange(-pre, post + 1) / MINISCOPE_FPS
    base = (t >= BASELINE_S[0]) & (t < BASELINE_S[1])
    per_group, table = {}, []
    for group in DREADD_DISPLAY_ORDER:
        gsub = df_runs_trace[df_runs_trace['group'] == group]
        if gsub.empty:
            raise RuntimeError(f'event_waveform: no events for group {group!r}.')
        rows = []
        for mouse, msub in gsub.groupby('mouse', sort=True):
            session = sessions[mouse]
            X = _trace_matrix(session, signal)
            row_of = _unit_id_rows(session, signal)
            segs, no_trace, truncated = [], 0, 0
            for cell, peak in zip(msub['cell'].astype(int), msub['peak_frame'].astype(int)):
                if cell not in row_of:
                    no_trace += 1
                    continue
                lo, hi = peak - pre, peak + post + 1
                seg = np.full(len(t), np.nan)
                a, b = max(lo, 0), min(hi, X.shape[1])
                seg[a - lo:b - lo] = X[row_of[cell], a:b]
                if a != lo or b != hi:
                    truncated += 1
                    if not np.isfinite(seg[base]).all():
                        raise RuntimeError(f'event_waveform: {mouse} event at frame {peak} has no '
                                           f'baseline samples ({signal}).')
                if signal != 'S':
                    seg = seg - np.nanmean(seg[base])
                segs.append(seg)
            if not segs:
                raise RuntimeError(f'event_waveform: {mouse} has no {signal} trace for any event.')
            mean = np.nanmean(np.vstack(segs), axis=0)
            rows.append(mean)
            table.append({'mouse': mouse, 'group': group, 'signal': signal, 'n_events': len(segs),
                          'n_events_without_trace': no_trace, 'n_events_truncated': truncated,
                          'peak': float(np.nanmax(mean)), 'peak_time_s': float(t[int(np.nanargmax(mean))]),
                          **{f'{x:+.3f}s': v for x, v in zip(t, mean)}})
        per_group[group] = np.vstack(rows)
    return per_group, t, pd.DataFrame(table)


def _draw(ax, per_group, t, lw):
    for group, mat in per_group.items():
        mu = mat.mean(axis=0)
        sem = mat.std(axis=0, ddof=1) / np.sqrt(mat.shape[0])
        ax.plot(t, mu, color=GROUP_COLOURS[group], lw=lw, label=GROUP_LABELS[group])
        ax.fill_between(t, mu - sem, mu + sem, color=GROUP_COLOURS[group], alpha=0.2, lw=0)


def plot_panel(by_epoch, t, signal, contrasts, values_by_epoch, out_root):
    """Figure 2 v2 h, Nature style at h's box size: per epoch (columns), top the average event, bottom
    the per-animal log amplitude the primary model was fitted on, with that model's Holm-adjusted
    stars vs Ctl for that epoch (no statistic is computed here)."""
    with nature_style():
        fig, axs = plt.subplots(2, len(PANEL_EPOCHS), figsize=panel_figsize('h', LAYOUT_V2_MM),
                                layout='constrained', sharey='row', gridspec_kw={'height_ratios': [1.25, 1]})
        for col, epoch in enumerate(PANEL_EPOCHS):
            ax, axb = axs[0, col], axs[1, col]
            _draw(ax, by_epoch[epoch], t, LINE_PT)
            ax.set_title(EPOCH_TITLES[epoch])
            ax.set_xlabel('Time from event (s)')
            values = {g: values_by_epoch[epoch].loc[values_by_epoch[epoch]['group'] == g,
                                                    'mouse_mean_log_amplitude'].to_numpy()
                      for g in DREADD_DISPLAY_ORDER}
            box_strip(axb, values, DREADD_DISPLAY_ORDER, GROUP_COLOURS, labels=GROUP_LABELS, point_size=4)
            c = contrasts[epoch]
            annotate_pairwise_brackets(axb, values, DREADD_DISPLAY_ORDER,
                                       _precomputed_stat_fn(c['hM3D']['p_holm'], c['hM4D']['p_holm']),
                                       fs=FONT_PT)
        axs[0, 0].set_ylabel(YLABEL[signal].split(' (')[0] + ' (a.u.)')
        axs[1, 0].set_ylabel('Log amplitude')
        axs[0, -1].legend(loc='upper right', handlelength=0.8, borderaxespad=0.1)
        save_paper_panel(fig, out_root)
    plt.close(fig)


def plot_signal_diagnostics(per_group, t, table, signal, epoch, out_dir):
    """Every signal: the group average and the per-animal averages (thin, labelled with n events)."""
    fig, axs = plt.subplots(1, 1 + len(per_group), figsize=(3.2 * (1 + len(per_group)), 3.0),
                            sharey=True, squeeze=False)
    _draw(axs[0, 0], per_group, t, 1.4)
    axs[0, 0].set_title(f'{signal}: group mean ± s.e.m. (animals)')
    axs[0, 0].legend(fontsize='x-small', frameon=False)
    colours = plt.get_cmap('tab10').colors
    for ax, (group, mat) in zip(axs[0, 1:], per_group.items()):
        recs = table[table['group'] == group].reset_index(drop=True)
        for i, row in enumerate(mat):
            ax.plot(t, row, color=colours[i % 10], lw=0.8,
                    label=f"{recs.loc[i, 'mouse']} (n={recs.loc[i, 'n_events']})")
        ax.plot(t, mat.mean(axis=0), color=GROUP_COLOURS[group], lw=2.2, label='mean')
        ax.set_title(f'{GROUP_LABELS[group]} ({mat.shape[0]} animals)')
        ax.legend(fontsize='xx-small', frameon=False)
    for ax in axs[0]:
        ax.set_xlabel('Time from event (s)')
    axs[0, 0].set_ylabel(YLABEL[signal])
    fig.suptitle(f'Average event, {EPOCH_TITLES[epoch]}, {signal}, aligned on the S event peak',
                 size='medium')
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, f'{PANEL_ROOT}_{signal}_{epoch}.png'))
    plt.close(fig)


def draw_event_waveforms(PLOTS_DIR, mice_per_group, TFC_cond, mapping='full', thres=None):
    """All three signals' tables and diagnostics per epoch, and the Figure 2 v2 h panel from
    PANEL_SIGNAL."""
    out_dir = os.path.join(PLOTS_DIR, OUT_SUBDIR)
    ensure_dirs(out_dir)
    _copy_analysis_methods_template(METHODS_FILENAME, out_dir)
    df_runs = build_run_structure_table(mice_per_group, TFC_cond, PANEL_EPOCHS, tfc_epoch_frames_fn(),
                                        mapping=mapping, thres=thres)
    contrasts = {e: {g: paper_contrast(PLOTS_DIR, 'amplitude', e, g) for g in ('hM3D', 'hM4D')}
                 for e in PANEL_EPOCHS}
    values_by_epoch = {e: trace_table(PLOTS_DIR, mice_per_group, epoch=e) for e in PANEL_EPOCHS}
    tables, panel_curves, t_panel = [], {}, None
    for signal in WINDOWS_S:
        for epoch in PANEL_EPOCHS:
            runs = df_runs[df_runs['epoch'] == epoch]
            if runs.empty:
                raise RuntimeError(f'event_waveform: no {epoch} events.')
            per_group, t, table = per_animal_waveforms(TFC_cond, runs, signal, mapping)
            tables.append(table.assign(epoch=epoch))
            plot_signal_diagnostics(per_group, t, table, signal, epoch, out_dir)
            if signal == PANEL_SIGNAL:
                panel_curves[epoch], t_panel = per_group, t
    plot_panel(panel_curves, t_panel, PANEL_SIGNAL, contrasts, values_by_epoch,
               os.path.join(out_dir, PANEL_ROOT))
    pd.concat(tables, ignore_index=True).to_csv(
        os.path.join(out_dir, f'{PANEL_ROOT}_per_animal.csv'), index=False)
