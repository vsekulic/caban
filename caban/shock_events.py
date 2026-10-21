"""Shock-locked pyramidal events (Figure 2 v2, panel i): per-event amplitude and event rate in 1 s
bins around shock onset, per animal, as group mean +/- s.e.m. across animals.

plans/figure2_redesign_plan.md §5: the event measures the rest of the figure uses, in place of the
YrA population mean of the old panel k. The panel is descriptive and adds no test: its one
comparison is the declared secondary of the primary conditioning model (post-shock per-event
amplitude, hM3D vs control), reported there. METHODS: analysis_methods_templates/shock_events_methods.md.

Events: find_event_runs_ca_S(S, thres) on each conditioning session's whole S (all cells, the
primary analysis's detector and threshold), each event placed at its peak frame, as in
sp_rates_lmm._iter_event_windows. Per animal and bin: event rate = events / (cells x trials x bin
seconds); amplitude = mean over that bin's events of log per-event amplitude (bins without an
event are left empty for that animal, never filled).
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from caban.utilities import MINISCOPE_FPS, find_event_runs_ca_S
from caban.epoch_analysis import get_epoch_frames
from caban.single_unit_common import (GROUP_COLOURS, GROUP_LABELS, DREADD_DISPLAY_ORDER, ensure_dirs,
                                      write_text)
from caban.decoder import _copy_analysis_methods_template
from caban.nature_comm_style import nature_style, save_paper_panel, AXIS_PT, LINE_PT, FONT_PT
from caban.figure2_composite import panel_figsize, LAYOUT_V2_MM

OUT_DIRNAME = 'shock_events'
METHODS_FILENAME = 'shock_events_methods.md'
PANEL_ROOT = 'shock_locked_events'
BIN_S = 1.0
PRE_S, POST_S = 5.0, 10.0
POST_SHOCK_WINDOW_S = 20.0     # the primary model's post-shock window, from shock offset


def mouse_shock_bins(session, mapping='full', thres=None):
    """Per bin: events, summed log amplitude, and exposure (cells x trials x bin s) for one mouse."""
    cell_thres = session.thres if thres is None else thres
    S, _spikes, _peak, _idx = session.get_S_mapping(mapping)
    if S.shape[0] == 0:
        raise RuntimeError(f'shock_events: mouse {session.mouse} has 0 cells for mapping={mapping!r}.')
    events = find_event_runs_ca_S(S, cell_thres)
    peak = np.concatenate([events[i][0] for i in range(S.shape[0])])
    log_amp = np.log(np.concatenate([events[i][1] for i in range(S.shape[0])]))
    edges = np.arange(-PRE_S, POST_S + BIN_S / 2, BIN_S)
    n_events = np.zeros(len(edges) - 1)
    sum_log_amp = np.zeros(len(edges) - 1)
    trials = list(session.periods)
    for t in trials:
        onset = get_epoch_frames(session, 'shock', t)[0]
        lo, hi = onset - int(round(PRE_S * MINISCOPE_FPS)), onset + int(round(POST_S * MINISCOPE_FPS))
        if lo < 0 or hi > S.shape[1]:
            raise RuntimeError(f'shock_events: mouse {session.mouse} trial {t} window [{lo}, {hi}) is '
                               f'outside its {S.shape[1]}-frame recording.')
        rel = (peak - onset) / MINISCOPE_FPS
        sel = (rel >= edges[0]) & (rel < edges[-1])
        idx = np.digitize(rel[sel], edges) - 1
        np.add.at(n_events, idx, 1)
        np.add.at(sum_log_amp, idx, log_amp[sel])
    exposure = S.shape[0] * len(trials) * BIN_S
    centres = edges[:-1] + BIN_S / 2
    return pd.DataFrame({'t': centres, 'n_events': n_events, 'rate': n_events / exposure,
                         'log_amplitude': np.where(n_events > 0, sum_log_amp / np.maximum(n_events, 1),
                                                   np.nan),
                         'n_cells': S.shape[0], 'n_trials': len(trials)})


def collect(mice_per_group, sessions, mapping='full', thres=None):
    rows = []
    for group in DREADD_DISPLAY_ORDER:
        for mouse in mice_per_group.get(group, []):
            if mouse not in sessions:
                raise KeyError(f'shock_events: no conditioning session for mouse {mouse!r}')
            rows.append(mouse_shock_bins(sessions[mouse], mapping, thres).assign(mouse=mouse, group=group))
    return pd.concat(rows, ignore_index=True)


def plot_panel(table, shock_s, out_path):
    """Figure 2 v2 i: event rate (left) and per-event log amplitude (right) around shock onset."""
    with nature_style():
        fig, axs = plt.subplots(1, 2, figsize=panel_figsize('i', LAYOUT_V2_MM), layout='constrained')
        for ax, (col, ylabel) in zip(axs, (('rate', 'Event rate (per cell per s)'),
                                           ('log_amplitude', 'Log per-event amplitude'))):
            ax.axvspan(0, shock_s, color='0.80', lw=0, zorder=0)
            ax.axvspan(shock_s, POST_S, color='0.92', lw=0, zorder=0)
            ax.axvline(0, color='k', lw=AXIS_PT)
            ax.axvline(shock_s, color='k', lw=AXIS_PT, ls='--')
            for g in DREADD_DISPLAY_ORDER:
                wide = table[table['group'] == g].pivot(index='mouse', columns='t', values=col)
                n = wide.notna().sum(axis=0)
                mu = wide.mean(axis=0)
                sem = wide.std(axis=0, ddof=1) / np.sqrt(n)
                ax.plot(mu.index, mu.values, color=GROUP_COLOURS[g], lw=LINE_PT, marker='o', ms=2,
                        label=GROUP_LABELS[g])
                ax.fill_between(mu.index, (mu - sem).values, (mu + sem).values, color=GROUP_COLOURS[g],
                                alpha=0.2, lw=0)
            ax.set_ylabel(ylabel)
            ax.set_xlabel('Time from shock onset (s)')
        axs[0].legend(loc='upper right', handlelength=1.0)
        save_paper_panel(fig, os.path.splitext(out_path)[0])
    plt.close(fig)


def run_shock_events(PLOTS_DIR, mice_per_group, TFC_cond, mapping='full', thres=None):
    out_dir = os.path.join(PLOTS_DIR, OUT_DIRNAME)
    ensure_dirs(out_dir)
    _copy_analysis_methods_template(METHODS_FILENAME, out_dir)
    table = collect(mice_per_group, TFC_cond, mapping, thres)
    shock_frames = [get_epoch_frames(TFC_cond[m], 'post_shock', t)[0] - get_epoch_frames(TFC_cond[m], 'shock', t)[0]
                    for ms in mice_per_group.values() for m in ms for t in TFC_cond[m].periods]
    if np.ptp(shock_frames) > 1:
        raise RuntimeError(f'shock_events: shock duration ranges {min(shock_frames)}-{max(shock_frames)} '
                           f'frames across trials.')
    shock_s = float(np.median(shock_frames)) / MINISCOPE_FPS
    table.to_csv(os.path.join(out_dir, 'shock_locked_bins_per_animal.csv'), index=False)
    plot_panel(table, shock_s, os.path.join(out_dir, PANEL_ROOT + '.png'))
    write_text(os.path.join(out_dir, 'shock_events_summary.txt'),
               'Shock-locked events, group mean over animals per bin (descriptive):\n'
               + table.groupby(['group', 't'])[['rate', 'log_amplitude']].mean().unstack('group')
               .round(4).to_string() + '\n')
    return table
