"""Shock-motion check: is the event-aligned response around the shock movement or calcium?

Diagnostic for Figure 2k (epoch_modulation panel M), not an analysis of its own. The YrA traces
swing fast and in both directions during the 2 s US, where the animal jumps; YrA is Minian's raw
residual, so brain movement the motion correction did not remove can enter it. This sets the
rigid motion-correction shifts Minian estimated (``motion.zarr``, exported per session by
``scripts/export_tfc_cond_motion.py``) against the activity, in exactly the frames the event-
proximal index uses (``epoch_modulation._index_window``: shock = the 2 s US, post-shock = 3 s
from shock offset, baseline = the trial's pre-tone window), per mouse and trial.

Motion is the frame-to-frame displacement of the estimated shift (pixels per frame). It sees
in-plane movement only: z-motion and non-rigid deformation are not in a rigid shift, so a clean
motion trace does not rule out a motion artifact; a motion trace that tracks the activity does
rule one in.

Everything here is descriptive. The unit is the mouse; there is no test.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from caban.utilities import CACHE_VOLUME, MINISCOPE_FPS
from caban.session_queue import session_tail
from caban.epoch_analysis import get_epoch_frames
from caban.single_unit_common import (GROUP_COLOURS, GROUP_LABELS, DREADD_DISPLAY_ORDER,
                                      ensure_dirs, save_fig, write_text)
from caban.epoch_modulation import (
    resolve_shared_cells, _standardized_trace, retained_trials, _index_window,
    _event_aligned_trial_traces, _shock_to_post_shock_frames, _window_frames, BASELINE_EPOCH, EVENT_PROXIMAL_WINDOW_S,
    EVENT_PROXIMAL_PRE_S, EVENT_PROXIMAL_POST_S, EVENT_PROXIMAL_DIRNAME)

# The export's keys are session_tail() with '/' written as this (an .npz key is a file name).
MOTION_KEY_SEP = '__'
MOTION_EXPORT_PATH = os.path.join(CACHE_VOLUME, 'caban_cache', 'shock_motion', 'tfc_cond_motion.npz')
SHOCK_MOTION_DIRNAME = 'shock_motion_check'
CHECK_WINDOWS = ('shock', 'post_shock')


def load_motion_shifts(path=MOTION_EXPORT_PATH):
    """{session_tail: (n_frames, 2) rigid shift in pixels} from the export."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f'{path} not found: run scripts/export_tfc_cond_motion.py on the '
                                f'Razer and copy its output here.')
    with np.load(path) as npz:
        return {k.replace(MOTION_KEY_SEP, '/'): npz[k] for k in npz.files}


def motion_speed(shifts):
    """Frame-to-frame displacement of the rigid shift, pixels per frame (frame 0: NaN)."""
    return np.r_[np.nan, np.linalg.norm(np.diff(shifts, axis=0), axis=1)]


def _window_mean(x, lo, hi, what):
    vals = x[..., lo:hi]
    if not np.isfinite(vals).all():
        raise RuntimeError(f'shock-motion check: non-finite values in {what} [{lo}, {hi}).')
    return vals.mean(axis=-1)


def session_motion_speed(session, speed_full):
    """The motion speed in the session's analysis frames.

    Minian's frames are the imaged timestamp rows (sessions.py checks C_full against them); the
    analysis then cuts C/S/YrA to the experiment, [miniscope_exp_fnum[start], miniscope_exp_fnum[stop]),
    and every epoch frame (shock onsets included) counts from that start. So analysis frame i is
    Minian frame i + miniscope_exp_fnum[start]. A re-timed session has no such mapping and fails.
    """
    if session.miniscope_retimed:
        raise RuntimeError(f'shock-motion check: {session.mouse} {session.session_type} was re-timed '
                           f'onto another frame grid; its frames are not Minian frames.')
    if speed_full.size != session.n_timestamp_rows_imaged:
        raise RuntimeError(f'shock-motion check: {session.mouse} motion has {speed_full.size} frames, '
                           f'the session {session.n_timestamp_rows_imaged} imaged frames (= Minian C).')
    start = session.miniscope_exp_fnum[session.start_idx]
    stop = session.miniscope_exp_fnum[session.stop_idx]
    return speed_full[start:stop]


def mouse_shock_motion(session, signal, speed_full, mapping='full', with_crossreg=None):
    """Per-trial table and shock-aligned traces (activity and motion) for one mouse."""
    cell_ids, _ = resolve_shared_cells(session, mapping=mapping, with_crossreg=with_crossreg)
    z = _standardized_trace(session, signal, cell_ids)
    n_frames = z.shape[1]
    speed = session_motion_speed(session, speed_full)
    if speed.size != n_frames:
        raise RuntimeError(f'shock-motion check: {session.mouse} motion in the experiment has '
                           f'{speed.size} frames, {signal} has {n_frames}.')
    trials, _ = retained_trials(session)

    rows = []
    for t in trials:
        base_on, base_off = get_epoch_frames(session, BASELINE_EPOCH, t)
        base_act = _window_mean(z, base_on, base_off, 'baseline activity')
        row = {'mouse': session.mouse, 'trial': int(t) + 1,
               'motion_baseline': float(_window_mean(speed, base_on, base_off, 'baseline motion'))}
        for epoch in CHECK_WINDOWS:
            lo, hi = _index_window(session, epoch, t, EVENT_PROXIMAL_WINDOW_S, n_frames)
            row[f'motion_{epoch}'] = float(_window_mean(speed, lo, hi, f'{epoch} motion'))
            row[f'activity_{epoch}'] = float(
                (_window_mean(z, lo, hi, f'{epoch} activity') - base_act).mean())
        rows.append(row)

    activity_trials, time_axis = _event_aligned_trial_traces(z, session, trials, 'shock')
    pre_f, post_f = _window_frames(EVENT_PROXIMAL_PRE_S), _window_frames(EVENT_PROXIMAL_POST_S)
    motion_trials = np.stack([speed[o - pre_f:o + post_f] for o in
                              (get_epoch_frames(session, 'shock', t)[0] for t in trials)])
    shock_frames = _shock_to_post_shock_frames(session, trials)
    return (pd.DataFrame(rows), activity_trials.mean(axis=0), motion_trials.mean(axis=0), time_axis,
            shock_frames)


def collect_shock_motion(mice_per_group, sessions, signal, shifts, mapping='full',
                         crossreg_to_use=None):
    tables, traces, time_axis, shock_frames = [], {}, None, []
    for group in DREADD_DISPLAY_ORDER:
        for mouse in mice_per_group.get(group, []):
            if mouse not in sessions:
                raise KeyError(f'shock-motion check: no conditioning session for mouse {mouse!r}')
            session = sessions[mouse]
            key = session_tail(session.dpath)
            if key not in shifts:
                raise KeyError(f'shock-motion check: no motion shifts for {key} in the export.')
            wc = crossreg_to_use[mouse] if (crossreg_to_use is not None and mapping != 'full') else None
            table, act, mot, time_axis, frames = mouse_shock_motion(
                session, signal, motion_speed(shifts[key]), mapping=mapping, with_crossreg=wc)
            shock_frames.append(frames)
            table.insert(1, 'group', group)
            tables.append(table)
            traces.setdefault(group, []).append((mouse, act, mot))
    # The shading's boundary: shock duration, the same on every trial within one frame (checked as
    # in epoch_modulation.collect_event_aligned_traces).
    shock_frames = np.concatenate(shock_frames)
    if np.ptp(shock_frames) > 1:
        raise RuntimeError(f'shock-motion check: shock duration ranges {shock_frames.min()}-'
                           f'{shock_frames.max()} frames across trials.')
    return (pd.concat(tables, ignore_index=True), traces, time_axis,
            float(np.median(shock_frames)) / MINISCOPE_FPS)


def plot_shock_aligned_motion(traces, time_axis, out_path, signal, post_shock_onset_s):
    """Rows: motion, activity; columns: groups. Thin = one mouse, thick = group mean."""
    groups = list(traces)
    fig, axs = plt.subplots(2, len(groups), sharex=True, sharey='row',
                            figsize=(3.2 * len(groups), 4.6), squeeze=False)
    mouse_colours = plt.get_cmap('tab10').colors
    for col, group in enumerate(groups):
        for row, (label, idx) in enumerate((('Motion (px/frame)', 2), (f'{signal} (s.d.)', 1))):
            ax = axs[row, col]
            ax.axvspan(0.0, post_shock_onset_s, color='0.80', lw=0, zorder=0)
            ax.axvspan(post_shock_onset_s, post_shock_onset_s + EVENT_PROXIMAL_WINDOW_S,
                       color='0.92', lw=0, zorder=0)
            ax.axvline(0.0, color='k', lw=0.6)
            ax.axvline(post_shock_onset_s, color='k', lw=0.6, ls='--')
            mat = np.stack([m[idx] for m in traces[group]])
            for i, m in enumerate(traces[group]):
                ax.plot(time_axis, m[idx], color=mouse_colours[i % 10], lw=0.6, alpha=0.85,
                        label=m[0])
            ax.plot(time_axis, mat.mean(axis=0), color=GROUP_COLOURS[group], lw=1.8,
                    label=f'{GROUP_LABELS[group]} mean')
            if row == 0:
                ax.set_title(f'{GROUP_LABELS[group]} (n = {mat.shape[0]})')
                ax.legend(loc='upper right', fontsize='x-small', frameon=False, ncol=2)
            if col == 0:
                ax.set_ylabel(label)
    for ax in axs[-1]:
        ax.set_xlabel('Time from shock onset (s)')
    fig.suptitle(f'Shock-aligned motion (rigid shift) and {signal}; darker = shock window, '
                 f'lighter = post-shock window', size='medium')
    fig.tight_layout()
    save_fig(fig, out_path)
    plt.close(fig)


def plot_motion_vs_activity(table, out_path, signal):
    """Per mouse x trial: motion in each window (minus baseline) against the activity index."""
    fig, axs = plt.subplots(1, len(CHECK_WINDOWS), figsize=(3.4 * len(CHECK_WINDOWS), 3.2),
                            squeeze=False)
    for ax, epoch in zip(axs[0], CHECK_WINDOWS):
        x = table[f'motion_{epoch}'] - table['motion_baseline']
        for group in DREADD_DISPLAY_ORDER:
            g = table['group'] == group
            ax.scatter(x[g], table.loc[g, f'activity_{epoch}'], s=16, alpha=0.85,
                       color=GROUP_COLOURS[group], edgecolor='k', linewidth=0.3,
                       label=GROUP_LABELS[group])
        ax.axhline(0, color='0.7', lw=0.5)
        ax.axvline(0, color='0.7', lw=0.5)
        ax.set_title({'shock': 'Shock (2 s US)', 'post_shock': 'Post-shock (3 s)'}[epoch])
        ax.set_xlabel('Motion − baseline (px/frame)')
    axs[0, 0].set_ylabel(f'{signal} index, trial (s.d.)')
    axs[0, 0].legend(fontsize='x-small', frameon=False)
    fig.suptitle('One point per mouse × trial', size='small')
    fig.tight_layout()
    save_fig(fig, out_path)
    plt.close(fig)


def plot_trial_course_per_animal(table, out_path, signal):
    """Per-trial index in the shock and post-shock windows, one line per animal (labelled), group mean
    thick; one column per group. Shows which animals carry a group's across-trial trend."""
    groups = [g for g in DREADD_DISPLAY_ORDER if (table['group'] == g).any()]
    fig, axs = plt.subplots(len(CHECK_WINDOWS), len(groups), sharex=True, sharey='row',
                            figsize=(3.3 * len(groups), 2.6 * len(CHECK_WINDOWS)), squeeze=False)
    mouse_colours = plt.get_cmap('tab10').colors
    for col, group in enumerate(groups):
        sub = table[table['group'] == group]
        for row, epoch in enumerate(CHECK_WINDOWS):
            ax = axs[row, col]
            ax.axhline(0, color='0.6', lw=0.6, ls='--', zorder=0)
            for i, (mouse, m) in enumerate(sub.groupby('mouse', sort=True)):
                m = m.sort_values('trial')
                ax.plot(m['trial'], m[f'activity_{epoch}'], marker='o', ms=3, lw=0.9,
                        color=mouse_colours[i % 10], alpha=0.9, label=mouse)
            mean = sub.groupby('trial')[f'activity_{epoch}'].mean()
            ax.plot(mean.index, mean.values, color=GROUP_COLOURS[group], lw=2.4, zorder=4,
                    label=f'{GROUP_LABELS[group]} mean')
            if row == 0:
                ax.set_title(f'{GROUP_LABELS[group]} (n = {sub["mouse"].nunique()})')
                ax.legend(fontsize='x-small', frameon=False, ncol=2)
            if col == 0:
                ax.set_ylabel({'shock': f'Shock (2 s US)\n{signal} index', 'post_shock':
                               f'Post-shock (3 s)\n{signal} index'}[epoch])
    for ax in axs[-1]:
        ax.set_xlabel('Conditioning trial')
        ax.set_xticks(sorted(table['trial'].unique()))
    fig.suptitle(f'Shock and post-shock response by trial, per animal ({signal}; cell-mean index, '
                 f'trial minus its pre-tone mean)', size='medium')
    fig.tight_layout()
    save_fig(fig, out_path)
    plt.close(fig)


def format_report(table, signal):
    lines = [f'Shock-motion check ({signal}) — descriptive; the unit is the mouse', '=' * 60, '',
             'Motion = frame-to-frame displacement of Minian\'s rigid motion-correction shift',
             '(px/frame), averaged over the same frames as the event-proximal index. In-plane only.',
             '']
    per_mouse = table.groupby(['group', 'mouse'], observed=True).mean(numeric_only=True)
    lines.append('Per group, mean (min-max) over mice of the mouse means:')
    for epoch in ('baseline',) + CHECK_WINDOWS:
        lines.append(f'  motion {epoch}:')
        for group in DREADD_DISPLAY_ORDER:
            v = per_mouse.loc[group, f'motion_{epoch}']
            lines.append(f'    {GROUP_LABELS[group]}: {v.mean():.3f} ({v.min():.3f}-{v.max():.3f})')
    lines += ['', 'Does activity track motion? Spearman rho, motion − baseline vs activity index:']
    for epoch in CHECK_WINDOWS:
        dx = per_mouse[f'motion_{epoch}'] - per_mouse['motion_baseline']
        rho_m, p_m = stats.spearmanr(dx, per_mouse[f'activity_{epoch}'])
        within = []
        for _, m in table.groupby('mouse'):
            r, _ = stats.spearmanr(m[f'motion_{epoch}'] - m['motion_baseline'], m[f'activity_{epoch}'])
            within.append(r)
        within = np.array(within)
        lines.append(f'  {epoch}: across mice rho = {rho_m:+.2f} (P = {p_m:.2f}, n = {len(per_mouse)}); '
                     f'within mouse across trials, mean rho = {np.nanmean(within):+.2f} '
                     f'(positive in {int((within > 0).sum())}/{np.isfinite(within).sum()} mice)')
    lines += ['', 'Per mouse (mean over trials):',
              per_mouse.round(3).to_string()]
    return '\n'.join(lines) + '\n'


def run_shock_motion_check(PLOTS_DIR, mice_per_group, TFC_cond, signal, motion_path=MOTION_EXPORT_PATH, mapping='full', crossreg_to_use=None):
    """Write the check into ``<PLOTS_DIR>/epoch_modulation/<signal>/event_proximal/shock_motion_check/``."""
    out_dir = os.path.join(PLOTS_DIR, 'epoch_modulation', signal, EVENT_PROXIMAL_DIRNAME,
                           SHOCK_MOTION_DIRNAME)
    ensure_dirs(out_dir)
    shifts = load_motion_shifts(motion_path)
    table, traces, time_axis, post_shock_onset_s = collect_shock_motion(mice_per_group, TFC_cond, signal, shifts,
                                                    mapping=mapping, crossreg_to_use=crossreg_to_use)
    table.to_csv(os.path.join(out_dir, 'shock_motion_per_trial.csv'), index=False)
    plot_shock_aligned_motion(traces, time_axis, os.path.join(out_dir, 'shock_aligned_motion.png'),
                              signal, post_shock_onset_s)
    plot_motion_vs_activity(table, os.path.join(out_dir, 'motion_vs_activity.png'), signal)
    plot_trial_course_per_animal(table, os.path.join(out_dir, 'shock_by_trial_per_animal.png'), signal)
    report = format_report(table, signal)
    write_text(os.path.join(out_dir, 'shock_motion_report.txt'), report)
    print(report)
    return table
