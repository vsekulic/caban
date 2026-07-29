"""Event-locked per-cell responsiveness (CS / trace / US) + peri-event heatmaps.

This is the single-cell successor to the "fraction of active cells" metric. Instead
of "active = >=1 event anywhere in the recording", each cell is tested for whether
its activity in a task epoch (tone/CS, trace interval, shock/US, post-shock, or the
recall tone) differs from its own pre-tone baseline. Cells are then summarised per
mouse as the FRACTION that are responsive and the response MAGNITUDE among
responders, and compared across the three DREADD groups. Because the manipulation
plausibly reorganises which cells respond rather than bulk rate, this event-locked,
per-cell view is where a group difference should surface.

Significance uses a circular-shift shuffle null on each cell's deconvolved activity
trace (as prototyped in debug_PSTH), which is robust to the low trial count
(5 conditioning trials, fewer for some mice) — a per-trial signed-rank test cannot
even reach p<0.05 at n=5. Per-cell p-values are FDR-corrected within mouse.

Reuses epoch frame windows from caban.epoch_analysis (get_epoch_frames /
get_testb_epoch_frames / get_testa_epoch_frames), the violin+stat panel from
caban.analysis (_draw_violin_triplet), and the shared group styling / IO in
caban.single_unit_common.
"""
import os

import time

import numpy as np
import matplotlib.pyplot as plt

from caban.utilities import MINISCOPE_FPS
from caban.epoch_analysis import (
    get_epoch_frames, get_testb_epoch_frames, get_testa_epoch_frames,
    TESTA_PSEUDO_TONE_ONSETS_S,
)
from caban.analysis import _draw_violin_triplet, do_pairwise_holm_plot
from caban.decoder import _copy_analysis_methods_template
from caban.single_unit_common import (
    GROUP_ORDER, GROUP_COLOURS, GROUP_LABELS,
    ensure_dirs, write_text, save_fig, shuffle_mask_contrast, shuffle_peak_contrast,
    get_mapping_signal, bracket_ylim,
)

# Baseline epoch and the response epochs tested against it, per session family.
_ENCODING_BASELINE = 'pre_tone'
_ENCODING_RESPONSE_EPOCHS = ['tone', 'trace', 'shock', 'post_shock']
_TESTB_BASELINE = 'pre_tone'
_TESTB_RESPONSE_EPOCHS = ['tone', 'post_tone']
_TESTA_BASELINE = 'pre_pseudo_tone'
_TESTA_RESPONSE_EPOCHS = ['pseudo_tone', 'post_pseudo_tone']

_FAMILY_CONFIG = {
    'encoding':     dict(baseline=_ENCODING_BASELINE, response=_ENCODING_RESPONSE_EPOCHS, recall_type=None),
    'recall_testB': dict(baseline=_TESTB_BASELINE,    response=_TESTB_RESPONSE_EPOCHS,    recall_type='testb'),
    'recall_testA': dict(baseline=_TESTA_BASELINE,    response=_TESTA_RESPONSE_EPOCHS,    recall_type='testa'),
}


def _epoch_frames(session, epoch, trial_idx, recall_type, pre_tone_duration_s=35.0):
    """Dispatch to the correct epoch-frame accessor for encoding vs recall.

    pre_tone_duration_s sets the baseline (pre-tone / pre-pseudo-tone) window length;
    it is ignored for the non-baseline epochs."""
    if recall_type is None:
        return get_epoch_frames(session, epoch, trial_idx, pre_tone_duration_s=pre_tone_duration_s)
    if recall_type == 'testb':
        return get_testb_epoch_frames(session, epoch, trial_idx, pre_tone_duration_s=pre_tone_duration_s)
    if recall_type == 'testa':
        return get_testa_epoch_frames(session, epoch, trial_idx, pre_tone_duration_s=pre_tone_duration_s)
    raise ValueError(f'Unknown recall_type {recall_type!r}')


def _n_trials(session, recall_type):
    if recall_type == 'testa':
        return len(TESTA_PSEUDO_TONE_ONSETS_S)
    return len(session.tone_onsets)


def _epoch_windows(session, epoch, recall_type, n_frames, pre_tone_duration_s=35.0, onset_window_s=None):
    """List of (onset, offset) frame pairs, one per trial, for an epoch.

    onset_window_s (if set) truncates each trial's window to the first N seconds from
    onset (onset-aligned response window)."""
    fps = MINISCOPE_FPS
    windows = []
    for t in range(_n_trials(session, recall_type)):
        result = _epoch_frames(session, epoch, t, recall_type, pre_tone_duration_s=pre_tone_duration_s)
        if result is None:
            continue
        onset, offset = result
        if onset_window_s is not None:
            offset = min(offset, onset + int(round(onset_window_s * fps)))
        offset = min(offset, n_frames)
        if offset > onset:
            windows.append((onset, offset))
    return windows


def _epoch_mask(session, epoch, recall_type, n_frames, pre_tone_duration_s=35.0, onset_window_s=None):
    """Boolean frame mask over all trials for one epoch (union of per-trial windows)."""
    mask = np.zeros(n_frames, dtype=float)
    for onset, offset in _epoch_windows(session, epoch, recall_type, n_frames,
                                        pre_tone_duration_s=pre_tone_duration_s,
                                        onset_window_s=onset_window_s):
        mask[onset:offset] = 1.0
    return mask


def classify_cells(sig, session, family, metric, n_shuffles=1000, seed=0,
                   baseline_s=10.0, onset_window_s=5.0, peak_smooth_s=1.0):
    """Per-cell event-locked responsiveness for every response epoch, under one metric.

    sig is the activity matrix (cells x frames), the denoised calcium trace C by
    default. The reference is the baseline epoch (pre-tone / pre-pseudo-tone), whose
    window length is baseline_s. The response statistic depends on metric:
      - 'peak'         : peak of the smoothed trace within the full epoch minus
                         baseline (phasic-response sensitive; peak_smooth_s smoothing).
      - 'onset_window' : mean over the first onset_window_s of the epoch minus baseline.
      - 'epoch_mean'   : mean over the whole epoch minus baseline.
    Significance is the z-scored circular-shift shuffle null, FDR-corrected across cells.

    Returns dict: epoch -> shuffle result dict.
    """
    cfg = _FAMILY_CONFIG[family]
    n_frames = sig.shape[1]

    base_mask = _epoch_mask(session, cfg['baseline'], cfg['recall_type'], n_frames,
                            pre_tone_duration_s=baseline_s)
    if base_mask.sum() == 0:
        raise RuntimeError(f'Baseline epoch {cfg["baseline"]!r} produced no frames for {family} '
                           f'(baseline_s={baseline_s}).')

    results = {}
    for epoch in cfg['response']:
        if metric == 'peak':
            windows = _epoch_windows(session, epoch, cfg['recall_type'], n_frames)
            if not windows:
                continue
            results[epoch] = shuffle_peak_contrast(
                sig, windows, base_mask,
                smooth_sigma_frames=peak_smooth_s * MINISCOPE_FPS,
                n_shuffles=n_shuffles, seed=seed)
        else:
            ow = onset_window_s if metric == 'onset_window' else None
            epoch_mask = _epoch_mask(session, epoch, cfg['recall_type'], n_frames, onset_window_s=ow)
            if epoch_mask.sum() == 0:
                continue
            results[epoch] = shuffle_mask_contrast(sig, epoch_mask, base_mask,
                                                   n_shuffles=n_shuffles, seed=seed)
    return results


def _peri_event_matrix(sig, session, anchor_epoch, recall_type, pre_s, post_s):
    """Per-cell trial-averaged activity (denoised C by default) aligned to an epoch
    onset.

    Returns (matrix cells x window_frames, time_axis_s). Cells are z-scored across
    the window for cross-cell comparability in the heatmap."""
    fps = MINISCOPE_FPS
    pre_f = int(round(pre_s * fps))
    post_f = int(round(post_s * fps))
    win = pre_f + post_f
    n_frames = sig.shape[1]

    per_trial = []
    for t in range(_n_trials(session, recall_type)):
        result = _epoch_frames(session, anchor_epoch, t, recall_type)
        if result is None:
            continue
        onset = result[0]
        lo, hi = onset - pre_f, onset + post_f
        if lo < 0 or hi > n_frames:
            continue
        per_trial.append(sig[:, lo:hi])
    if not per_trial:
        return None, None
    avg = np.mean(np.stack(per_trial, axis=0), axis=0)     # (n_cells, win)
    mu = avg.mean(axis=1, keepdims=True)
    sd = avg.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    z = (avg - mu) / sd
    time_axis = (np.arange(win) - pre_f) / fps
    return z, time_axis


def _plot_peri_event_heatmaps(group_mats, time_axis, anchor_label, out_path):
    """One sorted heatmap per group (cells pooled within group), shared colour scale."""
    fig, axs = plt.subplots(1, len(GROUP_ORDER), figsize=(4 * len(GROUP_ORDER), 4.2), sharey=False)
    if len(GROUP_ORDER) == 1:
        axs = np.array([axs])
    vmax = 2.0
    onset_frac = np.argmin(np.abs(time_axis)) / len(time_axis)
    for ax, group in zip(axs.flat, GROUP_ORDER):
        mat = group_mats.get(group)
        if mat is None or mat.size == 0:
            ax.set_visible(False)
            continue
        post_onset = time_axis >= 0
        order = np.argsort(-mat[:, post_onset].mean(axis=1))
        ax.imshow(mat[order], aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                  extent=[time_axis[0], time_axis[-1], mat.shape[0], 0])
        ax.axvline(0, color='k', lw=1.0)
        ax.set_title(f'{GROUP_LABELS[group]} (n={mat.shape[0]})', size='medium')
        ax.set_xlabel(f'Time from {anchor_label} onset (s)')
    axs.flat[0].set_ylabel('Cell (sorted by response)')
    fig.suptitle(f'{anchor_label}-aligned activity (z-scored per cell)')
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.13, top=0.86, wspace=0.15)
    save_fig(fig, out_path)
    plt.close(fig)


def _plot_responsiveness_panels(arr, response_epochs, family, metric, ylabel, filename, auto_close):
    """One row of violin+strip panels (one per epoch) across groups, with pairwise
    Welch+Holm significance brackets."""
    fig, axs = plt.subplots(1, len(response_epochs), figsize=(3 * len(response_epochs), 4), sharey='row')
    if len(response_epochs) == 1:
        axs = np.array([axs])
    ylim = bracket_ylim(arr, len(response_epochs))
    for i, (ax, epoch) in enumerate(zip(axs.flat, response_epochs)):
        _draw_violin_triplet(ax, arr, i, GROUP_ORDER, GROUP_COLOURS,
                             stat_fn=do_pairwise_holm_plot, ylim=ylim)
        ax.set_xticks(range(len(GROUP_ORDER)))
        ax.set_xticklabels([GROUP_LABELS[g] for g in GROUP_ORDER], size='medium')
        ax.set_title(epoch, size='medium')
    axs.flat[0].set_ylabel(ylabel)
    fig.suptitle(f'{ylabel} — {family} ({metric})')
    fig.subplots_adjust(left=0.08, bottom=0.1, right=0.98, top=0.85, wspace=0.2)
    save_fig(fig, filename)
    if auto_close:
        plt.close(fig)


def run_event_locked_responsiveness(PLOTS_DIR, mice_per_group, sessions, family,
                                    crossreg_to_use=None, mapping='full', signal='C',
                                    n_shuffles=1000, seed=0,
                                    response_metrics=('peak', 'onset_window'),
                                    baseline_s=10.0, onset_window_s=5.0, peak_smooth_s=1.0,
                                    auto_close=True):
    """Run the event-locked responsiveness analysis for one session family.

    PLOTS_DIR      : output root
    mice_per_group : dict group -> [mouse]
    sessions       : dict mouse -> session (TFC_cond, Test_A or Test_B)
    family         : 'encoding' | 'recall_testB' | 'recall_testA'
    crossreg_to_use: dict mouse -> CrossRegMapping (needed if mapping != 'full')
    mapping        : crossreg mapping string ('full' = all cells)
    signal         : activity trace attribute ('C' denoised calcium, default; 'S' spikes)
    response_metrics : which response statistics to compute (each gets its own panels/
                     tables): 'peak', 'onset_window', 'epoch_mean'
    baseline_s     : pre-tone baseline window length (s)
    onset_window_s : onset-aligned response window length (s) for the 'onset_window' metric
    peak_smooth_s  : gaussian smoothing (s) before peak detection for the 'peak' metric
    """
    cfg = _FAMILY_CONFIG[family]
    response_epochs = cfg['response']
    t_start = time.time()
    tag = f'[event-locked/{family}]'
    print(f'{tag} start — signal={signal}, metrics={tuple(response_metrics)}, '
          f'n_shuffles={n_shuffles}, baseline_s={baseline_s}, onset_window_s={onset_window_s}, '
          f'epochs={response_epochs}', flush=True)

    out_dir = os.path.join(PLOTS_DIR, 'event_locked_responsiveness', family)
    stats_dir = os.path.join(out_dir, 'stats')
    tables_dir = os.path.join(out_dir, 'tables')
    ensure_dirs(out_dir, stats_dir, tables_dir)
    _copy_analysis_methods_template('event_locked_responsiveness_methods.txt', out_dir)

    # Resolve each mouse's activity matrix once (reused across metrics + heatmaps).
    mouse_sig = {}
    for group in GROUP_ORDER:
        for mouse in mice_per_group.get(group, []):
            if mouse not in sessions:
                continue
            session = sessions[mouse]
            wc = crossreg_to_use[mouse] if (crossreg_to_use is not None and mapping != 'full') else None
            sig = get_mapping_signal(session, mapping, signal_attr=signal, with_crossreg=wc)
            if sig.shape[0] == 0:
                raise RuntimeError(f'No cells for mapping {mapping} mouse {mouse} ({family}).')
            mouse_sig[(group, mouse)] = sig
    print(f'{tag} resolved signals for {len(mouse_sig)} mice '
          f'({", ".join(g+":"+str(sum(1 for (gg,_) in mouse_sig if gg==g)) for g in GROUP_ORDER)})', flush=True)

    # ── Peri-event heatmaps (metric-independent) ──
    print(f'{tag} building peri-event heatmaps...', flush=True)
    anchor_shock = 'shock' if family == 'encoding' else None
    tone_mats = {g: [] for g in GROUP_ORDER}
    shock_mats = {g: [] for g in GROUP_ORDER}
    tone_axis = shock_axis = None
    for (group, mouse), sig in mouse_sig.items():
        session = sessions[mouse]
        tm, tax = _peri_event_matrix(sig, session, cfg['response'][0], cfg['recall_type'],
                                     pre_s=35.0, post_s=40.0)
        if tm is not None:
            tone_mats[group].append(tm)
            tone_axis = tax
        if anchor_shock is not None:
            sm, sax = _peri_event_matrix(sig, session, anchor_shock, cfg['recall_type'],
                                         pre_s=20.0, post_s=40.0)
            if sm is not None:
                shock_mats[group].append(sm)
                shock_axis = sax

    def _pool(mats):
        return {g: (np.concatenate(mats[g], axis=0) if mats[g] else np.empty((0, 0))) for g in GROUP_ORDER}

    if tone_axis is not None:
        _plot_peri_event_heatmaps(_pool(tone_mats), tone_axis, cfg['response'][0],
                                  os.path.join(out_dir, f'heatmap-{cfg["response"][0]}-aligned-{family}.png'))
    if anchor_shock is not None and shock_axis is not None:
        _plot_peri_event_heatmaps(_pool(shock_mats), shock_axis, anchor_shock,
                                  os.path.join(out_dir, f'heatmap-{anchor_shock}-aligned-{family}.png'))
    print(f'{tag} heatmaps done ({time.time() - t_start:.1f}s elapsed)', flush=True)

    # ── Per-metric responsiveness ──
    n_mice = len(mouse_sig)
    all_results = {}
    for metric in response_metrics:
        t_metric = time.time()
        print(f'{tag} metric {metric!r}: classifying {n_mice} mice '
              f'x {len(response_epochs)} epochs...', flush=True)
        frac = {g: [] for g in GROUP_ORDER}
        magn = {g: [] for g in GROUP_ORDER}
        mouse_order = {g: [] for g in GROUP_ORDER}
        table_rows = []
        done = 0
        for group in GROUP_ORDER:
            for mouse in mice_per_group.get(group, []):
                if (group, mouse) not in mouse_sig:
                    continue
                sig = mouse_sig[(group, mouse)]
                t_mouse = time.time()
                res = classify_cells(sig, sessions[mouse], family, metric,
                                     n_shuffles=n_shuffles, seed=seed, baseline_s=baseline_s,
                                     onset_window_s=onset_window_s, peak_smooth_s=peak_smooth_s)
                done += 1
                frac_row, magn_row = [], []
                for epoch in response_epochs:
                    if epoch not in res:
                        frac_row.append(np.nan)
                        magn_row.append(np.nan)
                        continue
                    responsive = res[epoch]['responsive']
                    observed = res[epoch]['observed']
                    n_cells = len(responsive)
                    frac_resp = responsive.sum() / n_cells if n_cells else np.nan
                    mag = float(np.mean(np.abs(observed[responsive]))) if responsive.any() else 0.0
                    frac_row.append(frac_resp)
                    magn_row.append(mag)
                    table_rows.append(f'{mouse},{group},{metric},{epoch},{n_cells},'
                                      f'{int(responsive.sum())},{frac_resp:.4f},{mag:.5f}')
                frac[group].append(frac_row)
                magn[group].append(magn_row)
                mouse_order[group].append(mouse)
                resp_str = ', '.join(f'{ep}:{int(res[ep]["responsive"].sum()) if ep in res else 0}'
                                     for ep in response_epochs)
                print(f'{tag}   [{metric}] {done}/{n_mice} {mouse} ({group}, {sig.shape[0]} cells) '
                      f'responsive[{resp_str}] {time.time() - t_mouse:.1f}s', flush=True)

        frac_arr = {g: np.array(frac[g], dtype=float) for g in GROUP_ORDER}
        magn_arr = {g: np.array(magn[g], dtype=float) for g in GROUP_ORDER}
        pooled_frac = {ep: np.nanmean([v for g in GROUP_ORDER for v in frac_arr[g][:, i]])
                       for i, ep in enumerate(response_epochs)}
        print(f'{tag} metric {metric!r} done in {time.time() - t_metric:.1f}s — '
              f'pooled mean fraction responsive: '
              f'{{{", ".join(f"{ep}:{pooled_frac[ep]:.3f}" for ep in response_epochs)}}}', flush=True)
        _plot_responsiveness_panels(frac_arr, response_epochs, family, metric,
                                    'Fraction of responsive cells',
                                    os.path.join(out_dir, f'fraction_responsive-{metric}-{family}.png'), auto_close)
        _plot_responsiveness_panels(magn_arr, response_epochs, family, metric,
                                    'Response magnitude (|Δ| among responders)',
                                    os.path.join(out_dir, f'response_magnitude-{metric}-{family}.png'), auto_close)
        write_text(os.path.join(tables_dir, f'per_mouse_responsiveness-{metric}-{family}.csv'),
                   'mouse,group,metric,epoch,n_cells,n_responsive,fraction_responsive,magnitude\n'
                   + '\n'.join(table_rows) + '\n')
        all_results[metric] = {'fraction': frac_arr, 'magnitude': magn_arr, 'mouse_order': mouse_order}

    print(f'{tag} ALL DONE — {len(response_metrics)} metric(s) in {time.time() - t_start:.1f}s, '
          f'outputs under {out_dir}', flush=True)
    return all_results
