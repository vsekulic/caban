"""Population coupling / co-activation structure.

A correlational axis independent of single-cell rate: how synchronized the
population is. Even when per-cell event rates and amplitudes match across groups,
the excitatory/inhibitory DREADD manipulation can change how coordinated cells are.
Two readouts per mouse:
  - mean pairwise correlation of binned activity (global synchrony), one scalar,
  - per-cell population coupling (Okun-style): each cell's correlation with the
    leave-one-out population sum, a per-cell distribution.
For the conditioning session, synchrony is also computed restricted to the task
epochs (pre-tone / tone / trace / shock) to test whether coordination is
event-specific.

Reuses the mapping cell subset (get_S_mapping), the epoch frame windows
(get_epoch_frames), the violin panel (caban.analysis._draw_violin_triplet), and the
shared ECDF / mixed-model / IO helpers in caban.single_unit_common.
"""
import os

import numpy as np
import matplotlib.pyplot as plt

from caban.epoch_analysis import get_epoch_frames
from caban.analysis import _draw_violin_triplet, do_pairwise_holm_plot
from caban.decoder import _copy_analysis_methods_template
from caban.single_unit_common import (
    GROUP_ORDER, GROUP_COLOURS, GROUP_LABELS,
    ensure_dirs, write_text, save_fig, get_mapping_signal,
    ecdf_panel, build_cell_records, fit_group_mixed_model, bracket_ylim,
)

_ENCODING_SYNC_EPOCHS = ['pre_tone', 'tone', 'trace', 'shock']


def _binned_matrix(S_sub, bin_width):
    """Bin (cells x frames) into (cells x n_bins) by summing within each bin."""
    n_cells, n_frames = S_sub.shape
    n_bins = n_frames // bin_width
    if n_bins < 2:
        return None
    trimmed = S_sub[:, :n_bins * bin_width]
    return trimmed.reshape(n_cells, n_bins, bin_width).sum(axis=2)


def _active_rows(binned):
    """Rows (cells) with non-zero variance — correlation is undefined otherwise."""
    return np.where(binned.std(axis=1) > 0)[0]


def _mean_pairwise_corr(binned):
    active = _active_rows(binned)
    if active.size < 2:
        return np.nan
    corr = np.corrcoef(binned[active])
    iu = np.triu_indices_from(corr, k=1)
    vals = corr[iu]
    vals = vals[np.isfinite(vals)]
    return float(np.mean(vals)) if vals.size else np.nan


def _population_coupling(binned):
    """Per-cell correlation with the leave-one-out population sum. Returns an array
    over ALL cells (NaN for zero-variance cells)."""
    n_cells = binned.shape[0]
    total = binned.sum(axis=0)
    coupling = np.full(n_cells, np.nan)
    for i in range(n_cells):
        if binned[i].std() == 0:
            continue
        rest = total - binned[i]
        if rest.std() == 0:
            continue
        coupling[i] = np.corrcoef(binned[i], rest)[0, 1]
    return coupling


def _epoch_frame_indices(session, epoch, n_frames):
    idx = []
    for t in range(len(session.tone_onsets)):
        result = get_epoch_frames(session, epoch, t)
        if result is None:
            continue
        onset, offset = result
        offset = min(offset, n_frames)
        if offset > onset:
            idx.append(np.arange(onset, offset))
    return np.concatenate(idx) if idx else np.array([], dtype=int)


def run_population_coupling(PLOTS_DIR, mice_per_group, sessions, family, bin_width,
                            mapping='full', crossreg_to_use=None, signal='C', auto_close=True):
    """Population coupling analysis for one session family. ``signal`` selects the
    activity trace binned for the correlations ('C' denoised calcium, default;
    'S' deconvolved spikes)."""
    out_dir = os.path.join(PLOTS_DIR, 'population_coupling', family)
    stats_dir = os.path.join(out_dir, 'stats')
    tables_dir = os.path.join(out_dir, 'tables')
    ensure_dirs(out_dir, stats_dir, tables_dir)
    _copy_analysis_methods_template('population_coupling_methods.txt', out_dir)

    synchrony = {g: [] for g in GROUP_ORDER}            # per mouse, whole session
    coupling_by_group = {g: {} for g in GROUP_ORDER}    # per mouse per-cell
    epoch_sync = {g: [] for g in GROUP_ORDER}           # per mouse, per epoch (encoding)
    table_rows = []
    do_epochs = (family == 'encoding')

    for group in GROUP_ORDER:
        for mouse in mice_per_group.get(group, []):
            if mouse not in sessions:
                continue
            session = sessions[mouse]
            wc = crossreg_to_use[mouse] if (crossreg_to_use is not None and mapping != 'full') else None
            sig = get_mapping_signal(session, mapping, signal_attr=signal, with_crossreg=wc)
            if sig.shape[0] < 2:
                raise RuntimeError(f'Need >=2 cells for coupling; mouse {mouse} ({family}) has {sig.shape[0]}.')

            binned = _binned_matrix(sig, bin_width)
            if binned is None:
                raise RuntimeError(f'Too few bins for mouse {mouse} ({family}).')
            sync = _mean_pairwise_corr(binned)
            coupling = _population_coupling(binned)
            synchrony[group].append([sync])
            coupling_by_group[group][mouse] = coupling[np.isfinite(coupling)]
            table_rows.append(f'{mouse},{group},{sig.shape[0]},{sync:.5f},'
                              f'{np.nanmean(coupling):.5f}')

            if do_epochs:
                row = []
                for epoch in _ENCODING_SYNC_EPOCHS:
                    frames = _epoch_frame_indices(session, epoch, sig.shape[1])
                    eb = _binned_matrix(sig[:, frames], bin_width) if frames.size else None
                    row.append(_mean_pairwise_corr(eb) if eb is not None else np.nan)
                epoch_sync[group].append(row)

    synchrony_arr = {g: np.array(synchrony[g], dtype=float) for g in GROUP_ORDER}

    # ── Figure 1: whole-session synchrony violin + per-cell coupling ECDF ──
    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    _draw_violin_triplet(axs[0], synchrony_arr, 0, GROUP_ORDER, GROUP_COLOURS,
                         stat_fn=do_pairwise_holm_plot, ylim=bracket_ylim(synchrony_arr, 1),
                         ylabel='Mean pairwise correlation')
    axs[0].set_xticks(range(len(GROUP_ORDER)))
    axs[0].set_xticklabels([GROUP_LABELS[g] for g in GROUP_ORDER], size='medium')
    axs[0].set_title('Mean pairwise correlation', size='medium')

    per_cell_by_group = {
        g: (np.concatenate(list(coupling_by_group[g].values())) if coupling_by_group[g] else np.array([]))
        for g in GROUP_ORDER
    }
    ecdf_panel(axs[1], per_cell_by_group, per_mouse_by_group=coupling_by_group,
               xlabel='Per-cell population coupling', title='Population coupling')
    fig.suptitle(f'Population coupling — {family}')
    fig.subplots_adjust(left=0.08, bottom=0.12, right=0.98, top=0.87, wspace=0.25)
    save_fig(fig, os.path.join(out_dir, f'population_coupling-{family}.png'))
    if auto_close:
        plt.close(fig)

    # ── Figure 2 (encoding): epoch-restricted synchrony ──
    if do_epochs:
        epoch_sync_arr = {g: np.array(epoch_sync[g], dtype=float) for g in GROUP_ORDER}
        fig, axs = plt.subplots(1, len(_ENCODING_SYNC_EPOCHS),
                                figsize=(3 * len(_ENCODING_SYNC_EPOCHS), 4), sharey='row')
        ep_ylim = bracket_ylim(epoch_sync_arr, len(_ENCODING_SYNC_EPOCHS))
        for i, (ax, epoch) in enumerate(zip(axs.flat, _ENCODING_SYNC_EPOCHS)):
            _draw_violin_triplet(ax, epoch_sync_arr, i, GROUP_ORDER, GROUP_COLOURS,
                                 stat_fn=do_pairwise_holm_plot, ylim=ep_ylim)
            ax.set_xticks(range(len(GROUP_ORDER)))
            ax.set_xticklabels([GROUP_LABELS[g] for g in GROUP_ORDER], size='medium')
            ax.set_title(epoch, size='medium')
        axs.flat[0].set_ylabel('Mean pairwise correlation')
        fig.suptitle(f'Epoch-restricted synchrony — {family}')
        fig.subplots_adjust(left=0.07, bottom=0.1, right=0.98, top=0.85, wspace=0.2)
        save_fig(fig, os.path.join(out_dir, f'epoch_synchrony-{family}.png'))
        if auto_close:
            plt.close(fig)

    # ── Stats: mixed model on per-cell coupling ──
    records = build_cell_records(coupling_by_group, value_name='value')
    model_text, method = fit_group_mixed_model(records, value_col='value')
    write_text(os.path.join(stats_dir, f'population_coupling_mixed_model-{family}.txt'),
               f'Per-cell population coupling — {family}\nmodel: {method}\n\n{model_text}')

    write_text(os.path.join(tables_dir, f'per_mouse_coupling-{family}.csv'),
               'mouse,group,n_cells,mean_pairwise_corr,mean_population_coupling\n'
               + '\n'.join(table_rows) + '\n')

    return {'synchrony': synchrony_arr, 'coupling_by_group': coupling_by_group}
