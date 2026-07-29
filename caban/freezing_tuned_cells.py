"""Freezing-tuned single units.

Classifies each cell as freezing-associated, movement-associated, or untuned by
contrasting its deconvolved activity during freezing frames vs moving frames, and
compares the fraction of freezing-preferring cells across the three DREADD groups.
Freezing is the behavioural readout of the fear memory, so a cell-level link to it
is a direct memory-relevant single-unit measure. Run on the conditioning session
and the recall tests.

Freezing is defined exactly as elsewhere in the codebase: imaging-frame velocity
(velocities_miniscope_smooth) below 2 cm/s (the same threshold behind S_mov/S_imm).
Significance uses the shared circular-shift shuffle contrast, which preserves each
trace's autocorrelation under the null.
"""
import os

import numpy as np
import matplotlib.pyplot as plt

from caban.analysis import _draw_violin_triplet, do_pairwise_holm_plot
from caban.decoder import _copy_analysis_methods_template
from caban.single_unit_common import (
    GROUP_ORDER, GROUP_COLOURS, GROUP_LABELS,
    ensure_dirs, write_text, save_fig, shuffle_mask_contrast, get_mapping_signal,
    ecdf_panel, build_cell_records, fit_group_mixed_model, bracket_ylim,
)

FREEZE_THRESH_CM_S = 2.0


def _freeze_move_masks(session, n_frames):
    """(freeze_mask, move_mask) over imaging frames, freezing = speed < threshold."""
    vel = np.asarray(session.velocities_miniscope_smooth[:n_frames], dtype=float)
    if vel.shape[0] < n_frames:
        raise RuntimeError(f'Velocity trace ({vel.shape[0]}) shorter than S frames ({n_frames}).')
    freeze_mask = (vel < FREEZE_THRESH_CM_S).astype(float)
    move_mask = (vel >= FREEZE_THRESH_CM_S).astype(float)
    return freeze_mask, move_mask


def run_freezing_tuned_cells(PLOTS_DIR, mice_per_group, sessions, family,
                             mapping='full', crossreg_to_use=None, signal='C',
                             n_shuffles=1000, seed=0, auto_close=True):
    """Freezing-tuning analysis for one session family ('encoding', 'recall_testA',
    'recall_testB'). ``signal`` selects the activity trace ('C' denoised calcium,
    default; 'S' deconvolved spikes)."""
    out_dir = os.path.join(PLOTS_DIR, 'freezing_tuned_cells', family)
    stats_dir = os.path.join(out_dir, 'stats')
    tables_dir = os.path.join(out_dir, 'tables')
    ensure_dirs(out_dir, stats_dir, tables_dir)
    _copy_analysis_methods_template('freezing_tuned_cells_methods.txt', out_dir)

    frac_freeze = {g: [] for g in GROUP_ORDER}     # fraction freezing-preferring
    frac_move = {g: [] for g in GROUP_ORDER}       # fraction movement-preferring
    effect_by_group = {g: {} for g in GROUP_ORDER}  # per-mouse per-cell observed contrast
    table_rows = []

    for group in GROUP_ORDER:
        for mouse in mice_per_group.get(group, []):
            if mouse not in sessions:
                continue
            session = sessions[mouse]
            wc = crossreg_to_use[mouse] if (crossreg_to_use is not None and mapping != 'full') else None
            sig = get_mapping_signal(session, mapping, signal_attr=signal, with_crossreg=wc)
            if sig.shape[0] == 0:
                raise RuntimeError(f'No cells for mapping {mapping} mouse {mouse} ({family}).')

            freeze_mask, move_mask = _freeze_move_masks(session, sig.shape[1])
            res = shuffle_mask_contrast(sig, freeze_mask, move_mask,
                                        n_shuffles=n_shuffles, seed=seed)
            responsive = res['responsive']
            observed = res['observed']
            n_cells = len(responsive)

            freeze_pref = responsive & (observed > 0)
            move_pref = responsive & (observed < 0)
            frac_freeze[group].append([freeze_pref.sum() / n_cells])
            frac_move[group].append([move_pref.sum() / n_cells])
            effect_by_group[group][mouse] = observed

            table_rows.append(f'{mouse},{group},{n_cells},{int(freeze_pref.sum())},'
                              f'{int(move_pref.sum())},{freeze_pref.sum() / n_cells:.4f},'
                              f'{move_pref.sum() / n_cells:.4f}')

    frac_freeze_arr = {g: np.array(frac_freeze[g], dtype=float) for g in GROUP_ORDER}
    frac_move_arr = {g: np.array(frac_move[g], dtype=float) for g in GROUP_ORDER}

    # ── Figure: fraction freezing / movement preferring + per-cell effect ECDF ──
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    frac_ylim = bracket_ylim({g: np.concatenate([frac_freeze_arr[g], frac_move_arr[g]], axis=0)
                              for g in GROUP_ORDER}, 1)
    for ax, arr, title in [
        (axs[0], frac_freeze_arr, 'Fraction freezing-preferring'),
        (axs[1], frac_move_arr, 'Fraction movement-preferring'),
    ]:
        _draw_violin_triplet(ax, arr, 0, GROUP_ORDER, GROUP_COLOURS,
                             stat_fn=do_pairwise_holm_plot, ylim=frac_ylim,
                             ylabel='Fraction of cells')
        ax.set_xticks(range(len(GROUP_ORDER)))
        ax.set_xticklabels([GROUP_LABELS[g] for g in GROUP_ORDER], size='medium')
        ax.set_title(title, size='medium')

    per_cell_by_group = {
        g: (np.concatenate(list(effect_by_group[g].values())) if effect_by_group[g] else np.array([]))
        for g in GROUP_ORDER
    }
    ecdf_panel(axs[2], per_cell_by_group, per_mouse_by_group=effect_by_group,
               xlabel='Freeze − move activity contrast', title='Per-cell effect')

    fig.suptitle(f'Freezing-tuned cells — {family}')
    fig.subplots_adjust(left=0.06, bottom=0.12, right=0.98, top=0.87, wspace=0.28)
    save_fig(fig, os.path.join(out_dir, f'freezing_tuned_cells-{family}.png'))
    if auto_close:
        plt.close(fig)

    # ── Stats: mixed model on the per-cell freeze−move contrast ──
    records = build_cell_records(effect_by_group, value_name='value')
    model_text, method = fit_group_mixed_model(records, value_col='value')
    write_text(os.path.join(stats_dir, f'freeze_move_contrast_mixed_model-{family}.txt'),
               f'Per-cell freeze−move activity contrast — {family}\nmodel: {method}\n\n{model_text}')

    write_text(os.path.join(tables_dir, f'per_mouse_freezing_tuning-{family}.csv'),
               'mouse,group,n_cells,n_freeze_pref,n_move_pref,'
               'fraction_freeze_pref,fraction_move_pref\n'
               + '\n'.join(table_rows) + '\n')

    return {'fraction_freeze': frac_freeze_arr, 'fraction_move': frac_move_arr}
