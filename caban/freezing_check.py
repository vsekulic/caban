"""Freezing check for the conditioning result (plans/figure2_redesign_plan.md §14, revised after
statistics-checker before it was built). METHODS: analysis_methods_templates/freezing_check_methods.md.

Freezing = speed below freezing_tuned_cells.FREEZE_THRESH_CM_S (behaviour-camera speed per
miniscope frame) sustained for at least MIN_BOUT_S; the frame-wise definition without the bout
criterion is reported beside it. Epochs and trials are the four-epoch lane's (sp_rates_lmm
TFC_FOUR_EPOCHS, exposure-matched (mouse, trial) pairs).

(a) Does freezing differ? Per animal and epoch, the fraction of frames freezing; per epoch,
    Exc - Ctl and Inh - Ctl on its empirical logit by exact pairwise mouse-label permutation (Holm
    across the two), Welch with its 95% CI beside.
(b) Does the effect persist at matched behaviour? The four-epoch model refitted on moving frames
    only (events whose peak frame is a moving frame; exposure = moving cell-seconds), compared with
    the all-frames model by effect size and CI. Animal x epoch cells without moving frames are
    listed, not filled. Secondary: the all-frames model with the freezing fraction split into its
    animal mean and the epoch deviation from it (logit scale). Freezing-only frames: descriptive.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from caban.utilities import MINISCOPE_FPS
from caban.single_unit_common import (GROUP_COLOURS, GROUP_LABELS, DREADD_DISPLAY_ORDER, ensure_dirs,
                                      save_fig, write_text, holm_correct, fit_mixed_model)
from caban.decoder import _copy_analysis_methods_template
from caban.nature_comm_style import box_strip
from caban.freezing_tuned_cells import _freeze_move_masks, FREEZE_THRESH_CM_S
from caban.sp_rates_lmm import (TFC_FOUR_EPOCHS, FOUR_EPOCH_SUBDIR, _iter_event_windows,
                                tfc_epoch_frames_fn, exposure_matched_event_table,
                                build_mouse_epoch_unified_table, fit_unified_group_epoch_model,
                                unified_posthoc_contrasts, mouse_label_permutation_test, _unified_formula,
                                _PAPER_EPOCH_LABELS, TFC_MATCHED_REFERENCE_EPOCH)

OUT_DIRNAME = 'freezing_check'
METHODS_FILENAME = 'freezing_check_methods.md'
MIN_BOUT_S = 1.0
OUTCOMES = {'amplitude': 'mouse_mean_log_amplitude', 'population_rate': 'log_population_rate'}


def bout_mask(frame_mask, min_frames):
    """Frames belonging to a run of True at least ``min_frames`` long."""
    out = np.zeros_like(frame_mask, dtype=bool)
    m = np.r_[False, frame_mask.astype(bool), False]
    starts, ends = np.flatnonzero(~m[:-1] & m[1:]), np.flatnonzero(m[:-1] & ~m[1:])
    for a, b in zip(starts, ends):
        if b - a >= min_frames:
            out[a:b] = True
    return out


def freeze_masks(session):
    """{'framewise': bool, 'bout': bool} freezing masks over the session's S frames."""
    freeze, _move = _freeze_move_masks(session, session.S.shape[1])
    framewise = freeze.astype(bool)
    return {'framewise': framewise, 'bout': bout_mask(framewise, int(round(MIN_BOUT_S * MINISCOPE_FPS)))}


def freezing_fractions(mice_per_group, TFC_cond, pairs):
    """Per (mouse, epoch): frames, frames freezing (both definitions) over the matched trials."""
    frames_fn = tfc_epoch_frames_fn()
    rows = []
    for group in DREADD_DISPLAY_ORDER:
        for mouse in mice_per_group.get(group, []):
            session = TFC_cond[mouse]
            masks = freeze_masks(session)
            for epoch in TFC_FOUR_EPOCHS:
                n = {'framewise': 0, 'bout': 0}
                total = 0
                for trial in session.periods:
                    if (mouse, int(trial)) not in pairs:
                        continue
                    lo, hi = frames_fn(session, epoch, trial)
                    total += hi - lo
                    for k in n:
                        n[k] += int(masks[k][lo:hi].sum())
                rows.append({'mouse': mouse, 'group': group, 'epoch': epoch, 'n_frames': total,
                             'freeze_frames': n['bout'], 'freeze_frames_framewise': n['framewise'],
                             'freeze_fraction': n['bout'] / total,
                             'freeze_fraction_framewise': n['framewise'] / total})
    return pd.DataFrame(rows)


def _elogit(k, n):
    return np.log((k + 0.5) / (n - k + 0.5))


def test_freezing(frac, mice_per_group):
    rows = []
    for definition, kcol in (('bout', 'freeze_frames'), ('framewise', 'freeze_frames_framewise')):
        for epoch in TFC_FOUR_EPOCHS:
            sub = frac[frac['epoch'] == epoch].set_index('mouse')
            val = _elogit(sub[kcol], sub['n_frames']).to_dict()
            for a in ('hM3D', 'hM4D'):
                b = 'mCherry'
                xa = np.array([val[m] for m in mice_per_group[a]])
                xb = np.array([val[m] for m in mice_per_group[b]])
                welch = stats.ttest_ind(xa, xb, equal_var=False)
                ci = welch.confidence_interval()

                def stat_fn(m2g, a=a, b=b, val=val):
                    return (np.mean([val[m] for m, g in m2g.items() if g == a])
                            - np.mean([val[m] for m, g in m2g.items() if g == b]))
                p_perm = mouse_label_permutation_test(stat_fn, mice_per_group, restrict_to_groups=(a, b),
                                                      exact=True)['p_two_sided']
                rows.append({'definition': definition, 'epoch': epoch,
                             'contrast': f'{GROUP_LABELS[a]} - {GROUP_LABELS[b]}',
                             'diff_logit': float(xa.mean() - xb.mean()), 'ci_low': float(ci.low),
                             'ci_high': float(ci.high), 'p_welch': float(welch.pvalue),
                             'p_permutation': float(p_perm)})
    out = pd.DataFrame(rows)
    out['p_holm'] = np.nan
    for (_d, _e), idx in out.groupby(['definition', 'epoch']).groups.items():
        out.loc[idx, 'p_holm'] = holm_correct(out.loc[idx, 'p_permutation'].to_numpy())[1]
    return out


def state_event_table(mice_per_group, TFC_cond, pairs, state, mapping='full', thres=None):
    """Fine (mouse, group, trial, epoch, cell) table restricted to one behavioural state
    ('moving' or 'freezing', bout definition): events whose peak frame is in that state, exposure
    = that state's frames in the window."""
    frames_fn = tfc_epoch_frames_fn()
    masks = {m: freeze_masks(TFC_cond[m])['bout'] for ms in mice_per_group.values() for m in ms}
    rows = []
    for (group, mouse, trial, epoch, cell_id, _exp, frameidx, amplitude, _nlm, _w, _ph) in \
            _iter_event_windows(mice_per_group, TFC_cond, TFC_FOUR_EPOCHS, frames_fn, mapping, thres):
        if (mouse, int(trial)) not in pairs:
            continue
        lo, hi = frames_fn(TFC_cond[mouse], epoch, trial)
        in_state = ~masks[mouse] if state == 'moving' else masks[mouse]
        keep = in_state[frameidx.astype(int)] if len(frameidx) else np.zeros(0, dtype=bool)
        rows.append({'mouse': mouse, 'group': group, 'trial': int(trial), 'epoch': epoch,
                     'cell': cell_id, 'n_events': int(keep.sum()),
                     'sum_amplitude': float(np.sum(amplitude[keep])) if keep.any() else 0.0,
                     'exposure_seconds': float(in_state[lo:hi].sum()) / MINISCOPE_FPS})
    return pd.DataFrame(rows)


def refit(df_state):
    me = build_mouse_epoch_unified_table(df_state, epochs=TFC_FOUR_EPOCHS, require_complete=False)
    fits = {k: fit_unified_group_epoch_model(me, col) for k, col in OUTCOMES.items()}
    return me, unified_posthoc_contrasts(fits, me, epochs=TFC_FOUR_EPOCHS, across_epoch_family=False)


def covariate_model(PLOTS_DIR, frac):
    """Secondary: the all-frames four-epoch model with the freezing fraction (logit, bout
    definition) split into animal mean and epoch deviation."""
    path = os.path.join(PLOTS_DIR, 'sp_rates_lmm', 'paper', FOUR_EPOCH_SUBDIR, 'stats',
                        'unified_lmm_mouse_epoch_values.csv')
    if not os.path.isfile(path):
        raise FileNotFoundError(f'{path} not found: run the four-epoch lane first.')
    me = pd.read_csv(path).merge(frac[['mouse', 'epoch', 'freeze_frames', 'n_frames']], on=['mouse', 'epoch'])
    me['freeze_logit'] = _elogit(me['freeze_frames'], me['n_frames'])
    me['freeze_between'] = me.groupby('mouse')['freeze_logit'].transform('mean')
    me['freeze_within'] = me['freeze_logit'] - me['freeze_between']
    texts = []
    for k, col in OUTCOMES.items():
        _res, method, text = fit_mixed_model(me, _unified_formula(col) + ' + freeze_between + freeze_within',
                                             group_col='mouse')
        texts.append(f'--- {k} ({method}) ---\n{text}')
    return '\n\n'.join(texts)


def plot_freezing(frac, out_path):
    fig, axs = plt.subplots(1, len(TFC_FOUR_EPOCHS), figsize=(2.4 * len(TFC_FOUR_EPOCHS), 2.8), sharey=True)
    for ax, epoch in zip(axs, TFC_FOUR_EPOCHS):
        sub = frac[frac['epoch'] == epoch]
        box_strip(ax, {g: sub.loc[sub['group'] == g, 'freeze_fraction'] for g in DREADD_DISPLAY_ORDER},
                  DREADD_DISPLAY_ORDER, GROUP_COLOURS, labels=GROUP_LABELS, point_size=14)
        ax.set_title(_PAPER_EPOCH_LABELS[epoch])
    axs[0].set_ylabel(f'Fraction freezing (< {FREEZE_THRESH_CM_S:g} cm/s, ≥ {MIN_BOUT_S:g} s)')
    fig.tight_layout()
    save_fig(fig, out_path)
    plt.close(fig)


def plot_effect_comparison(side, out_path):
    """Ratio vs mCherry with 95% CI: all frames, moving frames only, per epoch and outcome."""
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.2))
    for ax, outcome in zip(axs, OUTCOMES):
        sub = side[side['outcome'] == outcome].reset_index(drop=True)
        for i, r in sub.iterrows():
            for dx, sfx, mk in ((-0.12, 'all', 'o'), (0.12, 'moving', 's')):
                ax.errorbar(i + dx, r[f'ratio_{sfx}'], yerr=[[r[f'ratio_{sfx}'] - r[f'ratio_ci_low_{sfx}']],
                                                           [r[f'ratio_ci_high_{sfx}'] - r[f'ratio_{sfx}']]],
                            fmt=mk, color=GROUP_COLOURS[r['group']], ms=4, capsize=2,
                            label=f"{GROUP_LABELS[r['group']]} {sfx}" if i < 2 else None)
        ax.axhline(1, color='0.6', lw=0.6, ls='--')
        ax.set_yscale('log')
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels([f"{_PAPER_EPOCH_LABELS[e]}\n{GROUP_LABELS[g]}" for e, g in zip(sub['epoch'], sub['group'])],
                           fontsize='x-small')
        ax.set_title(f'{outcome}: ratio vs Ctl (circle all frames, square moving only)', fontsize='small')
    axs[0].legend(fontsize='xx-small', frameon=False)
    fig.tight_layout()
    save_fig(fig, out_path)
    plt.close(fig)


def run_freezing_check(PLOTS_DIR, mice_per_group, TFC_cond, mapping='full', thres=None):
    out_dir = os.path.join(PLOTS_DIR, OUT_DIRNAME)
    ensure_dirs(out_dir)
    _copy_analysis_methods_template(METHODS_FILENAME, out_dir)
    df_all, _cov = exposure_matched_event_table(mice_per_group, TFC_cond, TFC_FOUR_EPOCHS, mapping, thres)
    pairs = set(zip(df_all['mouse'], df_all['trial'].astype(int)))

    frac = freezing_fractions(mice_per_group, TFC_cond, pairs)
    tests = test_freezing(frac, mice_per_group)
    frac.to_csv(os.path.join(out_dir, 'freezing_fraction_per_animal.csv'), index=False)
    tests.to_csv(os.path.join(out_dir, 'freezing_group_tests.csv'), index=False)
    plot_freezing(frac, os.path.join(out_dir, 'freezing_fraction.png'))

    all_path = os.path.join(PLOTS_DIR, 'sp_rates_lmm', 'paper', FOUR_EPOCH_SUBDIR, 'stats',
                            'unified_lmm_posthoc_contrasts.csv')
    if not os.path.isfile(all_path):
        raise FileNotFoundError(f'{all_path} not found: run the four-epoch lane first.')
    keys, cols = ['outcome', 'epoch', 'group'], ['ratio', 'ratio_ci_low', 'ratio_ci_high', 'p_raw']
    side = pd.read_csv(all_path)[keys + cols]
    reports = []
    for state in ('moving', 'freezing'):
        df_state = state_event_table(mice_per_group, TFC_cond, pairs, state, mapping, thres)
        me, contrasts = refit(df_state)
        me.to_csv(os.path.join(out_dir, f'{state}_only_mouse_epoch_values.csv'), index=False)
        contrasts.to_csv(os.path.join(out_dir, f'{state}_only_contrasts.csv'), index=False)
        reports.append(f'{state}-only: animal x epoch cells without {state} frames or events: '
                       f'{me.attrs["incomplete"] or "none"}')
        if state == 'moving':
            side = side.merge(contrasts[keys + cols], on=keys, suffixes=('_all', '_moving'))
    side.to_csv(os.path.join(out_dir, 'all_vs_moving_only_contrasts.csv'), index=False)
    plot_effect_comparison(side, os.path.join(out_dir, 'all_vs_moving_only_contrasts.png'))

    text = ['Freezing check (plan §14; pre-specified) — the animal is the unit', '=' * 66, '',
            f'Freezing = speed < {FREEZE_THRESH_CM_S:g} cm/s for >= {MIN_BOUT_S:g} s (framewise beside).', '',
            '(a) Freezing fraction by group (empirical logit; exact pairwise permutation, Holm over 2; Welch CI):',
            tests.round(4).to_string(index=False), '',
            'Mean freezing fraction (bout definition) by group and epoch:',
            frac.groupby(['group', 'epoch'])['freeze_fraction'].mean().unstack().round(3).to_string(), '',
            '(b) Contrasts vs mCherry, all frames vs moving frames only (effect sizes, not P):',
            side.round(4).to_string(index=False), ''] + reports + [
            '', 'Secondary: freezing as covariate (animal mean + within-animal epoch deviation, logit):',
            covariate_model(PLOTS_DIR, frac)]
    write_text(os.path.join(out_dir, 'freezing_check_report.txt'), '\n'.join(text) + '\n')
    print('\n'.join(text[:30]))
    return tests, side
