"""Pyramidal events in the homecage before vs after CNO, on the conditioning day (Figure 2 v2, e).

Pre-specified in plans/figure2_redesign_plan.md §4.1.1 (revised after statistics-checker, before any
result was computed); METHODS: analysis_methods_templates/homecage_cno_methods.md.

- pre  = HC1;  post = the homecage recording that starts POST_WINDOW_MIN after the injection.
- The injection: CNO1's start in animals that also have a CNO2; CNO1's start minus
  CNO1_AFTER_INJECTION_MIN where CNO1 is the only CNO recording (VS, 2026-10-21: the label
  changed meaning during data collection).
- Events exactly as the primary conditioning analysis: find_event_runs_ca_S(S, EVENT_THRESHOLD) on
  each recording's whole S; per animal, mean log per-event amplitude over active cells and the
  population event rate over all cells (sp_rates_lmm.build_mouse_epoch_unified_table). Each
  recording has its own Minian units (no cross-registration), kept distinct.
- Test: per animal, delta = log(post) - log(pre); Exc vs Ctl and Inh vs Ctl by Welch t-test, Holm
  across the two within each measure, with the exact pairwise mouse-label randomization beside it;
  the group x window mixed model as a consistency check.

The S matrices and timestamps come from scripts/export_tfc_day_homecage_s.py (run on the Razer) in
HOMECAGE_EXPORT_DIR.
"""

import glob
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from caban.utilities import CACHE_VOLUME, MINISCOPE_FPS, find_event_runs_ca_S
from caban.sessions import check_miniscope_frame_rate
from caban.single_unit_common import (GROUP_COLOURS, GROUP_LABELS, DREADD_DISPLAY_ORDER,
                                      ensure_dirs, save_fig, write_text, holm_correct)
from caban.decoder import _copy_analysis_methods_template
from caban.nature_comm_style import nature_style, save_paper_panel, box_strip, AXIS_PT, FONT_PT
from caban.figure2_composite import panel_figsize, LAYOUT_V2_MM
from caban.sp_rates_lmm import (build_mouse_epoch_unified_table, fit_unified_group_epoch_model,
                                mouse_label_permutation_test)

HOMECAGE_EXPORT_DIR = os.path.join(CACHE_VOLUME, 'caban_cache', 'homecage_cno')
METHODS_FILENAME = 'homecage_cno_methods.md'
OUT_DIRNAME = 'homecage_cno'

EVENT_THRESHOLD = 2                 # sessions.BehaviourSession.thres, the primary analysis's value
# The primary analysis's own threshold-sensitivity values (paper_figure2.md, Controls).
SENSITIVITY_THRESHOLDS = (1.5, 2.0, 3.0)
POST_WINDOW_MIN = (15.0, 35.0)      # recording start, minutes after the injection
CNO1_AFTER_INJECTION_MIN = 30.0     # where CNO1 is the only CNO recording (VS, 2026-10-21)
PRE_LABEL = 'HC1'
# Animals without a CNO2 whose CNO1 nevertheless began at the injection, by name and reason. G12: under
# the CNO1-minus-30-min rule its drug-free LT1 (20 min before CNO1) would fall after the injection; its
# day has G13's layout (HC1, LT1, HC2, CNO1, LT2, HC3), and G13's CNO1 began at the injection.
# To be confirmed by VS (plans/figure2_redesign_plan.md §4.1.1).
CNO1_AT_INJECTION_WITHOUT_CNO2 = {'G12': 'drug-free LT1 would otherwise follow the injection'}
WINDOWS = ('pre_cno', 'post_cno')
MEASURES = {'amplitude': ('mouse_mean_log_amplitude', 'Δ log amplitude'),
            'rate': ('log_population_rate', 'Δ log rate')}
CONTRASTS = (('hM3D', 'mCherry'), ('hM4D', 'mCherry'))
EXPLORATORY_CONTRAST = ('hM3D', 'hM4D')

_KEY = re.compile(r'^(G\d\d)-[^_]+.*__(?P<day>[^_]+_[^_]+_[^_]+-TFC_cond)__'
                  r'(?P<hh>\d\d)_(?P<mm>\d\d)_(?P<ss>\d\d)-(?P<label>HC\d+|CNO\d?)$')


def recording_table(export_dir=HOMECAGE_EXPORT_DIR):
    """One row per exported recording: mouse, label, start (min of day), path."""
    rows = []
    for path in sorted(glob.glob(os.path.join(export_dir, '*.npz'))):
        key = os.path.basename(path)[:-4]
        m = _KEY.match(key)
        if m is None:
            raise RuntimeError(f'homecage_cno: unexpected export name {key!r}')
        start = int(m['hh']) * 60 + int(m['mm']) + int(m['ss']) / 60
        rows.append({'mouse': m.group(1), 'label': m['label'], 'start_min': start,
                     'tail': key.replace('__', '/'), 'path': path})
    if not rows:
        raise FileNotFoundError(f'homecage_cno: no exports in {export_dir}')
    return pd.DataFrame(rows)


def injection_minute(rec_mouse):
    """The injection, in minutes of the day, by the rule in the module docstring."""
    cno = rec_mouse[rec_mouse['label'].str.startswith('CNO')]
    labels = set(cno['label'])
    if 'CNO1' not in labels:
        raise RuntimeError(f'homecage_cno: {rec_mouse["mouse"].iloc[0]} has no CNO1 recording')
    cno1 = float(cno.loc[cno['label'] == 'CNO1', 'start_min'].iloc[0])
    if 'CNO2' in labels or rec_mouse['mouse'].iloc[0] in CNO1_AT_INJECTION_WITHOUT_CNO2:
        return cno1
    return cno1 - CNO1_AFTER_INJECTION_MIN


def select_recordings(rec, mice):
    """Per animal: its pre recording and its single post recording; every other recording with its
    minutes after the injection, for the record. Fails unless each animal has exactly one of each."""
    chosen, timing = [], []
    for mouse in mice:
        r = rec[rec['mouse'] == mouse].copy()
        if r.empty:
            raise RuntimeError(f'homecage_cno: no exported recordings for {mouse}')
        inj = injection_minute(r)
        r['min_after_injection'] = r['start_min'] - inj
        pre = r[r['label'] == PRE_LABEL]
        lo, hi = POST_WINDOW_MIN
        post = r[(r['min_after_injection'] >= lo) & (r['min_after_injection'] < hi)]
        if len(pre) != 1 or len(post) != 1:
            raise RuntimeError(f'homecage_cno: {mouse} has {len(pre)} {PRE_LABEL} and {len(post)} '
                               f'recordings {lo:g}-{hi:g} min after the injection; the '
                               f'pre-specification needs exactly one of each.')
        chosen += [dict(pre.iloc[0], window='pre_cno'), dict(post.iloc[0], window='post_cno')]
        timing.append(r[['mouse', 'label', 'min_after_injection', 'tail']])
    return pd.DataFrame(chosen), pd.concat(timing, ignore_index=True)


def recording_events(path, tail, thres=EVENT_THRESHOLD):
    """Per-unit events of one recording: n_events, sum_amplitude, exposure_seconds; plus its
    detection diagnostics."""
    with np.load(path) as z:
        S, unit_id, tstamp, imaged = z['S'], z['unit_id'], z['tstamp_ms'], z['imaged']
    fps = check_miniscope_frame_rate(tail, pd.Series(tstamp[imaged]))
    events = find_event_runs_ca_S(S, thres)
    exposure = S.shape[1] / MINISCOPE_FPS
    n_events = np.array([len(events[i][0]) for i in range(S.shape[0])])
    sum_amp = np.array([float(np.sum(events[i][1])) for i in range(S.shape[0])])
    cells = pd.DataFrame({'unit_id': unit_id, 'n_events': n_events, 'sum_amplitude': sum_amp,
                          'exposure_seconds': exposure})
    diag = {'n_units': int(S.shape[0]), 'fraction_active': float(np.mean(n_events > 0)),
            'events_per_unit_per_min': float(n_events.mean() / exposure * 60),
            'duration_min': exposure / 60, 'fps_measured': fps}
    return cells, diag


def build_tables(mice_per_group, export_dir=HOMECAGE_EXPORT_DIR, thres=EVENT_THRESHOLD):
    group_of = {m: g for g, ms in mice_per_group.items() for m in ms}
    rec = recording_table(export_dir)
    chosen, timing = select_recordings(rec, list(group_of))
    fine, diags = [], []
    for _, r in chosen.iterrows():
        cells, diag = recording_events(r['path'], r['tail'], thres)
        cells = cells.assign(mouse=r['mouse'], group=group_of[r['mouse']], trial=0,
                             epoch=r['window'], cell=r['label'] + ':' + cells['unit_id'].astype(str))
        fine.append(cells)
        diags.append({'mouse': r['mouse'], 'group': group_of[r['mouse']], 'window': r['window'],
                      'recording': r['label'], 'min_after_injection': r['min_after_injection'], **diag})
    fine = pd.concat(fine, ignore_index=True)
    per_mouse = build_mouse_epoch_unified_table(fine, epochs=WINDOWS, reference_epoch=WINDOWS[0])
    return per_mouse, pd.DataFrame(diags), timing


def per_animal_deltas(per_mouse):
    wide = per_mouse.pivot(index=['mouse', 'group'], columns='epoch',
                           values=[c for c, _ in MEASURES.values()])
    out = pd.DataFrame(index=wide.index)
    for name, (col, _) in MEASURES.items():
        out[f'delta_{name}'] = wide[(col, 'post_cno')] - wide[(col, 'pre_cno')]
    return out.reset_index()


def test_deltas(deltas, mice_per_group):
    rows = []
    for name in MEASURES:
        col = f'delta_{name}'
        for a, b in CONTRASTS + (EXPLORATORY_CONTRAST,):
            xa = deltas.loc[deltas['group'] == a, col].to_numpy()
            xb = deltas.loc[deltas['group'] == b, col].to_numpy()
            welch = stats.ttest_ind(xa, xb, equal_var=False)
            value = dict(zip(deltas['mouse'], deltas[col]))

            def stat_fn(mouse_to_group, a=a, b=b, value=value):
                va = [value[m] for m, g in mouse_to_group.items() if g == a]
                vb = [value[m] for m, g in mouse_to_group.items() if g == b]
                return float(np.mean(va) - np.mean(vb))

            perm = mouse_label_permutation_test(stat_fn, mice_per_group, restrict_to_groups=(a, b),
                                                exact=True)
            rows.append({'measure': name, 'contrast': f'{GROUP_LABELS[a]} - {GROUP_LABELS[b]}',
                         'estimate': float(xa.mean() - xb.mean()),
                         'ratio': float(np.exp(xa.mean() - xb.mean())),
                         'welch_t': float(welch.statistic), 'welch_df': float(welch.df),
                         'p_welch': float(welch.pvalue), 'p_permutation': float(perm['p_two_sided']),
                         'family': 'exploratory' if (a, b) == EXPLORATORY_CONTRAST else name})
    df = pd.DataFrame(rows)
    df['p_holm'] = np.nan
    for name in MEASURES:
        sel = df['family'] == name
        df.loc[sel, 'p_holm'] = holm_correct(df.loc[sel, 'p_welch'].to_numpy())[1]
    return df


def plot_panel(deltas, out_path):
    """Figure 2 v2 panel e: per-animal post - pre change, amplitude (left) and rate (right)."""
    with nature_style():
        fig, axs = plt.subplots(1, 2, figsize=panel_figsize('e', LAYOUT_V2_MM), layout='constrained')
        for ax, (name, (_, ylabel)) in zip(axs, MEASURES.items()):
            vals = {g: deltas.loc[deltas['group'] == g, f'delta_{name}'] for g in DREADD_DISPLAY_ORDER}
            box_strip(ax, vals, DREADD_DISPLAY_ORDER, GROUP_COLOURS, labels=GROUP_LABELS)
            ax.axhline(0, color='0.5', lw=AXIS_PT, ls='--', zorder=0)
            ax.set_ylabel(ylabel)
        axs[0].set_title('Per-event amplitude')
        axs[1].set_title('Population event rate')
        save_paper_panel(fig, os.path.splitext(out_path)[0])
    plt.close(fig)


def format_report(tests, diags, timing):
    lines = ['Homecage before vs after CNO (conditioning day) — pyramidal events',
             '=' * 66, '',
             f'pre = {PRE_LABEL}; post = the homecage recording starting {POST_WINDOW_MIN[0]:g}-'
             f'{POST_WINDOW_MIN[1]:g} min after the injection (pre-specified; plans/figure2_redesign_plan.md',
             '§4.1.1). Delta = log(post) - log(pre) per animal. Welch t-tests, Holm across the two',
             'treatment-vs-control contrasts within each measure; exact pairwise mouse-label',
             'randomization beside them. Exc - Inh is exploratory (no family).', '',
             tests.round(4).to_string(index=False), '',
             'Detection diagnostics per recording (mean over animals, by group and window):',
             diags.groupby(['group', 'window'])[['n_units', 'fraction_active',
                                                 'events_per_unit_per_min', 'min_after_injection']]
             .mean().round(3).to_string(), '',
             'Recordings used:', diags[['mouse', 'group', 'window', 'recording',
                                        'min_after_injection', 'n_units']].round(1).to_string(index=False),
             '', 'All homecage recordings, minutes after the injection:',
             timing.round(1).to_string(index=False)]
    return '\n'.join(lines) + '\n'


def run_homecage_cno(PLOTS_DIR, mice_per_group, export_dir=HOMECAGE_EXPORT_DIR):
    out_dir = os.path.join(PLOTS_DIR, OUT_DIRNAME)
    ensure_dirs(out_dir)
    _copy_analysis_methods_template(METHODS_FILENAME, out_dir)
    per_mouse, diags, timing = build_tables(mice_per_group, export_dir)
    deltas = per_animal_deltas(per_mouse)
    tests = test_deltas(deltas, mice_per_group)
    fits = {name: fit_unified_group_epoch_model(per_mouse, col, reference_epoch=WINDOWS[0])
            for name, (col, _) in MEASURES.items()}
    per_mouse.to_csv(os.path.join(out_dir, 'per_mouse_window.csv'), index=False)
    deltas.to_csv(os.path.join(out_dir, 'per_animal_delta.csv'), index=False)
    tests.to_csv(os.path.join(out_dir, 'delta_tests.csv'), index=False)
    diags.to_csv(os.path.join(out_dir, 'detection_diagnostics.csv'), index=False)
    timing.to_csv(os.path.join(out_dir, 'recording_timing.csv'), index=False)
    report = format_report(tests, diags, timing)
    report += '\nMixed-model consistency check (log(metric) ~ group * window + (1|animal)):\n'
    for name, fit in fits.items():
        report += f'\n--- {name} ---\n{fit["summary_text"]}\n'
    write_text(os.path.join(out_dir, 'homecage_cno_report.txt'), report)
    plot_panel(deltas, os.path.join(out_dir, 'homecage_cno_delta.png'))
    print(report)
    return tests


def run_threshold_sensitivity(PLOTS_DIR, mice_per_group, export_dir=HOMECAGE_EXPORT_DIR,
                              thresholds=SENSITIVITY_THRESHOLDS):
    """The per-animal changes and the same tests at each event threshold: a detection check
    (statistics-checker / code review, 2026-10-21), a diagnostic, not a figure panel. Writes
    <PLOTS_DIR>/homecage_cno/threshold_sensitivity/."""
    out_dir = os.path.join(PLOTS_DIR, OUT_DIRNAME, 'threshold_sensitivity')
    ensure_dirs(out_dir)
    all_deltas, all_tests = [], []
    for thres in thresholds:
        per_mouse, diags, _ = build_tables(mice_per_group, export_dir, thres)
        deltas = per_animal_deltas(per_mouse).assign(threshold=thres)
        all_deltas.append(deltas)
        all_tests.append(test_deltas(deltas.drop(columns='threshold'), mice_per_group).assign(threshold=thres))
    deltas = pd.concat(all_deltas, ignore_index=True)
    tests = pd.concat(all_tests, ignore_index=True)
    deltas.to_csv(os.path.join(out_dir, 'per_animal_delta_by_threshold.csv'), index=False)
    tests.to_csv(os.path.join(out_dir, 'delta_tests_by_threshold.csv'), index=False)

    fig, axs = plt.subplots(2, len(thresholds), figsize=(3.0 * len(thresholds), 5.4), sharey='row',
                            squeeze=False)
    for col, thres in enumerate(thresholds):
        d = deltas[deltas['threshold'] == thres]
        t = tests[tests['threshold'] == thres]
        for row, (name, (_, ylabel)) in enumerate(MEASURES.items()):
            ax = axs[row, col]
            box_strip(ax, {g: d.loc[d['group'] == g, f'delta_{name}'] for g in DREADD_DISPLAY_ORDER},
                      DREADD_DISPLAY_ORDER, GROUP_COLOURS, labels=GROUP_LABELS, point_size=14)
            ax.axhline(0, color='0.5', lw=0.6, ls='--', zorder=0)
            note = '\n'.join(f"{r['contrast']}: x{r['ratio']:.2f}, Holm P={r['p_holm']:.3f}"
                             for _, r in t[t['family'] == name].iterrows())
            ax.set_title(f'threshold {thres:g}\n{note}', fontsize='small')
            if col == 0:
                ax.set_ylabel(ylabel)
    fig.suptitle('Homecage pre/post-CNO: per-animal change at each event threshold '
                 '(diagnostic; primary threshold 2)', size='medium')
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, 'threshold_sensitivity.png'))
    plt.close(fig)
    report = tests.round(4).to_string(index=False)
    write_text(os.path.join(out_dir, 'threshold_sensitivity.txt'), report + '\n')
    print(report)
    return tests
