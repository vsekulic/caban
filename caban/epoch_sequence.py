"""Cross-validated test of sequential (time-cell-like) activity during conditioning.

EXPLORATORY. This module answers one question about a display artifact and does not change,
extend or reinterpret the epoch-modulation analysis it borrows from.

WHY THIS EXISTS
---------------
``epoch_modulation``'s panel K sorts cells by their trace modulation index and shows the
trial-averaged activity in that order. Sorted that way the heatmap appears to contain
sequential structure. It cannot be read that way, because THE ORDERING STATISTIC IS COMPUTED
FROM THE DATA BEING DISPLAYED: sorting any matrix by a statistic derived from it produces a
clean diagonal, including a matrix of pure noise. Most of the visible gradient in that panel
is in fact sort-induced -- ordering by (trace - pre_tone) necessarily puts high-trace,
low-pre-tone cells at the top and the reverse at the bottom.

The fix is to break the circularity with independent data: define each cell's peak latency on
one subset of trials, then ask whether that ordering survives on trials never used to define
it. That is what this module does.

THE MEASUREMENT
---------------
Per mouse, on the denoised calcium trace C, over the tone-onset-to-shock-onset window
(the CS-US interval a bridging sequence would have to span):

  1. split the retained conditioning trials into two DISJOINT halves (A = 1st, 3rd, ...;
     B = 2nd, 4th, ...),
  2. trial-average each half separately and smooth lightly along time,
  3. peak latency = argmax within the window, computed INDEPENDENTLY in each half,
  4. statistic = Spearman correlation between the two halves' peak latencies, ACROSS THE CELLS
     OF ONE MOUSE -- one value per mouse.

Because the halves share no trials, the correlation's expectation under "no reproducible
temporal tuning" is zero BY CONSTRUCTION. No shuffle, permutation or null model is required,
and none is used.

WHAT THE CORRELATION DOES NOT SHOW
----------------------------------
A high split-half correlation is NOT sufficient for a sequence: a sequence additionally requires
the peaks to TILE the interval, and the correlation is blind to whether they do. The failure mode
is specific, and worth stating precisely because the obvious guess about it is wrong:

  - PERFECT SYNCHRONY -- every cell peaking at the same moment -- does NOT produce a high
    correlation. Identical latencies are all ties, so Spearman is undefined on them and returns
    approximately zero (measured on a planted stub: rho = +0.01).
  - A FEW DISCRETE RESPONSE TIMES does. Two reproducible clusters, say cells peaking either early
    or late, give rho = +0.83 with only a handful of distinct latencies (measured on a planted
    stub) -- a strong, entirely genuine reproducibility that is nonetheless not a sequence.

The distribution of held-out peak latencies is therefore reported beside the correlation, and
that histogram, not the correlation, is what separates a tiling sequence from a small number of
reproducible response times.

SIGNAL
------
C, deliberately -- and for a different reason than YrA is primary in ``epoch_modulation``.
There the endpoint is a window MEAN, where C's sparsity (exactly zero between fitted
transients) is a liability. Here the endpoint is a TIMING statistic, where YrA's
sample-to-sample noise makes argmax unstable while C's denoised transients give a clean peak.

UNIT OF ANALYSIS
----------------
The mouse. Peak latencies are ranked and correlated WITHIN a mouse, never across pooled cells,
so between-animal differences in cell count or timing cannot manufacture a correlation. Cells
are pooled across mice only for DISPLAY.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr, ttest_1samp

from caban.utilities import MINISCOPE_FPS
from caban.single_unit_common import (
    GROUP_ORDER, GROUP_COLOURS, GROUP_LABELS, DREADD_DISPLAY_ORDER,
    ensure_dirs, write_text, save_fig,
)
from caban.epoch_modulation import (
    resolve_shared_cells, _standardized_trace, retained_trials, _tone_aligned_matrix,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Analysis window, in seconds from tone onset: tone onset (0 s) to shock onset (40 s), i.e. the
# 20 s tone followed by the 20 s trace interval. This is the CS-US interval that a bridging
# sequence would have to span, and it contains the tone period the original observation came
# from. Peaks are located within this window only.
WINDOW_START_S = 0.0
WINDOW_END_S = 40.0

# Temporal smoothing before taking the argmax. Enough to keep a single transient from splitting
# the peak across adjacent frames, short relative to the 40 s window so it cannot itself create
# apparent temporal structure.
SMOOTH_S = 0.5

SIGNAL = 'C'


# ─────────────────────────────────────────────────────────────────────────────
# Peak latencies
# ─────────────────────────────────────────────────────────────────────────────

def split_trials(trials):
    """Two disjoint trial halves, (A, B) = (even-position, odd-position) of ``trials``.

    With the usual 4 retained conditioning trials [1,2,3,4] this is A=[1,3], B=[2,4]. A mouse
    retaining only 3 gives A=[1,3], B=[2] -- thinner, but still disjoint, which is the property
    the whole analysis rests on. Raises if either half is empty, since a one-sided split cannot
    support a cross-validated statistic.
    """
    trials = list(trials)
    half_a, half_b = trials[0::2], trials[1::2]
    if not half_a or not half_b:
        raise RuntimeError(
            f'epoch_sequence: cannot split {trials} into two non-empty halves; a '
            f'cross-validated peak latency needs at least one trial on each side.')
    return half_a, half_b


def _window_slice(time_axis):
    """Boolean mask selecting the analysis window from a tone-aligned time axis."""
    mask = (time_axis >= WINDOW_START_S) & (time_axis <= WINDOW_END_S)
    if not mask.any():
        raise RuntimeError(
            f'epoch_sequence: the analysis window [{WINDOW_START_S}, {WINDOW_END_S}] s does not '
            f'intersect the tone-aligned axis spanning [{time_axis[0]:.1f}, {time_axis[-1]:.1f}] s.')
    return mask


def peak_latencies(mat, time_axis):
    """(latency in seconds, defined mask) for each cell of a trial-averaged aligned matrix.

    ** A cell whose windowed trace is CONSTANT has no defined peak latency. ** This is not a
    corner case here: C is the deconvolution fit and is exactly zero between fitted transients,
    so a cell with no transient anywhere in the window is identically zero across it. ``argmax``
    does not fail on such a row -- it silently returns index 0, i.e. "this cell peaked exactly at
    tone onset". A cell like that would be assigned the same latency in BOTH halves and would
    contribute a PERFECT agreement to the correlation. With enough silent cells the statistic
    would report a strong, reproducible sequence assembled entirely out of silence.

    Those cells are therefore marked undefined here and excluded downstream, rather than being
    allowed to pile up at zero.

    ** This is a definability requirement, not a responsiveness filter. ** It removes cells for
    which the statistic does not exist, not cells whose response is weak: any non-constant cell
    is kept however small or untuned its activity, and an untuned cell then has an essentially
    arbitrary argmax in each half and dilutes the correlation toward zero. Filtering on response
    SIZE would instead select cells on a noisy statistic and then report that statistic, which is
    the circularity this project avoids elsewhere.
    """
    mask = _window_slice(time_axis)
    windowed = mat[:, mask]
    smoothed = gaussian_filter1d(windowed, sigma=SMOOTH_S * MINISCOPE_FPS, axis=1)
    defined = smoothed.max(axis=1) > smoothed.min(axis=1)
    latency = time_axis[mask][np.argmax(smoothed, axis=1)]
    return latency, defined


def build_peak_table(mice_per_group, sessions, mapping='full', crossreg_to_use=None,
                     verbose=True):
    """Tidy per-cell split-half peak latencies, plus the aligned half-matrices for plotting.

    Returns (df, per_mouse_mats) where df has columns
    ``mouse, group, cell, latency_a, latency_b, defined`` and per_mouse_mats maps
    (group, mouse) -> dict(mat_a, mat_b, time_axis, defined) for the display panels.

    Cell sets, standardization and retained trials are taken unchanged from
    ``epoch_modulation`` -- this analysis re-measures nothing.
    """
    rows, mats = [], {}
    for group in GROUP_ORDER:
        for mouse in mice_per_group.get(group, []):
            if mouse not in sessions:
                continue
            session = sessions[mouse]
            wc = crossreg_to_use[mouse] if (crossreg_to_use is not None and mapping != 'full') else None

            cell_ids, _ = resolve_shared_cells(session, mapping=mapping, with_crossreg=wc)
            z = _standardized_trace(session, SIGNAL, cell_ids)
            trials, _ = retained_trials(session)
            half_a, half_b = split_trials(trials)

            mat_a, time_axis = _tone_aligned_matrix(z, session, half_a)
            mat_b, _ = _tone_aligned_matrix(z, session, half_b)
            if mat_a is None or mat_b is None:
                raise RuntimeError(
                    f'epoch_sequence: mouse {mouse!r} produced no tone-aligned window for one of '
                    f'the trial halves (A={half_a}, B={half_b}), although it retained {trials}.')

            lat_a, def_a = peak_latencies(mat_a, time_axis)
            lat_b, def_b = peak_latencies(mat_b, time_axis)
            defined = def_a & def_b

            for cid, la, lb, dd in zip(cell_ids, lat_a, lat_b, defined):
                rows.append({'mouse': mouse, 'group': group, 'cell': cid,
                             'latency_a': float(la), 'latency_b': float(lb),
                             'defined': bool(dd)})
            mats[(group, mouse)] = {'mat_a': mat_a, 'mat_b': mat_b,
                                    'time_axis': time_axis, 'defined': defined}
            if verbose:
                n = len(cell_ids)
                print(f'[epoch-sequence] {mouse} ({GROUP_LABELS[group]}): trials A={half_a} '
                      f'B={half_b}; {int(defined.sum())}/{n} cells with a defined peak latency '
                      f'in both halves ({100 * defined.mean():.1f}%)', flush=True)

    if not rows:
        raise RuntimeError('epoch_sequence: produced an empty peak table.')
    return pd.DataFrame(rows), mats


# ─────────────────────────────────────────────────────────────────────────────
# Statistic
# ─────────────────────────────────────────────────────────────────────────────

def split_half_correlations(df):
    """Per-mouse Spearman correlation between the two halves' peak latencies.

    Computed WITHIN each mouse over that mouse's cells with a defined latency in both halves.
    One row per mouse: ``mouse, group, n_cells_total, n_cells_defined, frac_defined, rho,
    fisher_z``.

    The correlation's null is zero by construction (the halves share no trials), so no shuffle
    is performed and none is needed.
    """
    rows = []
    for (group, mouse), sub in df.groupby(['group', 'mouse'], observed=True):
        ok = sub[sub['defined']]
        n_total, n_def = len(sub), len(ok)
        if n_def < 3:
            raise RuntimeError(
                f'epoch_sequence: mouse {mouse!r} has only {n_def} cells with a defined peak '
                f'latency in both halves (of {n_total}); a correlation over them would be '
                f'meaningless. Inspect the C traces for this mouse before proceeding.')
        rho = float(spearmanr(ok['latency_a'], ok['latency_b']).statistic)
        rows.append({
            'mouse': mouse, 'group': group,
            'n_cells_total': n_total, 'n_cells_defined': n_def,
            'frac_defined': n_def / n_total,
            'rho': rho,
            # arctanh is undefined at +/-1; clip only to keep the transform finite.
            'fisher_z': float(np.arctanh(np.clip(rho, -0.999999, 0.999999))),
        })
    return pd.DataFrame(rows).sort_values(['group', 'mouse']).reset_index(drop=True)


def format_report(per_mouse):
    """Human-readable summary: per-mouse correlations, retained fractions, and the mouse-level test."""
    z = per_mouse['fisher_z'].to_numpy()
    t, p = ttest_1samp(z, 0.0)
    lines = [
        'Cross-validated sequence test — split-half peak-latency reproducibility',
        '=' * 72,
        f'Signal: {SIGNAL};  window: {WINDOW_START_S:.0f}–{WINDOW_END_S:.0f} s from tone onset '
        f'(tone + trace);  smoothing {SMOOTH_S:g} s',
        'Peak latency defined per cell independently in two DISJOINT trial halves. The',
        'correlation between them has expectation zero under no reproducible temporal tuning,',
        'because the halves share no trials — no shuffle or null model is used.',
        '',
        'Cells with a CONSTANT windowed trace in either half have no defined peak latency and',
        'are excluded (argmax would silently return tone onset for them in both halves, which',
        'would contribute perfect agreement). This is a definability requirement, not a',
        'responsiveness filter: any non-constant cell is kept however weak its activity.',
        '',
        f'{"mouse":8s} {"group":>7s} {"cells":>7s} {"defined":>8s} {"frac":>7s} {"rho":>8s}',
    ]
    for _, r in per_mouse.iterrows():
        lines.append(f'{r["mouse"]:8s} {GROUP_LABELS[r["group"]]:>7s} {r["n_cells_total"]:7d} '
                     f'{r["n_cells_defined"]:8d} {r["frac_defined"]:7.3f} {r["rho"]:+8.4f}')
    lines += [
        '',
        f'Mouse-level test of Fisher-z against 0 (n = {len(z)} mice, the unit of analysis):',
        f'  mean rho = {per_mouse["rho"].mean():+.4f}   mean Fisher-z = {z.mean():+.4f}   '
        f't({len(z) - 1}) = {t:+.3f}   P = {p:.4g}',
        '',
        'A positive correlation alone does NOT establish a sequence. A few reproducible response',
        'times are enough to produce one: two clusters of cells peaking early or late give',
        'rho ~ +0.83 on a planted stub, with only a handful of distinct latencies. (Perfect',
        'synchrony is the opposite case and gives rho ~ 0, since identical latencies are ties.)',
        'Read the correlation together with the held-out latency histogram in the figure — a',
        'sequence requires the peaks to TILE the interval, and only the histogram shows that.',
    ]
    return '\n'.join(lines) + '\n'


# ─────────────────────────────────────────────────────────────────────────────
# Panels
# ─────────────────────────────────────────────────────────────────────────────

def _group_sorted_matrices(mats, group):
    """(sort-half, held-out-half, time axis, held-out latencies) for one group.

    Cells are ordered by their half-A peak latency, and the returned held-out latencies are in
    that same row order.

    ** The ordering is computed WITHIN each mouse and applied to that mouse's own rows before
    the mice are stacked. ** Ordering the pooled matrix instead would let between-animal
    differences in cell count or overall timing contribute to the apparent diagonal, which is a
    version of the very artifact this analysis exists to test.

    ** The latencies are returned rather than recomputed by the caller. ** They come from
    :func:`peak_latencies`, i.e. from the SMOOTHED trace, which is what the reported statistic
    uses. Taking a bare ``argmax`` of the returned (unsmoothed) matrices instead gives a
    different quantity -- for weakly-tuned cells an essentially unrelated one -- and would let
    the figure's latency histogram disagree with the correlation printed beside it.
    """
    blocks_a, blocks_b, blocks_lat_b, time_axis = [], [], [], None
    for (g, _mouse), d in mats.items():
        if g != group:
            continue
        keep = d['defined']
        time_axis = d['time_axis']
        mask = _window_slice(time_axis)
        lat_a, _ = peak_latencies(d['mat_a'], time_axis)
        lat_b, _ = peak_latencies(d['mat_b'], time_axis)
        order = np.argsort(lat_a[keep], kind='stable')
        blocks_a.append(d['mat_a'][keep][order][:, mask])
        blocks_b.append(d['mat_b'][keep][order][:, mask])
        blocks_lat_b.append(lat_b[keep][order])
    if not blocks_a:
        raise RuntimeError(f'epoch_sequence: no mice for group {group!r}.')
    return (np.concatenate(blocks_a, axis=0), np.concatenate(blocks_b, axis=0),
            time_axis[_window_slice(time_axis)], np.concatenate(blocks_lat_b))


def plot_cross_validated_sequence(mats, out_path, vmax=0.4, auto_close=True):
    """The main panel: sort half beside held-out half, cells ordered by sort-half peak latency.

    The LEFT column of each pair will always show a crisp diagonal -- it is sorted by a statistic
    computed from itself, so it would look like this even for pure noise. It is drawn deliberately
    as the comparison. **Only the RIGHT column carries evidence**: if the ordering reflects
    reproducible temporal tuning the diagonal survives on trials never used to define it, and if
    it does not, the right column is featureless.

    The bottom row shows the distribution of HELD-OUT peak latencies, which separates a genuine
    sequence (peaks tiling the interval) from synchrony (peaks piling at one moment).
    """
    n = len(DREADD_DISPLAY_ORDER)
    fig, axs = plt.subplots(2, 2 * n, figsize=(3.0 * 2 * n, 6.6),
                            gridspec_kw={'height_ratios': [3, 1]})

    for j, group in enumerate(DREADD_DISPLAY_ORDER):
        mat_a, mat_b, taxis, lat_b = _group_sorted_matrices(mats, group)
        extent = [taxis[0], taxis[-1], mat_a.shape[0], 0]
        for k, (mat, label) in enumerate(((mat_a, 'sort half (A)'),
                                          (mat_b, 'HELD-OUT half (B)'))):
            ax = axs[0, 2 * j + k]
            ax.imshow(mat, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax, extent=extent)
            ax.axvline(20.0, color='k', lw=0.8, ls=':')
            ax.set_title(f'{GROUP_LABELS[group]} — {label}',
                         size='medium', fontweight='bold' if k else 'normal')
            ax.set_xlabel('Time from tone onset (s)')
            if 2 * j + k == 0:
                ax.set_ylabel('Cell (sorted by peak latency in half A)')

        # Held-out latency distribution: tiling vs synchrony. Uses the SAME smoothed latencies
        # the correlation is computed from (see _group_sorted_matrices), so the histogram and
        # the reported statistic describe one quantity.
        ax = axs[1, 2 * j + 1]
        ax.hist(lat_b, bins=40, range=(taxis[0], taxis[-1]), color=GROUP_COLOURS[group], alpha=0.8)
        ax.axvline(20.0, color='k', lw=0.8, ls=':')
        ax.set_xlabel('Held-out peak latency (s)')
        ax.set_ylabel('cells')
        axs[1, 2 * j].axis('off')

    fig.suptitle('Cross-validated sequence test (C): left panel of each pair is sorted by itself '
                 'and is diagonal by construction; only the held-out panel is evidence',
                 size='medium')
    fig.subplots_adjust(left=0.05, right=0.99, bottom=0.09, top=0.88, wspace=0.30, hspace=0.45)
    save_fig(fig, out_path)
    if auto_close:
        plt.close(fig)


def plot_split_half_correlations(per_mouse, out_path, auto_close=True):
    """Per-mouse split-half peak-latency correlation, by group, against the zero null."""
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    rng = np.random.default_rng(0)
    for i, group in enumerate(DREADD_DISPLAY_ORDER):
        vals = per_mouse[per_mouse['group'] == group]['rho'].to_numpy()
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(vals)), vals, s=46,
                   color=GROUP_COLOURS[group], edgecolors='black', linewidths=0.6, zorder=3)
        ax.plot([i - 0.24, i + 0.24], [vals.mean()] * 2, color='black', lw=1.6, zorder=4)
    ax.axhline(0.0, color='0.5', lw=1.0, ls='--', zorder=1)
    ax.set_xticks(range(len(DREADD_DISPLAY_ORDER)))
    ax.set_xticklabels([GROUP_LABELS[g] for g in DREADD_DISPLAY_ORDER])
    ax.set_ylabel("Split-half peak-latency correlation (Spearman $\\rho$)")
    ax.set_title('One point per mouse; 0 = no reproducible tuning', size='medium')
    fig.subplots_adjust(left=0.20, right=0.97, bottom=0.11, top=0.91)
    save_fig(fig, out_path)
    if auto_close:
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch_sequence(PLOTS_DIR, mice_per_group, TFC_cond, mapping='full',
                       crossreg_to_use=None, vmax=0.4, auto_close=True):
    """Run the cross-validated sequence test. Exploratory; changes no existing analysis.

    Outputs under ``<PLOTS_DIR>/epoch_sequence/``:
      cross_validated_sequence.png/.svg   sort half vs held-out half, per group
      split_half_correlations.png/.svg    per-mouse correlation
      tables/per_cell_peak_latency.csv    per-cell latencies in both halves + defined flag
      tables/per_mouse_correlations.csv   per-mouse rho, Fisher-z, retained cell counts
      stats/sequence_report.txt           the above in readable form + the mouse-level test
    """
    out_dir = os.path.join(PLOTS_DIR, 'epoch_sequence')
    tables_dir, stats_dir = os.path.join(out_dir, 'tables'), os.path.join(out_dir, 'stats')
    ensure_dirs(out_dir, tables_dir, stats_dir)

    print(f'[epoch-sequence] building split-half peak latencies ({SIGNAL}, '
          f'{WINDOW_START_S:.0f}-{WINDOW_END_S:.0f} s)...', flush=True)
    df, mats = build_peak_table(mice_per_group, TFC_cond, mapping=mapping,
                                crossreg_to_use=crossreg_to_use)
    per_mouse = split_half_correlations(df)

    df.to_csv(os.path.join(tables_dir, 'per_cell_peak_latency.csv'), index=False)
    per_mouse.to_csv(os.path.join(tables_dir, 'per_mouse_correlations.csv'), index=False)
    report = format_report(per_mouse)
    write_text(os.path.join(stats_dir, 'sequence_report.txt'), report)

    plot_cross_validated_sequence(mats, os.path.join(out_dir, 'cross_validated_sequence.png'),
                                  vmax=vmax, auto_close=auto_close)
    plot_split_half_correlations(per_mouse,
                                 os.path.join(out_dir, 'split_half_correlations.png'),
                                 auto_close=auto_close)
    print(report, flush=True)
    return {'per_cell': df, 'per_mouse': per_mouse}
