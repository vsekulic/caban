"""Navigation-aware single-cell rate analyses.

The whole-session spike-rate / activity panels produced by ``run_sp_rates`` average over
*every* cell in a mapping and over *every* frame of the recording. A DREADD group that simply
navigates more will therefore show a higher average rate for purely behavioural reasons --
more running recruits more place cells, and fewer immobile frames dilute the average.

This module decomposes that single number along two orthogonal axes:

  * **cell class**  -- place cell vs non-place cell vs all
  * **frame class** -- all frames vs movement-only vs immobility-only

crossed with two analysis windows (whole session, and the pre-tone period used by the occupancy
and 2D spatial-information suites) and both metrics (event rate, peak-S activity).

Everything here is *additive*: no existing plot, directory, or place-field routine is modified.
Place fields are read from the ``*-PlaceFields.npz`` caches that ``run_pf_and_loc`` /
``run_LT_pfs`` already wrote, so these analyses have no ordering dependency on those sections.

Companion module: :mod:`caban.locomotion`, which supplies the locomotion covariates.
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt

from caban.utilities import (
    MINISCOPE_FPS,
    VELOCITY_THRESHOLD,
    get_avg_activity_in_frame_mask,
    get_avg_sp_rate_in_frame_mask,
    get_spike_count_in_frame_mask,
)
from caban.spatial import PlaceFields, _build_pretone_mask
# caban.decoder must be imported BEFORE caban.analysis: the two are mutually dependent, and
# decoder defines the constants analysis needs before its own deferred import of analysis.
# Entering the cycle from analysis instead leaves decoder's import of it half-initialised.
# This mirrors the bootstrap ordering in caban/sections.py.
from caban.decoder import _copy_analysis_methods_template, pf_bin_area_cm2
from caban.analysis import _WHOLE_SESSION_YLABEL, _draw_violin_triplet, group_colours

# ---------------------------------------------------------------------------
# Shared vocabulary
# ---------------------------------------------------------------------------

#: Top-level directory under PLOTS_DIR that every analysis in this suite writes beneath.
NAV_AWARE_DIR = 'navigation_aware_single_cell'

GROUP_ORDER = ['hM3D', 'mCherry', 'hM4D']
GROUP_LABELS = {'hM3D': 'Exc', 'mCherry': 'Ctl', 'hM4D': 'Inh'}

WINDOW_WHOLE_SESSION = 'whole_session'
WINDOW_PRETONE = 'pretone_180s'
WINDOWS = [WINDOW_WHOLE_SESSION, WINDOW_PRETONE]

FRAME_CLASS_ALL = 'all_frames'
FRAME_CLASS_MOVEMENT = 'movement_only'
FRAME_CLASS_IMMOBILITY = 'immobility_only'
FRAME_CLASSES = [FRAME_CLASS_ALL, FRAME_CLASS_MOVEMENT, FRAME_CLASS_IMMOBILITY]

CELL_CLASSES = ['place', 'non_place', 'all']
CELL_CLASS_TITLES = {'place': 'Place cells', 'non_place': 'Non-place cells', 'all': 'All cells'}

#: Axis labels, imported rather than restated so these panels stay identical to the whole-session
#: sp_rates panels they decompose. See the note there on why it is "Avg." and not "Population".
_METRIC_YLABEL = _WHOLE_SESSION_YLABEL

#: Sessions with no tones, for which the pre-tone window is undefined.
_SESSIONS_WITHOUT_TONES = ('LT1', 'LT2')


def windows_for_session(session_type):
    """Analysis windows applicable to *session_type*.

    The linear-track sessions have no tones, so ``_build_pretone_mask`` would silently fall back
    to "first 180 s", which is not a meaningful epoch there -- they get the whole session only.
    """
    if session_type in _SESSIONS_WITHOUT_TONES:
        return [WINDOW_WHOLE_SESSION]
    return list(WINDOWS)


# ---------------------------------------------------------------------------
# Place-field loading -- standalone, off the cached npz
# ---------------------------------------------------------------------------

#: {(session_type, mouse): PlaceFields} -- avoids re-reading the same npz once per
#: (window x frame class x metric) combination, which is 12 reads per mouse per mapping.
_PF_CACHE = {}


def clear_place_field_cache():
    """Drop the in-process place-field cache (call after re-fitting place fields)."""
    _PF_CACHE.clear()


def load_place_fields(sess, mouse):
    """Return a populated ``PlaceFields`` for *mouse*'s *sess*.

    Prefers ``sess.fm.pf`` when a FluorescenceMap is already attached (i.e. ``run_pf_and_loc`` /
    ``run_LT_pfs`` ran in this kernel). Otherwise reconstitutes it from the cached
    ``<session_type>-<mouse>-<n>cells-PlaceFields.npz``, which is the same load path
    ``PlaceFields.__init__`` uses for the pickle round-trip -- no new file-format code, and no
    re-fitting.

    Raises when neither source is available, naming the section that produces the cache.
    """
    fm = getattr(sess, 'fm', None)
    if fm is not None and getattr(fm, 'pf', None) is not None:
        return fm.pf

    key = (sess.session_type, mouse)
    if key in _PF_CACHE:
        return _PF_CACHE[key]

    pf = PlaceFields(to_pickle=True, sess=sess, mouse=mouse)
    if not pf.loaded:
        raise RuntimeError(
            'No place fields for {} {}: neither sess.fm.pf in memory nor a cached npz at {}. '
            'Run run_pf_and_loc(ds, cfg) (chamber sessions) or run_LT_pfs(ds, cfg) (LT1/LT2) '
            'once to produce it.'.format(mouse, sess.session_type, pf.load_pickle_path)
        )
    _PF_CACHE[key] = pf
    return pf


def _place_cell_positions(pf):
    """Row indices (positions into ``sess.S``) of cells with at least one detected place field.

    The PF dicts are keyed by position index into S -- see the ``_pf_pos`` docstring in
    caban.spatial -- which is the same index space ``get_S_mapping`` keys ``S_spikes`` by, so
    the two intersect directly. This mirrors what ``plot_pf_analyses`` does.
    """
    return {int(k) for k, sizes in pf.pf_size.items() if len(sizes) > 0}


def partition_place_cells(sess, mouse, mapping, s_spikes=None):
    """Split a mapping's cells into place vs non-place.

    *s_spikes* may be passed in to avoid re-resolving the mapping; when omitted it is derived
    from ``sess.get_S_mapping(mapping)``.

    Returns ``{'place': [...], 'non_place': [...], 'all': [...], 'n_place': int, 'n_total': int}``
    where the lists hold row indices into ``sess.S``.
    """
    if s_spikes is None:
        _, s_spikes, _, _ = sess.get_S_mapping(mapping)

    mapping_positions = [int(i) for i in s_spikes.keys()]
    pf_positions = _place_cell_positions(load_place_fields(sess, mouse))

    place = [i for i in mapping_positions if i in pf_positions]
    non_place = [i for i in mapping_positions if i not in pf_positions]

    if len(place) + len(non_place) != len(mapping_positions):
        raise RuntimeError(
            'place/non-place partition lost cells for {} {} mapping={}: '
            '{} + {} != {}'.format(mouse, sess.session_type, mapping,
                                   len(place), len(non_place), len(mapping_positions))
        )

    return {
        'place': place,
        'non_place': non_place,
        'all': mapping_positions,
        'n_place': len(place),
        'n_non_place': len(non_place),
        'n_total': len(mapping_positions),
    }


# ---------------------------------------------------------------------------
# Frame masks
# ---------------------------------------------------------------------------

def build_frame_mask(sess, window, frame_class, first_n_sec=180.0,
                     speed_thresh=VELOCITY_THRESHOLD):
    """Boolean mask over frames of the trimmed ``sess.S`` for one (window, frame_class) cell.

    The window component reuses ``caban.spatial._build_pretone_mask`` so these panels align
    exactly with the occupancy and 2D spatial-information suites. The movement component uses
    the identical velocity expression as ``BehaviourSession._compute_velocity_masked_activity``
    (which is what ``S_mov`` / ``S_imm`` are built from).
    """
    n_frames = sess.S.shape[1]

    if window == WINDOW_WHOLE_SESSION:
        mask = np.ones(n_frames, dtype=bool)
    elif window == WINDOW_PRETONE:
        if sess.session_type in _SESSIONS_WITHOUT_TONES:
            raise ValueError(
                'Window {!r} is undefined for {} (no tones) -- use windows_for_session().'.format(
                    window, sess.session_type)
            )
        mask = np.asarray(_build_pretone_mask(sess, first_n_sec=first_n_sec), dtype=bool)
    else:
        raise ValueError('Unknown window {!r}; expected one of {}.'.format(window, WINDOWS))

    if frame_class == FRAME_CLASS_ALL:
        return mask

    velocities = np.asarray(sess.velocities_miniscope_smooth[:n_frames], dtype=float)
    if len(velocities) != n_frames:
        raise RuntimeError(
            'Velocity trace for {} {} has {} samples but S has {} frames; cannot build a '
            'movement mask.'.format(sess.mouse, sess.session_type, len(velocities), n_frames)
        )
    if frame_class == FRAME_CLASS_MOVEMENT:
        return mask & (velocities >= speed_thresh)
    if frame_class == FRAME_CLASS_IMMOBILITY:
        return mask & (velocities < speed_thresh)
    raise ValueError('Unknown frame_class {!r}; expected one of {}.'.format(frame_class, FRAME_CLASSES))


# ---------------------------------------------------------------------------
# Rate computation
# ---------------------------------------------------------------------------

def rates_by_cell_class(sess, mouse, mapping, *, want_peakval=False,
                        window=WINDOW_WHOLE_SESSION, frame_class=FRAME_CLASS_ALL,
                        first_n_sec=180.0, speed_thresh=VELOCITY_THRESHOLD):
    """Cell-averaged rate (or activity) for each cell class over one (window, frame_class) cell.

    Returns a dict with a float per entry of :data:`CELL_CLASSES` plus the cell counts, the
    number of eligible frames, and the total event count (used by the partition assertions).

    With ``window='whole_session', frame_class='all_frames'`` the ``'all'`` entry reproduces
    ``BehaviourSession.process_whole_session_sp_rates_mapping()`` exactly -- that equivalence is
    asserted by :func:`assert_decomposition_consistent`.
    """
    _, s_spikes, s_peakval, _ = sess.get_S_mapping(mapping, want_peakval=want_peakval)
    partition = partition_place_cells(sess, mouse, mapping, s_spikes=s_spikes)
    frame_mask = build_frame_mask(sess, window, frame_class,
                                  first_n_sec=first_n_sec, speed_thresh=speed_thresh)

    n_eligible = int(frame_mask.sum())
    out = {
        'n_place': partition['n_place'],
        'n_non_place': partition['n_non_place'],
        'n_total': partition['n_total'],
        'n_frames': n_eligible,
        'duration_s': float(n_eligible) / MINISCOPE_FPS,
        'defined': True,
    }

    if n_eligible < MINISCOPE_FPS:
        # Under a second of eligible time. This is a real behavioural property, not an error:
        # G06 never drops below 2 cm/s on LT1 (minimum smoothed speed 2.06 cm/s over 612 s), so
        # it has no immobility period and therefore no defined immobility rate. Report it as
        # undefined and let the caller exclude it explicitly and say so, rather than either
        # raising (which would kill the whole panel over one animal) or emitting a zero (which
        # would be read as a measured rate of zero).
        out['defined'] = False
        for cell_class in CELL_CLASSES:
            out[cell_class] = float('nan')
            out['n_events_' + cell_class] = 0
        return out

    for cell_class in CELL_CLASSES:
        cells = partition[cell_class]
        if len(cells) == 0:
            # A genuinely empty class (e.g. a mouse with no detected place fields) is data, not
            # an error -- but it must be visible as NaN rather than silently averaged as zero.
            out[cell_class] = float('nan')
            out['n_events_' + cell_class] = 0
            continue
        sub_spikes = {i: s_spikes[i] for i in cells}
        if want_peakval:
            out[cell_class] = get_avg_activity_in_frame_mask(
                sub_spikes, {i: s_peakval[i] for i in cells}, frame_mask)
        else:
            out[cell_class] = get_avg_sp_rate_in_frame_mask(sub_spikes, frame_mask)
        out['n_events_' + cell_class] = get_spike_count_in_frame_mask(sub_spikes, frame_mask)[0]

    return out


def assert_decomposition_consistent(sess, mouse, mapping, *, want_peakval=False, rtol=1e-9):
    """Assert the new machinery reproduces the existing whole-session number.

    ``rates_by_cell_class(window='whole_session', frame_class='all_frames')['all']`` must equal
    ``sess.process_whole_session_sp_rates_mapping(mapping)[0]``. A mismatch means the frame mask
    or the cell subsetting has drifted from the established convention, and must fail loudly.
    """
    new = rates_by_cell_class(sess, mouse, mapping, want_peakval=want_peakval,
                              window=WINDOW_WHOLE_SESSION, frame_class=FRAME_CLASS_ALL)['all']
    existing = sess.process_whole_session_sp_rates_mapping(mapping, want_peakval=want_peakval)[0]

    # process_whole_session_sp_rates_mapping() spans frames [0, n-1], i.e. n-1 frames' worth of
    # duration, whereas the all-True frame mask spans all n. Rescale before comparing.
    n_frames = sess.S.shape[1]
    expected = existing * (n_frames - 1) / n_frames

    if not np.isclose(new, expected, rtol=rtol, atol=0.0):
        raise AssertionError(
            'Whole-session decomposition mismatch for {} {} mapping={} (peakval={}): '
            'rates_by_cell_class -> {!r}, process_whole_session_sp_rates_mapping -> {!r} '
            '(duration-adjusted {!r}).'.format(
                mouse, sess.session_type, mapping, want_peakval, new, existing, expected)
        )
    return new, existing


def assert_frame_partition_additive(sess, mouse, mapping, *, window=WINDOW_WHOLE_SESSION):
    """Assert movement and immobility frames form a disjoint cover of the window.

    Event *counts* are additive across a disjoint frame partition (rates are not, because the
    denominators differ), so this checks counts and frame totals rather than rates.
    """
    _, s_spikes, _, _ = sess.get_S_mapping(mapping)

    totals = {}
    for frame_class in FRAME_CLASSES:
        mask = build_frame_mask(sess, window, frame_class)
        totals[frame_class] = get_spike_count_in_frame_mask(s_spikes, mask)

    n_events_all, n_frames_all = totals[FRAME_CLASS_ALL]
    n_events_split = totals[FRAME_CLASS_MOVEMENT][0] + totals[FRAME_CLASS_IMMOBILITY][0]
    n_frames_split = totals[FRAME_CLASS_MOVEMENT][1] + totals[FRAME_CLASS_IMMOBILITY][1]

    if n_events_all != n_events_split or n_frames_all != n_frames_split:
        raise AssertionError(
            'Movement/immobility frames do not partition the {} window for {} {} mapping={}: '
            'events {} != {}+{}, frames {} != {}+{}.'.format(
                window, mouse, sess.session_type, mapping,
                n_events_all, totals[FRAME_CLASS_MOVEMENT][0], totals[FRAME_CLASS_IMMOBILITY][0],
                n_frames_all, totals[FRAME_CLASS_MOVEMENT][1], totals[FRAME_CLASS_IMMOBILITY][1])
        )
    return totals


# ---------------------------------------------------------------------------
# Per-mouse place-field properties (the n = mice companion to plot_pf_analyses)
# ---------------------------------------------------------------------------

def pf_properties_per_mouse(sess, mouse, mapping):
    """Collapse a mouse's place-field measures to one value each.

    ``plot_pf_analyses`` pools these across mice and tests with n = cells; that pooled version is
    retained deliberately. This is the per-mouse counterpart, so a group comparison can use
    n = mice. Every mouse present in the session dict is included -- there is no blanket
    G07/G15 skip here.

    Per-field measures (size, compactness, spatial selectivity) are averaged over all fields of
    all place cells in the mapping. ``mean_n_pfs`` counts fields per *place cell*, not per cell,
    so it is not confounded by the proportion of place cells (which is reported separately).
    """
    pf = load_place_fields(sess, mouse)
    _, s_spikes, _, _ = sess.get_S_mapping(mapping)
    mapping_positions = {int(i) for i in s_spikes.keys()}

    n_pfs, sizes, compactness, selectivity = [], [], [], []
    n_place = 0
    for pos, field_sizes in pf.pf_size.items():
        pos = int(pos)
        if pos not in mapping_positions or len(field_sizes) == 0:
            continue
        n_place += 1
        n_pfs.append(len(field_sizes))
        sizes.extend(list(field_sizes))
        compactness.extend(list(pf.compactness_pf[pos]))
        selectivity.extend(list(pf.spatial_selectivity[pos]))

    n_total = len(mapping_positions)
    if n_total == 0:
        raise RuntimeError(
            'Mapping {!r} resolved to 0 cells for {} {}.'.format(mapping, mouse, sess.session_type))

    # pf_size is a bounding-box area in BINNED coordinates; scale to cm^2. The factor is
    # session-type specific (6.004 cm^2 per bin in the chambers, 8.123 on the linear track)
    # because the behaviour camera sat differently over the two rigs -- see _PF_MAP_CM_PER_PX.
    bin_area = pf_bin_area_cm2(sess.session_type)

    return {
        'pct_place_cells': 100.0 * n_place / n_total,
        'n_place_cells': n_place,
        'n_cells': n_total,
        'mean_n_pfs': float(np.mean(n_pfs)) if n_pfs else float('nan'),
        'mean_pf_size': float(np.mean(sizes)) * bin_area if sizes else float('nan'),
        'pf_size_unit': 'cm^2',
        'mean_pf_compactness': float(np.mean(compactness)) if compactness else float('nan'),
        'mean_spatial_selectivity': float(np.mean(selectivity)) if selectivity else float('nan'),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _values_per_group(per_mouse, mouse_groups, columns, context):
    """Build the ``{group: (n_mice, n_cols)}`` array _draw_violin_triplet expects.

    Groups are populated from the mice actually present in *per_mouse*, not from *mouse_groups* --
    Test_B has no G07 and Test_A_1wk / Test_B_1wk have no G15.
    """
    values, names = {}, {}
    for group in GROUP_ORDER:
        mice = sorted(m for m in per_mouse if mouse_groups[m] == group)
        if len(mice) < 2:
            raise RuntimeError(
                '{}: group {} has {} mouse/mice ({}); need >=2 for a group comparison.'.format(
                    context, group, len(mice), mice)
            )
        arr = np.array([[float(per_mouse[m][c]) for c in columns] for m in mice], dtype=float)
        if not np.all(np.isfinite(arr)):
            bad = [(mice[r], columns[c]) for r, c in zip(*np.where(~np.isfinite(arr)))]
            raise RuntimeError(
                '{}: non-finite value(s) for {} -- a NaN here means a cell class resolved to '
                'zero cells (e.g. a mouse with no detected place fields). Investigate rather '
                'than plotting around it.'.format(context, bad)
            )
        values[group] = arr
        names[group] = mice
    return values, names


def _panel_max(values, col_idx):
    """Largest value in one column across all groups; 1.0 when everything is zero."""
    top = max(float(np.max(values[g][:, col_idx])) for g in GROUP_ORDER)
    return top if top > 0 else 1.0


def _panel_row(values, names, columns, titles, ylabel, figsize, show_mouse_names, suptitle=None):
    """Draw one row of violin-triplet panels sharing a y-label; returns the figure.

    *suptitle* carries whatever is common to the row (session, mapping, window, frame class) so
    the per-panel titles stay short; tight_layout is given headroom for it rather than letting
    it land on top of the panel titles.
    """
    fig, axes = plt.subplots(1, len(columns), figsize=figsize)
    axes = np.atleast_1d(axes)
    for col_idx, (ax, title) in enumerate(zip(axes, titles)):
        ax.spines[['right', 'top']].set_visible(False)
        panel_max = _panel_max(values, col_idx)
        _draw_violin_triplet(
            ax, values, col_idx, GROUP_ORDER, group_colours,
            mouse_names_per_group=names if show_mouse_names else None,
            ylim=(0.0, panel_max * 1.35),
            ylabel=ylabel if col_idx == 0 else None,
        )
        ax.set_xticks(range(len(GROUP_ORDER)))
        ax.set_xticklabels([GROUP_LABELS[g] for g in GROUP_ORDER], size='medium')
        ax.set_title(title, size='small')
    if suptitle is not None:
        fig.suptitle(suptitle, size='small')
        fig.tight_layout(pad=0.5, rect=(0.0, 0.0, 1.0, 0.94))
    else:
        fig.tight_layout(pad=0.5)
    return fig


def _save(fig, save_dir, stem, auto_close=True):
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, stem + '.png'), format='png', dpi=300)
    fig.savefig(os.path.join(save_dir, stem + '.svg'), format='svg')
    if auto_close:
        plt.close(fig)


def plot_place_cell_rate_split(PLOTS_DIR, mouse_groups, per_mouse, session_type, mapping,
                               window, frame_class, want_peakval=False,
                               figsize=(7.2, 3.2), auto_close=True):
    """Place / non-place / all cell-averaged rate, one value per mouse, compared across groups.

    Unit of analysis is the MOUSE: one-way ANOVA gated with Tukey HSD, n = mice -- the same
    machinery as ``plot_whole_session_sp_rates``.

    per_mouse - {mouse: dict from rates_by_cell_class()}
    """
    context = 'plot_place_cell_rate_split({}, {}, {}, {})'.format(
        session_type, mapping, window, frame_class)
    if not per_mouse:
        raise RuntimeError(
            '{}: no mouse has a defined rate for this frame class -- every animal lacks frames '
            'of this kind. There is nothing to plot.'.format(context))
    values, names = _values_per_group(per_mouse, mouse_groups, CELL_CLASSES, context)

    titles = [CELL_CLASS_TITLES[c] for c in CELL_CLASSES]
    fig = _panel_row(values, names, CELL_CLASSES, titles, _METRIC_YLABEL[want_peakval],
                     figsize, show_mouse_names=False,
                     suptitle='{} — {} — {} / {}'.format(
                         session_type, mapping, window, frame_class))

    save_dir = os.path.join(PLOTS_DIR, NAV_AWARE_DIR, 'place_cell_rates',
                            session_type, window, frame_class)
    _save(fig, save_dir, '{}_place_cell_rate_split-{}'.format(session_type, mapping), auto_close)
    return save_dir


def plot_place_cell_proportion(PLOTS_DIR, mouse_groups, per_mouse, session_type, mapping,
                               figsize=(2.6, 3.2), auto_close=True):
    """Percentage of cells with at least one detected place field, one value per mouse."""
    context = 'plot_place_cell_proportion({}, {})'.format(session_type, mapping)
    values, names = _values_per_group(per_mouse, mouse_groups, ['pct_place_cells'], context)

    fig = _panel_row(values, names, ['pct_place_cells'], ['{}'.format(session_type)],
                     'Place cells (% of mapped cells)', figsize, show_mouse_names=True)

    save_dir = os.path.join(PLOTS_DIR, NAV_AWARE_DIR, 'place_cell_proportions')
    _save(fig, save_dir, 'place_cell_proportion-{}-{}'.format(session_type, mapping), auto_close)
    return save_dir


_PF_PROPERTY_COLUMNS = ['mean_n_pfs', 'mean_pf_size', 'mean_pf_compactness',
                        'mean_spatial_selectivity']
_PF_PROPERTY_TITLES = ['Number of PFs', 'PF size ($cm^2$)', 'PF compactness', 'Spatial selectivity']
_PF_PROPERTY_STEMS = ['num_pfs', 'pf_size', 'pf_compactness', 'spatial_selectivity']


def plot_pf_properties_per_mouse(PLOTS_DIR, mouse_groups, per_mouse, session_type, mapping,
                                 figsize=(9.0, 3.2), auto_close=True):
    """Per-mouse (n = mice) companion to ``plot_pf_analyses``'s pooled-cell CDFs and KS tests.

    Emits the same four measures under the same filename stems, so the two versions can be read
    side by side -- but into a separate directory tree. The pooled version is retained
    deliberately and is not modified.
    """
    context = 'plot_pf_properties_per_mouse({}, {})'.format(session_type, mapping)
    values, names = _values_per_group(per_mouse, mouse_groups, _PF_PROPERTY_COLUMNS, context)

    save_dir = os.path.join(PLOTS_DIR, NAV_AWARE_DIR, 'place_field_properties_per_mouse',
                            session_type)

    # Combined row first, then one file per measure (matching the per-panel convention used for
    # multi-panel DREADD figures elsewhere).
    fig = _panel_row(values, names, _PF_PROPERTY_COLUMNS, _PF_PROPERTY_TITLES,
                     'Per-mouse mean', figsize, show_mouse_names=False,
                     suptitle='{} — {} (n = mice)'.format(session_type, mapping))
    _save(fig, save_dir, 'pf_properties_4panel-{}-{}'.format(session_type, mapping), auto_close)

    for col_idx, (stem, title) in enumerate(zip(_PF_PROPERTY_STEMS, _PF_PROPERTY_TITLES)):
        single = {g: values[g][:, [col_idx]] for g in GROUP_ORDER}
        fig = _panel_row(single, names, [0], [title], title, (2.6, 3.2), show_mouse_names=True)
        _save(fig, save_dir, '{}-{}-{}'.format(stem, session_type, mapping), auto_close)

    return save_dir


def copy_methods_templates(PLOTS_DIR):
    """Drop the METHODS templates into the directories this module writes."""
    root = os.path.join(PLOTS_DIR, NAV_AWARE_DIR)
    for template, subdir in (
        ('place_cell_restricted_rates_methods.txt', 'place_cell_rates'),
        ('place_cell_proportion_methods.txt', 'place_cell_proportions'),
        ('place_field_properties_per_mouse_methods.txt', 'place_field_properties_per_mouse'),
    ):
        dest = os.path.join(root, subdir)
        os.makedirs(dest, exist_ok=True)
        _copy_analysis_methods_template(template, dest)
