"""Speed-binned robustness analysis for the navigation-aware single-cell suite.

``caban.place_cell_rates`` splits the cell-averaged rate by a *binary* movement / immobility frame
class at 2 cm/s, and ``caban.locomotion`` shows the viral groups do not differ on any locomotion
summary. Hippocampal firing is nevertheless *graded* with running speed, and matched **mean**
speed does not guarantee matched speed **distributions** -- so a binary threshold leaves a
residual, if narrow, behavioural explanation open.

This module closes it, using the two approaches the literature actually uses -- speed as a
covariate, and reweighting to a common speed distribution -- rather than dividing rate by speed
(which is not a hippocampal convention: the rate/speed relation has a nonzero intercept, so the
ratio diverges as speed goes to zero).

Four deliverables, in dependency order:

1. **Speed-occupancy distributions.** Does the *distribution* differ, not just the mean? This is
   the premise the whole analysis rests on and nothing else tests it. It also licenses (4): if
   the distributions superimpose, the standardized rate must equal the raw rate.
2. **Speed tuning curves.** Cell-averaged rate vs speed bin, per group, absolute and shape-normalized.
3. **Poisson GLM.** ``log E[count_b] = alpha + beta*speed_b + log(seconds_b)``, per cell and per
   mouse; the test is a **group x speed interaction** and a null interaction is the desired result.
   The log link is load-bearing -- see :func:`fit_speed_glm`.
4. **Speed-standardized rate.** Reweighting each mouse's per-bin rates to a common reference
   speed distribution, reported alongside the *shift* from the raw rate.

Everything is additive: output lands under ``PLOTS_DIR/navigation_aware_single_cell/speed_tuning/``
and no existing analysis is modified.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats

from caban.utilities import (
    MINISCOPE_FPS,
    VELOCITY_THRESHOLD,
    get_per_cell_spike_count_in_frame_mask,
    get_spike_count_in_frame_mask,
)
# caban.decoder must be imported BEFORE caban.analysis: the two are mutually dependent, and
# decoder defines the constants analysis needs before its own deferred import of analysis.
# This mirrors the bootstrap ordering in caban/sections.py and caban/place_cell_rates.py.
from caban.decoder import _copy_analysis_methods_template
from caban.analysis import _draw_violin_triplet, group_colours
# GROUP_ORDER is taken from place_cell_rates (hM3D, mCherry, hM4D -- the CLAUDE.md order), NOT
# from single_unit_common, which uses hM3D, hM4D, mCherry. This module is the third to trip over
# that collision; the fix is to thread the order through (see ecdf_panel's group_order kwarg),
# never to change single_unit_common, whose default four published analyses depend on.
from caban.place_cell_rates import (
    FRAME_CLASS_ALL,
    FRAME_CLASS_IMMOBILITY,
    FRAME_CLASS_MOVEMENT,
    GROUP_LABELS,
    GROUP_ORDER,
    NAV_AWARE_DIR,
    WINDOW_WHOLE_SESSION,
    _METRIC_YLABEL,
    _save,
    _values_per_group,
    build_frame_mask,
    partition_place_cells,
    set_session_title,
)
from caban.single_unit_common import (
    bracket_ylim,
    build_cell_records,
    ecdf_panel,
    fdr_correct,
    fit_group_mixed_model,
)

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

SPEED_TUNING_DIR = 'speed_tuning'

#: Speed bin edges in cm/s. Fixed and global, deliberately:
#:
#:  * NOT quantile-based -- the speed distribution is heavily zero-inflated, so deciles would
#:    spend most bins subdividing the sub-threshold tracking noise floor and give the whole
#:    locomotion range two or three bins.
#:  * NOT per-session -- retained bins would be incomparable across sessions and the
#:    standardization weights would become a session-specific artifact that cannot be audited.
#:
#: The first internal edge is VELOCITY_THRESHOLD, so bin 0 is EXACTLY the existing
#: ``immobility_only`` frame class and bins 1.. are EXACTLY ``movement_only``. That makes this
#: analysis nest inside caban.place_cell_rates and yields the headline wiring assertion,
#: :func:`assert_bin_zero_is_immobility`. Sub-threshold speed is one bin and not several because
#: below 2 cm/s the signal is mostly tracking jitter convolved with the 200 ms smoothing kernel --
#: subdividing it would measure the smoother, not the animal. The top bin is open so no fast
#: frames are discarded.
SPEED_BIN_EDGES = (0.0, VELOCITY_THRESHOLD, 4.0, 6.0, 8.0, 12.0, 16.0, np.inf)

N_SPEED_BINS = len(SPEED_BIN_EDGES) - 1

#: A bin is retained for a session only if EVERY mouse in it has at least this much occupancy.
#: See :func:`retained_bins_for_session` for why retention is common across mice rather than
#: per-mouse.
MIN_BIN_SECONDS = 3.0

#: Minimum number of retained bins for a session to be analysable at all.
MIN_RETAINED_BINS = 4

#: Fraction of a bin set's natural universe (see :func:`coverage_fraction`) that must survive the
#: retention rule for a mouse. Set at 0.75 rather than a rounder 0.80 for a specific, observed
#: reason: on LT1 the common-bin rule drops the immobility bin for the whole session because G06
#: never stops, which leaves the worst-covered mouse there at 0.783. Failing that session would
#: cost a real recording to no scientific end, so the threshold sits just below the observed
#: worst case and the actual value is always printed and put in the panel title.
MIN_COVERAGE_FRACTION = 0.75

#: Per-cell eligibility for the Poisson fit. Screening up front (rather than catching a fit
#: failure) keeps the estimator itself free to hard-fail, per CLAUDE.md.
MIN_CELL_EVENTS = 10
MIN_CELL_OCCUPIED_BINS = 2

#: Sanity bound on a fitted per-cell slope, in log-rate per cm/s. |beta| = 1 would mean an
#: e-fold rate change per cm/s, which is far outside anything physiological.
BETA_ABS_MAX = 5.0

BIN_SET_ALL = 'all_bins'
BIN_SET_MOVING = 'moving_bins'
BIN_SETS = [BIN_SET_ALL, BIN_SET_MOVING]

REFERENCE_ALL_MICE = 'all_mice_mean'
REFERENCE_CONTROL = 'mCherry_mean'
REFERENCES = [REFERENCE_ALL_MICE, REFERENCE_CONTROL]

#: Deconvolution latency sensitivity: S event frames lag the true spikes while the speed kernel is
#: symmetric, so misalignment during acceleration ATTENUATES the slope -- biasing towards the null
#: this analysis hopes to find. Reported as a sensitivity run, never silently.
SENSITIVITY_LAG_FRAMES = 4


def bin_set_indices(bin_set, retained):
    """Indices of the bins a given bin set uses, within the session's retained bins.

    ``moving_bins`` drops bin 0. Bin 0 holds freezing, which is itself a DREADD effect -- i.e. a
    mediator -- so standardizing over a bin set that includes it partially adjusts away part of
    the effect under study. Both variants are therefore always reported.
    """
    retained = np.asarray(retained, dtype=bool)
    if bin_set == BIN_SET_ALL:
        keep = retained.copy()
    elif bin_set == BIN_SET_MOVING:
        keep = retained.copy()
        keep[0] = False
    else:
        raise ValueError('Unknown bin_set {!r}; expected one of {}.'.format(bin_set, BIN_SETS))
    idx = np.flatnonzero(keep)
    if len(idx) < 2:
        raise RuntimeError(
            'bin_set {!r} leaves {} bin(s) from retained={}; a speed relationship needs at least '
            '2.'.format(bin_set, len(idx), np.flatnonzero(retained).tolist()))
    return idx


def bin_sets_for_session(retained):
    """Bin sets worth running for a session, given its retained bins.

    When the retention rule has already dropped bin 0 -- which happens on LT1, where G06 never
    stops -- ``all_bins`` and ``moving_bins`` are the same set, and running both would write two
    identical figures under different names. Such a session is legitimately a moving-bins-only
    one: it is a locomotion-constrained track where the animals essentially always run.
    """
    retained = np.asarray(retained, dtype=bool)
    if not retained[0]:
        return [BIN_SET_MOVING]
    return list(BIN_SETS)


def assert_self_standardization_identity(binned, retained, mouse, session_type, rtol=1e-9):
    """Standardizing a mouse to its OWN occupancy must reproduce its raw rate.

    ``sum_b (seconds_b / total) * (counts_b / (n_cells * seconds_b))`` telescopes to
    ``total_counts / (n_cells * total)``, so this is an exact algebraic identity -- which makes it
    a sharp check that the weights, the per-bin rates and the occupancy are all keyed to the same
    bins. Verified over the full retained set and, separately, over the moving bins only, where
    the target is the existing ``frame_class='movement_only'`` cell-averaged rate.
    """
    for bin_set in bin_sets_for_session(retained):
        idx = bin_set_indices(bin_set, retained)
        weights = occupancy_proportions(binned['seconds'], idx)
        got = standardized_rate(binned['rate'], weights, idx)
        expected = (binned['counts'][:, idx].sum()
                    / (binned['n_cells'] * binned['seconds'][idx].sum()))
        if not np.isclose(got, expected, rtol=rtol, atol=0.0):
            raise AssertionError(
                'Self-standardization identity failed for {} {} ({}): standardized {!r} != raw '
                '{!r}. The weights, per-bin rates and occupancy are not keyed to the same '
                'bins.'.format(mouse, session_type, bin_set, got, expected))
    return True


# ---------------------------------------------------------------------------
# Binning core
# ---------------------------------------------------------------------------

def session_speed(sess, lag_frames=0):
    """Smoothed speed trace (cm/s) aligned to the frames of the trimmed ``sess.S``.

    *lag_frames* shifts the speed trace forward relative to the events, to probe the deconvolution
    latency described at :data:`SENSITIVITY_LAG_FRAMES`. The leading frames take the first
    available speed sample rather than wrapping, since wrapping would pair end-of-session speed
    with start-of-session events.

    ``build_frame_mask`` only touches the velocity trace when frame_class != all_frames, so its
    length check does not run for every caller; this function therefore performs its own.
    """
    n_frames = sess.S.shape[1]
    velocities = np.asarray(sess.velocities_miniscope_smooth, dtype=float)
    if len(velocities) < n_frames:
        raise RuntimeError(
            'Velocity trace for {} {} has {} samples but S has {} frames; cannot bin by '
            'speed.'.format(sess.mouse, sess.session_type, len(velocities), n_frames))
    velocities = velocities[:n_frames]

    if not np.all(np.isfinite(velocities)):
        n_bad = int(np.count_nonzero(~np.isfinite(velocities)))
        raise RuntimeError(
            'Velocity trace for {} {} has {} non-finite sample(s) within the S window; cannot '
            'bin by speed.'.format(sess.mouse, sess.session_type, n_bad))
    if velocities.min() < 0.0:
        # np.digitize is called against the INTERNAL edges, so it never sees the leading 0.0 and a
        # negative speed would land silently in bin 0 (the immobility bin) instead of failing.
        raise RuntimeError(
            'Velocity trace for {} {} has negative speed(s), min={:.4f} cm/s. Speed is a '
            'magnitude; a negative value means the trace is corrupt.'.format(
                sess.mouse, sess.session_type, velocities.min()))

    if lag_frames:
        if lag_frames < 0:
            raise ValueError('lag_frames must be >= 0, got {}.'.format(lag_frames))
        velocities = np.concatenate([np.full(lag_frames, velocities[0]),
                                     velocities[:-lag_frames]])
    return velocities


def speed_bin_index(sess, *, window=WINDOW_WHOLE_SESSION, lag_frames=0):
    """(bin index per frame, window mask). Frames outside the window get index -1.

    ``np.digitize`` is applied to the INTERNAL edges only, so the result is in [0, N_SPEED_BINS)
    and reproduces the ``>= VELOCITY_THRESHOLD`` convention of ``build_frame_mask`` exactly.
    """
    window_mask = build_frame_mask(sess, window, FRAME_CLASS_ALL)
    velocities = session_speed(sess, lag_frames=lag_frames)

    idx = np.digitize(velocities, np.asarray(SPEED_BIN_EDGES[1:-1], dtype=float), right=False)
    idx = idx.astype(int)
    if idx.min() < 0 or idx.max() >= N_SPEED_BINS:
        raise RuntimeError(
            'Speed bin index out of range for {} {}: [{}, {}] outside [0, {}).'.format(
                sess.mouse, sess.session_type, idx.min(), idx.max(), N_SPEED_BINS))
    idx[~window_mask] = -1
    return idx, window_mask


def speed_bin_masks(sess, *, window=WINDOW_WHOLE_SESSION, lag_frames=0):
    """List of ``N_SPEED_BINS`` boolean frame masks, disjoint and covering the window."""
    idx, _ = speed_bin_index(sess, window=window, lag_frames=lag_frames)
    return [idx == b for b in range(N_SPEED_BINS)]


def bin_occupancy_and_speed(sess, *, window=WINDOW_WHOLE_SESSION, lag_frames=0):
    """Per-bin occupancy (seconds) and realized mean speed (cm/s).

    The speed used for every downstream fit is the occupancy-weighted mean speed actually
    observed in the bin, never the nominal bin centre. The top bin is open so it has no nominal
    centre at all, and within a wide bin a nominal centre would inject a spurious group
    difference if one group habitually sits at the bottom of the bin and another at the top.

    Bins with no frames get NaN speed; they are excluded by the retention rule downstream.
    """
    idx, window_mask = speed_bin_index(sess, window=window, lag_frames=lag_frames)
    velocities = session_speed(sess, lag_frames=lag_frames)

    seconds = np.zeros(N_SPEED_BINS, dtype=float)
    speed = np.full(N_SPEED_BINS, np.nan, dtype=float)
    for b in range(N_SPEED_BINS):
        in_bin = idx == b
        n = int(np.count_nonzero(in_bin))
        seconds[b] = n / MINISCOPE_FPS
        if n:
            speed[b] = float(np.mean(velocities[in_bin]))

    total = int(np.count_nonzero(window_mask))
    if not np.isclose(seconds.sum(), total / MINISCOPE_FPS, rtol=0, atol=1e-9):
        raise AssertionError(
            'Speed bins do not cover the {} window for {} {}: {} binned frames vs {} in '
            'window.'.format(window, sess.mouse, sess.session_type,
                             int(round(seconds.sum() * MINISCOPE_FPS)), total))
    return {'seconds': seconds, 'speed': speed, 'n_frames_window': total}


# ---------------------------------------------------------------------------
# Structural assertions -- run these before trusting any number out of this module
# ---------------------------------------------------------------------------

def assert_bin_zero_is_immobility(sess, *, window=WINDOW_WHOLE_SESSION):
    """Bin 0 must be EXACTLY the existing immobility frame class, and bins 1.. exactly movement.

    This is the headline wiring check. Because SPEED_BIN_EDGES[1] is VELOCITY_THRESHOLD, exact
    boolean equality must hold, and demanding it catches essentially every way the binning can be
    wrong: an off-by-one in the edges, an inverted ``right=``, ``>`` vs ``>=``, the wrong slice of
    the velocity trace, or a mis-trimmed S matrix.

    Only meaningful at zero lag -- a shifted speed trace is deliberately no longer aligned to the
    frame classes the rest of the suite uses.
    """
    masks = speed_bin_masks(sess, window=window, lag_frames=0)
    immobility = build_frame_mask(sess, window, FRAME_CLASS_IMMOBILITY)
    movement = build_frame_mask(sess, window, FRAME_CLASS_MOVEMENT)

    if not np.array_equal(masks[0], immobility):
        n_diff = int(np.count_nonzero(masks[0] != immobility))
        raise AssertionError(
            'Speed bin 0 is not the immobility frame class for {} {}: {} frame(s) differ. '
            'SPEED_BIN_EDGES[1]={} must equal VELOCITY_THRESHOLD={}.'.format(
                sess.mouse, sess.session_type, n_diff, SPEED_BIN_EDGES[1], VELOCITY_THRESHOLD))

    moving_union = np.zeros_like(movement)
    for b in range(1, N_SPEED_BINS):
        moving_union |= masks[b]
    if not np.array_equal(moving_union, movement):
        n_diff = int(np.count_nonzero(moving_union != movement))
        raise AssertionError(
            'Union of speed bins 1.. is not the movement frame class for {} {}: {} frame(s) '
            'differ.'.format(sess.mouse, sess.session_type, n_diff))
    return True


def assert_bins_partition_window(sess, mouse, mapping, *, window=WINDOW_WHOLE_SESSION,
                                 lag_frames=0):
    """Speed bins must be a disjoint cover of the window, in frames and in events.

    Event counts are additive across a disjoint frame partition (rates are not, because the
    denominators differ), so this checks counts -- mirroring
    ``place_cell_rates.assert_frame_partition_additive``.
    """
    _, s_spikes, _, _ = sess.get_S_mapping(mapping)
    masks = speed_bin_masks(sess, window=window, lag_frames=lag_frames)

    window_mask = build_frame_mask(sess, window, FRAME_CLASS_ALL)
    n_events_window, n_frames_window = get_spike_count_in_frame_mask(s_spikes, window_mask)

    n_events_split = 0
    n_frames_split = 0
    for b, mask in enumerate(masks):
        events_b, frames_b = get_spike_count_in_frame_mask(s_spikes, mask)
        n_events_split += events_b
        n_frames_split += frames_b

    if n_events_split != n_events_window or n_frames_split != n_frames_window:
        raise AssertionError(
            'Speed bins do not partition the {} window for {} {} mapping={}: events {} != {}, '
            'frames {} != {}.'.format(window, mouse, sess.session_type, mapping,
                                      n_events_window, n_events_split,
                                      n_frames_window, n_frames_split))

    overlap = np.zeros(sess.S.shape[1], dtype=int)
    for mask in masks:
        overlap += mask.astype(int)
    if overlap.max() > 1:
        raise AssertionError(
            'Speed bins overlap for {} {}: {} frame(s) belong to more than one bin.'.format(
                mouse, sess.session_type, int(np.count_nonzero(overlap > 1))))
    return n_events_window, n_frames_window


# ---------------------------------------------------------------------------
# Bin retention, reference weights, coverage
# ---------------------------------------------------------------------------

def retained_bins_for_session(occupancy_per_mouse, context):
    """Bins usable for a whole session: those EVERY mouse occupies for >= MIN_BIN_SECONDS.

    Retention is common across mice rather than per-mouse on purpose. With a per-mouse rule the
    fastest point of a group curve might rest on three Exc mice and one Ctl mouse, which is a
    selection artifact rather than a group comparison. Requiring a common set makes the
    (mouse x bin) matrix complete by construction, so ``_values_per_group``'s existing non-finite
    hard-fail applies verbatim and the SEM at every bin is over the same animals.

    occupancy_per_mouse - {mouse: seconds array over N_SPEED_BINS}

    Returns a boolean array over bins. Raises if the retained set has a hole or is too small.
    """
    if not occupancy_per_mouse:
        raise RuntimeError('{}: no mice supplied.'.format(context))
    seconds = np.vstack([occupancy_per_mouse[m] for m in sorted(occupancy_per_mouse)])
    retained = seconds.min(axis=0) >= MIN_BIN_SECONDS

    kept = np.flatnonzero(retained)
    if len(kept) < MIN_RETAINED_BINS:
        raise RuntimeError(
            '{}: only {} speed bin(s) clear the {} s minimum in every mouse (need >= {}). '
            'Per-bin minimum occupancy across mice (s): {}. Edges: {}.'.format(
                context, len(kept), MIN_BIN_SECONDS, MIN_RETAINED_BINS,
                np.round(seconds.min(axis=0), 2).tolist(), SPEED_BIN_EDGES))

    # The retained set must be a contiguous RUN, but it need not start at bin 0.
    #
    # Note that occupancy is NOT monotonic in speed in this data -- the observed distributions are
    # multi-modal, with local dips around 6-8 and 12-16 cm/s -- so contiguity is not derived from
    # a monotonicity argument. It is a conservative structural guard: an interior gap would mean
    # some animal barely samples one speed range while sampling both the slower and the faster
    # ranges around it, which is strange enough to warrant looking at the trace rather than
    # quietly binning around it.
    #
    # Truncation at the BOTTOM is a different matter and is legitimate: G06 never drops below
    # 2 cm/s on LT1 (minimum smoothed speed 2.06 cm/s over 612 s), so it has no immobility
    # whatsoever and the common-bin rule correctly drops bin 0 for that whole session. This is the
    # same animal and the same documented behavioural property that caban.place_cell_rates
    # handles with defined=False.
    if not np.array_equal(kept, np.arange(kept[0], kept[0] + len(kept))):
        raise RuntimeError(
            '{}: retained speed bins {} have an interior gap -- some mouse barely samples a speed '
            'range while sampling both slower and faster ranges. Inspect the velocity trace '
            'rather than binning around it. Per-bin minimum occupancy across mice (s): {}.'.format(
                context, kept.tolist(), np.round(seconds.min(axis=0), 2).tolist()))
    return retained


def occupancy_proportions(seconds, idx):
    """Normalized occupancy over the bins in *idx*; sums to 1 by construction."""
    seconds = np.asarray(seconds, dtype=float)
    total = seconds[idx].sum()
    if total <= 0:
        raise RuntimeError('occupancy_proportions: zero occupancy across bins {}.'.format(
            np.asarray(idx).tolist()))
    props = np.zeros(N_SPEED_BINS, dtype=float)
    props[idx] = seconds[idx] / total
    return props


def reference_weights(occupancy_per_mouse, idx, reference, mouse_groups, context):
    """Reference speed distribution the standardized rate reweights every mouse to.

    The weight is the **unweighted mean across mice** of each mouse's own occupancy proportions,
    not the proportions of the pooled frames: pooling would weight the reference by session
    length and by how many mice each group happens to contribute.

    ``all_mice_mean``  - every animal in the session contributes equally.
    ``mCherry_mean``   - the control distribution; the conventional epidemiological choice, run as
                         a sensitivity check. If the conclusion differs between the two references
                         it is not robust, which is exactly what needs to be visible.
    """
    if reference == REFERENCE_ALL_MICE:
        mice = sorted(occupancy_per_mouse)
    elif reference == REFERENCE_CONTROL:
        mice = sorted(m for m in occupancy_per_mouse if mouse_groups[m] == 'mCherry')
    else:
        raise ValueError('Unknown reference {!r}; expected one of {}.'.format(reference, REFERENCES))

    if len(mice) < 2:
        raise RuntimeError('{}: reference {!r} resolves to {} mouse/mice ({}); need >= 2.'.format(
            context, reference, len(mice), mice))

    weights = np.mean([occupancy_proportions(occupancy_per_mouse[m], idx) for m in mice], axis=0)

    outside = np.setdiff1d(np.arange(N_SPEED_BINS), idx)
    if outside.size and not np.allclose(weights[outside], 0.0):
        raise AssertionError('{}: reference weights are non-zero outside the bin set.'.format(context))
    if not np.isclose(weights.sum(), 1.0, rtol=0, atol=1e-12):
        raise AssertionError('{}: reference weights sum to {!r}, not 1.'.format(context, weights.sum()))
    return weights


def coverage_fraction(seconds, idx, universe_idx):
    """Fraction of the bin set's natural universe that survived the retention rule.

    The denominator is the universe the bin set is *meant* to cover, not the whole session:
    ``moving_bins`` excludes immobility by design, so measuring it against total session time
    would report a low number for a deliberate choice rather than for anything lost. So
    ``all_bins`` is measured against every bin, and ``moving_bins`` against every moving bin.

    What the number then means is how much the retained-bin truncation leaves out -- the estimand
    is the rate that would obtain if every animal distributed its time as the reference animal
    does, *within the speed range every animal samples adequately*, and this is the honest
    disclosure of how wide that range is.
    """
    seconds = np.asarray(seconds, dtype=float)
    total = seconds[universe_idx].sum()
    if total <= 0:
        raise RuntimeError('coverage_fraction: zero occupancy across the universe bins {}.'.format(
            np.asarray(universe_idx).tolist()))
    return float(seconds[idx].sum() / total)


# ---------------------------------------------------------------------------
# Per-bin rates
# ---------------------------------------------------------------------------

def per_cell_counts_per_bin(sess, mouse, mapping, *, cell_class='all',
                            window=WINDOW_WHOLE_SESSION, lag_frames=0):
    """Per-cell event counts in every speed bin, plus the bin occupancy and realized speeds.

    Returns a dict with ``counts`` (n_cells x N_SPEED_BINS int), ``cells`` (row order, pinned to
    ``partition_place_cells``' ordering so per-cell records can never be misjoined to the
    place/non-place labels), ``seconds``, ``speed``, ``n_cells``, and ``rate`` (the cell-averaged
    rate per bin -- the mean across cells of each cell's own rate, silent cells included).
    """
    _, s_spikes, _, _ = sess.get_S_mapping(mapping)
    partition = partition_place_cells(sess, mouse, mapping, s_spikes=s_spikes)
    if cell_class not in partition:
        raise ValueError('Unknown cell_class {!r}; expected one of place/non_place/all.'.format(
            cell_class))
    cells = list(partition[cell_class])
    if len(cells) == 0:
        raise RuntimeError(
            'Cell class {!r} is empty for {} {} mapping={}; there is nothing to bin.'.format(
                cell_class, mouse, sess.session_type, mapping))

    masks = speed_bin_masks(sess, window=window, lag_frames=lag_frames)
    occ = bin_occupancy_and_speed(sess, window=window, lag_frames=lag_frames)

    counts = np.zeros((len(cells), N_SPEED_BINS), dtype=int)
    for b, mask in enumerate(masks):
        counts_b, cells_b = get_per_cell_spike_count_in_frame_mask(s_spikes, mask, cells=cells)
        if cells_b != cells:
            raise AssertionError(
                'Cell ordering drifted while binning {} {} bin {}.'.format(mouse, sess.session_type, b))
        counts[:, b] = counts_b

    # Cell-averaged rate per bin: mean over ALL cells including silent ones, matching
    # get_avg_sp_rate_in_frame_mask. Bins with no occupancy are NaN, never zero -- an unobserved
    # bin is not a measured rate of zero. The retention rule excludes them downstream.
    seconds = occ['seconds']
    rate = np.full(N_SPEED_BINS, np.nan, dtype=float)
    occupied = seconds > 0
    rate[occupied] = counts.sum(axis=0)[occupied] / (len(cells) * seconds[occupied])

    return {'counts': counts, 'cells': cells, 'seconds': seconds, 'speed': occ['speed'],
            'rate': rate, 'n_cells': len(cells), 'n_frames_window': occ['n_frames_window']}


def standardized_rate(rate, weights, idx):
    """Cell-averaged rate reweighted to a reference speed distribution.

    ``sum_b w_b * rate_b`` over the bin set -- direct standardization, the reweighting answer to
    "should rate be corrected for movement", as opposed to dividing rate by speed (which is not a
    hippocampal convention and diverges as speed goes to zero).
    """
    rate = np.asarray(rate, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if not np.all(np.isfinite(rate[idx])):
        raise AssertionError(
            'standardized_rate: non-finite rate inside the bin set {}; the common-bin retention '
            'rule should have made this impossible.'.format(np.asarray(idx).tolist()))
    return float(np.sum(weights[idx] * rate[idx]) / np.sum(weights[idx]))


# ---------------------------------------------------------------------------
# Poisson GLM
# ---------------------------------------------------------------------------

def fit_speed_glm(counts, exposure, speed, *, context):
    """Poisson GLM of binned event counts on speed, with a log-exposure offset.

    ``log E[count_b] = alpha + beta * speed_b + log(exposure_b)``

    **Why the log link, and not OLS on binned rates.** The scientific claim this analysis exists
    to make is "if the group difference lives in the speed-independent component rather than in
    speed modulation, the effect is cell-intrinsic". On an *additive* scale that claim is invalid:
    a multiplicative gain change -- which is what a DREADD does -- multiplies slope and intercept
    by the same factor, so a purely cell-intrinsic effect would appear as a slope difference and
    be misread as locomotion-driven. On the log scale ``beta`` is the *fractional* change in rate
    per cm/s, so a pure gain change moves ``alpha`` alone and the claim becomes falsifiable.

    The offset also weights each bin by how long the animal actually spent there; OLS on binned
    rates would treat a 3 s bin and a 400 s bin as equally informative, which is the dominant
    noise source in a per-cell fit.

    Fitting binned counts rather than per-frame samples is not an approximation: the binned-count
    GLM with offset is the sufficient-statistic form of per-frame Poisson regression on a step
    function of speed. It is also the only defensible choice here, because the speed trace is
    upsampled from 15 Hz and smoothed over ~200 ms, so per-frame samples are roughly 10x
    oversampled and per-frame standard errors would be badly anticonservative.

    exposure - seconds x cells observed in each bin (seconds alone for a single cell).

    Returns a dict with alpha (log intercept), beta (slope, log-rate per cm/s), its standard error
    and 95% CI, plus the event and bin counts behind the fit.
    """
    counts = np.asarray(counts, dtype=float)
    exposure = np.asarray(exposure, dtype=float)
    speed = np.asarray(speed, dtype=float)

    if not (len(counts) == len(exposure) == len(speed)):
        raise ValueError('{}: counts/exposure/speed lengths differ ({}, {}, {}).'.format(
            context, len(counts), len(exposure), len(speed)))
    if len(counts) < 2:
        raise RuntimeError('{}: need >= 2 bins to fit a speed relationship, got {}.'.format(
            context, len(counts)))
    if np.any(exposure <= 0):
        raise RuntimeError('{}: non-positive exposure in bin(s) {}.'.format(
            context, np.flatnonzero(exposure <= 0).tolist()))
    if not np.all(np.isfinite(speed)):
        raise RuntimeError('{}: non-finite realized speed in bin(s) {}.'.format(
            context, np.flatnonzero(~np.isfinite(speed)).tolist()))
    if np.ptp(speed) <= 0:
        raise RuntimeError('{}: realized speed does not vary across bins ({}); the slope is not '
                           'identified.'.format(context, speed.tolist()))
    if np.any(counts < 0):
        raise RuntimeError('{}: negative counts.'.format(context))

    design = sm.add_constant(speed[:, None], has_constant='add')
    result = sm.GLM(counts, design, family=sm.families.Poisson(),
                    offset=np.log(exposure)).fit()

    alpha = float(result.params[0])
    beta = float(result.params[1])
    se_beta = float(result.bse[1])

    if not (np.isfinite(alpha) and np.isfinite(beta) and np.isfinite(se_beta)):
        raise RuntimeError('{}: Poisson fit returned non-finite parameters '
                           '(alpha={!r}, beta={!r}, se={!r}).'.format(context, alpha, beta, se_beta))
    if abs(beta) > BETA_ABS_MAX:
        raise RuntimeError(
            '{}: fitted slope beta={:.4g} exceeds the sanity bound {} (log-rate per cm/s). '
            'counts={}, exposure={}, speed={}.'.format(
                context, beta, BETA_ABS_MAX, counts.tolist(),
                np.round(exposure, 2).tolist(), np.round(speed, 2).tolist()))

    return {'alpha': alpha, 'beta': beta, 'se_beta': se_beta,
            'ci_beta_low': beta - 1.96 * se_beta, 'ci_beta_high': beta + 1.96 * se_beta,
            'n_events': int(counts.sum()), 'n_bins': int(len(counts))}


def fit_per_cell_speed_glm(binned, idx, *, context):
    """Per-cell Poisson speed fits, with an explicit eligibility screen and exclusion tally.

    Cells are screened up front (>= MIN_CELL_EVENTS events, spread over >=
    MIN_CELL_OCCUPIED_BINS bins) rather than by catching a fit failure, so :func:`fit_speed_glm`
    stays free to hard-fail on anything unexpected.

    The exclusion tally is not bookkeeping: if one viral group silences cells, proportionally more
    of its cells fall under the event threshold, and the surviving per-cell distribution is
    survivorship-biased. The per-group inclusion fraction has to be reported alongside the
    distribution for it to be interpretable at all.

    Returns (dict of per-cell arrays over the INCLUDED cells, tally dict).
    """
    counts = binned['counts'][:, idx]
    exposure = binned['seconds'][idx]
    speed = binned['speed'][idx]

    n_events = counts.sum(axis=1)
    n_occupied = (counts > 0).sum(axis=1)
    eligible = (n_events >= MIN_CELL_EVENTS) & (n_occupied >= MIN_CELL_OCCUPIED_BINS)

    alphas, betas, ses, cells_out, events_out = [], [], [], [], []
    for row in np.flatnonzero(eligible):
        cell = binned['cells'][row]
        fit = fit_speed_glm(counts[row], exposure, speed,
                            context='{} cell {}'.format(context, cell))
        alphas.append(fit['alpha'])
        betas.append(fit['beta'])
        ses.append(fit['se_beta'])
        cells_out.append(cell)
        events_out.append(fit['n_events'])

    tally = {
        'n_cells_total': int(len(binned['cells'])),
        'n_cells_fit': int(eligible.sum()),
        'frac_fit': float(eligible.sum() / len(binned['cells'])),
        'n_excluded_few_events': int(np.count_nonzero(n_events < MIN_CELL_EVENTS)),
        'n_excluded_one_bin': int(np.count_nonzero((n_events >= MIN_CELL_EVENTS)
                                                   & (n_occupied < MIN_CELL_OCCUPIED_BINS))),
    }
    return ({'alpha': np.asarray(alphas), 'beta': np.asarray(betas), 'se_beta': np.asarray(ses),
             'cells': cells_out, 'n_events': np.asarray(events_out)}, tally)


def assert_glm_recovers_known_tuning(seed=0, tolerance=0.12):
    """Recover a known (alpha, beta) from synthetic Poisson counts.

    This is the only check that can catch a sign or scale error in the exposure offset, which
    otherwise produces entirely plausible-looking slopes. Run as part of the section's verify pass.
    """
    rng = np.random.default_rng(seed)
    speed = np.array([1.0, 3.0, 5.0, 7.0, 10.0, 14.0, 20.0])
    exposure = np.array([400.0, 260.0, 180.0, 120.0, 90.0, 45.0, 20.0]) * 150.0
    alpha_true, beta_true = np.log(0.05), 0.06

    counts = rng.poisson(np.exp(alpha_true + beta_true * speed) * exposure)
    fit = fit_speed_glm(counts, exposure, speed, context='assert_glm_recovers_known_tuning')

    if abs(fit['beta'] - beta_true) > tolerance * abs(beta_true):
        raise AssertionError(
            'Poisson speed GLM did not recover the known slope: got {:.6g}, expected {:.6g} '
            '(tolerance {:.0%}). A sign or scale error in the log-exposure offset would look '
            'exactly like this.'.format(fit['beta'], beta_true, tolerance))
    if abs(fit['alpha'] - alpha_true) > tolerance * abs(alpha_true):
        raise AssertionError(
            'Poisson speed GLM did not recover the known intercept: got {:.6g}, expected '
            '{:.6g}.'.format(fit['alpha'], alpha_true))
    return fit


def group_difference_ci(per_mouse_beta, mouse_groups, group_a, group_b):
    """Welch 95% CI on the difference in mean per-mouse slope between two groups.

    With n = 5/6/6 a non-significant group x speed interaction on its own says very little, so the
    interval -- and what a difference at its limit would imply -- is what makes the null
    informative rather than merely underpowered.
    """
    a = np.array([per_mouse_beta[m] for m in per_mouse_beta if mouse_groups[m] == group_a], float)
    b = np.array([per_mouse_beta[m] for m in per_mouse_beta if mouse_groups[m] == group_b], float)
    if len(a) < 2 or len(b) < 2:
        raise RuntimeError('group_difference_ci: need >= 2 mice per group, got {} and {}.'.format(
            len(a), len(b)))

    diff = float(a.mean() - b.mean())
    se = float(np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))
    # Welch-Satterthwaite degrees of freedom.
    num = (a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)) ** 2
    den = ((a.var(ddof=1) / len(a)) ** 2 / (len(a) - 1)
           + (b.var(ddof=1) / len(b)) ** 2 / (len(b) - 1))
    dof = float(num / den) if den > 0 else float(len(a) + len(b) - 2)
    crit = float(stats.t.ppf(0.975, dof))

    return {'diff': diff, 'se': se, 'dof': dof,
            'ci_low': diff - crit * se, 'ci_high': diff + crit * se,
            'n_a': len(a), 'n_b': len(b)}


def slope_difference_rate_impact(ci, speed_span):
    """Fractional rate difference implied by a slope difference at each end of its CI.

    On the log scale a slope difference of ``d`` sustained across a speed range of ``speed_span``
    multiplies the rate by ``exp(d * speed_span)``. Expressing the CI limits this way converts an
    uninformative ``p > 0.05`` into a statement about how large a speed-driven contribution the
    data can still accommodate.
    """
    return {
        'speed_span_cms': float(speed_span),
        'pct_at_ci_low': float((np.exp(ci['ci_low'] * speed_span) - 1.0) * 100.0),
        'pct_at_ci_high': float((np.exp(ci['ci_high'] * speed_span) - 1.0) * 100.0),
        'pct_at_estimate': float((np.exp(ci['diff'] * speed_span) - 1.0) * 100.0),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _speed_axis_labels(idx):
    """Short bin labels like '2-4' / '16+' for the x axis."""
    labels = []
    for b in idx:
        low, high = SPEED_BIN_EDGES[b], SPEED_BIN_EDGES[b + 1]
        labels.append('{:g}+'.format(low) if not np.isfinite(high) else '{:g}-{:g}'.format(low, high))
    return labels


def _draw_group_curves(ax, per_mouse_curve, mouse_groups, idx, *, ylabel, xlabel='Speed (cm/s)'):
    """Per-group mean +/- SEM curve over speed bins, one line per group.

    The error band is the SEM **across mice** at each bin (n = 5/6/6) -- never across cells and
    never across frames, either of which would understate it by an order of magnitude. The
    common-bin retention rule guarantees the same animals contribute at every bin, so the band is
    comparable along the whole curve.
    """
    x = np.arange(len(idx))
    for group in GROUP_ORDER:
        mice = sorted(m for m in per_mouse_curve if mouse_groups[m] == group)
        if len(mice) < 2:
            raise RuntimeError(
                'Group {} has {} mouse/mice ({}); need >= 2 for a mean +/- SEM curve.'.format(
                    group, len(mice), mice))
        curves = np.vstack([np.asarray(per_mouse_curve[m], dtype=float)[idx] for m in mice])
        if not np.all(np.isfinite(curves)):
            bad = [mice[r] for r, _ in zip(*np.where(~np.isfinite(curves)))]
            raise RuntimeError(
                'Non-finite curve value(s) for {}; the common-bin retention rule should have '
                'made this impossible.'.format(sorted(set(bad))))
        mean = curves.mean(axis=0)
        sem = curves.std(axis=0, ddof=1) / np.sqrt(len(mice))
        ax.plot(x, mean, color=group_colours[group], lw=1.4, marker='o', ms=3,
                label='{} (n={})'.format(GROUP_LABELS[group], len(mice)))
        ax.fill_between(x, mean - sem, mean + sem, color=group_colours[group], alpha=0.15,
                        linewidth=0)

    ax.set_xticks(x)
    ax.set_xticklabels(_speed_axis_labels(idx), size='x-small', rotation=45)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines[['right', 'top']].set_visible(False)
    ax.legend(frameon=False, fontsize='xx-small')


def _draw_violin_triplet_panel(ax, values, names, col_idx, ylabel, show_mouse_names=False):
    """One violin-triplet panel on an existing axis, styled like the rest of the suite.

    ``_panel_row`` builds its own figure, so this is the single-axis entry point used when a
    violin panel has to sit beside a curve in a shared figure. The statistics are the same:
    ANOVA-gated Tukey via ``_draw_violin_triplet``'s default ``stat_fn``, n = mice.
    """
    ax.spines[['right', 'top']].set_visible(False)
    lo, hi = bracket_ylim(values, col_idx + 1)
    _draw_violin_triplet(
        ax, values, col_idx, GROUP_ORDER, group_colours,
        mouse_names_per_group=names if show_mouse_names else None,
        ylim=(lo, hi), ylabel=ylabel,
    )
    ax.set_xticks(range(len(GROUP_ORDER)))
    ax.set_xticklabels([GROUP_LABELS[g] for g in GROUP_ORDER], size='medium')


def plot_speed_occupancy(PLOTS_DIR, mouse_groups, per_mouse, session_type, idx,
                         figsize=(7.2, 3.2), auto_close=True):
    """Deliverable 1: do the speed DISTRIBUTIONS differ, not merely the means?

    This is the premise the rest of the suite rests on and nothing else in the codebase tests it:
    ``caban.locomotion`` compares mean speed and % time moving, both of which can match while the
    distributions differ. It also licenses the standardized rate -- if the distributions
    superimpose, the standardized rate must equal the raw rate, which turns a weak null into a
    strong one.

    per_mouse - {mouse: {'proportions': array over bins, 'tv_distance': float}}
    """
    context = 'plot_speed_occupancy({})'.format(session_type)
    fig, axes = plt.subplots(1, 2, figsize=figsize,
                             gridspec_kw={'width_ratios': [2.0, 1.0]})

    _draw_group_curves(axes[0], {m: v['proportions'] for m, v in per_mouse.items()},
                       mouse_groups, idx, ylabel='Fraction of time')
    axes[0].set_title('Speed occupancy', size='small')

    values, names = _values_per_group(per_mouse, mouse_groups, ['tv_distance'], context)
    _draw_violin_triplet_panel(axes[1], values, names, 0,
                               'Total-variation distance\nfrom reference', show_mouse_names=True)
    axes[1].set_title('Distributional distance', size='small')

    set_session_title(fig, session_type)
    fig.tight_layout(pad=0.5, rect=(0.0, 0.0, 1.0, 0.92))

    save_dir = os.path.join(PLOTS_DIR, NAV_AWARE_DIR, SPEED_TUNING_DIR, 'speed_occupancy')
    _save(fig, save_dir, 'speed_occupancy-{}'.format(session_type), auto_close)
    return save_dir


def plot_speed_tuning_curves(PLOTS_DIR, mouse_groups, per_mouse, session_type, mapping, idx,
                             figsize=(7.2, 3.2), auto_close=True):
    """Deliverable 2: cell-averaged rate vs speed bin, absolute and shape-normalized.

    The second panel divides each mouse's curve by its own mean rate over the bin set, which
    separates *height* from *shape* -- the intercept/slope dissociation as a picture. If the
    normalized curves superimpose while the absolute ones do not, the groups differ in overall
    excitability and not in how they track speed, which is the cell-intrinsic reading.

    per_mouse - {mouse: rate array over bins}
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    _draw_group_curves(axes[0], per_mouse, mouse_groups, idx,
                       ylabel=_METRIC_YLABEL[False])
    axes[0].set_title('Absolute', size='small')

    normalized = {}
    for mouse, rate in per_mouse.items():
        rate = np.asarray(rate, dtype=float)
        scale = np.mean(rate[idx])
        if not np.isfinite(scale) or scale <= 0:
            raise RuntimeError(
                'Mouse {} has mean rate {!r} over the bin set in {}; cannot shape-normalize a '
                'non-positive rate.'.format(mouse, scale, session_type))
        normalized[mouse] = rate / scale
    _draw_group_curves(axes[1], normalized, mouse_groups, idx,
                       ylabel='Rate / mouse mean rate')
    axes[1].set_title('Shape (each mouse normalized)', size='small')

    set_session_title(fig, session_type)
    fig.tight_layout(pad=0.5, rect=(0.0, 0.0, 1.0, 0.92))

    save_dir = os.path.join(PLOTS_DIR, NAV_AWARE_DIR, SPEED_TUNING_DIR, 'tuning_curves',
                            session_type)
    _save(fig, save_dir, 'speed_tuning_curve-{}-{}'.format(session_type, mapping), auto_close)
    return save_dir


def plot_speed_tuning_by_cell_class(PLOTS_DIR, mouse_groups, per_mouse_by_class, session_type,
                                    mapping, idx, figsize=(7.2, 3.2), auto_close=True):
    """Positive control: place cells should be the most strongly speed-modulated cell class.

    If place cells do not show stronger speed modulation than non-place cells, the binning or the
    cell partition is wrong -- this panel is here to make that failure visible rather than to
    support a group claim.

    per_mouse_by_class - {'place'|'non_place': {mouse: rate array}}
    """
    classes = ['place', 'non_place']
    titles = {'place': 'Place cells', 'non_place': 'Non-place cells'}
    fig, axes = plt.subplots(1, len(classes), figsize=figsize, sharey=True)

    for ax, cell_class in zip(np.atleast_1d(axes), classes):
        _draw_group_curves(ax, per_mouse_by_class[cell_class], mouse_groups, idx,
                           ylabel=_METRIC_YLABEL[False])
        ax.set_title(titles[cell_class], size='small')

    set_session_title(fig, session_type)
    fig.tight_layout(pad=0.5, rect=(0.0, 0.0, 1.0, 0.92))

    save_dir = os.path.join(PLOTS_DIR, NAV_AWARE_DIR, SPEED_TUNING_DIR, 'tuning_curves',
                            session_type)
    _save(fig, save_dir, 'speed_tuning_curve_by_cell_class-{}-{}'.format(session_type, mapping),
          auto_close)
    return save_dir


_GLM_COLUMNS = ['alpha', 'beta']
_GLM_TITLES = ['Speed-independent rate\n(log events/s at 0 cm/s)',
               'Speed modulation\n(log rate per cm/s)']


def plot_speed_glm_per_mouse(PLOTS_DIR, mouse_groups, per_mouse, session_type, mapping, bin_set,
                             figsize=(5.4, 3.2), auto_close=True):
    """Deliverable 3, mouse level: the group x speed interaction, with n = mice.

    Comparing the per-mouse slope across groups **is** the interaction test, done as a two-stage
    summary-statistic analysis so there is no pseudoreplication -- consistent with the rest of the
    suite. ``alpha`` and ``beta`` are separated by the log link, so a pure multiplicative gain
    change moves alpha alone (see :func:`fit_speed_glm`).

    per_mouse - {mouse: {'alpha': float, 'beta': float}}
    """
    context = 'plot_speed_glm_per_mouse({}, {}, {})'.format(session_type, mapping, bin_set)
    values, names = _values_per_group(per_mouse, mouse_groups, _GLM_COLUMNS, context)

    fig, axes = plt.subplots(1, len(_GLM_COLUMNS), figsize=figsize)
    for col_idx, (ax, title) in enumerate(zip(np.atleast_1d(axes), _GLM_TITLES)):
        _draw_violin_triplet_panel(ax, values, names, col_idx, title)
    set_session_title(fig, session_type)
    fig.tight_layout(pad=0.5, rect=(0.0, 0.0, 1.0, 0.88))

    save_dir = os.path.join(PLOTS_DIR, NAV_AWARE_DIR, SPEED_TUNING_DIR, 'speed_glm',
                            session_type, bin_set)
    _save(fig, save_dir, 'speed_glm_per_mouse-{}-{}'.format(session_type, mapping), auto_close)
    return save_dir


def plot_speed_slope_ecdf(PLOTS_DIR, per_cell_by_group, per_cell_by_group_mouse, session_type,
                          mapping, bin_set, figsize=(4.2, 3.4), auto_close=True):
    """Deliverable 3, cell level: distribution of per-cell speed-modulation slopes.

    The per-cell *intercept* distribution is deliberately not plotted -- it would very nearly
    duplicate the existing ``plot_cell_activity_distributions``. The slope is the new information.
    """
    fig, ax = plt.subplots(figsize=figsize)
    # GROUP_ORDER threaded in explicitly: single_unit_common's own order differs, and its default
    # must not change because four published analyses depend on it. title='' here (ecdf_panel
    # only sets a panel title when truthy) -- the session name goes on the figure instead, per the
    # paper-facing title convention.
    ecdf_panel(ax, per_cell_by_group, per_mouse_by_group=per_cell_by_group_mouse,
               xlabel='Speed modulation (log rate per cm/s)', title='',
               group_order=GROUP_ORDER)
    set_session_title(fig, session_type)
    fig.tight_layout(pad=0.5, rect=(0.0, 0.0, 1.0, 0.90))

    save_dir = os.path.join(PLOTS_DIR, NAV_AWARE_DIR, SPEED_TUNING_DIR, 'speed_glm',
                            session_type, bin_set)
    _save(fig, save_dir, 'speed_slope_ecdf-{}-{}'.format(session_type, mapping), auto_close)
    return save_dir


def plot_standardized_rate(PLOTS_DIR, mouse_groups, per_mouse, session_type, mapping, bin_set,
                           reference, figsize=(5.4, 3.2), auto_close=True):
    """Deliverable 4: the speed-standardized rate, and -- as its own panel -- the SHIFT.

    The shift panel is the point. Given that the groups do not differ in locomotion, the
    standardized rate should be almost identical to the raw one, and "the correction moves every
    animal by under a couple of percent" is a far stronger statement than a violin of the
    standardized value, which hides the magnitude entirely.

    per_mouse - {mouse: {'rate_standardized', 'rate_raw', 'shift_pct'}}
    """
    context = 'plot_standardized_rate({}, {}, {}, {})'.format(
        session_type, mapping, bin_set, reference)
    columns = ['rate_standardized', 'shift_pct']
    titles = ['Speed-standardized rate\n(events/s)', 'Shift from raw rate (%)']
    values, names = _values_per_group(per_mouse, mouse_groups, columns, context)

    fig, axes = plt.subplots(1, len(columns), figsize=figsize)
    for col_idx, (ax, title) in enumerate(zip(np.atleast_1d(axes), titles)):
        _draw_violin_triplet_panel(ax, values, names, col_idx, title)
    set_session_title(fig, session_type)
    fig.tight_layout(pad=0.5, rect=(0.0, 0.0, 1.0, 0.88))

    save_dir = os.path.join(PLOTS_DIR, NAV_AWARE_DIR, SPEED_TUNING_DIR, 'standardized_rate',
                            session_type, bin_set)
    _save(fig, save_dir, 'standardized_rate-{}-{}-{}'.format(session_type, mapping, reference),
          auto_close)
    return save_dir


# ---------------------------------------------------------------------------
# Statistics reporting
# ---------------------------------------------------------------------------

def write_slope_mixed_model(PLOTS_DIR, per_cell_by_group_mouse, tallies, session_type, mapping,
                            bin_set):
    """Per-cell slope LMM (cell nested in mouse) plus the exclusion-balance report.

    The exclusion fractions are printed in the same file as the model because the model is not
    interpretable without them: if one group's cells are systematically silenced below the event
    threshold, its surviving slope distribution is survivorship-biased.
    """
    df = build_cell_records(per_cell_by_group_mouse, value_name='beta')
    text, method = fit_group_mixed_model(df, value_col='beta', reference='mCherry')

    lines = ['Per-cell speed-modulation slope — {} — {} — {}'.format(
                 session_type, mapping, bin_set),
             'Slope is d(log rate)/d(speed), in log events/s per cm/s.', '',
             text, '', '--- cell inclusion balance ---',
             'Cells need >= {} events spread over >= {} retained speed bins to be fit. If one '
             'group is preferentially excluded, the'.format(MIN_CELL_EVENTS, MIN_CELL_OCCUPIED_BINS),
             'per-cell distribution above is survivorship-biased; compare the fractions below '
             'before reading it.', '']
    for group in GROUP_ORDER:
        for mouse in sorted(tallies.get(group, {})):
            t = tallies[group][mouse]
            lines.append('  {:<8s} {:<6s} fit {:5d} / {:5d}  ({:6.1%})   excluded: {} few events, '
                         '{} single-bin'.format(GROUP_LABELS[group], mouse, t['n_cells_fit'],
                                                t['n_cells_total'], t['frac_fit'],
                                                t['n_excluded_few_events'], t['n_excluded_one_bin']))
        totals = [tallies[group][m] for m in tallies.get(group, {})]
        if totals:
            fit = sum(t['n_cells_fit'] for t in totals)
            tot = sum(t['n_cells_total'] for t in totals)
            lines.append('  {:<8s} TOTAL  fit {:5d} / {:5d}  ({:6.1%})'.format(
                GROUP_LABELS[group], fit, tot, fit / tot if tot else float('nan')))
        lines.append('')

    save_dir = os.path.join(PLOTS_DIR, NAV_AWARE_DIR, SPEED_TUNING_DIR, 'speed_glm',
                            session_type, bin_set)
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, 'speed_slope_mixed_model-{}-{}.txt'.format(session_type, mapping))
    with open(path, 'w') as fh:
        fh.write('\n'.join(lines))
    return path, method


def write_group_stats(PLOTS_DIR, records):
    """Session-level interaction tests, CIs, and FDR correction across sessions.

    *records* - list of dicts, one per (session, bin_set), each carrying 'per_mouse_beta',
    'per_mouse_rate', 'mouse_groups', 'speed_span'.

    Three things this reports that a bare p-value does not:

    * the **95% CI** on each pairwise group difference in slope, and the fractional rate
      difference a slope difference at either limit would produce across the observed speed range.
      With n = 5/6/6 a null interaction is otherwise uninformative -- the interval is what makes
      it an actual bound on how much speed could be contributing;
    * the observed group difference in raw rate, so the two can be compared directly;
    * **FDR q-values across sessions**. The rest of the suite does not correct for multiplicity,
      but this is a *null* argument, where uncorrected multiplicity errs in the direction that
      destroys it -- one spurious interaction in fourteen tests and the defence collapses.
    """
    lines = ['Speed modulation: group x speed interaction',
             '',
             'The interaction is tested as a two-stage summary-statistic analysis: a Poisson',
             'speed slope is fit per mouse, then compared across groups with n = mice. A NULL',
             'interaction is the desired result -- it means the groups modulate their firing with',
             'speed identically, so the whole-session rate difference is not a speed effect.',
             '']

    pvalues, keys = [], []
    for rec in records:
        session, bin_set = rec['session'], rec['bin_set']
        groups = rec['mouse_groups']
        betas = rec['per_mouse_beta']
        arrays = [np.array([betas[m] for m in betas if groups[m] == g], float) for g in GROUP_ORDER]
        f_stat, p_value = stats.f_oneway(*arrays)
        pvalues.append(p_value)
        keys.append((session, bin_set))

        lines.append('=== {} — {} ==='.format(session, bin_set))
        lines.append('  one-way ANOVA on per-mouse slope: F = {:.4f}, p = {:.4g}'.format(
            f_stat, p_value))
        lines.append('  per-group mean slope (log rate per cm/s): {}'.format(
            ', '.join('{}={:+.5f}'.format(GROUP_LABELS[g], a.mean())
                      for g, a in zip(GROUP_ORDER, arrays))))

        rates = rec['per_mouse_rate']
        rate_means = {g: float(np.mean([rates[m] for m in rates if groups[m] == g]))
                      for g in GROUP_ORDER}
        lines.append('  per-group mean raw rate (events/s): {}'.format(
            ', '.join('{}={:.5f}'.format(GROUP_LABELS[g], rate_means[g]) for g in GROUP_ORDER)))
        lines.append('  observed speed span across retained bins: {:.2f} cm/s'.format(
            rec['speed_span']))
        lines.append('')

        for group_a, group_b in [('hM3D', 'mCherry'), ('hM4D', 'mCherry'), ('hM3D', 'hM4D')]:
            ci = group_difference_ci(betas, groups, group_a, group_b)
            impact = slope_difference_rate_impact(ci, rec['speed_span'])
            rate_pct = (100.0 * (rate_means[group_a] - rate_means[group_b]) / rate_means[group_b]
                        if rate_means[group_b] else float('nan'))
            lines.append('  {} vs {}:'.format(GROUP_LABELS[group_a], GROUP_LABELS[group_b]))
            lines.append('    slope difference = {:+.5f}  95% CI [{:+.5f}, {:+.5f}]  '
                         '(n = {} vs {})'.format(ci['diff'], ci['ci_low'], ci['ci_high'],
                                                 ci['n_a'], ci['n_b']))
            lines.append('    across the {:.1f} cm/s span this implies a rate difference of '
                         '{:+.2f}% (CI {:+.2f}% to {:+.2f}%)'.format(
                             impact['speed_span_cms'], impact['pct_at_estimate'],
                             impact['pct_at_ci_low'], impact['pct_at_ci_high']))
            lines.append('    observed raw rate difference: {:+.2f}%'.format(rate_pct))
            lines.append('    -> at the CI limit, graded speed modulation could account for at '
                         'most {:.1f}% of the {:.1f}% observed difference.'.format(
                             max(abs(impact['pct_at_ci_low']), abs(impact['pct_at_ci_high'])),
                             abs(rate_pct)))
        lines.append('')

    reject, qvalues = fdr_correct(np.asarray(pvalues))
    lines.append('=== Benjamini-Hochberg FDR across all {} interaction tests ==='.format(len(pvalues)))
    for (session, bin_set), p, q, rej in zip(keys, pvalues, qvalues, reject):
        lines.append('  {:<12s} {:<12s} p = {:.4g}   q = {:.4g}   {}'.format(
            session, bin_set, p, q, 'SIGNIFICANT' if rej else 'ns'))

    save_dir = os.path.join(PLOTS_DIR, NAV_AWARE_DIR, SPEED_TUNING_DIR)
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, 'speed_tuning_group_stats.txt')
    with open(path, 'w') as fh:
        fh.write('\n'.join(lines))
    return path


def write_verification(PLOTS_DIR, lines):
    """Persist the assertion log so a wiring failure is visible in the output tree."""
    save_dir = os.path.join(PLOTS_DIR, NAV_AWARE_DIR, SPEED_TUNING_DIR)
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, 'verification.txt')
    with open(path, 'w') as fh:
        fh.write('\n'.join(lines))
    return path


def write_table(PLOTS_DIR, rows, filename):
    """Write one tidy CSV under the speed_tuning root."""
    save_dir = os.path.join(PLOTS_DIR, NAV_AWARE_DIR, SPEED_TUNING_DIR)
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def copy_methods_templates(PLOTS_DIR):
    """Drop the METHODS template into the directory this module writes."""
    dest = os.path.join(PLOTS_DIR, NAV_AWARE_DIR, SPEED_TUNING_DIR)
    os.makedirs(dest, exist_ok=True)
    _copy_analysis_methods_template('speed_tuning_methods.txt', dest)
