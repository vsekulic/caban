"""Per-cell epoch modulation during trace fear conditioning (Figure 2, single-cell block).

This is the single-cell companion to the population-level conditioning analysis in
``caban.sp_rates_lmm``. That analysis asks how much calcium activity there is per group
per epoch; this one asks, for each individual cell, how far its activity in the tone,
trace, shock and post-shock windows departs from ITS OWN pre-tone baseline ON THE SAME
TRIAL, and whether the DREADD manipulations change the average cellular modulation
profile across those epochs.

WHAT THIS MODULE DELIBERATELY IS NOT
------------------------------------
It is neither ``sp_rates_lmm``'s S-event world nor ``event_locked_responsiveness``'s
shuffle/classification world, and it lives in its own module so it inherits neither.

- No binary responsive/non-responsive gate. The modulation index is continuous and the
  group comparison is made on it directly. In particular this module does NOT reproduce
  the ``event_locked_responsiveness`` pattern of "shuffle -> FDR-select responders ->
  compare the selecting statistic's magnitude across groups", which selects cells on a
  noisy statistic and then reports that same statistic as an unbiased treatment effect.
- No deconvolved S. At the population event rates this cohort shows (0.05-0.1 events per
  second per cell, see sp_rates_lmm), a 20 s window holds ~1-2 detected events and a 2 s
  shock window essentially none, so a per-cell-per-trial-per-epoch S quantity would be
  effectively 0/1/2 -- far too coarse for a continuous index. An S-based single-cell
  analysis would also largely restate the existing population figure.

THE MEASUREMENT, IN ONE PLACE
-----------------------------
Step 1, ONCE per cell, over the WHOLE session:

    z_c(t) = (x_c(t) - mean_over_session(x_c)) / sd_over_session(x_c)

Step 2, window means OF THAT SINGLE STANDARDIZED TRACE:

    index(cell, trial, epoch) = mean_{t in epoch window}    z_c(t)
                              - mean_{t in pre-tone window} z_c(t)

** THERE IS EXACTLY ONE STANDARDIZATION AND IT PRECEDES ALL WINDOWING. ** The epoch mean
and the pre-tone mean are two averages of the SAME standardized trace and therefore share
one denominator (that cell's session SD). Windows are NEVER standardized separately:
per-window z-scoring would force each window to unit variance by its own SD, so the two
means would no longer be on a common scale and their difference would carry no amplitude
information at all. Any change to this module that computes an SD inside a window is
wrong -- see ``_standardized_trace``, which is the only place a SD is ever taken.

The standardization is PER CELL, never pooled across a mouse's cells. Pooling would leave
per-cell scale differences (footprint amplitude, expression level, depth in the FOV)
inside the index and let the brightest cells dominate the mouse-level mean. The known
consequence, stated rather than corrected: a quiet cell has a small SD in the denominator,
so per-cell standardization amplifies quiet cells.

WHAT THE INDEX MEANS
--------------------
Modulation in units of that cell's OWN session-wide variability. It is NOT an absolute
response magnitude and is not comparable to the dF/F- or event-amplitude-scaled quantities
in the population figure. The same numerical index also means different physical things on
the two signals below, because YrA's per-cell SD is dominated by measurement noise while
C's is dominated by fitted transients -- so the signal comparison is about AGREEMENT OF
CONCLUSIONS, not equality of magnitudes.

SIGNALS: YrA PRIMARY, C CONFIRMATORY
------------------------------------
YrA is primary because C's sparsity is a specific liability for short, event-locked
windows: C is the deconvolution-constrained fit and is exactly zero between fitted
transients, so on a 2 s shock window a cell with no fitted transient contributes an
identically flat value, whereas YrA still carries graded fluorescence. C is re-run
identically as a confirmatory replicate; agreement shows the result is not carried by
residual neuropil/noise that CNMF-E assigns to background, nor by the deconvolution model.
Agreement is pre-specified as matching contrast signs with overlapping CIs, and
DISAGREEMENT IS REPORTED, never resolved by keeping whichever signal looks better.

UNIT OF INFERENCE
-----------------
The mouse. Cells are displayed, never counted as treatment replicates. Averaging cells
within a mouse costs essentially no treatment-level power -- Var(mouse mean) =
sigma^2_between + sigma^2_within/n_cells, and the second term is negligible at hundreds of
cells per mouse -- but it does discard distribution SHAPE, which the panels display and
the additive hierarchical cell-level companion models. ``summarize_mice`` reports each
mouse's SEM across cells alongside the between-mouse spread so that claim is demonstrated
on this dataset rather than asserted.
"""
import datetime
import os
import shutil
import time
import traceback

import numpy as np
import pandas as pd
import patsy
import matplotlib.pyplot as plt
from scipy.optimize import approx_fprime, minimize

from caban.utilities import MINISCOPE_FPS
from caban.epoch_analysis import get_epoch_frames, TRACE_MATCHED_WINDOW_S
from caban.decoder import _copy_analysis_methods_template
from caban.single_unit_common import (
    GROUP_ORDER, GROUP_COLOURS, GROUP_LABELS, DREADD_DISPLAY_ORDER,
    ensure_dirs, write_text, save_fig, holm_correct,
    fit_mixed_model, linear_contrast_test, joint_wald_test,
    draw_superplot_triplet, grow_ylim_for_bracket_headroom,
)
# The randomization machinery of the recall lane, reused verbatim rather than reimplemented
# (CLAUDE.md, Code Deduplication). `caban.sp_rates_lmm` does not import this module, so there is
# no cycle to resolve.
from caban.sp_rates_lmm import (mouse_label_permutation_test, n_distinct_relabelings,
                                make_contrast_stat)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# The within-trial baseline every index is constructed against. It is the CONSTRUCTION
# BASELINE, not an epoch level of the model -- including it as a level would add an
# identically-zero modeled epoch, since index(pre_tone) == 0 by definition.
BASELINE_EPOCH = 'pre_tone_matched'

# Model epoch levels, reference first. 'tone' is the reference level rather than a
# pre-tone level for the reason above.
RESPONSE_EPOCHS = ('tone', 'trace', 'shock', 'post_shock')
REFERENCE_EPOCH = 'tone'
REFERENCE_GROUP = 'mCherry'

def nominal_trace_seconds(session, trial_idx):
    """The trial's trace-interval length in seconds, from the session's DEFINED protocol times.

        shock_onsets_def[t] - (tone_onsets_def[t] + tone_duration)

    ** Exact, and deliberately not measured from frame numbers. ** The protocol times are
    declared outright in TraceFearCondSession.__init__ (``tone_onsets_def``, ``tone_duration``,
    ``shock_onsets_def``, ``shock_duration``); ``find_exp_boundaries`` then converts each one to
    a frame by snapping it to the nearest miniscope timestamp. That snapping is what makes a
    declared 20 s interval span 396 frames (19.8 s) rather than 400 -- it is quantization in the
    lookup, not a property of the trial.

    Asking the FRAMES how long the trace is therefore poses a question the definitions already
    answer exactly, and forces a tolerance to absorb the resulting jitter. Asking the
    DEFINITIONS gives 15 s on the acquisition trial and 20 s on the rest, as integers, so the
    retention rule needs no tolerance at all. This still reads the session's own timing rather
    than hard-coding a trial index, so it stays correct if the protocol changes.
    """
    return float(session.shock_onsets_def[trial_idx]
                 - (session.tone_onsets_def[trial_idx] + session.tone_duration))

# Signals, with the division of labour described in the module docstring. This is not an
# open-ended bake-off: the primary signal produces the panels, the confirmatory one
# produces a side-by-side contrast table.
PRIMARY_SIGNAL = 'YrA'
CONFIRMATORY_SIGNAL = 'C'
SIGNALS = (PRIMARY_SIGNAL, CONFIRMATORY_SIGNAL)

# Panel K alignment window, relative to tone onset. -20 s covers the pre-tone baseline and
# +65 s reaches past the post-shock window (20 s tone + 20 s trace + 2 s shock + 20 s
# post-shock = 62 s after tone onset).
HEATMAP_PRE_S = 20.0
HEATMAP_POST_S = 65.0

METHODS_FILENAME = 'epoch_modulation_methods.md'


# ─────────────────────────────────────────────────────────────────────────────
# Signal preparation
# ─────────────────────────────────────────────────────────────────────────────

_SIGNAL_INDEX_ATTR = {'C': 'C_idx', 'S': 'S_idx', 'YrA': 'YrA_idx'}


def _unit_id_rows(session, signal):
    """{unit_id -> row position} for one signal's own matrix.

    ** Cells are matched to traces BY UNIT ID, never by row position. ** This matters
    because ``S.zarr`` and ``YrA.zarr`` are exported independently (see the loading block
    at sessions.py:596-626, which reads ``S_idx``, ``C_idx`` and ``YrA_idx`` from three
    separate files) and on this dataset they DO NOT hold the same units. Measured over the
    conditioning cohort: every mouse has an equal COUNT of S and YrA cells -- so a length
    check passes -- but in 8 of 17 mice the id SETS differ by 1-4 cells, and where they
    differ the row alignment shears for the whole remainder of the matrix. For G05, S_idx
    holds unit 71 where YrA_idx holds unit 70, and 37 of 570 row positions thereafter refer
    to different cells. Indexing YrA by S-derived row positions would therefore pair most
    cells with another cell's trace and produce a confident, entirely wrong result.

    This is why ``get_mapping_signal`` is NOT used for the trace matrices here: it resolves
    cells through ``get_S_indeces``, i.e. positions in ``S_idx``, which is correct for C
    (C_idx equals S_idx exactly on this dataset) and wrong for YrA.
    """
    idx = getattr(session, _SIGNAL_INDEX_ATTR[signal], None)
    if idx is None:
        raise RuntimeError(
            f'epoch_modulation: session {session.mouse!r} has no '
            f'{_SIGNAL_INDEX_ATTR[signal]}; the {signal} matrix was never loaded (sessions.py '
            f'leaves YrA_full/YrA_idx as None when YrA.zarr is absent from the Minian output). '
            f'This analysis requires {signal}, so it cannot proceed for this mouse.')
    return {int(u): i for i, u in enumerate(np.asarray(idx))}


def resolve_shared_cells(session, mapping='full', with_crossreg=None):
    """(cell ids usable on EVERY signal, per-signal dropped ids) for one session.

    The analysed cell set is the mapping's cells that carry a trace in **both** signals,
    kept in the mapping's own order. Restricting both lanes to the same cells is deliberate:
    the C lane exists to confirm the YrA lane, and the pre-specified agreement criterion is
    only meaningful if the two are computed over identical cells.

    On this dataset the restriction is small -- 14 of 9531 conditioning cells (0.15%), in 8
    of 17 mice -- but it is REPORTED per mouse rather than absorbed silently, because it
    makes this analysis's denominators differ slightly from the population analysis's, and a
    reader comparing cell counts between figures is entitled to know why.

    A cell with no YrA trace has no YrA measurement; this is a definitional absence, like a
    window that does not fit, not an error to swallow. The upstream fix -- re-exporting
    YrA.zarr over the same unit set as S.zarr -- would remove the restriction entirely.
    """
    _, _, _, cell_ids = session.get_S_mapping(mapping, with_crossreg=with_crossreg)
    lookups = {sig: _unit_id_rows(session, sig) for sig in SIGNALS}
    dropped = {sig: {'missing': [], 'constant': []} for sig in SIGNALS}

    # A constant trace has no defined index -- the standardization would divide by a zero SD.
    # For C this is not a defect but the extreme of its sparsity: C is the deconvolution fit and
    # is exactly zero between fitted transients, so a cell for which CNMF-E fitted NO transient
    # inside the analysis window is identically zero across it. Measured on this cohort, 6 of
    # 9531 cells (0.06%, in 4 mice) are constant in C and NONE are constant in YrA -- which is
    # the same sparsity argument that made YrA the primary signal, showing up at the level of
    # whole cells.
    constant = {}
    for sig in SIGNALS:
        present = [c for c in cell_ids if int(c) in lookups[sig]]
        rows = [lookups[sig][int(c)] for c in present]
        if rows:
            sd = np.asarray(getattr(session, sig), dtype=float)[rows].std(axis=1)
            constant[sig] = {int(present[i]) for i in np.flatnonzero(sd == 0.0)}
        else:
            constant[sig] = set()

    shared = []
    for cid in cell_ids:
        unusable = False
        for sig in SIGNALS:
            if int(cid) not in lookups[sig]:
                dropped[sig]['missing'].append(int(cid)); unusable = True
            elif int(cid) in constant[sig]:
                dropped[sig]['constant'].append(int(cid)); unusable = True
        if not unusable:
            shared.append(cid)

    if not shared:
        raise RuntimeError(
            f'epoch_modulation: mouse {session.mouse!r} has no cell with a usable trace in every '
            f'signal {SIGNALS} for mapping {mapping!r}. Dropped per signal: '
            f'{ {s: {k: len(v) for k, v in d.items()} for s, d in dropped.items()} }.')
    return shared, dropped


def _standardized_trace(session, signal, cell_ids):
    """Per-cell whole-session standardized activity matrix for one signal, in ``cell_ids`` order.

    Rows are selected BY UNIT ID (see ``_unit_id_rows``), so row *i* of the result is
    ``cell_ids[i]`` on every signal and the two signals' matrices are row-comparable.

    ** THE ONLY PLACE THIS MODULE EVER COMPUTES A STANDARD DEVIATION. ** The SD is taken
    over the complete session for each cell independently; nothing downstream re-scales,
    and no window is ever standardized against itself (see the module docstring).

    For YrA the cell's session MEDIAN is subtracted first, matching the existing repo
    convention for this signal (``analysis.py:2323`` uses ``nanmedian`` as F0). Stated
    plainly rather than left to be discovered: this subtraction does NOT change the
    resulting index by even a floating-point ULP in exact arithmetic, because the index is
    a DIFFERENCE of two window means of a STANDARDIZED trace and a per-cell additive
    constant cancels from both the standardization's numerator and the difference. It is
    kept because it makes the trace itself interpretable as a fluorescence change, and
    because the baseline-subtracted trace is what the peri-event heatmap displays -- not
    because it does any statistical work.
    """
    if signal not in SIGNALS:
        raise ValueError(f'epoch_modulation: unknown signal {signal!r}; expected one of {SIGNALS}.')
    lookup = _unit_id_rows(session, signal)
    missing = [int(c) for c in cell_ids if int(c) not in lookup]
    if missing:
        raise RuntimeError(
            f'epoch_modulation: mouse {session.mouse!r} has no {signal} trace for cells '
            f'{missing[:10]}{"..." if len(missing) > 10 else ""}. Cell sets must be resolved '
            f'through resolve_shared_cells() before calling this.')
    rows = [lookup[int(c)] for c in cell_ids]

    x = np.asarray(getattr(session, signal), dtype=float)[rows]
    if x.ndim != 2 or x.shape[0] == 0:
        raise RuntimeError(f'epoch_modulation: {signal} matrix for mouse {session.mouse!r} has '
                           f'shape {x.shape}; expected (n_cells > 0, n_frames).')
    if not np.all(np.isfinite(x)):
        raise RuntimeError(f'epoch_modulation: {signal} matrix for mouse {session.mouse!r} '
                           f'contains {int((~np.isfinite(x)).sum())} non-finite samples. The '
                           f'index is a window mean, so a single NaN would silently poison an '
                           f'entire epoch; fix the source matrix rather than masking here.')

    if signal == 'YrA':
        x = x - np.median(x, axis=1, keepdims=True)

    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    flat = np.flatnonzero(sd[:, 0] == 0.0)
    if flat.size:
        # Defensive: resolve_shared_cells already excludes constant-trace cells, so reaching
        # here means a caller bypassed it. Not a QC verdict on the data -- a constant C is an
        # expected consequence of deconvolution sparsity for a cell with no fitted transients.
        raise RuntimeError(
            f'epoch_modulation: {flat.size} cell(s) of mouse {session.mouse!r} have zero '
            f'{signal} variance over the session (rows {flat[:10].tolist()}), so they cannot be '
            f'standardized. Cell sets must be resolved through resolve_shared_cells(), which '
            f'excludes them, before calling this.')
    return (x - mu) / sd


# ─────────────────────────────────────────────────────────────────────────────
# Trial retention
# ─────────────────────────────────────────────────────────────────────────────

def _window_frames(duration_s):
    return int(round(duration_s * MINISCOPE_FPS))


def _index_window(session, epoch, trial_idx, window_seconds, n_frames):
    """The [onset, offset) frames the index is averaged over, for one epoch on one trial.

    ``window_seconds is None`` -- the epoch's OWN full window, i.e. the analysis described in
    the module docstring. This is the default everywhere and nothing about it changes.

    ``window_seconds`` a float -- the EVENT-PROXIMAL window: that many seconds starting at the
    epoch's ONSET. The onset is read from ``get_epoch_frames`` rather than recomputed from the
    session's declared times, so the two lanes anchor on exactly the same frame and can differ
    only in how far forward they average.

    The offset guard is not defensive boilerplate: for 'post_shock' a short window is always
    inside the recording whenever the full one is (``retained_trials`` requires a 20 s ITI), so
    reaching it means the trial retention rule and this window disagree, which must be visible.
    """
    frames = get_epoch_frames(session, epoch, trial_idx)
    if frames is None:
        raise RuntimeError(
            f'epoch_modulation: mouse {session.mouse!r} trial {trial_idx} has no {epoch!r} '
            f'window, although retained_trials() retained the trial. The retention rule and '
            f'the windowing disagree.')
    if window_seconds is None:
        return int(frames[0]), int(frames[1])

    onset = int(frames[0])
    offset = onset + _window_frames(window_seconds)
    if offset > n_frames:
        raise RuntimeError(
            f'epoch_modulation: mouse {session.mouse!r} trial {trial_idx} {epoch!r} '
            f'event-proximal window [{onset}, {offset}) runs past the {n_frames}-frame '
            f'recording. The trial should not have been retained.')
    return onset, offset


def retained_trials(session):
    """(retained trial indices, per-trial rejection reasons) for one conditioning session.

    A trial is retained only when EVERY window this analysis needs is completely present:
    the 20 s pre-tone baseline, the tone, the FULL 20 s trace interval, the shock, and a
    complete 20 s post-shock window. This is the exposure-matched trial set the population
    conditioning analysis uses, and it drops trial 1 as a CONSEQUENCE of the rule rather
    than as a special case -- trial 1's trace interval is 15 s, not 20 s (its tone onset
    was set to 185 s rather than 180 s; see epoch_analysis.TRACE_MATCHED_WINDOW_S).

    Trial 1 is deliberately NOT recovered by truncating every trace to a common 15 s. It is
    biologically an ACQUISITION trial preceding the first US, and the final 5 s of the
    later trace intervals -- immediately before shock onset -- is exactly where an
    anticipatory signal would live.

    ** The post-shock fit is queried through the 'iti' window, not by calling for
    'post_shock' and catching its exception. ** ``get_epoch_frames`` RAISES for
    'post_shock' when the window runs past the end of the trial (deliberately: for the
    population analysis every trial must supply it), so asking it directly would abort the
    run on the truncated final trial before this function could decide to drop that trial.
    The 'iti' window starts at exactly the same frame -- ``post_shock_onsets[i]`` IS
    ``shock_offsets[i]`` (sessions.py:1337,1340) -- so "the 20 s post-shock window fits" is
    precisely "the ITI is at least 20 s long", asked through the public accessor with no
    duplicated guard logic and no swallowed error.
    """
    post_shock_needed = _window_frames(TRACE_MATCHED_WINDOW_S)

    retained, reasons = [], {}
    for trial_idx in session.periods:
        why = []

        # The one epoch whose declared length varies across trials. Exact, from the protocol
        # definitions -- see nominal_trace_seconds for why this is not measured from frames.
        trace_s = nominal_trace_seconds(session, trial_idx)
        if trace_s < TRACE_MATCHED_WINDOW_S:
            why.append(f'declared trace interval is {trace_s:.0f} s, short of the '
                       f'{TRACE_MATCHED_WINDOW_S:.0f} s exposure-matched window')

        # The remaining windows are checked for PRESENCE only, never for length. 'tone' and
        # 'shock' have fixed declared durations (tone_duration, shock_duration) so their length
        # is not in question, and 'pre_tone_matched' returns None precisely when it does not fit.
        # Their realised frame spans may sit a frame or two off the declared length through
        # timestamp snapping, which is immaterial here: the endpoint is a window MEAN, not a
        # count or a rate, so it is not exposure-scaled. Realised spans are still written to the
        # coverage table, so a genuinely degenerate window stays visible.
        for epoch in (BASELINE_EPOCH, 'tone', 'shock'):
            if get_epoch_frames(session, epoch, trial_idx) is None:
                why.append(f'{epoch} window is absent, empty or inverted')

        iti = get_epoch_frames(session, 'iti', trial_idx)
        if iti is None:
            why.append('inter-trial interval is empty (no room for a post-shock window)')
        elif (iti[1] - iti[0]) < post_shock_needed:
            why.append(f'only {(iti[1] - iti[0]) / MINISCOPE_FPS:.1f} s remain after shock '
                       f'offset, short of the {TRACE_MATCHED_WINDOW_S:.0f} s post_shock window')

        if why:
            reasons[int(trial_idx)] = '; '.join(why)
        else:
            retained.append(int(trial_idx))

    if not retained:
        raise RuntimeError(
            f'epoch_modulation: mouse {session.mouse!r} retained NO conditioning trial. '
            f'Per-trial reasons: {reasons}. Every mouse must contribute at least one fully '
            f'observed trial; an empty retained set is a data or timing problem, not something '
            f'to skip past.')
    return retained, reasons


# ─────────────────────────────────────────────────────────────────────────────
# The modulation table
# ─────────────────────────────────────────────────────────────────────────────

def build_modulation_table(mice_per_group, sessions, signal=PRIMARY_SIGNAL, mapping='full',
                           crossreg_to_use=None, verbose=True, window_seconds=None):
    """Tidy per-(mouse, group, trial, cell, epoch) modulation-index table for one signal.

    Returns (df, coverage) where df has columns
    ``mouse, group, trial, cell, epoch, index`` -- one row per retained trial per cell per
    response epoch -- and ``coverage`` is a per-mouse DataFrame recording how many trials
    were retained and why the others were not.

    Every cell of the mapping contributes to every retained trial and epoch: the index is
    defined from window means of a continuous trace, so unlike the event-based analyses
    there is no zero-event dropout, no eligibility set and no conditional estimand.

    ``window_seconds`` selects the response window and NOTHING else (see ``_index_window``):
    None gives each epoch's own full window -- the analysis this module was written for -- and
    a float gives the EVENT-PROXIMAL window of that many seconds from each epoch's onset. The
    standardization, the trial-matched pre-tone baseline, the retention rule and the resolved
    cell set are identical either way, which is what makes the two lanes a paired comparison
    on the same data rather than two separate analyses. The short-window path additionally
    records ``frac_flat_window`` in the coverage table (see below); the default path's columns
    are unchanged.
    """
    rows, coverage_rows = [], []
    for group in GROUP_ORDER:
        for mouse in mice_per_group.get(group, []):
            if mouse not in sessions:
                continue
            session = sessions[mouse]
            wc = crossreg_to_use[mouse] if (crossreg_to_use is not None and mapping != 'full') else None
            cell_ids, dropped = resolve_shared_cells(session, mapping=mapping, with_crossreg=wc)
            z = _standardized_trace(session, signal, cell_ids)

            trials, why_not = retained_trials(session)
            n_frames = int(z.shape[1])
            n_missing = sum(len(d['missing']) for d in dropped.values())
            n_constant = sum(len(d['constant']) for d in dropped.values())
            n_flat_windows = 0
            windows = {(e, t): _index_window(session, e, t, window_seconds, n_frames)
                       for t in trials for e in RESPONSE_EPOCHS}
            coverage_rows.append({
                'mouse': mouse, 'group': group, 'signal': signal,
                'n_cells': int(z.shape[0]),
                'n_cells_dropped_no_trace': n_missing,
                'n_cells_dropped_constant_trace': n_constant,
                'cells_dropped': '; '.join(
                    f'{s} {k}: {v}' for s, d in dropped.items()
                    for k, v in d.items() if v),
                'n_trials_total': int(len(session.periods)),
                'n_trials_retained': int(len(trials)),
                'retained_trials': ','.join(str(t) for t in trials),
                'window_seconds': '; '.join(
                    f'{e}: ' + ','.join(
                        f'{(windows[(e, t)][1] - windows[(e, t)][0]) / MINISCOPE_FPS:.2f}'
                        for t in trials)
                    for e in RESPONSE_EPOCHS),
                'dropped_trials': '; '.join(f'trial {t}: {r}' for t, r in sorted(why_not.items())),
            })

            for trial_idx in trials:
                # The baseline is ALWAYS the full pre-tone window, on both lanes. It is a mean
                # against a mean and is not exposure-scaled, so a longer baseline is simply a
                # more precise one; shortening it to match the response window would only add
                # noise to every index.
                base_on, base_off = get_epoch_frames(session, BASELINE_EPOCH, trial_idx)
                baseline = z[:, base_on:base_off].mean(axis=1)
                for epoch in RESPONSE_EPOCHS:
                    on, off = windows[(epoch, trial_idx)]
                    # Window MEAN of the already-standardized trace; no SD is taken here.
                    value = z[:, on:off].mean(axis=1) - baseline
                    if window_seconds is not None:
                        # How many (cell, trial, epoch) values come from a window in which the
                        # trace never changes. On C this is deconvolution sparsity at the scale
                        # of a single short window: a cell with no fitted transient inside it is
                        # identically zero there, so its index is exactly -(its pre-tone mean) --
                        # a deterministic constant rather than a measurement. resolve_shared_cells
                        # only excludes cells constant over the WHOLE session, so these stay in
                        # by design; this counts them instead of hiding them.
                        n_flat_windows += int((np.ptp(z[:, on:off], axis=1) == 0.0).sum())
                    for cell_id, v in zip(cell_ids, value):
                        rows.append({'mouse': mouse, 'group': group, 'trial': int(trial_idx),
                                     'cell': cell_id, 'epoch': epoch, 'index': float(v)})
            if window_seconds is not None:
                n_values = int(z.shape[0]) * len(trials) * len(RESPONSE_EPOCHS)
                coverage_rows[-1]['n_flat_windows'] = n_flat_windows
                coverage_rows[-1]['frac_flat_window'] = n_flat_windows / n_values
            if verbose:
                bits = ([f'{n_missing} no trace'] if n_missing else []) + \
                       ([f'{n_constant} constant trace'] if n_constant else [])
                note = f'  [dropped: {", ".join(bits)}]' if bits else ''
                if window_seconds is not None and n_flat_windows:
                    note += (f'  [{n_flat_windows} flat {window_seconds:g} s windows = '
                             f'{100 * coverage_rows[-1]["frac_flat_window"]:.1f}%]')
                print(f'[epoch-modulation/{signal}] {mouse} ({GROUP_LABELS[group]}): '
                      f'{z.shape[0]} cells x {len(trials)} retained trials{note}', flush=True)

    if not rows:
        raise RuntimeError('epoch_modulation: produced an empty modulation table -- check '
                           'mice_per_group / sessions / mapping.')
    return pd.DataFrame(rows), pd.DataFrame(coverage_rows)


def summarize_cells(df):
    """Per-(mouse, group, cell, epoch) index: the MEDIAN across that mouse's retained trials.

    The median rather than the mean so that a single anomalous conditioning trial cannot
    define a neuron's response -- the trial dimension is preserved all the way to here and
    collapsed only at this step.
    """
    out = (df.groupby(['mouse', 'group', 'cell', 'epoch'], observed=True)['index']
             .median().reset_index())
    return out


def summarize_mice(df_cell):
    """Per-(mouse, group, epoch) endpoint: the MEAN modulation index across that mouse's cells.

    Also carries ``n_cells`` and ``sem_cells`` -- the within-mouse standard error across
    cells. That column is the free diagnostic for the "does averaging cells throw away the
    per-cell information?" question: if each mouse's SEM is small relative to the spread of
    the mouse means themselves, then sigma^2_within/n_cells is negligible in
    Var(mouse mean) = sigma^2_between + sigma^2_within/n_cells, and collapsing to the mean
    costs essentially no treatment-level precision. It is computed from data already in
    hand, so it demonstrates the claim on this dataset instead of asserting it.
    """
    grouped = df_cell.groupby(['mouse', 'group', 'epoch'], observed=True)['index']
    out = grouped.agg(value='mean', n_cells='size', sd_cells='std').reset_index()
    out['sem_cells'] = out['sd_cells'] / np.sqrt(out['n_cells'])
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Mouse-level model
# ─────────────────────────────────────────────────────────────────────────────

def _coefficient_names(result):
    """Map the fitted model's coefficient names onto (group,) and (group, epoch) keys.

    statsmodels' dummy-name format depends on the formula, so the names are READ OFF the
    fitted result and matched by their level tokens rather than reconstructed by string
    formatting -- the convention ``linear_contrast_test``/``joint_wald_test`` require.
    """
    names = list(result.fe_params.index if hasattr(result, 'fe_params') else result.params.index)
    group_main, interaction = {}, {}
    treatments = [g for g in GROUP_ORDER if g != REFERENCE_GROUP]
    non_ref_epochs = [e for e in RESPONSE_EPOCHS if e != REFERENCE_EPOCH]

    for name in names:
        has_group = [g for g in treatments if f'[T.{g}]' in name]
        has_epoch = [e for e in non_ref_epochs if f'[T.{e}]' in name]
        if len(has_group) == 1 and not has_epoch:
            group_main[has_group[0]] = name
        elif len(has_group) == 1 and len(has_epoch) == 1 and ':' in name:
            interaction[(has_group[0], has_epoch[0])] = name

    missing_main = [g for g in treatments if g not in group_main]
    missing_int = [(g, e) for g in treatments for e in non_ref_epochs
                   if (g, e) not in interaction]
    if missing_main or missing_int:
        raise RuntimeError(
            f'epoch_modulation: could not resolve model coefficients. Missing group terms '
            f'{missing_main}, missing interaction terms {missing_int}. Fitted names: {names}')
    return group_main, interaction


def fit_mouse_level_model(df_mouse, alpha=0.05):
    """Fit the paper-facing mouse-level model and extract its contrasts.

        index ~ C(group, Treatment("mCherry")) * C(epoch, Treatment("tone")) + (1 | mouse)

    Epoch levels are tone/trace/shock/post_shock ONLY. Pre-tone is the within-trial
    construction baseline of the index and is not a level -- as a level it would be
    identically zero.

    The index is a SIGNED STANDARDIZED DIFFERENCE and is analysed UNTRANSFORMED: no log,
    no exponentiation, and effects are reported as differences in modulation index with
    95% CIs. No claim is made about the shape or symmetry of its distribution.

    Within each epoch the two treatment-versus-control contrasts are Holm-corrected
    together. At the reference epoch the contrast IS the group coefficient; at the other
    epochs it is the group coefficient combined with that epoch's interaction term through
    the fitted covariance, which is a contrast rather than a coefficient and so cannot be
    read off the model summary.

    Epoch dependence is assessed ONCE by the joint Wald test of all six interaction
    coefficients. Note what that omnibus does and does not do: a significant interaction
    establishes that the group effect is NOT UNIFORM ACROSS EPOCHS, but it does not itself
    identify WHICH epochs differ. The within-epoch contrasts describe where the effect
    sits; they do not upgrade to an epoch-specificity claim on their own, and the omnibus
    does not localize. Callers must not infer an epoch-specific effect from "significant in
    one epoch, not in another".

    Returns (result, method, text, contrasts_df, omnibus).
    """
    df = df_mouse.copy()
    df['group'] = pd.Categorical(
        df['group'], categories=[REFERENCE_GROUP] + [g for g in GROUP_ORDER if g != REFERENCE_GROUP])
    df['epoch'] = pd.Categorical(
        df['epoch'], categories=[REFERENCE_EPOCH] + [e for e in RESPONSE_EPOCHS if e != REFERENCE_EPOCH])

    n_mice = df['mouse'].nunique()
    formula = (f'value ~ C(group, Treatment("{REFERENCE_GROUP}")) '
               f'* C(epoch, Treatment("{REFERENCE_EPOCH}"))')
    header = (f'Unit of inference: mouse (n = {n_mice}); denominator df = n_mouse - 1 = '
              f'{n_mice - 1}.\nEpoch levels: {list(RESPONSE_EPOCHS)} '
              f'(reference {REFERENCE_EPOCH!r}); pre-tone is the construction baseline of the '
              f'index, not a level.\nResponse analysed untransformed (signed standardized '
              f'difference).\n\n')
    # NUMERICAL choice, not a statistical one (see fit_mixed_model's `method` docstring): on this
    # frame shape -- 68 rows, 4 per mouse, across 17 mice -- statsmodels' default 'lbfgs' raises
    # LinAlgError('Singular matrix') inside MixedLM.fit, which fit_mixed_model then reports as a
    # failed convergence and answers with mouse-clustered OLS. bfgs, cg, powell and nm all reach
    # the SAME interior optimum on the same data (verified: mouse variance 0.006746 to six
    # figures across all four), so starting the escalation at bfgs recovers the mixed model this
    # analysis's METHODS actually describes. Passing a list that BEGINS with lbfgs does not help:
    # the exception escapes statsmodels' own method escalation. The clustered-OLS fallback stays
    # available for genuine degeneracy; it is simply no longer triggered by an optimizer quirk.
    result, method, text = fit_mixed_model(df, formula, group_col='mouse', extra_header=header,
                                           method=['bfgs', 'cg', 'powell'])

    group_main, interaction = _coefficient_names(result)
    treatments = [g for g in GROUP_ORDER if g != REFERENCE_GROUP]

    records = []
    for epoch in RESPONSE_EPOCHS:
        per_epoch = []
        for g in treatments:
            weights = {group_main[g]: 1.0}
            if epoch != REFERENCE_EPOCH:
                weights[interaction[(g, epoch)]] = 1.0
            stat = linear_contrast_test(result, weights, n_groups=n_mice, alpha=alpha)
            stat.update({'epoch': epoch, 'group': g, 'reference': REFERENCE_GROUP})
            per_epoch.append(stat)
        # Holm across exactly the two treatment-versus-control contrasts within this epoch.
        _, padj = holm_correct([s['p'] for s in per_epoch], alpha=alpha)
        for s, pa in zip(per_epoch, padj):
            s['p_holm'] = float(pa)
        records.extend(per_epoch)
    contrasts = pd.DataFrame(records)

    omnibus = joint_wald_test(result, list(interaction.values()), n_groups=n_mice)
    return result, method, text, contrasts, omnibus


def format_variance_diagnostic(df_mouse):
    """Within-mouse (across-cell) SEM versus the between-mouse spread, per epoch.

    The endpoint of this analysis is a mean over each mouse's cells, which invites the
    objection that collapsing hundreds of cells throws away the single-cell information.
    For TREATMENT-level precision it does not, and this block demonstrates rather than
    asserts it on the actual data:

        Var(mouse mean) = sigma^2_between + sigma^2_within / n_cells

    The second term is what averaging removes. When the mean within-mouse SEM is small
    relative to the SD of the mouse means themselves, that term is negligible, each mouse's
    mean is already a near-noise-free estimate of that mouse's value, and a cell-level model
    would return essentially the same standard error on a group contrast. What averaging
    DOES discard is distribution shape, which the panels display and the hierarchical
    cell-level companion models -- that is a limitation of the endpoint, not a power deficit.

    Uses only columns already computed by ``summarize_mice``; nothing is refit.
    """
    lines = ['', 'Within-mouse versus between-mouse variability (diagnostic, nothing is refit)',
             '-' * 74,
             'Var(mouse mean) = sigma^2_between + sigma^2_within/n_cells. A mean within-mouse',
             'SEM well below the between-mouse SD means the second term is negligible, so',
             'collapsing cells to a mouse mean costs essentially no treatment-level precision.',
             'It does discard distribution SHAPE, which is a different matter from power.',
             '']
    for epoch in RESPONSE_EPOCHS:
        sub = df_mouse[df_mouse['epoch'] == epoch]
        between = float(sub['value'].std(ddof=1))
        within = float(sub['sem_cells'].mean())
        ratio = (within / between) if between > 0 else float('nan')
        lines.append(f'  {epoch:>11}: mean within-mouse SEM = {within:.4f}   '
                     f'between-mouse SD = {between:.4f}   ratio = {ratio:.3f}   '
                     f'(median n_cells = {sub["n_cells"].median():.0f})')
    return '\n'.join(lines) + '\n'


def format_model_report(contrasts, omnibus, method, n_mice, window_seconds=None):
    """Human-readable contrast/omnibus block for the stats output file.

    ``window_seconds`` mirrors ``build_modulation_table``: None describes the full-epoch
    analysis (unchanged), a float describes the event-proximal windows, whose per-window
    caveats are different ones -- the shock window is no longer the 2 s US, and the overlap
    with post-shock is created by the window length rather than by indicator kinetics alone.
    """
    if window_seconds is None:
        heading = 'Mouse-level model of per-cell epoch modulation'
        shock_note = '   [2 s window: fewer samples than the 20 s epochs]'
        tail = [
            'The shock epoch is a 2 s window (~40 frames at 20 Hz) and is therefore',
            'estimated from far less data than the 20 s epochs; the short window may',
            'contribute to greater dispersion of the shock index. Because of GCaMP',
            'kinetics, shock-evoked activity can also extend into the separate post-shock',
            'window, so those two epochs are not independent readouts.',
        ]
    else:
        heading = (f'Mouse-level model of per-cell EVENT-PROXIMAL modulation '
                   f'({window_seconds:g} s from onset)')
        shock_note = ''
        tail = [
            f'Every window is {window_seconds:g} s from its event onset, so all four are',
            'estimated from the same number of frames. Two consequences to keep in view:',
            f'  - the shock window OVERLAPS the post-shock window by '
            f'{max(0.0, window_seconds - 2.0):g} s, because the',
            '    US is only 2 s long. Those two estimates are correlated by construction, and',
            '    no epoch-specificity claim should be made across that boundary.',
            '  - the shock event is the one window the full-epoch analysis did NOT dilute:',
            '    its epoch was already the 2 s US, so agreement between the two lanes there',
            '    is an internal check rather than an independent result.',
            '',
            'This is a COMPANION lane. The full-epoch analysis beside it is unchanged and',
            'remains the paper-facing one.',
        ]
    lines = [
        heading,
        '=' * len(heading),
        f'Fit method: {method};  n mice = {n_mice};  denominator df = {n_mice - 1}',
        '',
        f'Group x epoch joint Wald test (all {omnibus["df1"]} interaction coefficients):',
        f'  F({omnibus["df1"]}, {omnibus["df2"]}) = {omnibus["F"]:.3f}, P = {omnibus["p"]:.4g}',
        '  Interpretation: this establishes whether the group effect is NON-UNIFORM across',
        '  epochs. It does NOT identify which epochs differ -- the within-epoch contrasts',
        '  below describe where the effect sits and do not, on their own, support an',
        '  epoch-specificity claim.',
        '',
        f'Within-epoch contrasts versus {REFERENCE_GROUP} (Holm-corrected across the two',
        'treatment-versus-control contrasts WITHIN each epoch); differences in modulation',
        'index, analysed untransformed:',
    ]
    for epoch in RESPONSE_EPOCHS:
        lines.append(f'  {epoch}:')
        for _, r in contrasts[contrasts['epoch'] == epoch].iterrows():
            note = shock_note if epoch == 'shock' else ''
            lines.append(
                f'    {GROUP_LABELS[r["group"]]:>3} vs {GROUP_LABELS[REFERENCE_GROUP]}: '
                f'{r["estimate"]:+.4f} [{r["ci_low"]:+.4f}, {r["ci_high"]:+.4f}]  '
                f't({int(r["df"])}) = {r["t"]:.3f}  P = {r["p"]:.4g}  '
                f'P_holm = {r["p_holm"]:.4g}{note}')
    lines.append('')
    lines.extend(tail)
    return '\n'.join(lines) + '\n'


# ─────────────────────────────────────────────────────────────────────────────
# Panels
# ─────────────────────────────────────────────────────────────────────────────

def _model_contrast_stat_fn(contrasts, epoch):
    """stat_fn adapter that reports THIS ANALYSIS'S model-derived p-values on the panel.

    ``draw_superplot_triplet``'s bracket path calls
    ``stat_fn(hM3D, hM4D, mCherry, ax, heights, group_order=..., annotate=False)`` and uses
    the three returned p-values, in pair order [(Exc,Inh),(Exc,Ctl),(Inh,Ctl)]. The default
    ``do_pairwise_holm_plot`` would compute a Welch test on the per-mouse means -- a
    DIFFERENT test from the one this module's METHODS describes. Substituting this closure
    keeps the drawing machinery and reports the mixed model's own Holm-corrected
    within-epoch contrasts instead, so the panel and the stats file cannot disagree.

    Exc-versus-Inh is returned as NaN: it is not in this analysis's correction family, and
    ``annotate_pairwise_brackets`` skips a NaN pair ("this comparison is not on this
    panel") rather than drawing it.
    """
    rows = contrasts[contrasts['epoch'] == epoch].set_index('group')

    def stat_fn(_g_hM3D, _g_hM4D, _g_mCherry, _ax, _heights, group_order=None, annotate=True):
        return np.array([np.nan,
                         float(rows.loc['hM3D', 'p_holm']),
                         float(rows.loc['hM4D', 'p_holm'])], dtype=float)

    return stat_fn


def plot_epoch_modulation_superplot(df_cell, df_mouse, contrasts, omnibus, out_path,
                                    signal=PRIMARY_SIGNAL, auto_close=True,
                                    title_note=None, footer=None,
                                    title_prefix='Per-cell epoch modulation',
                                    epoch_titles=None, ylabel=None):
    """Panel L: per-cell modulation index by epoch and group, mice overlaid.

    Cells are the visual cloud; every statistic comes from the per-mouse means, and the
    brackets are this module's model-derived within-epoch contrasts (see
    ``_model_contrast_stat_fn``). Group DISPLAY order is mCherry, hM3D, hM4D throughout.

    ``yscale='linear'`` is passed deliberately. ``draw_superplot_triplet`` defaults to
    'auto', which puts heavy-tailed POSITIVE quantities on a log/symlog axis; the
    modulation index is a signed standardized difference that takes negative values, so
    that default does not apply to it.

    ``title_note`` and ``footer`` both default to None, which reproduces panel L exactly as it
    has always been drawn. They exist so the hierarchical companion lane can reuse this function
    verbatim -- same panel grammar, same bracket path, different SOURCE of the p-values -- while
    saying on its own figure which analysis produced the numbers on it. A caller that supplies
    brackets from somewhere other than the mouse-level model must supply a ``title_note``.

    ``title_prefix``, ``epoch_titles`` and ``ylabel`` default to the full-epoch analysis's own
    wording. The event-proximal lane overrides them because its windows are NOT the epochs --
    leaving the defaults there would label a 3 s post-onset window "shock (2 s)" and describe
    the y-axis as "epoch - pre-tone", both of which would be false on that figure.
    """
    fig, axs = plt.subplots(1, len(RESPONSE_EPOCHS),
                            figsize=(3.1 * len(RESPONSE_EPOCHS), 4.2), sharey=True)
    axs = np.atleast_1d(axs)

    for ax, epoch in zip(axs.flat, RESPONSE_EPOCHS):
        cell_vals, mouse_means = {}, {}
        for group in DREADD_DISPLAY_ORDER:
            sub_cell = df_cell[(df_cell['group'] == group) & (df_cell['epoch'] == epoch)]
            sub_mouse = df_mouse[(df_mouse['group'] == group) & (df_mouse['epoch'] == epoch)]
            cell_vals[group] = {m: g['index'].to_numpy() for m, g in sub_cell.groupby('mouse')}
            mouse_means[group] = dict(zip(sub_mouse['mouse'], sub_mouse['value']))

        draw_superplot_triplet(
            ax, cell_vals, mouse_means, DREADD_DISPLAY_ORDER, GROUP_COLOURS,
            stat_fn=_model_contrast_stat_fn(contrasts, epoch), yscale='linear',
            annotate='stats')
        ax.axhline(0.0, color='0.6', lw=0.8, ls='--', zorder=1)
        ax.set_xticks(range(len(DREADD_DISPLAY_ORDER)))
        ax.set_xticklabels([GROUP_LABELS[g] for g in DREADD_DISPLAY_ORDER], size='medium')
        default_title = epoch if epoch != 'shock' else 'shock (2 s)'
        ax.set_title((epoch_titles or {}).get(epoch, default_title), size='medium')
        grow_ylim_for_bracket_headroom(ax)

    axs.flat[0].set_ylabel(ylabel or 'Modulation index\n(epoch - pre-tone, cell SD units)')
    title = (f'{title_prefix} ({signal}) — group x epoch '
             f'F({omnibus["df1"]},{omnibus["df2"]}) = {omnibus["F"]:.2f}, '
             f'P = {omnibus["p"]:.3g}')
    if title_note:
        title += f'\n{title_note}'
    fig.suptitle(title, size='medium')
    if footer:
        fig.text(0.5, 0.012, footer, ha='center', size='xx-small')
    fig.subplots_adjust(left=0.09, bottom=0.11 if not footer else 0.26,
                        right=0.98, top=0.84 if not title_note else 0.80, wspace=0.16)
    save_fig(fig, out_path)
    if auto_close:
        plt.close(fig)


def _tone_aligned_matrix(z, session, trials):
    """Trial-averaged standardized activity aligned to tone onset (cells x window frames).

    ** No re-standardization happens here. ** ``event_locked_responsiveness._peri_event_matrix``
    z-scores each cell ACROSS THE ALIGNMENT WINDOW, which is a perfectly reasonable choice
    for that analysis but would put this panel on a different scale from the index it is
    supposed to illustrate -- and would be a second standardization, which this module's
    measurement definition forbids. The trace arrives already standardized once over the
    whole session and is only averaged over trials here.

    Averaged over the SAME retained trials the index uses, so panel and statistics describe
    the same data.
    """
    pre_f = _window_frames(HEATMAP_PRE_S)
    post_f = _window_frames(HEATMAP_POST_S)
    n_frames = z.shape[1]

    per_trial = []
    for trial_idx in trials:
        onset = get_epoch_frames(session, 'tone', trial_idx)[0]
        lo, hi = onset - pre_f, onset + post_f
        if lo < 0 or hi > n_frames:
            continue
        per_trial.append(z[:, lo:hi])
    if not per_trial:
        return None, None
    avg = np.mean(np.stack(per_trial, axis=0), axis=0)
    time_axis = (np.arange(pre_f + post_f) - pre_f) / MINISCOPE_FPS
    return avg, time_axis


def plot_modulation_heatmaps(group_mats, group_sort_values, time_axis, out_path,
                             signal=PRIMARY_SIGNAL, vmax=1.0, auto_close=True):
    """Panel K: one tone-onset-aligned heatmap per group, cells sorted by trace modulation.

    Cells are sorted by their TRACE index rather than by mean post-onset activity, so the
    ordering is the same quantity the statistics are built on. Groups share one colour
    scale. Display order is mCherry, hM3D, hM4D.
    """
    fig, axs = plt.subplots(1, len(DREADD_DISPLAY_ORDER),
                            figsize=(4.1 * len(DREADD_DISPLAY_ORDER), 4.4))
    axs = np.atleast_1d(axs)

    for ax, group in zip(axs.flat, DREADD_DISPLAY_ORDER):
        mat = group_mats.get(group)
        if mat is None or mat.size == 0:
            raise RuntimeError(f'epoch_modulation: no heatmap rows for group {group!r}; every '
                               f'group must contribute cells to panel K.')
        order = np.argsort(-np.asarray(group_sort_values[group], dtype=float))
        ax.imshow(mat[order], aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                  extent=[time_axis[0], time_axis[-1], mat.shape[0], 0])
        # Epoch boundaries, so the reader can see which band is which epoch rather than
        # inferring it from the x-axis. Nominal offsets from tone onset: the tone runs
        # 0-20 s, the trace interval 20-40 s, the 2 s shock 40-42 s, and the post-shock
        # window 42-62 s (see epoch_analysis.TRACE_MATCHED_WINDOW_S and the session's
        # tone/shock definitions). The shock band is shaded rather than ruled because at
        # this width two lines 2 s apart are indistinguishable.
        for boundary in (0.0, 20.0, 40.0, 42.0, 62.0):
            ax.axvline(boundary, color='k', lw=0.8,
                       ls='-' if boundary == 0.0 else ':')
        ax.axvspan(40.0, 42.0, color='k', alpha=0.12, lw=0)
        ax.set_title(f'{GROUP_LABELS[group]} (n = {mat.shape[0]} cells)', size='medium')
        ax.set_xlabel('Time from tone onset (s)')
        ax.set_xticks([0, 20, 40, 62])
        sec = ax.secondary_xaxis('top')
        sec.set_xticks([10.0, 30.0, 41.0, 52.0])
        sec.set_xticklabels(['tone', 'trace', 'US', 'post-shock'], size='small')
        sec.tick_params(length=0)
    axs.flat[0].set_ylabel('Cell (sorted by trace modulation)')
    fig.suptitle(f'Tone-aligned per-cell activity ({signal}; standardized once per cell '
                 f'over the session)', size='medium')
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.13, top=0.86, wspace=0.15)
    save_fig(fig, out_path)
    if auto_close:
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Signal comparison
# ─────────────────────────────────────────────────────────────────────────────

def compare_signals(contrasts_by_signal, omnibus_by_signal):
    """Side-by-side within-epoch contrasts for the primary and confirmatory signals.

    Agreement is PRE-SPECIFIED: the two signals agree on a contrast when the estimates
    match in sign and their 95% CIs overlap. Disagreement is REPORTED -- this function
    never picks a winner, and nothing downstream is allowed to choose whichever signal
    looks better.
    """
    primary = contrasts_by_signal[PRIMARY_SIGNAL]
    confirm = contrasts_by_signal[CONFIRMATORY_SIGNAL]
    merged = primary.merge(confirm, on=['epoch', 'group'], suffixes=('_primary', '_confirm'))

    same_sign = np.sign(merged['estimate_primary']) == np.sign(merged['estimate_confirm'])
    overlap = ((merged['ci_low_primary'] <= merged['ci_high_confirm']) &
               (merged['ci_low_confirm'] <= merged['ci_high_primary']))
    merged['agrees'] = same_sign & overlap

    lines = [
        f'Signal comparison: {PRIMARY_SIGNAL} (primary) vs {CONFIRMATORY_SIGNAL} (confirmatory)',
        '=' * 72,
        'The same index, the same model, run twice on two signals. The comparison is about',
        'AGREEMENT OF CONCLUSIONS, not equality of magnitudes: YrA\'s per-cell SD is',
        'dominated by measurement noise while C\'s is dominated by fitted transients, so the',
        'same numerical index means different physical things on the two.',
        'Agreement criterion: matching sign AND overlapping 95% CIs.',
        '',
    ]
    for sig in SIGNALS:
        om = omnibus_by_signal[sig]
        lines.append(f'  group x epoch omnibus [{sig}]: F({om["df1"]},{om["df2"]}) = '
                     f'{om["F"]:.3f}, P = {om["p"]:.4g}')
    lines.append('')
    for _, r in merged.iterrows():
        verdict = 'agree' if r['agrees'] else 'DISAGREE'
        lines.append(
            f'  {r["epoch"]:>11} {GROUP_LABELS[r["group"]]:>3} vs {GROUP_LABELS[REFERENCE_GROUP]}: '
            f'{PRIMARY_SIGNAL} {r["estimate_primary"]:+.4f} '
            f'[{r["ci_low_primary"]:+.4f}, {r["ci_high_primary"]:+.4f}]  |  '
            f'{CONFIRMATORY_SIGNAL} {r["estimate_confirm"]:+.4f} '
            f'[{r["ci_low_confirm"]:+.4f}, {r["ci_high_confirm"]:+.4f}]  -> {verdict}')

    n_disagree = int((~merged['agrees']).sum())
    lines.append('')
    if n_disagree:
        lines.append(f'{n_disagree} of {len(merged)} contrasts DISAGREE between signals. This is '
                     f'reported, not adjudicated: report both rather than selecting a signal. '
                     f'Each signal has its own panels, so both can be inspected directly.')
        lines.append('')
        lines.append('CAVEAT on this criterion: requiring a matching SIGN is informative for a '
                     'non-null effect but degenerate near zero, where the sign of an estimate '
                     'is noise. A pair whose CIs overlap almost entirely and both span zero is '
                     'flagged as disagreeing purely because the point estimates fall on '
                     'opposite sides of zero -- which is agreement about there being no effect, '
                     'not a discrepancy. Check the estimates above before treating a flag as '
                     'a real signal difference.')
    else:
        lines.append(f'All {len(merged)} contrasts agree between signals.')
    return merged, '\n'.join(lines) + '\n'


# ─────────────────────────────────────────────────────────────────────────────
# Hierarchical cell-level companion lane
#
# The mouse-level model above is the paper-facing analysis and this changes nothing about
# it. What it adds is the piece that averaging to a mouse mean throws away: WITHIN-MOUSE
# CELLULAR COHERENCE. It does NOT add treatment-level power, and nothing here is ever
# reported as n = thousands of cells -- 17 animals were assigned a treatment, and every
# p-value below comes from permuting those 17 MOUSE labels with each cell fixed to its own
# animal.
#
#     index ~ group * epoch + (1|mouse) + (0 + epoch||mouse) + (1|mouse:cell)
#
# ** The animal-level epoch random effect is required, not optional. ** Without it,
# between-animal variation in the epoch effect -- which is the thing being compared across
# groups -- has nowhere to go, and the cells become replicates for the epoch contrast. It is
# INDEPENDENT (uncorrelated) across epochs with its own variance per epoch, mirroring the
# recall lane's frozen structure; the shock epoch's between-animal variance is an order of
# magnitude larger than tone's on these data, so a single shared epoch variance would be the
# wrong constraint.
#
# ONE GENUINE IMPROVEMENT OVER THE RECALL LANE
# --------------------------------------------
# That analysis needed a CONDITIONAL estimand, because pre and post amplitudes came from
# different sets of active cells and only their intersection could be paired. Here the index
# is within-cell and within-trial and every cell is defined in every window (see
# build_modulation_table), so the estimand is UNCONDITIONAL: there is no eligibility set, no
# eligibility table, and no fraction-eligible caveat travelling with the estimate.
# ─────────────────────────────────────────────────────────────────────────────

HIERARCHICAL_DIRNAME = 'hierarchical_cells'
HIERARCHICAL_METHODS_FILENAME = 'epoch_modulation_hierarchical_cells_methods.md'
HIERARCHICAL_SEED = 0

# Monte Carlo settings for the OMNIBUS only -- the pairwise tests are exact enumerations. The
# global null's 17!/(5!6!6!) = 5,717,712 relabelings cannot be enumerated even at a millisecond
# per refit, so the omnibus is Monte Carlo at a frozen seed, and both the seed and the draw count
# are written into the output so a development pass at a lower count can never be mistaken for a
# reported one.
#
# ** 2,000, the same frozen value as sp_rates_lmm.HIERARCHICAL_CELL_N_PERM_OMNIBUS, and for the
# same reason. ** This was briefly 20,000, chosen on a mis-measured refit cost: the Wald variant
# of the statistic costs ~39 ms per draw, not the ~10 ms the plain contrast refit costs, so 20,000
# draws is ~13 min per signal against ~1.3 min. What that bought is nothing: the Monte Carlo
# standard error on an omnibus P near 0.6 is sqrt(p(1-p)/n) = 0.011 at 2,000 draws and 0.0035 at
# 20,000, and no decision in this lane turns on a difference of one part in a hundred that far
# from any threshold. Raise it deliberately if an omnibus ever lands near 0.05, where the Monte
# Carlo error would actually be load-bearing -- and say so in the output, which already records
# the count.
HIERARCHICAL_N_PERM_OMNIBUS = 2000

# ** Refit the whole model through statsmodels on all 38,060 cell rows and compare? OFF. **
# This was mandatory in the first version of the lane and was 246 s (YrA) / 807 s (C) -- 17.5 of
# the 21 minutes the lane cost, to compute a number that is not reported and does not gate
# anything. It is a corroboration, and a weak one: measured on these data statsmodels converges to
# a WORSE point on the shared objective than the collapsed fit does (it stops short along the flat
# between-mouse variance directions that 17 groups leave poorly determined), so where the two
# disagree the collapsed fit is right and the cross-check is what needs explaining.
#
# What actually guarantees the collapsed fit is the model is assert_reml_objective_identity: it
# checks the collapsed objective against a naive full-covariance reference implementation at
# several unrelated parameter vectors and requires the difference to be constant. That is an
# EXACT check (agreement to ~1e-13 on these data), it is mandatory, and it costs under a second.
# The expensive fit adds only "and a third implementation lands nearby at the true cell counts".
#
# Turn it on deliberately -- after changing the model structure, or when a reviewer wants the
# belt-and-braces version -- with cross_check_full_fit=True.
HIERARCHICAL_CROSS_CHECK_FULL_FIT = False

# NUMERICAL optimizer for the full cell-level fit -- frozen, and not a statistical choice.
# fit_mixed_model's default 'lbfgs' is not used anywhere in this lane: measured on this dataset it
# lands on the boundary of the parameter space on a frame of this shape (38,060 rows, ~560 cells
# per mouse in each of 17 mice). bfgs/cg/powell reach the same interior optimum on the same
# likelihood. Same reasoning, and the same list, as sp_rates_lmm.HIERARCHICAL_CELL_LMM_OPTIMIZER.
HIERARCHICAL_LMM_OPTIMIZER = ['bfgs', 'cg', 'powell']

# All three pairwise comparisons are COMPUTED and tabulated. Only the two treatment-versus-control
# comparisons enter a Holm family, and the family is formed WITHIN each epoch -- exactly the
# structure of the mouse-level lane (see fit_mouse_level_model), so the companion answers the same
# question with the same multiplicity rule rather than importing the recall lane's three-member
# family. hM3D-vs-hM4D is in NO family here, is flagged as such in every output, and carries no
# bracket -- matching panel L, which skips it.
HIERARCHICAL_PAIRS = (('hM3D', 'mCherry'), ('hM4D', 'mCherry'), ('hM3D', 'hM4D'))
HIERARCHICAL_FAMILY_PAIRS = (('hM3D', 'mCherry'), ('hM4D', 'mCherry'))

# Tolerances for the collapse-equivalence assertion (see assert_collapse_equivalence). Absolute on
# the coefficients, relative on the standard errors: the two fits optimize the same REML objective
# with different optimizers, so they agree to their own convergence tolerance and not to machine
# precision. Measured agreement on this dataset is ~2e-6 (coefficients) and ~0.5% (SEs).
HIERARCHICAL_BETA_ATOL = 1e-4
HIERARCHICAL_SE_RTOL = 0.02

_EPOCH_INDICATOR = {e: f'is_{e}' for e in RESPONSE_EPOCHS}


def _hierarchical_frame(df_cell):
    """``df_cell`` with the columns the cell-level fit needs, and nothing else changed.

    ``cell`` ids are only unique WITHIN a mouse, so the nested ``(1|mouse:cell)`` grouping needs
    an explicit mouse-qualified id -- pooling on the bare id would merge one mouse's cell 7 with
    every other mouse's cell 7 into a single random effect.
    """
    df = _code_group_epoch(df_cell)
    df['cell_uid'] = df['mouse'].astype(str) + '_' + df['cell'].astype(str)
    for epoch, col in _EPOCH_INDICATOR.items():
        df[col] = (df['epoch'] == epoch).astype(float)
    return df


def _code_group_epoch(df):
    """``group`` and ``epoch`` as categoricals with the model's reference levels first.

    Split out from _hierarchical_frame so the tiny design grid in _design_blocks -- which has no
    mouse or cell columns -- gets exactly the same coding as the data. Patsy reads the reference
    level off the category order, so a grid coded any other way would silently produce a design
    whose columns do not line up with the fitted model's.
    """
    out = df.copy()
    out['group'] = pd.Categorical(
        out['group'], categories=[REFERENCE_GROUP] + [g for g in GROUP_ORDER if g != REFERENCE_GROUP])
    out['epoch'] = pd.Categorical(
        out['epoch'], categories=[REFERENCE_EPOCH] + [e for e in RESPONSE_EPOCHS if e != REFERENCE_EPOCH])
    return out


def _hierarchical_formula():
    return (f'index ~ C(group, Treatment("{REFERENCE_GROUP}")) '
            f'* C(epoch, Treatment("{REFERENCE_EPOCH}"))')


def _hierarchical_vc_formula():
    """Variance components for ``(0 + epoch||mouse) + (1|mouse:cell)``.

    One component per epoch, each with its own variance and uncorrelated with the others, plus
    the nested cell intercept. The mouse intercept is NOT here -- it comes from ``re_formula='1'``,
    which fit_mixed_model requires alongside a vc_formula precisely so it cannot go missing.
    """
    vc = {f'epoch_{e}': f'0 + {_EPOCH_INDICATOR[e]}' for e in RESPONSE_EPOCHS}
    vc['cell'] = '0 + C(cell_uid)'
    return vc


def fit_hierarchical_cell_model(df_cell):
    """The full model on all (cell x epoch) rows, fit through statsmodels.

    ** This is the INDEPENDENT CROSS-CHECK, not the fit the lane reports. ** The reported estimate,
    interval and p-value all come from CollapsedEpochModel, which optimizes the identical REML
    objective (assert_reml_objective_identity proves the identity exactly) and is the fit the
    randomization refits under every relabeling -- so all three describe one model rather than
    reading an interval off one fit and a p-value off another. This fit exists to corroborate that
    one from a completely different implementation, and where the two optimizers disagree
    assert_collapse_equivalence scores both on the shared objective and reports which converged.
    Its contrast is carried in the permutation CSV beside every estimate.

    ** No statistical fallback is permitted here. ** fit_mixed_model answers a failed or
    degenerate mixed-model fit with mouse-clustered OLS -- correct for the lanes that documented
    it, and exactly wrong for this one: clustered OLS over tens of thousands of cell rows with 17
    clusters is the anti-conservative cell-level inference this whole lane is built against, and
    it would also silently drop the random-effect structure that is the point of the model. A
    fallback therefore RAISES rather than being reported under this function's name.

    ** The summary's own P>|z| column is asymptotic over CELLS and is not paper-facing. ** On
    these data those standard errors are roughly HALF the mouse-level ones; that is a property of
    a cell-level asymptotic interval, not a sharpened estimate, and the inference lives in the
    randomization CSV instead. Every file this fit is written into says so in its header.

    Returns dict(result, method, summary_text, formula, n_mice, n_cells, fe_names, group_main,
    interaction, omnibus, fit_seconds).
    """
    df = _hierarchical_frame(df_cell)
    n_mice = int(df['mouse'].nunique())
    n_cells = int(df.groupby(['mouse', 'cell'], observed=True).ngroups)
    header = (f'Unit of inference: mouse (n = {n_mice}); denominator df = n_mouse - 1 = '
              f'{n_mice - 1}. {n_cells} cells are MODELLED, never counted as treatment '
              f'replicates.\n'
              f'** The P>|z| column below is asymptotic over cell rows and is NOT '
              f'paper-facing. ** Inference is by mouse-label randomization; see the '
              f'permutation table.\n\n')
    t0 = time.perf_counter()
    result, method, text = fit_mixed_model(
        df, _hierarchical_formula(), group_col='mouse', extra_header=header,
        method=HIERARCHICAL_LMM_OPTIMIZER, re_formula='1',
        vc_formula=_hierarchical_vc_formula())
    fit_seconds = time.perf_counter() - t0
    if method != 'mixedlm':
        raise RuntimeError(
            f'fit_hierarchical_cell_model: fit_mixed_model fell back to {method!r} instead of the '
            f'specified mixed model. This lane permits NO statistical fallback -- clustered OLS '
            f'over {len(df)} cell rows with {n_mice} clusters is precisely the cell-level '
            f'inference it is built against, and it would drop the random-effect structure that '
            f'is the point of the model. Stopping so the failure is visible; the fit text was:\n'
            f'{text}')
    group_main, interaction = _coefficient_names(result)
    fe_names = list(result.fe_params.index)
    omnibus = joint_wald_test(result, list(interaction.values()), n_groups=n_mice)
    return {'result': result, 'method': method, 'summary_text': text,
            'formula': _hierarchical_formula(), 'n_mice': n_mice, 'n_cells': n_cells,
            'fe_names': fe_names, 'group_main': group_main, 'interaction': interaction,
            'omnibus': omnibus, 'fit_seconds': fit_seconds}


def hierarchical_model_info(collapsed, observed, df_cell):
    """The model's identity -- coefficient names, counts, omnibus -- read off the REPORTED fit.

    The lane needs to know which coefficient is which in order to build contrasts, and it used to
    learn that from the full statsmodels fit. It does not need to: the collapsed fit carries the
    same coefficients under the same names (asserted by assert_collapse_equivalence whenever the
    cross-check runs, and by construction otherwise, since _design_blocks builds the design through
    patsy from the same formula string). Taking it from here is what lets the expensive cross-check
    be optional.
    """
    group_main, interaction = _coefficient_names(observed['fit'])
    n_mice = int(df_cell['mouse'].nunique())
    return {
        'formula': _hierarchical_formula(),
        'n_mice': n_mice,
        'n_cells': int(df_cell.groupby(['mouse', 'cell'], observed=True).ngroups),
        'fe_names': list(observed['fit'].fe_params.index),
        'group_main': group_main,
        'interaction': interaction,
        'omnibus': joint_wald_test(observed['fit'], list(interaction.values()), n_groups=n_mice),
    }


class _CollapsedFit:
    """The two attributes ``linear_contrast_test`` / ``joint_wald_test`` read off a fitted model.

    A shim, so the collapsed fit goes through exactly the same contrast and omnibus code as every
    other model in the codebase instead of a parallel re-derivation of the same algebra.
    """

    def __init__(self, fe_names, beta, cov):
        self.fe_params = pd.Series(np.asarray(beta, dtype=float), index=list(fe_names))
        self._cov = np.asarray(cov, dtype=float)

    def cov_params(self):
        return self._cov


class CollapsedEpochModel:
    """The cell-level model above, re-expressed EXACTLY on 68 (mouse x epoch) cell means.

    WHY THIS EXISTS
    ---------------
    The randomization test refits the model under every relabeling. The full fit costs ~3-4
    minutes on this dataset, and the lane needs ~27,000 refits per signal -- 60 to 150 hours,
    which is not a computation anyone runs. This class does the same fit in about a millisecond.

    WHY IT IS EXACT, AND NOT A REDUCED STRUCTURE
    --------------------------------------------
    The design is COMPLETELY BALANCED WITHIN CELL: every cell contributes to every retained trial
    and every epoch, so the fixed-effect design matrix is constant within each (mouse, epoch)
    block of cells. Two consequences follow, and together they are the whole argument:

    1. Writing the model as

           y[m,c,e] = x[m,e] . beta + u[m] + s[m,e] + a[m,c] + eps[m,c,e]

       each mouse's data splits ORTHOGONALLY into the part that is constant across its cells (the
       four cell means) and the part that contrasts its cells against each other. The covariance
       is block-diagonal with respect to that split -- every random term is a cell-structure
       (J or I) crossed with an epoch-structure -- and the fixed-effect design lies ENTIRELY in
       the first block, because x[m,e] does not vary across cells. So the REML log-likelihood is
       a sum of two pieces:

         * a CELL-CONTRAST piece that does not involve beta at all, and reduces to exactly two
           chi-square terms per mouse: the spread of cell means about the mouse mean, with
           variance n_epochs*sigma2_cell + sigma2_resid on n[m]-1 df, and the two-way residual,
           with variance sigma2_resid on 3*(n[m]-1) df;
         * a CELL-MEAN piece in the 68 numbers ybar[m,e], with

               Var(ybar[m,.]) = (sigma2_mouse + sigma2_cell/n[m]) J + diag(sigma2_epoch)
                                + (sigma2_resid/n[m]) I

       Nothing about the cellular hierarchy is discarded. It reappears twice: as the two
       sufficient statistics of the first piece, which is what estimates sigma2_cell and
       sigma2_resid, and as the /n[m] terms in the second, which is how a mouse with 89 cells
       comes to carry less weight than one with 1,047.

    2. The four sufficient statistics of the cell-contrast piece -- two sums of squares and their
       two degrees of freedom, per mouse -- are INVARIANT under every mouse relabeling. Group is
       a between-mouse factor, so the fixed-effect design is constant within each mouse's cell
       block and no relabeling can move a within-block residual. They are therefore computed
       ONCE, in __init__, and reused across the whole null distribution.

    ALL SEVEN variance parameters and beta are refit by REML under every relabeling. This is a
    full refit of the same likelihood, not a plug-in of anything: the objective below IS the full
    model's REML objective, up to terms free of every parameter.

    ** The argument is asserted, not trusted. ** assert_collapse_equivalence() checks this class's
    coefficients and standard errors against the full statsmodels fit on the observed labeling,
    on every run, and RAISES on disagreement. If the balance assumption above were ever violated
    -- a future change admitting cells that are missing an epoch -- that assertion is what would
    catch it, and _require_balanced() catches it earlier and more legibly.
    """

    # Bounds on the log-variances handed to the optimizer. The lower bound is effectively zero: a
    # variance component that wants to be zero is a statement about the data (reported as such),
    # not a failure, and is exactly what the boundary of the parameter space means here.
    _LOG_VAR_BOUNDS = (np.log(1e-12), np.log(1e3))

    def __init__(self, df_cell):
        df = _hierarchical_frame(df_cell)
        _require_balanced(df)
        self.epochs = list(RESPONSE_EPOCHS)
        self.n_epochs = len(self.epochs)

        # ---- sufficient statistics of the CELL-CONTRAST piece, invariant under relabeling ----
        # The two-way (cell x epoch) additive decomposition inside each mouse. Neither sum of
        # squares involves the group factor at all, which is precisely why permuting group labels
        # cannot change them.
        cell_mean = df.groupby(['mouse', 'cell_uid'], observed=True)['index'].transform('mean')
        me_mean = df.groupby(['mouse', 'epoch'], observed=True)['index'].transform('mean')
        m_mean = df.groupby('mouse', observed=True)['index'].transform('mean')
        resid = df['index'] - cell_mean - me_mean + m_mean
        n_cells_by_mouse = df.groupby('mouse', observed=True)['cell_uid'].nunique()
        # SS_resid ~ sigma2_resid on 3*(n-1) df per mouse.
        self.ss_resid = float((resid ** 2).sum())
        self.df_resid = float(sum((n - 1) * (self.n_epochs - 1) for n in n_cells_by_mouse))
        # SS_cell ~ (n_epochs*sigma2_cell + sigma2_resid) on (n-1) df per mouse. The factor of
        # n_epochs is the orthonormal scaling of the epoch-constant direction, not a fudge.
        cm = df.groupby(['mouse', 'cell_uid'], observed=True)['index'].mean()
        self.ss_cell = float(self.n_epochs * sum(((g - g.mean()) ** 2).sum()
                                                 for _, g in cm.groupby('mouse', observed=True)))
        self.df_cell = float(sum(n - 1 for n in n_cells_by_mouse))
        if self.df_resid <= 0 or self.df_cell <= 0:
            raise RuntimeError('CollapsedEpochModel: no within-mouse degrees of freedom -- every '
                               'mouse would need at least two cells for the cell variance '
                               'component to be estimable.')
        # Method-of-moments values, used ONLY as the optimizer's starting point.
        self._mom_resid = self.ss_resid / self.df_resid
        self._mom_cell = max((self.ss_cell / self.df_cell - self._mom_resid) / self.n_epochs,
                             1e-8)

        # ---- the 68 sufficient statistics --------------------------------------------------
        means = (df.groupby(['mouse', 'group', 'epoch'], observed=True)['index']
                   .mean().reset_index(name='ybar'))
        means['epoch_pos'] = means['epoch'].astype(str).map({e: i for i, e in enumerate(self.epochs)})
        means = means.sort_values(['mouse', 'epoch_pos'])
        self.mice = list(dict.fromkeys(means['mouse'].tolist()))
        self.true_assignment = dict(zip(means['mouse'], means['group'].astype(str)))
        self.y = (means['ybar'].to_numpy(dtype=float)
                  .reshape(len(self.mice), self.n_epochs))
        self.n_cells = n_cells_by_mouse.reindex(self.mice).to_numpy(dtype=float)
        self.n_cells_total = int(n_cells_by_mouse.sum())

        # ---- the three possible design blocks, one per group -------------------------------
        # Under a relabeling only WHICH block a mouse gets changes; the blocks themselves never
        # do. They are built through patsy on the same formula as the full fit and their column
        # names are asserted against it, so the collapsed coefficient vector cannot silently
        # permute relative to the model whose names the contrasts are written in terms of.
        self.fe_names, self._block_by_group = _design_blocks(self.epochs)

    # theta = log of [sigma2_mouse, sigma2_epoch (one per epoch), sigma2_cell, sigma2_resid].
    def _unpack(self, theta):
        v = np.exp(np.asarray(theta, dtype=float))
        return float(v[0]), v[1:1 + self.n_epochs], float(v[-2]), float(v[-1])

    def _covariances(self, s2_mouse, s2_epoch, s2_cell, s2_resid):
        """Stacked per-mouse 4x4 covariance of the CELL MEANS.

            Var(ybar[m,.]) = (sigma2_mouse + sigma2_cell/n[m]) J + diag(sigma2_epoch)
                             + (sigma2_resid/n[m]) I
        """
        j_scale = (s2_mouse + s2_cell / self.n_cells)[:, None, None]
        v = j_scale * np.ones((1, self.n_epochs, self.n_epochs))
        diag = s2_epoch[None, :] + (s2_resid / self.n_cells)[:, None]
        idx = np.arange(self.n_epochs)
        v[:, idx, idx] += diag
        return v

    def _gls(self, x, v):
        # Two batched matmuls rather than one three-operand einsum. Identical arithmetic; einsum
        # picks a contraction order without the intermediate here and was 4x slower, which is the
        # difference between a 10 ms and a 3 ms refit -- and this runs ~9,000 times per signal.
        v_inv = np.linalg.inv(v)
        vx = v_inv @ x                                    # (m, e, k)
        xtvx = np.einsum('mek,mel->kl', x, vx)
        xtvy = np.einsum('me,mek->k', self.y, vx)
        beta = np.linalg.solve(xtvx, xtvy)
        return beta, xtvx, v_inv, vx


    def _neg2_reml(self, theta, x):
        """The FULL model's -2 REML log-likelihood, up to terms free of every parameter.

        Two additive pieces, as the class docstring derives: the cell-contrast piece (which is
        where sigma2_cell and sigma2_resid are actually estimated from, and which involves neither
        beta nor the between-mouse variances) and the cell-mean piece.
        """
        s2_mouse, s2_epoch, s2_cell, s2_resid = self._unpack(theta)

        # ---- cell-contrast piece: two chi-squares per mouse, pooled over mice ----
        var_cell = self.n_epochs * s2_cell + s2_resid
        contrast = (self.df_cell * np.log(var_cell) + self.ss_cell / var_cell
                    + self.df_resid * np.log(s2_resid) + self.ss_resid / s2_resid)

        # ---- cell-mean piece ----
        v = self._covariances(s2_mouse, s2_epoch, s2_cell, s2_resid)
        sign, logdet = np.linalg.slogdet(v)
        if np.any(sign <= 0):
            return np.inf
        beta, xtvx, v_inv, _vx = self._gls(x, v)
        r = self.y - np.einsum('mij,j->mi', x, beta)
        quad = float(np.einsum('mi,mij,mj->', r, v_inv, r))
        sign_x, logdet_x = np.linalg.slogdet(xtvx)
        if sign_x <= 0:
            return np.inf
        return float(contrast + logdet.sum() + quad + logdet_x)

    def _neg2_reml_and_grad(self, theta, x):
        """``_neg2_reml`` and its ANALYTIC gradient with respect to the log-variances.

        ** Why this exists: it is the difference between a 33 ms and a 9 ms refit. ** L-BFGS-B
        with a finite-difference gradient spends 8 of every ~12 function evaluations just
        estimating the gradient of 7 parameters, and the randomization does ~9,400 refits per
        signal. Supplying the gradient cuts the evaluation count several-fold and changes nothing
        about the optimum, the model or the statistic -- it is the same objective, differentiated
        rather than probed.

        The derivatives are cheap because every dV/dphi in this model is J, a single diagonal
        entry, or I: each trace the gradient needs collapses onto quantities ``_gls`` has already
        formed (``v_inv``, ``vx = V^-1 X``, and ``s = V^-1 r``). beta is the GLS minimiser of the
        quadratic form, so by the envelope theorem it is held fixed while differentiating.

        Validated against scipy's own finite-difference gradient in
        ``verify_hierarchical_epoch_synthetic``; a wrong gradient here would show up as a worse
        optimum, so it is checked rather than trusted.
        """
        s2_mouse, s2_epoch, s2_cell, s2_resid = self._unpack(theta)
        n_e = self.n_epochs

        var_cell = n_e * s2_cell + s2_resid
        contrast = (self.df_cell * np.log(var_cell) + self.ss_cell / var_cell
                    + self.df_resid * np.log(s2_resid) + self.ss_resid / s2_resid)
        # d(contrast)/d(phi), for the two within-cell variances only.
        d_var_cell = self.df_cell / var_cell - self.ss_cell / var_cell ** 2
        dc_dcell = n_e * d_var_cell
        dc_dresid = d_var_cell + (self.df_resid / s2_resid - self.ss_resid / s2_resid ** 2)

        v = self._covariances(s2_mouse, s2_epoch, s2_cell, s2_resid)
        sign, logdet = np.linalg.slogdet(v)
        if np.any(sign <= 0):
            return np.inf, np.zeros_like(np.asarray(theta, dtype=float))
        beta, xtvx, a, vx = self._gls(x, v)
        r = self.y - np.einsum('mij,j->mi', x, beta)
        s = np.einsum('mij,mj->mi', a, r)                      # V^-1 r, per mouse
        quad = float(np.einsum('mi,mi->', r, s))
        sign_x, logdet_x = np.linalg.slogdet(xtvx)
        if sign_x <= 0:
            return np.inf, np.zeros_like(np.asarray(theta, dtype=float))
        f = float(contrast + logdet.sum() + quad + logdet_x)

        m_inv = np.linalg.inv(xtvx)
        inv_n = 1.0 / self.n_cells                             # (m,)
        s_sum = s.sum(axis=1)                                  # 1' V^-1 r
        u = vx.sum(axis=1)                                     # (m, k) = X' V^-1 1

        def _term(tr, rgr, xgx):
            """tr(V^-1 G) - r'V^-1 G V^-1 r - tr(M^-1 X'V^-1 G V^-1 X), summed over mice."""
            return float(tr.sum() - rgr.sum() - np.einsum('kl,lk->', m_inv, xgx))

        grad_phi = np.empty(n_e + 3)
        # sigma2_mouse: G = J
        grad_phi[0] = _term(a.sum(axis=(1, 2)), s_sum ** 2, np.einsum('mk,ml->kl', u, u))
        # sigma2_epoch[e]: G = e_e e_e'
        for e in range(n_e):
            grad_phi[1 + e] = _term(a[:, e, e], s[:, e] ** 2,
                                    np.einsum('mk,ml->kl', vx[:, e, :], vx[:, e, :]))
        # sigma2_cell: G = J / n_m
        grad_phi[-2] = dc_dcell + _term(a.sum(axis=(1, 2)) * inv_n, s_sum ** 2 * inv_n,
                                        np.einsum('m,mk,ml->kl', inv_n, u, u))
        # sigma2_resid: G = I / n_m
        grad_phi[-1] = dc_dresid + _term(np.trace(a, axis1=1, axis2=2) * inv_n,
                                         np.einsum('mi,mi->m', s, s) * inv_n,
                                         np.einsum('m,mek,mel->kl', inv_n, vx, vx))
        # chain rule: theta = log(phi)
        phi = np.concatenate([[s2_mouse], s2_epoch, [s2_cell, s2_resid]])
        return f, grad_phi * phi

    def design(self, assignment):
        """Stacked (n_mice, n_epochs, n_fixed) design under a mouse -> group mapping."""
        try:
            return np.stack([self._block_by_group[assignment[m]] for m in self.mice])
        except KeyError as exc:
            raise RuntimeError(f'CollapsedEpochModel.design: no design block for {exc} -- the '
                               f'relabeling names a group this model was not built for.') from exc

    def start_theta(self):
        """Starting variance parameters: method-of-moments where they are available, and a small
        positive value for the between-mouse terms, which the within-mouse statistics say nothing
        about."""
        return np.log(np.concatenate([[0.005], np.full(self.n_epochs, 0.005),
                                      [self._mom_cell, self._mom_resid]]))

    def fit(self, assignment=None, theta0=None):
        """REML fit under one labeling. Returns dict(fit, beta, cov, theta, neg2_reml, ...)."""
        assignment = self.true_assignment if assignment is None else assignment
        x = self.design(assignment)
        theta0 = self.start_theta() if theta0 is None else np.asarray(theta0, dtype=float)
        res = minimize(self._neg2_reml_and_grad, x0=theta0, args=(x,), jac=True,
                       method='L-BFGS-B',
                       bounds=[self._LOG_VAR_BOUNDS] * (self.n_epochs + 3))
        s2_mouse, s2_epoch, s2_cell, s2_resid = self._unpack(res.x)
        v = self._covariances(s2_mouse, s2_epoch, s2_cell, s2_resid)
        beta, xtvx, _v_inv, _vx = self._gls(x, v)
        cov = np.linalg.inv(xtvx)
        return {'fit': _CollapsedFit(self.fe_names, beta, cov), 'beta': beta, 'cov': cov,
                'theta': res.x, 'neg2_reml': float(res.fun),
                'sigma2_mouse': s2_mouse,
                'sigma2_epoch': dict(zip(self.epochs, s2_epoch)),
                'sigma2_cell': s2_cell, 'sigma2_resid': s2_resid,
                'converged': bool(res.success)}


def _require_balanced(df):
    """Hard-fail unless every cell carries every response epoch exactly once.

    The collapse is exact BECAUSE of this balance (see CollapsedEpochModel), so it is checked
    rather than assumed. build_modulation_table guarantees it today -- the index is a window mean
    of a continuous trace, so there is no dropout -- but a future change that admitted a partial
    cell would silently invalidate the algebra, and this is the cheap place to catch it.
    """
    counts = df.groupby(['mouse', 'cell'], observed=True)['epoch'].nunique()
    bad = counts[counts != len(RESPONSE_EPOCHS)]
    if len(bad):
        raise RuntimeError(
            f'epoch_modulation hierarchical lane: {len(bad)} cells do not carry all '
            f'{len(RESPONSE_EPOCHS)} response epochs (e.g. {bad.index[:5].tolist()}). The '
            f'collapsed representation is exact only for a design that is balanced within cell, '
            f'so an unbalanced table must not be fit through it.')
    dup = df.groupby(['mouse', 'cell', 'epoch'], observed=True).size()
    if (dup != 1).any():
        raise RuntimeError('epoch_modulation hierarchical lane: the cell-level table has more '
                           'than one row per (mouse, cell, epoch). Pass summarize_cells() output, '
                           'which has already collapsed the trial dimension.')


def _design_blocks(epochs):
    """(fixed-effect names, {group: (n_epochs, n_fixed) design block}) for the collapsed model.

    Built by running patsy over the SAME formula string the full fit uses, on a tiny 3-group x
    n-epoch frame, so the column set and its order are whatever statsmodels itself would produce
    rather than a hand-rolled dummy coding that could drift from it. The caller asserts the names
    against the fitted model.
    """
    grid = pd.DataFrame([{'group': g, 'epoch': e} for g in GROUP_ORDER for e in epochs])
    grid = _code_group_epoch(grid)
    x = patsy.dmatrix(_hierarchical_formula().split('~', 1)[1].strip(), grid,
                      return_type='dataframe')
    names = list(x.columns)
    blocks = {}
    for g in GROUP_ORDER:
        mask = (grid['group'].astype(str) == g).to_numpy()
        block = np.asarray(x[mask], dtype=float)
        order = [list(grid.loc[mask, 'epoch'].astype(str)).index(e) for e in epochs]
        blocks[g] = block[order]
    return names, blocks


def _statsmodels_theta(fit, collapsed):
    """The full fit's variance estimates, as CollapsedEpochModel's log-variance vector.

    Lets the two fits be scored against each other on one objective. statsmodels reports
    ``cov_re``, ``vcomp`` and ``scale`` in the response's own units, so no rescaling is involved;
    the component names are read off the fitted model rather than reconstructed.
    """
    result = fit['result']
    vc = dict(zip(list(result.model.exog_vc.names), np.asarray(result.vcomp, dtype=float)))
    missing = [f'epoch_{e}' for e in RESPONSE_EPOCHS if f'epoch_{e}' not in vc] + \
              ([] if 'cell' in vc else ['cell'])
    if missing:
        raise RuntimeError(f'_statsmodels_theta: variance components {missing} are absent from '
                           f'the fitted model, which has {sorted(vc)}.')
    values = ([float(np.asarray(result.cov_re)[0, 0])]
              + [vc[f'epoch_{e}'] for e in RESPONSE_EPOCHS]
              + [vc['cell'], float(result.scale)])
    return np.log(np.maximum(np.asarray(values, dtype=float), 1e-12))


def _brute_force_neg2_reml(df_cell, theta, fe_names):
    """-2 REML computed DIRECTLY from the cell-level rows, with no collapse and no shortcut.

    A deliberately naive reference implementation: it forms each mouse's full (n_cells*n_epochs)
    covariance matrix and takes its log-determinant. That is why it is only ever used on a
    subsample -- the point is that it is obviously correct, not that it is fast.

    ``theta`` is the same log-variance vector CollapsedEpochModel uses.
    """
    epochs = list(RESPONSE_EPOCHS)
    v = np.exp(np.asarray(theta, dtype=float))
    s2_mouse, s2_epoch, s2_cell, s2_resid = v[0], v[1:1 + len(epochs)], v[-2], v[-1]

    df = _code_group_epoch(df_cell)
    x_all = patsy.dmatrix(_hierarchical_formula().split('~', 1)[1].strip(), df,
                          return_type='dataframe')
    if list(x_all.columns) != list(fe_names):
        raise RuntimeError(f'_brute_force_neg2_reml: design columns {list(x_all.columns)} do not '
                           f'match the model coefficients {list(fe_names)}.')
    x_all = np.asarray(x_all, dtype=float)
    y_all = df['index'].to_numpy(dtype=float)
    cell_code = pd.factorize(df['mouse'].astype(str) + '_' + df['cell'].astype(str))[0]
    epoch_code = df['epoch'].astype(str).map({e: i for i, e in enumerate(epochs)}).to_numpy()

    xtvx = np.zeros((len(fe_names), len(fe_names)))
    xtvy = np.zeros(len(fe_names))
    logdet_total = 0.0
    blocks = []
    for mouse, idx in df.groupby('mouse', observed=True).indices.items():
        c, e = cell_code[idx], epoch_code[idx]
        cov = (s2_mouse
               + np.where(e[:, None] == e[None, :], s2_epoch[e][:, None], 0.0)
               + np.where(c[:, None] == c[None, :], s2_cell, 0.0)
               + np.eye(len(idx)) * s2_resid)
        cov_inv = np.linalg.inv(cov)
        logdet_total += np.linalg.slogdet(cov)[1]
        xm, ym = x_all[idx], y_all[idx]
        xtvx += xm.T @ cov_inv @ xm
        xtvy += xm.T @ cov_inv @ ym
        blocks.append((idx, cov_inv))
    beta = np.linalg.solve(xtvx, xtvy)
    quad = 0.0
    for idx, cov_inv in blocks:
        r = y_all[idx] - x_all[idx] @ beta
        quad += r @ cov_inv @ r
    return float(logdet_total + quad + np.linalg.slogdet(xtvx)[1]), beta


def assert_reml_objective_identity(collapsed, df_cell, n_cells_per_mouse=12, n_probes=4,
                                   seed=HIERARCHICAL_SEED, rtol=1e-6):
    """Hard-fail unless the collapsed objective IS the full model's REML objective.

    ** This is the load-bearing check of the whole approach, and it is exact. ** The claim
    CollapsedEpochModel rests on is that its objective equals the full cell-level REML objective
    up to terms free of every parameter. That is checked directly here: both objectives are
    evaluated at several unrelated parameter vectors and their DIFFERENCE must be the same
    constant at all of them, and the GLS coefficients must coincide.

    It is checked on a random SUBSAMPLE of cells per mouse, because the reference implementation
    forms each mouse's full covariance matrix explicitly. The identity is an algebraic property of
    the model structure and the within-cell balance, not of the number of cells, and the subsample
    preserves the structure that matters: the same 17 mice, the same four epochs, unequal cell
    counts per mouse, and the same fixed-effect design. What it does NOT re-verify is the exact
    cell counts of the real fit, which is what assert_collapse_equivalence covers.

    Returns a report string.
    """
    rng = np.random.default_rng(seed)
    keep = []
    for _mouse, sub in df_cell.groupby('mouse', observed=True):
        cells = sub['cell'].unique()
        n = min(n_cells_per_mouse, len(cells))
        keep.append(pd.Series(rng.choice(cells, size=n, replace=False)).to_frame('cell')
                    .assign(mouse=_mouse))
    keep = pd.concat(keep, ignore_index=True)
    df_sub = df_cell.merge(keep, on=['mouse', 'cell'], how='inner')
    sub_model = CollapsedEpochModel(df_sub)

    diffs, max_dbeta = [], 0.0
    for i in range(n_probes):
        theta = np.log(rng.uniform(0.002, 0.5, size=collapsed.n_epochs + 3))
        obj_collapsed = sub_model._neg2_reml(theta, sub_model.design(sub_model.true_assignment))
        obj_full, beta_full = _brute_force_neg2_reml(df_sub, theta, sub_model.fe_names)
        v = sub_model._unpack(theta)
        beta_col, _xtvx, _vi, _vx = sub_model._gls(sub_model.design(sub_model.true_assignment),
                                              sub_model._covariances(*v))
        diffs.append(obj_full - obj_collapsed)
        max_dbeta = max(max_dbeta, float(np.max(np.abs(beta_full - beta_col))))
    spread = float(np.max(diffs) - np.min(diffs))
    scale = max(abs(float(np.mean(diffs))), 1.0)
    if spread / scale > rtol or max_dbeta > 1e-8:
        raise RuntimeError(
            f'assert_reml_objective_identity FAILED: the collapsed objective is not the full REML '
            f'objective up to a constant. Across {n_probes} parameter vectors the offset varied by '
            f'{spread:.6g} (relative {spread / scale:.3g}, tolerance {rtol:g}) and the GLS '
            f'coefficients differed by up to {max_dbeta:.3g}. The collapse is exact only for a '
            f'design balanced within cell and for the random-effect structure the full fit was '
            f'given; refusing to permute a statistic that is not the model being reported.')
    return (f'REML objective identity CHECKED and PASSED (exact):\n'
            f'  probes                  = {n_probes} unrelated parameter vectors\n'
            f'  subsample               = {n_cells_per_mouse} cells per mouse '
            f'({len(df_sub)} rows), full covariance formed explicitly\n'
            f'  offset between the two objectives varied by {spread:.3g} across probes '
            f'(relative {spread / scale:.3g}, tolerance {rtol:g})\n'
            f'  max |d beta| between collapsed GLS and full GLS = {max_dbeta:.3g}\n')


def assert_collapse_equivalence(fit, collapsed_result, collapsed=None,
                                beta_atol=HIERARCHICAL_BETA_ATOL,
                                se_rtol=HIERARCHICAL_SE_RTOL):
    """Check the collapsed fit against the full cell-level fit on the OBSERVED labeling.

    ** What this can and cannot establish. ** assert_reml_objective_identity is what proves the
    two objectives are the same function; this checks that two different optimizers landed in the
    same place on it. When they do not, one of them stopped short, and which one is a question
    with an answer: the objectives are evaluated at BOTH parameter vectors and compared. A
    disagreement is accepted only when the collapsed fit attains a strictly better value of the
    shared objective -- i.e. statsmodels under-converged along the flat between-mouse directions,
    which is common with 17 groups -- and it is then REPORTED in the output rather than hidden.
    Any other disagreement raises.

    ``collapsed`` is the CollapsedEpochModel; without it the objective comparison cannot be made
    and a disagreement can only raise.

    Returns a report string for the output file.
    """
    names_full = list(fit['fe_names'])
    names_col = list(collapsed_result['fit'].fe_params.index)
    if names_full != names_col:
        raise RuntimeError(
            f'assert_collapse_equivalence: the collapsed design has different coefficients from '
            f'the fitted model, or the same ones in a different order.\n  full:      '
            f'{names_full}\n  collapsed: {names_col}')
    beta_full = np.asarray(fit['result'].fe_params, dtype=float)
    beta_col = np.asarray(collapsed_result['beta'], dtype=float)
    n_fixed = len(names_full)
    cov_full = np.asarray(fit['result'].cov_params())[:n_fixed, :n_fixed]
    se_full = np.sqrt(np.diag(cov_full))
    se_col = np.sqrt(np.diag(collapsed_result['cov']))

    d_beta = np.max(np.abs(beta_full - beta_col))
    d_se = np.max(np.abs(se_full - se_col) / np.maximum(se_full, 1e-12))
    convergence_note = ''
    if not (d_beta <= beta_atol and d_se <= se_rtol):
        detail = '\n'.join(
            f'    {n[:64]:<64} full {bf:+.6f} (se {sf:.6f})  collapsed {bc:+.6f} (se {sc:.6f})'
            for n, bf, bc, sf, sc in zip(names_full, beta_full, beta_col, se_full, se_col))
        obj_sm = obj_col = None
        if collapsed is not None:
            x = collapsed.design(collapsed.true_assignment)
            obj_sm = collapsed._neg2_reml(_statsmodels_theta(fit, collapsed), x)
            obj_col = collapsed._neg2_reml(collapsed_result['theta'], x)
        if obj_sm is None or not (obj_col < obj_sm):
            raise RuntimeError(
                f'assert_collapse_equivalence: the 68-row collapsed fit does NOT reproduce the '
                f'full cell-level fit (max |dbeta| = {d_beta:.3g}, tolerance {beta_atol:g}; max '
                f'relative dSE = {d_se:.3g}, tolerance {se_rtol:g}), and this is NOT attributable '
                f'to statsmodels stopping short: on the shared objective it scores '
                f'{"(not evaluated -- pass the CollapsedEpochModel to evaluate it)" if obj_sm is None else f"{obj_sm:.6f} against the collapsed fit s {obj_col:.6f}"}. '
                f'Refusing to permute a statistic that is not the model being '
                f'reported.\n{detail}')
        convergence_note = (
            f'  NOTE: the two fits differ by more than the tolerance (max |d beta| = '
            f'{d_beta:.3g}, max relative |d SE| = {d_se:.3g}).\n'
            f'  This is statsmodels stopping short, not a discrepancy in the model: on the SHARED '
            f'REML objective (whose\n'
            f'  identity is established exactly by assert_reml_objective_identity) the collapsed '
            f'fit scores {obj_col:.6f}\n'
            f'  against statsmodels\' {obj_sm:.6f} -- lower is better, and the difference lies '
            f'along the flat\n'
            f'  between-mouse variance directions that 17 groups leave poorly determined. The '
            f'collapsed fit is the\n'
            f'  better optimum of the same likelihood, and it is the one the reported estimates, '
            f'intervals and\n'
            f'  randomization p-values all come from -- so all three describe one fit. The full '
            f'statsmodels fit is\n'
            f'  retained as the independent cross-check it is, and its own summary is written out '
            f'beside this file.\n')
    return (convergence_note
            + f'Collapse equivalence CHECKED on the observed labeling:\n'
            f'  max |d beta|            = {d_beta:.3g}  (tolerance {beta_atol:g}, absolute)\n'
            f'  max relative |d SE|     = {d_se:.3g}  (tolerance {se_rtol:g})\n'
            f'  full fit                = {len(names_full)} fixed effects over cell x epoch rows, '
            f'{fit["fit_seconds"]:.0f} s\n'
            f'  collapsed fit           = the same {len(names_full)} fixed effects over 68 '
            f'(mouse x epoch) cell means\n')


def _contrast_weights(fit, group, epoch):
    """{coefficient: weight} for the group-versus-reference contrast AT one epoch.

    At the reference epoch the contrast IS the group coefficient; at every other epoch it is that
    coefficient plus that epoch's interaction term, combined through the fitted covariance. Same
    construction as fit_mouse_level_model, so the two lanes' contrasts mean the same thing.
    """
    weights = {fit['group_main'][group]: 1.0}
    if epoch != REFERENCE_EPOCH:
        weights[fit['interaction'][(group, epoch)]] = 1.0
    return weights


def _pair_weights(fit, group_a, group_b, epoch):
    """{coefficient: weight} for one pairwise comparison at one epoch.

    Against the reference group it is a single simple effect; between the two treatment groups it
    is their DIFFERENCE, which needs the coefficients' covariance and is therefore a linear
    contrast, not a subtraction of two published standard errors.
    """
    if group_b == REFERENCE_GROUP:
        return _contrast_weights(fit, group_a, epoch)
    if group_a == REFERENCE_GROUP:
        return {k: -v for k, v in _contrast_weights(fit, group_b, epoch).items()}
    weights = {k: v for k, v in _contrast_weights(fit, group_a, epoch).items()}
    for k, v in _contrast_weights(fit, group_b, epoch).items():
        weights[k] = weights.get(k, 0.0) - v
    return weights


def _make_collapsed_stat(collapsed, true_assignment, theta0, weights=None, kind='contrast',
                         interaction_names=None, n_mice=None):
    """stat_fn factory for mouse_label_permutation_test whose statistic is the REFIT MODEL.

    ** The full three-group model is refit on every draw -- never a two-group subset. ** When
    mouse_label_permutation_test is called with restrict_to_groups=(a, b) it hands this closure a
    mapping covering only those two groups' mice; the third group's mice keep their true labels
    and stay in every fit. The restriction is on the RANDOMIZATION, not on the data.

    ``theta0`` warm-starts the variance parameters from the observed fit. That is an optimizer
    start and nothing else -- the REML optimum is what it is, and the returned statistic is the
    converged one. It matters only because it cuts the refit cost several-fold.

    kind='contrast' returns the linear combination given by ``weights``; kind='wald' returns the
    joint Wald F over ``interaction_names`` (the omnibus statistic).
    """
    def stat_fn(mouse_to_group):
        assignment = dict(true_assignment)
        assignment.update(mouse_to_group)
        out = collapsed.fit(assignment, theta0=theta0)
        if kind == 'wald':
            return float(joint_wald_test(out['fit'], list(interaction_names),
                                         n_groups=n_mice)['F'])
        return float(sum(w * float(out['fit'].fe_params[n]) for n, w in weights.items()))
    return stat_fn


def hierarchical_epoch_permutation(df_cell, fit, collapsed, mice_per_group, observed,
                                   n_perm_omnibus=HIERARCHICAL_N_PERM_OMNIBUS,
                                   seed=HIERARCHICAL_SEED, alpha=0.05, verbose=True,
                                   full_fit=None):
    """Mouse-label randomization inference for the cell-level epoch-modulation model.

    ** PAIRWISE (primary): EXACT randomization of the model contrast. ** For each epoch and each
    comparison the statistic is the fitted three-group model's contrast at that epoch, and
    exchangeability is restricted to the two groups being compared. With a 5/6/6 cohort those
    restricted spaces hold C(11,5) = 462, C(12,6) = 924 and C(11,5) = 462 relabelings, so the
    model is refit for EVERY one of them and the p-values are exact -- no Monte Carlo error, and
    no +1/+1 correction, since the observed labeling is itself one of the enumerated draws.

    ** Holm follows the mouse-level lane, not the recall lane. ** Within each epoch the two
    treatment-versus-control comparisons are corrected together: four independent two-member
    families. hM3D-vs-hM4D is computed and tabulated but is in NO family, exactly as on panel L.

    ** OMNIBUS: Monte Carlo randomization of a 6-df model-based statistic. ** The joint Wald test
    of all six interaction coefficients of the same fit, permuted under the GLOBAL null. Its H0 is
    that those coefficients are jointly zero -- that the group effect is uniform across epochs. It
    tests THOSE COEFFICIENTS, not the shape, spread or tails of the cellular distribution.

    ** DESIGN-BASED SENSITIVITY, always run, in no family. ** The same restricted enumeration
    applied to the difference of group means of the per-mouse mean index (make_contrast_stat,
    weight='mouse'). Model-free, and therefore a genuine check -- but it collapses each animal to
    one number before testing, carries no cellular hierarchy, and is not the primary. No numerical
    equivalence with the model-based statistic is claimed: GLS weighting and equal-mouse weighting
    coincide only at equal cluster sizes, and cells per mouse range from 89 to 1,047 here.

    Returns one tidy DataFrame: an ``omnibus`` row, and per (epoch, comparison) a
    ``pairwise_primary`` row and a ``pairwise_sensitivity`` row.
    """
    n_mice = fit['n_mice']
    present = {g: [m for m in mice_per_group.get(g, []) if m in set(df_cell['mouse'])]
               for g in GROUP_ORDER}
    present = {g: ms for g, ms in present.items() if ms}
    true_assignment = {m: g for g, ms in present.items() for m in ms}
    if set(true_assignment) != set(df_cell['mouse']):
        raise RuntimeError('hierarchical_epoch_permutation: the cell-level table and '
                           'mice_per_group cover different animals.')
    theta0 = observed['theta']
    rows = []

    # ---- omnibus: 6-df model-based statistic, global null, Monte Carlo ---------------------
    interaction_names = list(fit['interaction'].values())
    # The observed omnibus F, off the same fit the null distribution is built from.
    observed_omnibus = joint_wald_test(observed['fit'], interaction_names, n_groups=n_mice)
    omnibus_stat = _make_collapsed_stat(collapsed, true_assignment, theta0, kind='wald',
                                        interaction_names=interaction_names, n_mice=n_mice)
    if verbose:
        print(f'[epoch-modulation/hier] omnibus: {n_perm_omnibus} Monte Carlo refits '
              f'(global null, seed {seed}); the global space holds '
              f'{n_distinct_relabelings([g for g, ms in present.items() for _ in ms])} '
              f'relabelings, so this cannot be exact...', flush=True)
    _t = time.perf_counter()
    omnibus_perm = mouse_label_permutation_test(
        omnibus_stat, present, n_perm=n_perm_omnibus, seed=seed,
        progress_label='omnibus', progress_every=max(1, n_perm_omnibus // 10))
    if verbose:
        print(f'[epoch-modulation/hier] omnibus done in '
              f'{(time.perf_counter() - _t) / 60:.1f} min: P = '
              f'{omnibus_perm["p_two_sided"]:.4g}', flush=True)
    rows.append({
        'block': 'omnibus', 'epoch': 'all', 'group_a': 'all', 'group_b': 'all',
        'statistic_kind': 'joint Wald F on all six group x epoch interaction coefficients',
        'estimate': float(observed_omnibus['F']), 'ci_low': np.nan, 'ci_high': np.nan,
        'se': np.nan, 'p_raw': float(omnibus_perm['p_two_sided']), 'p_holm_epoch': np.nan,
        'in_holm_family': False, 'estimate_mouse_weighted': np.nan,
        'p_raw_design_based': np.nan, 'null_type': 'global', 'exact': False,
        'n_relabelings': np.nan, 'n_perm': int(omnibus_perm['n_perm']), 'seed': seed,
        'n_mice_a': n_mice, 'n_mice_b': n_mice, 'n_cells_a': fit['n_cells'],
        'n_cells_b': fit['n_cells'],
        'asymptotic_p_not_for_inference': float(observed_omnibus['p']),
        'estimate_full_fit_crosscheck': (float(full_fit['omnibus']['F'])
                                         if full_fit is not None else np.nan),
        'se_full_fit_crosscheck': np.nan,
    })

    # ---- pairwise -------------------------------------------------------------------------
    for epoch in RESPONSE_EPOCHS:
        slice_epoch = df_cell[df_cell['epoch'] == epoch]
        per_epoch = []
        for group_a, group_b in HIERARCHICAL_PAIRS:
            if group_a not in present or group_b not in present:
                raise RuntimeError(f'hierarchical_epoch_permutation: {group_a}/{group_b} is not '
                                   f'present in this cohort; the declared comparison set cannot '
                                   f'be computed.')
            weights = _pair_weights(fit, group_a, group_b, epoch)
            # ** The estimate, its interval and its p-value all come from ONE fit. ** That fit is
            # the collapsed one: it is the same model and the same likelihood as the full
            # statsmodels fit (assert_reml_objective_identity establishes the identity exactly),
            # it is the fit the randomization refits under every relabeling, and where the two
            # optimizers disagree it is the better optimum (assert_collapse_equivalence checks
            # which, and says so in the output). Reading the estimate off one fit and the p-value
            # off another would let a published interval describe a different fit from its own
            # test.
            contrast = linear_contrast_test(observed['fit'], weights, n_groups=n_mice, alpha=alpha)
            contrast_full = (linear_contrast_test(full_fit['result'], weights, n_groups=n_mice,
                                                  alpha=alpha)
                             if full_fit is not None else None)
            stat = _make_collapsed_stat(collapsed, true_assignment, theta0, weights=weights)
            expected = n_distinct_relabelings([group_a] * len(present[group_a])
                                              + [group_b] * len(present[group_b]))
            if verbose:
                print(f'[epoch-modulation/hier] {epoch} {group_a} vs {group_b}: EXACT '
                      f'enumeration, {expected} refits...', flush=True)
            perm = mouse_label_permutation_test(stat, present, seed=seed,
                                                restrict_to_groups=(group_a, group_b), exact=True)
            if int(perm['n_relabelings']) != expected:
                raise RuntimeError(
                    f'hierarchical_epoch_permutation: {epoch} {group_a} vs {group_b} enumerated '
                    f'{perm["n_relabelings"]} relabelings but the multinomial coefficient for '
                    f'{len(present[group_a])}/{len(present[group_b])} mice is {expected}.')
            design_stat = make_contrast_stat(slice_epoch, 'index', group_a, group_b,
                                             weight='mouse')
            design_perm = mouse_label_permutation_test(
                design_stat, present, seed=seed, restrict_to_groups=(group_a, group_b), exact=True)
            in_family = (group_a, group_b) in HIERARCHICAL_FAMILY_PAIRS
            row = {
                'block': 'pairwise_primary', 'epoch': epoch,
                'group_a': group_a, 'group_b': group_b,
                'statistic_kind': 'cell-level model contrast ' + ' + '.join(
                    f'{w:+g}*{n}' for n, w in weights.items()),
                'estimate': contrast['estimate'], 'ci_low': contrast['ci_low'],
                'ci_high': contrast['ci_high'], 'se': contrast['se'],
                'p_raw': float(perm['p_two_sided']), 'p_holm_epoch': np.nan,
                'in_holm_family': in_family, 'estimate_mouse_weighted': np.nan,
                'p_raw_design_based': np.nan,
                'null_type': f'pairwise ({group_a}, {group_b})', 'exact': True,
                'n_relabelings': int(perm['n_relabelings']), 'n_perm': int(perm['n_perm']),
                'seed': seed, 'n_mice_a': len(present[group_a]), 'n_mice_b': len(present[group_b]),
                'n_cells_a': int((slice_epoch['group'] == group_a).sum()),
                'n_cells_b': int((slice_epoch['group'] == group_b).sum()),
                'asymptotic_p_not_for_inference': contrast['p'],
                # The same contrast off the independent statsmodels fit when that optional
                # cross-check was run; NaN when it was not. Never an input to anything.
                'estimate_full_fit_crosscheck': (contrast_full['estimate'] if contrast_full
                                                 else np.nan),
                'se_full_fit_crosscheck': contrast_full['se'] if contrast_full else np.nan,
            }
            per_epoch.append(row)
            rows.append(row)
            rows.append({
                'block': 'pairwise_sensitivity', 'epoch': epoch,
                'group_a': group_a, 'group_b': group_b,
                'statistic_kind': 'design-based: difference of group means of the per-mouse mean '
                                  'index at this epoch (model-free)',
                'estimate': np.nan, 'ci_low': np.nan, 'ci_high': np.nan, 'se': np.nan,
                'p_raw': np.nan, 'p_holm_epoch': np.nan, 'in_holm_family': False,
                'estimate_mouse_weighted': float(design_perm['observed']),
                'p_raw_design_based': float(design_perm['p_two_sided']),
                'null_type': f'pairwise ({group_a}, {group_b})', 'exact': True,
                'n_relabelings': int(design_perm['n_relabelings']),
                'n_perm': int(design_perm['n_perm']), 'seed': seed,
                'n_mice_a': len(present[group_a]), 'n_mice_b': len(present[group_b]),
                'n_cells_a': int((slice_epoch['group'] == group_a).sum()),
                'n_cells_b': int((slice_epoch['group'] == group_b).sum()),
                'asymptotic_p_not_for_inference': np.nan,
                'estimate_full_fit_crosscheck': np.nan,
                'se_full_fit_crosscheck': np.nan,
            })
        # Holm across EXACTLY the two treatment-versus-control comparisons within this epoch.
        family = [r for r in per_epoch if r['in_holm_family']]
        _reject, padj = holm_correct([r['p_raw'] for r in family], alpha=alpha)
        for r, pa in zip(family, padj):
            r['p_holm_epoch'] = float(pa)

    return pd.DataFrame(rows)


def format_hierarchical_report(perm_df, fit, collapsed, observed, equivalence_text, signal):
    """Human-readable block for the lane's stats file."""
    lines = [
        f'Hierarchical cell-level companion analysis ({signal})',
        '=' * 72,
        'COMPANION / SENSITIVITY ANALYSIS. It replaces nothing: the mouse-level model, its',
        'contrasts, its Holm families, its omnibus and every existing figure and table are',
        'unchanged and remain the paper-facing analysis.',
        '',
        f'Model: {fit["formula"]}',
        f'       + (1|mouse) + (0 + epoch||mouse) + (1|mouse:cell)',
        f'Rows: {fit["n_cells"]} cells x {len(RESPONSE_EPOCHS)} epochs; '
        f'mice: {fit["n_mice"]}.',
        '',
        f'** n = {fit["n_mice"]} MICE is the sample size. ** That is the number of independently',
        'assigned experimental units and it does not change because cells were modelled. Cell',
        'counts are descriptive. Every p-value below comes from permuting the mouse labels with',
        'each cell fixed to its own animal; the fitted model\'s own asymptotic P>|z| is written',
        'out in the CSV under a column named so it cannot be mistaken for inference.',
        '',
        'THE ANIMAL-LEVEL EPOCH RANDOM EFFECT IS LOAD-BEARING. Without it, between-animal',
        'variation in the epoch effect has nowhere to go and cells become replicates for the',
        'epoch contrast. Fitted between-animal variances:',
        f'  mouse intercept       {observed["sigma2_mouse"]:.6f}',
    ]
    for e, v in observed['sigma2_epoch'].items():
        lines.append(f'  epoch {e:<15} {v:.6f}')
    lines += [
        f'  cell (within mouse)   {observed["sigma2_cell"]:.6f}',
        f'  residual              {observed["sigma2_resid"]:.6f}',
        '',
        'THE ESTIMAND IS UNCONDITIONAL. Every cell is defined in every window (the index is a',
        'window mean of a continuous trace), so unlike the recall lane\'s paired-amplitude',
        'companion there is no eligibility set and no conditional estimand to caveat.',
        '',
        equivalence_text,
        f'Randomization: pairwise tests are EXACT enumerations of the restricted spaces;',
        f'the omnibus is Monte Carlo at seed {HIERARCHICAL_SEED} over the global null.',
        '',
    ]
    om = perm_df[perm_df['block'] == 'omnibus'].iloc[0]
    lines += [
        f'Group x epoch omnibus (all 6 interaction coefficients):',
        f'  observed F = {om["estimate"]:.3f}; mouse-label randomization P = {om["p_raw"]:.4g} '
        f'({int(om["n_perm"])} Monte Carlo draws)',
        '  This establishes whether the group effect is NON-UNIFORM across epochs. It does NOT',
        '  identify which epochs differ, and the within-epoch comparisons below do not, on their',
        '  own, license an epoch-specificity claim.',
        '',
        'Within-epoch comparisons (differences in modulation index, analysed untransformed).',
        'Holm is applied across exactly the two treatment-versus-control comparisons WITHIN each',
        'epoch; hM3D vs hM4D is in NO family and is marked accordingly.',
    ]
    for epoch in RESPONSE_EPOCHS:
        lines.append(f'  {epoch}:')
        prim = perm_df[(perm_df['epoch'] == epoch) & (perm_df['block'] == 'pairwise_primary')]
        sens = perm_df[(perm_df['epoch'] == epoch) & (perm_df['block'] == 'pairwise_sensitivity')]
        for _, r in prim.iterrows():
            holm = (f'  P_holm = {r["p_holm_epoch"]:.4g}' if r['in_holm_family']
                    else '  [in no Holm family]')
            d = sens[(sens['group_a'] == r['group_a']) & (sens['group_b'] == r['group_b'])]
            design = (f'  |  design-based P = {float(d["p_raw_design_based"].iloc[0]):.4g}'
                      if len(d) else '')
            lines.append(
                f'    {GROUP_LABELS[r["group_a"]]:>3} vs {GROUP_LABELS[r["group_b"]]:<3}: '
                f'{r["estimate"]:+.4f} [{r["ci_low"]:+.4f}, {r["ci_high"]:+.4f}]  '
                f'exact P = {r["p_raw"]:.4g}  ({int(r["n_relabelings"])} relabelings)'
                f'{holm}{design}')
    lines += [
        '',
        'INTERVAL PROVENANCE. The estimate and its interval are the cell-level model contrast and',
        'its own Wald interval, on the animal-level df = n_mice - 1. The design-based column is a',
        'DIFFERENT estimator (the difference of group means of the per-mouse mean index) and its',
        'p-value is never attached to the model estimate. The randomization p-values carry no',
        'interval of their own.',
        '',
        'WHAT THIS LANE ADDS, AND WHAT IT DOES NOT. It retains within-mouse cellular coherence and',
        'weights each animal by how precisely its own cells estimate it. It buys no treatment-',
        'level precision -- that is floored by between-animal variance and no number of cells',
        'reduces it -- and it is not a test of the shape, spread or tails of the cellular',
        'distribution.',
    ]
    return '\n'.join(lines) + '\n'


def _hierarchical_panel_contrasts(perm_df):
    """The frame ``_model_contrast_stat_fn`` reads, built from the randomization results.

    Only the two treatment-versus-control comparisons appear, keyed by treatment group and epoch
    exactly as the mouse-level contrasts frame is, so panel L's bracket path is reused unchanged
    and the panel reports THIS lane's Holm-adjusted exact randomization p-values.
    """
    prim = perm_df[(perm_df['block'] == 'pairwise_primary') & perm_df['in_holm_family']]
    out = prim.rename(columns={'group_a': 'group'})[['epoch', 'group', 'estimate', 'ci_low',
                                                     'ci_high', 'p_raw', 'p_holm_epoch']].copy()
    out['p'] = out['p_raw']
    out['p_holm'] = out['p_holm_epoch']
    return out


def write_hierarchical_vs_mouse_summary(save_dir, signal, fit, perm_df, mouse_contrasts,
                                        mouse_omnibus, observed, collapsed,
                                        filename='hierarchical_vs_mouse_level_summary.md'):
    """The two analyses side by side, every existing number READ from the frames the lane already
    produced rather than restated.

    ** The purpose is a question, not a verdict: ** does retaining the cellular hierarchy
    materially sharpen, weaken, or leave unchanged the conclusion? The smaller p-value is NOT
    automatically preferred, and this file says so.
    """
    om = perm_df[perm_df['block'] == 'omnibus'].iloc[0]
    lines = [
        f'# Hierarchical cell-level companion vs the mouse-level analysis — {signal}',
        '',
        '**This is a companion / sensitivity analysis. It replaces nothing.** The mouse-level '
        'model, its contrasts, its Holm families, its omnibus test and every existing figure and '
        'table are unchanged and remain the paper-facing analysis. Everything in the '
        '"hierarchical-cell" column below is additive evidence.',
        '',
        f'**n = {fit["n_mice"]} mice** in both analyses — the number of independently assigned '
        f'experimental units. It does not change because {fit["n_cells"]} cells were modelled; '
        f'cell counts are descriptive.',
        '',
        '## Omnibus — group × epoch',
        '',
        '| analysis | test | result |',
        '|---|---|---|',
        f'| mouse-level (authoritative) | joint Wald, 6 interaction coefficients | '
        f'F({mouse_omnibus["df1"]},{mouse_omnibus["df2"]}) = {mouse_omnibus["F"]:.3f}, '
        f'P = {mouse_omnibus["p"]:.4g} |',
        f'| hierarchical-cell (companion) | mouse-label randomization of the same 6-df joint '
        f'Wald statistic | F = {om["estimate"]:.3f}, P = {om["p_raw"]:.4g} '
        f'({int(om["n_perm"])} draws) |',
        '',
        '## Within-epoch comparisons',
        '',
        '| epoch | comparison | mouse-level estimate [95% CI] | mouse-level P | mouse-level '
        'P_holm | hierarchical estimate [95% CI] | exact randomization P | P_holm | design-based '
        'P |',
        '|---|---|---|---|---|---|---|---|---|',
    ]
    for epoch in RESPONSE_EPOCHS:
        for group_a, group_b in HIERARCHICAL_PAIRS:
            new = perm_df[(perm_df['block'] == 'pairwise_primary') & (perm_df['epoch'] == epoch)
                          & (perm_df['group_a'] == group_a) & (perm_df['group_b'] == group_b)]
            sens = perm_df[(perm_df['block'] == 'pairwise_sensitivity')
                           & (perm_df['epoch'] == epoch) & (perm_df['group_a'] == group_a)
                           & (perm_df['group_b'] == group_b)]
            n = new.iloc[0]
            old = mouse_contrasts[(mouse_contrasts['epoch'] == epoch)
                                  & (mouse_contrasts['group'] == group_a)]
            if group_b == REFERENCE_GROUP and len(old):
                o = old.iloc[0]
                old_cells = (f'{o["estimate"]:+.4f} [{o["ci_low"]:+.4f}, {o["ci_high"]:+.4f}] | '
                             f'{o["p"]:.4g} | {o["p_holm"]:.4g}')
            else:
                # hM3D vs hM4D is not computed by the mouse-level lane -- it is in no family
                # there either, and inventing it here would be a new, unstated comparison.
                old_cells = 'not computed | — | —'
            holm = f'{n["p_holm_epoch"]:.4g}' if n['in_holm_family'] else 'in no family'
            design = (f'{float(sens["p_raw_design_based"].iloc[0]):.4g}' if len(sens) else '—')
            lines.append(
                f'| {epoch} | {GROUP_LABELS[group_a]} vs {GROUP_LABELS[group_b]} | {old_cells} | '
                f'{n["estimate"]:+.4f} [{n["ci_low"]:+.4f}, {n["ci_high"]:+.4f}] | '
                f'{n["p_raw"]:.4g} | {holm} | {design} |')

    lines += [
        '',
        '## How to read the two columns',
        '',
        '**The smaller *P* is not automatically the better one.** The two analyses answer the '
        'same question with different amounts of structure, and where they agree the mouse-level '
        'number is the one to quote.',
        '',
        '**Why the estimates are so close.** They are close by construction, not by luck: the '
        'design is balanced within cell, so the cell-level model\'s fixed effects are a '
        'cell-count-weighted version of the mouse-level ones. What the hierarchical lane changes '
        'is the weighting (an animal with 1,047 cells estimates its own mean more precisely than '
        'one with 89) and the inference (exact mouse-label randomization rather than a t on '
        f'{fit["n_mice"] - 1} df).',
        '',
        '**What it does not do.** It buys no treatment-level precision — the contrast standard '
        'error is a between-animal quantity and no number of cells reduces it — and it is not a '
        'test of whether treatment reshapes the cell *distribution*. A manipulation that raised '
        'modulation in some cells and lowered it in others at constant mean would still be '
        'invisible.',
        '',
        '**Supportable wording**, where this lane strengthens a conclusion: *a hierarchical '
        'cell-level analysis, retaining within-mouse cellular variation while preserving '
        'mouse-level treatment assignment, provided additional evidence that …* — never '
        '`n = thousands of cells`.',
    ]
    ensure_dirs(save_dir)
    write_text(os.path.join(save_dir, filename), '\n'.join(lines) + '\n')


_HIERARCHICAL_COMPONENTS = (
    'cell-level table + balance check',
    'collapsed fit (the reported fit) + exact REML objective identity',
    'mouse-label randomization (exact within-epoch pairwise + MC omnibus + design-based '
    'sensitivity)',
    'modulation figure',
    'comparison against the mouse-level analysis',
)

# The optional statsmodels cross-check, inserted after the collapsed fit when it is requested.
_HIERARCHICAL_CROSSCHECK_COMPONENT = 'full cell-level MixedLM (optional independent cross-check)'


def run_hierarchical_cell_lane(sig_dir, signal, df_cell, df_mouse, mice_per_group,
                               mouse_contrasts, mouse_omnibus,
                               n_perm_omnibus=HIERARCHICAL_N_PERM_OMNIBUS,
                               seed=HIERARCHICAL_SEED, auto_close=True, verbose=True,
                               cross_check_full_fit=HIERARCHICAL_CROSS_CHECK_FULL_FIT,
                               panel_kwargs=None):
    """Run the whole hierarchical companion for one signal, ATOMICALLY.

    Output goes to ``<sig_dir>/hierarchical_cells/``, i.e. inside this analysis's own plots tree
    beside panels K and L, so everything for one signal lives in one place.

    ``cross_check_full_fit`` refits the whole model through statsmodels on all 38,060 cell rows
    and compares it against the reported collapsed fit. It is **False by default because it is
    slow and not load-bearing** -- see HIERARCHICAL_CROSS_CHECK_FULL_FIT. What guarantees the
    collapsed fit is the model is assert_reml_objective_identity, which is exact, mandatory, and
    costs under a second.

    ** One predefined flow, all-or-nothing. ** Every component in _HIERARCHICAL_COMPONENTS runs on
    every pass. There is no per-component switch and no result-dependent branch: nothing is
    activated or omitted because of what another component showed, and no model is substituted if
    a specified one fails. A failed fit, a failed equivalence assertion or an enumeration that
    does not match its multinomial coefficient RAISES. (``cross_check_full_fit`` adds a
    VALIDATION step; it is set before the run and by the caller, never by a result.)

    ``panel_kwargs`` is forwarded to ``plot_epoch_modulation_superplot`` for the figure's
    wording only (``title_prefix``/``epoch_titles``/``ylabel``). It exists so the event-proximal
    lane, which reuses this whole function on its own windows, does not draw a figure labelled
    with the full-epoch analysis's window names. It cannot change what is fit or plotted.

    ** Atomic output. ** Everything is written to a sibling ``<dir>__staging/`` and the previous
    complete output is replaced wholesale only after every component has succeeded, so a run that
    raises leaves the previous complete directory untouched and there is no window in which the
    promoted directory holds a mixture of two passes.
    """
    final_dir = os.path.join(sig_dir, HIERARCHICAL_DIRNAME)
    staging_dir = os.path.join(sig_dir, HIERARCHICAL_DIRNAME + '__staging')
    if os.path.isdir(staging_dir):
        shutil.rmtree(staging_dir)
    stats_dir = os.path.join(staging_dir, 'stats')
    tables_dir = os.path.join(staging_dir, 'tables')
    ensure_dirs(staging_dir, stats_dir, tables_dir)

    components = list(_HIERARCHICAL_COMPONENTS)
    if cross_check_full_fit:
        components.insert(2, _HIERARCHICAL_CROSSCHECK_COMPONENT)
    _remaining = iter(components)
    completed, current = [], next(_remaining)
    _n = len(components)
    _t_step = [time.perf_counter()]
    _t_lane = time.perf_counter()

    def _start():
        # DISPLAY ONLY. The full fit and the omnibus each run for minutes in total silence
        # otherwise, which is how a stalled run goes unnoticed.
        print(f'[hier/{signal} {len(completed) + 1:>2}/{_n} '
              f'{datetime.datetime.now():%H:%M:%S}] START {current}', flush=True)
        _t_step[0] = time.perf_counter()

    def _advance():
        nonlocal current
        print(f'[hier/{signal} {len(completed) + 1:>2}/{_n} '
              f'{datetime.datetime.now():%H:%M:%S}] DONE  {current} '
              f'({time.perf_counter() - _t_step[0]:.1f} s)', flush=True)
        completed.append(current)
        current = next(_remaining, 'done')

    try:
        _start()
        collapsed = CollapsedEpochModel(df_cell)
        _advance()

        _start()
        observed = collapsed.fit()
        fit = hierarchical_model_info(collapsed, observed, df_cell)
        equivalence_text = assert_reml_objective_identity(collapsed, df_cell)
        _advance()

        full_fit = None
        if cross_check_full_fit:
            _start()
            full_fit = fit_hierarchical_cell_model(df_cell)
            write_text(os.path.join(stats_dir, 'hierarchical_model_summary.txt'),
                       full_fit['summary_text'])
            equivalence_text += assert_collapse_equivalence(full_fit, observed,
                                                            collapsed=collapsed)
            _advance()
        else:
            equivalence_text += (
                'The optional full-data statsmodels cross-check was NOT run '
                '(cross_check_full_fit=False).\n'
                '  It refits the same model on every cell row through a second implementation and '
                'costs 4-14 min per\n'
                '  signal. It is a corroboration, not the guarantee: the identity above is what '
                'establishes that the\n'
                '  reported fit optimizes the full model\'s REML objective, and it is exact. Pass '
                'cross_check_full_fit=True\n'
                '  to run it; it then writes hierarchical_model_summary.txt beside this file and '
                'fills the\n'
                '  *_full_fit_crosscheck columns of the permutation table.\n')

        _start()
        perm_df = hierarchical_epoch_permutation(
            df_cell, fit, collapsed, mice_per_group, observed, full_fit=full_fit,
            n_perm_omnibus=n_perm_omnibus, seed=seed, verbose=verbose)
        perm_df.to_csv(os.path.join(tables_dir, 'hierarchical_permutation.csv'), index=False)
        write_text(os.path.join(stats_dir, 'hierarchical_contrasts.txt'),
                   format_hierarchical_report(perm_df, fit, collapsed, observed,
                                              equivalence_text, signal))
        _advance()

        _start()
        panel_contrasts = _hierarchical_panel_contrasts(perm_df)
        om = perm_df[perm_df['block'] == 'omnibus'].iloc[0]
        plot_epoch_modulation_superplot(
            df_cell, df_mouse, panel_contrasts,
            {'F': float(om['estimate']), 'df1': fit['omnibus']['df1'],
             'df2': fit['omnibus']['df2'], 'p': float(om['p_raw'])},
            os.path.join(staging_dir, 'hierarchical_epoch_modulation.png'),
            signal=signal, auto_close=auto_close,
            title_note='hierarchical cell-level companion; P values are mouse-label randomization',
            footer=(f'n = {fit["n_mice"]} mice — the unit of treatment assignment and of '
                    f'inference. {fit["n_cells"]} cells are modelled and drawn descriptively.\n'
                    f'Brackets are Holm-adjusted EXACT mouse-label randomization P-values of the '
                    f'cell-level model contrast within each epoch\n'
                    f'(tables/hierarchical_permutation.csv); no statistic is computed from the '
                    f'plotted values.'),
            **(panel_kwargs or {}))
        _advance()

        _start()
        write_hierarchical_vs_mouse_summary(stats_dir, signal, fit, perm_df, mouse_contrasts,
                                            mouse_omnibus, observed, collapsed)
        _advance()
    except BaseException as exc:
        write_text(os.path.join(staging_dir, 'FAILED.txt'),
                   f'Hierarchical cell-level lane FAILED ({signal}).\n'
                   f'Failing component: {current}\n'
                   f'Completed: {completed}\n\n{type(exc).__name__}: {exc}\n\n'
                   f'{traceback.format_exc()}')
        raise

    if os.path.isdir(final_dir):
        shutil.rmtree(final_dir)
    os.rename(staging_dir, final_dir)
    print(f'[hier/{signal}] lane complete in {(time.perf_counter() - _t_lane) / 60:.1f} min '
          f'-> {final_dir}', flush=True)
    return {'fit': fit, 'collapsed': collapsed, 'observed': observed, 'permutation': perm_df}


# ─────────────────────────────────────────────────────────────────────────────
# Implementation validation against planted truth
#
# ** DEVELOPMENT TOOLS, CALLED BY HAND. ** These validate the CODE, not the experiment, and
# run_hierarchical_cell_lane never calls them: a run that produces the reported numbers
# should compute those numbers and nothing else. Run verify_hierarchical_epoch_synthetic()
# after changing anything in the lane above.
# ─────────────────────────────────────────────────────────────────────────────

_SYNTH_GROUP_SIZES = (('mCherry', 6), ('hM3D', 5), ('hM4D', 6))
_SYNTH_N_CELLS = 60
_SYNTH_MOUSE_SD = 0.10        # between-animal SD of the animal intercept
_SYNTH_EPOCH_SD = 0.10        # between-animal SD of the per-epoch effect
_SYNTH_CELL_SD = 0.30         # within-animal, between-cell SD
_SYNTH_RESID_SD = 0.40        # within-cell residual SD
_SYNTH_SHIFT = 0.60           # planted group x epoch effect, index units
_SYNTH_ONE_MOUSE_SHIFT = 6.0  # deliberately absurd, and confined to ONE animal
_SYNTH_N_PERM_OMNIBUS = 300   # a VERIFICATION setting; reported runs use the frozen constant
_SYNTH_ALPHA = 0.05


def _synthetic_cell_epoch_table(group_sizes, shift_by_group_epoch, rng,
                                n_cells=_SYNTH_N_CELLS, one_mouse_shift=None):
    """A cell x epoch modulation table with a KNOWN planted structure.

    ``shift_by_group_epoch`` : {(group, epoch): shift} added to every cell of that group in that
    epoch -- the planted group x epoch effect. ``one_mouse_shift`` : (mouse, epoch, shift) added
    to a SINGLE animal, for the pseudoreplication check.

    Cells per animal are deliberately UNEQUAL (n_cells to 3*n_cells), because equal cluster sizes
    are the one case in which GLS weighting and equal-animal weighting coincide, and a verification
    run on that case would not exercise the weighting at all.
    """
    rows = []
    mice_per_group = {}
    counter = 0
    for group, n in group_sizes:
        mice_per_group[group] = []
        for _ in range(n):
            counter += 1
            mouse = f'S{counter:02d}'
            mice_per_group[group].append(mouse)
            n_c = int(rng.integers(n_cells, 3 * n_cells + 1))
            u = rng.normal(0.0, _SYNTH_MOUSE_SD)
            s = {e: rng.normal(0.0, _SYNTH_EPOCH_SD) for e in RESPONSE_EPOCHS}
            a = rng.normal(0.0, _SYNTH_CELL_SD, size=n_c)
            for e in RESPONSE_EPOCHS:
                shift = shift_by_group_epoch.get((group, e), 0.0)
                if one_mouse_shift is not None and (mouse, e) == one_mouse_shift[:2]:
                    shift += one_mouse_shift[2]
                vals = u + s[e] + a + shift + rng.normal(0.0, _SYNTH_RESID_SD, size=n_c)
                rows.append(pd.DataFrame({'mouse': mouse, 'group': group,
                                          'cell': np.arange(n_c), 'epoch': e, 'index': vals}))
    return pd.concat(rows, ignore_index=True), mice_per_group


def _synth_lookup(perm_df, epoch, group_a, group_b, column):
    sub = perm_df[(perm_df['block'] == 'pairwise_primary') & (perm_df['epoch'] == epoch)
                  & (perm_df['group_a'] == group_a) & (perm_df['group_b'] == group_b)]
    if len(sub) != 1:
        raise RuntimeError(f'_synth_lookup: {len(sub)} rows for {epoch} {group_a} vs {group_b}.')
    return float(sub.iloc[0][column])


def verify_hierarchical_epoch_synthetic(save_dir, seed=HIERARCHICAL_SEED,
                                        group_sizes=_SYNTH_GROUP_SIZES,
                                        n_perm_omnibus=_SYNTH_N_PERM_OMNIBUS):
    """Plant a known truth in SIMULATED data and hard-fail if this lane does not recover it.

    Five designs, and each one checks a property the lane's validity rests on:

    0. **The analytic gradient is the objective's gradient.** Every fit below optimizes with it,
       and a wrong one would not raise -- it would quietly return a worse optimum.

    1. **A planted group x epoch effect is recovered.** A shift confined to the shock epoch in one
       treatment group must produce a significant within-epoch comparison there AND a significant
       omnibus. Without this the rest is a test that the code returns large p-values.

    2. **A huge effect confined to ONE animal must NOT reach significance.** A cell-level test
       would call it overwhelming -- hundreds of cells all shifted by 6 index units. The
       mouse-label randomization cannot, because that animal's LABEL is what moves. This is the
       pseudoreplication guard, and it is the single most important check here.

    3. **Adding cells at a fixed number of animals sharpens each animal's own estimate and buys no
       treatment-level precision.** Measured over a 100x increase in cells per animal under a true
       null: the mean within-animal SEM must fall by roughly the square root of that factor while
       the group-contrast standard error must NOT track it downward. The contrast SE is a
       between-animal quantity floored by between-animal variance, and that gap is the guard.
       (Computed through the collapsed model alone, since a full statsmodels fit at 100x cells is
       not a computation that finishes; the collapse-equivalence check is design 4's job.)

    4. **The collapsed representation reproduces the full fit** on independently simulated data,
       not only on the one dataset it was developed against.

    Writes a report and raises on the first failure. This is a DEVELOPMENT TOOL: it is never
    called by run_hierarchical_cell_lane.
    """
    ensure_dirs(save_dir)
    rng = np.random.default_rng(seed)
    lines = ['Hierarchical cell-level lane — verification against planted synthetic truth',
             '=' * 78,
             f'seed = {seed};  group sizes = {dict(group_sizes)};  '
             f'omnibus draws = {n_perm_omnibus} (a VERIFICATION setting; reported runs use '
             f'{HIERARCHICAL_N_PERM_OMNIBUS})',
             '']

    def _run(df_cell, mice_per_group, n_perm=n_perm_omnibus):
        # The statsmodels cross-check IS run here, on every design, even though a real run leaves
        # it off. These are small simulated frames where it costs seconds, and validating the code
        # is exactly the occasion for the belt-and-braces check -- design 4 then does it again on
        # its own data and reports the numbers.
        collapsed = CollapsedEpochModel(df_cell)
        observed = collapsed.fit()
        fit = hierarchical_model_info(collapsed, observed, df_cell)
        full_fit = fit_hierarchical_cell_model(df_cell)
        assert_reml_objective_identity(collapsed, df_cell)
        assert_collapse_equivalence(full_fit, observed, collapsed=collapsed)
        perm = hierarchical_epoch_permutation(df_cell, fit, collapsed, mice_per_group, observed,
                                              full_fit=full_fit, n_perm_omnibus=n_perm,
                                              seed=seed, verbose=False)
        return fit, perm

    # ---- design 0: the analytic gradient is the objective's actual gradient ---------------
    # Checked first, because everything below optimizes with it. A wrong gradient would not raise;
    # it would quietly return a worse optimum, which is the failure mode hardest to notice.
    df0, _mpg0 = _synthetic_cell_epoch_table(group_sizes, {}, rng, n_cells=10)
    collapsed0 = CollapsedEpochModel(df0)
    x0 = collapsed0.design(collapsed0.true_assignment)
    worst = 0.0
    for _ in range(4):
        theta = np.log(rng.uniform(0.002, 0.5, size=collapsed0.n_epochs + 3))
        f_a, grad_a = collapsed0._neg2_reml_and_grad(theta, x0)
        grad_n = approx_fprime(theta, lambda t: collapsed0._neg2_reml(t, x0), 1e-6)
        if abs(f_a - collapsed0._neg2_reml(theta, x0)) > 1e-9:
            raise RuntimeError('verify_hierarchical_epoch_synthetic DESIGN 0 FAILED: '
                               '_neg2_reml_and_grad and _neg2_reml disagree on the objective.')
        worst = max(worst, float(np.max(np.abs(grad_a - grad_n)
                                        / np.maximum(np.abs(grad_n), 1e-6))))
    lines += ['DESIGN 0 — the analytic gradient against finite differences',
              f'  max relative error over 4 parameter vectors = {worst:.2e}', '']
    if worst > 1e-4:
        raise RuntimeError(
            f'verify_hierarchical_epoch_synthetic DESIGN 0 FAILED: the analytic gradient differs '
            f'from the finite-difference gradient by up to {worst:.3g} (relative). Every fit below '
            f'optimizes with it, so a wrong gradient silently degrades every estimate rather than '
            f'raising.')

    # ---- design 1: a planted group x epoch effect is recovered ---------------------------
    df1, mpg1 = _synthetic_cell_epoch_table(
        group_sizes, {('hM3D', 'shock'): _SYNTH_SHIFT}, rng)
    fit1, perm1 = _run(df1, mpg1)
    p_shock = _synth_lookup(perm1, 'shock', 'hM3D', 'mCherry', 'p_holm_epoch')
    p_tone = _synth_lookup(perm1, 'tone', 'hM3D', 'mCherry', 'p_holm_epoch')
    est_shock = _synth_lookup(perm1, 'shock', 'hM3D', 'mCherry', 'estimate')
    p_omni = float(perm1[perm1['block'] == 'omnibus'].iloc[0]['p_raw'])
    lines += [f'DESIGN 1 — planted +{_SYNTH_SHIFT} shift, hM3D at shock only',
              f'  shock hM3D vs Ctl: estimate {est_shock:+.3f}  Holm P = {p_shock:.4g}',
              f'  tone  hM3D vs Ctl: Holm P = {p_tone:.4g}  (nothing planted here)',
              f'  omnibus randomization P = {p_omni:.4g}', '']
    if not (p_shock < _SYNTH_ALPHA and p_omni < _SYNTH_ALPHA):
        raise RuntimeError(
            f'verify_hierarchical_epoch_synthetic DESIGN 1 FAILED: a planted +{_SYNTH_SHIFT} '
            f'group x epoch effect was not recovered (shock Holm P = {p_shock:.4g}, omnibus '
            f'P = {p_omni:.4g}). The lane cannot detect an effect it is given.')
    if abs(est_shock - _SYNTH_SHIFT) > 0.25:
        raise RuntimeError(
            f'verify_hierarchical_epoch_synthetic DESIGN 1 FAILED: the recovered estimate '
            f'{est_shock:+.3f} is far from the planted {_SYNTH_SHIFT:+.3f}.')

    # ---- design 2: a huge effect in ONE animal must not reach significance ---------------
    outlier_mouse = 'S08'
    df2, mpg2 = _synthetic_cell_epoch_table(
        group_sizes, {}, rng, one_mouse_shift=(outlier_mouse, 'shock', _SYNTH_ONE_MOUSE_SHIFT))
    group_of_outlier = next((g for g, ms in mpg2.items() if outlier_mouse in ms), None)
    if group_of_outlier != 'hM3D':
        raise RuntimeError(
            f'verify_hierarchical_epoch_synthetic DESIGN 2: the animal carrying the planted '
            f'single-animal effect ({outlier_mouse}) is in group {group_of_outlier!r}, not hM3D, '
            f'so the comparison checked below would not contain it. Fix the animal id to match '
            f'the group sizes rather than checking a comparison the effect is not in.')
    fit2, perm2 = _run(df2, mpg2)
    p_one = _synth_lookup(perm2, 'shock', 'hM3D', 'mCherry', 'p_raw')
    p_one_holm = _synth_lookup(perm2, 'shock', 'hM3D', 'mCherry', 'p_holm_epoch')
    lines += [f'DESIGN 2 — +{_SYNTH_ONE_MOUSE_SHIFT} at shock, confined to ONE animal '
              f'({outlier_mouse}, group {group_of_outlier}); every other animal is a true null',
              f'  shock hM3D vs Ctl: exact randomization P = {p_one:.4g}, Holm P = '
              f'{p_one_holm:.4g}',
              '  A cell-level test would call hundreds of identically shifted cells overwhelming.',
              '  The mouse-label randomization cannot, because that animal\'s LABEL is what moves.',
              '']
    if p_one_holm < _SYNTH_ALPHA:
        raise RuntimeError(
            f'verify_hierarchical_epoch_synthetic DESIGN 2 FAILED: an effect confined to ONE '
            f'animal reached significance (Holm P = {p_one_holm:.4g}). That is exactly the '
            f'pseudoreplication this lane exists to prevent.')

    # ---- design 3: more cells sharpen the animal, not the contrast ------------------------
    lines.append('DESIGN 3 — a true null, with cells per animal scaled 1x / 10x / 100x')
    sem_by_scale, se_by_scale = {}, {}
    for scale in (1, 10, 100):
        df3, mpg3 = _synthetic_cell_epoch_table(group_sizes, {}, rng, n_cells=6 * scale)
        collapsed = CollapsedEpochModel(df3)
        observed = collapsed.fit()
        sem = float(summarize_mice(df3)['sem_cells'].mean())
        # The contrast standard error, read off the same collapsed fit the randomization uses.
        names = collapsed.fe_names
        w = np.zeros(len(names))
        w[names.index([n for n in names if '[T.hM3D]' in n and ':' not in n][0])] = 1.0
        se = float(np.sqrt(w @ observed['cov'] @ w))
        sem_by_scale[scale], se_by_scale[scale] = sem, se
        lines.append(f'  {scale:>4}x cells: mean within-animal SEM = {sem:.4f}   '
                     f'group-contrast SE = {se:.4f}   '
                     f'(median cells/animal = {df3.groupby("mouse")["cell"].nunique().median():.0f})')
    sem_drop = sem_by_scale[1] / sem_by_scale[100]
    se_ratio = se_by_scale[100] / se_by_scale[1]
    lines += [f'  within-animal SEM fell by {sem_drop:.1f}x over a 100x increase in cells '
              f'(sqrt(100) = 10 is the expectation)',
              f'  group-contrast SE moved by a factor of {se_ratio:.2f} -- it must NOT track '
              f'cell count downward', '']
    if sem_drop < 5.0:
        raise RuntimeError(
            f'verify_hierarchical_epoch_synthetic DESIGN 3 FAILED: the within-animal SEM fell '
            f'only {sem_drop:.1f}x over a 100x increase in cells; adding cells is not sharpening '
            f'the per-animal estimate as it must.')
    if se_ratio < 0.5:
        raise RuntimeError(
            f'verify_hierarchical_epoch_synthetic DESIGN 3 FAILED: the group-contrast SE fell by '
            f'{1 / se_ratio:.1f}x when only the CELL COUNT increased. The contrast SE is a '
            f'between-animal quantity and must be floored by between-animal variance; a version '
            f'in which cells buy treatment-level precision is pseudoreplicating.')

    # ---- design 4: the collapse reproduces the full fit on fresh simulated data -----------
    df4, mpg4 = _synthetic_cell_epoch_table(group_sizes, {('hM4D', 'trace'): 0.3}, rng)
    fit4 = fit_hierarchical_cell_model(df4)
    collapsed4 = CollapsedEpochModel(df4)
    lines += ['DESIGN 4 — the collapse, on independently simulated data',
              '  ' + assert_reml_objective_identity(collapsed4, df4).replace('\n', '\n  '),
              '  ' + assert_collapse_equivalence(fit4, collapsed4.fit(),
                                                 collapsed=collapsed4).replace('\n', '\n  '), '']

    lines.append("ALL FIVE DESIGNS PASSED.")
    text = '\n'.join(lines) + '\n'
    write_text(os.path.join(save_dir, 'hierarchical_synthetic_verification.txt'), text)
    print(text, flush=True)
    return text


def compare_hierarchical_signals(perm_by_signal):
    """Side-by-side hierarchical contrasts for the primary and confirmatory signals.

    ** The agreement criterion here is NOT the one compare_signals() uses, deliberately. ** That
    criterion -- matching sign AND overlapping 95% CIs -- is informative for a non-null effect but
    degenerate near zero, where the sign of an estimate is noise: two estimates whose intervals
    overlap almost entirely and both span zero get flagged as disagreeing purely because they fall
    on opposite sides of it, which is agreement about there being no effect. This lane is new, so
    it is written with the fixed criterion from the start rather than inheriting a known defect:

        the signals AGREE when they license the same conclusion -- either both intervals contain
        zero (neither finds an effect), or the estimates match in sign with overlapping intervals.

    Changing compare_signals() itself is a separate matter: its numbers are already reported.
    """
    primary, confirm = perm_by_signal[PRIMARY_SIGNAL], perm_by_signal[CONFIRMATORY_SIGNAL]
    keys = ['epoch', 'group_a', 'group_b']
    cols = keys + ['estimate', 'ci_low', 'ci_high', 'p_raw', 'p_holm_epoch', 'in_holm_family']
    merged = (primary[primary['block'] == 'pairwise_primary'][cols]
              .merge(confirm[confirm['block'] == 'pairwise_primary'][cols], on=keys,
                     suffixes=('_primary', '_confirm')))

    spans_zero_p = (merged['ci_low_primary'] <= 0) & (merged['ci_high_primary'] >= 0)
    spans_zero_c = (merged['ci_low_confirm'] <= 0) & (merged['ci_high_confirm'] >= 0)
    same_sign = np.sign(merged['estimate_primary']) == np.sign(merged['estimate_confirm'])
    overlap = ((merged['ci_low_primary'] <= merged['ci_high_confirm']) &
               (merged['ci_low_confirm'] <= merged['ci_high_primary']))
    merged['agrees'] = (spans_zero_p & spans_zero_c) | (same_sign & overlap)

    lines = [
        f'Hierarchical cell-level companion: {PRIMARY_SIGNAL} (primary) vs '
        f'{CONFIRMATORY_SIGNAL} (confirmatory)',
        '=' * 78,
        'The same model and the same randomization, run twice on two signals. The comparison is',
        'about AGREEMENT OF CONCLUSIONS, not equality of magnitudes.',
        'Agreement criterion: both 95% CIs contain zero, OR matching sign with overlapping CIs.',
        '',
    ]
    for _, r in merged.iterrows():
        verdict = 'agree' if r['agrees'] else 'DISAGREE'
        holm = ('' if r['in_holm_family_primary'] else '  [in no Holm family]')
        lines.append(
            f'  {r["epoch"]:>11} {GROUP_LABELS[r["group_a"]]:>3} vs '
            f'{GROUP_LABELS[r["group_b"]]:<3}: '
            f'{PRIMARY_SIGNAL} {r["estimate_primary"]:+.4f} '
            f'[{r["ci_low_primary"]:+.4f}, {r["ci_high_primary"]:+.4f}] P={r["p_raw_primary"]:.3g}'
            f'  |  {CONFIRMATORY_SIGNAL} {r["estimate_confirm"]:+.4f} '
            f'[{r["ci_low_confirm"]:+.4f}, {r["ci_high_confirm"]:+.4f}] '
            f'P={r["p_raw_confirm"]:.3g}  -> {verdict}{holm}')
    n_disagree = int((~merged['agrees']).sum())
    lines.append('')
    lines.append(f'{len(merged) - n_disagree} of {len(merged)} comparisons agree between signals.'
                 if n_disagree else f'All {len(merged)} comparisons agree between signals.')
    if n_disagree:
        lines.append('Disagreement is REPORTED, never adjudicated: both signals have their own '
                     'full output and both should be reported rather than one selected.')
    return merged, '\n'.join(lines) + '\n'


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Event-proximal companion lane
#
# The full-epoch analysis averages three of its four windows over 20 s. A response locked
# to an event ONSET contributes only a few percent of such a mean, so a real brief
# transient can be diluted to invisibility -- which is one concrete explanation for the
# small cohort-wide modulation the main lane reports. This lane tests that directly by
# recomputing THE SAME INDEX over EVENT_PROXIMAL_WINDOW_S from each event onset and
# changing nothing else: same resolved cells, same retained trials, same single per-cell
# standardization, same trial-matched full-length pre-tone baseline.
#
# It is a COMPANION. Everything above it is unchanged and remains the paper-facing
# analysis, and the two lanes are a PAIRED comparison on identical data (see
# build_modulation_table's window_seconds) rather than two separate analyses.
#
# ** The inference here is the hierarchical cell-level model, not the mouse-level one. **
# The mouse-level model has a single residual variance shared across all four windows,
# and the hierarchical lane's fitted variance components show that assumption is wrong on
# these data by two orders of magnitude (the shock window's between-animal variance is
# ~373x the tone window's), which makes the quiet windows' intervals too wide and the loud
# one's too narrow. Short windows do not repair that and are unlikely to shrink the gap,
# so the mouse-level fit is still computed -- it is what the hierarchical lane compares
# itself against -- but it is written out as a labelled CROSS-CHECK and brackets nothing.
# ─────────────────────────────────────────────────────────────────────────────

# The one number that defines this lane. Uniform across all four events, so the four
# windows hold the same number of frames and the epoch factor stays a clean 4-level
# comparison. 3 s is short enough to be genuinely event-proximal (a seventh of the 20 s
# epochs) and long enough to contain a GCaMP transient's rise and part of its decay.
EVENT_PROXIMAL_WINDOW_S = 3.0

EVENT_PROXIMAL_DIRNAME = 'event_proximal'
EVENT_PROXIMAL_METHODS_FILENAME = 'epoch_modulation_event_proximal_methods.md'

# Panel M alignment window, relative to each event onset.
EVENT_PROXIMAL_PRE_S = 5.0
EVENT_PROXIMAL_POST_S = 10.0

# What the four onsets ARE, as opposed to what the four epochs are. The distinction is the
# whole point of the lane, so the figures never reuse the epoch names unqualified.
EVENT_ONSET_LABELS = {'tone': 'tone onset', 'trace': 'trace onset (tone offset)',
                      'shock': 'shock onset (US)', 'post_shock': 'post-shock onset'}


def _event_proximal_epoch_titles(window_seconds):
    return {e: f'{EVENT_ONSET_LABELS[e]}\n+{window_seconds:g} s' for e in RESPONSE_EPOCHS}


def _event_aligned_mouse_trace(z, session, trials, epoch):
    """One mouse's mean standardized activity around one event onset (a vector over frames).

    Baseline-subtracted exactly as the index is -- each trial's own full-length pre-tone
    mean, per cell -- then averaged over trials and over cells. ** No re-standardization
    happens here ** (see ``_tone_aligned_matrix`` for why that matters): the trace arrives
    standardized once over the whole session and is only averaged.

    The one deliberate mismatch with the index: this averages over trials, whereas
    ``summarize_cells`` takes each cell's MEDIAN over trials. So the mean of this curve over
    the shaded window is close to, but not identical to, the mouse's plotted index. Stated
    rather than papered over by switching either one to match the other -- the median is
    there so a single anomalous trial cannot define a neuron, and a median is the wrong
    thing to draw a time course with.
    """
    pre_f = _window_frames(EVENT_PROXIMAL_PRE_S)
    post_f = _window_frames(EVENT_PROXIMAL_POST_S)
    n_frames = z.shape[1]

    per_trial = []
    for trial_idx in trials:
        onset = get_epoch_frames(session, epoch, trial_idx)[0]
        lo, hi = onset - pre_f, onset + post_f
        if lo < 0 or hi > n_frames:
            raise RuntimeError(
                f'epoch_modulation: mouse {session.mouse!r} trial {trial_idx} {epoch!r} '
                f'alignment window [{lo}, {hi}) falls outside the {n_frames}-frame recording. '
                f'The trial was retained, so every window this lane needs must be present.')
        base_on, base_off = get_epoch_frames(session, BASELINE_EPOCH, trial_idx)
        baseline = z[:, base_on:base_off].mean(axis=1, keepdims=True)
        per_trial.append(z[:, lo:hi] - baseline)

    avg = np.mean(np.stack(per_trial, axis=0), axis=0)      # (n_cells, win)
    time_axis = (np.arange(pre_f + post_f) - pre_f) / MINISCOPE_FPS
    return avg.mean(axis=0), time_axis


def collect_event_aligned_traces(mice_per_group, sessions, signal, mapping='full',
                                 crossreg_to_use=None):
    """{group: {epoch: (n_mice, win) array}} plus the time axis, for panel M.

    Built from the same ``resolve_shared_cells`` / ``retained_trials`` pair the index uses, so
    the curves and the numbers describe the same cells and the same trials.
    """
    traces = {g: {e: [] for e in RESPONSE_EPOCHS} for g in DREADD_DISPLAY_ORDER}
    time_axis = None
    for group in DREADD_DISPLAY_ORDER:
        for mouse in mice_per_group.get(group, []):
            if mouse not in sessions:
                continue
            session = sessions[mouse]
            wc = crossreg_to_use[mouse] if (crossreg_to_use is not None and mapping != 'full') else None
            cell_ids, _ = resolve_shared_cells(session, mapping=mapping, with_crossreg=wc)
            z = _standardized_trace(session, signal, cell_ids)
            trials, _ = retained_trials(session)
            for epoch in RESPONSE_EPOCHS:
                vec, time_axis = _event_aligned_mouse_trace(z, session, trials, epoch)
                traces[group][epoch].append(vec)
    out = {g: {e: np.stack(v, axis=0) for e, v in per_epoch.items() if v}
           for g, per_epoch in traces.items()}
    out = {g: v for g, v in out.items() if v}
    if not out:
        raise RuntimeError('epoch_modulation: collected no event-aligned traces for panel M.')
    return out, time_axis


def plot_event_aligned_traces(traces, time_axis, out_path, window_seconds, signal=PRIMARY_SIGNAL,
                              auto_close=True):
    """Panel M: group-mean activity around each event onset, with the analysed window shaded.

    ** This is the panel the whole lane rests on. ** If there is no onset-locked transient
    here, then a null on the short windows means "there is nothing brief to find", not "the
    brief response does not differ between groups" -- a different statement, and the results
    write-up has to say which one it is.

    Mean +/- SEM ACROSS MICE (n = mice, the unit of inference), never across cells.
    """
    fig, axs = plt.subplots(1, len(RESPONSE_EPOCHS),
                            figsize=(3.1 * len(RESPONSE_EPOCHS), 3.6), sharey=True)
    axs = np.atleast_1d(axs)

    for ax, epoch in zip(axs.flat, RESPONSE_EPOCHS):
        ax.axvspan(0.0, window_seconds, color='0.85', lw=0, zorder=0)
        ax.axvline(0.0, color='k', lw=0.8, zorder=1)
        ax.axhline(0.0, color='0.6', lw=0.8, ls='--', zorder=1)
        for group in DREADD_DISPLAY_ORDER:
            mat = traces[group][epoch]
            mu = mat.mean(axis=0)
            sem = mat.std(axis=0, ddof=1) / np.sqrt(mat.shape[0])
            ax.plot(time_axis, mu, color=GROUP_COLOURS[group], lw=1.4,
                    label=f'{GROUP_LABELS[group]} (n = {mat.shape[0]})', zorder=3)
            ax.fill_between(time_axis, mu - sem, mu + sem, color=GROUP_COLOURS[group],
                            alpha=0.18, lw=0, zorder=2)
        ax.set_title(EVENT_ONSET_LABELS[epoch], size='medium')
        ax.set_xlabel('Time from onset (s)')
    axs.flat[0].set_ylabel('Activity − pre-tone\n(cell SD units)')
    axs.flat[0].legend(frameon=False, fontsize='x-small', loc='upper left')
    fig.suptitle(f'Event-aligned activity ({signal}); shaded band = the analysed '
                 f'{window_seconds:g} s window', size='medium')
    fig.text(0.5, 0.015,
             'Mean ± SEM across MICE. Standardized once per cell over the whole session and '
             'baseline-subtracted per trial;\nno re-scaling inside the alignment window. Curves '
             'average over trials where the index takes each cell\'s median, so the shaded '
             'window\'s\nmean is close to but not identical to the plotted index.',
             ha='center', size='xx-small')
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.28, top=0.84, wspace=0.12)
    save_fig(fig, out_path)
    if auto_close:
        plt.close(fig)


def dilution_table(df_mouse_event, df_mouse_full):
    """Per-(mouse, group, epoch) event-proximal index beside the full-epoch one.

    The lane's own premise, quantified: if the 20 s mean was diluting a brief response, the
    short window's index is the larger of the two in magnitude. The two columns are paired --
    same mouse, same cells, same trials, same baseline -- so their difference isolates the
    window and nothing else.
    """
    merged = df_mouse_event.merge(
        df_mouse_full[['mouse', 'group', 'epoch', 'value']], on=['mouse', 'group', 'epoch'],
        how='inner', suffixes=('_event', '_full'))
    if len(merged) != len(df_mouse_event):
        raise RuntimeError(
            f'epoch_modulation: the event-proximal lane has {len(df_mouse_event)} mouse x epoch '
            f'rows but only {len(merged)} of them matched the full-epoch lane. Both lanes must '
            f'describe the same mice and epochs.')
    merged['difference'] = merged['value_event'] - merged['value_full']
    # Magnitude ratio, not signed: "did the short window see MORE modulation" is a question
    # about size. A sign flip between the two lanes is visible in the two value columns.
    merged['abs_ratio'] = np.abs(merged['value_event']) / np.abs(merged['value_full'])
    return merged


def plot_dilution_comparison(dilution, out_path, window_seconds, signal=PRIMARY_SIGNAL,
                             auto_close=True):
    """Per-mouse event-proximal index against the full-epoch index, one facet per event."""
    fig, axs = plt.subplots(1, len(RESPONSE_EPOCHS),
                            figsize=(2.9 * len(RESPONSE_EPOCHS), 3.3))
    axs = np.atleast_1d(axs)
    for ax, epoch in zip(axs.flat, RESPONSE_EPOCHS):
        sub = dilution[dilution['epoch'] == epoch]
        lim = float(np.abs(np.concatenate([sub['value_full'].to_numpy(),
                                           sub['value_event'].to_numpy()])).max()) * 1.15
        ax.plot([-lim, lim], [-lim, lim], color='0.6', lw=0.8, ls='--', zorder=1)
        ax.axhline(0.0, color='0.85', lw=0.6, zorder=0)
        ax.axvline(0.0, color='0.85', lw=0.6, zorder=0)
        for group in DREADD_DISPLAY_ORDER:
            g = sub[sub['group'] == group]
            ax.scatter(g['value_full'], g['value_event'], s=26, alpha=0.85,
                       color=GROUP_COLOURS[group], edgecolor='k', linewidth=0.3,
                       label=GROUP_LABELS[group], zorder=3)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(EVENT_ONSET_LABELS[epoch], size='small')
        ax.set_xlabel('full-epoch index')
    axs.flat[0].set_ylabel(f'{window_seconds:g} s event-proximal index')
    axs.flat[0].legend(frameon=False, fontsize='x-small', loc='upper left')
    fig.suptitle(f'Does the full-epoch mean dilute an event-locked response? ({signal}) — '
                 f'one point per mouse', size='medium')
    fig.text(0.5, 0.015,
             'Points above the diagonal in the upper-right quadrant (or below it in the '
             'lower-left) are mice whose modulation is LARGER in the short window,\ni.e. mice '
             'for which the 20 s mean was diluting. The two axes are paired: same mouse, same '
             'cells, same trials, same baseline.', ha='center', size='xx-small')
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.26, top=0.85, wspace=0.25)
    save_fig(fig, out_path)
    if auto_close:
        plt.close(fig)


def format_dilution_report(dilution, coverage, window_seconds, signal):
    """The lane's own overview: what the window changed, and how much of C is flat windows."""
    heading = f'Event-proximal modulation lane — {signal}'
    lines = [
        heading,
        '=' * len(heading),
        '',
        f'Response window: {window_seconds:g} s from each event onset (tone, trace, shock,',
        f'post-shock), uniform across all four. Baseline: the SAME trial-matched '
        f'{TRACE_MATCHED_WINDOW_S:.0f} s',
        'pre-tone window the full-epoch analysis uses -- a mean against a mean is not',
        'exposure-scaled, so a longer baseline is simply a more precise one.',
        '',
        'Same resolved cells, same retained trials and the same single per-cell whole-session',
        'standardization as the full-epoch lane, so the comparison below is PAIRED.',
        '',
        'The paper-facing inference for this lane is the hierarchical cell-level model in',
        f'{HIERARCHICAL_DIRNAME}/, which gives each event window its own between-animal',
        'variance. The mouse-level fit beside this file is a labelled CROSS-CHECK: its single',
        'shared residual across windows is the assumption the hierarchical variance components',
        'contradict, and it brackets nothing.',
        '',
        'Event-proximal versus full-epoch index, mean over mice (paired)',
        '-' * 74,
    ]
    for epoch in RESPONSE_EPOCHS:
        sub = dilution[dilution['epoch'] == epoch]
        n_larger = int((sub['abs_ratio'] > 1.0).sum())
        lines.append(
            f'  {epoch:>11}: event-proximal {sub["value_event"].mean():+.4f}   '
            f'full-epoch {sub["value_full"].mean():+.4f}   '
            f'difference {sub["difference"].mean():+.4f}   '
            f'median |ratio| {sub["abs_ratio"].median():.2f}   '
            f'{n_larger}/{len(sub)} mice larger in the short window')
    lines += [
        '',
        'The shock row is the internal check, not a result: the full-epoch shock window was',
        'already the 2 s US, so the two lanes are measuring nearly the same thing there and',
        'should agree. A large divergence in that row means the windowing is wrong.',
        '',
    ]
    if 'frac_flat_window' in coverage.columns:
        frac = coverage['frac_flat_window'].astype(float)
        lines += [
            f'Flat windows ({window_seconds:g} s windows in which the trace never changes)',
            '-' * 74,
            'On C these are deconvolution sparsity at the scale of a single short window: a',
            'cell with no fitted transient inside the window is identically zero there, so its',
            'index is exactly -(its pre-tone mean) -- a deterministic constant rather than a',
            'measurement. resolve_shared_cells only excludes cells constant over the WHOLE',
            'session, so these stay in by design. They are counted here instead of hidden.',
            '',
            f'  across mice: median {frac.median() * 100:.1f}%   range '
            f'{frac.min() * 100:.1f}%-{frac.max() * 100:.1f}%   '
            f'total {int(coverage["n_flat_windows"].sum())} of '
            f'{int((coverage["n_cells"] * coverage["n_trials_retained"] * len(RESPONSE_EPOCHS)).sum())} '
            f'(cell x trial x event) values',
            '',
            'This is why YrA is the primary signal, and the argument is stronger here than in',
            'the full-epoch lane: the shorter the window, the more of C is constant inside it.',
            '',
        ]
    return '\n'.join(lines) + '\n'


def run_event_proximal_lane(sig_dir, signal, mice_per_group, sessions, df_mouse_full,
                            mapping='full', crossreg_to_use=None,
                            window_seconds=EVENT_PROXIMAL_WINDOW_S, auto_close=True,
                            verbose=True, run_hierarchical_cells=True,
                            hierarchical_cross_check_full_fit=HIERARCHICAL_CROSS_CHECK_FULL_FIT):
    """Run the event-proximal companion for one signal, into ``<sig_dir>/event_proximal/``.

    ``df_mouse_full`` is the full-epoch lane's per-mouse endpoint for the SAME signal, already
    in hand at the call site -- the dilution comparison is paired against it rather than
    recomputing the full-epoch table a second time.

    Everything it needs is the existing machinery: ``build_modulation_table`` with a window,
    ``summarize_cells``/``summarize_mice``, ``fit_mouse_level_model`` for the cross-check, and
    ``run_hierarchical_cell_lane`` for the reported inference. Nothing is reimplemented.
    """
    out_dir = os.path.join(sig_dir, EVENT_PROXIMAL_DIRNAME)
    tables_dir = os.path.join(out_dir, 'tables')
    stats_dir = os.path.join(out_dir, 'stats')
    ensure_dirs(out_dir, tables_dir, stats_dir)

    if verbose:
        print(f'[event-proximal/{signal}] building {window_seconds:g} s modulation table...',
              flush=True)
    df, coverage = build_modulation_table(
        mice_per_group, sessions, signal=signal, mapping=mapping,
        crossreg_to_use=crossreg_to_use, verbose=verbose, window_seconds=window_seconds)
    df_cell = summarize_cells(df)
    df_mouse = summarize_mice(df_cell)
    n_mice = df_mouse['mouse'].nunique()

    # The mouse-level fit: needed as the hierarchical lane's comparison arm, reported as a
    # cross-check only (see this section's header).
    _result, method, model_text, contrasts, omnibus = fit_mouse_level_model(df_mouse)

    dilution = dilution_table(df_mouse, df_mouse_full)

    df_cell.to_csv(os.path.join(tables_dir, 'per_cell_modulation.csv'), index=False)
    df_mouse.to_csv(os.path.join(tables_dir, 'per_mouse_modulation.csv'), index=False)
    coverage.to_csv(os.path.join(tables_dir, 'trial_coverage.csv'), index=False)
    contrasts.to_csv(os.path.join(tables_dir, 'model_contrasts_crosscheck.csv'), index=False)
    dilution.to_csv(os.path.join(tables_dir, 'dilution_vs_full_epoch.csv'), index=False)
    write_text(os.path.join(stats_dir, 'event_proximal_report.txt'),
               format_dilution_report(dilution, coverage, window_seconds, signal))
    write_text(os.path.join(stats_dir, 'mouse_level_crosscheck.txt'),
               'CROSS-CHECK ONLY — not the reported inference for this lane.\n'
               'This model gives all four event windows a single shared residual variance. The\n'
               f'hierarchical fit in {HIERARCHICAL_DIRNAME}/ estimates a separate between-animal\n'
               'variance per window and is what the figure brackets and the write-up report.\n\n'
               + format_model_report(contrasts, omnibus, method, n_mice,
                                     window_seconds=window_seconds)
               + format_variance_diagnostic(df_mouse)
               + '\n\nStatsmodels summary of the cross-check fit\n'
               + '-' * 74 + '\n' + model_text)

    # ── Panel M: the premise ──
    traces, time_axis = collect_event_aligned_traces(
        mice_per_group, sessions, signal, mapping=mapping, crossreg_to_use=crossreg_to_use)
    plot_event_aligned_traces(traces, time_axis,
                              os.path.join(out_dir, 'panel_M_event_aligned_traces.png'),
                              window_seconds, signal=signal, auto_close=auto_close)
    plot_dilution_comparison(dilution, os.path.join(out_dir, 'dilution_vs_full_epoch.png'),
                             window_seconds, signal=signal, auto_close=auto_close)

    panel_kwargs = {
        'title_prefix': f'Per-cell event-proximal modulation, {window_seconds:g} s from onset',
        'epoch_titles': _event_proximal_epoch_titles(window_seconds),
        'ylabel': f'Modulation index\n({window_seconds:g} s from onset − pre-tone, cell SD units)',
    }

    perm_df = None
    if run_hierarchical_cells:
        perm_df = run_hierarchical_cell_lane(
            out_dir, signal, df_cell, df_mouse, mice_per_group, contrasts, omnibus,
            auto_close=auto_close, verbose=verbose,
            cross_check_full_fit=hierarchical_cross_check_full_fit,
            panel_kwargs=panel_kwargs)['permutation']
        panel_contrasts = _hierarchical_panel_contrasts(perm_df)
        om = perm_df[perm_df['block'] == 'omnibus'].iloc[0]
        panel_omnibus = {'F': float(om['estimate']), 'df1': omnibus['df1'],
                         'df2': omnibus['df2'], 'p': float(om['p_raw'])}
        title_note = (f'{window_seconds:g} s from event onset; P values are mouse-label '
                      f'randomization of the hierarchical cell-level model')
        footer = (f'n = {n_mice} mice — the unit of treatment assignment and of inference.\n'
                  f'Brackets are Holm-adjusted EXACT mouse-label randomization P-values '
                  f'({HIERARCHICAL_DIRNAME}/tables/hierarchical_permutation.csv), whose model '
                  f'gives each event window its own between-animal variance.\n'
                  f'The shock window overlaps the post-shock window by '
                  f'{max(0.0, window_seconds - 2.0):g} s, because the US is only 2 s long.')
    else:
        # Stated on the figure rather than silently substituted: with the hierarchical lane
        # off, the only brackets available are the ones this lane does not report.
        panel_contrasts, panel_omnibus = contrasts, omnibus
        title_note = (f'{window_seconds:g} s from event onset; P values are the MOUSE-LEVEL '
                      f'cross-check (equal variance across windows) — the hierarchical lane '
                      f'was not run')
        footer = None

    plot_epoch_modulation_superplot(
        df_cell, df_mouse, panel_contrasts, panel_omnibus,
        os.path.join(out_dir, 'panel_N_event_proximal_index.png'),
        signal=signal, auto_close=auto_close, title_note=title_note, footer=footer,
        **panel_kwargs)

    if verbose:
        n_cells_total = df_cell.groupby(['mouse', 'cell'], observed=True).ngroups
        print(f'[event-proximal/{signal}] {n_mice} mice, {n_cells_total} cells, '
              f'{window_seconds:g} s windows -> {out_dir}', flush=True)
    return {'df_cell': df_cell, 'df_mouse': df_mouse, 'coverage': coverage,
            'contrasts': contrasts, 'omnibus': omnibus, 'dilution': dilution,
            'permutation': perm_df}


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch_modulation(PLOTS_DIR, mice_per_group, TFC_cond, mapping='full',
                         crossreg_to_use=None, auto_close=True,
                         run_hierarchical_cells=True,
                         hierarchical_cross_check_full_fit=HIERARCHICAL_CROSS_CHECK_FULL_FIT,
                         run_event_proximal=True):
    """Run the per-cell epoch modulation analysis (Figure 2 single-cell block, panels K-L).

    PLOTS_DIR       : output root
    mice_per_group  : dict group -> [mouse]
    TFC_cond        : dict mouse -> conditioning session
    mapping         : cross-registration mapping string ('full' = all cells of the session)
    crossreg_to_use : dict mouse -> CrossRegMapping (required when mapping != 'full')

    run_hierarchical_cells : run the hierarchical cell-level companion per signal
                    (run_hierarchical_cell_lane), into ``<signal>/hierarchical_cells/`` beside
                    this analysis's own panels. True by default, so a normal pipeline run
                    produces it. It is a COMPANION: it changes nothing above it, and the
                    mouse-level analysis remains the paper-facing one and is complete without it.
                    Roughly a minute per signal. Pass False (or
                    cfg.epoch_modulation_hierarchical_cells = False) to skip it.

    run_event_proximal : run the event-proximal companion per signal (run_event_proximal_lane),
                    into ``<signal>/event_proximal/``, beside the hierarchical companion. True by
                    default. It recomputes the same index over EVENT_PROXIMAL_WINDOW_S from each
                    event onset to test whether the 20 s epoch means dilute a brief event-locked
                    response; it changes nothing above it. It carries its own hierarchical lane,
                    so it roughly doubles the run's cost. Pass False (or
                    cfg.epoch_modulation_event_proximal = False) to skip it.

    hierarchical_cross_check_full_fit : also refit the whole model through statsmodels on every
                    cell row and compare it against the reported collapsed fit. False by default
                    because it costs 4-14 min per signal and is a corroboration rather than the
                    guarantee -- see HIERARCHICAL_CROSS_CHECK_FULL_FIT.

    Produces, under ``<PLOTS_DIR>/epoch_modulation/``:
      <signal>/panel_K_tone_aligned_heatmaps.png/.svg   (both signals)
      <signal>/panel_L_epoch_modulation.png/.svg        (both signals)
      <signal>/tables/{per_cell,per_mouse,coverage,contrasts}.csv
      <signal>/stats/{model_summary,contrasts}.txt
      signal_comparison.txt
      epoch_modulation_methods.md
    and, when run_hierarchical_cells:
      <signal>/hierarchical_cells/hierarchical_epoch_modulation.png/.svg
      <signal>/hierarchical_cells/tables/hierarchical_permutation.csv
      <signal>/hierarchical_cells/stats/{hierarchical_contrasts,hierarchical_model_summary}.txt
      <signal>/hierarchical_cells/stats/hierarchical_vs_mouse_level_summary.md
      hierarchical_cells/signal_comparison.{txt,csv}
      epoch_modulation_hierarchical_cells_methods.md
    """
    out_dir = os.path.join(PLOTS_DIR, 'epoch_modulation')
    ensure_dirs(out_dir)
    _copy_analysis_methods_template(METHODS_FILENAME, out_dir)

    contrasts_by_signal, omnibus_by_signal, hier_by_signal = {}, {}, {}
    event_prox_by_signal = {}
    if run_hierarchical_cells:
        _copy_analysis_methods_template(HIERARCHICAL_METHODS_FILENAME, out_dir)
    if run_event_proximal:
        _copy_analysis_methods_template(EVENT_PROXIMAL_METHODS_FILENAME, out_dir)

    for signal in SIGNALS:
        sig_dir = os.path.join(out_dir, signal)
        tables_dir = os.path.join(sig_dir, 'tables')
        stats_dir = os.path.join(sig_dir, 'stats')
        ensure_dirs(sig_dir, tables_dir, stats_dir)
        print(f'[epoch-modulation/{signal}] building modulation table...', flush=True)

        df, coverage = build_modulation_table(
            mice_per_group, TFC_cond, signal=signal, mapping=mapping,
            crossreg_to_use=crossreg_to_use)
        df_cell = summarize_cells(df)
        df_mouse = summarize_mice(df_cell)
        n_mice = df_mouse['mouse'].nunique()

        result, method, model_text, contrasts, omnibus = fit_mouse_level_model(df_mouse)
        contrasts_by_signal[signal] = contrasts
        omnibus_by_signal[signal] = omnibus

        df_cell.to_csv(os.path.join(tables_dir, 'per_cell_modulation.csv'), index=False)
        df_mouse.to_csv(os.path.join(tables_dir, 'per_mouse_modulation.csv'), index=False)
        coverage.to_csv(os.path.join(tables_dir, 'trial_coverage.csv'), index=False)
        contrasts.to_csv(os.path.join(tables_dir, 'model_contrasts.csv'), index=False)
        write_text(os.path.join(stats_dir, 'model_summary.txt'), model_text)
        write_text(os.path.join(stats_dir, 'contrasts.txt'),
                   format_model_report(contrasts, omnibus, method, n_mice)
                   + format_variance_diagnostic(df_mouse))

        n_cells_total = df_cell.groupby(['mouse', 'cell'], observed=True).ngroups
        print(f'[epoch-modulation/{signal}] {n_mice} mice, {n_cells_total} cells, '
              f'{len(df_mouse)} mouse x epoch rows; group x epoch '
              f'F({omnibus["df1"]},{omnibus["df2"]}) = {omnibus["F"]:.3f}, '
              f'P = {omnibus["p"]:.4g}', flush=True)

        # ── Panels, for EVERY signal ──
        # Both lanes get the full panel set so the per-signal output directories are
        # structurally identical and the confirmatory lane can be inspected the same way as the
        # primary one. An earlier version drew panels only for PRIMARY_SIGNAL and left the
        # supplementary two-signal figure to a disagreement trigger; that made the two
        # directories asymmetric and left the trigger unwired.
        group_mats, group_sort = {g: [] for g in DREADD_DISPLAY_ORDER}, {g: [] for g in DREADD_DISPLAY_ORDER}
        time_axis = None
        for group in DREADD_DISPLAY_ORDER:
            for mouse in mice_per_group.get(group, []):
                if mouse not in TFC_cond:
                    continue
                session = TFC_cond[mouse]
                wc = crossreg_to_use[mouse] if (crossreg_to_use is not None and mapping != 'full') else None
                cell_ids, _ = resolve_shared_cells(session, mapping=mapping, with_crossreg=wc)
                z = _standardized_trace(session, signal, cell_ids)
                trials, _ = retained_trials(session)
                mat, axis = _tone_aligned_matrix(z, session, trials)
                if mat is None:
                    raise RuntimeError(
                        f'epoch_modulation: mouse {mouse!r} produced no tone-aligned window for '
                        f'panel K, although it retained trials {trials}. The alignment window '
                        f'(-{HEATMAP_PRE_S:.0f} s to +{HEATMAP_POST_S:.0f} s) must fit inside the '
                        f'recording for at least one retained trial.')
                # The sort values MUST be put back into the matrix's own row order. `mat` rows
                # follow `cell_ids` order, whereas df_cell came from a groupby and is therefore
                # sorted by cell id. Those two coincide only if the mapping's ids happen to be
                # ascending, which nothing guarantees -- so reindex by cell id explicitly rather
                # than relying on it. A length check alone would NOT catch a permutation.
                sub = (df_cell[(df_cell['mouse'] == mouse) & (df_cell['epoch'] == 'trace')]
                       .set_index('cell'))
                if len(sub) != mat.shape[0]:
                    raise RuntimeError(
                        f'epoch_modulation: mouse {mouse!r} has {len(sub)} trace-index cells but '
                        f'{mat.shape[0]} heatmap rows -- the sort order would be misaligned.')
                missing = [c for c in cell_ids if c not in sub.index]
                if missing:
                    raise RuntimeError(
                        f'epoch_modulation: mouse {mouse!r} is missing trace indices for cells '
                        f'{missing[:10]} -- cannot align the panel K sort to the matrix rows.')
                group_mats[group].append(mat)
                group_sort[group].append(sub.loc[list(cell_ids), 'index'].to_numpy())
                time_axis = axis

        group_mats = {g: np.concatenate(v, axis=0) for g, v in group_mats.items() if v}
        group_sort = {g: np.concatenate(v, axis=0) for g, v in group_sort.items() if v}

        plot_modulation_heatmaps(group_mats, group_sort, time_axis,
                                 os.path.join(sig_dir, 'panel_K_tone_aligned_heatmaps.png'),
                                 signal=signal, auto_close=auto_close)
        plot_epoch_modulation_superplot(df_cell, df_mouse, contrasts, omnibus,
                                        os.path.join(sig_dir, 'panel_L_epoch_modulation.png'),
                                        signal=signal, auto_close=auto_close)

        # ── Hierarchical cell-level companion, only when explicitly asked for ──
        # It runs on the SAME df_cell the mouse-level lane just fit, so the two analyses cannot
        # describe different cells or different trials.
        if run_hierarchical_cells:
            hier_by_signal[signal] = run_hierarchical_cell_lane(
                sig_dir, signal, df_cell, df_mouse, mice_per_group, contrasts, omnibus,
                auto_close=auto_close,
                cross_check_full_fit=hierarchical_cross_check_full_fit)['permutation']

        # ── Event-proximal companion, on the same cells and trials ──
        # Run here rather than from its own entry point so it is fit on the SAME resolved cell
        # set and retained trials as the lane above, and so the full-epoch per-mouse endpoint
        # its dilution comparison is paired against is the one just computed rather than a
        # second, separately rebuilt copy of it.
        if run_event_proximal:
            event_prox_by_signal[signal] = run_event_proximal_lane(
                sig_dir, signal, mice_per_group, TFC_cond, df_mouse,
                mapping=mapping, crossreg_to_use=crossreg_to_use, auto_close=auto_close,
                run_hierarchical_cells=run_hierarchical_cells,
                hierarchical_cross_check_full_fit=hierarchical_cross_check_full_fit)

    merged, comparison_text = compare_signals(contrasts_by_signal, omnibus_by_signal)
    merged.to_csv(os.path.join(out_dir, 'signal_comparison.csv'), index=False)
    write_text(os.path.join(out_dir, 'signal_comparison.txt'), comparison_text)
    print(comparison_text, flush=True)

    out = {'contrasts': contrasts_by_signal, 'omnibus': omnibus_by_signal,
           'signal_comparison': merged}
    if run_hierarchical_cells:
        hier_merged, hier_text = compare_hierarchical_signals(hier_by_signal)
        hier_dir = os.path.join(out_dir, HIERARCHICAL_DIRNAME)
        ensure_dirs(hier_dir)
        hier_merged.to_csv(os.path.join(hier_dir, 'signal_comparison.csv'), index=False)
        write_text(os.path.join(hier_dir, 'signal_comparison.txt'), hier_text)
        print(hier_text, flush=True)
        out['hierarchical'] = hier_by_signal
        out['hierarchical_signal_comparison'] = hier_merged

    if run_event_proximal:
        ep_dir = os.path.join(out_dir, EVENT_PROXIMAL_DIRNAME)
        ensure_dirs(ep_dir)
        out['event_proximal'] = event_prox_by_signal
        if run_hierarchical_cells:
            # The lane's own two-signal agreement, on the same fixed criterion the hierarchical
            # companion uses (both CIs contain zero, OR matching sign with overlapping CIs).
            ep_merged, ep_text = compare_hierarchical_signals(
                {s: r['permutation'] for s, r in event_prox_by_signal.items()})
            ep_merged.to_csv(os.path.join(ep_dir, 'signal_comparison.csv'), index=False)
            write_text(os.path.join(ep_dir, 'signal_comparison.txt'), ep_text)
            print(ep_text, flush=True)
            out['event_proximal_signal_comparison'] = ep_merged
        dilution = pd.concat(
            [r['dilution'].assign(signal=s) for s, r in event_prox_by_signal.items()],
            ignore_index=True)
        dilution.to_csv(os.path.join(ep_dir, 'dilution_vs_full_epoch.csv'), index=False)
    return out
