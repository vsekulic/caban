"""SST DIO-GCaMP6f mice: the DREADDs validated on the SST cells themselves (Figure 2 v2, d).

plans/sst_dio_dreadd_validation_plan.md. GCaMP6f is Cre-dependent in SST-Cre mice, so the imaged
cells are the DREADD-expressing SST cells. This module holds the parameter-check diagnostics first;
the analysis follows once the plan's §8 is fixed.

Raw projections (:func:`raw_projections`) are computed from the session's .avi files directly,
before any Minian step, so the four fields of view can be compared before the baseline Minian runs.
"""

import concurrent.futures
import glob
import json
import os
import re
import threading
import time

import cv2
import numpy as np
import pandas as pd
import scipy.ndimage
import scipy.sparse
import matplotlib.pyplot as plt

from caban import minian_runner as mr
from caban import yra_recompute as yr
from caban.nature_comm_style import MM, FONT_PT
from caban.sst_dio_common import SEED_MASKS_PROPOSED, inside_ellipse, footprint_centroids, unit_metrics

SST_DIO_DREADD = {'G22': 'hM3D', 'G23': 'hM3D', 'G26': 'hM4D', 'G27': 'hM4D'}
SST_DIO_DISPLAY_ORDER = ('G22', 'G23', 'G26', 'G27')     # hM3D then hM4D (no mCherry-DIO control)
SST_DIO_MOUSE_DIRS = {
    'G22': 'G22-ST875-hM3D-DIOGC', 'G23': 'G23-ST856-hM3D-DIOGC',
    'G26': 'G26-ST894-hM4D-DIOGC', 'G27': 'G27-ST895-hM4D-DIOGC',
}
CONDITIONING_DAYS = {
    'G22': '2023_03_20-TFC_cond', 'G23': '2023_03_20-TFC_cond',
    'G26': '2023_05_22-TFC_cond', 'G27': '2023_05_22-TFC_cond',
}
# The parameter check's sessions (plan §6): HC2 of the conditioning day, the last pre-injection homecage.
PARAMETER_CHECK_SESSIONS = {
    'G22': '14_15_20-HC2', 'G23': '16_26_10-HC2', 'G26': '13_23_16-HC2', 'G27': '15_21_41-HC2',
}
OUT_DIRNAME = 'sst_dio_validation'
PROJECTION_PERCENTILES = (0.5, 99.5)   # display range of each projection image


def parameter_check_session_dir(data_root, mouse):
    """``<data_root>/<mouse dir>/<conditioning day>/<HC2>`` for one DIO mouse."""
    return os.path.join(data_root, SST_DIO_MOUSE_DIRS[mouse], CONDITIONING_DAYS[mouse],
                        PARAMETER_CHECK_SESSIONS[mouse])


def session_videos(session_dir):
    """The session's Miniscope videos ``0.avi, 1.avi, ...`` in numeric order; hard-fails on a gap."""
    paths = glob.glob(os.path.join(session_dir, 'Miniscope', '*.avi'))
    numbered = sorted((int(m.group(1)), p) for p in paths
                      for m in [re.match(r'^(\d+)\.avi$', os.path.basename(p))] if m)
    if not numbered:
        raise FileNotFoundError('no numbered .avi in {}/Miniscope'.format(session_dir))
    numbers = [n for n, _ in numbered]
    if numbers != list(range(len(numbers))):
        raise ValueError('{}: video numbers are not 0..{}: {}'.format(session_dir, len(numbers) - 1,
                                                                      numbers))
    return [p for _, p in numbered]


def iter_frames(session_dir):
    """Every raw frame of the session, in order, as float64 (Miniscope v4 grayscale, read by OpenCV
    as three equal channels). The frame index is the position in this sequence, which is Minian's
    ``frame`` coordinate for the same session."""
    for path in session_videos(session_dir):
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            raise IOError('cannot open {}'.format(path))
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            yield frame[:, :, 0].astype(np.float64)
        capture.release()


def raw_projections(session_dir):
    """Mean, max - mean and standard-deviation images over every frame of the session's raw video.

    Streamed one frame at a time (float64 accumulators). ``max - mean`` removes the static
    vignetting and background that dominate a raw max projection, so transiently bright cells
    stand out; it is a preview of the field, not Minian's (motion-corrected) ``max_proj``.
    """
    total = total_sq = peak = None
    n_frames = 0
    for frame in iter_frames(session_dir):
        if total is None:
            total, total_sq, peak = np.zeros_like(frame), np.zeros_like(frame), frame.copy()
        total += frame
        total_sq += frame ** 2
        np.maximum(peak, frame, out=peak)
        n_frames += 1
    if n_frames == 0:
        raise ValueError('{}: no frames decoded'.format(session_dir))
    mean = total / n_frames
    std = np.sqrt(np.maximum(total_sq / n_frames - mean ** 2, 0.0))
    return {'mean': mean, 'max_minus_mean': peak - mean, 'std': std, 'n_frames': n_frames}


def plot_projection_grid(projections, out_path):
    """One row per mouse (:data:`SST_DIO_DISPLAY_ORDER`), columns mean | max - mean | s.d."""
    columns = (('mean', 'Mean'), ('max_minus_mean', 'Max − mean'), ('std', 'S.d.'))
    mice = [m for m in SST_DIO_DISPLAY_ORDER if m in projections]
    fig, axes = plt.subplots(len(mice), len(columns), figsize=(180 * MM, 60 * MM * len(mice)),
                             squeeze=False)
    for row, mouse in enumerate(mice):
        for col, (key, title) in enumerate(columns):
            image = projections[mouse][key]
            low, high = np.percentile(image, PROJECTION_PERCENTILES)
            ax = axes[row, col]
            ax.imshow(image, cmap='gray', vmin=low, vmax=high, interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(title, fontsize=FONT_PT)
            if col == 0:
                ax.set_ylabel('{} {}\n{} frames, {}×{} px'.format(
                    mouse, SST_DIO_DREADD[mouse], projections[mouse]['n_frames'],
                    image.shape[1], image.shape[0]), fontsize=FONT_PT)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


# --- line artefacts: whole bright rows from the Miniscope's connection drop-outs ---------------
# VS's practice in the hand-run notebooks (template cell "CUSTOM SUBSETS FOR FRAMES"): a per-session
# list of the frames carrying such lines, dropped from varr_ref before motion correction, frame
# labels kept. Here the frames are found by score, then every flagged frame is shown to VS.
ROW_ARTEFACT_Z = 12.0           # flag a frame when one row stands out this many robust s.d. (proposal)
ROW_ARTEFACT_SHEET_FRAMES = 40  # frames per contact sheet


def row_profiles(session_dir):
    """Per-frame row means, shape (n_frames, height), and each frame's row-mean contrast against its
    two neighbouring rows -- a thin bright line is a single-row spike in that contrast."""
    profiles = np.array([frame.mean(axis=1) for frame in iter_frames(session_dir)])
    if profiles.size == 0:
        raise ValueError('{}: no frames decoded'.format(session_dir))
    contrast = profiles[:, 1:-1] - 0.5 * (profiles[:, :-2] + profiles[:, 2:])
    return profiles, contrast


def row_artefact_scores(contrast):
    """Robust z of each row's contrast against the same row over the whole recording.

    Steady features (the lens edge, a static scratch) have a high contrast in every frame and so a
    high median, not a high z; a line present in a few frames does. Returns z (n_frames, height - 2)
    and each frame's maximum z.
    """
    median = np.median(contrast, axis=0)
    spread = 1.4826 * np.median(np.abs(contrast - median), axis=0)
    if np.any(spread == 0):
        raise ValueError('rows with zero spread over frames: {}'.format(np.where(spread == 0)[0][:10]))
    z = (contrast - median) / spread
    return z, z.max(axis=1)


def plot_row_artefact_scores(frame_max_z, threshold, title, out_path):
    """Each frame's maximum row z over the recording, the threshold, and the flagged count."""
    flagged = np.flatnonzero(frame_max_z > threshold)
    fig, ax = plt.subplots(figsize=(180 * MM, 50 * MM))
    ax.plot(frame_max_z, lw=0.5, color='0.3')
    ax.axhline(threshold, color='C3', lw=0.75)
    ax.set_yscale('log')
    ax.set_xlabel('Frame', fontsize=FONT_PT)
    ax.set_ylabel('Max row z', fontsize=FONT_PT)
    ax.set_title('{}: {} of {} frames above z = {:g}'.format(title, len(flagged), len(frame_max_z),
                                                            threshold), fontsize=FONT_PT)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_frame_sheet(session_dir, frames, frame_max_z, title, out_path, n_columns=8):
    """The listed raw frames, each labelled with its index and max row z, for a frame-by-frame look."""
    wanted = set(int(f) for f in frames)
    images = {i: frame for i, frame in enumerate(iter_frames(session_dir)) if i in wanted}
    missing = wanted - set(images)
    if missing:
        raise ValueError('{}: frames not in the video: {}'.format(session_dir, sorted(missing)[:10]))
    n_rows = int(np.ceil(len(frames) / n_columns))
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(180 * MM, 24 * MM * n_rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis('off')
    for ax, frame in zip(axes.ravel(), frames):
        image = images[int(frame)]
        low, high = np.percentile(image, PROJECTION_PERCENTILES)
        ax.imshow(image, cmap='gray', vmin=low, vmax=high, interpolation='nearest')
        ax.set_title('{} (z {:.0f})'.format(int(frame), frame_max_z[int(frame)]), fontsize=5)
    fig.suptitle(title, fontsize=FONT_PT)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


# --- lens-interior seed mask (session_queue.SEED_MASKS) -----------------------------------------


def plot_seed_mask(max_proj, A, mask, title, out_path, raw_mean=None):
    """``max_proj`` with the ellipse, and each unit's centroid: kept (inside) or not (outside).
    ``A`` is None for a session with no units yet (only the ellipse is drawn). With ``raw_mean``
    (the raw movie's mean, :func:`raw_projections`), the same ellipse over it in a second panel:
    the lens edge shows there when the max projection hides it."""
    panels = 1 if raw_mean is None else 2
    fig, axes = plt.subplots(1, panels, figsize=(120 * MM * panels, 120 * MM), squeeze=False)
    theta = np.linspace(0, 2 * np.pi, 400)
    for ax, image in zip(axes[0], (max_proj, raw_mean)):
        low, high = np.percentile(image, PROJECTION_PERCENTILES)
        ax.imshow(image, cmap='gray', vmin=low, vmax=high, interpolation='nearest')
        ax.plot(mask['center_width'] + mask['radius_width'] * np.cos(theta),
                mask['center_height'] + mask['radius_height'] * np.sin(theta), color='C1', lw=0.75)
        ax.set_xticks([])
        ax.set_yticks([])
    if raw_mean is not None:
        axes[0, 1].set_title('raw mean (lens edge)', fontsize=FONT_PT)
    ax = axes[0, 0]
    if A is None:
        ax.set_title('{}: no units yet'.format(title), fontsize=FONT_PT)
    else:
        height, width = footprint_centroids(A)
        inside = inside_ellipse(height, width, mask)
        ax.scatter(width[inside], height[inside], s=6, facecolor='none', edgecolor='C0', lw=0.5)
        ax.scatter(width[~inside], height[~inside], s=6, marker='x', color='C3', lw=0.5)
        ax.set_title('{}: {} units inside (o), {} outside (x)'.format(title, int(inside.sum()),
                                                                      int((~inside).sum())), fontsize=FONT_PT)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path



# --- parameter grids (plan §6): full Minian runs outside the session folder ----------------------
# One value changed at a time from the frozen set; every variant gets the same seed mask and the
# del_frames production would get. VS judges them in the review notebook.
PARAMETER_GRID_ONE_AT_A_TIME = {
    'frozen': {},
    'background_wnd_10': {'param_background_removal': {'wnd': 10}},
    'background_wnd_20': {'param_background_removal': {'wnd': 20}},
    'seeds_max_wnd_10': {'param_seeds_init': {'max_wnd': 10}},
    'seeds_max_wnd_20': {'param_seeds_init': {'max_wnd': 20}},
    'pnr_thres_1_5': {'param_pnr_refine': {'thres': 1.5}},
    'pnr_thres_2': {'param_pnr_refine': {'thres': 2.0}},
    'seeds_merge_dist_5': {'param_seeds_merge': {'thres_dist': 5}},
    'seeds_merge_dist_15': {'param_seeds_merge': {'thres_dist': 15}},
    'merge_corr_0_7': {name: {'thres_corr': 0.7} for name in
                       ('param_seeds_merge', 'param_init_merge', 'param_first_merge')},
    'merge_corr_0_9': {name: {'thres_corr': 0.9} for name in
                       ('param_seeds_merge', 'param_init_merge', 'param_first_merge')},
    'spatial_penal_1e-3': {name: {'sparse_penal': 1e-3} for name in
                           ('param_first_spatial', 'param_second_spatial')},
    'spatial_penal_1e-5': {name: {'sparse_penal': 1e-5} for name in
                           ('param_first_spatial', 'param_second_spatial')},
    'spatial_dl_wnd_8': {name: {'dl_wnd': 8} for name in
                         ('param_first_spatial', 'param_second_spatial')},
}
GRID_NOTEBOOK_FLAGS = dict(mr.NOTEBOOK_FLAGS, want_mc_video=False, want_final_video=False)
FRAGMENT_DISTANCE_PX = 8.0      # two units' centroids closer than this: likely one cell split
KERNEL_START_SPACING_S = 30.0   # between two variants' notebook starts


def close_pairs(height, width, max_dist=FRAGMENT_DISTANCE_PX):
    """Number of unit pairs whose centroids lie within ``max_dist`` px of each other."""
    points = np.column_stack([height, width])
    dist = np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(-1))
    return int(np.triu(dist < max_dist, k=1).sum())


def run_parameter_variant(session_dir, name, overrides, out_root, seed_mask=None, n_workers=6,
                          cut_after=None):
    """One Minian run of ``session_dir`` with ``overrides``, written to ``out_root/name``.

    As :func:`caban.minian_runner.reestimate_motion`: ``intpath`` and ``minian_ds_path`` both under
    the variant's folder (verified from the kernel), no videos, the session folder's listing
    checked unchanged. ``del_frames`` come from ``yra_recompute.NOTEBOOK_DEL_FRAMES`` as in
    production. With ``cut_after`` (:data:`STAGE_CUT_AFTER`) the run stops after that stage and
    has no units to report. Intermediates are kept for the review. Writes ``variant.json``; raises
    on failure.
    """
    work = os.path.join(out_root, name)
    if os.path.lexists(work):
        raise FileExistsError('{} exists; remove it to re-run the variant'.format(work))
    intpath, output = os.path.join(work, 'minian_intermediate'), os.path.join(work, mr.OUTPUT_NAME)
    os.makedirs(intpath)
    os.makedirs(output)
    del_frames = yr.notebook_del_frames(session_dir)
    nb, edits = mr.build_run_notebook(mr.load_template(), session_dir,
                                      os.path.join(work, mr.PARAMETERS_JSON), scratch=intpath,
                                      flags=GRID_NOTEBOOK_FLAGS, minian_ds_path=output,
                                      del_frames=del_frames, seed_mask=seed_mask,
                                      parameter_overrides=overrides, cut_after=cut_after)
    before = sorted(os.listdir(session_dir))
    record = {'session_dir': session_dir, 'variant': name, 'overrides': overrides, 'cut_after': cut_after,
              'seed_mask': seed_mask, 'del_frames': del_frames, 'notebook_edits': edits,
              'caban': mr._git_state(mr.REPO_DIR), 'minian_fork': mr._git_state(mr.MINIAN_FORK_DIR),
              'started': pd.Timestamp.now().isoformat(timespec='seconds'), 'timings': {}}
    started = time.time()
    try:
        mr.execute_and_check(nb, work, record, n_workers=n_workers)
        with open(os.path.join(work, mr.PARAMETERS_JSON)) as fh:
            parameters = json.load(fh)
        mr._check_edits_held(parameters, session_dir, GRID_NOTEBOOK_FLAGS, intpath, output)
        mr.overrides_held(parameters, overrides)
        if sorted(os.listdir(session_dir)) != before:
            raise RuntimeError('the run changed {}'.format(session_dir))
        record['parameters'] = parameters['param']
        if cut_after is None:
            record['report'] = mr.report_session(output, work, parameters['FRAMERATE'])
            A = yr.open_minian_array(output, 'A').compute()
            height, width = footprint_centroids(A)
            record['report']['close_pairs'] = close_pairs(height, width)
        record['status'] = 'done'
    except BaseException as error:
        record['status'] = 'failed'
        record['error'] = '{}: {}'.format(type(error).__name__, error)
        raise
    finally:
        record['timings']['total_s'] = round(time.time() - started, 1)
        with open(os.path.join(work, 'variant.json'), 'w') as fh:
            json.dump(record, fh, indent=1, default=str)
    return record


def run_parameter_grid(session_dir, grid, out_root, seed_mask=None, n_parallel=4, n_workers=6,
                       cut_after=None):
    """Every variant of ``grid``, ``n_parallel`` at a time. A failed variant is recorded in its
    ``variant.json`` and printed, and the others carry on (an unattended grid; each failure is a
    result to look at, e.g. the empty-merge failure). Returns {name: 'done' | error}."""
    outcome = {}
    start_lock, last_start = threading.Lock(), [0.0]

    def staggered(name, overrides):
        # Kernels started in the same second can die before replying (seen 2026-10-21), so
        # starts are spaced by KERNEL_START_SPACING_S.
        with start_lock:
            wait = last_start[0] + KERNEL_START_SPACING_S - time.time()
            if wait > 0:
                time.sleep(wait)
            last_start[0] = time.time()
        return run_parameter_variant(session_dir, name, overrides, out_root, seed_mask, n_workers,
                                     cut_after)

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_parallel) as pool:
        futures = {pool.submit(staggered, name, overrides): name for name, overrides in grid.items()}
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            error = future.exception()
            outcome[name] = 'done' if error is None else '{}: {}'.format(type(error).__name__, error)
            print('{} {}: {}'.format(pd.Timestamp.now().strftime('%H:%M:%S'), name, outcome[name]),
                  flush=True)
    return outcome


def grid_summary(out_root):
    """One row per variant: status, unit count, footprint size, close pairs, run time."""
    rows = []
    for path in sorted(glob.glob(os.path.join(out_root, '*', 'variant.json'))):
        with open(path) as fh:
            r = json.load(fh)
        report = r.get('report', {})
        rows.append({'variant': r['variant'], 'status': r['status'],
                     'n_units': report.get('n_units'),
                     'footprint_px_median': report.get('footprint_px', {}).get('median'),
                     'close_pairs': report.get('close_pairs'),
                     'minutes': round(r['timings']['total_s'] / 60, 1),
                     'error': r.get('error', '')})
    return pd.DataFrame(rows).set_index('variant')


# --- staged parameter walk (VS, 2026-10-22): one pipeline stage at a time, judged at its own ------
# output, decided before the next stage's grid runs on top of it. The one-at-a-time grid above
# stays as a sensitivity map.
PIPELINE_STAGES = ('preprocessing', 'seeds', 'initialisation', 'cnmf')
# Seeds and initialisation (stages 2-3) are explored in Minian's env from saved movies
# (caban.sst_dio_seed_grid, caban.sst_dio_init_grid); stage 4 runs the whole notebook (None: no cut).
STAGE_CUT_AFTER = {'preprocessing': mr.MOTION_CUT_ANCHOR, 'cnmf': None}
PREPROCESSING_KSIZES = (3, 5, 7)
PREPROCESSING_WNDS = (10, 15, 20, 25, 30)
CNMF_FIRST_MERGE_THRES_CORRS = (0.6, 0.7, 0.8)
CNMF_DL_WNDS = (5, 8)
STAGE_GRIDS = {
    'preprocessing': {'ksize_{}_wnd_{}'.format(k, w): {'param_denoise': {'ksize': k},
                                                       'param_background_removal': {'wnd': w}}
                      for k in PREPROCESSING_KSIZES for w in PREPROCESSING_WNDS},
    # The merge after the first CNMF update, and the spatial updates' dilation window (both).
    'cnmf': {'first_merge_{}_dl_wnd_{}'.format(str(m).replace('.', '_'), d): {
                 'param_first_merge': {'thres_corr': m},
                 'param_first_spatial': {'dl_wnd': d}, 'param_second_spatial': {'dl_wnd': d}}
             for m in CNMF_FIRST_MERGE_THRES_CORRS for d in CNMF_DL_WNDS},
}
# VS's decisions, stage by stage (the parameter log), carried into every later stage's grid.
# Explicit even where a value equals the frozen set, so the record says it was decided.
DECIDED_OVERRIDES = {
    # preprocessing, VS 2026-10-22 (G27 HC2): same as the pyramidal cohort.
    'param_denoise': {'ksize': 5},
    'param_background_removal': {'wnd': 15},
    # seeds, VS 2026-10-22: same as the pyramidal cohort (plus the lens mask, applied per session).
    'param_seeds_init': {'max_wnd': 15},
    'param_pnr_refine': {'thres': 1.0},
    'param_ks_refine': {'sig': 0.05},
    'param_seeds_merge': {'thres_dist': 10, 'thres_corr': 0.8},
    # initialisation and its merge, VS 2026-10-22: same as the pyramidal cohort.
    'param_initialize': {'thres_corr': 0.8, 'wnd': 10},
    'param_init_merge': {'thres_corr': 0.8},
}
# Reference cells for the stage measurements: the units of the G27 grid's merge_corr_0_7 run
# (inside the lens mask, the cleanest separation of the one-at-a-time grid). Only their positions
# are used, the same for every variant, so the variants are compared on the same cells.
REFERENCE_UNITS_VARIANT = 'merge_corr_0_7'
CELL_DISC_RADIUS_PX = 3.0
ANNULUS_PX = (12.0, 18.0)
NEIGHBOUR_EXCLUSION_PX = 8.0    # annulus pixels this close to another reference cell are left out
BACKGROUND_SMOOTH_SIGMA_PX = 10.0
FRAME_BLOCK = 500


def merged_overrides(*parts):
    """Overrides combined left to right, per parameter and key (later wins)."""
    out = {}
    for part in parts:
        for name, values in part.items():
            out.setdefault(name, {}).update(values)
    return out


def reference_cells(grid_root, variant=REFERENCE_UNITS_VARIANT):
    """(height, width) centroids of the reference units."""
    A = yr.open_minian_array(os.path.join(grid_root, variant, mr.OUTPUT_NAME), 'A').compute()
    return footprint_centroids(A)


def _region_weights(cells, shape, seed_mask):
    """Sparse row-normalised weights: one disc row and one annulus row per cell (pixels flattened),
    and the background pixels (inside the mask, beyond the annulus of every cell)."""
    hh, ww = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    height, width = cells
    dist = np.sqrt((hh[None] - np.asarray(height)[:, None, None]) ** 2
                   + (ww[None] - np.asarray(width)[:, None, None]) ** 2)
    if len(height) < 2:
        raise ValueError('need at least two reference cells')
    nearest = np.partition(dist, 1, axis=0)[:2]      # the two smallest distances at each pixel
    rows = []
    for i in range(len(height)):
        disc = dist[i] <= CELL_DISC_RADIUS_PX
        others = np.where(dist[i] <= nearest[0], nearest[1], nearest[0])   # nearest *other* cell
        ring = (dist[i] >= ANNULUS_PX[0]) & (dist[i] <= ANNULUS_PX[1]) & (others > NEIGHBOUR_EXCLUSION_PX)
        if not disc.any() or not ring.any():
            raise ValueError('reference cell {} has an empty disc or annulus'.format(i))
        rows += [disc.ravel(), ring.ravel()]
    weights = scipy.sparse.csr_matrix(np.array(rows, dtype=np.float32))
    weights = scipy.sparse.diags(1.0 / np.asarray(weights.sum(axis=1)).ravel()) @ weights
    background = (nearest[0] > ANNULUS_PX[1]) & inside_ellipse(hh, ww, seed_mask)
    return weights, background, dist


def preprocessing_metrics(variant_dir, cells, seed_mask):
    """Stage-1 measurements of one variant on the reference cells (see the section comment above).

    contrast_z: median over cells of (disc - annulus) in max_proj, in background s.d.;
    transient_snr: median over cells of the 99.9th-percentile excursion of (disc - annulus) trace
    of Y_fm_chk over its noise (robust s.d. of frame differences / sqrt 2);
    residual_background: s.d. of the smoothed max_proj over background pixels / median cell
    contrast (structure left that is not cells); core_fwhm_px: full width at half maximum of the
    mean radial profile (max_proj, annulus subtracted).
    """
    max_proj = yr.open_minian_array(os.path.join(variant_dir, mr.OUTPUT_NAME), 'max_proj').compute().values
    weights, background, dist = _region_weights(cells, max_proj.shape, seed_mask)
    regions = weights @ max_proj.ravel()
    disc, ring = regions[0::2], regions[1::2]
    bg = max_proj[background]
    bg_sd = np.std(bg)    # plain s.d.: max_proj is integer-valued, so its MAD can be 0
    if bg_sd == 0:
        raise ValueError('{}: background of max_proj is flat'.format(variant_dir))
    cell_contrast = disc - ring
    smooth = scipy.ndimage.gaussian_filter(max_proj, BACKGROUND_SMOOTH_SIGMA_PX)

    Y = yr.open_minian_array(os.path.join(variant_dir, 'minian_intermediate'), 'Y_fm_chk')
    n_frames = Y.sizes['frame']
    traces = np.empty((weights.shape[0], n_frames), dtype=np.float64)
    for start in range(0, n_frames, FRAME_BLOCK):
        block = Y.isel(frame=slice(start, start + FRAME_BLOCK)).values
        traces[:, start:start + block.shape[0]] = weights @ block.reshape(block.shape[0], -1).T
    signal = traces[0::2] - traces[1::2]
    step = np.diff(signal, axis=1)
    noise = 1.4826 * np.median(np.abs(step - np.median(step, axis=1, keepdims=True)), axis=1) / np.sqrt(2)
    excursion = np.percentile(signal - np.median(signal, axis=1, keepdims=True), 99.9, axis=1)

    radii = np.arange(0, int(ANNULUS_PX[0]))       # bins [r, r + 1) px from the centroid
    profile = np.array([[max_proj[(dist[i] >= r) & (dist[i] < r + 1)].mean() - ring[i]
                         for r in radii] for i in range(len(disc))]).mean(axis=0)
    profile = profile / profile[0]
    below = np.flatnonzero(profile < 0.5)
    if len(below) == 0:
        raise ValueError('{}: the radial profile never falls below half its peak'.format(variant_dir))
    return {'contrast_z': float(np.median(cell_contrast) / bg_sd),
            'transient_snr': float(np.median(excursion / noise)),
            'residual_background': float(np.std(smooth[background]) / np.median(cell_contrast)),
            'core_fwhm_px': float(2 * below[0]),     # in 1-px radial bins: a coarse width
            'n_cells': int(len(disc)), 'n_frames': int(n_frames)}


def plot_preprocessing_grid(stage_root, done, out_path, crop=((180, 480), (150, 450))):
    """The max projection of every finished preprocessing variant (``done``: their names), rows
    ksize, columns wnd, one crop; a variant that did not finish is drawn as a labelled empty panel."""
    (h0, h1), (w0, w1) = crop
    fig, axes = plt.subplots(len(PREPROCESSING_KSIZES), len(PREPROCESSING_WNDS),
                             figsize=(3.2 * len(PREPROCESSING_WNDS), 3.3 * len(PREPROCESSING_KSIZES)),
                             squeeze=False)
    for i, k in enumerate(PREPROCESSING_KSIZES):
        for j, w in enumerate(PREPROCESSING_WNDS):
            name = 'ksize_{}_wnd_{}'.format(k, w)
            ax = axes[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            if name not in done:
                ax.set_title('ksize {}, wnd {}: NOT FINISHED'.format(k, w), fontsize=9, color='C3')
                continue
            image = yr.open_minian_array(os.path.join(stage_root, name, mr.OUTPUT_NAME),
                                         'max_proj').compute().values[h0:h1, w0:w1]
            low, high = np.percentile(image, PROJECTION_PERCENTILES)
            ax.imshow(image, cmap='gray', vmin=low, vmax=high, interpolation='nearest')
            ax.set_title('ksize {}, wnd {}'.format(k, w), fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def cnmf_metrics(variant_dir, cells):
    """sst_dio_common.unit_metrics of a full run's final A."""
    return unit_metrics(yr.open_minian_array(os.path.join(variant_dir, mr.OUTPUT_NAME), 'A').compute(), cells)


def plot_cnmf_metrics(metrics, out_path):
    """Each measurement against the first-merge threshold, one line per dl_wnd."""
    columns = ('n_units', 'recall', 'cells_with_two_or_more', 'fused_units', 'unmatched_units',
               'footprint_px_median')
    fig, axes = plt.subplots(1, len(columns), figsize=(3.8 * len(columns), 3.4))
    for ax, column in zip(axes, columns):
        for dl_wnd in CNMF_DL_WNDS:
            part = metrics[metrics['dl_wnd'] == dl_wnd].sort_values('first_merge')
            ax.plot(part['first_merge'], part[column], marker='o', label='dl_wnd {}'.format(dl_wnd))
        ax.set_xlabel('first-merge thres_corr')
        ax.set_title(column, fontsize=10)
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def plot_preprocessing_metrics(metrics, out_path):
    """Each measurement against wnd, one line per ksize."""
    columns = ('contrast_z', 'transient_snr', 'residual_background', 'core_fwhm_px')
    fig, axes = plt.subplots(1, len(columns), figsize=(4.2 * len(columns), 3.4))
    for ax, column in zip(axes, columns):
        for k in PREPROCESSING_KSIZES:
            part = metrics[metrics['ksize'] == k].sort_values('wnd')
            ax.plot(part['wnd'], part[column], marker='o', label='ksize {}'.format(k))
        ax.set_xlabel('background wnd (px)')
        ax.set_title(column, fontsize=10)
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
