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
import time

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from caban import minian_runner as mr
from caban import yra_recompute as yr
from caban.nature_comm_style import MM, FONT_PT

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


def inside_ellipse(height, width, mask):
    """True where (height, width) lies inside the ellipse ``mask`` (minian_runner.SEED_MASK_KEYS)."""
    return (((np.asarray(height) - mask['center_height']) / mask['radius_height']) ** 2
            + ((np.asarray(width) - mask['center_width']) / mask['radius_width']) ** 2) <= 1


def footprint_centroids(A):
    """(height, width) centroid of each unit's footprint, weighted by footprint value."""
    values = A.values.reshape(A.sizes['unit_id'], -1)
    hh, ww = np.meshgrid(A.coords['height'].values, A.coords['width'].values, indexing='ij')
    weights = values / values.sum(axis=1, keepdims=True)
    return weights @ hh.ravel(), weights @ ww.ravel()


def plot_seed_mask(max_proj, A, mask, title, out_path):
    """``max_proj`` with the ellipse, and each unit's centroid: kept (inside) or not (outside)."""
    height, width = footprint_centroids(A)
    inside = inside_ellipse(height, width, mask)
    fig, ax = plt.subplots(figsize=(120 * MM, 120 * MM))
    low, high = np.percentile(max_proj, PROJECTION_PERCENTILES)
    ax.imshow(max_proj, cmap='gray', vmin=low, vmax=high, interpolation='nearest')
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(mask['center_width'] + mask['radius_width'] * np.cos(theta),
            mask['center_height'] + mask['radius_height'] * np.sin(theta), color='C1', lw=0.75)
    ax.scatter(width[inside], height[inside], s=6, facecolor='none', edgecolor='C0', lw=0.5)
    ax.scatter(width[~inside], height[~inside], s=6, marker='x', color='C3', lw=0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('{}: {} units inside (o), {} outside (x)'.format(title, int(inside.sum()),
                                                                  int((~inside).sum())), fontsize=FONT_PT)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


# Lens-interior ellipses proposed for the parameter check, before VS approves them into
# session_queue.SEED_MASKS (which production runs read). Drawn over the baseline max_proj.
SEED_MASKS_PROPOSED = {
    'G27-ST895-hM4D-DIOGC/2023_05_22-TFC_cond/15_21_41-HC2': {
        'center_height': 335.0, 'center_width': 300.0, 'radius_height': 265.0, 'radius_width': 285.0},
}


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


def close_pairs(height, width, max_dist=FRAGMENT_DISTANCE_PX):
    """Number of unit pairs whose centroids lie within ``max_dist`` px of each other."""
    points = np.column_stack([height, width])
    dist = np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(-1))
    return int(np.triu(dist < max_dist, k=1).sum())


def run_parameter_variant(session_dir, name, overrides, out_root, seed_mask=None, n_workers=6):
    """One full Minian run of ``session_dir`` with ``overrides``, written to ``out_root/name``.

    As :func:`caban.minian_runner.reestimate_motion`: ``intpath`` and ``minian_ds_path`` both under
    the variant's folder (verified from the kernel), no videos, the session folder's listing
    checked unchanged. ``del_frames`` come from ``yra_recompute.NOTEBOOK_DEL_FRAMES`` as in
    production. Intermediates are kept for the review. Writes ``variant.json``; raises on failure.
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
                                      parameter_overrides=overrides)
    before = sorted(os.listdir(session_dir))
    record = {'session_dir': session_dir, 'variant': name, 'overrides': overrides,
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


def run_parameter_grid(session_dir, grid, out_root, seed_mask=None, n_parallel=4, n_workers=6):
    """Every variant of ``grid``, ``n_parallel`` at a time. A failed variant is recorded in its
    ``variant.json`` and printed, and the others carry on (an unattended grid; each failure is a
    result to look at, e.g. the empty-merge failure). Returns {name: 'done' | error}."""
    outcome = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_parallel) as pool:
        futures = {pool.submit(run_parameter_variant, session_dir, name, overrides, out_root,
                               seed_mask, n_workers): name for name, overrides in grid.items()}
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
