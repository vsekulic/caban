"""SST DIO-GCaMP6f mice: the DREADDs validated on the SST cells themselves (Figure 2 v2, d).

plans/sst_dio_dreadd_validation_plan.md. GCaMP6f is Cre-dependent in SST-Cre mice, so the imaged
cells are the DREADD-expressing SST cells. This module holds the parameter-check diagnostics first;
the analysis follows once the plan's §8 is fixed.

Raw projections (:func:`raw_projections`) are computed from the session's .avi files directly,
before any Minian step, so the four fields of view can be compared before the baseline Minian runs.
"""

import glob
import os
import re

import cv2
import numpy as np
import matplotlib.pyplot as plt

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


def raw_projections(session_dir):
    """Mean, max - mean and standard-deviation images over every frame of the session's raw video.

    Streamed one frame at a time (float64 accumulators). ``max - mean`` removes the static
    vignetting and background that dominate a raw max projection, so transiently bright cells
    stand out; it is a preview of the field, not Minian's (motion-corrected) ``max_proj``.
    """
    total = total_sq = peak = None
    n_frames = 0
    for path in session_videos(session_dir):
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            raise IOError('cannot open {}'.format(path))
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame = frame[:, :, 0].astype(np.float64)   # Miniscope v4 grayscale, read as 3 equal channels
            if total is None:
                total, total_sq, peak = np.zeros_like(frame), np.zeros_like(frame), frame.copy()
            total += frame
            total_sq += frame ** 2
            np.maximum(peak, frame, out=peak)
            n_frames += 1
        capture.release()
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
