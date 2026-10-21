"""Find the frames with whole bright rows (Miniscope connection drop-outs) in the four SST DIO
parameter-check sessions, on the Razer.

    python scripts/sst_dio_row_artefacts.py <data_root> <out_dir>

Per mouse: <mouse>_HC2_row_scores.png (each frame's max row z against the threshold),
<mouse>_HC2_flagged_frames.png (every flagged frame), <mouse>_HC2_near_threshold.png (the highest-
scoring unflagged frames, to check the threshold misses nothing), and <mouse>_HC2_row_scores.npz.
plans/sst_dio_dreadd_validation_plan.md §6.
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from caban.sst_dio_validation import (SST_DIO_DISPLAY_ORDER, ROW_ARTEFACT_Z, ROW_ARTEFACT_SHEET_FRAMES,
                                      parameter_check_session_dir, row_profiles, row_artefact_scores,
                                      plot_row_artefact_scores, plot_frame_sheet)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    data_root, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    for mouse in SST_DIO_DISPLAY_ORDER:
        session_dir = parameter_check_session_dir(data_root, mouse)
        stem = os.path.join(out_dir, '{}_HC2'.format(mouse))
        profiles, contrast = row_profiles(session_dir)
        z, frame_max_z = row_artefact_scores(contrast)
        np.savez_compressed(stem + '_row_scores.npz', profiles=profiles, frame_max_z=frame_max_z)
        flagged = np.flatnonzero(frame_max_z > ROW_ARTEFACT_Z)
        print('{}: {} of {} frames flagged: {}'.format(mouse, len(flagged), len(frame_max_z),
                                                       flagged.tolist()), flush=True)
        plot_row_artefact_scores(frame_max_z, ROW_ARTEFACT_Z, mouse, stem + '_row_scores.png')
        for start in range(0, len(flagged), ROW_ARTEFACT_SHEET_FRAMES):
            part = flagged[start:start + ROW_ARTEFACT_SHEET_FRAMES]
            plot_frame_sheet(session_dir, part, frame_max_z,
                             '{} flagged frames {}-{} of {}'.format(mouse, start + 1, start + len(part),
                                                                    len(flagged)),
                             '{}_flagged_frames_{:03d}.png'.format(stem, start // ROW_ARTEFACT_SHEET_FRAMES))
        unflagged = np.setdiff1d(np.argsort(frame_max_z)[::-1], flagged, assume_unique=True)
        near = np.sort(unflagged[:16])
        plot_frame_sheet(session_dir, near, frame_max_z,
                         '{}: the 16 highest-scoring unflagged frames'.format(mouse),
                         stem + '_near_threshold.png')
