"""Every session of each SST DIO mouse's conditioning day: max - mean and line-artefact frames, on
the Razer (plan §6.1, before the queue). VS judges the lines by eye on the max - mean panels; the
frame-by-frame detection (row_artefact_scores) beside it gives the frames to drop.

    python scripts/sst_dio_day_overview.py <data_root> <out_dir> <mouse> ...

Writes <out_dir>/<mouse>_conditioning_day.png, <mouse>_line_frames.csv (session, frames, flagged
frames, highest unflagged z) and a contact sheet of the flagged frames of every session that has any.
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from caban import session_queue as sq
from caban import sst_dio_validation as sdv

if __name__ == '__main__':
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    data_root, out_dir, mice = sys.argv[1], sys.argv[2], sys.argv[3:]
    os.makedirs(out_dir, exist_ok=True)
    for mouse in mice:
        hc2_tail = sq.session_tail(sdv.parameter_check_session_dir(data_root, mouse))
        seed_mask = sq.SEED_MASKS.get(hc2_tail, sdv.SEED_MASKS_PROPOSED[hc2_tail])
        overviews, rows = {}, []
        for session_dir in sdv.conditioning_day_sessions(data_root, mouse):
            name = os.path.basename(session_dir)
            overview = sdv.session_overview(session_dir)
            overviews[name] = overview
            rows.append({'session': name, 'n_frames': overview['n_frames'],
                         'n_flagged': len(overview['flagged_frames']),
                         'flagged_frames': ' '.join(str(f) for f in overview['flagged_frames']),
                         'max_z_unflagged': round(overview['max_z_unflagged'], 1)})
            print('{} {}: {} frames, {} flagged {}'.format(mouse, name, overview['n_frames'],
                                                          len(overview['flagged_frames']),
                                                          overview['flagged_frames'][:20]), flush=True)
            if overview['flagged_frames']:
                sdv.plot_frame_sheet(session_dir, overview['flagged_frames'][:sdv.ROW_ARTEFACT_SHEET_FRAMES],
                                     overview['frame_max_z'], '{} {}: flagged frames'.format(mouse, name),
                                     os.path.join(out_dir, '{}_{}_flagged_frames.png'.format(mouse, name)))
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, '{}_line_frames.csv'.format(mouse)), index=False)
        print(sdv.plot_day_overview(overviews, mouse, seed_mask,
                                    os.path.join(out_dir, '{}_conditioning_day.png'.format(mouse))), flush=True)
