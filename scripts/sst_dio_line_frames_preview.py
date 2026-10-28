"""Raw max projection of a mouse's conditioning-day sessions with and without their line-artefact
frames (sst_dio_validation.max_projection_frames_removed), so VS can see the lines are gone, as
in the Minian notebook. On the Razer.

    python scripts/sst_dio_line_frames_preview.py <data_root> <out_png> <mouse> <session>=<frames> ...

<frames>: comma-separated 0-based frames, ranges a-b, or a- (frame a to the end), e.g.
    15_05_08-TFC1=2426,3188,9918-
plans/sst_dio_dreadd_validation_plan.md §6.1.
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from caban import sst_dio_validation as sdv


def parse_frames(spec, n_frames):
    frames = []
    for item in spec.split(','):
        if '-' not in item:
            frames.append(int(item))
            continue
        start, end = item.split('-')
        frames.extend(range(int(start), (int(end) if end else n_frames - 1) + 1))
    return frames


if __name__ == '__main__':
    if len(sys.argv) < 5:
        raise SystemExit(__doc__)
    data_root, out_png, mouse = sys.argv[1:4]
    sessions = {os.path.basename(s): s for s in sdv.conditioning_day_sessions(data_root, mouse)}
    previews = {}
    for name, spec in (a.split('=', 1) for a in sys.argv[4:]):
        session_dir = sessions[name]
        n_frames = sum(1 for _ in sdv.iter_frames(session_dir)) if spec.endswith('-') else None
        frames = parse_frames(spec, n_frames)
        previews[name] = sdv.max_projection_frames_removed(session_dir, frames)
        print('{}: {} of {} frames removed'.format(name, previews[name]['n_removed'], previews[name]['n_frames']), flush=True)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    print(sdv.plot_frames_removed(previews, '{} conditioning day: raw max projection, line frames removed'.format(mouse), out_png))
