"""Export S and the Miniscope timestamps of every homecage recording on the conditioning day.

    python scripts/export_tfc_day_homecage_s.py <data_root> <out_dir>

For the homecage pre/post-CNO analysis (caban.homecage_cno; plans/figure2_redesign_plan.md §4.1).
Runs on the Razer, where the batch outputs are; imports only what its environment has. For every
G05-G21 ``*-TFC_cond`` day, each named HC* / CNO* recording with a batch ``Miniscope/minian``
output is written to ``<out_dir>/<mouse>__<day>__<session>.npz`` holding

    S          (units, frames) float32, Minian's S
    unit_id    (units,)
    tstamp_ms  (rows,) the Miniscope timeStamps.csv column, every row
    imaged     (rows,) bool, session_queue.imaged_timestamp_rows

S must have exactly one frame per imaged timestamp row, or the export fails. A recording in
session_queue.SESSIONS_NOT_RUN or PARTIAL_OUTPUT_NO_S is reported with its reason, not exported.
"""

import glob
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from caban import session_queue as sq
from caban import yra_recompute as yr

# Recordings whose Minian output is an old partial run with no S, listed by name so any OTHER missing S
# still fails. Reported on every export.
PARTIAL_OUTPUT_NO_S = {
    'G09-ST702_mCherry/2021_11_08-TFC_cond/19_28_33-HC3':
        'old partial output (no S, all-NaN motion), never rerun; after conditioning; VS to decide '
        '(minian_batch_status.md, open items 10)',
}

HOMECAGE_LABEL = re.compile(r'^\d\d_\d\d_\d\d-(HC\d+|CNO\d?)$')
MICE = [f'G{i:02d}' for i in range(5, 22)]


def recordings(data_root):
    out = []
    for mouse in MICE:
        days = glob.glob(os.path.join(data_root, f'{mouse}-*', '*-TFC_cond'))
        if len(days) != 1:
            raise RuntimeError(f'{mouse}: expected one TFC_cond day under {data_root}, found {days}')
        for session_dir in sorted(glob.glob(os.path.join(days[0], '*'))):
            if os.path.isdir(session_dir) and HOMECAGE_LABEL.match(os.path.basename(session_dir)):
                out.append(session_dir)
    return out


def main(data_root, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for session_dir in recordings(data_root):
        tail = sq.session_tail(session_dir)
        if tail in sq.SESSIONS_NOT_RUN:
            print(f'{tail}: not exported -- {sq.SESSIONS_NOT_RUN[tail]}')
            continue
        if tail in PARTIAL_OUTPUT_NO_S:
            print(f'{tail}: not exported -- {PARTIAL_OUTPUT_NO_S[tail]}')
            continue
        minian_dir = os.path.join(session_dir, 'Miniscope', 'minian')
        if not os.path.isdir(minian_dir):
            raise RuntimeError(f'{tail}: no Miniscope/minian output')
        S = yr.open_minian_array(minian_dir, 'S').transpose('unit_id', 'frame')
        ts = pd.read_csv(os.path.join(session_dir, 'Miniscope', 'timeStamps.csv'))
        tstamp = ts.iloc[:, 1].to_numpy(dtype=float)
        imaged = sq.imaged_timestamp_rows(session_dir, len(tstamp))
        if S.sizes['frame'] != int(imaged.sum()):
            raise RuntimeError(f'{tail}: S has {S.sizes["frame"]} frames, timestamps {int(imaged.sum())} '
                               f'imaged rows')
        values = S.values.astype(np.float32)
        if not np.isfinite(values).all():
            raise RuntimeError(f'{tail}: S has non-finite values')
        out = os.path.join(out_dir, tail.replace('/', '__') + '.npz')
        np.savez_compressed(out, S=values, unit_id=S.coords['unit_id'].values, tstamp_ms=tstamp,
                            imaged=imaged)
        print(f'{tail}: {values.shape[0]} units x {values.shape[1]} frames')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
