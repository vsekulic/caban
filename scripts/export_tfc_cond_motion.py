"""Export Minian's motion-correction shifts of every conditioning (TFC_cond) session to one .npz.

    python scripts/export_tfc_cond_motion.py <data_root> <out.npz>

For the shock-motion check (caban.shock_motion): the analysis reads each session's production
``Miniscope/minian_crossreg*`` output, and its frame i is that output's frame i, so the shifts are
taken from the same folder. Keys are ``<mouse>/<day>/<session>`` (session_queue.session_tail) with ``/`` written as
``__`` (caban.shock_motion.MOTION_KEY_SEP);
values are (n_frames, 2) arrays of the rigid (height, width) shift in pixels. Each is checked
against its C frame count and for NaN; any mismatch fails.
"""

import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from caban import session_queue as sq
from caban import yra_recompute as yr


def minian_dirs(data_root: str) -> list:
    """The production Minian output of every TFC_cond session under ``data_root``."""
    out = []
    for miniscope in sorted(glob.glob(os.path.join(data_root, 'G*', '*-TFC_cond', '*-TFC_cond',
                                                   'Miniscope'))):
        found = sorted(glob.glob(os.path.join(miniscope, 'minian_crossreg*')))
        if len(found) > 1:
            raise RuntimeError(f'{miniscope}: more than one minian_crossreg* output: {found}')
        if found:
            out.append(found[0])
    if not out:
        raise RuntimeError(f'no TFC_cond minian_crossreg* outputs under {data_root}')
    return out


def main(data_root: str, out_path: str) -> None:
    shifts = {}
    for minian_dir in minian_dirs(data_root):
        key = sq.session_tail(os.path.dirname(os.path.dirname(minian_dir)))
        motion = yr.open_minian_array(minian_dir, 'motion')
        n_c = yr.open_minian_array(minian_dir, 'C').sizes['frame']
        values = motion.transpose('frame', 'shift_dim').values
        if list(motion.coords['shift_dim'].values) != ['height', 'width']:
            raise RuntimeError(f'{key}: unexpected shift_dim {motion.coords["shift_dim"].values}')
        if values.shape[0] != n_c:
            raise RuntimeError(f'{key}: motion has {values.shape[0]} frames, C has {n_c}')
        if np.isnan(values).any():
            raise RuntimeError(f'{key}: motion.zarr has {int(np.isnan(values).sum())} NaN values')
        shifts[key.replace('/', '__')] = values
        print(f'{key}: {values.shape[0]} frames')
    np.savez_compressed(out_path, **shifts)
    print(f'wrote {len(shifts)} sessions -> {out_path}')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
