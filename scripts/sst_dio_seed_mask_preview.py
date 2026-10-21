"""Draw a candidate lens-interior ellipse over a session's baseline max_proj, with its units, on the Razer.

    python scripts/sst_dio_seed_mask_preview.py <minian_dir> <title> <out_png> \
        <center_height> <center_width> <radius_height> <radius_width>

The ellipse, once VS approves it, goes into caban.session_queue.SEED_MASKS.
plans/sst_dio_dreadd_validation_plan.md §6.
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from caban import yra_recompute as yr
from caban.minian_runner import SEED_MASK_KEYS
from caban.sst_dio_validation import plot_seed_mask

if __name__ == '__main__':
    if len(sys.argv) != 8:
        raise SystemExit(__doc__)
    minian_dir, title, out_png = sys.argv[1:4]
    mask = dict(zip(SEED_MASK_KEYS, (float(v) for v in sys.argv[4:8])))
    max_proj = yr.open_minian_array(minian_dir, 'max_proj').compute().values
    A = yr.open_minian_array(minian_dir, 'A').compute()
    print(plot_seed_mask(max_proj, A, mask, '{} {}'.format(title, mask), out_png))
