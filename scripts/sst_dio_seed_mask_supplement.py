"""Supplementary figure of the lens-interior seed masks (sst_dio_validation.plot_seed_mask_supplement),
on the Razer: per mouse, the HC2 raw mean (scripts/sst_dio_raw_projections.py) and a masked run's
max projection and units, with the ellipse; plus each run's count of seeds the mask removed (the
run notebook's "seed mask:" line), for the caption.

    python scripts/sst_dio_seed_mask_supplement.py <raw_projections_dir> <out_root> <mouse>=<run_dir> ...

<run_dir> is a finished run with the seed mask (a decided-set check run). Output:
<out_root>.svg, <out_root>.png, <out_root>_seeds_removed.csv.
plans/sst_dio_dreadd_validation_plan.md §6.1.
"""

import os
import re
import sys

import matplotlib
import numpy as np
import pandas as pd
matplotlib.use('Agg')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from caban import session_queue as sq
from caban import sst_dio_validation as sdv
from caban import yra_recompute as yr

SEED_MASK_LINE = re.compile(r'seed mask: (\d+) of (\d+) seeds outside the lens interior removed')

if __name__ == '__main__':
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    projections_dir, out_root = sys.argv[1:3]
    panels, rows = {}, []
    for mouse, run_dir in (a.split('=', 1) for a in sys.argv[3:]):
        tail = sq.session_tail(os.path.join(sdv.parameter_check_session_dir('', mouse), 'Miniscope'))
        mask = sq.SEED_MASKS.get(tail, sdv.SEED_MASKS_PROPOSED.get(tail))
        if mask is None:
            raise SystemExit('no seed mask for {}'.format(tail))
        with open(os.path.join(run_dir, 'pipeline.ipynb')) as f:
            found = SEED_MASK_LINE.findall(f.read())
        if len(found) != 1:
            raise ValueError('{}: expected one seed-mask line in pipeline.ipynb, found {}'.format(run_dir, found))
        minian_dir = os.path.join(run_dir, 'minian')
        panels[mouse] = {'raw_mean': np.load(os.path.join(projections_dir, '{}_HC2_projections.npz'.format(mouse)))['mean'],
                         'max_proj': yr.open_minian_array(minian_dir, 'max_proj').compute().values,
                         'A': yr.open_minian_array(minian_dir, 'A').compute(), 'mask': mask}
        rows.append({'mouse': mouse, 'run_dir': run_dir, 'seeds_removed': int(found[0][0]),
                     'seeds': int(found[0][1]), 'units': panels[mouse]['A'].sizes['unit_id'], **mask})
    os.makedirs(os.path.dirname(out_root), exist_ok=True)
    sdv.plot_seed_mask_supplement(panels, out_root)
    table = pd.DataFrame(rows)
    table.to_csv(out_root + '_seeds_removed.csv', index=False)
    print(table.to_string())
