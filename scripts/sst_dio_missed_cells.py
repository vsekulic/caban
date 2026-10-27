"""Cell-like spots with no final unit in a decided-set check run, followed back through Minian's
seed steps; Minian's env on the Razer (plan §6.1).

    python scripts/sst_dio_missed_cells.py <check_root> <out_root> <n_workers> <mouse> ...

<check_root>/<mouse>_HC2/decided_set/ is a run of scripts/sst_dio_final_set_check.py (its saved
movies, max projection, units and parameters). The seed steps are re-run there at max_wnd 15 and
PNR 1.0 (decided), 0.75 and 0.5. Writes <out_root>/<mouse>_HC2/: spot_fates.csv and
missed_spots.png.
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import xarray as xr

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.expanduser('~/code/minian_vsekulic'))
from caban.sst_dio_common import SEED_MASKS_PROPOSED
from caban import sst_dio_seed_grid as sg

PNR_THRESHOLDS = (1.0, 0.75, 0.5)

if __name__ == '__main__':
    if len(sys.argv) < 5:
        raise SystemExit(__doc__)
    check_root, out_root, n_workers, mice = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4:]
    client = sg.start_cluster(n_workers)
    for mouse in mice:
        run_dir = os.path.join(check_root, '{}_HC2'.format(mouse), 'decided_set')
        tail = [k for k in SEED_MASKS_PROPOSED if k.startswith(mouse + '-') and k.endswith('-HC2')]
        if len(tail) != 1:
            raise ValueError('expected one HC2 seed mask for {}, found {}'.format(mouse, tail))
        seed_mask = SEED_MASKS_PROPOSED[tail[0]]
        out_dir = os.path.join(out_root, '{}_HC2'.format(mouse))
        os.makedirs(out_dir)
        os.environ['MINIAN_INTERMEDIATE'] = os.path.join(out_dir, 'minian_intermediate')
        os.makedirs(os.environ['MINIAN_INTERMEDIATE'])
        tables, max_proj = sg.run_seed_grid(run_dir, out_dir, seed_mask, max_wnds=(15,), thresholds=PNR_THRESHOLDS)
        tables = {name.replace('max_wnd_15_', ''): table for name, table in tables.items()}
        arrays, _ = sg.open_stage1(run_dir)
        A = xr.open_zarr(os.path.join(run_dir, 'minian', 'A.zarr'))['A'].compute()
        spots = sg.candidate_spots(max_proj.values, seed_mask)
        fates = sg.spot_fates(spots, tables, A)
        fates.to_csv(os.path.join(out_dir, 'spot_fates.csv'), index=False)
        print(mouse, fates.to_string(), flush=True)
        print(sg.plot_missed_spots(max_proj.values, spots, fates, arrays['Y_hw_chk'], seed_mask,
                                   '{} HC2, decided set'.format(mouse), os.path.join(out_dir, 'missed_spots.png')))
    client.close()
