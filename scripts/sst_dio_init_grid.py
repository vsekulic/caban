"""Stage 3 (initialisation and its merge) of the staged parameter walk, in Minian's env on the Razer.

    python scripts/sst_dio_init_grid.py <stages_root>/<mouse>_HC2 <grid_root>/<mouse>_HC2 <n_workers>

From stage 1's decided movies and stage 2's decided seeds; writes <stages_root>/<mouse>_HC2/
initialisation/<variant>/ and stage_metrics.{csv,png}. Plan §6.1.
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.expanduser('~/code/minian_vsekulic'))
from caban import sst_dio_seed_grid as sg
from caban import sst_dio_init_grid as ig

DECIDED_PREPROCESSING_VARIANT = 'ksize_5_wnd_15'    # stage 1, VS 2026-10-22
DECIDED_SEED_VARIANT = 'max_wnd_15_pnr_1_0'         # stage 2, VS 2026-10-22

if __name__ == '__main__':
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    stages_root, grid_root, n_workers = sys.argv[1], sys.argv[2], int(sys.argv[3])
    out_dir = os.path.join(stages_root, 'initialisation')
    os.makedirs(out_dir)
    os.environ['MINIAN_INTERMEDIATE'] = os.path.join(out_dir, 'minian_intermediate')
    os.makedirs(os.environ['MINIAN_INTERMEDIATE'])
    client = sg.start_cluster(n_workers)
    arrays, params = sg.open_stage1(os.path.join(stages_root, 'preprocessing', DECIDED_PREPROCESSING_VARIANT))
    seeds = ig.decided_seeds(os.path.join(stages_root, 'seeds', 'seeds_{}.csv'.format(DECIDED_SEED_VARIANT)))
    names = ig.run_init_grid(arrays, params, seeds, out_dir, os.environ['MINIAN_INTERMEDIATE'])
    cells = sg.reference_cells(os.path.join(grid_root, 'merge_corr_0_7'))
    rows = []
    for init_corr in ig.INIT_THRES_CORRS:
        for init_wnd in ig.INIT_WNDS:
            for merge_corr in ig.INIT_MERGE_THRES_CORRS:
                name = ig.variant_name(init_corr, init_wnd, merge_corr)
                rows.append(dict(variant=name, init_corr=init_corr, init_wnd=init_wnd, merge_corr=merge_corr,
                                 **ig.init_metrics(out_dir, name, cells)))
    metrics = pd.DataFrame(rows).set_index('variant')
    metrics.to_csv(os.path.join(out_dir, 'stage_metrics.csv'))
    ig.plot_init_metrics(metrics, os.path.join(out_dir, 'stage_metrics.png'))
    print(metrics.to_string())
    client.close()
