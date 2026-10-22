"""Stage 2 (seeds) of the staged parameter walk, in Minian's env on the Razer (plan §6.1).

    conda activate minian-native
    python scripts/sst_dio_seed_grid.py <stages_root>/<mouse>_HC2 <grid_root>/<mouse>_HC2 <n_workers>

Starts from <stages_root>/<mouse>_HC2/preprocessing/<decided variant> (stage 1's decision); writes
<stages_root>/<mouse>_HC2/seeds/: seeds_<variant>.csv, seed_steps_<variant>.png, stage_metrics.csv
and stage_metrics.png. Reference cells: <grid_root>/<mouse>_HC2/merge_corr_0_7.
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.expanduser('~/code/minian_vsekulic'))
from caban.sst_dio_common import SEED_MASKS_PROPOSED
from caban import sst_dio_seed_grid as sg

DECIDED_PREPROCESSING_VARIANT = 'ksize_5_wnd_15'    # stage 1, VS 2026-10-22
SESSION_TAILS = {'G27_HC2': 'G27-ST895-hM4D-DIOGC/2023_05_22-TFC_cond/15_21_41-HC2'}

if __name__ == '__main__':
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    stages_root, grid_root, n_workers = sys.argv[1], sys.argv[2], int(sys.argv[3])
    key = os.path.basename(os.path.normpath(stages_root))
    seed_mask = SEED_MASKS_PROPOSED[SESSION_TAILS[key]]
    out_dir = os.path.join(stages_root, 'seeds')
    os.makedirs(out_dir, exist_ok=True)
    client = sg.start_cluster(n_workers)
    tables, max_proj = sg.run_seed_grid(os.path.join(stages_root, 'preprocessing', DECIDED_PREPROCESSING_VARIANT),
                                        out_dir, seed_mask)
    cells = sg.reference_cells(os.path.join(grid_root, 'merge_corr_0_7'))
    rows = []
    for name, table in tables.items():
        max_wnd = int(name.split('_')[2])
        pnr_thres = float(name.split('_pnr_')[1].replace('_', '.'))
        row = dict(variant=name, max_wnd=max_wnd, pnr_thres=pnr_thres, **sg.seed_metrics(table, cells))
        rows.append(row)
        sg.plot_seed_steps(table, max_proj.values, cells, name, os.path.join(out_dir, 'seed_steps_{}.png'.format(name)))
    metrics = pd.DataFrame(rows).set_index('variant')
    metrics.to_csv(os.path.join(out_dir, 'stage_metrics.csv'))
    sg.plot_seed_metrics(metrics, os.path.join(out_dir, 'stage_metrics.png'))
    print(metrics.to_string())
    client.close()
