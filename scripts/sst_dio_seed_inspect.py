"""Draw the unmatched-seed check (caban.sst_dio_seed_grid.plot_unmatched_seeds) for chosen seed-grid
variants, from their saved tables; Minian's env on the Razer (plan §6.1).

    python scripts/sst_dio_seed_inspect.py <stages_root>/<mouse>_HC2 <grid_root>/<mouse>_HC2 <variant> ...

Writes <stages_root>/<mouse>_HC2/seeds/unmatched_<variant>.png.
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

DECIDED_PREPROCESSING_VARIANT = 'ksize_5_wnd_15'    # stage 1, VS 2026-10-22

if __name__ == '__main__':
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    stages_root, grid_root, variants = sys.argv[1], sys.argv[2], sys.argv[3:]
    arrays, _ = sg.open_stage1(os.path.join(stages_root, 'preprocessing', DECIDED_PREPROCESSING_VARIANT))
    cells = sg.reference_cells(os.path.join(grid_root, 'merge_corr_0_7'))
    for variant in variants:
        table = pd.read_csv(os.path.join(stages_root, 'seeds', 'seeds_{}.csv'.format(variant)))
        print(sg.plot_unmatched_seeds(table, arrays['Y_hw_chk'], arrays['max_proj'].values, cells, variant,
                                      os.path.join(stages_root, 'seeds', 'unmatched_{}.png'.format(variant))))
