"""Run a Minian parameter grid on one SST DIO session, on the Razer (plan §6).

    python scripts/sst_dio_parameter_grid.py <mouse> <data_root> <out_root> [<n_parallel>]

Session: that mouse's parameter-check session (HC2 of the conditioning day). Grid:
sst_dio_validation.PARAMETER_GRID_ONE_AT_A_TIME. Seed mask: session_queue.SEED_MASKS, else
sst_dio_validation.SEED_MASKS_PROPOSED. Output: <out_root>/<mouse>_HC2/<variant>/ and
grid_summary.csv there.
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from caban import session_queue as sq
from caban.sst_dio_validation import (PARAMETER_GRID_ONE_AT_A_TIME, SEED_MASKS_PROPOSED,
                                      parameter_check_session_dir, run_parameter_grid, grid_summary)

if __name__ == '__main__':
    if len(sys.argv) not in (4, 5):
        raise SystemExit(__doc__)
    mouse, data_root, out_root = sys.argv[1:4]
    n_parallel = int(sys.argv[4]) if len(sys.argv) == 5 else 4
    session_dir = os.path.join(parameter_check_session_dir(data_root, mouse), 'Miniscope')
    tail = sq.session_tail(session_dir)
    seed_mask = sq.SEED_MASKS.get(tail, SEED_MASKS_PROPOSED.get(tail))
    if seed_mask is None:
        raise SystemExit('no seed mask for {}: add one to SEED_MASKS_PROPOSED first'.format(tail))
    grid_root = os.path.join(out_root, '{}_HC2'.format(mouse))
    os.makedirs(grid_root, exist_ok=True)
    print('{}: {} variants, seed mask {}'.format(tail, len(PARAMETER_GRID_ONE_AT_A_TIME), seed_mask),
          flush=True)
    run_parameter_grid(session_dir, PARAMETER_GRID_ONE_AT_A_TIME, grid_root, seed_mask, n_parallel)
    summary = grid_summary(grid_root)
    summary.to_csv(os.path.join(grid_root, 'grid_summary.csv'))
    print(summary.to_string())
